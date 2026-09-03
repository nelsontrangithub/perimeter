"""FastAPI application: /health, the MCP endpoint, and (later) the admin API.

Every HTTP request runs inside a connector-token request scope
(:func:`perimeter.server.auth.request_scope`), entered by a pure ASGI
middleware so the scope lives in the same task as the handler and is wiped
when the response is done.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

from perimeter.core.errors import (
    AuthError,
    ConnectorError,
    InvalidPrincipalError,
    InvalidRequestError,
)
from perimeter.core.principal import Principal, parse_group_id, parse_principal_id
from perimeter.core.query import RetrievalRequest
from perimeter.server.auth import request_scope, tokens_from_headers
from perimeter.server.connectors import ConnectorConfig, IngestRun, Kind
from perimeter.server.mcp import result_to_payload
from perimeter.server.wiring import Runtime

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class RequestScopeMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers", [])}
        try:
            tokens_from_headers(headers)
        except AuthError as exc:
            await _json_response(send, 401, {"error": str(exc)})
            return
        with request_scope(headers):
            await self._app(scope, receive, send)


async def _json_response(send: Send, status: int, body: dict[str, Any]) -> None:
    payload = json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": payload})


def transport_security(allowed_hosts: tuple[str, ...]) -> TransportSecuritySettings:
    if "*" in allowed_hosts:
        return TransportSecuritySettings(enable_dns_rebinding_protection=False)
    origins = [f"{scheme}://{h}" for h in allowed_hosts for scheme in ("http", "https")]
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(allowed_hosts),
        allowed_origins=origins,
    )


def build_app(runtime: Runtime) -> FastAPI:
    mcp_app = runtime.mcp_server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        transport_security=transport_security(runtime.settings.allowed_hosts),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    app = FastAPI(
        title="Perimeter",
        version=runtime.version,
        lifespan=lifespan,
        docs_url="/admin/api/docs",
        openapi_url="/admin/api/openapi.json",
        redoc_url=None,
    )
    app.add_middleware(RequestScopeMiddleware)

    @app.get("/health", tags=["ops"], operation_id="health")
    def health() -> Health:
        return Health(
            status="ok",
            version=runtime.version,
            documents=runtime.store.count_documents(),
            chunks=runtime.store.count_chunks(),
            index_size=runtime.index.size,
            embedder=runtime.embedder_name,
            store=runtime.store_name,
            reranker="cohere" if runtime.reranker is not None else "none",
        )

    @app.post("/admin/api/acl/invalidate", tags=["admin"], operation_id="invalidateAcl")
    def acl_invalidate(body: AclInvalidate) -> dict[str, Any]:
        """The ADR-004 invalidation hook. Call on any membership change, either direction."""
        try:
            if body.all:
                runtime.acl_cache.invalidate_all()
                return {"invalidated": "all"}
            if body.principal is not None:
                runtime.acl_cache.invalidate(parse_principal_id(body.principal))
                return {"invalidated": body.principal}
            if body.group is not None:
                runtime.acl_cache.invalidate_group(parse_group_id(body.group))
                return {"invalidated": body.group}
        except InvalidPrincipalError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        raise HTTPException(status_code=422, detail="one of principal, group, or all is required")

    @app.get("/admin/api/connectors", tags=["admin"], operation_id="listConnectors")
    def list_connectors() -> list[ConnectorView]:
        return [_connector_view(runtime, c) for c in runtime.connectors.list()]

    @app.post(
        "/admin/api/connectors", tags=["admin"], operation_id="createConnector", status_code=201
    )
    def create_connector(body: ConnectorCreate) -> ConnectorView:
        config = ConnectorConfig(name=body.name, kind=body.kind, root=body.root)
        try:
            runtime.connectors.add(config)
        except ConnectorError as exc:
            status = 409 if "already exists" in str(exc) else 422
            raise HTTPException(status_code=status, detail=str(exc)) from None
        return _connector_view(runtime, config)

    @app.delete(
        "/admin/api/connectors/{name}",
        tags=["admin"],
        operation_id="deleteConnector",
        status_code=204,
        response_class=Response,
    )
    def delete_connector(name: str) -> Response:
        try:
            runtime.connectors.remove(name)
        except ConnectorError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        return Response(status_code=204)

    @app.post("/admin/api/connectors/{name}/ingest", tags=["admin"], operation_id="ingestConnector")
    def ingest_connector(name: str) -> IngestRunView:
        try:
            runtime.connectors.get(name)
        except ConnectorError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        run = runtime.connectors.ingest(name, runtime.ingestor, runtime.index)
        return IngestRunView.from_run(run)

    @app.get("/admin/api/index", tags=["admin"], operation_id="indexHealth")
    def index_health() -> IndexHealth:
        stats = runtime.index.stats()
        cache = runtime.acl_cache.stats
        return IndexHealth(
            rows=stats.rows,
            staged=stats.staged,
            dimension=stats.dimension,
            quantizer_fitted=stats.quantizer_fitted,
            rescore_multiplier=stats.rescore_multiplier,
            acl_principals=stats.acl_principals,
            files=stats.files,
            bytes_on_disk=stats.bytes_on_disk,
            bytes_per_chunk=stats.bytes_per_chunk,
            documents=runtime.store.count_documents(),
            chunks=runtime.store.count_chunks(),
            acl_cache=AclCacheView(
                ttl_seconds=runtime.acl_cache.ttl_seconds,
                hits=cache.hits,
                misses=cache.misses,
                errors=cache.errors,
                evictions=cache.evictions,
                size=cache.size,
            ),
        )

    @app.post("/admin/api/simulate", tags=["admin"], operation_id="simulate")
    def simulate(body: SimulateRequest) -> Simulation:
        """Preview the corpus as any principal: what they would see, and why."""
        try:
            principal = Principal(
                id=parse_principal_id(body.principal),
                groups=frozenset(parse_group_id(g) for g in body.groups),
            )
            request = RetrievalRequest(principal=principal, query=body.query or "-", k=body.k)
        except (InvalidPrincipalError, InvalidRequestError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        permitted = runtime.retriever.permissions_for(request)
        documents = [
            _explain(summary, permitted) for summary in runtime.store.catalog(limit=body.limit)
        ]
        results = None
        if body.query and body.query.strip():
            results = result_to_payload(runtime.retriever.retrieve_with(request, permitted))
        return Simulation(
            principal=principal.id,
            effective_principals=sorted(permitted),
            documents=documents,
            visible_count=sum(1 for d in documents if d.visible),
            total=len(documents),
            results=results,
        )

    for route in mcp_app.routes:
        app.router.routes.append(route)

    dist = runtime.settings.admin_dist
    if dist is not None and dist.is_dir():
        app.mount("/admin", StaticFiles(directory=dist, html=True), name="admin-console")
    return app


class ConnectorCreate(BaseModel):
    name: str
    kind: Kind
    root: str | None = None


class IngestRunView(BaseModel):
    started_at: str
    duration_seconds: float
    documents: int
    chunks: int
    skipped_unchanged: int
    unreadable: int
    error: str | None

    @classmethod
    def from_run(cls, run: IngestRun) -> IngestRunView:
        return cls(
            started_at=run.started_at,
            duration_seconds=run.duration_seconds,
            documents=run.documents,
            chunks=run.chunks,
            skipped_unchanged=run.skipped_unchanged,
            unreadable=run.unreadable,
            error=run.error,
        )


class ConnectorView(BaseModel):
    name: str
    kind: Kind
    root: str | None
    needs_request_token: bool
    last_run: IngestRunView | None


def _connector_view(runtime: Runtime, config: ConnectorConfig) -> ConnectorView:
    run = runtime.connectors.last_run(config.name)
    return ConnectorView(
        name=config.name,
        kind=config.kind,
        root=config.root,
        needs_request_token=config.kind == "gdrive",
        last_run=IngestRunView.from_run(run) if run else None,
    )


class AclCacheView(BaseModel):
    ttl_seconds: float
    hits: int
    misses: int
    errors: int
    evictions: int
    size: int


class IndexHealth(BaseModel):
    rows: int
    staged: int
    dimension: int
    quantizer_fitted: bool
    rescore_multiplier: int
    acl_principals: int
    files: dict[str, int]
    bytes_on_disk: int
    bytes_per_chunk: float
    documents: int
    chunks: int
    acl_cache: AclCacheView


class SimulateRequest(BaseModel):
    principal: str
    groups: list[str] = []
    query: str | None = None
    k: int = 10
    limit: int = 500


class DocumentDecision(BaseModel):
    id: str
    title: str
    uri: str
    connector: str
    grants: list[str]
    denies: list[str]
    visible: bool
    reason: str


class Simulation(BaseModel):
    principal: str
    effective_principals: list[str]
    documents: list[DocumentDecision]
    visible_count: int
    total: int
    results: dict[str, Any] | None


def _explain(summary: Any, permitted: Any) -> DocumentDecision:
    decision = summary.policy.explain(permitted)
    return DocumentDecision(
        id=summary.id,
        title=summary.source.title,
        uri=summary.source.uri,
        connector=summary.source.connector,
        grants=sorted(summary.policy.grants),
        denies=sorted(summary.policy.denies),
        visible=decision.admitted,
        reason=decision.reason,
    )


class Health(BaseModel):
    status: str
    version: str
    documents: int
    chunks: int
    index_size: int
    embedder: str
    store: str
    reranker: str


class AclInvalidate(BaseModel):
    principal: str | None = None
    group: str | None = None
    all: bool = False
