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

from fastapi import FastAPI
from mcp.server.transport_security import TransportSecuritySettings

from perimeter.core.errors import AuthError
from perimeter.server.auth import request_scope, tokens_from_headers
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
        docs_url="/admin/docs",
        openapi_url="/admin/openapi.json",
        redoc_url=None,
    )
    app.add_middleware(RequestScopeMiddleware)

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": runtime.version,
            "documents": runtime.store.count_documents(),
            "chunks": runtime.store.count_chunks(),
            "index_size": runtime.index.size,
            "embedder": runtime.embedder_name,
            "store": runtime.store_name,
            "reranker": "cohere" if runtime.reranker is not None else "none",
        }

    for route in mcp_app.routes:
        app.router.routes.append(route)
    return app
