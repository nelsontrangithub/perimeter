"""FastAPI app: /health, MCP mounted, request scope per request."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from perimeter.core.acl import AccessPolicy
from perimeter.server.auth import PRINCIPAL_HEADER, TOKEN_HEADER_PREFIX, current_tokens
from perimeter.server.http import build_app
from perimeter.server.settings import Settings
from perimeter.server.wiring import Runtime, build_runtime
from tests.pipeline.fakes import raw

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def runtime(tmp_path: Any) -> Runtime:
    settings = Settings(data_dir=tmp_path / "data", cohere_api_key=None, database_url=None)
    rt = build_runtime(settings)
    rt.ingestor.ingest([raw("d", "apples pears plums", AccessPolicy.public())])
    return rt


@pytest.fixture
async def client(runtime: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = build_app(runtime)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as c:
            yield c


async def test_health_reports_ok_and_counts(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents"] == 1
    assert body["chunks"] >= 1
    assert body["index_size"] == body["chunks"]
    assert body["embedder"] == "local"
    assert body["store"] == "memory"
    assert body["version"]


async def test_mcp_is_mounted_and_identity_flows_through(client: httpx.AsyncClient) -> None:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "whoami", "arguments": {}},
    }
    response = await client.post(
        "/mcp",
        json=body,
        headers={"Accept": "application/json, text/event-stream", PRINCIPAL_HEADER: "alice"},
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    payload = result.get("structuredContent") or {}
    assert payload["principal"] == "alice"


async def test_request_scope_is_per_request(runtime: Runtime) -> None:
    seen: list[list[str]] = []
    app = build_app(runtime)

    @app.get("/_probe")
    def probe() -> dict[str, list[str]]:
        seen.append(sorted(current_tokens()))
        return {"tokens": sorted(current_tokens())}

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as c:
            r1 = await c.get("/_probe", headers={f"{TOKEN_HEADER_PREFIX}gdrive": "tok"})
            r2 = await c.get("/_probe")
    assert r1.json()["tokens"] == ["gdrive"]
    assert r2.json()["tokens"] == []
    assert current_tokens() == {}


async def test_malformed_token_header_is_rejected_with_401(client: httpx.AsyncClient) -> None:
    response = await client.get("/health", headers={f"{TOKEN_HEADER_PREFIX}gdrive": "   "})
    assert response.status_code == 401


async def test_admin_acl_invalidate_endpoint(client: httpx.AsyncClient, runtime: Runtime) -> None:
    from perimeter.core.principal import Principal, PrincipalId
    from perimeter.core.query import RetrievalRequest

    runtime.retriever.permissions_for(
        RetrievalRequest(principal=Principal(PrincipalId("alice")), query="x", k=1)
    )
    assert runtime.acl_cache.stats.size == 1
    response = await client.post("/admin/acl/invalidate", json={"principal": "alice"})
    assert response.status_code == 200
    assert response.json() == {"invalidated": "alice"}
    assert runtime.acl_cache.stats.size == 0
    response = await client.post("/admin/acl/invalidate", json={"all": True})
    assert response.status_code == 200
    response = await client.post("/admin/acl/invalidate", json={})
    assert response.status_code == 422
