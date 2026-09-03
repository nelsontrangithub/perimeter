"""MCP tool server: identity is forwarded in headers; without it, nothing is served."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from mcp.client import Client

from perimeter.core.acl import AccessPolicy, Grant
from perimeter.core.principal import PrincipalId
from perimeter.pipeline.retrieve import Retriever
from perimeter.server.auth import GROUPS_HEADER, PRINCIPAL_HEADER
from perimeter.server.mcp import build_mcp_server
from tests.pipeline.fakes import ListIndex, StaticResolver, build_corpus, raw

pytestmark = pytest.mark.anyio

DOCS = [
    raw("pub", "MARKER_PUB apples apples apples", AccessPolicy.public()),
    raw(
        "eng",
        "MARKER_ENG apples apples apples",
        AccessPolicy.from_rules([Grant(PrincipalId("eng"))]),
    ),
    raw(
        "alice",
        "MARKER_ALICE apples apples apples",
        AccessPolicy.from_rules([Grant(PrincipalId("alice"))]),
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def stack() -> tuple[Retriever, ListIndex]:
    store, index, embedder = build_corpus(DOCS)
    retriever = Retriever(
        resolver=StaticResolver(), embedder=embedder, index=index, store=store, reranker=None
    )
    return retriever, index


@pytest.fixture
async def http(stack: tuple[Retriever, ListIndex]) -> AsyncIterator[httpx.AsyncClient]:
    app = build_mcp_server(stack[0]).streamable_http_app(stateless_http=True, json_response=True)
    # httpx's ASGI transport does not drive lifespan events; the session manager needs them.
    # The SDK's DNS-rebinding guard only accepts the configured host, hence 127.0.0.1.
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8000"
        ) as c:
            yield c


async def _call(
    http: httpx.AsyncClient, tool: str, arguments: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    response = await http.post(
        "/mcp",
        json=body,
        headers={
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": "2025-06-18",
            **headers,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert "result" in payload, payload
    result: dict[str, Any] = payload["result"]
    return result


def _structured(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("structuredContent"):
        out: dict[str, Any] = result["structuredContent"]
        return out
    parsed: dict[str, Any] = json.loads(result["content"][0]["text"])
    return parsed


async def test_tool_is_listed_with_identity_instructions(
    stack: tuple[Retriever, ListIndex],
) -> None:
    async with Client(build_mcp_server(stack[0])) as client:
        tools = (await client.list_tools()).tools
    names = {t.name for t in tools}
    assert "retrieve" in names
    retrieve = next(t for t in tools if t.name == "retrieve")
    assert PRINCIPAL_HEADER in (retrieve.description or "")


async def test_call_without_identity_is_an_error_and_touches_nothing(
    stack: tuple[Retriever, ListIndex],
) -> None:
    retriever, index = stack
    async with Client(build_mcp_server(retriever)) as client:
        result = await client.call_tool("retrieve", {"query": "apples", "k": 5})
    assert result.is_error
    assert "identity" in result.content[0].text.lower()  # type: ignore[union-attr]
    assert index.search_calls == []


async def test_forwarded_identity_scopes_results(http: httpx.AsyncClient) -> None:
    result = _structured(
        await _call(
            http,
            "retrieve",
            {"query": "apples", "k": 10},
            {PRINCIPAL_HEADER: "bob", GROUPS_HEADER: "eng"},
        )
    )
    docs = {r["citation"]["document_id"] for r in result["results"]}
    assert docs == {"pub", "eng"}
    assert result["requested_k"] == 10
    assert result["returned"] == len(result["results"])
    texts = " ".join(r["text"] for r in result["results"])
    assert "MARKER_ALICE" not in texts
    for r in result["results"]:
        assert set(r["citation"]) >= {"chunk_id", "document_id", "uri", "title", "start", "end"}
        assert isinstance(r["score"], float)


async def test_other_identity_sees_other_documents(http: httpx.AsyncClient) -> None:
    result = _structured(
        await _call(http, "retrieve", {"query": "apples", "k": 10}, {PRINCIPAL_HEADER: "alice"})
    )
    assert {r["citation"]["document_id"] for r in result["results"]} == {"pub", "alice"}


async def test_missing_identity_over_http_is_tool_error(http: httpx.AsyncClient) -> None:
    result = await _call(http, "retrieve", {"query": "apples", "k": 3}, {})
    assert result.get("isError") is True


async def test_invalid_request_is_tool_error_without_query_echo(http: httpx.AsyncClient) -> None:
    result = await _call(
        http, "retrieve", {"query": "SECRET QUERY TEXT", "k": 0}, {PRINCIPAL_HEADER: "alice"}
    )
    assert result.get("isError") is True
    assert "SECRET" not in json.dumps(result)


async def test_whoami_reports_forwarded_identity(http: httpx.AsyncClient) -> None:
    result = _structured(
        await _call(http, "whoami", {}, {PRINCIPAL_HEADER: "bob", GROUPS_HEADER: "eng"})
    )
    assert result["principal"] == "bob"
    assert set(result["effective_principals"]) == {"bob", "eng", "everyone"}
