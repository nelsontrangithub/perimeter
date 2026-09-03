"""Google Drive connector over a fake transport, using the request-scoped token."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from perimeter.connectors.gdrive import GoogleDriveConnector
from perimeter.core.acl import AccessPolicy, Grant
from perimeter.core.document import SourceRef
from perimeter.core.errors import ConnectorError
from perimeter.core.principal import EVERYONE, PrincipalId
from perimeter.server.auth import TOKEN_HEADER_PREFIX, request_scope

SECRET = "ya29.A0ARrdaM-super-secret-token-value-1234567890"
Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler), base_url="https://www.googleapis.com"
    )


def _files_response(files: list[dict[str, Any]], next_page: str | None = None) -> httpx.Response:
    body: dict[str, Any] = {"files": files}
    if next_page:
        body["nextPageToken"] = next_page
    return httpx.Response(200, json=body)


def _handler(seen: list[httpx.Request]) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if path == "/drive/v3/files":
            if request.url.params.get("pageToken") == "p2":
                return _files_response(
                    [{"id": "f2", "name": "Plain", "mimeType": "text/plain", "modifiedTime": "t2"}]
                )
            return _files_response(
                [
                    {
                        "id": "f1",
                        "name": "Doc",
                        "mimeType": "application/vnd.google-apps.document",
                        "modifiedTime": "t1",
                    }
                ],
                next_page="p2",
            )
        if path == "/drive/v3/files/f1/export":
            assert request.url.params["mimeType"] == "text/plain"
            return httpx.Response(200, text="exported doc text")
        if path == "/drive/v3/files/f2":
            assert request.url.params["alt"] == "media"
            return httpx.Response(200, text="plain file text")
        if path == "/drive/v3/files/f1/permissions":
            return httpx.Response(
                200,
                json={
                    "permissions": [
                        {"type": "user", "emailAddress": "alice@example.com", "role": "reader"},
                        {"type": "group", "emailAddress": "eng@example.com", "role": "writer"},
                        {"type": "domain", "domain": "example.com", "role": "reader"},
                    ]
                },
            )
        if path == "/drive/v3/files/f2/permissions":
            return httpx.Response(200, json={"permissions": [{"type": "anyone", "role": "reader"}]})
        return httpx.Response(404)

    return handle


def test_enumerate_pages_through_files_with_bearer_token() -> None:
    seen: list[httpx.Request] = []
    conn = GoogleDriveConnector(client=_client(_handler(seen)))
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}):
        refs = list(conn.enumerate())
    assert [r.uri for r in refs] == ["gdrive://document/f1", "gdrive://file/f2"]
    assert refs[0].title == "Doc" and refs[0].version == "t1"
    assert all(r.headers["authorization"] == f"Bearer {SECRET}" for r in seen)
    assert len([r for r in seen if r.url.path == "/drive/v3/files"]) == 2


def test_fetch_exports_google_docs_and_downloads_plain_files() -> None:
    seen: list[httpx.Request] = []
    conn = GoogleDriveConnector(client=_client(_handler(seen)))
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}):
        refs = list(conn.enumerate())
        assert conn.fetch(refs[0]) == "exported doc text"
        assert conn.fetch(refs[1]) == "plain file text"


def test_acl_maps_drive_permissions_to_grants_only() -> None:
    seen: list[httpx.Request] = []
    conn = GoogleDriveConnector(client=_client(_handler(seen)))
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}):
        refs = list(conn.enumerate())
        doc_policy = conn.acl_for(refs[0])
        anyone_policy = conn.acl_for(refs[1])
    assert doc_policy == AccessPolicy.from_rules(
        [
            Grant(PrincipalId("alice@example.com")),
            Grant(PrincipalId("eng@example.com")),
            Grant(PrincipalId("domain:example.com")),
        ]
    )
    assert doc_policy.denies == frozenset()
    assert anyone_policy == AccessPolicy.from_rules([Grant(EVERYONE)])


def test_without_token_in_scope_every_operation_fails_closed() -> None:
    conn = GoogleDriveConnector(client=_client(_handler([])))
    with pytest.raises(ConnectorError) as info:
        list(conn.enumerate())
    assert "gdrive" in str(info.value)
    with pytest.raises(ConnectorError):
        conn.acl_for(SourceRef("gdrive", "gdrive://f1", "Doc"))


def test_http_failure_is_a_typed_error_without_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": f"denied for {SECRET}"}})

    conn = GoogleDriveConnector(client=_client(handler))
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}):
        with pytest.raises(ConnectorError) as info:
            list(conn.enumerate())
    assert SECRET not in str(info.value)
    assert "403" in str(info.value)


def test_unknown_permission_type_is_ignored_not_widened() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/permissions"):
            return httpx.Response(
                200, json={"permissions": [{"type": "mystery"}, {"type": "user"}]}
            )
        return httpx.Response(404)

    conn = GoogleDriveConnector(client=_client(handler))
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}):
        policy = conn.acl_for(SourceRef("gdrive", "gdrive://x", "X"))
    assert policy == AccessPolicy.nobody()


def test_malformed_permissions_payload_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"permissions": "nope"})

    conn = GoogleDriveConnector(client=_client(handler))
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}):
        with pytest.raises(ConnectorError):
            conn.acl_for(SourceRef("gdrive", "gdrive://x", "X"))


def test_token_is_read_at_call_time_not_construction() -> None:
    seen: list[httpx.Request] = []
    conn = GoogleDriveConnector(client=_client(_handler(seen)))
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": "first"}):
        list(conn.enumerate())
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": "second"}):
        list(conn.enumerate())
    tokens = [json.dumps(r.headers["authorization"]) for r in seen]
    assert any("first" in t for t in tokens) and any("second" in t for t in tokens)
