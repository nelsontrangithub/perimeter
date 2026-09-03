"""Google Drive connector (Drive API v3) acting as the calling user.

The OAuth token is read from the request scope at call time
(:func:`perimeter.server.auth.token_for`), never stored on the connector, so
the same connector object serves every request with that request's token and
holds nothing between them (ADR-005, INV-3).

Drive permissions map to grants only; Drive has no deny concept:

    type=user    -> Grant(emailAddress)
    type=group   -> Grant(emailAddress)          (a group is a principal)
    type=domain  -> Grant("domain:<domain>")     (front door forwards domain groups)
    type=anyone  -> Grant(EVERYONE)
    anything else -> ignored (never widens)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from perimeter.core.acl import AccessPolicy, Grant
from perimeter.core.document import SourceRef
from perimeter.core.errors import ConnectorError
from perimeter.core.principal import EVERYONE, PrincipalId
from perimeter.server.auth import token_for

DRIVE_BASE_URL = "https://www.googleapis.com"
GOOGLE_DOC = "application/vnd.google-apps.document"
TEXT_MIME_TYPES = ("text/plain", "text/markdown")
_URI_PREFIX = "gdrive://"


class GoogleDriveConnector:
    name = "gdrive"

    def __init__(self, *, client: httpx.Client | None = None, page_size: int = 100) -> None:
        self._client = client or httpx.Client(base_url=DRIVE_BASE_URL, timeout=30.0)
        self._page_size = page_size

    def __repr__(self) -> str:
        return "GoogleDriveConnector()"

    # -- plumbing ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        token = token_for(self.name)
        if token is None:
            raise ConnectorError(f"no {self.name} token in request scope")
        return {"Authorization": f"Bearer {token.reveal()}"}

    def _get(self, path: str, params: dict[str, Any]) -> httpx.Response:
        try:
            response = self._client.get(path, params=params, headers=self._headers())
        except httpx.HTTPError as exc:
            raise ConnectorError(f"drive: transport error ({type(exc).__name__})") from None
        if response.status_code != 200:
            raise ConnectorError(f"drive: HTTP {response.status_code} on {path.split('/')[-1]}")
        return response

    def _get_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            data = self._get(path, params).json()
        except ValueError:
            raise ConnectorError("drive: response is not JSON") from None
        if not isinstance(data, dict):
            raise ConnectorError("drive: response is not an object")
        return data

    @staticmethod
    def _parse_uri(ref: SourceRef) -> tuple[str, str]:
        """``gdrive://document/<id>`` (needs export) or ``gdrive://file/<id>``."""
        if not ref.uri.startswith(_URI_PREFIX):
            raise ConnectorError("not a gdrive:// uri")
        rest = ref.uri[len(_URI_PREFIX) :]
        kind, sep, file_id = rest.partition("/")
        if not sep:
            return "file", kind
        if kind not in ("document", "file") or not file_id:
            raise ConnectorError("malformed gdrive:// uri")
        return kind, file_id

    # -- connector protocol ----------------------------------------------------------

    def enumerate(self) -> Iterator[SourceRef]:
        mime_filter = " or ".join(f"mimeType='{m}'" for m in (GOOGLE_DOC, *TEXT_MIME_TYPES))
        params: dict[str, Any] = {
            "q": f"({mime_filter}) and trashed=false",
            "fields": "nextPageToken, files(id, name, mimeType, modifiedTime)",
            "pageSize": self._page_size,
        }
        while True:
            data = self._get_json("/drive/v3/files", params)
            files = data.get("files")
            if not isinstance(files, list):
                raise ConnectorError("drive: malformed files listing")
            for f in files:
                if not isinstance(f, dict) or "id" not in f:
                    continue
                kind = "document" if f.get("mimeType") == GOOGLE_DOC else "file"
                yield SourceRef(
                    connector=self.name,
                    uri=f"{_URI_PREFIX}{kind}/{f['id']}",
                    title=str(f.get("name", f["id"])),
                    version=str(f.get("modifiedTime", "")) or None,
                )
            token = data.get("nextPageToken")
            if not token:
                return
            params = {**params, "pageToken": token}

    def fetch(self, ref: SourceRef) -> str:
        kind, file_id = self._parse_uri(ref)
        if kind == "document":
            return self._get(f"/drive/v3/files/{file_id}/export", {"mimeType": "text/plain"}).text
        return self._get(f"/drive/v3/files/{file_id}", {"alt": "media"}).text

    def acl_for(self, ref: SourceRef) -> AccessPolicy:
        _, file_id = self._parse_uri(ref)
        data = self._get_json(
            f"/drive/v3/files/{file_id}/permissions",
            {"fields": "permissions(type, emailAddress, domain, role)", "pageSize": 100},
        )
        permissions = data.get("permissions")
        if not isinstance(permissions, list):
            raise ConnectorError("drive: malformed permissions")
        grants: list[Grant] = []
        for p in permissions:
            if not isinstance(p, dict):
                continue
            kind = p.get("type")
            if kind in ("user", "group") and isinstance(p.get("emailAddress"), str):
                grants.append(Grant(PrincipalId(p["emailAddress"])))
            elif kind == "domain" and isinstance(p.get("domain"), str):
                grants.append(Grant(PrincipalId(f"domain:{p['domain']}")))
            elif kind == "anyone":
                grants.append(Grant(EVERYONE))
        return AccessPolicy.from_rules(grants)
