"""MCP tool server over StreamableHTTP.

One retrieval tool. The caller's identity arrives in request headers
(:mod:`perimeter.server.auth`), never in tool arguments, so a model cannot
talk itself into a different user's permissions by changing a parameter.
"""

from __future__ import annotations

from typing import Any, Protocol

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from perimeter.core.acl import PermissionSet
from perimeter.core.errors import AuthError, InvalidRequestError, PerimeterError
from perimeter.core.query import RetrievalRequest, ScopedResult
from perimeter.server.auth import GROUPS_HEADER, PRINCIPAL_HEADER, identity_from_headers

RETRIEVE_DESCRIPTION = (
    "Retrieve the chunks most relevant to `query` that the calling user is permitted to see. "
    f"Identity is taken from the {PRINCIPAL_HEADER} and {GROUPS_HEADER} request headers, "
    "never from arguments. Documents the caller may not read are never scored, never "
    "ranked, and never returned. `returned` may be less than `k` when the caller's "
    "permitted set is small; that is the honest answer, not a filtering artefact."
)


class RetrievalService(Protocol):
    """What the tool needs: a Retriever, traced or not."""

    def permissions_for(self, request: RetrievalRequest) -> PermissionSet: ...

    def retrieve(self, request: RetrievalRequest) -> ScopedResult: ...


def result_to_payload(result: ScopedResult) -> dict[str, Any]:
    return {
        "requested_k": result.requested_k,
        "returned": result.returned,
        "candidates": result.candidates,
        "results": [
            {
                "text": sc.text,
                "score": float(sc.score),
                "citation": {
                    "chunk_id": sc.citation.chunk_id,
                    "document_id": sc.citation.document_id,
                    "connector": sc.citation.source.connector,
                    "uri": sc.citation.source.uri,
                    "title": sc.citation.source.title,
                    "version": sc.citation.source.version,
                    "start": sc.citation.start,
                    "end": sc.citation.end,
                },
            }
            for sc in result.chunks
        ],
    }


def build_mcp_server(retriever: RetrievalService, *, name: str = "perimeter") -> MCPServer[Any]:
    server: MCPServer[Any] = MCPServer(
        name,
        instructions=(
            "Permission-aware retrieval. Every call is scoped to the identity forwarded in "
            f"{PRINCIPAL_HEADER}; results never include text the caller may not read."
        ),
    )

    @server.tool(name="retrieve", description=RETRIEVE_DESCRIPTION)
    def retrieve(query: str, k: int, ctx: Context[Any, Any]) -> dict[str, Any]:
        try:
            principal = identity_from_headers(ctx.headers)
        except AuthError as exc:
            raise ToolError(str(exc)) from None
        try:
            request = RetrievalRequest(principal=principal, query=query, k=k)
        except InvalidRequestError as exc:
            raise ToolError(f"invalid request: {exc}") from None
        try:
            return result_to_payload(retriever.retrieve(request))
        except PerimeterError as exc:
            raise ToolError(f"retrieval failed: {type(exc).__name__}") from None

    @server.tool(
        name="whoami",
        description=(
            "Report the identity Perimeter sees for this call and its effective principals."
        ),
    )
    def whoami(ctx: Context[Any, Any]) -> dict[str, Any]:
        try:
            principal = identity_from_headers(ctx.headers)
        except AuthError as exc:
            raise ToolError(str(exc)) from None
        permitted: PermissionSet = retriever.permissions_for(
            RetrievalRequest(principal=principal, query="-", k=1)
        )
        return {
            "principal": principal.id,
            "forwarded_groups": sorted(principal.groups),
            "effective_principals": sorted(permitted),
        }

    return server
