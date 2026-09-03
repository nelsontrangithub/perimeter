"""ACL resolver that trusts the forwarded identity and a static nested-group graph.

This is the resolver for deployments where the front door already resolves
direct group memberships and forwards them. Nested groups are expanded from a
graph loaded at start (``PERIMETER_GROUPS_FILE``, ``{child: [parents]}``).
"""

from __future__ import annotations

import json
from pathlib import Path

from perimeter.core.acl import PermissionSet
from perimeter.core.errors import AclResolutionError, InvalidPrincipalError
from perimeter.core.principal import GroupGraph, Principal


class StaticAclResolver:
    def __init__(self, graph: GroupGraph | None = None) -> None:
        self._graph = graph or GroupGraph.empty()

    @classmethod
    def from_file(cls, path: Path) -> StaticAclResolver:
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise AclResolutionError(f"groups file unreadable ({type(exc).__name__})") from None
        if not isinstance(raw, dict):
            raise AclResolutionError("groups file must be a JSON object of child -> [parents]")
        try:
            return cls(GroupGraph.from_edges(raw))
        except InvalidPrincipalError as exc:
            raise AclResolutionError(f"groups file invalid: {exc}") from None

    def resolve(self, principal: Principal) -> PermissionSet:
        return PermissionSet.resolve(principal, self._graph)
