"""Principals, groups, and effective-principal resolution.

A *principal* is anything a grant or deny can name: a user, a group, or the
public principal :data:`EVERYONE`. Access policies (:mod:`perimeter.core.acl`)
are evaluated against a caller's *effective principals*: the caller's own ID,
every group they belong to transitively, and :data:`EVERYONE`.

Group membership is a directed graph (a group can be nested inside other
groups). Resolution is a bounded closure that tolerates cycles.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import NewType

from perimeter.core.errors import InvalidPrincipalError

PrincipalId = NewType("PrincipalId", str)
"""Identifier of a user or other subject. Opaque to Perimeter."""

GroupId = NewType("GroupId", PrincipalId)
"""Identifier of a group. A ``GroupId`` is a ``PrincipalId``: grants can name groups."""

EVERYONE = PrincipalId("everyone")
"""The public principal. Added to every effective set by resolution; never claimable."""

_MAX_ID_LENGTH = 256


def _validate_id(raw: str, kind: str) -> str:
    if not raw or raw != raw.strip():
        raise InvalidPrincipalError(f"{kind} id must be non-empty with no surrounding whitespace")
    if len(raw) > _MAX_ID_LENGTH:
        raise InvalidPrincipalError(f"{kind} id exceeds {_MAX_ID_LENGTH} characters")
    if any(ch.isspace() or ord(ch) < 0x20 for ch in raw):
        raise InvalidPrincipalError(f"{kind} id contains whitespace or control characters")
    if raw == EVERYONE:
        raise InvalidPrincipalError(f"{kind} id {raw!r} is reserved")
    return raw


def parse_principal_id(raw: str) -> PrincipalId:
    """Validate an untrusted string as a :data:`PrincipalId`."""
    return PrincipalId(_validate_id(raw, "principal"))


def parse_group_id(raw: str) -> GroupId:
    """Validate an untrusted string as a :data:`GroupId`."""
    return GroupId(PrincipalId(_validate_id(raw, "group")))


@dataclass(frozen=True, slots=True)
class Principal:
    """A caller identity as forwarded by the client: an ID and direct group memberships."""

    id: PrincipalId
    groups: frozenset[GroupId] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class GroupGraph:
    """Nested-group membership: for each group, the groups it is directly a member of."""

    _parents: Mapping[GroupId, frozenset[GroupId]]

    @classmethod
    def empty(cls) -> GroupGraph:
        return cls({})

    @classmethod
    def from_edges(cls, edges: Mapping[str, Iterable[str]]) -> GroupGraph:
        """Build from ``{child_group: [parent_group, ...]}`` using untrusted strings."""
        parents: dict[GroupId, frozenset[GroupId]] = {}
        for child, ps in edges.items():
            parents[parse_group_id(child)] = frozenset(parse_group_id(p) for p in ps)
        return cls(parents)

    def parents_of(self, group: GroupId) -> frozenset[GroupId]:
        return self._parents.get(group, frozenset())

    def with_edge(self, child: GroupId, parent: GroupId) -> GroupGraph:
        """A new graph in which ``child`` is also a member of ``parent``."""
        parents = dict(self._parents)
        parents[child] = parents.get(child, frozenset()) | {parent}
        return GroupGraph(parents)


def effective_principals(principal: Principal, graph: GroupGraph) -> frozenset[PrincipalId]:
    """Resolve the full set of principals a caller *is*.

    The result is the caller's own ID, the transitive closure of their group
    memberships over ``graph``, and :data:`EVERYONE`. Cycles in the graph are
    tolerated: each group is visited at most once.
    """
    seen: set[GroupId] = set()
    frontier: list[GroupId] = list(principal.groups)
    while frontier:
        group = frontier.pop()
        if group in seen:
            continue
        seen.add(group)
        frontier.extend(graph.parents_of(group))
    return frozenset({principal.id, EVERYONE, *seen})
