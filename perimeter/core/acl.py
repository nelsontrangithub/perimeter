"""Access policies and permission sets.

An :class:`AccessPolicy` is attached to every document (and inherited by its
chunks). It is evaluated against a caller's :class:`PermissionSet`, which is the
frozen set of principals the caller *is* (see
:func:`perimeter.core.principal.effective_principals`).

Semantics, in priority order:

1. An empty permission set is never admitted. Fail closed (INV-4).
2. Any explicit :class:`Deny` that intersects the permission set refuses.
3. Otherwise the caller is admitted if any :class:`Grant` intersects it.
4. A policy with no grants admits nobody, including the public principal.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from perimeter.core.principal import (
    EVERYONE,
    GroupGraph,
    Principal,
    PrincipalId,
    effective_principals,
)


@dataclass(frozen=True, slots=True)
class PermissionSet:
    """The principals a caller is. Produced by resolution, consumed by policy checks."""

    principals: frozenset[PrincipalId]

    @classmethod
    def empty(cls) -> PermissionSet:
        return cls(frozenset())

    @classmethod
    def of(cls, *ids: PrincipalId) -> PermissionSet:
        return cls(frozenset(ids))

    @classmethod
    def resolve(cls, principal: Principal, graph: GroupGraph) -> PermissionSet:
        return cls(effective_principals(principal, graph))

    @property
    def is_empty(self) -> bool:
        return not self.principals

    def __contains__(self, item: object) -> bool:
        return item in self.principals

    def __iter__(self) -> Iterator[PrincipalId]:
        return iter(self.principals)

    def __len__(self) -> int:
        return len(self.principals)

    def without(self, *ids: PrincipalId) -> PermissionSet:
        """A strictly-not-larger permission set. Used when a membership is revoked."""
        return PermissionSet(self.principals.difference(ids))

    def is_subset_of(self, other: PermissionSet) -> bool:
        return self.principals <= other.principals


@dataclass(frozen=True, slots=True)
class Grant:
    """Admit any caller whose permission set contains ``principal``."""

    principal: PrincipalId


@dataclass(frozen=True, slots=True)
class Deny:
    """Refuse any caller whose permission set contains ``principal``. Wins over grants."""

    principal: PrincipalId


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    """Who may read a document. Immutable; mutation methods return a new policy."""

    grants: frozenset[PrincipalId]
    denies: frozenset[PrincipalId] = field(default_factory=frozenset)

    @classmethod
    def from_rules(cls, rules: Iterable[Grant | Deny]) -> AccessPolicy:
        grants: set[PrincipalId] = set()
        denies: set[PrincipalId] = set()
        for rule in rules:
            if isinstance(rule, Grant):
                grants.add(rule.principal)
            else:
                denies.add(rule.principal)
        return cls(frozenset(grants), frozenset(denies))

    @classmethod
    def public(cls) -> AccessPolicy:
        """Readable by every resolved caller."""
        return cls(frozenset({EVERYONE}))

    @classmethod
    def nobody(cls) -> AccessPolicy:
        """Readable by no one. The safe default for a document with unknown ACLs."""
        return cls(frozenset())

    def admits(self, permissions: PermissionSet) -> bool:
        if permissions.is_empty:
            return False
        if not self.denies.isdisjoint(permissions.principals):
            return False
        return not self.grants.isdisjoint(permissions.principals)

    def revoke(self, principal: PrincipalId) -> AccessPolicy:
        return AccessPolicy(self.grants - {principal}, self.denies)

    def with_grant(self, principal: PrincipalId) -> AccessPolicy:
        return AccessPolicy(self.grants | {principal}, self.denies)

    def with_deny(self, principal: PrincipalId) -> AccessPolicy:
        return AccessPolicy(self.grants, self.denies | {principal})
