"""Unit tests for access policies: grants, explicit denies, permission sets. No I/O."""

from __future__ import annotations

from perimeter.core.acl import AccessPolicy, Deny, Grant, PermissionSet
from perimeter.core.principal import EVERYONE, GroupGraph, GroupId, Principal, PrincipalId

ALICE = PrincipalId("alice")
BOB = PrincipalId("bob")
ENG = GroupId(PrincipalId("eng"))
CONTRACTORS = GroupId(PrincipalId("contractors"))


def _perms(*ids: PrincipalId) -> PermissionSet:
    return PermissionSet(frozenset(ids))


def test_grant_to_user_admits_that_user() -> None:
    policy = AccessPolicy.from_rules([Grant(ALICE)])
    assert policy.admits(_perms(ALICE, EVERYONE))


def test_grant_to_user_does_not_admit_other_user() -> None:
    policy = AccessPolicy.from_rules([Grant(ALICE)])
    assert not policy.admits(_perms(BOB, EVERYONE))


def test_grant_to_group_admits_member() -> None:
    policy = AccessPolicy.from_rules([Grant(ENG)])
    assert policy.admits(_perms(ALICE, ENG, EVERYONE))


def test_no_grants_admits_nobody_even_with_everyone() -> None:
    policy = AccessPolicy.nobody()
    assert not policy.admits(_perms(ALICE, EVERYONE))


def test_public_policy_admits_any_resolved_caller() -> None:
    assert AccessPolicy.public().admits(_perms(BOB, EVERYONE))


def test_public_policy_requires_everyone_in_permission_set() -> None:
    """A permission set without EVERYONE was not produced by resolution; do not trust it."""
    assert not AccessPolicy.public().admits(_perms(BOB))


def test_empty_permission_set_is_never_admitted() -> None:
    """INV-4 at the policy level: fail closed."""
    assert not AccessPolicy.public().admits(PermissionSet.empty())
    assert not AccessPolicy.from_rules([Grant(ALICE)]).admits(PermissionSet.empty())


def test_deny_overrides_grant_for_same_principal() -> None:
    policy = AccessPolicy.from_rules([Grant(ALICE), Deny(ALICE)])
    assert not policy.admits(_perms(ALICE, EVERYONE))


def test_deny_on_group_overrides_direct_user_grant() -> None:
    policy = AccessPolicy.from_rules([Grant(ALICE), Deny(CONTRACTORS)])
    assert not policy.admits(_perms(ALICE, CONTRACTORS, EVERYONE))


def test_deny_on_everyone_locks_document() -> None:
    policy = AccessPolicy.from_rules([Grant(ALICE), Deny(EVERYONE)])
    assert not policy.admits(_perms(ALICE, EVERYONE))


def test_from_rules_separates_grants_and_denies() -> None:
    policy = AccessPolicy.from_rules([Grant(ALICE), Grant(ENG), Deny(BOB)])
    assert policy.grants == frozenset({ALICE, ENG})
    assert policy.denies == frozenset({BOB})


def test_revoke_removes_grant() -> None:
    policy = AccessPolicy.from_rules([Grant(ALICE), Grant(BOB)]).revoke(ALICE)
    assert not policy.admits(_perms(ALICE, EVERYONE))
    assert policy.admits(_perms(BOB, EVERYONE))


def test_with_deny_returns_new_policy_and_leaves_original_untouched() -> None:
    original = AccessPolicy.from_rules([Grant(ALICE)])
    denied = original.with_deny(ALICE)
    assert original.admits(_perms(ALICE, EVERYONE))
    assert not denied.admits(_perms(ALICE, EVERYONE))


def test_policy_is_hashable() -> None:
    a = AccessPolicy.from_rules([Grant(ALICE)])
    b = AccessPolicy.from_rules([Grant(ALICE)])
    assert hash(a) == hash(b)
    assert a == b


def test_permission_set_resolve_matches_effective_principals() -> None:
    graph = GroupGraph.from_edges({"eng": ["staff"]})
    caller = Principal(id=ALICE, groups=frozenset({ENG}))
    perms = PermissionSet.resolve(caller, graph)
    assert set(perms) == {ALICE, ENG, GroupId(PrincipalId("staff")), EVERYONE}
    assert ALICE in perms
    assert not perms.is_empty
    assert len(perms) == 4


def test_permission_set_empty_is_empty() -> None:
    assert PermissionSet.empty().is_empty
    assert len(PermissionSet.empty()) == 0


def test_permission_set_restrict_to_is_never_larger() -> None:
    """Used by INV-5: a stale entry may only shrink, never grow."""
    full = _perms(ALICE, ENG, EVERYONE)
    smaller = full.without(ENG)
    assert set(smaller) == {ALICE, EVERYONE}
    assert smaller.is_subset_of(full)
    assert not full.is_subset_of(smaller)
