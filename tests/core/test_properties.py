"""Property-based tests for principal resolution and policy evaluation.

These state the algebraic laws the rest of Perimeter relies on. If one of them
fails, an invariant test somewhere above it will eventually fail too; this file
is where the failure is cheapest to understand.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from perimeter.core.acl import AccessPolicy, Deny, Grant, PermissionSet
from perimeter.core.principal import (
    EVERYONE,
    GroupGraph,
    GroupId,
    Principal,
    PrincipalId,
    effective_principals,
)

_NAMES = [f"g{i}" for i in range(8)]
_USERS = [f"u{i}" for i in range(4)]

group_ids = st.sampled_from(_NAMES).map(lambda s: GroupId(PrincipalId(s)))
user_ids = st.sampled_from(_USERS).map(PrincipalId)
any_principal = st.one_of(user_ids, group_ids, st.just(EVERYONE))

group_sets = st.frozensets(group_ids, max_size=5)
edge_maps = st.dictionaries(
    st.sampled_from(_NAMES), st.lists(st.sampled_from(_NAMES), max_size=3), max_size=8
)
graphs = edge_maps.map(GroupGraph.from_edges)
principals = st.builds(Principal, id=user_ids, groups=group_sets)

policies = st.builds(
    AccessPolicy,
    grants=st.frozensets(any_principal, max_size=5),
    denies=st.frozensets(any_principal, max_size=3),
)
permission_sets = st.frozensets(any_principal, max_size=6).map(PermissionSet)


# --- principal resolution -------------------------------------------------


@given(principals, graphs)
def test_effective_always_contains_self_and_everyone(p: Principal, g: GroupGraph) -> None:
    eff = effective_principals(p, g)
    assert p.id in eff
    assert EVERYONE in eff


@given(principals, graphs)
def test_effective_is_closed_under_parent_edges(p: Principal, g: GroupGraph) -> None:
    eff = effective_principals(p, g)
    for member in eff:
        if member == p.id or member == EVERYONE:
            continue
        assert g.parents_of(GroupId(member)) <= eff


@given(principals, graphs, group_ids, group_ids)
def test_adding_an_edge_never_shrinks_effective_set(
    p: Principal, g: GroupGraph, child: GroupId, parent: GroupId
) -> None:
    before = effective_principals(p, g)
    after = effective_principals(p, g.with_edge(child, parent))
    assert before <= after


@given(principals, graphs, group_ids)
def test_adding_a_direct_membership_never_shrinks_effective_set(
    p: Principal, g: GroupGraph, extra: GroupId
) -> None:
    before = effective_principals(p, g)
    after = effective_principals(Principal(p.id, p.groups | {extra}), g)
    assert before <= after
    assert extra in after


@given(principals, graphs)
def test_resolution_is_idempotent_over_its_own_output(p: Principal, g: GroupGraph) -> None:
    eff = effective_principals(p, g)
    groups_only = frozenset(GroupId(m) for m in eff if m not in (p.id, EVERYONE))
    again = effective_principals(Principal(p.id, groups_only), g)
    assert again == eff


# --- policy evaluation ----------------------------------------------------


@given(policies)
def test_empty_permission_set_is_never_admitted(policy: AccessPolicy) -> None:
    assert not policy.admits(PermissionSet.empty())


@given(policies, permission_sets)
def test_admission_characterisation(policy: AccessPolicy, perms: PermissionSet) -> None:
    expected = (
        not perms.is_empty
        and policy.denies.isdisjoint(perms.principals)
        and not policy.grants.isdisjoint(perms.principals)
    )
    assert policy.admits(perms) == expected


@given(policies, permission_sets)
def test_any_intersecting_deny_refuses(policy: AccessPolicy, perms: PermissionSet) -> None:
    if not policy.denies.isdisjoint(perms.principals):
        assert not policy.admits(perms)


@given(policies, permission_sets, any_principal)
def test_revoke_never_widens(policy: AccessPolicy, perms: PermissionSet, x: PrincipalId) -> None:
    if policy.revoke(x).admits(perms):
        assert policy.admits(perms)


@given(policies, permission_sets, any_principal)
def test_with_deny_never_widens(policy: AccessPolicy, perms: PermissionSet, x: PrincipalId) -> None:
    if policy.with_deny(x).admits(perms):
        assert policy.admits(perms)


@given(policies, permission_sets, any_principal)
def test_with_grant_never_narrows(
    policy: AccessPolicy, perms: PermissionSet, x: PrincipalId
) -> None:
    if policy.admits(perms):
        assert policy.with_grant(x).admits(perms)


@given(
    st.builds(AccessPolicy, grants=st.frozensets(any_principal, max_size=5)),
    permission_sets,
    any_principal,
)
@settings(max_examples=300)
def test_grant_only_policies_are_monotone_in_permission_set(
    policy: AccessPolicy, perms: PermissionSet, dropped: PrincipalId
) -> None:
    """For grant-only policies, a smaller permission set is never more permissive.

    This is the law the ACL cache relies on (ADR-004, INV-5): a stale entry
    that is missing a newly added membership can only refuse, never admit.
    """
    smaller = perms.without(dropped)
    if policy.admits(smaller):
        assert policy.admits(perms)


def test_explicit_denies_break_monotonicity_known_and_documented() -> None:
    """Counterexample, kept on purpose.

    With an explicit deny, *removing* a principal from a permission set can turn
    a refusal into an admission. So a membership addition is a revocation from
    the point of view of every document that denies that group, and it must be
    signalled to the ACL cache like any other revocation. ADR-004 records this;
    the caching resolver's invalidation hook covers both directions.
    """
    contractors = GroupId(PrincipalId("contractors"))
    alice = PrincipalId("alice")
    policy = AccessPolicy.from_rules([Grant(alice), Deny(contractors)])
    full = PermissionSet.of(alice, contractors, EVERYONE)
    assert not policy.admits(full)
    assert policy.admits(full.without(contractors))


@given(policies, permission_sets)
def test_explain_agrees_with_admits(policy: AccessPolicy, perms: PermissionSet) -> None:
    decision = policy.explain(perms)
    assert decision.admitted == policy.admits(perms)
    assert decision.matched_grants <= policy.grants
    assert decision.matched_denies <= policy.denies
