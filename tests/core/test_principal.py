"""Unit tests for principal identity and effective-principal resolution. No I/O."""

from __future__ import annotations

import pytest

from perimeter.core.errors import InvalidPrincipalError
from perimeter.core.principal import (
    EVERYONE,
    GroupGraph,
    GroupId,
    Principal,
    PrincipalId,
    effective_principals,
    parse_group_id,
    parse_principal_id,
)


def _g(name: str) -> GroupId:
    return GroupId(PrincipalId(name))


def _p(user: str, *groups: str) -> Principal:
    return Principal(id=PrincipalId(user), groups=frozenset(_g(g) for g in groups))


def test_effective_set_contains_own_id() -> None:
    eff = effective_principals(_p("alice"), GroupGraph.empty())
    assert PrincipalId("alice") in eff


def test_effective_set_contains_everyone() -> None:
    eff = effective_principals(_p("alice"), GroupGraph.empty())
    assert EVERYONE in eff


def test_effective_set_contains_direct_groups() -> None:
    eff = effective_principals(_p("alice", "eng"), GroupGraph.empty())
    assert _g("eng") in eff


def test_effective_set_contains_transitive_parent_groups() -> None:
    graph = GroupGraph.from_edges({"eng": ["staff"], "staff": ["all-hands"]})
    eff = effective_principals(_p("alice", "eng"), graph)
    assert {_g("eng"), _g("staff"), _g("all-hands")} <= eff


def test_group_absent_from_graph_is_still_a_leaf_membership() -> None:
    graph = GroupGraph.from_edges({"eng": ["staff"]})
    eff = effective_principals(_p("alice", "sales"), graph)
    assert _g("sales") in eff
    assert _g("staff") not in eff


def test_cycle_in_group_graph_terminates_and_includes_cycle_members() -> None:
    graph = GroupGraph.from_edges({"a": ["b"], "b": ["c"], "c": ["a"]})
    eff = effective_principals(_p("alice", "a"), graph)
    assert {_g("a"), _g("b"), _g("c")} <= eff


def test_effective_set_is_exactly_id_groups_closure_and_everyone() -> None:
    graph = GroupGraph.from_edges({"eng": ["staff"]})
    eff = effective_principals(_p("alice", "eng"), graph)
    assert eff == frozenset({PrincipalId("alice"), _g("eng"), _g("staff"), EVERYONE})


def test_result_is_immutable_frozenset() -> None:
    eff = effective_principals(_p("alice"), GroupGraph.empty())
    assert isinstance(eff, frozenset)


@pytest.mark.parametrize("raw", ["", "   ", "a\nb", "x" * 257, "tab\there"])
def test_parse_principal_id_rejects_malformed(raw: str) -> None:
    with pytest.raises(InvalidPrincipalError):
        parse_principal_id(raw)


def test_parse_principal_id_accepts_email_like_and_opaque_ids() -> None:
    assert parse_principal_id("alice@example.com") == PrincipalId("alice@example.com")
    assert parse_principal_id("user:1234") == PrincipalId("user:1234")


def test_parse_group_id_rejects_malformed() -> None:
    with pytest.raises(InvalidPrincipalError):
        parse_group_id("")


def test_parse_principal_id_rejects_reserved_everyone() -> None:
    """Callers cannot claim to *be* the public principal; it is added by resolution."""
    with pytest.raises(InvalidPrincipalError):
        parse_principal_id(str(EVERYONE))


def test_invalid_principal_error_is_typed_domain_error() -> None:
    from perimeter.core.errors import PerimeterError

    assert issubclass(InvalidPrincipalError, PerimeterError)


def test_group_graph_parents_of_unknown_group_is_empty() -> None:
    graph = GroupGraph.from_edges({"eng": ["staff"]})
    assert graph.parents_of(_g("nope")) == frozenset()
