"""Allow-list filtering inside the scan."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from perimeter.core.acl import AccessPolicy, Deny, Grant, PermissionSet
from perimeter.core.document import ChunkId
from perimeter.core.ports import IndexEntry, VectorIndex, as_vector
from perimeter.core.principal import EVERYONE, PrincipalId
from perimeter.index.filtered_search import filtered_search, permitted_rows
from perimeter.index.flat import AclTable, FlatIndex
from perimeter.index.quantize import l2_normalize

DIM = 32
ALICE = PrincipalId("alice")
BOB = PrincipalId("bob")
ENG = PrincipalId("eng")
CONTRACTORS = PrincipalId("contractors")

_ids = st.sampled_from([ALICE, BOB, ENG, CONTRACTORS, EVERYONE])
_policies = st.builds(
    AccessPolicy, grants=st.frozensets(_ids, max_size=3), denies=st.frozensets(_ids, max_size=2)
)
_perms = st.frozensets(_ids, max_size=4).map(PermissionSet)


def _vecs(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal((n, DIM)).astype(np.float32))


def _index(tmp_path: Path, policies: list[AccessPolicy], seed: int = 0) -> FlatIndex:
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    vecs = _vecs(len(policies), seed)
    idx.add(
        IndexEntry(ChunkId(f"c{i}"), as_vector(vecs[i]), policies[i]) for i in range(len(policies))
    )
    idx.flush()
    return idx


@given(st.lists(_policies, max_size=12), _perms)
@settings(max_examples=200)
def test_permitted_rows_agrees_exactly_with_core_policy_semantics(
    policies: list[AccessPolicy], perms: PermissionSet
) -> None:
    """The vectorised mask and AccessPolicy.admits must be the same function."""
    table = AclTable.build(policies)
    expected = {r for r, p in enumerate(policies) if p.admits(perms)}
    assert set(permitted_rows(table, perms).tolist()) == expected


def test_permitted_rows_is_sorted_int32() -> None:
    table = AclTable.build([AccessPolicy.public()] * 5)
    rows = permitted_rows(table, PermissionSet.of(ALICE, EVERYONE))
    assert rows.dtype == np.int32
    assert rows.tolist() == [0, 1, 2, 3, 4]


def test_permitted_rows_empty_permission_set_is_empty_even_for_public_rows() -> None:
    table = AclTable.build([AccessPolicy.public()] * 3)
    assert permitted_rows(table, PermissionSet.empty()).shape == (0,)


def test_permitted_rows_unknown_principals_match_nothing() -> None:
    table = AclTable.build([AccessPolicy.from_rules([Grant(ALICE)])])
    assert permitted_rows(table, PermissionSet.of(BOB)).shape == (0,)


def test_filtered_search_returns_only_admitted_hits_and_preserves_k(tmp_path: Path) -> None:
    policies = [
        AccessPolicy.from_rules([Grant(ALICE)]) if i % 2 else AccessPolicy.nobody()
        for i in range(40)
    ]
    idx = _index(tmp_path, policies)
    query = as_vector(_vecs(40)[0])
    hits = filtered_search(idx, query, PermissionSet.of(ALICE, EVERYONE), k=10)
    assert len(hits) == 10
    for h in hits:
        assert idx.policy_at(idx.row_of(h.chunk_id)).admits(PermissionSet.of(ALICE, EVERYONE))
        assert int(h.chunk_id[1:]) % 2 == 1


def test_filtered_search_ranks_best_permitted_first(tmp_path: Path) -> None:
    policies = [AccessPolicy.public()] * 50
    idx = _index(tmp_path, policies, seed=4)
    vecs = _vecs(50, seed=4)
    hits = filtered_search(idx, as_vector(vecs[17]), PermissionSet.of(BOB, EVERYONE), k=3)
    assert hits[0].chunk_id == ChunkId("c17")
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_filtered_search_scan_never_receives_unpermitted_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policies = [AccessPolicy.public() if i < 5 else AccessPolicy.nobody() for i in range(10)]
    idx = _index(tmp_path, policies)
    seen: list[np.ndarray] = []
    original = idx.scan_rows

    def spy(query: np.ndarray, rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        seen.append(rows.copy())
        return original(query, rows, k)

    monkeypatch.setattr(idx, "scan_rows", spy)
    filtered_search(idx, as_vector(_vecs(10)[0]), PermissionSet.of(ALICE, EVERYONE), k=10)
    assert len(seen) == 1
    assert seen[0].tolist() == [0, 1, 2, 3, 4]


def test_filtered_search_empty_permission_set_never_touches_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    idx = _index(tmp_path, [AccessPolicy.public()] * 10)
    calls = 0

    def spy(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        raise AssertionError("scan must not run for an empty permission set")

    monkeypatch.setattr(idx, "scan_rows", spy)
    assert filtered_search(idx, as_vector(_vecs(10)[0]), PermissionSet.empty(), k=5) == []
    assert calls == 0


def test_deny_excludes_row_inside_scan(tmp_path: Path) -> None:
    policies = [AccessPolicy.from_rules([Grant(EVERYONE), Deny(CONTRACTORS)])] * 5
    idx = _index(tmp_path, policies)
    perms = PermissionSet.of(ALICE, CONTRACTORS, EVERYONE)
    assert filtered_search(idx, as_vector(_vecs(5)[0]), perms, k=5) == []


def test_flat_index_search_satisfies_vector_index_port(tmp_path: Path) -> None:
    idx = _index(tmp_path, [AccessPolicy.public()] * 3)
    assert isinstance(idx, VectorIndex)
    hits = idx.search(as_vector(_vecs(3)[1]), PermissionSet.of(ALICE, EVERYONE), k=2)
    assert hits[0].chunk_id == ChunkId("c1")


def test_search_on_empty_index_returns_empty(tmp_path: Path) -> None:
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    assert idx.search(as_vector([1.0] * DIM), PermissionSet.of(ALICE, EVERYONE), k=3) == []
