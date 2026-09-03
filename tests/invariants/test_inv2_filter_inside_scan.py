"""INV-2: the candidate set entering the reranker is a strict subset of the caller's
permitted set. Filtering happens inside the index scan, never after.

This file holds the index-level half: the scan is only ever handed permitted rows,
and the permitted-row mask is exactly the core policy semantics. The pipeline-level
half (what the reranker receives) is added with the retrieval orchestrator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from perimeter.core.acl import AccessPolicy, PermissionSet
from perimeter.core.document import ChunkId
from perimeter.core.ports import IndexEntry, as_vector
from perimeter.core.principal import EVERYONE, PrincipalId
from perimeter.index.filtered_search import filtered_search
from perimeter.index.flat import FlatIndex
from perimeter.index.quantize import l2_normalize

pytestmark = pytest.mark.invariant

DIM = 32
_ids = st.sampled_from(
    [PrincipalId(p) for p in ("alice", "bob", "eng", "contractors")] + [EVERYONE]
)
_policies = st.builds(
    AccessPolicy, grants=st.frozensets(_ids, max_size=3), denies=st.frozensets(_ids, max_size=2)
)
_perms = st.frozensets(_ids, max_size=4).map(PermissionSet)


@pytest.fixture(scope="module")
def index_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("inv2")


@given(policies=st.lists(_policies, min_size=1, max_size=25), perms=_perms, seed=st.integers(0, 50))
@settings(max_examples=150, deadline=None)
def test_inv2_index_scores_only_permitted_rows(
    index_dir: Path, policies: list[AccessPolicy], perms: PermissionSet, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    vecs = l2_normalize(rng.standard_normal((len(policies), DIM)).astype(np.float32))
    idx = FlatIndex.open(index_dir / f"i{seed}-{len(policies)}", dimension=DIM)
    idx.add(IndexEntry(ChunkId(f"c{i}"), as_vector(vecs[i]), p) for i, p in enumerate(policies))
    idx.flush()

    scanned: list[np.ndarray] = []
    original = idx.scan_rows

    def spy(query: np.ndarray, rows: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scanned.append(rows.copy())
        return original(query, rows, k)

    idx.scan_rows = spy  # type: ignore[method-assign]
    hits = filtered_search(idx, as_vector(vecs[0]), perms, k=len(policies))

    admitted = {i for i, p in enumerate(policies) if p.admits(perms)}
    for rows in scanned:
        assert set(rows.tolist()) <= admitted, "scan received a row the caller is not permitted"
    assert {int(h.chunk_id[1:]) for h in hits} <= admitted
    if not admitted:
        assert scanned == [] and hits == []
    else:
        assert {int(h.chunk_id[1:]) for h in hits} == admitted, (
            "k is preserved over the permitted set"
        )


# --- pipeline-level half -----------------------------------------------------


def test_inv2_reranker_input_is_subset_of_permitted() -> None:
    """Every chunk the reranker sees is admitted by the caller's permission set, and
    the index was asked with that permission set (filtering requested inside the
    scan, not applied after)."""
    from perimeter.core.acl import Deny, Grant
    from perimeter.core.principal import GroupGraph, GroupId, Principal
    from perimeter.core.query import RetrievalRequest
    from perimeter.pipeline.retrieve import Retriever
    from tests.pipeline.fakes import SpyReranker, StaticResolver, build_corpus, raw

    eng = GroupId(PrincipalId("eng"))
    contractors = GroupId(PrincipalId("contractors"))
    docs = [
        raw("pub", "apples apples apples", AccessPolicy.public()),
        raw("eng", "apples apples apples eng", AccessPolicy.from_rules([Grant(eng)])),
        raw(
            "noc",
            "apples apples apples noc",
            AccessPolicy.from_rules([Grant(EVERYONE), Deny(contractors)]),
        ),
        raw("nobody", "apples apples apples nobody", AccessPolicy.nobody()),
    ]
    store, index, embedder = build_corpus(docs)
    reranker = SpyReranker()
    retriever = Retriever(
        resolver=StaticResolver(), embedder=embedder, index=index, store=store, reranker=reranker
    )

    for caller in (
        Principal(id=PrincipalId("alice")),
        Principal(id=PrincipalId("bob"), groups=frozenset({eng})),
        Principal(id=PrincipalId("carol"), groups=frozenset({contractors})),
    ):
        permitted = PermissionSet.resolve(caller, GroupGraph.empty())
        retriever.retrieve(RetrievalRequest(principal=caller, query="apples", k=10))
        assert index.search_calls[-1][0] == permitted
        for chunk in reranker.received[-1]:
            assert chunk.policy.admits(permitted), "reranker received an unpermitted chunk"
        assert reranker.received[-1], "reranker should have received permitted candidates"
