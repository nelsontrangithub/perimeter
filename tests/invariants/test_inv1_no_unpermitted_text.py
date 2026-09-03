"""INV-1: no chunk text is ever returned for a chunk whose access policy does not
admit the caller.

The test deliberately uses a *leaky* index that ignores the permission set and
returns every chunk. If the pipeline and store enforce INV-1 on their own, the
leak stops there. If this test fails, two independent checks failed at once.
"""

from __future__ import annotations

import pytest

from perimeter.core.acl import AccessPolicy, Deny, Grant, PermissionSet
from perimeter.core.principal import EVERYONE, GroupGraph, GroupId, Principal, PrincipalId
from perimeter.core.query import RetrievalRequest
from perimeter.pipeline.retrieve import Retriever
from tests.pipeline.fakes import LeakyIndex, StaticResolver, build_corpus, raw

pytestmark = pytest.mark.invariant

ENG = GroupId(PrincipalId("eng"))
CONTRACTORS = GroupId(PrincipalId("contractors"))
DOCS = [
    raw("pub", "MARKER_PUB apples apples apples", AccessPolicy.public()),
    raw("eng", "MARKER_ENG apples apples apples", AccessPolicy.from_rules([Grant(ENG)])),
    raw(
        "alice",
        "MARKER_ALICE apples apples apples",
        AccessPolicy.from_rules([Grant(PrincipalId("alice"))]),
    ),
    raw(
        "noc",
        "MARKER_NOC apples apples apples",
        AccessPolicy.from_rules([Grant(EVERYONE), Deny(CONTRACTORS)]),
    ),
    raw("nobody", "MARKER_NOBODY apples apples apples", AccessPolicy.nobody()),
]
CALLERS = [
    Principal(id=PrincipalId("alice")),
    Principal(id=PrincipalId("bob"), groups=frozenset({ENG})),
    Principal(id=PrincipalId("carol"), groups=frozenset({CONTRACTORS})),
    Principal(id=PrincipalId("dave"), groups=frozenset({ENG, CONTRACTORS})),
]


@pytest.mark.parametrize("caller", CALLERS, ids=lambda p: str(p.id))
def test_inv1_no_chunk_text_for_unpermitted_caller(caller: Principal) -> None:
    store, index, embedder = build_corpus(DOCS, index=LeakyIndex())
    retriever = Retriever(
        resolver=StaticResolver(), embedder=embedder, index=index, store=store, reranker=None
    )
    permitted = PermissionSet.resolve(caller, GroupGraph.empty())

    result = retriever.retrieve(RetrievalRequest(principal=caller, query="apples", k=50))

    admitted_docs = {d.id for d in DOCS if d.policy.admits(permitted)}
    forbidden_docs = {d.id for d in DOCS} - admitted_docs
    assert admitted_docs, "test setup: every caller should see something"
    returned_docs = {sc.citation.document_id for sc in result.chunks}
    assert returned_docs <= admitted_docs
    assert returned_docs == admitted_docs, "k is large enough to see every permitted doc"
    all_text = " ".join(sc.text for sc in result.chunks)
    for doc_id in forbidden_docs:
        assert f"MARKER_{doc_id.upper()}" not in all_text
    for sc in result.chunks:
        assert sc.chunk.policy.admits(permitted)
