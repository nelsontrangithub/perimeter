"""INV-1, universally: for every generated group graph, corpus of policies, and caller,
retrieval returns only chunks the policy admits, and (with k large enough) all of them.

Two configurations run the same property:

* the real stack: MemoryStore + on-disk FlatIndex + Retriever;
* the real stack with a broken index that ignores permissions, proving the store and
  orchestrator hold INV-1 without the index's help.

The completeness half (returned == admitted when k covers the corpus) matters: an
implementation that returned nothing would satisfy the leak half trivially.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from perimeter.adapters.memory_store import MemoryStore
from perimeter.core.acl import AccessPolicy, PermissionSet
from perimeter.core.principal import EVERYONE, GroupGraph, GroupId, Principal, PrincipalId
from perimeter.core.query import RetrievalRequest
from perimeter.index.flat import FlatIndex
from perimeter.pipeline.ingest import ChunkingConfig, Ingestor
from perimeter.pipeline.retrieve import Retriever
from tests.pipeline.fakes import DIM, BagEmbedder, LeakyIndex, StaticResolver, raw

pytestmark = pytest.mark.invariant

_GROUPS = [f"g{i}" for i in range(5)]
_USERS = [f"u{i}" for i in range(3)]
group_ids = st.sampled_from(_GROUPS).map(lambda s: GroupId(PrincipalId(s)))
user_ids = st.sampled_from(_USERS).map(PrincipalId)
any_principal = st.one_of(user_ids, group_ids, st.just(EVERYONE))

graphs = st.dictionaries(
    st.sampled_from(_GROUPS), st.lists(st.sampled_from(_GROUPS), max_size=2), max_size=5
).map(GroupGraph.from_edges)
policies = st.builds(
    AccessPolicy,
    grants=st.frozensets(any_principal, max_size=3),
    denies=st.frozensets(any_principal, max_size=2),
)
callers = st.builds(Principal, id=user_ids, groups=st.frozensets(group_ids, max_size=3))

_counter = itertools.count()


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("leak")


def _run(
    base_dir: Path,
    graph: GroupGraph,
    doc_policies: list[AccessPolicy],
    caller: Principal,
    *,
    leaky: bool,
) -> None:
    docs = [raw(f"d{i}", f"MARKER_D{i} apples pears plums", p) for i, p in enumerate(doc_policies)]
    store, embedder = MemoryStore(), BagEmbedder()
    index = (
        LeakyIndex() if leaky else FlatIndex.open(base_dir / f"idx-{next(_counter)}", dimension=DIM)
    )
    Ingestor(
        store=store,
        index=index,
        embedder=embedder,
        chunking=ChunkingConfig(max_chars=200, overlap_chars=0),
    ).ingest(docs)
    retriever = Retriever(
        resolver=StaticResolver(graph), embedder=embedder, index=index, store=store, reranker=None
    )

    permitted = PermissionSet.resolve(caller, graph)
    admitted = {d.id for d in docs if d.policy.admits(permitted)}
    forbidden = {d.id for d in docs} - admitted

    result = retriever.retrieve(
        RetrievalRequest(principal=caller, query="apples pears", k=len(docs))
    )

    returned = {sc.citation.document_id for sc in result.chunks}
    assert returned <= admitted, f"leak: returned {returned - admitted}"
    assert returned == admitted, f"over-filtered: missing {admitted - returned}"
    text = " ".join(sc.text for sc in result.chunks)
    for doc_id in forbidden:
        assert f"MARKER_{doc_id.upper()} " not in text + " "
    for sc in result.chunks:
        assert sc.chunk.policy.admits(permitted)
    if not admitted:
        assert result.is_empty


@given(graph=graphs, doc_policies=st.lists(policies, min_size=1, max_size=8), caller=callers)
@settings(max_examples=150, deadline=None)
def test_inv1_holds_for_all_policy_graphs(
    base_dir: Path, graph: GroupGraph, doc_policies: list[AccessPolicy], caller: Principal
) -> None:
    _run(base_dir, graph, doc_policies, caller, leaky=False)


@given(graph=graphs, doc_policies=st.lists(policies, min_size=1, max_size=8), caller=callers)
@settings(max_examples=150, deadline=None)
def test_inv1_holds_even_with_a_leaky_index(
    base_dir: Path, graph: GroupGraph, doc_policies: list[AccessPolicy], caller: Principal
) -> None:
    _run(base_dir, graph, doc_policies, caller, leaky=True)
