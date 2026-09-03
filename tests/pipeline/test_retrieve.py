"""Retrieval orchestrator: resolve -> filtered search -> fetch -> rerank -> assemble."""

from __future__ import annotations

from perimeter.core.acl import AccessPolicy, Grant
from perimeter.core.principal import GroupGraph, GroupId, Principal, PrincipalId
from perimeter.core.query import RetrievalRequest, ScopedResult
from perimeter.pipeline.retrieve import Retriever
from tests.pipeline.fakes import SpyReranker, StaticResolver, build_corpus, raw

ALICE = Principal(id=PrincipalId("alice"))
BOB = Principal(id=PrincipalId("bob"), groups=frozenset({GroupId(PrincipalId("eng"))}))
ENG_ONLY = AccessPolicy.from_rules([Grant(PrincipalId("eng"))])

DOCS = [
    raw("pub", "apples pears apples pears public", AccessPolicy.public()),
    raw("eng", "apples pears engineering only apples", ENG_ONLY),
    raw(
        "alice",
        "apples for alice alone apples",
        AccessPolicy.from_rules([Grant(PrincipalId("alice"))]),
    ),
]


def _retriever(
    reranker: SpyReranker | None = None, resolver: StaticResolver | None = None
) -> tuple[Retriever, SpyReranker | None]:
    store, index, embedder = build_corpus(DOCS)
    r = Retriever(
        resolver=resolver or StaticResolver(GroupGraph.empty()),
        embedder=embedder,
        index=index,
        store=store,
        reranker=reranker,
    )
    return r, reranker


def test_returns_scoped_result_with_citations_and_text() -> None:
    retriever, _ = _retriever()
    result = retriever.retrieve(RetrievalRequest(principal=ALICE, query="apples", k=5))
    assert isinstance(result, ScopedResult)
    assert result.requested_k == 5
    assert result.returned >= 1
    for sc in result.chunks:
        assert sc.text
        assert sc.citation.chunk_id == sc.chunk.id
        assert sc.citation.source.uri.startswith("file:///")


def test_only_permitted_documents_are_returned() -> None:
    retriever, _ = _retriever()
    for principal, allowed in ((ALICE, {"pub", "alice"}), (BOB, {"pub", "eng"})):
        result = retriever.retrieve(
            RetrievalRequest(principal=principal, query="apples pears", k=10)
        )
        assert {sc.citation.document_id for sc in result.chunks} <= allowed
        assert result.returned > 0


def test_k_is_preserved_when_enough_permitted_chunks_exist() -> None:
    docs = [raw(f"d{i}", f"apples number {i} " * 3, AccessPolicy.public()) for i in range(30)]
    store, index, embedder = build_corpus(docs, max_chars=200)
    retriever = Retriever(
        resolver=StaticResolver(), embedder=embedder, index=index, store=store, reranker=None
    )
    result = retriever.retrieve(RetrievalRequest(principal=ALICE, query="apples", k=10))
    assert result.returned == 10


def test_reranker_order_is_used_when_present() -> None:
    reranker = SpyReranker()
    retriever, _ = _retriever(reranker=reranker)
    result = retriever.retrieve(RetrievalRequest(principal=ALICE, query="apples", k=3))
    received_ids = [c.id for c in reranker.received[0]]
    returned_ids = [sc.chunk.id for sc in result.chunks]
    assert returned_ids == list(reversed(received_ids))[:3]
    assert [sc.score for sc in result.chunks] == sorted(
        (sc.score for sc in result.chunks), reverse=True
    )


def test_without_reranker_index_order_and_scores_are_used() -> None:
    retriever, _ = _retriever(reranker=None)
    result = retriever.retrieve(RetrievalRequest(principal=ALICE, query="apples", k=3))
    scores = [sc.score for sc in result.chunks]
    assert scores == sorted(scores, reverse=True)


def test_index_receives_candidate_k_and_permission_set() -> None:
    store, index, embedder = build_corpus(DOCS)
    retriever = Retriever(
        resolver=StaticResolver(), embedder=embedder, index=index, store=store, reranker=None
    )
    retriever.retrieve(RetrievalRequest(principal=BOB, query="apples", k=2, candidate_multiplier=5))
    permitted, k = index.search_calls[0]
    assert k == 10
    assert PrincipalId("eng") in permitted and PrincipalId("bob") in permitted


def test_resolver_error_yields_empty_result_and_no_index_call() -> None:
    store, index, embedder = build_corpus(DOCS)
    retriever = Retriever(
        resolver=StaticResolver(fail=True),
        embedder=embedder,
        index=index,
        store=store,
        reranker=None,
    )
    result = retriever.retrieve(RetrievalRequest(principal=ALICE, query="apples", k=3))
    assert result.is_empty
    assert index.search_calls == []
    assert embedder.query_calls == 0


def test_result_reports_candidate_count() -> None:
    retriever, _ = _retriever()
    result = retriever.retrieve(RetrievalRequest(principal=ALICE, query="apples", k=2))
    assert result.candidates >= result.returned
