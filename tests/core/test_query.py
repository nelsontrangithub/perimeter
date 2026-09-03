"""Unit tests for retrieval request/result types. No I/O."""

from __future__ import annotations

import pytest

from perimeter.core.acl import AccessPolicy, Grant
from perimeter.core.document import Chunk, Document, DocumentId, SourceRef
from perimeter.core.errors import InvalidRequestError
from perimeter.core.principal import Principal, PrincipalId
from perimeter.core.query import Citation, RetrievalRequest, ScopedResult, ScoredChunk

ALICE = Principal(id=PrincipalId("alice"))
SRC = SourceRef(connector="filesystem", uri="file:///a.md", title="A", version="v1")
DOC = Document.create(
    id=DocumentId("d1"),
    source=SRC,
    policy=AccessPolicy.from_rules([Grant(PrincipalId("alice"))]),
    text="alpha beta gamma",
)


def test_request_defaults_and_validation() -> None:
    req = RetrievalRequest(principal=ALICE, query="alpha", k=5)
    assert req.k == 5
    assert req.candidate_multiplier >= 1


@pytest.mark.parametrize("k", [0, -1, 1001])
def test_request_rejects_out_of_range_k(k: int) -> None:
    with pytest.raises(InvalidRequestError):
        RetrievalRequest(principal=ALICE, query="alpha", k=k)


@pytest.mark.parametrize("query", ["", "   "])
def test_request_rejects_blank_query(query: str) -> None:
    with pytest.raises(InvalidRequestError):
        RetrievalRequest(principal=ALICE, query=query, k=3)


def test_request_rejects_oversized_query() -> None:
    with pytest.raises(InvalidRequestError):
        RetrievalRequest(principal=ALICE, query="x" * 20_001, k=3)


def test_request_repr_omits_query_text() -> None:
    req = RetrievalRequest(principal=ALICE, query="SENSITIVE QUERY", k=3)
    assert "SENSITIVE" not in repr(req)


def test_citation_from_chunk_carries_source_and_span() -> None:
    chunk = Chunk.from_document(DOC, ordinal=0, start=0, end=5)
    cite = Citation.from_chunk(chunk)
    assert cite.chunk_id == chunk.id
    assert cite.document_id == DOC.id
    assert cite.source == SRC
    assert (cite.start, cite.end) == (0, 5)


def test_scored_chunk_builds_its_citation() -> None:
    chunk = Chunk.from_document(DOC, ordinal=0, start=0, end=5)
    scored = ScoredChunk.of(chunk, score=0.9)
    assert scored.citation == Citation.from_chunk(chunk)
    assert scored.text == "alpha"


def test_scoped_result_empty_reports_requested_k() -> None:
    result = ScopedResult.empty(principal=ALICE, k=7)
    assert result.chunks == ()
    assert result.requested_k == 7
    assert result.returned == 0
    assert result.is_empty


def test_scoped_result_returned_counts_chunks() -> None:
    chunk = Chunk.from_document(DOC, ordinal=0, start=0, end=5)
    result = ScopedResult(principal=ALICE, requested_k=3, chunks=(ScoredChunk.of(chunk, 1.0),))
    assert result.returned == 1
    assert not result.is_empty
