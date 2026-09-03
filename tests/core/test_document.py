"""Unit tests for the document, chunk, and source-reference model. No I/O."""

from __future__ import annotations

import pytest

from perimeter.core.acl import AccessPolicy, Grant
from perimeter.core.document import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    SourceRef,
    chunk_id_for,
    parse_document_id,
)
from perimeter.core.errors import InvalidDocumentError
from perimeter.core.principal import PrincipalId

SRC = SourceRef(connector="filesystem", uri="file:///docs/a.md", title="A")
POLICY = AccessPolicy.from_rules([Grant(PrincipalId("alice"))])


def test_document_create_computes_stable_content_hash() -> None:
    a = Document.create(id=DocumentId("d1"), source=SRC, policy=POLICY, text="hello")
    b = Document.create(id=DocumentId("d1"), source=SRC, policy=POLICY, text="hello")
    c = Document.create(id=DocumentId("d1"), source=SRC, policy=POLICY, text="hello!")
    assert a.content_hash == b.content_hash
    assert a.content_hash != c.content_hash
    assert len(a.content_hash) == 64  # sha256 hex


def test_document_create_rejects_empty_id() -> None:
    with pytest.raises(InvalidDocumentError):
        Document.create(id=DocumentId(""), source=SRC, policy=POLICY, text="x")


def test_chunk_id_is_deterministic_and_distinct_per_ordinal() -> None:
    assert chunk_id_for(DocumentId("d1"), 0) == chunk_id_for(DocumentId("d1"), 0)
    assert chunk_id_for(DocumentId("d1"), 0) != chunk_id_for(DocumentId("d1"), 1)
    assert chunk_id_for(DocumentId("d1"), 0) != chunk_id_for(DocumentId("d2"), 0)


def test_chunk_from_document_slices_text_and_inherits_policy_and_source() -> None:
    doc = Document.create(id=DocumentId("d1"), source=SRC, policy=POLICY, text="hello world")
    chunk = Chunk.from_document(doc, ordinal=1, start=6, end=11)
    assert chunk.text == "world"
    assert chunk.document_id == doc.id
    assert chunk.policy == POLICY
    assert chunk.source == SRC
    assert chunk.ordinal == 1
    assert chunk.id == chunk_id_for(doc.id, 1)


@pytest.mark.parametrize(("start", "end"), [(5, 5), (6, 3), (-1, 4), (0, 12), (11, 12)])
def test_chunk_from_document_rejects_bad_span(start: int, end: int) -> None:
    doc = Document.create(id=DocumentId("d1"), source=SRC, policy=POLICY, text="hello world")
    with pytest.raises(InvalidDocumentError):
        Chunk.from_document(doc, ordinal=0, start=start, end=end)


def test_chunk_from_document_rejects_negative_ordinal() -> None:
    doc = Document.create(id=DocumentId("d1"), source=SRC, policy=POLICY, text="hello")
    with pytest.raises(InvalidDocumentError):
        Chunk.from_document(doc, ordinal=-1, start=0, end=5)


def test_parse_document_id_validates() -> None:
    assert parse_document_id("gdrive:abc") == DocumentId("gdrive:abc")
    with pytest.raises(InvalidDocumentError):
        parse_document_id(" spaced ")
    with pytest.raises(InvalidDocumentError):
        parse_document_id("")


def test_chunk_id_is_a_str_newtype_usable_as_dict_key() -> None:
    cid: ChunkId = chunk_id_for(DocumentId("d1"), 0)
    assert {cid: 1}[cid] == 1


def test_source_ref_optional_version() -> None:
    assert SRC.version is None
    versioned = SourceRef(connector="gdrive", uri="https://x", title="T", version="etag-1")
    assert versioned.version == "etag-1"


def test_document_repr_does_not_include_text() -> None:
    """Document text must not leak through repr into logs or error messages."""
    doc = Document.create(id=DocumentId("d1"), source=SRC, policy=POLICY, text="SECRET BODY")
    assert "SECRET BODY" not in repr(doc)
    chunk = Chunk.from_document(doc, ordinal=0, start=0, end=6)
    assert "SECRET" not in repr(chunk)
