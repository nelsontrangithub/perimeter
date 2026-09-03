"""Documents, chunks, and source references.

A :class:`Document` is the unit of access control: it carries an
:class:`~perimeter.core.acl.AccessPolicy`. A :class:`Chunk` is the unit of
retrieval: a span of a document's text with the document's policy and source
denormalised onto it, so the store can check permission and the pipeline can
build a citation without a second lookup.

``repr`` of these types never includes text. Text must not reach a log line or
an error message through an accidental f-string.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import NewType

from perimeter.core.acl import AccessPolicy
from perimeter.core.errors import InvalidDocumentError

DocumentId = NewType("DocumentId", str)
ChunkId = NewType("ChunkId", str)

_MAX_ID_LENGTH = 512


def parse_document_id(raw: str) -> DocumentId:
    if not raw or raw != raw.strip() or len(raw) > _MAX_ID_LENGTH:
        raise InvalidDocumentError("document id must be non-empty, trimmed, and short")
    if any(ch.isspace() or ord(ch) < 0x20 for ch in raw):
        raise InvalidDocumentError("document id contains whitespace or control characters")
    return DocumentId(raw)


def chunk_id_for(document_id: DocumentId, ordinal: int) -> ChunkId:
    """Deterministic chunk ID: re-ingesting the same document yields the same IDs."""
    return ChunkId(f"{document_id}#{ordinal}")


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Where a document came from, for citations and re-fetching."""

    connector: str
    uri: str
    title: str
    version: str | None = None


@dataclass(frozen=True, slots=True)
class Document:
    id: DocumentId
    source: SourceRef
    policy: AccessPolicy
    text: str = field(repr=False)
    content_hash: str

    @classmethod
    def create(
        cls, *, id: DocumentId, source: SourceRef, policy: AccessPolicy, text: str
    ) -> Document:
        parse_document_id(id)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(id=id, source=source, policy=policy, text=text, content_hash=digest)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: ChunkId
    document_id: DocumentId
    ordinal: int
    start: int
    end: int
    text: str = field(repr=False)
    policy: AccessPolicy
    source: SourceRef

    @classmethod
    def from_document(cls, document: Document, *, ordinal: int, start: int, end: int) -> Chunk:
        if ordinal < 0:
            raise InvalidDocumentError("chunk ordinal must be non-negative")
        if start < 0 or end > len(document.text) or start >= end:
            raise InvalidDocumentError("chunk span must satisfy 0 <= start < end <= len(text)")
        return cls(
            id=chunk_id_for(document.id, ordinal),
            document_id=document.id,
            ordinal=ordinal,
            start=start,
            end=end,
            text=document.text[start:end],
            policy=document.policy,
            source=document.source,
        )
