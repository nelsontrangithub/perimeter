"""Ingestion: chunk, embed, extract ACLs, write.

The ingestor depends on ports only. Per document it:

1. compares the store's fingerprint (content hash + policy, no text) with the
   incoming document and skips it when nothing changed;
2. splits the text into spans at paragraph, sentence, or whitespace boundaries;
3. writes the document and its chunks to the store (which replaces any earlier
   version) and removes the document's earlier rows from the index;
4. embeds chunk text in batches and adds ``IndexEntry`` rows carrying the
   chunk ID, the vector, and the document's policy. No text goes to the index.

ACL extraction happens in the connector, which produces the ``AccessPolicy`` on
the ``RawDocument``. A connector that cannot read a document's ACL must hand
over ``AccessPolicy.nobody()``; the ingestor never defaults to public.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from perimeter.core.acl import AccessPolicy
from perimeter.core.document import Chunk, Document, DocumentId, SourceRef
from perimeter.core.errors import InvalidDocumentError
from perimeter.core.ports import DocumentStore, EmbeddingModel, IndexEntry, VectorIndex

_PARAGRAPH = "\n\n"
_SENTENCE_ENDS = (". ", ".\n", "? ", "?\n", "! ", "!\n")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    max_chars: int = 1200
    overlap_chars: int = 150

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise InvalidDocumentError("max_chars must be positive")
        if not 0 <= self.overlap_chars < self.max_chars:
            raise InvalidDocumentError("overlap_chars must be in [0, max_chars)")


def _boundary(text: str, start: int, end: int) -> int:
    """Best cut position in (start, end]: after a paragraph, else a sentence, else whitespace."""
    floor = start + max(1, (end - start) // 4)
    cut = text.rfind(_PARAGRAPH, floor, end)
    if cut != -1:
        return cut + len(_PARAGRAPH)
    best = -1
    for marker in _SENTENCE_ENDS:
        pos = text.rfind(marker, floor, end)
        if pos != -1:
            best = max(best, pos + len(marker))
    if best != -1:
        return best
    for i in range(end - 1, floor, -1):
        if text[i].isspace():
            return i + 1
    return end


def split_text(text: str, config: ChunkingConfig) -> list[tuple[int, int]]:
    """Contiguous (start, end) spans covering ``text``; overlapping when configured."""
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + config.max_chars, n)
        if end < n:
            end = _boundary(text, pos, end)
        spans.append((pos, end))
        if end >= n:
            break
        next_pos = end - config.overlap_chars if config.overlap_chars else end
        pos = max(next_pos, pos + 1)
    return spans


@dataclass(frozen=True, slots=True)
class RawDocument:
    """What a connector produces: identity, source, the extracted policy, and text."""

    id: DocumentId
    source: SourceRef
    policy: AccessPolicy
    text: str = field(repr=False)

    def to_document(self) -> Document:
        return Document.create(id=self.id, source=self.source, policy=self.policy, text=self.text)


@dataclass(slots=True)
class IngestReport:
    documents: int = 0
    chunks: int = 0
    skipped_unchanged: int = 0


class Ingestor:
    def __init__(
        self,
        *,
        store: DocumentStore,
        index: VectorIndex,
        embedder: EmbeddingModel,
        chunking: ChunkingConfig | None = None,
        embed_batch: int = 96,
    ) -> None:
        self._store = store
        self._index = index
        self._embedder = embedder
        self._chunking = chunking or ChunkingConfig()
        self._embed_batch = max(1, embed_batch)

    def ingest(self, documents: Iterable[RawDocument]) -> IngestReport:
        report = IngestReport()
        pending: list[Chunk] = []
        for raw in documents:
            doc = raw.to_document()
            if self._store.fingerprint(doc.id) == doc.fingerprint:
                report.skipped_unchanged += 1
                continue
            chunks = self._chunk(doc)
            self._store.put(doc, chunks)
            self._index.remove_document(doc.id)
            report.documents += 1
            report.chunks += len(chunks)
            pending.extend(chunks)
            if len(pending) >= self._embed_batch:
                self._embed_and_index(pending)
                pending = []
        if pending:
            self._embed_and_index(pending)
        return report

    def _chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for ordinal, (start, end) in enumerate(split_text(doc.text, self._chunking)):
            if doc.text[start:end].strip():
                chunks.append(Chunk.from_document(doc, ordinal=ordinal, start=start, end=end))
        return chunks

    def _embed_and_index(self, chunks: list[Chunk]) -> None:
        vectors = self._embedder.embed_documents([c.text for c in chunks])
        self._index.add(
            IndexEntry(chunk_id=c.id, vector=v, policy=c.policy)
            for c, v in zip(chunks, vectors, strict=True)
        )
