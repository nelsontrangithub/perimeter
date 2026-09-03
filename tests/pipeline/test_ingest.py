"""Ingestion: chunk, embed, extract ACLs, write to store and index."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path

import pytest

from perimeter.adapters.memory_store import MemoryStore
from perimeter.core.acl import AccessPolicy, Grant, PermissionSet
from perimeter.core.document import Document, DocumentId, SourceRef
from perimeter.core.errors import InvalidDocumentError
from perimeter.core.ports import IndexEntry, IndexHit, Vector, as_vector
from perimeter.core.principal import EVERYONE, PrincipalId
from perimeter.index.flat import FlatIndex
from perimeter.pipeline.ingest import (
    ChunkingConfig,
    Ingestor,
    IngestReport,
    RawDocument,
    split_text,
)

DIM = 16
ALICE = PrincipalId("alice")
AS_ALICE = PermissionSet.of(ALICE, EVERYONE)


class FakeEmbedder:
    """Deterministic: vector is a bag of character codes, so similar text is close."""

    dimension = DIM

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        self.calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> Vector:
        v = [0.0] * DIM
        for ch in text:
            v[ord(ch) % DIM] += 1.0
        return as_vector(v)


class RecordingIndex:
    def __init__(self) -> None:
        self.entries: list[IndexEntry] = []

    @property
    def size(self) -> int:
        return len(self.entries)

    def add(self, entries: Iterable[IndexEntry]) -> None:
        self.entries.extend(entries)

    def remove_document(self, document_id: DocumentId) -> None:
        self.entries = [e for e in self.entries if not e.chunk_id.startswith(f"{document_id}#")]

    def search(self, query: Vector, permitted: PermissionSet, k: int) -> Sequence[IndexHit]:
        return []


# --- split_text ------------------------------------------------------------


def test_split_text_covers_whole_document_in_order() -> None:
    text = "para one.\n\npara two is a bit longer.\n\npara three."
    spans = split_text(text, ChunkingConfig(max_chars=20, overlap_chars=0))
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)
    for (s0, e0), (s1, _e1) in pairwise(spans):
        assert s1 == e0
        assert 0 < e0 - s0 <= 20


def test_split_text_prefers_paragraph_then_sentence_boundaries() -> None:
    text = "First sentence. Second sentence.\n\nNew paragraph here."
    spans = split_text(text, ChunkingConfig(max_chars=40, overlap_chars=0))
    pieces = [text[s:e] for s, e in spans]
    assert pieces[0].rstrip() == "First sentence. Second sentence."


def test_split_text_overlap_makes_windows_share_text() -> None:
    text = "a" * 100
    spans = split_text(text, ChunkingConfig(max_chars=40, overlap_chars=10))
    assert spans[0] == (0, 40)
    assert spans[1][0] == 30
    assert spans[-1][1] == 100


def test_split_text_empty_or_blank_yields_no_spans() -> None:
    assert split_text("", ChunkingConfig(max_chars=10, overlap_chars=0)) == []
    assert split_text("   \n\n ", ChunkingConfig(max_chars=10, overlap_chars=0)) == []


def test_chunking_config_validates() -> None:
    with pytest.raises(InvalidDocumentError):
        ChunkingConfig(max_chars=0, overlap_chars=0)
    with pytest.raises(InvalidDocumentError):
        ChunkingConfig(max_chars=10, overlap_chars=10)


# --- Ingestor --------------------------------------------------------------


def _raw(doc_id: str, text: str, policy: AccessPolicy) -> RawDocument:
    return RawDocument(
        id=DocumentId(doc_id),
        source=SourceRef("fs", f"file:///{doc_id}", doc_id),
        policy=policy,
        text=text,
    )


def test_ingest_writes_chunks_to_store_and_entries_to_index() -> None:
    store, index, embedder = MemoryStore(), RecordingIndex(), FakeEmbedder()
    ingestor = Ingestor(
        store=store,
        index=index,
        embedder=embedder,
        chunking=ChunkingConfig(max_chars=12, overlap_chars=0),
    )
    report = ingestor.ingest(
        [_raw("a", "hello world. this is text.", AccessPolicy.from_rules([Grant(ALICE)]))]
    )
    assert isinstance(report, IngestReport)
    assert report.documents == 1
    assert report.chunks == store.count_chunks() == index.size
    assert report.chunks >= 2
    for entry in index.entries:
        assert entry.policy == AccessPolicy.from_rules([Grant(ALICE)])
        assert len(entry.vector) == DIM
    ids = [e.chunk_id for e in index.entries]
    assert [c.id for c in store.get_chunks(ids, AS_ALICE)] == ids


def test_ingest_embeds_chunk_text_not_document_text() -> None:
    store, index, embedder = MemoryStore(), RecordingIndex(), FakeEmbedder()
    ingestor = Ingestor(
        store=store,
        index=index,
        embedder=embedder,
        chunking=ChunkingConfig(max_chars=6, overlap_chars=0),
    )
    ingestor.ingest([_raw("a", "abcdef ghijkl", AccessPolicy.public())])
    embedded = [t for call in embedder.calls for t in call]
    assert embedded == ["abcdef", " ghijk", "l"] or all(len(t) <= 6 for t in embedded)


def test_ingest_document_with_no_acl_is_written_as_nobody() -> None:
    """A document whose ACL could not be read must not become public by default."""
    store, index, embedder = MemoryStore(), RecordingIndex(), FakeEmbedder()
    ingestor = Ingestor(store=store, index=index, embedder=embedder)
    ingestor.ingest([_raw("a", "secret text", AccessPolicy.nobody())])
    assert store.get_chunks([e.chunk_id for e in index.entries], AS_ALICE) == []
    assert all(e.policy == AccessPolicy.nobody() for e in index.entries)


def test_ingest_index_entries_carry_no_text() -> None:
    store, index, embedder = MemoryStore(), RecordingIndex(), FakeEmbedder()
    Ingestor(store=store, index=index, embedder=embedder).ingest(
        [_raw("a", "PLAINTEXT MARKER", AccessPolicy.public())]
    )
    assert not any("PLAINTEXT" in repr(e) for e in index.entries)
    assert not hasattr(index.entries[0], "text")


def test_ingest_skips_unchanged_documents_by_content_hash() -> None:
    store, index, embedder = MemoryStore(), RecordingIndex(), FakeEmbedder()
    ingestor = Ingestor(store=store, index=index, embedder=embedder)
    doc = _raw("a", "same text", AccessPolicy.public())
    first = ingestor.ingest([doc])
    second = ingestor.ingest([doc])
    assert first.documents == 1 and second.documents == 0
    assert second.skipped_unchanged == 1
    assert len(embedder.calls) == 1


def test_ingest_reingests_when_policy_changes_even_if_text_unchanged() -> None:
    store, index, embedder = MemoryStore(), RecordingIndex(), FakeEmbedder()
    ingestor = Ingestor(store=store, index=index, embedder=embedder)
    ingestor.ingest([_raw("a", "same text", AccessPolicy.public())])
    report = ingestor.ingest([_raw("a", "same text", AccessPolicy.nobody())])
    assert report.documents == 1
    assert index.entries[-1].policy == AccessPolicy.nobody()
    assert store.get_document(DocumentId("a"), AS_ALICE) is None


def test_ingest_empty_document_writes_document_but_no_chunks() -> None:
    store, index, embedder = MemoryStore(), RecordingIndex(), FakeEmbedder()
    report = Ingestor(store=store, index=index, embedder=embedder).ingest(
        [_raw("empty", "   ", AccessPolicy.public())]
    )
    assert report.documents == 1 and report.chunks == 0
    assert store.count_documents() == 1


def test_ingest_end_to_end_with_real_flat_index(tmp_path: Path) -> None:
    store, embedder = MemoryStore(), FakeEmbedder()
    index = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    ingestor = Ingestor(
        store=store,
        index=index,
        embedder=embedder,
        chunking=ChunkingConfig(max_chars=30, overlap_chars=0),
    )
    ingestor.ingest(
        [
            _raw("pub", "public document about apples and pears", AccessPolicy.public()),
            _raw(
                "priv",
                "private document about apples and pears",
                AccessPolicy.from_rules([Grant(ALICE)]),
            ),
        ]
    )
    index.flush()
    hits = index.search(
        embedder.embed_query("apples"), PermissionSet.of(PrincipalId("bob"), EVERYONE), k=10
    )
    assert hits and all(h.chunk_id.startswith("pub#") for h in hits)


def test_document_from_raw_builds_hash(tmp_path: Path) -> None:
    raw = _raw("a", "text", AccessPolicy.public())
    doc = raw.to_document()
    assert isinstance(doc, Document)
    assert doc.content_hash


def test_source_failure_mid_stream_still_indexes_received_documents() -> None:
    from collections.abc import Iterator

    from perimeter.core.errors import ConnectorError

    store, index, embedder = MemoryStore(), RecordingIndex(), FakeEmbedder()

    def source() -> Iterator[RawDocument]:
        yield _raw("ok", "arrived fine", AccessPolicy.public())
        raise ConnectorError("upstream failed")

    with pytest.raises(ConnectorError):
        Ingestor(store=store, index=index, embedder=embedder).ingest(source())
    assert store.count_documents() == 1
    assert index.size >= 1, "stored documents must be indexed even when the source fails later"


def test_embedding_failure_removes_stored_documents_so_retry_can_reingest() -> None:
    from perimeter.core.errors import EmbeddingError

    class FailingEmbedder(FakeEmbedder):
        def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
            raise EmbeddingError("embed: HTTP 500")

    store, index = MemoryStore(), RecordingIndex()
    doc = _raw("a", "text", AccessPolicy.public())
    with pytest.raises(EmbeddingError):
        Ingestor(store=store, index=index, embedder=FailingEmbedder()).ingest([doc])
    assert store.count_documents() == 0
    assert store.fingerprint(DocumentId("a")) is None
    report = Ingestor(store=store, index=index, embedder=FakeEmbedder()).ingest([doc])
    assert report.documents == 1
