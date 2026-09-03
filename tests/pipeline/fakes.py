"""Shared port fakes for pipeline and invariant tests. No I/O."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from perimeter.adapters.memory_store import MemoryStore
from perimeter.core.acl import AccessPolicy, PermissionSet
from perimeter.core.document import Chunk, DocumentId, SourceRef
from perimeter.core.errors import AclResolutionError
from perimeter.core.ports import IndexEntry, IndexHit, RerankHit, Vector, as_vector
from perimeter.core.principal import GroupGraph, Principal
from perimeter.pipeline.ingest import ChunkingConfig, Ingestor, RawDocument

DIM = 16


class BagEmbedder:
    dimension = DIM

    def __init__(self) -> None:
        self.query_calls = 0

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        self.query_calls += 1
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> Vector:
        v = [0.0] * DIM
        for ch in text.lower():
            v[ord(ch) % DIM] += 1.0
        return as_vector(v)


class ListIndex:
    """Exact cosine over a list, filtering inside its own scan loop."""

    def __init__(self) -> None:
        self.entries: list[IndexEntry] = []
        self.search_calls: list[tuple[PermissionSet, int]] = []

    @property
    def size(self) -> int:
        return len(self.entries)

    def add(self, entries: Iterable[IndexEntry]) -> None:
        self.entries.extend(entries)

    def remove_document(self, document_id: DocumentId) -> None:
        self.entries = [e for e in self.entries if not e.chunk_id.startswith(f"{document_id}#")]

    def search(self, query: Vector, permitted: PermissionSet, k: int) -> Sequence[IndexHit]:
        self.search_calls.append((permitted, k))
        if permitted.is_empty:
            return []
        scored = [
            (self._dot(query, e.vector), e.chunk_id)
            for e in self.entries
            if e.policy.admits(permitted)
        ]
        scored.sort(key=lambda t: -t[0])
        return [IndexHit(chunk_id=cid, score=s) for s, cid in scored[:k]]

    @staticmethod
    def _dot(a: Vector, b: Vector) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))


class LeakyIndex(ListIndex):
    """A broken index that ignores the permission set. Used to prove the pipeline
    and store still refuse (INV-1 defense in depth)."""

    def search(self, query: Vector, permitted: PermissionSet, k: int) -> Sequence[IndexHit]:
        self.search_calls.append((permitted, k))
        scored = sorted(
            ((self._dot(query, e.vector), e.chunk_id) for e in self.entries), key=lambda t: -t[0]
        )
        return [IndexHit(chunk_id=cid, score=s) for s, cid in scored[:k]]


class SpyReranker:
    def __init__(self) -> None:
        self.received: list[list[Chunk]] = []

    def rerank(self, query: str, chunks: Sequence[Chunk], k: int) -> Sequence[RerankHit]:
        self.received.append(list(chunks))
        # Reverse the order so tests can tell the reranker's ordering was used;
        # scores decrease along the returned order, as a real reranker's do.
        ordered = list(reversed(chunks))
        n = len(ordered)
        return [RerankHit(chunk_id=c.id, score=float(n - i)) for i, c in enumerate(ordered)][:k]


class StaticResolver:
    def __init__(self, graph: GroupGraph | None = None, *, fail: bool = False) -> None:
        self._graph = graph or GroupGraph.empty()
        self._fail = fail
        self.calls = 0

    def resolve(self, principal: Principal) -> PermissionSet:
        self.calls += 1
        if self._fail:
            raise AclResolutionError("resolver unavailable")
        return PermissionSet.resolve(principal, self._graph)


def raw(doc_id: str, text: str, policy: AccessPolicy) -> RawDocument:
    return RawDocument(
        id=DocumentId(doc_id),
        source=SourceRef("fs", f"file:///{doc_id}", doc_id),
        policy=policy,
        text=text,
    )


def build_corpus(
    docs: Sequence[RawDocument], *, index: ListIndex | None = None, max_chars: int = 40
) -> tuple[MemoryStore, ListIndex, BagEmbedder]:
    store, idx, embedder = MemoryStore(), index or ListIndex(), BagEmbedder()
    Ingestor(
        store=store,
        index=idx,
        embedder=embedder,
        chunking=ChunkingConfig(max_chars=max_chars, overlap_chars=0),
    ).ingest(docs)
    return store, idx, embedder
