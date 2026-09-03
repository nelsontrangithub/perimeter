"""The ports are structural Protocols; these tests pin their shapes with minimal fakes."""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from perimeter.core.acl import PermissionSet
from perimeter.core.document import Chunk, ChunkId, Document, DocumentId
from perimeter.core.ports import (
    AclResolver,
    Clock,
    DocumentStore,
    EmbeddingModel,
    IndexEntry,
    IndexHit,
    Reranker,
    RerankHit,
    Vector,
    VectorIndex,
    as_vector,
)
from perimeter.core.principal import Principal


class _Embedder:
    dimension = 4

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return [as_vector([1.0, 0.0, 0.0, 0.0]) for _ in texts]

    def embed_query(self, text: str) -> Vector:
        return as_vector([1.0, 0.0, 0.0, 0.0])


class _Reranker:
    def rerank(self, query: str, chunks: Sequence[Chunk], k: int) -> Sequence[RerankHit]:
        return [RerankHit(chunk_id=c.id, score=1.0) for c in chunks[:k]]


class _Index:
    def __init__(self) -> None:
        self._entries: list[IndexEntry] = []

    @property
    def size(self) -> int:
        return len(self._entries)

    def add(self, entries: Iterable[IndexEntry]) -> None:
        self._entries.extend(entries)

    def search(self, query: Vector, permitted: PermissionSet, k: int) -> Sequence[IndexHit]:
        return [
            IndexHit(chunk_id=e.chunk_id, score=0.0)
            for e in self._entries
            if e.policy.admits(permitted)
        ][:k]


class _Store:
    def put(self, document: Document, chunks: Sequence[Chunk]) -> None:
        return None

    def get_chunks(self, ids: Sequence[ChunkId], permitted: PermissionSet) -> Sequence[Chunk]:
        return []

    def get_document(self, id: DocumentId, permitted: PermissionSet) -> Document | None:
        return None

    def delete(self, id: DocumentId) -> None:
        return None

    def count_documents(self) -> int:
        return 0

    def count_chunks(self) -> int:
        return 0


class _Resolver:
    def resolve(self, principal: Principal) -> PermissionSet:
        return PermissionSet.empty()


class _Clock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def monotonic(self) -> float:
        return 0.0


def test_fakes_satisfy_protocols() -> None:
    assert isinstance(_Embedder(), EmbeddingModel)
    assert isinstance(_Reranker(), Reranker)
    assert isinstance(_Index(), VectorIndex)
    assert isinstance(_Store(), DocumentStore)
    assert isinstance(_Resolver(), AclResolver)
    assert isinstance(_Clock(), Clock)


def test_as_vector_produces_float32_array() -> None:
    v = as_vector([0.5, 1.5])
    assert isinstance(v, array)
    assert v.typecode == "f"
    assert list(v) == [0.5, 1.5]


def test_as_vector_passes_through_existing_float_array() -> None:
    src = array("f", [1.0, 2.0])
    assert as_vector(src) is src
