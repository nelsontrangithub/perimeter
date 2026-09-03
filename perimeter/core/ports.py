"""I/O ports.

Everything Perimeter needs from the outside world is one of these Protocols.
Implementations live in :mod:`perimeter.adapters` and :mod:`perimeter.index`;
the pipeline depends only on the Protocols. Every Protocol here is
``runtime_checkable`` so tests can assert an adapter satisfies its port.

Vectors cross this boundary as ``array.array('f')``: a standard-library type
that NumPy wraps without copying, so the core never imports NumPy and the index
never converts a list of Python floats.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from perimeter.core.acl import AccessPolicy, PermissionSet
from perimeter.core.document import Chunk, ChunkId, Document, DocumentId
from perimeter.core.principal import Principal

Vector = array[float]
"""A float32 vector. Construct with :func:`as_vector`."""


def as_vector(values: Iterable[float]) -> Vector:
    """Coerce to a float32 ``array`` without copying when it already is one."""
    if isinstance(values, array) and values.typecode == "f":
        return values
    return array("f", values)


@dataclass(frozen=True, slots=True)
class IndexEntry:
    """What the index stores per chunk: an ID, a vector, and the policy to filter on.

    No text (ADR-006)."""

    chunk_id: ChunkId
    vector: Vector
    policy: AccessPolicy


@dataclass(frozen=True, slots=True)
class IndexHit:
    chunk_id: ChunkId
    score: float


@dataclass(frozen=True, slots=True)
class RerankHit:
    chunk_id: ChunkId
    score: float


@runtime_checkable
class EmbeddingModel(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders *already permitted* chunks (INV-2). Sees text; never sees anything else."""

    def rerank(self, query: str, chunks: Sequence[Chunk], k: int) -> Sequence[RerankHit]: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Filtered nearest-neighbour search. ``search`` applies ``permitted`` inside the scan."""

    @property
    def size(self) -> int: ...

    def add(self, entries: Iterable[IndexEntry]) -> None: ...

    def search(self, query: Vector, permitted: PermissionSet, k: int) -> Sequence[IndexHit]: ...


@runtime_checkable
class DocumentStore(Protocol):
    """Holds text. Every read takes the caller's permission set and enforces it (INV-1)."""

    def put(self, document: Document, chunks: Sequence[Chunk]) -> None: ...

    def get_chunks(self, ids: Sequence[ChunkId], permitted: PermissionSet) -> Sequence[Chunk]: ...

    def get_document(self, id: DocumentId, permitted: PermissionSet) -> Document | None: ...

    def delete(self, id: DocumentId) -> None: ...

    def count_documents(self) -> int: ...

    def count_chunks(self) -> int: ...


@runtime_checkable
class AclResolver(Protocol):
    """Expands a forwarded identity into its effective permission set.

    Implementations must fail closed: on any error, raise
    :class:`~perimeter.core.errors.AclResolutionError` or return the empty set.
    """

    def resolve(self, principal: Principal) -> PermissionSet: ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...
