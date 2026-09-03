"""Retrieval requests and scoped results."""

from __future__ import annotations

from dataclasses import dataclass, field

from perimeter.core.document import Chunk, ChunkId, DocumentId, SourceRef
from perimeter.core.errors import InvalidRequestError
from perimeter.core.principal import Principal

MAX_K = 1000
MAX_QUERY_CHARS = 20_000
DEFAULT_CANDIDATE_MULTIPLIER = 4


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """One retrieval call on behalf of ``principal``.

    ``candidate_multiplier`` controls how many permitted candidates the index
    returns to the reranker (``k * candidate_multiplier``). It is a recall knob,
    not a security knob: every candidate is already permitted (INV-2).
    """

    principal: Principal
    query: str = field(repr=False)
    k: int = 10
    candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER

    def __post_init__(self) -> None:
        if not 1 <= self.k <= MAX_K:
            raise InvalidRequestError(f"k must be in [1, {MAX_K}]")
        if not self.query.strip():
            raise InvalidRequestError("query must not be blank")
        if len(self.query) > MAX_QUERY_CHARS:
            raise InvalidRequestError(f"query exceeds {MAX_QUERY_CHARS} characters")
        if self.candidate_multiplier < 1:
            raise InvalidRequestError("candidate_multiplier must be >= 1")

    @property
    def candidate_k(self) -> int:
        return min(self.k * self.candidate_multiplier, MAX_K * DEFAULT_CANDIDATE_MULTIPLIER)


@dataclass(frozen=True, slots=True)
class Citation:
    chunk_id: ChunkId
    document_id: DocumentId
    source: SourceRef
    start: int
    end: int

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> Citation:
        return cls(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            source=chunk.source,
            start=chunk.start,
            end=chunk.end,
        )


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    citation: Citation

    @classmethod
    def of(cls, chunk: Chunk, score: float) -> ScoredChunk:
        return cls(chunk=chunk, score=score, citation=Citation.from_chunk(chunk))

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass(frozen=True, slots=True)
class ScopedResult:
    """The chunks the caller is permitted to see, best first, with citations."""

    principal: Principal
    requested_k: int
    chunks: tuple[ScoredChunk, ...]

    @classmethod
    def empty(cls, *, principal: Principal, k: int) -> ScopedResult:
        return cls(principal=principal, requested_k=k, chunks=())

    @property
    def returned(self) -> int:
        return len(self.chunks)

    @property
    def is_empty(self) -> bool:
        return not self.chunks
