"""Retrieval orchestrator: resolve principals -> filtered search -> rerank -> assemble.

Order matters and is the point:

1. Resolve the caller's permission set first. If it is empty, or resolution
   fails, return an empty result *before* embedding the query (INV-4). No API
   call is spent on a caller who can see nothing, and there is no path from
   "no permissions" to an unfiltered scan.
2. Embed the query and ask the index for ``candidate_k`` hits *with the
   permission set*. The index filters inside its scan (INV-2).
3. Fetch chunk text from the store *with the permission set*. The store
   refuses text for anything the policy does not admit (INV-1, check two).
4. Re-check every fetched chunk's policy here (INV-1, check three). The
   ``tests/invariants`` suite drives this orchestrator with an index that
   ignores permissions and asserts nothing leaks past steps 3 and 4.
5. Rerank the permitted candidates (the reranker only ever sees permitted
   text), take the top ``k``, attach citations.
"""

from __future__ import annotations

import logging

from perimeter.core.acl import PermissionSet
from perimeter.core.document import Chunk, ChunkId
from perimeter.core.errors import AclResolutionError
from perimeter.core.ports import AclResolver, DocumentStore, EmbeddingModel, Reranker, VectorIndex
from perimeter.core.query import RetrievalRequest, ScopedResult, ScoredChunk

log = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        *,
        resolver: AclResolver,
        embedder: EmbeddingModel,
        index: VectorIndex,
        store: DocumentStore,
        reranker: Reranker | None,
    ) -> None:
        self._resolver = resolver
        self._embedder = embedder
        self._index = index
        self._store = store
        self._reranker = reranker

    def permissions_for(self, request: RetrievalRequest) -> PermissionSet:
        """Resolve, failing closed. Logs the failure class only, never the principal."""
        try:
            return self._resolver.resolve(request.principal)
        except AclResolutionError as exc:
            log.warning(
                "acl resolution failed; treating as empty permission set: %s", type(exc).__name__
            )
            return PermissionSet.empty()

    def retrieve(self, request: RetrievalRequest) -> ScopedResult:
        permitted = self.permissions_for(request)
        if permitted.is_empty:
            return ScopedResult.empty(principal=request.principal, k=request.k)

        query_vector = self._embedder.embed_query(request.query)
        hits = self._index.search(query_vector, permitted, request.candidate_k)
        if not hits:
            return ScopedResult.empty(principal=request.principal, k=request.k)

        fetched = self._store.get_chunks([h.chunk_id for h in hits], permitted)
        chunks = [c for c in fetched if c.policy.admits(permitted)]
        if not chunks:
            return ScopedResult.empty(principal=request.principal, k=request.k)

        index_scores = {h.chunk_id: h.score for h in hits}
        scored = self._order(request, chunks, index_scores)
        return ScopedResult(
            principal=request.principal,
            requested_k=request.k,
            chunks=tuple(scored[: request.k]),
            candidates=len(chunks),
        )

    def _order(
        self, request: RetrievalRequest, chunks: list[Chunk], index_scores: dict[ChunkId, float]
    ) -> list[ScoredChunk]:
        if self._reranker is None:
            ranked = sorted(chunks, key=lambda c: -index_scores.get(c.id, 0.0))
            return [ScoredChunk.of(c, index_scores.get(c.id, 0.0)) for c in ranked]
        by_id = {c.id: c for c in chunks}
        out: list[ScoredChunk] = []
        for hit in self._reranker.rerank(request.query, chunks, request.k):
            chunk = by_id.get(hit.chunk_id)
            if chunk is not None:
                out.append(ScoredChunk.of(chunk, hit.score))
        out.sort(key=lambda sc: -sc.score)
        return out
