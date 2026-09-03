"""Cohere rerank-v4.0-fast behind the :class:`~perimeter.core.ports.Reranker` port.

The reranker sees chunk text. It is the pipeline's job (INV-2) to make sure
every chunk handed here is already permitted; this adapter only sends what it
is given and maps the response back to chunk IDs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from perimeter.adapters._cohere import make_client, post_json
from perimeter.core.document import Chunk
from perimeter.core.errors import RerankError
from perimeter.core.ports import RerankHit

MODEL = "rerank-v4.0-fast"


class CohereReranker:
    def __init__(
        self, *, api_key: str, client: httpx.Client | None = None, model: str = MODEL
    ) -> None:
        self._api_key = api_key
        self._client = client or make_client()
        self._model = model

    def __repr__(self) -> str:
        return f"CohereReranker(model={self._model!r})"

    def rerank(self, query: str, chunks: Sequence[Chunk], k: int) -> Sequence[RerankHit]:
        if not chunks or k <= 0:
            return []
        data = post_json(
            self._client,
            "/v2/rerank",
            self._api_key,
            {
                "model": self._model,
                "query": query,
                "documents": [c.text for c in chunks],
                "top_n": min(k, len(chunks)),
            },
            RerankError,
            "rerank",
        )
        results = data.get("results")
        if not isinstance(results, list):
            raise RerankError("rerank: malformed response")
        hits: list[RerankHit] = []
        for item in results:
            index, score = self._parse(item, len(chunks))
            hits.append(RerankHit(chunk_id=chunks[index].id, score=score))
        return hits

    @staticmethod
    def _parse(item: Any, n: int) -> tuple[int, float]:
        if not isinstance(item, dict):
            raise RerankError("rerank: malformed result item")
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or not 0 <= index < n:
            raise RerankError("rerank: result index out of range")
        if not isinstance(score, int | float):
            raise RerankError("rerank: missing relevance score")
        return index, float(score)
