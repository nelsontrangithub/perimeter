"""Cohere embed-v4.0 behind the :class:`~perimeter.core.ports.EmbeddingModel` port.

Talks to ``POST /v2/embed`` over plain HTTPS with httpx. The vendor SDK is not
used: a fake transport tests the adapter completely, the runtime image stays
small, and there is one fewer dependency in the air-gapped build.
"""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from typing import Any

import httpx

from perimeter.adapters._cohere import make_client, post_json
from perimeter.core.errors import EmbeddingError
from perimeter.core.ports import Vector

MODEL = "embed-v4.0"
DEFAULT_DIMENSION = 1024
DEFAULT_BATCH_SIZE = 96


class CohereEmbeddings:
    def __init__(
        self,
        *,
        api_key: str,
        client: httpx.Client | None = None,
        dimension: int = DEFAULT_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        model: str = MODEL,
    ) -> None:
        self._api_key = api_key
        self._client = client or make_client()
        self._dimension = dimension
        self._batch_size = max(1, batch_size)
        self._model = model

    def __repr__(self) -> str:
        return f"CohereEmbeddings(model={self._model!r}, dimension={self._dimension})"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        out: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            out.extend(
                self._embed(list(texts[start : start + self._batch_size]), "search_document")
            )
        return out

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], "search_query")[0]

    def _embed(self, texts: list[str], input_type: str) -> list[Vector]:
        if not texts:
            return []
        data = post_json(
            self._client,
            "/v2/embed",
            self._api_key,
            {
                "model": self._model,
                "texts": texts,
                "input_type": input_type,
                "embedding_types": ["float"],
                "output_dimension": self._dimension,
            },
            EmbeddingError,
            "embed",
        )
        return [self._to_vector(row) for row in self._rows(data, len(texts))]

    @staticmethod
    def _rows(data: dict[str, Any], expected: int) -> list[Any]:
        embeddings = data.get("embeddings")
        rows = embeddings.get("float") if isinstance(embeddings, dict) else None
        if not isinstance(rows, list) or len(rows) != expected:
            raise EmbeddingError("embed: malformed response")
        return rows

    def _to_vector(self, row: Any) -> Vector:
        if not isinstance(row, list) or len(row) != self._dimension:
            raise EmbeddingError(f"embed: expected {self._dimension} dimensions")
        try:
            return array("f", (float(x) for x in row))
        except (TypeError, ValueError):
            raise EmbeddingError("embed: non-numeric embedding value") from None
