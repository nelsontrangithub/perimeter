"""Deterministic local embedder: feature-hashed word and character n-grams.

Used when no Cohere key is configured (air-gapped demo, CI, benchmarks). It
is not a semantic model and does not pretend to be one; it is a stable,
network-free stand-in with the same port and the same vector shape, so every
other component can be exercised exactly as it will run in production.
"""

from __future__ import annotations

import math
import re
import zlib
from array import array
from collections.abc import Iterator, Sequence
from itertools import pairwise

from perimeter.core.ports import Vector

_WORD = re.compile(r"[a-z0-9]+")


class LocalEmbeddings:
    name = "local"

    def __init__(self, *, dimension: int = 1024) -> None:
        self._dimension = dimension

    def __repr__(self) -> str:
        return f"LocalEmbeddings(dimension={self._dimension})"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    @staticmethod
    def _features(text: str) -> Iterator[str]:
        words = _WORD.findall(text.lower())
        yield from words
        for a, b in pairwise(words):
            yield f"{a} {b}"
        for w in words:
            padded = f" {w} "
            for i in range(len(padded) - 2):
                yield "#" + padded[i : i + 3]

    def _embed(self, text: str) -> Vector:
        acc = [0.0] * self._dimension
        for feature in self._features(text):
            h = zlib.crc32(feature.encode("utf-8"))
            bucket = h % self._dimension
            sign = 1.0 if (h >> 31) & 1 else -1.0
            acc[bucket] += sign
        norm = math.sqrt(sum(x * x for x in acc))
        if norm > 0:
            acc = [x / norm for x in acc]
        return array("f", acc)
