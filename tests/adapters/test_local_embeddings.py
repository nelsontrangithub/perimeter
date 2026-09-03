"""Deterministic local embedder for air-gapped runs, tests, and benchmarks."""

from __future__ import annotations

import math

from perimeter.adapters.local_embeddings import LocalEmbeddings
from perimeter.core.ports import EmbeddingModel


def test_satisfies_port_with_requested_dimension() -> None:
    model = LocalEmbeddings(dimension=64)
    assert isinstance(model, EmbeddingModel)
    assert model.dimension == 64
    assert len(model.embed_query("hello")) == 64
    assert [len(v) for v in model.embed_documents(["a", "b"])] == [64, 64]


def test_is_deterministic_and_unit_length() -> None:
    model = LocalEmbeddings(dimension=128)
    a = model.embed_query("the quick brown fox")
    b = model.embed_query("the quick brown fox")
    assert list(a) == list(b)
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-5)


def test_similar_text_is_closer_than_unrelated_text() -> None:
    model = LocalEmbeddings(dimension=256)
    q = model.embed_query("quarterly revenue report for finance")
    near = model.embed_query("finance quarterly revenue figures")
    far = model.embed_query("gardening tips for tomatoes in spring")

    def cos(x: object, y: object) -> float:
        return sum(a * b for a, b in zip(x, y, strict=True))  # type: ignore[call-overload]

    assert cos(q, near) > cos(q, far)


def test_empty_text_embeds_to_zero_vector_without_nan() -> None:
    v = LocalEmbeddings(dimension=32).embed_query("")
    assert all(x == 0.0 for x in v)
