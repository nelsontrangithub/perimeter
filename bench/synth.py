"""Deterministic synthetic corpora for benchmarks and gates.

Real chunk embeddings are not isotropic noise: chunks of one document cluster,
and a query lands near a handful of chunks. The generator models that with
``clusters`` centres, per-chunk Gaussian spread around a centre, and queries
that are a perturbed corpus vector, so the exact top-k is well defined and the
quantized index has something to recover.
"""

from __future__ import annotations

import numpy as np

from perimeter.index.quantize import F32, l2_normalize


def synthetic_corpus(
    n: int, dim: int, *, clusters: int = 512, spread: float = 0.6, seed: int = 0
) -> F32:
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((clusters, dim)).astype(np.float32)
    assignment = rng.integers(0, clusters, size=n)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * np.float32(spread)
    return l2_normalize(centres[assignment] + noise)


def synthetic_queries(corpus: F32, n_queries: int, *, jitter: float = 0.35, seed: int = 1) -> F32:
    """Perturbed corpus vectors. ``jitter`` is relative to the unit vector's norm, so the
    expected cosine between a query and its source is ``1 / sqrt(1 + jitter**2)``."""
    rng = np.random.default_rng(seed)
    dim = corpus.shape[1]
    picks = rng.integers(0, corpus.shape[0], size=n_queries)
    sigma = np.float32(jitter / np.sqrt(dim))
    noise = rng.standard_normal((n_queries, dim)).astype(np.float32) * sigma
    return l2_normalize(corpus[picks] + noise)
