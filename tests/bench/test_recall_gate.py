"""Regression gate: recall@10 of the quantized filtered scan vs exact float32.

Budget lives in CLAUDE.md ("Performance budget"). This test fails the build if
the index's recall drops below it, whether from a quantizer change, a scan
change, or a rescoring change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.recall import build_index, recall_at_k
from bench.synth import synthetic_corpus, synthetic_queries
from perimeter.index.flat import FlatIndex

pytestmark = pytest.mark.bench

RECALL_AT_10_FLOOR = 0.95
CORPUS_SIZE = 50_000
DIM = 1024
QUERIES = 200


@pytest.fixture(scope="module")
def corpus() -> np.ndarray:
    return synthetic_corpus(CORPUS_SIZE, DIM, seed=0)


@pytest.fixture(scope="module")
def queries(corpus: np.ndarray) -> np.ndarray:
    return synthetic_queries(corpus, QUERIES, seed=1)


@pytest.mark.parametrize("permitted_fraction", [1.0, 0.1])
def test_recall_at_10_meets_budget(
    tmp_path: Path, corpus: np.ndarray, queries: np.ndarray, permitted_fraction: float
) -> None:
    rng = np.random.default_rng(7)
    mask = rng.random(CORPUS_SIZE) < permitted_fraction
    index = FlatIndex.open(tmp_path / f"idx-{permitted_fraction}", dimension=DIM)
    build_index(index, corpus, mask)
    result = recall_at_k(index, corpus, queries, mask, k=10)
    assert result.recall >= RECALL_AT_10_FLOOR, (
        f"recall@10={result.recall:.4f} below floor {RECALL_AT_10_FLOOR} "
        f"(permitted fraction {permitted_fraction})"
    )
