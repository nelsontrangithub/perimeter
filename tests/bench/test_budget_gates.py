"""CI gates over the performance budget (bench/budget.py).

Runs the benchmark harness exactly as `make bench` does, both phases in their
own processes, at the full 50k-chunk corpus, and fails the build if any
measured figure exceeds its ceiling. Marked `bench`; run with `make bench-gate`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench import budget
from bench.harness import (
    CORPUS_SIZE,
    BuildResult,
    QueryResult,
    load_results,
    run_phase_subprocess,
)

pytestmark = pytest.mark.bench


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> tuple[BuildResult, QueryResult]:
    work: Path = tmp_path_factory.mktemp("gates")
    run_phase_subprocess("build", work, corpus_size=CORPUS_SIZE)
    run_phase_subprocess("query", work, corpus_size=CORPUS_SIZE)
    return load_results(work)


def test_gate_corpus_is_the_budgeted_size(measured: tuple[BuildResult, QueryResult]) -> None:
    build, _ = measured
    assert build.corpus_size == CORPUS_SIZE == 50_000


@pytest.mark.parametrize("caller", ["all", "selective"])
def test_gate_p95_latency(measured: tuple[BuildResult, QueryResult], caller: str) -> None:
    _, query = measured
    loop = next(x for x in query.loops if x.caller == caller)
    assert loop.p95_ms <= budget.P95_LATENCY_MS, (
        f"p95 {loop.p95_ms} ms for {caller} exceeds {budget.P95_LATENCY_MS} ms"
    )


def test_gate_peak_rss(measured: tuple[BuildResult, QueryResult]) -> None:
    _, query = measured
    mib = query.peak_rss_bytes / (1024 * 1024)
    assert mib <= budget.PEAK_RSS_MIB, f"peak RSS {mib:.0f} MiB exceeds {budget.PEAK_RSS_MIB} MiB"


def test_gate_index_bytes_per_chunk(measured: tuple[BuildResult, QueryResult]) -> None:
    build, _ = measured
    assert build.bytes_per_chunk <= budget.INDEX_BYTES_PER_CHUNK, (
        f"{build.bytes_per_chunk} B/chunk exceeds {budget.INDEX_BYTES_PER_CHUNK}"
    )


def test_gate_recall_at_10(measured: tuple[BuildResult, QueryResult]) -> None:
    build, _ = measured
    assert build.recall_at_10_all >= budget.RECALL_AT_10, f"recall {build.recall_at_10_all}"
    assert build.recall_at_10_selective >= budget.RECALL_AT_10, (
        f"selective recall {build.recall_at_10_selective}"
    )


def test_gate_cohere_calls_per_query(measured: tuple[BuildResult, QueryResult]) -> None:
    _, query = measured
    for loop in query.loops:
        calls = loop.embed_calls_per_query + loop.rerank_calls_per_query
        assert calls <= budget.COHERE_CALLS_PER_QUERY, f"{calls} API calls/query ({loop.caller})"
        assert loop.embed_calls_per_query == 1.0
        assert loop.rerank_calls_per_query == 1.0


def test_gate_k_is_preserved_when_permitted_set_covers_it(
    measured: tuple[BuildResult, QueryResult],
) -> None:
    """Both callers can see far more than k rows, so every query must return exactly k."""
    _, query = measured
    for loop in query.loops:
        assert loop.returned_min == loop.returned_max == 10, loop
