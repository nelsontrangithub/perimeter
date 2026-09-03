"""Benchmark harness: builds a 50k-chunk corpus, measures the budgeted metrics.

Two phases run in separate processes so that peak RSS is that of a *serving*
process (memory-mapped index, in-memory store, runtime) and not of the build
process that holds the float32 corpus:

* ``build`` (bench/build.py): synthetic corpus -> FlatIndex on disk; recall@10
  against an exact float32 scan for two callers (100% and 10% permitted);
  writes queries.npy and build.json into the work directory.
* ``query`` (bench/query.py): opens the index, populates a MemoryStore with one
  chunk per row, runs the real Retriever with zero-cost stubs for the Cohere
  ports, and records latency percentiles, peak RSS, and port calls per query.

``bench/run.py`` orchestrates both and renders the markdown table.
"""

from __future__ import annotations

import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bench.recall import ALLOWED, CALLER, SELECT, SELECT_CALLER, build_index, recall_at_k
from bench.synth import synthetic_corpus, synthetic_queries
from perimeter import __version__
from perimeter.adapters.caching_acl_resolver import CachingAclResolver
from perimeter.adapters.clock import SystemClock
from perimeter.adapters.memory_store import MemoryStore
from perimeter.adapters.static_acl_resolver import StaticAclResolver
from perimeter.core.acl import PermissionSet
from perimeter.core.document import Chunk, Document, DocumentId, SourceRef
from perimeter.core.ports import RerankHit, Vector, as_vector
from perimeter.core.principal import GroupId, Principal, PrincipalId
from perimeter.core.query import RetrievalRequest
from perimeter.index.flat import FlatIndex
from perimeter.pipeline.retrieve import Retriever

CORPUS_SIZE = 50_000
DIM = 1024
RECALL_QUERIES = 200
LOOP_QUERIES = 500
WARMUP_QUERIES = 50
K = 10
SELECTIVE_FRACTION = 0.1


@dataclass(frozen=True, slots=True)
class BuildResult:
    corpus_size: int
    dimension: int
    build_seconds: float
    recall_at_10_all: float
    recall_at_10_selective: float
    selective_fraction: float
    bytes_on_disk: int
    bytes_per_chunk: float
    files: dict[str, int]


@dataclass(frozen=True, slots=True)
class LoopResult:
    caller: str
    permitted_fraction: float
    queries: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    embed_calls_per_query: float
    rerank_calls_per_query: float
    resolver_calls_per_query: float
    returned_min: int
    returned_max: int


@dataclass(frozen=True, slots=True)
class QueryResult:
    peak_rss_bytes: int
    loops: list[LoopResult]


# --- build phase ----------------------------------------------------------------------


def run_build(work: Path, *, corpus_size: int = CORPUS_SIZE) -> BuildResult:
    t0 = time.perf_counter()
    corpus = synthetic_corpus(corpus_size, DIM, seed=0)
    queries = synthetic_queries(corpus, max(RECALL_QUERIES, LOOP_QUERIES + WARMUP_QUERIES), seed=1)
    mask = np.random.default_rng(7).random(corpus_size) < SELECTIVE_FRACTION
    index = FlatIndex.open(work / "index", dimension=DIM)
    build_index(index, corpus, mask)
    build_seconds = time.perf_counter() - t0
    np.save(work / "queries.npy", queries)
    np.save(work / "mask.npy", mask)
    recall_all = recall_at_k(
        index,
        corpus,
        queries[:RECALL_QUERIES],
        np.ones(corpus_size, dtype=bool),
        caller=CALLER,
        k=K,
    )
    recall_sel = recall_at_k(
        index, corpus, queries[:RECALL_QUERIES], mask, caller=SELECT_CALLER, k=K
    )
    stats = index.stats()
    result = BuildResult(
        corpus_size=corpus_size,
        dimension=DIM,
        build_seconds=round(build_seconds, 2),
        recall_at_10_all=round(recall_all.recall, 4),
        recall_at_10_selective=round(recall_sel.recall, 4),
        selective_fraction=SELECTIVE_FRACTION,
        bytes_on_disk=stats.bytes_on_disk,
        bytes_per_chunk=round(stats.bytes_per_chunk, 1),
        files=stats.files,
    )
    (work / "build.json").write_text(json.dumps(asdict(result), indent=2))
    return result


# --- query phase ----------------------------------------------------------------------


class ReplayEmbedder:
    """Zero-cost stand-in for the embedding API: replays precomputed query vectors."""

    dimension = DIM

    def __init__(self, queries: np.ndarray) -> None:
        self._queries = queries
        self.calls = 0

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        raise NotImplementedError("query phase never embeds documents")

    def embed_query(self, text: str) -> Vector:
        self.calls += 1
        return as_vector(self._queries[int(text)])


class PassThroughReranker:
    """Zero-cost stand-in for the rerank API: keeps index order, counts calls."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, chunks: Sequence[Chunk], k: int) -> Sequence[RerankHit]:
        self.calls += 1
        n = len(chunks)
        return [RerankHit(chunk_id=c.id, score=float(n - i)) for i, c in enumerate(chunks[:k])]


class CountingResolver:
    def __init__(self, inner: StaticAclResolver) -> None:
        self._inner = inner
        self.calls = 0

    def resolve(self, principal: Principal) -> PermissionSet:
        self.calls += 1
        return self._inner.resolve(principal)


def _populate_store(store: MemoryStore, index: FlatIndex) -> None:
    from bench.recall import policies_for

    mask = np.load(index.path.parent / "mask.npy")
    policies = policies_for(mask)
    for row in range(index.size):
        cid = index.chunk_id_at(row)
        doc = Document.create(
            id=DocumentId(str(cid)),
            source=SourceRef("bench", f"bench://{cid}", str(cid)),
            policy=policies[row],
            text=f"chunk {row} of the synthetic corpus",
        )
        chunk = Chunk(
            id=cid,
            document_id=doc.id,
            ordinal=0,
            start=0,
            end=len(doc.text),
            text=doc.text,
            policy=doc.policy,
            source=doc.source,
        )
        store.put(doc, [chunk])


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if sys.platform == "darwin" else int(rss) * 1024


def run_query(work: Path) -> QueryResult:
    queries = np.load(work / "queries.npy")
    index = FlatIndex.open(work / "index", dimension=DIM)
    store = MemoryStore()
    _populate_store(store, index)
    embedder = ReplayEmbedder(queries)
    reranker = PassThroughReranker()
    counting = CountingResolver(StaticAclResolver())
    resolver = CachingAclResolver(counting, clock=SystemClock(), ttl_seconds=60)
    retriever = Retriever(
        resolver=resolver, embedder=embedder, index=index, store=store, reranker=reranker
    )
    callers = [
        ("all", 1.0, Principal(PrincipalId("bench-all"), frozenset({GroupId(ALLOWED)}))),
        (
            "selective",
            SELECTIVE_FRACTION,
            Principal(PrincipalId("bench-sel"), frozenset({GroupId(SELECT)})),
        ),
    ]
    loops: list[LoopResult] = []
    for name, fraction, principal in callers:
        for i in range(WARMUP_QUERIES):
            retriever.retrieve(RetrievalRequest(principal=principal, query=str(i), k=K))
        embed0, rerank0, resolve0 = embedder.calls, reranker.calls, counting.calls
        latencies: list[float] = []
        returned: list[int] = []
        for i in range(LOOP_QUERIES):
            request = RetrievalRequest(principal=principal, query=str(WARMUP_QUERIES + i), k=K)
            t0 = time.perf_counter_ns()
            scoped = retriever.retrieve(request)
            latencies.append((time.perf_counter_ns() - t0) / 1e6)
            returned.append(scoped.returned)
        latencies.sort()
        q = statistics.quantiles(latencies, n=100)
        loops.append(
            LoopResult(
                caller=name,
                permitted_fraction=fraction,
                queries=LOOP_QUERIES,
                p50_ms=round(q[49], 3),
                p95_ms=round(q[94], 3),
                p99_ms=round(q[98], 3),
                mean_ms=round(statistics.fmean(latencies), 3),
                embed_calls_per_query=(embedder.calls - embed0) / LOOP_QUERIES,
                rerank_calls_per_query=(reranker.calls - rerank0) / LOOP_QUERIES,
                resolver_calls_per_query=(counting.calls - resolve0) / LOOP_QUERIES,
                returned_min=min(returned),
                returned_max=max(returned),
            )
        )
    result = QueryResult(peak_rss_bytes=_peak_rss_bytes(), loops=loops)
    (work / "query.json").write_text(json.dumps(asdict(result), indent=2))
    return result


# --- orchestration --------------------------------------------------------------------


def run_phase_subprocess(phase: str, work: Path, *, corpus_size: int = CORPUS_SIZE) -> None:
    cmd = [
        sys.executable,
        "-m",
        f"bench.{phase}",
        "--work",
        str(work),
        "--corpus-size",
        str(corpus_size),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603 - fixed argv, no shell


def load_results(work: Path) -> tuple[BuildResult, QueryResult]:
    build = BuildResult(**json.loads((work / "build.json").read_text()))
    raw = json.loads((work / "query.json").read_text())
    query = QueryResult(
        peak_rss_bytes=raw["peak_rss_bytes"], loops=[LoopResult(**loop) for loop in raw["loops"]]
    )
    return build, query


def environment() -> dict[str, str]:
    return {
        "perimeter": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "cpu": platform.processor() or platform.machine(),
    }


def render_markdown(build: BuildResult, query: QueryResult, env: dict[str, str]) -> str:
    all_loop = next(loop for loop in query.loops if loop.caller == "all")
    sel_loop = next(loop for loop in query.loops if loop.caller == "selective")
    api_calls = all_loop.embed_calls_per_query + all_loop.rerank_calls_per_query
    mib = query.peak_rss_bytes / (1024 * 1024)
    rows = [
        ("p95 retrieval latency, all rows permitted", f"{all_loop.p95_ms:.2f} ms", "<= 30 ms"),
        (
            f"p95 retrieval latency, {int(build.selective_fraction * 100)}% of rows permitted",
            f"{sel_loop.p95_ms:.2f} ms",
            "<= 30 ms",
        ),
        ("Peak RSS, sustained query loop", f"{mib:.0f} MiB", "<= 512 MiB"),
        ("Index bytes per chunk", f"{build.bytes_per_chunk:.0f} B", "<= 1,280 B"),
        (
            "recall@10 vs exact float32, all rows permitted",
            f"{build.recall_at_10_all:.3f}",
            ">= 0.95",
        ),
        (
            f"recall@10 vs exact float32, {int(build.selective_fraction * 100)}% permitted",
            f"{build.recall_at_10_selective:.3f}",
            ">= 0.95",
        ),
        (
            "Cohere API calls per query",
            f"{api_calls:.0f} ({all_loop.embed_calls_per_query:.0f} embed, "
            f"{all_loop.rerank_calls_per_query:.0f} rerank)",
            "<= 2",
        ),
    ]
    lines = [
        "| Metric | Measured | Budget |",
        "|--------|---------:|-------:|",
        *[f"| {m} | {v} | {b} |" for m, v, b in rows],
        "",
        f"Corpus: {build.corpus_size:,} chunks x {build.dimension} dims (synthetic, clustered); "
        f"k={K}; {all_loop.queries} timed queries per caller after {WARMUP_QUERIES} warm-up; "
        f"Cohere ports stubbed at zero cost (calls counted); "
        f"p50 {all_loop.p50_ms:.2f} ms / p99 {all_loop.p99_ms:.2f} ms (all rows); "
        f"ACL resolver calls per query {all_loop.resolver_calls_per_query:.3f} (cached); "
        f"index build {build.build_seconds:.1f} s; "
        f"on disk {build.bytes_on_disk / (1024 * 1024):.1f} MiB.",
        "",
        "Environment: " + ", ".join(f"{k} {v}" for k, v in env.items()) + ".",
    ]
    return "\n".join(lines) + "\n"
