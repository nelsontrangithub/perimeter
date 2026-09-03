"""Query phase of the benchmark (run in its own process). See bench/harness.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.harness import run_query


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--corpus-size", type=int, default=0)  # accepted for symmetry
    args = parser.parse_args()
    result = run_query(Path(args.work))
    for loop in result.loops:
        print(f"{loop.caller}: p50 {loop.p50_ms} ms, p95 {loop.p95_ms} ms, p99 {loop.p99_ms} ms")
    print(f"peak rss {result.peak_rss_bytes / (1024 * 1024):.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
