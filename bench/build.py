"""Build phase of the benchmark (run in its own process). See bench/harness.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.harness import CORPUS_SIZE, run_build


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--corpus-size", type=int, default=CORPUS_SIZE)
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    result = run_build(work, corpus_size=args.corpus_size)
    print(
        f"built {result.corpus_size} rows in {result.build_seconds}s; "
        f"recall@10 all={result.recall_at_10_all} selective={result.recall_at_10_selective}; "
        f"{result.bytes_per_chunk} B/chunk"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
