"""Run the benchmark and emit the results table pasted into the README.

    uv run python -m bench.run --out bench/results.md

Every number in the table is measured by this harness in this run. Nothing is
typed in by hand.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from bench.harness import (
    CORPUS_SIZE,
    environment,
    load_results,
    render_markdown,
    run_phase_subprocess,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench/results.md")
    parser.add_argument("--json", default="bench/results.json")
    parser.add_argument("--work", default=None, help="work directory (default: temporary)")
    parser.add_argument("--corpus-size", type=int, default=CORPUS_SIZE)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(args.work) if args.work else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        run_phase_subprocess("build", work, corpus_size=args.corpus_size)
        run_phase_subprocess("query", work, corpus_size=args.corpus_size)
        build, query = load_results(work)

    env = environment()
    table = render_markdown(build, query, env)
    Path(args.out).write_text(table)
    Path(args.json).write_text(
        json.dumps({"build": asdict(build), "query": asdict(query), "environment": env}, indent=2)
        + "\n"
    )
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
