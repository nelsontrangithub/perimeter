"""``perimeter`` command line: serve."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from perimeter.server.logging import configure_logging
from perimeter.server.settings import Settings

log = logging.getLogger(__name__)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from perimeter.server.http import build_app
    from perimeter.server.telemetry import Telemetry
    from perimeter.server.wiring import build_runtime

    settings = Settings.from_env()
    telemetry = Telemetry.configure(console=settings.trace_console)
    runtime = build_runtime(settings, telemetry=telemetry)
    host = args.host or settings.host
    port = args.port or settings.port
    log.info(
        "perimeter %s serving on %s:%d (embedder=%s store=%s index=%d)",
        runtime.version,
        host,
        port,
        runtime.embedder_name,
        runtime.store_name,
        runtime.index.size,
    )
    uvicorn.run(build_app(runtime), host=host, port=port, log_config=None)
    return 0


def _ingest(args: argparse.Namespace) -> int:
    from pathlib import Path

    from perimeter.connectors.base import documents_from
    from perimeter.connectors.filesystem import FilesystemConnector
    from perimeter.server.wiring import build_runtime

    settings = Settings.from_env()
    runtime = build_runtime(settings)
    connector = FilesystemConnector(Path(args.path))
    skipped: list[str] = []
    report = runtime.ingestor.ingest(
        documents_from(connector, on_skip=lambda ref: skipped.append(ref.uri))
    )
    runtime.index.flush()
    log.info(
        "ingested %d documents (%d chunks), skipped %d unchanged, %d unreadable; index=%d",
        report.documents,
        report.chunks,
        report.skipped_unchanged,
        len(skipped),
        runtime.index.size,
    )
    if runtime.store_name == "memory":
        log.warning(
            "store is in-memory: documents are not persisted across processes; "
            "set PERIMETER_DATABASE_URL, or ingest from the running server instead"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="perimeter", description="Permission-aware retrieval.")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the MCP + HTTP server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=_serve)
    ingest = sub.add_parser("ingest", help="ingest a directory of text files with ACL sidecars")
    ingest.add_argument("path")
    ingest.set_defaults(func=_ingest)
    args = parser.parse_args(argv)
    configure_logging()
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
