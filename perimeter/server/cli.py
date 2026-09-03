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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="perimeter", description="Permission-aware retrieval.")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the MCP + HTTP server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=_serve)
    args = parser.parse_args(argv)
    configure_logging()
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
