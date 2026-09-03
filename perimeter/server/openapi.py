"""Export the admin API's OpenAPI document without running a server.

The admin console's typed client is generated from this document
(``make generate-client``); CI regenerates it and fails if the committed
client is stale, so the console cannot drift from the API silently.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from perimeter.server.http import build_app
from perimeter.server.settings import Settings
from perimeter.server.wiring import build_runtime


def export_openapi(data_dir: Path) -> dict[str, Any]:
    runtime = build_runtime(Settings(data_dir=data_dir))
    return build_app(runtime).openapi()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        schema = export_openapi(Path(tmp))
    json.dump(schema, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
