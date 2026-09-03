"""Connector configuration registry.

Configuration is a small JSON file under the data directory. It holds names,
kinds, and non-secret settings (a filesystem root). It never holds a token:
Google Drive is configured by name only and receives its credential per
request (ADR-005).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from perimeter.connectors.base import Connector, documents_from
from perimeter.connectors.filesystem import FilesystemConnector
from perimeter.connectors.gdrive import GoogleDriveConnector
from perimeter.core.errors import ConnectorError, PerimeterError
from perimeter.index.flat import FlatIndex
from perimeter.pipeline.ingest import Ingestor

Kind = Literal["filesystem", "gdrive"]
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True, slots=True)
class ConnectorConfig:
    name: str
    kind: Kind
    root: str | None = None


@dataclass(frozen=True, slots=True)
class IngestRun:
    started_at: str
    duration_seconds: float
    documents: int
    chunks: int
    skipped_unchanged: int
    unreadable: int
    error: str | None


class ConnectorRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._configs: dict[str, ConnectorConfig] = {}
        self._last_runs: dict[str, IngestRun] = {}
        self._load()

    # -- persistence ---------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ConnectorError(f"connectors file unreadable ({type(exc).__name__})") from None
        if not isinstance(raw, list):
            raise ConnectorError("connectors file must be a list")
        for item in raw:
            if isinstance(item, dict):
                self._configs[str(item["name"])] = ConnectorConfig(
                    name=str(item["name"]), kind=item["kind"], root=item.get("root")
                )

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps([asdict(c) for c in self._configs.values()], indent=2))

    # -- configuration ---------------------------------------------------------------

    def list(self) -> list[ConnectorConfig]:
        return list(self._configs.values())

    def get(self, name: str) -> ConnectorConfig:
        try:
            return self._configs[name]
        except KeyError:
            raise ConnectorError(f"no connector named {name!r}") from None

    def last_run(self, name: str) -> IngestRun | None:
        return self._last_runs.get(name)

    def add(self, config: ConnectorConfig) -> None:
        if not _NAME.match(config.name):
            raise ConnectorError("connector name must be lowercase letters, digits, '-' or '_'")
        if config.name in self._configs:
            raise ConnectorError(f"connector {config.name!r} already exists")
        if config.kind == "filesystem":
            if not config.root:
                raise ConnectorError("filesystem connector requires a root directory")
            if not Path(config.root).expanduser().is_dir():
                raise ConnectorError("filesystem root is not a directory")
        elif config.kind != "gdrive":
            raise ConnectorError("unknown connector kind")
        self._configs[config.name] = config
        self._save()

    def remove(self, name: str) -> None:
        self.get(name)
        del self._configs[name]
        self._last_runs.pop(name, None)
        self._save()

    def build(self, name: str) -> Connector:
        config = self.get(name)
        if config.kind == "filesystem":
            return FilesystemConnector(Path(config.root or "").expanduser())
        return GoogleDriveConnector()

    # -- ingestion --------------------------------------------------------------------

    def ingest(self, name: str, ingestor: Ingestor, index: FlatIndex) -> IngestRun:
        """Run one ingestion for ``name`` inside the current request scope; record the result."""
        connector = self.build(name)
        started = datetime.now(tz=UTC).isoformat(timespec="seconds")
        t0 = time.perf_counter()
        skipped: list[str] = []
        error: str | None = None
        documents = chunks = unchanged = 0
        try:
            report = ingestor.ingest(
                documents_from(connector, on_skip=lambda ref: skipped.append(ref.uri))
            )
            documents, chunks, unchanged = report.documents, report.chunks, report.skipped_unchanged
        except PerimeterError as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            index.flush()
        run = IngestRun(
            started_at=started,
            duration_seconds=round(time.perf_counter() - t0, 3),
            documents=documents,
            chunks=chunks,
            skipped_unchanged=unchanged,
            unreadable=len(skipped),
            error=error,
        )
        self._last_runs[name] = run
        return run
