"""Process configuration from the environment. Secrets are excluded from repr."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from perimeter.core.errors import InvalidRequestError

DEFAULT_ALLOWED_HOSTS = ("127.0.0.1:*", "localhost:*", "[::1]:*")


@dataclass(frozen=True, slots=True)
class Settings:
    data_dir: Path
    cohere_api_key: str | None = field(default=None, repr=False)
    database_url: str | None = field(default=None, repr=False)
    acl_ttl_seconds: float = 60.0
    embedding_dimension: int = 1024
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    groups_file: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    trace_console: bool = False
    admin_dist: Path | None = None

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, default_data_dir: Path | None = None
    ) -> Settings:
        e = os.environ if env is None else env
        data_dir = Path(e.get("PERIMETER_DATA_DIR") or (default_data_dir or Path.cwd() / "data"))
        try:
            ttl = float(e.get("PERIMETER_ACL_TTL_SECONDS", "60"))
            dim = int(e.get("PERIMETER_EMBEDDING_DIMENSION", "1024"))
            port = int(e.get("PERIMETER_PORT", "8000"))
        except ValueError:
            raise InvalidRequestError("numeric setting is not a number") from None
        if ttl < 0:
            raise InvalidRequestError("PERIMETER_ACL_TTL_SECONDS must be >= 0")
        if dim <= 0 or dim % 8:
            raise InvalidRequestError(
                "PERIMETER_EMBEDDING_DIMENSION must be a positive multiple of 8"
            )
        hosts_raw = e.get("PERIMETER_ALLOWED_HOSTS", "")
        hosts = tuple(h.strip() for h in hosts_raw.split(",") if h.strip()) or DEFAULT_ALLOWED_HOSTS
        groups = e.get("PERIMETER_GROUPS_FILE")
        admin = e.get("PERIMETER_ADMIN_DIST")
        return cls(
            data_dir=data_dir,
            cohere_api_key=e.get("COHERE_API_KEY") or None,
            database_url=e.get("PERIMETER_DATABASE_URL") or None,
            acl_ttl_seconds=ttl,
            embedding_dimension=dim,
            allowed_hosts=hosts,
            groups_file=Path(groups) if groups else None,
            host=e.get("PERIMETER_HOST", "127.0.0.1"),
            port=port,
            log_level=e.get("PERIMETER_LOG_LEVEL", "INFO").upper(),
            trace_console=e.get("PERIMETER_TRACE_CONSOLE", "").lower() in ("1", "true", "yes"),
            admin_dist=Path(admin) if admin else None,
        )
