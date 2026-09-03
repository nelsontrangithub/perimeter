"""Structured logging with a redaction filter (INV-3 backstop).

Nothing in Perimeter should ever put a token in a log call, and the typed
errors and ``ConnectorToken.__repr__`` make that hard to do by accident. The
filter here is the layer beneath that: it rewrites every record's message,
arguments, and exception text so that the exact values of the tokens
currently in request scope, and anything shaped like a bearer credential,
are replaced with ``[REDACTED]`` before a handler sees them.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Literal

from perimeter.server.auth import current_tokens

REDACTED = "[REDACTED]"

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*"),
    re.compile(r"\bya29\.[A-Za-z0-9\-._]+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16,}"),
    re.compile(r"(?i)\b(token|secret|password|passwd|api[_-]?key|authorization)(\s*[=:]\s*)\S+"),
)


def _scrub_patterns(text: str) -> str:
    text = _PATTERNS[0].sub(f"Bearer {REDACTED}", text)
    for pattern in _PATTERNS[1:4]:
        text = pattern.sub(REDACTED, text)
    return _PATTERNS[4].sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}", text)


def redact(text: str) -> str:
    """Replace in-scope token values and token-shaped strings with ``[REDACTED]``."""
    for token in sorted(current_tokens().values(), key=lambda t: -len(t.reveal())):
        secret = token.reveal()
        if secret:
            text = text.replace(secret, REDACTED)
    return _scrub_patterns(text)


class RedactionFilter(logging.Filter):
    """Rewrites the record in place. Attach to handlers, where every record passes."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        if record.exc_info and record.exc_info[0] is not None:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        if record.stack_info:
            record.stack_info = redact(record.stack_info)
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.now(tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_text:
            payload["exception"] = record.exc_text
        return json.dumps(payload, ensure_ascii=False)


_INSTALLED = "_perimeter_handler"


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
) -> logging.Logger:
    """Install one JSON handler with the redaction filter on the root logger. Idempotent."""
    root = logging.getLogger()
    root.setLevel(level)
    for handler in root.handlers:
        if getattr(handler, _INSTALLED, False):
            return root
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RedactionFilter())
    setattr(handler, _INSTALLED, True)
    root.addHandler(handler)
    return root
