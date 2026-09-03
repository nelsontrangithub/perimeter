"""Wall and monotonic clocks behind the Clock port."""

from __future__ import annotations

import time
from datetime import UTC, datetime


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)

    def monotonic(self) -> float:
        return time.monotonic()
