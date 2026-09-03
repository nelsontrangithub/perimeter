"""The performance budget. Build gates, not aspirations (CLAUDE.md, "Performance budget").

These constants are the single source of truth: the gate test asserts them and
the results table prints them. Changing a ceiling is a visible, reviewable diff.
"""

from __future__ import annotations

P95_LATENCY_MS = 30.0
PEAK_RSS_MIB = 512.0
INDEX_BYTES_PER_CHUNK = 1280.0
RECALL_AT_10 = 0.95
COHERE_CALLS_PER_QUERY = 2.0
