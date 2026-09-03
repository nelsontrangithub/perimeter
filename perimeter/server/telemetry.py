"""OpenTelemetry tracing with sensitive data disabled.

Spans carry only attributes under the ``perimeter.`` prefix, and string
values pass through the log redaction filter. Query text, principal IDs,
document text, and headers never become span attributes; exceptions are
recorded by type name only, never as events with messages. With
``record_sensitive_data=False`` (the only value the wiring uses) this is
enforced in :meth:`Telemetry.span` rather than left to call sites.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Span, StatusCode

from perimeter.core.acl import PermissionSet
from perimeter.core.query import RetrievalRequest, ScopedResult
from perimeter.pipeline.retrieve import Retriever
from perimeter.server.logging import redact

ALLOWED_ATTRIBUTE_PREFIX = "perimeter."
AttrValue = str | int | float | bool


@dataclass(slots=True)
class SpanHandle:
    _span: Span | None
    _sensitive_ok: bool

    def set(self, **attributes: AttrValue) -> None:
        if self._span is None:
            return
        for key, value in attributes.items():
            if not self._sensitive_ok and not key.startswith(ALLOWED_ATTRIBUTE_PREFIX):
                continue
            self._span.set_attribute(key, redact(value) if isinstance(value, str) else value)


class Telemetry:
    def __init__(
        self, *, provider: TracerProvider | None, record_sensitive_data: bool = False
    ) -> None:
        self._tracer = provider.get_tracer("perimeter") if provider is not None else None
        self._sensitive_ok = record_sensitive_data

    @classmethod
    def disabled(cls) -> Telemetry:
        return cls(provider=None)

    @classmethod
    def configure(cls, *, service_name: str = "perimeter", console: bool = False) -> Telemetry:
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if console:
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        return cls(provider=provider, record_sensitive_data=False)

    @contextmanager
    def span(self, name: str, **attributes: AttrValue) -> Iterator[SpanHandle]:
        if self._tracer is None:
            yield SpanHandle(None, self._sensitive_ok)
            return
        with self._tracer.start_as_current_span(
            name, record_exception=False, set_status_on_exception=False
        ) as span:
            handle = SpanHandle(span, self._sensitive_ok)
            handle.set(**attributes)
            try:
                yield handle
            except Exception as exc:
                span.set_attribute("perimeter.error", type(exc).__name__)
                span.set_status(StatusCode.ERROR)
                raise


class TracedRetriever:
    """Wraps a Retriever with one span per call carrying counts only."""

    def __init__(self, inner: Retriever, telemetry: Telemetry) -> None:
        self._inner = inner
        self._telemetry = telemetry

    def permissions_for(self, request: RetrievalRequest) -> PermissionSet:
        return self._inner.permissions_for(request)

    def retrieve(self, request: RetrievalRequest) -> ScopedResult:
        return self.retrieve_with(request, None)

    def retrieve_with(
        self, request: RetrievalRequest, permitted: PermissionSet | None
    ) -> ScopedResult:
        with self._telemetry.span("perimeter.retrieve", **{"perimeter.k": request.k}) as span:
            started = time.perf_counter()
            if permitted is None:
                permitted = self._inner.permissions_for(request)
            span.set(**{"perimeter.permitted": not permitted.is_empty})
            result = self._inner.retrieve_with(request, permitted)
            span.set(
                **{
                    "perimeter.returned": result.returned,
                    "perimeter.candidates": result.candidates,
                    "perimeter.duration_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            )
            return result
