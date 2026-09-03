"""OpenTelemetry tracing with sensitive data disabled."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from perimeter.core.acl import AccessPolicy
from perimeter.core.principal import Principal, PrincipalId
from perimeter.core.query import RetrievalRequest
from perimeter.pipeline.retrieve import Retriever
from perimeter.server.auth import TOKEN_HEADER_PREFIX, request_scope
from perimeter.server.telemetry import ALLOWED_ATTRIBUTE_PREFIX, Telemetry, TracedRetriever
from tests.pipeline.fakes import StaticResolver, build_corpus, raw

SECRET = "ya29.A0ARrdaM-super-secret-token-value-1234567890"


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter
    provider.shutdown()


@pytest.fixture
def telemetry(exporter: InMemorySpanExporter) -> Telemetry:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Telemetry(provider=provider, record_sensitive_data=False)


def test_traced_retrieval_records_counts_but_never_query_or_principal(
    telemetry: Telemetry, exporter: InMemorySpanExporter
) -> None:
    store, index, embedder = build_corpus([raw("d", "apples apples", AccessPolicy.public())])
    retriever = TracedRetriever(
        Retriever(
            resolver=StaticResolver(), embedder=embedder, index=index, store=store, reranker=None
        ),
        telemetry,
    )
    request = RetrievalRequest(
        principal=Principal(id=PrincipalId("alice@example.com")), query="SECRET QUERY", k=3
    )
    result = retriever.retrieve(request)
    assert result.returned >= 1

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["perimeter.retrieve"]
    attrs = dict(spans[0].attributes or {})
    assert attrs["perimeter.k"] == 3
    assert attrs["perimeter.returned"] == result.returned
    assert attrs["perimeter.candidates"] == result.candidates
    assert attrs["perimeter.permitted"] is True
    dumped = repr(attrs)
    assert "SECRET QUERY" not in dumped
    assert "alice" not in dumped
    assert all(k.startswith(ALLOWED_ATTRIBUTE_PREFIX) for k in attrs)


def test_span_attributes_are_whitelisted_and_redacted(
    telemetry: Telemetry, exporter: InMemorySpanExporter
) -> None:
    attributes: dict[str, str | int | float | bool] = {
        "perimeter.count": 2,
        "perimeter.note": f"used {SECRET}",
        "http.request.header.authorization": "Bearer abc",
        "query": "text",
    }
    with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}):
        with telemetry.span("perimeter.test", **attributes):
            pass
    attrs = dict(exporter.get_finished_spans()[0].attributes or {})
    assert set(attrs) == {"perimeter.count", "perimeter.note"}
    assert SECRET not in attrs["perimeter.note"]  # type: ignore[operator]


def test_error_inside_span_is_recorded_by_type_only(
    telemetry: Telemetry, exporter: InMemorySpanExporter
) -> None:
    from perimeter.core.errors import ConnectorError

    with pytest.raises(ConnectorError):
        with telemetry.span("perimeter.failing"):
            raise ConnectorError(f"failed with {SECRET}")
    span = exporter.get_finished_spans()[0]
    attrs = dict(span.attributes or {})
    assert attrs["perimeter.error"] == "ConnectorError"
    assert SECRET not in repr(attrs)
    assert not span.events, "exception events would carry the message; they are disabled"


def test_noop_telemetry_is_safe_without_provider() -> None:
    t = Telemetry.disabled()
    with t.span("x", **{"perimeter.a": 1}):
        pass
