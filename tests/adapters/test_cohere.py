"""Cohere adapters over a fake httpx transport. No network."""

from __future__ import annotations

import json
from array import array
from collections.abc import Callable

import httpx
import pytest

from perimeter.adapters.cohere_embeddings import CohereEmbeddings
from perimeter.adapters.cohere_rerank import CohereReranker
from perimeter.core.acl import AccessPolicy
from perimeter.core.document import Chunk, Document, DocumentId, SourceRef
from perimeter.core.errors import EmbeddingError, RerankError
from perimeter.core.ports import EmbeddingModel, Reranker

DIM = 8
Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.cohere.com")


def _embed_ok(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    n = len(body["texts"])
    vec = [float(i) for i in range(DIM)]
    return httpx.Response(200, json={"embeddings": {"float": [vec] * n}})


def _chunks(n: int) -> list[Chunk]:
    doc = Document.create(
        id=DocumentId("d"),
        source=SourceRef("fs", "file:///d", "D"),
        policy=AccessPolicy.public(),
        text=" ".join(f"w{i}" for i in range(n)),
    )
    out = []
    pos = 0
    for i in range(n):
        word = f"w{i}"
        out.append(Chunk.from_document(doc, ordinal=i, start=pos, end=pos + len(word)))
        pos += len(word) + 1
    return out


# --- embeddings ------------------------------------------------------------


def test_embeddings_satisfy_port_and_return_float32_arrays() -> None:
    model = CohereEmbeddings(api_key="k", client=_client(_embed_ok), dimension=DIM)
    assert isinstance(model, EmbeddingModel)
    vecs = model.embed_documents(["a", "b"])
    assert len(vecs) == 2
    assert isinstance(vecs[0], array) and vecs[0].typecode == "f"
    assert list(vecs[0]) == [float(i) for i in range(DIM)]
    assert model.dimension == DIM


def test_embed_documents_sends_expected_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _embed_ok(request)

    model = CohereEmbeddings(api_key="secret-key", client=_client(handler), dimension=DIM)
    model.embed_documents(["hello"])
    req = seen[0]
    assert req.url.path == "/v2/embed"
    assert req.headers["authorization"] == "Bearer secret-key"
    body = json.loads(req.content)
    assert body["model"] == "embed-v4.0"
    assert body["input_type"] == "search_document"
    assert body["embedding_types"] == ["float"]
    assert body["output_dimension"] == DIM
    assert body["texts"] == ["hello"]


def test_embed_query_uses_search_query_input_type() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _embed_ok(request)

    CohereEmbeddings(api_key="k", client=_client(handler), dimension=DIM).embed_query("q")
    assert seen[0]["input_type"] == "search_query"


def test_embed_documents_batches_requests() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(request.content)["texts"]))
        return _embed_ok(request)

    model = CohereEmbeddings(api_key="k", client=_client(handler), dimension=DIM, batch_size=4)
    out = model.embed_documents([f"t{i}" for i in range(10)])
    assert len(out) == 10
    assert sizes == [4, 4, 2]


def test_embed_documents_empty_input_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert (
        CohereEmbeddings(api_key="k", client=_client(handler), dimension=DIM).embed_documents([])
        == []
    )


def test_embed_http_error_raises_typed_error_without_text_or_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "rate limited"})

    model = CohereEmbeddings(api_key="secret-key", client=_client(handler), dimension=DIM)
    with pytest.raises(EmbeddingError) as info:
        model.embed_documents(["VERY PRIVATE TEXT"])
    assert "VERY PRIVATE" not in str(info.value)
    assert "secret-key" not in str(info.value)
    assert "429" in str(info.value)


def test_embed_transport_error_raises_typed_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(EmbeddingError):
        CohereEmbeddings(api_key="k", client=_client(handler), dimension=DIM).embed_query("q")


def test_embed_rejects_wrong_dimension_from_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": {"float": [[1.0, 2.0]]}})

    with pytest.raises(EmbeddingError):
        CohereEmbeddings(api_key="k", client=_client(handler), dimension=DIM).embed_query("q")


def test_embed_rejects_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": 1})

    with pytest.raises(EmbeddingError):
        CohereEmbeddings(api_key="k", client=_client(handler), dimension=DIM).embed_query("q")


def test_embeddings_repr_hides_api_key() -> None:
    model = CohereEmbeddings(api_key="secret-key", client=_client(_embed_ok), dimension=DIM)
    assert "secret-key" not in repr(model)


# --- rerank ----------------------------------------------------------------


def test_reranker_satisfies_port_and_maps_indices_to_chunk_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v2/rerank"
        assert body["model"] == "rerank-v4.0-fast"
        assert body["query"] == "q"
        assert body["top_n"] == 2
        assert body["documents"] == ["w0", "w1", "w2"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.4},
                ]
            },
        )

    reranker = CohereReranker(api_key="k", client=_client(handler))
    assert isinstance(reranker, Reranker)
    chunks = _chunks(3)
    hits = reranker.rerank("q", chunks, k=2)
    assert [h.chunk_id for h in hits] == [chunks[2].id, chunks[0].id]
    assert hits[0].score == pytest.approx(0.9)


def test_rerank_empty_candidates_makes_no_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert CohereReranker(api_key="k", client=_client(handler)).rerank("q", [], k=5) == []


def test_rerank_http_error_raises_typed_error_without_chunk_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    with pytest.raises(RerankError) as info:
        CohereReranker(api_key="k", client=_client(handler)).rerank("q", _chunks(2), k=1)
    assert "w0" not in str(info.value)


def test_rerank_rejects_out_of_range_index() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 7, "relevance_score": 0.1}]})

    with pytest.raises(RerankError):
        CohereReranker(api_key="k", client=_client(handler)).rerank("q", _chunks(2), k=1)
