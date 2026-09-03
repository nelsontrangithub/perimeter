"""Composition root: build every port implementation from Settings, once per process."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp.server.mcpserver import MCPServer

from perimeter import __version__
from perimeter.adapters.caching_acl_resolver import CachingAclResolver
from perimeter.adapters.clock import SystemClock
from perimeter.adapters.cohere_embeddings import CohereEmbeddings
from perimeter.adapters.cohere_rerank import CohereReranker
from perimeter.adapters.local_embeddings import LocalEmbeddings
from perimeter.adapters.memory_store import MemoryStore
from perimeter.adapters.static_acl_resolver import StaticAclResolver
from perimeter.core.ports import AclResolver, DocumentStore, EmbeddingModel, Reranker
from perimeter.index.flat import FlatIndex
from perimeter.pipeline.ingest import Ingestor
from perimeter.pipeline.retrieve import Retriever
from perimeter.server.connectors import ConnectorRegistry
from perimeter.server.mcp import build_mcp_server
from perimeter.server.settings import Settings
from perimeter.server.telemetry import Telemetry, TracedRetriever


@dataclass(slots=True)
class Runtime:
    settings: Settings
    store: DocumentStore
    store_name: str
    index: FlatIndex
    embedder: EmbeddingModel
    embedder_name: str
    reranker: Reranker | None
    resolver: AclResolver
    acl_cache: CachingAclResolver
    retriever: TracedRetriever
    ingestor: Ingestor
    telemetry: Telemetry
    mcp_server: MCPServer[Any]
    connectors: ConnectorRegistry
    version: str = __version__


def build_store(settings: Settings) -> tuple[DocumentStore, str]:
    if settings.database_url:
        from perimeter.adapters.postgres_store import PostgresStore

        store = PostgresStore.connect(settings.database_url)
        store.create_schema()
        return store, "postgres"
    return MemoryStore(), "memory"


def build_embedder(settings: Settings) -> tuple[EmbeddingModel, Reranker | None, str]:
    if settings.cohere_api_key:
        return (
            CohereEmbeddings(
                api_key=settings.cohere_api_key, dimension=settings.embedding_dimension
            ),
            CohereReranker(api_key=settings.cohere_api_key),
            "cohere",
        )
    return LocalEmbeddings(dimension=settings.embedding_dimension), None, "local"


def build_resolver(settings: Settings) -> CachingAclResolver:
    inner: AclResolver = (
        StaticAclResolver.from_file(settings.groups_file)
        if settings.groups_file
        else StaticAclResolver()
    )
    return CachingAclResolver(inner, clock=SystemClock(), ttl_seconds=settings.acl_ttl_seconds)


def build_runtime(settings: Settings, *, telemetry: Telemetry | None = None) -> Runtime:
    telemetry = telemetry or Telemetry.disabled()
    store, store_name = build_store(settings)
    embedder, reranker, embedder_name = build_embedder(settings)
    index = FlatIndex.open(settings.data_dir / "index", dimension=embedder.dimension)
    resolver = build_resolver(settings)
    retriever = TracedRetriever(
        Retriever(
            resolver=resolver, embedder=embedder, index=index, store=store, reranker=reranker
        ),
        telemetry,
    )
    ingestor = Ingestor(store=store, index=index, embedder=embedder)
    return Runtime(
        settings=settings,
        store=store,
        store_name=store_name,
        index=index,
        embedder=embedder,
        embedder_name=embedder_name,
        reranker=reranker,
        resolver=resolver,
        acl_cache=resolver,
        retriever=retriever,
        ingestor=ingestor,
        telemetry=telemetry,
        mcp_server=build_mcp_server(retriever),
        connectors=ConnectorRegistry(settings.data_dir / "connectors.json"),
    )
