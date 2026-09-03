"""INV-3: connector OAuth tokens never enter logs, traces, error messages, or
persistent storage. They live in request scope and are dropped when the request ends."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from perimeter.adapters.memory_store import MemoryStore
from perimeter.core.acl import AccessPolicy, PermissionSet
from perimeter.core.document import DocumentId, SourceRef
from perimeter.core.errors import AuthError, ConnectorError
from perimeter.core.principal import EVERYONE, PrincipalId
from perimeter.index.flat import FlatIndex
from perimeter.pipeline.ingest import Ingestor, RawDocument
from perimeter.server.auth import TOKEN_HEADER_PREFIX, current_tokens, request_scope, token_for
from perimeter.server.logging import RedactionFilter
from tests.pipeline.fakes import DIM, BagEmbedder

pytestmark = pytest.mark.invariant

SECRET = "ya29.A0ARrdaM-super-secret-token-value-1234567890"


def _connector_fetch() -> Iterator[RawDocument]:
    """A stand-in connector: uses the token, logs (badly), and may fail (badly)."""
    token = token_for("gdrive")
    assert token is not None
    logging.getLogger("perimeter.connectors.fake").info(
        "fetching with %r / %s", token, token.reveal()
    )
    yield RawDocument(
        id=DocumentId("doc"),
        source=SourceRef("gdrive", "https://drive.example/doc", "Doc", version="1"),
        policy=AccessPolicy.public(),
        text=f"content fetched using a token of length {len(token.reveal())}",
    )
    raise ConnectorError("upstream said no")


def test_inv3_connector_token_never_persisted_or_logged(tmp_path: Path) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactionFilter())
    root = logging.getLogger()
    root.addHandler(handler)
    previous = root.level
    root.setLevel(logging.DEBUG)

    store, embedder = MemoryStore(), BagEmbedder()
    index = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    try:
        with request_scope({f"{TOKEN_HEADER_PREFIX}gdrive": SECRET}) as creds:
            with pytest.raises(ConnectorError) as info:
                Ingestor(store=store, index=index, embedder=embedder).ingest(_connector_fetch())
            root.exception("ingest failed", exc_info=info.value)
            held = creds.tokens["gdrive"]
    finally:
        root.removeHandler(handler)
        root.setLevel(previous)

    # 1. Not in logs, whatever the log call did.
    assert SECRET not in stream.getvalue()
    # 2. Not in error messages or their repr.
    assert SECRET not in str(info.value) and SECRET not in repr(info.value)
    # 3. Not in persistent storage: nothing under the index directory or in the store.
    index.flush()
    for file in (tmp_path / "idx").rglob("*"):
        if file.is_file():
            assert SECRET.encode() not in file.read_bytes()
    everyone = PermissionSet.of(PrincipalId("x"), EVERYONE)
    for doc in store.list_documents(everyone, limit=100):
        assert SECRET not in doc.text
    chunk_ids = [index.chunk_id_at(r) for r in range(index.size)]
    assert chunk_ids, "the connector did yield a document before failing"
    assert all(SECRET not in c.text for c in store.get_chunks(chunk_ids, everyone))
    # 4. Dropped when the request ends.
    assert current_tokens() == {}
    with pytest.raises(AuthError):
        held.reveal()
