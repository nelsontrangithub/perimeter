"""Postgres document store (psycopg 3).

The permission check runs twice: once in SQL (``grants && caller AND NOT
(denies && caller)``) so unpermitted rows are never transferred, and once in
Python via :meth:`AccessPolicy.admits` on whatever comes back, so the SQL
predicate and the core semantics cannot silently diverge. An empty permission
set returns before any query is issued (INV-4).

Policies are denormalised onto chunk rows as text arrays, so a chunk read is a
single indexed lookup with no join for the security check.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import psycopg

from perimeter.core.acl import AccessPolicy, PermissionSet
from perimeter.core.document import Chunk, ChunkId, Document, DocumentId, SourceRef
from perimeter.core.errors import StoreError
from perimeter.core.principal import PrincipalId

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    connector     TEXT NOT NULL,
    uri           TEXT NOT NULL,
    title         TEXT NOT NULL,
    version       TEXT,
    content_hash  TEXT NOT NULL,
    grants        TEXT[] NOT NULL,
    denies        TEXT[] NOT NULL,
    body          TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal       INTEGER NOT NULL,
    span_start    INTEGER NOT NULL,
    span_end      INTEGER NOT NULL,
    body          TEXT NOT NULL,
    grants        TEXT[] NOT NULL,
    denies        TEXT[] NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
"""

_SELECT_CHUNKS = (
    "SELECT c.id, c.document_id, c.ordinal, c.span_start, c.span_end, c.body, c.grants,"
    " c.denies, d.connector, d.uri, d.title, d.version"
    " FROM chunks c JOIN documents d ON d.id = c.document_id"
    " WHERE c.id = ANY(%s) AND c.grants && %s AND NOT (c.denies && %s)"
)
_SELECT_DOCUMENT = (
    "SELECT id, connector, uri, title, version, content_hash, grants, denies, body"
    " FROM documents WHERE id = %s AND grants && %s AND NOT (denies && %s)"
)
_SELECT_DOCUMENTS = (
    "SELECT id, connector, uri, title, version, content_hash, grants, denies, body"
    " FROM documents WHERE grants && %s AND NOT (denies && %s) ORDER BY id LIMIT %s"
)
_INSERT_DOCUMENT = (
    "INSERT INTO documents"
    " (id, connector, uri, title, version, content_hash, grants, denies, body)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
)
_INSERT_CHUNK = (
    "INSERT INTO chunks (id, document_id, ordinal, span_start, span_end, body, grants, denies)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
)


class PostgresStore:
    def __init__(self, conn: psycopg.Connection[tuple[Any, ...]]) -> None:
        self._conn = conn

    @classmethod
    def connect(cls, url: str) -> PostgresStore:
        try:
            return cls(psycopg.connect(url))
        except psycopg.Error as exc:
            raise StoreError(f"postgres: cannot connect ({type(exc).__name__})") from None

    def close(self) -> None:
        self._conn.close()

    def create_schema(self) -> None:
        self._run(SCHEMA)

    def truncate(self) -> None:
        self._run("TRUNCATE chunks, documents")

    def _run(self, sql: str, params: Any = None) -> None:
        try:
            with self._conn.transaction():
                self._conn.execute(sql, params)
        except psycopg.Error as exc:
            raise StoreError(f"postgres: {type(exc).__name__}") from None

    def _fetch(self, sql: str, params: Any) -> list[tuple[Any, ...]]:
        try:
            with self._conn.transaction():
                return self._conn.execute(sql, params).fetchall()
        except psycopg.Error as exc:
            raise StoreError(f"postgres: {type(exc).__name__}") from None

    # -- writes ----------------------------------------------------------------

    def put(self, document: Document, chunks: Sequence[Chunk]) -> None:
        try:
            with self._conn.transaction():
                self._conn.execute("DELETE FROM documents WHERE id = %s", (document.id,))
                self._conn.execute(
                    _INSERT_DOCUMENT,
                    (
                        document.id,
                        document.source.connector,
                        document.source.uri,
                        document.source.title,
                        document.source.version,
                        document.content_hash,
                        sorted(document.policy.grants),
                        sorted(document.policy.denies),
                        document.text,
                    ),
                )
                with self._conn.cursor() as cur:
                    cur.executemany(
                        _INSERT_CHUNK,
                        [
                            (
                                c.id,
                                c.document_id,
                                c.ordinal,
                                c.start,
                                c.end,
                                c.text,
                                sorted(c.policy.grants),
                                sorted(c.policy.denies),
                            )
                            for c in chunks
                        ],
                    )
        except psycopg.Error as exc:
            raise StoreError(f"postgres: put failed ({type(exc).__name__})") from None

    def delete(self, id: DocumentId) -> None:
        self._run("DELETE FROM documents WHERE id = %s", (id,))

    # -- reads -----------------------------------------------------------------

    def get_chunks(self, ids: Sequence[ChunkId], permitted: PermissionSet) -> Sequence[Chunk]:
        if permitted.is_empty or not ids:
            return []
        caller = sorted(permitted.principals)
        rows = self._fetch(_SELECT_CHUNKS, (list(ids), caller, caller))
        by_id = {}
        for row in rows:
            chunk = self._chunk_from_row(row)
            if chunk.policy.admits(permitted):
                by_id[chunk.id] = chunk
        return [by_id[cid] for cid in ids if cid in by_id]

    def get_document(self, id: DocumentId, permitted: PermissionSet) -> Document | None:
        if permitted.is_empty:
            return None
        caller = sorted(permitted.principals)
        rows = self._fetch(_SELECT_DOCUMENT, (id, caller, caller))
        if not rows:
            return None
        doc = self._document_from_row(rows[0])
        return doc if doc.policy.admits(permitted) else None

    def list_documents(self, permitted: PermissionSet, *, limit: int) -> Sequence[Document]:
        if permitted.is_empty or limit <= 0:
            return []
        caller = sorted(permitted.principals)
        rows = self._fetch(_SELECT_DOCUMENTS, (caller, caller, limit))
        docs = [self._document_from_row(r) for r in rows]
        return [d for d in docs if d.policy.admits(permitted)]

    def count_documents(self) -> int:
        return int(self._fetch("SELECT count(*) FROM documents", None)[0][0])

    def count_chunks(self) -> int:
        return int(self._fetch("SELECT count(*) FROM chunks", None)[0][0])

    # -- mapping ---------------------------------------------------------------

    @staticmethod
    def _policy(grants: list[str], denies: list[str]) -> AccessPolicy:
        return AccessPolicy(
            frozenset(PrincipalId(g) for g in grants), frozenset(PrincipalId(d) for d in denies)
        )

    @classmethod
    def _chunk_from_row(cls, row: tuple[Any, ...]) -> Chunk:
        (cid, doc_id, ordinal, start, end, body, grants, denies, connector, uri, title, version) = (
            row
        )
        return Chunk(
            id=ChunkId(cid),
            document_id=DocumentId(doc_id),
            ordinal=int(ordinal),
            start=int(start),
            end=int(end),
            text=body,
            policy=cls._policy(grants, denies),
            source=SourceRef(connector=connector, uri=uri, title=title, version=version),
        )

    @classmethod
    def _document_from_row(cls, row: tuple[Any, ...]) -> Document:
        (doc_id, connector, uri, title, version, content_hash, grants, denies, body) = row
        return Document(
            id=DocumentId(doc_id),
            source=SourceRef(connector=connector, uri=uri, title=title, version=version),
            policy=cls._policy(grants, denies),
            text=body,
            content_hash=content_hash,
        )
