"""Document store contract. Runs against the in-memory store always, and against
Postgres when PERIMETER_TEST_DATABASE_URL is set (CI provides one).

The contract is the security contract: every read takes a permission set and
returns nothing the policy does not admit (INV-1), and an empty set reads
nothing (INV-4).
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from perimeter.adapters.memory_store import MemoryStore
from perimeter.core.acl import AccessPolicy, Deny, Grant, PermissionSet
from perimeter.core.document import Chunk, ChunkId, Document, DocumentId, SourceRef
from perimeter.core.ports import DocumentStore
from perimeter.core.principal import EVERYONE, PrincipalId

ALICE = PrincipalId("alice")
BOB = PrincipalId("bob")
CONTRACTORS = PrincipalId("contractors")
AS_ALICE = PermissionSet.of(ALICE, EVERYONE)
AS_BOB = PermissionSet.of(BOB, EVERYONE)
AS_CONTRACTOR = PermissionSet.of(BOB, CONTRACTORS, EVERYONE)


def _doc(
    doc_id: str, policy: AccessPolicy, text: str = "alpha beta gamma delta"
) -> tuple[Document, list[Chunk]]:
    doc = Document.create(
        id=DocumentId(doc_id),
        source=SourceRef("fs", f"file:///{doc_id}", doc_id.upper(), version="1"),
        policy=policy,
        text=text,
    )
    words = text.split(" ")
    chunks = []
    pos = 0
    for i, w in enumerate(words):
        chunks.append(Chunk.from_document(doc, ordinal=i, start=pos, end=pos + len(w)))
        pos += len(w) + 1
    return doc, chunks


@pytest.fixture(params=["memory", "postgres"])
def store(request: pytest.FixtureRequest) -> Iterator[DocumentStore]:
    if request.param == "memory":
        yield MemoryStore()
        return
    url = os.environ.get("PERIMETER_TEST_DATABASE_URL")
    if not url:
        pytest.skip("PERIMETER_TEST_DATABASE_URL not set")
    from perimeter.adapters.postgres_store import PostgresStore

    pg = PostgresStore.connect(url)
    pg.create_schema()
    pg.truncate()
    try:
        yield pg
    finally:
        pg.truncate()
        pg.close()


def test_store_satisfies_port(store: DocumentStore) -> None:
    assert isinstance(store, DocumentStore)


def test_get_chunks_returns_only_admitted_in_requested_order(store: DocumentStore) -> None:
    doc, chunks = _doc("a", AccessPolicy.from_rules([Grant(ALICE)]))
    store.put(doc, chunks)
    ids = [chunks[2].id, chunks[0].id]
    got = store.get_chunks(ids, AS_ALICE)
    assert [c.id for c in got] == ids
    assert got[0].text == "gamma"
    assert store.get_chunks(ids, AS_BOB) == []


def test_get_chunks_empty_permission_set_returns_nothing_even_for_public(
    store: DocumentStore,
) -> None:
    doc, chunks = _doc("p", AccessPolicy.public())
    store.put(doc, chunks)
    assert store.get_chunks([c.id for c in chunks], PermissionSet.empty()) == []


def test_get_chunks_skips_unknown_ids(store: DocumentStore) -> None:
    doc, chunks = _doc("a", AccessPolicy.public())
    store.put(doc, chunks)
    got = store.get_chunks([ChunkId("nope#0"), chunks[1].id], AS_ALICE)
    assert [c.id for c in got] == [chunks[1].id]


def test_deny_excludes_chunks_and_document(store: DocumentStore) -> None:
    doc, chunks = _doc("d", AccessPolicy.from_rules([Grant(EVERYONE), Deny(CONTRACTORS)]))
    store.put(doc, chunks)
    assert store.get_chunks([chunks[0].id], AS_BOB) != []
    assert store.get_chunks([chunks[0].id], AS_CONTRACTOR) == []
    assert store.get_document(doc.id, AS_CONTRACTOR) is None


def test_get_document_respects_policy(store: DocumentStore) -> None:
    doc, chunks = _doc("a", AccessPolicy.from_rules([Grant(ALICE)]))
    store.put(doc, chunks)
    got = store.get_document(doc.id, AS_ALICE)
    assert got is not None and got.text == doc.text and got.content_hash == doc.content_hash
    assert store.get_document(doc.id, AS_BOB) is None
    assert store.get_document(DocumentId("missing"), AS_ALICE) is None


def test_reput_replaces_chunks(store: DocumentStore) -> None:
    doc, chunks = _doc("a", AccessPolicy.public(), text="one two three")
    store.put(doc, chunks)
    doc2, chunks2 = _doc("a", AccessPolicy.public(), text="uno")
    store.put(doc2, chunks2)
    assert store.count_documents() == 1
    assert store.count_chunks() == 1
    assert store.get_chunks([chunks[2].id], AS_ALICE) == []
    assert store.get_chunks([chunks2[0].id], AS_ALICE)[0].text == "uno"


def test_delete_removes_document_and_chunks(store: DocumentStore) -> None:
    doc, chunks = _doc("a", AccessPolicy.public())
    store.put(doc, chunks)
    store.delete(doc.id)
    assert store.count_documents() == 0
    assert store.count_chunks() == 0
    assert store.get_document(doc.id, AS_ALICE) is None
    store.delete(doc.id)  # idempotent


def test_list_documents_filters_by_permission_and_limits(store: DocumentStore) -> None:
    for i in range(3):
        d, c = _doc(f"pub{i}", AccessPolicy.public())
        store.put(d, c)
    d, c = _doc("private", AccessPolicy.from_rules([Grant(ALICE)]))
    store.put(d, c)
    as_bob = store.list_documents(AS_BOB, limit=10)
    assert sorted(x.id for x in as_bob) == ["pub0", "pub1", "pub2"]
    as_alice = store.list_documents(AS_ALICE, limit=10)
    assert len(as_alice) == 4
    assert len(store.list_documents(AS_ALICE, limit=2)) == 2
    assert store.list_documents(PermissionSet.empty(), limit=10) == []


def test_policy_and_source_roundtrip(store: DocumentStore) -> None:
    policy = AccessPolicy.from_rules([Grant(ALICE), Grant(BOB), Deny(CONTRACTORS)])
    doc, chunks = _doc("r", policy)
    store.put(doc, chunks)
    got = store.get_document(doc.id, AS_ALICE)
    assert got is not None
    assert got.policy == policy
    assert got.source == doc.source
    chunk = store.get_chunks([chunks[1].id], AS_ALICE)[0]
    assert chunk.policy == policy
    assert chunk.source == doc.source
    assert (chunk.ordinal, chunk.start, chunk.end) == (1, 6, 10)


def test_fingerprint_changes_with_text_or_policy_and_is_none_when_missing(
    store: DocumentStore,
) -> None:
    assert store.fingerprint(DocumentId("missing")) is None
    doc, chunks = _doc("f", AccessPolicy.public(), text="one")
    store.put(doc, chunks)
    fp = store.fingerprint(doc.id)
    assert fp == doc.fingerprint
    doc2, chunks2 = _doc("f", AccessPolicy.public(), text="two")
    store.put(doc2, chunks2)
    assert store.fingerprint(doc.id) != fp
    doc3, chunks3 = _doc("f", AccessPolicy.from_rules([Grant(ALICE)]), text="two")
    store.put(doc3, chunks3)
    assert store.fingerprint(doc.id) not in (fp, doc2.fingerprint)
