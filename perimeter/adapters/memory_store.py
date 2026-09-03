"""In-memory document store: tests and the air-gapped demo.

Every read applies the caller's permission set before any text leaves the
store (INV-1). An empty permission set reads nothing (INV-4).
"""

from __future__ import annotations

from collections.abc import Sequence

from perimeter.core.acl import PermissionSet
from perimeter.core.document import Chunk, ChunkId, Document, DocumentId


class MemoryStore:
    def __init__(self) -> None:
        self._documents: dict[DocumentId, Document] = {}
        self._chunks: dict[ChunkId, Chunk] = {}
        self._chunks_by_document: dict[DocumentId, list[ChunkId]] = {}

    def put(self, document: Document, chunks: Sequence[Chunk]) -> None:
        self.delete(document.id)
        self._documents[document.id] = document
        ids: list[ChunkId] = []
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            ids.append(chunk.id)
        self._chunks_by_document[document.id] = ids

    def fingerprint(self, id: DocumentId) -> str | None:
        doc = self._documents.get(id)
        return None if doc is None else doc.fingerprint

    def get_chunks(self, ids: Sequence[ChunkId], permitted: PermissionSet) -> Sequence[Chunk]:
        if permitted.is_empty:
            return []
        out: list[Chunk] = []
        for cid in ids:
            chunk = self._chunks.get(cid)
            if chunk is not None and chunk.policy.admits(permitted):
                out.append(chunk)
        return out

    def get_document(self, id: DocumentId, permitted: PermissionSet) -> Document | None:
        if permitted.is_empty:
            return None
        doc = self._documents.get(id)
        if doc is None or not doc.policy.admits(permitted):
            return None
        return doc

    def list_documents(self, permitted: PermissionSet, *, limit: int) -> Sequence[Document]:
        if permitted.is_empty or limit <= 0:
            return []
        out: list[Document] = []
        for doc in self._documents.values():
            if doc.policy.admits(permitted):
                out.append(doc)
                if len(out) >= limit:
                    break
        return out

    def delete(self, id: DocumentId) -> None:
        for cid in self._chunks_by_document.pop(id, []):
            self._chunks.pop(cid, None)
        self._documents.pop(id, None)

    def count_documents(self) -> int:
        return len(self._documents)

    def count_chunks(self) -> int:
        return len(self._chunks)
