"""Allow-list filtering applied inside the scan (ADR-001, INV-2, INV-4).

:func:`permitted_rows` turns the caller's permission set into the exact set of
index rows whose policy admits it, using the same semantics as
:meth:`perimeter.core.acl.AccessPolicy.admits` (a property test asserts the two
agree on every generated policy). :func:`filtered_search` hands *only* those
rows to the scan. There is no code path that scans a row outside that set, and
an empty permission set returns before any scan happens.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from perimeter.core.acl import PermissionSet
from perimeter.core.document import ChunkId
from perimeter.core.ports import IndexHit, Vector
from perimeter.index.flat import AclTable
from perimeter.index.quantize import F32, I32


class Scannable(Protocol):
    """What filtered search needs from an index: its ACL table and a row-restricted scan."""

    @property
    def acl(self) -> AclTable: ...

    def scan_rows(self, query: F32, rows: I32, k: int) -> tuple[I32, F32]: ...

    def chunk_id_at(self, row: int) -> ChunkId: ...


def _rows_with_any_hit(indptr: I32, indices: I32, wanted: I32, n_rows: int) -> np.ndarray:
    """Boolean mask over rows: does the row's CSR segment contain any of ``wanted``?"""
    mask = np.zeros(n_rows, dtype=bool)
    if indices.shape[0] == 0 or wanted.shape[0] == 0:
        return mask
    hit = np.isin(indices, wanted)
    row_of_entry = np.repeat(np.arange(n_rows, dtype=np.int32), np.diff(indptr))
    mask[row_of_entry[hit]] = True
    return mask


def permitted_rows(acl: AclTable, permitted: PermissionSet) -> I32:
    """Rows whose policy admits ``permitted``, ascending. Empty set in, empty rows out."""
    empty = np.zeros((0,), dtype=np.int32)
    if permitted.is_empty or acl.rows == 0:
        return empty
    vocab = {p: i for i, p in enumerate(acl.principals)}
    wanted = np.asarray([vocab[p] for p in permitted if p in vocab], dtype=np.int32)
    if wanted.shape[0] == 0:
        return empty
    allowed = _rows_with_any_hit(acl.allow_indptr, acl.allow_indices, wanted, acl.rows)
    denied = _rows_with_any_hit(acl.deny_indptr, acl.deny_indices, wanted, acl.rows)
    out: I32 = np.flatnonzero(allowed & ~denied).astype(np.int32)
    return out


def filtered_search(
    index: Scannable, query: Vector, permitted: PermissionSet, k: int
) -> list[IndexHit]:
    """Top-``k`` over the caller's permitted rows only."""
    rows = permitted_rows(index.acl, permitted)
    if rows.shape[0] == 0:
        return []
    q = np.frombuffer(query, dtype=np.float32)
    hit_rows, scores = index.scan_rows(q, rows, k)
    return [
        IndexHit(chunk_id=index.chunk_id_at(int(r)), score=float(s))
        for r, s in zip(hit_rows.tolist(), scores.tolist(), strict=True)
    ]
