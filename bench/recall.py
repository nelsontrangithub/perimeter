"""recall@k of the quantized, filtered index against an exact float32 scan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from perimeter.core.acl import AccessPolicy, PermissionSet
from perimeter.core.document import ChunkId
from perimeter.core.ports import IndexEntry, as_vector
from perimeter.core.principal import EVERYONE, PrincipalId
from perimeter.index.filtered_search import filtered_search
from perimeter.index.flat import FlatIndex
from perimeter.index.quantize import F32

ALLOWED = PrincipalId("allowed")
CALLER = PermissionSet.of(PrincipalId("caller"), ALLOWED, EVERYONE)


def build_index(index: FlatIndex, corpus: F32, permitted_mask: np.ndarray) -> None:
    """Load ``corpus`` into ``index``; rows where ``permitted_mask`` is False admit nobody."""
    granted = AccessPolicy(frozenset({ALLOWED}))
    nobody = AccessPolicy.nobody()
    index.add(
        IndexEntry(ChunkId(f"c{i}"), as_vector(corpus[i]), granted if permitted_mask[i] else nobody)
        for i in range(corpus.shape[0])
    )
    index.flush()


def exact_top_k(corpus: F32, permitted_mask: np.ndarray, query: F32, k: int) -> set[int]:
    rows = np.flatnonzero(permitted_mask)
    scores = corpus[rows] @ query
    top = np.argpartition(-scores, min(k, rows.shape[0]) - 1)[:k]
    return set(rows[top].tolist())


@dataclass(frozen=True, slots=True)
class RecallResult:
    k: int
    queries: int
    permitted_fraction: float
    recall: float


def recall_at_k(
    index: FlatIndex, corpus: F32, queries: F32, permitted_mask: np.ndarray, *, k: int = 10
) -> RecallResult:
    hits_total = 0
    for q in queries:
        exact = exact_top_k(corpus, permitted_mask, q, k)
        got = {int(h.chunk_id[1:]) for h in filtered_search(index, as_vector(q), CALLER, k)}
        hits_total += len(exact & got)
    return RecallResult(
        k=k,
        queries=int(queries.shape[0]),
        permitted_fraction=float(permitted_mask.mean()),
        recall=hits_total / (k * queries.shape[0]),
    )
