"""Memory-mapped flat index.

On disk an index is a directory:

    meta.json     format, dimension, row count
    binary.bin    uint8 [count, dim/8]   sign-bit codes, scanned on every query
    int8.bin      int8  [count, dim]     affine codes, used to rescore candidates
    ids.json      chunk IDs, row order
    acl.npz       CSR-style allow/deny rows over a principal vocabulary
    quant.npz     int8 quantizer parameters

No text, ever (ADR-006). ``open`` memory-maps the code arrays read-only; the
operating system's page cache is the memory strategy (ADR-002).

Writes are staged in memory by :meth:`FlatIndex.add` and made durable by
:meth:`FlatIndex.flush`, which appends to the existing rows, rewrites the
directory, and re-maps. The int8 quantizer is fitted on the first flush and
reused afterwards so codes stay comparable across appends.

Searching is split in two so the security-relevant half is isolated:
:meth:`scan_rows` scores an *explicit* set of rows and nothing else;
:mod:`perimeter.index.filtered_search` decides which rows those are.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from perimeter.core.acl import AccessPolicy, PermissionSet
from perimeter.core.document import ChunkId
from perimeter.core.errors import VectorIndexError
from perimeter.core.ports import IndexEntry, IndexHit, Vector
from perimeter.core.principal import PrincipalId
from perimeter.index.quantize import (
    F32,
    I8,
    I32,
    U8,
    Int8Params,
    Int8Quantizer,
    binarize,
    hamming_distances,
    l2_normalize,
)

FORMAT = "perimeter-flat-v1"
DEFAULT_RESCORE_MULTIPLIER = 64
"""Binary-scan candidates rescored per requested result. A recall knob, not a security one."""


@dataclass(frozen=True, slots=True)
class AclTable:
    """Per-row allow and deny lists in CSR form over a principal vocabulary.

    ``allow_indices[allow_indptr[r]:allow_indptr[r+1]]`` are the vocabulary
    indices granted on row ``r``; likewise for denies.
    """

    principals: tuple[str, ...]
    allow_indptr: I32
    allow_indices: I32
    deny_indptr: I32
    deny_indices: I32

    @classmethod
    def build(cls, policies: Sequence[AccessPolicy]) -> AclTable:
        vocab: dict[str, int] = {}

        def idx(p: str) -> int:
            if p not in vocab:
                vocab[p] = len(vocab)
            return vocab[p]

        allow_ptr = [0]
        allow_idx: list[int] = []
        deny_ptr = [0]
        deny_idx: list[int] = []
        for policy in policies:
            allow_idx.extend(idx(p) for p in sorted(policy.grants))
            allow_ptr.append(len(allow_idx))
            deny_idx.extend(idx(p) for p in sorted(policy.denies))
            deny_ptr.append(len(deny_idx))
        return cls(
            principals=tuple(vocab),
            allow_indptr=np.asarray(allow_ptr, dtype=np.int32),
            allow_indices=np.asarray(allow_idx, dtype=np.int32),
            deny_indptr=np.asarray(deny_ptr, dtype=np.int32),
            deny_indices=np.asarray(deny_idx, dtype=np.int32),
        )

    @property
    def rows(self) -> int:
        return int(self.allow_indptr.shape[0]) - 1

    def policy_at(self, row: int) -> AccessPolicy:
        a0, a1 = int(self.allow_indptr[row]), int(self.allow_indptr[row + 1])
        d0, d1 = int(self.deny_indptr[row]), int(self.deny_indptr[row + 1])
        grants = frozenset(PrincipalId(self.principals[i]) for i in self.allow_indices[a0:a1])
        denies = frozenset(PrincipalId(self.principals[i]) for i in self.deny_indices[d0:d1])
        return AccessPolicy(grants, denies)

    def policies(self) -> list[AccessPolicy]:
        return [self.policy_at(r) for r in range(self.rows)]

    def save(self, path: Path) -> None:
        np.savez(
            path,
            principals=np.asarray(self.principals, dtype=np.str_),
            allow_indptr=self.allow_indptr,
            allow_indices=self.allow_indices,
            deny_indptr=self.deny_indptr,
            deny_indices=self.deny_indices,
        )

    @classmethod
    def load(cls, path: Path) -> AclTable:
        with np.load(path) as z:
            return cls(
                principals=tuple(str(p) for p in z["principals"]),
                allow_indptr=z["allow_indptr"].astype(np.int32),
                allow_indices=z["allow_indices"].astype(np.int32),
                deny_indptr=z["deny_indptr"].astype(np.int32),
                deny_indices=z["deny_indices"].astype(np.int32),
            )


@dataclass(slots=True)
class _Staged:
    vectors: list[F32]
    ids: list[ChunkId]
    policies: list[AccessPolicy]

    def __len__(self) -> int:
        return len(self.ids)


class FlatIndex:
    """See module docstring. Not thread-safe; one writer, single process (ADR-002)."""

    def __init__(
        self, path: Path, dimension: int, *, rescore_multiplier: int = DEFAULT_RESCORE_MULTIPLIER
    ) -> None:
        self._path = path
        self._dimension = dimension
        self._rescore_multiplier = max(1, rescore_multiplier)
        self._quantizer: Int8Quantizer | None = None
        self._binary: U8 = np.zeros((0, dimension // 8), dtype=np.uint8)
        self._int8: I8 = np.zeros((0, dimension), dtype=np.int8)
        self._ids: list[ChunkId] = []
        self._row_of: dict[ChunkId, int] = {}
        self._acl: AclTable = AclTable.build([])
        self._staged = _Staged([], [], [])

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(
        cls, path: Path, *, dimension: int, rescore_multiplier: int = DEFAULT_RESCORE_MULTIPLIER
    ) -> FlatIndex:
        if dimension <= 0 or dimension % 8 != 0:
            raise VectorIndexError("dimension must be a positive multiple of 8")
        index = cls(path, dimension, rescore_multiplier=rescore_multiplier)
        meta_path = path / "meta.json"
        if meta_path.exists():
            index._load()
        else:
            path.mkdir(parents=True, exist_ok=True)
            index._write_all()
        return index

    def _load(self) -> None:
        try:
            meta = json.loads((self._path / "meta.json").read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorIndexError("index meta.json is unreadable") from exc
        if meta.get("format") != FORMAT:
            raise VectorIndexError("index directory is not a perimeter flat index")
        if int(meta.get("dimension", -1)) != self._dimension:
            raise VectorIndexError(
                f"index dimension {meta.get('dimension')} != requested {self._dimension}"
            )
        count = int(meta["count"])
        self._binary = self._map_u8(self._path / "binary.bin", (count, self._dimension // 8))
        self._int8 = self._map_i8(self._path / "int8.bin", (count, self._dimension))
        self._ids = [ChunkId(s) for s in json.loads((self._path / "ids.json").read_text())]
        self._row_of = {cid: i for i, cid in enumerate(self._ids)}
        self._acl = AclTable.load(self._path / "acl.npz")
        quant = self._path / "quant.npz"
        if quant.exists():
            with np.load(quant) as z:
                self._quantizer = Int8Quantizer.from_params(
                    Int8Params(minimum=z["minimum"], scale=z["scale"])
                )
        if len(self._ids) != count or self._acl.rows != count:
            raise VectorIndexError("index files disagree on row count")

    @staticmethod
    def _map_u8(path: Path, shape: tuple[int, int]) -> U8:
        if shape[0] == 0:
            return np.zeros(shape, dtype=np.uint8)
        try:
            return cast(U8, np.memmap(path, dtype=np.uint8, mode="r", shape=shape))
        except (OSError, ValueError) as exc:
            raise VectorIndexError(f"cannot map {path.name}") from exc

    @staticmethod
    def _map_i8(path: Path, shape: tuple[int, int]) -> I8:
        if shape[0] == 0:
            return np.zeros(shape, dtype=np.int8)
        try:
            return cast(I8, np.memmap(path, dtype=np.int8, mode="r", shape=shape))
        except (OSError, ValueError) as exc:
            raise VectorIndexError(f"cannot map {path.name}") from exc

    def _write_all(self) -> None:
        tmp = self._path / ".write"
        tmp.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(self._binary).tofile(tmp / "binary.bin")
        np.ascontiguousarray(self._int8).tofile(tmp / "int8.bin")
        (tmp / "ids.json").write_text(json.dumps(list(self._ids)))
        self._acl.save(tmp / "acl.npz")
        if self._quantizer is not None:
            p = self._quantizer.params()
            np.savez(tmp / "quant.npz", minimum=p.minimum, scale=p.scale)
        (tmp / "meta.json").write_text(
            json.dumps({"format": FORMAT, "dimension": self._dimension, "count": len(self._ids)})
        )
        for name in ("binary.bin", "int8.bin", "ids.json", "acl.npz", "quant.npz", "meta.json"):
            src = tmp / name
            if src.exists():
                os.replace(src, self._path / name)
        tmp.rmdir()

    # -- properties --------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return len(self._ids) + len(self._staged)

    @property
    def quantizer(self) -> Int8Quantizer:
        if self._quantizer is None:
            raise VectorIndexError("index is empty; no quantizer has been fitted")
        return self._quantizer

    @property
    def binary_codes(self) -> U8:
        return self._binary

    @property
    def int8_codes(self) -> I8:
        return self._int8

    @property
    def acl(self) -> AclTable:
        return self._acl

    def chunk_id_at(self, row: int) -> ChunkId:
        self._flush_if_staged()
        return self._ids[row]

    def row_of(self, chunk_id: ChunkId) -> int:
        self._flush_if_staged()
        try:
            return self._row_of[chunk_id]
        except KeyError as exc:
            raise VectorIndexError("chunk id not in index") from exc

    def policy_at(self, row: int) -> AccessPolicy:
        self._flush_if_staged()
        return self._acl.policy_at(row)

    # -- writes ------------------------------------------------------------

    def add(self, entries: Iterable[IndexEntry]) -> None:
        for entry in entries:
            if len(entry.vector) != self._dimension:
                raise VectorIndexError(
                    f"vector has {len(entry.vector)} dims; index expects {self._dimension}"
                )
            vec = np.frombuffer(entry.vector, dtype=np.float32).reshape(1, -1)
            self._staged.vectors.append(l2_normalize(vec)[0])
            self._staged.ids.append(entry.chunk_id)
            self._staged.policies.append(entry.policy)

    def _flush_if_staged(self) -> None:
        if len(self._staged):
            self.flush()

    def flush(self) -> None:
        """Append staged rows to disk (later duplicates replace earlier rows) and re-map."""
        staged, self._staged = self._staged, _Staged([], [], [])
        all_ids = [*self._ids, *staged.ids]
        last_pos = {cid: i for i, cid in enumerate(all_ids)}
        keep = [i for i, cid in enumerate(all_ids) if last_pos[cid] == i]
        keep_old = np.asarray([i for i in keep if i < len(self._ids)], dtype=np.int64)
        keep_new = [i - len(self._ids) for i in keep if i >= len(self._ids)]

        new_vecs = (
            np.stack([staged.vectors[i] for i in keep_new])
            if keep_new
            else np.zeros((0, self._dimension), dtype=np.float32)
        )
        if self._quantizer is None and new_vecs.shape[0] > 0:
            self._quantizer = Int8Quantizer.fit(new_vecs)

        old_bin = np.asarray(self._binary)[keep_old]
        old_i8 = np.asarray(self._int8)[keep_old]
        if new_vecs.shape[0] > 0 and self._quantizer is not None:
            new_bin = binarize(new_vecs)
            new_i8 = self._quantizer.encode(new_vecs)
        else:
            new_bin = np.zeros((0, self._dimension // 8), dtype=np.uint8)
            new_i8 = np.zeros((0, self._dimension), dtype=np.int8)

        old_policies = self._acl.policies()
        self._binary = np.concatenate([old_bin, new_bin]).astype(np.uint8)
        self._int8 = np.concatenate([old_i8, new_i8]).astype(np.int8)
        self._ids = [all_ids[i] for i in keep]
        self._row_of = {cid: i for i, cid in enumerate(self._ids)}
        self._acl = AclTable.build(
            [old_policies[i] for i in keep_old.tolist()] + [staged.policies[i] for i in keep_new]
        )
        self._write_all()
        self._load()

    # -- reads -------------------------------------------------------------

    def search(self, query: Vector, permitted: PermissionSet, k: int) -> Sequence[IndexHit]:
        """The VectorIndex port. Filtering is applied inside the scan; see filtered_search."""
        from perimeter.index.filtered_search import filtered_search

        self._flush_if_staged()
        if len(self._ids) == 0:
            return []
        return filtered_search(self, query, permitted, k)

    def scan_rows(self, query: F32, rows: I32, k: int) -> tuple[I32, F32]:
        """Score exactly ``rows`` against ``query``; return up to ``k`` best, best first.

        Two stages: Hamming distance on the binary codes of ``rows`` selects
        ``k * rescore_multiplier`` candidates; int8 rescoring against the float
        query orders them. Rows outside ``rows`` are never read.
        """
        self._flush_if_staged()
        if k <= 0:
            raise VectorIndexError("k must be positive")
        if rows.shape[0] == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
        if self._quantizer is None:
            raise VectorIndexError("index is empty")
        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        qcode = binarize(q.reshape(1, -1))[0]

        distances = hamming_distances(qcode, self._binary[rows])
        n_rescore = min(rows.shape[0], max(k, k * self._rescore_multiplier))
        if n_rescore < rows.shape[0]:
            local = np.argpartition(distances, n_rescore - 1)[:n_rescore]
        else:
            local = np.arange(rows.shape[0])
        candidate_rows = rows[local]

        scores = self._quantizer.rescore(q, self._int8[candidate_rows])
        top = np.argsort(-scores, kind="stable")[:k]
        return candidate_rows[top].astype(np.int32), scores[top].astype(np.float32)
