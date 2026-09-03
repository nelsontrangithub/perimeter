"""Memory-mapped flat index: persistence, mmap-at-open, staged adds, two-stage scan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from perimeter.core.acl import AccessPolicy, Deny, Grant
from perimeter.core.document import ChunkId
from perimeter.core.errors import VectorIndexError
from perimeter.core.ports import IndexEntry, as_vector
from perimeter.core.principal import EVERYONE, PrincipalId
from perimeter.index.flat import FlatIndex
from perimeter.index.quantize import l2_normalize

DIM = 64
PUBLIC = AccessPolicy.public()


def _vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal((n, DIM)).astype(np.float32))


def _entries(
    vecs: np.ndarray, prefix: str = "c", policy: AccessPolicy = PUBLIC
) -> list[IndexEntry]:
    return [
        IndexEntry(chunk_id=ChunkId(f"{prefix}{i}"), vector=as_vector(v), policy=policy)
        for i, v in enumerate(vecs)
    ]


def test_open_creates_empty_index(tmp_path: Path) -> None:
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    assert idx.size == 0
    assert idx.dimension == DIM
    assert (tmp_path / "idx" / "meta.json").is_file()


def test_add_then_flush_persists_and_reopens(tmp_path: Path) -> None:
    path = tmp_path / "idx"
    idx = FlatIndex.open(path, dimension=DIM)
    idx.add(_entries(_vectors(10)))
    assert idx.size == 10
    idx.flush()
    reopened = FlatIndex.open(path, dimension=DIM)
    assert reopened.size == 10
    assert reopened.chunk_id_at(3) == ChunkId("c3")


def test_flushed_arrays_are_memory_mapped_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "idx"
    idx = FlatIndex.open(path, dimension=DIM)
    idx.add(_entries(_vectors(5)))
    idx.flush()
    assert isinstance(idx.binary_codes, np.memmap)
    assert isinstance(idx.int8_codes, np.memmap)
    assert Path(str(idx.binary_codes.filename)).parent == path
    assert idx.binary_codes.shape == (5, DIM // 8)
    assert idx.int8_codes.shape == (5, DIM)


def test_add_rejects_dimension_mismatch(tmp_path: Path) -> None:
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    bad = IndexEntry(chunk_id=ChunkId("x"), vector=as_vector([1.0] * 16), policy=PUBLIC)
    with pytest.raises(VectorIndexError):
        idx.add([bad])


def test_scan_rows_returns_only_requested_rows_best_first(tmp_path: Path) -> None:
    vecs = _vectors(100)
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    idx.add(_entries(vecs))
    idx.flush()
    query = vecs[42]
    rows = np.array([1, 42, 77], dtype=np.int32)
    got_rows, scores = idx.scan_rows(query, rows, k=2)
    assert set(got_rows.tolist()) <= {1, 42, 77}
    assert got_rows[0] == 42
    assert scores[0] >= scores[1]


def test_scan_rows_over_all_rows_recovers_exact_top1(tmp_path: Path) -> None:
    vecs = _vectors(500, seed=7)
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    idx.add(_entries(vecs))
    idx.flush()
    query = l2_normalize((vecs[9] * 0.95 + vecs[10] * 0.05)[None, :])[0]
    exact_top = int(np.argmax(vecs @ query))
    got_rows, _ = idx.scan_rows(query, np.arange(500, dtype=np.int32), k=5)
    assert got_rows[0] == exact_top


def test_scan_rows_with_empty_rows_returns_empty(tmp_path: Path) -> None:
    vecs = _vectors(10)
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    idx.add(_entries(vecs))
    idx.flush()
    rows, scores = idx.scan_rows(vecs[0], np.zeros((0,), dtype=np.int32), k=3)
    assert rows.shape == (0,)
    assert scores.shape == (0,)


def test_scan_rows_k_larger_than_rows_returns_all_rows(tmp_path: Path) -> None:
    vecs = _vectors(10)
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    idx.add(_entries(vecs))
    idx.flush()
    rows, _ = idx.scan_rows(vecs[0], np.array([2, 5], dtype=np.int32), k=50)
    assert sorted(rows.tolist()) == [2, 5]


def test_second_flush_appends_and_keeps_quantizer(tmp_path: Path) -> None:
    path = tmp_path / "idx"
    idx = FlatIndex.open(path, dimension=DIM)
    idx.add(_entries(_vectors(20, seed=1), prefix="a"))
    idx.flush()
    first_params = idx.quantizer.params()
    idx.add(_entries(_vectors(5, seed=2), prefix="b"))
    idx.flush()
    assert idx.size == 25
    assert idx.chunk_id_at(24) == ChunkId("b4")
    np.testing.assert_array_equal(idx.quantizer.params().minimum, first_params.minimum)


def test_scan_sees_staged_entries_by_flushing_lazily(tmp_path: Path) -> None:
    vecs = _vectors(30, seed=3)
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    idx.add(_entries(vecs[:20]))
    idx.flush()
    idx.add(_entries(vecs[20:], prefix="late"))
    rows, _ = idx.scan_rows(vecs[25], np.arange(30, dtype=np.int32), k=1)
    assert idx.chunk_id_at(int(rows[0])) == ChunkId("late5")


def test_duplicate_chunk_id_later_entry_replaces_earlier(tmp_path: Path) -> None:
    vecs = _vectors(4)
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    idx.add(_entries(vecs[:2], prefix="d"))
    idx.flush()
    idx.add([IndexEntry(ChunkId("d1"), as_vector(vecs[3]), AccessPolicy.nobody())])
    idx.flush()
    assert idx.size == 2
    row = idx.row_of(ChunkId("d1"))
    assert idx.policy_at(row) == AccessPolicy.nobody()


def test_policies_roundtrip_through_disk(tmp_path: Path) -> None:
    path = tmp_path / "idx"
    alice = PrincipalId("alice")
    contractors = PrincipalId("contractors")
    policy = AccessPolicy.from_rules([Grant(alice), Grant(EVERYONE), Deny(contractors)])
    idx = FlatIndex.open(path, dimension=DIM)
    idx.add(_entries(_vectors(2), policy=policy))
    idx.flush()
    reopened = FlatIndex.open(path, dimension=DIM)
    assert reopened.policy_at(0) == policy
    assert reopened.policy_at(1) == policy


def test_index_stores_no_text_fields(tmp_path: Path) -> None:
    """ADR-006: nothing under the index directory is a text payload."""
    path = tmp_path / "idx"
    idx = FlatIndex.open(path, dimension=DIM)
    idx.add(_entries(_vectors(3)))
    idx.flush()
    names = sorted(p.name for p in path.iterdir())
    assert names == ["acl.npz", "binary.bin", "ids.json", "int8.bin", "meta.json", "quant.npz"]


def test_open_rejects_dimension_mismatch_with_existing_index(tmp_path: Path) -> None:
    path = tmp_path / "idx"
    FlatIndex.open(path, dimension=DIM).flush()
    with pytest.raises(VectorIndexError):
        FlatIndex.open(path, dimension=DIM * 2)


def test_open_rejects_corrupt_meta(tmp_path: Path) -> None:
    path = tmp_path / "idx"
    path.mkdir()
    (path / "meta.json").write_text(json.dumps({"format": "something-else"}))
    with pytest.raises(VectorIndexError):
        FlatIndex.open(path, dimension=DIM)


def test_row_of_unknown_id_raises(tmp_path: Path) -> None:
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    with pytest.raises(VectorIndexError):
        idx.row_of(ChunkId("nope"))


def test_remove_document_drops_its_rows_at_flush(tmp_path: Path) -> None:
    from perimeter.core.document import DocumentId

    vecs = _vectors(4)
    idx = FlatIndex.open(tmp_path / "idx", dimension=DIM)
    idx.add([IndexEntry(ChunkId(f"docA#{i}"), as_vector(vecs[i]), PUBLIC) for i in range(2)])
    idx.add([IndexEntry(ChunkId(f"docB#{i}"), as_vector(vecs[2 + i]), PUBLIC) for i in range(2)])
    idx.flush()
    idx.remove_document(DocumentId("docA"))
    idx.add([IndexEntry(ChunkId("docA#0"), as_vector(vecs[0]), PUBLIC)])
    idx.flush()
    assert idx.size == 3
    assert sorted(idx.chunk_id_at(r) for r in range(3)) == ["docA#0", "docB#0", "docB#1"]
