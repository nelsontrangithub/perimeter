"""Binary and int8 quantization with float rescoring."""

from __future__ import annotations

import numpy as np
import pytest

from perimeter.core.errors import VectorIndexError
from perimeter.index.quantize import (
    Int8Quantizer,
    binarize,
    hamming_distances,
    l2_normalize,
)


def _random_unit(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return l2_normalize(rng.standard_normal((n, d)).astype(np.float32))


def test_binarize_packs_sign_bits_big_endian_within_byte() -> None:
    v = np.array([[1, -1, 1, -1, 1, -1, 1, -1]], dtype=np.float32)
    code = binarize(v)
    assert code.dtype == np.uint8
    assert code.shape == (1, 1)
    assert int(code[0, 0]) == 0b10101010


def test_binarize_treats_zero_as_negative() -> None:
    v = np.zeros((1, 8), dtype=np.float32)
    assert int(binarize(v)[0, 0]) == 0


def test_binarize_rejects_dimension_not_multiple_of_eight() -> None:
    with pytest.raises(VectorIndexError):
        binarize(np.ones((1, 12), dtype=np.float32))


def test_binarize_output_width_is_dim_over_eight() -> None:
    assert binarize(_random_unit(5, 1024)).shape == (5, 128)


def test_hamming_self_is_zero_and_complement_is_dim() -> None:
    codes = binarize(_random_unit(1, 64))
    assert hamming_distances(codes[0], codes)[0] == 0
    complement = np.bitwise_not(codes)
    assert hamming_distances(codes[0], complement)[0] == 64


def test_hamming_equals_count_of_sign_disagreements() -> None:
    vecs = _random_unit(50, 128)
    codes = binarize(vecs)
    q = vecs[0]
    expected = np.sum((vecs > 0) != (q > 0), axis=1)
    got = hamming_distances(codes[0], codes)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, expected)


def test_hamming_on_empty_code_set_returns_empty() -> None:
    codes = binarize(_random_unit(1, 64))
    assert hamming_distances(codes[0], codes[:0]).shape == (0,)


def test_l2_normalize_produces_unit_rows_and_leaves_zero_rows_zero() -> None:
    v = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    n = l2_normalize(v)
    np.testing.assert_allclose(n[0], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_array_equal(n[1], [0.0, 0.0])


def test_int8_roundtrip_error_is_bounded_by_step_size() -> None:
    vecs = _random_unit(200, 64)
    q = Int8Quantizer.fit(vecs)
    codes = q.encode(vecs)
    assert codes.dtype == np.int8
    decoded = q.decode(codes)
    step = q.scale / 255.0
    assert np.all(np.abs(decoded - vecs) <= step / 2 + 1e-6)


def test_int8_rescore_approximates_dot_product() -> None:
    vecs = _random_unit(500, 128, seed=1)
    q = Int8Quantizer.fit(vecs)
    codes = q.encode(vecs)
    query = _random_unit(1, 128, seed=2)[0]
    exact = vecs @ query
    approx = q.rescore(query, codes)
    assert approx.dtype == np.float32
    assert np.max(np.abs(approx - exact)) < 0.02


def test_int8_rescore_preserves_exact_ranking_on_separated_scores() -> None:
    vecs = _random_unit(20, 64, seed=3)
    q = Int8Quantizer.fit(vecs)
    codes = q.encode(vecs)
    query = vecs[7] * 0.9 + vecs[3] * 0.1
    exact_order = np.argsort(-(vecs @ query))
    approx_order = np.argsort(-q.rescore(query, codes))
    assert exact_order[0] == approx_order[0] == 7


def test_int8_rescore_on_subset_of_rows_matches_full() -> None:
    vecs = _random_unit(30, 64, seed=4)
    q = Int8Quantizer.fit(vecs)
    codes = q.encode(vecs)
    query = _random_unit(1, 64, seed=5)[0]
    full = q.rescore(query, codes)
    rows = np.array([3, 9, 27])
    np.testing.assert_allclose(q.rescore(query, codes[rows]), full[rows])


def test_int8_quantizer_params_roundtrip() -> None:
    vecs = _random_unit(10, 32)
    q = Int8Quantizer.fit(vecs)
    restored = Int8Quantizer.from_params(q.params())
    np.testing.assert_array_equal(q.encode(vecs), restored.encode(vecs))


def test_int8_fit_rejects_empty_and_wrong_dtype() -> None:
    with pytest.raises(VectorIndexError):
        Int8Quantizer.fit(np.zeros((0, 8), dtype=np.float32))
    with pytest.raises(VectorIndexError):
        Int8Quantizer.fit(np.zeros((2, 8), dtype=np.float64))


def test_int8_encode_rejects_dimension_mismatch() -> None:
    q = Int8Quantizer.fit(_random_unit(4, 16))
    with pytest.raises(VectorIndexError):
        q.encode(_random_unit(4, 32))


def test_int8_encode_clips_out_of_range_values_instead_of_wrapping() -> None:
    q = Int8Quantizer.fit(np.array([[0.0, 1.0]], dtype=np.float32))
    codes = q.encode(np.array([[-5.0, 5.0]], dtype=np.float32))
    assert int(codes[0, 0]) == -128
    assert int(codes[0, 1]) == 127
