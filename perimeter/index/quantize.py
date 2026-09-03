"""Binary and int8 quantization with float rescoring.

The index keeps two compressed views of every vector (ADR-002):

* a 1-bit **binary code** (sign of each dimension, packed to ``d/8`` bytes)
  used for the scan: Hamming distance via XOR + popcount over the permitted rows;
* an **int8 code** (per-dimension affine quantization to 256 levels) used to
  rescore the scan's candidates against the float32 query.

Neither is lossless. The recall benchmark (commit 11) measures what the pair
costs against an exact float32 scan and gates CI on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from perimeter.core.errors import VectorIndexError

F32 = npt.NDArray[np.float32]
U8 = npt.NDArray[np.uint8]
I8 = npt.NDArray[np.int8]
I32 = npt.NDArray[np.int32]


def l2_normalize(vectors: F32) -> F32:
    """Row-wise unit length. All-zero rows stay zero rather than becoming NaN."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    out: F32 = (vectors / safe).astype(np.float32, copy=False)
    return out


def binarize(vectors: F32) -> U8:
    """Pack the sign bit of each dimension: ``d/8`` bytes per row, MSB first."""
    if vectors.ndim != 2 or vectors.shape[1] % 8 != 0:
        raise VectorIndexError("binary quantization requires 2-D input with dim % 8 == 0")
    packed: U8 = np.packbits(vectors > 0, axis=1)
    return packed


def hamming_distances(query_code: U8, codes: U8) -> I32:
    """Hamming distance from one packed code to each row of ``codes``."""
    if codes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    xored = np.bitwise_xor(codes, query_code)
    out: I32 = np.bitwise_count(xored).sum(axis=1, dtype=np.int32)
    return out


@dataclass(frozen=True, slots=True)
class Int8Params:
    minimum: F32
    scale: F32


@dataclass(frozen=True, slots=True)
class Int8Quantizer:
    """Per-dimension affine int8 quantizer with a dot-product rescoring shortcut."""

    minimum: F32
    scale: F32

    @classmethod
    def fit(cls, vectors: F32) -> Int8Quantizer:
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise VectorIndexError("int8 quantizer needs a non-empty 2-D sample")
        if vectors.dtype != np.float32:
            raise VectorIndexError("int8 quantizer expects float32 input")
        minimum = vectors.min(axis=0).astype(np.float32)
        scale = (vectors.max(axis=0) - minimum).astype(np.float32)
        scale = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
        return cls(minimum=minimum, scale=scale)

    @classmethod
    def from_params(cls, params: Int8Params) -> Int8Quantizer:
        return cls(minimum=params.minimum, scale=params.scale)

    def params(self) -> Int8Params:
        return Int8Params(minimum=self.minimum, scale=self.scale)

    @property
    def dimension(self) -> int:
        return int(self.minimum.shape[0])

    def encode(self, vectors: F32) -> I8:
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise VectorIndexError("int8 encode: dimension mismatch")
        levels = np.rint((vectors - self.minimum) / self.scale * 255.0) - 128.0
        out: I8 = np.clip(levels, -128, 127).astype(np.int8)
        return out

    def decode(self, codes: I8) -> F32:
        out: F32 = ((codes.astype(np.float32) + 128.0) / 255.0 * self.scale + self.minimum).astype(
            np.float32
        )
        return out

    def rescore(self, query: F32, codes: I8) -> F32:
        """Approximate ``codes_as_float @ query`` without materialising the floats.

        ``v ≈ (c + 128) * scale/255 + min``, so
        ``q·v = c·(q*scale/255) + q·(128*scale/255 + min)``: one int8-to-float
        matmul over the candidate rows plus a constant.
        """
        if query.shape != (self.dimension,):
            raise VectorIndexError("int8 rescore: query dimension mismatch")
        weighted = (query * self.scale / 255.0).astype(np.float32)
        bias = np.float32(np.dot(query, 128.0 * self.scale / 255.0 + self.minimum))
        out: F32 = (codes.astype(np.float32) @ weighted + bias).astype(np.float32)
        return out
