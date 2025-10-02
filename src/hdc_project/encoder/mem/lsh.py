"""Sign-based locality sensitive hashing utilities for MEM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = [
    "SignLSH",
    "MultiSignLSH",
    "code_to_bucket",
]


def _mix32(x: int) -> int:
    """Xorshift-style mixer to decorrelate low bits before modulo."""
    x &= 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


def code_to_bucket(code: int, B: int) -> int:
    """Map an integer code to a bucket in ``[0, B-1]``."""
    if B <= 0:
        raise ValueError("B must be positive")
    return _mix32(int(code)) % int(B)


@dataclass(frozen=True)
class SignLSH:
    """Sign-based LSH that projects onto ``k`` randomly-chosen coordinates."""

    idx_bits: np.ndarray  # (k,) int64

    @property
    def k(self) -> int:
        return int(self.idx_bits.shape[0])

    @staticmethod
    def with_k_bits(D: int, k: int, seed: int | None = None) -> "SignLSH":
        if not 1 <= k <= D:
            raise ValueError("k must satisfy 1 <= k <= D")
        g = np.random.default_rng(seed)
        bits = g.choice(int(D), size=int(k), replace=False)
        return SignLSH(idx_bits=bits.astype(np.int64, copy=False))

    def code(self, z: np.ndarray) -> int:
        vec = np.asarray(z, dtype=np.int8)
        if vec.ndim != 1 or vec.size < self.k:
            raise ValueError("vector must be 1-D with length >= k")
        bits = (vec[self.idx_bits] > 0).astype(np.uint8, copy=False)
        code = 0
        for bit in bits:
            code = (code << 1) | int(bit)
        return int(code)

    def bucket_mod(self, z: np.ndarray, B: int) -> int:
        return self.code(z) % int(B)

    def bucket_unbiased(self, z: np.ndarray, B: int) -> int:
        if B <= 0:
            raise ValueError("B must be positive")
        code = self.code(z)
        return int((code * int(B)) >> self.k)

    def bucket(self, z: np.ndarray, B: int, unbiased: bool = True) -> int:
        return self.bucket_unbiased(z, B) if unbiased else self.bucket_mod(z, B)


@dataclass
class MultiSignLSH:
    """Collection of independent sign-LSH tables with XOR fusion."""

    tables: list[SignLSH]

    @staticmethod
    def build(D: int, k: int, T: int, seed: int | None = None) -> "MultiSignLSH":
        if T <= 0:
            raise ValueError("T must be positive")
        g = np.random.default_rng(seed)
        tables = [SignLSH.with_k_bits(D, k, int(g.integers(0, 2**31 - 1))) for _ in range(T)]
        return MultiSignLSH(tables=tables)

    def code(self, z: np.ndarray) -> int:
        code = 0
        for table in self.tables:
            code ^= table.code(z)
        return int(code)

    def bucket(self, z: np.ndarray, B: int) -> int:
        return code_to_bucket(self.code(z), B)

    def bucket_stream(self, zs: Iterable[np.ndarray], B: int) -> np.ndarray:
        return np.array([self.bucket(z, B) for z in zs], dtype=np.int64)
