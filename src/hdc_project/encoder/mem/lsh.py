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


# --- utils 64-bit mixing & unbiased bucket ------------------

def _mix64(x: int) -> int:
    """SplitMix64-style mixer (fort, 64-bit)."""
    x &= 0xFFFFFFFFFFFFFFFF
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31)
    return x & 0xFFFFFFFFFFFFFFFF


def _bucket_unbiased_64(code: int, B: int, seed: int) -> int:
    """Réduction quasi-uniforme: hache en 64-bit puis Lemire fastmod."""
    if B <= 0:
        raise ValueError("B must be positive")
    h = _mix64((int(code) ^ (seed & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF)
    # Lemire: floor(h * B / 2^64) ≡ (h * B) >> 64
    return int(((h * int(B)) >> 64) & 0xFFFFFFFFFFFFFFFF)

def code_to_bucket(code: int, B: int, seed: int = 0) -> int:
    """Map code -> bucket via mix 64-bit + Lemire."""
    return _bucket_unbiased_64(code, B, seed)


from dataclasses import dataclass

@dataclass(frozen=True)
class SignLSH:
    idx_bits: np.ndarray  # (k,) int64
    seed: int = 0         # NEW: seed pour mixer/réduction

    @property
    def k(self) -> int:
        return int(self.idx_bits.shape[0])

    @staticmethod
    def with_k_bits(D: int, k: int, seed: int | None = None) -> "SignLSH":
        if not 1 <= k <= D:
            raise ValueError("k must satisfy 1 <= k <= D")
        g = np.random.default_rng(seed)
        bits = g.choice(int(D), size=int(k), replace=False)
        # seed secondaire pour le mixeur, indépendant du tirage d’indices:
        seed_mix = int(g.integers(1, 2**63 - 1))
        return SignLSH(idx_bits=bits.astype(np.int64, copy=False), seed=seed_mix)

    def code(self, z: np.ndarray) -> int:
        vec = np.asarray(z, dtype=np.int8)
        if vec.ndim != 1 or vec.size < self.k:
            raise ValueError("vector must be 1-D with length >= k")
        # pack MSB->LSB
        bits = (vec[self.idx_bits] > 0).astype(np.uint8, copy=False)
        code = 0
        for b in bits:
            code = (code << 1) | int(b)
        return int(code)

    def bucket_mod(self, z: np.ndarray, B: int) -> int:
        # évitable, mais on le garde pour compat
        return int(self.code(z) % int(B))

    def bucket_unbiased(self, z: np.ndarray, B: int) -> int:
        # NEW: passe par mix64 + Lemire (décorrèle "le rang" du code)
        return _bucket_unbiased_64(self.code(z), B, self.seed)

    def bucket(self, z: np.ndarray, B: int, unbiased: bool = True) -> int:
        return self.bucket_unbiased(z, B) if unbiased else self.bucket_mod(z, B)

@dataclass
class MultiSignLSH:
    tables: list[SignLSH]
    seed: int = 0  # NEW

    @staticmethod
    def build(D: int, k: int, T: int, seed: int | None = None) -> "MultiSignLSH":
        if T <= 0:
            raise ValueError("T must be positive")
        g = np.random.default_rng(seed)
        tables = [SignLSH.with_k_bits(D, k, int(g.integers(0, 2**31 - 1))) for _ in range(T)]
        seed_mix = int(g.integers(1, 2**63 - 1))
        return MultiSignLSH(tables=tables, seed=seed_mix)

    def code(self, z: np.ndarray) -> int:
        # XOR de codes k-bits → k-bits, puis on mixe dans bucket()
        code = 0
        for table in self.tables:
            code ^= table.code(z)
        return int(code)

    def bucket(self, z: np.ndarray, B: int) -> int:
        # NEW: mix64+Lemire sur le code fusionné
        return _bucket_unbiased_64(self.code(z), B, self.seed)

    def bucket_stream(self, zs: Iterable[np.ndarray], B: int) -> np.ndarray:
        return np.array([self.bucket(z, B) for z in zs], dtype=np.int64)
