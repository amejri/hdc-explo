"""Query construction utilities for MEM retrieval."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .binding import to_mem_tranche
from ..utils import ensure_pm1_int8

__all__ = [
    "apply_perm_power",
    "superpose_signed",
    "build_query_from_context",
    "build_query_mem",
]


def apply_perm_power(x: np.ndarray, pi: np.ndarray, power: int) -> np.ndarray:
    """Apply ``pi`` to ``x`` ``power`` times (negative powers use the inverse)."""
    vec = ensure_pm1_int8(x)
    perm = np.asarray(pi, dtype=np.int64)
    if perm.ndim != 1 or perm.shape[0] != vec.shape[0]:
        raise ValueError("pi must be a permutation of length D")
    if power == 0:
        return vec
    if power > 0:
        idx = np.arange(vec.shape[0], dtype=np.int64)
        for _ in range(power):
            idx = perm[idx]
        return vec[idx]
    inv = np.empty_like(perm)
    inv[perm] = np.arange(vec.shape[0], dtype=np.int64)
    idx = np.arange(vec.shape[0], dtype=np.int64)
    for _ in range(-power):
        idx = inv[idx]
    return vec[idx]


def superpose_signed(vectors: Sequence[np.ndarray], weights: Optional[Sequence[int]] = None) -> np.ndarray:
    if not vectors:
        raise ValueError("at least one vector required")
    D = vectors[0].shape[0]
    acc = np.zeros((D,), dtype=np.int16)
    if weights is None:
        weights = [1] * len(vectors)
    for vec, w in zip(vectors, weights):
        vv = ensure_pm1_int8(vec)
        if vv.shape != (D,):
            raise ValueError("vectors must share the same shape")
        acc += int(w) * vv.astype(np.int16, copy=False)
    return np.where(acc >= 0, 1, -1).astype(np.int8, copy=False)


def build_query_from_context(
    H_window: Sequence[np.ndarray],
    pi: np.ndarray,
    w_left: int,
    w_right: int,
    weights_ctx: Optional[Sequence[int]] = None,
    targets_hist: Optional[Sequence[Tuple[np.ndarray, int, int]]] = None,
) -> np.ndarray:
    if w_left < 0 or w_right < 0:
        raise ValueError("window sizes must be non-negative")
    expected = w_left + w_right + 1
    if len(H_window) != expected:
        raise ValueError("len(H_window) must equal w_left + w_right + 1")
    if weights_ctx is None:
        weights_ctx = [1] * expected
    if len(weights_ctx) != expected:
        raise ValueError("weights_ctx must match window length")
    pieces = []
    weights = []
    for offset, H in zip(range(-w_left, w_right + 1), H_window):
        pieces.append(apply_perm_power(H, pi, offset))
        weights.append(int(weights_ctx[offset + w_left]))
    if targets_hist is not None:
        for proto, beta, gamma in targets_hist:
            pieces.append(apply_perm_power(proto, pi, gamma))
            weights.append(int(beta))
    return superpose_signed(pieces, weights)


def build_query_mem(R: np.ndarray, Gmem: np.ndarray) -> np.ndarray:
    return to_mem_tranche(R, Gmem)
