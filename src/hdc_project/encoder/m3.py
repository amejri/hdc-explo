"""Module M3: Hadamard binding primitives."""
from __future__ import annotations

import numpy as np

from .utils import ensure_pm1_int8

__all__ = ["M3_bind", "M3_unbind", "M3_bind_batch", "M3_xnor_bind_bits"]


def M3_bind(X: np.ndarray, J: np.ndarray) -> np.ndarray:
    Xv = ensure_pm1_int8(X)
    Jv = ensure_pm1_int8(J)
    return (Xv * Jv).astype(np.int8, copy=False)


def M3_unbind(Y: np.ndarray, J: np.ndarray) -> np.ndarray:
    Yv = ensure_pm1_int8(Y)
    Jv = ensure_pm1_int8(J)
    return (Yv * Jv).astype(np.int8, copy=False)


def M3_bind_batch(X: np.ndarray, J: np.ndarray) -> np.ndarray:
    Xv = ensure_pm1_int8(X)
    Jv = ensure_pm1_int8(J)
    if Xv.ndim != 2:
        raise ValueError(f"X must be rank-2, got shape={Xv.shape}")
    if Jv.ndim != 1:
        raise ValueError(f"J must be 1-D, got shape={Jv.shape}")
    if Xv.shape[1] != Jv.size:
        raise ValueError("incompatible shapes")
    return (Xv * Jv).astype(np.int8, copy=False)


def M3_xnor_bind_bits(xbits: np.ndarray, jbits: np.ndarray) -> np.ndarray:
    if xbits.dtype != np.uint8 or jbits.dtype != np.uint8:
        raise ValueError("bit arrays must be uint8")
    if xbits.shape != jbits.shape:
        raise ValueError("shape mismatch")
    return np.bitwise_not(np.bitwise_xor(xbits, jbits))
