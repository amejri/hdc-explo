"""Binding helpers for the MEM tranche."""
from __future__ import annotations

import numpy as np

from ..utils import ensure_pm1_int8

__all__ = [
    "to_mem_tranche",
    "from_mem_tranche",
    "bind_tranche_batch",
]


def to_mem_tranche(x: np.ndarray, g_mem: np.ndarray) -> np.ndarray:
    """Bind a vector with the MEM key (element-wise product in ±1)."""
    xv = ensure_pm1_int8(x)
    gv = ensure_pm1_int8(g_mem)
    if xv.shape != gv.shape:
        raise ValueError("x and g_mem must share the same shape")
    return (xv.astype(np.int16, copy=False) * gv.astype(np.int16, copy=False)).astype(np.int8, copy=False)


from_mem_tranche = to_mem_tranche


def bind_tranche_batch(x: np.ndarray, g_mem: np.ndarray) -> np.ndarray:
    """Batch-capable binding: broadcast ``g_mem`` over the leading dimension."""
    gv = ensure_pm1_int8(g_mem)
    if gv.ndim != 1:
        raise ValueError("g_mem must be 1-D")
    if x.ndim == 1:
        return to_mem_tranche(x, gv)
    if x.ndim != 2:
        raise ValueError("x must be 1-D or 2-D")
    if x.shape[1] != gv.size:
        raise ValueError("x and g_mem dimensions do not match")
    xv = ensure_pm1_int8(x)
    return (xv.astype(np.int16, copy=False) * gv.astype(np.int16, copy=False)).astype(np.int8, copy=False)
