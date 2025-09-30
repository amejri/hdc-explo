"""Module M0: Rademacher key generation primitives."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .utils import rng

__all__ = [
    "M0_NewKey",
    "M0_Rad",
    "M0_min_D",
]


def M0_NewKey(seed: int, D: int) -> np.ndarray:
    """Return a single Rademacher vector of length *D* in int8."""
    g = rng(seed)
    bits = g.integers(0, 2, size=int(D), dtype=np.int8)
    return ((bits << 1) - 1).astype(np.int8, copy=False)


def M0_Rad(n: int, D: int, seed: Optional[int] = None) -> np.ndarray:
    """Return an ``(n, D)`` array whose rows are i.i.d. Rademacher vectors."""
    g = rng(seed)
    bits = g.integers(0, 2, size=(int(n), int(D)), dtype=np.int8)
    return ((bits << 1) - 1).astype(np.int8, copy=False)


def M0_min_D(eps: float, delta: float) -> int:
    """Minimum dimension so that Hoeffding's tail is bounded by ``delta``."""
    if eps <= 0 or not (0 < delta < 1):
        raise ValueError("invalid parameters")
    return int(math.ceil((2.0 / (eps * eps)) * math.log(2.0 / delta)))
