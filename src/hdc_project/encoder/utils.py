"""Shared utilities for the encoder brick.

These helpers regroup the repeated snippets that were scattered across the
`notebooks/explore.ipynb` prototype. Keeping them here avoids copy/paste and
makes the numbered Mx modules slimmer.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


__all__ = [
    "rng",
    "rand_pm1",
    "as_vec",
    "as_mat",
    "ensure_pm1_int8",
    "strict_sign_int8",
    "unbiased_sign_int8",
    "sign_int",
]


def rng(seed: int | None) -> np.random.Generator:
    """Return a PCG64 generator initialised from *seed* or the global RNG."""
    if seed is None:
        return np.random.default_rng()
    return np.random.Generator(np.random.PCG64(seed))


def rand_pm1(n: int, D: int, seed: int | None = None) -> np.ndarray:
    """Sample *n* Rademacher vectors of length *D* in int8."""
    g = rng(seed)
    bits = g.integers(0, 2, size=(n, D), dtype=np.int8)
    return ((bits << 1) - 1).astype(np.int8, copy=False)


def as_vec(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D vector, got shape={arr.shape}")
    return arr


def as_mat(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape={arr.shape}")
    return arr


def ensure_pm1_int8(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.dtype != np.int8:
        arr = arr.astype(np.int8, copy=False)
    return arr


def strict_sign_int8(x: np.ndarray) -> np.ndarray:
    """Strict sign: +1 for positive entries, -1 otherwise (ties treated as -1)."""
    return np.where(x > 0, 1, -1).astype(np.int8, copy=False)


def unbiased_sign_int8(x: np.ndarray, rng_obj: Optional[np.random.Generator] = None) -> np.ndarray:
    """Unbiased sign: randomises the ±1 choice on zero coordinates."""
    if rng_obj is None:
        rng_obj = np.random.default_rng()
    arr = np.empty_like(x, dtype=np.int8)
    gt = x > 0
    lt = x < 0
    eq = ~(gt | lt)
    arr[gt] = 1
    arr[lt] = -1
    if np.any(eq):
        toss = rng_obj.integers(0, 2, size=int(eq.sum()), dtype=np.int8)
        arr[eq] = (toss << 1) - 1
    return arr


def sign_int(x: np.ndarray) -> np.ndarray:
    """Map integer accumulator values to ±1 (ties -> +1 by convention)."""
    bits = (np.asarray(x) >= 0).astype(np.int8, copy=False)
    return ((bits << 1) - 1).astype(np.int8, copy=False)
