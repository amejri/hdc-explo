"""Module M2: Permutation and rotation primitives."""
from __future__ import annotations

from typing import List

import numpy as np


__all__ = ["M2_roll", "M2_plan_perm", "M2_pow_index", "M2_perm_pow"]


def M2_roll(X: np.ndarray, Delta: int) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim == 1:
        return np.roll(arr, shift=int(Delta), axis=0)
    if arr.ndim == 2:
        return np.roll(arr, shift=int(Delta), axis=1)
    raise ValueError(f"unsupported shape {arr.shape}")


def M2_plan_perm(D: int, seed: int | None = None) -> np.ndarray:
    g = np.random.default_rng(seed)
    pi = g.permutation(int(D)).astype(np.int64, copy=False)
    if np.unique(pi).size != D:
        raise RuntimeError("permutation is not bijective")
    return pi


def _cycles_of(pi: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    D = pi.size
    seen = np.zeros(D, dtype=bool)
    cycles: list[np.ndarray] = []
    cycle_id = -np.ones(D, dtype=np.int64)
    pos = -np.ones(D, dtype=np.int64)
    cid = 0
    for i in range(D):
        if seen[i]:
            continue
        cur: list[int] = []
        j = i
        while not seen[j]:
            seen[j] = True
            cur.append(j)
            j = int(pi[j])
        cyc = np.array(cur, dtype=np.int64)
        for p, idx in enumerate(cyc):
            cycle_id[idx] = cid
            pos[idx] = p
        cycles.append(cyc)
        cid += 1
    return cycle_id, pos, cycles


def M2_pow_index(pi: np.ndarray, k: int) -> np.ndarray:
    cycle_id, pos, cycles = _cycles_of(pi)
    D = pi.size
    idx = np.empty(D, dtype=np.int64)
    for cyc in cycles:
        L = cyc.size
        tgt = np.roll(cyc, shift=int(k) % L)
        idx[cyc] = tgt
    return idx


def M2_perm_pow(X: np.ndarray, pi: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(X)
    idx = M2_pow_index(pi, int(k))
    if arr.ndim == 1:
        return arr[idx]
    if arr.ndim == 2:
        return arr[:, idx]
    raise ValueError(f"unsupported shape {arr.shape}")
