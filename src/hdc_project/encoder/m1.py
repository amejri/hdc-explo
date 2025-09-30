"""Module M1: Similarity and distance primitives."""
from __future__ import annotations

import numpy as np

from .utils import as_vec, as_mat

__all__ = ["M1_sim", "M1_dH", "M1_sim_batch"]


def M1_sim(X: np.ndarray, Y: np.ndarray) -> float:
    Xv = as_vec(np.asarray(X))
    Yv = as_vec(np.asarray(Y))
    if Xv.shape != Yv.shape:
        raise ValueError(f"shape mismatch: {Xv.shape} vs {Yv.shape}")
    if Xv.dtype == np.int8:
        Xv = Xv.astype(np.int32, copy=False)
    if Yv.dtype == np.int8:
        Yv = Yv.astype(np.int32, copy=False)
    return float(np.dot(Xv, Yv) / Xv.shape[0])


def M1_dH(X: np.ndarray, Y: np.ndarray) -> int:
    Xv = as_vec(np.asarray(X))
    Yv = as_vec(np.asarray(Y))
    if Xv.shape != Yv.shape:
        raise ValueError(f"shape mismatch: {Xv.shape} vs {Yv.shape}")
    D = Xv.shape[0]
    sim = M1_sim(Xv, Yv)
    return int(round((D / 2.0) * (1.0 - sim)))


def M1_sim_batch(Xs: np.ndarray, Ys: np.ndarray) -> np.ndarray:
    Xm = as_mat(np.asarray(Xs))
    Ym = as_mat(np.asarray(Ys))
    if Xm.shape != Ym.shape:
        raise ValueError(f"shape mismatch: {Xm.shape} vs {Ym.shape}")
    D = Xm.shape[1]
    if Xm.dtype == np.int8:
        Xm = Xm.astype(np.int32, copy=False)
    if Ym.dtype == np.int8:
        Ym = Ym.astype(np.int32, copy=False)
    dots = (Xm * Ym).sum(axis=1, dtype=np.int64)
    return (dots / D).astype(np.float64, copy=False)
