"""Associative memory bank for MEM."""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..utils import ensure_pm1_int8

__all__ = [
    "MemBank",
    "mem_scores",
    "mem_scores_chunked",
    "mem_topk_stream",
    "mem_argmax",
    "mem_payload",
    "topk_indices",
    "margin_top1",
    "argmax_tie_break",
]


@dataclass
class MemBank:
    """Associative HD memory storing class accumulators and prototypes."""

    B: int
    D: int
    thresh: bool = True

    def __post_init__(self) -> None:
        if self.B <= 0 or self.D <= 0:
            raise ValueError("B and D must be positive")
        self.M = np.zeros((self.B, self.D), dtype=np.int32)
        self.H = np.zeros((self.B, self.D), dtype=np.int8)
        self.n = np.zeros((self.B,), dtype=np.int32)

    def add(self, c: int, payload: np.ndarray) -> None:
        if not (0 <= c < self.B):
            raise IndexError("class index out of range")
        vec = ensure_pm1_int8(payload)
        if vec.shape != (self.D,):
            raise ValueError("payload must have shape (D,)")
        self.M[c, :] += vec.astype(np.int32, copy=False)
        self.n[c] += 1
        if self.thresh:
            self.H[c, :] = np.where(self.M[c, :] >= 0, 1, -1).astype(np.int8, copy=False)

    def seal(self, c: int) -> None:
        if not (0 <= c < self.B):
            raise IndexError("class index out of range")
        self.H[c, :] = np.where(self.M[c, :] >= 0, 1, -1).astype(np.int8, copy=False)

    def empirical_mean(self, c: int) -> np.ndarray:
        if self.n[c] == 0:
            return np.full((self.D,), np.nan, dtype=np.float64)
        return self.M[c, :].astype(np.float64) / float(self.n[c])

    def sign_error_rate(self, c: int, reference: np.ndarray) -> float:
        ref = ensure_pm1_int8(reference)
        if ref.shape != (self.D,):
            raise ValueError("reference must have shape (D,)")
        sign = np.where(self.M[c, :] >= 0, 1, -1).astype(np.int8, copy=False)
        return float(np.mean(sign != ref))

    def inf_norm_error(self, c: int, reference: np.ndarray) -> float:
        ref = np.asarray(reference, dtype=np.float64)
        return float(np.max(np.abs(self.empirical_mean(c) - ref)))


def mem_scores(mem: MemBank, R_mem: np.ndarray, use_thresh: bool = True) -> np.ndarray:
    vec = ensure_pm1_int8(R_mem)
    if vec.shape != (mem.D,):
        raise ValueError("query must have shape (D,)")
    proto = mem.H if use_thresh else np.where(mem.M >= 0, 1, -1).astype(np.int8, copy=False)
    dots = proto.astype(np.int32, copy=False) @ vec.astype(np.int32, copy=False)
    return (dots / float(mem.D)).astype(np.float64, copy=False)


def mem_scores_chunked(mem: MemBank, R_mem: np.ndarray, chunk: int = 4096, use_thresh: bool = True) -> np.ndarray:
    vec = ensure_pm1_int8(R_mem)
    if vec.shape != (mem.D,):
        raise ValueError("query must have shape (D,)")
    proto = mem.H if use_thresh else np.where(mem.M >= 0, 1, -1).astype(np.int8, copy=False)
    out = np.empty((mem.B,), dtype=np.float64)
    R32 = vec.astype(np.int32, copy=False)
    for start in range(0, mem.B, chunk):
        end = min(mem.B, start + chunk)
        dots = proto[start:end, :].astype(np.int32, copy=False) @ R32
        out[start:end] = dots / float(mem.D)
    return out


def mem_topk_stream(mem: MemBank, R_mem: np.ndarray, k: int = 5, use_thresh: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)
    vec = ensure_pm1_int8(R_mem)
    proto = mem.H if use_thresh else np.where(mem.M >= 0, 1, -1).astype(np.int8, copy=False)
    R32 = vec.astype(np.int32, copy=False)
    heap: list[tuple[float, int]] = []
    for c in range(mem.B):
        score = float((proto[c, :].astype(np.int32, copy=False) @ R32) / float(mem.D))
        if len(heap) < k:
            heapq.heappush(heap, (score, c))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, c))
    heap.sort(reverse=True)
    scores = np.array([s for (s, _) in heap], dtype=np.float64)
    idx = np.array([c for (_, c) in heap], dtype=np.int64)
    return idx, scores


def mem_argmax(scores: np.ndarray) -> int:
    return int(np.argmax(np.asarray(scores)))


def mem_payload(mem: MemBank, c_star: int) -> np.ndarray:
    if not (0 <= c_star < mem.B):
        raise IndexError("class index out of range")
    out = mem.H[c_star, :].view()
    out.setflags(write=False)
    return out


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    arr = np.asarray(scores)
    k = int(min(k, arr.shape[0]))
    part = np.argpartition(arr, -k)[-k:]
    return part[np.argsort(arr[part])[::-1]]


def margin_top1(scores: np.ndarray) -> float:
    arr = np.asarray(scores)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return float(arr[0])
    idx = topk_indices(arr, 2)
    return float(arr[idx[0]] - arr[idx[1]])


def argmax_tie_break(scores: np.ndarray, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.0, 1e-9, size=np.asarray(scores).shape)
    return int(np.argmax(np.asarray(scores) + jitter))
