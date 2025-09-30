"""Module M4: English lexical hypervectors."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, List, Optional

import numpy as np

from .utils import rng

__all__ = [
    "M4_LexEN",
    "M4_LexEN_new",
    "M4_get",
    "M4_get_many",
    "M4_collision_audit",
    "M4_pair_stats",
]


def _canon_token(w: str) -> str:
    return w.strip().lower()


class M4_LexEN:
    """Seeded lexicon that maps tokens to Rademacher vectors."""

    def __init__(self, seed: int, D: int, reserve_pool: int = 0):
        self.D = int(D)
        self._rng = rng(seed)
        self._table: Dict[str, np.ndarray] = {}
        self._reserve: List[np.ndarray] = []
        for _ in range(int(reserve_pool)):
            v = self._new_key()
            v.setflags(write=False)
            self._reserve.append(v)
        self._pool_idx = 0

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    def _new_key(self) -> np.ndarray:
        bits = self._rng.integers(0, 2, size=self.D, dtype=np.int8)
        return ((bits << 1) - 1).astype(np.int8, copy=False)

    def get(self, token: str, use_pool: bool = False) -> np.ndarray:
        key = _canon_token(token)
        vec = self._table.get(key)
        if vec is not None:
            return vec
        if use_pool and self._reserve:
            vec = self._reserve[self._pool_idx]
            self._pool_idx = (self._pool_idx + 1) % len(self._reserve)
        else:
            vec = self._new_key()
            vec.setflags(write=False)
        self._table[key] = vec
        return vec

    def get_many(self, words: Iterable[str], use_pool: bool = False) -> np.ndarray:
        words_list = list(words)
        mat = np.empty((len(words_list), self.D), dtype=np.int8)
        for idx, token in enumerate(words_list):
            mat[idx, :] = self.get(token, use_pool=use_pool)
        return mat

    def contains(self, token: str) -> bool:
        return _canon_token(token) in self._table

    def size(self) -> int:
        return len(self._table)

    def save(self, path: str) -> None:
        keys = np.array(list(self._table.keys()), dtype=object)
        mats = np.stack([self._table[k] for k in keys], axis=0).astype(np.int8, copy=False)
        np.savez_compressed(path, D=self.D, keys=keys, mats=mats)

    @staticmethod
    def load(path: str) -> "M4_LexEN":
        data = np.load(path, allow_pickle=True)
        lex = M4_LexEN(seed=0, D=int(data["D"]))
        for key, mat in zip(data["keys"].tolist(), data["mats"].astype(np.int8, copy=False)):
            mat.setflags(write=False)
            lex._table[key] = mat
        return lex


def M4_LexEN_new(seed: int, D: int, reserve_pool: int = 0) -> M4_LexEN:
    return M4_LexEN(seed, D, reserve_pool)


def M4_get(lex: M4_LexEN, token: str, use_pool: bool = False) -> np.ndarray:
    return lex.get(token, use_pool=use_pool)


def M4_get_many(lex: M4_LexEN, words: Iterable[str], use_pool: bool = False) -> np.ndarray:
    return lex.get_many(words, use_pool=use_pool)


def M4_collision_audit(lex: M4_LexEN, vocab: List[str]) -> float:
    mats = np.stack([lex.get(w) for w in vocab], axis=0).astype(np.int16, copy=False)
    D = mats.shape[1]
    gram = (mats @ mats.T) / D
    np.fill_diagonal(gram, 0.0)
    return float(np.max(np.abs(gram)))


def M4_pair_stats(lex: M4_LexEN, vocab: List[str], n_pairs: int = 50_000, seed: int = 5) -> tuple[float, float]:
    g = np.random.default_rng(seed)
    V = len(vocab)
    sims = np.empty(int(n_pairs), dtype=np.float64)
    for k in range(int(n_pairs)):
        i = int(g.integers(0, V))
        j = int(g.integers(0, V))
        while j == i:
            j = int(g.integers(0, V))
        Xi = lex.get(vocab[i]).astype(np.int32, copy=False)
        Xj = lex.get(vocab[j]).astype(np.int32, copy=False)
        sims[k] = float(np.dot(Xi, Xj) / lex.D)
    return float(sims.mean()), float(sims.var(ddof=1))
