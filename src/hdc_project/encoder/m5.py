"""Module M5: n-gram construction primitives."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List

import numpy as np

from .m4 import M4_LexEN
from .utils import unbiased_sign_int8, strict_sign_int8, sign_int

__all__ = [
    "M5_precompute_pi_pows",
    "M5_ngram_cached",
    "M5_ngram",
    "M5_ngram_cached_strict",
    "M5_ngram_cached_unbiased",
    "M5_window_stream",
]


def M5_precompute_pi_pows(pi: np.ndarray, n: int) -> list[np.ndarray]:
    D = int(pi.shape[0])
    idxs = [np.arange(D, dtype=np.int64)]
    for _ in range(1, int(n)):
        idxs.append(pi[idxs[-1]])
    for arr in idxs:
        arr.setflags(write=False)
    return idxs


def _accumulate_ngram(lex: M4_LexEN, pi_pows: Sequence[np.ndarray], tokens: Sequence[str]) -> np.ndarray:
    D = lex.D
    acc = np.zeros(D, dtype=np.int16)
    for j, token in enumerate(reversed(tokens)):
        vec = lex.get(token).astype(np.int16, copy=False)
        acc += vec[pi_pows[j]]
    return acc


def M5_ngram_cached(lex: M4_LexEN, pi_pows: Sequence[np.ndarray], tokens: Sequence[str]) -> np.ndarray:
    return sign_int(_accumulate_ngram(lex, pi_pows, tokens))


def M5_ngram_cached_strict(lex: M4_LexEN, pi_pows: Sequence[np.ndarray], tokens: Sequence[str]) -> np.ndarray:
    return strict_sign_int8(_accumulate_ngram(lex, pi_pows, tokens))


def M5_ngram_cached_unbiased(
    lex: M4_LexEN,
    pi_pows: Sequence[np.ndarray],
    tokens: Sequence[str],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    return unbiased_sign_int8(_accumulate_ngram(lex, pi_pows, tokens), rng_obj=rng)


def M5_ngram(lex: M4_LexEN, pi: np.ndarray, tokens: Sequence[str]) -> np.ndarray:
    n = len(tokens)
    pi_pows = M5_precompute_pi_pows(pi, n if n > 0 else 1)
    return M5_ngram_cached(lex, pi_pows, tokens)


def M5_window_stream(lex: M4_LexEN, pi: np.ndarray, words: Sequence[str], n: int) -> List[np.ndarray]:
    pi_pows = M5_precompute_pi_pows(pi, n)
    out: List[np.ndarray] = []
    for t in range(n - 1, len(words)):
        window = words[max(0, t - n + 1): t + 1]
        out.append(M5_ngram_cached(lex, pi_pows, window))
    return out
