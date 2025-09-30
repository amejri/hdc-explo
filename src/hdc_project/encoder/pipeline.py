"""High-level encoder pipeline (M8) and evaluation helpers."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .m2 import M2_perm_pow
from .m3 import M3_bind
from .m4 import M4_LexEN, M4_LexEN_new
from .m5 import (
    M5_ngram,
    M5_ngram_cached_strict,
    M5_ngram_cached_unbiased,
)
from .m6 import (
    M6_SegAcc_init,
    M6_SegAcc_push,
    M6_SegAcc_sign,
)
from .m7 import M7_SegMgr_new, M7_curKey
from .utils import unbiased_sign_int8, strict_sign_int8

__all__ = [
    "M8_ENC",
    "enc_sentence_ENC",
    "encode_corpus_ENC",
    "intra_inter_ngram_sims",
    "inter_segment_similarity",
    "majority_error_curve",
    "majority_curve_repeated_vector",
    "tokenize_en",
    "build_vocab_EN",
    "sentence_to_tokens_EN",
    "opus_load_subset",
    "validate_on_opus",
    "validate_on_opus_enc3",
]


def _ensure_pi_pows(pi: np.ndarray, n: int) -> list[np.ndarray]:
    pi_pows = [np.arange(pi.shape[0], dtype=np.int64)]
    for _ in range(n):
        pi_pows.append(pi[pi_pows[-1]])
    return pi_pows


def M8_ENC(
    tokens: Iterable[str],
    pi: np.ndarray,
    n: int,
    LexEN: M4_LexEN,
    D: int,
    segmgr: Any | None = None,
    acc_S: np.ndarray | None = None,
    return_bound: bool = False,
    pi_pows: list[np.ndarray] | None = None,
    majority_mode: str = "unbiased",
    m5_variant: str = "auto",
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray] | Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray
]:
    tokens_list = list(tokens)
    if segmgr is None:
        seg_seed = int(LexEN.rng.integers(1, 2**31 - 1))
        segmgr = M7_SegMgr_new(seg_seed, D)
    K_s = M7_curKey(segmgr)

    S = acc_S if acc_S is not None else M6_SegAcc_init(D)

    if pi_pows is None:
        pi_pows = _ensure_pi_pows(pi, max(n, len(tokens_list)) + 2)

    rng = getattr(LexEN, "rng", np.random.default_rng())

    def pick_ngram(window: Sequence[str]) -> np.ndarray:
        if m5_variant in {"unbiased", "auto"}:
            try:
                return M5_ngram_cached_unbiased(LexEN, pi_pows, window, rng=rng)
            except NameError:
                if m5_variant == "unbiased":
                    raise
        if m5_variant in {"strict", "auto"}:
            try:
                return M5_ngram_cached_strict(LexEN, pi_pows, window)
            except NameError:
                if m5_variant == "strict":
                    raise
        return M5_ngram(LexEN, pi, window)

    E_seq: List[np.ndarray] = []
    X_seq: List[np.ndarray] = []
    Xb_seq: Optional[List[np.ndarray]] = [] if return_bound else None

    for t in range(len(tokens_list)):
        left = max(0, t - n + 1)
        window = tokens_list[left : t + 1]
        E_t = pick_ngram(window)
        Delta = t - left
        idx = pi_pows[Delta]
        X_t = E_t[idx].astype(np.int8, copy=False)
        S = M6_SegAcc_push(S, X_t, K_s)
        if return_bound and Xb_seq is not None:
            Xb_seq.append(M3_bind(X_t, K_s))
        E_seq.append(E_t)
        X_seq.append(X_t)

    if majority_mode == "unbiased":
        H = unbiased_sign_int8(S, rng_obj=rng)
    elif majority_mode == "strict":
        H = strict_sign_int8(S)
    else:
        raise ValueError("majority_mode must be 'unbiased' or 'strict'")

    if return_bound:
        return E_seq, X_seq, Xb_seq or [], S, H
    return E_seq, X_seq, S, H


def enc_sentence_ENC(
    sentence_tokens: List[str],
    n: int,
    pi: np.ndarray,
    LexEN: M4_LexEN,
    D: int,
    seg_seed: int,
) -> Dict[str, Any]:
    tokens = list(sentence_tokens)
    segmgr = M7_SegMgr_new(seg_seed, D)
    E_seq, X_seq, S, H = M8_ENC(tokens, pi, n, LexEN, D, segmgr=segmgr)
    return {
        "E_seq": E_seq,
        "X_seq": X_seq,
        "S": S,
        "H": H,
        "len": len(tokens),
        "seg_seed": seg_seed,
    }


def encode_corpus_ENC(
    sentences: List[str],
    LexEN: M4_LexEN,
    pi: np.ndarray,
    D: int,
    n: int,
    seg_seed0: int = 10_001,
    log_every: int = 1_000,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seg_seed0)
    for idx, sent in enumerate(sentences, 1):
        toks = sentence_to_tokens_EN(sent, vocab=set())
        seg_seed = int(rng.integers(1, 2**31 - 1))
        out.append(enc_sentence_ENC(toks, n, pi, LexEN, D, seg_seed))
        if log_every and idx % log_every == 0:
            pass  # caller configures logging if needed
    return out


def intra_inter_ngram_sims(E_seq_list: List[List[np.ndarray]], D: int) -> Tuple[float, float]:
    intra_vals: List[float] = []
    for seq in E_seq_list:
        for a in range(len(seq) - 1):
            intra_vals.append(float(np.dot(seq[a], seq[a + 1]) / D))
    s_intra = float(np.mean(intra_vals)) if intra_vals else 0.0

    inter_vals: List[float] = []
    for i in range(len(E_seq_list) - 1):
        if E_seq_list[i] and E_seq_list[i + 1]:
            inter_vals.append(abs(float(np.dot(E_seq_list[i][0], E_seq_list[i + 1][0]) / D)))
    s_inter = float(np.mean(inter_vals)) if inter_vals else 0.0
    return s_intra, s_inter


def inter_segment_similarity(H_list: List[np.ndarray]) -> float:
    vals = [abs(float(np.dot(H_list[i], H_list[i + 1]) / H_list[i].size)) for i in range(len(H_list) - 1)]
    return float(np.mean(vals)) if vals else 0.0


def majority_error_curve(
    E_seq_list: List[List[np.ndarray]],
    pi: np.ndarray,
    D: int,
    eta_list: Tuple[float, ...] = (0.0, 0.05, 0.10),
    seed_noise: int = 303,
) -> Dict[float, List[Tuple[int, float]]]:
    rng = np.random.default_rng(seed_noise)
    results: Dict[float, List[Tuple[int, float]]] = {eta: [] for eta in eta_list}

    for E_seq in E_seq_list:
        m = len(E_seq)
        if m == 0:
            continue
        key_seed = int(rng.integers(1, 2**31 - 1))
        segmgr = M7_SegMgr_new(key_seed, D)
        K = M7_curKey(segmgr)

        S_clean = M6_SegAcc_init(D)
        for t, E_t in enumerate(E_seq):
            X_t = M2_perm_pow(E_t, pi, t).astype(np.int8, copy=False)
            S_clean = M6_SegAcc_push(S_clean, X_t, K)
        H_clean = M6_SegAcc_sign(S_clean)

        for eta in eta_list:
            S_noisy = M6_SegAcc_init(D)
            for t, E_t in enumerate(E_seq):
                X_t = M2_perm_pow(E_t, pi, t).astype(np.int8, copy=False)
                if eta > 0.0:
                    mask = rng.random(X_t.shape[0]) < eta
                    X_t = X_t.copy()
                    X_t[mask] = -X_t[mask]
                S_noisy = M6_SegAcc_push(S_noisy, X_t, K)
            H_noisy = M6_SegAcc_sign(S_noisy)
            diff = (H_clean.astype(np.int8) * H_noisy.astype(np.int8) == -1).mean()
            results[eta].append((m, float(diff)))

    aggregated: Dict[float, List[Tuple[int, float]]] = {}
    for eta, pairs in results.items():
        by_m: Dict[int, List[float]] = {}
        for m, val in pairs:
            by_m.setdefault(m, []).append(val)
        aggregated[eta] = sorted((m, float(np.mean(vals))) for m, vals in by_m.items())
    return aggregated


def majority_curve_repeated_vector(
    E_seq_list: List[List[np.ndarray]],
    pi: np.ndarray,
    D: int,
    eta_list: Tuple[float, ...] = (0.0, 0.05, 0.10),
    trials_per_m: int = 4_000,
    seed: int = 4242,
) -> Dict[float, List[Tuple[int, float]]]:
    m_grid = sorted({len(seq) for seq in E_seq_list if len(seq) > 0})
    if not m_grid:
        return {eta: [] for eta in eta_list}
    rng = np.random.default_rng(seed)
    out: Dict[float, List[Tuple[int, float]]] = {eta: [] for eta in eta_list}
    for eta in eta_list:
        p = 1.0 - float(eta)
        for m in m_grid:
            B = rng.random(size=(trials_per_m, m)) < p
            Y = np.where(B, 1, -1).astype(np.int16)
            S = Y.sum(axis=1)
            err = float((S <= 0).mean())
            out[eta].append((m, err))
    for eta in eta_list:
        out[eta].sort(key=lambda t: t[0])
    return out


# --- basic NLP helpers -----------------------------------------------------

def tokenize_en(s: str) -> List[str]:
    return [w.strip().lower() for w in s.split() if w.strip()]


def build_vocab_EN(ens: Iterable[str], V: int = 5_000) -> set[str]:
    counter = Counter()
    for sent in ens:
        counter.update(tokenize_en(sent))
    return {token for token, _ in counter.most_common(V)}


def sentence_to_tokens_EN(sentence: str, vocab: set[str]) -> List[str]:
    return tokenize_en(sentence)


# --- validation helper -----------------------------------------------------

def opus_load_subset(
    name: str = "opus_books",
    config: str = "en-fr",
    split: str = "train",
    N: int = 10_000,
    seed: int = 123,
) -> Tuple[List[str], List[str]]:
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(N, len(ds))))
    ens = [ex["translation"]["en"] for ex in ds]
    frs = [ex["translation"]["fr"] for ex in ds]
    return ens, frs


def validate_on_opus(
    D: int = 16_384,
    n: int = 3,
    N: int = 1_000,
    seed_lex: int = 10_123,
    seed_pi: int = 10_456,
) -> Dict[str, Any]:
    ens, _ = opus_load_subset("opus_books", "en-fr", "train", N=N, seed=2024)
    Lex = M4_LexEN_new(seed_lex, D, reserve_pool=4_096)
    pi = np.random.default_rng(seed_pi).permutation(D).astype(np.int64)
    encoded = encode_corpus_ENC(ens, Lex, pi, D, n, seg_seed0=9_001)
    E_list = [e["E_seq"] for e in encoded]
    H_list = [e["H"] for e in encoded]
    s_intra, s_inter = intra_inter_ngram_sims(E_list, D)
    s_inter_seg = inter_segment_similarity(H_list)
    maj_curves = majority_error_curve(E_list, pi, D, eta_list=(0.0, 0.05, 0.1))
    return {
        "D": D,
        "n": n,
        "N_pairs": N,
        "seed_lex": seed_lex,
        "seed_pi": seed_pi,
        "intra_ngram_mean_sim": s_intra,
        "inter_ngram_abs_mean_sim": s_inter,
        "inter_segment_abs_mean_sim": s_inter_seg,
        "majority_curves": maj_curves,
    }


def validate_on_opus_enc3(
    D: int = 16_384,
    n: int = 3,
    N: int = 1_000,
    seed_lex: int = 10_123,
    seed_pi: int = 10_456,
) -> Dict[str, Any]:
    ens, _ = opus_load_subset("opus_books", "en-fr", "train", N=N, seed=2024)
    Lex = M4_LexEN_new(seed_lex, D, reserve_pool=4_096)
    pi = np.random.default_rng(seed_pi).permutation(D).astype(np.int64)
    encoded = encode_corpus_ENC(ens, Lex, pi, D, n, seg_seed0=9_001)
    E_list = [e["E_seq"] for e in encoded]
    H_list = [e["H"] for e in encoded]
    s_intra, s_inter = intra_inter_ngram_sims(E_list, D)
    s_inter_seg = inter_segment_similarity(H_list)
    maj_curves = majority_curve_repeated_vector(
        E_list, pi, D, eta_list=(0.0, 0.05, 0.1), trials_per_m=4_000, seed=4_242
    )
    return {
        "D": D,
        "n": n,
        "N_pairs": N,
        "seed_lex": seed_lex,
        "seed_pi": seed_pi,
        "intra_ngram_mean_sim": s_intra,
        "inter_ngram_abs_mean_sim": s_inter,
        "inter_segment_abs_mean_sim": s_inter_seg,
        "majority_curves": maj_curves,
    }
