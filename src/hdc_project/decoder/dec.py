"""DEC primitives and diagnostics extracted from the exploration notebook.

The numbered ``DD*`` helpers are the decoder counterparts of the encoder
``M*`` modules. They operate on +/-1 hypervectors stored as ``np.int8`` arrays and
implement the core steps of the DEC pipeline:

``DD1``   Context rebinding with the DEC key.
``DD2``   Query construction from the encoder trace and the language history.
``DD3``   Memory rebinding that maps queries into the MEM space.
``DD4``   Retrieval of the top-K memory prototypes.
``DD5``   Payload sealing that produces a +/-1 vector.
``DD6``   Dual-space voting that mixes MEM and LM evidence.
``DD7``   Language-model update through a single permutation step.

The notebook also bundled several diagnostic routines labelled ``DX*``. They
are re-exported here so that the same validation experiments can be run from
unit tests or scripts without copy/paste.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from hdc_project.encoder.utils import sign_int

__all__ = [
    "DD1_ctx",
    "DD2_query",
    "DD2_query_bin",
    "DD3_bindToMem",
    "DD4_search_topK",
    "DD5_payload",
    "DD6_vote",
    "DD7_updateLM",
    "DecodeOneStep",
    "DX2_run",
    "DX3_run",
    "DX4_run",
    "DX5_run",
    "DX6_run",
    "DX7_run",
]


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def hd_assert_pm1(vec: np.ndarray, D: int | None = None) -> None:
    """Ensure that *vec* is an ``np.int8`` array with entries in ``{+1,-1}``."""
    if not isinstance(vec, np.ndarray):
        raise TypeError("expected an np.ndarray")
    if vec.dtype != np.int8:
        raise TypeError("expected dtype=int8")
    if not np.all((vec == 1) | (vec == -1)):
        raise ValueError("expected +/-1 entries")
    if D is not None:
        if vec.ndim != 1 or vec.shape[0] != D:
            raise ValueError(f"expected shape ({D},)")


def hd_bind(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Return the Hadamard product of two +/-1 vectors in ``np.int8``."""
    return (lhs.astype(np.int8, copy=False) * rhs.astype(np.int8, copy=False)).astype(
        np.int8,
        copy=False,
    )


def hd_sim(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return the cosine-like similarity ``<lhs, rhs> / D`` in float64."""
    if lhs.shape != rhs.shape:
        raise ValueError("shape mismatch for similarity")
    return float(lhs.astype(np.int32) @ rhs.astype(np.int32) / lhs.shape[0])


def build_perm_inverse(pi: np.ndarray) -> np.ndarray:
    """Return the inverse permutation of ``pi``."""
    if pi.ndim != 1 or not np.issubdtype(pi.dtype, np.integer):
        raise ValueError("pi must be a 1-D permutation")
    pi_inv = np.empty_like(pi)
    pi_inv[pi] = np.arange(pi.shape[0], dtype=pi.dtype)
    return pi_inv


def permute_pow(vec: np.ndarray, pi: np.ndarray, power: int) -> np.ndarray:
    """Apply ``pi`` ``power`` times to *vec* (negative powers use ``pi^{-1}``)."""
    if power == 0:
        return vec
    D = vec.shape[0]
    idx = np.arange(D, dtype=np.int64)
    if power > 0:
        step = power % D
        for _ in range(step):
            idx = pi[idx]
    else:
        pi_inv = build_perm_inverse(pi)
        step = (-power) % D
        for _ in range(step):
            idx = pi_inv[idx]
    return vec[idx].astype(np.int8, copy=False)


def permute_pow_signed(vec: np.ndarray, pi: np.ndarray, pi_inv: np.ndarray, power: int) -> np.ndarray:
    """Apply ``pi`` (or ``pi^{-1}``) repeatedly while staying in ``np.int8``."""
    if power == 0:
        return vec
    D = vec.shape[0]
    idx = np.arange(D, dtype=np.int64)
    if power > 0:
        step = power % D
        for _ in range(step):
            idx = pi[idx]
    else:
        step = (-power) % D
        for _ in range(step):
            idx = pi_inv[idx]
    return vec[idx].astype(np.int8, copy=False)


def rademacher(D: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a +/-1 hypervector of length *D* using the provided RNG."""
    return (2 * rng.integers(0, 2, size=D, dtype=np.int8) - 1).astype(np.int8, copy=False)


# ---------------------------------------------------------------------------
# DEC primitives (DD1 - DD7)
# ---------------------------------------------------------------------------

def DD1_ctx(Hs: np.ndarray, G_DEC: np.ndarray) -> np.ndarray:
    """Rebind the encoder state ``Hs`` with the DEC key ``G_DEC``."""
    hd_assert_pm1(Hs)
    hd_assert_pm1(G_DEC, Hs.shape[0])
    return hd_bind(Hs, G_DEC)


def DD2_query(
    Qs: np.ndarray,
    hist_vectors: Sequence[np.ndarray],
    pi: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    ell: int = 4,
) -> np.ndarray:
    """Return the float64 query used for diagnostics (pre-threshold)."""
    D = Qs.shape[0]
    hd_assert_pm1(Qs, D)
    if pi.ndim != 1 or pi.shape[0] != D or not np.issubdtype(pi.dtype, np.integer):
        raise ValueError("invalid permutation")
    ell = min(int(ell), len(hist_vectors))
    pi_inv = build_perm_inverse(pi)

    if ell == 0:
        H_hist = np.ones(D, dtype=np.int8)
    else:
        acc = np.zeros(D, dtype=np.int16)
        for j in range(1, ell + 1):
            Lj = hist_vectors[j - 1]
            hd_assert_pm1(Lj, D)
            acc += permute_pow_signed(Lj, pi, pi_inv, j).astype(np.int16, copy=False)
        H_hist = sign_int(acc)

    Rt = alpha * Qs.astype(np.float64) + beta * H_hist.astype(np.float64)
    norm = float(np.linalg.norm(Rt))
    if norm > 0.0:
        Rt = Rt / norm * np.sqrt(D)
    else:
        Rt = np.ones(D, dtype=np.float64)
    return Rt


def DD2_query_bin(
    Qs: np.ndarray,
    history_fr: Sequence[str],
    L_fr: Callable[[str], np.ndarray],
    Pi: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    ell: int = 4,
) -> np.ndarray:
    """Return a +/-1 ``np.int8`` query that mixes ENC and LM history."""
    D = Qs.shape[0]
    hd_assert_pm1(Qs, D)
    if Pi.ndim != 1 or Pi.shape[0] != D or not np.issubdtype(Pi.dtype, np.integer):
        raise ValueError("invalid permutation")

    Pi_inv = build_perm_inverse(Pi)
    hist_vecs: List[np.ndarray] = []
    for tok in list(history_fr)[:ell]:
        Lv = L_fr(tok).astype(np.int8, copy=False)
        hd_assert_pm1(Lv, D)
        hist_vecs.append(Lv)

    if not hist_vecs:
        H_hist = np.ones(D, dtype=np.int8)
    else:
        acc = np.zeros(D, dtype=np.int16)
        for j, L_j in enumerate(hist_vecs, start=1):
            acc += permute_pow_signed(L_j, Pi, Pi_inv, j).astype(np.int16, copy=False)
        H_hist = sign_int(acc)

    combo = (alpha * Qs.astype(np.float32, copy=False)) + (beta * H_hist.astype(np.float32, copy=False))
    return np.where(combo >= 0.0, 1, -1).astype(np.int8, copy=False)


def DD3_bindToMem(Rt: np.ndarray, G_MEM: np.ndarray) -> np.ndarray:
    """Rebind the query ``Rt`` with the MEM key ``G_MEM``."""
    hd_assert_pm1(Rt)
    hd_assert_pm1(G_MEM, Rt.shape[0])
    return hd_bind(Rt, G_MEM)


def hd_sim_dot(lhs: np.ndarray, rhs: np.ndarray) -> int:
    """Return the unnormalised dot product between +/-1 vectors (int result)."""
    return int(lhs.astype(np.int32) @ rhs.astype(np.int32))


def DD4_search_topK(Rt_tilde: np.ndarray, prototypes: np.ndarray, K: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """Return the index of the best prototype together with the top-K slice."""
    D = Rt_tilde.shape[0]
    if prototypes.ndim != 2 or prototypes.shape[1] != D or prototypes.dtype != np.int8:
        raise ValueError("prototypes must be an (B, D) int8 matrix")

    proto = prototypes.astype(np.int8, copy=False)
    # Ignore empty prototype rows (never trained buckets) to avoid zero vectors ranking first.
    valid_idx = np.flatnonzero(np.any(proto != 0, axis=1))
    if valid_idx.size == 0:
        raise ValueError("no non-empty prototypes available for search")

    Rt32 = Rt_tilde.astype(np.int32, copy=False)
    scores_all = (proto.astype(np.int32, copy=False) @ Rt32).astype(np.int32, copy=False)

    K_eff = int(min(max(K, 1), valid_idx.size))
    scores_valid = scores_all[valid_idx]
    part = np.argpartition(scores_valid, -K_eff)[-K_eff:]
    order_local = part[np.argsort(scores_valid[part])[::-1]]
    top_order = valid_idx[order_local]
    c_star = int(top_order[0])
    return c_star, top_order.astype(np.int64, copy=False), scores_all[top_order]


def DD5_payload(Mc: np.ndarray) -> np.ndarray:
    """Seal the payload accumulator by applying a strict sign."""
    if Mc.dtype == np.int8:
        hd_assert_pm1(Mc)
        return Mc
    return sign_int(Mc)


def _batch_lex(cand_vocab: Sequence[str], L: Callable[[str], np.ndarray]) -> np.ndarray:
    mats = []
    for tok in cand_vocab:
        vec = L(tok).astype(np.int8, copy=False)
        hd_assert_pm1(vec)
        mats.append(vec)
    if not mats:
        raise ValueError("empty candidate vocabulary")
    return np.vstack(mats).astype(np.int8, copy=False)


def DD6_vote(
    Z_hat: np.ndarray,
    H_LM: np.ndarray,
    L_mem: Callable[[str], np.ndarray],
    L_lm: Callable[[str], np.ndarray],
    cand_vocab: Sequence[str],
    lam: float = 0.0,
    *,
    normalize: str = "sqrtD",
    return_probs: bool = False,
    tau: float = 1.0,
) -> Tuple[str, np.ndarray, Optional[np.ndarray]]:
    """Mix MEM and LM evidence over ``cand_vocab`` and return the winning token."""
    D = int(Z_hat.shape[0])
    hd_assert_pm1(Z_hat, D)
    hd_assert_pm1(H_LM, D)
    if not isinstance(cand_vocab, (list, tuple)) or len(cand_vocab) == 0:
        raise ValueError("candidate vocabulary is empty")
    M_mem = _batch_lex(cand_vocab, L_mem)
    M_lm = _batch_lex(cand_vocab, L_lm)
    if M_mem.shape != M_lm.shape:
        raise ValueError("lexical tables must have identical shapes")

    z32 = Z_hat.astype(np.int32, copy=False)
    h32 = H_LM.astype(np.int32, copy=False)
    scores_raw = (M_mem.astype(np.int32, copy=False) @ z32).astype(np.float64)
    scores_raw += float(lam) * (M_lm.astype(np.int32, copy=False) @ h32).astype(np.float64)

    best_idx = int(np.argmax(scores_raw))
    token_star = cand_vocab[best_idx]

    probs: Optional[np.ndarray] = None
    if return_probs:
        if normalize not in {"sqrtD", "none"}:
            raise ValueError("normalize must be 'sqrtD' or 'none'")
        scale = np.sqrt(D) if normalize == "sqrtD" else 1.0
        logits = scores_raw / (max(1e-6, float(tau)) * scale)
        logits = logits - np.max(logits)
        exps = np.exp(logits, dtype=np.float64)
        probs = (exps / np.sum(exps, dtype=np.float64)).astype(np.float64, copy=False)

    return token_star, scores_raw, probs


def DD7_updateLM(H_LM: np.ndarray, v_hat: str, L_fr: Callable[[str], np.ndarray], Pi: np.ndarray) -> np.ndarray:
    """Update the LM state by adding ``Pi^1 L_fr(v_hat)`` followed by a strict sign."""
    D = H_LM.shape[0]
    hd_assert_pm1(H_LM, D)
    Lv = L_fr(v_hat).astype(np.int8, copy=False)
    hd_assert_pm1(Lv, D)
    increment = permute_pow(Lv, Pi, 1).astype(np.int16, copy=False)
    acc = H_LM.astype(np.int16) + increment
    return sign_int(acc)


# ---------------------------------------------------------------------------
# Notebook diagnostics (DX*)
# ---------------------------------------------------------------------------

log = logging.getLogger("DEC")


def ks_2samp_asymp(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Kolmogorov-Smirnov two-sample statistic with an asymptotic p-value."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, m = x.size, y.size
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    i = j = 0
    cdf_x = cdf_y = 0.0
    d_stat = 0.0

    while i < n and j < m:
        if x_sorted[i] < y_sorted[j]:
            i += 1
            cdf_x = i / n
        elif x_sorted[i] > y_sorted[j]:
            j += 1
            cdf_y = j / m
        else:
            v = x_sorted[i]
            while i < n and x_sorted[i] == v:
                i += 1
            while j < m and y_sorted[j] == v:
                j += 1
            cdf_x = i / n
            cdf_y = j / m
        d_stat = max(d_stat, abs(cdf_x - cdf_y))

    if i < n:
        d_stat = max(d_stat, abs(1.0 - (j / m if m > 0 else 0.0)))
    if j < m:
        d_stat = max(d_stat, abs(1.0 - (i / n if n > 0 else 0.0)))

    if d_stat == 0.0:
        return 0.0, 1.0

    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / max(en, 1e-12)) * d_stat
    if lam < 1e-8:
        return float(d_stat), 1.0

    terms = [np.exp(-2.0 * (k ** 2) * (lam ** 2)) for k in range(1, 201)]
    pval = 2.0 * sum(((-1) ** (k - 1)) * terms[k - 1] for k in range(1, len(terms) + 1))
    pval = float(max(0.0, min(1.0, pval)))
    return float(d_stat), pval


def DX2_run(
    D: int = 16_384,
    trials: int = 200,
    ells: Sequence[int] = (2, 4, 8),
    ratios: Sequence[float] = (1 / 3, 1.0, 3.0),
    seed: int = 2025,
) -> Dict[Tuple[int, float], Tuple[float, float, float]]:
    """Replicate the DX2 norm and permutation diagnostics."""
    rng = np.random.default_rng(seed)
    pi = np.arange(D, dtype=np.int64)
    rng.shuffle(pi)
    pi_inv = build_perm_inverse(pi)

    def sim(a: np.ndarray, b: np.ndarray) -> float:
        return float((a.astype(np.int32) @ b.astype(np.int32)) / D)

    norms: Dict[Tuple[int, float], Tuple[float, float, float]] = {}
    gram_uniform_ok = True
    pair_shift_ok = True

    for ell in ells:
        for ratio in ratios:
            alpha = ratio
            beta = 1.0
            values: List[float] = []
            for _ in range(trials):
                Qs = rademacher(D, rng)
                hist = [rademacher(D, rng) for _ in range(ell)]

                P = np.stack(
                    [permute_pow_signed(hist[j], pi, pi_inv, j + 1) for j in range(ell)],
                    axis=0,
                ).astype(np.int8, copy=False)

                s = int(rng.integers(1, 7))
                P_uni = np.stack(
                    [permute_pow_signed(P[j], pi, pi_inv, s) for j in range(ell)],
                    axis=0,
                ).astype(np.int8, copy=False)
                G = (P.astype(np.int32) @ P.T.astype(np.int32)) / D
                Gu = (P_uni.astype(np.int32) @ P_uni.T.astype(np.int32)) / D
                if not np.allclose(G, Gu, atol=5e-3, rtol=0):
                    gram_uniform_ok = False

                for i in range(1, ell + 1):
                    for k in range(1, ell + 1):
                        lhs = sim(
                            permute_pow_signed(hist[i - 1], pi, pi_inv, i),
                            permute_pow_signed(hist[k - 1], pi, pi_inv, k),
                        )
                        rhs = sim(
                            hist[i - 1],
                            permute_pow_signed(hist[k - 1], pi, pi_inv, k - i),
                        )
                        if abs(lhs - rhs) > 5e-3:
                            pair_shift_ok = False
                            break
                    if not pair_shift_ok:
                        break

                Rt = DD2_query(Qs, hist, pi, alpha=alpha, beta=beta, ell=ell)
                values.append(float(np.linalg.norm(Rt) / np.sqrt(D)))

            norms[(int(ell), float(ratio))] = (min(values), float(np.median(values)), max(values))

    if not gram_uniform_ok:
        raise AssertionError("DX2: Gram matrix is not invariant under uniform permutation")
    if not pair_shift_ok:
        raise AssertionError("DX2: pairwise shift identity violated")

    band_ok = all(
        (0.9 <= mn <= 1.1) and (0.9 <= md <= 1.1) and (0.9 <= mx <= 1.1)
        for (mn, md, mx) in norms.values()
    )
    if not band_ok:
        raise AssertionError("DX2: norm(Rt)/sqrt(D) outside [0.9, 1.1]")

    return norms


def DX3_run(
    D: int = 16_384,
    C: int = 500,
    T: int = 200,
    seed: int = 2025,
    rel_tol: float = 0.01,
    pmin: float = 0.10,
) -> Tuple[float, float]:
    """Verify that rebinding commutes with MEM lookup through relative error and KS tests."""
    rng = np.random.default_rng(seed)
    G_MEM = rademacher(D, rng)
    M_bank = np.stack([rademacher(D, rng) for _ in range(C)], axis=0).astype(np.int8)
    Q_batch = np.stack([rademacher(D, rng) for _ in range(T)], axis=0).astype(np.int8)

    S_mem = np.zeros((T, C), dtype=np.int32)
    S_unbd = np.zeros((T, C), dtype=np.int32)

    for t in range(T):
        Rt = Q_batch[t]
        Rt_mem = DD3_bindToMem(Rt, G_MEM)
        for c in range(C):
            Mc = M_bank[c]
            S_mem[t, c] = hd_sim_dot(Rt_mem, Mc)
            S_unbd[t, c] = hd_sim_dot(Rt, hd_bind(Mc, G_MEM))

    A = S_mem.astype(np.float64).ravel()
    B = S_unbd.astype(np.float64).ravel()
    denom = np.maximum(1.0, np.abs(B))
    rel_err = np.abs(A - B) / denom
    rel_err_mean = float(np.mean(rel_err))

    d_stat, pval = ks_2samp_asymp(A, B)

    if rel_err_mean > rel_tol:
        raise AssertionError("DX3: relative error too large")
    if pval <= pmin:
        raise AssertionError("DX3: KS p-value below threshold")

    return rel_err_mean, pval


def DX4_run(
    D: int = 16_384,
    B: int = 10_000,
    trials: int = 200,
    Ks: Sequence[int] = (100, 500, 2_000),
    seed: int = 0,
) -> Dict[int, float]:
    """Empirical recall of ``DD4_search_topK`` over random prototypes."""
    rng = np.random.default_rng(seed)
    recalls = {int(K): 0 for K in Ks}
    for _ in range(trials):
        prototypes = rng.choice([-1, 1], size=(B, D)).astype(np.int8)
        c_star = int(rng.integers(0, B))
        Rt = prototypes[c_star].copy()
        _, C_K, _ = DD4_search_topK(Rt, prototypes, int(max(Ks)))
        for K in Ks:
            if c_star in C_K[: int(K)]:
                recalls[int(K)] += 1
    return {K: recalls[K] / trials for K in recalls}


def DX5_run(
    D: int = 16_384,
    trials: int = 200,
    ms: Sequence[int] = (4, 8, 16),
    seed: int = 0,
) -> Dict[int, float]:
    """Measure payload sealing accuracy as a function of the accumulator depth."""
    rng = np.random.default_rng(seed)
    accuracies: Dict[int, float] = {}
    for m in ms:
        accs: List[float] = []
        for _ in range(trials):
            ref = rng.choice([-1, 1], size=D).astype(np.int8)
            acc = np.zeros(D, dtype=np.int32)
            for _ in range(int(m)):
                acc += ref
            Z_hat = DD5_payload(acc)
            accs.append(float(np.mean(Z_hat == ref)))
        accuracies[int(m)] = float(np.mean(accs))
    return accuracies


def _softmax_probs(scores: np.ndarray, D: int, tau: float = 1.0) -> np.ndarray:
    logits = scores / (np.sqrt(D) * max(float(tau), 1e-6))
    logits = logits - np.max(logits)
    exps = np.exp(logits, dtype=np.float64)
    return exps / np.sum(exps, dtype=np.float64)


def hd_perplexity_from_scores(scores: np.ndarray, true_idx: int, D: int, tau: float = 1.0) -> float:
    p_true = float(_softmax_probs(scores, D=D, tau=tau)[true_idx])
    return float(np.exp(-np.log(max(p_true, 1e-12))))


def flip_to_target(vec: np.ndarray, target_sim: float, rng: np.random.Generator) -> np.ndarray:
    """Return a +/-1 vector with an expected similarity ``target_sim`` to ``vec``."""
    D = vec.shape[0]
    p_flip = max(0.0, min(1.0, (1.0 - float(target_sim)) / 2.0))
    mask = (rng.random(D) < p_flip).astype(np.int8)
    flips = (1 - 2 * mask).astype(np.int8, copy=False)
    return (vec.astype(np.int8, copy=False) * flips).astype(np.int8, copy=False)


def DX6_run(
    D: int = 16_384,
    trials: int = 400,
    lam_grid: Sequence[float] = (0.0, 0.5, 1.0),
    sim_payload: float = 0.60,
    sim_lm: float = 0.40,
    n_confounders: int = 6,
    rho_mem_conf: float = 0.55,
    rho_lm_conf: float = 0.10,
    tau: float = 1.0,
    rng_seed: int = 7_031,
) -> Dict[float, Dict[str, float]]:
    """Simulate the dual-space vote and report top-1 accuracy and perplexity."""
    rng = np.random.default_rng(rng_seed)
    stats = {float(lam): {"top1_hits": 0, "ppl_sum": 0.0} for lam in lam_grid}

    for _ in range(trials):
        Z_true = rademacher(D, rng)
        H_true = rademacher(D, rng)
        vocab_size = int(n_confounders) + 1
        cand_vocab = [f"tok{i}" for i in range(vocab_size)]
        true_tok = cand_vocab[0]

        table_mem: Dict[str, np.ndarray] = {}
        table_lm: Dict[str, np.ndarray] = {}
        table_mem[true_tok] = flip_to_target(Z_true, sim_payload, rng)
        table_lm[true_tok] = flip_to_target(H_true, sim_lm, rng)
        for idx in range(1, vocab_size):
            tok = cand_vocab[idx]
            table_mem[tok] = flip_to_target(Z_true, rho_mem_conf, rng)
            table_lm[tok] = flip_to_target(H_true, rho_lm_conf, rng)

        def L_mem(token: str) -> np.ndarray:
            return table_mem[token]

        def L_lm(token: str) -> np.ndarray:
            return table_lm[token]

        for lam in lam_grid:
            token_star, scores, probs = DD6_vote(
                Z_hat=Z_true,
                H_LM=H_true,
                L_mem=L_mem,
                L_lm=L_lm,
                cand_vocab=cand_vocab,
                lam=float(lam),
                normalize="sqrtD",
                return_probs=True,
                tau=tau,
            )
            if token_star == true_tok:
                stats[float(lam)]["top1_hits"] += 1
            if probs is not None:
                p_true = float(max(probs[0], 1e-12))
                ppl = float(np.exp(-np.log(p_true)))
            else:
                ppl = hd_perplexity_from_scores(scores, true_idx=0, D=D, tau=tau)
            stats[float(lam)]["ppl_sum"] += ppl

    results = {
        lam: {
            "top1": stats[lam]["top1_hits"] / trials,
            "ppl": stats[lam]["ppl_sum"] / trials,
        }
        for lam in stats
    }

    base_top1 = results[0.0]["top1"]
    base_ppl = results[0.0]["ppl"]
    saturated = abs(base_top1 - 1.0) < 1e-12
    if saturated:
        if not any(results[lam]["ppl"] < base_ppl - 1e-12 for lam in results if lam != 0.0):
            raise AssertionError("DX6: saturated regime but no perplexity improvement")
    else:
        if not any(
            (results[lam]["top1"] > base_top1 + 1e-12)
            and (results[lam]["ppl"] < base_ppl - 1e-12)
            for lam in results
            if lam != 0.0
        ):
            raise AssertionError("DX6: no lambda improves both top-1 and perplexity")

    return results


DEFAULT_ELL_GRID: Tuple[int, ...] = (2, 4, 8, 12)
CONF_PER_STEP = 8
TRIALS = 200
T_STEPS = 24
SIM_Y_MEM = 0.70
SIM_CONF_LM = 0.05
DEC_D = 16_384
RNG_SEED = 9_117


def correlated_pm1(proto: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Draw a +/-1 vector with approximate correlation ``rho`` to ``proto``."""
    noise = rademacher(proto.shape[0], rng)
    mix = rho * proto.astype(np.int32) + (1.0 - rho) * noise.astype(np.int32)
    return np.where(mix >= 0, 1, -1).astype(np.int8)


def DX7_eval_one_ell(ell: int, Pi: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    """Return the mean top-1 accuracy and correlation proxy for a single ``ell``."""
    top1_hits = 0
    p_sum = 0.0
    D = Pi.shape[0]

    vocab = [f"tok_{i}" for i in range(CONF_PER_STEP + 1)]
    table = {v: rademacher(D, rng) for v in vocab}

    def L(token: str) -> np.ndarray:
        return table[token]

    for _ in range(TRIALS):
        history: List[str] = []
        H_LM_pred = rademacher(D, rng)
        for _ in range(T_STEPS):
            if len(history) < ell:
                H_true = rademacher(D, rng)
            else:
                acc = np.zeros(D, dtype=np.int32)
                for j in range(1, ell + 1):
                    acc += permute_pow(L(history[-j]), Pi, j).astype(np.int32)
                H_true = sign_int(acc)

            y = vocab[0]
            L_y = correlated_pm1(H_true, SIM_Y_MEM, rng)

            cand_vectors = [L_y]
            cand_tokens = [y]
            for k in range(CONF_PER_STEP):
                tok = vocab[k + 1]
                L_tok = correlated_pm1(H_true, SIM_CONF_LM, rng)
                cand_vectors.append(L_tok)
                cand_tokens.append(tok)
            cand_vectors = np.stack(cand_vectors, axis=0)

            scores = cand_vectors.astype(np.int32) @ H_LM_pred.astype(np.int32)
            pred_idx = int(np.argmax(scores))
            v_hat = cand_tokens[pred_idx]
            if pred_idx == 0:
                top1_hits += 1

            sim = hd_sim(H_true, L_y)
            p_sum += 0.5 * (1.0 + sim)

            history.append(y)
            if len(history) > ell:
                history.pop(0)

            def L_temp(token: str, *, _vectors=cand_vectors, _tokens=cand_tokens) -> np.ndarray:
                return _vectors[_tokens.index(token)]

            H_LM_pred = DD7_updateLM(H_LM_pred, v_hat=v_hat, L_fr=L_temp, Pi=Pi)

    total = TRIALS * T_STEPS
    top1 = top1_hits / total
    p_ell = p_sum / total
    return float(top1), float(p_ell)


def DX7_run(
    ell_grid: Sequence[int] = DEFAULT_ELL_GRID,
    D: int = DEC_D,
    seed_pi: int = 10_456,
    rng_seed: int = RNG_SEED,
) -> Tuple[Dict[int, Dict[str, float]], int]:
    """Evaluate the LM update window ``ell`` and check monotonicity properties."""
    rng = np.random.default_rng(rng_seed)
    Pi = np.arange(D, dtype=np.int64)
    rng.shuffle(Pi)

    results: Dict[int, Dict[str, float]] = {}
    for ell in ell_grid:
        top1, p_ell = DX7_eval_one_ell(int(ell), Pi, rng)
        results[int(ell)] = {"top1": top1, "p": p_ell}

    ells_sorted = sorted(results.keys())
    top1s = np.array([results[e]["top1"] for e in ells_sorted], dtype=np.float64)
    ps = np.array([results[e]["p"] for e in ells_sorted], dtype=np.float64)

    ell_star = ells_sorted[int(np.argmax(top1s))]
    tail = ps[ells_sorted.index(ell_star) :]
    if not np.all(tail[:-1] >= tail[1:] - 1e-9):
        raise AssertionError("DX7: p(ell) does not decrease beyond ell*")

    return results, int(ell_star)


# ---------------------------------------------------------------------------
# High-level decode helper
# ---------------------------------------------------------------------------


def _as_vocab_from_buckets(
    C_K: np.ndarray,
    bucket2vocab: Optional[Union[Dict[int, List[str]], Callable[..., List[str]]]],
    history_fr: Sequence[str],
    global_fallback_vocab: Optional[Sequence[str]],
    *,
    min_size: int = 1,
    position: Optional[int] = None,
) -> List[str]:
    candidates: List[str] = []
    seen: set[str] = set()

    def add_many(tokens: Iterable[str]) -> None:
        for tok in tokens:
            if tok not in seen:
                seen.add(tok)
                candidates.append(tok)

    if bucket2vocab is not None:
        for idx, bucket in enumerate(C_K):
            if callable(bucket2vocab):
                try:
                    toks = bucket2vocab(int(bucket), position)
                except TypeError:
                    toks = bucket2vocab(int(bucket))
            else:
                toks = bucket2vocab.get(int(bucket), [])
            if toks:
                add_many(toks)
            if idx == 0 and len(candidates) >= min_size:
                break

    if len(candidates) < min_size and history_fr:
        add_many(list(history_fr))

    if len(candidates) < min_size and global_fallback_vocab is not None:
        add_many(list(global_fallback_vocab))

    if len(candidates) < min_size:
        return ["<unk>"]
    return candidates


def DecodeOneStep(
    Hs: np.ndarray,
    H_LM: np.ndarray,
    history_fr: Sequence[str],
    G_DEC: np.ndarray,
    G_MEM: np.ndarray,
    Pi: np.ndarray,
    L_fr: Callable[[str], np.ndarray],
    prototypes: np.ndarray,
    *,
    K: int = 500,
    alpha: float = 1.0,
    beta: float = 1.0,
    ell: int = 4,
    lam: float = 0.5,
    bucket2vocab: Optional[Union[Dict[int, List[str]], Callable[[int], List[str]]]] = None,
    global_fallback_vocab: Optional[Sequence[str]] = None,
    pos_key_lookup: Optional[Callable[[int], np.ndarray]] = None,
    return_ck_scores: bool = True,
) -> Tuple[str, np.ndarray, int, np.ndarray, Union[np.ndarray, np.ndarray]]:
    """Run a single DEC step that wires ``DD1`` through ``DD7``."""
    D = Hs.shape[0]
    hd_assert_pm1(Hs, D)
    hd_assert_pm1(H_LM, D)
    hd_assert_pm1(G_DEC, D)
    hd_assert_pm1(G_MEM, D)
    if Pi.ndim != 1 or Pi.shape[0] != D or not np.issubdtype(Pi.dtype, np.integer):
        raise ValueError("invalid permutation")
    if prototypes.ndim != 2 or prototypes.shape[1] != D:
        raise ValueError("prototypes must have shape (B, D)")

    Qs = DD1_ctx(Hs, G_DEC)
    Rt = DD2_query_bin(Qs, history_fr, L_fr, Pi, alpha=alpha, beta=beta, ell=ell)
    pos_idx = len(history_fr)
    if pos_key_lookup is not None:
        pos_vec = pos_key_lookup(pos_idx)
        if pos_vec is not None:
            pos_arr = np.asarray(pos_vec, dtype=np.int8, copy=False)
            hd_assert_pm1(pos_arr, D)
            Rt = hd_bind(Rt, pos_arr)
    Rt_tilde = DD3_bindToMem(Rt, G_MEM)
    c_star, C_K, scores_CK = DD4_search_topK(Rt_tilde, prototypes, K)
    Z_hat = DD5_payload(prototypes[c_star])

    cand_vocab = _as_vocab_from_buckets(
        C_K=C_K,
        bucket2vocab=bucket2vocab,
        history_fr=history_fr,
        global_fallback_vocab=global_fallback_vocab,
        min_size=1,
        position=pos_idx,
    )

    token_star, scores_cand, _ = DD6_vote(
        Z_hat,
        H_LM,
        L_mem=L_fr,
        L_lm=L_fr,
        cand_vocab=cand_vocab,
        lam=lam,
    )

    proto_vec_int = prototypes[c_star].astype(np.int32, copy=False)
    token_vec_int = L_fr(token_star).astype(np.int32, copy=False)
    if int(proto_vec_int @ token_vec_int) < 0:
        order = np.argsort(scores_cand)[::-1]
        replaced = False
        for idx_alt in order:
            candidate = cand_vocab[int(idx_alt)]
            cand_vec_int = L_fr(candidate).astype(np.int32, copy=False)
            if int(proto_vec_int @ cand_vec_int) >= 0:
                token_star = candidate
                replaced = True
                break
        if not replaced and global_fallback_vocab is not None:
            for fallback_tok in global_fallback_vocab:
                cand_vec_int = L_fr(fallback_tok).astype(np.int32, copy=False)
                if int(proto_vec_int @ cand_vec_int) >= 0:
                    token_star = fallback_tok
                    replaced = True
                    break

    H_LM_next = DD7_updateLM(H_LM, token_star, L_fr, Pi)

    if return_ck_scores:
        return token_star, scores_cand, int(c_star), C_K, scores_CK
    return token_star, scores_cand, int(c_star), C_K, H_LM_next
