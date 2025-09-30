"""Module M6: Segment accumulation and majority."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Iterable

import numpy as np

from .m3 import M3_bind
from .utils import unbiased_sign_int8

__all__ = [
    "M6_SegAcc_init",
    "M6_SegAcc_push",
    "M6_SegAcc_pushPos",
    "M6_SegAcc_pushE",
    "M6_SegAcc_sign",
    "kl_half_vs_p",
    "M6_simulate_majority_error_fast",
    "M6_fit_loglinear",
]


def M6_SegAcc_init(D: int) -> np.ndarray:
    return np.zeros(int(D), dtype=np.int16)


def M6_SegAcc_push(S: np.ndarray, X_t: np.ndarray, K_s: np.ndarray | None = None) -> np.ndarray:
    """Accumulate a (possibly bound) vector into *S*.

    If *K_s* is provided, *X_t* is assumed to be the unbound positional vector and
    binding is performed inside the function. Otherwise *X_t* must already be bound.
    """
    if S.dtype != np.int16:
        S = S.astype(np.int16, copy=False)
    if K_s is not None:
        Xb = M3_bind(X_t, K_s).astype(np.int16, copy=False)
    else:
        Xb = np.asarray(X_t, dtype=np.int16, copy=False)
    S += Xb
    return S


def M6_SegAcc_pushPos(S: np.ndarray, X_t: np.ndarray, K_s: np.ndarray) -> np.ndarray:
    return M6_SegAcc_push(S, X_t, K_s)


def M6_SegAcc_pushE(
    S: np.ndarray,
    E_t: np.ndarray,
    Delta: int,
    K_s: np.ndarray,
    pi_pows: Sequence[np.ndarray],
) -> np.ndarray:
    idx = pi_pows[int(Delta)]
    X_t = E_t[idx].astype(np.int8, copy=False)
    return M6_SegAcc_push(S, X_t, K_s)


def M6_SegAcc_sign(S: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    return unbiased_sign_int8(S, rng_obj=rng)


def kl_half_vs_p(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0, 1)")
    return 0.5 * np.log(1.0 / (4.0 * p * (1.0 - p)))


def M6_simulate_majority_error_fast(
    D: int,
    m_list: Iterable[int],
    p: float,
    trials: int = 5_000,
    seed: int | None = None,
    strict_tie: bool = True,
    batch_trials: int = 2_048,
) -> list[tuple[int, float, float]]:
    g = np.random.default_rng(seed)
    out: list[tuple[int, float, float]] = []
    for m in m_list:
        thr = (m // 2) + 1 if strict_tie else (m + 1) // 2
        total_ok = 0
        done = 0
        while done < trials:
            b = min(batch_trials, trials - done)
            cnt = g.binomial(n=m, p=p, size=(b, D)).astype(np.int16, copy=False)
            total_ok += int((cnt >= thr).sum())
            done += b
        err_emp = 1.0 - (total_ok / float(trials * D))
        hoeff = float(np.exp(-0.5 * m * (2 * p - 1) ** 2))
        out.append((int(m), float(err_emp), hoeff))
    return out


def M6_fit_loglinear(points: Sequence[tuple[int, float, float]], m_min_for_fit: int = 32) -> dict[str, float]:
    arr = np.asarray(points, dtype=np.float64)
    m_all = arr[:, 0]
    y_all = np.log(np.maximum(arr[:, 1], 1e-14))
    mask = m_all >= m_min_for_fit
    m = m_all[mask]
    y = y_all[mask]
    A = np.vstack([np.ones_like(m), m]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {"intercept": float(coef[0]), "slope": float(coef[1]), "R2": R2}
