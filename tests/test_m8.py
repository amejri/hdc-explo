import numpy as np

from hdc_project.encoder import m2, m4, pipeline


def _build_pi_pows_with_neg(pi: np.ndarray, max_abs_k: int) -> dict[int, np.ndarray]:
    D = pi.size
    pows_pos = [np.arange(D, dtype=np.int64)]
    for _ in range(max_abs_k):
        pows_pos.append(pi[pows_pos[-1]])
    pi_inv = np.empty_like(pi)
    pi_inv[pi] = np.arange(D, dtype=np.int64)
    pows_neg = [np.arange(D, dtype=np.int64)]
    for _ in range(max_abs_k):
        pows_neg.append(pi_inv[pows_neg[-1]])
    out = {}
    for k in range(max_abs_k + 1):
        out[k] = pows_pos[k]
        out[-k] = pows_neg[k]
    return out


def deltas_for_len(T: int, n: int) -> list[int]:
    out = []
    for t in range(T):
        left = max(0, t - n + 1)
        out.append(t - left)
    return out


def test_m8_isometry_alignment():
    D = 8192
    n = 3
    Lex = m4.M4_LexEN_new(202, D)
    pi = m2.M2_plan_perm(D, seed=203)

    sent1 = ["i", "love", "cats", "very", "much"]
    sent2 = ["we", "like", "music"]

    E1, X1, S1, H1 = pipeline.M8_ENC(sent1, pi, n, Lex, D)
    E2, X2, S2, H2 = pipeline.M8_ENC(sent2, pi, n, Lex, D)

    pows = _build_pi_pows_with_neg(pi, max_abs_k=n - 1)

    def gram_aligned(E_seq, X_seq, n):
        T = len(E_seq)
        D = E_seq[0].size
        delta = deltas_for_len(T, n)
        A = np.stack(X_seq).astype(np.int16)
        Gx = (A @ A.T) / D
        Ge = np.empty_like(Gx)
        B = np.stack(E_seq).astype(np.int16)
        for a in range(T):
            Ea = B[a]
            for b in range(T):
                k = delta[b] - delta[a]
                Eb_shift = B[b, pows[k]]
                Ge[a, b] = float(np.dot(Ea, Eb_shift) / D)
        return Gx, Ge

    Gx1, Ge1 = gram_aligned(E1, X1, n)
    Gx2, Ge2 = gram_aligned(E2, X2, n)

    assert np.max(np.abs(Gx1 - Ge1)) <= 1e-12
    assert np.max(np.abs(Gx2 - Ge2)) <= 1e-12
