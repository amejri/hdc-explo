import numpy as np

from hdc_project.encoder import m1, m2, m4, m5, m6, m7


def _build_pi_pows(pi: np.ndarray, max_power: int) -> list[np.ndarray]:
    pi_pows = [np.arange(pi.shape[0], dtype=np.int64)]
    for _ in range(max_power):
        pi_pows.append(pi[pi_pows[-1]])
    return pi_pows


def _build_H_segment(tokens, Lex, pi_pows, K, n, rng):
    D = Lex.D
    S = m6.M6_SegAcc_init(D)
    for t in range(len(tokens)):
        left = max(0, t - n + 1)
        window = tokens[left : t + 1]
        E_t = m5.M5_ngram_cached_unbiased(Lex, pi_pows, window, rng=rng)
        Delta = t - left
        S = m6.M6_SegAcc_push(S, E_t[pi_pows[Delta]], K)
    return m6.M6_SegAcc_sign(S, rng=rng)


def test_m7_resets_keep_segments_independent():
    D = 8192
    n = 3
    trials = 120
    g = np.random.default_rng(101)
    Lex = m4.M4_LexEN_new(102, D)
    pi = m2.M2_plan_perm(D, seed=103)
    pi_pows = _build_pi_pows(pi, 16)

    sims = []
    for _ in range(trials):
        SM1 = m7.M7_SegMgr_new(int(g.integers(1, 2**31 - 1)), D)
        SM2 = m7.M7_SegMgr_new(int(g.integers(1, 2**31 - 1)), D)
        sent1 = [f"w{int(g.integers(0, 2000))}" for _ in range(10)]
        sent2 = [f"w{int(g.integers(0, 2000))}" for _ in range(10)]
        H1 = _build_H_segment(sent1, Lex, pi_pows, m7.M7_curKey(SM1), n, g)
        H2 = _build_H_segment(sent2, Lex, pi_pows, m7.M7_curKey(SM2), n, g)
        sims.append(m1.M1_sim(H1, H2))
    sims = np.asarray(sims)
    assert abs(float(sims.mean())) <= 2e-2
    assert float((np.abs(sims) > 0.08).mean()) < 0.05


def test_m7_correlated_keys_raise_bias():
    D = 4096
    n = 3
    g = np.random.default_rng(202)
    Lex = m4.M4_LexEN_new(203, D)
    pi = m2.M2_plan_perm(D, seed=204)
    pi_pows = _build_pi_pows(pi, 16)
    rhos = [0.0, 0.05, 0.1]
    means = []
    for rho in rhos:
        vals = []
        for _ in range(60):
            SM = m7.M7_SegMgr_new(int(g.integers(1, 2**31 - 1)), D)
            base_key = m7.M7_curKey(SM)
            sent = [f"w{int(g.integers(0, 2000))}" for _ in range(10)]
            H1 = _build_H_segment(sent, Lex, pi_pows, base_key, n, g)
            corr_key = SM.nextKey_correlated(rho)
            sent2 = [f"w{int(g.integers(0, 2000))}" for _ in range(10)]
            H2 = _build_H_segment(sent2, Lex, pi_pows, corr_key, n, g)
            vals.append(m1.M1_sim(H1, H2))
        means.append(float(np.mean(vals)))
    assert means[0] <= means[1] + 0.01 <= means[2] + 0.02


def test_m7_leak_vs_delta_is_monotone():
    D = 8192
    n = 3
    g = np.random.default_rng(150)
    Lex = m4.M4_LexEN_new(151, D)
    pi = m2.M2_plan_perm(D, seed=152)
    pi_pows = _build_pi_pows(pi, 16)
    deltas = [0, 1, 3]
    sims = [[] for _ in deltas]
    for _ in range(80):
        sent = [f"w{int(g.integers(0, 2000))}" for _ in range(12)]
        sentp = [f"w{int(g.integers(0, 2000))}" for _ in range(12)]
        SM1 = m7.M7_SegMgr_new(int(g.integers(1, 2**31 - 1)), D)
        SM2 = m7.M7_SegMgr_new(int(g.integers(1, 2**31 - 1)), D)
        H1 = _build_H_segment(sent, Lex, pi_pows, m7.M7_curKey(SM1), n, g)
        for idx, delta in enumerate(deltas):
            S = m6.M6_SegAcc_init(D)
            K1 = m7.M7_curKey(SM2)
            K2 = m7.M7_onBoundary(SM2)
            for t, tok in enumerate(sentp):
                left = max(0, t - n + 1)
                window = sentp[left : t + 1]
                E_t = m5.M5_ngram_cached_unbiased(Lex, pi_pows, window, rng=g)
                Delta = t - left
                K = K1 if t < delta else K2
                S = m6.M6_SegAcc_push(S, E_t[pi_pows[Delta]], K)
            H2 = m6.M6_SegAcc_sign(S, rng=g)
            sims[idx].append(m1.M1_sim(H1, H2))
    means = [float(np.mean(s)) for s in sims]
    ses = [float(np.std(s, ddof=1) / np.sqrt(len(s))) for s in sims]
    for i in range(len(deltas) - 1):
        assert means[i] <= means[i + 1] + 2.0 * (ses[i] + ses[i + 1])
