import numpy as np

from hdc_project.encoder import m3, m6, utils


def test_m6_binding_basic():
    D = 2048
    g = np.random.default_rng(12)
    X = utils.rand_pm1(1, D, seed=12)[0]
    Y = utils.rand_pm1(1, D, seed=13)[0]
    K = utils.rand_pm1(1, D, seed=14)[0]
    Xb = m3.M3_bind(X, K)
    Yb = m3.M3_bind(Y, K)
    assert int(np.dot(X.astype(np.int32), Y.astype(np.int32))) == int(np.dot(Xb.astype(np.int32), Yb.astype(np.int32)))
    S = m6.M6_SegAcc_init(D)
    S = m6.M6_SegAcc_push(S, X, K)
    S = m6.M6_SegAcc_push(S, Y, K)
    H = m6.M6_SegAcc_sign(S)
    assert set(np.unique(H)).issubset({-1, 1})


def test_m6_majority_fit():
    D = 4096
    p = 0.6
    pts = m6.M6_simulate_majority_error_fast(D, [8, 16, 32, 64, 96], p, trials=2000, seed=44)
    fit = m6.M6_fit_loglinear(pts, m_min_for_fit=32)
    slope_th = -m6.kl_half_vs_p(p)
    rel_gap = abs(fit["slope"] - slope_th) / abs(slope_th)
    assert fit["R2"] > 0.97
    assert rel_gap <= 0.35
