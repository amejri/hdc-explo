import numpy as np

from hdc_project.encoder import m0, m1, m2, m3, utils


def test_m0_statistics():
    res = []
    D = 4096
    eps = 0.07
    n_pairs = 2_000
    g = np.random.default_rng(123)
    sims = np.empty(n_pairs, dtype=np.float64)
    for k in range(n_pairs):
        J = m0.M0_NewKey(int(g.integers(0, 2**31 - 1)), D)
        Jp = m0.M0_NewKey(int(g.integers(0, 2**31 - 1)), D)
        sims[k] = m1.M1_sim(J, Jp)
    assert abs(float(sims.mean())) < 1e-2
    tail = float((np.abs(sims) > eps).mean())
    hoeff = 2.0 * np.exp(-D * eps**2 / 2.0)
    assert tail <= hoeff * 1.5


def test_m1_identities():
    D = 4096
    n = 512
    X = utils.rand_pm1(n, D, seed=7)
    Y = utils.rand_pm1(n, D, seed=8)
    sims = m1.M1_sim_batch(X, Y)
    dh = (D / 2.0 * (1.0 - sims)).astype(int)
    mismatch = ((X * Y) == -1).sum(axis=1)
    assert np.all(dh == mismatch)
    for i in range(5):
        assert m1.M1_sim(X[i], X[i]) == 1.0
        assert m1.M1_dH(X[i], X[i]) == 0
        assert m1.M1_sim(X[i], -X[i]) == -1.0
        assert m1.M1_dH(X[i], -X[i]) == D


def test_m1_dtype_equivalence():
    D = 8192
    n = 256
    X = utils.rand_pm1(n, D, seed=11)
    Y = utils.rand_pm1(n, D, seed=12)
    s_int = m1.M1_sim_batch(X, Y)
    s_f32 = m1.M1_sim_batch(X.astype(np.float32), Y.astype(np.float32))
    s_f64 = m1.M1_sim_batch(X.astype(np.float64), Y.astype(np.float64))
    assert np.allclose(s_int, s_f32)
    assert np.allclose(s_int, s_f64)


def test_m2_permutation_isometry():
    D = 2048
    n = 128
    g = np.random.default_rng(21)
    pi = m2.M2_plan_perm(D, seed=21)
    X = utils.rand_pm1(n, D, seed=22)
    Y = utils.rand_pm1(n, D, seed=23)
    for k in (-257, -5, 0, 3, 511):
        Xp = m2.M2_perm_pow(X, pi, k)
        Yp = m2.M2_perm_pow(Y, pi, k)
        for i in range(5):
            assert m1.M1_sim(X[i], Y[i]) == m1.M1_sim(Xp[i], Yp[i])
        assert np.array_equal(m2.M2_perm_pow(Xp, pi, -k), X)


def test_m3_isometry_and_batch():
    D = 4096
    n = 256
    X = utils.rand_pm1(n, D, seed=31)
    Y = utils.rand_pm1(n, D, seed=32)
    J = utils.rand_pm1(1, D, seed=33)[0]
    Xb = m3.M3_bind_batch(X, J)
    Yb = m3.M3_bind_batch(Y, J)
    assert np.allclose(m1.M1_sim_batch(X, Y), m1.M1_sim_batch(Xb, Yb))
    assert np.array_equal(m3.M3_unbind(m3.M3_bind(X[0], J), J), X[0])
