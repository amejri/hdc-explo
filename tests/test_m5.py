import numpy as np

from hdc_project.encoder import m1, m2, m4, m5, utils


def _rand_word(g, V):
    return f"w{int(g.integers(0, V))}"


def test_m5_separability():
    D = 8192
    n = 3
    V = 1_000
    trials = 200
    g = np.random.default_rng(77)
    Lex = m4.M4_LexEN_new(78, D)
    pi = m2.M2_plan_perm(D, seed=79)
    pi_pows = m5.M5_precompute_pi_pows(pi, n)

    base = [[_rand_word(g, V) for _ in range(n)] for _ in range(trials)]
    fam1 = [m5.M5_ngram_cached(Lex, pi_pows, toks) for toks in base]

    fam1b = []
    for toks in base:
        toks2 = toks.copy()
        if g.random() < 0.3:
            j = int(g.integers(0, n))
            toks2[j] = _rand_word(g, V)
        fam1b.append(m5.M5_ngram_cached(Lex, pi_pows, toks2))

    fam2 = [m5.M5_ngram_cached(Lex, pi_pows, [_rand_word(g, V) for _ in range(n)]) for _ in range(trials)]

    A = np.stack(fam1).astype(np.int16)
    B = np.stack(fam1b).astype(np.int16)
    C = np.stack(fam2).astype(np.int16)

    intra = float(np.mean(np.sum(A * B, axis=1) / D))
    inter = float(np.mean(np.abs((A @ C.T) / D)))
    assert intra > 0.2
    assert inter < 0.08


def test_m5_drop_robustness():
    D = 8192
    V = 1_000
    drop = 0.1
    g = np.random.default_rng(99)
    Lex = m4.M4_LexEN_new(100, D)
    pi = m2.M2_plan_perm(D, seed=101)
    results = []
    for n in (2, 3, 5):
        pi_pows = m5.M5_precompute_pi_pows(pi, n)
        fam = [[_rand_word(g, V) for _ in range(n)] for _ in range(100)]
        E_ref = [m5.M5_ngram_cached(Lex, pi_pows, toks) for toks in fam]
        E_drop = []
        for toks in fam:
            toks2 = [tok if g.random() > drop else _rand_word(g, V) for tok in toks]
            E_drop.append(m5.M5_ngram_cached(Lex, pi_pows, toks2))
        A = np.stack(E_ref).astype(np.int16)
        B = np.stack(E_drop).astype(np.int16)
        sim = float(np.mean(np.sum(A * B, axis=1) / D))
        results.append((n, sim))
    assert results[1][1] <= results[0][1] + 1e-3
    assert results[2][1] <= results[1][1] + 1e-3


def test_m5_edges():
    D = 4096
    n = 3
    Lex = m4.M4_LexEN_new(5, D)
    pi = m2.M2_plan_perm(D, seed=6)
    pi_pows = m5.M5_precompute_pi_pows(pi, n)
    toks = ["a", "b", "c"][:n]
    E = m5.M5_ngram_cached(Lex, pi_pows, toks)
    assert m1.M1_sim(E, E) == 1.0
    assert m1.M1_dH(E, E) == 0
    Epi = m2.M2_perm_pow(E, pi, 1)
    assert m1.M1_sim(Epi, Epi) == 1.0
