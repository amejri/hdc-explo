import numpy as np

from hdc_project.encoder import m4, pipeline


def _random_sentences(g, vocab_size, count, length):
    return [
        " ".join(f"w{int(g.integers(0, vocab_size))}" for _ in range(length))
        for _ in range(count)
    ]


def test_pipeline_encode_and_metrics():
    D = 1024
    n = 3
    g = np.random.default_rng(123)
    Lex = m4.M4_LexEN_new(321, D)
    pi = g.permutation(D).astype(np.int64)

    sentences = _random_sentences(g, vocab_size=500, count=3, length=8)
    encoded = pipeline.encode_corpus_ENC(sentences, Lex, pi, D, n, seg_seed0=999)

    assert len(encoded) == len(sentences)
    for segment in encoded:
        assert segment["H"].dtype == np.int8
        assert segment["S"].dtype == np.int16
        assert len(segment["E_seq"]) == segment["len"] == len(segment["X_seq"])
        # ensure all hypervectors have the expected dimensionality
        assert all(vec.shape == (D,) for vec in segment["E_seq"])
        assert all(vec.shape == (D,) for vec in segment["X_seq"])

    E_list = [entry["E_seq"] for entry in encoded]
    H_list = [entry["H"] for entry in encoded]

    s_intra, s_inter = pipeline.intra_inter_ngram_sims(E_list, D)
    assert -1.0 <= s_intra <= 1.0
    assert 0.0 <= s_inter <= 1.0

    majority_curves = pipeline.majority_error_curve(E_list, pi, D, eta_list=(0.0, 0.05))
    assert set(majority_curves) == {0.0, 0.05}
    for items in majority_curves.values():
        for _, err in items:
            assert 0.0 <= err <= 1.0

    repeated_curves = pipeline.majority_curve_repeated_vector(
        E_list, pi, D, eta_list=(0.0, 0.05), trials_per_m=200, seed=456
    )
    assert set(repeated_curves) == {0.0, 0.05}
    for items in repeated_curves.values():
        for _, err in items:
            assert 0.0 <= err <= 1.0

    inter_seg = pipeline.inter_segment_similarity(H_list)
    assert 0.0 <= inter_seg <= 1.0


def test_pipeline_m8_modes():
    D = 512
    n = 2
    Lex = m4.M4_LexEN_new(777, D)
    pi = np.random.default_rng(888).permutation(D).astype(np.int64)
    tokens = "hyper dimensional encoding rocks".split()

    unbiased = pipeline.M8_ENC(tokens, pi, n, Lex, D, majority_mode="unbiased", return_bound=True)
    strict = pipeline.M8_ENC(tokens, pi, n, Lex, D, majority_mode="strict")

    E_unbiased, X_unbiased, Xb_seq, S_unbiased, H_unbiased = unbiased
    E_strict, X_strict, S_strict, H_strict = strict

    assert len(E_unbiased) == len(tokens)
    assert len(X_unbiased) == len(tokens)
    assert len(Xb_seq) == len(tokens)
    assert S_unbiased.shape == (D,)
    assert S_strict.shape == (D,)
    assert H_unbiased.shape == (D,)
    assert H_strict.shape == (D,)
    # strict majority must be deterministic, unbiased can differ but both stay in {-1,+1}
    assert set(np.unique(H_strict)).issubset({-1, 1})
    assert set(np.unique(H_unbiased)).issubset({-1, 1})
