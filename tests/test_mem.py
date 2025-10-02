import numpy as np

from hdc_project.encoder.mem import binding, bank, lsh, pipeline, query


def test_sign_lsh_code_and_bucket():
    D = 256
    k = 8
    rng = np.random.default_rng(0)
    hasher = lsh.SignLSH.with_k_bits(D, k, seed=1)
    vec = rng.integers(0, 2, size=D, dtype=np.int8)
    vec = (vec << 1) - 1
    code = hasher.code(vec)
    assert 0 <= code < 2**k
    bucket = hasher.bucket(vec, B=17)
    assert 0 <= bucket < 17
    multi = lsh.MultiSignLSH.build(D, k, T=3, seed=2)
    bucket_multi = multi.bucket(vec, B=19)
    assert 0 <= bucket_multi < 19


def test_binding_involution_batch():
    D = 512
    X = np.ones((4, D), dtype=np.int8)
    G = np.ones((D,), dtype=np.int8)
    X[:, ::2] = -1
    G[1::2] = -1
    Xg = binding.bind_tranche_batch(X, G)
    assert Xg.shape == (4, D)
    X_back = binding.bind_tranche_batch(Xg, G)
    np.testing.assert_array_equal(X_back, X)
    x_single = X[0]
    x_mem = binding.to_mem_tranche(x_single, G)
    np.testing.assert_array_equal(binding.from_mem_tranche(x_mem, G), x_single)


def test_mem_bank_scoring_and_topk():
    D = 256
    B = 32
    mem = bank.MemBank(B=B, D=D, thresh=True)
    rng = np.random.default_rng(3)
    payloads = (rng.integers(0, 2, size=(B, D), dtype=np.int8) << 1) - 1
    for c in range(B):
        mem.add(c, payloads[c])
    query_vec = payloads[5]
    scores = bank.mem_scores(mem, query_vec)
    assert scores.shape == (B,)
    idx = bank.mem_argmax(scores)
    assert isinstance(idx, int)
    top_idx = bank.topk_indices(scores, 3)
    assert len(top_idx) == 3
    margin = bank.margin_top1(scores)
    assert margin <= 2.0
    idx_stream, scores_stream = bank.mem_topk_stream(mem, query_vec, k=3)
    np.testing.assert_array_equal(np.sort(top_idx), np.sort(idx_stream))
    assert scores_stream.shape == (3,)


def test_mem_pipeline_end_to_end():
    cfg = pipeline.MemConfig(D=512, B=16, k=6, seed_lsh=10, seed_gmem=11)
    comp = pipeline.make_mem_pipeline(cfg)
    data = pipeline.make_aligned_pairs(
        B=cfg.B,
        D=cfg.D,
        m_per_class=4,
        noise_fr=0.05,
        noise_en=0.05,
        seed_proto=20,
        seed_stream=21,
    )
    pipeline.train_one_pass_MEM(comp, data["pairs"])
    for c in range(cfg.B):
        R = data["Q_clean"][c]
        pred, score = pipeline.infer_map_top1(comp, R)
        assert 0 <= pred < cfg.B
        assert -1.0 <= score <= 1.0
        idx, scores = pipeline.infer_map_topk(comp, R, k=3)
        assert len(idx) == len(scores)
        assert np.all((idx >= 0) & (idx < cfg.B))


def test_query_builder_with_targets():
    D = 128
    pi = np.random.default_rng(0).permutation(D)
    H_window = [(np.random.default_rng(i).integers(0, 2, size=D, dtype=np.int8) << 1) - 1 for i in range(3)]
    proto = (np.random.default_rng(10).integers(0, 2, size=D, dtype=np.int8) << 1) - 1
    R = query.build_query_from_context(H_window, pi, w_left=1, w_right=1,
                                       weights_ctx=[1, 2, 1],
                                       targets_hist=[(proto, 2, 1)])
    assert R.shape == (D,)
    G = (np.random.default_rng(1).integers(0, 2, size=D, dtype=np.int8) << 1) - 1
    R_mem = query.build_query_mem(R, G)
    assert R_mem.shape == (D,)
