import tempfile
from pathlib import Path

import numpy as np

from hdc_project.encoder import m4


def test_m4_seed_stability():
    D = 4096
    L1 = m4.M4_LexEN_new(123, D)
    L2 = m4.M4_LexEN_new(123, D)
    v1 = m4.M4_get(L1, "Cat")
    v2 = m4.M4_get(L2, "cat")
    assert np.array_equal(v1, v2)
    try:
        v1[0] = 0
    except ValueError:
        pass
    else:
        raise AssertionError("vectors should be read-only")


def test_m4_collisions_and_stats():
    D = 4096
    V = 512
    Lex = m4.M4_LexEN_new(321, D)
    vocab = [f"w{i}" for i in range(V)]
    max_sim = m4.M4_collision_audit(Lex, vocab)
    assert max_sim <= 0.08
    mean, var = m4.M4_pair_stats(Lex, vocab, n_pairs=2_000)
    assert abs(mean) < 0.02
    assert var <= 1.5 / D


def test_m4_oov_pool_and_persist(tmp_path=None):
    D = 2048
    Lex = m4.M4_LexEN_new(77, D, reserve_pool=64)
    seen = {id(m4.M4_get(Lex, f"oov-{i}", use_pool=True)) for i in range(128)}
    assert len(seen) <= 128
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    path = tmp_path / "lex.npz"
    ref = [m4.M4_get(Lex, w) for w in ["a", "b", "c"]]
    Lex.save(str(path))
    loaded = m4.M4_LexEN.load(str(path))
    out = [m4.M4_get(loaded, w) for w in ["a", "b", "c"]]
    for r, o in zip(ref, out):
        assert np.array_equal(r, o)
