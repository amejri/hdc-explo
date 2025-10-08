from pathlib import Path

import numpy as np

from hdc_project.artifacts_io import load_artifacts, save_artifacts
from hdc_project.artifacts_types import SerializableArtifacts
from hdc_project.encoder import m4


def test_artifacts_roundtrip(tmp_path: Path) -> None:
    prefix = tmp_path / "opus_model"

    lex = m4.M4_LexEN_new(seed=7, D=8)
    _ = lex.get("bonjour")

    artifacts = SerializableArtifacts(
        D=8,
        prototypes=np.ones((2, 8), dtype=np.int8),
        bucket2vocab={0: ["bonjour"]},
        bucket2vocab_pos={(0, 0): ["bonjour"]},
        global_vocab=["bonjour"],
        fallback_vocab=["bonjour"],
        Pi=np.arange(8, dtype=np.int64),
        G_MEM=np.ones(8, dtype=np.int8),
        G_DEC=-np.ones(8, dtype=np.int8),
        LM_prior=np.ones(8, dtype=np.int8),
        pos_base_key=np.ones(8, dtype=np.int8),
        pos_key_seed=42,
        mem_stats={"B": 2},
        freq_lm={"bonjour": 2},
        bigrams_lm={"bonjour": {"bonjour": 1}},
    )

    save_artifacts(prefix, artifacts, lex)

    loaded_artifacts, loaded_lex = load_artifacts(prefix)

    assert loaded_artifacts.D == artifacts.D
    assert np.array_equal(loaded_artifacts.prototypes, artifacts.prototypes)
    assert loaded_artifacts.bucket2vocab == artifacts.bucket2vocab
    assert loaded_artifacts.bucket2vocab_pos == artifacts.bucket2vocab_pos
    assert loaded_artifacts.freq_lm == artifacts.freq_lm
    assert loaded_artifacts.bigrams_lm == artifacts.bigrams_lm
    assert np.array_equal(loaded_lex.get("bonjour"), lex.get("bonjour"))
