from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np

from hdc_project.encoder import m4
from hdc_project.artifacts_types import SerializableArtifacts


def _ensure_path(prefix: str | Path) -> Path:
    path = Path(prefix)
    if path.suffix:
        return path.with_suffix("")
    return path


def save_artifacts(prefix: str | Path, artifacts: SerializableArtifacts, lexicon: m4.M4_LexEN) -> None:
    base = _ensure_path(prefix)
    base.parent.mkdir(parents=True, exist_ok=True)

    arrays_path = base.with_suffix(".npz")
    np.savez_compressed(
        arrays_path,
        prototypes=artifacts.prototypes.astype(np.int8, copy=False),
        Pi=artifacts.Pi.astype(np.int64, copy=False),
        G_MEM=artifacts.G_MEM.astype(np.int8, copy=False),
        G_DEC=artifacts.G_DEC.astype(np.int8, copy=False),
        LM_prior=artifacts.LM_prior.astype(np.int8, copy=False),
        pos_base_key=artifacts.pos_base_key.astype(np.int8, copy=False),
    )

    def _cast_keys(mapping: dict[int, list[str]]) -> dict[str, list[str]]:
        return {str(k): v for k, v in mapping.items()}

    def _cast_pos_keys(mapping: dict[tuple[int, int], list[str]]) -> dict[str, list[str]]:
        return {f"{k[0]}|{k[1]}": v for k, v in mapping.items()}

    meta = {
        "version": 1,
        "D": artifacts.D,
        "bucket2vocab": _cast_keys(artifacts.bucket2vocab),
        "bucket2vocab_pos": _cast_pos_keys(artifacts.bucket2vocab_pos),
        "global_vocab": artifacts.global_vocab,
        "fallback_vocab": artifacts.fallback_vocab,
        "mem_stats": artifacts.mem_stats,
        "freq_lm": artifacts.freq_lm,
        "bigrams_lm": artifacts.bigrams_lm,
        "pos_key_seed": artifacts.pos_key_seed,
    }
    meta_path = base.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    lex_path = base.with_suffix(".lex.npz")
    lexicon.save(str(lex_path))


def load_artifacts(prefix: str | Path) -> Tuple[SerializableArtifacts, m4.M4_LexEN]:
    base = _ensure_path(prefix)
    arrays_path = base.with_suffix(".npz")
    meta_path = base.with_suffix(".json")
    lex_path = base.with_suffix(".lex.npz")

    arrays = np.load(arrays_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("version") != 1:
        raise ValueError(f"Unsupported artifact version: {meta.get('version')}")

    def _restore_keys(mapping: dict[str, list[str]]) -> dict[int, list[str]]:
        return {int(k): list(v) for k, v in mapping.items()}

    def _restore_pos_keys(mapping: dict[str, list[str]]) -> dict[tuple[int, int], list[str]]:
        restored: dict[tuple[int, int], list[str]] = {}
        for key, value in mapping.items():
            bucket_str, pos_str = key.split("|", 1)
            restored[(int(bucket_str), int(pos_str))] = list(value)
        return restored

    serializable = SerializableArtifacts(
        D=int(meta["D"]),
        prototypes=arrays["prototypes"].astype(np.int8, copy=False),
        bucket2vocab=_restore_keys(meta["bucket2vocab"]),
        bucket2vocab_pos=_restore_pos_keys(meta["bucket2vocab_pos"]),
        global_vocab=list(meta["global_vocab"]),
        fallback_vocab=list(meta["fallback_vocab"]),
        Pi=arrays["Pi"].astype(np.int64, copy=False),
        G_MEM=arrays["G_MEM"].astype(np.int8, copy=False),
        G_DEC=arrays["G_DEC"].astype(np.int8, copy=False),
        LM_prior=arrays["LM_prior"].astype(np.int8, copy=False),
        pos_base_key=arrays["pos_base_key"].astype(np.int8, copy=False),
        pos_key_seed=int(meta["pos_key_seed"]),
        mem_stats={k: float(v) if isinstance(v, (int, float)) else v for k, v in meta["mem_stats"].items()},
        freq_lm={k: int(v) for k, v in meta["freq_lm"].items()},
        bigrams_lm={k: {kk: int(vv) for kk, vv in inner.items()} for k, inner in meta["bigrams_lm"].items()},
    )
    lexicon = m4.M4_LexEN.load(str(lex_path))
    return serializable, lexicon
