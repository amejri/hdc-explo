from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RuntimeData:
    metas: List[Dict[str, object]]
    split_metas: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)


@dataclass
class SerializableArtifacts:
    D: int
    prototypes: np.ndarray
    bucket2vocab: Dict[int, List[str]]
    bucket2vocab_pos: Dict[Tuple[int, int], List[str]]
    global_vocab: List[str]
    fallback_vocab: List[str]
    Pi: np.ndarray
    G_MEM: np.ndarray
    G_DEC: np.ndarray
    LM_prior: np.ndarray
    pos_base_key: np.ndarray
    pos_key_seed: int
    mem_stats: Dict[str, object]
    freq_lm: Dict[str, int]
    bigrams_lm: Dict[str, Dict[str, int]]


@dataclass
class PipelineArtifacts:
    runtime: RuntimeData
    model: SerializableArtifacts
    Lex_fr: object | None = None
    freq_train: Counter[str] = field(default_factory=Counter)
    bigrams_train: Dict[str, Counter[str]] = field(default_factory=dict)
    pos_keys: Dict[int, np.ndarray] = field(default_factory=dict)
