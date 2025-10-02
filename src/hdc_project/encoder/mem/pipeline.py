"""MEM training and inference pipeline built on top of the core primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .binding import to_mem_tranche
from .bank import MemBank, mem_scores, mem_argmax, topk_indices
from .lsh import SignLSH, code_to_bucket
from .query import build_query_mem
from ..utils import rand_pm1

__all__ = [
    "MemConfig",
    "MemComponents",
    "make_mem_pipeline",
    "train_one_pass_MEM",
    "infer_map_top1",
    "infer_map_topk",
    "make_aligned_pairs",
]


@dataclass(frozen=True)
class MemConfig:
    """Configuration for instantiating the MEM pipeline.

    Attributes:
        D: Hypervector dimensionality used by the bank.
        B: Number of addressable memory buckets/classes.
        k: Number of sign-LSH bits.
        seed_lsh: Seed controlling the random LSH projection.
        seed_gmem: Seed used to draw the MEM binding key ``G_mem``.
        thresh: Whether the :class:`MemBank` seals prototypes online (``True``)
            or lets the caller perform explicit sealing.
    """

    D: int
    B: int
    k: int
    seed_lsh: int
    seed_gmem: int
    thresh: bool = True


@dataclass
class MemComponents:
    """Concrete MEM components produced by :func:`make_mem_pipeline`.

    Attributes:
        mem: The associative memory bank.
        lsh: Sign-based LSH indexer.
        Gmem: MEM binding key (vector of ±1).
        meta: Metadata describing the instantiation (seeds, sizes, etc.).
    """

    mem: MemBank
    lsh: SignLSH
    Gmem: np.ndarray
    meta: Dict[str, int | bool]


def make_mem_pipeline(cfg: MemConfig) -> MemComponents:
    """Instantiate MEM components according to ``cfg``.

    Args:
        cfg: Configuration dataclass describing dimensionality, capacity and
            seeds for the various primitives.

    Returns:
        A :class:`MemComponents` object bundling the bank, the LSH indexer, the
        MEM binding key and an informational metadata dictionary.
    """

    mem = MemBank(B=cfg.B, D=cfg.D, thresh=cfg.thresh)
    lsh = SignLSH.with_k_bits(cfg.D, cfg.k, cfg.seed_lsh)
    rng = np.random.default_rng(cfg.seed_gmem)
    gmem = ((rng.integers(0, 2, size=(cfg.D,), dtype=np.int8) << 1) - 1).astype(np.int8, copy=False)
    meta: Dict[str, int | bool] = {
        "B": cfg.B,
        "D": cfg.D,
        "k": cfg.k,
        "seed_lsh": cfg.seed_lsh,
        "seed_gmem": cfg.seed_gmem,
        "thresh": cfg.thresh,
    }
    return MemComponents(mem=mem, lsh=lsh, Gmem=gmem, meta=meta)


def _bucket(lsh: SignLSH, z_mem: np.ndarray, B: int) -> int:
    """Route ``z_mem`` to a memory bucket using the configured LSH.

    The helper tries the unbiased bucketiser first, then falls back to a simple
    modulo or to the mixed ``code_to_bucket`` reducer.
    """

    try:
        return lsh.bucket_unbiased(z_mem, B)
    except AttributeError:
        return lsh.bucket_mod(z_mem, B)
    except Exception:
        return code_to_bucket(lsh.code(z_mem), B)


def train_one_pass_MEM(components: MemComponents,
                       pairs_en_fr: Iterable[Tuple[np.ndarray, np.ndarray]]) -> None:
    """Perform a single training pass over aligned encoder pairs.

    Args:
        components: MEM components returned by :func:`make_mem_pipeline`.
        pairs_en_fr: Iterable of ``(Z_en, Z_fr)`` pairs where both entries have
            shape ``(D,)`` and live in ``{-1, +1}``.

    Raises:
        ValueError: If the payloads do not have the expected dimensionality.
    """

    mem, lsh, gmem = components.mem, components.lsh, components.Gmem
    D = gmem.shape[0]
    for Z_en, Z_fr in pairs_en_fr:
        ze = np.asarray(Z_en, dtype=np.int8)
        zf = np.asarray(Z_fr, dtype=np.int8)
        if ze.shape != (D,) or zf.shape != (D,):
            raise ValueError("payloads must have shape (D,)")
        ze_mem = to_mem_tranche(ze, gmem)
        bucket = _bucket(lsh, ze_mem, mem.B)
        mem.add(bucket, zf)


def infer_map_top1(components: MemComponents,
                   R: np.ndarray,
                   use_thresh: bool = True) -> Tuple[int, float]:
    """Run MEM inference and return the top-1 bucket id and score.

    Args:
        components: MEM components returned by :func:`make_mem_pipeline`.
        R: Query vector in ``{-1, +1}`` (shape ``(D,)``).
        use_thresh: Selects whether the bank uses sealed prototypes (``True``)
            or the sign of the raw accumulators (``False``).

    Returns:
        ``(bucket_index, score)`` where ``score`` is the similarity between the
        query and the selected prototype.
    """

    R_mem = build_query_mem(np.asarray(R, dtype=np.int8), components.Gmem)
    scores = mem_scores(components.mem, R_mem, use_thresh=use_thresh)
    c_star = mem_argmax(scores)
    return c_star, float(scores[c_star])


def infer_map_topk(components: MemComponents,
                   R: np.ndarray,
                   k: int = 5,
                   use_thresh: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return the indices and scores of the top-``k`` buckets for query ``R``."""

    R_mem = build_query_mem(np.asarray(R, dtype=np.int8), components.Gmem)
    scores = mem_scores(components.mem, R_mem, use_thresh=use_thresh)
    idx = topk_indices(scores, k)
    return idx, scores[idx]


def make_aligned_pairs(B: int,
                       D: int,
                       m_per_class: int,
                       noise_fr: float,
                       noise_en: float,
                       seed_proto: int,
                       seed_stream: int) -> Dict[str, List | np.ndarray]:
    """Generate synthetic aligned EN/FR pairs for MEM experiments.

    The function creates ``B`` prototype hypervectors, applies flips to model
    noise, and returns the resulting training pairs alongside the clean
    prototypes for both languages.

    Args:
        B: Number of classes/buckets.
        D: Hypervector dimensionality.
        m_per_class: Number of samples per class.
        noise_fr: Flip probability applied to the FR payload.
        noise_en: Flip probability applied to the EN payload.
        seed_proto: Seed used to draw base prototypes and the permutation.
        seed_stream: Seed that drives the noise stream.

    Returns:
        Dictionary containing the training pairs and the clean references for
        evaluation purposes.

    Raises:
        ValueError: If ``B``, ``D`` or ``m_per_class`` is non positive.
    """

    if B <= 0 or D <= 0 or m_per_class <= 0:
        raise ValueError("B, D and m_per_class must be positive")
    g_proto = np.random.default_rng(seed_proto)
    pi = g_proto.permutation(D).astype(np.int64)
    prototypes = rand_pm1(B, D, seed_proto + 1)
    en_prototypes = prototypes[:, pi]
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    g_stream = np.random.default_rng(seed_stream)
    for c in range(B):
        base_fr = prototypes[c]
        base_en = en_prototypes[c]
        for _ in range(m_per_class):
            z_fr = base_fr.copy()
            z_en = base_en.copy()
            if noise_fr > 0:
                mask = g_stream.random(D) < noise_fr
                z_fr[mask] = -z_fr[mask]
            if noise_en > 0:
                mask = g_stream.random(D) < noise_en
                z_en[mask] = -z_en[mask]
            pairs.append((z_en.astype(np.int8, copy=False), z_fr.astype(np.int8, copy=False)))
    return {
        "pairs": pairs,
        "P": prototypes,
        "pi": pi,
        "R_clean": prototypes.copy(),
        "Q_clean": en_prototypes.copy(),
    }
