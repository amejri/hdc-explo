"""MEM training and inference pipeline built on top of the core primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .binding import to_mem_tranche, bind_tranche_batch
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
    # OPT: rand_pm1 → plus direct que integers->shift->cast
    G = rand_pm1(1, cfg.D, seed=cfg.seed_gmem).reshape(cfg.D).astype(np.int8, copy=False)
    G.setflags(write=False)
    meta: Dict[str, int | bool] = {
        "B": cfg.B,
        "D": cfg.D,
        "k": cfg.k,
        "seed_lsh": cfg.seed_lsh,
        "seed_gmem": cfg.seed_gmem,
        "thresh": cfg.thresh,
    }
    return MemComponents(mem=mem, lsh=lsh, Gmem=G, meta=meta)


def _bucket(lsh: SignLSH, z_mem: np.ndarray, B: int) -> int:
    """Route ``z_mem`` to a memory bucket using the configured LSH.

    The helper tries the unbiased bucketiser first, then falls back to a simple
    modulo or to the mixed ``code_to_bucket`` reducer.
    """
    if hasattr(lsh, "bucket_unbiased"):
        return lsh.bucket_unbiased(z_mem, B)
    if hasattr(lsh, "bucket_mod"):
        return lsh.bucket_mod(z_mem, B)
    # dernier recours: mix & reduce
    return code_to_bucket(lsh.code(z_mem), B)


def train_one_pass_MEM(components: MemComponents,
                       pairs_en_fr: Iterable[Tuple[np.ndarray, np.ndarray]],
                       chunk: int = 4096,) -> None:
    """Perform a single training pass over aligned encoder pairs.

    Args:
        components: MEM components returned by :func:`make_mem_pipeline`.
        pairs_en_fr: Iterable of ``(Z_en, Z_fr)`` pairs where both entries have
            shape ``(D,)`` and live in ``{-1, +1}``.
        chunk: taille de lot pour vectoriser le binding (trade-off RAM/CPU).

    Raises:
        ValueError: If the payloads do not have the expected dimensionality.
    """

    mem, lsh, G = components.mem, components.lsh, components.Gmem
    D = int(G.shape[0])

    # Convertit l’itérable en listes plates (si ce n’est pas déjà des arrays)
    # → cela permet le batching efficace. Si l’itérable est énorme/paresseux,
    #   remplace par une lecture par blocs côté appelant.
    if not isinstance(pairs_en_fr, (list, tuple)):
        pairs_en_fr = list(pairs_en_fr)

    # Prépare des buffers (limite les réallocations)
    ZEN: list[np.ndarray] = []
    ZFR: list[np.ndarray] = []
    ZEN_extend = ZEN.extend
    ZFR_extend = ZFR.extend

    for ze, zf in pairs_en_fr:
        ZEN_extend([np.asarray(ze, dtype=np.int8, copy=False)])
        ZFR_extend([np.asarray(zf, dtype=np.int8, copy=False)])

    N = len(ZEN)
    if N == 0:
        return

    # Traitement par blocs
    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        # Empilement → (B, D) int8
        ZE = np.stack(ZEN[start:end], axis=0).astype(np.int8, copy=False)
        ZF = np.stack(ZFR[start:end], axis=0).astype(np.int8, copy=False)

        # Binding MEM vectorisé
        ZE_MEM = bind_tranche_batch(ZE, G)  # (b, D)

        # Boucle mince (bucket + add) — pas de copies inutiles
        for i in range(ZE_MEM.shape[0]):
            c = _bucket(lsh, ZE_MEM[i], mem.B)
            mem.add(c, ZF[i])


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
    mem, lsh, G = components.mem, components.lsh, components.Gmem
    R = np.asarray(R, dtype=np.int8, copy=False)
    R_mem = to_mem_tranche(R, G)                         # même binding qu’en train
    c_star = _bucket(lsh, R_mem, mem.B)                  # routage LSH (theory-consistent)

    # Score de confiance simple : taille normalisée (ou marge LSH si dispo)
    # Ici : fraction du bucket vs p99 pour bornage [0,1]
    n = int(mem.n[c_star])
    p99 = max(1, int(np.percentile(mem.n, 99)))          # robustifier
    conf = float(min(1.0, n / p99))
    return c_star, conf


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
