"""High-level encoder pipeline (M8) and evaluation helpers."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .m2 import M2_perm_pow
from .m3 import M3_bind
from .m4 import M4_LexEN, M4_LexEN_new
from .m5 import (
    M5_ngram,
    M5_ngram_cached_strict,
    M5_ngram_cached_unbiased,
)
from .m6 import (
    M6_SegAcc_init,
    M6_SegAcc_push,
    M6_SegAcc_sign,
)
from .m7 import M7_SegMgr_new, M7_curKey
from .utils import unbiased_sign_int8, strict_sign_int8

__all__ = [
    "M8_ENC",
    "enc_sentence_ENC",
    "encode_corpus_ENC",
    "intra_inter_ngram_sims",
    "inter_segment_similarity",
    "majority_error_curve",
    "majority_curve_repeated_vector",
    "tokenize_en",
    "build_vocab_EN",
    "sentence_to_tokens_EN",
    "opus_load_subset",
    "validate_on_opus",
    "validate_on_opus_enc3",
]


def _ensure_pi_pows(pi: np.ndarray, n: int) -> list[np.ndarray]:
    """Return cached permutation powers ``[Pi^0, Pi^1, ..., Pi^n]``.

    The helper precomputes forward permutation indices that are reused by the
    encoder inner loop. The first entry is ``Pi^0`` (identity) and each
    subsequent entry composes the previous one with ``pi``.

    Args:
        pi: Permutation array of shape ``(D,)`` containing a bijection over
            ``0..D-1``.
        n: Maximum power that must be available in the cache. Values greater
            than the longest window length guarantee that the cache can be used
            directly inside the sliding encoder.

    Returns:
        A list of ``n + 1`` ``numpy.ndarray`` objects describing the reindexing
        to apply for ``Pi^k``.
    """

    pi_pows = [np.arange(pi.shape[0], dtype=np.int64)]
    for _ in range(n):
        pi_pows.append(pi[pi_pows[-1]])
    return pi_pows


def M8_ENC(
    tokens: Iterable[str],
    pi: np.ndarray,
    n: int,
    LexEN: M4_LexEN,
    D: int,
    segmgr: Any | None = None,
    acc_S: np.ndarray | None = None,
    return_bound: bool = False,
    pi_pows: list[np.ndarray] | None = None,
    majority_mode: str = "unbiased",
    m5_variant: str = "auto",
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray] | Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray
]:
    """Encode a token sequence with the ENC brick (modules M5-M7).

    The pipeline slides an ``n``-gram window over ``tokens``. Each window is
    converted into an HDC vector ``E_t`` (via module M5), reindexed according to
    the relative offset (module M2), bound with the current segment key (module
    M3) and accumulated (module M6). The final segment signature ``H`` is
    obtained by applying either the unbiased or the strict majority policy.

    Args:
        tokens: Sequence of string tokens describing the segment to encode.
        pi: Permutation array returned by :func:`~hdc_project.encoder.m2.M2_plan_perm`.
        n: Order of the ``n``-gram (context window length).
        LexEN: Lexicon that maps tokens to hypervectors (module M4).
        D: Dimensionality of the hypervectors.
        segmgr: Optional segment manager (module M7). When ``None`` a fresh one
            is instantiated using a seed drawn from ``LexEN``'s RNG.
        acc_S: Optional accumulator to reuse across calls. When ``None`` a fresh
            zero accumulator is created.
        return_bound: If ``True`` the returned tuple includes the bound
            intermediates representing the bound vectors ``X_t`` hadamard-multiplied with ``K_s`` for inspection or debugging.
        pi_pows: Optional cache of permutation powers produced by
            :func:`_ensure_pi_pows`. The cache is expanded automatically when it
            is not provided.
        majority_mode: Either ``"unbiased"`` (default) or ``"strict"`` to select
            the majority policy applied to the accumulator when producing the
            final signature.
        m5_variant: Selects which implementation of the M5 encoder is used.
            ``"auto"`` attempts the unbiased variant first, falling back to the
            strict or generic implementations when unavailable.

    Returns:
        A tuple containing the list of raw ``E_t`` vectors, the permutation
        aligned ``X_t`` vectors, optionally the bound vectors, the final
        accumulator and the segment signature ``H``.

    Raises:
        ValueError: If ``majority_mode`` is not ``"unbiased"`` or ``"strict"``.
    """

    tokens_list = list(tokens)
    if segmgr is None:
        seg_seed = int(LexEN.rng.integers(1, 2**31 - 1))
        segmgr = M7_SegMgr_new(seg_seed, D)
    K_s = M7_curKey(segmgr)

    S = acc_S if acc_S is not None else M6_SegAcc_init(D)

    if pi_pows is None:
        pi_pows = _ensure_pi_pows(pi, max(n, len(tokens_list)) + 2)

    rng = getattr(LexEN, "rng", np.random.default_rng())

    def pick_ngram(window: Sequence[str]) -> np.ndarray:
        if m5_variant in {"unbiased", "auto"}:
            try:
                return M5_ngram_cached_unbiased(LexEN, pi_pows, window, rng=rng)
            except NameError:
                if m5_variant == "unbiased":
                    raise
        if m5_variant in {"strict", "auto"}:
            try:
                return M5_ngram_cached_strict(LexEN, pi_pows, window)
            except NameError:
                if m5_variant == "strict":
                    raise
        return M5_ngram(LexEN, pi, window)

    E_seq: List[np.ndarray] = []
    X_seq: List[np.ndarray] = []
    Xb_seq: Optional[List[np.ndarray]] = [] if return_bound else None

    for t in range(len(tokens_list)):
        left = max(0, t - n + 1)
        window = tokens_list[left : t + 1]
        E_t = pick_ngram(window)
        Delta = t - left
        idx = pi_pows[Delta]
        X_t = E_t[idx].astype(np.int8, copy=False)
        S = M6_SegAcc_push(S, X_t, K_s)
        if return_bound and Xb_seq is not None:
            Xb_seq.append(M3_bind(X_t, K_s))
        E_seq.append(E_t)
        X_seq.append(X_t)

    if majority_mode == "unbiased":
        H = unbiased_sign_int8(S, rng_obj=rng)
    elif majority_mode == "strict":
        H = strict_sign_int8(S)
    else:
        raise ValueError("majority_mode must be 'unbiased' or 'strict'")

    if return_bound:
        return E_seq, X_seq, Xb_seq or [], S, H
    return E_seq, X_seq, S, H


def enc_sentence_ENC(
    sentence_tokens: List[str],
    n: int,
    pi: np.ndarray,
    LexEN: M4_LexEN,
    D: int,
    seg_seed: int,
) -> Dict[str, Any]:
    """Encode a single sentence into the ENC representation.

    This convenience wrapper instantiates a deterministic segment manager using
    ``seg_seed`` and delegates the encoding to :func:`M8_ENC`.

    Args:
        sentence_tokens: Token list describing the sentence.
        n: ``n``-gram order for M5.
        pi: Permutation used by the encoder.
        LexEN: Lexicon providing token hypervectors.
        D: Dimensionality of the hypervectors.
        seg_seed: Seed used to initialise the segment manager.

    Returns:
        A dictionary matching the structure produced by :func:`M8_ENC` with
        additional metadata (sentence length and seed).
    """

    tokens = list(sentence_tokens)
    segmgr = M7_SegMgr_new(seg_seed, D)
    E_seq, X_seq, S, H = M8_ENC(tokens, pi, n, LexEN, D, segmgr=segmgr)
    return {
        "E_seq": E_seq,
        "X_seq": X_seq,
        "S": S,
        "H": H,
        "len": len(tokens),
        "seg_seed": seg_seed,
    }


def encode_corpus_ENC(
    sentences: List[str],
    LexEN: M4_LexEN,
    pi: np.ndarray,
    D: int,
    n: int,
    seg_seed0: int = 10_001,
    log_every: int = 1_000,
) -> List[Dict[str, Any]]:
    """Encode a list of sentences with ENC, returning full traces.

    Args:
        sentences: Raw text sentences that are tokenised using
            :func:`sentence_to_tokens_EN`.
        LexEN: Lexicon for token hypervectors.
        pi: Permutation used for positional binding.
        D: Dimensionality of the hypervectors.
        n: ``n``-gram order for ENC.
        seg_seed0: Master seed used to draw per sentence segment seeds.
        log_every: Progress logging cadence. The function itself does not log
            but callers may hook on the iteration index.

    Returns:
        A list of dictionaries mirroring the structure returned by
        :func:`enc_sentence_ENC` for each sentence.
    """

    out: List[Dict[str, Any]] = []
    rng = np.random.default_rng(seg_seed0)
    for idx, sent in enumerate(sentences, 1):
        toks = sentence_to_tokens_EN(sent, vocab=set())
        seg_seed = int(rng.integers(1, 2**31 - 1))
        out.append(enc_sentence_ENC(toks, n, pi, LexEN, D, seg_seed))
        if log_every and idx % log_every == 0:
            pass  # caller configures logging if needed
    return out


def intra_inter_ngram_sims(E_seq_list: List[List[np.ndarray]], D: int) -> Tuple[float, float]:
    """Compute intra and inter sentence n-gram similarities.

    Args:
        E_seq_list: Collection of ``E_t`` sequences (one per sentence).
        D: Dimensionality used to normalise the dot products.

    Returns:
        ``(s_intra, s_inter)`` where ``s_intra`` is the mean similarity between
        consecutive n-grams inside the same sentence and ``s_inter`` is the mean
        absolute similarity between the first n-gram of neighbouring sentences.
    """

    intra_vals: List[float] = []
    for seq in E_seq_list:
        for a in range(len(seq) - 1):
            intra_vals.append(float(np.dot(seq[a], seq[a + 1]) / D))
    s_intra = float(np.mean(intra_vals)) if intra_vals else 0.0

    inter_vals: List[float] = []
    for i in range(len(E_seq_list) - 1):
        if E_seq_list[i] and E_seq_list[i + 1]:
            inter_vals.append(abs(float(np.dot(E_seq_list[i][0], E_seq_list[i + 1][0]) / D)))
    s_inter = float(np.mean(inter_vals)) if inter_vals else 0.0
    return s_intra, s_inter


def inter_segment_similarity(H_list: List[np.ndarray]) -> float:
    """Compute the mean absolute similarity between consecutive ENC segments.

    Args:
        H_list: Sequence of segment signatures ``H`` produced by the encoder.

    Returns:
        The arithmetic mean of ``|<H_i, H_{i+1}>| / D`` for consecutive pairs.
        Returns ``0.0`` when fewer than two signatures are provided.
    """

    vals = [abs(float(np.dot(H_list[i], H_list[i + 1]) / H_list[i].size)) for i in range(len(H_list) - 1)]
    return float(np.mean(vals)) if vals else 0.0


def majority_error_curve(
    E_seq_list: List[List[np.ndarray]],
    pi: np.ndarray,
    D: int,
    eta_list: Tuple[float, ...] = (0.0, 0.05, 0.10),
    seed_noise: int = 303,
) -> Dict[float, List[Tuple[int, float]]]:
    """Empirically estimate majority vote error rates on real sequences.

    Each sequence ``E_seq`` is replayed twice (clean and noisy) using the same
    segment key. The noisy run flips individual coordinates with probability
    ``eta`` before accumulation. The proportion of coordinates that disagree
    between the clean and noisy signatures is returned for each sequence length.

    Args:
        E_seq_list: List of n-gram sequences obtained from the encoder.
        pi: Permutation used for relative reindexing.
        D: Hypervector dimensionality.
        eta_list: Noise levels to simulate.
        seed_noise: RNG seed used to generate the key and the noise masks.

    Returns:
        A dictionary ``eta -> [(m, error_rate), ...]`` aggregated by sequence
        length ``m``.
    """

    rng = np.random.default_rng(seed_noise)
    results: Dict[float, List[Tuple[int, float]]] = {eta: [] for eta in eta_list}

    for E_seq in E_seq_list:
        m = len(E_seq)
        if m == 0:
            continue
        key_seed = int(rng.integers(1, 2**31 - 1))
        segmgr = M7_SegMgr_new(key_seed, D)
        K = M7_curKey(segmgr)

        S_clean = M6_SegAcc_init(D)
        for t, E_t in enumerate(E_seq):
            X_t = M2_perm_pow(E_t, pi, t).astype(np.int8, copy=False)
            S_clean = M6_SegAcc_push(S_clean, X_t, K)
        H_clean = M6_SegAcc_sign(S_clean)

        for eta in eta_list:
            S_noisy = M6_SegAcc_init(D)
            for t, E_t in enumerate(E_seq):
                X_t = M2_perm_pow(E_t, pi, t).astype(np.int8, copy=False)
                if eta > 0.0:
                    mask = rng.random(X_t.shape[0]) < eta
                    X_t = X_t.copy()
                    X_t[mask] = -X_t[mask]
                S_noisy = M6_SegAcc_push(S_noisy, X_t, K)
            H_noisy = M6_SegAcc_sign(S_noisy)
            diff = (H_clean.astype(np.int8) * H_noisy.astype(np.int8) == -1).mean()
            results[eta].append((m, float(diff)))

    aggregated: Dict[float, List[Tuple[int, float]]] = {}
    for eta, pairs in results.items():
        by_m: Dict[int, List[float]] = {}
        for m, val in pairs:
            by_m.setdefault(m, []).append(val)
        aggregated[eta] = sorted((m, float(np.mean(vals))) for m, vals in by_m.items())
    return aggregated


def majority_curve_repeated_vector(
    E_seq_list: List[List[np.ndarray]],
    pi: np.ndarray,
    D: int,
    eta_list: Tuple[float, ...] = (0.0, 0.05, 0.10),
    trials_per_m: int = 4_000,
    seed: int = 4242,
) -> Dict[float, List[Tuple[int, float]]]:
    """Simulate majority vote error curves on repeated vectors.

    Instead of replaying the full pipeline this helper draws Bernoulli samples
    for the majority vote and returns error estimates for the lengths observed
    in ``E_seq_list``.

    Args:
        E_seq_list: Source sequences used to collect the empirical length grid.
        pi: Unused but kept for signature compatibility with callers.
        D: Unused dimensionality (kept for signature uniformity).
        eta_list: Noise levels to evaluate.
        trials_per_m: Number of Monte Carlo trials for each length.
        seed: RNG seed.

    Returns:
        Dictionary ``eta -> [(m, error_rate), ...]`` sorted by ``m``.
    """

    m_grid = sorted({len(seq) for seq in E_seq_list if len(seq) > 0})
    if not m_grid:
        return {eta: [] for eta in eta_list}
    rng = np.random.default_rng(seed)
    out: Dict[float, List[Tuple[int, float]]] = {eta: [] for eta in eta_list}
    for eta in eta_list:
        p = 1.0 - float(eta)
        for m in m_grid:
            B = rng.random(size=(trials_per_m, m)) < p
            Y = np.where(B, 1, -1).astype(np.int16)
            S = Y.sum(axis=1)
            err = float((S <= 0).mean())
            out[eta].append((m, err))
    for eta in eta_list:
        out[eta].sort(key=lambda t: t[0])
    return out


# --- basic NLP helpers -----------------------------------------------------

def tokenize_en(s: str) -> List[str]:
    """Tokenise English text by lowercasing and dropping empty chunks.

    Args:
        s: Raw string to tokenise.

    Returns:
        A list of non empty lowercase tokens separated on whitespace.
    """

    return [w.strip().lower() for w in s.split() if w.strip()]


def build_vocab_EN(ens: Iterable[str], V: int = 5_000) -> set[str]:
    """Extract the ``V`` most frequent tokens from a corpus.

    Args:
        ens: Iterable of sentences.
        V: Maximum vocabulary size to keep.

    Returns:
        A set containing the most frequent tokens according to a simple
        frequency count.
    """

    counter = Counter()
    for sent in ens:
        counter.update(tokenize_en(sent))
    return {token for token, _ in counter.most_common(V)}


def sentence_to_tokens_EN(sentence: str, vocab: set[str]) -> List[str]:
    """Tokenise ``sentence`` and optionally filter using ``vocab``.

    The current implementation mirrors the historical behaviour and does not
    drop out-of-vocabulary tokens, but the ``vocab`` parameter is kept for API
    stability.

    Args:
        sentence: Raw sentence to tokenise.
        vocab: Token set (currently unused).

    Returns:
        The tokenised sentence.
    """

    return tokenize_en(sentence)


# --- validation helper -----------------------------------------------------

def opus_load_subset(
    name: str = "opus_books",
    config: str = "en-fr",
    split: str = "train",
    N: int = 10_000,
    seed: int = 123,
) -> Tuple[List[str], List[str]]:
    """Download a subset of the OPUS parallel corpus using ``datasets``.

    Args:
        name: Dataset name on Hugging Face.
        config: Dataset configuration (language pair).
        split: Split to download.
        N: Maximum number of sentence pairs to retain.
        seed: Shuffle seed applied before slicing ``N`` samples.

    Returns:
        Two lists containing the English and French sentences.
    """

    from datasets import load_dataset

    ds = load_dataset(name, config, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(N, len(ds))))
    ens = [ex["translation"]["en"] for ex in ds]
    frs = [ex["translation"]["fr"] for ex in ds]
    return ens, frs


def validate_on_opus(
    D: int = 16_384,
    n: int = 3,
    N: int = 1_000,
    seed_lex: int = 10_123,
    seed_pi: int = 10_456,
) -> Dict[str, Any]:
    """Run the full ENC validation pipeline on an OPUS subset.

    Args:
        D: Hypervector dimensionality.
        n: ``n``-gram order used by the encoder.
        N: Number of sentence pairs to sample.
        seed_lex: Seed for the English lexicon.
        seed_pi: Seed for the permutation plan.

    Returns:
        A dictionary containing summary statistics (similarities and majority
        curves) that can be logged or persisted by the caller.
    """

    ens, _ = opus_load_subset("opus_books", "en-fr", "train", N=N, seed=2024)
    Lex = M4_LexEN_new(seed_lex, D, reserve_pool=4_096)
    pi = np.random.default_rng(seed_pi).permutation(D).astype(np.int64)
    encoded = encode_corpus_ENC(ens, Lex, pi, D, n, seg_seed0=9_001)
    E_list = [e["E_seq"] for e in encoded]
    H_list = [e["H"] for e in encoded]
    s_intra, s_inter = intra_inter_ngram_sims(E_list, D)
    s_inter_seg = inter_segment_similarity(H_list)
    maj_curves = majority_error_curve(E_list, pi, D, eta_list=(0.0, 0.05, 0.1))
    return {
        "D": D,
        "n": n,
        "N_pairs": N,
        "seed_lex": seed_lex,
        "seed_pi": seed_pi,
        "intra_ngram_mean_sim": s_intra,
        "inter_ngram_abs_mean_sim": s_inter,
        "inter_segment_abs_mean_sim": s_inter_seg,
        "majority_curves": maj_curves,
    }


def validate_on_opus_enc3(
    D: int = 16_384,
    n: int = 3,
    N: int = 1_000,
    seed_lex: int = 10_123,
    seed_pi: int = 10_456,
) -> Dict[str, Any]:
    """Variant of :func:`validate_on_opus` that replaces real replays by simulations.

    The function returns the same summary structure but uses
    :func:`majority_curve_repeated_vector` to approximate majority vote errors
    instead of replaying the noisy pipeline.

    Args:
        D: Hypervector dimensionality.
        n: ``n``-gram order used by the encoder.
        N: Number of sentence pairs to sample.
        seed_lex: Seed for the English lexicon.
        seed_pi: Seed for the permutation plan.

    Returns:
        A summary dictionary matching :func:`validate_on_opus` but with
        simulated majority curves.
    """

    ens, _ = opus_load_subset("opus_books", "en-fr", "train", N=N, seed=2024)
    Lex = M4_LexEN_new(seed_lex, D, reserve_pool=4_096)
    pi = np.random.default_rng(seed_pi).permutation(D).astype(np.int64)
    encoded = encode_corpus_ENC(ens, Lex, pi, D, n, seg_seed0=9_001)
    E_list = [e["E_seq"] for e in encoded]
    H_list = [e["H"] for e in encoded]
    s_intra, s_inter = intra_inter_ngram_sims(E_list, D)
    s_inter_seg = inter_segment_similarity(H_list)
    maj_curves = majority_curve_repeated_vector(
        E_list, pi, D, eta_list=(0.0, 0.05, 0.1), trials_per_m=4_000, seed=4_242
    )
    return {
        "D": D,
        "n": n,
        "N_pairs": N,
        "seed_lex": seed_lex,
        "seed_pi": seed_pi,
        "intra_ngram_mean_sim": s_intra,
        "inter_ngram_abs_mean_sim": s_inter,
        "inter_segment_abs_mean_sim": s_inter_seg,
        "majority_curves": maj_curves,
    }


def content_signature_from_Xseq(X_seq: List[np.ndarray],
                                majority: str = "strict") -> np.ndarray:
    """
    Construit une signature de contenu à partir des X_t (déjà réindexés par Pi^Δ),
    sans binding par la clé de segment K_s. On somme puis on seuillle.
    """
    if not X_seq:
        raise ValueError("X_seq vide")
    D = X_seq[0].shape[0]
    S = np.zeros((D,), dtype=np.int32)
    for x in X_seq:
        S += x.astype(np.int32, copy=False)
    if majority == "strict":
        return np.where(S >= 0, 1, -1).astype(np.int8, copy=False)
    elif majority == "unbiased":
        # même seuil, mais vous pouvez ajouter du dithering si besoin
        return np.where(S >= 0, 1, -1).astype(np.int8, copy=False)
    else:
        raise ValueError("majority must be 'strict' or 'unbiased'")

def span_signatures_from_trace(X_seq: List[np.ndarray],
                               win: int = 8,
                               stride: int = 4,
                               majority: str = "strict") -> List[np.ndarray]:
    """
    Balaye X_seq avec une fenêtre glissante pour produire plusieurs signatures de contenu.
    Chaque span donne un hypervecteur de contenu (±1 int8).
    """
    D = X_seq[0].shape[0] if X_seq else 0
    if D == 0:
        return []
    out: List[np.ndarray] = []
    T = len(X_seq)
    if T == 0:
        return out
    for start in range(0, max(1, T - win + 1), max(1, stride)):
        stop = min(T, start + win)
        sign = content_signature_from_Xseq(X_seq[start:stop], majority=majority)
        out.append(sign)
    # si la phrase est plus courte que win, on a au moins un span (start=0, stop=T)
    if not out and T > 0:
        out.append(content_signature_from_Xseq(X_seq, majority=majority))
    return out

def build_mem_pairs_from_encoded(encoded_en: List[Dict[str, Any]],
                                 encoded_fr: List[Dict[str, Any]],
                                 win: int = 8,
                                 stride: int = 4,
                                 majority: str = "strict",
                                 max_pairs: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Construit des paires (Z_en, Z_fr) *sans K_s* à partir des traces M8.
    - Pour chaque phrase parallélisée, on extrait des spans côté EN et FR.
    - On les aligne naïvement par rang (min(#spans_en, #spans_fr)).
    - Retourne une liste de (Z_en, Z_fr) en int8 ±1.
    """
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    N = min(len(encoded_en), len(encoded_fr))
    for i in range(N):
        X_en = encoded_en[i]["X_seq"]
        X_fr = encoded_fr[i]["X_seq"]
        spans_en = span_signatures_from_trace(X_en, win=win, stride=stride, majority=majority)
        spans_fr = span_signatures_from_trace(X_fr, win=win, stride=stride, majority=majority)
        L = min(len(spans_en), len(spans_fr))
        for t in range(L):
            ze = spans_en[t].astype(np.int8, copy=False)
            zf = spans_fr[t].astype(np.int8, copy=False)
            pairs.append((ze, zf))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs