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
):
    tokens_list = list(tokens)
    if segmgr is None:
        seg_seed = int(LexEN.rng.integers(1, 2**31 - 1))
        segmgr = M7_SegMgr_new(seg_seed, D)
    K_s = M7_curKey(segmgr)

    S = acc_S if acc_S is not None else M6_SegAcc_init(D)

    # OPT: cache minimal pour Δ<n
    if pi_pows is None:
        pi_pows = _ensure_pi_pows(pi, n)

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

    # Local bindings pour éviter les recherches d’attribut en boucle
    _M6_push = M6_SegAcc_push
    _M3_bind = M3_bind
    _pi_pows = pi_pows
    _K = K_s

    L = len(tokens_list)
    for t in range(L):
        left = t - n + 1
        if left < 0:
            left = 0
        window = tokens_list[left : t + 1]
        E_t = pick_ngram(window)
        Delta = t - left  # < n
        idx = _pi_pows[Delta]
        X_t = E_t[idx]           # déjà int8
        S = _M6_push(S, X_t, _K)
        if Xb_seq is not None:
            Xb_seq.append(_M3_bind(X_t, _K))
        E_seq.append(E_t)
        X_seq.append(X_t)

    if majority_mode == "unbiased":
        H = unbiased_sign_int8(S, rng_obj=rng)
    elif majority_mode == "strict":
        H = strict_sign_int8(S)
    else:
        raise ValueError("majority_mode must be 'unbiased' or 'strict'")

    if Xb_seq is not None:
        return E_seq, X_seq, Xb_seq, S, H
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
    empty_vocab: set[str] = set()  # OPT: réutilisé
    for idx, sent in enumerate(sentences, 1):
        toks = sentence_to_tokens_EN(sent, vocab=empty_vocab)
        toks = [
            (f"{tok}_{idx}" if tok.startswith("__sent_marker_") else tok)
            for tok in toks
        ]
        toks.extend([f"__sent_{idx}_marker_{j}" for j in range(3)])
        seg_seed = int(rng.integers(1, 2**31 - 1))
        out.append(enc_sentence_ENC(toks, n, pi, LexEN, D, seg_seed))
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
    # intra: moyenne des <E_t, E_{t+1}>/D à l’intérieur d’une phrase
    intra_vals: List[float] = []
    for seq in E_seq_list:
        if len(seq) >= 2:
            M = np.stack(seq, axis=0).astype(np.int16, copy=False)
            dots = (M[:-1] * M[1:]).sum(axis=1, dtype=np.int64) / float(D)
            intra_vals.append(float(dots.mean()))
    s_intra = float(np.mean(intra_vals)) if intra_vals else 0.0

    # inter: moyenne des |<E^i_0, E^{i+1}_0>|/D entre phrases consécutives
    heads = [seq[0] for seq in E_seq_list if len(seq) > 0]
    if len(heads) >= 2:
        H = np.stack(heads, axis=0).astype(np.int16, copy=False)
        dots = (H[:-1] * H[1:]).sum(axis=1, dtype=np.int64) / float(D)
        s_inter = float(np.mean(np.abs(dots)))
    else:
        s_inter = 0.0
    return s_intra, s_inter


def inter_segment_similarity(H_list: List[np.ndarray]) -> float:
    """Compute the mean absolute similarity between consecutive ENC segments.

    Args:
        H_list: Sequence of segment signatures ``H`` produced by the encoder.

    Returns:
        The arithmetic mean of ``|<H_i, H_{i+1}>| / D`` for consecutive pairs.
        Returns ``0.0`` when fewer than two signatures are provided.
    """
    if len(H_list) < 2:
        return 0.0
    H = np.stack(H_list, axis=0).astype(np.int16, copy=False)
    dots = (H[:-1] * H[1:]).sum(axis=1, dtype=np.int64) / float(H.shape[1])
    return float(np.mean(np.abs(dots)))


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

        # 1) Pré-compute tous les X_t = Pi^t(E_t)
        #    NB: on garde int16 pour la somme
        X_stack = np.empty((m, D), dtype=np.int16)
        for t, E_t in enumerate(E_seq):
            X_stack[t] = M2_perm_pow(E_t, pi, t).astype(np.int16, copy=False)

        # 2) Somme "clean" + signature
        S_clean = X_stack.sum(axis=0, dtype=np.int32)           # pas besoin de K pour la décision
        H_clean = M6_SegAcc_sign(S_clean.astype(np.int16, copy=False))

        # 3) Runs bruités (tous vectorisés)
        for eta in eta_list:
            if eta <= 0.0:
                diff = 0.0
            else:
                flips = rng.random(size=X_stack.shape) < float(eta)  # (m,D) bool
                signs = np.where(flips, -1, 1).astype(np.int16, copy=False)
                S_noisy = (X_stack * signs).sum(axis=0, dtype=np.int32)
                H_noisy = M6_SegAcc_sign(S_noisy.astype(np.int16, copy=False))
                diff = float((H_clean.astype(np.int8) * H_noisy.astype(np.int8) == -1).mean())
            results[eta].append((m, diff))

    # Agrégation par longueur m
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
    """Tokenise ``sentence`` and augment it with contiguous bigram features.

    The output preserves the base tokens and appends ``token_i_token_{i+1}``
    bigrams to encourage richer context windows for downstream MEM training.
    The ``vocab`` parameter is kept for API compatibility and is currently
    ignored.

    Args:
        sentence: Raw sentence to tokenise.
        vocab: Token set (ignored).

    Returns:
        A list containing the original tokens followed by their contiguous
        bigrams.
    """

    base_tokens = tokenize_en(sentence)
    tokens = list(base_tokens)
    max_n = min(4, len(base_tokens))
    for n in range(2, max_n + 1):
        ngrams = [
            "_".join(base_tokens[i : i + n])
            for i in range(len(base_tokens) - n + 1)
        ]
        tokens += ngrams

    markers = [f"__sent_marker_{j}" for j in range(3)]
    if tokens:
        segment = max(1, len(tokens) // (len(markers) + 1))
        for j, marker in enumerate(markers, start=1):
            pos = min(len(tokens), j * segment)
            tokens.insert(pos, marker)
    else:
        tokens.extend(markers)

    target_len = 24
    if base_tokens and len(tokens) < target_len:
        needed = target_len - len(tokens)
        for idx in range(needed):
            tok = base_tokens[idx % len(base_tokens)]
            tokens.append(f"{tok}_dup{idx}")

    return tokens


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
    Xm = np.stack(X_seq, axis=0).astype(np.int16, copy=False)
    S = Xm.sum(axis=0, dtype=np.int32)
    if majority == "strict":
        return np.where(S > 0, 1, -1).astype(np.int8, copy=False)
    elif majority == "unbiased":
        return unbiased_sign_int8(S)
    else:
        raise ValueError("majority must be 'strict' or 'unbiased'")

def span_signatures_from_trace(X_seq: List[np.ndarray],
                               win: int = 8,
                               stride: int = 4,
                               majority: str = "strict",
                               min_win: int = 3,
                               multiscale_small: bool = True,) -> List[np.ndarray]:
    """ 
    Balaye X_seq avec une fenêtre glissante pour produire plusieurs signatures de contenu.
    Chaque span donne un hypervecteur de contenu (±1 int8).
    """
    T = len(X_seq)
    if T == 0:
        return []

    out: List[np.ndarray] = []

    # Cas nominal : fenêtre fixe w = min(win, T)
    w = min(win, T)
    for start in range(0, T - w + 1, max(1, stride)):
        stop = start + w
        out.append(content_signature_from_Xseq(X_seq[start:stop], majority))

    # Si rien (typiquement T < w et stride trop grand), ajoute au moins 1 span
    if not out:
        out.append(content_signature_from_Xseq(X_seq, majority))

    # Si phrase courte, enrichir via multi-échelles : w2 = min_win..T
    if multiscale_small and T < win:
        w_min = max( min_win, 2 )
        for w2 in range(w_min, T):  # génère tailles 2..(T-1)
            # on peut balayer aussi avec stride= max(1, w2//2) pour varier un peu
            step = max(1, min(stride, w2))
            for start in range(0, T - w2 + 1, step):
                stop = start + w2
                out.append(content_signature_from_Xseq(X_seq[start:stop], majority))

    return out

def _span_signatures_fast(
    X_seq: List[np.ndarray],
    win: int,
    stride: int,
    majority: str = "strict",
) -> List[np.ndarray]:
    """
    Version vectorisée pour produire les signatures de contenu sur fenêtres glissantes :
    - empile X_seq en matrice (T,D)
    - cumule (cumsum) sur l'axe temporel
    - récupère toutes les sommes de fenêtres d'un coup via la technique pad+cumsum
    - seuillage vectorisé
    """
    T = len(X_seq)
    if T == 0:
        return []

    # Empilement en (T,D) en int16 (accumulation exacte jusqu'à ~32767)
    X = np.vstack(X_seq).astype(np.int16, copy=False)
    D = X.shape[1]

    # Cas T <= win : un seul span (équivalent au code précédent)
    if T <= win:
        S = X.sum(axis=0, dtype=np.int32)
        if majority == "strict":
            Z = np.where(S > 0, 1, -1).astype(np.int8, copy=False)
        elif majority == "unbiased":
            # tie-break non biaisé : on peut injecter un léger dithering si besoin
            # pour rester 100% déterministe, on garde >= 0:
            Z = np.where(S >= 0, 1, -1).astype(np.int8, copy=False)
        else:
            raise ValueError("majority must be 'strict' or 'unbiased'")
        return [Z]

    # Indices de départ des fenêtres
    starts = np.arange(0, T - win + 1, max(1, stride), dtype=np.int64)
    ends = starts + win - 1

    # Sommes préfixes (T,D) -> padding (T+1,D)
    cs = np.cumsum(X, axis=0, dtype=np.int32)
    pad = np.vstack([np.zeros((1, D), dtype=np.int32), cs])

    # Sommes de fenêtres : S[s:e] = pad[e+1] - pad[s] pour tous s, e
    S = pad[ends + 1] - pad[starts]              # (num_windows, D)

    # Seuillage vectorisé
    if majority == "strict":
        Z = np.where(S > 0, 1, -1).astype(np.int8, copy=False)
    elif majority == "unbiased":
        Z = np.where(S >= 0, 1, -1).astype(np.int8, copy=False)
    else:
        raise ValueError("majority must be 'strict' or 'unbiased'")

    # Retourne une liste de vues par ligne (évite des copies inutiles)
    return [Z[i] for i in range(Z.shape[0])]


def build_mem_pairs_from_encoded(
    encoded_en: List[Dict[str, Any]],
    encoded_fr: List[Dict[str, Any]],
    win: int = 8,
    stride: int = 4,
    majority: str = "strict",
    max_pairs: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Construit des paires (Z_en, Z_fr) *sans K_s* à partir des traces M8.
    Optimisations:
    - concaténation adaptative de phrases courtes pour garantir assez de fenêtres
    - calcul des spans entièrement vectorisé (_span_signatures_fast)
    - évite les casts répétés et copies superflues
    """
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    N = min(len(encoded_en), len(encoded_fr))
    if N == 0:
        return pairs

    # Heuristique : estimer longueur médiane et choisir k pour atteindre win
    # (échantillonner au plus 256 phrases pour ne pas payer une passe complète)
    sample_n = min(N, 256)
    lengths = np.fromiter((len(encoded_en[i]["X_seq"]) for i in range(sample_n)), count=sample_n, dtype=np.int32)
    med = int(np.median(lengths)) if sample_n > 0 else 0
    k = max(1, int(np.ceil(win / max(1, med))))   # nb de phrases concaténées

    i = 0
    while i < N:
        j = min(N, i + k)

        # Concat sans copies inutiles : on construit des listes puis on empile
        X_en_list = []
        X_fr_list = []
        for t in range(i, j):
            X_en_list.extend(encoded_en[t]["X_seq"])
            X_fr_list.extend(encoded_fr[t]["X_seq"])

        # Spans vectorisés
        spans_en = _span_signatures_fast(X_en_list, win=win, stride=stride, majority=majority)
        spans_fr = _span_signatures_fast(X_fr_list, win=win, stride=stride, majority=majority)

        # Aligner par rang
        L = min(len(spans_en), len(spans_fr))
        if L:
            # Empiler en batch, puis appairer sans copies
            Ze = np.stack(spans_en[:L], axis=0)  # (L,D) int8
            Zf = np.stack(spans_fr[:L], axis=0)  # (L,D) int8

            # On étire par blocs pour limiter le coût Python
            if max_pairs is not None:
                remaining = max_pairs - len(pairs)
                L = min(L, remaining)
                Ze = Ze[:L]
                Zf = Zf[:L]

            pairs.extend((Ze[r], Zf[r]) for r in range(L))

            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs

        i = j

    return pairs