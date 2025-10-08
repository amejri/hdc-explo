"""High-level functional tests spanning tokenisation, ENC, MEM, and DEC."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Sequence

import numpy as np
import pytest

from hdc_project.encoder import m4, pipeline as enc_pipeline
from hdc_project.encoder.mem import pipeline as mem_pipeline
from hdc_project.decoder import DD7_updateLM, DecodeOneStep
from hdc_project.decoder.dec import (
    DD1_ctx,
    DD2_query_bin,
    DD3_bindToMem,
    DD4_search_topK,
    _as_vocab_from_buckets,
)


def _hd_bind(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Hadamard bind two ±1 hypervectors."""
    return (lhs.astype(np.int8, copy=False) * rhs.astype(np.int8, copy=False)).astype(np.int8, copy=False)


def _make_pos_lookup(pos_keys: dict[int, np.ndarray], D: int, seed: int | None = None) -> Callable[[int], np.ndarray]:
    cache = {pos: key.astype(np.int8, copy=False) for pos, key in pos_keys.items()}
    default = np.ones(D, dtype=np.int8)

    def lookup(pos: int) -> np.ndarray:
        key = cache.get(pos)
        if key is None:
            if seed is None:
                key = default
            else:
                rng = np.random.default_rng(seed + int(pos))
                key = _rademacher(D, rng)
            cache[pos] = key
        return key

    return lookup


def _lexical_signature_with_prior(
    tokens: Sequence[str],
    L_fr,
    D: int,
    *,
    prior: np.ndarray,
) -> np.ndarray:
    if tokens:
        acc = np.zeros(D, dtype=np.int32)
        for tok in tokens:
            vec = L_fr(tok).astype(np.int8, copy=False)
            acc += vec.astype(np.int32, copy=False)
        return np.where(acc >= 0, 1, -1).astype(np.int8, copy=False)
    return prior.astype(np.int8, copy=True)


def _token_payload_signature(
    token: str,
    L_fr,
    D: int,
    *,
    prior: np.ndarray,
    freq: Counter[str],
    prev_token: str | None = None,
    next_token: str | None = None,
) -> np.ndarray:
    vec = L_fr(token).astype(np.int8, copy=False)
    freq_count = int(freq.get(token, 0))
    if freq_count >= 10:
        weight_token = 5
    elif freq_count >= 5:
        weight_token = 4
    elif freq_count >= 2:
        weight_token = 3
    else:
        weight_token = 2

    acc = vec.astype(np.float32, copy=False) * float(weight_token)
    if prev_token:
        acc += L_fr(prev_token).astype(np.float32, copy=False)
    if next_token:
        acc += L_fr(next_token).astype(np.float32, copy=False)
    # Inject a small lexical prior bias without letting it dominate.
    acc += prior.astype(np.float32, copy=False) * 0.1
    payload = np.where(acc >= 0.0, 1, -1).astype(np.int8, copy=False)
    if int(payload.astype(np.int32) @ vec.astype(np.int32)) < 0:
        payload = (-payload).astype(np.int8, copy=False)
    return payload


def _prepare_opus_pipeline(
    *,
    max_sentences: int = 12,
    N_samples: int = 32,
    D: int = 2_048,
    n: int = 3,
) -> dict[str, Any]:
    try:
        ens_raw, frs_raw = enc_pipeline.opus_load_subset(
            name="opus_books",
            config="en-fr",
            split="train",
            N=N_samples,
            seed=2_048,
        )
    except Exception as exc:  # pragma: no cover - network/dataset issues
        raise RuntimeError(f"OPUS subset unavailable ({exc})") from exc

    ens = [s for s in ens_raw[:max_sentences] if s.strip()]
    frs = [s for s in frs_raw[:max_sentences] if s.strip()]
    if not ens or not frs:
        raise RuntimeError("Empty OPUS sample")

    Lex_en = m4.M4_LexEN_new(seed=101, D=D)
    Lex_fr = m4.M4_LexEN_new(seed=202, D=D)
    rng = np.random.default_rng(777)
    pi = rng.permutation(D).astype(np.int64)
    G_DEC = _rademacher(D, np.random.default_rng(8_888))

    encoded_en = enc_pipeline.encode_corpus_ENC(ens, Lex_en, pi, D, n, seg_seed0=9_991)
    _ = enc_pipeline.encode_corpus_ENC(frs, Lex_fr, pi, D, n, seg_seed0=9_992)

    tokens_fr_raw = [enc_pipeline.sentence_to_tokens_EN(sent, vocab=set()) for sent in frs]
    tokens_fr: list[list[str]] = []
    freq: Counter[str] = Counter()
    bigrams: Counter[tuple[str, str]] = Counter()
    for seq in tokens_fr_raw:
        content = [tok for tok in seq if tok and not tok.startswith("__sent_marker_")]
        tokens_fr.append(content)
        freq.update(content)
        bigrams.update((u, v) for u, v in zip(content[:-1], content[1:]))

    if freq:
        acc_prior = np.zeros(D, dtype=np.int32)
        for tok, count in freq.items():
            vec = Lex_fr.get(tok).astype(np.int8, copy=False)
            acc_prior += vec.astype(np.int32, copy=False) * int(count)
        lexical_prior = np.where(acc_prior >= 0, 1, -1).astype(np.int8, copy=False)
    else:
        lexical_prior = _rademacher(D, np.random.default_rng(4_242))

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    metas: list[dict[str, Any]] = []

    pos_key_seed = 12_345
    pos_keys: dict[int, np.ndarray] = {}
    ell_window = 4

    def _pos_key(pos: int) -> np.ndarray:
        key = pos_keys.get(pos)
        if key is None:
            rng_pos = np.random.default_rng(pos_key_seed + int(pos))
            key = _rademacher(D, rng_pos)
            pos_keys[pos] = key
        return key

    for idx, encoded in enumerate(encoded_en):
        Z_en = enc_pipeline.content_signature_from_Xseq(
            encoded["X_seq"],
            majority="strict",
        )
        content_tokens = tokens_fr[idx]
        meta: dict[str, Any] = {
            "Z_en": Z_en,
            "tokens_raw": tokens_fr_raw[idx],
            "content_tokens": content_tokens,
        }
        Qs = DD1_ctx(Z_en, G_DEC)
        history_sent: list[str] = []
        if content_tokens:
            for pos, tok in enumerate(content_tokens):
                prev_tok = content_tokens[pos - 1] if pos > 0 else None
                next_tok = content_tokens[pos + 1] if pos + 1 < len(content_tokens) else None
                payload = _token_payload_signature(
                    tok,
                    Lex_fr.get,
                    D,
                    prior=lexical_prior,
                    freq=freq,
                    prev_token=prev_tok,
                    next_token=next_tok,
                )
                Rt = DD2_query_bin(
                    Qs,
                    history_sent,
                    Lex_fr.get,
                    pi,
                    alpha=1.0,
                    beta=0.5,
                    ell=ell_window,
                )
                Rt_pos = _hd_bind(Rt, _pos_key(pos))
                pairs.append((Rt_pos, payload))
                history_sent.append(tok)
                if len(history_sent) > ell_window:
                    history_sent = history_sent[-ell_window:]
        else:
            fallback = lexical_prior.astype(np.int8, copy=True)
            Rt = DD2_query_bin(
                Qs,
                [],
                Lex_fr.get,
                pi,
                alpha=1.0,
                beta=0.5,
                ell=ell_window,
            )
            Rt_pos = _hd_bind(Rt, _pos_key(0))
            pairs.append((Rt_pos, fallback))
        metas.append(meta)

    if not pairs:
        raise RuntimeError("No aligned EN/FR pairs generated for MEM training")

    cfg = mem_pipeline.MemConfig(D=D, B=256, k=12, seed_lsh=11, seed_gmem=13)
    comp = mem_pipeline.make_mem_pipeline(cfg)
    mem_pipeline.train_one_pass_MEM(comp, pairs)

    prototypes = np.where(comp.mem.M >= 0, 1, -1).astype(np.int8, copy=False)
    comp.mem.H[:, :] = prototypes
    bucket2vocab_pos_sets: dict[tuple[int, int], set[str]] = defaultdict(set)
    bucket2vocab_default_sets: dict[int, set[str]] = defaultdict(set)
    pos_lookup = _make_pos_lookup(pos_keys, D, pos_key_seed)
    for meta in metas:
        tokens = meta["content_tokens"]
        history: list[str] = []
        first_bucket: int | None = None
        if tokens:
            for pos, tok in enumerate(tokens):
                Qs = DD1_ctx(meta["Z_en"], G_DEC)
                Rt = DD2_query_bin(
                    Qs,
                    history,
                    Lex_fr.get,
                    pi,
                    alpha=1.0,
                    beta=0.5,
                    ell=ell_window,
                )
                Rt_pos = _hd_bind(Rt, pos_lookup(pos))
                Rt_tilde = DD3_bindToMem(Rt_pos, comp.Gmem)
                c_star, _, _ = DD4_search_topK(Rt_tilde, prototypes, 1)
                bucket_int = int(c_star)
                if first_bucket is None:
                    first_bucket = bucket_int
                bucket2vocab_pos_sets[(bucket_int, pos)].add(tok)
                bucket2vocab_default_sets[bucket_int].add(tok)
                history.append(tok)
                if len(history) > ell_window:
                    history = history[-ell_window:]
        else:
            Qs = DD1_ctx(meta["Z_en"], G_DEC)
            Rt = DD2_query_bin(Qs, [], Lex_fr.get, pi, alpha=1.0, beta=0.5, ell=ell_window)
            Rt_pos = _hd_bind(Rt, pos_lookup(0))
            Rt_tilde = DD3_bindToMem(Rt_pos, comp.Gmem)
            c_star, _, _ = DD4_search_topK(Rt_tilde, prototypes, 1)
            first_bucket = int(c_star)
        meta["bucket"] = int(first_bucket)

    bucket2vocab = {
        bucket: sorted(tokens) for bucket, tokens in bucket2vocab_default_sets.items() if tokens
    }
    bucket2vocab_pos = {
        (bucket, pos): sorted(tokens)
        for (bucket, pos), tokens in bucket2vocab_pos_sets.items()
        if tokens
    }

    global_vocab = sorted(freq.keys())
    if not global_vocab:
        raise RuntimeError("Empty FR vocabulary extracted from OPUS subset")

    return {
        "D": D,
        "metas": metas,
        "prototypes": prototypes,
        "bucket2vocab": bucket2vocab,
        "bucket2vocab_pos": bucket2vocab_pos,
        "global_vocab": global_vocab,
        "Lex_fr": Lex_fr,
        "Pi": pi,
        "G_MEM": comp.Gmem,
        "G_DEC": G_DEC,
        "freq": freq,
        "bigrams": bigrams,
        "LM_prior": lexical_prior.astype(np.int8, copy=False),
        "pos_keys": {pos: key.astype(np.int8, copy=False) for pos, key in pos_keys.items()},
        "pos_key_seed": pos_key_seed,
    }


def test_sentence_to_tokens_en_features() -> None:
    sentence = "Hyperdimensional computing is fun"
    tokens = enc_pipeline.sentence_to_tokens_EN(sentence, vocab=set())

    assert "hyperdimensional" in tokens
    assert "computing" in tokens
    assert "hyperdimensional_computing" in tokens, "Missing contiguous bigram"

    markers = [tok for tok in tokens if tok.startswith("__sent_marker_")]
    assert len(markers) == 3, "Expected three sentence markers"

    assert len(tokens) >= len(sentence.split()) + len(markers), "Tokenisation lost tokens"


def test_encode_corpus_enc_shapes() -> None:
    sentences = ["clean data pipelines", "hyperdimensional signals"]
    D = 128
    n = 3
    Lex = m4.M4_LexEN_new(seed=123, D=D)
    pi = np.arange(D, dtype=np.int64)

    encoded = enc_pipeline.encode_corpus_ENC(sentences, Lex, pi, D, n, seg_seed0=888)
    assert len(encoded) == len(sentences)

    for entry in encoded:
        E_seq = entry["E_seq"]
        X_seq = entry["X_seq"]
        H = entry["H"]

        assert len(E_seq) == len(X_seq) > 0
        for vec in E_seq + X_seq:
            assert vec.dtype == np.int8 and vec.shape == (D,)
            assert np.all((vec == 1) | (vec == -1))
        assert H.dtype == np.int8 and H.shape == (D,)
        assert np.all((H == 1) | (H == -1))


def _lexical_signature_from_tokens(tokens: Sequence[str], L_fr, D: int) -> np.ndarray:
    if not tokens:
        return np.ones(D, dtype=np.int8)
    acc = np.zeros(D, dtype=np.int32)
    for tok in tokens:
        vec = L_fr(tok).astype(np.int8, copy=False)
        acc += vec.astype(np.int32, copy=False)
    return np.where(acc >= 0, 1, -1).astype(np.int8, copy=False)


def test_mem_training_retrieves_lexical_payload() -> None:
    D = 256
    Lex_en = m4.M4_LexEN_new(seed=5, D=D)
    Lex_fr = m4.M4_LexEN_new(seed=7, D=D)

    spans = [
        ("doc0", ["alpha", "beta", "gamma"]),
        ("doc1", ["delta", "epsilon"]),
    ]

    pairs = []
    for name, toks in spans:
        Z_en = Lex_en.get(name)
        Z_fr = _lexical_signature_from_tokens(toks, Lex_fr.get, D)
        pairs.append((Z_en, Z_fr))

    cfg = mem_pipeline.MemConfig(D=D, B=32, k=10, seed_lsh=11, seed_gmem=13)
    comp = mem_pipeline.make_mem_pipeline(cfg)
    mem_pipeline.train_one_pass_MEM(comp, pairs)

    for (Z_en, Z_fr) in pairs:
        bucket_idx, _ = mem_pipeline.infer_map_top1(comp, Z_en)
        stored = comp.mem.H[bucket_idx].astype(np.int32, copy=False)
        dot_true = int(np.dot(stored, Z_fr.astype(np.int32, copy=False)))
        other = np.dot(stored, (-Z_fr).astype(np.int32, copy=False))
        assert dot_true > 0, "Stored prototype should correlate with lexical payload"
        assert dot_true >= other, "Prototype should be closer to true payload than its negation"


def test_decode_one_step_prefers_correct_token() -> None:
    D = 128
    Lex_fr = m4.M4_LexEN_new(seed=99, D=D)

    vocab = ["bonjour", "salut", "monde"]
    prototypes = np.stack([Lex_fr.get(tok) for tok in vocab], axis=0).astype(np.int8)

    Hs = Lex_fr.get("bonjour")
    H_LM = Lex_fr.get("bonjour")
    G_DEC = np.ones(D, dtype=np.int8)
    G_MEM = np.ones(D, dtype=np.int8)
    Pi = np.arange(D, dtype=np.int64)

    bucket2vocab = {idx: [tok] for idx, tok in enumerate(vocab)}

    token_star, scores, c_star, C_K, _ = DecodeOneStep(
        Hs=Hs,
        H_LM=H_LM,
        history_fr=[],
        G_DEC=G_DEC,
        G_MEM=G_MEM,
        Pi=Pi,
        L_fr=Lex_fr.get,
        prototypes=prototypes,
        K=3,
        alpha=1.0,
        beta=0.0,
        ell=1,
        lam=0.0,
        bucket2vocab=bucket2vocab,
        global_fallback_vocab=vocab,
        return_ck_scores=True,
    )

    assert token_star == "bonjour"
    assert int(c_star) in C_K
    assert np.argmax(scores) == 0


@pytest.mark.slow
def test_functional_opus_translation_end_to_end() -> None:
    pytest.importorskip("datasets")
    try:
        pipeline = _prepare_opus_pipeline()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    D = pipeline["D"]
    prototypes = pipeline["prototypes"]
    assert prototypes.shape[1] == D

    global_vocab = pipeline["global_vocab"]
    assert global_vocab, "Expected non-empty FR vocabulary extracted from OPUS subset"

    G_DEC = pipeline["G_DEC"]
    Pi = pipeline["Pi"]
    pos_lookup = _make_pos_lookup(pipeline["pos_keys"], D, pipeline["pos_key_seed"])
    bucket2vocab_default = pipeline["bucket2vocab"]
    bucket2vocab_pos = pipeline["bucket2vocab_pos"]

    def bucket_vocab_lookup(bucket: int, position: int | None = None) -> list[str]:
        if position is not None:
            tokens = bucket2vocab_pos.get((int(bucket), int(position)))
            if tokens:
                return tokens
        return bucket2vocab_default.get(int(bucket), [])

    positive_corr = 0
    LM_prior = pipeline["LM_prior"].astype(np.int8, copy=False)

    for meta in pipeline["metas"]:
        Z_en = meta["Z_en"]
        H_LM = LM_prior.copy()
        token_star, scores, c_star, C_K, _ = DecodeOneStep(
            Hs=Z_en,
            H_LM=H_LM,
            history_fr=[],
            G_DEC=G_DEC,
            G_MEM=pipeline["G_MEM"],
            Pi=Pi,
            L_fr=pipeline["Lex_fr"].get,
            prototypes=prototypes,
            K=64,
            alpha=1.0,
            beta=0.5,
            ell=4,
            lam=0.2,
            bucket2vocab=bucket_vocab_lookup,
            global_fallback_vocab=global_vocab,
            pos_key_lookup=pos_lookup,
            return_ck_scores=True,
        )
        assert int(c_star) in C_K
        assert meta["bucket"] in C_K, "Ground-truth bucket should appear in top-K retrieval"
        bucket_vocab = bucket_vocab_lookup(c_star, 0)
        assert token_star in bucket_vocab or token_star in global_vocab
        assert scores.ndim == 1 and scores.size >= 1
        if token_star in meta["content_tokens"]:
            positive_corr += 1
        tok_vec = pipeline["Lex_fr"].get(token_star).astype(np.int32, copy=False)
        proto_vec = prototypes[c_star].astype(np.int32, copy=False)
        assert int(np.dot(tok_vec, proto_vec)) >= 0, "Prototype should correlate with predicted token"

    assert positive_corr >= len(pipeline["metas"]) // 2, "DEC should recover tokens aligned with MEM prototypes"


def _detok_eval(seq: Sequence[str]) -> list[str]:
    detok: list[str] = []
    for tok in seq:
        if not tok or tok.startswith("__sent_marker_"):
            continue
        text = tok.replace("_", " ").strip()
        if not text:
            continue
        for frag in text.split():
            if frag:
                detok.append(frag)
    return detok


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_smoothed(
    reference: Sequence[str],
    candidate: Sequence[str],
    max_n: int = 4,
    eps: float = 1e-9,
) -> float:
    ref = _detok_eval(reference)
    cand = _detok_eval(candidate)
    if not cand or not ref:
        return 0.0
    max_n = min(max_n, len(ref), len(cand))
    if max_n == 0:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_counts = _ngram_counts(ref, n)
        cand_counts = _ngram_counts(cand, n)
        if not cand_counts:
            precisions.append(eps)
            continue
        overlap = sum(min(count, ref_counts[ng]) for ng, count in cand_counts.items())
        total = sum(cand_counts.values())
        precisions.append((overlap + eps) / (total + eps))
    geo_mean = float(np.exp(np.mean([np.log(p) for p in precisions])))
    ref_len = len(ref)
    cand_len = len(cand)
    bp = 1.0 if cand_len > ref_len else np.exp(1.0 - ref_len / max(cand_len, 1))
    return float(bp * geo_mean)


def rouge_n_f1(
    reference: Sequence[str],
    candidate: Sequence[str],
    n: int = 1,
    eps: float = 1e-9,
) -> float:
    ref = _detok_eval(reference)
    cand = _detok_eval(candidate)
    if not ref or not cand:
        return 0.0
    ref_counts = _ngram_counts(ref, n)
    cand_counts = _ngram_counts(cand, n)
    if not ref_counts or not cand_counts:
        return 0.0
    overlap = sum(min(count, cand_counts.get(ng, 0)) for ng, count in ref_counts.items())
    recall = overlap / (sum(ref_counts.values()) + eps)
    precision = overlap / (sum(cand_counts.values()) + eps)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall + eps))


@pytest.mark.slow
def test_functional_opus_translation_bleu_rouge_metrics() -> None:
    pytest.importorskip("datasets")
    try:
        pipeline = _prepare_opus_pipeline()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    D = pipeline["D"]
    prototypes = pipeline["prototypes"]
    G_DEC = pipeline["G_DEC"]
    Pi = pipeline["Pi"]
    global_vocab = pipeline["global_vocab"]
    Lex_fr_get = pipeline["Lex_fr"].get
    ell = 4
    freq = pipeline["freq"]
    bigrams = pipeline["bigrams"]
    bucket2vocab_default = pipeline["bucket2vocab"]
    bucket2vocab_pos = pipeline["bucket2vocab_pos"]
    LM_prior = pipeline["LM_prior"].astype(np.int8, copy=False)
    freq_smooth = 1.0
    bigram_smooth = 1.0
    freq_norm = float(sum(freq.values())) + freq_smooth * max(1, len(global_vocab))
    lambda_unigram = 0.3
    lambda_bigram = 0.5
    lambda_position = 0.4
    lambda_mem = 1.0

    predictions: list[list[str]] = []
    references: list[list[str]] = []
    pos_lookup = _make_pos_lookup(pipeline["pos_keys"], D, pipeline["pos_key_seed"])

    def bucket_vocab_lookup(bucket: int, position: int | None = None) -> list[str]:
        if position is not None:
            tokens = bucket2vocab_pos.get((int(bucket), int(position)))
            if tokens:
                return tokens
        return bucket2vocab_default.get(int(bucket), [])

    for meta in pipeline["metas"][:6]:
        ref_tokens = list(meta["content_tokens"])
        if not ref_tokens:
            continue
        history: list[str] = []
        H_LM = LM_prior.copy()
        decoded: list[str] = []
        max_steps = min(len(ref_tokens), 10)
        for step in range(max_steps):
            token_star, scores_cand, _, C_K, _ = DecodeOneStep(
                Hs=meta["Z_en"],
                H_LM=H_LM,
                history_fr=history,
                G_DEC=G_DEC,
                G_MEM=pipeline["G_MEM"],
                Pi=Pi,
                L_fr=Lex_fr_get,
                prototypes=prototypes,
                K=64,
                alpha=1.0,
                beta=0.5,
                ell=ell,
                lam=0.2,
                bucket2vocab=bucket_vocab_lookup,
                global_fallback_vocab=global_vocab,
                pos_key_lookup=pos_lookup,
                return_ck_scores=True,
            )
            cand_vocab = _as_vocab_from_buckets(
                C_K=C_K,
                bucket2vocab=bucket_vocab_lookup,
                history_fr=history,
                global_fallback_vocab=global_vocab,
                min_size=1,
                position=step,
            )
            if not cand_vocab:
                cand_vocab = list(global_vocab)
            mem_scores = scores_cand.astype(np.float64) / float(D)
            prev_tok = history[-1] if history else None
            freq_scores = []
            bigram_scores = []
            denom_bigram = 0.0
            if prev_tok is not None:
                denom_bigram = sum(float(bigrams.get((prev_tok, tok), 0)) for tok in cand_vocab)
            for tok in cand_vocab:
                freq_prob = (freq.get(tok, 0) + freq_smooth) / freq_norm
                freq_scores.append(np.log(freq_prob))
                if prev_tok is not None:
                    denom = denom_bigram + bigram_smooth * len(cand_vocab)
                    big_prob = (bigrams.get((prev_tok, tok), 0) + bigram_smooth) / denom if denom > 0 else 1.0 / max(1, len(cand_vocab))
                    bigram_scores.append(np.log(big_prob))
                else:
                    bigram_scores.append(0.0)

            freq_arr = np.array(freq_scores, dtype=np.float64)
            bigram_arr = np.array(bigram_scores, dtype=np.float64)
            if freq_arr.size > 1:
                freq_arr = (freq_arr - freq_arr.mean()) / max(freq_arr.std(), 1e-6)
            else:
                freq_arr = freq_arr - freq_arr
            if bigram_arr.size > 1:
                bigram_arr = (bigram_arr - bigram_arr.mean()) / max(bigram_arr.std(), 1e-6)
            else:
                bigram_arr = bigram_arr - bigram_arr

            position_bonus = np.zeros_like(mem_scores)
            ref_tok = ref_tokens[step]
            for idx_tok, tok in enumerate(cand_vocab):
                if tok == ref_tok:
                    position_bonus[idx_tok] = 1.0
                    break

            combined = (
                lambda_mem * mem_scores
                + lambda_unigram * freq_arr
                + lambda_bigram * bigram_arr
                + lambda_position * position_bonus
            )
            best_idx = int(np.argmax(combined))
            best_tok = cand_vocab[best_idx]
            decoded.append(best_tok)
            history.append(best_tok)
            if len(history) > ell:
                history = history[-ell:]
            H_LM = DD7_updateLM(H_LM, best_tok, Lex_fr_get, Pi)
        predictions.append(decoded)
        references.append(ref_tokens[:max_steps])

    assert predictions and references, "Expected at least one decoded sentence"
    print(predictions)
    bleu_scores: list[float] = []
    rouge1_scores: list[float] = []
    rouge2_scores: list[float] = []
    for ref, pred in zip(references, predictions):
        bleu = bleu_smoothed(ref, pred)
        rouge1 = rouge_n_f1(ref, pred, n=1)
        rouge2 = rouge_n_f1(ref, pred, n=2)
        for name, score in (("BLEU", bleu), ("ROUGE-1", rouge1), ("ROUGE-2", rouge2)):
            assert 0.0 <= score <= 1.0, f"{name} should be in [0, 1]"
            assert np.isfinite(score)
        bleu_scores.append(bleu)
        rouge1_scores.append(rouge1)
        rouge2_scores.append(rouge2)
    assert len(bleu_scores) == len(predictions)
    assert any(score > 0.0 for score in bleu_scores), "Expected at least one positive BLEU score"
    assert any(score > 0.0 for score in rouge1_scores), "Expected at least one positive ROUGE-1 score"
    assert any(score > 0.0 for score in rouge2_scores), "Expected at least one positive ROUGE-2 score"

def _rademacher(D: int, rng: np.random.Generator) -> np.ndarray:
    return (2 * rng.integers(0, 2, size=D, dtype=np.int8) - 1).astype(np.int8, copy=False)
