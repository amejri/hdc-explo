"""Functional tests for the ENC → MEM → DEC translation pipeline."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Sequence, Tuple

import numpy as np

from hdc_project.encoder import m4, pipeline as enc_pipeline
from hdc_project.encoder.mem import pipeline as mem_pipeline


def _lexical_signature_from_tokens(
    tokens: Sequence[str],
    L_fr_mem,
    D: int,
) -> np.ndarray:
    if not tokens:
        return np.ones(D, dtype=np.int8)
    acc = np.zeros(D, dtype=np.int32)
    for tok in tokens:
        vec = L_fr_mem(tok).astype(np.int8, copy=False)
        acc += vec.astype(np.int32, copy=False)
    return np.where(acc >= 0, 1, -1).astype(np.int8, copy=False)


def _content_signature_from_Xseq(X_seq: Sequence[np.ndarray]) -> np.ndarray:
    if not X_seq:
        raise ValueError("X_seq vide")
    acc = np.zeros(X_seq[0].shape[0], dtype=np.int32)
    for x in X_seq:
        acc += x.astype(np.int32, copy=False)
    return np.where(acc >= 0, 1, -1).astype(np.int8, copy=False)


def _span_signatures_from_trace(
    X_seq: Sequence[np.ndarray],
    *,
    win: int,
    stride: int,
) -> List[Tuple[np.ndarray, int, int]]:
    T = len(X_seq)
    if T == 0:
        return []
    spans: List[Tuple[np.ndarray, int, int]] = []
    if T <= win:
        spans.append((_content_signature_from_Xseq(X_seq), 0, T))
        return spans
    for start in range(0, T - win + 1, max(1, stride)):
        stop = start + win
        spans.append((_content_signature_from_Xseq(X_seq[start:stop]), start, stop))
    return spans


def _build_mem_pairs_with_meta(
    encoded_en: Sequence[dict],
    encoded_fr: Sequence[dict],
    tokens_fr: Sequence[Sequence[str]],
    *,
    L_fr_mem,
    D: int,
    win: int = 8,
    stride: int = 4,
    max_pairs: int | None = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[dict]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    meta: List[dict] = []
    N = min(len(encoded_en), len(encoded_fr))
    for idx in range(N):
        spans_en = _span_signatures_from_trace(encoded_en[idx]["X_seq"], win=win, stride=stride)
        spans_fr = _span_signatures_from_trace(encoded_fr[idx]["X_seq"], win=win, stride=stride)
        tok_fr = list(tokens_fr[idx]) if idx < len(tokens_fr) else []
        L = min(len(spans_en), len(spans_fr))
        for (ze, start_en, stop_en), (_, start_fr, stop_fr) in zip(spans_en[:L], spans_fr[:L]):
            span_tokens = tok_fr[start_fr:stop_fr] if tok_fr else []
            zf_lex = _lexical_signature_from_tokens(span_tokens, L_fr_mem, D)
            pairs.append((ze, zf_lex))
            meta.append(
                {
                    "sentence_idx": idx,
                    "start": start_fr,
                    "stop": stop_fr,
                    "history_tokens": tok_fr[max(0, start_fr - stride):start_fr] if tok_fr else [],
                    "span_tokens": span_tokens,
                    "Z_en": ze,
                }
            )
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs, meta
    return pairs, meta


def _bucket_vocab_freq(comp: mem_pipeline.MemComponents, span_meta: Sequence[dict], tokens_fr: Sequence[Sequence[str]]) -> dict[int, list[tuple[str, int]]]:
    bucket_counts: dict[int, Counter] = defaultdict(Counter)
    for meta in span_meta:
        bucket_idx, _ = mem_pipeline.infer_map_top1(comp, meta["Z_en"])
        bucket_idx = int(bucket_idx)
        meta["bucket_idx"] = bucket_idx
        for tok in meta.get("span_tokens", []):
            bucket_counts[bucket_idx][tok] += 1
    return {
        bucket: sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        for bucket, counter in bucket_counts.items()
    }


def test_enc_mem_dec_pipeline_smoke() -> None:
    sentences_en = [
        "hyperdimensional computing is fun",
        "vector symbolic architectures are powerful",
        "encoding words into hyperspace",
        "memory augmented networks love clean data",
    ]
    sentences_fr = [
        "le calcul hyperdimensionnel est amusant",
        "les architectures symboliques vectorielles sont puissantes",
        "encoder des mots dans l'hyperspace",
        "les réseaux augmentés de mémoire aiment les données propres",
    ]

    D = 512
    n = 3
    rng = np.random.default_rng(123)
    Lex_en = m4.M4_LexEN_new(seed=1, D=D)
    Lex_fr_mem = m4.M4_LexEN_new(seed=2, D=D)
    pi = rng.permutation(D).astype(np.int64)

    encoded_en = enc_pipeline.encode_corpus_ENC(sentences_en, Lex_en, pi, D, n, seg_seed0=999)
    encoded_fr = enc_pipeline.encode_corpus_ENC(sentences_fr, Lex_fr_mem, pi, D, n, seg_seed0=1999)

    tokens_fr = [enc_pipeline.sentence_to_tokens_EN(sent, vocab=set()) for sent in sentences_fr]
    pairs_mem, span_meta = _build_mem_pairs_with_meta(
        encoded_en,
        encoded_fr,
        tokens_fr,
        L_fr_mem=Lex_fr_mem.get,
        D=D,
        win=6,
        stride=3,
        max_pairs=10_000,
    )
    assert pairs_mem, "aucune paire MEM générée"

    cfg = mem_pipeline.MemConfig(D=D, B=128, k=12, seed_lsh=10, seed_gmem=11)
    comp = mem_pipeline.make_mem_pipeline(cfg)
    mem_pipeline.train_one_pass_MEM(comp, pairs_mem)

    bucket2vocab_freq = _bucket_vocab_freq(comp, span_meta, tokens_fr)
    assert bucket2vocab_freq, "vocabulaire par bucket vide"

    meta = next((m for m in span_meta if m.get("span_tokens")), span_meta[0])
    bucket_idx = int(meta.get("bucket_idx", 0))
    prototype = comp.mem.H[bucket_idx].astype(np.int8, copy=False)

    # Payload lexical alignment: every token in the span should correlate positively.
    positives = 0
    for tok in meta.get("span_tokens", []):
        vec = Lex_fr_mem.get(tok).astype(np.int8, copy=False)
        dot = int(np.dot(prototype.astype(np.int32), vec.astype(np.int32)))
        if dot > 0:
            positives += 1
    if meta.get("span_tokens"):
        ratio = positives / len(meta["span_tokens"])
        assert ratio >= 0.5, f"corrélation payload/lexique trop faible ({ratio:.3f})"

    # Candidate gating: reference tokens appear among the frequent bucket entries.
    freq_list = bucket2vocab_freq.get(bucket_idx, [])[:32]
    top_tokens = {tok for tok, _ in freq_list}
    if meta.get("span_tokens"):
        assert any(tok in top_tokens for tok in meta["span_tokens"]), "span hors vocabulaire du bucket"
