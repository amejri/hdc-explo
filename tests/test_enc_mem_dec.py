"""Functional tests for the ENC → MEM → DEC translation pipeline."""
from __future__ import annotations

from collections import defaultdict
from typing import List, Sequence, Tuple

import numpy as np

from hdc_project.encoder import m4, pipeline as enc_pipeline
from hdc_project.encoder.mem import pipeline as mem_pipeline
from hdc_project.decoder import (
    DD7_updateLM,
    DecodeOneStep,
)
from hdc_project.decoder.dec import (
    rademacher,
)


def _content_signature_from_Xseq(
    X_seq: Sequence[np.ndarray],
    *,
    majority: str = "strict",
) -> np.ndarray:
    if not X_seq:
        raise ValueError("X_seq vide")
    acc = np.zeros(X_seq[0].shape[0], dtype=np.int32)
    for x in X_seq:
        acc += x.astype(np.int32, copy=False)
    if majority == "strict":
        return np.where(acc >= 0, 1, -1).astype(np.int8, copy=False)
    if majority == "unbiased":
        rng_local = np.random.default_rng(0)
        ties = acc == 0
        acc[ties] = rng_local.integers(0, 2, size=int(ties.sum()), dtype=np.int32) * 2 - 1
        return np.where(acc >= 0, 1, -1).astype(np.int8, copy=False)
    raise ValueError("majority must be 'strict' or 'unbiased'")


def _span_signatures_from_trace(
    X_seq: Sequence[np.ndarray],
    *,
    win: int,
    stride: int,
    majority: str,
) -> List[Tuple[np.ndarray, int, int]]:
    T = len(X_seq)
    if T == 0:
        return []
    spans: List[Tuple[np.ndarray, int, int]] = []
    if T <= win:
        spans.append((_content_signature_from_Xseq(X_seq, majority=majority), 0, T))
        return spans
    for start in range(0, T - win + 1, max(1, stride)):
        stop = start + win
        spans.append(
            (
                _content_signature_from_Xseq(X_seq[start:stop], majority=majority),
                start,
                stop,
            )
        )
    return spans


def _build_mem_pairs_with_meta(
    encoded_en: Sequence[dict],
    encoded_fr: Sequence[dict],
    tokens_fr: Sequence[Sequence[str]],
    *,
    win: int = 8,
    stride: int = 4,
    majority: str = "strict",
    max_pairs: int | None = None,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[dict]]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    meta: List[dict] = []
    N = min(len(encoded_en), len(encoded_fr))
    for idx in range(N):
        spans_en = _span_signatures_from_trace(
            encoded_en[idx]["X_seq"],
            win=win,
            stride=stride,
            majority=majority,
        )
        spans_fr = _span_signatures_from_trace(
            encoded_fr[idx]["X_seq"],
            win=win,
            stride=stride,
            majority=majority,
        )
        tok_fr = list(tokens_fr[idx]) if idx < len(tokens_fr) else []
        L = min(len(spans_en), len(spans_fr))
        for (ze, start_en, stop_en), (zf, start_fr, stop_fr) in zip(spans_en[:L], spans_fr[:L]):
            pairs.append((ze, zf))
            span_tokens = tok_fr[start_fr:stop_fr] if tok_fr else []
            history = tok_fr[max(0, start_fr - stride):start_fr] if tok_fr else []
            meta.append(
                {
                    "sentence_idx": idx,
                    "start": start_fr,
                    "stop": stop_fr,
                    "history_tokens": history,
                    "span_tokens": span_tokens,
                    "Z_en": ze,
                    "Z_fr": zf,
                }
            )
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs, meta
    return pairs, meta


def _bucket_vocab(comp: mem_pipeline.MemComponents, span_meta: Sequence[dict]) -> dict[int, list[str]]:
    bucket_vocab: dict[int, set[str]] = defaultdict(set)
    for meta in span_meta:
        bucket_idx, _ = mem_pipeline.infer_map_top1(comp, meta["Z_en"])
        meta["bucket_idx"] = int(bucket_idx)
        for tok in meta["span_tokens"]:
            bucket_vocab[int(bucket_idx)].add(tok)
    return {bucket: sorted(tokens) for bucket, tokens in bucket_vocab.items()}


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
    Lex_fr = m4.M4_LexEN_new(seed=2, D=D)
    pi = rng.permutation(D).astype(np.int64)

    encoded_en = enc_pipeline.encode_corpus_ENC(sentences_en, Lex_en, pi, D, n, seg_seed0=999)
    encoded_fr = enc_pipeline.encode_corpus_ENC(sentences_fr, Lex_fr, pi, D, n, seg_seed0=1999)

    tokens_fr = [enc_pipeline.sentence_to_tokens_EN(sent, vocab=set()) for sent in sentences_fr]
    pairs_mem, span_meta = _build_mem_pairs_with_meta(
        encoded_en,
        encoded_fr,
        tokens_fr,
        win=6,
        stride=3,
        majority="strict",
        max_pairs=10_000,
    )
    assert pairs_mem, "aucune paire MEM générée"

    cfg = mem_pipeline.MemConfig(D=D, B=128, k=12, seed_lsh=10, seed_gmem=11)
    comp = mem_pipeline.make_mem_pipeline(cfg)
    mem_pipeline.train_one_pass_MEM(comp, pairs_mem)

    bucket2vocab = _bucket_vocab(comp, span_meta)
    assert bucket2vocab, "vocabulaire par bucket vide"

    demo = next((m for m in span_meta if m["span_tokens"]), span_meta[0])
    history = list(demo["history_tokens"][-4:])

    rng_demo = np.random.default_rng(4242)
    H_LM = rademacher(D, rng_demo)
    for tok in history:
        H_LM = DD7_updateLM(H_LM, tok, Lex_fr.get, pi)

    G_DEC = rademacher(D, np.random.default_rng(2025))
    G_MEM = comp.Gmem
    prototypes = comp.mem.H.astype(np.int8, copy=False)

    token_star, scores_cand, c_star, C_K, _ = DecodeOneStep(
        Hs=demo["Z_en"],
        H_LM=H_LM,
        history_fr=history,
        G_DEC=G_DEC,
        G_MEM=G_MEM,
        Pi=pi,
        L_fr=Lex_fr.get,
        prototypes=prototypes,
        K=32,
        alpha=1.0,
        beta=1.0,
        ell=max(1, len(history)),
        lam=0.5,
        bucket2vocab=bucket2vocab,
        global_fallback_vocab=sorted({tok for tokens in bucket2vocab.values() for tok in tokens}),
        return_ck_scores=True,
    )

    # Assertions ensure the pipeline runs end-to-end and produces coherent outputs.
    assert isinstance(token_star, str) and token_star
    assert scores_cand.ndim == 1 and scores_cand.size >= 1
    assert int(c_star) in C_K
