"""Train a minimal ENC→MEM→DEC pipeline on an OPUS subset and run inference.

The script mirrors the functional workflow tests but turns them into a small
CLI utility that:
    1. Loads a slice of the OPUS English→French corpus (via `datasets`).
    2. Encodes the parallel sentences with the ENC pipeline.
    3. Trains a MEM bank with simple positional binding of the payloads.
    4. Runs the DEC loop to predict the first few tokens of each FR sentence.

Usage (from project root):
    uv run python -m hdc_project.infer_opus --max-sentences 12 --topk 64
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from hdc_project.encoder import m4, pipeline as enc_pipeline
from hdc_project.encoder.mem import pipeline as mem_pipeline
from hdc_project.encoder.utils import rand_pm1
from hdc_project.decoder import DecodeOneStep, DD7_updateLM
from hdc_project.decoder.dec import (
    DD1_ctx,
    DD2_query_bin,
    DD3_bindToMem,
    DD4_search_topK,
    _as_vocab_from_buckets,
    hd_bind,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper dataclasses / containers
# ---------------------------------------------------------------------------


@dataclass
class PipelineArtifacts:
    D: int
    metas: list[dict[str, object]]
    prototypes: np.ndarray
    bucket2vocab: dict[int, list[str]]
    bucket2vocab_pos: dict[tuple[int, int], list[str]]
    global_vocab: list[str]
    Lex_fr: m4.M4_LexEN
    Pi: np.ndarray
    G_MEM: np.ndarray
    G_DEC: np.ndarray
    freq: Counter[str]
    LM_prior: np.ndarray
    pos_keys: dict[int, np.ndarray]
    pos_key_seed: int


# ---------------------------------------------------------------------------
# Utility helpers mirrored from the functional test
# ---------------------------------------------------------------------------


def _make_pos_lookup(
    pos_keys: dict[int, np.ndarray],
    D: int,
    seed: int | None = None,
) -> callable:
    default = np.ones(D, dtype=np.int8)
    cache: dict[int, np.ndarray] = {pos: key.astype(np.int8, copy=False) for pos, key in pos_keys.items()}

    def lookup(pos: int) -> np.ndarray:
        key = cache.get(pos)
        if key is None:
            if seed is None:
                key = default
            else:
                rng = np.random.default_rng(seed + int(pos))
                key = rand_pm1(1, D, seed=rng.integers(0, 1_000_000))[0]
            cache[pos] = key.astype(np.int8, copy=False)
        return key

    return lookup


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
    acc += prior.astype(np.float32, copy=False) * 0.1

    payload = np.where(acc >= 0.0, 1, -1).astype(np.int8, copy=False)
    # Keep payload aligned with lexical vector to avoid destructive cancellation.
    if int(payload.astype(np.int32) @ vec.astype(np.int32)) < 0:
        payload = (-payload).astype(np.int8, copy=False)
    return payload


# ---------------------------------------------------------------------------
# Pipeline preparation
# ---------------------------------------------------------------------------


def _prepare_opus_pipeline(
    *,
    max_sentences: int,
    N_samples: int,
    D: int,
    n: int,
    ell_window: int,
    alpha: float,
    beta: float,
    pos_key_seed: int,
) -> PipelineArtifacts:
    try:
        ens_raw, frs_raw = enc_pipeline.opus_load_subset(
            name="opus_books",
            config="en-fr",
            split="train",
            N=N_samples,
            seed=2_048,
        )
    except Exception as exc:  # pragma: no cover - dataset availability
        raise RuntimeError(f"OPUS subset unavailable ({exc})") from exc

    ens = [s for s in ens_raw[:max_sentences] if s.strip()]
    frs = [s for s in frs_raw[:max_sentences] if s.strip()]
    if not ens or not frs:
        raise RuntimeError("Empty OPUS sample")

    Lex_en = m4.M4_LexEN_new(seed=101, D=D)
    Lex_fr = m4.M4_LexEN_new(seed=202, D=D)
    rng = np.random.default_rng(777)
    Pi = rng.permutation(D).astype(np.int64)
    G_DEC = rand_pm1(1, D, seed=8_888)[0].astype(np.int8, copy=False)

    encoded_en = enc_pipeline.encode_corpus_ENC(ens, Lex_en, Pi, D, n, seg_seed0=9_991)
    _ = enc_pipeline.encode_corpus_ENC(frs, Lex_fr, Pi, D, n, seg_seed0=9_992)

    tokens_fr_raw = [enc_pipeline.sentence_to_tokens_EN(sent, vocab=set()) for sent in frs]
    tokens_fr: list[list[str]] = []
    freq: Counter[str] = Counter()
    for seq in tokens_fr_raw:
        content = [tok for tok in seq if tok and not tok.startswith("__sent_marker_")]
        tokens_fr.append(content)
        freq.update(content)

    if freq:
        acc_prior = np.zeros(D, dtype=np.int32)
        for tok, count in freq.items():
            vec = Lex_fr.get(tok).astype(np.int8, copy=False)
            acc_prior += vec.astype(np.int32, copy=False) * int(count)
        lexical_prior = np.where(acc_prior >= 0, 1, -1).astype(np.int8, copy=False)
    else:
        lexical_prior = rand_pm1(1, D, seed=4_242)[0].astype(np.int8, copy=False)

    pos_keys: dict[int, np.ndarray] = {}

    def pos_key(pos: int) -> np.ndarray:
        key = pos_keys.get(pos)
        if key is None:
            rng_pos = np.random.default_rng(pos_key_seed + int(pos))
            key = rand_pm1(1, D, seed=rng_pos.integers(0, 1_000_000))[0]
            pos_keys[pos] = key.astype(np.int8, copy=False)
        return key

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    metas: list[dict[str, object]] = []

    for idx, encoded in enumerate(encoded_en):
        Z_en = enc_pipeline.content_signature_from_Xseq(encoded["X_seq"], majority="strict")
        content_tokens = tokens_fr[idx]
        meta: dict[str, object] = {
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
                    Pi,
                    alpha=alpha,
                    beta=beta,
                    ell=ell_window,
                )
                Rt_pos = hd_bind(Rt, pos_key(pos))
                pairs.append((Rt_pos, payload))
                history_sent.append(tok)
                if len(history_sent) > ell_window:
                    history_sent = history_sent[-ell_window:]
        else:
            Rt = DD2_query_bin(
                Qs,
                [],
                Lex_fr.get,
                Pi,
                alpha=alpha,
                beta=beta,
                ell=ell_window,
            )
            Rt_pos = hd_bind(Rt, pos_key(0))
            pairs.append((Rt_pos, lexical_prior.astype(np.int8, copy=True)))
        metas.append(meta)

    if not pairs:
        raise RuntimeError("No aligned EN/FR pairs generated for MEM training")

    cfg = mem_pipeline.MemConfig(D=D, B=256, k=12, seed_lsh=11, seed_gmem=13)
    comp = mem_pipeline.make_mem_pipeline(cfg)
    mem_pipeline.train_one_pass_MEM(comp, pairs)

    prototypes = np.where(comp.mem.M >= 0, 1, -1).astype(np.int8, copy=False)
    comp.mem.H[:, :] = prototypes

    bucket2vocab_pos: dict[tuple[int, int], set[str]] = defaultdict(set)
    bucket2vocab_default: dict[int, set[str]] = defaultdict(set)
    pos_lookup = _make_pos_lookup(pos_keys, D, pos_key_seed)
    for meta in metas:
        tokens = meta["content_tokens"]
        history_sent = []
        buckets_for_sentence: list[int] = []
        Qs = DD1_ctx(meta["Z_en"], G_DEC)
        if tokens:
            for pos, tok in enumerate(tokens):
                Rt = DD2_query_bin(
                    Qs,
                    history_sent,
                    Lex_fr.get,
                    Pi,
                    alpha=alpha,
                    beta=beta,
                    ell=ell_window,
                )
                Rt_pos = hd_bind(Rt, pos_lookup(pos))
                Rt_tilde = DD3_bindToMem(Rt_pos, comp.Gmem)
                bucket_int = int(DD4_search_topK(Rt_tilde, prototypes, 1)[0])
                if pos == 0:
                    meta["bucket"] = bucket_int
                buckets_for_sentence.append(bucket_int)
                bucket2vocab_pos[(bucket_int, pos)].add(tok)
                bucket2vocab_default[bucket_int].add(tok)
                history_sent.append(tok)
                if len(history_sent) > ell_window:
                    history_sent = history_sent[-ell_window:]
        else:
            Rt = DD2_query_bin(Qs, [], Lex_fr.get, Pi, alpha=alpha, beta=beta, ell=ell_window)
            Rt_pos = hd_bind(Rt, pos_lookup(0))
            Rt_tilde = DD3_bindToMem(Rt_pos, comp.Gmem)
            meta["bucket"] = int(DD4_search_topK(Rt_tilde, prototypes, 1)[0])

    bucket2vocab = {bucket: sorted(tokens) for bucket, tokens in bucket2vocab_default.items()}
    bucket2vocab_pos_lists = {
        (bucket, pos): sorted(tokens) for (bucket, pos), tokens in bucket2vocab_pos.items()
    }

    global_vocab = sorted(freq.keys())
    if not global_vocab:
        raise RuntimeError("Empty FR vocabulary extracted from OPUS subset")

    return PipelineArtifacts(
        D=D,
        metas=metas,
        prototypes=prototypes,
        bucket2vocab=bucket2vocab,
        bucket2vocab_pos=bucket2vocab_pos_lists,
        global_vocab=global_vocab,
        Lex_fr=Lex_fr,
        Pi=Pi,
        G_MEM=comp.Gmem,
        G_DEC=G_DEC,
        freq=freq,
        LM_prior=lexical_prior.astype(np.int8, copy=False),
        pos_keys={pos: key.astype(np.int8, copy=False) for pos, key in pos_keys.items()},
        pos_key_seed=pos_key_seed,
    )


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------


def _bucket_vocab_lookup_factory(
    default_map: dict[int, list[str]],
    pos_map: dict[tuple[int, int], list[str]],
) -> callable:
    def lookup(bucket: int, position: int | None = None) -> list[str]:
        if position is not None:
            tokens = pos_map.get((int(bucket), int(position)))
            if tokens:
                return tokens
        return default_map.get(int(bucket), [])

    return lookup


def run_inference(
    pipeline: PipelineArtifacts,
    *,
    max_sentences: int,
    max_steps: int,
    topk: int,
    alpha: float,
    beta: float,
    ell_window: int,
) -> None:
    pos_lookup = _make_pos_lookup(pipeline.pos_keys, pipeline.D, pipeline.pos_key_seed)
    bucket_vocab_lookup = _bucket_vocab_lookup_factory(pipeline.bucket2vocab, pipeline.bucket2vocab_pos)

    hits = 0
    total = 0
    print("\n=== OPUS EN→FR inference preview ===")
    for idx, meta in enumerate(pipeline.metas[:max_sentences], start=1):
        ref_tokens = list(meta["content_tokens"])
        if not ref_tokens:
            continue
        history: list[str] = []
        H_LM = pipeline.LM_prior.copy()
        decoded: list[str] = []
        steps = min(len(ref_tokens), max_steps)
        for step in range(steps):
            token_star, _, _, C_K, _ = DecodeOneStep(
                Hs=meta["Z_en"],
                H_LM=H_LM,
                history_fr=history,
                G_DEC=pipeline.G_DEC,
                G_MEM=pipeline.G_MEM,
                Pi=pipeline.Pi,
                L_fr=pipeline.Lex_fr.get,
                prototypes=pipeline.prototypes,
                K=topk,
                alpha=alpha,
                beta=beta,
                ell=ell_window,
                lam=0.2,
                bucket2vocab=bucket_vocab_lookup,
                global_fallback_vocab=pipeline.global_vocab,
                pos_key_lookup=pos_lookup,
                return_ck_scores=True,
            )
            decoded.append(token_star)
            history.append(token_star)
            if len(history) > ell_window:
                history = history[-ell_window:]
            H_LM = DD7_updateLM(H_LM, token_star, pipeline.Lex_fr.get, pipeline.Pi)

        ref_slice = ref_tokens[:steps]
        step_hits = sum(int(p == r) for p, r in zip(decoded, ref_slice))
        hits += step_hits
        total += steps
        print(f"\nSentence {idx}")
        print("  EN:", ens := meta["tokens_raw"])
        print("  REF:", " ".join(ref_slice))
        print("  DEC:", " ".join(decoded))
        print(f"  Match {step_hits}/{steps}")

    if total:
        print(f"\nToken accuracy over preview: {hits}/{total} ({hits / total:.2%})")
    else:
        print("\nNo tokens decoded (empty sample).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sentences", type=int, default=12, help="Number of OPUS sentences to use.")
    parser.add_argument("--samples", type=int, default=64, help="Sample size drawn from OPUS before filtering.")
    parser.add_argument("--dim", type=int, default=2048, help="Hypervector dimensionality.")
    parser.add_argument("--ngram", type=int, default=3, help="n-gram size used by the encoder pipeline.")
    parser.add_argument("--ell", type=int, default=4, help="Language-model history window (ell).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight of ENC query in DD2.")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight of LM history in DD2.")
    parser.add_argument(
        "--pos-key-seed",
        type=int,
        default=12_345,
        help="Seed used to derive deterministic positional binding keys.",
    )
    parser.add_argument("--max-steps", type=int, default=10, help="Number of decoding steps per sentence.")
    parser.add_argument("--topk", type=int, default=64, help="Number of MEM buckets considered at decode time.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    try:
        pipeline = _prepare_opus_pipeline(
            max_sentences=args.max_sentences,
            N_samples=args.samples,
            D=args.dim,
            n=args.ngram,
            ell_window=args.ell,
            alpha=args.alpha,
            beta=args.beta,
            pos_key_seed=args.pos_key_seed,
        )
    except RuntimeError as exc:
        log.error(str(exc))
        return

    run_inference(
        pipeline,
        max_sentences=args.max_sentences,
        max_steps=args.max_steps,
        topk=args.topk,
        alpha=args.alpha,
        beta=args.beta,
        ell_window=args.ell,
    )


if __name__ == "__main__":
    main()
