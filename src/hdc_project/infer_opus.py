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
from dataclasses import dataclass, field
from typing import Iterable, Sequence
import math
from tqdm import tqdm

import numpy as np

from hdc_project import artifacts_io
from hdc_project.artifacts_types import PipelineArtifacts, RuntimeData, SerializableArtifacts
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
    permute_pow,
)

log = logging.getLogger(__name__)
LOG_EPS = math.log(np.finfo(np.float64).tiny)


# ---------------------------------------------------------------------------
# Helper dataclasses / containers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Utility helpers mirrored from the functional test
# ---------------------------------------------------------------------------


def _make_pos_lookup(
    base_key: np.ndarray,
    Pi: np.ndarray,
    cache: dict[int, np.ndarray],
) -> callable:
    def lookup(pos: int) -> np.ndarray:
        key = cache.get(pos)
        if key is None:
            key = permute_pow(base_key, Pi, pos)
            cache[pos] = key.astype(np.int8, copy=False)
        return key

    return lookup


def _is_candidate_token(token: str) -> bool:
    if not token:
        return False
    if token.startswith("__") or "_dup" in token:
        return False
    if token in {"--", "--_il", "--_il_me"}:
        return False
    return True


def _logsumexp(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("-inf")
    max_val = float(np.max(arr))
    if math.isinf(max_val):
        return max_val
    shifted = np.exp(arr - max_val, dtype=np.float64)
    return max_val + float(np.log(np.sum(shifted, dtype=np.float64)))


def _make_dynamic_mem_config(
    D: int,
    num_pairs: int,
    *,
    base_buckets: int,
    target_load: float,
    max_buckets: int | None,
    k_bits: int,
    seed_lsh: int,
    seed_gmem: int,
) -> mem_pipeline.MemConfig:
    target_load = max(1.0, float(target_load))
    B = max(base_buckets, 1)
    if num_pairs > 0:
        desired = int(math.ceil(num_pairs / target_load))
        B = max(B, desired)
    if max_buckets is not None:
        B = min(B, max_buckets)
    if B & (B - 1) != 0:
        B = 1 << (B - 1).bit_length()
    return mem_pipeline.MemConfig(D=D, B=int(B), k=int(k_bits), seed_lsh=seed_lsh, seed_gmem=seed_gmem)


def _prepare_lm_resources(
    freq_train: Counter[str],
    bigram_train: dict[str, Counter[str]],
    lexicon_sequences: Iterable[Sequence[str]],
) -> tuple[Counter[str], dict[str, Counter[str]], list[str], list[str], bool]:
    freq_lm: Counter[str] = Counter()
    bigram_counts_lm: dict[str, Counter[str]] = defaultdict(Counter)
    for seq in lexicon_sequences:
        filtered = [
            tok for tok in seq if tok and not tok.startswith("__sent_marker_") and _is_candidate_token(tok)
        ]
        if not filtered:
            continue
        freq_lm.update(filtered)
        for prev, curr in zip(filtered[:-1], filtered[1:]):
            bigram_counts_lm[prev][curr] += 1

    used_fallback = False
    if not freq_lm:
        freq_lm = Counter(freq_train)
        bigram_counts_lm = {k: Counter(v) for k, v in bigram_train.items()}
        used_fallback = True
    else:
        bigram_counts_lm = {k: Counter(v) for k, v in bigram_counts_lm.items()}

    train_vocab = {tok for tok in freq_train if _is_candidate_token(tok)}
    lexicon_vocab = {tok for tok in freq_lm if _is_candidate_token(tok)}
    fallback_vocab = sorted(lexicon_vocab) if lexicon_vocab else sorted(train_vocab)
    global_vocab = sorted(train_vocab.union(lexicon_vocab))
    return freq_lm, bigram_counts_lm, fallback_vocab, global_vocab, used_fallback


def _token_payload_signature(
    token: str,
    L_fr,
    D: int,
    *,
    prior: np.ndarray,
    freq: Counter[str],
    pos_vec: np.ndarray,
    Pi: np.ndarray,
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

    core = hd_bind(vec, pos_vec)
    acc = core.astype(np.float32, copy=False) * float(weight_token)
    if prev_token:
        prev_vec = permute_pow(L_fr(prev_token).astype(np.int8, copy=False), Pi, -1)
        bind_prev = hd_bind(prev_vec, pos_vec)
        acc += bind_prev.astype(np.float32, copy=False)
    if next_token:
        next_vec = permute_pow(L_fr(next_token).astype(np.int8, copy=False), Pi, 1)
        bind_next = hd_bind(next_vec, pos_vec)
        acc += bind_next.astype(np.float32, copy=False)
    acc += hd_bind(prior, pos_vec).astype(np.float32, copy=False) * 0.15

    payload = np.where(acc >= 0.0, 1, -1).astype(np.int8, copy=False)
    if int(payload.astype(np.int32) @ core.astype(np.int32)) < 0:
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
    train_frac: float,
    val_frac: float,
    test_frac: float,
    split_seed: int,
    mem_base_buckets: int,
    mem_target_load: float,
    mem_max_buckets: int | None,
    mem_k_bits: int,
    mem_alert_load: float,
    mem_alert_bucket: int,
    lexicon_samples: int,
    preloaded_model: SerializableArtifacts | None = None,
    preloaded_lex: m4.M4_LexEN | None = None,
) -> PipelineArtifacts:
    log.info(
        "Preparing pipeline (max_sentences=%d, samples=%d, D=%d, train/val/test=%.2f/%.2f/%.2f)",
        max_sentences,
        N_samples,
        D,
        train_frac,
        val_frac,
        test_frac,
    )
    try:
        ens_raw, frs_raw = enc_pipeline.opus_load_subset(
            name="opus_books",
            config="en-fr",
            split="train",
            N=N_samples,
            seed=2_048,
        )
        log.info("Loaded OPUS subset (EN=%d, FR=%d)", len(ens_raw), len(frs_raw))
    except Exception as exc:  # pragma: no cover - dataset availability
        raise RuntimeError(f"OPUS subset unavailable ({exc})") from exc

    ens = [s for s in ens_raw[:max_sentences] if s.strip()]
    frs = [s for s in frs_raw[:max_sentences] if s.strip()]
    if not ens or not frs:
        raise RuntimeError("Empty OPUS sample")

    use_preloaded = preloaded_model is not None and preloaded_lex is not None

    total_pairs = min(len(ens), len(frs))
    if total_pairs < 3:
        raise RuntimeError("Need at least 3 sentence pairs for train/val/test split")

    frac_vec = np.array([train_frac, val_frac, test_frac], dtype=np.float64)
    if np.any(frac_vec < 0):
        raise ValueError("Split fractions must be non-negative")
    if frac_vec.sum() == 0:
        raise ValueError("At least one split fraction must be positive")
    frac_vec = frac_vec / frac_vec.sum()
    rng_split = np.random.default_rng(split_seed)
    permuted_idx = rng_split.permutation(total_pairs)

    counts = np.floor(frac_vec * total_pairs).astype(int)
    remainder = total_pairs - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(frac_vec - counts / max(total_pairs, 1)))
        for idx in order:
            if remainder == 0:
                break
            if frac_vec[idx] > 0:
                counts[idx] += 1
                remainder -= 1

    for idx, frac in enumerate(frac_vec):
        if frac > 0.0 and counts[idx] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] <= 1:
                continue
            counts[donor] -= 1
            counts[idx] += 1

    if counts.sum() != total_pairs:
        raise RuntimeError("Split allocation failed to cover all samples")

    train_count, val_count, test_count = counts.tolist()
    train_idx = permuted_idx[:train_count]
    val_idx = permuted_idx[train_count : train_count + val_count]
    test_idx = permuted_idx[train_count + val_count :]

    split_indices = {
        "train": sorted(int(i) for i in train_idx),
        "val": sorted(int(i) for i in val_idx),
        "test": sorted(int(i) for i in test_idx),
    }
    log.info(
        "Data splits → train=%d | val=%d | test=%d (total=%d)",
        len(split_indices["train"]),
        len(split_indices["val"]),
        len(split_indices["test"]),
        total_pairs,
    )

    idx_to_split = {}
    for split_name, idxs in split_indices.items():
        for idx in idxs:
            idx_to_split[idx] = split_name

    Lex_en = m4.M4_LexEN_new(seed=101, D=D)
    Lex_fr = preloaded_lex if preloaded_lex is not None else m4.M4_LexEN_new(seed=202, D=D)
    if use_preloaded:
        Pi = preloaded_model.Pi.astype(np.int64, copy=False)
        G_DEC = preloaded_model.G_DEC.astype(np.int8, copy=False)
        log.info("Reusing preloaded Pi/G_DEC/Lex_fr artefacts.")
    else:
        rng = np.random.default_rng(777)
        Pi = rng.permutation(D).astype(np.int64)
        G_DEC = rand_pm1(1, D, seed=8_888)[0].astype(np.int8, copy=False)
        log.info("Instantiated new Lex_FR and permutation keys for training.")

    encoded_en = enc_pipeline.encode_corpus_ENC(ens, Lex_en, Pi, D, n, seg_seed0=9_991)
    _ = enc_pipeline.encode_corpus_ENC(frs, Lex_fr, Pi, D, n, seg_seed0=9_992)
    log.info("Encoded corpora: EN traces=%d FR traces=%d", len(encoded_en), len(frs))

    tokens_fr_raw = [enc_pipeline.sentence_to_tokens_EN(sent, vocab=set()) for sent in frs]
    tokens_fr: list[list[str]] = []
    for seq in tokens_fr_raw:
        content = [tok for tok in seq if tok and not tok.startswith("__sent_marker_")]
        tokens_fr.append(content)

    freq_train: Counter[str] = Counter()
    bigram_counts_train: dict[str, Counter[str]] = defaultdict(Counter)
    for idx in split_indices["train"]:
        content = tokens_fr[idx]
        if not content:
            continue
        freq_train.update(content)
        for prev, curr in zip(content[:-1], content[1:]):
            bigram_counts_train[prev][curr] += 1
    bigrams_train_dict = {k: Counter(v) for k, v in bigram_counts_train.items()}
    log.info(
        "Training vocab stats: tokens=%d bigrams=%d avg_len=%.2f",
        len(freq_train),
        len(bigrams_train_dict),
        np.mean([len(tokens_fr[idx]) for idx in split_indices["train"]]) if split_indices["train"] else 0.0,
    )

    if use_preloaded:
        freq_lm = Counter(preloaded_model.freq_lm)
        bigram_counts_lm = {k: Counter(v) for k, v in preloaded_model.bigrams_lm.items()}
        fallback_vocab = list(preloaded_model.fallback_vocab)
        global_vocab = list(preloaded_model.global_vocab)
        used_fallback = False
    else:
        lexicon_sequences = [
            enc_pipeline.sentence_to_tokens_EN(sent, vocab=set())
            for sent in frs_raw[max_sentences : max_sentences + max(0, lexicon_samples)]
            if sent.strip()
        ]
        train_vocab_size = len({tok for tok in freq_train if _is_candidate_token(tok)})
        freq_lm, bigram_counts_lm, fallback_vocab, global_vocab, used_fallback = _prepare_lm_resources(
            freq_train,
            bigrams_train_dict,
            lexicon_sequences,
        )
        if used_fallback:
            log.warning("Held-out lexicon empty; falling back to training vocabulary for LM/fallback.")
        else:
            log.info(
                "Lexicon fallback built from %d held-out sentences (vocab=%d, train_vocab=%d)",
                len(lexicon_sequences),
                len(fallback_vocab),
                train_vocab_size,
            )

    if use_preloaded:
        lexical_prior = preloaded_model.LM_prior.astype(np.int8, copy=False)
    else:
        if freq_train:
            acc_prior = np.zeros(D, dtype=np.int32)
            for tok, count in freq_train.items():
                vec = Lex_fr.get(tok).astype(np.int8, copy=False)
                acc_prior += vec.astype(np.int32, copy=False) * int(count)
            lexical_prior = np.where(acc_prior >= 0, 1, -1).astype(np.int8, copy=False)
        else:
            lexical_prior = rand_pm1(1, D, seed=4_242)[0].astype(np.int8, copy=False)

    if use_preloaded:
        base_pos_key = preloaded_model.pos_base_key.astype(np.int8, copy=False)
        pos_key_seed = preloaded_model.pos_key_seed
    else:
        base_pos_key = rand_pm1(1, D, seed=pos_key_seed)[0].astype(np.int8, copy=False)
    pos_keys: dict[int, np.ndarray] = {}

    def pos_key(pos: int) -> np.ndarray:
        key = pos_keys.get(pos)
        if key is None:
            key = permute_pow(base_pos_key, Pi, pos)
            pos_keys[pos] = key.astype(np.int8, copy=False)
        return key

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    metas: list[dict[str, object]] = []
    split_metas: dict[str, list[dict[str, object]]] = {name: [] for name in ("train", "val", "test")}

    for idx, encoded in enumerate(encoded_en):
        split_name = idx_to_split.get(idx)
        if split_name is None:
            continue
        Z_en = enc_pipeline.content_signature_from_Xseq(encoded["X_seq"], majority="strict")
        content_tokens = tokens_fr[idx]
        meta: dict[str, object] = {
            "Z_en": Z_en,
            "split": split_name,
            "index": idx,
            "en_text": ens[idx],
            "fr_text": frs[idx],
            "fr_tokens_raw": tokens_fr_raw[idx],
            "content_tokens": content_tokens,
        }

        Qs = DD1_ctx(Z_en, G_DEC)
        history_sent: list[str] = []
        if split_name == "train" and not use_preloaded:
            if content_tokens:
                for pos, tok in enumerate(tqdm(content_tokens, desc="Processing tokens", unit="tok")):
                    prev_tok = content_tokens[pos - 1] if pos > 0 else None
                    next_tok = content_tokens[pos + 1] if pos + 1 < len(content_tokens) else None
                    payload = _token_payload_signature(
                        tok,
                        Lex_fr.get,
                        D,
                        prior=lexical_prior,
                        freq=freq_train,
                        pos_vec=pos_key(pos),
                        Pi=Pi,
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
                fallback = hd_bind(lexical_prior.astype(np.int8, copy=True), pos_key(0))
                pairs.append((Rt_pos, fallback))
        metas.append(meta)
        split_metas[split_name].append(meta)

    if not use_preloaded:
        if not pairs:
            raise RuntimeError("No aligned EN/FR pairs generated for MEM training")

        cfg = _make_dynamic_mem_config(
            D=D,
            num_pairs=len(pairs),
            base_buckets=mem_base_buckets,
            target_load=mem_target_load,
            max_buckets=mem_max_buckets,
            k_bits=mem_k_bits,
            seed_lsh=11,
            seed_gmem=13,
        )
        log.info(
            "Training MEM with %d pairs (base=%d target_load=%.2f → B=%d)",
            len(pairs),
            mem_base_buckets,
            mem_target_load,
            cfg.B,
        )
        comp = mem_pipeline.make_mem_pipeline(cfg)
        mem_pipeline.train_one_pass_MEM(comp, pairs)
        log.info("MEM training pass complete; sealing prototypes.")

        counts = comp.mem.n.astype(np.float32, copy=False)
        accum = comp.mem.M.astype(np.float32, copy=False)
        denom = counts[:, None]
        mean = np.divide(accum, denom, out=np.zeros_like(accum), where=denom != 0)
        base_sign = np.where(mean >= 0, 1, -1).astype(np.int8, copy=False)
        prototypes = np.where(mean >= 0.2, 1, np.where(mean <= -0.2, -1, base_sign)).astype(np.int8, copy=False)
        comp.mem.H[:, :] = prototypes

        counts_int = comp.mem.n.astype(np.int32, copy=False)
        non_empty = int(np.count_nonzero(counts_int))
        total_pairs_trained = int(counts_int.sum())
        collision_events = max(0, total_pairs_trained - non_empty)
        load_factor = float(total_pairs_trained / max(cfg.B, 1))
        occupancy_ratio = float(non_empty / max(cfg.B, 1))
        non_zero_counts = counts_int[counts_int > 0]
        avg_bucket_fill = float(non_zero_counts.mean()) if non_empty else 0.0
        max_bucket_fill = int(non_zero_counts.max()) if non_empty else 0
        p95_bucket_fill = float(np.percentile(non_zero_counts, 95)) if non_empty else 0.0
        recommended_topk = int(max(1, min(cfg.B, math.ceil(avg_bucket_fill * 4)) if avg_bucket_fill else 1))
        collision_rate = float(collision_events / max(total_pairs_trained, 1))
        alerts: list[str] = []
        if load_factor > mem_alert_load:
            alerts.append(f"load_factor {load_factor:.2f} exceeds threshold {mem_alert_load:.2f}")
        if mem_alert_bucket > 0 and max_bucket_fill > mem_alert_bucket:
            alerts.append(f"max_bucket_fill {max_bucket_fill} exceeds threshold {mem_alert_bucket}")
        for message in alerts:
            log.warning("MEM saturation alert: %s (B=%d pairs=%d)", message, cfg.B, total_pairs_trained)
        mem_stats = {
            "B": cfg.B,
            "pairs_trained": total_pairs_trained,
            "non_empty_buckets": non_empty,
            "load_factor": load_factor,
            "target_load": float(mem_target_load),
            "occupancy_ratio": occupancy_ratio,
            "avg_bucket_fill": avg_bucket_fill,
            "max_bucket_fill": max_bucket_fill,
            "p95_bucket_fill": p95_bucket_fill,
            "collision_events": collision_events,
            "collision_rate": collision_rate,
            "recommended_topk": recommended_topk,
            "alerts": alerts,
        }
        log.info(
            "MEM stats → load=%.2f occ=%.2f avg_bucket=%.2f max_bucket=%d recommended_topk=%d collisions=%d",
            load_factor,
            occupancy_ratio,
            avg_bucket_fill,
            max_bucket_fill,
            recommended_topk,
            collision_events,
        )
        G_MEM = comp.Gmem
    else:
        prototypes = preloaded_model.prototypes.astype(np.int8, copy=False)
        mem_stats = preloaded_model.mem_stats
        G_MEM = preloaded_model.G_MEM.astype(np.int8, copy=False)
        bucket2vocab = {int(k): list(v) for k, v in preloaded_model.bucket2vocab.items()}
        bucket2vocab_pos_lists = {(int(k[0]), int(k[1])): list(v) for k, v in preloaded_model.bucket2vocab_pos.items()}
        log.info("Loaded MEM artefacts from disk (B=%d)", int(mem_stats.get("B", 0)) if mem_stats else -1)

    if not use_preloaded:
        bucket2vocab_pos_counts: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
        bucket2vocab_default_counts: dict[int, Counter[str]] = defaultdict(Counter)
        pos_lookup = _make_pos_lookup(base_pos_key, Pi, pos_keys)
        for meta in split_metas["train"]:
            tokens = meta["content_tokens"]
            history_sent = []
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
                    bucket2vocab_pos_counts[(bucket_int, pos)][tok] += 1
                    bucket2vocab_default_counts[bucket_int][tok] += 1
                    history_sent.append(tok)
                    if len(history_sent) > ell_window:
                        history_sent = history_sent[-ell_window:]
            else:
                Rt = DD2_query_bin(Qs, [], Lex_fr.get, Pi, alpha=alpha, beta=beta, ell=ell_window)
                Rt_pos = hd_bind(Rt, pos_lookup(0))
                Rt_tilde = DD3_bindToMem(Rt_pos, comp.Gmem)
                meta["bucket"] = int(DD4_search_topK(Rt_tilde, prototypes, 1)[0])

        def _finalize_candidates(counter: Counter[str], limit: int) -> list[str]:
            tokens = [tok for tok, _ in counter.most_common() if _is_candidate_token(tok)]
            return tokens[:limit]

        CAND_LIMIT = 32
        bucket2vocab = {
            bucket: _finalize_candidates(counter, CAND_LIMIT)
            for bucket, counter in bucket2vocab_default_counts.items()
            if counter
        }
        bucket2vocab_pos_lists = {
            (bucket, pos): _finalize_candidates(counter, max(1, CAND_LIMIT // 2))
            for (bucket, pos), counter in bucket2vocab_pos_counts.items()
            if counter
        }
        log.info(
            "Bucket vocab built: buckets=%d pos_entries=%d",
            len(bucket2vocab),
            len(bucket2vocab_pos_lists),
        )

    if not global_vocab:
        raise RuntimeError("Empty FR vocabulary extracted from OPUS subset")

    runtime_data = RuntimeData(metas=metas, split_metas=split_metas)
    mem_stats = mem_stats or {}
    serializable = SerializableArtifacts(
        D=D,
        prototypes=prototypes,
        bucket2vocab=bucket2vocab,
        bucket2vocab_pos=bucket2vocab_pos_lists,
        global_vocab=global_vocab,
        fallback_vocab=fallback_vocab,
        Pi=Pi,
        G_MEM=G_MEM,
        G_DEC=G_DEC,
        LM_prior=lexical_prior.astype(np.int8, copy=False),
        pos_base_key=base_pos_key.astype(np.int8, copy=False),
        pos_key_seed=pos_key_seed,
        mem_stats=mem_stats,
        freq_lm=dict(freq_lm),
        bigrams_lm={k: dict(v) for k, v in bigram_counts_lm.items()},
    )
    log.info(
        "Pipeline ready (global_vocab=%d, fallback_vocab=%d, splits=%s)",
        len(serializable.global_vocab),
        len(serializable.fallback_vocab),
        {k: len(v) for k, v in split_metas.items()},
    )
    return PipelineArtifacts(
        runtime=runtime_data,
        model=serializable,
        Lex_fr=Lex_fr,
        freq_train=freq_train,
        bigrams_train=bigrams_train_dict,
        pos_keys={pos: key.astype(np.int8, copy=False) for pos, key in pos_keys.items()},
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
    split: str,
    max_sentences: int,
    max_steps: int,
    topk: int,
    alpha: float,
    beta: float,
    ell_window: int,
    lambda_mem: float,
    repeat_penalty: float,
    teacher_forcing: int,
    lambda_bigram: float,
) -> dict[str, float]:
    log.info(
        "Running inference on split=%s (max_sentences=%d, max_steps=%d, topk=%d)",
        split,
        max_sentences,
        max_steps,
        topk,
    )
    metas = pipeline.runtime.split_metas.get(split, [])
    if not metas:
        log.warning("No sentences available for split '%s'", split)
        return {
            "split": split,
            "accuracy": 0.0,
            "perplexity": float("inf"),
            "tokens": 0,
            "missing_reference": 0,
            "hits": 0,
            "predictions": 0,
        }

    base_pos_key = pipeline.model.pos_base_key.astype(np.int8, copy=False)
    pos_lookup = _make_pos_lookup(base_pos_key, pipeline.model.Pi, dict(pipeline.pos_keys))
    bucket_vocab_lookup = _bucket_vocab_lookup_factory(pipeline.model.bucket2vocab, pipeline.model.bucket2vocab_pos)
    bigrams = pipeline.model.bigrams_lm if pipeline.model.bigrams_lm else pipeline.bigrams_train
    freq_unigram = Counter(pipeline.model.freq_lm) if pipeline.model.freq_lm else pipeline.freq_train
    fallback_vocab = pipeline.model.fallback_vocab if pipeline.model.fallback_vocab else pipeline.model.global_vocab
    vocab_size = max(1, len(fallback_vocab))
    unigram_total = float(sum(freq_unigram.values()) + vocab_size)

    hits = 0
    total = 0
    log_prob_sum = 0.0
    evaluated_tokens = 0
    missing_reference = 0

    print(f"\n=== OPUS EN→FR inference preview ({split} split) ===")
    rec_topk = int(pipeline.model.mem_stats.get("recommended_topk", 0)) if pipeline.model.mem_stats else 0
    if rec_topk:
        print(f"Suggested MEM top-K (train stats): {rec_topk}")
    for idx, meta in enumerate(metas[:max_sentences], start=1):
        ref_tokens = list(meta["content_tokens"])
        if not ref_tokens:
            continue
        prefill = min(max(0, teacher_forcing), len(ref_tokens))
        history: list[str] = ref_tokens[:prefill]
        H_LM = pipeline.model.LM_prior.copy()
        for forced_tok in history:
            H_LM = DD7_updateLM(H_LM, forced_tok, pipeline.Lex_fr.get, pipeline.model.Pi)

        decoded: list[str] = []
        steps = min(len(ref_tokens), max_steps)
        for step in range(prefill, steps):
            token_star, scores_cand, _, C_K, _ = DecodeOneStep(
                Hs=meta["Z_en"],
                H_LM=H_LM,
                history_fr=history,
                G_DEC=pipeline.model.G_DEC,
                G_MEM=pipeline.model.G_MEM,
                Pi=pipeline.model.Pi,
                L_fr=pipeline.Lex_fr.get,
                prototypes=pipeline.model.prototypes,
                K=topk,
                alpha=alpha,
                beta=beta,
                ell=ell_window,
                lam=lambda_mem,
                bucket2vocab=bucket_vocab_lookup,
                global_fallback_vocab=fallback_vocab,
                pos_key_lookup=pos_lookup,
                return_ck_scores=True,
            )
            cand_vocab = _as_vocab_from_buckets(
                C_K=C_K,
                bucket2vocab=bucket_vocab_lookup,
                history_fr=history,
                global_fallback_vocab=fallback_vocab,
                min_size=1,
                position=step,
            )
            scores_arr = np.asarray(scores_cand, dtype=np.float64)
            adjusted = None
            if scores_arr.size == len(cand_vocab) and len(cand_vocab) > 0:
                penalties = np.array([history.count(tok) * repeat_penalty for tok in cand_vocab], dtype=np.float64)
                adjusted = scores_arr - penalties

                if history:
                    prev_tok = history[-1]
                    prev_counts = bigrams.get(prev_tok, {})
                    denom = float(sum(prev_counts.values()) + vocab_size)
                    bigram_scores = np.array(
                        [np.log((prev_counts.get(tok, 0) + 1.0) / denom) for tok in cand_vocab],
                        dtype=np.float64,
                    )
                else:
                    bigram_scores = np.array(
                        [np.log((freq_unigram.get(tok, 0) + 1.0) / unigram_total) for tok in cand_vocab],
                        dtype=np.float64,
                    )

                adjusted = adjusted + lambda_bigram * bigram_scores
                best_idx = int(np.argmax(adjusted))
                token_star = cand_vocab[best_idx]

            decoded.append(token_star)
            predicted_token = token_star

            if adjusted is not None and len(cand_vocab) > 0:
                log_probs = adjusted - _logsumexp(adjusted)
            else:
                log_probs = None

            ref_token = ref_tokens[step] if step < len(ref_tokens) else None
            if ref_token is not None:
                if log_probs is not None and ref_token in cand_vocab:
                    ref_idx = cand_vocab.index(ref_token)
                    log_prob_sum += float(log_probs[ref_idx])
                    evaluated_tokens += 1
                else:
                    log_prob_sum += LOG_EPS
                    evaluated_tokens += 1
                    missing_reference += 1

            history.append(predicted_token)
            if len(history) > ell_window:
                history = history[-ell_window:]
            H_LM = DD7_updateLM(H_LM, predicted_token, pipeline.Lex_fr.get, pipeline.model.Pi)

        ref_slice = ref_tokens[prefill:steps]
        step_hits = sum(int(p == r) for p, r in zip(decoded, ref_slice))
        predicted_len = max(0, steps - prefill)
        hits += step_hits
        total += predicted_len
        print(f"\nSentence {idx} [{split}]")
        print("  EN:", meta["en_text"])
        print("  REF:", " ".join(ref_tokens[:steps]))
        print("  DEC:", " ".join(history[:prefill] + decoded))
        if predicted_len:
            print(f"  Match {step_hits}/{predicted_len}")
        else:
            print("  Match -- (teacher forcing only)")

    accuracy = hits / total if total else 0.0
    perplexity = math.exp(-log_prob_sum / evaluated_tokens) if evaluated_tokens else float("inf")
    if total:
        print(f"\nToken accuracy over preview ({split}): {hits}/{total} ({accuracy:.2%})")
    else:
        print(f"\nNo tokens decoded on split '{split}'.")
    if evaluated_tokens:
        print(f"Perplexity ({split}): {perplexity:.2f} over {evaluated_tokens} tokens (missing refs: {missing_reference})")

    log.info(
        "Inference summary split=%s accuracy=%.4f perplexity=%s tokens=%d missing=%d",
        split,
        accuracy,
        "inf" if not math.isfinite(perplexity) else f"{perplexity:.3f}",
        evaluated_tokens,
        missing_reference,
    )
    return {
        "split": split,
        "accuracy": accuracy,
        "perplexity": perplexity,
        "tokens": evaluated_tokens,
        "missing_reference": missing_reference,
        "hits": hits,
        "predictions": total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-sentences", type=int, default=8_000, help="Number of OPUS sentences to use.")
    parser.add_argument("--samples", type=int, default=40_000, help="Sample size drawn from OPUS before filtering.")
    parser.add_argument("--dim", type=int, default=8_192, help="Hypervector dimensionality.")
    parser.add_argument("--ngram", type=int, default=3, help="n-gram size used by the encoder pipeline.")
    parser.add_argument("--ell", type=int, default=6, help="Language-model history window (ell).")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight of ENC query in DD2.")
    parser.add_argument("--beta", type=float, default=0.75, help="Weight of LM history in DD2.")
    parser.add_argument(
        "--pos-key-seed",
        type=int,
        default=12_345,
        help="Seed used to derive deterministic positional binding keys.",
    )
    parser.add_argument("--train-frac", type=float, default=0.82, help="Fraction of samples used for MEM/LM training.")
    parser.add_argument("--val-frac", type=float, default=0.09, help="Fraction of samples reserved for validation.")
    parser.add_argument("--test-frac", type=float, default=0.09, help="Fraction of samples reserved for testing.")
    parser.add_argument("--split-seed", type=int, default=73, help="Seed controlling the train/val/test partition.")
    parser.add_argument(
        "--eval-split",
        choices=("train", "val", "test"),
        default="val",
        help="Dataset split used for reporting metrics during inference.",
    )
    parser.add_argument("--max-steps", type=int, default=18, help="Number of decoding steps per sentence.")
    parser.add_argument("--topk", type=int, default=64, help="Number of MEM buckets considered at decode time.")
    parser.add_argument("--lambda-mem", type=float, default=0.6, help="Weight of MEM evidence in DD6.")
    parser.add_argument("--lambda-bigram", type=float, default=0.4, help="Weight of bigram LM log-probabilities during decoding.")
    parser.add_argument("--repeat-penalty", type=float, default=1.05, help="Penalty applied per previous use of a token.")
    parser.add_argument("--teacher-forcing", type=int, default=4, help="Number of reference tokens to seed the history with.")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    parser.add_argument("--mem-base-buckets", type=int, default=512, help="Minimum number of MEM buckets.")
    parser.add_argument("--mem-target-load", type=float, default=16.0, help="Target average items per bucket used to scale MEM capacity.")
    parser.add_argument("--mem-max-buckets", type=int, default=32_768, help="Maximum number of MEM buckets (<=0 disables the cap).")
    parser.add_argument("--mem-hash-bits", type=int, default=14, help="Number of sign-LSH bits used by MEM.")
    parser.add_argument("--mem-alert-load", type=float, default=0.75, help="Alert threshold for global MEM load factor.")
    parser.add_argument("--mem-alert-bucket", type=int, default=192, help="Alert threshold for maximum bucket occupancy.")
    parser.add_argument("--lexicon-samples", type=int, default=4_000, help="Number of held-out OPUS sentences reserved for LM/vocabulary fallback.")
    parser.add_argument("--save-artifacts-prefix", type=str, default=None, help="Prefix used to persist trained encoder/decoder artifacts.")
    parser.add_argument("--load-artifacts-prefix", type=str, default=None, help="Prefix used to load pre-trained encoder/decoder artifacts (skips MEM training).")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    preloaded_model: SerializableArtifacts | None = None
    preloaded_lex: m4.M4_LexEN | None = None
    if args.load_artifacts_prefix:
        try:
            preloaded_model, preloaded_lex = artifacts_io.load_artifacts(args.load_artifacts_prefix)
            log.info("Loaded artifacts from %s", args.load_artifacts_prefix)
        except Exception as exc:  # pragma: no cover - IO guard
            log.error("Failed to load artifacts ('%s'): %s", args.load_artifacts_prefix, exc)
            return

    try:
        log.info("=== Stage: pipeline preparation ===")
        pipeline = _prepare_opus_pipeline(
            max_sentences=args.max_sentences,
            N_samples=args.samples,
            D=args.dim,
            n=args.ngram,
            ell_window=args.ell,
            alpha=args.alpha,
            beta=args.beta,
            pos_key_seed=args.pos_key_seed,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            split_seed=args.split_seed,
            mem_base_buckets=args.mem_base_buckets,
            mem_target_load=args.mem_target_load,
            mem_max_buckets=None if args.mem_max_buckets <= 0 else args.mem_max_buckets,
            mem_k_bits=args.mem_hash_bits,
            mem_alert_load=args.mem_alert_load,
            mem_alert_bucket=args.mem_alert_bucket,
            lexicon_samples=args.lexicon_samples,
            preloaded_model=preloaded_model,
            preloaded_lex=preloaded_lex,
        )
    except RuntimeError as exc:
        log.error(str(exc))
        return

    if args.save_artifacts_prefix:
        try:
            artifacts_io.save_artifacts(args.save_artifacts_prefix, pipeline.model, pipeline.Lex_fr)
            log.info("Saved artifacts to %s", args.save_artifacts_prefix)
        except Exception as exc:  # pragma: no cover - IO guard
            log.error("Failed to save artifacts ('%s'): %s", args.save_artifacts_prefix, exc)

    if pipeline.model.mem_stats:
        mem_stats = pipeline.model.mem_stats
        log.info(
            "MEM stats B=%d pairs=%d load=%.2f target_load=%.2f avg_bucket=%.2f max_bucket=%d collision_rate=%.3f recommended_topk=%d alerts=%s",
            int(mem_stats.get("B", 0)),
            int(mem_stats.get("pairs_trained", 0)),
            float(mem_stats.get("load_factor", 0.0)),
            float(mem_stats.get("target_load", 0.0)),
            float(mem_stats.get("avg_bucket_fill", 0.0)),
            int(mem_stats.get("max_bucket_fill", 0)),
            float(mem_stats.get("collision_rate", 0.0)),
            int(mem_stats.get("recommended_topk", 0)),
            ",".join(mem_stats.get("alerts", [])),
        )

    log.info("=== Stage: inference (%s split) ===", args.eval_split)
    metrics = run_inference(
        pipeline,
        split=args.eval_split,
        max_sentences=args.max_sentences,
        max_steps=args.max_steps,
        topk=args.topk,
        alpha=args.alpha,
        beta=args.beta,
        ell_window=args.ell,
        lambda_mem=args.lambda_mem,
        repeat_penalty=args.repeat_penalty,
        teacher_forcing=args.teacher_forcing,
        lambda_bigram=args.lambda_bigram,
    )
    log.info(
        "Metrics split=%s accuracy=%.4f perplexity=%.3f tokens=%d missing=%d",
        metrics["split"],
        metrics["accuracy"],
        metrics["perplexity"],
        metrics["tokens"],
        metrics["missing_reference"],
    )


if __name__ == "__main__":
    main()
