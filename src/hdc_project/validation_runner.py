from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Sequence

from hdc_project import artifacts_io
from hdc_project.artifacts_types import SerializableArtifacts
from hdc_project.infer_opus import _prepare_opus_pipeline, run_inference
from hdc_project.encoder import m4


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-sentences", type=int, default=5_000)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--dim", type=int, default=4_096)
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--ell", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--topk", type=int, default=96)
    parser.add_argument("--lambda-mem", type=float, default=0.45)
    parser.add_argument("--lambda-bigram", type=float, default=0.3)
    parser.add_argument("--repeat-penalty", type=float, default=0.9)
    parser.add_argument("--teacher-forcing", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=73)
    parser.add_argument("--mem-base-buckets", type=int, default=256)
    parser.add_argument("--mem-target-load", type=float, default=24.0)
    parser.add_argument("--mem-max-buckets", type=int, default=16_384)
    parser.add_argument("--mem-hash-bits", type=int, default=12)
    parser.add_argument("--mem-alert-load", type=float, default=0.85)
    parser.add_argument("--mem-alert-bucket", type=int, default=128)
    parser.add_argument("--lexicon-samples", type=int, default=2_000)


def _normalise_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    normalised: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            normalised[key] = None
        else:
            normalised[key] = value
    return normalised


def _train_and_report(args: argparse.Namespace) -> tuple[Any, Any, Dict[str, Any]]:
    logging.info("Starting training phase.")
    pipeline = _prepare_opus_pipeline(
        max_sentences=args.max_sentences,
        N_samples=args.samples,
        D=args.dim,
        n=args.ngram,
        ell_window=args.ell,
        alpha=args.alpha,
        beta=args.beta,
        pos_key_seed=12_345,
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
    )

    logging.info("Running validation inference on freshly trained pipeline.")
    metrics_val = run_inference(
        pipeline,
        split="val",
        max_sentences=args.max_sentences,
        max_steps=args.ell,
        topk=args.topk,
        alpha=args.alpha,
        beta=args.beta,
        ell_window=args.ell,
        lambda_mem=args.lambda_mem,
        repeat_penalty=args.repeat_penalty,
        teacher_forcing=args.teacher_forcing,
        lambda_bigram=args.lambda_bigram,
    )
    metrics_test = run_inference(
        pipeline,
        split="test",
        max_sentences=args.max_sentences,
        max_steps=args.ell,
        topk=args.topk,
        alpha=args.alpha,
        beta=args.beta,
        ell_window=args.ell,
        lambda_mem=args.lambda_mem,
        repeat_penalty=args.repeat_penalty,
        teacher_forcing=args.teacher_forcing,
        lambda_bigram=args.lambda_bigram,
    )
    return pipeline, metrics_val, {"val": metrics_val, "test": metrics_test}


def _load_and_report(
    args: argparse.Namespace,
    model: SerializableArtifacts,
    lex: m4.M4_LexEN,
) -> Dict[str, Any]:
    logging.info("Rebuilding pipeline from persisted artefacts.")
    pipeline_loaded = _prepare_opus_pipeline(
        max_sentences=args.max_sentences,
        N_samples=args.samples,
        D=model.D,
        n=args.ngram,
        ell_window=args.ell,
        alpha=args.alpha,
        beta=args.beta,
        pos_key_seed=model.pos_key_seed,
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
        preloaded_model=model,
        preloaded_lex=lex,
    )

    logging.info("Evaluating reloaded pipeline on validation and test splits.")
    metrics_val = run_inference(
        pipeline_loaded,
        split="val",
        max_sentences=args.max_sentences,
        max_steps=args.ell,
        topk=args.topk,
        alpha=args.alpha,
        beta=args.beta,
        ell_window=args.ell,
        lambda_mem=args.lambda_mem,
        repeat_penalty=args.repeat_penalty,
        teacher_forcing=args.teacher_forcing,
        lambda_bigram=args.lambda_bigram,
    )
    metrics_test = run_inference(
        pipeline_loaded,
        split="test",
        max_sentences=args.max_sentences,
        max_steps=args.ell,
        topk=args.topk,
        alpha=args.alpha,
        beta=args.beta,
        ell_window=args.ell,
        lambda_mem=args.lambda_mem,
        repeat_penalty=args.repeat_penalty,
        teacher_forcing=args.teacher_forcing,
        lambda_bigram=args.lambda_bigram,
    )
    return {"val": metrics_val, "test": metrics_test}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train, persist, reload and evaluate the OPUS pipeline.")
    _add_common_args(parser)
    parser.add_argument("--artifacts-prefix", type=str, default="artifacts/opus_run")
    parser.add_argument("--report", type=str, default="artifacts/opus_report.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    logging.info("=== Phase 1: Training & validation ===")
    pipeline, _, trained_metrics = _train_and_report(args)

    artifacts_prefix = Path(args.artifacts_prefix)
    artifacts_prefix.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Persisting artefacts to %s", artifacts_prefix)
    artifacts_io.save_artifacts(artifacts_prefix, pipeline.model, pipeline.Lex_fr)

    logging.info("=== Phase 2: Reload & evaluation ===")
    loaded_model, loaded_lex = artifacts_io.load_artifacts(artifacts_prefix)
    loaded_metrics = _load_and_report(args, loaded_model, loaded_lex)

    report = {
        "artifacts_prefix": str(artifacts_prefix),
        "trained": {split: _normalise_metrics(metrics) for split, metrics in trained_metrics.items()},
        "loaded": {split: _normalise_metrics(metrics) for split, metrics in loaded_metrics.items()},
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    logging.info("Validation report written to %s", report_path)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
