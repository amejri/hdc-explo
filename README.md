## HDC OPUS Pipeline

This project trains a minimalist hyperdimensional encoder → memory → decoder stack on
OPUS (EN→FR) and provides tooling to evaluate inference quality, persist artefacts and
re-run validation without re-training.

### Core CLI (`hdc_project.infer_opus`)

```bash
uv run python -m hdc_project.infer_opus \
  --max-sentences 12 \
  --train-frac 0.75 --val-frac 0.15 --test-frac 0.10 \
  --mem-base-buckets 256 --mem-target-load 24 \
  --save-artifacts-prefix artifacts/opus_demo
```

- **Train/val/test split**: controlled via `--train-frac` / `--val-frac` / `--test-frac`.
  Accuracy and perplexity are reported on the evaluation split (`--eval-split`).
- **MEM capacity**: automatically scaled with `--mem-target-load`. Warnings are emitted when
  load-factor or per-bucket occupancy passes the thresholds (`--mem-alert-*`).
- **Vocabulary & LM**: candidate fallbacks and bigram priors are sourced from a held-out
  lexicon (`--lexicon-samples`), so MEM training no longer leaks tokens or n-grams.
- **Persistence**: `--save-artifacts-prefix` serialises prototypes, buckets, permutation keys
  and the FR lexicon (npz + json). Use `--load-artifacts-prefix` to skip MEM training entirely
  and run inference with previously saved artefacts.

### Automated validation runner

The helper orchestrates a full train → save → reload → validate loop and writes a JSON report
for regression tracking.

```bash
uv run python -m hdc_project.validation_runner \
  --artifacts-prefix artifacts/opus_run \
  --report artifacts/opus_report.json --verbose
```

The generated report contains val/test accuracy + perplexity for both the freshly trained
pipeline (`trained`) and the reload pass (`loaded`), making it trivial to chart metrics across
experiments.

### Keeping metrics visible

The JSON time-series produced by `validation_runner` can be versioned or fed into your
dashboarding tool of choice. Each entry includes the split name, token counts, accuracy,
perplexity and the number of missing references so you can track regressions quickly.

### Tests

Two lightweight pytest suites cover LM fallback computation and artefact round-tripping:

```bash
python3 -m pytest tests/test_lm_resources.py tests/test_artifacts_io.py
```

These ensure that held-out vocabulary tokens remain eligible during decoding and that the
serialised artefacts can be restored without loss.
