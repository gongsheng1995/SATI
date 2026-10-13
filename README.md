# SATI Reproduction Materials Package

Assembled 2026-10-10; updated 2026-10-13 to cover every reported table.
Contains the small per-run artifacts needed to verify each reported number without
redistributing benchmark inputs or trained weights.

## Contents

- `runs/<queue>/<run_name>/` — 92 completed audited runs, each with:
  - `config.json` — frozen per-run protocol (all hyperparameters, seeds, paths)
  - `test_predictions.jsonl` — per-example test predictions (IDs, labels, probabilities, argmax)
  - `test_results.json` — saved metrics
  - `prediction_validation.json` — reload-and-recompute validation receipt
  - `history.json` — per-epoch training/dev curves (checkpoint-selection audit)
  - `resource_usage_training.json` / `resource_usage_export.json` — completion receipts
    for the architecture-baseline queues (where applicable)
- Queue directories:
  - `runs/reruns/`, `runs/followup/` — core same-predictor conditions, supervision controls
    (seeds 42–44 each), text-only conditions, and coefficient/rank sensitivity (2026-10-10 set)
  - `runs/architecture_mintrec2/` — the five MIntRec2.0 architecture baselines
    (MISA, MulT, MAG-BERT, TCL-MAP, MIntOOD), three optimization seeds each
  - `runs/journal_mintrec1_misa_v1/`, `runs/journal_mintrec1_baselines_v1/` — the
    MIntRec (20-class) architecture comparison, three seeds per method
- `priors/` — the audited MIntRec adapted prior (`train_prior.jsonl`, 1,334 strict-JSON
  records; SHA-256 7ddbe6f1ff0ab33bfbfaa414c2dd2f35f81d8eff4740540deb8df0c3fb06a7a7)
  and its generation manifest (pinned Qwen/Qwen2.5-7B-Instruct revision
  a09a35458c702b33eeacc393d103063234e8bc28, frozen prompts, raw responses).
- `aggregations/` — three-seed aggregation records for the MIntRec comparison,
  the independently audited MIntRec1 baseline audit record
  (`MIntRec1_architecture_baselines_audit.json`), and `six_metric_results.json` —
  the canonical six-metric recomputation (ACC/MF1/P/R/WF1/WP, fixed class universes)
  from which the manuscript tables are derived.
- `history/aggregations/` — original source receipts and progress snapshots.
- `SHA256SUMS.json` — SHA-256 of every content file (`.git` and tool directories excluded).

## Verification procedure

For any run, recompute ACC / weighted F1 / fixed-universe macro F1 directly from
`test_predictions.jsonl` (argmax of `probabilities`; label mapping frozen in
`config.json:label2id`) and compare with `test_results.json`. Every reported table,
including the main architecture comparison on both datasets, derives from these files
and from `aggregations/six_metric_results.json`. The analysis universe is fixed at 30
classes for MIntRec2.0 (28 observed on the debiased split) and 20 classes for MIntRec.

Path note: `aggregations/six_metric_results.json` references workspace-relative paths
such as `results/second_review/...`; within this package the same runs live under
`runs/` with identical run-directory names. One deliberate rename:
`results/misa_review/original_native_seed42` appears here as
`runs/architecture_mintrec2/misa_original_native_seed42`.

## Excluded by design

- Trained checkpoint weights (not redistributed).
- Original benchmark inputs and third-party features (available from the official
  repositories and Zenodo DOI listed in the manuscript).
- The historical MIntRec2.0 prior cache is documented in the manuscript but its full
  generation provenance does not exist; the audited MIntRec prior here is fully traceable.
