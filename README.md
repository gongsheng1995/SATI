# SATI Reproduction Materials Package

Assembled 2026-10-10 from the independently audited experiment records of the
SATI manuscript. Contains the small per-run artifacts needed to verify every
reported number without redistributing benchmark inputs or trained weights.

## Contents

- `runs/<queue>/<run_name>/` — for each of the 54 completed audited runs:
  - `config.json` — frozen per-run protocol (all hyperparameters, seeds, paths)
  - `test_predictions.jsonl` — per-example test predictions (IDs, labels,
    probabilities, argmax, fixed-model zeroing diagnostics)
  - `test_results.json` — saved metrics
  - `prediction_validation.json` — reload-and-recompute validation receipt
  - `history.json` — per-epoch training/dev curves (checkpoint-selection audit)
  - `checkpoint_storage_policy.json` — training-completion/policy receipt where applicable
- `priors/` — the audited MIntRec adapted prior (`train_prior.jsonl`, 1,334
  strict-JSON records; SHA-256 7ddbe6f1ff0ab33bfbfaa414c2dd2f35f81d8eff4740540deb8df0c3fb06a7a7)
  and its generation manifest (pinned Qwen/Qwen2.5-7B-Instruct revision
  a09a35458c702b33eeacc393d103063234e8bc28, frozen prompts, raw responses).
- `aggregations/` — three-seed aggregation records for the MIntRec comparison
  (SATI / all-sample / MISA) with per-run audited metrics.
- `SHA256SUMS.json` — SHA-256 of every file in this package.

## Verification procedure

For any run, recompute ACC / weighted F1 / fixed-universe macro F1 directly from
`test_predictions.jsonl` (argmax of `probabilities`; label mapping frozen in
`config.json:label2id`) and compare with `test_results.json`; all reported
tables derive from these files. The analysis universe is fixed at 30 classes for
MIntRec2.0 (28 observed on the debiased split) and 20 classes for MIntRec.

## Excluded by design

- Trained checkpoint weights (not redistributed).
- Original benchmark inputs and third-party features (available from the
  official repositories and Zenodo DOI listed in the manuscript).
- The historical MIntRec2.0 prior cache is documented in the manuscript but its
  full generation provenance does not exist; the audited MIntRec prior here is
  fully traceable.
