# SATI results and reproducibility materials

Updated locally on 2026-10-13. This folder is a publication candidate; completion
of the local package does not confirm synchronization of the public repository.

## Scope

- 98 existing completed runs in `runs/`, including the six original/debiased
  joint-within-class permutation controls under `runs/second_review/`.
- `aggregations/six_metric_results.json`: the 26 canonical conditions used in
  the main result tables, with run prediction hashes, means and sample SDs.
- `classifier_inputs/data/`: byte-identical historical MIntRec2.0 priors,
  all/no-auxiliary and frozen random/joint control masks, and MIntRec masks.
  `classifier_inputs/mask_manifest.json` maps frozen config paths to package paths.
  Historical development/test judgments are included only to audit descriptive
  strata; they must not be consumed by classifier training or checkpoint selection.
- `priors/mintrec1_bf16_v3/`: the successful 1,334-example train-only prior,
  frozen prompt messages, raw generated text/token records, generation reservations,
  model/preparation manifests and original independent audit receipt. No v1/v2
  partial outputs enter this prior. The top-level prior files remain compatibility copies.
- `implementation/`: local SATI training/model/loss and preparation implementation
  snapshot with source hashes. This permits inspection and reuse, not certification
  that every historical experiment binary equals the current snapshot. Baseline
  configs retain their pinned source revisions and adaptation protocols; their
  small pinned vendor source files and baseline adapters/drivers are included,
  with original provenance hashes; upstream licenses and recorded revisions apply.
- `SHA256SUMS.json`: all scientific package content hashes, excluding itself,
  Git/tool/session files and screenshots.
- `analysis/`: diagnostic-analysis scripts and result records that consume external
  human annotations or released model probabilities (dialogue-cluster bootstrap,
  split-membership audit, human-reference audit, loss-based selector audit,
  prior strata and prior-structure statistics), with a provenance README.

## Three different reproduction tasks

1. **Recompute reported scores:** only saved predictions and frozen label maps are
   needed. Use the bundled `verification/recompute_predictions.py`; it verifies
   IDs, probability vectors, six metrics and canonical aggregates. No GPU work.
2. **Retrain a classifier with frozen masks:** obtain official transcripts/features
   from the benchmark sources and the pinned BERT checkpoint; restore the mask
   paths using `classifier_inputs/mask_manifest.json`. Preserve each run's config,
   label order and feature hashes. The code snapshot is inspectable, but rerunning
   all adapted recipes uses the included pinned source snapshots and drivers,
   which retain execution guards/bindings; prepare a new local execution configuration
   rather than editing an immutable historical recipe. Exact cross-hardware equality
   and execution without obtaining the external data/models are not certified.
   Original MIntRec2.0 uses WavLM/Swin ROI features; MIntRec uses official
   wav2vec2/Faster R-CNN ROI features converted losslessly to sequence containers.
3. **Audit the adapted prior generation:** frozen messages, raw responses, parsed
   bits and model hashes are supplied. Downloading Qwen weights or generating a new
   prior is a separate operation. The successful v3 archive is fully auditable;
   historical MIntRec2.0 model versions/captions/raw responses/fallback decisions
   remain unavailable and exact historical regeneration is not claimed.

Masks are weak supervision, not verified modality-necessity labels. A strict JSON
and provenance audit is not human validation of the semantic judgments.

## External assets and paths

Trained weights, original feature archives and pretrained model weights are not
redistributed. Original benchmark inputs and any excerpts in generation records
remain subject to their source licenses. This is a local candidate for author
publication; no upload was performed during this repair.

Official benchmarks: https://github.com/thuiar/MIntRec and
https://github.com/thuiar/MIntRec2.0 ; MIntRec2.0 feature release:
https://doi.org/10.5281/zenodo.8041676 .

Aggregation paths use historical workspace-relative paths; within this package,
resolve each run by its prediction SHA-256 or run-directory name under `runs/`.
The original MISA run `original_native_seed42` is named
`misa_original_native_seed42` in this package. Source scientific records are unchanged.
