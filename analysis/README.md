# Diagnostic analysis scripts and result records

This directory supplies the reproducibility chain for the manuscript's
diagnostic analyses that consume external human annotations or released model
probabilities, complementing the saved-prediction verification entry point in
`verification/`. No GPU work or retraining is required to rerun any of these.

## Scripts and the results they produce

| Script | Result record | Manuscript analysis |
|---|---|---|
| `scripts/analyze_dialogue_cluster_controls.py` | `results/dialogue_cluster_control_analysis.json` | Dialogue-cluster bootstrap sensitivity of the paired control contrasts (5,000 draws) |
| `scripts/audit_dialogue_split_membership.py` | `results/dialogue_split_membership_audit.json` | ID-level train/dev/test split disjointness audit |
| `scripts/audit_priors_against_human.py` | `results/human_prior_audit.json`, `results/human_per_class.json` | SATI-mask vs. public human modality-annotation audit incl. 10,000-permutation nulls |
| `scripts/audit_loss_selector.py` | `results/loss_selector_audit.json` | External Delta-NLL selector diagnostic on the same annotated subset |
| `scripts/disagreement_strata_analysis.py` | `results/disagreement_strata_results.json` | Held-out prior-stratum descriptive comparison of SATI vs. all-sample |
| `scripts/analyze_prior_structure.py` | (prints; numbers quoted in the manuscript) | Prior coverage, entropy decomposition, total-variation distances |
| `scripts/analyze_review_results.py`, `scripts/analyze_second_review_results.py` | — | Local module dependencies of the dialogue-cluster script |

## How to run each script

Working-directory convention: all commands below assume the workspace layout
described under "Path conventions" (a root containing `SATI 9/` and
`paper_work/`). Python 3.12 with `numpy` is required; no GPU is used.

| Script | Invocation | Inputs | Output |
|---|---|---|---|
| `audit_priors_against_human.py` | `python audit_priors_against_human.py` (defaults) or `--prior --human_dir --out --seed --permutations` | SATI prior archive (`--prior`, maps to the package's historical prior under `classifier_inputs/data/`); public human annotations (`--human_dir`, external download) | `human_prior_audit.json` (seed 20261006, 10,000 permutations by default) |
| `audit_loss_selector.py` | `python audit_loss_selector.py` (paths fixed at module top: `ROOT/HUMAN/DATA/PRIOR/OUT`) | released Phase-2 MulT/BERT probabilities (external download); prior archive; official test order | `loss_selector_audit.json` |
| `audit_dialogue_split_membership.py` | `python audit_dialogue_split_membership.py` (paths relative to the code root) | official train/dev/test text tables (external download, not redistributed) | `dialogue_split_membership_audit.json` |
| `analyze_dialogue_cluster_controls.py` | `python analyze_dialogue_cluster_controls.py` (no arguments) | `results/second_review/combined_review_analysis.json` (the audited 36-run report, included in `results/` here); official `test.tsv`; per-run prediction directories under `results/reruns/` and `results/second_review/` (package: `runs/`) | `dialogue_cluster_control_analysis.json` (5,000 paired dialogue/seed bootstrap draws) |
| `disagreement_strata_analysis.py` | `python disagreement_strata_analysis.py` | prior archive; three-seed SATI and all-sample predictions under `results/reruns/` | `disagreement_strata_results.json` |
| `analyze_prior_structure.py` | `python analyze_prior_structure.py` (prints statistics) | prior archive | entropy decomposition, per-class rates and total-variation distances quoted in the manuscript |

Reproduction note: the result records shipped in `results/` were produced with
the seeds and input revisions recorded inside each JSON; rerunning after
obtaining the external files reproduces them deterministically. The scripts
verify prediction SHA-256 values against the combined report before
resampling, so substituted inputs fail loudly.

## External inputs (not redistributed)

- The modality-bias study's released files: adjudicated human modality
  annotations (`Human-Annotation/MIntRec2.0/merged_test.tsv`) and Phase-2
  automated MulT/BERT per-example probabilities. Obtain them from the official
  repository of that study; the scripts match records by dialogue and
  utterance identifiers.
- The official MIntRec2.0 test text table used for dialogue clustering.
- The SATI prior archive and per-run test predictions, both included in this
  package (`classifier_inputs/`, `priors/`, `runs/`).

## Path conventions

The scripts were executed from a workspace in which the code repository was
`SATI 9/` and working materials lived under `paper_work/`; scripts that define
`ROOT = Path(__file__).resolve().parents[1]` resolved one level above their
original `scripts/` location. When rerunning from this package, map
`SATI 9/results/...` to `runs/...`, and `SATI 9/data/llm_prior.jsonl` to the
historical prior archive under `classifier_inputs/data/` (see
`classifier_inputs/mask_manifest.json`). One packaged copy
(`disagreement_strata_analysis.py`) had absolute workspace paths rewritten to
the same relative conventions; no analysis logic was changed.

## Seeds, universes and sampling units

Permutation audits use 10,000 shuffles and the dialogue-cluster bootstrap uses
5,000 draws; the concrete seeds are recorded inside the corresponding result
JSONs. All metrics use the fixed 30-class MIntRec2.0 universe. Sampling units
are test utterances grouped by dialogue for the cluster analysis; the strata
analysis groups held-out examples by cached prior judgment.
