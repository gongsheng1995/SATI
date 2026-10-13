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
