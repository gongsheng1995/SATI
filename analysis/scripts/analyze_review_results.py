"""Audit and aggregate saved review reruns without inferring missing results.

Primary comparisons use paired seeds 42/43/44. Bootstrap intervals resample
seeds and test examples jointly; with three seeds they are approximate,
exploratory uncertainty summaries. No test-set hyperparameter selection.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATCH_KEYS = ("text_encoder", "hidden_dim", "tensor_rank", "dropout", "freeze_bert",
              "label_smoothing", "lambda_rdrop", "lambda_aux", "use_focal",
              "batch_size", "lr", "weight_decay", "epochs", "warmup_ratio",
              "grad_clip", "patience", "use_swa", "swa_start", "num_workers",
              "audio_dim", "video_dim", "num_classes", "label2id")


def metrics_from_codes(codes, weights, classes):
    cm = np.bincount(codes, weights=weights, minlength=classes * classes).reshape(classes, classes)
    support, predicted, correct = cm.sum(1), cm.sum(0), np.diag(cm)
    denom = support + predicted
    f1 = np.divide(2 * correct, denom, out=np.zeros(classes), where=denom > 0)
    n = cm.sum()
    return np.array([correct.sum() / n, np.dot(support, f1) / n, f1.mean()])


def load_run(directory):
    required = ("config.json", "history.json", "test_results.json", "test_predictions.jsonl", "prediction_validation.json")
    if any(not (directory / x).exists() for x in required):
        return None
    config = json.loads((directory / "config.json").read_text())
    saved = json.loads((directory / "test_results.json").read_text())
    history = json.loads((directory / "history.json").read_text())
    records = [json.loads(x) for x in (directory / "test_predictions.jsonl").read_text().splitlines() if x.strip()]
    records.sort(key=lambda r: r["id"])
    ids = [r["id"] for r in records]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Empty or duplicated IDs: {directory}")
    classes = config["num_classes"]
    y = np.array([r["target_id"] for r in records], dtype=int)
    pred = np.array([r["pred_id"] for r in records], dtype=int)
    if np.any(y < 0) or np.any(pred < 0) or np.any(y >= classes) or np.any(pred >= classes):
        raise ValueError(f"Invalid label ID: {directory}")
    codes = y * classes + pred
    actual = metrics_from_codes(codes, np.ones(len(y)), classes)
    expected = np.array([saved["acc"], saved["weighted_f1"], saved["f1"]])
    if not np.allclose(actual, expected, rtol=0, atol=1e-10):
        raise ValueError(f"Predictions do not reproduce saved metrics: {directory}")
    suppression = {}
    for modality in ('audio', 'video'):
        key = f'{modality}_zero_pred_id'
        if not all(key in r for r in records):
            raise ValueError(f'Missing fixed-model suppression predictions: {directory}')
        zero_pred = np.array([r[key] for r in records], dtype=int)
        if np.any(zero_pred < 0) or np.any(zero_pred >= classes):
            raise ValueError(f'Invalid suppression label ID: {directory}')
        zero_metrics = metrics_from_codes(y * classes + zero_pred, np.ones(len(y)), classes)
        full_correct, zero_correct = pred == y, zero_pred == y
        suppression[modality] = {
            'zero_metrics_percent': (100 * zero_metrics).tolist(),
            'full_minus_zero_percentage_points': (100 * (actual - zero_metrics)).tolist(),
            'prediction_changed_count': int(np.sum(pred != zero_pred)),
            'correct_to_incorrect_count': int(np.sum(full_correct & ~zero_correct)),
            'incorrect_to_correct_count': int(np.sum(~full_correct & zero_correct))
        }
    return {"name": directory.name, "config": config, "ids": ids, "y": y, "pred": pred,
            "codes": codes, "metrics": actual, "epochs_completed": len(history["train_loss"]),
            "fixed_model_suppression": suppression,
            "prediction_sha256": hashlib.sha256((directory / "test_predictions.jsonl").read_bytes()).hexdigest()}


def paired_comparison(left, right, repetitions, seed, required_seeds):
    seeds = sorted(set(left) & set(right))
    result = {"paired_seeds": seeds, "required_seeds": list(required_seeds), "per_seed": [],
              "bootstrap_interval": None, "status": "partial"}
    for s in seeds:
        a, b = left[s], right[s]
        if a["ids"] != b["ids"] or not np.array_equal(a["y"], b["y"]):
            raise ValueError("Paired prediction IDs or ground-truth labels differ")
        if any(a["config"][k] != b["config"][k] for k in MATCH_KEYS):
            raise ValueError("A matched comparison has inconsistent hyperparameters")
        ac, bc = a["pred"] == a["y"], b["pred"] == b["y"]
        n10, n01 = int(np.sum(ac & ~bc)), int(np.sum(~ac & bc))
        disagreements = n10 + n01
        p = min(1.0, 2 * sum(math.comb(disagreements, k) for k in range(min(n10, n01) + 1)) / 2**disagreements) if disagreements else 1.0
        result["per_seed"].append({"seed": s, "delta_percentage_points": (100 * (a["metrics"] - b["metrics"])).tolist(),
                                  "left_only_correct": n10, "right_only_correct": n01,
                                  "mcnemar_two_sided_exact_p": p})
    if not seeds:
        return result
    delta = np.array([left[s]["metrics"] - right[s]["metrics"] for s in seeds])
    result["mean_delta_percentage_points"] = (100 * delta.mean(0)).tolist()
    result["sd_delta_percentage_points"] = (100 * delta.std(0, ddof=1)).tolist() if len(seeds) > 1 else None
    if set(seeds) != set(required_seeds):
        return result
    result["status"] = "complete" if len(seeds) >= 3 else "single_seed_control"
    if len(seeds) < 3:
        result["interpretation"] = "Single training/mask seed; no across-seed robustness claim."
        return result
    first = left[seeds[0]]
    for s in seeds:
        if first["ids"] != left[s]["ids"] or any(first["config"][k] != left[s]["config"][k] for k in MATCH_KEYS):
            raise ValueError("Sample indexing or hyperparameters differ across seeds")
    rng = np.random.default_rng(seed)
    n, classes = len(first["ids"]), first["config"]["num_classes"]
    samples = np.empty((repetitions, 3))
    for i in range(repetitions):
        weights = np.bincount(rng.integers(n, size=n), minlength=n)
        chosen = rng.integers(len(seeds), size=len(seeds))
        d = []
        for j in chosen:
            s = seeds[j]
            d.append(metrics_from_codes(left[s]["codes"], weights, classes) - metrics_from_codes(right[s]["codes"], weights, classes))
        samples[i] = np.mean(d, axis=0)
    result["bootstrap_interval"] = {"method": "Paired percentile bootstrap, shared test-example resampling and paired seed resampling",
                                    "repetitions": repetitions, "random_seed": seed,
                                    "lower_95_percentage_points": (100 * np.quantile(samples, .025, axis=0)).tolist(),
                                    "upper_95_percentage_points": (100 * np.quantile(samples, .975, axis=0)).tolist(),
                                    "caution": "Approximate exploratory uncertainty with only three training seeds; not a proof of equivalence or broad superiority."}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    folder = ROOT / "results" / "reruns"
    queue = json.loads((folder / "queue_progress.json").read_text())
    balancing_path = folder / 'balancing_progress.json'
    balancing = json.loads(balancing_path.read_text()) if balancing_path.exists() else {'jobs': []}
    jobs = queue['jobs'] + balancing['jobs']
    runs, groups = {}, {}
    missing = []
    for job in jobs:
        run = load_run(folder / job["name"])
        if run is None:
            missing.append(job["name"])
            continue
        runs[job["name"]] = run
        protocol = "debiased" if job["protocol"] == "debiased" else "original"
        groups.setdefault((protocol, job["condition"]), {})[job["seed"]] = run
    output = {"status": "complete" if not missing else "partial", "complete_runs": len(runs),
              "planned_runs": len(jobs), "missing_runs": missing,
              "complete_core_runs": sum(j['name'] in runs for j in queue['jobs']),
              "planned_core_runs": len(queue['jobs']),
              "complete_balancing_runs": sum(j['name'] in runs for j in balancing['jobs']),
              "planned_balancing_runs": len(balancing['jobs']),
              "metric_order": ["ACC", "WF1", "MF1_fixed_30_classes"],
              "inference_caution": "Bootstrap intervals are exploratory pointwise intervals. Per-seed McNemar p-values are nominal and not adjusted for multiple comparisons; no automatic significance claim is made.",
              "checkpoint_selection": "Best ordinary validation-accuracy checkpoint (best_model.pt), as actually tested by train.py. SWA is tracked separately and does not replace the tested checkpoint.",
              "suppression_caution": "Audio/video feature zeroing at a fixed checkpoint is a diagnostic intervention. It is not a retrained unimodal baseline, a temporal-model comparison, or causal proof of modality necessity.",
              "fixed_model_suppression": [{"run": name, "n_test": len(run['ids']), **run['fixed_model_suppression']} for name, run in sorted(runs.items())],
              "groups": [], "paired_comparisons": [], "human_subset_analysis": []}
    for (protocol, condition), data in sorted(groups.items()):
        seeds = sorted(data)
        matrix = np.array([data[s]["metrics"] for s in seeds])
        output["groups"].append({"protocol": protocol, "condition": condition, "seeds": seeds,
                                 "mean_percent": (100 * matrix.mean(0)).tolist(),
                                 "sample_sd_percent": (100 * matrix.std(0, ddof=1)).tolist() if len(seeds) > 1 else None,
                                 "n_test": len(data[seeds[0]]["ids"]),
                                 "per_seed": [{"seed": s, "metrics_percent": (100 * data[s]["metrics"]).tolist(),
                                               "epochs_completed": data[s]["epochs_completed"],
                                               "prediction_sha256": data[s]["prediction_sha256"]} for s in seeds]})
    for protocol in ("original", "debiased"):
        a = groups.get((protocol, "sati_focal"), {})
        controls = ["all_true_focal", "no_prior_focal"]
        if protocol == "original":
            controls += ["density_matched_focal", "class_conditioned_focal"]
            if balancing['jobs']:
                controls += ['uncertainty_all_true_focal', 'pcgrad_all_true_focal']
        for index, control in enumerate(controls):
            result = paired_comparison(a, groups.get((protocol, control), {}), args.repetitions,
                                       args.seed + index + (100 if protocol == "debiased" else 0),
                                       (42,) if control in ("density_matched_focal", "class_conditioned_focal", 'uncertainty_all_true_focal', 'pcgrad_all_true_focal') else (42, 43, 44))
            result.update(protocol=protocol, left="sati_focal", right=control)
            output["paired_comparisons"].append(result)
    human = ROOT.parent / "paper_work" / "modality_bias_repo" / "Dataset" / "Human-Annotation" / "MIntRec2.0" / "merged_test.tsv"
    if human.exists():
        with human.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        annotations = {}
        for row in rows:
            modality = (row.get("Modality") or "").strip()
            if not modality or modality.lower() == "na":
                continue
            sid = f"dia{row['Dialogue_id']}_utt{row['Utterance_id']}"
            annotations[sid] = "minimum_set_contains_nonverbal" if {"Audio", "Video"} & set(modality.split("+")) else "minimum_set_text_only"
        output["human_annotation_sha256"] = hashlib.sha256(human.read_bytes()).hexdigest()
        output["human_subset_caution"] = "Descriptive held-out subgroup evaluation against public adjudicated minimum-sufficiency labels; distinct from the SATI helpfulness prompt. No new human annotations or reconstructed annotator votes."
        for name, run in runs.items():
            if name.startswith("debiased_"):
                continue
            for group in ("minimum_set_contains_nonverbal", "minimum_set_text_only"):
                mask = np.array([annotations.get(sid) == group for sid in run["ids"]])
                if mask.any():
                    value = metrics_from_codes(run["codes"][mask], np.ones(int(mask.sum())), run["config"]["num_classes"])
                    output["human_subset_analysis"].append({"run": name, "group": group, "n": int(mask.sum()), "metrics_percent": (100 * value).tolist()})
    target = folder / "review_analysis.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"status": output["status"], "complete_runs": len(runs), "planned_runs": len(jobs),
                      "complete_core_runs": output['complete_core_runs'],
                      "complete_balancing_runs": output['complete_balancing_runs'], "output": str(target)}))


if __name__ == "__main__":
    main()
