"""Compare cached SATI modality masks with the public human modality annotations.

The human labels describe a minimal sufficient modality set; SATI prompts ask
whether a modality would help. This is a criterion-mismatch audit, not a gold
label validation of causal modality necessity. No model training is performed.
"""
import argparse, csv, json, math, random, hashlib, subprocess
from collections import Counter, defaultdict
from pathlib import Path


def read_tsv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def summarize(y, pred):
    tp = sum(a and b for a, b in zip(y, pred))
    fp = sum((not a) and b for a, b in zip(y, pred))
    tn = sum((not a) and (not b) for a, b in zip(y, pred))
    fn = sum(a and (not b) for a, b in zip(y, pred))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    acc = (tp + tn) / len(y) if y else 0.0
    pe = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (len(y) ** 2) if y else 0.0
    kappa = (acc - pe) / (1 - pe) if 1 - pe else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "human_positive": sum(y), "mask_positive": sum(pred),
            "accuracy": acc, "precision": p, "recall": r, "f1": f,
            "cohen_kappa": kappa}


def wilson(k, n, z=1.959963984540054):
    if not n:
        return [None, None]
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [c-h, c+h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", default="SATI 9/data/llm_prior.jsonl")
    ap.add_argument("--human_dir", default="paper_work/modality_bias_repo/Dataset/Human-Annotation/MIntRec2.0")
    ap.add_argument("--out", default="paper_work/human_prior_audit.json")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--permutations", type=int, default=10000)
    args = ap.parse_args()
    prior = {r["id"].lower(): r for r in (json.loads(s) for s in open(args.prior, encoding="utf-8") if s.strip())}
    split_files = {"train": "merged_train.tsv", "dev": "merged_dev.tsv", "test": "merged_test.tsv"}
    human_root = Path(args.human_dir)
    repo_root = human_root.parents[2]
    try:
        revision = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        revision = "unknown"
    def sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    summary = {"source": "Text-Takes-Over-EMNLP-2025/MultiModal-Intent-EMNLP-2025, Dataset/Human-Annotation/MIntRec2.0",
               "source_repo_revision": revision,
               "human_test_file_sha256": sha256(human_root / "merged_test.tsv"),
               "prior_file_sha256": sha256(args.prior),
               "criteria": "Human annotation is the smallest sufficient modality combination; SATI asks whether a modality would help. Agreement is descriptive, not causal necessity accuracy.",
               "seed": args.seed, "permutations": args.permutations, "splits": {}}
    matched = {}
    for split, filename in split_files.items():
        human_rows = read_tsv(Path(args.human_dir) / filename)
        use = []
        excluded = Counter()
        for h in human_rows:
            label = (h.get("Modality") or "").strip()
            if not label or label.lower() == "na":
                excluded[label or "blank"] += 1
                continue
            key = f"dia{h['Dialogue_id']}_utt{h['Utterance_id']}".lower()
            if key not in prior:
                excluded["unmatched_id"] += 1
                continue
            p = prior[key]
            tokens = set(label.split("+"))
            use.append({"id": key, "intent": p["intent"], "label": label,
                        "ha": "Audio" in tokens, "hv": "Video" in tokens,
                        "pa": bool(p["audio_necessary"]), "pv": bool(p["video_necessary"])})
        matched[split] = use
        report = {"raw_human_rows": len(human_rows), "matched_valid_rows": len(use),
                  "excluded": dict(excluded), "human_label_counts": dict(Counter(x["label"] for x in use)),
                  "audio": summarize([x["ha"] for x in use], [x["pa"] for x in use]),
                  "video": summarize([x["hv"] for x in use], [x["pv"] for x in use]),
                  "joint_exact_match": sum(x["ha"] == x["pa"] and x["hv"] == x["pv"] for x in use) / len(use) if use else None,
                  "joint_all_text_baseline": sum(not x["ha"] and not x["hv"] for x in use) / len(use) if use else None}
        report["audio"]["accuracy_wilson95"] = wilson(report["audio"]["tp"] + report["audio"]["tn"], len(use))
        report["video"]["accuracy_wilson95"] = wilson(report["video"]["tp"] + report["video"]["tn"], len(use))
        report["joint_exact_wilson95"] = wilson(round(report["joint_exact_match"] * len(use)), len(use)) if use else [None, None]
        summary["splits"][split] = report

    # Prespecified independent audit: test split only. Compare the observed
    # joint-mask agreement with shuffles preserving either global density or
    # intent-specific rates, preventing post-hoc tuning on the human test subset.
    test = matched["test"]
    rng = random.Random(args.seed)
    observed = sum(x["ha"] == x["pa"] and x["hv"] == x["pv"] for x in test) / len(test)
    grouped = defaultdict(list)
    for i, x in enumerate(test):
        grouped[x["intent"]].append(i)
    null_global, null_class = [], []
    for _ in range(args.permutations):
        ga, gv = [x["pa"] for x in test], [x["pv"] for x in test]
        rng.shuffle(ga); rng.shuffle(gv)
        null_global.append(sum(x["ha"] == ga[i] and x["hv"] == gv[i] for i, x in enumerate(test)) / len(test))
        ca, cv = [False] * len(test), [False] * len(test)
        for idxs in grouped.values():
            aa = [test[i]["pa"] for i in idxs]; vv = [test[i]["pv"] for i in idxs]
            rng.shuffle(aa); rng.shuffle(vv)
            for j, i in enumerate(idxs): ca[i], cv[i] = aa[j], vv[j]
        null_class.append(sum(x["ha"] == ca[i] and x["hv"] == cv[i] for i, x in enumerate(test)) / len(test))
    def null_report(values):
        greater = sum(v >= observed for v in values)
        return {"mean": sum(values)/len(values), "sd": (sum((x-sum(values)/len(values))**2 for x in values)/(len(values)-1))**0.5,
                "q025": sorted(values)[int(.025*len(values))], "q975": sorted(values)[int(.975*len(values))-1],
                "one_sided_mc_p_ge_observed": (greater+1)/(len(values)+1)}
    summary["test_split_permutation_audit"] = {"observed_joint_exact_match": observed,
        "global_density_matched_shuffle": null_report(null_global),
        "within_intent_class_shuffle": null_report(null_class),
        "interpretation": "Tests whether cached sample assignments align with the external minimal-sufficient-modality labels beyond global or within-intent rates; it does not test downstream classifier performance."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

