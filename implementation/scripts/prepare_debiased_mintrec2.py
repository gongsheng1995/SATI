"""Prepare the released MIntRec2.0 debiased membership using official SATI inputs."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    source_repo = root.parent / "paper_work" / "modality_bias_repo"
    source = source_repo / "Dataset" / "Debiased" / "MIntRec-2"
    original_path = root / "data" / "retrain_inputs" / "data.json"
    original = json.loads(original_path.read_text(encoding="utf-8"))
    out = root / "data" / "debiased_mintrec2"
    out.mkdir(parents=True, exist_ok=True)
    priors = {r["id"] for r in map(json.loads, (root / "data" / "llm_prior.jsonl").read_text(encoding="utf-8").splitlines())}
    result, manifest, seen = {}, {}, set()
    for split in ("train", "dev", "test"):
        official = {r["id"]: r for r in original[split]}
        path = source / f"debiased_{split}.tsv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        result[split] = []
        changed_text = 0
        for row in rows:
            sid = f"dia{row['Dialogue_id']}_utt{row['Utterance_id']}"
            if sid in seen or sid not in official or sid not in priors:
                raise ValueError(f"Duplicate, wrong split, or uncovered prior: {sid}")
            item = official[sid]
            released_label = {"ask_help": "Ask for help", "ask_opinions": "Asking for opinions"}.get(row["Label"].strip(), row["Label"].strip())
            if item["label"].casefold() != released_label.casefold():
                raise ValueError(f"Label mismatch: {sid}")
            changed_text += item["text"] != row["Text"]
            seen.add(sid)
            result[split].append(dict(item))
        manifest[split] = {"n": len(rows), "original_n": len(original[split]),
                           "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "text_strings_differ_from_official": changed_text}
    manifest["source_repo"] = "https://github.com/Text-Takes-Over-EMNLP-2025/MultiModal-Intent-EMNLP-2025"
    manifest["source_revision"] = subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip()
    manifest["protocol"] = "Released debiased split membership; official original text and canonical labels by ID. No examples move across train/dev/test. Same official nonverbal features and cached priors."
    manifest["original_json_sha256"] = hashlib.sha256(original_path.read_bytes()).hexdigest()
    manifest["labels_present"] = sorted({x["label"] for rows in result.values() for x in rows})
    label_list = sorted({x["label"] for rows in original.values() for x in rows})
    manifest["label_universe"] = label_list
    manifest["label_aliases"] = {"ask_help": "Ask for help", "ask_opinions": "Asking for opinions"}
    manifest["unique_ids"] = len(seen)
    result["label_list"] = label_list
    manifest["evaluation"] = "Retain original 30-way classifier and fixed 30-class macro metrics; absent classes have zero support."
    target = out / "data.json"
    target.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    manifest["output_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
