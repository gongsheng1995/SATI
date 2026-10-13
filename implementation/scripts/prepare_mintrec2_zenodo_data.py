"""Build SATI's compact split JSON from the official Zenodo text archive."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", default="data/retrain_inputs/text")
    parser.add_argument("--output", default="data/retrain_inputs/data.json")
    args = parser.parse_args()
    source = Path(args.text_dir)
    result = {}
    all_ids = set()
    summary = {}
    for split in ("train", "dev", "test"):
        path = source / f"{split}.tsv"
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                sid = f"dia{row['Dialogue_id']}_utt{row['Utterance_id']}"
                if sid in all_ids:
                    raise ValueError(f"duplicate sample id across splits: {sid}")
                all_ids.add(sid)
                rows.append({"id": sid, "text": row["Text"], "label": row["Label"]})
        result[split] = rows
        summary[split] = {"n": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    labels = sorted({r["label"] for rows in result.values() for r in rows})
    if len(labels) != 30:
        raise ValueError(f"expected 30 in-scope labels, got {len(labels)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    summary["total_unique_ids"] = len(all_ids)
    summary["labels"] = labels
    summary["output"] = str(output.resolve())
    output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
