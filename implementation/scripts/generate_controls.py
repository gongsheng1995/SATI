"""Generate control prior files for the sample-selection analysis.

Controls built from the archived LLM prior file (data/llm_prior.jsonl):

1. class_conditioned_prior.jsonl
   Within each training class, the audio (resp. video) mask assignments are
   randomly permuted among that class's examples. Per-class positive counts,
   class proportions, and global coverage are preserved exactly; only the
   sample-level assignment inside each class changes. Tests whether the
   within-class selection carries information beyond the class label.

2. density_matched_prior.jsonl
   Training masks are re-drawn uniformly without replacement so that the
   global number of selected examples equals the original (audio 4,394,
   video 5,408 over 6,165 training examples). Destroys both class and sample
   structure while preserving the amount of supervision and the batch-level
   normalization behavior.

3. no_prior.jsonl
   All audio/video masks are zero, which disables both auxiliary losses
   (train.py guards each term with ``mask.sum() > 0``). Backfills the mask
   file referenced by, but not archived with, the recorded no-prior run.

4. all_true.jsonl
   All audio/video masks are one. Mirrors the archived all_true prior used
   by the all-sample control.

Dev/test records are unchanged in the two random controls. The no-prior and
all-true controls replace masks across the archive, but only training masks
enter the loss. Post-hoc groups use the original archive, never all-true fields.
All randomness is seeded for reproducibility.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(records):>5} records -> {path}")


def binary_entropy(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def mask_stats(records, field):
    n = len(records)
    pos = sum(1 for r in records if r[field])
    by_class = defaultdict(list)
    for r in records:
        by_class[r["intent"]].append(r[field])
    h = binary_entropy(pos / n)
    h_cond = sum(len(v) / n * binary_entropy(sum(v) / len(v)) for v in by_class.values())
    class_dist = Counter(r["intent"] for r in records)
    sel_dist = Counter(r["intent"] for r in records if r[field])
    tv = 0.5 * sum(abs(sel_dist[c] / pos - class_dist[c] / n) for c in class_dist)
    return {
        "n": n,
        "pos": pos,
        "rate": pos / n,
        "I(M;Y)": h - h_cond,
        "H(M)": h,
        "TV": tv,
    }


def fmt(field, s):
    return (f"  {field:<18} pos={s['pos']:>5}/{s['n']} rate={s['rate']:.4f} "
            f"H={s['H(M)']:.3f} I(M;Y)={s['I(M;Y)']:.3f} TV={s['TV']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/llm_prior.jsonl")
    ap.add_argument("--out_dir", default="data/controls")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_records(args.input)
    train_idx = [i for i, r in enumerate(records) if r["split"] == "train"]
    train_n = len(train_idx)
    print(f"loaded {len(records)} records ({train_n} train) from {args.input}")

    print("original training-mask statistics:")
    print(fmt("audio_necessary", mask_stats([records[i] for i in train_idx], "audio_necessary")))
    print(fmt("video_necessary", mask_stats([records[i] for i in train_idx], "video_necessary")))

    rng = random.Random(args.seed)

    # ---- 1. class-conditioned shuffle (train only, per class, per modality)
    cc = [dict(r) for r in records]
    for field in ("audio_necessary", "video_necessary"):
        by_class = defaultdict(list)
        for i in train_idx:
            by_class[records[i]["intent"]].append(i)
        for intent, idxs in by_class.items():
            values = [records[i][field] for i in idxs]
            rng.shuffle(values)
            for i, v in zip(idxs, values):
                cc[i][field] = v
    for i in train_idx:
        cc[i]["source"] = "class_conditioned"
    write_records(cc, Path(args.out_dir) / "class_conditioned_prior.jsonl")

    # ---- 2. density-matched random selection (train only, exact counts)
    dm = [dict(r) for r in records]
    for field in ("audio_necessary", "video_necessary"):
        n_pos = sum(1 for i in train_idx if records[i][field])
        chosen = set(rng.sample(train_idx, n_pos))
        for i in train_idx:
            dm[i][field] = i in chosen
    for i in train_idx:
        dm[i]["source"] = "density_matched"
    write_records(dm, Path(args.out_dir) / "density_matched_prior.jsonl")

    # ---- 3. no-prior (all zero masks disable both auxiliary losses)
    np_ = [dict(r) for r in records]
    for r in np_:
        r["audio_necessary"] = False
        r["video_necessary"] = False
        r["source"] = "no_prior"
    write_records(np_, Path(args.out_dir) / "no_prior.jsonl")

    # ---- 4. all-true (mirrors the archived all-sample prior convention)
    at = [dict(r) for r in records]
    for r in at:
        r["audio_necessary"] = True
        r["video_necessary"] = True
        r["text_sufficient"] = False
        r["source"] = "all_true"
    write_records(at, Path(args.out_dir) / "all_true.jsonl")

    # ---- verification
    print("\nverification (training split):")
    for name in ("class_conditioned_prior", "density_matched_prior"):
        rec = load_records(Path(args.out_dir) / f"{name}.jsonl")
        tr = [r for r in rec if r["split"] == "train"]
        changed_a = sum(1 for i, r in zip(train_idx, tr) if r["audio_necessary"] != records[i]["audio_necessary"])
        changed_v = sum(1 for i, r in zip(train_idx, tr) if r["video_necessary"] != records[i]["video_necessary"])
        print(f"  {name}:")
        print(fmt("  audio_necessary", mask_stats(tr, "audio_necessary")))
        print(fmt("  video_necessary", mask_stats(tr, "video_necessary")))
        print(f"    changed assignments vs original: audio {changed_a}, video {changed_v}")


if __name__ == "__main__":
    main()
