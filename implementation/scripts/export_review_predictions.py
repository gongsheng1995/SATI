"""Export held-out predictions and verify they reproduce the saved run metrics."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.sati import SATI
from scripts.train import MIntRecDataset, collate_fn


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    c = json.loads((run / "config.json").read_text())
    dataset = MIntRecDataset(c["data_json"], c["audio_feats"], c["video_feats"],
                            c["prior_file"], c["text_encoder"], split="test")
    if dataset.label2id != c["label2id"]:
        raise ValueError("Label mapping differs from training")
    loader = DataLoader(dataset, batch_size=c["batch_size"], shuffle=False,
                        num_workers=4, collate_fn=collate_fn)
    device = torch.device("cuda")
    model = SATI(text_encoder_name=c["text_encoder"], audio_dim=c["audio_dim"],
                 video_dim=c["video_dim"], hidden_dim=c["hidden_dim"],
                 num_classes=c["num_classes"], tensor_rank=c["tensor_rank"],
                 dropout=c["dropout"], freeze_bert=c["freeze_bert"]).to(device)
    checkpoint = torch.load(run / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    del checkpoint
    model.eval()
    records, offset = [], 0
    for batch in loader:
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        audio, video = batch["audio"].to(device), batch["video"].to(device)
        labels = batch["labels"].numpy().tolist()
        probs = model(ids, mask, audio, video).logits.softmax(-1).cpu().numpy()
        pa = model(ids, mask, torch.zeros_like(audio), video).logits.softmax(-1).cpu().numpy()
        pv = model(ids, mask, audio, torch.zeros_like(video)).logits.softmax(-1).cpu().numpy()
        for j, label in enumerate(labels):
            item = dataset.samples[offset + j]
            pred = int(probs[j].argmax())
            records.append({"id": item["id"], "target_id": label, "pred_id": pred,
                            "target_label": dataset.label_list[label],
                            "pred_label": dataset.label_list[pred],
                            "probabilities": probs[j].astype(float).tolist(),
                            "audio_zero_pred_id": int(pa[j].argmax()),
                            "video_zero_pred_id": int(pv[j].argmax()),
                            "audio_zero_true_probability": float(pa[j, label]),
                            "video_zero_true_probability": float(pv[j, label])})
        offset += len(labels)
    y, pred = [r["target_id"] for r in records], [r["pred_id"] for r in records]
    calculated = {"acc": float(accuracy_score(y, pred)),
                  "weighted_f1": float(f1_score(y, pred, average="weighted")),
                  "f1": float(f1_score(y, pred, average="macro", labels=list(range(c["num_classes"])), zero_division=0))}
    saved = json.loads((run / "test_results.json").read_text())
    for key, value in calculated.items():
        if abs(value - saved[key]) > 1e-12:
            raise ValueError(f"Metric mismatch {key}: {value} vs {saved[key]}")
    with (run / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    report = {"n": len(records), "reproduced_saved_metrics": calculated,
              "checkpoint": "best_model.pt", "auxiliary_masks_used_at_prediction": False,
              "audio_zero_acc": float(accuracy_score(y, [r["audio_zero_pred_id"] for r in records])),
              "video_zero_acc": float(accuracy_score(y, [r["video_zero_pred_id"] for r in records])),
              "suppression_note": "Fixed-model feature zeroing; not a retrained unimodal baseline or causal modality necessity test."}
    (run / "prediction_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
