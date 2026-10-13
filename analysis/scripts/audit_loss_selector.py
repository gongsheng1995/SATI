"""Audit a prespecified confidence/loss selector against public human labels.

Uses released Mult/BERT per-example true-label probabilities. This is a
criterion-alignment diagnostic only: it is not a retrained, SATI-matched
downstream comparison.
"""
import csv, json, math, hashlib, subprocess
from pathlib import Path

ROOT = Path("paper_work/modality_bias_repo/Dataset")
HUMAN = ROOT / "Human-Annotation/MIntRec2.0/merged_test.tsv"
DATA = ROOT / "Phase-2-Automated-Annotation/MIntRec-2-auto-annotate"
PRIOR = Path("SATI 9/data/llm_prior.jsonl")
OUT = Path("paper_work/loss_selector_audit.json")

def read(path, delimiter):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))

def score(y, pred):
    tp = sum(a and b for a,b in zip(y,pred)); fp = sum(not a and b for a,b in zip(y,pred))
    fn = sum(a and not b for a,b in zip(y,pred)); tn = len(y)-tp-fp-fn
    p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
    f = 2*p*r/(p+r) if p+r else 0; acc=(tp+tn)/len(y)
    pe=((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/len(y)**2
    return {"positive":sum(pred),"tp":tp,"fp":fp,"tn":tn,"fn":fn,
            "precision":p,"recall":r,"f1":f,"cohen_kappa":(acc-pe)/(1-pe) if pe<1 else 0}

prior = {r["id"].lower():r for r in map(json.loads, open(PRIOR, encoding="utf-8")) if r.get("id")}
human = {}
for r in read(HUMAN, "\t"):
    label=(r.get("Modality") or "").strip(); key=f"dia{r['Dialogue_id']}_utt{r['Utterance_id']}".lower()
    if label and label.lower() != "na" and key in prior: human[key]=label.split("+")
rows=read(DATA/"test.tsv", "\t")
index={f"dia{r['Dialogue_id']}_utt{r['Utterance_id']}".lower():i for i,r in enumerate(rows)}
ids=[k for k in human if k in index]
assert len(ids)==398
labels=sorted({r["Label"] for r in rows})
outputs={m:read(DATA/f"mult_bert_{m}.csv", ",") for m in ("text","text_audio","text_video")}
assert all(len(v)==len(rows) for v in outputs.values())
assert all(int(outputs["text"][i]["actual_label"])==labels.index(r["Label"]) for i,r in enumerate(rows))
preds={}
report={"n":len(ids),"selector":"select modality when adding it to text decreases true-label NLL (equivalently increases true-label probability); strict threshold delta>0, no tuning",
        "source":"Text-Takes-Over-EMNLP-2025/MultiModal-Intent-EMNLP-2025, Phase-2-Automated-Annotation/MIntRec-2-auto-annotate; released Mult/BERT outputs",
        "caveat":"External model outputs; not retrained with SATI features, splits, or protocol. Alignment to minimum-sufficient human labels is descriptive and criteria differ.","modalities":{}}
try:
    report["source_repo_revision"] = subprocess.check_output(
        ["git", "-C", str(ROOT.parent), "rev-parse", "HEAD"], text=True,
        stderr=subprocess.DEVNULL
    ).strip()
except Exception:
    report["source_repo_revision"] = "unknown"
for modality in ("Audio","Video"):
    m=modality.lower(); variant=f"text_{m}"
    y=[modality in human[k] for k in ids]; pred=[]
    for k in ids:
        i=index[k]; p0=max(float(outputs["text"][i]["true_label_prob"]),1e-12)
        p1=max(float(outputs[variant][i]["true_label_prob"]),1e-12)
        pred.append(math.log(p0)-math.log(p1)>0)
    preds[m]=pred; report["modalities"][m]=score(y,pred)
report["joint_exact"]={"count":sum(("Audio" in human[k])==preds["audio"][j] and ("Video" in human[k])==preds["video"][j] for j,k in enumerate(ids)),"fraction":sum(("Audio" in human[k])==preds["audio"][j] and ("Video" in human[k])==preds["video"][j] for j,k in enumerate(ids))/len(ids)}
report["mapping_checks"]={"official_test_rows":len(rows),"label_index_matches":len(rows),"matched_annotated_test_rows":len(ids)}
report["sha256"]={}
for name,path in (("human_test",HUMAN),("test_rows",DATA/"test.tsv"),("text_probs",DATA/"mult_bert_text.csv"),("text_audio_probs",DATA/"mult_bert_text_audio.csv"),("text_video_probs",DATA/"mult_bert_text_video.csv")):
    report["sha256"][name]=hashlib.sha256(path.read_bytes()).hexdigest()
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
