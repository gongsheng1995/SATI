from pathlib import Path
import hashlib, json, math, statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=Path.cwd()
PACKAGE=ROOT
keys=['acc','f1','prec','rec','weighted_f1','weighted_p']
aggregate=json.loads((PACKAGE/'aggregations/six_metric_results.json').read_text())
files=list((PACKAGE/'runs').rglob('test_predictions.jsonl'))
extra=[]
runs=[]
for p in sorted(files):
    folder=p.parent
    records=[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]
    cfg=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    c=cfg.get('num_classes') or len(records[0]['probabilities'])
    cm=[[0]*c for _ in range(c)]
    issues=[]
    ids=[r['id'] for r in records]
    if len(ids)!=len(set(ids)): issues.append('Duplicate test IDs')
    label2id=cfg.get('label2id',{})
    for r in records:
        y,q=r['target_id'],r['pred_id']
        ps=r['probabilities']
        if len(ps)!=c or not all(math.isfinite(v) and 0<=v<=1 for v in ps):
            issues.append('Invalid probability vector'); continue
        if abs(sum(ps)-1)>1e-5 or max(range(c),key=lambda j:ps[j])!=q:
            issues.append('Probability sum/argmax mismatch')
        if not(0<=y<c and 0<=q<c): issues.append('Class ID out of universe'); continue
        if label2id and (label2id.get(r['target_label'])!=y or label2id.get(r['pred_label'])!=q):
            issues.append('Label mapping mismatch')
        cm[y][q]+=1
    n=len(records)
    support=[sum(row) for row in cm]
    predictions=[sum(cm[y][q] for y in range(c)) for q in range(c)]
    pr=[cm[k][k]/predictions[k] if predictions[k] else 0 for k in range(c)]
    rc=[cm[k][k]/support[k] if support[k] else 0 for k in range(c)]
    fs=[2*p*r/(p+r) if p+r else 0 for p,r in zip(pr,rc)]
    met=dict(acc=sum(cm[k][k] for k in range(c))/n,f1=sum(fs)/c,
             prec=sum(pr)/c,rec=sum(rc)/c,weighted_f1=sum(s*f for s,f in zip(support,fs))/n,
             weighted_p=sum(s*p for s,p in zip(support,pr))/n)
    saved=json.loads((folder/'test_results.json').read_text())
    available=[k for k in keys if k in saved]
    maxerr=max(abs(met[k]-saved[k]) for k in available)
    if maxerr>1e-10: issues.append('Saved metric mismatch')
    packaged=p.is_relative_to(PACKAGE)
    runs.append(dict(path=str(folder.relative_to(PACKAGE if packaged else ROOT)),
                     in_public_package_candidate=packaged,name=folder.name,n=n,classes=c,
                     prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                     metrics_percent=[met[k]*100 for k in keys],max_saved_error=maxerr,
                     issues=sorted(set(issues))))

agg_issues=[]
source_issues=[]
for name,rec in aggregate['records'].items():
    vectors=[]
    for src in rec['runs']:
        candidates=[r for r in runs if r['prediction_sha256']==src['prediction_sha256']]
        if not any(r['in_public_package_candidate'] for r in candidates):
            agg_issues.append(f'{name}: missing prediction hash in public-package candidate {src["path"]}')
        if not candidates:
            source_issues.append(f'{name}: missing prediction hash in both package and workspace {src["path"]}'); continue
        vector=candidates[0]['metrics_percent']; vectors.append(vector)
        if max(abs(x-y) for x,y in zip(vector,src['metrics']))>1e-8:
            source_issues.append(f'{name}: run metric mismatch {src["path"]}')
    if len(vectors)==len(rec['runs']):
        means=[statistics.mean(v[i] for v in vectors) for i in range(6)]
        if max(abs(x-y) for x,y in zip(means,rec['mean']))>1e-8:
            source_issues.append(f'{name}: aggregate mean mismatch')
        if len(vectors)>1 and 'sd' in rec:
            sds=[statistics.stdev(v[i] for v in vectors) for i in range(6)]
            if max(abs(x-y) for x,y in zip(sds,rec['sd']))>1e-8:
                source_issues.append(f'{name}: aggregate SD mismatch')
result=dict(scope='Local public-package candidate prediction audit; does not confirm remote publication or replicate training',
            n_runs=len(runs),n_packaged_runs=sum(r['in_public_package_candidate'] for r in runs),
            n_additional_workspace_runs=len(extra),n_predictions=sum(r['n'] for r in runs),
            n_canonical_conditions=len(aggregate['records']),runs=runs,canonical_aggregation_issues=agg_issues,
            workspace_canonical_issues=source_issues,
            prediction_metrics_all_pass=not source_issues and not any(r['issues'] for r in runs),
            all_pass=not agg_issues and not source_issues and not any(r['issues'] for r in runs))
(OUT/'package_prediction_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='runs'},ensure_ascii=True))
