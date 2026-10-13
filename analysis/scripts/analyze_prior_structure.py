from pathlib import Path
import json,math,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
rows=[json.loads(l) for l in (ROOT/'SATI 9/data/llm_prior.jsonl').read_text(encoding='utf-8').splitlines()]
train=[x for x in rows if x['split']=='train'];groups={c:[x for x in train if x['intent']==c] for c in sorted({x['intent'] for x in train})}
def H(p):return -sum(v*math.log2(v) for v in p if v>0)
out={}
for k in ['text_sufficient','audio_necessary','video_necessary']:
 p=np.mean([x[k] for x in train]);hc=sum(len(v)/len(train)*H([np.mean([x[k] for x in v]),1-np.mean([x[k] for x in v])]) for v in groups.values());h=H([p,1-p])
 orig=np.array([len(v)/len(train) for v in groups.values()]);selected=np.array([sum(x[k] for x in v)/(p*len(train)) for v in groups.values()])
 mixed=[c for c,v in groups.items() if 0<sum(x[k] for x in v)<len(v)]
 out[k]={'coverage':float(p),'entropy':h,'conditional_entropy':hc,'mutual_information':h-hc,'explained_fraction':(h-hc)/h,'mixed_classes':len(mixed),'constant_classes':30-len(mixed),'class_distribution_tv':float(.5*np.abs(orig-selected).sum()),'empty_batch_probability_B48':float((1-p)**48)}
out['class_rates']={c:{'n':len(v),'text':float(np.mean([x['text_sufficient'] for x in v])),'audio':float(np.mean([x['audio_necessary'] for x in v])),'video':float(np.mean([x['video_necessary'] for x in v]))} for c,v in groups.items()}
(ROOT/'paper_work/prior_structure_analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
for k,v in out.items():
 if k!='class_rates':print(k,v)

