"""Prepare new controls and a14-run plan; NEVER start training or renew a rental."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def joint_counts(records, ids):
    byclass=defaultdict(Counter)
    for r in records:
        if r['id'] in ids:
            byclass[r['intent']][f"{int(r['audio_necessary'])}{int(r['video_necessary'])}"]+=1
    return {c:dict(sorted(counts.items())) for c,counts in sorted(byclass.items())}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mask-seed',type=int,default=42)
    args=ap.parse_args()
    prior=ROOT/'data/llm_prior.jsonl'
    records=[json.loads(s) for s in prior.read_text(encoding='utf-8').splitlines() if s.strip()]
    ids=[r['id'] for r in records]
    assert len(set(ids))==len(ids)
    config=json.loads((ROOT/'results/reruns/sati_focal_seed42/config.json').read_text())
    newdir=ROOT/'data/second_review_controls'
    newdir.mkdir(exist_ok=True)
    control_metadata={}
    for protocol,data in [('original','data/retrain_inputs/data.json'),('debiased','data/debiased_mintrec2/data.json')]:
        dataset=json.loads((ROOT/data).read_text(encoding='utf-8'))
        train={r['id']:r['label'] for r in dataset['train']}
        assert set(train)<=set(ids)
        indices=defaultdict(list)
        for i,r in enumerate(records):
            if r['id'] in train:
                assert r['split']=='train' and r['intent']==train[r['id']]
                indices[r['intent']].append(i)
        shuffled=[dict(r) for r in records]
        rng=random.Random(args.mask_seed)
        for c,idxs in sorted(indices.items()):
            pairs=[(records[i]['audio_necessary'],records[i]['video_necessary']) for i in idxs]
            rng.shuffle(pairs)
            for i,(a,v) in zip(idxs,pairs):
                shuffled[i].update(audio_necessary=a,video_necessary=v,
                                   source='joint_class_permutation',selection_seed=args.mask_seed)
        assert joint_counts(records,train)==joint_counts(shuffled,train)
        assert all(a==b for a,b in zip(records,shuffled) if a['id'] not in train)
        target=newdir/f'{protocol}_joint_class_seed{args.mask_seed}.jsonl'
        payload=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in shuffled)
        if target.exists() and target.read_text(encoding='utf-8')!=payload:
            raise ValueError(f'Refusing to overwrite a different control archive: {target}')
        target.write_text(payload,encoding='utf-8')
        control_metadata[protocol]={'path':target.relative_to(ROOT).as_posix(),'sha256':sha(target),
                                    'train_n':len(train),'data_sha256':sha(ROOT/data),
                                    'joint_counts_by_class':joint_counts(shuffled,train),
                                    'changed_joint_assignment_n':sum((a['audio_necessary'],a['video_necessary'])!=(b['audio_necessary'],b['video_necessary']) for a,b in zip(records,shuffled) if a['id'] in train)}
    keys=['text_encoder','hidden_dim','tensor_rank','dropout','label_smoothing','lambda_rdrop','lambda_aux',
          'use_focal','batch_size','lr','weight_decay','epochs','warmup_ratio','grad_clip','patience',
          'use_swa','swa_start','num_workers']
    common={k:config[k] for k in keys}
    common.update(audio_feats=config['audio_feats'],video_feats=config['video_feats'])
    jobs=[]
    def add(name,seed,prior,data='data/retrain_inputs/data.json',method=None):
        params={**common,'data_json':data,'prior_file':prior,'seed':seed,
                'output_dir':f'results/second_review/{name}'}
        script='scripts/train_matched_balancing.py' if method else 'scripts/train.py'
        command=[script]
        if method:command+=['--balancing',method]
        for k,v in params.items():
            if isinstance(v,bool):
                if v:command.append('--'+k)
            else:command+=['--'+k,str(v)]
        jobs.append({'name':name,'seed':seed,'parameters':params,'method':method,
                     'command':command,'prior_sha256':sha(ROOT/prior),
                     'data_sha256':sha(ROOT/data),'status':'prepared_not_run'})
    # Preserve the exact original mask realization42; vary downstream optimization seeds.
    for seed in (43,44):
        add(f'density_matched_focal_seed{seed}',seed,'data/controls/density_matched_prior.jsonl')
        add(f'class_conditioned_focal_seed{seed}',seed,'data/controls/class_conditioned_prior.jsonl')
        add(f'uncertainty_all_true_focal_seed{seed}',seed,'data/controls/all_true.jsonl',method='uncertainty')
        add(f'pcgrad_all_true_focal_seed{seed}',seed,'data/controls/all_true.jsonl',method='pcgrad')
    for protocol,data in [('original','data/retrain_inputs/data.json'),('debiased','data/debiased_mintrec2/data.json')]:
        for seed in (42,43,44):
            add(f'{protocol}_joint_class_mask{args.mask_seed}_focal_seed{seed}',seed,control_metadata[protocol]['path'],data=data)
    out=ROOT/'plans'
    out.mkdir(exist_ok=True)
    plan={'status':'prepared_not_run','jobs':jobs,'planned_new_training_runs':len(jobs),
          'old_completed_training_runs':22,'original_prior_sha256':sha(prior),'mask_seed':args.mask_seed,
          'control_metadata':control_metadata,
          'scope':'14 newly planned runs. Existing22 are not restarted. Both masks are shuffled jointly within class; four joint states and per-class modality counts are preserved exactly.',
          'inference_caution':'Each control uses one fixed mask realization across three optimization seeds. This does not estimate mask-generation variability.',
          'authorization':'The original14h allowance expired. Execution needs a separately agreed new runtime cap. Preparing this plan authorizes no GPU spending or API calls.',
          'required_feature_md5':{'audio_feats':'81d075ffd7fba4c0abdfa72fb77aa53c','video_feats':'c9516cade914af6f9fca2b92004208b4'},
          'required_code_sha256':{p:sha(ROOT/p) for p in ('scripts/train.py','scripts/train_matched_balancing.py','models/sati.py','models/fusion.py','scripts/export_review_predictions.py')},
          'required_versions':json.loads((ROOT/'results/reruns/environment.json').read_text())['versions'],
          'estimate':{'historical_original_mean_minutes':46.930692995,'historical_debiased_mean_minutes':26.424405290740744,
                      'training_hours_estimate':(11*46.930692995+3*26.424405290740744)/60,
                      'proposed_cap_hours':12,'price_assumption_usd_per_hour':0.48,
                      'price_status':'previous instance price; current account price must be confirmed',
                      'compute_cost_at_cap_usd':5.76,'suggested_prepaid_balance_usd':8},
          'later_batches_not_in_this_budget':[
              'Auxiliary coefficient0.1/0.6 crossed with SATI/all-sample, three seeds; fixed rank64',
              'Rank32/128 crossed with SATI/all-sample, three seeds; fixed auxiliary coefficient0.3',
              'Sequence cross-attention adaptation crossed with no-aux/all-sample/SATI on both splits, three seeds',
              'Traceable new prior generation and matched downstream reruns; needs documented captions/model/API budget']}
    path=out/'second_review_phase1.json'
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'plan':str(path),'new_runs':len(jobs),'estimated_training_hours':plan['estimate']['training_hours_estimate'],'status':plan['status']}))

if __name__=='__main__':main()
