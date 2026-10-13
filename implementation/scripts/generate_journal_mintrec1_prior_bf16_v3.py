"""Prepared pinned local teacher generator. Default prepares only, no GPU work.

Requires a new explicit GPU allowance, isolated Linux session, fixed deadline,
local verified model files and lifetime generation cap. No API calls/downloads.
Only the prepared training inputs are processed; no heuristic mask fallback.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys

from generate_auditable_prior import archive_lock,save_manifest

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('text_sufficient','audio_necessary','video_necessary')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()] if p.exists() else []
def append(p,record):
    with p.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--completion-authorization',required=True);parser.add_argument('--execute',action='store_true');options=parser.parse_args()
    auth_path=ROOT/options.completion_authorization;auth=read(auth_path);plan_path=ROOT/auth['plan_path'];plan=read(plan_path)
    if not options.execute:print(json.dumps({'status':'prepared_not_run','requests':1334}));return
    if sys.platform!='linux' or os.getsid(0)!=os.getpid() or auth['status']!='approved_completion_based_generation' or auth['plan_sha256']!=sha(plan_path):raise ValueError('Approved isolated Linux generation required')
    binding=read(ROOT/auth['instance_binding_path'])
    if sha(ROOT/auth['instance_binding_path'])!=auth['instance_binding_sha256'] or not binding['gpu_verified']:raise ValueError('Verified instance binding required')
    for k in ('host','port','instance_id','verified_hourly_usd','billing_start_utc'):
        if auth[k]!=binding[k]:raise ValueError('Instance/fee binding differs')
    for f,h in {**plan['required_code_sha256'],**plan['input_sha256']}.items():
        if sha(ROOT/f)!=h:raise ValueError('Source/input differs '+f)
    options.prepared_dir=plan['prepared_dir'];options.snapshot_dir=auth['snapshot_dir'];options.output_dir=plan['output_dir'];options.authorization_reference=auth['authorization_reference'];options.max_generations=1334;options.batch_size=1
    preparation=read(ROOT/options.prepared_dir/'preparation_manifest.json');inputs=lines(ROOT/options.prepared_dir/'training_inputs.jsonl');byid={r['id']:r for r in inputs}
    truth=read(ROOT/'data/prepared_mintrec1/original/data.json')['train']
    if len(inputs)!=1334 or len(byid)!=1334 or set(byid)!={r['id'] for r in truth} or any(r['split']!='train' for r in inputs):raise ValueError('Training-only input mismatch')
    import shutil,urllib.request,urllib.parse,fcntl
    queue_locks=[]
    for p in (ROOT/'results/reruns/queue.lock',ROOT/'results/second_review/queue.lock'):
        p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);queue_locks.append(f)
    source=Path(options.snapshot_dir);source.mkdir(parents=True,exist_ok=True)
    required=sum(i['size'] for i in preparation['expected_model_files'] if not (source/i['rfilename']).exists())+2*1024**3
    if shutil.disk_usage(source).free<required:raise ValueError('Teacher snapshot disk reserve unavailable')
    folder=ROOT/options.output_dir;folder.mkdir(parents=True,exist_ok=True)
    with archive_lock(folder):
        save_manifest(folder/'download_status.json',{'status':'downloading','model':preparation['model'],'revision':preparation['revision'],'at_utc':now()})
        for item in preparation['expected_model_files']:
            name=item['rfilename'];dest=source/name
            if '/' in name or name.startswith('..'):raise ValueError('Unsafe model file')
            if not dest.exists():
                temp=dest.with_suffix(dest.suffix+'.partial')
                if temp.exists():raise ValueError('Partial download retained; no silent overwrite')
                url='https://huggingface.co/'+preparation['model']+'/resolve/'+preparation['revision']+'/'+urllib.parse.quote(name)
                with urllib.request.urlopen(url,timeout=120) as response,temp.open('xb') as out:
                    n=0
                    while True:
                        block=response.read(4*1024**2)
                        if not block:break
                        n+=len(block)
                        if n>item['size']:raise ValueError('Oversized model file')
                        out.write(block)
                if n!=item['size']:raise ValueError('Incomplete model file')
                temp.replace(dest)
            if dest.stat().st_size!=item['size']:raise ValueError('Snapshot size differs')
            if item.get('lfs'):
                if sha(dest)!=item['lfs']['sha256']:raise ValueError('Weight digest differs')
            elif hashlib.sha1(b'blob '+str(item['size']).encode()+b'\0'+dest.read_bytes()).hexdigest()!=item['blobId']:raise ValueError('Tokenizer git digest differs')
            save_manifest(folder/'download_status.json',{'status':'downloading','last_verified_file':name,'at_utc':now()})
        save_manifest(folder/'download_status.json',{'status':'verified','at_utc':now()})
        generate(options,preparation,inputs,byid,folder,None)

def generate(options,preparation,inputs,byid,folder,deadline):
    source=Path(options.snapshot_dir).resolve()
    protocol={'model':preparation['model'],'revision':preparation['revision'],
        'preparation_sha256':sha((ROOT/options.prepared_dir)/'preparation_manifest.json'),
        'training_inputs_sha256':preparation['training_inputs_sha256'],
        'source_sha256':sha(Path(__file__)),
        'archive_helpers_sha256':sha(Path(__file__).with_name('generate_auditable_prior.py')),
        'batch_size':options.batch_size,'max_generations':options.max_generations,
        'absolute_deadline':None,
        'authorization_reference':options.authorization_reference,
        'decoding':'Greedy, max100 new tokens, repetition_penalty1, one beam; no sampling.',
        'unknown_outcome_policy':'Reserved generations count against the lifetime cap even after a crash; at most3 reservations per sample. Explicit invalid JSON is not retried with identical greedy input.',
        'input_scope':'Prepared train transcripts/known labels, explicit no-video variant; no held-out labels/observations.',
        'status':'preflight','gpu_generation_started':False,'api_calls':0}
    manifest_path=folder/'generation_manifest.json'
    if manifest_path.exists():
        previous=read(manifest_path)
        for k,v in protocol.items():
            if k not in ('status','gpu_generation_started','api_calls') and previous.get(k)!=v:
                raise ValueError('Archive resume cannot change input/model/code/decoding/cap/deadline: '+k)
        protocol=previous
    save_manifest(manifest_path,protocol)
    verified=[]
    for item in preparation['expected_model_files']:
        file=source/item['rfilename']
        if not file.is_file() or file.stat().st_size!=item['size']:raise ValueError('Model snapshot file missing or wrong size: '+item['rfilename'])
        h=hashlib.sha256()
        git=hashlib.sha1(b'blob '+str(item['size']).encode()+b'\0') if not item.get('lfs') else None
        with file.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                h.update(block)
                if git:git.update(block)
        if item.get('lfs') and h.hexdigest()!=item['lfs']['sha256']:raise ValueError('Weight SHA256 differs.')
        if git and git.hexdigest()!=item['blobId']:raise ValueError('Model/tokenizer git blob differs.')
        verified.append({'file':item['rfilename'],'sha256':h.hexdigest(),'bytes':item['size']})
    raw_path=folder/'raw_outputs.jsonl';reservation_path=folder/'generation_reservations.jsonl'
    prior_path=folder/'train_prior.jsonl'
    reservations=lines(reservation_path);raw=lines(raw_path);priors=lines(prior_path)
    completed={r['id']:r for r in priors};attempts={sid:0 for sid in byid};responses={}
    if len(completed)!=len(priors) or set(completed)-set(byid):raise ValueError('Duplicate/foreign prior IDs.')
    def validate_prior(row,sid):
        if (row['id']!=sid or row.get('split')!='train' or row.get('intent')!=byid[sid]['intent'] or
            row.get('model')!=protocol['model'] or row.get('revision')!=protocol['revision'] or
            any(type(row.get(k)) is not bool for k in FIELDS) or row.get('fallback') is not False):
            raise ValueError('Parsed prior differs from the pinned training protocol.')
    for sid,row in completed.items():validate_prior(row,sid)
    for i,r in enumerate(reservations,1):
        if r['index']!=i or r['id'] not in attempts or r['attempt']!=attempts[r['id']]+1:
            raise ValueError('Reservation ledger differs from frozen input order/attempt cap.')
        attempts[r['id']]+=1
        if attempts[r['id']]>3:raise ValueError('Per-sample reservation cap exceeded.')
    for r in raw:
        index=r['reservation_index']
        if not 1<=index<=len(reservations) or index in responses or r['id']!=reservations[index-1]['id']:
            raise ValueError('Raw response lacks a unique matching reservation.')
        responses[index]=r
        if r['parse_success']:
            row=r['parsed_prior'];validate_prior(row,r['id'])
            if r['id'] in completed and completed[r['id']]!=row:raise ValueError('Raw and appended prior disagree.')
            if r['id'] not in completed:append(prior_path,row);completed[row['id']]=row
    raw_successes={r['id'] for r in raw if r['parse_success']}
    if set(completed)!=raw_successes:raise ValueError('Every completed prior requires a retained successful raw response.')
    failed={r['id'] for r in raw if not r['parse_success']}
    if len(reservations)>options.max_generations:raise ValueError('Lifetime generation cap already exceeded.')
    pending=[r for r in inputs if r['id'] not in completed and r['id'] not in failed and attempts[r['id']]<3]
    used=len(reservations)
    if not pending or used==options.max_generations:
        protocol.update(status='generated' if len(completed)==len(inputs) else 'partial_or_failed',
            reserved_generations=used,completed_train_samples=len(completed),failed_ids=sorted(failed))
        save_manifest(manifest_path,protocol)
        print(json.dumps({'status':protocol['status'],'generated_train_samples':len(completed),'gpu_work_started_this_invocation':False}));return
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig,set_seed,LogitsProcessor,LogitsProcessorList
    versions={p:importlib.metadata.version(p) for p in ('torch','transformers','numpy')}
    if versions['transformers']!='4.45.2' or versions['torch']!='2.8.0+cu128':raise ValueError('Prepared model environment changed; freeze a reviewed new protocol instead.')
    if not torch.cuda.is_available():raise RuntimeError('Authorized CUDA GPU required; no CPU fallback.')
    set_seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True,trust_remote_code=False,padding_side='left')
    model=AutoModelForCausalLM.from_pretrained(source,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,attn_implementation='eager').cuda().eval()
    assert torch.cuda.is_bf16_supported()
    assert all(bool(torch.isfinite(v).all()) for v in model.parameters())
    class FiniteScores(LogitsProcessor):
        def __call__(self,input_ids,scores):
            if not torch.isfinite(scores).all(): raise FloatingPointError('Nonfinite generation logits; no repair')
            return scores
    config=GenerationConfig.from_dict(model.generation_config.to_dict())
    config.do_sample=False;config.temperature=1.;config.top_p=1.;config.top_k=50
    config.repetition_penalty=1.;config.max_new_tokens=100;config.num_beams=1
    protocol.update(status='running',model_files=verified,versions=versions,
        tokenizer_chat_template=tokenizer.chat_template,generation_config=config.to_dict(),
        effective_sampling_temperature=None,gpu=torch.cuda.get_device_name(0),
        torch_dtype='bfloat16',attention_implementation='eager',gpu_generation_started=True,
        reproducibility_caution='Exact files, inputs, token IDs and outputs retained; bitwise equality across hardware or resumed batch composition is not guaranteed.')
    save_manifest(manifest_path,protocol)
    for start in range(0,len(pending),options.batch_size):
        remaining=options.max_generations-used
        if remaining<=0:break
        batch=pending[start:start+min(options.batch_size,remaining)]
        chats=[tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True) for r in batch]
        encoded=tokenizer(chats,return_tensors='pt',padding=True,truncation=False)
        if encoded['input_ids'].shape[1]+100>model.config.max_position_embeddings:
            raise ValueError('Prepared prompt exceeds context; no silent truncation.')
        indices=[]
        for row in batch:
            used+=1;attempts[row['id']]+=1;indices.append(used)
            append(reservation_path,{'id':row['id'],'index':used,'attempt':attempts[row['id']],
                'batch_ids':[r['id'] for r in batch],'reserved_at_utc':now()})
        with torch.no_grad():
            cuda_inputs={k:v.cuda() for k,v in encoded.items()}
            assert torch.isfinite(model(**cuda_inputs).logits).all()
            generated=model.generate(**cuda_inputs,generation_config=config,logits_processor=LogitsProcessorList([FiniteScores()]))
        generated=generated[:,encoded['input_ids'].shape[1]:].cpu()
        for i,row in enumerate(batch):
            output_tokens=generated[i].tolist();text=tokenizer.decode(output_tokens,skip_special_tokens=True)
            record={'id':row['id'],'reservation_index':indices[i],'finished_at_utc':now(),
                'messages':row['messages'],'rendered_chat':chats[i],
                'input_token_ids':encoded['input_ids'][i][encoded['attention_mask'][i].bool()].tolist(),
                'output_token_ids':output_tokens,'raw_text':text,'parse_success':False,'fallback':False}
            try:
                bits=json.loads(text.strip())
                if set(bits)!=set(FIELDS) or any(type(bits[k]) is not bool for k in FIELDS):raise ValueError('Expected exactly three JSON booleans.')
                prior={'id':row['id'],'split':'train','intent':row['intent'],**bits,
                    'source':'pinned_offline_teacher_no_video','model':protocol['model'],'revision':protocol['revision'],'fallback':False}
                record.update(parse_success=True,parsed_prior=prior)
            except (ValueError,TypeError) as error:record['parse_error_type']=type(error).__name__;failed.add(row['id'])
            append(raw_path,record)
            if record['parse_success']:append(prior_path,prior);completed[row['id']]=prior
        protocol.update(reserved_generations=used,completed_train_samples=len(completed),failed_ids=sorted(failed));save_manifest(manifest_path,protocol)
        if failed:
            protocol.update(status='strict_parse_failure_stopped');save_manifest(manifest_path,protocol)
            raise ValueError('Strict JSON failure; raw response preserved, no fallback')
    protocol.update(status='generated' if len(completed)==len(inputs) else 'partial_or_failed',
        reserved_generations=used,completed_train_samples=len(completed),failed_ids=sorted(failed),
        raw_outputs_sha256=sha(raw_path) if raw_path.exists() else None,
        train_prior_sha256=sha(prior_path) if prior_path.exists() else None,
        caution='Independent completeness/quality audit and new downstream runs required before manuscript claims.')
    save_manifest(manifest_path,protocol)
    print(json.dumps({'status':protocol['status'],'generated_train_samples':len(completed),'api_calls':0}))

if __name__=='__main__':main()
