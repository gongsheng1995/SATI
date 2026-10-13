"""Prepare a traceable NEW prior archive. Never reconstruct historical provenance.

Default is offline preparation. API execution requires explicit --execute and
--max-requests. API key is read from an environment variable and never persisted.
Source prompts are transcribed via AST; importing the old async generator is avoided.
Captions must have a separate provenance manifest. Test labels are not sent to the API.
"""
import argparse
import ast
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def save_manifest(path, value):
    temporary=path.with_suffix('.json.tmp')
    with temporary.open('w',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2))
        f.flush();os.fsync(f.fileno())
    temporary.replace(path)

def append_record(path, value, key):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False).replace(key,'[REDACTED_CREDENTIAL]')+'\n')
        f.flush();os.fsync(f.fileno())

@contextmanager
def archive_lock(folder):
    # OS advisory lock is automatically released on a crash; never race writers.
    with (folder/'archive.lock').open('a+b') as f:
        if f.seek(0,2)==0:
            f.write(b'0');f.flush()
        f.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-json',required=True)
    ap.add_argument('--captions',required=True)
    ap.add_argument('--caption-provenance',required=True)
    ap.add_argument('--model',required=True,help='Documented model identifier; prefer a pinned snapshot.')
    ap.add_argument('--endpoint',required=True,help='Authorized HTTPS chat/completions endpoint.')
    ap.add_argument('--output-dir',required=True)
    ap.add_argument('--api-key-env',default='SATI_PRIOR_API_KEY')
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--max-requests',type=int)
    args=ap.parse_args()
    if not args.endpoint.startswith('https://'):ap.error('Endpoint must use HTTPS.')
    endpoint=urlsplit(args.endpoint)
    if not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        ap.error('Endpoint must contain no credentials, query or fragment.')
    data_path,caption_path,caption_provenance=map(Path,(args.data_json,args.captions,args.caption_provenance))
    data=json.loads(data_path.read_text(encoding='utf-8'))
    cp=json.loads(caption_provenance.read_text(encoding='utf-8'))
    if not cp.get('source') or not cp.get('generation_or_collection_protocol'):
        raise ValueError('Caption provenance needs source and generation_or_collection_protocol.')
    caption_rows=[json.loads(s) for s in caption_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    captions={r['id']:r.get('video_caption','') for r in caption_rows}
    if len(captions)!=len(caption_rows):raise ValueError('Duplicate caption IDs.')
    if any(not isinstance(caption,str) for caption in captions.values()):raise ValueError('Captions must be strings.')
    ids=[r['id'] for r in data['train']]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate training IDs.')
    if set(ids)-set(captions):raise ValueError('Every training ID needs an explicit caption record; missing captions use an empty string.')
    old=ROOT/'scripts/generate_prior.py'
    prompts={}
    for node in ast.parse(old.read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            for name in node.targets:
                if isinstance(name,ast.Name) and name.id in ('SYSTEM_PROMPT','USER_PROMPT_TEMPLATE','USER_PROMPT_NO_VIDEO'):
                    prompts[name.id]=node.value.value
    if len(prompts)!=3:raise ValueError('Prompt source changed; manual review required.')
    folder=Path(args.output_dir);folder.mkdir(exist_ok=True,parents=True)
    if args.execute and (args.max_requests is None or args.max_requests<=0):
        ap.error('A separately authorized positive --max-requests is required.')
    with archive_lock(folder):
        run_archive(args,folder,data,captions,prompts,data_path,caption_path,caption_provenance,old)

def run_archive(args,folder,data,captions,prompts,data_path,caption_path,caption_provenance,old):
    ids=[r['id'] for r in data['train']]
    manifest={'new_archive':True,'historical_archive_reconstructed':False,
              'source_data_sha256':sha(data_path),'captions_sha256':sha(caption_path),
              'caption_provenance_sha256':sha(caption_provenance),'model_requested':args.model,
              'endpoint':args.endpoint,'temperature':0.1,'max_tokens':100,
              'max_attempts_per_sample':3,'prompts':prompts,'prompt_source_sha256':sha(old),
              'splits_sent_to_generator':['train'],
              'fallback_policy':'No heuristic fallback. Failed rows are explicitly unsuccessful and must not silently enter training.',
              'raw_record_policy':'Store input, exact payload, complete response/model/fingerprint/usage, parsing status, attempt count; never headers or API key.',
              'status':'prepared_not_run'}
    manifest_path=folder/'generation_manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text(encoding='utf-8'))
        for k in ('source_data_sha256','captions_sha256','caption_provenance_sha256','model_requested','endpoint','prompts',
                  'temperature','max_tokens','max_attempts_per_sample','prompt_source_sha256'):
            if previous[k]!=manifest[k]:raise ValueError('Existing archive has a different protocol; use a new directory.')
        manifest=previous
    save_manifest(manifest_path,manifest)
    if not args.execute:
        print(json.dumps({'status':manifest['status'],'train_samples':len(ids),'api_calls_this_invocation':0}));return
    if manifest.get('request_cap') is not None and manifest['request_cap']!=args.max_requests:
        raise ValueError('An archive resume cannot reset its original API request cap.')
    manifest['request_cap']=args.max_requests
    key=os.environ.get(args.api_key_env)
    if not key:raise RuntimeError('API key environment variable is not set.')
    raw_path=folder/'raw_responses.jsonl';prior_path=folder/'train_prior.jsonl'
    ledger_path=folder/'request_reservations.jsonl'
    completed={}
    if prior_path.exists():
        for line in prior_path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['id'] in completed or r['id'] not in ids:raise ValueError('Duplicate or foreign completed ID.')
            completed[r['id']]=r
    records=[json.loads(line) for line in raw_path.read_text(encoding='utf-8').splitlines() if line.strip()] if raw_path.exists() else []
    reservations=[json.loads(line) for line in ledger_path.read_text(encoding='utf-8').splitlines() if line.strip()] if ledger_path.exists() else []
    if records and not reservations:
        raise ValueError('Legacy response-only archive has no durable request reservations; do not reset its cap.')
    reserved_indices=[r['request_index'] for r in reservations]
    if reserved_indices!=list(range(1,len(reservations)+1)):
        raise ValueError('Request reservation ledger is not contiguous.')
    attempts={sid:0 for sid in ids}
    for r in reservations:
        if r['id'] not in attempts or r['attempt']!=attempts[r['id']]+1:
            raise ValueError('Request reservation IDs/attempts differ from the training protocol.')
        attempts[r['id']]=r['attempt']
        if attempts[r['id']]>3:raise ValueError('Per-sample lifetime attempt cap exceeded.')
    seen_responses=set()
    reservation_by_index={r['request_index']:r for r in reservations}
    for r in records:
        index=r.get('request_index')
        reserved=reservation_by_index.get(index)
        if (index in seen_responses or not reserved or
            r['id']!=reserved['id'] or r['attempt']!=reserved['attempt']):
            raise ValueError('Response does not match a unique durable reservation.')
        seen_responses.add(index)
        if r.get('parse_success') and r['id'] not in completed:
            # A crash after durable response but before prior append needs no API retry.
            result=r.get('parsed_prior')
            if (not result or result['id']!=r['id'] or result['split']!='train' or
                any(type(result.get(k)) is not bool for k in ('text_sufficient','audio_necessary','video_necessary'))):
                raise ValueError('Successful response lacks a valid recoverable parsed prior.')
            append_record(prior_path,result,key);completed[r['id']]=result
    prior_calls=len(reservations)
    if prior_calls>args.max_requests:raise ValueError('Saved request reservations exceed the original cap.')
    manifest.update(status='running',request_accounting='Durable reservations count against lifetime cap, including calls with unknown outcome after a crash.')
    save_manifest(manifest_path,manifest)
    calls=0;failed=[]
    for item in data['train']:
        if item['id'] in completed:continue
        caption=captions[item['id']]
        template=prompts['USER_PROMPT_TEMPLATE' if caption else 'USER_PROMPT_NO_VIDEO']
        user=template.format(text=item['text'],video_caption=caption,intent=item['label'])
        payload={'model':args.model,'messages':[{'role':'system','content':prompts['SYSTEM_PROMPT']},
                                             {'role':'user','content':user}],
                 'temperature':0.1,'max_tokens':100}
        success=False
        for attempt in range(attempts[item['id']]+1,4):
            if prior_calls+calls>=args.max_requests:break
            calls+=1
            record={'id':item['id'],'split':'train','intent':item['label'],'caption':caption,
                    'attempt':attempt,'started_at_utc':datetime.now(timezone.utc).isoformat(),
                    'payload':payload,'fallback':False,'parse_success':False,
                    'request_index':prior_calls+calls}
            # Persist the original cap and reservation BEFORE any network request.
            append_record(ledger_path,{'id':item['id'],'attempt':attempt,
                'request_index':prior_calls+calls,'reserved_at_utc':record['started_at_utc']},key)
            request=urllib.request.Request(args.endpoint,data=json.dumps(payload).encode(),
                                          headers={'Content-Type':'application/json','Authorization':'Bearer '+key})
            try:
                with urllib.request.urlopen(request,timeout=60) as response:
                    body=response.read().decode('utf-8')
                record['raw_response']=body.replace(key,'[REDACTED_CREDENTIAL]')
                decoded=json.loads(body)
                record['model_returned']=decoded.get('model')
                record['system_fingerprint']=decoded.get('system_fingerprint')
                record['usage']=decoded.get('usage')
                content=decoded['choices'][0]['message']['content'].strip()
                if content.startswith('```'):
                    content=content.split('\n',1)[1].rsplit('```',1)[0].strip()
                bits=json.loads(content)
                fields=('text_sufficient','audio_necessary','video_necessary')
                if any(type(bits.get(k)) is not bool for k in fields):raise ValueError('Not three JSON booleans.')
                record['parse_success']=True
                result={'id':item['id'],'split':'train','intent':item['label'],
                        **{k:bits[k] for k in fields},'source':'new_auditable_llm',
                        'model_requested':args.model,'model_returned':decoded.get('model'),'fallback':False}
                record['parsed_prior']=result
                success=True
            except Exception as error:
                # Store exception class only: a provider message might echo credentials.
                record['error_type']=type(error).__name__
            append_record(raw_path,record,key)
            if success:
                append_record(prior_path,result,key);completed[item['id']]=result
                break
            time.sleep(min(attempt,3))
        if not success:failed.append(item['id'])
        if prior_calls+calls>=args.max_requests:break
    manifest.update(status='partial_or_failed' if len(completed)<len(ids) else 'generated',
                    api_calls_this_invocation=calls,failed_ids_this_invocation=failed,
                    api_calls_total=prior_calls+calls,completed_train_samples=len(completed),
                    warning='New outputs require completeness/provenance audit and new downstream training before manuscript claims.')
    save_manifest(manifest_path,manifest)
    print(json.dumps({'api_calls':calls,'status':manifest['status']}))

if __name__=='__main__':main()
