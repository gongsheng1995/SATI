"""Prepared storage-only wrapper for future pooled sensitivity runs.

Keep native training, SWA tracking/evaluation and ordinary-best test selection.
Omit serialization of the optional unused SWA weight file, retain its dev metrics
and tensor byte estimate. Existing checkpoints are never deleted or moved.
Default reads the planned job only; no model imports/GPU work until --execute.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--absolute-deadline')
    parser.add_argument('--authorization-reference')
    parser.add_argument('--inherited-lock-fds')
    options=parser.parse_args()
    path=(ROOT/options.plan).resolve()
    if not path.is_relative_to((ROOT/'plans').resolve()):raise ValueError('Plan must be inside the repository plans directory.')
    plan=json.loads(path.read_text(encoding='utf-8'))
    if plan['stage'] not in ('auxiliary_sensitivity','rank_sensitivity'):raise ValueError('Only pooled sensitivity stages are prepared.')
    matches=[job for job in plan['jobs'] if job['name']==options.job]
    if len(matches)!=1:raise ValueError('Expected exactly one frozen job.')
    job=matches[0];parameters=job['parameters']
    if not parameters['use_swa']:raise ValueError('This wrapper preserves the existing SWA-enabled protocol.')
    if not options.execute:
        print(json.dumps({'status':'prepared_not_run','job':job['name'],'swa_tracking_preserved':True,
            'ordinary_best_test_preserved':True,'optional_swa_weight_serialization':False,'gpu_jobs_started':0}));return
    if sys.platform!='linux' or not options.absolute_deadline or not options.authorization_reference:
        parser.error('New explicit authorization and a fixed deadline on Linux are required.')
    if (plan.get('checkpoint_storage_profile')!='ordinary_best_only_with_swa_metrics'
        or plan['required_code_sha256'].get('scripts/train_pooled_storage_review.py')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise ValueError('Storage policy and wrapper source must be explicitly sealed in a new plan; old plans cannot be reused for execution.')
    deadline=datetime.fromisoformat(options.absolute_deadline)
    if deadline.tzinfo is None:raise ValueError('Deadline requires a timezone.')
    seconds=(deadline-datetime.now(timezone.utc)).total_seconds()
    if not 0<seconds<=12*3600 or os.getsid(0)!=os.getpid():raise ValueError('Expired/invalid deadline or nonisolated training session.')
    import signal
    signal.signal(signal.SIGALRM,lambda *_:os.killpg(os.getpgrp(),signal.SIGKILL))
    signal.setitimer(signal.ITIMER_REAL,seconds)
    os.chdir(ROOT);sys.path.insert(0,str(ROOT))
    from baseline_queue_locks import acquire_queue_locks
    locks=acquire_queue_locks((ROOT/'results/reruns/queue.lock',ROOT/'results/second_review/queue.lock'),options.inherited_lock_fds)
    output=(ROOT/parameters['output_dir']).resolve()
    if not output.is_relative_to((ROOT/'results/followup').resolve()):raise ValueError('Unexpected future pooled output directory.')
    if (output/'config.json').exists():raise ValueError('Existing config refused; preserve partial run for explicit same-deadline recovery review.')
    for key,digest in (('data_json','data_sha256'),('prior_file','prior_sha256')):
        if hashlib.sha256((ROOT/parameters[key]).read_bytes()).hexdigest()!=job[digest]:raise ValueError('Frozen input changed.')
    for name,digest in plan['required_code_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+name)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Authorized CUDA GPU required; no CPU training fallback.')
    import train as base
    output.mkdir(parents=True,exist_ok=True)
    metadata={'job':job['name'],'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'absolute_deadline':deadline.astimezone(timezone.utc).isoformat(),'authorization_reference':options.authorization_reference,
        'policy':'Keep all training/SWA computation, dev evaluation and ordinary-best test checkpoint; omit only optional SWA file serialization',
        'existing_weight_files_deleted_or_moved':False,'optional_swa_serializations_omitted':[],
        'training_completed':False,'gpu_runtime_validated_before_this_run':False}
    def record():
        temporary=output/'checkpoint_storage_policy.tmp'
        temporary.write_text(json.dumps(metadata,indent=2));temporary.replace(output/'checkpoint_storage_policy.json')
    original_save=torch.save
    def save(payload,destination,*args,**kwargs):
        if isinstance(destination,(str,os.PathLike)):
            file=Path(destination)
            if file.parent.resolve()==output:
                state=payload.get('model_state_dict') if isinstance(payload,dict) else None
                if state is not None:
                    tensor_bytes=sum(value.numel()*value.element_size() for value in state.values() if torch.is_tensor(value))
                    if file.name=='swa_model.pt':
                        metadata['optional_swa_serializations_omitted'].append({'tensor_bytes':tensor_bytes,
                            'val_metrics':{key:float(value) for key,value in payload['val_metrics'].items()}})
                        record();print('Optional SWA weight file omitted by declared storage policy; dev metrics retained.',flush=True);return
                    if file.name=='best_model.pt':
                        disk=file.resolve()
                        while not disk.exists():disk=disk.parent
                        required=tensor_bytes+1024**2+2*1024**3
                        if shutil.disk_usage(disk).free<required:raise RuntimeError('Insufficient existing disk for actual ordinary checkpoint tensors plus reserve.')
        return original_save(payload,destination,*args,**kwargs)
    parent=base.Trainer
    class Trainer(parent):
        def __init__(self,*args,**kwargs):
            training_args=kwargs['args']
            training_args.checkpoint_storage_policy=metadata['policy']
            training_args.storage_plan_sha256=metadata['plan_sha256']
            training_args.absolute_deadline=metadata['absolute_deadline']
            super().__init__(*args,**kwargs)
    base.Trainer=Trainer;torch.save=save
    command=['scripts/train.py']
    for key,value in parameters.items():
        if isinstance(value,bool):
            if value:command.append('--'+key)
        else:command.extend(('--'+key,str(value)))
    sys.argv=command;record()
    try:
        base.main();metadata['training_completed']=True
    finally:
        torch.save=original_save;record()


if __name__=='__main__':main()
