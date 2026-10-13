"""Prepared allocation across existing data/system disks; no deletion or moves.

Only a separately authorized Linux supervisor may call commit_layout. Planning
reserves one ordinary checkpoint per missing job and2GiB per filesystem. No
SWA weights are budgeted for the explicit ordinary-only storage profile.
"""
import os
from pathlib import Path
import re
import shutil
import sys


def ancestor(path):
    while not path.exists():path=path.parent
    return path


def prepare_layout(root,jobs,stage,checkpoint_bytes,reserve_bytes=2*1024**3):
    if sys.platform!='linux':raise RuntimeError('Existing-disk allocation is only prepared for the authorized Linux host.')
    root=root.resolve()
    if root!=Path('/root/autodl-tmp/SATI') or stage not in ('journal_mintrec1_sati_seed42','journal_mintrec1_sati_seed43','journal_mintrec1_sati_seed44'):
        raise ValueError('Unexpected workspace or pooled stage.')
    weight_root=Path('/root/sati_followup_weights')
    if weight_root.resolve()!=weight_root:raise ValueError('System weight root must not be redirected by a symlink.')
    disks={}
    for directory in (root/'results/followup',weight_root):
        present=ancestor(directory);device=present.stat().st_dev
        if device not in disks:
            free=shutil.disk_usage(present).free
            disks[device]={'device':device,'path':str(present),'free_bytes':free,
                'reserve_bytes':reserve_bytes,'remaining_bytes':free-reserve_bytes}
    assignments=[]
    for job in jobs:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',job['name']):raise ValueError('Unsafe job name.')
        run=(root/job['parameters']['output_dir']).resolve()
        if not run.is_relative_to(root/'results/followup'):raise ValueError('Unexpected future output directory.')
        link=run/'best_model.pt'
        if os.path.lexists(link):
            target=link.resolve()
            if not (target.is_relative_to(root/'results/followup') or target.is_relative_to(weight_root)):
                raise ValueError('Existing checkpoint redirects outside the named output roots.')
            if link.exists():
                assignments.append({'job':job['name'],'link':str(link),'target':str(target),
                    'existing_file_preserved':True,'create_symlink':False});continue
            device=ancestor(target).stat().st_dev
            if device not in disks:raise ValueError('Existing dangling link targets an unexpected filesystem.')
            chosen=disks[device]
            create=False
        else:
            available=[disk for disk in disks.values() if disk['remaining_bytes']>=checkpoint_bytes]
            if not available:raise RuntimeError('Insufficient existing disks for all planned ordinary checkpoints and reserve.')
            chosen=max(available,key=lambda disk:(disk['remaining_bytes'],-disk['device']))
            if chosen['device']==ancestor(run).stat().st_dev:target=link;create=False
            else:target=weight_root/stage/job['name']/'best_model.pt';create=True
        chosen['remaining_bytes']-=checkpoint_bytes
        if chosen['remaining_bytes']<0:raise RuntimeError('Existing links exceed remaining disk capacity.')
        assignments.append({'job':job['name'],'link':str(link),'target':str(target),
            'existing_file_preserved':False,'create_symlink':create,'device':chosen['device'],
            'reserved_checkpoint_bytes':checkpoint_bytes})
    return {'stage':stage,'assignments':assignments,'filesystems':list(disks.values()),
        'checkpoint_profile':'ordinary_best_only_with_swa_metrics','no_deletion_or_moves':True}


def commit_layout(layout):
    if sys.platform!='linux':raise RuntimeError('Prepared layout cannot be committed on the local Windows workspace.')
    weight_root=Path('/root/sati_followup_weights')
    data_root=Path('/root/autodl-tmp/SATI/results/followup').resolve()
    # Validate every target before the first mutation. Existing files are retained.
    for item in layout['assignments']:
        link=Path(item['link']);target=Path(item['target'])
        if not link.parent.resolve().is_relative_to(data_root):raise ValueError('Link path left the declared results root.')
        if item['create_symlink']:
            if not target.parent.resolve().is_relative_to(weight_root) or os.path.lexists(link) or target.exists():
                raise ValueError('Target changed after preflight; refusing overwrite.')
    for item in layout['assignments']:
        link=Path(item['link']);target=Path(item['target'])
        link.parent.mkdir(parents=True,exist_ok=True)
        if item['create_symlink']:
            target.parent.mkdir(parents=True,exist_ok=True)
            link.symlink_to(target)
