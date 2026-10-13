"""Prepared official MulT modern training adaptation; default has no GPU work.

Native pinned MIntRec model with MIntRec2 model hyperparameters. Shared SATI
trainer AdamW/linear warmup/norm clip replaces native Adam/plateau/value clip,
which must be disclosed. No trained strong-baseline results exist yet.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time


def official_mult_dataset_class(parent):
    import numpy as np
    import torch
    class Dataset(parent):
        def __init__(self,*args,**kwargs):
            kwargs['max_length']=50;super().__init__(*args,**kwargs)
            if self.num_classes!=20:raise ValueError('Prepared MIntRec1 MulT protocol requires fixed20 classes.')
            if self.split=='train' and any(row['id'] not in self.prior_dict for row in self.samples):
                raise ValueError('Missing frozen training prior; implicit fallback refused.')
        @staticmethod
        def sequence(feature,cap,dim):
            if isinstance(feature,list):feature=np.concatenate(feature,axis=0)
            value=torch.as_tensor(feature,dtype=torch.float32)
            if value.ndim==3 and value.shape[1]==1:value=value.squeeze(1)
            if value.ndim!=2 or value.shape[-1]!=dim or not len(value) or not torch.isfinite(value).all():
                raise ValueError('Expected finite nonempty original frame sequence with frozen dimensions.')
            result=value.new_full((cap,dim),float('nan'))
            count=min(len(value),cap);result[:count]=value[:count]
            return result
        def _get_audio_feat(self,sid):return self.sequence(self.audio_feats[sid],400,768)
        def _get_video_feat(self,sid):return self.sequence(self.video_feats[sid],180,256)
    return Dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--absolute-deadline')
    parser.add_argument('--authorization-reference')
    parser.add_argument('--inherited-lock-fds',help='Bounded supervisor lock descriptors; never substitutes for authorization.')
    parser.add_argument('--condition',choices=('native','none','all','sati'),default='native')
    parser.add_argument('--protocol',choices=('original','debiased'),default='original')
    parser.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    parser.add_argument('--output-dir')
    options=parser.parse_args()
    split='data/prepared_mintrec1/original/data.json'
    prior='data/llm_prior.jsonl' if options.condition=='sati' else 'data/prepared_mintrec1/controls/all_true.jsonl'
    output=options.output_dir or f'results/journal_mintrec1_baselines_v1/journal_mintrec1_mult_seed{options.seed}'
    protocol={'backbone':'pinned_official_mult_modern_training_adaptation',
        'source_revision':'6bc239695df342c22c324bb3a76789fa295ff999','condition':options.condition,
        'text_length':50,'audio_length':400,'video_length':180,'padding':'Native zero_end; NaN suffix markers in loader only, replaced by zeros before native forward',
        'truncation':'first_frames','hidden_dim':80,'heads':2,'cross_layers':2,'memory_layers':3,
        'text_conv_kernel':5,'audio_video_conv_kernel':1,'causal_attention':True,
        'model_hyperparameters':'MIntRec2 official selected settings reused unchanged for the MIntRec1 instantiation; MIntRec1 defaults120/8/8 not substituted',
        'dropouts':{'attention_text':0.,'attention_audio_video':.2,'relu':.3,'embedding':.1,'residual':0.,'output':.2,'text':.4},
        'batch_size':16,'epochs':40,'patience':8,'lr':5e-6,'weight_decay':.01,'warmup_ratio':.15,
        'grad_clip_norm':.5,'optimizer':'torch AdamW and linear warmup; native manager Adam/ReduceLROnPlateau/value clip differs',
        'focal':False,'label_smoothing':0.,'rdrop':0.,'swa':False,'main_loss':'CE',
        'lambda_aux':.3,'auxiliary_loss':'Each modality selected-mean CE on shared native convolutional projection valid-length mean; no text auxiliary',
        'matched_aux_architecture':'none/all/sati all retain identical heads; native bare model has no added heads',
        'checkpoint':'ordinary best dev accuracy','fixed_num_classes':20,
        'tokenization':'SATI AutoTokenizer on the same transcripts; not source manual tokenization',
        'alignment_caution':'Native cross-attention uses sequences but no exact timestamps or main key padding masks',
        'data_json':split,'prior_file':prior,'seed':options.seed,'output_dir':output}
    if not options.execute:
        print(json.dumps({'status':'prepared_not_run','gpu_runtime_validated':False,'protocol':protocol}));return
        from journal_mintrec1_baselines_guard import check as _guard; _guard(options, output)
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));os.chdir(root)
    from baseline_queue_locks import acquire_queue_locks
    locks=acquire_queue_locks((root/'results/reruns/queue.lock',root/'results/second_review/queue.lock'),options.inherited_lock_fds)
    if (root/output/'config.json').exists():raise ValueError('Existing run refused; never silently retrain/overwrite.')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('No authorized CUDA GPU; CPU training fallback refused.')
    torch.cuda.reset_peak_memory_stats();resource_started=time.monotonic()
    import train as base
    from models.official_mult_journal_mintrec1 import OfficialMulTForSATI,OfficialMulTTrainerObjective
    protocol.update(absolute_deadline=None,authorization_reference=options.authorization_reference)
    parent=base.Trainer
    class Trainer(parent):
        def __init__(self,*args,**kwargs):
            training_args=kwargs['args'];training_args.use_swa=False
            training_args.backbone=protocol['backbone'];training_args.official_mult_protocol=protocol
            super().__init__(*args,**kwargs)
    base.MIntRecDataset=official_mult_dataset_class(base.MIntRecDataset)
    base.SATI=lambda **kwargs:OfficialMulTForSATI(**kwargs,with_auxiliary_heads=options.condition!='native')
    base.SATILoss=lambda **kwargs:OfficialMulTTrainerObjective(options.condition,.3)
    base.Trainer=Trainer
    sys.argv=[sys.argv[0],'--data_json',split,'--audio_feats','data/prepared_mintrec1/audio_sequence_feats.pkl',
        '--video_feats','data/prepared_mintrec1/video_sequence_feats.pkl','--prior_file',prior,'--output_dir',output,
        '--text_encoder','bert-large-uncased','--hidden_dim','80','--dropout','.2','--label_smoothing','0',
        '--lambda_rdrop','0','--lambda_aux','.3','--batch_size','16','--epochs','40','--patience','8',
        '--lr','5e-6','--weight_decay','.01','--warmup_ratio','.15','--grad_clip','.5',
        '--seed',str(options.seed),'--num_workers','8']
    completed=False
    try:
        base.main();completed=True
    finally:
        folder=root/output;folder.mkdir(parents=True,exist_ok=True)
        (folder/'resource_usage_training.json').write_text(json.dumps({'completed':completed,
            'measurement':'Torch process allocator peak, not total device VRAM or a sampled nvidia-smi peak',
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'measured_wall_seconds':time.monotonic()-resource_started,'absolute_deadline':protocol['absolute_deadline']},indent=2))


if __name__=='__main__':main()
