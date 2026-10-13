"""Prospective MIntOOD matched two-pass driver; default starts no compute.

Not run or runtime validated. Requires a new sealed bounded supervisor/allocation.
"""
import argparse,json,os,sys,time
from pathlib import Path
from datetime import datetime,timezone

def prepared_protocol(split,seed,output):
    return dict(backbone='pinned_mintood_init_complete_adaptation',
        source_revision='ad0cd88980c970366491a3d17268932b86bdeed1',
        text_encoder='bert-large-uncased',audio_dim=768,video_dim=256,hidden_dim=1024,
        dropout=.1,freeze_bert=False,num_classes=20,seed=seed,output_dir=output,
        data_json=split,prior_file='data/prepared_mintrec1/controls/all_true.jsonl',
        audio_feats='data/prepared_mintrec1/audio_sequence_feats.pkl',
        video_feats='data/prepared_mintrec1/video_sequence_feats.pkl',
        batch_size=32,eval_batch_size=8,test_batch_size=8,epochs=100,patience=8,
        lr=3e-5,weight_decay=.01,warmup_ratio=.1,
        optimizer='Transformers4.45.2 AdamW correct_bias=True; native bias/LayerNorm no decay groups',
        scheduler='floor(N/32)*100 steps; int(N*100*0.1/32) warmup; step in BOTH epoch passes, native behavior',
        epoch_order=['binary CE original+mixtures','original CE plus independently mixed two-view SupCon'],
        synthesis='Training minibatch only; N mixtures; Dirichlet alpha2,2-3 distinct source indices,>=2 source classes',
        contrastive_temperature=2,initialization='HF4.45.2 _fast_init=False; changed RNG stream disclosed',
        checkpoint='best dev closed-set ACC; snapshot encoder and BOTH heads together',
        native_binary_dev_diagnostic='Omitted; no labeled dev mixture diagnostic or its RNG consumption. Dev/test prediction sees no labels.',
        mask_compatibility='Shortest finite valid-token count replaces overflowing additive-mask sum, native argmin choice retained',
        sequence_caps=[50,400,180],padding='NaN suffix converted to native zero_end; first-frame truncation',
        auxiliary_selection=False,rdrop=False,swa=False,grad_clip=0.,
        numerical_policy='Finite initial parameters, loss and gradients; abort before optimizer update; native cosine denominator unchanged')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--protocol',choices=('original','debiased'),default='original')
    parser.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    parser.add_argument('--output-dir');parser.add_argument('--absolute-deadline')
    parser.add_argument('--authorization-reference');parser.add_argument('--inherited-lock-fds')
    o=parser.parse_args()
    split='data/prepared_mintrec1/original/data.json'
    output=o.output_dir or f'results/journal_mintrec1_baselines_v1/journal_mintrec1_mintood_seed{o.seed}'
    protocol=prepared_protocol(split,o.seed,output)
    if not o.execute:
        print(json.dumps({'status':'prepared_not_run','protocol':protocol,'gpu_jobs_started':0}));return
        from journal_mintrec1_baselines_guard import check as _guard; _guard(options, output)
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));os.chdir(root)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    from baseline_queue_locks import acquire_queue_locks
    locks=acquire_queue_locks((root/'results/reruns/queue.lock',root/'results/second_review/queue.lock'),o.inherited_lock_fds)
    if (root/output/'config.json').exists():raise ValueError('Existing configuration refused; no implicit rerun')
    import torch,transformers
    if transformers.__version__!='4.45.2':raise ValueError('Pinned Transformers4.45.2 required')
    if not torch.cuda.is_available():raise RuntimeError('Authorized CUDA required; no CPU training')
    from transformers.optimization import AdamW,get_linear_schedule_with_warmup
    import train as base
    from journal_mintrec1_sequence_dataset import official_mult_dataset_class
    from models.mintood_journal_mintrec1 import MIntOODMIntRec1ForSATI as MIntOODForSATI
    protocol.update(absolute_deadline=None,authorization_reference=o.authorization_reference)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();created=[];train_sizes=[]
    def model_factory(**kw):
        if kw.pop('dropout')!=.1:raise ValueError('Fixed MLP dropout required')
        kw.pop('tensor_rank')
        model=MIntOODForSATI(**kw)
        for n,p in model.named_parameters():
            if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite initial parameter '+n)
        created.append(model);return model
    parent_dataset=official_mult_dataset_class(base.MIntRecDataset)
    class Dataset(parent_dataset):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            if self.split=='train':train_sizes.append(len(self))
        def __getitem__(self,i):
            item=super().__getitem__(i);item['sample_id']=self.samples[i]['id'];return item
    original_collate=base.collate_fn
    def collate(batch):
        result=original_collate(batch);result['sample_ids']=[r['sample_id'] for r in batch];return result
    def optimizer_factory(parameters,lr,weight_decay):
        no_decay=('bias','LayerNorm.bias','LayerNorm.weight')
        groups=[{'params':[p for n,p in created[-1].named_parameters() if not any(k in n for k in no_decay)],'weight_decay':weight_decay},
                {'params':[p for n,p in created[-1].named_parameters() if any(k in n for k in no_decay)],'weight_decay':0.}]
        return AdamW(groups,lr=lr,correct_bias=True)
    def scheduler_factory(optimizer,**kw):
        if len(train_sizes)!=1:raise ValueError('Expected one training dataset')
        n=train_sizes[0]
        return get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(n*100*.1/32),num_training_steps=int(n/32)*100)
    parent_trainer=base.Trainer
    class Trainer(parent_trainer):
        def __init__(self,*a,**kw):
            args=kw['args'];args.use_swa=False;args.backbone=protocol['backbone']
            args.mintood_protocol=protocol;args.bert_loading_audit=created[-1].loading_audit
            super().__init__(*a,**kw)
        def train_epoch(self,epoch):
            self.model.train();self.model.multiclass_classifier.eval()
            totals={'binary':0.,'multiclass':0.};weights={'binary':0,'multiclass':0};correct=0
            trace=self.output_dir/'mixture_trace.jsonl'
            for stage in ('binary','multiclass'):
                if stage=='multiclass':
                    self.model.binary_classifier.eval();self.model.encoder.train();self.model.multiclass_classifier.train()
                for index,batch in enumerate(self.train_loader):
                    tensors=[batch[k].to(self.device) for k in ('input_ids','attention_mask','audio','video','labels')]
                    self.optimizer.zero_grad()
                    fn=self.model.binary_training_loss if stage=='binary' else self.model.multiclass_training_loss
                    loss,details=fn(*tensors)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite '+stage+' loss')
                    record={k:(float(v.detach()) if torch.is_tensor(v) else v) for k,v in details.items()}
                    record.update(epoch=epoch,stage=stage,batch=index,source_ids=batch['sample_ids'],loss=float(loss.detach()))
                    with trace.open('a',encoding='utf-8') as handle:handle.write(json.dumps(record)+'\n')
                    loss.backward()
                    for n,p in self.model.named_parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient '+n)
                    self.optimizer.step();self.scheduler.step()
                    weight=details['n_mixed'] if stage=='binary' else details['n_original']
                    totals[stage]+=float(loss.detach())*weight;weights[stage]+=weight
                    if stage=='multiclass':correct+=details['original_correct']
                self.history[stage+'_loss'].append(totals[stage]/weights[stage])
                (self.output_dir/'history.json').write_text(json.dumps(dict(self.history),indent=2))
            return {'loss':totals['multiclass']/weights['multiclass'],'accuracy':correct/weights['multiclass']}
    loader=base.DataLoader
    def loader_factory(dataset,*a,**kw):
        if dataset.split!='train':kw['batch_size']=8
        return loader(dataset,*a,**kw)
    base.DataLoader=loader_factory;base.collate_fn=collate;base.MIntRecDataset=Dataset
    base.SATI=model_factory;base.SATILoss=lambda **kw:torch.nn.Identity()
    base.AdamW=optimizer_factory;base.get_linear_schedule_with_warmup=scheduler_factory;base.Trainer=Trainer
    sys.argv=[sys.argv[0],'--data_json',split,'--audio_feats',protocol['audio_feats'],'--video_feats',protocol['video_feats'],
        '--prior_file',protocol['prior_file'],'--output_dir',output,'--text_encoder',protocol['text_encoder'],
        '--hidden_dim','1024','--dropout','.1','--label_smoothing','0','--lambda_rdrop','0','--lambda_aux','0',
        '--batch_size','32','--epochs','100','--patience','8','--lr','3e-5','--weight_decay','.01',
        '--warmup_ratio','.1','--grad_clip','0','--seed',str(o.seed),'--num_workers','8']
    complete=False
    try:base.main();complete=True
    finally:
        folder=root/output;folder.mkdir(parents=True,exist_ok=True)
        (folder/'resource_usage_training.json').write_text(json.dumps(dict(completed=complete,
            measurement='Torch allocator peak, not whole device VRAM',peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),measured_wall_seconds=time.monotonic()-start,
            absolute_deadline=protocol['absolute_deadline']),indent=2))

if __name__=='__main__':main()
