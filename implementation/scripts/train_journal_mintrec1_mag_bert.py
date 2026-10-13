"""Prepared MAG-BERT matched driver; default starts no model/GPU execution.

Native MAG gating, soft CTC alignment, CE, AdamW correct_bias=False, grouping
and floor-based warmup counts retained. Modern data/tokenization and best dev
accuracy (native dev F1) are disclosed adaptations. No encoder hook added.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


def prepared_protocol(split, seed, output):
    return {
        'backbone': 'pinned_mag_bert_init_complete_adaptation',
        'source_revision': '6bc239695df342c22c324bb3a76789fa295ff999',
        'text_length': 50, 'audio_length': 400, 'video_length': 180,
        'padding': 'NaN metadata restored to native zero_end before alignment', 'truncation': 'first_frames',
        'alignment': 'Native CTC-style soft alignment, no CTC supervision or real timestamps; padded frames retained',
        'encoder_output': 'Native indexing/slicing retained; Transformers4.45.2 ModelOutput supports tuple slicing, no hook added',
        'text_encoder': 'bert-large-uncased', 'audio_dim': 768, 'video_dim': 256,
        'hidden_dim': 1024, 'dropout': .5, 'beta_shift': .005,
        'classification_dropout': 'BERT config hidden_dropout_prob, independent from gate dropout0.5',
        'batch_size': 16, 'eval_batch_size': 8, 'test_batch_size': 8, 'epochs': 100, 'patience': 8,
        'lr': 2e-5, 'weight_decay': .03, 'warmup_ratio': .1,
        'optimizer': 'Transformers4.45.2 AdamW correct_bias=False, native defaults; not torch AdamW bias-corrected adaptation',
        'optimizer_groups': 'Native bias/LayerNorm.bias/LayerNorm.weight no decay',
        'scheduler': 'Native floor(n_train/16)*100 linear steps, int(n_train*100*0.1/16) warmup',
        'checkpoint': 'ordinary best dev accuracy; native source selects dev F1',
        'grad_clip': 0., 'main_loss': 'CE', 'label_smoothing': 0., 'focal': False, 'rdrop': 0., 'swa': False,
        'fixed_num_classes': 20, 'freeze_bert': False, 'tensor_rank_unused': True,
        'initialization': 'HF4.45.2 _fast_init=False; native CTC LSTM constructors; changed RNG stream',
        'numerical_guards': 'Finite initial parameters/logits/loss/gradients; not yet runtime validated',
        'prior_consumed_by_objective': False, 'selection_heads_added': False,
        'tokenization': 'SATI AutoTokenizer, not native manual tokenizer',
        'data_json': split, 'prior_file': 'data/prepared_mintrec1/controls/all_true.jsonl',
        'audio_feats': 'data/prepared_mintrec1/audio_sequence_feats.pkl',
        'video_feats': 'data/prepared_mintrec1/video_sequence_feats.pkl',
        'seed': seed, 'output_dir': output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--absolute-deadline')
    parser.add_argument('--authorization-reference')
    parser.add_argument('--inherited-lock-fds')
    parser.add_argument('--protocol', choices=('original', 'debiased'), default='original')
    parser.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
    parser.add_argument('--output-dir')
    options = parser.parse_args()
    split = 'data/prepared_mintrec1/original/data.json'
    output = options.output_dir or f'results/journal_mintrec1_baselines_v1/journal_mintrec1_mag_bert_seed{options.seed}'
    protocol = prepared_protocol(split, options.seed, output)
    if not options.execute:
        print(json.dumps({'status': 'prepared_not_run', 'gpu_runtime_validated': False, 'protocol': protocol})); return
        from journal_mintrec1_baselines_guard import check as _guard; _guard(options, output)
    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root)); os.chdir(root)
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from baseline_queue_locks import acquire_queue_locks
    locks = acquire_queue_locks((root/'results/reruns/queue.lock', root/'results/second_review/queue.lock'), options.inherited_lock_fds)
    if (root/output/'config.json').exists(): raise ValueError('Existing configuration refused; never silently retrain.')
    import torch
    import transformers
    if transformers.__version__ != '4.45.2': raise ValueError('Pinned Transformers4.45.2 required for prepared native optimizer.')
    from transformers.optimization import AdamW as NativeAdamW, get_linear_schedule_with_warmup
    if not torch.cuda.is_available(): raise RuntimeError('Authorized CUDA required; no CPU training fallback.')
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    import train as base
    from journal_mintrec1_sequence_dataset import official_mult_dataset_class
    from models.mag_bert_repaired_journal_mintrec1 import MAGBERTMIntRec1ForSATI as MAGBERTForSATI, MAGBERTObjective
    protocol.update(absolute_deadline=None, authorization_reference=options.authorization_reference)
    created = []; train_sizes = []
    def model_factory(**kwargs):
        model = MAGBERTForSATI(**kwargs); created.append(model); return model
    parent_dataset = official_mult_dataset_class(base.MIntRecDataset)
    class Dataset(parent_dataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.split == 'train': train_sizes.append(len(self))
    def optimizer_factory(parameters, lr, weight_decay):
        no_decay = ('bias', 'LayerNorm.bias', 'LayerNorm.weight')
        groups = [{'params': [p for n,p in created[-1].named_parameters() if not any(k in n for k in no_decay)], 'weight_decay': weight_decay},
                  {'params': [p for n,p in created[-1].named_parameters() if any(k in n for k in no_decay)], 'weight_decay': 0.}]
        return NativeAdamW(groups, lr=lr, correct_bias=False)
    def scheduler_factory(optimizer, num_warmup_steps, num_training_steps):
        if len(train_sizes) != 1: raise ValueError('Expected one frozen training dataset.')
        n = train_sizes[0]
        return get_linear_schedule_with_warmup(optimizer,
            num_warmup_steps=int(n*100*.1/16), num_training_steps=int(n/16)*100)
    parent = base.Trainer
    class Trainer(parent):
        def __init__(self, *args, **kwargs):
            args_ = kwargs['args']; args_.use_swa = False
            args_.backbone = protocol['backbone']; args_.mag_bert_protocol = protocol
            args_.bert_loading_audit = created[-1].loading_audit
            super().__init__(*args, **kwargs)
    original_loader = base.DataLoader
    def loader_factory(dataset, *args, **kwargs):
        if dataset.split != 'train': kwargs['batch_size'] = 8
        return original_loader(dataset, *args, **kwargs)
    base.DataLoader = loader_factory; base.MIntRecDataset = Dataset
    base.SATI = model_factory; base.SATILoss = lambda **kwargs: MAGBERTObjective()
    base.AdamW = optimizer_factory; base.get_linear_schedule_with_warmup = scheduler_factory; base.Trainer = Trainer
    sys.argv = [sys.argv[0], '--data_json', split, '--audio_feats', protocol['audio_feats'], '--video_feats', protocol['video_feats'],
        '--prior_file', protocol['prior_file'], '--output_dir', output, '--text_encoder', protocol['text_encoder'],
        '--hidden_dim', '1024', '--dropout', '.5', '--label_smoothing', '0', '--lambda_rdrop', '0', '--lambda_aux', '0',
        '--batch_size', '16', '--epochs', '100', '--patience', '8', '--lr', '2e-5', '--weight_decay', '.03',
        '--warmup_ratio', '.1', '--grad_clip', '0', '--seed', str(options.seed), '--num_workers', '8']
    completed = False
    try:
        base.main(); completed = True
    finally:
        folder = root/output; folder.mkdir(parents=True, exist_ok=True)
        (folder/'resource_usage_training.json').write_text(json.dumps({'completed': completed,
            'measurement': 'Torch allocator peak, not total device VRAM', 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'measured_wall_seconds': time.monotonic()-started,
            'absolute_deadline': protocol['absolute_deadline']}, indent=2))


if __name__ == '__main__': main()
