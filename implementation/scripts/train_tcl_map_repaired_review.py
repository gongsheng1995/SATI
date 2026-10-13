"""Prepared TCL-MAP matched driver; default starts no model/GPU execution.

Native prompt/MAG, soft alignment, CE+instance contrastive loss, AdamW grouping
and floor-based warmup counts retained. Ground truth enters only training teacher views. Modern data/tokenization and best dev
accuracy (native dev F1) are disclosed adaptations. Eval computes no teacher branch. No encoder hook added.
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
        'initialization': 'HF4.45.2 _fast_init=False; native custom constructors retained; different RNG stream disclosed',
        'numerical_checks': 'Finite parameters/logits/context/teacher hidden/loss/gradients; abort before optimizer update',
        'backbone': 'pinned_tcl_map_init_repaired_adaptation',
        'source_revision': '37095a591c9eb8d93f71190a78ee07bd8e78a6ea',
        'text_length': 50, 'total_prompt_text_length': 57, 'prompt_length': 3, 'label_length': 4, 'audio_length': 400, 'video_length': 180,
        'padding': 'NaN metadata restored to native zero_end before alignment', 'truncation': 'first_frames',
        'alignment': 'Native CTC-style soft alignment, no CTC supervision or real timestamps; padded frames retained',
        'encoder_output': 'Native indexing/slicing retained; Transformers4.45.2 ModelOutput supports tuple slicing, no hook added',
        'text_encoder': 'bert-large-uncased', 'audio_dim': 768, 'video_dim': 256,
        'hidden_dim': 1024, 'dropout': .5, 'beta_shift': .006,
        'classification_dropout': 'BERT config hidden_dropout_prob, independent from gate dropout0.5',
        'batch_size': 16, 'eval_batch_size': 8, 'test_batch_size': 8, 'epochs': 100, 'patience': 8,
        'lr': 2e-5, 'weight_decay': .1, 'warmup_ratio': .1,
        'optimizer': 'Transformers4.45.2 AdamW correct_bias=False, native defaults; not torch AdamW bias-corrected adaptation',
        'optimizer_groups': 'Native bias/LayerNorm.bias/LayerNorm.weight no decay',
        'scheduler': 'Native floor(n_train/16)*100 linear steps, int(n_train*100*0.1/16) warmup',
        'checkpoint': 'ordinary best dev accuracy; native source selects dev F1',
        'grad_clip': 0., 'main_loss': 'CE+native SupCon(two normalized views, temperature0.5, no class-positive labels)', 'label_smoothing': 0., 'focal': False, 'rdrop': 0., 'swa': False,
        'fixed_num_classes': 30, 'freeze_bert': False, 'tensor_rank_unused': True, 'encoders': 2, 'prediction_label_view': False,
        'literal_MASK_token': 'Native literal MASK retained as UNK100; label slots are MASK103',
        'teacher_view': 'Canonical training label WordPieces padded to4 with native literalMASK/UNK100; absent from eval',
        'prompt': '3 learned tokens,5 transformer layers,8 heads, similarity+CTC soft alignment, native gamma1e-4',
        'prior_consumed_by_objective': False, 'selection_heads_added': False,
        'tokenization': 'SATI AutoTokenizer base content50 plus native3+4 slots; frozen canonical label BertTokenizer audit',
        'data_json': split, 'prior_file': 'data/controls/all_true.jsonl',
        'audio_feats': 'data/retrain_inputs/in-scope_audio_data_wavlm_feats_binary.pkl',
        'video_feats': 'data/retrain_inputs/in-scope_video_data_swin_roi_binary.pkl',
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
    split = 'data/retrain_inputs/data.json' if options.protocol == 'original' else 'data/debiased_mintrec2/data.json'
    output = options.output_dir or f'results/tcl_map_init_repaired_review/{options.protocol}_native_seed{options.seed}'
    protocol = prepared_protocol(split, options.seed, output)
    if not options.execute:
        print(json.dumps({'status': 'prepared_not_run', 'gpu_runtime_validated': False, 'protocol': protocol})); return
    if sys.platform != 'linux' or not options.absolute_deadline or not options.authorization_reference:
        parser.error('Explicit new allocation/reference and fixed deadline on Linux required.')
    deadline = datetime.fromisoformat(options.absolute_deadline)
    if deadline.tzinfo is None: parser.error('Timezone-aware deadline required.')
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if not 0 < remaining <= 12*3600: raise ValueError('Expired or over12h deadline; no renewal allowed.')
    if os.getsid(0) != os.getpid(): raise RuntimeError('Use an isolated bounded supervisor session.')
    import signal
    signal.signal(signal.SIGALRM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    signal.setitimer(signal.ITIMER_REAL, remaining)
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
    from train_official_mult_review import official_mult_dataset_class
    from models.tcl_map_repaired_baseline import TCLMAPForSATI, TCLMAPObjective
    protocol.update(absolute_deadline=deadline.astimezone(timezone.utc).isoformat(), authorization_reference=options.authorization_reference)
    created = []; train_sizes = []; training_labels = []
    def model_factory(**kwargs):
        model = TCLMAPForSATI(**kwargs, label_list=training_labels[0]); created.append(model); return model
    parent_dataset = official_mult_dataset_class(base.MIntRecDataset)
    class Dataset(parent_dataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.split == 'train':
                train_sizes.append(len(self)); training_labels.append(self.label_list)
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
            args_.backbone = protocol['backbone']; args_.tcl_map_protocol = protocol
            args_.bert_loading_audit = created[-1].loading_audit
            super().__init__(*args, **kwargs)
    original_loader = base.DataLoader
    def loader_factory(dataset, *args, **kwargs):
        if dataset.split != 'train': kwargs['batch_size'] = 8
        return original_loader(dataset, *args, **kwargs)
    base.DataLoader = loader_factory; base.MIntRecDataset = Dataset
    base.SATI = model_factory; base.SATILoss = lambda **kwargs: TCLMAPObjective(created[-1])
    base.AdamW = optimizer_factory; base.get_linear_schedule_with_warmup = scheduler_factory; base.Trainer = Trainer
    sys.argv = [sys.argv[0], '--data_json', split, '--audio_feats', protocol['audio_feats'], '--video_feats', protocol['video_feats'],
        '--prior_file', protocol['prior_file'], '--output_dir', output, '--text_encoder', protocol['text_encoder'],
        '--hidden_dim', '1024', '--dropout', '.5', '--label_smoothing', '0', '--lambda_rdrop', '0', '--lambda_aux', '0',
        '--batch_size', '16', '--epochs', '100', '--patience', '8', '--lr', '2e-5', '--weight_decay', '.1',
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
