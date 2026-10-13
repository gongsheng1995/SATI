"""MISA length-corrected matched preparation; default starts no GPU work.

Retain the pinned native model/objective, Adam and exponential decay after
non-improving epochs. SATI data/tokenization and best dev accuracy selection
are declared adaptations; native source selects dev F1. No trained result yet.
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
        'backbone': 'pinned_misa_length_corrected_adaptation',
        'source_revision': '6bc239695df342c22c324bb3a76789fa295ff999',
        'text_length': 50, 'audio_length': 400, 'video_length': 180,
        'padding': 'NaN suffix length metadata converted to native zero_end before forward',
        'truncation': 'first_frames', 'length_adaptation': 'True retained A/V lengths replace native text lengths in packed RNNs; no timestamp alignment',
        'text_encoder': 'bert-large-uncased', 'audio_dim': 768, 'video_dim': 256,
        'hidden_dim': 256, 'dropout': .1, 'rnncell': 'lstm',
        'use_cmd_sim': False, 'reverse_grad_weight': .8,
        'loss_weights': {'diff': .7, 'similarity': .7, 'reconstruction': .6},
        'main_loss': 'CE', 'similarity': 'Native three-domain adversarial CE',
        'microbatching': 'None; six DiffLoss terms computed on entire batch',
        'batch_size': 16, 'eval_batch_size': 8, 'test_batch_size': 8,
        'epochs': 100, 'patience': 8, 'lr': 3e-5, 'weight_decay': 0.,
        'optimizer': 'torch.optim.Adam, native default betas/eps and no weight decay',
        'scheduler': 'ExponentialLR gamma0.5 after each non-improving dev epoch; no per-batch decay or warmup',
        'checkpoint': 'ordinary best dev accuracy; native source instead selects dev F1',
        'grad_clip': 0., 'focal': False, 'label_smoothing': 0., 'rdrop': 0., 'swa': False,
        'fixed_num_classes': 30, 'tensor_rank_unused': True,
        'prior_consumed_by_objective': False, 'selection_heads_added': False,
        'tokenization': 'SATI AutoTokenizer; not native manual tokenizer',
        'data_json': split, 'prior_file': 'data/controls/all_true.jsonl',
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
    output = options.output_dir or f'results/misa_review/{options.protocol}_native_seed{options.seed}'
    protocol = prepared_protocol(split, options.seed, output)
    if not options.execute:
        print(json.dumps({'status': 'prepared_not_run', 'gpu_runtime_validated': False, 'protocol': protocol})); return
    from completion_baseline_guard import check
    check(options,output)

    root = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(root)); os.chdir(root)
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    from baseline_queue_locks import acquire_queue_locks
    locks = acquire_queue_locks((root/'results/reruns/queue.lock', root/'results/second_review/queue.lock'), options.inherited_lock_fds)
    if (root/output/'config.json').exists(): raise ValueError('Existing configuration refused; no overwrite/retraining.')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Authorized CUDA required; CPU training fallback refused.')
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    import train as base
    from train_official_mult_review_completion import official_mult_dataset_class
    from models.misa_baseline import MISAForSATI, MISAObjective
    protocol.update(absolute_deadline=None, authorization_reference=options.authorization_reference)

    class EpochScheduler:
        """Trainer invokes step per batch; native MISA changes LR per bad epoch."""
        def __init__(self, optimizer): self.native = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.5)
        def step(self): pass
        def get_last_lr(self): return self.native.get_last_lr()
        def nonimproving_epoch(self): self.native.step()

    parent = base.Trainer
    class Trainer(parent):
        def __init__(self, *args, **kwargs):
            training_args = kwargs['args']; training_args.use_swa = False
            training_args.backbone = protocol['backbone']; training_args.misa_protocol = protocol
            super().__init__(*args, **kwargs)
            self.scheduler = EpochScheduler(self.optimizer)
        def evaluate(self, dataloader):
            metrics = super().evaluate(dataloader)
            if dataloader is self.val_loader and metrics['accuracy'] <= self.best_val_acc:
                self.scheduler.nonimproving_epoch()
            return metrics

    original_loader = base.DataLoader
    def loader_factory(dataset, *args, **kwargs):
        if dataset.split != 'train': kwargs['batch_size'] = 8
        return original_loader(dataset, *args, **kwargs)
    base.DataLoader = loader_factory
    base.MIntRecDataset = official_mult_dataset_class(base.MIntRecDataset)
    base.SATI = MISAForSATI
    base.SATILoss = lambda **kwargs: MISAObjective()
    base.AdamW = lambda parameters, lr, weight_decay: torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    base.Trainer = Trainer
    sys.argv = [sys.argv[0], '--data_json', split,
        '--audio_feats', 'data/retrain_inputs/in-scope_audio_data_wavlm_feats_binary.pkl',
        '--video_feats', 'data/retrain_inputs/in-scope_video_data_swin_roi_binary.pkl',
        '--prior_file', protocol['prior_file'], '--output_dir', output, '--text_encoder', protocol['text_encoder'],
        '--hidden_dim', '256', '--dropout', '.1', '--label_smoothing', '0', '--lambda_rdrop', '0', '--lambda_aux', '0',
        '--batch_size', '16', '--epochs', '100', '--patience', '8', '--lr', '3e-5', '--weight_decay', '0',
        '--warmup_ratio', '0', '--grad_clip', '0', '--seed', str(options.seed), '--num_workers', '8']
    completed = False
    try:
        base.main(); completed = True
    finally:
        folder = root/output; folder.mkdir(parents=True, exist_ok=True)
        (folder/'resource_usage_training.json').write_text(json.dumps({'completed': completed,
            'measurement': 'Torch allocator peak, not total device VRAM',
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'measured_wall_seconds': time.monotonic()-started, 'absolute_deadline': protocol['absolute_deadline']}, indent=2))


if __name__ == '__main__': main()
