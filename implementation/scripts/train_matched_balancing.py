#!/usr/bin/env python3
"""Matched all-sample loss-balancing controls using the existing SATI trainer.

Prepared controls, not evidence until full training and prediction audit finish.
Pass --balancing uncertainty|pcgrad followed by the ordinary train.py arguments.
This entry point modifies only its own imported Python module, never train.py.

Uncertainty: exp(-s)*[L_main, lambda_aux*L_a, lambda_aux*L_v] + sum(s)/2,
plus the unchanged R-Drop penalty. Kendall et al., CVPR 2018, Eq. 10.
Focal/smoothed losses and the initial auxiliary scale are disclosed adaptations
of the classification approximation, not an exact probabilistic likelihood.

PCGrad: project the three task gradients on the union of audio_proj/video_proj
parameters, then sum them. All other parameters receive the ordinary summed
gradient. Task 0 includes R-Drop; tasks 1/2 are scaled auxiliary CE losses.
This is a PCGrad adaptation on the shared nonverbal projection blocks.
Sources: https://arxiv.org/abs/1705.07115v3 and
https://arxiv.org/abs/2001.06782 / https://github.com/tianheyu927/PCGrad .
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import torch
from torch import nn

import train as base


def project_task_gradients(vectors, rng):
    """Project conflicts against ORIGINAL task gradients; aggregate by sum.

    Random orders use a separate seeded RNG, leaving DataLoader/dropout random
    streams unchanged. Zero-norm task gradients produce a zero correction.
    """
    projected = []
    conflicts = 0
    comparisons = 0
    for i, original in enumerate(vectors):
        gradient = original.clone()
        order = [j for j in range(len(vectors)) if j != i]
        rng.shuffle(order)
        for j in order:
            other = vectors[j]
            dot = torch.dot(gradient, other)
            norm_sq = torch.dot(other, other)
            conflicts += int(dot.detach().item() < 0)
            comparisons += 1
            gradient = gradient - (dot / norm_sq.clamp_min(1e-12)).clamp_max(0) * other
        projected.append(gradient)
    return torch.stack(projected).sum(dim=0), conflicts / max(comparisons, 1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument('--balancing', choices=['uncertainty', 'pcgrad'], required=True)
    extra.add_argument('--weight_lr', type=float, default=1e-3)
    options, ordinary_args = extra.parse_known_args()
    sys.argv = [sys.argv[0], *ordinary_args]
    original_loss = base.SATILoss
    original_trainer = base.Trainer
    original_adamw = base.AdamW
    active_criterion = []

    class BalancingLoss(original_loss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.lambda_aux <= 0:
                raise ValueError('Balancing controls require positive lambda_aux')
            if options.balancing == 'uncertainty':
                self.log_variances = nn.Parameter(torch.zeros(3))
                self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            active_criterion.append(self)

        def forward(self, output1, output2, labels, llm_prior=None):
            if llm_prior is None or not (
                llm_prior.audio_necessary.bool().all()
                and llm_prior.video_necessary.bool().all()
            ):
                raise ValueError('Every training sample must have both auxiliary masks true')
            losses = super().forward(output1, output2, labels, llm_prior)
            audio_ce = self.ce_loss(output1.logits_audio, labels).mean()
            video_ce = self.ce_loss(output1.logits_video, labels).mean()
            weighted_tasks = torch.stack([
                losses['ce_loss'], self.lambda_aux * audio_ce, self.lambda_aux * video_ce
            ])
            if options.balancing == 'uncertainty':
                losses['total_loss'] = (
                    (self.log_variances.neg().exp() * weighted_tasks).sum()
                    + 0.5 * self.log_variances.sum()
                    + self.lambda_rdrop * losses['rdrop_loss']
                )
            else:
                losses['_tasks'] = [
                    weighted_tasks[0] + self.lambda_rdrop * losses['rdrop_loss'],
                    weighted_tasks[1], weighted_tasks[2]
                ]
            return losses

    def balancing_optimizer(params, **kwargs):
        groups = [{'params': list(params)}]
        if options.balancing == 'uncertainty':
            groups.append({
                'params': list(active_criterion[-1].parameters()),
                'lr': options.weight_lr, 'weight_decay': 0.0
            })
        return original_adamw(groups, **kwargs)

    class BalancingTrainer(original_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.projection_rng = random.Random(self.args.seed)
            self.balancing_history = []
            self.shared_parameters = [
                parameter for module in [self.model.audio_proj, self.model.video_proj]
                for parameter in module.parameters() if parameter.requires_grad
            ]

        def train(self):
            config_path = self.output_dir / 'config.json'
            config = json.loads(config_path.read_text(encoding='utf-8'))
            config['balancing'] = {
                'method': options.balancing,
                'auxiliary_coverage': 'all training samples for audio and video',
                'initial_task_scales': [1.0, self.args.lambda_aux, self.args.lambda_aux],
                'weight_lr': options.weight_lr if options.balancing == 'uncertainty' else None,
                'log_variance_initialization': [0.0, 0.0, 0.0] if options.balancing == 'uncertainty' else None,
                'pcgrad_parameter_scope': ['audio_proj', 'video_proj'] if options.balancing == 'pcgrad' else None,
                'pcgrad_reduction': 'sum',
                'pcgrad_order_rng_seed': self.args.seed,
                'test_checkpoint': 'ordinary best validation accuracy; best_model.pt',
                'base_train_sha256': digest(base.__file__),
                'balancing_source_sha256': digest(__file__),
                'source_urls': ['https://arxiv.org/abs/1705.07115v3', 'https://arxiv.org/abs/2001.06782'],
                'adaptations': 'Focal/smoothed losses with initial auxiliary scaling; PCGrad on nonverbal projection union only'
            }
            config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
            return super().train()

        def train_epoch(self, epoch):
            self.model.train()
            self.criterion.train()
            components = defaultdict(float)
            total = correct = 0
            pbar = base.tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['labels'].to(self.device)
                llm_prior = base.LLMPrior(
                    text_sufficient=batch['llm_prior'].text_sufficient.to(self.device),
                    audio_necessary=batch['llm_prior'].audio_necessary.to(self.device),
                    video_necessary=batch['llm_prior'].video_necessary.to(self.device)
                )
                self.optimizer.zero_grad()
                output1 = self.model(input_ids, attention_mask, audio, video)
                output2 = self.model(input_ids, attention_mask, audio, video) if self.args.lambda_rdrop > 0 else None
                losses = self.criterion(output1, output2, labels, llm_prior)
                projected = None
                if options.balancing == 'pcgrad':
                    vectors = []
                    for task in losses.pop('_tasks'):
                        grads = torch.autograd.grad(task, self.shared_parameters, retain_graph=True, allow_unused=True)
                        vectors.append(torch.cat([
                            (torch.zeros_like(p) if g is None else g).reshape(-1)
                            for p, g in zip(self.shared_parameters, grads)
                        ]))
                    projected, conflict_rate = project_task_gradients(vectors, self.projection_rng)
                    components['pcgrad_conflict_rate'] += conflict_rate * len(labels)
                loss = losses['total_loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite balancing loss; preserve run logs')
                loss.backward()
                if projected is not None:
                    offset = 0
                    for parameter in self.shared_parameters:
                        length = parameter.numel()
                        parameter.grad = projected[offset:offset+length].view_as(parameter).clone()
                        offset += length
                if self.args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                for name, value in losses.items():
                    components[name] += value.detach().item() * len(labels)
                correct += (output1.logits.argmax(-1) == labels).sum().item()
                total += len(labels)
                pbar.set_postfix(loss=loss.item(), acc=correct/total, lr=self.scheduler.get_last_lr()[0])
            metrics = {name: value/total for name, value in components.items()}
            metrics.update(loss=metrics['total_loss'], accuracy=correct/total)
            record = {'epoch': epoch+1, **metrics}
            if options.balancing == 'uncertainty':
                record['log_variances_end_epoch'] = self.criterion.log_variances.detach().cpu().tolist()
                record['multipliers_end_epoch'] = self.criterion.log_variances.detach().neg().exp().cpu().tolist()
            self.balancing_history.append(record)
            (self.output_dir / 'balancing_history.json').write_text(
                json.dumps(self.balancing_history, indent=2), encoding='utf-8'
            )
            return metrics

    # The inherited main keeps datasets/model/R-Drop/SWA/selection/test identical.
    # No subprocess or additional run is launched by importing this module.
    base.SATILoss = BalancingLoss
    base.Trainer = BalancingTrainer
    base.AdamW = balancing_optimizer
    base.main()


if __name__ == '__main__':
    main()
