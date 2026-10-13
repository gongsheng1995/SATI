#!/usr/bin/env python3
"""
SATI 训练脚本 - 在线 BERT encoder 版本

特性：
- 在线 BERT encoder (可微调)
- R-Drop 正则化
- SWA (Stochastic Weight Averaging)
- Warmup + Linear Schedule
- LLM Prior Sample-Adaptive Loss
- Focal Loss / Class Weight

Usage:
    python train.py \
        --data_json /path/to/data.json \
        --audio_feats /path/to/wavlm_feats.pkl \
        --video_feats /path/to/swin_roi.pkl \
        --prior_file /path/to/llm_prior.jsonl \
        --output_dir results/sati_online
"""

import os
import json
import pickle
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.sati import SATI, SATIOutput
from utils.losses import LLMPrior


# ============================================================
# 数据集
# ============================================================

class MIntRecDataset(Dataset):
    """MIntRec2.0 数据集"""
    
    def __init__(
        self,
        data_json: str,
        audio_feats_path: str,
        video_feats_path: str,
        prior_file: str,
        tokenizer_name: str = "bert-large-uncased",
        max_length: int = 128,
        split: str = 'train'
    ):
        self.split = split
        self.max_length = max_length
        
        # 加载数据
        with open(data_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.samples = data[split]
        
        # 标签映射
        self.label_list = data.get('label_list') or sorted(list(set(
            item['label'] for s in ['train', 'dev', 'test'] 
            for item in data.get(s, [])
        )))
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.num_classes = len(self.label_list)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 加载音频特征
        print(f"Loading audio features from {audio_feats_path}...")
        with open(audio_feats_path, 'rb') as f:
            self.audio_feats = pickle.load(f)
        
        # 加载视频特征
        print(f"Loading video features from {video_feats_path}...")
        with open(video_feats_path, 'rb') as f:
            self.video_feats = pickle.load(f)
        
        # 加载 LLM Prior
        print(f"Loading LLM priors from {prior_file}...")
        self.prior_dict = {}
        with open(prior_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    self.prior_dict[item['id']] = {
                        'text_sufficient': item.get('text_sufficient', True),
                        'audio_necessary': item.get('audio_necessary', False),
                        'video_necessary': item.get('video_necessary', False)
                    }
        
        print(f"[{split}] Loaded {len(self.samples)} samples")
    
    def _get_audio_feat(self, sample_id: str) -> torch.Tensor:
        """获取音频特征 (mean pooling)"""
        feat = self.audio_feats[sample_id]
        if isinstance(feat, torch.Tensor):
            if feat.dim() == 2:
                return feat.mean(dim=0).float()
            return feat.float()
        elif isinstance(feat, np.ndarray):
            if feat.ndim == 2:
                return torch.tensor(feat.mean(axis=0), dtype=torch.float32)
            return torch.tensor(feat, dtype=torch.float32)
        return torch.tensor(feat, dtype=torch.float32)
    
    def _get_video_feat(self, sample_id: str) -> torch.Tensor:
        """获取视频特征 (mean pooling)"""
        feat = self.video_feats[sample_id]
        if isinstance(feat, list):
            stacked = np.concatenate(feat, axis=0)
            return torch.tensor(stacked.mean(axis=0), dtype=torch.float32)
        elif isinstance(feat, torch.Tensor):
            if feat.dim() == 2:
                return feat.mean(dim=0).float()
            return feat.float()
        elif isinstance(feat, np.ndarray):
            if feat.ndim == 2:
                return torch.tensor(feat.mean(axis=0), dtype=torch.float32)
            return torch.tensor(feat, dtype=torch.float32)
        return torch.tensor(feat, dtype=torch.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        sample_id = item['id']
        text = item['text']
        label = self.label2id[item['label']]
        
        # 文本编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取特征
        audio_feat = self._get_audio_feat(sample_id)
        video_feat = self._get_video_feat(sample_id)
        
        # 获取 LLM Prior
        prior = self.prior_dict.get(sample_id, {
            'text_sufficient': True,
            'audio_necessary': False,
            'video_necessary': False
        })
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'audio': audio_feat,
            'video': video_feat,
            'label': torch.tensor(label, dtype=torch.long),
            'text_sufficient': prior['text_sufficient'],
            'audio_necessary': prior['audio_necessary'],
            'video_necessary': prior['video_necessary']
        }


def collate_fn(batch):
    """整理 batch"""
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'audio': torch.stack([b['audio'] for b in batch]),
        'video': torch.stack([b['video'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch]),
        'llm_prior': LLMPrior(
            text_sufficient=torch.tensor([b['text_sufficient'] for b in batch]),
            audio_necessary=torch.tensor([b['audio_necessary'] for b in batch]),
            video_necessary=torch.tensor([b['video_necessary'] for b in batch])
        )
    }


# ============================================================
# 损失函数
# ============================================================

class SATILoss(nn.Module):
    """
    SATI 损失函数
    
    包含：
    - CE Loss (带 label smoothing)
    - R-Drop KL Loss
    - Sample-Adaptive Loss (基于 LLM Prior)
    """
    
    def __init__(
        self,
        num_classes: int = 30,
        label_smoothing: float = 0.1,
        lambda_rdrop: float = 0.5,
        lambda_aux: float = 0.3,
        use_focal: bool = False,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.lambda_rdrop = lambda_rdrop
        self.lambda_aux = lambda_aux
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )
    
    def focal_loss(self, logits, labels, gamma=2.0):
        """Focal Loss"""
        ce = self.ce_loss(logits, labels)
        pt = torch.exp(-ce)
        return ((1 - pt) ** gamma * ce).mean()
    
    def kl_divergence(self, p, q):
        """对称 KL 散度"""
        p = F.log_softmax(p, dim=-1)
        q = F.log_softmax(q, dim=-1)
        
        kl_pq = F.kl_div(p, q.exp(), reduction='batchmean')
        kl_qp = F.kl_div(q, p.exp(), reduction='batchmean')
        
        return (kl_pq + kl_qp) / 2
    
    def forward(
        self,
        output1: SATIOutput,
        output2: Optional[SATIOutput],
        labels: torch.Tensor,
        llm_prior: Optional[LLMPrior] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            output1: 第一次前向的输出
            output2: 第二次前向的输出 (R-Drop)
            labels: 真实标签
            llm_prior: LLM 先验
        """
        device = labels.device
        
        # 1. CE Loss
        if self.use_focal:
            ce_loss = self.focal_loss(output1.logits, labels, self.focal_gamma)
        else:
            ce_loss = self.ce_loss(output1.logits, labels).mean()
        
        # 2. R-Drop Loss
        rdrop_loss = torch.tensor(0.0, device=device)
        if output2 is not None and self.lambda_rdrop > 0:
            rdrop_loss = self.kl_divergence(output1.logits, output2.logits)
            
            # 对第二次前向也计算 CE
            if self.use_focal:
                ce_loss = (ce_loss + self.focal_loss(output2.logits, labels, self.focal_gamma)) / 2
            else:
                ce_loss = (ce_loss + self.ce_loss(output2.logits, labels).mean()) / 2
        
        # 3. 辅助分类器 Loss (Sample-Adaptive)
        aux_loss = torch.tensor(0.0, device=device)
        if llm_prior is not None and self.lambda_aux > 0:
            # Audio 辅助损失 (仅对 audio_necessary 样本)
            audio_mask = llm_prior.audio_necessary.float().to(device)
            if audio_mask.sum() > 0:
                audio_ce = self.ce_loss(output1.logits_audio, labels)
                aux_loss = aux_loss + (audio_mask * audio_ce).sum() / audio_mask.sum()
            
            # Video 辅助损失 (仅对 video_necessary 样本)
            video_mask = llm_prior.video_necessary.float().to(device)
            if video_mask.sum() > 0:
                video_ce = self.ce_loss(output1.logits_video, labels)
                aux_loss = aux_loss + (video_mask * video_ce).sum() / video_mask.sum()
        
        # 总损失
        total_loss = ce_loss + self.lambda_rdrop * rdrop_loss + self.lambda_aux * aux_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'rdrop_loss': rdrop_loss,
            'aux_loss': aux_loss
        }


# ============================================================
# SWA
# ============================================================

class SWA:
    """Stochastic Weight Averaging"""
    
    def __init__(self, model, swa_start=30):
        self.model = model
        self.swa_model = deepcopy(model)
        self.swa_start = swa_start
        self.swa_n = 0
    
    def update(self, epoch):
        if epoch >= self.swa_start:
            self.swa_n += 1
            for param, swa_param in zip(self.model.parameters(), self.swa_model.parameters()):
                swa_param.data = (swa_param.data * (self.swa_n - 1) + param.data) / self.swa_n
    
    def get_swa_model(self):
        return self.swa_model


# ============================================================
# 训练器
# ============================================================

class Trainer:
    """SATI 训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        args
    ):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.args = args
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_acc = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        
        self.history = defaultdict(list)
        
        # SWA
        self.swa = SWA(model, args.swa_start) if args.use_swa else None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        loss_components = defaultdict(float)
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # 准备数据
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['labels'].to(self.device)
            llm_prior = LLMPrior(
                text_sufficient=batch['llm_prior'].text_sufficient.to(self.device),
                audio_necessary=batch['llm_prior'].audio_necessary.to(self.device),
                video_necessary=batch['llm_prior'].video_necessary.to(self.device)
            )
            
            # 前向传播
            self.optimizer.zero_grad()
            
            output1 = self.model(input_ids, attention_mask, audio, video)
            
            # R-Drop: 第二次前向
            output2 = None
            if self.args.lambda_rdrop > 0:
                output2 = self.model(input_ids, attention_mask, audio, video)
            
            # 计算损失
            losses = self.criterion(output1, output2, labels, llm_prior)
            loss = losses['total_loss']
            
            # 反向传播
            loss.backward()
            
            if self.args.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item() * len(labels)
            for k, v in losses.items():
                loss_components[k] += v.item() * len(labels)
            
            pred = output1.logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += len(labels)
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total,
                'lr': self.scheduler.get_last_lr()[0]
            })
        
        return {
            'loss': total_loss / total,
            'accuracy': correct / total,
            **{k: v / total for k, v in loss_components.items()}
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            output = self.model(input_ids, attention_mask, audio, video)
            
            loss = F.cross_entropy(output.logits, labels)
            total_loss += loss.item() * len(labels)
            
            pred = output.logits.argmax(dim=-1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
        macro_f1 = f1_score(all_labels, all_preds, average='macro', labels=list(range(self.model.num_classes)), zero_division=0)
        
        return {
            'loss': total_loss / len(all_labels),
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1
        }
    
    def train(self):
        """完整训练流程"""
        print(f"\nStarting training for {self.args.epochs} epochs...")
        print(f"Output dir: {self.output_dir}")
        
        for epoch in range(self.args.epochs):
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.evaluate(self.val_loader)
            
            # SWA
            if self.swa:
                self.swa.update(epoch)
            
            # 记录
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_wf1'].append(val_metrics['weighted_f1'])
            
            # 打印
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, WF1: {val_metrics['weighted_f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_model_state = deepcopy(self.model.state_dict())
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_metrics': val_metrics
                }, self.output_dir / 'best_model.pt')
                
                print(f"  ✓ New best model (acc: {self.best_val_acc:.4f})")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # 保存历史
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        
        print(f"\nTraining finished! Best val acc: {self.best_val_acc:.4f}")
        
        return self.best_val_acc, self.best_model_state


# ============================================================
# 主函数
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description='SATI Training (Online BERT)')
    
    # 数据
    parser.add_argument('--data_json', type=str, required=True)
    parser.add_argument('--audio_feats', type=str, required=True)
    parser.add_argument('--video_feats', type=str, required=True)
    parser.add_argument('--prior_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    # 模型
    parser.add_argument('--text_encoder', type=str, default='bert-large-uncased')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--tensor_rank', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_bert', action='store_true', default=False)
    
    # 损失
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--lambda_rdrop', type=float, default=0.5)
    parser.add_argument('--lambda_aux', type=float, default=0.3)
    parser.add_argument('--use_focal', action='store_true', default=False)
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=5)
    
    # SWA
    parser.add_argument('--use_swa', action='store_true', default=True)
    parser.add_argument('--swa_start', type=int, default=15)
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据集
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    
    train_dataset = MIntRecDataset(
        data_json=args.data_json,
        audio_feats_path=args.audio_feats,
        video_feats_path=args.video_feats,
        prior_file=args.prior_file,
        tokenizer_name=args.text_encoder,
        split='train'
    )
    
    val_dataset = MIntRecDataset(
        data_json=args.data_json,
        audio_feats_path=args.audio_feats,
        video_feats_path=args.video_feats,
        prior_file=args.prior_file,
        tokenizer_name=args.text_encoder,
        split='dev'
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
    )
    
    # 模型
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    # 获取特征维度
    sample = train_dataset[0]
    audio_dim = sample['audio'].shape[-1]
    video_dim = sample['video'].shape[-1]
    
    model = SATI(
        text_encoder_name=args.text_encoder,
        audio_dim=audio_dim,
        video_dim=video_dim,
        hidden_dim=args.hidden_dim,
        num_classes=train_dataset.num_classes,
        tensor_rank=args.tensor_rank,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 损失函数
    criterion = SATILoss(
        num_classes=train_dataset.num_classes,
        label_smoothing=args.label_smoothing,
        lambda_rdrop=args.lambda_rdrop,
        lambda_aux=args.lambda_aux,
        use_focal=args.use_focal
    )
    
    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args
    )
    
    # 保存配置
    config = vars(args)
    config['timestamp'] = datetime.now().isoformat()
    config['audio_dim'] = audio_dim
    config['video_dim'] = video_dim
    config['num_classes'] = train_dataset.num_classes
    config['label2id'] = train_dataset.label2id
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 训练
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_acc, best_state = trainer.train()
    
    # 测试 SWA 模型
    if args.use_swa and trainer.swa and trainer.swa.swa_n > 0:
        print("\n" + "="*60)
        print("Testing SWA model...")
        print("="*60)
        
        swa_model = trainer.swa.get_swa_model()
        trainer.model = swa_model
        swa_metrics = trainer.evaluate(val_loader)
        
        print(f"SWA Val Acc: {swa_metrics['accuracy']:.4f}")
        print(f"SWA Val WF1: {swa_metrics['weighted_f1']:.4f}")
        
        if swa_metrics['accuracy'] > best_acc:
            print("✓ SWA model is better!")
            torch.save({
                'model_state_dict': swa_model.state_dict(),
                'val_metrics': swa_metrics
            }, Path(args.output_dir) / 'swa_model.pt')
    
    print(f"\n✓ Training complete. Best accuracy: {best_acc:.4f}")
    
    # ============================================================
    # 自动测试
    # ============================================================
    print("\n" + "="*60)
    print("Testing on TEST set...")
    print("="*60)
    
    # 加载测试数据
    test_dataset = MIntRecDataset(
        data_json=args.data_json,
        audio_feats_path=args.audio_feats,
        video_feats_path=args.video_feats,
        prior_file=args.prior_file,
        tokenizer_name=args.text_encoder,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers
    )
    
    # 加载最佳模型
    best_checkpoint = torch.load(
        Path(args.output_dir) / 'best_model.pt', 
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    # 测试
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            labels = batch['labels'].to(device)
            
            output = model(input_ids, attention_mask, audio, video)
            preds = output.logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    test_results = {
        'acc': float(accuracy_score(all_labels, all_preds)),
        'weighted_f1': float(f1_score(all_labels, all_preds, average='weighted')),
        'weighted_p': float(precision_score(all_labels, all_preds, average='weighted')),
        'f1': float(f1_score(all_labels, all_preds, average='macro', labels=list(range(test_dataset.num_classes)), zero_division=0)),
        'prec': float(precision_score(all_labels, all_preds, average='macro', labels=list(range(test_dataset.num_classes)), zero_division=0)),
        'rec': float(recall_score(all_labels, all_preds, average='macro', labels=list(range(test_dataset.num_classes)), zero_division=0))
    }
    
    # 打印结果
    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  acc:         {test_results['acc']*100:.2f}%")
    print(f"  weighted_f1: {test_results['weighted_f1']*100:.2f}%")
    print(f"  weighted_p:  {test_results['weighted_p']*100:.2f}%")
    print(f"  f1 (macro):  {test_results['f1']*100:.2f}%")
    print(f"  prec (macro):{test_results['prec']*100:.2f}%")
    print(f"  rec (macro): {test_results['rec']*100:.2f}%")
    print(f"{'='*60}")
    
    # 保存测试结果
    with open(Path(args.output_dir) / 'test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTest results saved to: {Path(args.output_dir) / 'test_results.json'}")


if __name__ == "__main__":
    main()
