"""
Sample-Adaptive Loss - SATI 的核心损失函数

根据 LLM 语义先验，对不同样本采用不同的训练策略：
1. L_main: 主分类损失（所有样本）
2. L_audio: 音频分支强化损失（仅 audio_necessary 样本）
3. L_video: 视频分支强化损失（仅 video_necessary 样本）
4. L_counterfactual: 反事实惩罚（仅 !text_sufficient 样本）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class LLMPrior:
    """LLM 语义先验"""
    text_sufficient: torch.Tensor   # [B], bool, 文本是否足够
    audio_necessary: torch.Tensor   # [B], bool, 是否需要音频
    video_necessary: torch.Tensor   # [B], bool, 是否需要视频


class SampleAdaptiveLoss(nn.Module):
    """
    Sample-Adaptive Loss
    
    根据 LLM 先验动态调整损失：
    - 对于"需要 audio"的样本，额外训练 audio 分支
    - 对于"需要 video"的样本，额外训练 video 分支
    - 对于"text 不够"的样本，惩罚 text-only 预测正确
    
    Args:
        lambda_audio: 音频强化损失权重
        lambda_video: 视频强化损失权重
        lambda_cf: 反事实惩罚权重
        label_smoothing: 标签平滑
    """
    
    def __init__(
        self,
        lambda_audio: float = 0.5,
        lambda_video: float = 0.5,
        lambda_cf: float = 0.3,
        label_smoothing: float = 0.1
    ):
        super().__init__()
        
        self.lambda_audio = lambda_audio
        self.lambda_video = lambda_video
        self.lambda_cf = lambda_cf
        self.label_smoothing = label_smoothing
        
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )
    
    def forward(
        self,
        logits_main: torch.Tensor,
        logits_text: torch.Tensor,
        logits_audio: torch.Tensor,
        logits_video: torch.Tensor,
        labels: torch.Tensor,
        llm_prior: LLMPrior
    ) -> Dict[str, torch.Tensor]:
        """
        计算 Sample-Adaptive Loss
        
        Args:
            logits_main: [B, C] 主分类器输出
            logits_text: [B, C] Text-only 分类器输出
            logits_audio: [B, C] Audio 辅助分类器输出
            logits_video: [B, C] Video 辅助分类器输出
            labels: [B] 真实标签
            llm_prior: LLM 语义先验
        
        Returns:
            Dict containing:
                - loss_total: 总损失
                - loss_main: 主分类损失
                - loss_audio: 音频强化损失
                - loss_video: 视频强化损失
                - loss_cf: 反事实惩罚
        """
        batch_size = labels.size(0)
        device = labels.device
        
        # ========== 1. 主分类损失 ==========
        loss_main = self.ce_loss(logits_main, labels).mean()
        
        # ========== 2. 音频强化损失 ==========
        audio_mask = llm_prior.audio_necessary.float()  # [B]
        n_audio = audio_mask.sum()
        
        if n_audio > 0:
            loss_audio_per_sample = self.ce_loss(logits_audio, labels)  # [B]
            loss_audio = (audio_mask * loss_audio_per_sample).sum() / n_audio
        else:
            loss_audio = torch.tensor(0.0, device=device)
        
        # ========== 3. 视频强化损失 ==========
        video_mask = llm_prior.video_necessary.float()  # [B]
        n_video = video_mask.sum()
        
        if n_video > 0:
            loss_video_per_sample = self.ce_loss(logits_video, labels)  # [B]
            loss_video = (video_mask * loss_video_per_sample).sum() / n_video
        else:
            loss_video = torch.tensor(0.0, device=device)
        
        # ========== 4. 反事实惩罚 ==========
        # 对于"text 不够"的样本，惩罚 text-only 预测正确
        need_multimodal_mask = (~llm_prior.text_sufficient).float()  # [B]
        n_need_multimodal = need_multimodal_mask.sum()
        
        if n_need_multimodal > 0:
            # Text-only 预测正确的概率
            prob_text = F.softmax(logits_text, dim=-1)  # [B, C]
            prob_text_correct = prob_text.gather(1, labels.unsqueeze(1)).squeeze(1)  # [B]
            
            # 惩罚：text-only 预测越准确，惩罚越大
            loss_cf = (need_multimodal_mask * prob_text_correct).sum() / n_need_multimodal
        else:
            loss_cf = torch.tensor(0.0, device=device)
        
        # ========== 5. 总损失 ==========
        loss_total = (
            loss_main +
            self.lambda_audio * loss_audio +
            self.lambda_video * loss_video +
            self.lambda_cf * loss_cf
        )
        
        return {
            'loss_total': loss_total,
            'loss_main': loss_main,
            'loss_audio': loss_audio,
            'loss_video': loss_video,
            'loss_cf': loss_cf,
            # 统计信息
            'n_audio_samples': n_audio,
            'n_video_samples': n_video,
            'n_need_multimodal': n_need_multimodal
        }


class SampleAdaptiveLossV2(nn.Module):
    """
    Sample-Adaptive Loss V2 - 增强版
    
    额外添加：
    - 模态一致性损失：融合预测应该与各模态预测一致
    - 难样本挖掘：对难样本增加权重
    """
    
    def __init__(
        self,
        lambda_audio: float = 0.5,
        lambda_video: float = 0.5,
        lambda_cf: float = 0.3,
        lambda_consist: float = 0.1,
        label_smoothing: float = 0.1,
        hard_sample_weight: float = 2.0
    ):
        super().__init__()
        
        self.lambda_audio = lambda_audio
        self.lambda_video = lambda_video
        self.lambda_cf = lambda_cf
        self.lambda_consist = lambda_consist
        self.hard_sample_weight = hard_sample_weight
        
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction='none'
        )
    
    def forward(
        self,
        logits_main: torch.Tensor,
        logits_text: torch.Tensor,
        logits_audio: torch.Tensor,
        logits_video: torch.Tensor,
        labels: torch.Tensor,
        llm_prior: LLMPrior
    ) -> Dict[str, torch.Tensor]:
        """计算损失"""
        batch_size = labels.size(0)
        device = labels.device
        
        # ========== 1. 主分类损失（带难样本挖掘）==========
        loss_main_per_sample = self.ce_loss(logits_main, labels)  # [B]
        
        # 难样本：需要多模态但 loss 较大的样本
        need_multimodal = ~llm_prior.text_sufficient
        is_hard = need_multimodal & (loss_main_per_sample > loss_main_per_sample.mean())
        
        sample_weights = torch.ones(batch_size, device=device)
        sample_weights[is_hard] = self.hard_sample_weight
        
        loss_main = (sample_weights * loss_main_per_sample).mean()
        
        # ========== 2. 音频强化损失 ==========
        audio_mask = llm_prior.audio_necessary.float()
        n_audio = audio_mask.sum()
        
        if n_audio > 0:
            loss_audio_per_sample = self.ce_loss(logits_audio, labels)
            loss_audio = (audio_mask * loss_audio_per_sample).sum() / n_audio
        else:
            loss_audio = torch.tensor(0.0, device=device)
        
        # ========== 3. 视频强化损失 ==========
        video_mask = llm_prior.video_necessary.float()
        n_video = video_mask.sum()
        
        if n_video > 0:
            loss_video_per_sample = self.ce_loss(logits_video, labels)
            loss_video = (video_mask * loss_video_per_sample).sum() / n_video
        else:
            loss_video = torch.tensor(0.0, device=device)
        
        # ========== 4. 反事实惩罚 ==========
        need_multimodal_mask = (~llm_prior.text_sufficient).float()
        n_need_multimodal = need_multimodal_mask.sum()
        
        if n_need_multimodal > 0:
            prob_text = F.softmax(logits_text, dim=-1)
            prob_text_correct = prob_text.gather(1, labels.unsqueeze(1)).squeeze(1)
            loss_cf = (need_multimodal_mask * prob_text_correct).sum() / n_need_multimodal
        else:
            loss_cf = torch.tensor(0.0, device=device)
        
        # ========== 5. 模态一致性损失 ==========
        # 融合预测的分布应该与各模态预测的加权分布一致
        prob_main = F.softmax(logits_main, dim=-1)
        prob_audio = F.softmax(logits_audio, dim=-1)
        prob_video = F.softmax(logits_video, dim=-1)
        
        # 对于需要 audio 的样本，融合预测应该与 audio 预测一致
        if n_audio > 0:
            kl_audio = F.kl_div(
                prob_main.log(),
                prob_audio,
                reduction='none'
            ).sum(dim=-1)  # [B]
            loss_consist_audio = (audio_mask * kl_audio).sum() / n_audio
        else:
            loss_consist_audio = torch.tensor(0.0, device=device)
        
        # 对于需要 video 的样本，融合预测应该与 video 预测一致
        if n_video > 0:
            kl_video = F.kl_div(
                prob_main.log(),
                prob_video,
                reduction='none'
            ).sum(dim=-1)
            loss_consist_video = (video_mask * kl_video).sum() / n_video
        else:
            loss_consist_video = torch.tensor(0.0, device=device)
        
        loss_consist = loss_consist_audio + loss_consist_video
        
        # ========== 6. 总损失 ==========
        loss_total = (
            loss_main +
            self.lambda_audio * loss_audio +
            self.lambda_video * loss_video +
            self.lambda_cf * loss_cf +
            self.lambda_consist * loss_consist
        )
        
        return {
            'loss_total': loss_total,
            'loss_main': loss_main,
            'loss_audio': loss_audio,
            'loss_video': loss_video,
            'loss_cf': loss_cf,
            'loss_consist': loss_consist,
            'n_audio_samples': n_audio,
            'n_video_samples': n_video,
            'n_need_multimodal': n_need_multimodal,
            'n_hard_samples': is_hard.sum()
        }


def create_loss_function(
    loss_type: str = "v1",
    **kwargs
) -> nn.Module:
    """
    创建损失函数
    
    Args:
        loss_type: "v1" (基础版) 或 "v2" (增强版)
    """
    if loss_type == "v1":
        return SampleAdaptiveLoss(**kwargs)
    elif loss_type == "v2":
        return SampleAdaptiveLossV2(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
