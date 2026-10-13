"""
SATI: Sample-Adaptive Training with LLM Semantic Prior

在线 BERT encoder 版本，借鉴 TensorGated 的设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModel

from .fusion import TensorInteractionModule


@dataclass
class SATIOutput:
    """SATI 模型输出"""
    logits: torch.Tensor           # [B, C] 主分类器输出
    logits_text: torch.Tensor      # [B, C] Text-only 分类器输出
    logits_audio: torch.Tensor     # [B, C] Audio 辅助分类器输出
    logits_video: torch.Tensor     # [B, C] Video 辅助分类器输出
    h_fused: torch.Tensor          # [B, D] 融合特征
    h_text: torch.Tensor           # [B, D] Text 特征


class SATI(nn.Module):
    """
    SATI: Sample-Adaptive Training with LLM Semantic Prior
    
    在线 BERT encoder 版本
    
    架构：
    1. BERT Encoder (text) + 投影层 (audio/video)
    2. TensorInteractionModule (三模态交互)
    3. 残差连接: z_final = z_text + interaction
    4. 主分类器 + 辅助分类器
    """
    
    def __init__(
        self,
        text_encoder_name: str = "bert-large-uncased",
        text_dim: int = 1024,
        audio_dim: int = 768,
        video_dim: int = 256,
        hidden_dim: int = 768,
        num_classes: int = 30,
        tensor_rank: int = 64,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # ============ BERT Encoder ============
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        encoder_dim = self.text_encoder.config.hidden_size
        
        if freeze_bert:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Text 投影
        self.text_proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # ============ Audio/Video 投影 ============
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # ============ Tensor Interaction ============
        self.tensor_interaction = TensorInteractionModule(
            hidden_dim=hidden_dim,
            rank=tensor_rank,
            dropout=dropout
        )
        
        # ============ 主分类器 ============
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # ============ 辅助分类器 ============
        # Text-only 分类器
        self.text_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Audio 分类器
        self.audio_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Video 分类器
        self.video_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # ============ 对比学习投影头 ============
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化非 BERT 部分的权重"""
        for name, module in self.named_modules():
            if 'text_encoder' in name:
                continue  # 跳过 BERT
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """编码文本"""
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 取 [CLS] token 或 pooler_output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            h = outputs.pooler_output
        else:
            h = outputs.last_hidden_state[:, 0]
        
        return self.text_proj(h)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_feat: torch.Tensor,
        video_feat: torch.Tensor
    ) -> SATIOutput:
        """
        前向传播
        
        Args:
            input_ids: [B, seq_len] 文本 token IDs
            attention_mask: [B, seq_len] attention mask
            audio_feat: [B, audio_dim] 音频特征
            video_feat: [B, video_dim] 视频特征
        
        Returns:
            SATIOutput
        """
        # 1. 编码各模态
        h_text = self.encode_text(input_ids, attention_mask)  # [B, hidden_dim]
        h_audio = self.audio_proj(audio_feat)  # [B, hidden_dim]
        h_video = self.video_proj(video_feat)  # [B, hidden_dim]
        
        # 2. 计算张量交互
        interaction = self.tensor_interaction(h_text, h_audio, h_video)  # [B, hidden_dim]
        
        # 3. 残差融合
        h_fused = h_text + interaction  # [B, hidden_dim]
        
        # 4. 主分类
        logits = self.classifier(h_fused)
        
        # 5. 辅助分类
        logits_text = self.text_classifier(h_text)
        logits_audio = self.audio_classifier(h_audio)
        logits_video = self.video_classifier(h_video)
        
        return SATIOutput(
            logits=logits,
            logits_text=logits_text,
            logits_audio=logits_audio,
            logits_video=logits_video,
            h_fused=h_fused,
            h_text=h_text
        )
    
    def get_text_confidence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 text-only 预测和置信度"""
        h_text = self.encode_text(input_ids, attention_mask)
        logits = self.text_classifier(h_text)
        probs = F.softmax(logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)
        return predictions, confidence
