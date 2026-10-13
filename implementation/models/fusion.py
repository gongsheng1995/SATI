"""
融合模块 - 借鉴 TensorGated 的设计

核心改进：
1. 统一维度：先把所有模态投影到 hidden_dim
2. 两两交互：text-audio, text-video, audio-video 三组交互
3. 残差连接：z_final = z_text + interaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class LowRankBilinear(nn.Module):
    """
    低秩双线性交互 (来自 TensorGated)
    
    计算: output = (x @ W1) ⊙ (y @ W2)
    """
    
    def __init__(self, input_dim: int, rank: int, dropout: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(input_dim, rank, bias=False)
        self.W2 = nn.Linear(input_dim, rank, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.xavier_uniform_(self.W2.weight)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_proj = self.W1(x)
        y_proj = self.W2(y)
        return self.dropout(x_proj * y_proj)


class TensorInteractionModule(nn.Module):
    """
    张量交互模块 (来自 TensorGated)
    
    计算三组两两交互：text-audio, text-video, audio-video
    然后拼接融合
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        rank: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # 三组双线性交互
        self.text_audio_interaction = LowRankBilinear(hidden_dim, rank, dropout)
        self.text_video_interaction = LowRankBilinear(hidden_dim, rank, dropout)
        self.audio_video_interaction = LowRankBilinear(hidden_dim, rank, dropout)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(rank * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(
        self,
        z_text: torch.Tensor,
        z_audio: torch.Tensor,
        z_video: torch.Tensor
    ) -> torch.Tensor:
        # 三组交互
        I_ta = self.text_audio_interaction(z_text, z_audio)
        I_tv = self.text_video_interaction(z_text, z_video)
        I_av = self.audio_video_interaction(z_audio, z_video)
        
        # 拼接融合
        I_concat = torch.cat([I_ta, I_tv, I_av], dim=-1)
        interaction = self.fusion(I_concat)
        interaction = self.layer_norm(interaction) * self.scale
        
        return interaction


class TensorFusionV2(nn.Module):
    """
    完整的 Tensor 融合模块 V2 (借鉴 TensorGated)
    
    包含：
    1. 模态投影（统一维度）
    2. TensorInteractionModule
    3. 残差连接
    """
    
    def __init__(
        self,
        text_dim: int = 1024,
        audio_dim: int = 768,
        video_dim: int = 256,
        hidden_dim: int = 768,
        rank: int = 64,
        output_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1. 模态投影（统一到 hidden_dim）
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
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
        
        # 2. 张量交互模块
        self.tensor_interaction = TensorInteractionModule(
            hidden_dim=hidden_dim,
            rank=rank,
            dropout=dropout
        )
        
        # 3. 输出投影
        if output_dim != hidden_dim:
            self.output_proj = nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.output_proj = nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        z_t: torch.Tensor,
        z_a: torch.Tensor,
        z_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_t: [B, text_dim]
            z_a: [B, audio_dim]
            z_v: [B, video_dim]
        
        Returns:
            output: [B, output_dim] 融合特征
        """
        # 投影到统一维度
        h_t = self.text_proj(z_t)
        h_a = self.audio_proj(z_a)
        h_v = self.video_proj(z_v)
        
        # 计算交互
        interaction = self.tensor_interaction(h_t, h_a, h_v)
        
        # 残差连接：text + interaction
        h_fused = h_t + interaction
        
        # 输出投影
        output = self.output_proj(h_fused)
        
        return output


class ConcatFusion(nn.Module):
    """
    拼接融合（Baseline）
    """
    
    def __init__(
        self,
        text_dim: int = 1024,
        audio_dim: int = 768,
        video_dim: int = 256,
        output_dim: int = 768,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        total_dim = text_dim + audio_dim + video_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        z_t: torch.Tensor,
        z_a: torch.Tensor,
        z_v: torch.Tensor
    ) -> torch.Tensor:
        h_concat = torch.cat([z_t, z_a, z_v], dim=-1)
        return self.fusion(h_concat)


class AttentionFusion(nn.Module):
    """
    跨模态注意力融合
    """
    
    def __init__(
        self,
        text_dim: int = 1024,
        audio_dim: int = 768,
        video_dim: int = 256,
        hidden_dim: int = 768,
        num_heads: int = 8,
        output_dim: int = 768,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 投影到统一维度
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 输出投影
        if output_dim != hidden_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.output_proj = nn.Identity()
    
    def forward(
        self,
        z_t: torch.Tensor,
        z_a: torch.Tensor,
        z_v: torch.Tensor
    ) -> torch.Tensor:
        # 投影到统一维度
        h_t = self.text_proj(z_t)  # [B, H]
        h_a = self.audio_proj(z_a)  # [B, H]
        h_v = self.video_proj(z_v)  # [B, H]
        
        # 构造序列: [text], [audio, video]
        query = h_t.unsqueeze(1)  # [B, 1, H]
        kv = torch.stack([h_a, h_v], dim=1)  # [B, 2, H]
        
        # 跨模态注意力
        attn_out, _ = self.cross_attn(query, kv, kv)  # [B, 1, H]
        attn_out = attn_out.squeeze(1)  # [B, H]
        
        # 残差 + LayerNorm
        h = self.norm1(h_t + attn_out)
        
        # FFN + 残差
        h = self.norm2(h + self.ffn(h))
        
        return self.output_proj(h)


class GatedFusion(nn.Module):
    """
    门控融合
    """
    
    def __init__(
        self,
        text_dim: int = 1024,
        audio_dim: int = 768,
        video_dim: int = 256,
        hidden_dim: int = 768,
        output_dim: int = 768,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        # 投影到统一维度
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        z_t: torch.Tensor,
        z_a: torch.Tensor,
        z_v: torch.Tensor
    ) -> torch.Tensor:
        # 投影
        h_t = self.text_proj(z_t)
        h_a = self.audio_proj(z_a)
        h_v = self.video_proj(z_v)
        
        # 计算门控
        h_concat = torch.cat([h_t, h_a, h_v], dim=-1)
        gates = self.gate_net(h_concat)  # [B, 3]
        
        # 加权融合
        h_fused = (
            gates[:, 0:1] * h_t +
            gates[:, 1:2] * h_a +
            gates[:, 2:3] * h_v
        )
        
        return self.output_proj(h_fused)


# ============================================================
# 工厂函数
# ============================================================

def create_fusion_module(
    fusion_type: str = "tensor",
    text_dim: int = 1024,
    audio_dim: int = 768,
    video_dim: int = 256,
    output_dim: int = 768,
    hidden_dim: int = 768,
    rank: int = 64,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    创建融合模块
    
    Args:
        fusion_type: "tensor", "attention", "gated", "concat"
    """
    if fusion_type == "tensor":
        return TensorFusionV2(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            hidden_dim=hidden_dim,
            rank=rank,
            output_dim=output_dim,
            dropout=dropout
        )
    elif fusion_type == "attention":
        return AttentionFusion(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
    elif fusion_type == "gated":
        return GatedFusion(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout
        )
    elif fusion_type == "concat":
        return ConcatFusion(
            text_dim=text_dim,
            audio_dim=audio_dim,
            video_dim=video_dim,
            output_dim=output_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("Testing fusion modules...")
    
    B = 4
    z_t = torch.randn(B, 1024)
    z_a = torch.randn(B, 768)
    z_v = torch.randn(B, 256)
    
    for fusion_type in ["tensor", "attention", "gated", "concat"]:
        print(f"\nTesting {fusion_type} fusion...")
        
        module = create_fusion_module(
            fusion_type=fusion_type,
            text_dim=1024,
            audio_dim=768,
            video_dim=256,
            output_dim=768,
            hidden_dim=768,
            dropout=0.1
        )
        
        output = module(z_t, z_a, z_v)
        print(f"  Input: text={z_t.shape}, audio={z_a.shape}, video={z_v.shape}")
        print(f"  Output: {output.shape}")
        print(f"  Parameters: {sum(p.numel() for p in module.parameters()):,}")
    
    print("\n✓ All fusion tests passed!")
