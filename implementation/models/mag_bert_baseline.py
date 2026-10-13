"""Pinned MAG-BERT matched wrapper with audited pretrained initialization.

Native gating, CTC-style soft alignment and CE remain. No new temporal labels,
timestamp alignment, SATI selection heads, model loading or GPU result asserted.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

from .vendor.mag_bert.FusionNets.MAG_BERT import MAG_BertForSequenceClassification
from .official_mult_baseline import OfficialMulTForSATI


@dataclass
class MAGBERTOutput:
    logits: torch.Tensor


class MAGBERTForSATI(nn.Module):
    def __init__(self, text_encoder_name='bert-large-uncased', audio_dim=768, video_dim=256,
                 hidden_dim=1024, num_classes=30, dropout=.5, freeze_bert=False, tensor_rank=None):
        super().__init__()
        if (audio_dim, video_dim, hidden_dim, num_classes, dropout) != (768, 256, 1024, 30, .5):
            raise ValueError('Prepared MAG-BERT protocol requires WavLM768/Swin256/BERT-large1024/30 labels/gate dropout0.5.')
        directory = Path(__file__).parent/'vendor/mag_bert'
        provenance = json.loads((directory/'source_provenance.json').read_text(encoding='utf-8'))
        for name, digest in provenance['files_sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Pinned MAG-BERT source changed.')
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(text_encoder_name, local_files_only=True)
        if config.hidden_size != 1024 or config.num_hidden_layers != 24:
            raise ValueError('BERT-large fixed dimensions/layers required.')
        args = SimpleNamespace(text_feat_dim=1024, audio_feat_dim=768, video_feat_dim=256,
            text_seq_len=50, audio_seq_len=400, video_seq_len=180, need_aligned=True,
            aligned_method='ctc', beta_shift=.005, dropout_prob=.5, num_labels=30, device=torch.device('cpu'))
        self.native, loading = MAG_BertForSequenceClassification.from_pretrained(
            text_encoder_name, config=config, args=args, local_files_only=True, output_loading_info=True)
        # MAG and classifier are new. All pretrained BERT layers must load.
        missing = loading.get('missing_keys', [])
        unexpected = loading.get('unexpected_keys', [])
        if any(not key.startswith(('bert.MAG.', 'classifier.')) for key in missing):
            raise ValueError('Missing pretrained BERT key; initialization audit failed: '+str(missing))
        if any(not key.startswith('cls.') for key in unexpected) or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError('Unexpected pretrained loading mismatch: '+str(loading))
        self.loading_audit = {'missing_new_component_keys': missing, 'unused_pretraining_head_keys': unexpected}
        self.num_classes = num_classes
        if freeze_bert:
            for name, parameter in self.native.bert.named_parameters():
                if not name.startswith('MAG.'): parameter.requires_grad = False

    def forward(self, input_ids, attention_mask, audio, video):
        if input_ids.shape[1] != 50: raise ValueError('Prepared text cap50 required.')
        audio, _ = OfficialMulTForSATI.finite_inputs(audio, 400, 768)
        video, _ = OfficialMulTForSATI.finite_inputs(video, 180, 256)
        # Native alignment consumes zero-end padded sequences without length masks.
        # Native temporary tensors use args.device; update it to the actual device.
        self.native.bert.MAG.args.device = input_ids.device
        text = torch.stack((input_ids, attention_mask, torch.zeros_like(input_ids)), dim=1)
        return MAGBERTOutput(self.native(text, video, audio)[0])


class MAGBERTObjective(nn.Module):
    def forward(self, output1, output2, labels, llm_prior=None):
        if output2 is not None: raise ValueError('Native MAG-BERT uses one forward, no R-Drop.')
        main = F.cross_entropy(output1.logits, labels)
        return {'total_loss': main, 'ce_loss': main, 'rdrop_loss': main.new_zeros(()), 'aux_loss': main.new_zeros(())}
