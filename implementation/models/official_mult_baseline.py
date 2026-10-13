"""Prepared wrapper for pinned official MIntRec MulT, not GPU validated.

The native model forward/classifier remains unchanged. An optional training-only
extension routes CE on mean valid A/V convolutional projection outputs, shared
with the main cross-attention path. No LLM mask enters the main prediction.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

from .vendor.mintrec_mult.FusionNets.MULT import MULT


@dataclass
class OfficialMulTOutput:
    logits:torch.Tensor
    h_fused:torch.Tensor
    logits_audio:torch.Tensor|None=None
    logits_video:torch.Tensor|None=None


class OfficialMulTBaseline(nn.Module):
    def __init__(self,args:SimpleNamespace,with_auxiliary_heads:bool=False):
        super().__init__()
        directory=Path(__file__).parent/'vendor/mintrec_mult'
        provenance=json.loads((directory/'source_provenance.json').read_text(encoding='utf-8'))
        for name,digest in provenance['files_sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Pinned official MulT source differs: '+name)
        if args.audio_feat_dim==args.dst_feature_dims or args.video_feat_dim==args.dst_feature_dims:
            raise ValueError('This wrapper requires the native forward to use both A/V convolutional projections.')
        if args.conv1d_kernel_size_a!=1 or args.conv1d_kernel_size_v!=1:
            raise ValueError('Prepared valid-length auxiliary pooling assumes the official A/V kernel1 configuration.')
        self.native=MULT(args)
        self.num_classes=args.num_labels
        self.with_auxiliary_heads=with_auxiliary_heads
        if with_auxiliary_heads:
            self.audio_head=nn.Linear(args.dst_feature_dims,args.num_labels)
            self.video_head=nn.Linear(args.dst_feature_dims,args.num_labels)

    @staticmethod
    def valid_mean(projected,lengths):
        if lengths is None:raise ValueError('Actual retained sequence lengths are mandatory for auxiliary pooling.')
        lengths=lengths.to(device=projected.device)
        if lengths.dtype not in (torch.int32,torch.int64) or lengths.ndim!=1 or lengths.shape[0]!=projected.shape[0] or torch.any(lengths<=0) or torch.any(lengths>projected.shape[-1]):
            raise ValueError('Invalid sequence lengths; no zero/missing feature fallback.')
        valid=torch.arange(projected.shape[-1],device=projected.device)[None,:]<lengths[:,None]
        return (projected*valid[:,None,:]).sum(-1)/lengths[:,None]

    def forward(self,text_feats,video_feats,audio_feats,*,audio_lengths=None,video_lengths=None):
        if not self.with_auxiliary_heads:
            logits,features=self.native(text_feats,video_feats,audio_feats)
            return OfficialMulTOutput(logits,features)
        captured={}
        def capture(name):
            def hook(module,inputs,output):captured[name]=output
            return hook
        handles=[self.native.proj_a.register_forward_hook(capture('audio')),
                 self.native.proj_v.register_forward_hook(capture('video'))]
        try:
            logits,features=self.native(text_feats,video_feats,audio_feats)
        finally:
            for handle in handles:handle.remove()
        if set(captured)!= {'audio','video'}:raise RuntimeError('Shared native projections were not captured.')
        audio_logits=self.audio_head(self.valid_mean(captured['audio'],audio_lengths))
        video_logits=self.video_head(self.valid_mean(captured['video'],video_lengths))
        return OfficialMulTOutput(logits,features,audio_logits,video_logits)


class OfficialMulTObjective(nn.Module):
    """Native main CE, optional A/V selected-mean CE extension.

All/none/SATI must share this exact architecture and coefficient when comparing
selection. The bare native baseline uses a separate model without added heads.
"""
    def __init__(self,condition='native',lambda_aux=.3):
        super().__init__()
        if condition not in ('native','none','all','sati'):raise ValueError('Unknown auxiliary condition.')
        if lambda_aux<0:raise ValueError('Negative auxiliary coefficient.')
        self.condition=condition;self.lambda_aux=lambda_aux

    def forward(self,output,targets,*,audio_mask=None,video_mask=None):
        main=F.cross_entropy(output.logits,targets)
        if self.condition=='native':return main
        if output.logits_audio is None or output.logits_video is None:
            raise ValueError('Matched auxiliary architecture required for none/all/SATI.')
        auxiliary=[]
        for logits,mask in ((output.logits_audio,audio_mask),(output.logits_video,video_mask)):
            per_sample=F.cross_entropy(logits,targets,reduction='none')
            if self.condition=='all':mask=torch.ones_like(targets,dtype=torch.bool)
            elif self.condition=='none':mask=torch.zeros_like(targets,dtype=torch.bool)
            elif mask is None or mask.dtype!=torch.bool or mask.shape!=targets.shape:
                raise ValueError('SATI requires an explicitly audited boolean training mask for each modality.')
            mask=mask.to(logits.device)
            selected=mask.to(per_sample.dtype)
            auxiliary.append((per_sample*selected).sum()/selected.sum().clamp_min(1))
        return main+self.lambda_aux*sum(auxiliary)


class OfficialMulTForSATI(nn.Module):
    """SATI trainer signature; NaN padding is metadata, never a native input.

Only the fixed MIntRec2 adaptation (80 dimensions, two cross layers/heads) is
prepared here. Actual retained lengths come from wholly NaN suffix rows.
"""
    def __init__(self,text_encoder_name='bert-large-uncased',audio_dim=768,video_dim=256,
                 hidden_dim=80,num_classes=30,dropout=.2,freeze_bert=False,
                 tensor_rank=None,with_auxiliary_heads=True):
        super().__init__()
        if hidden_dim!=80 or num_classes!=30:raise ValueError('Prepared MulT driver requires fixed80 hidden/30 classes.')
        from transformers import AutoConfig
        config=AutoConfig.from_pretrained(text_encoder_name,local_files_only=True)
        args=SimpleNamespace(text_backbone=text_encoder_name,cache_path=None,text_feat_dim=config.hidden_size,
            audio_feat_dim=audio_dim,video_feat_dim=video_dim,dst_feature_dims=80,nheads=2,n_levels=2,
            attn_dropout=0.,attn_dropout_a=.2,attn_dropout_v=.2,relu_dropout=.3,embed_dropout=.1,
            res_dropout=0.,output_dropout=.2,text_dropout=.4,attn_mask=True,
            conv1d_kernel_size_l=5,conv1d_kernel_size_a=1,conv1d_kernel_size_v=1,num_labels=num_classes)
        self.model=OfficialMulTBaseline(args,with_auxiliary_heads)
        self.num_classes=num_classes
        if freeze_bert:
            for parameter in self.model.native.text_subnet.parameters():parameter.requires_grad=False

    @staticmethod
    def finite_inputs(raw,expected_length,expected_dim):
        if raw.ndim!=3 or raw.shape[1:]!=(expected_length,expected_dim):raise ValueError('Fixed MulT sequence protocol differs.')
        valid=torch.isfinite(raw).all(-1)
        padding=torch.isnan(raw).all(-1)
        if not (valid|padding).all():raise ValueError('Invalid partial NaN/Inf frame; only finite values or padding metadata allowed.')
        lengths=valid.sum(-1)
        prefix=torch.arange(expected_length,device=raw.device)[None,:]<lengths[:,None]
        if (lengths<=0).any() or not torch.equal(valid,prefix):raise ValueError('Expected nonempty finite prefix followed only by NaN padding.')
        return raw.masked_fill(~valid[:,:,None],0),lengths

    def forward(self,input_ids,attention_mask,audio,video):
        if input_ids.shape[1]!=50:raise ValueError('Prepared official MulT text length is50.')
        audio,audio_lengths=self.finite_inputs(audio,400,self.model.native.orig_d_a)
        video,video_lengths=self.finite_inputs(video,180,self.model.native.orig_d_v)
        tokens=torch.stack((input_ids,attention_mask,torch.zeros_like(input_ids)),dim=1)
        return self.model(tokens,video,audio,audio_lengths=audio_lengths,video_lengths=video_lengths)


class OfficialMulTTrainerObjective(OfficialMulTObjective):
    def forward(self,output1,output2,labels,llm_prior=None):
        if output2 is not None:raise ValueError('Prepared MulT protocol has one forward, no R-Drop.')
        masks={}
        if self.condition=='sati':
            if llm_prior is None:raise ValueError('Explicit frozen training prior required.')
            masks={'audio_mask':llm_prior.audio_necessary,'video_mask':llm_prior.video_necessary}
        total=super().forward(output1,labels,**masks)
        main=F.cross_entropy(output1.logits,labels)
        zero=main.new_zeros(())
        return {'total_loss':total,'ce_loss':main,'rdrop_loss':zero,
            'aux_loss':(total-main)/self.lambda_aux if self.condition!='native' and self.lambda_aux>0 else zero}
