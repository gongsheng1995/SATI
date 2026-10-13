"""Prepared MISA with explicit per-modality sequence-length correction.

Pinned MIntRec MISA native model/loss sources remain unchanged. A subclass uses
retained A/V frame lengths for packed RNNs; the source forward uses text lengths
for both. This is a declared adaptation, not a bitwise native reproduction.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F

from .vendor.misa.FusionNets.MISA import MISA
from .vendor.misa.native_losses import DiffLoss,CMD,MSE
from .official_mult_baseline import OfficialMulTForSATI


class LengthAwareMISA(MISA):
    def _extract_features(self,sequence,text_lengths,rnn1,rnn2,layer_norm):
        if self._retained_lengths is None:raise RuntimeError('Retained A/V lengths must be provided for this forward.')
        if rnn1 is self.vrnn1:lengths=self._retained_lengths[1]
        elif rnn1 is self.arnn1:lengths=self._retained_lengths[0]
        else:raise ValueError('Unexpected sequence extractor; no implicit text-length substitution.')
        return super()._extract_features(sequence,lengths.detach().cpu(),rnn1,rnn2,layer_norm)


@dataclass
class MISAOutput:
    logits:torch.Tensor
    native_states:dict


class MISAForSATI(nn.Module):
    def __init__(self,text_encoder_name='bert-large-uncased',audio_dim=768,video_dim=256,
                 hidden_dim=256,num_classes=30,dropout=.1,freeze_bert=False,tensor_rank=None,
                 use_cmd_sim=False,reverse_grad_weight=.8):
        super().__init__()
        if (audio_dim,video_dim,hidden_dim,num_classes)!=(768,256,256,30):
            raise ValueError('Prepared MISA protocol is WavLM768/Swin256, hidden256 and fixed30 classes.')
        directory=Path(__file__).parent/'vendor/misa'
        provenance=json.loads((directory/'source_provenance.json').read_text(encoding='utf-8'))
        for name,digest in provenance['files_sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:raise ValueError('Pinned MISA source changed.')
        from transformers import AutoConfig
        config=AutoConfig.from_pretrained(text_encoder_name,local_files_only=True)
        args=SimpleNamespace(text_backbone=text_encoder_name,cache_path=None,text_feat_dim=config.hidden_size,
            audio_feat_dim=audio_dim,video_feat_dim=video_dim,num_labels=num_classes,hidden_size=hidden_dim,
            dropout_rate=dropout,rnncell='lstm',use_cmd_sim=use_cmd_sim,reverse_grad_weight=reverse_grad_weight)
        self.native=LengthAwareMISA(args);self.native._retained_lengths=None
        self.num_classes=num_classes;self.use_cmd_sim=use_cmd_sim
        if freeze_bert:
            for parameter in self.native.text_subnet.parameters():parameter.requires_grad=False

    def forward(self,input_ids,attention_mask,audio,video):
        if input_ids.shape[1]!=50:raise ValueError('Prepared MISA text length50 required.')
        audio,audio_lengths=OfficialMulTForSATI.finite_inputs(audio,400,768)
        video,video_lengths=OfficialMulTForSATI.finite_inputs(video,180,256)
        tokens=torch.stack((input_ids,attention_mask,torch.zeros_like(input_ids)),dim=1)
        self.native._retained_lengths=(audio_lengths,video_lengths)
        try:logits=self.native(tokens,video,audio)
        finally:self.native._retained_lengths=None
        states={name:getattr(self.native,name) for name in (
            'utt_shared_t','utt_shared_v','utt_shared_a','utt_private_t','utt_private_v','utt_private_a',
            'utt_t_orig','utt_v_orig','utt_a_orig','utt_t_recon','utt_v_recon','utt_a_recon',
            'domain_label_t','domain_label_v','domain_label_a')}
        return MISAOutput(logits,states)


class MISAObjective(nn.Module):
    """Native CE plus six difference terms, similarity and mean reconstruction.

Full-batch terms are retained. No selected auxiliary classification heads are
added to this method baseline, and its objective does not consume LLM masks.
"""
    def __init__(self,diff_weight=.7,sim_weight=.7,recon_weight=.6,use_cmd_sim=False):
        super().__init__()
        if min(diff_weight,sim_weight,recon_weight)<0:raise ValueError('Nonnegative native weights required.')
        self.weights=(diff_weight,sim_weight,recon_weight);self.use_cmd_sim=use_cmd_sim
        self.diff=DiffLoss();self.cmd=CMD();self.reconstruction=MSE()

    def forward(self,output1,output2,labels,llm_prior=None):
        if output2 is not None:raise ValueError('MISA native objective uses one forward, no R-Drop.')
        s=output1.native_states
        shared=[s['utt_shared_'+m] for m in ('t','v','a')]
        private=[s['utt_private_'+m] for m in ('t','v','a')]
        difference=sum(self.diff(p,q) for p,q in zip(private,shared))
        difference+=sum(self.diff(private[i],private[j]) for i,j in ((2,0),(2,1),(0,1)))
        if self.use_cmd_sim:
            similarity=sum(self.cmd(shared[i],shared[j],5) for i,j in ((0,1),(0,2),(2,1)))/3
        else:
            predictions=[s['domain_label_'+m] for m in ('t','v','a')]
            if any(p is None for p in predictions):raise ValueError('Native domain similarity branch mismatch.')
            domain_targets=torch.cat([torch.full((p.shape[0],),index,device=labels.device,dtype=torch.long) for index,p in enumerate(predictions)])
            similarity=F.cross_entropy(torch.cat(predictions),domain_targets)
        reconstruction=sum(self.reconstruction(s['utt_'+m+'_recon'],s['utt_'+m+'_orig']) for m in ('t','v','a'))/3
        main=F.cross_entropy(output1.logits,labels)
        total=main+self.weights[0]*difference+self.weights[1]*similarity+self.weights[2]*reconstruction
        return {'total_loss':total,'ce_loss':main,'rdrop_loss':main.new_zeros(()),
            'aux_loss':total-main,'diff_loss':difference,'similarity_loss':similarity,'reconstruction_loss':reconstruction}
