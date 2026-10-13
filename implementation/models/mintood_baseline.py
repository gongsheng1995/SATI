"""Prepared complete MIntOOD model and two-pass objectives; no runtime claim.

Leaf training/export and sealed scheduling are still required. Importing this
module is not model loading; the constructor must run under an authorized leaf.
"""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.mintood.FusionNets.MIntOOD_Fusion import Fusion_Model
from .vendor.mintood.FusionNets.MIntOOD import MLP_head
from .vendor.mintood.FusionNets.sampler import MIntOODSampler
from .vendor.mintood.loss import SupConLoss
from .official_mult_baseline import OfficialMulTForSATI

@dataclass
class MIntOODOutput:
    logits:torch.Tensor

class MIntOODForSATI(nn.Module):
    def __init__(self,text_encoder_name='bert-large-uncased',audio_dim=768,video_dim=256,
                 hidden_dim=1024,num_classes=30,freeze_bert=False):
        super().__init__()
        if (audio_dim,video_dim,hidden_dim,num_classes,freeze_bert)!=(768,256,1024,30,False):
            raise ValueError('Matched full BERT-large/WavLM768/Swin256/fixed30 required')
        directory=Path(__file__).parent/'vendor/mintood'
        provenance=json.loads((directory/'source_provenance.json').read_text())
        for name,r in provenance['files'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=r['adapter_sha256']:
                raise ValueError('Pinned source adapter differs')
        from transformers import AutoConfig
        config=AutoConfig.from_pretrained(text_encoder_name,local_files_only=True)
        if (config.hidden_size,config.num_hidden_layers)!=(1024,24):raise ValueError('BERT-large required')
        self.args=SimpleNamespace(text_pretrained_model=text_encoder_name,text_feat_dim=1024,
            audio_feat_dim=768,video_feat_dim=256,base_dim=1024,num_labels=30,ood_label_id=30,
            device=torch.device('cpu'),ablation_type='none',scale=16,mlp_hidden_size=256,mlp_dropout=.1,
            nheads=2,attn_dropout=0.,relu_dropout=.1,embed_dropout=0.,res_dropout=.2,attn_mask=False,
            encoder_layers_a=1,encoder_layers_v=2,weight_hidden_dim=256,weight_dropout=.5,
            binary_multiple_ood=1.,multiple_ood=1.,select_number_min=2,select_number_max=3,alpha=2)
        self.encoder,loading=Fusion_Model.from_pretrained(text_encoder_name,args=self.args,config=config,
            local_files_only=True,output_loading_info=True,_fast_init=False)
        missing=loading.get('missing_keys',[]);unexpected=loading.get('unexpected_keys',[])
        if any(not k.startswith('fusion.') for k in missing):raise ValueError('Missing pretrained BERT weights')
        if any(not k.startswith(('cls.','bert.pooler.','pooler.')) for k in unexpected) or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError('Pretrained loading mismatch')
        self.loading_audit={'new_fusion_keys':missing,'unused_pretraining_keys':unexpected,
            'initialization':'HF4.45.2 _fast_init=False; native custom constructors before pretrained loading'}
        self.sampler=MIntOODSampler(self.args)
        self.binary_classifier=MLP_head(self.args,2)
        self.multiclass_classifier=MLP_head(self.args,30)
        self.contrastive=SupConLoss()
        self.num_classes=30

    def _encode(self,input_ids,attention_mask,audio,video,labels=None,binary=False):
        if input_ids.ndim!=2 or input_ids.shape[1]!=50 or input_ids.shape!=attention_mask.shape:
            raise ValueError('Matched text cap50 required')
        self.args.device=input_ids.device
        audio,_=OfficialMulTForSATI.finite_inputs(audio,400,768)
        video,_=OfficialMulTForSATI.finite_inputs(video,180,256)
        if labels is not None:
            if not self.training:raise RuntimeError('Labeled mixture is training-only')
            if len(labels)<3 or labels.unique().numel()<2:raise ValueError('Sampler requires >=3 rows and >=2 classes; no silent batch modification')
            if ((labels<0)|(labels>=30)).any():raise ValueError('Invalid training classes')
        text=torch.stack((input_ids,attention_mask,torch.zeros_like(input_ids)),dim=1)
        return self.encoder(text,video,audio,self.sampler,labels,labels is not None,False,None,binary,None)

    def forward(self,input_ids,attention_mask,audio,video):
        # No labels or synthesis in the prediction interface, including dev/test.
        features=self._encode(input_ids,attention_mask,audio,video)
        scores=F.softmax(self.binary_classifier(features),dim=1)[:,1]
        logits,_=self.multiclass_classifier(features,binary_scores=scores)
        if not torch.isfinite(logits).all():raise FloatingPointError('Native cosine logits nonfinite')
        return MIntOODOutput(logits)

    def binary_training_loss(self,input_ids,attention_mask,audio,video,labels):
        features,mixed=self._encode(input_ids,attention_mask,audio,video,labels,binary=True)
        logits=self.binary_classifier(features)
        loss=F.cross_entropy(logits,mixed['binary'])
        if not torch.isfinite(loss):raise FloatingPointError('Binary loss nonfinite')
        return loss,{'select_elems':mixed['select_elems'],'n_original':len(labels),'n_mixed':len(mixed['binary'])}

    def multiclass_training_loss(self,input_ids,attention_mask,audio,video,labels):
        # Caller sets binary head eval, encoder/multiclass train for native second pass.
        first,mixed1=self._encode(input_ids,attention_mask,audio,video,labels)
        probs=F.softmax(self.binary_classifier(first),dim=1)
        scores=torch.cat((probs[:len(labels),1],probs[len(labels):,0])).detach()
        logits,contrast=self.multiclass_classifier(first,binary_scores=scores)
        main=F.cross_entropy(logits[:len(labels)],labels)
        second,mixed2=self._encode(input_ids,attention_mask,audio,video,labels)
        _,contrast2=self.multiclass_classifier(second)
        semi=mixed2['semi'];size=len(semi)
        expanded=semi.expand(size,size)
        mask=torch.eq(expanded,expanded.T).long()
        mask[semi==self.args.ood_label_id,:]=0
        mask=torch.scatter(mask,0,torch.arange(size,device=semi.device).unsqueeze(0),1)
        views=torch.stack((F.normalize(contrast),F.normalize(contrast2)),dim=1)
        aux=self.contrastive(views,mask=mask,temperature=2,device=semi.device)
        total=main+aux
        if not torch.isfinite(total):raise FloatingPointError('Multiclass/contrastive loss nonfinite')
        return total,{'ce_loss':main,'contrastive_loss':aux,'first_select_elems':mixed1['select_elems'],
                      'second_select_elems':mixed2['select_elems'],'n_original':len(labels),'n_mixed':size,
                      'original_correct':int((logits[:len(labels)].detach().argmax(-1)==labels).sum())}
