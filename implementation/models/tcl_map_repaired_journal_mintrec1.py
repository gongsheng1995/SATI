"""Prepared TCL-MAP adaptation; labels enter only its training contrastive view.

Native model/loss files are unchanged. Matching BERT-large and MIntRec2 features,
sequence caps and checkpoint selection are adaptations, not native score replicas.
Prospective repair after the invalid seed42 pilot. No runtime validation or new
training has been performed. Original executed adapter and sealed plan unchanged.
Disable HF fast initialization to preserve native custom-module constructors.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import weakref
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.tcl_map.model import MAP_Model, Cons_Model
from .vendor.tcl_map.loss import SupConLoss
from .official_mult_baseline import OfficialMulTForSATI


@dataclass
class TCLMAPOutput:
    logits: torch.Tensor
    main_ids: torch.Tensor | None = None
    main_mask: torch.Tensor | None = None
    condition_idx: torch.Tensor | None = None
    condition: torch.Tensor | None = None
    generated_ctx: torch.Tensor | None = None


class TCLMAPMIntRec1ForSATI(nn.Module):
    def __init__(self, text_encoder_name='bert-large-uncased', audio_dim=768,
                 video_dim=256, hidden_dim=1024, num_classes=20, dropout=.5,
                 freeze_bert=False, tensor_rank=None, label_list=None):
        super().__init__()
        if (audio_dim, video_dim, hidden_dim, num_classes, dropout, freeze_bert) != (768,256,1024,20,.5,False):
            raise ValueError('Prepared MIntRec1 protocol requires full BERT-large, WavLM768, Swin256,20 classes.')
        directory = Path(__file__).parent/'vendor/tcl_map'
        source = json.loads((directory/'source_provenance.json').read_text(encoding='utf-8'))
        for name,digest in source['files_sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
                raise ValueError('Pinned TCL-MAP source changed.')
        from transformers import AutoConfig, BertTokenizer
        config = AutoConfig.from_pretrained(text_encoder_name, local_files_only=True)
        if (config.hidden_size,config.num_hidden_layers) != (1024,24):
            raise ValueError('Pinned BERT-large required.')
        tokenizer = BertTokenizer.from_pretrained(text_encoder_name, local_files_only=True)
        audit = json.loads((directory/'tokenizer_audit_mintrec1.json').read_text(encoding='utf-8'))
        expected_labels = [r['label'] for r in audit['labels']]
        if list(label_list or []) != expected_labels:
            raise ValueError('Explicit canonical20-label MIntRec1 mapping must match frozen vocabulary audit.')
        if len(tokenizer) != 30522 or (tokenizer.cls_token_id,tokenizer.sep_token_id,tokenizer.mask_token_id,tokenizer.pad_token_id) != (101,102,103,0):
            raise ValueError('Pinned BERT vocabulary IDs required.')
        self.literal_mask_id = tokenizer.convert_tokens_to_ids('MASK')
        if self.literal_mask_id != 100: raise ValueError('Native literal MASK must retain UNK100 behavior.')
        table=[]
        for row in audit['labels']:
            ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(row['label']))
            if ids != row['ids'] or not 0<len(ids)<=4:
                raise ValueError('Canonical label tokenization differs from frozen audit.')
            table.append(ids+[self.literal_mask_id]*(4-len(ids)))
        self.register_buffer('label_token_ids',torch.tensor(table,dtype=torch.long))
        args = SimpleNamespace(text_feat_dim=1024,audio_feat_dim=768,video_feat_dim=256,
            text_seq_len=50,max_cons_seq_length=57,audio_seq_len=400,video_seq_len=180,
            need_aligned=True,mag_aligned_method='ctc',aligned_method='sim',shared_dim=256,
            eps=1e-9,beta_shift=.006,dropout_prob=.5,num_labels=20,label_len=4,
            prompt_len=3,nheads=8,n_levels=5,attn_dropout=.1,relu_dropout=0.,
            embed_dropout=.2,res_dropout=.1,attn_mask=True)
        self.main_model, main_loading = MAP_Model.from_pretrained(text_encoder_name,
            config=config,args=args,local_files_only=True,output_loading_info=True,_fast_init=False)
        self.cons_model, cons_loading = Cons_Model.from_pretrained(text_encoder_name,
            config=config,args=args,local_files_only=True,output_loading_info=True,_fast_init=False)
        self.loading_audit={}
        for name,loading,prefixes in (
            ('main',main_loading,('classifier.','bert.MAG.','bert.alignNet.','bert.audio_proj.',
                'bert.video_proj.','bert.text_proj.','bert.out_proj.','bert.trans_a_with_l.','bert.gamma')),
            ('contrastive',cons_loading,())):
            missing=loading.get('missing_keys',[]); unexpected=loading.get('unexpected_keys',[])
            if any(not k.startswith(prefixes) for k in missing):
                raise ValueError('Missing pretrained BERT layer in '+name+': '+str(missing))
            if any(not k.startswith('cls.') for k in unexpected) or loading.get('mismatched_keys') or loading.get('error_msgs'):
                raise ValueError('Unexpected pretrained mismatch in '+name+': '+str(loading))
            self.loading_audit[name]={'missing_new_component_keys':missing,'unused_pretraining_head_keys':unexpected}
        self.ctx=nn.Parameter(nn.init.trunc_normal_(torch.empty(3,1024)))
        self.num_classes=30
        self.loading_audit['initialization_policy'] = 'HF4.45.2 _fast_init=False; native constructors execute before pretrained load'
        for name,parameter in self.named_parameters():
            _require_finite(parameter, 'initial_parameter:'+name)
            if parameter.requires_grad:
                parameter.register_hook(_gradient_guard(name))

    def main_prompt_inputs(self, input_ids, attention_mask):
        # Input content is fixed basecap50 including CLS/SEP. Append native3+4 slots.
        if input_ids.ndim!=2 or input_ids.shape[1]!=50 or input_ids.shape!=attention_mask.shape:
            raise ValueError('Matched base text cap50 required.')
        lengths=attention_mask.sum(-1).long()
        positions=torch.arange(50,device=input_ids.device)[None,:]
        if not torch.equal(attention_mask.bool(),positions<lengths[:,None]):
            raise ValueError('Text must be contiguous end-padded.')
        if ((lengths<2)|(lengths>50)).any() or (input_ids[:,0]!=101).any():
            raise ValueError('Valid BERT CLS/content/SEP input required.')
        ids=input_ids.new_zeros((input_ids.shape[0],57)); mask=attention_mask.new_zeros(ids.shape)
        indices=lengths-1+3
        for i,n in enumerate(lengths.tolist()):
            if input_ids[i,n-1]!=102: raise ValueError('Last valid token must be SEP.')
            ids[i,:n-1]=input_ids[i,:n-1]
            ids[i,n-1:n+2]=self.literal_mask_id
            ids[i,n+2:n+6]=103
            ids[i,n+6]=102
            mask[i,:n+7]=1
        return ids,mask,indices

    def forward(self,input_ids,attention_mask,audio,video):
        ids,mask,indices=self.main_prompt_inputs(input_ids,attention_mask)
        audio,_=OfficialMulTForSATI.finite_inputs(audio,400,768)
        video,_=OfficialMulTForSATI.finite_inputs(video,180,256)
        text=torch.stack((ids,mask,torch.zeros_like(ids)),dim=1)
        outputs,_,condition,ctx=self.main_model(text,video,audio,indices,self.ctx)
        _require_finite(outputs[0], 'main_logits')
        _require_finite(condition, 'main_condition')
        _require_finite(ctx, 'generated_prompt')
        # Heldout prediction never calls Cons_Model and receives no example labels.
        if not self.training: return TCLMAPOutput(outputs[0])
        return TCLMAPOutput(outputs[0],ids,mask,indices,condition.mean(dim=1),ctx)

    def training_contrastive_view(self, output, labels):
        if not self.training or output.main_ids is None:
            raise RuntimeError('Ground-truth contrastive view is training-only.')
        ids=output.main_ids.clone()
        rows=torch.arange(len(labels),device=labels.device)[:,None]
        columns=output.condition_idx[:,None]+torch.arange(4,device=labels.device)[None,:]
        ids[rows,columns]=self.label_token_ids[labels]
        hidden=self.cons_model(condition_idx=output.condition_idx,ctx=output.generated_ctx,
            input_ids=ids,attention_mask=output.main_mask,token_type_ids=torch.zeros_like(ids)).last_hidden_state
        _require_finite(hidden, 'teacher_view_hidden')
        return hidden[rows,columns].mean(dim=1)


class TCLMAPObjective(nn.Module):
    def __init__(self,model):
        super().__init__()
        self._model_ref=weakref.ref(model)
        self.contrastive=SupConLoss(temperature=.5)

    def forward(self,output1,output2,labels,llm_prior=None):
        if output2 is not None: raise ValueError('Native TCL-MAP uses one paired view, no R-Drop.')
        model=self._model_ref()
        if model is None: raise RuntimeError('Training model is unavailable.')
        augmented=model.training_contrastive_view(output1,labels)
        features=torch.stack((output1.condition,augmented),dim=1)
        # Native manager supplies no class labels to SupCon: instance-pair positives.
        contrastive=self.contrastive(features)
        ce=F.cross_entropy(output1.logits,labels)
        _require_finite(ce, 'cross_entropy')
        _require_finite(contrastive, 'contrastive_loss')
        return {'total_loss':ce+contrastive,'ce_loss':ce,'rdrop_loss':ce.new_zeros(()),'aux_loss':contrastive}


def _require_finite(value, stage):
    if not torch.isfinite(value.detach()).all():
        raise FloatingPointError('TCL-MAP nonfinite value at '+stage+'; abort without optimizer update or valid-result claim.')


def _gradient_guard(name):
    def guard(gradient):
        _require_finite(gradient, 'gradient:'+name)
        return gradient
    return guard
