"""Independently trained text baseline; no audio/video modules or fusion."""
import json
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from models.sati import SATIOutput

class TextOnly(nn.Module):
    def __init__(self,text_encoder_name='bert-large-uncased',hidden_dim=768,num_classes=30,dropout=.2,freeze_bert=False,**unused):
        super().__init__();self.num_classes=num_classes
        self.text_encoder=AutoModel.from_pretrained(text_encoder_name)
        self.text_proj=nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size,hidden_dim),nn.LayerNorm(hidden_dim),nn.Dropout(dropout))
        self.classifier=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Dropout(dropout),nn.Linear(hidden_dim,num_classes))
        for name,module in self.named_modules():
            if 'text_encoder' not in name and isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight);nn.init.zeros_(module.bias)
        if freeze_bert:
            for p in self.text_encoder.parameters():p.requires_grad=False
        for p in self.parameters():
            if not torch.isfinite(p).all():raise ValueError('Nonfinite initial text parameters')
            if p.requires_grad:p.register_hook(self.finite_gradient)
    @staticmethod
    def finite_gradient(g):
        if not torch.isfinite(g).all():raise ValueError('Nonfinite text gradient')
        return g
    def forward(self,input_ids,attention_mask,audio_feat=None,video_feat=None):
        encoded=self.text_encoder(input_ids=input_ids,attention_mask=attention_mask)
        h=self.text_proj(encoded.pooler_output if encoded.pooler_output is not None else encoded.last_hidden_state[:,0])
        logits=self.classifier(h)
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite text logits')
        zeros=torch.zeros_like(logits)
        return SATIOutput(logits,logits,zeros,zeros,h,h)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,data_json,audio_feats_path=None,video_feats_path=None,prior_file=None,tokenizer_name='bert-large-uncased',max_length=128,split='train'):
        data=json.load(open(data_json,encoding='utf-8'));self.samples=data[split];self.max_length=max_length
        self.label_list=data.get('label_list') or sorted({r['label'] for s in ('train','dev','test') for r in data[s]});self.label2id={x:i for i,x in enumerate(self.label_list)};self.num_classes=len(self.label_list)
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_name)
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        row=self.samples[i];tokens=self.tokenizer(row['text'],max_length=self.max_length,padding='max_length',truncation=True,return_tensors='pt')
        return dict(input_ids=tokens['input_ids'].squeeze(0),attention_mask=tokens['attention_mask'].squeeze(0),
                    audio=torch.zeros(1),video=torch.zeros(1),label=torch.tensor(self.label2id[row['label']]),
                    text_sufficient=True,audio_necessary=False,video_necessary=False)
