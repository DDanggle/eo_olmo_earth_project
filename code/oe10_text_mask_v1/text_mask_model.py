"""Minimal shared conditional-mask baseline, not a proposed novel architecture.

Frozen Qwen text-only prefill -> request representation -> dense EO mask head.
P2 EO feature extraction is copied immutably. Qwen RGB/EO-slot generation and
language CE are deliberately not run by this module. It is a different baseline
from P2; compare its text conditions within this same model, not as P2 scores.
"""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
import torch
from torch import nn
import torch.nn.functional as F
from contracts import load_base,verify_base,render_context,context_sha

Base=load_base('episode_model')


def last_valid_hidden(hidden,attention_mask):
 """Correct for left/right padding and holes; do not assume length-1 is an index."""
 if hidden.ndim!=3 or attention_mask.shape!=hidden.shape[:2]:raise ValueError('Hidden/mask shape')
 if not bool(((attention_mask==0)|(attention_mask==1)).all()):raise ValueError('Nonbinary attention mask')
 mask=attention_mask.bool()
 if not bool(mask.any(dim=1).all()):raise ValueError('All-padding text is not allowed')
 pos=torch.arange(mask.shape[1],device=mask.device)[None,:].expand_as(mask)
 at=pos.masked_fill(~mask,-1).amax(1)
 return hidden[torch.arange(hidden.shape[0],device=hidden.device),at]


class ConditionalDenseHead(nn.Module):
 def __init__(self,dimension,text_dimension):
  super().__init__();self.dimension=dimension
  self.norm=nn.LayerNorm(dimension)
  self.text=nn.Sequential(nn.LayerNorm(text_dimension),nn.Linear(text_dimension,dimension),nn.GELU())
  self.net=nn.Sequential(nn.Linear(4*dimension+2,256),nn.GELU(),nn.Linear(256,1))
 def forward(self,query,positive,counter,text):
  q=self.norm(query);p=self.norm(positive);n=self.norm(counter)
  ep=p[:,None,None,:].expand_as(q);en=n[:,None,None,:].expand_as(q)
  et=self.text(text)[:,None,None,:].expand_as(q)
  cp=F.cosine_similarity(q,ep,dim=-1)[...,None];cn=F.cosine_similarity(q,en,dim=-1)[...,None]
  z=self.net(torch.cat([q,ep,en,cp,cn,et],-1)).permute(0,3,1,2)
  return F.interpolate(z,size=(128,128),mode='bilinear',align_corners=False)[0,0]


class TextConditionedEpisodeModel(Base.EpisodeModel):
 def __init__(self,*args,reader_identity,eo_identity,max_text_tokens=1024,max_text_cache_entries=0,**kwargs):
  verify_base()
  super().__init__(*args,**kwargs)
  if type(reader_identity) is not dict or not reader_identity:raise ValueError('Frozen reader/tokenizer identity required')
  self.reader_identity=json.loads(json.dumps(reader_identity,sort_keys=True,allow_nan=False))
  if type(eo_identity) is not dict or not eo_identity:raise ValueError('Frozen EO checkpoint and normalization identity required')
  self.eo_identity=json.loads(json.dumps(eo_identity,sort_keys=True,allow_nan=False))
  self.max_text_tokens=int(max_text_tokens);self.max_text_cache_entries=int(max_text_cache_entries)
  if not 1<=self.max_text_tokens<=8192 or self.max_text_cache_entries<0:raise ValueError('Text budget')
  # Remove the unused injection hook. Only public request tokens enter Qwen here.
  if self._embedding_hook is not None:self._embedding_hook.remove();self._embedding_hook=None
  self.connector.requires_grad_(False)
  self.head=ConditionalDenseHead(self.dimension,int(self.qwen.config.text_config.hidden_size)).to(self.device)
  self._text_cache=OrderedDict();self._reader_versions=self._text_versions()
  self.costs.update(text_prefill_calls=0,text_cache_hits=0,text_tokens_prefilled=0)

 def _text_versions(self):
  return tuple((id(t),t._version,str(t.device),str(t.dtype))
               for t in list(self.qwen.parameters())+list(self.qwen.buffers()))

 def clear_text_cache(self):
  self._text_cache.clear();self._reader_versions=self._text_versions()

 def text_representation(self,context):
  if any(p.requires_grad for p in self.qwen.parameters()):raise RuntimeError('Reader must stay frozen')
  if self._injection is not None:raise RuntimeError('EO/answer injection must not enter text prefill')
  request=render_context(context)
  message=[{'role':'user','content':[{'type':'text','text':request}]}]
  rendered=self.processor.apply_chat_template(message,tokenize=False,add_generation_prompt=False)
  batch=dict(self.processor.tokenizer(rendered,return_tensors='pt',padding=False,truncation=False,add_special_tokens=False))
  if 'input_ids' not in batch or 'attention_mask' not in batch:raise ValueError('Tokenizer missing IDs/mask')
  # Drop non-model tokenizer metadata; never pass labels, answer tokens or pixel data.
  batch={k:batch[k].to(self.device) for k in ('input_ids','attention_mask')}
  ids=batch['input_ids'];mask=batch['attention_mask']
  if ids.ndim!=2 or ids.shape[0]!=1 or mask.shape!=ids.shape:raise ValueError('Single request token shape')
  count=int(mask.sum())
  if count<1 or ids.shape[1]>self.max_text_tokens:raise ValueError('Text token budget exceeded; truncation forbidden')
  if not bool((mask==1).all()):raise ValueError('Production request prefill is unpadded batch size 1')
  versions=self._text_versions()
  if versions!=self._reader_versions:self.clear_text_cache()
  identity={'reader':self.reader_identity,'text':rendered,'token_ids':ids.detach().cpu().tolist(),
            'attention_mask':mask.detach().cpu().tolist(),'max_tokens':self.max_text_tokens}
  key=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  if key in self._text_cache:
   self.costs['text_cache_hits']+=1
   value=self._text_cache.pop(key);self._text_cache[key]=value
   return value.to(self.device),{'tokens':count,'cache_hit':True,'context_sha256':context_sha(context)}
  self.qwen.eval()
  with torch.no_grad():
   # Qwen3VLModel returns last_hidden_state without allocating LM-vocabulary logits
   # or every layer's hidden states. Actual pinned HF path needs a later real check.
   result=self.qwen.model(**batch,use_cache=False,output_hidden_states=False,return_dict=True)
   hidden=getattr(result,'last_hidden_state',None)
   if hidden is None:raise RuntimeError('Qwen base model did not expose last_hidden_state')
   text=last_valid_hidden(hidden,mask).detach().float()
  if text.shape!=(1,self.qwen.config.text_config.hidden_size) or not bool(torch.isfinite(text).all()):
   raise RuntimeError('Invalid frozen request representation')
  self.costs['text_prefill_calls']+=1;self.costs['text_tokens_prefilled']+=count
  if self.max_text_cache_entries:
   self._text_cache[key]=text.cpu().clone()
   while len(self._text_cache)>self.max_text_cache_entries:self._text_cache.popitem(last=False)
  return text,{'tokens':count,'cache_hit':False,'context_sha256':context_sha(context)}

 def forward(self,model_input,context,target=None):
  # Context is validated before any optional supervised target is read.
  text,text_audit=self.text_representation(context)
  query,positive,counter=self.features(model_input)
  logits=self.head(query,positive.mean(1),counter.mean(1),text)
  loss=None
  if target is not None:
   y,valid=self._target_arrays(target,self.device);z=logits[valid];yt=y[valid];p=z.sigmoid()
   loss=F.binary_cross_entropy_with_logits(z,yt)+1-(2*(p*yt).sum()+1)/(p.sum()+yt.sum()+1)
   if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite mask loss')
  return {'logits':logits,'loss':loss,'mask_loss':loss,'metrics':{
   'text':text_audit,'k_pairs':len(model_input['support_pairs']),'query_dates':2,
   'support_date_instances':16*len(model_input['support_pairs']),'language_ce_used':False,
   'language_generation_evaluated':False,'gold_in_text_prefill':False,
   'costs_cumulative':dict(self.costs)}}

 def trainable_parameters(self):
  for obj in [self.head]+([self.encoder] if self.arm=='B2' else []):
   yield from (p for p in obj.parameters() if p.requires_grad)

 def trainable_state_dict(self):
  return {'identity':{'base':verify_base(),'reader':self.reader_identity,'eo':self.eo_identity,'arm':self.arm,
                     'max_text_tokens':self.max_text_tokens,'architecture':'conditional_dense_v0'},
          'head':self.head.state_dict(),**({'encoder':self.encoder.state_dict()} if self.arm=='B2' else {})}

 def load_trainable_state_dict(self,state,strict=True):
  expected={'identity','head'}|({'encoder'} if self.arm=='B2' else set())
  if set(state)!=expected or state['identity']!=self.trainable_state_dict()['identity']:
   raise ValueError('Conditional baseline checkpoint identity mismatch')
  self.head.load_state_dict(state['head'],strict=strict)
  if self.arm=='B2':self.encoder.load_state_dict(state['encoder'],strict=strict)
  self.clear_cache();self.clear_text_cache()

 def close(self):
  super().close();self.clear_text_cache()
