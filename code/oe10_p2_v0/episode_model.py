"""Shared B0/B2 P1-derived episode model. No optimizer, scorer, native replay or I/O loader.

The single-episode interface accepts only EpisodeLoader.model_input. Query labels
are a separate optional supervised target. Frozen Qwen is differentiable with
respect to connector slots. This module does not implement free-form generation.
"""
from __future__ import annotations
from collections import OrderedDict
from contextlib import nullcontext
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PROMPT = ('Find regions matching the positive examples and exclude the counterexamples. '
          'Return a mask and the acquired observation dates supporting it.')
PACKET_KEYS = {'s2','raw_s2','timestamps','dates_yyyymmdd','observation_valid','band_observed'}


class DenseHead(nn.Module):
    """Unchanged P1 head; each role receives the mean of its K object prototypes."""
    def __init__(self, dimension: int):
        super().__init__()
        self.norm=nn.LayerNorm(dimension)
        self.net=nn.Sequential(nn.Linear(3*dimension+2,256),nn.GELU(),nn.Linear(256,1))

    def forward(self, query, positive, counterexample):
        q=self.norm(query);p=self.norm(positive);n=self.norm(counterexample)
        ep=p[:,None,None,:].expand_as(q);en=n[:,None,None,:].expand_as(q)
        cp=F.cosine_similarity(q,ep,dim=-1)[...,None]
        cn=F.cosine_similarity(q,en,dim=-1)[...,None]
        logits=self.net(torch.cat([q,ep,en,cp,cn],-1)).permute(0,3,1,2)
        return F.interpolate(logits,size=(128,128),mode='bilinear',align_corners=False)[0,0]


class Connector(nn.Module):
    """P1's 64 learned queries and three roles; all K exemplars retained as tokens.

    No position embedding was added to P1. Tensor order is preserved, while this
    cross-attention is mathematically invariant to a within-role permutation.
    """
    def __init__(self, dimension: int, language_dimension: int):
        super().__init__()
        if dimension%8:raise ValueError('P1 connector requires eight attention heads')
        self.dimension=dimension
        self.queries=nn.Parameter(torch.randn(1,64,dimension)*.02)
        self.role=nn.Parameter(torch.randn(3,dimension)*.02)
        self.norm=nn.LayerNorm(dimension)
        self.attn=nn.MultiheadAttention(dimension,8,batch_first=True,dropout=0)
        self.proj=nn.Linear(dimension,language_dimension)

    def forward(self, query, positive_all, counterexample_all):
        kv=torch.cat([query.reshape(1,-1,self.dimension)+self.role[0],
                      positive_all+self.role[1],counterexample_all+self.role[2]],1)
        value=self.norm(kv)
        return self.proj(self.attn(self.queries,value,value,need_weights=False)[0])


class EpisodeModel(nn.Module):
    def __init__(self, encoder, arm, device, qwen_path, gradient_checkpointing=True,
                 *, qwen_model=None, processor=None, embedding_dim=None,
                 sample_factory=None, online_mask_value=None, max_cache_entries=128):
        super().__init__()
        if arm not in {'B0','B2'}:raise ValueError('arm must be B0 or B2')
        self.arm=arm;self.device=torch.device(device)
        self.gradient_checkpointing=bool(gradient_checkpointing)
        self.encoder=encoder.to(self.device,dtype=torch.float32)
        self.encoder.requires_grad_(arm=='B2');self.encoder.eval()
        for module in self.encoder.modules():
            if getattr(module,'use_flash_attn',False):module.use_flash_attn=False
        config=getattr(self.encoder,'tokenization_config',None)
        if config is not None and config.get_num_bandsets('sentinel2_l2a')!=1:
            raise ValueError('P1 representation requires one sentinel2 bandset')
        self.dimension=int(embedding_dim or getattr(self.encoder,'embedding_size',0))
        if self.dimension<=0:raise ValueError('Encoder embedding_size is unavailable; supply embedding_dim')
        if sample_factory is None or online_mask_value is None:
            from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample,MaskValue
            sample_factory=sample_factory or MaskedOlmoEarthSample
            online_mask_value=MaskValue.ONLINE_ENCODER.value if online_mask_value is None else online_mask_value
        self._sample_factory=sample_factory;self._online_mask_value=int(online_mask_value)
        if qwen_model is None:
            if qwen_path is None:raise ValueError('A local pinned qwen_path is required')
            from transformers import Qwen3VLForConditionalGeneration
            qwen_model=Qwen3VLForConditionalGeneration.from_pretrained(str(qwen_path),
                local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')
        if processor is None:
            from transformers import AutoProcessor
            processor=AutoProcessor.from_pretrained(str(qwen_path),local_files_only=True)
        self.qwen=qwen_model.to(self.device,dtype=torch.float32)
        self.qwen.requires_grad_(False);self.qwen.eval();self.processor=processor
        self.head=DenseHead(self.dimension).to(self.device)
        self.connector=Connector(self.dimension,int(self.qwen.config.text_config.hidden_size)).to(self.device)
        self.costs={'EOencoder_calls':0,'patch_date_encodes':0,'vlm_forward_calls':0,
                    'cache_hits':0,'cache_misses':0,'cache_key_bytes_hashed':0,
                    'checkpoint_recompute_calls':0}
        self._cache=OrderedDict();self._cache_mode='frozen' if arm=='B0' else None
        self._max_cache_entries=int(max_cache_entries)
        if self._max_cache_entries<0:raise ValueError('max_cache_entries must be nonnegative')
        self._encoder_versions=self._parameter_versions()
        self._injection=None
        self._embedding_hook=self.qwen.get_input_embeddings().register_forward_hook(self._inject_slots)

    def train(self, mode=True):
        super().train(mode)
        # Eval mode fixes P1 dropout/stochastic behavior but does not freeze B2 gradients.
        self.encoder.eval();self.qwen.eval()
        if mode and self.arm=='B2':
            self.clear_cache()
            self._cache_mode=None
        return self

    def trainable_parameters(self):
        modules=[self.head,self.connector]+([self.encoder] if self.arm=='B2' else [])
        for module in modules:
            yield from (p for p in module.parameters() if p.requires_grad)

    def trainable_state_dict(self):
        result={'head':self.head.state_dict(),'connector':self.connector.state_dict()}
        if self.arm=='B2':result['encoder']=self.encoder.state_dict()
        return result

    def load_trainable_state_dict(self, state, strict=True):
        expected={'head','connector'}|({'encoder'} if self.arm=='B2' else set())
        if set(state)!=expected:raise ValueError('Trainable checkpoint modules differ from arm')
        for key in expected:getattr(self,key).load_state_dict(state[key],strict=strict)
        self.clear_cache()

    def _parameter_versions(self):
        # Parameters plus buffers protect normal optimizer/load_state_dict mutations.
        return tuple((id(t),t._version,tuple(t.shape),str(t.device),str(t.dtype))
                     for t in list(self.encoder.parameters())+list(self.encoder.buffers()))

    def clear_cache(self):
        self._cache.clear()
        self._encoder_versions=self._parameter_versions()

    @property
    def cache_mode(self):return self._cache_mode

    @cache_mode.setter
    def cache_mode(self, value):self.set_cache_mode(value)

    def set_cache_mode(self, mode):
        if mode not in {None,'none','frozen','evaluation'}:raise ValueError('Invalid cache mode')
        if mode=='frozen' and self.arm!='B0':raise ValueError('B2 cannot use a training feature cache')
        self.clear_cache();self._cache_mode=None if mode=='none' else mode
        return self

    def _cache_allowed(self):
        if self._max_cache_entries==0 or self._cache_mode is None:return False
        if self._cache_mode=='frozen':
            if any(p.requires_grad for p in self.encoder.parameters()):
                raise RuntimeError('Frozen cache cannot serve a trainable encoder')
        elif self.training or torch.is_grad_enabled():
            raise RuntimeError('Evaluation feature cache requires model.eval() and torch.no_grad()')
        versions=self._parameter_versions()
        if versions!=self._encoder_versions:self.clear_cache()
        return True

    def _content_key(self, observation):
        h=hashlib.sha256()
        h.update(f'oe10-p1-grid-v1:patch4:D{self.dimension}'.encode())
        for key in ['s2','timestamps','observation_valid','band_observed']:
            value=np.ascontiguousarray(observation[key])
            h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode())
            h.update(memoryview(value).cast('B'))
            self.costs['cache_key_bytes_hashed']+=value.nbytes
        return h.hexdigest()

    @staticmethod
    def _validate_observation(observation, dates, support=False):
        if set(observation)!=PACKET_KEYS|({'mask'} if support else set()):
            raise ValueError('Unexpected model packet fields; IDs/audits/gold are not model input')
        if np.shape(observation['s2'])!=(128,128,dates,12) or np.shape(observation['raw_s2'])!=(dates,10,128,128):
            raise ValueError('Expected original128 acquired EO observations')
        if np.shape(observation['timestamps'])!=(dates,3) or np.shape(observation['dates_yyyymmdd'])!=(dates,):
            raise ValueError('Timestamp/date shape mismatch')
        valid=np.asarray(observation['observation_valid'])
        if valid.shape!=(dates,128,128) or valid.dtype!=np.bool_ or not bool(valid.all()):
            raise ValueError('Strict baseline requires no sentinel-missing acquired observations; no cloud guarantee')
        if not np.isfinite(observation['s2']).all():raise ValueError('Nonfinite normalized EO input')
        if np.asarray(observation['band_observed']).tolist()!=[True]*10+[False]*2:
            raise ValueError('Expected explicit B01/B09 imputation flags')
        if support:
            mask=np.asarray(observation['mask'])
            if mask.shape!=(128,128) or mask.dtype!=np.bool_ or not bool(mask.any()):
                raise ValueError('Support mask must be a nonempty boolean original-grid object mask')

    def _encode_tensor(self, data, timestamps):
        self.costs['EOencoder_calls']+=1;self.costs['patch_date_encodes']+=int(data.shape[3])
        mask=torch.full((*data.shape[:-1],1),self._online_mask_value,dtype=torch.int32,device=self.device)
        sample=self._sample_factory(sentinel2_l2a=data,sentinel2_l2a_mask=mask,timestamps=timestamps)
        result=self.encoder(sample,patch_size=4,fast_pass=True)['tokens_and_masks'].sentinel2_l2a
        if result.ndim!=6 or result.shape[:3]!=(1,32,32) or result.shape[4]!=1 or result.shape[-1]!=self.dimension:
            raise ValueError('Unexpected native EO grid: '+str(tuple(result.shape)))
        return result.mean(dim=(3,4))

    def _encode(self, observation):
        use_cache=self._cache_allowed()
        key=self._content_key(observation) if use_cache else None
        if use_cache and key in self._cache:
            self.costs['cache_hits']+=1
            value=self._cache.pop(key);self._cache[key]=value
            return value.to(self.device)
        if use_cache:self.costs['cache_misses']+=1
        data=torch.as_tensor(observation['s2'],dtype=torch.float32,device=self.device).unsqueeze(0)
        timestamps=torch.as_tensor(observation['timestamps'],dtype=torch.int64,device=self.device).unsqueeze(0)
        context=torch.no_grad() if self.arm=='B0' else nullcontext()
        with context:
            if self.arm=='B2' and self.gradient_checkpointing and torch.is_grad_enabled():
                calls=[0]
                def encode_again(x,t):
                    if calls[0]:self.costs['checkpoint_recompute_calls']+=1
                    calls[0]+=1
                    return self._encode_tensor(x,t)
                grid=checkpoint(encode_again,data,timestamps,use_reentrant=False)
            else:grid=self._encode_tensor(data,timestamps)
        if use_cache:
            self._cache[key]=grid.detach().to('cpu').clone()
            while len(self._cache)>self._max_cache_entries:self._cache.popitem(last=False)
        return grid

    def _prototype(self, grid, mask):
        weight=F.avg_pool2d(torch.as_tensor(mask,dtype=torch.float32,device=self.device)[None,None],4).squeeze(0).squeeze(0)
        if float(weight.sum())<=0:raise ValueError('Empty pooled support')
        return (grid*weight[None,:,:,None]).sum((1,2))/weight.sum()

    def features(self, model_input):
        if set(model_input)!={'prompt','query','support_pairs'} or model_input['prompt']!=PROMPT:
            raise ValueError('Only fixed prompt and observation/support arrays may enter the model')
        pairs=model_input['support_pairs']
        if len(pairs) not in (1,2,4,8):raise ValueError('Expected K=1/2/4/8 support pairs')
        self._validate_observation(model_input['query'],2)
        self.encoder.eval()  # Also restore P1 mode if a shared native-replay wrapper called train().
        query=self._encode(model_input['query'])
        positive=[];counter=[]
        for pair in pairs:
            if set(pair)!={'positive','counterexample'}:raise ValueError('Invalid support roles')
            for role,destination in [('positive',positive),('counterexample',counter)]:
                self._validate_observation(pair[role],8,support=True)
                destination.append(self._prototype(self._encode(pair[role]),pair[role]['mask']))
        return query,torch.stack(positive,dim=1),torch.stack(counter,dim=1)

    @staticmethod
    def _target_arrays(target, device):
        y=torch.as_tensor(target['target_mask'],dtype=torch.float32,device=device)
        valid=torch.as_tensor(target['label_valid'],dtype=torch.bool,device=device)
        if y.shape!=(128,128) or valid.shape!=y.shape or not bool(valid.any()):
            raise ValueError('Expected nonempty original-grid source label validity')
        if not bool(((y==0)|(y==1)).all()):raise ValueError('Expected binary target mask')
        return y,valid

    def _language_batch(self, model_input, target):
        from PIL import Image
        raw=model_input['query']['raw_s2']
        rgb=np.clip(raw[-1,[2,1,0]].transpose(1,2,0)/3000,0,1)
        image=Image.fromarray(np.round(255*rgb).astype(np.uint8))
        message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':model_input['prompt']+
          ' Give target coverage as an integer percentage and reference region_1. The following EO slots encode the acquired query and positive/counterexample supports.'}]}]
        text=self.processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
        batch=dict(self.processor(text=[text],images=[image],return_tensors='pt',max_pixels=128*128))
        end=self.processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        positions=(batch['input_ids'][0]==end).nonzero().flatten()
        if not len(positions):raise RuntimeError('Qwen chat template has no user end position')
        at=int(positions[-1])
        placeholder=self.processor.tokenizer.encode('x',add_special_tokens=False)
        if not placeholder:raise RuntimeError('Missing placeholder token')
        keys=['input_ids','attention_mask','mm_token_type_ids','token_type_ids']
        for key in keys:
            if key not in batch:continue
            value=placeholder[0] if key=='input_ids' else (1 if key=='attention_mask' else 0)
            batch[key]=torch.cat([batch[key][:,:at],torch.full((1,64),value,dtype=batch[key].dtype),batch[key][:,at:]],1)
        prompt_length=batch['input_ids'].shape[1]
        answer=None
        if target is not None:
            y,valid=self._target_arrays(target,'cpu')
            fraction=round(100*float(y[valid].sum())/int(valid.sum()))
            answer=f'Target cover: {fraction} percent. Reference: region_1.'
            tids=torch.tensor([self.processor.tokenizer.encode(answer,add_special_tokens=False)],dtype=torch.int64)
            if not tids.shape[1]:raise RuntimeError('Empty language supervision')
            for key in keys:
                if key not in batch:continue
                extra=tids if key=='input_ids' else torch.full_like(tids,1 if key=='attention_mask' else 0)
                batch[key]=torch.cat([batch[key],extra],1)
            labels=torch.full_like(batch['input_ids'],-100);labels[:,prompt_length:]=tids
            batch['labels']=labels
        return {key:value.to(self.device) for key,value in batch.items()},at,prompt_length,answer

    def _inject_slots(self, module, args, output):
        if self._injection is None:return output
        ids,at,slots=self._injection
        if not torch.equal(args[0],ids):raise RuntimeError('EO injection input-ID mismatch')
        if output.shape[0]!=1 or slots.shape!=(1,64,output.shape[-1]):raise RuntimeError('EO slot shape mismatch')
        result=output.clone();result[:,at:at+64]=slots
        return result

    def forward(self, model_input, target=None, with_language=True):
        query,positive,counter=self.features(model_input)
        logits=self.head(query,positive.mean(dim=1),counter.mean(dim=1))
        mask_loss=None
        if target is not None:
            y,valid=self._target_arrays(target,self.device)
            z=logits[valid];yt=y[valid];p=z.sigmoid()
            mask_loss=F.binary_cross_entropy_with_logits(z,yt)+1-(2*(p*yt).sum()+1)/(p.sum()+yt.sum()+1)
        language_loss=logits.new_zeros(())
        last_logits=None;prompt_tokens=0
        if with_language:
            if self._injection is not None:raise RuntimeError('Concurrent EpisodeModel forward is unsupported')
            batch,at,prompt_tokens,_=self._language_batch(model_input,target)
            slots=self.connector(query,positive,counter)
            self._injection=(batch['input_ids'],at,slots)
            try:
                self.qwen.eval()
                result=self.qwen(**batch,use_cache=False)
                self.costs['vlm_forward_calls']+=1
                if target is not None:language_loss=result.loss
                last_logits=result.logits[:,-1].detach()
            finally:self._injection=None
        loss=None if mask_loss is None else mask_loss+.05*language_loss
        if loss is not None and not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite episode training loss')
        return {'logits':logits,'mask_loss':mask_loss,'language_loss':language_loss,'loss':loss,
                'vlm_last_logits':last_logits,'metrics':{'k_pairs':len(model_input['support_pairs']),
                    'query_dates':2,'support_date_instances':16*len(model_input['support_pairs']),
                    'connector_slots':64,'language_prompt_tokens':prompt_tokens,
                    'language_supervision':bool(with_language and target is not None),
                    'gold_in_model_input':False,'costs_cumulative':dict(self.costs)}}

    def close(self):
        if self._embedding_hook is not None:
            self._embedding_hook.remove();self._embedding_hook=None
        self.clear_cache()
