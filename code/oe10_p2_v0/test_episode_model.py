"""Small CPU substitutes validate wiring only, not EO/Qwen numeric or GPU behavior."""
from types import SimpleNamespace
import copy
import unittest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from episode_model import EpisodeModel,PROMPT


class ToyEncoder(nn.Module):
    embedding_size=8
    def __init__(self):
        super().__init__();self.project=nn.Linear(12,8)
    def forward(self,sample,patch_size,fast_pass):
        data=sample.sentinel2_l2a;t=data.shape[3]
        pooled=data.reshape(1,32,4,32,4,t,12).mean((2,4))
        out=self.project(pooled).unsqueeze(4)
        return {'tokens_and_masks':SimpleNamespace(sentinel2_l2a=out)}

class ToyTokenizer:
    def convert_tokens_to_ids(self,value):return 2
    def encode(self,text,add_special_tokens=False):return [7] if text=='x' else [10,11,12]

class ToyProcessor:
    tokenizer=ToyTokenizer()
    def apply_chat_template(self,*args,**kwargs):return 'fixed synthetic chat'
    def __call__(self,text,images,**kwargs):
        assert images[0].size==(128,128)
        return {'input_ids':torch.tensor([[1,8,2,9]]),'attention_mask':torch.ones(1,4,dtype=torch.int64),
                'mm_token_type_ids':torch.zeros(1,4,dtype=torch.int64),
                'pixel_values':torch.zeros(1,3,128,128)}

class ToyQwen(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=16))
        self.embed=nn.Embedding(64,16);self.lm=nn.Linear(16,64)
    def get_input_embeddings(self):return self.embed
    def forward(self,input_ids,labels=None,**kwargs):
        x=self.embed(input_ids)
        x=x.cumsum(1)/torch.arange(1,x.shape[1]+1,device=x.device)[None,:,None]
        logits=self.lm(x)
        loss=None if labels is None else F.cross_entropy(logits[:,:-1].reshape(-1,64),labels[:,1:].reshape(-1),ignore_index=-100)
        return SimpleNamespace(logits=logits,loss=loss)

def model(arm,checkpoint=True):
    return EpisodeModel(ToyEncoder(),arm,'cpu',None,checkpoint,qwen_model=ToyQwen(),
        processor=ToyProcessor(),sample_factory=SimpleNamespace,online_mask_value=1)

def observation(seed,dates,support=False):
    rng=np.random.default_rng(seed)
    result={'s2':rng.normal(size=(128,128,dates,12)).astype(np.float32),
        'raw_s2':np.full((dates,10,128,128),1000,dtype=np.int16),
        'timestamps':np.asarray([[i+1,0,2019] for i in range(dates)],dtype=np.int64),
        'dates_yyyymmdd':np.asarray([20190101+i for i in range(dates)],dtype=np.int64),
        'observation_valid':np.ones((dates,128,128),dtype=np.bool_),
        'band_observed':np.asarray([True]*10+[False]*2,dtype=np.bool_)}
    if support:
        mask=np.zeros((128,128),dtype=np.bool_);mask[16:72,24:84]=True;result['mask']=mask
    return result

def inputs(k=2):
    data={'prompt':PROMPT,'query':observation(0,2),'support_pairs':[
        {'positive':observation(10+i,8,True),'counterexample':observation(30+i,8,True)} for i in range(k)]}
    y=np.zeros((128,128),dtype=np.bool_);y[:64,:64]=True
    target={'target_mask':y,'label_valid':np.ones_like(y)}
    return data,target


class EpisodeModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_b0_all_k_supports_frozen_encoder_and_content_cache(self):
        torch.manual_seed(1);m=model('B0');x,y=inputs(8)
        out=m(x,y,with_language=False)
        self.assertEqual(tuple(out['logits'].shape),(128,128))
        self.assertEqual(m.costs['EOencoder_calls'],17)
        self.assertEqual(m.costs['patch_date_encodes'],130)
        self.assertEqual(m.costs['vlm_forward_calls'],0)
        out['loss'].backward()
        self.assertTrue(any(p.grad is not None for p in m.head.parameters()))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.encoder.parameters()))
        excluded={id(p) for p in m.encoder.parameters()}|{id(p) for p in m.qwen.parameters()}
        self.assertFalse(excluded & {id(p) for p in m.trainable_parameters()})
        m(x,y,with_language=False)
        self.assertEqual(m.costs['cache_hits'],17)
        before=m.costs['EOencoder_calls']
        with torch.no_grad():m.encoder.project.weight.add_(.01)
        m(x,y,with_language=False)
        self.assertEqual(m.costs['EOencoder_calls']-before,17)
        m.close()

    def test_b2_checkpoint_matches_live_encoder_gradients(self):
        torch.manual_seed(2);a=model('B2',True)
        torch.manual_seed(2);b=model('B2',False)
        x,y=inputs(2)
        aa=a(x,y,with_language=False);bb=b(x,y,with_language=False)
        self.assertTrue(torch.allclose(aa['logits'],bb['logits'],atol=1e-6))
        aa['loss'].backward();bb['loss'].backward()
        self.assertTrue(torch.allclose(a.encoder.project.weight.grad,b.encoder.project.weight.grad,atol=1e-6))
        self.assertGreater(float(a.encoder.project.weight.grad.norm()),0)
        self.assertGreater(a.costs['checkpoint_recompute_calls'],0)
        self.assertGreater(a.costs['EOencoder_calls'],5)
        self.assertEqual(b.costs['EOencoder_calls'],5)
        a.close();b.close()

    def test_every_support_pair_contributes_and_all_tokens_reach_connector(self):
        torch.manual_seed(3);m=model('B2',False);x,_=inputs(8)
        q,p,n=m.features(x)
        self.assertEqual(tuple(p.shape),(1,8,8));self.assertEqual(tuple(n.shape),(1,8,8))
        slots=m.connector(q,p,n)
        self.assertEqual(tuple(slots.shape),(1,64,16))
        before=m.head(q,p.mean(1),n.mean(1))
        changed=copy.deepcopy(x);changed['support_pairs'][-1]['positive']['s2']+=3
        q2,p2,n2=m.features(changed)
        after=m.head(q2,p2.mean(1),n2.mean(1))
        self.assertGreater(float((before-after).abs().max()),1e-6)
        self.assertGreater(float((slots-m.connector(q2,p2,n2)).abs().max()),1e-6)
        m.close()

    def test_frozen_language_model_preserves_language_gradient_to_b2_encoder(self):
        torch.manual_seed(4);m=model('B2',True);x,y=inputs(2)
        out=m(x,y,with_language=True)
        out['language_loss'].backward()
        self.assertGreater(float(m.encoder.project.weight.grad.norm()),0)
        self.assertTrue(any(p.grad is not None and float(p.grad.norm())>0 for p in m.connector.parameters()))
        self.assertTrue(all(p.grad is None for p in m.qwen.parameters()))
        self.assertTrue(all(p.grad is None for p in m.head.parameters()))
        self.assertIsNone(m._injection)
        self.assertEqual(m.costs['vlm_forward_calls'],1)
        m.eval()
        with torch.no_grad():prediction=m(x,target=None,with_language=True)
        self.assertIsNone(prediction['loss']);self.assertIsNone(prediction['mask_loss'])
        self.assertFalse(prediction['metrics']['language_supervision'])
        self.assertIsNotNone(prediction['vlm_last_logits'])
        m.close()

    def test_b2_cache_requires_eval_no_grad_and_state_excludes_qwen(self):
        torch.manual_seed(5);m=model('B2');x,y=inputs(1)
        with self.assertRaisesRegex(ValueError,'training feature cache'):m.set_cache_mode('frozen')
        m.set_cache_mode('evaluation')
        with self.assertRaisesRegex(RuntimeError,'eval'):m(x,y,with_language=False)
        m.eval()
        with torch.no_grad():
            m(x,with_language=False);calls=m.costs['EOencoder_calls'];m(x,with_language=False)
        self.assertEqual(m.costs['EOencoder_calls'],calls)
        self.assertEqual(m.costs['cache_hits'],3)
        self.assertEqual(set(m.trainable_state_dict()),{'encoder','head','connector'})
        m.train()
        self.assertEqual(m.cache_mode,None);self.assertEqual(len(m._cache),0)
        self.assertFalse(m.encoder.training);self.assertFalse(m.qwen.training)
        m.close()

if __name__=='__main__':unittest.main()
