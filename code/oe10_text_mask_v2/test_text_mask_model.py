"""CPU toy encoder/tokenizer/reader wiring only. Not actual EO/Qwen or semantics."""
from types import SimpleNamespace
import copy,json,unittest
import numpy as np
import torch
from torch import nn
from contracts import PROMPT
from text_mask_model import TextConditionedEpisodeModel,last_valid_hidden

class ToyEncoder(nn.Module):
 embedding_size=8
 def __init__(self):super().__init__();self.project=nn.Linear(12,8)
 def forward(self,sample,patch_size,fast_pass):
  x=sample.sentinel2_l2a;t=x.shape[3]
  z=self.project(x.reshape(1,32,4,32,4,t,12).mean((2,4))).unsqueeze(4)
  return {'tokens_and_masks':SimpleNamespace(sentinel2_l2a=z)}
class ToyTokenizer:
 def __call__(self,text,**kwargs):
  ids=torch.tensor([[ord(c)%255+1 for c in text]])
  return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
class ToyProcessor:
 tokenizer=ToyTokenizer()
 def apply_chat_template(self,messages,tokenize,add_generation_prompt):
  assert not tokenize and not add_generation_prompt and len(messages)==1
  assert messages[0]['role']=='user'
  return messages[0]['content'][0]['text']+'\nUSER_END'
class ToyBackbone(nn.Module):
 def __init__(self):super().__init__();self.embed=nn.Embedding(256,16)
 def forward(self,input_ids,attention_mask,use_cache,output_hidden_states,return_dict):
  assert not use_cache and not output_hidden_states and return_dict
  x=self.embed(input_ids);weight=torch.arange(1,x.shape[1]+1,dtype=x.dtype,device=x.device)[None,:,None]
  h=(x*weight).cumsum(1)/weight.cumsum(1)
  return SimpleNamespace(last_hidden_state=h)
class ToyQwen(nn.Module):
 def __init__(self):
  super().__init__();self.model=ToyBackbone();self.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=16))
 def get_input_embeddings(self):return self.model.embed

def model(arm='B2',seed=41,**kw):
 torch.manual_seed(seed)
 return TextConditionedEpisodeModel(ToyEncoder(),arm,'cpu',None,True,qwen_model=ToyQwen(),processor=ToyProcessor(),
  sample_factory=SimpleNamespace,online_mask_value=1,reader_identity={'kind':'toy_cpu_wiring_not_semantics','seed':seed},eo_identity={'kind':'toy_cpu_encoder','seed':seed,'normalization':'synthetic normal arrays'},**kw)
def packet(seed,t,support=False):
 rng=np.random.default_rng(seed)
 p={'s2':rng.normal(size=(128,128,t,12)).astype(np.float32),'raw_s2':np.ones((t,10,128,128),dtype=np.int16),
 'timestamps':np.asarray([[1+i,0,2019] for i in range(t)],dtype=np.int64),'dates_yyyymmdd':np.arange(20190101,20190101+t),
 'observation_valid':np.ones((t,128,128),dtype=bool),'band_observed':np.array([True]*10+[False]*2)}
 if support:
  p['mask']=np.zeros((128,128),dtype=bool);p['mask'][12:92,16:80]=True
 return p
def inputs():
 x={'prompt':PROMPT,'query':packet(1,2),'support_pairs':[{'positive':packet(2,8,True),'counterexample':packet(3,8,True)}]}
 y=np.zeros((128,128),dtype=bool);y[:72,:48]=True
 return x,{'target_mask':y,'label_valid':np.ones_like(y)}
def context(a='grapevine',b='leguminous fodder'):
 return {'instruction':PROMPT,'positive':a,'counterexample':b}

class ModelTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(1)
 def test_b2_text_changes_logits_and_eo_gradients(self):
  m=model();x,y=inputs();a=m(x,context(),y);a['loss'].backward()
  g=m.encoder.project.weight.grad.clone()
  self.assertGreater(float(g.norm()),0)
  self.assertTrue(all(p.grad is None for p in m.qwen.parameters()))
  self.assertTrue(all(p.grad is None for p in m.connector.parameters()))
  self.assertGreater(float(m.head.text[1].weight.grad.norm()),0)
  m.zero_grad(set_to_none=True)
  b=m(x,context('perennial woody vines with dormancy','forage legumes harvested repeatedly'),y);b['loss'].backward()
  self.assertGreater(float((a['logits']-b['logits']).abs().max()),1e-8)
  self.assertGreater(float((g-m.encoder.project.weight.grad).abs().max()),1e-10)
  m.close()
 def test_b0_freeze_roles_and_gold_invariance(self):
  m=model('B0');x,y=inputs();a=m(x,context(),y)
  altered={'target_mask':~y['target_mask'],'label_valid':y['label_valid']}
  b=m(x,context(),altered)
  self.assertTrue(torch.equal(a['logits'],b['logits']))
  self.assertFalse(torch.equal(a['loss'],b['loss']))
  a['loss'].backward();self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.encoder.parameters()))
  changed=copy.deepcopy(x);p=changed['support_pairs'][0];p['positive'],p['counterexample']=p['counterexample'],p['positive']
  c=m(changed,context())
  self.assertGreater(float((a['logits']-c['logits']).abs().max()),1e-8)
  query_changed=copy.deepcopy(x);query_changed['query']['s2']+=2
  q=m(query_changed,context())
  self.assertGreater(float((a['logits']-q['logits']).abs().max()),1e-8)
  self.assertTrue(all(not p.requires_grad for p in m.qwen.parameters()))
  m.close()
 def test_padding_pool_reject_empty(self):
  h=torch.arange(2*5*3).reshape(2,5,3).float();mask=torch.tensor([[0,0,1,1,1],[1,1,1,0,0]])
  out=last_valid_hidden(h,mask)
  self.assertTrue(torch.equal(out[0],h[0,4]));self.assertTrue(torch.equal(out[1],h[1,2]))
  with self.assertRaises(ValueError):last_valid_hidden(h,torch.zeros_like(mask))
 def test_cache_content_and_reader_mutation(self):
  m=model('B0',max_text_cache_entries=2);a,_=m.text_representation(context());b,meta=m.text_representation(context())
  self.assertTrue(meta['cache_hit']);self.assertTrue(torch.equal(a,b))
  _,meta=m.text_representation(context('other crop'));self.assertFalse(meta['cache_hit'])
  with torch.no_grad():m.qwen.model.embed.weight.add_(.2)
  c,meta=m.text_representation(context());self.assertFalse(meta['cache_hit']);self.assertFalse(torch.equal(a,c))
  m.close()
 def test_token_budget_no_silent_truncation(self):
  m=model(max_text_tokens=4)
  with self.assertRaisesRegex(ValueError,'truncation'):m.text_representation(context())
  m.close()
 def test_save_restore_and_wrong_reader_rejected(self):
  m=model();x,y=inputs();optimizer=torch.optim.SGD(list(m.trainable_parameters()),lr=.01)
  m(x,context(),y)['loss'].backward();optimizer.step()
  state=copy.deepcopy(m.trainable_state_dict());expected=m(x,context())['logits'].detach()
  self.assertEqual(set(state),{'identity','head','encoder'})
  other=model();other.load_trainable_state_dict(state);actual=other(x,context())['logits'].detach()
  self.assertTrue(torch.equal(expected,actual))
  wrong=model(seed=42)
  with self.assertRaisesRegex(ValueError,'identity'):wrong.load_trainable_state_dict(state)
  wrong.eo_identity={'kind':'different frozen encoder'}
  wrong.reader_identity=copy.deepcopy(m.reader_identity)
  with self.assertRaisesRegex(ValueError,'identity'):wrong.load_trainable_state_dict(state)
  m.close();other.close();wrong.close()
 def test_b2_training_cache_rejected_and_eval_mutation_invalidates(self):
  m=model();x,_=inputs()
  with self.assertRaises(ValueError):m.set_cache_mode('frozen')
  m.eval();m.set_cache_mode('evaluation')
  with torch.no_grad():
   a=m(x,context());before=m.costs['EOencoder_calls']
   m(x,context());self.assertEqual(m.costs['EOencoder_calls'],before)
   m.encoder.project.weight.add_(.01)
   m(x,context());self.assertGreater(m.costs['EOencoder_calls'],before)
  m.train();self.assertIsNone(m.cache_mode);m.close()
 def test_unknown_input_and_context_fields_rejected(self):
  m=model();x,_=inputs()
  with self.assertRaises(ValueError):m(dict(x,target_mask=np.zeros((128,128))),context())
  with self.assertRaises(ValueError):m(x,dict(context(),answer='yes'))
  m.close()
if __name__=='__main__':unittest.main()
