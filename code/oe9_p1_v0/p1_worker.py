#!/usr/bin/env python3
"""Bounded P1 engineering run, not B0/B2 evaluation or novelty evidence."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT=Path('/home/work/data/olmoearth')
SOURCE=ROOT/'oe4_native_v12_v0/source'
DEPS=ROOT/'oe4_native_v12_v0/deps'
EO=ROOT/'oe4_native_v12_v0/models/OlmoEarth-v1_2-Base'
VLM=ROOT/'models/Qwen3-VL-8B-Instruct'
PREP=ROOT/'oe8_pastis_prepare_v0/prepared_v0'
EPISODES=ROOT/'oe8_pastis_prepare_v0/episodes_v0'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def save_json(p,data):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--head-steps',type=int,default=192)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 started=time.monotonic()
 receipt={'status':'initializing','started_utc':datetime.now(timezone.utc).isoformat(),
          'scope':'P1 engineering with two training cases; no development scoring or B0/B2 comparison',
          'script_sha256':sha(__file__),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'support_dates':8,'query_dates':[2,5], 'spatial_resolution':[128,128],
          'new_method_tested':False,'native_replay_included':False,
          'p2_baseline_complete':False,'head_overfit_steps':a.head_steps}
 def update(**kw):
  receipt.update(kw);receipt['elapsed_seconds']=time.monotonic()-started
  save_json(a.out/'receipt.json',receipt)
  print(json.dumps({'status':receipt['status'],'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
 update()
 try:
  sys.path[:0]=[str(SOURCE),str(DEPS)]
  import numpy as np
  import torch
  import torch.nn as nn
  import torch.nn.functional as F
  import transformers
  from PIL import Image
  from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
  from olmoearth_pretrain.model_loader import load_model_from_path
  from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample,MaskValue
  from episode_loader import EpisodeLoader
  torch.set_num_threads(4)
  random.seed(270927);np.random.seed(270927);torch.manual_seed(270927);torch.cuda.manual_seed_all(270927)
  torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
  torch.backends.cuda.matmul.allow_tf32=False
  dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.cuda.reset_peak_memory_stats(dev)
  loader=EpisodeLoader(PREP,EPISODES,'train')
  public={r['episode_id']:r for r in map(json.loads,(EPISODES/'episodes_train.jsonl').read_text().splitlines())}
  scores=list(map(json.loads,(EPISODES/'scoring/scoring_train.jsonl').read_text().splitlines()))
  eligible=[r for r in scores if r['k_pairs']==1 and .10<=r['target_pixels']/r['label_valid_pixels']<=.50]
  eligible.sort(key=lambda r:hashlib.sha256(('oe9-p1-two-case:'+r['episode_id']).encode()).hexdigest())
  selected=[];seen=set()
  for r in eligible:
   if r['query_patch_id'] not in seen:
    selected.append(r);seen.add(r['query_patch_id'])
   if len(selected)==2:break
  if len(selected)!=2:raise RuntimeError('Need two nonempty training cases')
  examples=[]
  for r in selected:
   data=loader.load(r['episode_id'],acquired_positions=[2,5])
   gold=loader.training_target(r['episode_id'],training=True)
   examples.append({'input':data['model_input'],'audit':data['audit'],'gold':gold,'source':r})
  save_json(a.out/'selected_examples.json',{'selection':'train only, K1, target fraction .10-.50, fixed salted hash, unique patches',
    'rows':selected,'audits':[e['audit'] for e in examples]})
  hashes={x:sha(EO/x) for x in ['config.json','weights.pth']}
  if hashes!={'config.json':'0d531a67ad3e477e7011efabcceb01ed80f430aa0a0a3d344fe18cec0f229b8a','weights.pth':'57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec'}:raise RuntimeError('Pinned EO identity mismatch')
  full=load_model_from_path(str(EO));encoder=full.encoder;del full
  encoder.to(dev,dtype=torch.float32).eval()
  for mod in encoder.modules():
   if getattr(mod,'use_flash_attn',False):mod.use_flash_attn=False
  if encoder.tokenization_config.get_num_bandsets('sentinel2_l2a')!=1:raise RuntimeError('Unexpected bandsets')
  cost={'eo_forward_calls':0,'patch_date_encodes':0,'cached_head_steps':0,'vlm_forward_calls':0}
  shapes=[]
  def encode(obs):
   arr=obs['s2'];valid=obs['observation_valid']
   if arr.shape[:2]!=(128,128) or not bool(np.asarray(valid).all()):raise RuntimeError('P1 requires original128 no-sentinel observations')
   data=torch.as_tensor(arr,dtype=torch.float32,device=dev).unsqueeze(0)
   mask=torch.full((*data.shape[:-1],1),MaskValue.ONLINE_ENCODER.value,dtype=torch.int32,device=dev)
   sample=MaskedOlmoEarthSample(sentinel2_l2a=data,sentinel2_l2a_mask=mask,
      timestamps=torch.as_tensor(obs['timestamps'],dtype=torch.int64,device=dev).unsqueeze(0))
   out=encoder(sample,patch_size=4,fast_pass=True)['tokens_and_masks'].sentinel2_l2a
   if out.ndim!=6 or out.shape[:3]!=(1,32,32) or out.shape[4]!=1:raise RuntimeError('Unexpected native grid '+str(out.shape))
   cost['eo_forward_calls']+=1;cost['patch_date_encodes']+=arr.shape[2]
   if list(out.shape) not in shapes:shapes.append(list(out.shape))
   return out.mean(dim=(3,4))  # One annual spatial representation, not a per-date crop label.
  def proto(grid,mask):
   weight=F.avg_pool2d(torch.as_tensor(mask,dtype=torch.float32,device=dev)[None,None],4).squeeze()
   if float(weight.sum())<=0:raise RuntimeError('Empty support')
   return (grid*weight[None,:,:,None]).sum((1,2))/weight.sum()
  def features(e):
   q=encode(e['input']['query']);pair=e['input']['support_pairs'][0]
   p=proto(encode(pair['positive']),pair['positive']['mask'])
   n=proto(encode(pair['counterexample']),pair['counterexample']['mask'])
   return q,p,n
  update(status='encoding_original_resolution_training_examples',eo_checkpoint_hashes=hashes)
  with torch.no_grad():cached=[tuple(x.detach() for x in features(e)) for e in examples]
  D=cached[0][0].shape[-1]
  class DenseHead(nn.Module):
   def __init__(self):
    super().__init__();self.norm=nn.LayerNorm(D);self.net=nn.Sequential(nn.Linear(3*D+2,256),nn.GELU(),nn.Linear(256,1))
   def forward(self,q,p,n):
    q=self.norm(q);p=self.norm(p);n=self.norm(n)
    ep=p[:,None,None,:].expand_as(q);en=n[:,None,None,:].expand_as(q)
    cp=F.cosine_similarity(q,ep,dim=-1)[...,None];cn=F.cosine_similarity(q,en,dim=-1)[...,None]
    logits=self.net(torch.cat([q,ep,en,cp,cn],-1)).permute(0,3,1,2)
    return F.interpolate(logits,size=(128,128),mode='bilinear',align_corners=False)[0,0]
  head=DenseHead().to(dev)
  def gold(e):
   return (torch.as_tensor(e['gold']['target_mask'],dtype=torch.float32,device=dev),
           torch.as_tensor(e['gold']['label_valid'],dtype=torch.bool,device=dev))
  def mask_loss(logit,e):
   y,v=gold(e);z=logit[v];y=y[v];prob=z.sigmoid()
   return F.binary_cross_entropy_with_logits(z,y)+1-(2*(prob*y).sum()+1)/(prob.sum()+y.sum()+1)
  def measure():
   rows=[]
   with torch.no_grad():
    for f,e in zip(cached,examples):
     logit=head(*f);y,v=gold(e);pred=(logit>0)&v;target=y.bool()&v
     rows.append({'iou':float((pred&target).sum()/((pred|target).sum().clamp_min(1))),
       'predicted_fraction':float(pred.sum()/v.sum()),'target_fraction':float(target.sum()/v.sum()),
       'loss':float(mask_loss(logit,e))})
   return rows
  opt_head=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
  initial=measure();curve=[]
  for step in range(a.head_steps):
   idx=step%2;opt_head.zero_grad(set_to_none=True)
   loss=mask_loss(head(*cached[idx]),examples[idx]);loss.backward();opt_head.step();cost['cached_head_steps']+=1
   if (step+1)%16==0:
    rows=measure();curve.append({'step':step+1,'cases':rows})
    save_json(a.out/'head_curve.json',curve)
    update(status='two_case_frozen_head_fit',head_latest=curve[-1])
  head_final=measure()
  for i,(f,e) in enumerate(zip(cached,examples)):
   with torch.no_grad():pred=head(*f).sigmoid().cpu().numpy()
   np.savez_compressed(a.out/f'train_prediction_{i}.npz',probability=pred,target=e['gold']['target_mask'],valid=e['gold']['label_valid'])
  receipt['head_fit']={'initial':initial,'final':head_final,'all_iou_at_least_0_80':all(x['iou']>=.8 for x in head_final),
       'frozen_feature_cache_valid_only_for_this_head_fit':True,'interpretation':'training-set engineering fit, no generalization score'}
  update(status='loading_frozen_qwen_for_live_encoder_continuation',token_shapes=shapes)
  # Frozen VLM remains a real differentiable part of the following joint step.
  vlm_files=sorted([p for p in VLM.glob('*.safetensors')]+[VLM/'config.json',VLM/'model.safetensors.index.json'])
  vlm_identity={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in vlm_files if p.is_file()}
  if not any(k.endswith('.safetensors') for k in vlm_identity):raise RuntimeError('Missing VLM weights')
  model=Qwen3VLForConditionalGeneration.from_pretrained(str(VLM),local_files_only=True,dtype=torch.float32,attn_implementation='sdpa').to(dev)
  model.requires_grad_(False);model.eval()
  processor=AutoProcessor.from_pretrained(str(VLM),local_files_only=True)
  HD=model.config.text_config.hidden_size
  class Connector(nn.Module):
   def __init__(self):
    super().__init__();self.queries=nn.Parameter(torch.randn(1,64,D)*.02)
    self.role=nn.Parameter(torch.randn(3,D)*.02)
    self.norm=nn.LayerNorm(D);self.attn=nn.MultiheadAttention(D,8,batch_first=True,dropout=0)
    self.proj=nn.Linear(D,HD)
   def forward(self,q,p,n):
    kv=torch.cat([q.reshape(1,-1,D)+self.role[0],p[:,None]+self.role[1],n[:,None]+self.role[2]],1)
    x=self.attn(self.queries,self.norm(kv),self.norm(kv),need_weights=False)[0]
    return self.proj(x)
  connector=Connector().to(dev)
  modules=nn.ModuleDict({'encoder':encoder,'head':head,'connector':connector})
  trainable=[p for p in modules.parameters() if p.requires_grad]
  optimizer=torch.optim.AdamW([{'params':encoder.parameters(),'lr':1e-6},
       {'params':head.parameters(),'lr':1e-4},{'params':connector.parameters(),'lr':1e-4}],weight_decay=0)
  batches=[]
  for e in examples:
   rgb=np.clip(e['input']['query']['raw_s2'][-1,[2,1,0]].transpose(1,2,0)/3000,0,1)
   img=Image.fromarray(np.round(255*rgb).astype(np.uint8))
   message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':e['input']['prompt']+' Give target coverage as an integer percentage and reference region_1. The following EO slots encode the acquired query and positive/counterexample supports.'}]}]
   text=processor.apply_chat_template(message,tokenize=False,add_generation_prompt=True)
   batch=dict(processor(text=[text],images=[img],return_tensors='pt',max_pixels=128*128))
   end=processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
   at=int((batch['input_ids'][0]==end).nonzero().flatten()[-1])
   placeholder=processor.tokenizer.encode('x',add_special_tokens=False)[0]
   for key in ['input_ids','attention_mask','mm_token_type_ids','token_type_ids']:
    if key not in batch:continue
    value=placeholder if key=='input_ids' else (1 if key=='attention_mask' else 0)
    batch[key]=torch.cat([batch[key][:,:at],torch.full((1,64),value,dtype=batch[key].dtype),batch[key][:,at:]],1)
   plen=batch['input_ids'].shape[1]
   frac=round(100*e['source']['target_pixels']/e['source']['label_valid_pixels'])
   target=f'Target cover: {frac} percent. Reference: region_1.'
   tids=torch.tensor([processor.tokenizer.encode(target,add_special_tokens=False)])
   for key in ['input_ids','attention_mask','mm_token_type_ids','token_type_ids']:
    if key not in batch:continue
    extra=tids if key=='input_ids' else torch.full_like(tids,1 if key=='attention_mask' else 0)
    batch[key]=torch.cat([batch[key],extra],1)
   labels=torch.full_like(batch['input_ids'],-100);labels[:,plen:]=tids
   batch['labels']=labels
   batch={k:v.to(dev) for k,v in batch.items()}
   batches.append((batch,at,plen,target))
  state={'slots':None,'at':None,'expected_ids':None}
  def hook(mod,args,output):
   if state['slots'] is None:return output
   if not torch.equal(args[0],state['expected_ids']):raise RuntimeError('EO injection prefix mismatch')
   out=output.clone();at=state['at'];out[:,at:at+64]=state['slots'];return out
  handle=model.get_input_embeddings().register_forward_hook(hook)
  anchor_name,anchor=next((n,p) for n,p in encoder.named_parameters() if 'sentinel2' in n and p.requires_grad and p.ndim>1)
  def step(idx,do_step=True,ce_check=False):
   optimizer.zero_grad(set_to_none=True)
   q,p,n=features(examples[idx]);logit=head(q,p,n);ml=mask_loss(logit,examples[idx])
   batch,at,plen,target=batches[idx]
   state.update(slots=connector(q,p,n),at=at,expected_ids=batch['input_ids'])
   result=model(**batch,use_cache=False);cost['vlm_forward_calls']+=1;ce=result.loss
   ce_grad=None
   if ce_check:
    g=torch.autograd.grad(ce,anchor,retain_graph=True,allow_unused=True)[0]
    ce_grad={'name':anchor_name,'norm':None if g is None else float(g.norm()),'finite':g is not None and bool(torch.isfinite(g).all())}
   loss=ml+.05*ce
   if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite joint loss')
   loss.backward()
   stats={}
   for name,module in modules.items():
    gs=[p.grad for p in module.parameters() if p.requires_grad and p.grad is not None]
    stats[name]={'tensor_count':len(gs),'nonzero':sum(bool(g.abs().max()>0) for g in gs),'finite':all(bool(torch.isfinite(g).all()) for g in gs)}
   if not all(x['finite'] and x['nonzero']>0 for x in stats.values()):raise RuntimeError('Joint gradient contract failed: '+str(stats))
   torch.nn.utils.clip_grad_norm_(trainable,1)
   out={'loss':float(loss),'mask_loss':float(ml),'ce_loss':float(ce),'gradient':stats,'ce_encoder_gradient':ce_grad}
   pred=logit.detach().cpu();last=result.logits[:,-1].detach().cpu()
   if do_step:optimizer.step()
   state['slots']=None
   return out,pred,last
  def rng_state():
   return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
  def set_rng(r):
   random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
  update(status='live_joint_step_and_checkpoint',vlm_identity=vlm_identity,versions={'torch':torch.__version__,'transformers':transformers.__version__})
  before=anchor.detach().clone();first,_,_=step(0,ce_check=True)
  cg=first['ce_encoder_gradient']
  if not cg['finite'] or cg['norm'] is None or cg['norm']<=0:raise RuntimeError('No finite nonzero language gradient to encoder anchor')
  changed=bool((before!=anchor).any());del before
  checkpoint=a.out/'trainer_step1.pt'
  torch.save({'modules':modules.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),
      'step':1,'next_case_index':1,'source_sha':sha(__file__),'loader_sha':sha(Path(__file__).parent/'episode_loader.py'),
      'config':{'precision':'float32','patch_size':4,'query_positions':[2,5],'support_positions':list(range(8)),
                'mask_plus_ce_weight':.05,'native_replay':False,'scope':'engineering'},
      'frozen_vlm_identity':vlm_identity,'eo_initial_identity':hashes,'selected_episode_ids':[r['episode_id'] for r in selected]},checkpoint)
  update(status='checking_identical_continuation',first_joint_step=first,encoder_anchor_changed=changed,checkpoint_sha256=sha(checkpoint))
  reference,refmask,reflogit=step(1)
  expected={k:v.detach().cpu().clone() for k,v in modules.state_dict().items()}
  restored=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if restored['frozen_vlm_identity']!=vlm_identity:raise RuntimeError('Frozen VLM identity mismatch')
  modules.load_state_dict(restored['modules']);optimizer.load_state_dict(restored['optimizer']);set_rng(restored['rng']);del restored
  resumed,newmask,newlogit=step(1)
  delta=max(float((v.detach().cpu().double()-expected[k].double()).abs().max()) for k,v in modules.state_dict().items())
  parity={'loss_abs_difference':abs(reference['loss']-resumed['loss']),
          'mask_max_abs_difference':float((refmask-newmask).abs().max()),
          'vlm_logit_max_abs_difference':float((reflogit-newlogit).abs().max()),
          'all_trainable_state_max_abs_difference':delta,
          'atol':1e-6,'rtol':1e-5,
          'pass':bool(torch.allclose(refmask,newmask,atol=1e-6,rtol=1e-5) and torch.allclose(reflogit,newlogit,atol=1e-6,rtol=1e-5) and delta<=1e-6 and abs(reference['loss']-resumed['loss'])<=1e-6)}
  handle.remove()
  torch.cuda.synchronize()
  full_pass=bool(parity['pass'] and changed and receipt['head_fit']['all_iou_at_least_0_80'])
  final_status=('completed_p1_engineering' if full_pass else 'completed_engineering_checks_head_fit_gate_unmet') if parity['pass'] and changed else 'failed_continuation_contract'
  update(status=final_status,
         continuation=parity,reference_step2=reference,resumed_step2=resumed,cost_ledger=cost,
         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
         p1_two_case_overfit_pass=receipt['head_fit']['all_iou_at_least_0_80'],
         p1_full_gate_pass=full_pass,
         remaining=['No development metrics','No B0/B2 comparison','No native replay in this engineering trainer','No human responses','No second independent region'])
  if not parity['pass']:raise RuntimeError('Continuation mismatch')
 except Exception as e:
  update(status='failed',error=repr(e),traceback=traceback.format_exc())
  raise

if __name__=='__main__':main()
