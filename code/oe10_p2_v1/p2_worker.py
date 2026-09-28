"""Common B0/B2 worker; train-only engineering precedes locked development runs."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT=Path('/home/work/data/olmoearth')
SOURCE=ROOT/'oe4_native_v12_v0/source'
DEPS=ROOT/'oe4_native_v12_v0/deps'
EO=ROOT/'oe4_native_v12_v0/models/OlmoEarth-v1_2-Base'
VLM=ROOT/'models/Qwen3-VL-8B-Instruct'
NATIVE=ROOT/'oe4_native_v12_v0/runs/data_prepare_v1/data_manifest.json'
PREP=ROOT/'oe8_pastis_prepare_v0/prepared_v0'
EPISODES=ROOT/'oe8_pastis_prepare_v0/episodes_v0'
SNAP=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def dump(path,value):
 path=Path(path);tmp=path.with_suffix('.tmp')
 tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def sources():
 manifest=json.loads((SNAP/'source_manifest.json').read_text())['files']
 for name,h in manifest.items():
  if sha(SNAP/name)!=h:raise RuntimeError('source changed '+name)
 return manifest

def episode_order(catalog,seed,count):
 # Each four updates exposes each K once; each K permutation repeats only after
 # every episode of that K. Identical draws for both arms of a paired seed.
 buckets={k:sorted(eid for eid,e in catalog.items() if e['k_pairs']==k) for k in (1,2,4,8)}
 out=[];epoch=0
 while len(out)<count:
  shuffled={}
  for k,ids in buckets.items():
   ids=ids.copy();random.Random(seed+epoch*1009+k).shuffle(ids);shuffled[k]=ids
  for i in range(min(map(len,shuffled.values()))):
   for k in (1,2,4,8):out.append(shuffled[k][i])
  epoch+=1
 return out[:count]

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--mode',choices=['reference','resume','train'],required=True)
 ap.add_argument('--arm',choices=['B0','B2'],required=True)
 ap.add_argument('--seed',type=int,default=270927)
 ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--reference-dir',type=Path)
 ap.add_argument('--protocol',type=Path)
 a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 started=time.monotonic()
 receipt={'status':'initializing','arm':a.arm,'seed':a.seed,'mode':a.mode,
          'pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
          'source_hashes':sources(),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),
          'development_metrics_accessed':False,'new_method_tested':False}
 def update(**values):
  receipt.update(values);receipt['elapsed_seconds']=time.monotonic()-started
  dump(a.out/'receipt.json',receipt)
  print(json.dumps({'status':receipt['status'],'step':receipt.get('completed_updates',0),'seconds':receipt['elapsed_seconds']}),flush=True)
 update()
 try:
  sys.path[:0]=[str(SOURCE),str(DEPS)]
  import numpy as np
  import torch
  from native_replay import make_replay
  from episode_loader import EpisodeLoader
  from episode_model import EpisodeModel
  torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True
  random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
  device=torch.device('cuda:0');torch.cuda.set_device(device)
  protocol=json.loads(a.protocol.read_text()) if a.protocol else None
  if a.mode=='train' and (protocol is None or not protocol.get('locked_before_development_results')):
   raise RuntimeError('Training requires locked protocol')
  receipt['protocol_sha256']=sha(a.protocol) if a.protocol else None
  loader=EpisodeLoader(PREP,EPISODES,'train');catalog=loader.episodes
  replay=make_replay(SOURCE,DEPS,EO,NATIVE,device)
  if a.arm=='B0':replay.model.requires_grad_(False)
  module=EpisodeModel(replay.encoder,a.arm,device,VLM,gradient_checkpointing=True)
  # Encoder appears both in official native model and episode wrapper: dedupe.
  parameters=[];seen=set()
  for p in list(replay.model.parameters())+list(module.head.parameters())+list(module.connector.parameters()):
   if p.requires_grad and id(p) not in seen:parameters.append(p);seen.add(id(p))
  encoder_ids={id(p) for p in replay.encoder.parameters()}
  native_ids={id(p) for p in replay.model.parameters()}
  groups=[{'params':[p for p in parameters if id(p) in encoder_ids],'lr':1e-5,'role':'encoder'},
          {'params':[p for p in parameters if id(p) in native_ids and id(p) not in encoder_ids],'lr':1e-4,'role':'native_decoder'},
          {'params':[p for p in parameters if id(p) not in native_ids],'lr':1e-3,'role':'readouts'}]
  groups=[g for g in groups if g['params']]
  for g in groups:g['base_lr']=g['lr']
  optimizer=torch.optim.AdamW(groups,weight_decay=.01,foreach=False)
  total=protocol['updates_per_run'] if protocol else 2
  if a.mode=='train':order=episode_order(catalog,a.seed,total)
  else:
   ids=sorted(eid for eid,e in catalog.items() if e['k_pairs']==8)
   random.Random(a.seed).shuffle(ids);order=ids[:2]
  order_sha=hashlib.sha256(('\n'.join(order)+'\n').encode()).hexdigest()
  vlm_files=['config.json','model.safetensors.index.json']+sorted(p.name for p in VLM.glob('model-*.safetensors'))
  vlm_identity={name:{'sha256':sha(VLM/name),'bytes':(VLM/name).stat().st_size} for name in vlm_files}
  identity={'sources':receipt['source_hashes'],'arm':a.arm,'seed':a.seed,'order_sha256':order_sha,
            'protocol_sha256':receipt['protocol_sha256'],'vlm_identity':vlm_identity}
  def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
  def set_rng(r):
   random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
  def state():return {'native':replay.model.state_dict(),'head':module.head.state_dict(),'connector':module.connector.state_dict()}
  def save_checkpoint(path,step):
   tmp=path.with_suffix('.tmp');torch.save({'modules':state(),'optimizer':optimizer.state_dict(),'rng':rng(),'step':step,'identity':identity},tmp);tmp.replace(path)
   return sha(path)
  def restore(path):
   c=torch.load(path,map_location='cpu',weights_only=False)
   if c['identity']!=identity:raise RuntimeError('resume identity mismatch')
   replay.model.load_state_dict(c['modules']['native']);module.head.load_state_dict(c['modules']['head']);module.connector.load_state_dict(c['modules']['connector'])
   optimizer.load_state_dict(c['optimizer']);set_rng(c['rng']);return c['step']
  initial_anchor={n:p.detach().cpu().clone() for n,p in replay.encoder.named_parameters() if 'sentinel2' in n and p.ndim>1}
  costs={'training_observation_exposures':0,'native_raw_patch_date_instances':0,
         'native_online_patch_date_instances':0,'native_explicit_target_patch_date_instances':0,
         'training_seconds':0.,'evaluation_seconds':0.}
  logs=[];windows=[]
  def train_step(index):
   start=time.monotonic();eid=order[index]
   data=loader.load(eid,acquired_positions=[2,5]);gold=loader.training_target(eid,training=True)
   if protocol:
    warmup=protocol['warmup_updates'];progress=(index+1-warmup)/max(1,total-warmup)
    factor=min(1.,(index+1)/warmup) if index<warmup else .1+.9*.5*(1+math.cos(math.pi*min(1.,progress)))
    for g in optimizer.param_groups:g['lr']=g['base_lr']*factor
   optimizer.zero_grad(set_to_none=True)
   # Both arms observe identical replay data. B0 has no trainable native params.
   with torch.set_grad_enabled(a.arm=='B2'):
    nl,nm=replay.loss(index,a.seed+index+1)
   if not bool(torch.isfinite(nl)):raise RuntimeError('nonfinite native loss')
   if a.arm=='B2':nl.backward()
   native_grads=sum(p.grad is not None and bool(p.grad.abs().max()>0) for p in replay.encoder.parameters())
   output=module(data['model_input'],target=gold,with_language=True)
   downstream=output['mask_loss']+.05*output['language_loss']
   if not bool(torch.isfinite(downstream)):raise RuntimeError('nonfinite downstream loss')
   ce_gradient=None
   if index==0 and a.arm=='B2':
    anchor=next(p for n,p in replay.encoder.named_parameters() if 'sentinel2' in n and p.requires_grad and p.ndim>1)
    cg=torch.autograd.grad(output['language_loss'],anchor,retain_graph=True,allow_unused=True)[0]
    ce_gradient=None if cg is None else float(cg.norm())
    if ce_gradient is None or not math.isfinite(ce_gradient) or ce_gradient<=0:raise RuntimeError('no language gradient to encoder')
   downstream.backward()
   nonzero={name:sum(p.grad is not None and bool(p.grad.abs().max()>0) for p in obj.parameters())
            for name,obj in [('encoder',replay.encoder),('head',module.head),('connector',module.connector)]}
   if nonzero['head']==0 or nonzero['connector']==0:raise RuntimeError('readout gradient absent')
   if (a.arm=='B0' and nonzero['encoder']!=0) or (a.arm=='B2' and (nonzero['encoder']==0 or native_grads==0)):
    raise RuntimeError('encoder gradient arm contract')
   if any(p.grad is not None for p in replay.model.target_encoder.parameters()):raise RuntimeError('target gradient')
   grad=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
   pred=output['logits'].detach().float().cpu();ml=float(output['mask_loss']);ce=float(output['language_loss'])
   optimizer.step();torch.cuda.synchronize()
   elapsed=time.monotonic()-start
   costs['training_seconds']+=elapsed;costs['training_observation_exposures']+=2+16*catalog[eid]['k_pairs'];costs['native_raw_patch_date_instances']+=4
   costs['native_online_patch_date_instances']+=nm['online_patch_date_instances']
   costs['native_explicit_target_patch_date_instances']+=nm['explicit_target_patch_date_instances']
   row={'step':index+1,'episode_id':eid,'k':catalog[eid]['k_pairs'],'mask_loss':ml,'language_loss':ce,
        'native_loss':float(nl),'total_loss':float(nl)+ml+.05*ce,'native_metrics':nm,
        'native_encoder_gradient_tensors':native_grads,'language_encoder_anchor_gradient_norm':ce_gradient,
        'nonzero_gradients':nonzero,'grad_norm':float(grad),
        'seconds':elapsed,'learning_rates':{g['role']:g['lr'] for g in optimizer.param_groups},'model_metrics':output.get('metrics',{})}
   with (a.out/'train_log.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
   logs.append(row);del output,downstream,nl
   return row,pred
  def evaluate(step):
   # Public catalog drives inference. Query gold is opened only by separate scorer.
   start=time.monotonic();devloader=EpisodeLoader(PREP,EPISODES,'development');public=devloader.episodes
   bases={e['base_id'] for e in public.values() if e['k_pairs']==8}
   selected=sorted((e for e in public.values() if e['base_id'] in bases),key=lambda e:e['episode_id'])
   if len(bases)!=96 or len(selected)!=384:raise RuntimeError('common96 count mismatch')
   folder=a.out/f'predictions_step_{step:06d}';folder.mkdir()
   module.eval();module.set_cache_mode('evaluation')
   with (folder/'predictions.jsonl').open('x') as manifest,torch.no_grad():
    for i,e in enumerate(selected):
     inp=devloader.load(e['episode_id'],acquired_positions=[2,5])['model_input']
     output=module(inp,with_language=False)
     path=folder/f'{i:04d}.npz';np.savez_compressed(path,probability=output['logits'].sigmoid().cpu().numpy().astype(np.float32))
     manifest.write(json.dumps({'episode_id':e['episode_id'],'base_id':e['base_id'],'k':e['k_pairs'],'npz_path':path.name,'sha256':sha(path)})+'\n')
   module.set_cache_mode('frozen' if a.arm=='B0' else None);module.train()
   score=a.out/f'score_step_{step:06d}.json'
   subprocess.run([sys.executable,str(SNAP/'score_predictions.py'),'--prepared-root',str(PREP),'--episodes-root',str(EPISODES),
     '--prediction-dir',str(folder),'--gate-config',str(SNAP/'gate_config.json'),'--out',str(score)],check=True,stdout=subprocess.DEVNULL)
   result=json.loads(score.read_text());windows.append({'step':step,'auc':result['target_iou_auc'],'score_path':str(score),'score_sha256':sha(score)})
   costs['evaluation_seconds']+=time.monotonic()-start
   update(status='training',development_metrics_accessed=True,evaluation_windows=windows)
  torch.cuda.reset_peak_memory_stats()
  update(status='ready',native_identity=replay.identity,trainable_parameters=sum(p.numel() for p in parameters),episode_order_sha256=order_sha,
         episode_count=len(order),vlm_identity=vlm_identity)
  if a.mode=='resume':
   if a.reference_dir is None:raise RuntimeError('reference directory required')
   restored=restore(a.reference_dir/'step1.pt')
   if restored!=1:raise RuntimeError('expected step1')
   row,pred=train_step(1)
   expected=torch.load(a.reference_dir/'expected_step2.pt',map_location='cpu',weights_only=False)
   detail=[{'module':name,'parameter':key,'max_abs_difference':float((v.detach().cpu().double()-expected['modules'][name][key].double()).abs().max())}
           for name,values in state().items() for key,v in values.items()]
   differences={name:max(r['max_abs_difference'] for r in detail if r['module']==name) for name in state()}
   md=float((pred-expected['prediction']).abs().max());ld=abs(row['total_loss']-expected['row']['total_loss'])
   passed=max(differences.values())<=1e-6 and md<=1e-6 and ld<=1e-6
   update(status='cold_resume_passed' if passed else 'cold_resume_failed',fresh_process_resume=True,
          reference_pid=expected['pid'],current_pid=os.getpid(),state_max_abs_differences=differences,
          prediction_max_abs_difference=md,total_loss_abs_difference=ld,pass_result=passed,
          largest_state_differences=sorted(detail,key=lambda r:r['max_abs_difference'],reverse=True)[:20])
   if not passed or expected['pid']==os.getpid():raise RuntimeError('cold resume failed')
  elif a.mode=='reference':
   row,pred=train_step(0);checkpoint_sha=save_checkpoint(a.out/'step1.pt',1)
   row,pred=train_step(1)
   torch.save({'modules':state(),'row':row,'prediction':pred,'pid':os.getpid()},a.out/'expected_step2.pt')
   update(status='reference_completed',completed_updates=2,step1_sha256=checkpoint_sha,
          max_k=8,profile_steps=[{'seconds':r['seconds'],'k':r['k'],'native_loss':r['native_loss']} for r in logs])
  else:
   for index in range(total):
    if time.monotonic()-started>protocol['max_worker_seconds']:raise RuntimeError('worker resource cap; comparison inconclusive')
    row,_=train_step(index)
    if (index+1)%protocol['progress_interval_updates']==0:update(status='training',completed_updates=index+1,last_train_step=row)
    if (index+1)%protocol['evaluation_interval_updates']==0:evaluate(index+1)
    if (index+1)%protocol['checkpoint_interval_updates']==0:
     h=save_checkpoint(a.out/'latest.pt',index+1);update(completed_updates=index+1,latest_checkpoint_sha256=h)
   h=save_checkpoint(a.out/'final.pt',total)
   update(status='training_completed',completed_updates=total,checkpoint_sha256=h,evaluation_windows=windows)
  delta={n:float((p.detach().cpu()-initial_anchor[n]).abs().max()) for n,p in replay.encoder.named_parameters() if n in initial_anchor}
  update(encoder_anchor_max_abs_deltas=delta,costs=costs,model_costs=module.costs,
         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
         finished_utc=datetime.now(timezone.utc).isoformat())
 except Exception as exc:
  update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
