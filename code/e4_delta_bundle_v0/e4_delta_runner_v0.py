#!/usr/bin/env python3
"""Frozen E4 runner. Post-E3 exploratory delta routing probe; no training."""
import argparse, collections, hashlib, json, os, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from e4_delta_probe_v0 import ARMS, transform_pair
ROOT=Path('/home/work/data/olmoearth')
PHEN={'landslide':('Sentinel-2','landslide'),'flood':('Sentinel-1','flood')}
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def require(c,msg):
 if not c:raise ValueError(msg)
def verify_parent(cfg):
 parent=ROOT/cfg['parent']['directory'];pins=cfg['parent']
 for name,key in [('manifest.json','manifest_sha256'),('scores.json','scores_sha256'),('items.jsonl','items_sha256'),('pairs.npz','pairs_sha256'),('saved_e2_real.json','saved_e2_real_sha256')]:
  require(sha(parent/name)==pins[key],'Pinned E3 parent changed: '+name)
 for seed,h in pins['real_answer_sha256'].items():require(sha(parent/f'answers_seed{seed}_real.jsonl')==h,'Parent real answers changed')
 require(read(parent/'status.json')['status']=='completed' and read(parent/'scores.json')['valid'],'E3 parent is not valid and complete')
 return parent,read(parent/'manifest.json')
def prepare(a):
 cfg=read(a.config);parent,pm=verify_parent(cfg);out=ROOT/a.out
 require(a.out=='e4_delta_probe_v0','Unexpected output path')
 require(cfg['population']['n_items']==209 and cfg['population']['n_generations']==2508,'Plan population mismatch')
 require(list(cfg['arms'])==list(ARMS),'Plan arms mismatch')
 require(all(pm['models'][seed]['sha256']==h for seed,h in cfg['checkpoints'].items()),'Checkpoint pin mismatch')
 for f,key in [('items.jsonl','items_sha256'),('pairs.npz','pairs_sha256')]:
  shutil.copyfile(parent/f,out/f)
  require(sha(out/f)==cfg['parent'][key],'Copied parent bytes changed: '+f)
 items=lines(out/'items.jsonl');ids={i['id'] for i in items}
 require(len(ids)==len(items)==209,'Parent item coverage')
 require(collections.Counter(i['phen'] for i in items)=={'flood':165,'landslide':44},'Parent phenomenon counts')
 ref={'e2_real':read(parent/'saved_e2_real.json'),'e3_real':{}}
 for seed in ['1','2','3']:
  rows=lines(parent/f'answers_seed{seed}_real.jsonl');mapping={x['id']:x['parsed'] for x in rows}
  require(len(rows)==len(mapping)==209 and set(mapping)==ids,'Parent real coverage')
  ref['e3_real'][seed]=mapping
 for name,mapping in ref.items():
  require(set(mapping)=={'1','2','3'},'Reference seed coverage')
  require(all(set(v)==ids and all(x in ('yes','no') for x in v.values()) for v in mapping.values()),'Reference invalid')
 write(out/'references.json',ref)
 audit=[]
 with np.load(out/'pairs.npz') as data:
  require(set(data.files)=={i['pair_key'] for i in items},'Pair archive coverage')
  for it in items:
   pair=data[it['pair_key']]
   require(pair.dtype==np.float32 and pair.shape==(2,64,768) and np.isfinite(pair).all(),'Invalid frozen pair')
   original=pair.copy();delta=pair[1]-pair[0];maxerr=0.
   for arm in ARMS:
    t,ty=transform_pair(pair,arm)
    require(t.dtype==np.float32 and t.shape==(192,768) and np.isfinite(t).all(),'Invalid transformed tokens')
    require(np.array_equal(ty,np.repeat([0,1,3],64)) and np.array_equal(pair,original),'Type/input mutation')
    if arm=='delta_feature_permute':
     require(np.array_equal(np.sort(delta,axis=1),np.sort(t[128:],axis=1)),'Permutation multiset changed')
     x,y=delta.astype('float64'),t[128:].astype('float64')
     for stat in [lambda z:z.mean(1),lambda z:z.var(1),lambda z:np.linalg.norm(z,axis=1)]:
      u,v=stat(x),stat(y);require(np.allclose(u,v,rtol=1e-6,atol=1e-6),'Permutation moment changed');maxerr=max(maxerr,float(np.abs(u-v).max()))
   audit.append({'id':it['id'],'pair_key':it['pair_key'],'all_four_finite':True,'permutation_stat_max_abs_error':maxerr})
 write(out/'input_audit.json',{'n_items':len(items),'all':audit})
 snap=out/'code_snapshot';snap.mkdir()
 sources=['e4_delta_runner_v0.py','e4_delta_probe_v0.py','run_e4_when_idle_v0.py']
 for fn in sources:shutil.copyfile(Path(__file__).parent/fn,snap/fn)
 shutil.copyfile(a.config,out/'prereg.json')
 ps=out/'parent_snapshot';ps.mkdir()
 for fn in ['manifest.json','scores.json','status.json','prereg.json']:shutil.copyfile(parent/fn,ps/fn)
 manifest={'schema':'e4-delta-prepared-v0','prepared_at':now(),'scope':cfg['scope'],
  'items_sha256':sha(out/'items.jsonl'),'pairs_sha256':sha(out/'pairs.npz'),'references_sha256':sha(out/'references.json'),
  'prereg_sha256':sha(out/'prereg.json'),'input_audit_sha256':sha(out/'input_audit.json'),
  'parent_snapshot_sha256':{p.name:sha(p) for p in ps.iterdir()},'models':pm['models'],'llm_files_sha256':pm['llm_files_sha256'],
  'code_snapshot_sha256':{fn:sha(snap/fn) for fn in sources},'n_items':209,'n_generations':2508,
  'gpu_index':0,'environment':{'python':sys.version,'numpy':np.__version__}}
 verify_parent(cfg)
 write(out/'manifest.json',manifest);write(out/'status.json',{'status':'prepared','at':now()});print('E4 PREPARED '+json.dumps({'items':209,'generations':2508,'manifest_sha256':sha(out/'manifest.json')}),flush=True)

def run(a):
 import re
 import torch
 import torch.nn as nn
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from e4_delta_probe_v0 import score_run
 torch.set_num_threads(2);out=ROOT/a.out;m=read(out/'manifest.json');cfg=read(out/'prereg.json')
 for fn,key in [('items.jsonl','items_sha256'),('pairs.npz','pairs_sha256'),('references.json','references_sha256'),('prereg.json','prereg_sha256')]:
  if sha(out/fn)!=m[key]:raise ValueError('Frozen prepared input changed: '+fn)
 for fn,h in m['code_snapshot_sha256'].items():
  if sha(Path(__file__).parent/fn)!=h:raise ValueError('Must run unchanged frozen source snapshot')
 if sha(out/'input_audit.json')!=m['input_audit_sha256']:raise ValueError('Input audit changed')
 for fn,h in m['parent_snapshot_sha256'].items():
  if sha(out/'parent_snapshot'/fn)!=h:raise ValueError('Parent snapshot changed')
 if read(out/'status.json')['status']!='prepared':raise ValueError('Run already started/completed; use new version for repair')
 for v in m['models'].values():
  if sha(v['path'])!=v['sha256']:raise ValueError('Frozen checkpoint changed')
 for path,hsh in m['llm_files_sha256'].items():
  if sha(path)!=hsh:raise ValueError('Frozen LLM/tokenizer file changed: '+path)
 verify_parent(cfg)
 # The launcher also checks; keep a final guard immediately before allocation.
 uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 if uuid in busy:raise RuntimeError('GPU0 occupied; do not allocate')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise RuntimeError('Explicit GPU0 mapping required')
 items=lines(out/'items.jsonl');data=np.load(out/'pairs.npz');references=read(out/'references.json');start=time.perf_counter();rows=[];validity={}
 def emit_status(status,**kw):write(out/'status.json',{'status':status,'at':now(),'elapsed_s':time.perf_counter()-start,**kw})
 emit_status('running',stage='load_model');dev=torch.device('cuda');torch.manual_seed(0)
 try:
  llm_dir=ROOT/'olmo_llm/Olmo-3-7B-Instruct';tok=AutoTokenizer.from_pretrained(str(llm_dir));llm=AutoModelForCausalLM.from_pretrained(str(llm_dir),dtype=torch.bfloat16).to(dev).eval()
  for p in llm.parameters():p.requires_grad_(False)
  emb=llm.get_input_embeddings();h=emb.weight.shape[1];rms=float(emb.weight.detach().float().pow(2).mean().sqrt())
  class Projector(nn.Module):
   def __init__(self):
    super().__init__();self.mlp=nn.Sequential(nn.Linear(768,2048),nn.GELU(),nn.Linear(2048,h));self.ttype=nn.Embedding(4,h);self.norm=nn.LayerNorm(768);self.out=nn.LayerNorm(h);self.gain=nn.Parameter(torch.tensor(1.0))
   def forward(self,t,ty):return self.out(self.mlp(self.norm(t)))*(rms*self.gain)+self.ttype(ty)*rms
  env={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(0),'gpu_uuid':uuid,'llm_config_sha256':sha(llm_dir/'config.json'),'llm_weight_files':{p.name:{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in llm_dir.glob('*.safetensors')}}
  write(out/'runtime_environment.json',env)
  projs={}
  for seed in [1,2,3]:
   proj=Projector().to(dev);state=torch.load(m['models'][str(seed)]['path'],map_location='cpu',weights_only=True)
   if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite projector')
   proj.load_state_dict(state,strict=True);projs[seed]=proj.eval()
  prompts={};prompt_snapshot={}
  for it in items:
   sensor,word=PHEN[it['phen']];user=f"These are 2 {sensor} observations of the same area in chronological order, taken on {', '.join(it['dates'])}: <EO> Did a {word} occur between the two observations? Answer with yes or no."
   text=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True);pre,post=text.split('<EO>')
   prompts[it['id']]=tuple(tok(v,add_special_tokens=False,return_tensors='pt').input_ids[0].to(dev) for v in [pre,post])
   prompt_snapshot[it['id']]={'user_text':user,'chat_text':text,'prefix_ids':prompts[it['id']][0].cpu().tolist(),'suffix_ids':prompts[it['id']][1].cpu().tolist(),'n_eo_tokens':192}
  write(out/'prompt_snapshot.json',prompt_snapshot)
  def generate(seed,arm,it):
   if time.perf_counter()-start>cfg['compute']['max_runtime_minutes']*60:raise TimeoutError('Pre-registered runtime budget reached')
   t,ty=transform_pair(data[it['pair_key']],arm);e=projs[seed](torch.from_numpy(t).to(dev),torch.from_numpy(ty).to(dev)).to(torch.bfloat16)
   if not torch.isfinite(e).all():raise ValueError('Nonfinite projected tokens')
   pre,post=prompts[it['id']];inputs=torch.cat([emb(pre),e,emb(post)])
   gen=llm.generate(inputs_embeds=inputs[None],attention_mask=torch.ones(1,len(inputs),dtype=torch.long,device=dev),max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id,return_dict_in_generate=True,output_scores=True)
   if any(torch.isnan(s).any() or torch.isposinf(s).any() or not torch.isfinite(s).any() for s in gen.scores):raise ValueError('Invalid generation scores')
   raw=tok.decode(gen.sequences[0],skip_special_tokens=True);parsed=re.search(r'\b(yes|no)\b',raw.strip().lower())
   return {'seed':seed,'arm':arm,'id':it['id'],'tile':it['tile'],'cluster':it['cluster'],'phen':it['phen'],'kind':it['kind'],'source_gold':it['answer'],'transformed_gold':None,'parsed':parsed.group(1) if parsed else None,'answer_raw':raw,'pair_key':it['pair_key']}
  def arm_run(seed,arm):
   selected=list(items);dest=out/f'answers_seed{seed}_{arm}.jsonl';generated=[]
   with dest.open('x') as f,torch.inference_mode():
    for j,it in enumerate(selected):
     r=generate(seed,arm,it);generated.append(r);rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
     if j%40==0:emit_status('running',seed=seed,arm=arm,completed=j,total=len(selected));print(f'seed{seed} {arm} {j}/{len(selected)} {time.perf_counter()-start:.0f}s',flush=True)
   return generated
  # Verify all original arms before opening any transformed output.
  for seed in [1,2,3]:
   real=arm_run(seed,'real');validity[str(seed)]={}
   for phen in PHEN:
    subset=[r for r in real if r['phen']==phen]
    pf=sum(r['parsed'] is None for r in subset)/len(subset)
    if pf>cfg['validity']['max_parse_fail']:raise ValueError('Real parse gate failed')
    validity[str(seed)][phen]={}
    for refname,reference in references.items():
     rep=sum(r['parsed'] in ('yes','no') and r['parsed']==reference[str(seed)][r['id']] for r in subset)/len(subset)
     validity[str(seed)][phen][refname]={'n':len(subset),'reproduction_rate':rep,'parse_failure_rate':pf}
     if rep<cfg['validity']['real_reproduction_min']:raise ValueError(f'Reproduction failed {refname} seed{seed}/{phen}: {rep}')
  write(out/'reproduction.json',validity);print('REPRODUCTION PASS '+json.dumps(validity),flush=True)
  for seed in [1,2,3]:
   for arm in ARMS[1:]:
    rr=arm_run(seed,arm)
    for phen in PHEN:
     sub=[r for r in rr if r['phen']==phen]
     if sub and sum(r['parsed'] is None for r in sub)/len(sub)>cfg['validity']['max_parse_fail']:raise ValueError('Parse validity failed')
  scores=score_run(rows,items,references);scores['runtime_reproduction']=validity;scores['manifest_sha256']=sha(out/'manifest.json');scores['prereg_sha256']=m['prereg_sha256'];scores['prompt_snapshot_sha256']=sha(out/'prompt_snapshot.json');scores['elapsed_s']=time.perf_counter()-start
  write(out/'scores.json',scores)
  if not scores['valid']:raise ValueError('Scoring validity failed: '+str(scores.get('invalid_reason')))
  emit_status('completed',verdict=scores['verdict']);print('E4 DONE '+scores['verdict'],flush=True)
 except BaseException as exc:
  write(out/'failure.json',{'at':now(),'reason':repr(exc),'traceback':traceback.format_exc(),'rows_completed':len(rows),'reproduction_so_far':validity});emit_status('invalid',reason=repr(exc));raise


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run']);p.add_argument('--out',default='e4_delta_probe_v0');p.add_argument('--config',type=Path,default=Path(__file__).with_name('e4_delta_probe_prereg_v0.json'));a=p.parse_args()
 require(a.out=='e4_delta_probe_v0','Only dedicated E4 output permitted')
 if a.command=='prepare':
  dest=ROOT/a.out;dest.mkdir(parents=True,exist_ok=False)
  try:prepare(a)
  except BaseException as exc:
   write(dest/'failure.json',{'phase':'prepare','at':now(),'reason':repr(exc),'traceback':traceback.format_exc()});write(dest/'status.json',{'status':'invalid_preparation','at':now(),'reason':repr(exc)});raise
 else:
  try:run(a)
  except BaseException as exc:
   dest=ROOT/a.out
   if dest.exists() and (dest/'status.json').exists() and read(dest/'status.json').get('status') in ('prepared','running') and not (dest/'failure.json').exists() and not (dest/'scores.json').exists():
    write(dest/'failure.json',{'phase':'run_preflight_or_run','at':now(),'reason':repr(exc),'traceback':traceback.format_exc()})
    write(dest/'status.json',{'status':'invalid','at':now(),'reason':repr(exc)})
   raise
