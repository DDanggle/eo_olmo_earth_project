#!/usr/bin/env python3
"""E3-PD: pre-registered, inference-only pair information diagnosis.
prepare is CPU only. run consumes a frozen prepare manifest and source snapshot.
"""
import argparse, collections, hashlib, json, os, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from e3_pair_transforms_v0 import transform_pair

ROOT=Path('/home/work/data/olmoearth')
ARMS=['real','earlier_only','later_only','repeat_earlier','repeat_later','no_delta','reverse']
HARD_ARMS=['real','later_only','repeat_later']
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
def rank(it):return hashlib.sha256(f"e3-pd-v0-20260925|{it['phen']}|{it['cluster']}|{it['tile']}".encode()).hexdigest()
def kept_dates(rec,tile):
 r=rec[tile];q=r['scl_clear_fraction'];k=sorted(sorted(range(15),key=lambda i:(-float(q[i]),i))[:12]);return [str(r['times'][i])[:10] for i in k]
def check_id_rows(rows):
 ids=[x['id'] for x in rows]
 if len(ids)!=len(set(ids)):raise ValueError('Duplicate item IDs')
 return set(ids)
def prepare(a):
 import torch
 import torch.nn.functional as F
 torch.set_num_threads(2)
 cfg=read(a.config);out=ROOT/a.out
 paths={'landslide':ROOT/'sentinel_qa_v0_1/items.jsonl','flood':ROOT/'flood_qa_v0/items.jsonl','contract':ROOT/'sen12_gp_contract/sample_contract.jsonl'}
 rec={x['sample_id']:x for x in lines(paths['contract'])}
 reference={str(seed):{r['id']:r for r in lines(ROOT/f'e2_multi_reader_v0/reader_seed{seed}/answers_real_all.jsonl')} for seed in [1,2,3]}
 for seed in ['2','3']:
  if set(reference[seed])!=set(reference['1']):raise ValueError('E2 seed IDs differ')
 items=[]
 for phen in PHEN:
  for it in lines(paths[phen]):
   if it.get('type')!='Q1' or (phen=='flood' and it['fold']!='test'):continue
   if it['id'] not in reference['1']:continue
   it={**it,'phen':phen,'kind':it.get('kind','pos' if it['answer']=='yes' else 'neg')}
   it['cluster']=str(it['event']) if phen=='flood' else it['fold']
   it['cache_path']=str(ROOT/('kurosiwo_s1_cache/single_fp16' if phen=='flood' else 'olmo_streaming_dev/single_fp16')/(it['tile']+'.npy'))
   it['indices']=[{'pre_1':0,'pre_2':1,'post':2}[x] for x in it['slots']] if phen=='flood' else [kept_dates(rec,it['tile']).index(d) for d in it['dates']]
   for seed in ['1','2','3']:
    saved=reference[seed][it['id']]
    if any(saved[k]!=it[k] for k in ['tile','phen','kind','fold']) or saved['text_gold']!=it['answer']:raise ValueError('E2 item contract mismatch')
   items.append(it)
 if check_id_rows(items)!=set(reference['1']):raise ValueError('E2 population coverage differs')
 bytile=collections.defaultdict(list)
 for it in items:
  if it['kind'] in ('pos','neg'):bytile[(it['phen'],it['tile'])].append(it)
 grouped=collections.defaultdict(list)
 for key,pair in bytile.items():
  if len(pair)!=2 or {i['answer'] for i in pair}!={'yes','no'}:raise ValueError('Incomplete pair')
  pair.sort(key=lambda i:i['kind']=='neg');grouped[(key[0],pair[0]['cluster'])].append(pair)
 selected=[]
 for (phen,cluster),pairs in sorted(grouped.items()):
  selected.extend(i for pair in sorted(pairs,key=lambda q:rank(q[0]))[:8 if phen=='flood' else 16] for i in pair)
 hard=collections.defaultdict(list)
 for it in items:
  if it['phen']=='flood' and it['kind']=='hard_neg':hard[it['cluster']].append(it)
 for cluster,its in sorted(hard.items()):selected.extend(sorted(its,key=rank)[:8])
 selected.sort(key=lambda i:(i['phen'],i['cluster'],i['tile'],i['kind']))
 # Audit the actually used E2 dates, not unrelated cached frames. No exclusion.
 usage=collections.defaultdict(set)
 for it in items:usage[it['cache_path']].update(it['indices'])
 audit=[]
 for j,(path,idx) in enumerate(sorted(usage.items())):
  s=np.load(path,mmap_mode='r');x=np.asarray(s[sorted(idx)])
  audit.append({'path':path,'indices':sorted(idx),'shape':list(s.shape),'nonfinite_used_values':int((~np.isfinite(x)).sum())})
  if j%200==0:print(f'CPU cache audit {j}/{len(usage)}',flush=True)
 pairs={};selected_sources={};invalid_selected=[]
 for j,it in enumerate(selected):
  path=it['cache_path'];s=np.load(path,mmap_mode='r');expected=(3,768,48,48) if it['phen']=='flood' else (12,768,32,32)
  if tuple(s.shape)!=expected:raise ValueError(f'Unexpected cache shape {path}: {s.shape}')
  t=torch.from_numpy(np.asarray(s[it['indices']],dtype='float32'));pool=6 if it['phen']=='flood' else 4
  pair=F.avg_pool2d(t,pool).flatten(2).permute(0,2,1).contiguous().numpy()
  if not np.isfinite(pair).all():invalid_selected.append(it['id'])
  pairs[f'pair_{j}']=pair;it['pair_key']=f'pair_{j}';it['allowed_arms']=HARD_ARMS if it['kind']=='hard_neg' else ARMS
  if path not in selected_sources:selected_sources[path]=sha(path)
 llm_dir=ROOT/'olmo_llm/Olmo-3-7B-Instruct'
 llm_files=sorted(p for p in llm_dir.iterdir() if p.is_file() and p.suffix in ('.json','.jinja','.safetensors','.txt','.model'))
 print('CPU hash frozen LLM and tokenizer files',flush=True)
 llm_hashes={str(p):sha(p) for p in llm_files}
 models={}
 for seed in ['1','2','3']:
  path=ROOT/f'e2_multi_reader_v0/reader_seed{seed}/projector.pt';actual=sha(path)
  if actual!=cfg['checkpoints'][seed]:raise ValueError('Checkpoint changed')
  models[seed]={'path':str(path),'sha256':actual}
 write(out/'input_audit.json',{'n_unique_caches':len(audit),'nonfinite_caches':[r for r in audit if r['nonfinite_used_values']],'selected_nonfinite_ids':invalid_selected,'all':audit})
 if invalid_selected:
  write(out/'failure.json',{'phase':'prepare','reason':'nonfinite selected source','ids':invalid_selected,'at':now()});raise ValueError('Nonfinite selected data: no silent reselection')
 np.savez_compressed(out/'pairs.npz',**pairs)
 (out/'items.jsonl').write_text(''.join(json.dumps(it)+'\n' for it in selected))
 write(out/'saved_e2_real.json',{seed:{it['id']:reference[seed][it['id']]['parsed'] for it in selected} for seed in reference})
 shutil=__import__('shutil');snap=out/'code_snapshot';snap.mkdir()
 sources=['e3_pair_dependence_v0.py','e3_pair_transforms_v0.py','e3_pair_scoring.py']
 for fn in sources:shutil.copyfile(Path(__file__).parent/fn,snap/fn)
 shutil.copyfile(a.config,out/'prereg.json')
 manifest={'schema':'e3-pd-prepared-v0','prepared_at':now(),'scope':cfg['scope'],'selection_uses_predictions':False,'items_sha256':sha(out/'items.jsonl'),'pairs_sha256':sha(out/'pairs.npz'),'saved_e2_real_sha256':sha(out/'saved_e2_real.json'),'prereg_sha256':sha(out/'prereg.json'),'source_input_sha256':{str(p):sha(p) for p in paths.values()},'selected_cache_sha256':selected_sources,'models':models,'llm_files_sha256':llm_hashes,'e2_reference_sha256':{str(seed):sha(ROOT/f'e2_multi_reader_v0/reader_seed{seed}/answers_real_all.jsonl') for seed in [1,2,3]},'code_snapshot_sha256':{fn:sha(snap/fn) for fn in sources},'selected_counts':dict(collections.Counter(f"{i['phen']}|{i['cluster']}|{i['kind']}" for i in selected)),'n_items':len(selected),'n_generations':3*sum(len(i['allowed_arms']) for i in selected),'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__},'full_E2_nonfinite_caches':sum(bool(r['nonfinite_used_values']) for r in audit)}
 write(out/'manifest.json',manifest);write(out/'status.json',{'status':'prepared','at':now()});print(json.dumps({k:manifest[k] for k in ['n_items','n_generations','selected_counts','full_E2_nonfinite_caches']},indent=2),flush=True)

def run(a):
 import re
 import torch
 import torch.nn as nn
 from transformers import AutoTokenizer,AutoModelForCausalLM
 from e3_pair_scoring import score_run
 torch.set_num_threads(2);out=ROOT/a.out;m=read(out/'manifest.json');cfg=read(out/'prereg.json')
 for fn,key in [('items.jsonl','items_sha256'),('pairs.npz','pairs_sha256'),('saved_e2_real.json','saved_e2_real_sha256'),('prereg.json','prereg_sha256')]:
  if sha(out/fn)!=m[key]:raise ValueError('Frozen prepared input changed: '+fn)
 for fn,h in m['code_snapshot_sha256'].items():
  if sha(Path(__file__).parent/fn)!=h:raise ValueError('Must run unchanged frozen source snapshot')
 if read(out/'status.json')['status']!='prepared':raise ValueError('Run already started/completed; use new version for repair')
 for v in m['models'].values():
  if sha(v['path'])!=v['sha256']:raise ValueError('Frozen checkpoint changed')
 for path,hsh in m['llm_files_sha256'].items():
  if sha(path)!=hsh:raise ValueError('Frozen LLM/tokenizer file changed: '+path)
 # The launcher also checks; keep a final guard immediately before allocation.
 uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 if uuid in busy:raise RuntimeError('GPU1 occupied; do not allocate')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise RuntimeError('Explicit GPU1 mapping required')
 items=lines(out/'items.jsonl');data=np.load(out/'pairs.npz');saved=read(out/'saved_e2_real.json');start=time.perf_counter();rows=[];validity={}
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
  prompts={}
  for it in items:
   sensor,word=PHEN[it['phen']];user=f"These are 2 {sensor} observations of the same area in chronological order, taken on {', '.join(it['dates'])}: <EO> Did a {word} occur between the two observations? Answer with yes or no."
   text=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True);pre,post=text.split('<EO>')
   prompts[it['id']]=tuple(tok(v,add_special_tokens=False,return_tensors='pt').input_ids[0].to(dev) for v in [pre,post])
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
   selected=[i for i in items if arm in i['allowed_arms']];dest=out/f'answers_seed{seed}_{arm}.jsonl';generated=[]
   with dest.open('x') as f,torch.inference_mode():
    for j,it in enumerate(selected):
     r=generate(seed,arm,it);generated.append(r);rows.append(r);f.write(json.dumps(r)+'\n');f.flush()
     if j%40==0:emit_status('running',seed=seed,arm=arm,completed=j,total=len(selected));print(f'seed{seed} {arm} {j}/{len(selected)} {time.perf_counter()-start:.0f}s',flush=True)
   return generated
  # Verify all original arms before opening any transformed output.
  for seed in [1,2,3]:
   real=arm_run(seed,'real');validity[str(seed)]={}
   for phen in PHEN:
    subset=[r for r in real if r['phen']==phen];rep=sum(r['parsed']==saved[str(seed)][r['id']] for r in subset)/len(subset);pf=sum(r['parsed'] is None for r in subset)/len(subset)
    validity[str(seed)][phen]={'n':len(subset),'reproduction_rate':rep,'parse_failure_rate':pf}
    if rep<cfg['validity']['real_reproduction_min'] or pf>cfg['validity']['max_parse_fail']:raise ValueError(f'Reproduction failed seed{seed}/{phen}: {rep}, {pf}')
  write(out/'reproduction.json',validity);print('REPRODUCTION PASS '+json.dumps(validity),flush=True)
  for seed in [1,2,3]:
   for arm in ARMS[1:]:
    rr=arm_run(seed,arm)
    for phen in PHEN:
     sub=[r for r in rr if r['phen']==phen]
     if sub and sum(r['parsed'] is None for r in sub)/len(sub)>cfg['validity']['max_parse_fail']:raise ValueError('Parse validity failed')
  scores=score_run(rows,items,seeds=(1,2,3));scores['reproduction']=validity;scores['manifest_sha256']=sha(out/'manifest.json');scores['prereg_sha256']=m['prereg_sha256'];scores['elapsed_s']=time.perf_counter()-start
  write(out/'scores.json',scores)
  if not scores['valid']:raise ValueError('Scoring validity failed: '+str(scores.get('invalid_reasons')))
  emit_status('completed',verdict=scores['verdict']);print('E3 DONE '+scores['verdict'],flush=True)
 except BaseException as exc:
  write(out/'failure.json',{'at':now(),'reason':repr(exc),'traceback':traceback.format_exc(),'rows_completed':len(rows),'reproduction_so_far':validity});emit_status('invalid',reason=repr(exc));raise

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run']);p.add_argument('--out',default='e3_pair_dependence_v0');p.add_argument('--config',type=Path,default=ROOT/'config/e3_pair_dependence_prereg_v0.json');a=p.parse_args()
 if a.command=='prepare':
  dest=ROOT/a.out
  dest.mkdir(parents=True,exist_ok=False)
  try:prepare(a)
  except BaseException as exc:
   dest.mkdir(parents=True,exist_ok=True)
   if not (dest/'failure.json').exists():write(dest/'failure.json',{'phase':'prepare','at':now(),'reason':repr(exc),'traceback':traceback.format_exc()})
   write(dest/'status.json',{'status':'invalid_preparation','at':now(),'reason':repr(exc)})
   raise
 else:run(a)
