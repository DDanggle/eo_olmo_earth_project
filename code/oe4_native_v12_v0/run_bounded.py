#!/usr/bin/env python3
"""One bounded real-data preparation + idle-GPU experiment sequence, no daemon."""
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/work/data/olmoearth/oe4_native_v12_v0')
PYTHON=ROOT.parent/'.venv-master/bin/python'
RUNTIME=ROOT/'code_snapshot/runtime_v0/runtime_smoke.py'
EXPECTED='5a038e059364bac95375861b1b3369c4e6c8f261a10047f13803bd8f49bf3eee'
PROTECTED={
 'pilot_sen12_gp_heads.py':('8f805359c2989cd474e89d9086545698c0acdbb9673b27a82cc1ad00743a1af1',1788238578),
 'sen12_official_baselines.py':('19232b2ae122dd6529bac56977e075ce7b94c0daf90f24c21557b4d5cc213f5b',1787683824),
 'extract_sen12_fold_cache.py':('197895a05f2f6dddff12f6385c01aacf4f72bdbb877d273b473c83dc5d3a5e0a',1787659265),
 'audit_sen12_fold_cache.py':('4287dbcdef64c9ea9ea2a707e961ce6ba606b85405ee794fa91f44ded2419ec9',1787668636),
}
RUNS=[{'name':'native_smoke_v0','train':8,'dev':4,'steps':8},
      {'name':'native_development_v0','train':64,'dev':8,'steps':32}]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def timestamp():return datetime.now(timezone.utc).isoformat()
def protected():
 values={}
 for name,(expected,mtime) in PROTECTED.items():
  p=ROOT.parent/'code'/name;values[name]={'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns}
  if values[name]['sha256']!=expected or int(p.stat().st_mtime)!=mtime:raise RuntimeError(f'Protected source changed: {name}')
 return values

def idle():
 cmd=['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']
 gpu=list(csv.reader(subprocess.check_output(cmd,text=True,timeout=15).splitlines()))
 busy={x.split(',')[0].strip() for x in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15).splitlines() if ',' in x}
 free=[]
 for idx,uuid,mem,util in gpu:
  if uuid.strip() not in busy and int(mem.strip())<512 and int(util.strip())<5:free.append(int(idx))
 return [i for i in [1,0] if i in free]

def main():
 lock=(ROOT/'bounded_sequence.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=ROOT/'bounded_sequence_status.json'
 if status.exists():raise FileExistsError(status)
 state={'started_utc':timestamp(),'status':'waiting_for_assets','run_plan':RUNS,'runtime_sha256':EXPECTED,'maximum_gpu_wait_seconds':5400,'maximum_worker_seconds':1100,'results':[],'controller_pid':os.getpid(),'scope':'engineering_development_not_novelty_or_confirmatory_result','protected_before':protected()}
 def write():
  tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status)
  print(json.dumps({'utc':timestamp(),'status':state['status'],'detail':state.get('detail')}),flush=True)
 try:
  if sha(RUNTIME)!=EXPECTED:raise RuntimeError('Runtime source hash mismatch')
  write();start=time.monotonic()
  while not (ROOT/'asset_receipt.json').exists():
   if time.monotonic()-start>2400:raise TimeoutError('Asset acquisition did not finish within 40 minutes')
   time.sleep(10)
  state['asset_receipt_sha256']=sha(ROOT/'asset_receipt.json')
  base=[str(PYTHON),'-u',str(RUNTIME),'--source-root',str(ROOT/'source'),'--deps-root',str(ROOT/'deps'),'--checkpoint-dir',str(ROOT/'models/OlmoEarth-v1_2-Base'),'--h5-root',str(ROOT/'data/official_subset_1k')]
  env=dict(os.environ);env.pop('PYTHONPATH',None);env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
  prep=ROOT/'runs/data_prepare_v1'
  state['status']='preparing_real_data';write()
  with (ROOT/'prepare.log').open('x') as log:
   subprocess.run(base+['--out-dir',str(prep),'--train-files','64','--dev-files','8','--steps','32','--prepare-only'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
  for run in RUNS:
   state['status']='waiting_for_idle_gpu';state['detail']=run['name'];write()
   wait_start=time.monotonic();stable=None;streak=0
   while True:
    choices=idle();current=choices[0] if choices else None
    streak=streak+1 if current is not None and current==stable else (1 if current is not None else 0)
    stable=current
    if streak>=3:break
    if time.monotonic()-wait_start>5400:raise TimeoutError('No idle GPU within 90 minutes; no jobs interrupted')
    time.sleep(10)
   # Recheck immediately before launching; never share a GPU with an active job.
   if stable not in idle():raise RuntimeError('GPU became busy before launch')
   protected()
   env['CUDA_VISIBLE_DEVICES']=str(stable)
   out=ROOT/'runs'/run['name']
   argv=base+['--out-dir',str(out),'--device','cuda:0','--train-files',str(run['train']),'--dev-files',str(run['dev']),'--steps',str(run['steps']),'--save-weights','--max-wall-seconds','900']
   item={'name':run['name'],'gpu_physical_index':stable,'started_utc':timestamp(),'argv':argv}
   state['results'].append(item);state['status']='running';write()
   with (ROOT/(run['name']+'.log')).open('x') as log:
    result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1100)
   item['returncode']=result.returncode;item['finished_utc']=timestamp()
   if result.returncode!=0:raise RuntimeError(f'Worker failed: {run["name"]}')
   receipt=json.loads((out/'receipt.json').read_text())
   if receipt['status']!='PASS_bounded_encoder_update':raise RuntimeError('Worker did not establish pass')
   item['receipt_sha256']=sha(out/'receipt.json')
   state['protected_after']=protected();write()
  state['status']='completed';state['detail']='Two bounded engineering runs passed; inspect receipts for scientific limits.';state['completed_utc']=timestamp();write()
 except BaseException as e:
  state['status']='failed';state['detail']=repr(e);state['finished_utc']=timestamp();write();raise

if __name__=='__main__':main()
