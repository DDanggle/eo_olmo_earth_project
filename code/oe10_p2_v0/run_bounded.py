"""Bounded sequential workers on verified idle physical GPU1 only."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from guard_reference import idle_gpu_snapshot,protected_records

SNAP=Path(__file__).resolve().parent
ROOT=Path('/home/work/data/olmoearth/oe10_p2_v0')
PYTHON='/home/work/data/olmoearth/.venv-master/bin/python'
GPU='GPU-8b485982-e8bb-004a-a10a-e37637e5e2bb'

def hashes():
 m=json.loads((SNAP/'source_manifest.json').read_text())['files']
 for name,h in m.items():
  if hashlib.sha256((SNAP/name).read_bytes()).hexdigest()!=h:raise RuntimeError('source changed '+name)
 return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['engineering','training'],required=True)
 ap.add_argument('--tag',required=True);ap.add_argument('--protocol',type=Path)
 a=ap.parse_args()
 if not a.tag.replace('_','').isalnum():raise RuntimeError('invalid tag')
 if not SNAP.is_relative_to(ROOT/'code_snapshot'):raise RuntimeError('run frozen snapshot')
 ROOT.mkdir(exist_ok=True);lock=(ROOT/'gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 run=ROOT/a.tag;run.mkdir(exist_ok=False)
 state={'status':'initializing','controller_pid':os.getpid(),'gpu_uuid':GPU,'stage':a.stage,'workers':[],
        'started_utc':datetime.now(timezone.utc).isoformat()}
 def save():
  tmp=run/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(run/'status.json')
 child=None;save();started=time.monotonic()
 def cleanup():
  if child is not None and child.poll() is None:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 try:
  state['protected_before']=protected_records();state['source_hashes']=hashes()
  env=os.environ.copy();env.pop('PYTHONPATH',None)
  env.update(CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
    TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  jobs=[]
  if a.stage=='engineering':
   for arm in ['B0','B2']:
    for mode in ['reference','resume']:jobs.append((arm,270927,mode,1800))
   total_limit=7200
  else:
   if a.protocol is None:raise RuntimeError('protocol required')
   protocol=json.loads(a.protocol.read_text())
   state['protocol_sha256']=hashlib.sha256(a.protocol.read_bytes()).hexdigest()
   if not protocol['locked_before_development_results']:raise RuntimeError('unlocked protocol')
   engineering=Path(protocol['engineering_results_root'])
   engineering_status=json.loads((engineering/'status.json').read_text())
   if engineering_status['status']!='completed' or not engineering_status['protected_and_snapshot_unchanged']:
    raise RuntimeError('engineering integrity gate failed')
   for arm in ['B0','B2']:
    r=json.loads((engineering/f'{arm}_270927_resume'/'receipt.json').read_text())
    if r['status']!='cold_resume_passed' or not r['pass_result']:raise RuntimeError('engineering gate failed')
    if r['source_hashes']!=state['source_hashes']:raise RuntimeError('cold resume tested a different source')
   for seed in protocol['paired_seed_ids']:
    for arm in ['B0','B2']:jobs.append((arm,seed,'train',protocol['max_worker_seconds']+120))
   total_limit=protocol['max_sequence_seconds']
  for arm,seed,mode,limit in jobs:
   if time.monotonic()-started>=total_limit:raise RuntimeError('sequence resource cap')
   if hashes()!=state['source_hashes'] or protected_records()!=state['protected_before']:raise RuntimeError('integrity changed')
   idle,obs=idle_gpu_snapshot()
   if not any(r['index']==1 and r['uuid']==GPU for r in idle):raise RuntimeError('GPU1 not idle; no launch')
   name=f'{arm}_{seed}_{mode}';out=run/name
   args=[PYTHON,str(SNAP/'p2_worker.py'),'--mode',mode,'--arm',arm,'--seed',str(seed),'--out',str(out)]
   if mode=='resume':args+=['--reference-dir',str(run/f'{arm}_{seed}_reference')]
   if a.protocol:args+=['--protocol',str(a.protocol)]
   row={'name':name,'status':'running','output':str(out),'gpu_observations':obs,'timeout_seconds':limit}
   state['workers'].append(row);state['status']='running';state['current_worker']=name
   with (run/f'{name}.log').open('x') as log:
    child=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row['pid']=child.pid;save()
    try:code=child.wait(timeout=min(limit,total_limit-(time.monotonic()-started)))
    except subprocess.TimeoutExpired:
     cleanup();row['status']='own_worker_timeout';save();raise RuntimeError('worker timeout')
   row.update(status='completed' if code==0 else 'failed',exit_code=code)
   if (out/'receipt.json').exists():row['receipt_status']=json.loads((out/'receipt.json').read_text())['status']
   save()
   if code!=0:raise RuntimeError('worker failed '+name)
  state['status']='completed'
 except Exception as e:
  state.update(status='failed',error=repr(e),traceback=traceback.format_exc())
 finally:
  cleanup()
  try:
   state['protected_after']=protected_records()
   state['protected_and_snapshot_unchanged']=state.get('protected_before')==state['protected_after'] and state.get('source_hashes')==hashes()
  except Exception as exc:
   state['protected_and_snapshot_unchanged']=False;state['integrity_error']=repr(exc)
  if not state['protected_and_snapshot_unchanged']:state['status']='failed'
  state['elapsed_seconds']=time.monotonic()-started;state['finished_utc']=datetime.now(timezone.utc).isoformat();save()
 if state['status']!='completed':return 1
 return 0

if __name__=='__main__':raise SystemExit(main())
