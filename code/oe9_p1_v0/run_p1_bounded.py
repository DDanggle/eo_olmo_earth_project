#!/usr/bin/env python3
"""One shot, only physical GPU1, frozen sources, bounded own child process."""
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

ROOT=Path('/home/work/data/olmoearth/oe9_review_p1_v0')
SNAP=ROOT/'code_snapshot/p1_v0'
PYTHON='/home/work/data/olmoearth/.venv-master/bin/python'
EXPECTED_GPU='GPU-8b485982-e8bb-004a-a10a-e37637e5e2bb'

def hashes():
 m=json.loads((SNAP/'source_manifest.json').read_text())['files']
 for name,digest in m.items():
  if hashlib.sha256((SNAP/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Snapshot changed '+name)
 return m

def main():
 if Path(__file__).resolve()!=SNAP/'run_p1_bounded.py':raise RuntimeError('Run snapshot')
 lock=(ROOT/'gpu1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=ROOT/'p1_status_v0.json'
 with status.open('x') as f:json.dump({'status':'initializing'},f)
 state={'status':'initializing','controller_pid':os.getpid(),'gpu_uuid':EXPECTED_GPU,'max_worker_seconds':7200}
 def save():
  temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(status)
 worker=None
 try:
  before=protected_records();sources=hashes();state.update(protected_before=before,source_hashes=sources)
  idle,observations=idle_gpu_snapshot();state['gpu_observations']=observations
  if not any(r['index']==1 and r['uuid']==EXPECTED_GPU for r in idle):
   state['status']='not_launched_gpu1_not_idle';save();return
  env=os.environ.copy();env.pop('PYTHONPATH',None)
  env.update(CUDA_VISIBLE_DEVICES=EXPECTED_GPU,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  start=time.monotonic()
  with (ROOT/'p1_worker_v0.log').open('x') as log:
   worker=subprocess.Popen([PYTHON,str(SNAP/'p1_worker.py'),'--out',str(ROOT/'p1_run_v0')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   state.update(status='running',worker_pid=worker.pid);save()
   try:code=worker.wait(timeout=7200)
   except subprocess.TimeoutExpired:
    os.killpg(worker.pid,signal.SIGTERM)
    try:worker.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
    state.update(status='own_worker_time_limit',elapsed_seconds=time.monotonic()-start);save();return
  after=protected_records();unchanged=after==before and hashes()==sources
  state.update(status='worker_completed' if code==0 and unchanged else 'worker_failed',exit_code=code,
    protected_after=after,protected_and_snapshot_unchanged=unchanged,elapsed_seconds=time.monotonic()-start)
  receipt=ROOT/'p1_run_v0/receipt.json'
  if receipt.is_file():
   report=json.loads(receipt.read_text());state['worker_result_status']=report['status'];state['p1_full_gate_pass']=report.get('p1_full_gate_pass',False)
   if code==0 and unchanged and not state['p1_full_gate_pass']:state['status']='completed_p1_gate_not_met'
  save()
 except Exception as e:
  state.update(status='controller_failed',error=repr(e),traceback=traceback.format_exc());save();raise
 finally:
  if worker is not None and worker.poll() is None:
   os.killpg(worker.pid,signal.SIGTERM)
   try:worker.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
  try:
   state['protected_after']=protected_records()
   state['protected_and_snapshot_unchanged']=(state.get('protected_before')==state['protected_after'] and state.get('source_hashes')==hashes())
  except Exception as e:
   state['integrity_check_error']=repr(e)
   state['protected_and_snapshot_unchanged']=False
  save()

if __name__=='__main__':main()
