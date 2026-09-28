#!/usr/bin/env python3
"""Run one frozen E3 job after GPU1 is idle. No interaction with other jobs."""
import fcntl,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/home/work/data/olmoearth');OUT=ROOT/'e3_pair_dependence_v0';PY=ROOT/'.venv-master/bin/python'
lock=(ROOT/'.eo_reader_gpu1.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Another cooperative EO reader job holds the lock')
queue=OUT/'queue.json';started=time.monotonic()
def status(s,**kw):
 d={'status':s,'at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'elapsed_wait_s':time.monotonic()-started,**kw};queue.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
if json.loads((OUT/'status.json').read_text())['status']!='prepared':raise SystemExit('E3 preparation must succeed before enqueue')
uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
while True:
 try:busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
 except subprocess.CalledProcessError as exc:status('gpu_query_failed',error=str(exc));raise
 if uuid not in busy:break
 status('waiting_gpu1')
 if time.monotonic()-started>6*3600:status('wait_expired_without_launch');raise SystemExit(3)
 time.sleep(45)
status('launching_frozen_snapshot');env=dict(os.environ);env.pop('PYTHONPATH',None);env.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
with (OUT/'run.log').open('a') as log:
 rc=subprocess.call([str(PY),'-B',str(OUT/'code_snapshot/e3_pair_dependence_v0.py'),'run','--out','e3_pair_dependence_v0'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
status('completed' if rc==0 else 'failed',returncode=rc)
raise SystemExit(rc)
