#!/usr/bin/env python3
"""Launch only the frozen E4 snapshot with cooperative inherited locks."""
from contextlib import ExitStack
import fcntl, json, os, subprocess, time
from pathlib import Path
from e4_delta_runner_v0 import read, write, sha, now, require
ROOT=Path('/home/work/data/olmoearth');OUT=ROOT/'e4_delta_probe_v0';PY=ROOT/'.venv-master/bin/python'
def main():
 with ExitStack() as stack:
  handles=[]
  for name in ['.eo_e3_pair_dependence.lock','.eo_reader_gpu0.lock','.eo_reader_gpu1.lock','.eo_e4_delta_probe.lock']:
   handle=stack.enter_context((ROOT/name).open('a+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(handle)
  m=read(OUT/'manifest.json')
  require(sha(__file__)==m['code_snapshot_sha256']['run_e4_when_idle_v0.py'],'Launcher not frozen')
  require(read(OUT/'status.json')['status']=='prepared','Already run or unprepared')
  require(not list(OUT.glob('answers_*.jsonl')) and not any((OUT/n).exists() for n in ['failure.json','reproduction.json','scores.json','runtime_environment.json','run.log']),'Output already exists')
  started=time.monotonic()
  def status(s,**kw):
   d={'status':s,'at':now(),'pid':os.getpid(),'elapsed_wait_s':time.monotonic()-started,**kw};write(OUT/'queue.json',d);print(json.dumps(d),flush=True)
  uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();require(bool(uuid) and '\n' not in uuid,'Bad GPU UUID')
  while True:
   busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
   if uuid not in [line.strip() for line in busy]:break
   status('waiting_gpu0',gpu_uuid=uuid)
   if time.monotonic()-started>6*3600:status('wait_expired_without_launch');return 3
   time.sleep(45)
  status('launching_frozen_snapshot',gpu_uuid=uuid,gpu_index=0,manifest_sha256=sha(OUT/'manifest.json'))
  env=dict(os.environ);env.pop('PYTHONPATH',None);env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
  with (OUT/'run.log').open('x') as log:
   rc=subprocess.call([str(PY),'-B',str(OUT/'code_snapshot/e4_delta_runner_v0.py'),'run'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(h.fileno() for h in handles))
  status('completed' if rc==0 else 'failed',returncode=rc);return rc
if __name__=='__main__':raise SystemExit(main())
