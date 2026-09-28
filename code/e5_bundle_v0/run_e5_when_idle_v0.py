#!/usr/bin/env python3
"""One frozen E5 chain on an idle GPU with inherited cooperative leases."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path

ROOT=Path('/home/work/data/olmoearth')
LOCKS=['.eo_e3_pair_dependence.lock','.eo_reader_gpu0.lock','.eo_reader_gpu1.lock','.eo_e5_equal_budget.lock']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(c,s):
 if not c:raise ValueError(s)
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main(argv=None):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'e5_equal_budget_v0');a=ap.parse_args(argv)
 out=a.out.resolve();require(out.parent==ROOT and out.name=='e5_equal_budget_v0','Dedicated E5 output required')
 with ExitStack() as stack:
  handles={}
  for name in LOCKS:
   handle=stack.enter_context((ROOT/name).open('a+'));fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles[name]=handle
  manifest=read(out/'manifest.json');cfg=read(out/'prereg.json');initial_manifest_sha=sha(out/'manifest.json')
  require(Path(__file__).resolve().parent==out/'code_snapshot','Only frozen launcher permitted')
  require(manifest['code_snapshot_sha256'][Path(__file__).name]==sha(__file__),'Frozen launcher changed')
  require(sha(out/'prereg.json')==manifest['files_sha256']['prereg.json'],'Frozen plan changed')
  require(read(out/'status.json')['status']=='prepared','E5 already started or invalid')
  require(not any((out/n).exists() for n in ['run.log','scores.json','failure.json','runtime_environment.json','training_summary.json','prompts.jsonl','run_claim.json','predictions.jsonl','initial_states','models']),'Existing E5 execution artifacts')
  require(cfg['compute']['gpu_index']==0 and cfg['compute']['lock_names']==LOCKS,'GPU/lease contract changed')
  started=time.monotonic();queue=out/'queue.json'
  def status(s,**kw):
   value={'status':s,'at':now(),'pid':os.getpid(),'elapsed_wait_s':time.monotonic()-started,**kw};write(queue,value);print(json.dumps(value),flush=True)
  uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  require(bool(uuid) and '\n' not in uuid,'Expected one GPU0 UUID')
  while True:
   busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
   if uuid not in [line.strip() for line in busy]:break
   status('waiting_gpu0',gpu_uuid=uuid)
   if time.monotonic()-started>=cfg['compute']['queue_wait_hours']*3600:
    status('wait_expired_without_launch');return 3
   time.sleep(45)
  require(read(out/'status.json')['status']=='prepared','Prepared status changed before launch')
  require(sha(out/'manifest.json')==initial_manifest_sha and sha(out/'prereg.json')==manifest['files_sha256']['prereg.json'],'Prepared freeze changed while waiting')
  for name,h in manifest['code_snapshot_sha256'].items():require(sha(out/'code_snapshot'/name)==h,'Source snapshot changed: '+name)
  manifest_sha=sha(out/'manifest.json')
  env=dict(os.environ);env.pop('PYTHONPATH',None)
  env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',
    E5_LOCK_FDS=json.dumps({n:h.fileno() for n,h in handles.items()}),E5_GPU_INDEX='0',E5_GPU_UUID=uuid,E5_MANIFEST_SHA256=manifest_sha)
  status('launching_frozen_snapshot',gpu_uuid=uuid,gpu_index=0,manifest_sha256=manifest_sha)
  with (out/'run.log').open('x') as log:
   rc=subprocess.call([str(ROOT/'.venv-master/bin/python'),'-B',str(out/'code_snapshot/e5_train_v0.py'),'--out',str(out)],
    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(h.fileno() for h in handles.values()))
  good=rc==0 and read(out/'status.json').get('status')=='completed' and (out/'scores.json').exists() and read(out/'scores.json').get('valid') is True
  status('completed' if good else 'failed',returncode=rc,scientifically_valid=good)
  return 0 if good else rc or 1
if __name__=='__main__':raise SystemExit(main())
