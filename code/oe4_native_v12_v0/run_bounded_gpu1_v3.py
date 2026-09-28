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
RUNTIME=ROOT/'code_snapshot/runtime_v1/runtime_smoke_v1.py'
EXPECTED='b737a7254c2d66518e3acc8b8314918c8529d21d33e04347b49be6029cac0b2c'
PROTECTED={
 'pilot_sen12_gp_heads.py':('8f805359c2989cd474e89d9086545698c0acdbb9673b27a82cc1ad00743a1af1',1788238578),
 'sen12_official_baselines.py':('19232b2ae122dd6529bac56977e075ce7b94c0daf90f24c21557b4d5cc213f5b',1787683824),
 'extract_sen12_fold_cache.py':('197895a05f2f6dddff12f6385c01aacf4f72bdbb877d273b473c83dc5d3a5e0a',1787659265),
 'audit_sen12_fold_cache.py':('4287dbcdef64c9ea9ea2a707e961ce6ba606b85405ee794fa91f44ded2419ec9',1787668636),
}
RUNS=[{'name':'native_smoke_gpu1_v2','train':8,'dev':4,'steps':8},
      {'name':'native_development_gpu1_v2','train':64,'dev':8,'steps':32}]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def timestamp():return datetime.now(timezone.utc).isoformat()
def protected():
 values={}
 for name,(expected,mtime) in PROTECTED.items():
  p=ROOT.parent/'code'/name;values[name]={'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns}
  if values[name]['sha256']!=expected or int(p.stat().st_mtime)!=mtime:raise RuntimeError(f'Protected source changed: {name}')
 return values

LAST_GPU_OBSERVATIONS=[]

def parse_gpu1_idle(gpu_text, process_text):
 busy=set()
 for row in csv.reader(process_text.splitlines()):
  if not row or not any(x.strip() for x in row):continue
  if len(row)!=2 or not row[0].strip().startswith('GPU-') or not row[1].strip().isdigit():
   raise ValueError('Malformed process query')
  busy.add(row[0].strip())
 seen=set();observations=[];choices=[]
 for row in csv.reader(gpu_text.splitlines()):
  if not row or not any(x.strip() for x in row):continue
  if len(row)!=4:raise ValueError('Malformed GPU query')
  idx,uuid,mem,util=[x.strip() for x in row]
  if not idx.isdigit() or not mem.isdigit() or not uuid.startswith('GPU-') or int(idx) in seen:
   raise ValueError('Malformed GPU identity/memory')
  idx=int(idx);mem=int(mem);seen.add(idx)
  unavailable=util in ('[Not Found]','N/A')
  if not unavailable and not util.isdigit():raise ValueError('Malformed GPU utilization')
  value=None if unavailable else int(util)
  allowed=idx==1 and uuid not in busy and ((mem==0) if unavailable else (mem<512 and value<5))
  observations.append({'index':idx,'uuid':uuid,'memory_mib':mem,'utilization_percent':value,
   'utilization_unavailable':unavailable,'has_compute_pid':uuid in busy,
   'allowed':allowed,'idle_basis':'memory_zero_and_no_compute_PID' if unavailable else 'memory_process_and_utilization'})
  if allowed:choices.append((idx,uuid))
 if 1 not in seen:raise ValueError('GPU1 missing')
 return choices,observations

def idle():
 global LAST_GPU_OBSERVATIONS
 try:
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
  choices,LAST_GPU_OBSERVATIONS=parse_gpu1_idle(gpu,apps)
  return choices
 except (OSError,subprocess.SubprocessError,ValueError,csv.Error) as e:
  LAST_GPU_OBSERVATIONS=[{'query_error':repr(e),'treated_as':'busy'}]
  return []

def main():
 lock=(ROOT/'bounded_sequence_gpu1_v3.lock').open('a')
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status=ROOT/'bounded_sequence_gpu1_v3_status.json'
 if status.exists():raise FileExistsError(status)
 state={'started_utc':timestamp(),'status':'waiting_for_assets','run_plan':RUNS,'runtime_sha256':EXPECTED,'maximum_gpu_wait_seconds':600,'maximum_worker_seconds':1100,'results':[],'controller_pid':os.getpid(),'scope':'engineering_development_not_novelty_or_confirmatory_result','protected_before':protected()}
 def write():
  tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status)
  print(json.dumps({'utc':timestamp(),'status':state['status'],'detail':state.get('detail')}),flush=True)
 try:
  if sha(RUNTIME)!=EXPECTED:raise RuntimeError('Runtime source hash mismatch')
  cpu_receipt=ROOT/'runs/native_cpu_contract_v1/receipt.json'
  cpu=json.loads(cpu_receipt.read_text())
  if cpu.get('status')!='PASS_bounded_encoder_update' or cpu.get('runtime_script_sha256')!=EXPECTED:raise RuntimeError('Require actual CPU runtime v1 contract PASS before GPU dispatch')
  state['cpu_preflight_receipt_sha256']=sha(cpu_receipt)
  state['controller_sha256']=sha(Path(__file__))
  write();start=time.monotonic()
  while not (ROOT/'asset_receipt.json').exists():
   if time.monotonic()-start>2400:raise TimeoutError('Asset acquisition did not finish within 40 minutes')
   time.sleep(10)
  state['asset_receipt_sha256']=sha(ROOT/'asset_receipt.json')
  base=[str(PYTHON),'-u',str(RUNTIME),'--source-root',str(ROOT/'source'),'--deps-root',str(ROOT/'deps'),'--checkpoint-dir',str(ROOT/'models/OlmoEarth-v1_2-Base'),'--h5-root',str(ROOT/'data/official_subset_1k')]
  env=dict(os.environ);env.pop('PYTHONPATH',None);env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2'
  prep=ROOT/'runs/data_prepare_v1'
  # v0 failed in status polling, after preparation and before any GPU run.
  previous=json.loads((ROOT/'bounded_sequence_status.json').read_text())
  if previous['status']!='failed' or previous['results']:raise RuntimeError('Unexpected v0 execution state')
  if '[Not Found]' not in previous.get('detail',''):raise RuntimeError('Unexpected v0 failure')
  receipt=json.loads((prep/'receipt.json').read_text())
  if receipt['status']!='data_prepared_not_gpu_validated':raise RuntimeError('Invalid existing preparation')
  manifest=prep/'data_manifest.json'
  if sha(manifest)!='54d12227d7319aec7aa0eefccc3d65e07039b6784134fe47d27f69b6b1b25e4e':raise RuntimeError('Prepared manifest changed')
  state['prepared_manifest_sha256']=sha(manifest)
  state['revision_reason']='User requested GPU1 retry; permit exact missing utilization only with zero used memory and no compute PID, verified q utilization samples and pmon idle; GPU0 excluded; runtime v1 unchanged'
  state['status']='reused_validated_preparation';write()
  for run in RUNS:
   state['status']='waiting_for_idle_gpu';state['detail']=run['name'];write()
   wait_start=time.monotonic();stable=None;streak=0
   while True:
    choices=idle();current=choices[0] if choices else None
    streak=streak+1 if current is not None and current==stable else (1 if current is not None else 0)
    stable=current
    state['gpu_last_observations']=LAST_GPU_OBSERVATIONS
    if streak>=3:break
    if time.monotonic()-wait_start>600:raise TimeoutError('GPU1 unavailable within 10 minutes; no jobs interrupted')
    time.sleep(10)
   # Recheck immediately before launching; never share a GPU with an active job.
   if stable not in idle():raise RuntimeError('GPU became busy before launch')
   protected()
   env['CUDA_VISIBLE_DEVICES']=stable[1]
   out=ROOT/'runs'/run['name']
   argv=base+['--out-dir',str(out),'--device','cuda:0','--train-files',str(run['train']),'--dev-files',str(run['dev']),'--steps',str(run['steps']),'--save-weights','--max-wall-seconds','900']
   item={'name':run['name'],'gpu_physical_index':stable[0],'gpu_uuid':stable[1],'started_utc':timestamp(),'argv':argv}
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
