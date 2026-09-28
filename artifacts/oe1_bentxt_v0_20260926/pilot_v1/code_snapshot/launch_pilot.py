#!/usr/bin/env python3
"""Freeze OE1 source then execute it on idle physical GPU1 under cooperative locks."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path('/home/work/data/olmoearth')
PROTECTED={
 'pilot_sen12_gp_heads.py':(45602,1788238578,'8f805359c2989cd474e89d9086545698c0acdbb9673b27a82cc1ad00743a1af1'),
 'sen12_official_baselines.py':(10926,1787683824,'19232b2ae122dd6529bac56977e075ce7b94c0daf90f24c21557b4d5cc213f5b'),
 'extract_sen12_fold_cache.py':(10983,1787659265,'197895a05f2f6dddff12f6385c01aacf4f72bdbb877d273b473c83dc5d3a5e0a'),
 'audit_sen12_fold_cache.py':(7314,1787668636,'4287dbcdef64c9ea9ea2a707e961ce6ba606b85405ee794fa91f44ded2419ec9')}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')

def protected():
    state={}
    for name,(size,mtime,digest) in PROTECTED.items():
        p=ROOT/'code'/name;s=p.stat();actual=(s.st_size,int(s.st_mtime),sha(p))
        if actual!=(size,mtime,digest): raise RuntimeError('Protected prior code changed: '+name)
        state[name]={'bytes':actual[0],'mtime':actual[1],'sha256':actual[2]}
    return state

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-name',default='pilot_v0')
    a=ap.parse_args()
    if not a.run_name.startswith('pilot_') or '/' in a.run_name: raise ValueError('Use a dedicated pilot_* run name')
    base=ROOT/'oe1_bentxt_v0'
    run=base/'runs'/a.run_name
    run.mkdir(parents=True,exist_ok=False)
    status={'status':'preflight','started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()}
    write(run/'launch_status.json',status)
    try:
        with ExitStack() as stack:
            for name in ('.eo_reader_gpu1.lock','.oe1_bentxt_gpu1.lock'):
                handle=stack.enter_context((ROOT/name).open('a+'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            uuid=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
            if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):
                raise RuntimeError('Physical GPU1 is occupied; no training launched')
            before=protected()
            snapshot=run/'code_snapshot';snapshot.mkdir()
            sources={}
            for name in ('train_pilot.py','eo_model.py','pilot_plan.json','data_source_contract.json','launch_pilot.py'):
                src=Path(__file__).resolve().parent/name
                shutil.copy2(src,snapshot/name)
                if sha(src)!=sha(snapshot/name): raise RuntimeError('Snapshot copy mismatch')
                sources[name]=sha(snapshot/name)
            plan=json.loads((snapshot/'pilot_plan.json').read_text())
            data=base/'data'
            manifest=json.loads((data/'manifest.json').read_text())
            if manifest['status']!='complete': raise RuntimeError('Data preparation not complete')
            model=base/'models/OlmoEarth-v1-Tiny'
            if sha(model/'weights.pth')!=plan['models']['encoder_weights_sha256']:
                raise RuntimeError('Encoder checkpoint does not match fixed plan')
            # Byte provenance of the public corpus; no download and no shared mutation.
            source_paths={'tortilla':ROOT/'geobench2/benv2/geobench_benv2.tortilla',
                          'text':ROOT/'region_language_contract_20260921/BigEarthNet.txt.parquet'}
            source_hashes={k:sha(p) for k,p in source_paths.items()}
            if source_hashes['tortilla']!='821c2f429c3e85c158c758bbb215bf61170a2451a11284efaf0f89cef97e468a':
                raise RuntimeError('GEO-Bench original corpus SHA differs from official pinned file')
            environment=dict(os.environ);environment.pop('PYTHONPATH',None)
            environment.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                               TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            command=[str(ROOT/'.venv-master/bin/python'),'-B',str(snapshot/'train_pilot.py'),
                     '--data-dir',str(data),'--out',str(run/'training'),'--model-dir',str(model),
                     '--steps','256','--seed','17','--max-wall-seconds','3600']
            write(run/'launch_manifest.json',{'physical_gpu':1,'gpu_uuid':uuid,'command':command,
                  'source_snapshot_sha256':sources,'data_manifest_sha256':sha(data/'manifest.json'),
                  'public_source_sha256':source_hashes,'protected_code_before':before,
                  'classification':'development feasibility; no final benchmark claim'})
            apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
            if any(line.split(',')[0].strip()==uuid for line in apps.splitlines()):
                raise RuntimeError('Physical GPU1 became occupied during preflight; no training launched')
            status.update(status='training',gpu_uuid=uuid)
            write(run/'launch_status.json',status)
            with (run/'train.log').open('x') as log:
                result=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=3700)
            after=protected()
            write(run/'protected_code_after.json',after)
            if before!=after: raise RuntimeError('Protected prior code changed during run')
            if any(sha(snapshot/n)!=h for n,h in sources.items()): raise RuntimeError('Frozen source snapshot changed')
            completed=json.loads((run/'training/status.json').read_text())
            good=result.returncode==0 and completed.get('status')=='completed' and completed.get('valid') is True
            status.update(status='completed' if good else 'failed',returncode=result.returncode,
                          valid=good,finished_utc=datetime.now(timezone.utc).isoformat())
            write(run/'launch_status.json',status)
            return 0 if good else 1
    except BaseException as exc:
        status.update(status='failed',valid=False,error=repr(exc),finished_utc=datetime.now(timezone.utc).isoformat())
        write(run/'launch_status.json',status)
        raise

if __name__=='__main__': raise SystemExit(main())
