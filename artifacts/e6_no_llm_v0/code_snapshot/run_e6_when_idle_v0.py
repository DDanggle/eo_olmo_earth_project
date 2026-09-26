#!/usr/bin/env python3
"""Run one prepared E6 snapshot on idle GPU0 with inherited cooperative locks."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/work/data/olmoearth')
PLAN_SHA='66f820db8bc6a3d03c07d66676adcbe6170c012649a824e83b02a94d03194840'
LOCKS=['.eo_e3_pair_dependence.lock','.eo_reader_gpu0.lock','.eo_reader_gpu1.lock','.eo_e5_equal_budget.lock','.eo_e6_no_llm.lock']
SOURCE_FILES={'e6_prepare_v0.py','e6_head_model_v0.py','e6_train_v0.py','e6_scoring_v0.py','run_e6_when_idle_v0.py'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate_prepared(out):
    manifest=read(out/'manifest.json')
    cfg=read(out/'prereg.json')
    require(manifest['schema']=='e6-no-llm-prepared-v0','Unexpected prepared manifest')
    require(sha(out/'prereg.json')==PLAN_SHA==manifest['files_sha256']['prereg.json'],'E6 plan changed')
    require(Path(__file__).resolve().parent==out/'code_snapshot','Only frozen launcher permitted')
    require(set(cfg['source_files'])==SOURCE_FILES and len(cfg['source_files'])==5
            and set(manifest['code_snapshot_sha256'])==SOURCE_FILES,'Incomplete or extra frozen executable membership')
    for name,digest in manifest['code_snapshot_sha256'].items():
        require(Path(name).name==name and sha(out/'code_snapshot'/name)==digest,'Frozen executable changed')
    require(cfg['compute']['gpu_index']==0 and cfg['compute']['lock_names']==LOCKS,'GPU/lease contract changed')
    require(read(out/'status.json')['status']=='prepared','Run already started or invalid')
    require(not any((out/n).exists() for n in ('run.log','run_claim.json','scores.json','failure.json','runtime_environment.json','training_summary.json','predictions.jsonl','initial_states','models')),
            'Existing E6 execution artifacts; no resume')
    gate=read(out/'reference_audit.json')
    require(gate.get('valid') is True and gate.get('independent_consistent') is True,'E5 reference gate failed')
    require(sha(out/'reference_audit.json')==manifest['files_sha256']['reference_audit.json'],'E5 gate changed')
    return manifest,cfg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'e6_no_llm_v0')
    args=p.parse_args();out=args.out.resolve()
    require(out==ROOT/'e6_no_llm_v0','Dedicated E6 output required')
    manifest,cfg=validate_prepared(out)
    initial_sha=sha(out/'manifest.json')
    with ExitStack() as stack:
        handles={}
        for name in LOCKS:
            handle=stack.enter_context((ROOT/name).open('a+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);handles[name]=handle
        started=time.monotonic()
        def status(stage,**extra):
            value={'status':stage,'at':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'elapsed_wait_s':time.monotonic()-started,**extra}
            (out/'queue.json').write_text(json.dumps(value,indent=2)+'\n')
            print(json.dumps(value),flush=True)
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=20).strip()
        require(uuid and '\n' not in uuid,'Expected one GPU0 UUID')
        while True:
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True,timeout=20)
            if time.monotonic()-started>=cfg['compute']['queue_wait_hours']*3600:
                status('wait_expired_without_launch');return 3
            if uuid not in {line.strip() for line in active.splitlines()}:break
            status('waiting_gpu0',gpu_uuid=uuid)
            time.sleep(45)
        validate_prepared(out)
        require(sha(out/'manifest.json')==initial_sha,'Prepared manifest changed while waiting')
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
            CUBLAS_WORKSPACE_CONFIG=':4096:8',E6_LOCK_FDS=json.dumps({k:v.fileno() for k,v in handles.items()}),
            E6_GPU_INDEX='0',E6_GPU_UUID=uuid,E6_MANIFEST_SHA256=initial_sha)
        status('launching_frozen_snapshot',gpu_uuid=uuid,manifest_sha256=initial_sha)
        with (out/'run.log').open('x') as log:
            rc=subprocess.call([str(ROOT/'.venv-master/bin/python'),'-B',str(out/'code_snapshot/e6_train_v0.py'),'--out',str(out)],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(h.fileno() for h in handles.values()))
        good=rc==0 and read(out/'status.json').get('status')=='completed' and (out/'scores.json').exists() and read(out/'scores.json').get('valid') is True
        status('completed' if good else 'failed',returncode=rc,scientifically_valid=good)
        return 0 if good else rc or 1


if __name__=='__main__':raise SystemExit(main())
