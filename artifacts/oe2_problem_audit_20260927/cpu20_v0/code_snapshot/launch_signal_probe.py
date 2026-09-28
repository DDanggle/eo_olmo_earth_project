#!/usr/bin/env python3
"""Run a bounded CPU audit from a pre-execution snapshot; no GPU or training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path('/home/work/data/olmoearth')
EXPECTED = {
 'pilot_sen12_gp_heads.py': (45602,1788238578,'8f805359c2989cd474e89d9086545698c0acdbb9673b27a82cc1ad00743a1af1'),
 'sen12_official_baselines.py': (10926,1787683824,'19232b2ae122dd6529bac56977e075ce7b94c0daf90f24c21557b4d5cc213f5b'),
 'extract_sen12_fold_cache.py': (10983,1787659265,'197895a05f2f6dddff12f6385c01aacf4f72bdbb877d273b473c83dc5d3a5e0a'),
 'audit_sen12_fold_cache.py': (7314,1787668636,'4287dbcdef64c9ea9ea2a707e961ce6ba606b85405ee794fa91f44ded2419ec9')}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write(p,v):
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def protected():
    result={}
    for name,expected in EXPECTED.items():
        p=ROOT/'code'/name; s=p.stat(); actual=(s.st_size,int(s.st_mtime),sha(p))
        if actual!=expected: raise RuntimeError('Protected source differs: '+name)
        result[name]={'size':s.st_size,'mtime_ns':s.st_mtime_ns,'sha256':actual[2]}
    return result

def main():
    run=ROOT/'oe2_problem_audit_20260927'/'cpu20_v0'
    run.mkdir(parents=True,exist_ok=False)
    status={'status':'preflight','started_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()}
    write(run/'launch_status.json',status)
    before=None
    try:
        before=protected()
        snap=run/'code_snapshot';snap.mkdir()
        source=Path(__file__).resolve().parent
        for name in ('signal_probe.py','launch_signal_probe.py','eo_model.py','candidates20_manifest.json','probe_plan.json'):
            shutil.copy2(source/name,snap/name)
            if sha(source/name)!=sha(snap/name): raise RuntimeError('Copy changed bytes')
        pins={p.name:sha(p) for p in snap.iterdir() if p.is_file()}
        plan=json.loads((snap/'probe_plan.json').read_text())
        model=ROOT/'oe1_bentxt_v0/models/OlmoEarth-v1-Tiny'
        if sha(model/'weights.pth')!=plan['encoder_weights_sha256']:
            raise RuntimeError('Wrong encoder checkpoint')
        if sha(snap/'candidates20_manifest.json')!=plan['candidate_manifest_sha256']:
            raise RuntimeError('Candidate manifest mismatch')
        command=[str(ROOT/'.venv-master/bin/python'),'-B',str(snap/'signal_probe.py'),
                 '--manifest',str(snap/'candidates20_manifest.json'),
                 '--data-dir',str(ROOT/'oe1_bentxt_v0/data'),
                 '--eo-ckpt',str(model),'--eo-module',str(snap/'eo_model.py'),
                 '--out',str(run/'probe'),'--device','cpu','--wall-seconds','540']
        environment=dict(os.environ);environment.pop('PYTHONPATH',None)
        environment.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        write(run/'launch_manifest.json',{'command':command,'source_snapshot_sha256':pins,
              'protected_code_before':before,'device':'cpu','thread_limit':4,
              'classification':'exploratory development sensor-signal audit; no VLM or training'})
        status['status']='running';write(run/'launch_status.json',status)
        with (run/'probe.log').open('x') as log:
            result=subprocess.run(command,cwd=snap,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=590)
        after=protected();write(run/'protected_code_after.json',after)
        if before!=after: raise RuntimeError('Protected source changed during run')
        if any(sha(snap/n)!=h for n,h in pins.items()): raise RuntimeError('Snapshot changed')
        child=json.loads((run/'probe/status.json').read_text())
        good=result.returncode==0 and child['status']=='completed'
        status.update(status='completed' if good else 'failed',returncode=result.returncode,valid=good,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write(run/'launch_status.json',status)
        return 0 if good else 1
    except BaseException as exc:
        status.update(status='failed',valid=False,error=repr(exc),finished_utc=datetime.now(timezone.utc).isoformat())
        if before is not None:
            try:
                after=protected();write(run/'protected_code_after.json',after)
                status['protected_unchanged']=before==after
            except Exception as guard_error: status['protected_error']=repr(guard_error)
        write(run/'launch_status.json',status)
        raise

if __name__=='__main__': raise SystemExit(main())
