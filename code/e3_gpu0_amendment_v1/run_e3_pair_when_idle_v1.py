#!/usr/bin/env python3
"""Frozen E3 operational GPU0 launcher; preserve cooperative old/new exclusion."""
import json, os, subprocess, time
from pathlib import Path
from prepare_e3_gpu0_v1 import assert_quiescent, locks, now, read, require, sha
ROOT=Path('/home/work/data/olmoearth');OUT=ROOT/'e3_pair_dependence_v1';OLD=ROOT/'e3_pair_dependence_v0';PY=ROOT/'.venv-master/bin/python'

def main():
    with locks(ROOT) as handles:
        manifest=read(OUT/'manifest.json'); amendment=read(OUT/'amendment.json')
        require(sha(OUT/'amendment.json')==manifest['operational_amendment']['sha256'],'Amendment changed')
        require(sha(__file__)==amendment['launcher_sha256'],'Launcher changed')
        require(sha(Path(__file__).with_name('prepare_e3_gpu0_v1.py'))==amendment['prepare_helper_sha256'],'Helper changed')
        require(sha(OLD/'manifest.json')==amendment['parent_manifest_sha256'],'Parent freeze changed')
        require(sha(OUT/'operational_handoff.json')==amendment['handoff_sha256'],'Handoff changed')
        assert_quiescent(OLD)
        require(read(OUT/'status.json')['status']=='prepared','V1 already started or not prepared')
        require(not list(OUT.glob('answers_*.jsonl')) and not any((OUT/f).exists() for f in ['failure.json','reproduction.json','scores.json','runtime_environment.json']),'V1 contains inference outputs')
        queue=OUT/'queue.json';started=time.monotonic()
        def status(s,**kw):
            d={'status':s,'at':now(),'pid':os.getpid(),'elapsed_wait_s':time.monotonic()-started,**kw};queue.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        require(bool(uuid) and '\n' not in uuid,'Expected exactly one GPU0 UUID')
        while True:
            busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()
            if uuid not in [line.strip() for line in busy]:break
            status('waiting_gpu0')
            if time.monotonic()-started>6*3600:status('wait_expired_without_launch');return 3
            time.sleep(45)
        assert_quiescent(OLD)
        require(read(OUT/'status.json')['status']=='prepared','V1 preparation changed before launch')
        status('launching_frozen_snapshot',gpu_uuid=uuid)
        env=dict(os.environ);env.pop('PYTHONPATH',None);env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
        with (OUT/'run.log').open('x') as log:
            rc=subprocess.call([str(PY),'-B',str(OUT/'code_snapshot/e3_pair_dependence_v0.py'),'run','--out','e3_pair_dependence_v1'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=tuple(h.fileno() for h in handles))
        status('completed' if rc==0 else 'failed',returncode=rc);return rc

if __name__=='__main__':raise SystemExit(main())
