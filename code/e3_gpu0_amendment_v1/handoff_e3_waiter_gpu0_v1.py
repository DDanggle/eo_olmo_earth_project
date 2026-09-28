#!/usr/bin/env python3
"""Quiesce only the known, still-waiting E3 v0 launcher before a GPU0 amendment.

Never stops a model child or a foreign GPU process. Any ambiguity aborts and
resumes a stopped waiter. This script does not launch the amended experiment.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import select
import subprocess
import time
from datetime import datetime, timezone

ROOT=Path('/home/work/data/olmoearth')
PID=955997
START_TICKS=1728957463
WAITER_SHA='19bab30ccf73a3b096c563e76c47340386a701ceba06d37cbbcf6c6723cc5b6a'
EXPECTED_CMD=['.venv-master/bin/python','-B','code/run_e3_pair_when_idle_v0.py']

def require(ok, reason):
    if not ok: raise RuntimeError(reason)

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()

def process(pid):
    p=Path('/proc')/str(pid)
    try:
        stat=(p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':pid,'state':stat[0],'start_ticks':int(stat[19]),
                'argv':(p/'cmdline').read_bytes().decode().strip('\0').split('\0'),
                'uid':p.stat().st_uid,'cwd':str((p/'cwd').resolve()),
                'children':(p/'task'/str(pid)/'children').read_text().split()}
    except (FileNotFoundError, ProcessLookupError): return None

def verify_waiting(info, status, queue, output_names):
    require(info is not None and info['pid']==PID and info['start_ticks']==START_TICKS,'Wrong or recycled process')
    require(info['uid']==os.getuid() and info['cwd']==str(ROOT),'Different owner or workspace')
    require(info['argv']==EXPECTED_CMD,'Unexpected launcher command')
    require(info['state'] not in ('Z','X'),'Waiter is not live')
    require(not info['children'],'Waiter has a child; do not stop model or nvidia-smi')
    require(status.get('status')=='prepared','E3 already started')
    require(queue.get('pid')==PID and queue.get('status')=='waiting_gpu1','E3 is no longer waiting')
    require(not output_names,'E3 output already exists')

def output_names(out):
    return sorted(p.name for p in out.iterdir() if p.name.startswith('answers_') or p.name in
                  ('reproduction.json','runtime_environment.json','scores.json','failure.json'))

def ensure_no_runner():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try: argv=(p/'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        is_runner=any(Path(arg).name=='e3_pair_dependence_v0.py' for arg in argv) and 'run' in argv
        require(not is_runner,
                'An original E3 frozen runner exists; abort handoff')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    a=parser.parse_args(); dest=a.out.resolve()
    require(dest.parent==ROOT,'Handoff record must be directly under research root')
    require(not dest.exists(),'Never overwrite a handoff record')
    require(hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'),'Stable Linux pidfd support is required')
    out=ROOT/'e3_pair_dependence_v0'; stopped=False; terminated=False; pidfd=None
    report={'started_at':now(),'script_sha256':sha(Path(__file__)),'pid':PID,'start_ticks':START_TICKS,'status':'checking'}
    with dest.open('x') as stream: stream.write(json.dumps(report,indent=2)+'\n')
    def save(): dest.write_text(json.dumps(report,indent=2)+'\n')
    def interrupted(signum,frame): raise RuntimeError('Handoff interrupted by signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGHUP,interrupted)
    try:
        require(sha(ROOT/'code/run_e3_pair_when_idle_v0.py')==WAITER_SHA,'Waiter source changed')
        uuid=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
        require(uuid and uuid not in busy,'GPU0 is not empty')
        pidfd=os.pidfd_open(PID)
        exit_poll=select.poll();exit_poll.register(pidfd,select.POLLIN)
        report.update(gpu0_uuid=uuid,original_manifest_sha256=sha(out/'manifest.json'),
                      original_queue=read(out/'queue.json'),original_status=read(out/'status.json'),
                      original_process=process(PID))
        require(report['original_process'] is not None and report['original_process']['state'] not in ('T','t'),
                'Waiter was already stopped; do not take over another pause')
        verify_waiting(report['original_process'],report['original_status'],report['original_queue'],output_names(out))
        ensure_no_runner(); save()
        stopped=True
        signal.pidfd_send_signal(pidfd,signal.SIGSTOP)
        for _ in range(50):
            info=process(PID)
            if info and info['state'] in ('T','t'): break
            time.sleep(.1)
        require(info is not None and info['state'] in ('T','t'),'Could not quiesce waiter')
        verify_waiting(info,read(out/'status.json'),read(out/'queue.json'),output_names(out))
        ensure_no_runner()
        report['quiesced_process']=info; report['status']='quiesced_without_child'; save()
        report['status']='termination_requested';save()
        signal.pidfd_send_signal(pidfd,signal.SIGTERM)
        signal.pidfd_send_signal(pidfd,signal.SIGCONT); stopped=False
        for _ in range(100):
            if exit_poll.poll(0): terminated=True; break
            time.sleep(.1)
        require(terminated,'Waiter termination not confirmed; do not start another run')
        ensure_no_runner()
        require(read(out/'status.json')['status']=='prepared' and not output_names(out),'Original run began during handoff')
        report.update(status='stopped_waiter_before_inference',completed_at=now(),no_child_or_model_stopped=True)
        save(); print(json.dumps(report))
    except BaseException as exc:
        report.update(status='handoff_aborted',error=repr(exc),at=now()); save(); raise
    finally:
        if stopped and not terminated and pidfd is not None:
            try: signal.pidfd_send_signal(pidfd,signal.SIGCONT)
            except ProcessLookupError: pass
        if pidfd is not None: os.close(pidfd)

if __name__=='__main__': main()
