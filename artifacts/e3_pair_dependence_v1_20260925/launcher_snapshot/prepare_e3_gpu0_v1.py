#!/usr/bin/env python3
"""Clone frozen E3 v0 preparation; change only GPU guard and compute destination."""
import argparse, contextlib, copy, fcntl, hashlib, json, os, shutil
from datetime import datetime, timezone
from pathlib import Path

PARENT_SHA = 'ac71805461a087fbc6fb81fd552977b7197c4e1b2c6f6d2ec07e6643948897a5'
RUNNER = 'e3_pair_dependence_v0.py'
CORE = {'items.jsonl':'items_sha256', 'pairs.npz':'pairs_sha256', 'saved_e2_real.json':'saved_e2_real_sha256'}
REPLACEMENTS = [
    ("['nvidia-smi','-i','1','--query-gpu=uuid'", "['nvidia-smi','-i','0','--query-gpu=uuid'"),
    ("RuntimeError('GPU1 occupied; do not allocate')", "RuntimeError('GPU0 occupied; do not allocate')"),
    ("os.environ.get('CUDA_VISIBLE_DEVICES')!='1'", "os.environ.get('CUDA_VISIBLE_DEVICES')!='0'"),
    ("RuntimeError('Explicit GPU1 mapping required')", "RuntimeError('Explicit GPU0 mapping required')")]

def require(ok, msg):
    if not ok: raise ValueError(msg)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,data):
    with Path(path).open('x') as stream: json.dump(data,stream,indent=2,allow_nan=False);stream.write('\n')
def now(): return datetime.now(timezone.utc).isoformat()
def patched_runner(original):
    for old,new in REPLACEMENTS:
        require(original.count(old)==1,'Runtime replacement anchor mismatch')
        original=original.replace(old,new)
    return original

def assert_quiescent(old):
    require(read(old/'status.json')['status']=='prepared','Parent is no longer prepared')
    forbidden=list(old.glob('answers_*.jsonl'))+[old/f for f in ['failure.json','reproduction.json','scores.json','runtime_environment.json'] if (old/f).exists()]
    require(not forbidden,'Parent inference/output artifacts exist')
    require(Path('/proc').is_dir(), 'Linux /proc required for handoff verification')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try: argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,ProcessLookupError):continue
        except PermissionError:raise RuntimeError('Cannot verify process cmdline: '+proc.name)
        names={Path(a).name for a in argv if a}
        require('run_e3_pair_when_idle_v0.py' not in names,'Old waiter remains alive: '+proc.name)
        if RUNNER in names and 'run' in argv:
            dest=argv[argv.index('--out')+1] if '--out' in argv else 'e3_pair_dependence_v0'
            require(dest not in (old.name,str(old)),'Old runner remains alive: '+proc.name)

@contextlib.contextmanager
def locks(root):
    handles=[]
    try:
        for name in ['.eo_e3_pair_dependence.lock','.eo_reader_gpu0.lock','.eo_reader_gpu1.lock']:
            f=(root/name).open('a'); handles.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield handles
    finally:
        for handle in reversed(handles):handle.close()

def prepare(root,old,out,runner,config,launcher,handoff,parent_sha=PARENT_SHA):
    root,old,out,runner,config,launcher,handoff=map(lambda p:Path(p).resolve(),(root,old,out,runner,config,launcher,handoff))
    require(old.parent==out.parent==root and old.name=='e3_pair_dependence_v0' and out.name=='e3_pair_dependence_v1', 'Unexpected output relationship')
    require(not out.exists(),'Refusing existing v1 output')
    with locks(root):
        assert_quiescent(old)
        require(sha(old/'manifest.json')==parent_sha,'Parent manifest differs from reviewed freeze')
        parent=read(old/'manifest.json');cfg=read(old/'prereg.json')
        require(sha(old/'prereg.json')==parent['prereg_sha256'],'Parent prereg changed')
        for f,k in CORE.items():require(sha(old/f)==parent[k],'Parent prepared bytes changed: '+f)
        for f,h in parent['code_snapshot_sha256'].items():require(sha(old/'code_snapshot'/f)==h,'Parent snapshot changed: '+f)
        require(runner.read_text()==patched_runner((old/'code_snapshot'/RUNNER).read_text()),'Runner must differ in exactly four GPU guard anchors')
        expected=copy.deepcopy(cfg)
        expected['compute']['gpu']='0 only after no other process, with GPU0 and legacy GPU1 research locks'
        expected['compute']['outputs']=str(out)+'/'
        require(read(config)==expected,'Prereg may change only compute.gpu and compute.outputs')
        require(handoff.is_file(),'Require audited quiescent handoff record')
        handoff_record=read(handoff)
        require(handoff_record.get('status')=='stopped_waiter_before_inference'
                and handoff_record.get('original_manifest_sha256')==parent_sha
                and handoff_record.get('pid')==955997
                and handoff_record.get('start_ticks')==1728957463
                and handoff_record.get('no_child_or_model_stopped') is True,
                'Handoff must prove the reviewed waiter stopped before inference')
        handoff_sha=sha(handoff)
        out.mkdir();(out/'code_snapshot').mkdir();(out/'parent_code_snapshot').mkdir();(out/'launcher_snapshot').mkdir()
        for f in CORE:shutil.copyfile(old/f,out/f)
        for f in ['manifest.json','prereg.json','status.json']:shutil.copyfile(old/f,out/('parent_'+f))
        for f in parent['code_snapshot_sha256']:
            shutil.copyfile(old/'code_snapshot'/f,out/'parent_code_snapshot'/f)
            shutil.copyfile(runner if f==RUNNER else old/'code_snapshot'/f,out/'code_snapshot'/f)
        shutil.copyfile(config,out/'prereg.json');shutil.copyfile(old/'input_audit.json',out/'input_audit.json')
        shutil.copyfile(handoff,out/'operational_handoff.json');shutil.copyfile(launcher,out/'launcher_snapshot/run_e3_pair_when_idle_v1.py')
        shutil.copyfile(__file__,out/'launcher_snapshot/prepare_e3_gpu0_v1.py')
        amendment={'schema':'e3-gpu-operational-amendment-v1','at':now(),'parent_out':str(old),'output':str(out),
                   'parent_manifest_sha256':parent_sha,'parent_prereg_sha256':parent['prereg_sha256'],
                   'reason':'GPU0 became idle while GPU1 remained occupied by unrelated work; pre-inference operational relocation',
                   'changed_fields':['compute.gpu','compute.outputs','runner.final_gpu_guard'],
                   'scientific_rules_unchanged':True,'selection_recomputed':False,'parent_inference_outputs_absent':True,
                   'old_waiter_and_runner_absent_checked':True,'handoff_sha256':handoff_sha,
                   'launcher_sha256':sha(launcher),'prepare_helper_sha256':sha(__file__),
                   'parent_input_audit_sha256':sha(old/'input_audit.json'),'parent_code_snapshot_sha256':parent['code_snapshot_sha256']}
        write(out/'amendment.json',amendment)
        manifest=copy.deepcopy(parent);manifest['prereg_sha256']=sha(out/'prereg.json')
        manifest['code_snapshot_sha256']={f:sha(out/'code_snapshot'/f) for f in parent['code_snapshot_sha256']}
        manifest['operational_amendment']={'path':'amendment.json','sha256':sha(out/'amendment.json'),'parent_manifest_sha256':parent_sha}
        for f,k in CORE.items():require(sha(out/f)==parent[k],'Copied prepared bytes changed')
        require(sha(old/'manifest.json')==parent_sha and sha(out/'operational_handoff.json')==handoff_sha,'Provenance changed during copy')
        assert_quiescent(old)
        write(out/'manifest.json',manifest);write(out/'status.json',{'status':'prepared','at':now(),'operational_amendment':'GPU0 only; original preparation reused'})
        return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/home/work/data/olmoearth'))
    for name in ['runner','config','launcher','handoff']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-manifest-sha256',default=PARENT_SHA);a=p.parse_args()
    result=prepare(a.root,a.root/'e3_pair_dependence_v0',a.root/'e3_pair_dependence_v1',a.runner,a.config,a.launcher,a.handoff,a.parent_manifest_sha256)
    print(json.dumps({k:result[k] for k in ['n_items','n_generations','items_sha256','pairs_sha256']}))
