"""Manual, bounded three-process OE11 pilot; importing/preflight reserves nothing."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,os,re,signal,subprocess,sys,time
from pathlib import Path

HERE=Path(__file__).absolute().parent
DEPENDENCIES={'_ledger_v1.py':'ef844bed9eed10cbabf67fe8d6019477d0d817fdf932b98eaf42e8c6fc4079b4',
              '_worker_entry_v1.py':'eb61444a881840f838ee81dd4481ff93e8773c97848862203b1cdb84ff1cc6b0'}
for name,digest in DEPENDENCIES.items():
    if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Pinned reused dependency changed: '+name)
_spec=importlib.util.spec_from_file_location('_oe11_reused_ledger',HERE/'_ledger_v1.py')
L=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(L)

DATA=L.DATA
ROOT=DATA/'oe11_training_pilot_v0'
SNAP=ROOT/'code_snapshot/oe11_training_launcher_v0'
RUNS=ROOT/'runs'
LEDGER=L.LEDGER  # Existing ledger only. No new/reset budget lineage.
PYTHON=L.PYTHON
INITIAL_SHA='0c7a2c815fc12270a68fa5cb9e8739a36614efe52837d94dcd13bd47b094cd20'
GPU_UUIDS={1:'GPU-8b485982-e8bb-004a-a10a-e37637e5e2bb',0:'GPU-83516c9a-4344-a891-ccf3-621c862d2468'}
STAGES=('uninterrupted','split','resume')
EXPECTED_STATUS={'uninterrupted':'uninterrupted_completed','split':'split_completed','resume':'resume_passed'}

def require(ok,message):L.require(ok,message)

def gpu_inventory(run=L.read_command):
    inventory=[[v.strip() for v in r] for r in csv.reader(run(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']).splitlines())]
    apps=[[v.strip() for v in r] for r in csv.reader(run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).splitlines())]
    require(inventory and all(len(r)==4 for r in inventory),'Malformed GPU inventory')
    require(all(len(r)==2 and r[1].isdigit() for r in apps),'Malformed GPU process inventory')
    results={}
    for index,uuid in GPU_UUIDS.items():
        row=[r for r in inventory if r[0]==str(index)]
        require(len(row)==1 and row[0][1]==uuid,'GPU index/UUID identity differs')
        memory=float(row[0][2]);require(math.isfinite(memory) and memory>=0,'Invalid memory telemetry')
        raw=row[0][3];unavailable=raw in {'[Not Found]','N/A'}
        utilization=None if unavailable else float(raw)
        require(unavailable or (math.isfinite(utilization) and 0<=utilization<=100),'Invalid utilization telemetry')
        pids=[int(r[1]) for r in apps if r[0]==uuid]
        idle=not pids and (memory==0 if unavailable else memory<=256 and utilization==0)
        results[index]={'index':index,'uuid':uuid,'memory_mib':memory,'utilization_percent':utilization,
            'utilization_raw':raw,'utilization_telemetry':'unavailable' if unavailable else 'numeric',
            'compute_pids':pids,'idle':bool(idle),'checked_utc':L.utc()}
    return results

def select_gpu(inventory,index=None):
    require(index is None or type(index) is int and index in GPU_UUIDS,'Only physical GPU0/1 permitted')
    for i in ([index] if index is not None else [1,0]):
        if inventory[i]['idle']:return inventory[i]
    raise ValueError('Selected GPU occupied' if index is not None else 'Both permitted GPUs occupied; no queue or reservation')

def verify_protocol(protocol,*,scope=DATA):
    require(protocol.get('backend')=='actual','Launcher accepts actual backend only')
    paths=protocol['paths'];pins=protocol['file_sha256']
    require(type(pins) is dict and pins,'Pinned input/source files required')
    required={'worker','cases','contexts','model_identity','prepared','episodes','eo_source','deps','eo_weights','qwen_weights'}
    require(required<=set(paths),'Protocol path coverage incomplete')
    for name,path in paths.items():
        require(type(path) is str,'Protocol paths must be strings')
        L.safe(path,within=scope)
    required_files=[paths[k] for k in ('worker','cases','contexts','model_identity')]
    required_files+=[str(Path(paths['prepared'])/'manifest.jsonl')]
    required_files+=[str(Path(paths['episodes'])/n) for n in ('episode_contract.json','episodes_train.jsonl','scoring/scoring_train.jsonl')]
    require(set(required_files)<=set(pins),'Required case/context/catalog identity missing')
    for name,h in pins.items():
        f=L.safe(name,within=scope,file=True)
        require(type(h) is str and re.fullmatch('[0-9a-f]{64}',h) and L.sha(f)==h,'Protocol file identity differs: '+name)
    execution=protocol.get('execution',{})
    arm=execution.get('arm',protocol['allowed_arms'][0]);condition=execution.get('condition',protocol['allowed_conditions'][0])
    require(arm in ('B0','B2') and arm in protocol['allowed_arms'],'Arm not allowed')
    require(condition in ('names_only','matched_knowledge') and condition in protocol['allowed_conditions'],'Condition not allowed')
    require(tuple(execution.get('stages',STAGES))==STAGES,'Required fresh-process stage sequence differs')
    index=execution.get('gpu_index');require(index is None or type(index) is int and index in GPU_UUIDS,'GPU selection differs')
    # Optional root-produced case preflight is independently hash-bound by file_sha256.
    if 'case_preflight' in paths:
        require(paths['case_preflight'] in pins,'Case preflight not pinned')
        report=L.pinned_json(paths['case_preflight'],pins[paths['case_preflight']])
        require(report.get('status') in {'PASS','passed'} or report.get('passed') is True,'Case preflight did not pass')
        coverage={'cases_sha256':paths['cases'],'contexts_sha256':paths['contexts'],
            'model_identity_sha256':paths['model_identity'],
            'prepared_manifest_sha256':str(Path(paths['prepared'])/'manifest.jsonl'),
            'train_catalog_sha256':str(Path(paths['episodes'])/'episodes_train.jsonl')}
        require(all(report.get(field)==pins[path] for field,path in coverage.items()),'Case preflight describes different input identities')
    return {'files_verified':len(pins),'arm':arm,'condition':condition,'stages':list(STAGES),'gpu_index':index}

def verify_models(protocol,*,scope=DATA):
    paths=protocol['paths'];identity=L.pinned_json(paths['model_identity'],protocol['file_sha256'][paths['model_identity']])
    result={}
    for location,key in [('qwen_weights','reader_files_sha256'),('eo_weights','eo_files_sha256'),('eo_source','eo_source_files_sha256')]:
        root=L.safe(paths[location],within=scope);items=identity[key]
        require(type(items) is dict and bool(items),'Empty model identity group')
        for name,digest in items.items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Invalid model identity path')
            require(L.sha(L.safe(root/name,within=root,file=True))==digest,'Model asset hash differs: '+name)
        result[key]=len(items)
    reader=L.safe(paths['qwen_weights'],within=scope)
    discovered={str(f.relative_to(reader)) for f in reader.rglob('*') if f.is_file() and f.suffix in {'.json','.jinja','.safetensors','.txt'}}
    require(discovered==set(identity['reader_files_sha256']),'Reader config/tokenizer/weight coverage differs')
    return result

def ledger_view(args):
    require(args.initial_budget_sha==INITIAL_SHA,'Existing bootstrap lineage required')
    require(LEDGER.is_file() and LEDGER.with_suffix('.initialized.json').is_file(),'Existing ledger/witness required; initialization prohibited')
    value=json.loads(L.safe(LEDGER,file=True).read_text());charged=L.validate_ledger(value)
    witness=json.loads(L.safe(LEDGER.with_suffix('.initialized.json'),file=True).read_text())
    require(value['initial_snapshot_sha256']==witness['initial_snapshot_sha256']==INITIAL_SHA,'Existing ledger lineage differs')
    L.pinned_json(args.initial_budget,INITIAL_SHA)
    old=[j for j in value['jobs'] if j['job_id']=='connection_20260928_01']
    require(len(old)==1 and old[0]['status']=='completed' and old[0]['charged_seconds']==48,'Original completed48-second charge not preserved')
    require(not any(j['status'] in {'reserved','running','cleanup_unconfirmed'} for j in value['jobs']),'Unresolved prior reservation')
    require(args.job_id not in {j['job_id'] for j in value['jobs']},'Job ID already consumed; retries need a new ID')
    require(type(args.seconds) is int and L.GRACE_SECONDS<args.seconds<=L.MAX_JOB and charged+args.seconds<=L.MAX_TOTAL,'Budget cap exceeded')
    require(not (RUNS/args.job_id).exists(),'Job output already exists')
    return {'charged_seconds':charged,'would_reserve_seconds':args.seconds,'read_only':True,'original_48_seconds_preserved':True}

def preflight(args,*,run=L.read_command):
    inputs_only=bool(getattr(args,'verify_inputs_only',False))
    report={'ready':False,'execution_ready':False,'reasons':[],'evidence':{},'read_only':True,
        'mode':'verify_inputs_only' if inputs_only else 'launch_preflight'}
    def check(name,fn):
        try:report['evidence'][name]=fn()
        except Exception as exc:report['reasons'].append(name+': '+str(exc))
    def server_scope():
        require(sys.platform.startswith('linux') and HERE==SNAP,'Actual preflight/launch requires frozen Linux server snapshot')
        L.safe(args.protocol,within=ROOT,file=True);L.safe(ROOT);L.safe(RUNS);L.safe(LEDGER)
        require(PYTHON.is_file(),'Expected worker Python missing')
        return True
    check('server_scope',server_scope)
    if report['reasons']:return report
    protocol=None
    try:
        protocol=L.pinned_json(args.protocol,args.protocol_sha256)
        report['protocol']=protocol
        report['evidence']['launcher_source']=L.verify_tree(HERE,'source_manifest.json',args.launcher_sha,{'launcher.py','_ledger_v1.py','_worker_entry_v1.py'})
    except Exception as exc:
        report['reasons'].append('protocol/source: '+str(exc));return report
    if not inputs_only:
        check('gpu_inventory',lambda:gpu_inventory(run))
        if 'gpu_inventory' in report['evidence']:
            check('selected_gpu',lambda:select_gpu(report['evidence']['gpu_inventory'],protocol.get('execution',{}).get('gpu_index')))
    report['gpu_check_performed']=not inputs_only
    check('p2_terminal_audit',lambda:L.terminal_audit(L.P2_STATUS,args.attestation,args.attestation_sha,run=run))
    check('ledger',lambda:ledger_view(args))
    if not inputs_only and 'selected_gpu' not in report['evidence']:
        report['evidence']['large_model_and_input_identity']='not_attempted_while_GPU_blocked'
        return report
    check('protocol_files',lambda:verify_protocol(protocol))
    if 'protocol_files' in report['evidence']:check('model_assets',lambda:verify_models(protocol))
    report['ready']=not report['reasons'];report['execution_ready']=report['ready'] and not inputs_only
    return report

def environment(index):
    require(type(index) is int and index in GPU_UUIDS,'Invalid physical GPU index')
    env=L.clean_environment();env['CUDA_VISIBLE_DEVICES']=GPU_UUIDS[index]
    env['OE11_PHYSICAL_GPU_INDEX']=str(index)
    env['OE11_EXPECTED_GPU_UUID']=GPU_UUIDS[index]
    env['OE11_MANUAL_BOUNDED_JOB']='1'
    return env

def worker_command(protocol,protocol_path,protocol_sha,stage,job):
    require(stage in STAGES,'Unexpected stage')
    execution=protocol.get('execution',{})
    arm=execution.get('arm',protocol['allowed_arms'][0]);condition=execution.get('condition',protocol['allowed_conditions'][0])
    argv=[str(PYTHON),'-B',protocol['paths']['worker'],'--protocol',str(protocol_path),'--protocol-sha256',protocol_sha,
          '--stage',stage,'--arm',arm,'--condition',condition,'--out',str(job/stage)]
    dynamic={}
    if stage=='resume':
        for flag,hashflag,path in [('--resume-from','--resume-sha256',job/'split/checkpoint.pt'),
                                   ('--reference-result','--reference-sha256',job/'uninterrupted/final_comparison.pt')]:
            f=L.safe(path,within=job,file=True);digest=L.sha(f)
            argv.extend([flag,str(f),hashflag,digest]);dynamic[str(f)]=digest
    return argv,dynamic

def run_stages(args,protocol,job,started,gpu,row,value,lock,*,clock=time.monotonic,run=L.read_command,popen=subprocess.Popen,supervise=L.supervise):
    stages=[];row['stages']=stages;deadline=started+args.seconds
    for stage in STAGES:
        if clock()>=deadline-L.GRACE_SECONDS:return {'status':'timeout','stages':stages,'reason':'No time remains for next stage and cleanup'}
        command,dynamic=worker_command(protocol,args.protocol,args.protocol_sha256,stage,job)
        before=select_gpu(gpu_inventory(run),gpu['index'])
        require(before['uuid']==gpu['uuid'],'Selected physical GPU changed')
        entry=[str(PYTHON),'-B',str(HERE/'_worker_entry_v1.py'),str(job/(stage+'_environment.json')),str(os.getpid()),'--',*command]
        record={'stage':stage,'status':'starting','gpu':before,'argv':entry,'shell':False,'dynamic_artifact_sha256':dynamic}
        L.atomic_json(job/(stage+'_command.json'),record,create=True)
        stage_start=clock();child=None
        try:
            with (job/(stage+'.log')).open('x') as log:
                env=environment(gpu['index'])
                # Telemetry and artifact hashing above consume the same job budget.
                # Recheck after that I/O, immediately before starting another GPU process.
                if clock()>=deadline-L.GRACE_SECONDS:
                    record.update(status='timeout',reason='Deadline reached during stage preparation; no worker started')
                    stages.append(record)
                    return {'status':'timeout','stages':stages,'reason':record['reason']}
                child=popen(entry,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=HERE)
                row.update(status='running',child_pid=child.pid,active_stage=stage,started_utc=row.get('started_utc',L.utc()))
                lock.update(value);record.update(status='running',pid=child.pid)
                L.atomic_json(job/'status.json',{'status':'running','active_stage':record,'finished_stages':stages})
                result=supervise(child,started,args.seconds)
            record.update(result,stage_elapsed_seconds=clock()-stage_start)
            if result['status']=='completed':
                receipt_path=L.safe(job/stage/'receipt.json',within=job,file=True)
                receipt=json.loads(receipt_path.read_text())
                require(receipt.get('status')==EXPECTED_STATUS[stage],'Stage exited zero without expected success receipt: '+stage)
                record.update(receipt_sha256=L.sha(receipt_path),worker_receipt_status=receipt['status'])
            stages.append(record);L.atomic_json(job/'status.json',{'status':result['status'],'finished_stages':stages})
            if result['status']!='completed':return {'status':result['status'],'stages':stages}
        except BaseException as exc:
            record.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',error=repr(exc))
            if record not in stages:stages.append(record)
            if child is not None and (child.poll() is None or L.group_alive(child.pid)):
                L.terminate_owned(child,deadline)
            # The caller settles the shared ledger and preserves unconfirmed cleanup.
            if child is not None and (child.poll() is None or L.group_alive(child.pid)):
                row['cleanup_unconfirmed_pid']=child.pid
            raise
    return {'status':'completed','stages':stages}

def settle(lock,value,row,started,seconds,*,clock=time.monotonic):
    elapsed=clock()-started;require(math.isfinite(elapsed) and elapsed>=0,'Invalid elapsed time')
    if row.get('cleanup_unconfirmed_pid') is not None and L.group_alive(row['cleanup_unconfirmed_pid']):row['status']='cleanup_unconfirmed'
    row.update(actual_elapsed_seconds=elapsed,finished_utc=L.utc())
    unresolved=row['status'] in {'reserved','running','cleanup_unconfirmed'}
    row['charged_seconds']=max(seconds,math.ceil(elapsed)) if unresolved else math.ceil(elapsed)
    lock.update(value)

def execute(args,report):
    require(report['ready'] and report.get('execution_ready') is True,'Cannot execute blocked or identity-only preflight')
    def interrupted(signum,frame):raise KeyboardInterrupt('Launcher interrupted by signal '+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupted)
    with L.LedgerLock(LEDGER) as lock:
        report=preflight(args);require(report['ready'],'Preflight changed: '+'; '.join(report['reasons']))
        require(LEDGER.is_file(),'Existing ledger disappeared; no reinitialization')
        value=lock.read_or_initialize(args.initial_budget,args.initial_budget_sha)
        row=lock.reserve(value,args.job_id,args.seconds,{'purpose':'oe11_training_resume_pilot','protocol_sha256':args.protocol_sha256,
            'launcher_manifest_sha256':args.launcher_sha,'preflight':report['evidence']})
        started=time.monotonic();job=RUNS/args.job_id
        try:
            RUNS.mkdir(exist_ok=True);job.mkdir(exist_ok=False)
            L.atomic_json(job/'preflight.json',report,create=True)
            L.atomic_json(job/'reservation.json',{'job_id':args.job_id,'status':'reserved','purpose':'oe11_training_resume_pilot',
                'reserved_seconds':args.seconds,'prior_gpu_used_seconds':value['charged_seconds']-args.seconds,
                'p2_finished_and_audited':True,'selected_gpu':report['evidence']['selected_gpu']},create=True)
            result=run_stages(args,report['protocol'],job,started,report['evidence']['selected_gpu'],row,value,lock)
            row.update(result)
        except BaseException as exc:
            row.update(status='interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',error=repr(exc))
        finally:
            settle(lock,value,row,started,args.seconds)
            if job.exists():
                L.atomic_json(job/'launcher_receipt.json',row)
                L.atomic_json(job/'status.json',{'status':row['status'],'finished_utc':row['finished_utc'],
                    'charged_seconds':row['charged_seconds'],'actual_elapsed_seconds':row['actual_elapsed_seconds'],
                    'stages':row.get('stages',[]),'error':row.get('error')})
        return row

def main():
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--preflight-only',action='store_true');m.add_argument('--execute',action='store_true')
    m.add_argument('--verify-inputs-only',action='store_true')
    for name in ('protocol','initial-budget','attestation'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('protocol-sha256','launcher-sha','initial-budget-sha','attestation-sha','job-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--seconds',type=int,default=1800);args=p.parse_args()
    require(re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}',args.job_id),'Invalid job ID')
    report=preflight(args)
    if args.preflight_only or args.verify_inputs_only or not report['ready']:
        print(json.dumps(report,indent=2));return 0 if report['ready'] else 2
    row=execute(args,report);print(json.dumps(row,indent=2));return 0 if row['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
