"""CPU-only local tests with fake telemetry/workers and isolated existing ledgers."""
import argparse,json,multiprocessing,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest import mock
import launcher as N
L=N.L

def lock_probe(path,pipe):
    try:
        with L.LedgerLock(Path(path)):pipe.send('acquired')
    except BlockingIOError:pipe.send('blocked')
    finally:pipe.close()

class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now
    def sleep(self,seconds):self.now+=seconds

class TimeoutChild:
    pid=765432
    def __init__(self,clock,unkillable=False):self.clock=clock;self.returncode=None;self.alive=True;self.signals=[];self.unkillable=unkillable
    def poll(self):return self.returncode
    def wait(self,timeout):self.clock.now+=timeout;raise subprocess.TimeoutExpired('test-child',timeout)
    def kill(self,pgid,sig):
        assert pgid==self.pid;self.signals.append((pgid,sig,self.clock()))
        if sig==signal.SIGKILL and not self.unkillable:self.alive=False;self.returncode=-9
    def group_alive(self,pgid):assert pgid==self.pid;return self.alive

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/private/tmp' if Path('/private/tmp').is_dir() else None);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.initial=self.root/'initial.json'
        self.initial.write_text(json.dumps({'pilot_budget':{'used_seconds':0,'reserved_seconds':0,'jobs':[],
            'total_gpu_occupancy_seconds_max':7200,'per_job_wall_seconds_max':1800,'reset_each_heartbeat':False}}))
        self.pin=L.sha(self.initial);self.ledger=self.root/'ledger.json';self.runs=self.root/'runs'
        with L.LedgerLock(self.ledger) as lock:
            value=lock.read_or_initialize(self.initial,self.pin)
            old=lock.reserve(value,'connection_20260928_01',1800,{})
            old.update(status='completed',charged_seconds=48,actual_elapsed_seconds=47.45);lock.update(value)
        for name,val in [('LEDGER',self.ledger),('RUNS',self.runs),('INITIAL_SHA',self.pin)]:
            p=mock.patch.object(N,name,val);p.start();self.addCleanup(p.stop)
        self.args=argparse.Namespace(initial_budget=self.initial,initial_budget_sha=self.pin,job_id='pilot_01',seconds=1800,
            protocol=self.root/'protocol.json',protocol_sha256='1'*64,verify_inputs_only=False)
        self.protocol={'backend':'actual','paths':{'worker':'/frozen/worker.py'},'allowed_arms':['B0','B2'],
            'allowed_conditions':['names_only','matched_knowledge'],'execution':{'arm':'B2','condition':'matched_knowledge','stages':list(N.STAGES)}}
    def telemetry(self,memory1='0',util1='0',pid1=None,memory0='0',util0='0',pid0=None):
        def run(argv):
            if '--query-gpu=index,uuid,memory.used,utilization.gpu' in argv:
                return f"0, {N.GPU_UUIDS[0]}, {memory0}, {util0}\n1, {N.GPU_UUIDS[1]}, {memory1}, {util1}\n"
            return ''.join(f'{N.GPU_UUIDS[i]}, {pid}\n' for i,pid in [(0,pid0),(1,pid1)] if pid is not None)
        return run
    def test_gpu1_preferred_and_gpu0_fallback(self):
        self.assertEqual(N.select_gpu(N.gpu_inventory(self.telemetry()))['index'],1)
        self.assertEqual(N.select_gpu(N.gpu_inventory(self.telemetry(memory1='90000',util1='99',pid1=1472164)))['index'],0)
    def test_both_busy_and_fixed_index_refused(self):
        both=N.gpu_inventory(self.telemetry(memory1='90000',util1='99',pid1=1472164,memory0='60000',util0='99',pid0=1474335))
        with self.assertRaisesRegex(ValueError,'Both'):N.select_gpu(both)
        one=N.gpu_inventory(self.telemetry(pid1=111))
        with self.assertRaisesRegex(ValueError,'occupied'):N.select_gpu(one,1)
    def test_unavailable_utilization_requires_zero_memory_and_no_process(self):
        for value in ('[Not Found]','N/A'):
            self.assertTrue(N.gpu_inventory(self.telemetry(util1=value))[1]['idle'])
            self.assertFalse(N.gpu_inventory(self.telemetry(memory1='1',util1=value))[1]['idle'])
            self.assertFalse(N.gpu_inventory(self.telemetry(util1=value,pid1=111))[1]['idle'])
    def test_nonfinite_telemetry_and_uuid_fail_closed(self):
        for field in ('memory','utilization'):
            fn=self.telemetry(memory1='nan') if field=='memory' else self.telemetry(util1='inf')
            with self.assertRaises(ValueError):N.gpu_inventory(fn)
        normal=self.telemetry()
        with self.assertRaisesRegex(ValueError,'UUID'):N.gpu_inventory(lambda args:normal(args).replace(N.GPU_UUIDS[1],'GPU-wrong'))
    def test_existing48_ledger_preserved_and_view_readonly(self):
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report=N.ledger_view(self.args);self.assertEqual(report['charged_seconds'],48)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
    def test_missing_ledger_never_reinitialized(self):
        self.ledger.unlink()
        with self.assertRaisesRegex(ValueError,'initialization prohibited'):N.ledger_view(self.args)
        self.assertFalse(self.ledger.exists())
    def test_legacy_charge_mutation_refused(self):
        value=json.loads(self.ledger.read_text());value['jobs'][0].update(charged_seconds=47,actual_elapsed_seconds=47);value['charged_seconds']=47;L.atomic_json(self.ledger,value)
        with self.assertRaisesRegex(ValueError,'48-second'):N.ledger_view(self.args)
    def test_no_budget_or_jobid_reset_on_retry(self):
        with L.LedgerLock(self.ledger) as lock:
            value=lock.read_or_initialize(self.initial,self.pin);row=lock.reserve(value,'pilot_01',1800,{})
            row.update(status='failed',actual_elapsed_seconds=12.2,charged_seconds=13);lock.update(value)
        with self.assertRaisesRegex(ValueError,'consumed'):N.ledger_view(self.args)
        self.assertEqual(json.loads(self.ledger.read_text())['charged_seconds'],61)
    def test_unsettled_reservation_blocks_another_job(self):
        with L.LedgerLock(self.ledger) as lock:
            value=lock.read_or_initialize(self.initial,self.pin);lock.reserve(value,'crashed_job',1800,{})
        with self.assertRaisesRegex(ValueError,'Unresolved'):N.ledger_view(self.args)
        self.assertEqual(json.loads(self.ledger.read_text())['charged_seconds'],1848)
    def test_cumulative_and_per_job_cap(self):
        self.args.seconds=1801
        with self.assertRaisesRegex(ValueError,'cap'):N.ledger_view(self.args)
        with L.LedgerLock(self.ledger) as lock:
            value=lock.read_or_initialize(self.initial,self.pin)
            for i in range(3):
                row=lock.reserve(value,'done_'+str(i),1800,{});row.update(status='completed',actual_elapsed_seconds=1800,charged_seconds=1800);lock.update(value)
        self.args.seconds=1800
        with self.assertRaisesRegex(ValueError,'cap'):N.ledger_view(self.args)
    def test_ledger_lock_blocks_concurrent_reservation(self):
        with L.LedgerLock(self.ledger):
            ctx=multiprocessing.get_context('fork');parent,child=ctx.Pipe(False);proc=ctx.Process(target=lock_probe,args=(str(self.ledger),child));proc.start()
            self.assertTrue(parent.poll(3));self.assertEqual(parent.recv(),'blocked');proc.join(3);self.assertEqual(proc.exitcode,0)
    def test_failure_settlement_rounds_up_and_preserves48(self):
        with L.LedgerLock(self.ledger) as lock:
            value=lock.read_or_initialize(self.initial,self.pin);row=lock.reserve(value,'pilot_01',1800,{})
            row['status']='failed';N.settle(lock,value,row,100.,1800,clock=lambda:112.2)
            self.assertEqual(row['charged_seconds'],13);self.assertEqual(value['charged_seconds'],61)
    def test_unconfirmed_cleanup_retains_full_reservation(self):
        with L.LedgerLock(self.ledger) as lock:
            value=lock.read_or_initialize(self.initial,self.pin);row=lock.reserve(value,'pilot_01',1800,{})
            row['status']='cleanup_unconfirmed';N.settle(lock,value,row,0,1800,clock=lambda:10.2)
            self.assertEqual(value['charged_seconds'],1848)
    def test_timeout_grace_is_within_total_cap_own_group_only(self):
        c=Clock();child=TimeoutChild(c)
        result=L.supervise(child,0,30,clock=c,kill_group=child.kill,alive_group=child.group_alive,sleep=c.sleep)
        self.assertEqual(result['status'],'timeout');self.assertLessEqual(c(),30)
        self.assertEqual([r[1] for r in child.signals],[signal.SIGTERM,signal.SIGKILL])
        self.assertTrue(all(r[0]==child.pid for r in child.signals))
    def test_unclean_timeout_full_status(self):
        c=Clock();child=TimeoutChild(c,unkillable=True)
        result=L.supervise(child,0,30,clock=c,kill_group=child.kill,alive_group=child.group_alive,sleep=c.sleep)
        self.assertEqual(result['status'],'cleanup_unconfirmed');self.assertLessEqual(c(),30)
    def test_environment_fixed_single_gpu_no_pythonpath(self):
        with mock.patch.dict(os.environ,{'PYTHONPATH':'bad'}):
            env=N.environment(0);self.assertNotIn('PYTHONPATH',env);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],N.GPU_UUIDS[0]);self.assertEqual(env['HF_HUB_OFFLINE'],'1')
            self.assertEqual(env['OE11_PHYSICAL_GPU_INDEX'],'0');self.assertEqual(env['OE11_EXPECTED_GPU_UUID'],N.GPU_UUIDS[0])
    def test_resume_uses_own_artifact_hashes_and_new_process_stages(self):
        job=self.root/'job';job.mkdir();clock=Clock();calls=[]
        class Child:
            def __init__(self,pid):self.pid=pid;self.returncode=0
            def poll(self):return 0
        def popen(argv,**kwargs):
            calls.append((argv,kwargs));stage=argv[argv.index('--stage')+1];out=Path(argv[argv.index('--out')+1]);out.mkdir()
            (out/'receipt.json').write_text(json.dumps({'status':N.EXPECTED_STATUS[stage]}))
            if stage=='uninterrupted':(out/'final_comparison.pt').write_bytes(b'reference-owned')
            if stage=='split':(out/'checkpoint.pt').write_bytes(b'checkpoint-owned')
            return Child(1000+len(calls))
        def supervise(*args):clock.now+=2;return {'status':'completed','exit_code':0,'actual_elapsed_seconds':clock()}
        gpu=N.select_gpu(N.gpu_inventory(self.telemetry()))
        with L.LedgerLock(self.ledger) as lock:
            value=lock.read_or_initialize(self.initial,self.pin);row=lock.reserve(value,'pilot_01',1800,{})
            result=N.run_stages(self.args,self.protocol,job,0,gpu,row,value,lock,clock=clock,run=self.telemetry(),popen=popen,supervise=supervise)
        self.assertEqual(result['status'],'completed');self.assertEqual(len(calls),3)
        self.assertEqual(len({r['pid'] for r in result['stages']}),3)
        last=calls[-1][0];self.assertEqual(last[last.index('--resume-sha256')+1],L.sha(job/'split/checkpoint.pt'))
        self.assertEqual(last[last.index('--reference-sha256')+1],L.sha(job/'uninterrupted/final_comparison.pt'))
        self.assertTrue(all(kw['start_new_session'] and kw['env']['CUDA_VISIBLE_DEVICES']==N.GPU_UUIDS[1] for _,kw in calls))
    def test_missing_checkpoint_refused_before_resume(self):
        job=self.root/'job';job.mkdir()
        with self.assertRaises(ValueError):N.worker_command(self.protocol,self.args.protocol,'1'*64,'resume',job)
    def test_resume_symlink_to_external_artifact_rejected(self):
        job=self.root/'job';(job/'split').mkdir(parents=True)
        other=self.root/'other.pt';other.write_bytes(b'not owned')
        (job/'split/checkpoint.pt').symlink_to(other)
        with self.assertRaisesRegex(ValueError,'Symlink'):N.worker_command(self.protocol,self.args.protocol,'1'*64,'resume',job)
    def test_resume_parent_directory_symlink_rejected(self):
        job=self.root/'job';job.mkdir();other=self.root/'other';other.mkdir()
        (other/'checkpoint.pt').write_bytes(b'not owned');(job/'split').symlink_to(other,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'Symlink'):N.worker_command(self.protocol,self.args.protocol,'1'*64,'resume',job)
    def test_deadline_during_telemetry_cannot_start_stage(self):
        job=self.root/'job';job.mkdir();clock=Clock();self.args.seconds=30
        normal=self.telemetry();gpu=N.select_gpu(N.gpu_inventory(normal))
        def delayed(argv):clock.now+=8;return normal(argv)
        popen=mock.Mock(side_effect=AssertionError('Worker must not start'))
        with L.LedgerLock(self.ledger) as lock:
            value=lock.read_or_initialize(self.initial,self.pin);row=lock.reserve(value,'pilot_01',30,{})
            result=N.run_stages(self.args,self.protocol,job,0,gpu,row,value,lock,clock=clock,run=delayed,popen=popen)
            row.update(result);N.settle(lock,value,row,0,30,clock=clock)
            self.assertEqual(value['charged_seconds'],64)
        self.assertEqual(result['status'],'timeout');popen.assert_not_called();self.assertEqual(len(result['stages']),1)
        self.assertFalse((job/'uninterrupted').exists())
    def test_first_stage_failure_blocks_resume_and_charges_elapsed(self):
        job=self.root/'job';job.mkdir();clock=Clock();calls=[]
        class Child:
            pid=1001
            def poll(self):return 7
        def popen(argv,**kwargs):calls.append(argv);return Child()
        def supervise(*args):clock.now+=12.2;return {'status':'failed','exit_code':7}
        gpu=N.select_gpu(N.gpu_inventory(self.telemetry()))
        with L.LedgerLock(self.ledger) as lock:
            value=lock.read_or_initialize(self.initial,self.pin);row=lock.reserve(value,'pilot_01',1800,{})
            result=N.run_stages(self.args,self.protocol,job,0,gpu,row,value,lock,clock=clock,run=self.telemetry(),popen=popen,supervise=supervise)
            row.update(result);N.settle(lock,value,row,0,1800,clock=clock)
            self.assertEqual(value['charged_seconds'],61)
        self.assertEqual(result['status'],'failed');self.assertEqual(len(calls),1)
        self.assertEqual(result['stages'][0]['stage'],'uninterrupted');self.assertFalse((job/'split').exists())
    def test_busy_server_preflight_readonly_no_reservation_or_jobdir(self):
        self.args.protocol.write_text(json.dumps(self.protocol));self.args.protocol_sha256=L.sha(self.args.protocol)
        self.args.launcher_sha='f'*64;self.args.attestation=self.root/'attestation.json';self.args.attestation_sha='e'*64
        fake_python=self.root/'python';fake_python.write_text('placeholder')
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        busy=self.telemetry(memory1='90000',util1='99',pid1=1472164,memory0='60000',util0='99',pid0=1474335)
        with mock.patch.object(N.sys,'platform','linux'),mock.patch.object(N,'SNAP',N.HERE),mock.patch.object(N,'ROOT',self.root),mock.patch.object(N,'PYTHON',fake_python),mock.patch.object(L,'verify_tree',return_value={'files_verified':3}),mock.patch.object(L,'terminal_audit',return_value={'passed':True}),mock.patch.object(N,'verify_protocol') as vp,mock.patch.object(N,'verify_models') as vm:
            report=N.preflight(self.args,run=busy)
            self.assertFalse(report['ready']);self.assertFalse(report['execution_ready'])
            self.assertTrue(any('Both permitted GPUs occupied' in r for r in report['reasons']))
            self.assertEqual(report['evidence']['large_model_and_input_identity'],'not_attempted_while_GPU_blocked')
            vp.assert_not_called();vm.assert_not_called()
            with self.assertRaisesRegex(ValueError,'blocked'):N.execute(self.args,report)
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertFalse(self.runs.exists());self.assertEqual(json.loads(self.ledger.read_text())['charged_seconds'],48)
    def test_inputs_only_report_cannot_execute(self):
        with self.assertRaisesRegex(ValueError,'identity-only'):N.execute(self.args,{'ready':True,'execution_ready':False})
    def test_case_preflight_is_bound_to_exact_case_and_catalog_inputs(self):
        root=self.root/'inputs';root.mkdir()
        paths={name:str(root/name) for name in ('worker','cases','contexts','model_identity','prepared','episodes','eo_source','deps','eo_weights','qwen_weights','case_preflight')}
        for name in ('prepared','episodes','eo_source','deps','eo_weights','qwen_weights'):Path(paths[name]).mkdir()
        required=[Path(paths[name]) for name in ('worker','cases','contexts','model_identity')]
        required+=[Path(paths['prepared'])/'manifest.jsonl']
        required+=[Path(paths['episodes'])/name for name in ('episode_contract.json','episodes_train.jsonl','scoring/scoring_train.jsonl')]
        pins={}
        for path in required:path.parent.mkdir(exist_ok=True);path.write_text('{}');pins[str(path)]=L.sha(path)
        report={'status':'PASS','cases_sha256':pins[paths['cases']],'contexts_sha256':pins[paths['contexts']],
            'model_identity_sha256':pins[paths['model_identity']],'prepared_manifest_sha256':pins[str(Path(paths['prepared'])/'manifest.jsonl')],
            'train_catalog_sha256':pins[str(Path(paths['episodes'])/'episodes_train.jsonl')]}
        pf=Path(paths['case_preflight']);pf.write_text(json.dumps(report));pins[str(pf)]=L.sha(pf)
        protocol={**self.protocol,'paths':paths,'file_sha256':pins}
        self.assertEqual(N.verify_protocol(protocol,scope=root)['files_verified'],len(pins))
        report['cases_sha256']='f'*64;pf.write_text(json.dumps(report));pins[str(pf)]=L.sha(pf)
        with self.assertRaisesRegex(ValueError,'different input identities'):N.verify_protocol(protocol,scope=root)
    def test_local_preflight_is_readonly_and_does_not_launch(self):
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        report=N.preflight(self.args,run=self.telemetry());self.assertFalse(report['ready'])
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

if __name__=='__main__':unittest.main()
