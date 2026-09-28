"""CPU-only fake GPU/clock/worker tests and isolated temporary-ledger processes."""
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import launcher as L


def lock_probe(path, pipe):
    try:
        with L.LedgerLock(path):
            pipe.send('acquired')
    except BlockingIOError:
        pipe.send('blocked')
    finally:
        pipe.close()


class FakeClock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now
    def sleep(self, duration): self.now += duration


class FakeChild:
    pid = 654321
    def __init__(self, clock, *, normal=False, descendant=False, unkillable=False):
        self.clock = clock; self.returncode = None; self.normal = normal
        self.descendant = descendant; self.alive = True; self.unkillable = unkillable
        self.signals = []; self.waits = []
    def poll(self): return self.returncode
    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is not None: return self.returncode
        if self.normal:
            self.clock.now += min(1, timeout); self.returncode = 0
            self.alive = self.descendant; return 0
        self.clock.now += timeout
        raise subprocess.TimeoutExpired('fake owned worker', timeout)
    def kill(self, pgid, signum):
        assert pgid == self.pid
        self.signals.append((pgid, signum, self.clock()))
        if signum == signal.SIGKILL and not self.unkillable:
            self.alive = False; self.returncode = -9
    def group_alive(self, pgid):
        assert pgid == self.pid
        return self.alive


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp' if Path('/private/tmp').exists() else None)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.initial = self.root / 'initial.json'
        self.initial.write_text(json.dumps({'pilot_budget': {'used_seconds':0,'reserved_seconds':0,
            'jobs':[],'total_gpu_occupancy_seconds_max':7200,'per_job_wall_seconds_max':1800,
            'reset_each_heartbeat':False}}))
        self.pin = L.sha(self.initial)
        self.ledger = self.root / 'ledger.json'
    def gpu_run(self, args):
        if '--query-gpu=index,uuid,memory.used,utilization.gpu' in args:
            return f'0, GPU-other, 90000, 100\n1, {L.GPU_UUID}, 0, 0\n'
        return 'GPU-other, 555\n'
    def test_gpu1_idle_other_gpu_busy_allowed(self):
        self.assertEqual(L.gpu_idle(self.gpu_run)['uuid'], L.GPU_UUID)
    def test_gpu_wrong_uuid_occupied_utilization_and_parse_fail_closed(self):
        for kind in ('uuid','occupied','utilization','parse'):
            def run(args):
                s = self.gpu_run(args)
                if kind == 'uuid': return s.replace(L.GPU_UUID, 'GPU-wrong')
                if kind == 'occupied' and '--query-compute-apps=gpu_uuid,pid' in args:
                    return s + f'{L.GPU_UUID}, 777\n'
                if kind == 'utilization': return s.replace(', 0, 0', ', 0, 30')
                if kind == 'parse': return 'unknown'
                return s
            with self.subTest(kind=kind), self.assertRaises(ValueError): L.gpu_idle(run)
    def test_concurrent_reservation_lock_is_exclusive(self):
        with L.LedgerLock(self.ledger) as lock:
            value = lock.read_or_initialize(self.initial,self.pin)
            lock.reserve(value,'locked_reservation',1800,{})
            ctx = multiprocessing.get_context('fork')
            parent, child = ctx.Pipe(False)
            process = ctx.Process(target=lock_probe, args=(self.ledger, child)); process.start()
            self.assertTrue(parent.poll(3)); self.assertEqual(parent.recv(), 'blocked')
            process.join(3); self.assertEqual(process.exitcode, 0)
    def test_initial_snapshot_pin_and_nonzero_refused(self):
        with L.LedgerLock(self.ledger) as lock:
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                lock.read_or_initialize(self.initial, '0' * 64)
            self.initial.write_text(self.initial.read_text().replace('"used_seconds": 0', '"used_seconds": 1'))
            with self.assertRaisesRegex(ValueError, 'initial zero budget'):
                lock.read_or_initialize(self.initial, L.sha(self.initial))
        self.assertFalse(self.ledger.exists())
    def test_crash_reservation_fully_charged_and_blocks_next(self):
        with L.LedgerLock(self.ledger) as lock:
            value = lock.read_or_initialize(self.initial, self.pin)
            lock.reserve(value, 'job_a', 1800, {})
        with L.LedgerLock(self.ledger) as lock:
            value = lock.read_or_initialize(self.initial, self.pin)
            self.assertEqual(value['charged_seconds'], 1800)
            with self.assertRaisesRegex(ValueError, 'Unresolved'):
                lock.reserve(value, 'job_b', 1800, {})
    def test_duplicate_job_id_no_retry_and_cumulative_no_reset(self):
        with L.LedgerLock(self.ledger) as lock:
            value = lock.read_or_initialize(self.initial, self.pin)
            for i in range(4):
                row = lock.reserve(value, 'job_' + str(i), 1800, {})
                row.update(status='completed',actual_elapsed_seconds=1800,charged_seconds=1800)
                lock.update(value)
            with self.assertRaisesRegex(ValueError, 'Job ID already'):
                lock.reserve(value, 'job_0', 16, {})
            with self.assertRaisesRegex(ValueError, 'budget exceeded'):
                lock.reserve(value, 'job_4', 16, {})
        self.assertEqual(json.loads(self.ledger.read_text())['charged_seconds'], 7200)
    def test_deleted_ledger_cannot_reinitialize(self):
        with L.LedgerLock(self.ledger) as lock:
            lock.read_or_initialize(self.initial, self.pin)
            self.ledger.unlink()
            with self.assertRaisesRegex(ValueError, 'no reset'):
                lock.read_or_initialize(self.initial, self.pin)
    def test_elapsed_not_undercharged_and_unconfirmed_full_charge(self):
        with L.LedgerLock(self.ledger) as lock:
            value = lock.read_or_initialize(self.initial, self.pin)
            row = lock.reserve(value, 'job_a', 1800, {})
            row.update(status='completed',actual_elapsed_seconds=14.2,charged_seconds=14)
            with self.assertRaisesRegex(ValueError, 'not fully charged'): lock.update(value)
            row.update(status='cleanup_unconfirmed',charged_seconds=15)
            with self.assertRaisesRegex(ValueError, 'fully charged'): lock.update(value)
    def test_timeout_grace_inside_cap_and_only_owned_group(self):
        clock = FakeClock(); child = FakeChild(clock)
        result = L.supervise(child, 0, 30, clock=clock,kill_group=child.kill,
                             alive_group=child.group_alive,sleep=clock.sleep)
        self.assertEqual(result['status'], 'timeout')
        self.assertLessEqual(clock(), 30)
        self.assertEqual([s[1] for s in child.signals], [signal.SIGTERM,signal.SIGKILL])
        self.assertTrue(all(s[0] == child.pid and s[2] <= 30 for s in child.signals))
    def test_leader_exits_descendant_cleaned_not_success(self):
        clock = FakeClock(); child = FakeChild(clock,normal=True,descendant=True)
        result = L.supervise(child, 0, 30, clock=clock,kill_group=child.kill,
                             alive_group=child.group_alive,sleep=clock.sleep)
        self.assertEqual(result['status'], 'failed'); self.assertFalse(child.alive)
        self.assertEqual(len(child.signals), 2)
    def test_unreaped_group_stays_unconfirmed_at_deadline(self):
        clock = FakeClock(); child = FakeChild(clock,unkillable=True)
        result = L.supervise(child, 0, 30, clock=clock,kill_group=child.kill,
                             alive_group=child.group_alive,sleep=clock.sleep)
        self.assertEqual(result['status'], 'cleanup_unconfirmed'); self.assertLessEqual(clock(), 30)
    def test_exception_cleanup_starts_immediately(self):
        clock = FakeClock(); child = FakeChild(clock)
        L.terminate_owned(child, 30, clock=clock,kill_group=child.kill,
                          alive_group=child.group_alive,sleep=clock.sleep)
        self.assertEqual(child.signals[0][2], 0)
        self.assertLess(clock(), 30)
    def test_terminal_audit_allows_honest_incomplete_and_rejects_live(self):
        status = self.root / 'status.json'; audit = self.root / 'audit.json'; attest = self.root / 'attest.json'
        status.write_text(json.dumps({'status':'running','controller_pid':111}))
        with self.assertRaisesRegex(ValueError, 'P2 still running'):
            L.terminal_audit(status, None, None,scope=self.root,alive=lambda p:False,run=lambda a:'')
        value = {'status':'failed','finished_utc':'2026-09-28T00:00:00Z','controller_pid':111,
                 'protected_and_snapshot_unchanged':True,'workers':[{'status':'completed',
                 'receipt_status':'training_completed','pid':i} for i in range(200,205)]}
        status.write_text(json.dumps(value)); audit.write_text('{}')
        record = {'schema_version':'oe10_p2_terminal_audit_attestation_v1','audit_complete':True,
                  'p2_status_sha256':L.sha(status),'p2_terminal_status':'failed','completed_runs':5,
                  'expected_runs':6,'p2_result':'incomplete','audited_utc':'2026-09-28T00:01:00Z',
                  'audit_files':[{'path':str(audit),'sha256':L.sha(audit)}]}
        attest.write_text(json.dumps(record)); pin = L.sha(attest)
        result = L.terminal_audit(status,attest,pin,scope=self.root,alive=lambda p:False,run=lambda a:'')
        self.assertEqual(result['p2_result'],'incomplete')
        with self.assertRaisesRegex(ValueError, 'still active'):
            L.terminal_audit(status,attest,pin,scope=self.root,alive=lambda p:p==111,run=lambda a:'')
        value['protected_and_snapshot_unchanged'] = False; status.write_text(json.dumps(value))
        record['p2_status_sha256'] = L.sha(status); attest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'integrity not verified'):
            L.terminal_audit(status,attest,L.sha(attest),scope=self.root,alive=lambda p:False,run=lambda a:'')
    def test_argv_no_shell_and_environment_cleared(self):
        protocol = {'source':{'server':'/frozen/model','manifest_sha256':'sourcehash'},
                    'identity':{'server':'/identity.json','sha256':'inputhash'},
                    'sample':{'episode_id':'episode:k1'},'configuration':{'arm':'B2'},
                    'paths':{'source':'/eo/source','contexts':'/contexts'}}
        argv = L.worker_argv(protocol,'/reservation','/output')
        self.assertEqual(argv[2],'/frozen/model/check_connection.py')
        self.assertIn('--execute',argv); self.assertEqual(argv[argv.index('--arm')+1],'B2')
        with mock.patch.dict(os.environ, {'PYTHONPATH':'bad'}):
            env = L.clean_environment()
        self.assertNotIn('PYTHONPATH',env); self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
    def test_local_preflight_readonly_and_server_scope_failclosed(self):
        args = argparse.Namespace(protocol=self.initial,attestation=None,initial_budget=self.initial,
                initial_budget_sha=self.pin,launcher_sha='0'*64,attestation_sha=None,job_id='a',seconds=1800)
        before = sorted(str(p) for p in self.root.rglob('*'))
        report = L.preflight(args,run=self.gpu_run)
        self.assertFalse(report['ready']); self.assertIn('server_scope',report['reasons'][0])
        self.assertEqual(before,sorted(str(p) for p in self.root.rglob('*')))


@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux-only PR_SET_PDEATHSIG actual exec test')
class ParentDeathTests(unittest.TestCase):
    def test_parent_death_survives_normal_exec(self):
        with tempfile.TemporaryDirectory() as temp:
            ready = Path(temp)/'ready'; source = str(Path(__file__).resolve().parent)
            child_code = ('import os,sys;sys.path.insert(0,sys.argv[1]);from worker_entry import arm_parent_death;'
                          'arm_parent_death(int(sys.argv[2]));os.execv(sys.executable,[sys.executable,"-c",'
                          '"import os,time;from pathlib import Path;Path("+repr(sys.argv[3])+ ").write_text(str(os.getpid()));time.sleep(60)"])')
            parent_code = ('import os,subprocess,sys,time;subprocess.Popen([sys.executable,"-c",sys.argv[1],'
                           'sys.argv[2],str(os.getpid()),sys.argv[3]]);time.sleep(60)')
            parent = subprocess.Popen([sys.executable,'-B','-c',parent_code,child_code,source,str(ready)])
            child_pid = None
            try:
                end = time.monotonic()+5
                while not ready.exists() and time.monotonic()<end: time.sleep(.02)
                self.assertTrue(ready.exists()); child_pid = int(ready.read_text())
                parent.kill(); parent.wait(timeout=3)
                def living():
                    stat = Path(f'/proc/{child_pid}/stat')
                    return stat.exists() and stat.read_text().split()[2] != 'Z'
                end = time.monotonic()+3
                while living() and time.monotonic()<end: time.sleep(.02)
                self.assertFalse(living(),'Direct worker survived parent death across exec')
            finally:
                if parent.poll() is None: parent.kill(); parent.wait(timeout=3)
                if child_pid is not None:
                    try: os.kill(child_pid,signal.SIGKILL)
                    except ProcessLookupError: pass
    def test_parent_race_mismatch_refused(self):
        source = str(Path(__file__).resolve().parent)
        code = 'import os,sys;sys.path.insert(0,sys.argv[1]);from worker_entry import arm_parent_death;arm_parent_death(os.getppid()+100000)'
        result = subprocess.run([sys.executable,'-B','-c',code,source],capture_output=True,text=True,timeout=3)
        self.assertNotEqual(result.returncode,0); self.assertIn('Parent PID changed',result.stderr)


if __name__ == '__main__': unittest.main()
