"""One-shot connection launcher. Import/tests never launch or reserve GPU work."""
from __future__ import annotations
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time

DATA = Path('/home/work/data/olmoearth')
ROOT = DATA / 'oe10_text_mask_identity_v2'
SNAP = ROOT / 'code_snapshot/oe10_connection_launcher_v1'
RUNS = ROOT / 'connection_runs_v0'
LEDGER = ROOT / 'pilot_budget_ledger_v0.json'
P2_ROOT = DATA / 'oe10_p2_v0'
P2_STATUS = P2_ROOT / 'training_v0/status.json'
PYTHON = DATA / '.venv-master/bin/python'
PROTOCOL_SHA = 'bb721164b472cf6b66f2e3833a2e99689d59e59ee0c1c604d05b4f2894f5b778'
GPU_UUID = 'GPU-8b485982-e8bb-004a-a10a-e37637e5e2bb'
MAX_JOB = 1800
MAX_TOTAL = 7200
GRACE_SECONDS = 15


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def safe(path, within=None, file=False):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink path rejected')
    if within is not None:
        require(path.is_relative_to(within), 'Path outside fixed server scope')
    if file:
        require(path.is_file(), 'Required regular file missing: ' + str(path))
    return path


def pinned_json(path, digest):
    path = safe(path, file=True)
    require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest), 'External SHA required')
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, 'External hash mismatch: ' + str(path))
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(payload, object_pairs_hook=unique)


def atomic_json(path, value, create=False):
    path = safe(path)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        if create:
            os.link(temp, path)  # Fails if a ledger/job already exists; never overwrite initialization.
            temp.unlink()
        else:
            os.replace(temp, path)
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if temp.exists():
            temp.unlink()


def verify_tree(root, manifest, digest, required):
    root = safe(root, file=False)
    m = pinned_json(root / manifest, digest)
    require(type(m.get('files')) is list and m['files'], 'Nonempty file list required')
    names = set()
    for item in m['files']:
        name = item['path']; rel = PurePosixPath(name)
        require(type(name) is str and not rel.is_absolute() and '..' not in rel.parts
                and str(rel) == name and '\\' not in name and name not in names
                and name != manifest, 'Bad/duplicate manifest path')
        names.add(name)
        p = safe(root / name, within=root, file=True)
        require(p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], 'Payload identity mismatch: ' + name)
    actual = set()
    for p in root.rglob('*'):
        safe(p)
        if '__pycache__' in p.relative_to(root).parts:
            require(p.is_dir() or p.suffix == '.pyc', 'Unexpected bytecode-directory entry')
            continue
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
        else:
            require(p.is_dir(), 'Nonregular payload')
    require(set(required) <= names and actual == names | {manifest}, 'Incomplete source/input coverage')
    return {'manifest_sha256': digest, 'files_verified': len(names)}


def read_command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=15, check=True)
    return result.stdout


def gpu_idle(run=read_command):
    rows = list(csv.reader(run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits']).splitlines()))
    rows = [[v.strip() for v in r] for r in rows]
    require(rows and all(len(r) == 4 for r in rows), 'Malformed GPU inventory')
    gpu = [r for r in rows if r[0] == '1']
    require(len(gpu) == 1 and gpu[0][1] == GPU_UUID, 'Physical GPU1 UUID mismatch')
    memory = float(gpu[0][2])
    require(math.isfinite(memory) and memory >= 0, 'Invalid GPU1 memory telemetry')
    utilization_raw = gpu[0][3]
    unavailable = utilization_raw in {'[Not Found]', 'N/A'}
    if unavailable:
        # Missing utilization is not treated as measured zero. Require the
        # stricter zero-memory condition plus the process check below.
        require(memory == 0, 'GPU1 unavailable utilization requires exactly zero memory')
        utilization = None
    else:
        utilization = float(utilization_raw)
        require(math.isfinite(utilization) and 0 <= memory <= 256 and utilization == 0,
                'GPU1 memory/utilization not idle')
    apps = [[v.strip() for v in r] for r in csv.reader(run([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']).splitlines())]
    require(all(len(r) == 2 and r[1].isdigit() for r in apps), 'Malformed GPU process inventory')
    require(not any(r[0] == GPU_UUID for r in apps), 'GPU1 occupied by a compute process')
    return {'index': 1, 'uuid': GPU_UUID, 'memory_mib': memory,
            'utilization_percent': utilization, 'utilization_raw': utilization_raw,
            'utilization_telemetry': 'unavailable' if unavailable else 'numeric',
            'compute_processes': [], 'checked_utc': utc()}


def pid_alive(pid):
    require(type(pid) is int and pid > 1, 'Invalid recorded P2 PID')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminal_audit(status_path, attestation_path, attestation_sha, *, alive=pid_alive,
                   run=read_command, scope=P2_ROOT):
    status_path = safe(status_path, within=scope, file=True)
    live = json.loads(status_path.read_text())
    require(live.get('status') in {'completed', 'failed'} and live.get('finished_utc'),
            'P2 still running or not durably terminal: ' + str(live.get('status')))
    require(attestation_path is not None and attestation_sha is not None, 'Terminal audit attestation missing')
    attestation = pinned_json(attestation_path, attestation_sha)
    require(attestation.get('schema_version') == 'oe10_p2_terminal_audit_attestation_v1'
            and attestation.get('audit_complete') is True, 'Terminal audit attestation not complete')
    status = pinned_json(status_path, attestation.get('p2_status_sha256'))
    require(status.get('status') in {'completed', 'failed'} and status.get('finished_utc'), 'P2 not terminal')
    require(status['status'] == attestation.get('p2_terminal_status'), 'Terminal status attestation mismatch')
    require(status.get('protected_and_snapshot_unchanged') is True, 'P2 protected/source integrity not verified')
    completed = sum(w.get('status') == 'completed' and w.get('receipt_status') == 'training_completed'
                    for w in status.get('workers', []))
    require(attestation.get('completed_runs') == completed and attestation.get('expected_runs') == 6,
            'Completed-run audit count mismatch')
    result = 'complete' if completed == 6 and status['status'] == 'completed' else 'incomplete'
    require(attestation.get('p2_result') == result, 'P2 completeness must be reported faithfully')
    finished = datetime.fromisoformat(status['finished_utc'].replace('Z', '+00:00'))
    audited = datetime.fromisoformat(attestation['audited_utc'].replace('Z', '+00:00'))
    require(finished.tzinfo is not None and audited.tzinfo is not None and audited >= finished,
            'Audit must follow termination')
    audits = attestation.get('audit_files')
    require(type(audits) is list and audits, 'Audit file hashes missing')
    for item in audits:
        p = safe(item['path'], within=scope, file=True)
        require(sha(p) == item['sha256'], 'Terminal audit payload mismatch')
    pids = [status.get('controller_pid')] + [w.get('pid') for w in status.get('workers', []) if 'pid' in w]
    require(pids and not any(alive(pid) for pid in pids), 'P2 controller/worker still active')
    process_rows = run(['ps', '-eo', 'pid=,args=']).splitlines()
    leftovers = [r for r in process_rows if str(scope) in r
                 and ('run_bounded.py' in r or 'p2_worker.py' in r)]
    require(not leftovers, 'P2 command-line process still active')
    return {'p2_result': result, 'completed_runs': completed, 'expected_runs': 6,
            'p2_status_sha256': attestation['p2_status_sha256'],
            'attestation_sha256': attestation_sha, 'audited_utc': attestation['audited_utc']}


def validate_ledger(value):
    require(value.get('schema_version') == 'oe10_gpu_pilot_ledger_v1'
            and value.get('maximum_seconds') == MAX_TOTAL, 'Ledger version/cap changed')
    jobs = value.get('jobs'); require(type(jobs) is list, 'Malformed ledger jobs')
    ids = [j['job_id'] for j in jobs]; require(len(ids) == len(set(ids)), 'Duplicate ledger job ID')
    total = 0
    for j in jobs:
        require(type(j['reserved_seconds']) is int and 0 < j['reserved_seconds'] <= MAX_JOB,
                'Invalid historic reservation')
        charge = j['charged_seconds']
        require(type(charge) is int and charge >= 0, 'Invalid charged time')
        if j['status'] in {'reserved', 'running', 'cleanup_unconfirmed'}:
            require(charge >= j['reserved_seconds'], 'Unfinished reservation must remain fully charged')
        else:
            require(j['status'] in {'completed', 'failed', 'timeout', 'interrupted'}
                    and math.isfinite(j['actual_elapsed_seconds'])
                    and charge >= math.ceil(j['actual_elapsed_seconds']), 'Finished occupancy not fully charged')
        total += charge
    require(value.get('charged_seconds') == total, 'Ledger arithmetic mismatch')
    return total


class LedgerLock:
    """One lock is retained through launch/settlement. Crash leaves full charge."""
    def __init__(self, path):
        self.path = safe(path); self.stream = None
    def __enter__(self):
        lock = safe(self.path.with_suffix('.lock'))
        fd = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(fd, 'a+')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.stream.close(); raise
        return self
    def __exit__(self, *exc):
        fcntl.flock(self.stream, fcntl.LOCK_UN); self.stream.close()
    def read_or_initialize(self, initial, initial_sha):
        marker = self.path.with_suffix('.initialized.json')
        if self.path.exists():
            require(marker.is_file(), 'Ledger initialization witness missing')
            witness = json.loads(safe(marker, file=True).read_text())
            value = json.loads(safe(self.path, file=True).read_text()); validate_ledger(value)
            require(value['initial_snapshot_sha256'] == witness['initial_snapshot_sha256'] == initial_sha,
                    'Initial ledger lineage changed')
            return value
        require(not marker.exists(), 'Ledger was initialized already; no reset/recreation')
        snapshot = pinned_json(initial, initial_sha)
        budget = snapshot.get('pilot_budget', snapshot)
        require(budget.get('used_seconds') == 0 and budget.get('reserved_seconds') == 0
                and budget.get('jobs') == [] and budget.get('total_gpu_occupancy_seconds_max') == MAX_TOTAL
                and budget.get('per_job_wall_seconds_max') == MAX_JOB
                and budget.get('reset_each_heartbeat') is False, 'Expected frozen initial zero budget required')
        value = {'schema_version': 'oe10_gpu_pilot_ledger_v1', 'maximum_seconds': MAX_TOTAL,
                 'initial_snapshot_sha256': initial_sha, 'created_utc': utc(),
                 'jobs': [], 'charged_seconds': 0}
        atomic_json(marker, {'initial_snapshot_sha256':initial_sha,'created_utc':utc()}, create=True)
        atomic_json(self.path, value, create=True); return value
    def reserve(self, value, job_id, seconds, metadata):
        charged = validate_ledger(value)
        require(type(job_id) is str and re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}', job_id), 'Invalid job ID')
        require(job_id not in {j['job_id'] for j in value['jobs']}, 'Job ID already consumed; no retry')
        require(not any(j['status'] in {'reserved', 'running', 'cleanup_unconfirmed'} for j in value['jobs']),
                'Unresolved reservation; no automatic recovery/reset')
        require(type(seconds) is int and GRACE_SECONDS < seconds <= MAX_JOB
                and charged + seconds <= MAX_TOTAL, 'Cumulative or per-job budget exceeded')
        row = {'job_id': job_id, 'reserved_seconds': seconds, 'charged_seconds': seconds,
               'status': 'reserved', 'reserved_utc': utc(), 'metadata': metadata}
        value['jobs'].append(row); value['charged_seconds'] += seconds
        atomic_json(self.path, value); return row
    def update(self, value):
        value['charged_seconds'] = sum(j['charged_seconds'] for j in value['jobs'])
        validate_ledger(value); atomic_json(self.path, value)


def group_alive(pgid):
    require(type(pgid) is int and pgid > 1 and pgid != os.getpgrp(), 'Invalid owned process group')
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_owned(child, deadline, *, clock=time.monotonic, kill_group=os.killpg,
                    alive_group=group_alive, sleep=time.sleep):
    """Clean only our recorded new-session group, including a surviving descendant."""
    require(child.pid > 1 and child.pid != os.getpgrp(), 'Invalid owned child group')
    def wait_until(end):
        while alive_group(child.pid) and clock() < end:
            duration = min(0.1, end - clock())
            if child.poll() is None:
                try: child.wait(timeout=duration)
                except subprocess.TimeoutExpired: pass
            else:
                sleep(duration)
        child.poll()  # Reap our direct child, if it exited after the last poll.
    if not alive_group(child.pid):
        return True
    try:
        kill_group(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    wait_until(min(clock() + GRACE_SECONDS - 1, max(clock(), deadline - 1)))
    if alive_group(child.pid):
        try: kill_group(child.pid, signal.SIGKILL)
        except ProcessLookupError: return True
        wait_until(deadline)
    return not alive_group(child.pid)


def supervise(child, started, seconds, *, clock=time.monotonic, kill_group=os.killpg,
              alive_group=group_alive, sleep=time.sleep):
    """The grace period is part of, never added to, the reservation."""
    deadline = started + seconds
    result = 'completed'
    try:
        child.wait(timeout=max(0.0, deadline - clock() - GRACE_SECONDS))
        if child.returncode:
            result = 'failed'
        if alive_group(child.pid):
            result = 'failed'  # An exited leader with a remaining group is not success.
            terminate_owned(child, deadline, clock=clock, kill_group=kill_group,
                            alive_group=alive_group, sleep=sleep)
    except subprocess.TimeoutExpired:
        result = 'timeout'
        terminate_owned(child, deadline, clock=clock, kill_group=kill_group,
                        alive_group=alive_group, sleep=sleep)
    except BaseException:
        terminate_owned(child, deadline, clock=clock, kill_group=kill_group,
                        alive_group=alive_group, sleep=sleep)
        raise
    if child.poll() is None or alive_group(child.pid):
        result = 'cleanup_unconfirmed'
    return {'status': result, 'exit_code': child.poll(), 'actual_elapsed_seconds': clock() - started}


def clean_environment():
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env.update(CUDA_VISIBLE_DEVICES='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return env


def worker_argv(protocol, reservation, output):
    args = [str(PYTHON), '-B', str(Path(protocol['source']['server']) / 'check_connection.py'),
            '--execute', '--source-manifest-sha256', protocol['source']['manifest_sha256'],
            '--identity-manifest', protocol['identity']['server'],
            '--identity-manifest-sha256', protocol['identity']['sha256'],
            '--reservation', str(reservation), '--episode-id', protocol['sample']['episode_id'],
            '--arm', protocol['configuration']['arm'], '--out', str(output)]
    for key, value in protocol['paths'].items():
        args += ['--' + key, value]
    return args


def preflight(args, *, run=read_command):
    evidence = {}; reasons = []
    def check(name, function):
        try:
            evidence[name] = function()
        except Exception as exc:
            reasons.append(name + ': ' + str(exc))
    def server_scope():
        require(sys.platform.startswith('linux') and Path(__file__).absolute().parent == SNAP,
                'Actual preflight/launch requires frozen fixed Linux server path')
        for p in (args.protocol, args.initial_budget):
            safe(p, within=ROOT, file=True)
        if args.attestation is not None:
            safe(args.attestation, within=ROOT)
        require(PYTHON.is_file(), 'Expected server Python missing')
        safe(ROOT); safe(RUNS); safe(LEDGER)
        return True
    check('server_scope', server_scope)
    if reasons:
        return {'ready': False, 'reasons': reasons, 'evidence': evidence}
    protocol = None
    try:
        protocol = pinned_json(args.protocol, PROTOCOL_SHA)
        evidence['protocol_sha256'] = PROTOCOL_SHA
        evidence['launcher_source'] = verify_tree(SNAP, 'source_manifest.json', args.launcher_sha,
                                                   {'launcher.py', 'worker_entry.py'})
        evidence['model_source'] = verify_tree(Path(protocol['source']['server']), 'source_manifest.json',
            protocol['source']['manifest_sha256'], {'check_connection.py', 'input_identity.py', 'gradient_gate.py'})
        identity = pinned_json(protocol['identity']['server'], protocol['identity']['sha256'])
        evidence['episodes'] = verify_tree(Path(protocol['paths']['episodes']), 'export_manifest.json',
            identity['episodes_export_manifest_sha256'], {'episode_contract.json','episodes_train.jsonl','scoring/scoring_train.jsonl'})
        require(sha(Path(protocol['paths']['prepared']) / 'manifest.jsonl') == identity['prepared_manifest_sha256'],
                'Prepared manifest identity mismatch')
        require(sha(protocol['paths']['contexts']) == identity['contexts_sha256'], 'Context identity mismatch')
        evidence['full_reader_eo_hash_recheck_pending_reserved_child'] = True
    except Exception as exc:
        reasons.append('identity: ' + str(exc))
    check('p2_terminal_audit', lambda: terminal_audit(P2_STATUS, args.attestation, args.attestation_sha, run=run))
    check('gpu', lambda: gpu_idle(run))
    def ledger_view():
        if LEDGER.exists():
            value = json.loads(safe(LEDGER, file=True).read_text())
            charged = validate_ledger(value)
            require(not any(j['status'] in {'reserved','running','cleanup_unconfirmed'} for j in value['jobs']),
                    'Unresolved prior reservation')
            require(args.job_id not in {j['job_id'] for j in value['jobs']}, 'Job ID consumed')
        else:
            initial = pinned_json(args.initial_budget, args.initial_budget_sha)
            b = initial.get('pilot_budget', initial)
            require(b.get('used_seconds') == 0 and b.get('reserved_seconds') == 0 and b.get('jobs') == [],
                    'Initial budget not zero')
            charged = 0
        require(charged + args.seconds <= MAX_TOTAL and GRACE_SECONDS < args.seconds <= MAX_JOB,
                'Insufficient cumulative/per-job budget')
        require(not (RUNS / args.job_id).exists(), 'Job output already exists')
        return {'charged_seconds': charged, 'would_reserve_seconds': args.seconds, 'read_only': True}
    check('ledger', ledger_view)
    return {'ready': not reasons, 'reasons': reasons, 'evidence': evidence, 'protocol': protocol}


def main():
    p = argparse.ArgumentParser()
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight-only', action='store_true')
    modes.add_argument('--execute', action='store_true')
    for name in ('protocol', 'initial-budget'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--attestation', type=Path)
    p.add_argument('--attestation-sha')
    for name in ('launcher-sha', 'initial-budget-sha', 'job-id'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--seconds', type=int, default=1800)
    args = p.parse_args()
    require(re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}', args.job_id), 'Invalid job ID')
    report = preflight(args)
    if args.preflight_only or not report['ready']:
        print(json.dumps(report, indent=2)); return 0 if report['ready'] else 2
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Launcher interrupted by signal ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with LedgerLock(LEDGER) as lock:
        # Recheck under ownership lock immediately before reservation/launch.
        report = preflight(args); require(report['ready'], 'Preflight changed: ' + '; '.join(report['reasons']))
        value = lock.read_or_initialize(args.initial_budget, args.initial_budget_sha)
        row = lock.reserve(value, args.job_id, args.seconds,
                           {'protocol_sha256': PROTOCOL_SHA, 'preflight': report['evidence']})
        started = time.monotonic(); child = None
        job = RUNS / args.job_id
        try:
            RUNS.mkdir(exist_ok=True); job.mkdir(exist_ok=False)
            reservation = {'status':'reserved','purpose':'text_mask_connection_check',
                           'p2_finished_and_audited':True,'reserved_seconds':args.seconds,
                           'prior_gpu_used_seconds':value['charged_seconds'] - args.seconds,
                           'job_id':args.job_id,'p2_result':report['evidence']['p2_terminal_audit']['p2_result']}
            atomic_json(job / 'reservation.json', reservation, create=True)
            atomic_json(job / 'preflight.json', report, create=True)
            gpu = gpu_idle(); row['gpu_immediately_before_launch'] = gpu
            command = worker_argv(report['protocol'], job / 'reservation.json', job / 'model_output')
            entry = [str(PYTHON), '-B', str(SNAP / 'worker_entry.py'), str(job / 'environment.json'), str(os.getpid()), '--', *command]
            atomic_json(job / 'command.json', {'argv':entry,'shell':False,'env':{k:v for k,v in clean_environment().items()
                if k in {'CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','PYTHONNOUSERSITE','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'}},
                'PYTHONPATH_removed':True}, create=True)
            with (job / 'worker.log').open('x') as log:
                child = subprocess.Popen(entry, env=clean_environment(), stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=SNAP)
                row.update(status='running', child_pid=child.pid, started_utc=utc()); lock.update(value)
                result = supervise(child, started, args.seconds)
            row.update(result)
            if result['status'] == 'completed':
                model_receipt = json.loads((job / 'model_output/receipt.json').read_text())
                require(model_receipt.get('status') == 'actual_connection_check_passed'
                        and model_receipt.get('mask_loss_and_all_present_gradients_finite') is True,
                        'Child exit zero without frozen model success receipt')
        except BaseException as exc:
            row.update(status='interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed', error=repr(exc))
            if child is not None and (child.poll() is None or group_alive(child.pid)):
                # supervise has only the recorded child process group; unresolved children retain full charge.
                terminate_owned(child, started + args.seconds)
        finally:
            elapsed = time.monotonic() - started
            row['actual_elapsed_seconds'] = elapsed; row['finished_utc'] = utc()
            if child is not None and (child.poll() is None or group_alive(child.pid)):
                row['status'] = 'cleanup_unconfirmed'
            row['charged_seconds'] = max(args.seconds, math.ceil(elapsed)) if row['status'] in {
                'reserved','running','cleanup_unconfirmed'} else math.ceil(elapsed)
            lock.update(value)
            if job.exists():
                atomic_json(job / 'launcher_receipt.json', row)
        print(json.dumps(row, indent=2))
        return 0 if row['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
