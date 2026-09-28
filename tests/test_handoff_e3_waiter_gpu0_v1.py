"""Synthetic E3 handoff guards. NEVER sends signals or reads real /proc.

Every main() invocation mocks all signal/process/GPU/pidfd operations. Signal
calls are recorded against a fake fd only; os.kill always raises if attempted.
"""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / 'code/e3_gpu0_amendment_v1/handoff_e3_waiter_gpu0_v1.py'
spec = importlib.util.spec_from_file_location('handoff_under_test', MODULE)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def good_info():
    return dict(pid=h.PID, start_ticks=h.START_TICKS, uid=os.getuid(), cwd=str(h.ROOT),
                argv=list(h.EXPECTED_CMD), state='S', children=[])


class MetadataTests(unittest.TestCase):
    def test_exact_waiting_identity_accepts_live_and_quiesced(self):
        for state in ('S', 'R', 'T', 't'):
            info = good_info(); info['state'] = state
            h.verify_waiting(info, {'status': 'prepared'}, {'pid': h.PID, 'status': 'waiting_gpu1'}, [])

    def test_changed_pid_start_uid_cwd_argv_or_dead_state_rejected(self):
        changes = [('pid', h.PID + 1), ('start_ticks', h.START_TICKS + 1), ('uid', os.getuid() + 1),
                   ('cwd', '/wrong'), ('argv', ['python', 'other_job.py']), ('state', 'Z'), ('state', 'X'),
                   ('children', ['123456'])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                info = good_info(); info[key] = value
                with self.assertRaises(RuntimeError):
                    h.verify_waiting(info, {'status': 'prepared'}, {'pid': h.PID, 'status': 'waiting_gpu1'}, [])
        with self.assertRaises(RuntimeError):
            h.verify_waiting(None, {'status': 'prepared'}, {'pid': h.PID, 'status': 'waiting_gpu1'}, [])

    def test_started_state_queue_mismatch_and_any_output_rejected(self):
        cases = [({'status': 'running'}, {'pid': h.PID, 'status': 'waiting_gpu1'}, []),
                 ({'status': 'prepared'}, {'pid': h.PID + 1, 'status': 'waiting_gpu1'}, []),
                 ({'status': 'prepared'}, {'pid': h.PID, 'status': 'launching_frozen_snapshot'}, []),
                 ({'status': 'prepared'}, {'pid': h.PID, 'status': 'waiting_gpu1'}, ['answers_seed1_real.jsonl'])]
        for status, queue, outputs in cases:
            with self.subTest(status=status, queue=queue, outputs=outputs), self.assertRaises(RuntimeError):
                h.verify_waiting(good_info(), status, queue, outputs)

    def test_output_markers_detect_even_empty_started_files(self):
        with tempfile.TemporaryDirectory(prefix='handoff_output_guard_') as directory:
            root = Path(directory)
            markers = ['answers_seed1_real.jsonl', 'runtime_environment.json', 'reproduction.json', 'scores.json', 'failure.json']
            for name in markers + ['queue.json', 'status.json', 'manifest.json', 'run.log']:
                (root / name).touch()
            self.assertEqual(h.output_names(root), sorted(markers))

    def test_direct_and_snapshot_runner_detection_with_fake_proc_only(self):
        with tempfile.TemporaryDirectory(prefix='handoff_proc_guard_') as directory:
            root = Path(directory)
            (root / '123').mkdir()
            for argv in ([b'python', b'code/e3_pair_dependence_v0.py', b'run'],
                         [b'python', b'/home/work/data/olmoearth/e3_pair_dependence_v0/code_snapshot/e3_pair_dependence_v0.py', b'run']):
                (root / '123/cmdline').write_bytes(b'\0'.join(argv) + b'\0')
                with mock.patch.object(h, 'Path', side_effect=lambda value: root if value == '/proc' else Path(value)):
                    with self.assertRaises(RuntimeError):
                        h.ensure_no_runner()


class MainFlowTests(unittest.TestCase):
    def scenario(self, mode='success'):
        temporary = tempfile.TemporaryDirectory(prefix='handoff_main_guard_')
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve(); output = root / 'e3_pair_dependence_v0'; output.mkdir()
        (output / 'status.json').write_text(json.dumps({'status': 'prepared'}))
        (output / 'queue.json').write_text(json.dumps({'pid': h.PID, 'status': 'waiting_gpu1'}))
        (output / 'manifest.json').write_text('{}')
        destination = root / 'handoff.json'
        if mode == 'record_exists': destination.write_text('preserve me')
        state = {'value': 'T' if mode == 'initial_stopped' else 'S', 'children': [], 'terminated': False}
        sent, scans = [], []
        fakefd = 123456789

        def process(pid):
            self.assertEqual(pid, h.PID)
            info = good_info(); info.update(state=state['value'], children=list(state['children']))
            return info

        def send(fd, sig, *args, **kwargs):
            self.assertEqual(fd, fakefd)
            sent.append(sig)
            if sig == h.signal.SIGSTOP:
                state['value'] = 'T'
                if mode == 'child_after_stop': state['children'] = ['900001']
                if mode == 'interrupted_after_stop': raise KeyboardInterrupt('synthetic interrupt after STOP delivered')
            elif sig == h.signal.SIGTERM:
                state['terminated'] = True
            elif sig == h.signal.SIGCONT:
                state['value'] = 'Z' if state['terminated'] else 'S'

        def runner_scan():
            scans.append(1)
            if mode == 'runner_after_stop' and len(scans) == 2:
                raise RuntimeError('Synthetic runner appeared during handoff')

        poll = mock.Mock()
        poll.poll.side_effect = lambda timeout=0: [(fakefd, 1)] if state['terminated'] else []
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for patcher in (
            mock.patch.object(h, 'ROOT', root),
            mock.patch.object(h, 'process', side_effect=process),
            mock.patch.object(h, 'ensure_no_runner', side_effect=runner_scan),
            mock.patch.object(h, 'sha', return_value=h.WAITER_SHA),
            mock.patch.object(h.os, 'pidfd_open', return_value=fakefd, create=True),
            mock.patch.object(h.os, 'close'),
            mock.patch.object(h.os, 'kill', side_effect=AssertionError('A raw PID signal is forbidden in tests')),
            mock.patch.object(h.signal, 'pidfd_send_signal', side_effect=send, create=True),
            mock.patch.object(h.signal, 'signal'),
            mock.patch.object(h.select, 'poll', return_value=poll, create=True),
            mock.patch.object(h.time, 'sleep'),
            mock.patch.object(h.subprocess, 'check_output', side_effect=['GPU-free\n', 'GPU-other\n']),
            mock.patch('sys.argv', ['handoff', '--out', str(destination)]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            stack.enter_context(patcher)
        return sent, state, destination

    def test_success_uses_pidfd_stop_term_cont_only(self):
        sent, state, destination = self.scenario()
        h.main()
        self.assertEqual(sent, [h.signal.SIGSTOP, h.signal.SIGTERM, h.signal.SIGCONT])
        self.assertEqual(json.loads(destination.read_text())['status'], 'stopped_waiter_before_inference')
        self.assertTrue(state['terminated'])
        h.os.kill.assert_not_called()

    def test_child_appearing_at_stop_aborts_and_resumes_without_term(self):
        sent, state, destination = self.scenario('child_after_stop')
        with self.assertRaisesRegex(RuntimeError, 'child'):
            h.main()
        self.assertEqual(sent, [h.signal.SIGSTOP, h.signal.SIGCONT])
        self.assertEqual(state['value'], 'S')
        self.assertFalse(state['terminated'])

    def test_runner_appearing_after_stop_aborts_and_resumes(self):
        sent, state, destination = self.scenario('runner_after_stop')
        with self.assertRaisesRegex(RuntimeError, 'runner appeared'):
            h.main()
        self.assertEqual(sent, [h.signal.SIGSTOP, h.signal.SIGCONT])
        self.assertEqual(json.loads(destination.read_text())['status'], 'handoff_aborted')

    def test_interrupt_after_stop_delivery_cannot_leave_waiter_stopped(self):
        sent, state, destination = self.scenario('interrupted_after_stop')
        with self.assertRaises(KeyboardInterrupt):
            h.main()
        self.assertEqual(sent, [h.signal.SIGSTOP, h.signal.SIGCONT])
        self.assertEqual(state['value'], 'S')

    def test_initially_stopped_waiter_is_not_resumed_or_signalled(self):
        sent, state, destination = self.scenario('initial_stopped')
        with self.assertRaises(RuntimeError):
            h.main()
        self.assertEqual(sent, [])
        self.assertEqual(state['value'], 'T')

    def test_existing_record_is_preserved_without_any_signal(self):
        sent, state, destination = self.scenario('record_exists')
        with self.assertRaises(RuntimeError):
            h.main()
        self.assertEqual(sent, [])
        self.assertEqual(destination.read_text(), 'preserve me')


if __name__ == '__main__':
    unittest.main()
