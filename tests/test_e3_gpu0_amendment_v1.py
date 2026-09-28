import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / 'code/e3_gpu0_amendment_v1'
sys.path.insert(0, str(BUNDLE))
import prepare_e3_gpu0_v1 as prepare




class GpuAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.old = self.root / 'e3_pair_dependence_v0'
        self.out = self.root / 'e3_pair_dependence_v1'
        self.old.mkdir()
        (self.old / 'code_snapshot').mkdir()
        self.original_runner = (REPO / 'code/e3_pair_dependence_v0.py').read_text()
        (self.old / 'code_snapshot/e3_pair_dependence_v0.py').write_text(self.original_runner)
        for name in ('e3_pair_scoring.py', 'e3_pair_transforms_v0.py'):
            (self.old / 'code_snapshot' / name).write_text('# synthetic frozen unchanged source\n')
        self.cfg = json.loads((REPO / 'config/e3_pair_dependence_prereg_v0.json').read_text())
        (self.old / 'prereg.json').write_text(json.dumps(self.cfg))
        for name in prepare.CORE:
            (self.old / name).write_bytes(b'synthetic frozen data\n')
        (self.old / 'status.json').write_text('{"status":"prepared"}')
        (self.old / 'input_audit.json').write_text('{"synthetic":true}')
        manifest = {key: prepare.sha(self.old / name) for name, key in prepare.CORE.items()}
        manifest.update(prereg_sha256=prepare.sha(self.old / 'prereg.json'), n_items=209, n_generations=3777,
                        code_snapshot_sha256={p.name: prepare.sha(p) for p in (self.old / 'code_snapshot').iterdir()})
        (self.old / 'manifest.json').write_text(json.dumps(manifest))
        self.parent_sha = prepare.sha(self.old / 'manifest.json')
        self.runner = self.root / 'runner.py'
        self.runner.write_text(prepare.patched_runner(self.original_runner))
        self.config = self.root / 'config.json'
        changed = copy.deepcopy(self.cfg)
        changed['compute']['gpu'] = '0 only after no other process, with GPU0 and legacy GPU1 research locks'
        changed['compute']['outputs'] = str(self.out) + '/'
        self.config.write_text(json.dumps(changed))
        self.launcher = BUNDLE / 'run_e3_pair_when_idle_v1.py'
        self.handoff = self.root / 'handoff.json'
        self.handoff.write_text(json.dumps({'status': 'stopped_waiter_before_inference',
            'original_manifest_sha256': self.parent_sha, 'pid': 955997, 'start_ticks': 1728957463,
            'no_child_or_model_stopped': True}))

    def tearDown(self):
        self.tmp.cleanup()

    def run_prepare(self):
        with mock.patch.object(prepare, 'assert_quiescent'):
            return prepare.prepare(self.root, self.old, self.out, self.runner, self.config,
                                   self.launcher, self.handoff, self.parent_sha)

    def test_actual_bundle_only_changes_four_runtime_literals_and_two_compute_fields(self):
        self.assertEqual((BUNDLE / 'e3_pair_dependence_v0.py').read_text(),
                         prepare.patched_runner(self.original_runner))
        changed = copy.deepcopy(self.cfg)
        changed['compute']['gpu'] = '0 only after no other process, with GPU0 and legacy GPU1 research locks'
        changed['compute']['outputs'] = '/home/work/data/olmoearth/e3_pair_dependence_v1/'
        self.assertEqual(json.loads((BUNDLE / 'e3_pair_dependence_prereg_v1.json').read_text()), changed)

    def test_prepare_preserves_parent_and_core_artifact_bytes(self):
        status_hash = prepare.sha(self.old / 'status.json')
        manifest = self.run_prepare()
        for filename, key in prepare.CORE.items():
            self.assertEqual(prepare.sha(self.out / filename), prepare.sha(self.old / filename))
            self.assertEqual(manifest[key], prepare.sha(self.old / filename))
        self.assertEqual(prepare.sha(self.old / 'manifest.json'), self.parent_sha)
        self.assertEqual(prepare.sha(self.old / 'status.json'), status_hash)
        self.assertEqual(json.loads((self.out / 'status.json').read_text())['status'], 'prepared')
        self.assertEqual(prepare.sha(self.out / 'parent_manifest.json'), self.parent_sha)

    def test_parent_snapshot_tamper_is_rejected(self):
        (self.old / 'code_snapshot/e3_pair_scoring.py').write_text('# changed\n')
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_scientific_config_change_is_rejected(self):
        changed = json.loads(self.config.read_text())
        changed['validity']['real_reproduction_min'] = .90
        self.config.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_extra_runner_edit_is_rejected(self):
        self.runner.write_text(self.runner.read_text() + '\n# unreviewed change\n')
        with self.assertRaises(ValueError):
            self.run_prepare()

    def test_existing_output_is_not_overwritten(self):
        self.out.mkdir()
        marker = self.out / 'marker'
        marker.write_text('preserve')
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.assertEqual(marker.read_text(), 'preserve')

    def test_arbitrary_handoff_record_is_rejected(self):
        self.handoff.write_text('{}')
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.assertFalse(self.out.exists())

    def test_launcher_inherits_all_locks_and_closes_without_explicit_unlock(self):
        tree = ast.parse(self.launcher.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'call' and isinstance(n.func.value, ast.Name) and n.func.value.id == 'subprocess']
        self.assertEqual(len(calls), 1)
        keyword = next(k for k in calls[0].keywords if k.arg == 'pass_fds')
        self.assertEqual(ast.unparse(keyword.value), 'tuple((h.fileno() for h in handles))')
        with mock.patch.object(prepare.fcntl, 'flock', wraps=prepare.fcntl.flock) as flock:
            with prepare.locks(self.root) as handles:
                saved = list(handles)
                self.assertEqual([Path(h.name).name for h in handles],
                                 ['.eo_e3_pair_dependence.lock', '.eo_reader_gpu0.lock', '.eo_reader_gpu1.lock'])
                self.assertTrue(all(not h.closed for h in handles))
            self.assertTrue(all(h.closed for h in saved))
            self.assertEqual(flock.call_count, 3)
            self.assertTrue(all(call.args[1] == prepare.fcntl.LOCK_EX | prepare.fcntl.LOCK_NB
                                for call in flock.call_args_list))


if __name__ == '__main__':
    unittest.main()
