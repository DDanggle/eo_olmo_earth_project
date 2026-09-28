"""CPU-only contract tests. No actual E5 outputs, EO training, or GPU calls."""
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import e6_train_v0 as runner


class NoDeadline:
    def check(self):
        pass


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, pairs, metadata):
        return self.linear(pairs[:, 0, 0, :1]).squeeze(-1)


class RunnerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_complete_frozen_exposure_schedule_and_tail(self):
        ids = [f'train_{i}' for i in range(4234)]
        for seed in (1, 2, 3):
            epochs = runner.expected_batches(ids, seed)
            self.assertEqual([len(epoch) for epoch in epochs], [530] * 3)
            for epoch in epochs:
                self.assertEqual([len(x) for x in epoch], [8] * 529 + [2])
                self.assertCountEqual([i for batch in epoch for i in batch], ids)
            self.assertEqual(sum(len(batch) for epoch in epochs for batch in epoch), 12702)
            self.assertNotEqual(epochs[0], epochs[1])
            self.assertEqual(epochs, runner.expected_batches(ids, seed))
        self.assertNotEqual(runner.expected_batches(ids, 1), runner.expected_batches(ids, 2))

    def test_features_use_global_indices_and_preserve_readonly_source(self):
        pairs = np.zeros((3, 2, 64, 768), dtype=np.float32)
        pairs[0] = 10; pairs[1] = 20; pairs[2] = 30
        pairs.setflags(write=False)
        metadata = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        x, m = runner.inputs(torch, pairs, metadata, [2, 0], 'cpu')
        self.assertEqual(x[:, 0, 0, 0].tolist(), [30, 10])
        self.assertEqual(m.tolist(), metadata[[2, 0]].tolist())
        x.zero_()
        self.assertEqual(float(pairs[2, 0, 0, 0]), 30)
        bad = np.array(pairs, copy=True); bad[2, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite batch'):
            runner.inputs(torch, bad, metadata, [2], 'cpu')
        with self.assertRaisesRegex(ValueError, 'Batch shape'):
            runner.inputs(torch, pairs, metadata[:, :7], [0], 'cpu')

    def test_target_source_mapping_and_test_partition_rejection(self):
        items = [{'answer': 'no', 'partition': 'train', 'kind': 'pos'},
                 {'answer': 'yes', 'partition': 'train', 'kind': 'hard_neg'},
                 {'answer': 'yes', 'partition': 'test'}]
        # Target mapping uses explicit source answer, never infers from kind.
        self.assertEqual(runner.targets(torch, items, [1, 0], 'cpu').tolist(), [1, 0])
        with self.assertRaisesRegex(ValueError, 'Test target'):
            runner.targets(torch, items, [2], 'cpu')
        items[0]['answer'] = 'unknown'
        with self.assertRaisesRegex(ValueError, 'Unknown binary'):
            runner.targets(torch, items, [0], 'cpu')

    def test_prediction_threshold_sigmoid_and_source_gold(self):
        item = dict(id='x', tile='t', cluster='c', phen='flood', kind='hard_neg', answer='no')
        for z, p, expected in [(0., .5, 'yes'), (-1000., 0., 'no'), (1000., 1., 'yes')]:
            row = runner.prediction_row(1, item, 73, z, p)
            self.assertEqual(row['source_gold'], 'no')
            self.assertIsNone(row['transformed_gold'])
            self.assertEqual(row['prediction'], expected)
            self.assertEqual(row['pair_index'], 73)
            self.assertEqual(row['model_arm'], 'full_head')
        for z, p in [(0., .7), (float('nan'), .5), (1., float('inf'))]:
            with self.assertRaises(ValueError):
                runner.prediction_row(1, item, 73, z, p)
        with self.assertRaises(ValueError):
            runner.prediction_row(True, item, 73, 0., .5)

    def training_fixture(self):
        items = [{'id': f'id{i}', 'answer': 'yes' if i % 2 else 'no', 'partition': 'train'} for i in range(10)]
        # Global item order intentionally differs from the scheduled sequence.
        pairs = np.zeros((10, 2, 64, 768), dtype=np.float32)
        pairs[:, 0, 0, 0] = np.arange(10) / 10
        metadata = torch.zeros((10, 8), dtype=torch.float32)
        schedule = [[list(reversed([x['id'] for x in items[2:]])), [x['id'] for x in items[:2]]] for _ in range(3)]
        return items, pairs, metadata, schedule

    def test_training_audits_actual_steps_tail_weighting_and_gradients(self):
        items, pairs, metadata, schedule = self.training_fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'steps.jsonl'
            torch.manual_seed(91); model = TinyHead()
            before = [p.detach().clone() for p in model.parameters()]
            result = runner.train_model(torch, model, pairs, metadata, items,
                                        {x['id']: i for i, x in enumerate(items)}, schedule,
                                        'cpu', path, NoDeadline(), seed=2)
            rows = runner.lines(path)
            self.assertEqual((result['updates'], result['exposures']), (6, 30))
            self.assertEqual([r['exposures'] for r in rows], [8, 10, 18, 20, 28, 30])
            self.assertEqual(rows[0]['pair_indices'], list(range(9, 1, -1)))
            self.assertEqual(rows[1]['pair_indices'], [0, 1])
            self.assertEqual([r['seed'] for r in rows], [2] * 6)
            self.assertEqual([r['ids'] for r in rows], [b for e in schedule for b in e])
            for epoch in range(3):
                a, b = rows[2 * epoch:2 * epoch + 2]
                self.assertAlmostEqual(result['epoch_example_mean_bce'][epoch], (a['loss'] * 8 + b['loss'] * 2) / 10)
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
            self.assertTrue(all(r['gradients_finite'] and r['parameters_finite'] and math.isfinite(r['loss']) for r in rows))
            group = result['optimizer_groups'][0]
            self.assertEqual((group['lr'], group['weight_decay'], group['betas']), (.0001, .01, (.9, .999)))
            self.assertIs(group['foreach'], False); self.assertIs(group['fused'], False)
            self.assertEqual(result['steps_sha256'], runner.sha(path))
            with self.assertRaises(FileExistsError):
                runner.train_model(torch, model, pairs, metadata, items, {x['id']: i for i, x in enumerate(items)},
                                   schedule, 'cpu', path, NoDeadline())

    def test_nonfinite_gradient_stops_before_parameter_update(self):
        items, pairs, metadata, schedule = self.training_fixture()
        with tempfile.TemporaryDirectory() as temp:
            model = TinyHead(); before = [p.detach().clone() for p in model.parameters()]
            hook = model.linear.weight.register_hook(lambda grad: torch.full_like(grad, float('nan')))
            path = Path(temp) / 'steps.jsonl'
            with self.assertRaisesRegex(ValueError, 'Nonfinite trainable gradient'):
                runner.train_model(torch, model, pairs, metadata, items, {x['id']: i for i, x in enumerate(items)},
                                   schedule, 'cpu', path, NoDeadline())
            hook.remove()
            self.assertEqual(path.read_text(), '')
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, model.parameters())))

    def frozen_fixture(self, root):
        source_names = ['e6_prepare_v0.py', 'e6_head_model_v0.py', 'e6_train_v0.py', 'e6_scoring_v0.py', 'run_e6_when_idle_v0.py']
        snapshot = root / 'code_snapshot'; snapshot.mkdir()
        for name in source_names:
            (snapshot / name).write_text('# synthetic ' + name + '\n')
        for name in runner.REQUIRED - {'prereg.json'}:
            (root / name).write_text('{}\n')
        inherited = {name: runner.sha(root / name) for name in ['items.jsonl', 'pairs.npy', 'ordered_ids.json', 'batches.json', 'eval_sets.json']}
        cfg = {'source_files': source_names, 'reference': {'files_sha256': inherited}}
        runner.write(root / 'prereg.json', cfg)
        (root / 'reference_extra.json').write_text('{}\n')
        manifest = {'schema': 'e6-no-llm-prepared-v0', 'parent_e5_prepared_manifest_sha256': runner.PARENT_SHA,
                    'n_items': 5989, 'n_train': 4234, 'n_test': 1755,
                    'files_sha256': {n: runner.sha(root / n) for n in runner.REQUIRED | {'reference_extra.json'}},
                    'code_snapshot_sha256': {n: runner.sha(snapshot / n) for n in source_names}}
        return manifest, snapshot

    def test_frozen_contract_allows_audit_extras_but_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, snapshot = self.frozen_fixture(root)
            with mock.patch.object(runner, '__file__', str(snapshot / 'e6_train_v0.py')), mock.patch.object(runner, 'PLAN_SHA', runner.sha(root / 'prereg.json')):
                runner.verify_files(root, manifest)
                (root / 'reference_extra.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                    runner.verify_files(root, manifest)
                (root / 'reference_extra.json').write_text('{}\n')
                (snapshot / 'e6_head_model_v0.py').write_text('# altered\n')
                with self.assertRaisesRegex(ValueError, 'Frozen code changed'):
                    runner.verify_files(root, manifest)

    def test_rehashed_input_cannot_override_inherited_parent_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, snapshot = self.frozen_fixture(root)
            (root / 'items.jsonl').write_text('{"replacement":"self consistent manifest"}\n')
            manifest['files_sha256']['items.jsonl'] = runner.sha(root / 'items.jsonl')
            with mock.patch.object(runner, '__file__', str(snapshot / 'e6_train_v0.py')), mock.patch.object(runner, 'PLAN_SHA', runner.sha(root / 'prereg.json')):
                with self.assertRaisesRegex(ValueError, 'Inherited E5 input pin differs'):
                    runner.verify_files(root, manifest)

    def test_code_snapshot_exact_coverage_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, snapshot = self.frozen_fixture(root)
            with mock.patch.object(runner, '__file__', str(snapshot / 'e6_train_v0.py')), mock.patch.object(runner, 'PLAN_SHA', runner.sha(root / 'prereg.json')):
                del manifest['code_snapshot_sha256']['e6_prepare_v0.py']
                with self.assertRaisesRegex(ValueError, 'Exact code snapshot coverage'):
                    runner.verify_files(root, manifest)
            for name in ['/etc/passwd', '../escape', 'missing']:
                with self.assertRaises(ValueError):
                    runner.relative(root, name)
            (root / 'outside').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                runner.relative(root, 'outside')

    def test_inherited_locks_must_already_be_held(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fds = {name: os.open(root / name, os.O_CREAT | os.O_RDWR, 0o600) for name in runner.LOCKS}
            env = {'E6_GPU_INDEX': '0', 'CUDA_VISIBLE_DEVICES': '0', 'E6_LOCK_FDS': json.dumps(fds)}
            try:
                with mock.patch.dict(os.environ, env):
                    with self.assertRaisesRegex(ValueError, 'not already held'):
                        runner.check_locks(root)
                    for fd in fds.values():
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.assertEqual(runner.check_locks(root), fds)
                    # Verification preserves ownership, rather than releasing inherited leases.
                    probe = os.open(root / runner.LOCKS[0], os.O_RDWR)
                    try:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    finally:
                        os.close(probe)
                    altered = dict(fds); altered[runner.LOCKS[0]] = fds[runner.LOCKS[1]]
                    with mock.patch.dict(os.environ, {'E6_LOCK_FDS': json.dumps(altered)}):
                        with self.assertRaisesRegex(ValueError, 'identity differs'):
                            runner.check_locks(root)
                    altered = dict(fds); altered[runner.LOCKS[0]] = 1
                    with mock.patch.dict(os.environ, {'E6_LOCK_FDS': json.dumps(altered)}):
                        with self.assertRaisesRegex(ValueError, 'Invalid inherited descriptor'):
                            runner.check_locks(root)
            finally:
                for fd in fds.values():
                    os.close(fd)

    def test_idle_guard_uuid_busy_and_command_timeout(self):
        with mock.patch.dict(os.environ, {'E6_GPU_UUID': 'GPU-test'}):
            with mock.patch.object(runner.subprocess, 'check_output', side_effect=['GPU-test\n', 'GPU-other\n']) as command:
                self.assertEqual(runner.check_idle(), 'GPU-test')
                self.assertTrue(all(call.kwargs['timeout'] == 20 for call in command.call_args_list))
            with mock.patch.object(runner.subprocess, 'check_output', side_effect=['GPU-test\n', 'GPU-test\n']):
                with self.assertRaisesRegex(ValueError, 'became busy'):
                    runner.check_idle()
            with mock.patch.object(runner.subprocess, 'check_output', return_value='GPU-wrong\n'):
                with self.assertRaisesRegex(ValueError, 'UUID differs'):
                    runner.check_idle()

    def test_budget_train_and_eval_share_per_seed_limit(self):
        clock = [0.]
        with mock.patch.object(runner.time, 'monotonic', side_effect=lambda: clock[0]), mock.patch.object(runner.signal, 'setitimer') as alarm:
            budget = runner.Budget(); budget.arm(1)
            clock[0] = 1100; budget.finish()
            self.assertEqual(budget.spent[1], 1100)
            clock[0] = 1400; budget.arm(2)
            clock[0] = 1500; budget.finish()
            budget.arm(1)
            self.assertEqual(alarm.call_args.args[1], 100)
            clock[0] = 1600
            with self.assertRaisesRegex(ValueError, 'seed 20-minute'):
                budget.check()
            clock[0] = 3600
            with self.assertRaisesRegex(ValueError, 'total 60-minute'):
                budget.check()

    def test_strict_json_rejects_duplicate_nonfinite_and_exclusive_write(self):
        for text in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
            with self.assertRaises(ValueError):
                runner.decode(text)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'a.json'; runner.write(path, {'a': 1})
            with self.assertRaises(FileExistsError):
                runner.write(path, {'a': 2})
            self.assertEqual(runner.read(path), {'a': 1})

    def test_actual_prereg_matches_fixed_runner_contract(self):
        candidates = [Path(__file__).resolve().parent / 'e6_no_llm_prereg_v0.json',
                      Path(__file__).resolve().parent.parent / 'config/e6_no_llm_prereg_v0.json']
        path = next((p for p in candidates if p.is_file()), None)
        self.assertIsNotNone(path, 'Frozen prereg fixture is missing')
        cfg = runner.read(path)
        self.assertEqual(runner.sha(path), runner.PLAN_SHA)
        runner.validate_config(cfg)
        altered = copy.deepcopy(cfg); altered['training']['lr'] *= 10
        with self.assertRaisesRegex(ValueError, 'Training settings'):
            runner.validate_config(altered)


if __name__ == '__main__':
    unittest.main(verbosity=2)
