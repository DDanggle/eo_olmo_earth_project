"""CPU-only synthetic contracts for the E5 runner. Never loads a model/GPU."""
import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import e5_train_v0 as runner


class RunnerContracts(unittest.TestCase):
    def test_exact_schedule_and_tail_exposure(self):
        ids = [f'i{x}' for x in range(4234)]
        batches = runner.expected_batches(ids, 2)
        self.assertEqual(sum(map(len, batches)), 1590)
        self.assertEqual(sum(len(b) for ep in batches for b in ep), 12702)
        rng = np.random.default_rng(2)
        for ep in batches:
            self.assertEqual(len(ep), 530)
            self.assertEqual(len(ep[-1]), 2)
            self.assertEqual([i for b in ep for i in b], [ids[j] for j in rng.permutation(4234)])
        self.assertNotEqual(batches[0], batches[1])
        self.assertNotEqual(batches, runner.expected_batches(ids, 1))

    def test_common_initial_state_and_serialization_independent_hash(self):
        torch.manual_seed(2)
        canonical = runner.make_projector(torch, 8, .03)
        state = {k: v.clone() for k, v in canonical.state_dict().items()}
        expected = runner.tensor_state_hash(state)
        for arm in runner.ARMS:
            torch.manual_seed(91)
            model = runner.make_projector(torch, 8, .03)
            model.load_state_dict({k: v.clone() for k, v in state.items()}, strict=True)
            self.assertEqual(runner.tensor_state_hash(model.state_dict()), expected, arm)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.pt', Path(tmp) / 'other.pt'
            torch.save(state, a)
            torch.save(state, b)
            self.assertEqual(runner.tensor_state_hash(torch.load(a, weights_only=True)), expected)
            self.assertEqual(runner.tensor_state_hash(torch.load(b, weights_only=True)), expected)
        changed = {k: v.clone() for k, v in state.items()}
        changed['gain'].add_(1)
        self.assertNotEqual(runner.tensor_state_hash(changed), expected)

    def test_projector_exact_original_e2_initialization(self):
        class Original(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = torch.nn.Sequential(torch.nn.Linear(768, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 8))
                self.ttype = torch.nn.Embedding(4, 8)
                self.norm = torch.nn.LayerNorm(768)
                self.out = torch.nn.LayerNorm(8)
                self.gain = torch.nn.Parameter(torch.tensor(1.0))
                torch.nn.init.normal_(self.ttype.weight, std=.02)
        torch.manual_seed(3)
        expected = Original()
        torch.manual_seed(3)
        observed = runner.make_projector(torch, 8, .02)
        self.assertEqual(runner.tensor_state_hash(expected.state_dict()), runner.tensor_state_hash(observed.state_dict()))
        values = torch.randn(192, 768)
        types = torch.tensor([0] * 64 + [1] * 64 + [3] * 64)
        reference = expected.out(expected.mlp(expected.norm(values))) * (.02 * expected.gain) + expected.ttype(types) * .02
        self.assertTrue(torch.equal(observed(values, types), reference))

    def test_eval_never_embeds_source_answer_and_uses_masked_labels(self):
        calls = []
        def emb(ids):
            calls.append(ids.tolist())
            return ids.float()[:, None].repeat(1, 4)
        ids = [torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([99, 100])]
        projected = torch.zeros(192, 4)
        e, y = runner.assemble_sequence(torch, emb, ids, projected, False, 'cpu')
        self.assertEqual(calls, [[1, 2], [3]])
        self.assertEqual(e.shape, (195, 4))
        self.assertTrue(torch.all(y == -100))
        self.assertFalse(torch.any(e == 99))
        calls.clear()
        e, y = runner.assemble_sequence(torch, emb, ids, projected, True, 'cpu')
        self.assertEqual(calls, [[1, 2], [3], [99, 100]])
        self.assertEqual(y[-2:].tolist(), [99, 100])
        self.assertTrue(torch.all(y[:-2] == -100))
        self.assertEqual(e.shape, (197, 4))

    def test_right_padded_copy_retains_projected_input_gradients(self):
        projected = torch.randn(192, 4, requires_grad=True)
        def emb(ids):
            return torch.ones(len(ids), 4)
        ids = [torch.tensor([1]), torch.tensor([2, 3]), torch.tensor([4, 5])]
        e, _ = runner.assemble_sequence(torch, emb, ids, projected, True, 'cpu')
        batch = torch.zeros(1, len(e) + 2, 4)
        mask = torch.zeros(1, len(e) + 2, dtype=torch.long)
        batch[0, :len(e)] = e
        mask[0, :len(e)] = 1
        (batch.square() * mask[..., None]).sum().backward()
        self.assertTrue(torch.equal(projected.grad, 2 * projected.detach()))
        self.assertEqual(mask[0, -2:].tolist(), [0, 0])

    def test_row_contract_source_gold_and_intervention_role(self):
        item = {'id': 'tile_pos', 'tile': 'tile', 'cluster': 'evt', 'phen': 'flood', 'kind': 'pos', 'answer': 'yes'}
        row = runner.output_row(1, 'full', 'full_no_delta', item, 712, 'No.')
        self.assertEqual(row['parsed'], 'no')
        self.assertEqual(row['source_gold'], 'yes')
        self.assertIsNone(row['transformed_gold'])
        self.assertEqual(row['pair_index'], 712)
        self.assertIsNone(runner.output_row(1, 'later', 'native', item, 712, 'unclear')['parsed'])
        with self.assertRaises(ValueError):
            runner.output_row(1, 'later', 'full_no_delta', item, 712, 'yes')
        with self.assertRaises(ValueError):
            runner.output_row(True, 'full', 'native', item, 712, 'yes')

    def test_source_prompt_has_no_ids_or_labels(self):
        item = {'phen': 'flood', 'dates': ['2020-01-01', '2020-01-13'], 'tile': 'SECRET_TILE',
                'id': 'SECRET_ID', 'cluster': 'SECRET_CLUSTER', 'answer': 'SECRET_LABEL'}
        text = runner.source_prompt(item)
        self.assertNotIn('SECRET', text)
        self.assertEqual(text, 'These are 2 Sentinel-1 observations of the same area in chronological order, taken on 2020-01-01, 2020-01-13: <EO> Did a flood occur between the two observations? Answer with yes or no.')

    def test_actual_plan_recipe_and_changed_budget_rejected(self):
        cfg = runner.read(Path(__file__).parent / 'e5_equal_budget_prereg_v0.json')
        runner.validate_training_config(cfg)
        for section, key, value in [('training', 'epochs', 4), ('training', 'lr', .001),
                                    ('evaluation', 'n_answers', 21060), ('validity', 'max_parse_fail', .1)]:
            changed = copy.deepcopy(cfg)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                runner.validate_training_config(changed)

    def test_relative_file_cannot_escape_prepared_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'input.json').write_text('{}')
            self.assertEqual(runner.relative_file(root, 'input.json'), (root / 'input.json').resolve())
            for name in ('../elsewhere', '/absolute', 'missing'):
                with self.assertRaises(ValueError):
                    runner.relative_file(root, name)

    def test_lock_identity_and_lock_ownership_cpu_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptors = {}
            try:
                for name in runner.LOCK_NAMES:
                    fd = os.open(root / name, os.O_RDWR | os.O_CREAT, 0o600)
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    descriptors[name] = fd
                with patch.dict(os.environ, {'E5_LOCK_FDS': json.dumps(descriptors)}):
                    self.assertEqual(runner.check_locks(root), descriptors)
                    selected = next(iter(descriptors))
                    fcntl.flock(descriptors[selected], fcntl.LOCK_UN)
                    with self.assertRaises(ValueError):
                        runner.check_locks(root)
            finally:
                for fd in descriptors.values():
                    os.close(fd)

    def test_time_caps_are_model_and_total_not_per_epoch(self):
        with patch.object(runner.time, 'monotonic', return_value=100):
            budget = runner.Budget()
        budget.model_started = 200
        with patch.object(runner.time, 'monotonic', return_value=2901):
            with self.assertRaisesRegex(ValueError, 'Model'):
                budget.check()
        budget.model_started = None
        with patch.object(runner.time, 'monotonic', return_value=14501):
            with self.assertRaisesRegex(ValueError, 'Overall'):
                budget.check()

    def test_full_population_sequence_and_batch_mutation_guards(self):
        items = []
        for part, nf, nl in [('train', 1066, 518), ('test', 457, 192)]:
            for phen, n, nc in [('flood', nf, 27 if part == 'train' else 10),
                                ('landslide', nl, 7 if part == 'train' else 2)]:
                for kind in (['pos', 'neg', 'hard_neg'] if phen == 'flood' else ['pos', 'neg']):
                    for k in range(n):
                        tile = f'{part}_{phen}_{"hard" if kind == "hard_neg" else "pair"}_{k}'
                        items.append({'id': f'{tile}_{kind}', 'tile': tile, 'partition': part, 'phen': phen,
                                      'kind': kind, 'cluster': f'{part}_{k % nc}',
                                      'answer': 'yes' if kind == 'pos' else 'no', 'dates': ['2020-01-01', '2020-01-13']})
        items.sort(key=lambda x: (x['partition'], x['phen'], x['id']))
        ordered = {p: [i['id'] for i in items if i['partition'] == p] for p in ['train', 'test']}
        batches = {str(s): runner.expected_batches(ordered['train'], s) for s in (1, 2, 3)}
        pairs = np.broadcast_to(np.zeros((2, 64, 768), dtype=np.float32), (5989, 2, 64, 768))
        lookup = runner.validate_population(items, ordered, batches, pairs)
        self.assertEqual(lookup[items[5000]['id']], 5000)
        changed = copy.deepcopy(ordered)
        changed['train'][0], changed['train'][1] = changed['train'][1], changed['train'][0]
        with self.assertRaisesRegex(ValueError, 'Ordered sequence'):
            runner.validate_population(items, changed, batches, pairs)
        mutated = copy.deepcopy(batches)
        mutated['1'][0][0][0] = mutated['1'][0][0][1]
        with self.assertRaisesRegex(ValueError, 'Batch order'):
            runner.validate_population(items, ordered, mutated, pairs)

    def test_snapshot_and_llm_file_membership_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp).resolve()
            snap, llm = out / 'code_snapshot', out / 'llm'
            snap.mkdir()
            llm.mkdir()
            (out / 'parent_snapshot').mkdir()
            (out / 'parent_snapshot' / 'source.json').write_text('{}')
            for name in runner.REQUIRED_FILES:
                (out / name).write_text('{}')
            for name in ['e5_train_v0.py', 'e5_scoring_v0.py']:
                (snap / name).write_text('# frozen source\n')
            (llm / 'config.json').write_text('{}')
            (llm / 'model.safetensors').write_bytes(b'fixture')
            (llm / 'README.md').write_text('Nonexecuted model card')
            manifest = {'schema': 'e5-equal-budget-prepared-v0', 'n_items': 5989, 'n_train': 4234, 'n_test': 1755,
                        'llm_dir': str(llm), 'files_sha256': {name: runner.sha(out / name) for name in runner.REQUIRED_FILES},
                        'parent_snapshot_sha256': {'source.json': runner.sha(out / 'parent_snapshot' / 'source.json')},
                        'code_snapshot_sha256': {name: runner.sha(snap / name) for name in ['e5_train_v0.py', 'e5_scoring_v0.py']},
                        'llm_files_sha256': {str(p): runner.sha(p) for p in [llm / 'config.json', llm / 'model.safetensors']}}
            with patch.object(runner, '__file__', str(snap / 'e5_train_v0.py')):
                self.assertEqual(runner.verify_files(out, manifest), llm)
                (llm / 'tokenizer.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'file membership'):
                    runner.verify_files(out, manifest)
                (llm / 'tokenizer.json').unlink()
                (out / 'ordered_ids.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                    runner.verify_files(out, manifest)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
