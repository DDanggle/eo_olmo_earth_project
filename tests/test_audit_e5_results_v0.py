"""Synthetic checks for the independent E5 auditor; no production imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import audit_e5_results_v0 as audit


def primary_fixture():
    items = []
    for event, n_pos, n_neg in zip(range(8), (56, 56, 56, 56, 56, 55, 55, 55), (57, 57, 57, 57, 57, 57, 57, 58)):
        for kind, n in (('pos', n_pos), ('hard_neg', n_neg)):
            for index in range(n):
                dates = ['2020-01-01', '2020-01-13']
                if event == 0 and index == 0:
                    dates = ['2021-01-01', '2021-01-13']
                items.append({'id': f'{event}_{kind}_{index}', 'cluster': str(event), 'dates': dates,
                              'slots': ['pre_2', 'post'], 'kind': kind, 'answer': 'yes' if kind == 'pos' else 'no'})
    indexed = {(seed, arm, evaluation, item['id']): {'parsed': item['answer']}
               for seed in audit.SEEDS for arm, evaluation in audit.EVALUATIONS for item in items}
    return items, indexed


def full_prediction_fixture():
    # Metadata-only fixture for the entire 26325 expected output keys.
    items = [{'id': f'train_{i}', 'partition': 'train', 'tile': f'train_{i}', 'cluster': 'train',
              'phen': 'flood', 'kind': 'pos', 'answer': 'yes'} for i in range(4234)]
    for phen, n_pos, n_neg, n_hard in (('flood', 457, 457, 457), ('landslide', 192, 192, 0)):
        for kind, count in (('pos', n_pos), ('neg', n_neg), ('hard_neg', n_hard)):
            for index in range(count):
                items.append({'id': f'{phen}_{kind}_{index}', 'tile': f'{phen}_{kind}_{index}', 'cluster': 'test',
                              'partition': 'test', 'phen': phen, 'kind': kind, 'answer': 'yes' if kind == 'pos' else 'no'})
    rows = []
    for seed in audit.SEEDS:
        for arm, evaluation in audit.EVALUATIONS:
            for index in range(4234, len(items)):
                item = items[index]
                rows.append({'seed': seed, 'model_arm': arm, 'eval_arm': evaluation,
                    **{field: item[field] for field in ('id', 'tile', 'cluster', 'phen', 'kind')},
                    'source_gold': item['answer'], 'transformed_gold': None,
                    'answer_raw': item['answer'].upper() + '.', 'parsed': item['answer'], 'pair_index': index})
    return items, rows


def step_fixture():
    ids = [f'train_{index}' for index in range(4234)]
    schedule = audit.expected_schedule(ids, 2)
    rows, exposure = [], 0
    for epoch, batches in enumerate(schedule):
        for batch_index, batch in enumerate(batches):
            exposure += len(batch)
            step = len(rows) + 1
            rows.append({'step': step, 'epoch': epoch, 'batch_index': batch_index,
                         'ids': batch, 'batch_size': len(batch), 'exposures': exposure,
                         'loss': float(epoch + 1), 'elapsed_s': step / 10,
                         'finite_loss_grad_parameters': True})
    return schedule, rows


class PrimaryTests(unittest.TestCase):
    def test_stratum_then_event_balance_and_conservative_null_negative(self):
        primary, indexed = primary_fixture()
        for seed in audit.SEEDS:
            indexed[(seed, 'pair', 'native', '0_pos_0')]['parsed'] = 'no'
            indexed[(seed, 'pair', 'native', '0_hard_neg_0')]['parsed'] = None
        result = audit.primary_metrics(primary, indexed)
        pair = result['metrics']['1']['evaluations']['pair/native']
        self.assertEqual(pair['n_items'], 902)
        self.assertEqual(pair['n_strata'], 9)
        self.assertEqual(pair['events']['0']['ba'], .5)
        self.assertEqual(pair['macro_ba'], 7.5 / 8)
        tiny = next(group for group in pair['events']['0']['strata'] if group['n_pos'] == 1)
        self.assertEqual(tiny['ba'], 0.)
        self.assertEqual(tiny['fpr'], 0.)
        self.assertEqual(tiny['specificity'], 0.)
        self.assertEqual(tiny['parse_failures'], 1)

    def test_two_seed_primary_rule_and_descriptive_recovery(self):
        primary, indexed = primary_fixture()
        for key, row in indexed.items():
            seed, arm, evaluation, item_id = key
            if arm in ('later', 'delta') or evaluation == 'full_no_delta' or (seed == 3 and arm == 'pair'):
                row['parsed'] = 'no'
        result = audit.primary_metrics(primary, indexed)
        self.assertEqual(result['verdict'], 'pair_preserves_source_discrimination_at_equal_budget')
        self.assertEqual(result['pair_preserves_seeds'], [1, 2])
        self.assertEqual(result['explicit_difference_helps_seeds'], [3])
        self.assertEqual(result['metrics']['1']['contrasts']['pair_minus_full_no_delta']['ci95_delta'], [.5, .5])
        for key, row in indexed.items():
            if key[1] == 'pair':
                row['parsed'] = 'no'
        result = audit.primary_metrics(primary, indexed)
        self.assertEqual(result['verdict'], 'explicit_difference_helps_at_this_budget')
        for key, row in indexed.items():
            if key[1:3] == ('full', 'native'):
                row['parsed'] = 'no'
        self.assertEqual(audit.primary_metrics(primary, indexed)['verdict'], 'mixed_or_inconclusive')

    def test_bootstrap_hand_computable_constant_and_two_event_extremes(self):
        self.assertEqual(audit.event_interval([-.5] * 8), [-.5, -.5])
        self.assertEqual(audit.event_interval([-.5, .5]), [-.5, .5])
        with self.assertRaises(ValueError):
            audit.event_interval([np.nan, 1])


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prediction_fixture = full_prediction_fixture()

    def test_exact_prediction_coverage_raw_gold_index_and_train_rejection(self):
        items, original = self.prediction_fixture
        indexed, rates = audit.index_predictions(items, original)
        self.assertEqual(len(indexed), 26325)
        self.assertEqual(len(rates), 30)
        mutations = {
            'missing': lambda rows: rows.pop(),
            'duplicate': lambda rows: rows.append(copy.deepcopy(rows[0])),
            'raw': lambda rows: rows[0].update(answer_raw='NO, yes appears second'),
            'gold': lambda rows: rows[0].update(source_gold='no'),
            'pseudo': lambda rows: rows[0].update(transformed_gold='yes'),
            'missing_null': lambda rows: rows[0].pop('transformed_gold'),
            'global_index': lambda rows: rows[0].update(pair_index=0),
            'boolean_index': lambda rows: rows[0].update(pair_index=True),
            'train': lambda rows: rows[0].update(id='train_0'),
            'wrong_intervention': lambda rows: rows[0].update(model_arm='delta', eval_arm='full_no_delta'),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                rows = copy.deepcopy(original)
                mutate(rows)
                with self.assertRaises(ValueError):
                    audit.index_predictions(items, rows)

    def test_parse_gate_is_per_seed_model_evaluation_phenomenon(self):
        items, rows = copy.deepcopy(self.prediction_fixture)
        chosen = [row for row in rows if row['seed'] == 1 and row['model_arm'] == 'pair' and row['phen'] == 'landslide']
        for row in chosen[:3]:
            row.update(answer_raw='uncertain', parsed=None)
        _, rates = audit.index_predictions(items, rows)
        self.assertEqual(rates['1|pair|native|landslide']['failed'], 3)
        chosen[3].update(answer_raw='uncertain', parsed=None)
        with self.assertRaisesRegex(ValueError, 'Parse failure'):
            audit.index_predictions(items, rows)

    def test_secondary_paired_event_ba_hard_fpr_and_land_no_interval(self):
        items, rows = copy.deepcopy(self.prediction_fixture)
        source = {item['id']: item for item in items}
        for item in items[4234:]:
            n_events = 2 if item['phen'] == 'landslide' else 8 if item['kind'] == 'hard_neg' else 10
            item['cluster'] = str(int(item['id'].split('_')[-1]) % n_events)
        for row in rows:
            row['cluster'] = source[row['id']]['cluster']
            if row['seed'] == 1 and row['model_arm'] == 'pair' and row['kind'] == 'hard_neg' and row['cluster'] == '0':
                row.update(parsed='yes', answer_raw='yes')
            if row['seed'] == 1 and row['model_arm'] == 'pair' and row['id'] == 'flood_neg_0':
                row.update(parsed=None, answer_raw='uncertain')
        indexed, _ = audit.index_predictions(items, rows)
        result = audit.secondary_metrics(items[4234:], indexed)['1']
        hard = result['hard_negative']['pair/native']
        self.assertEqual(hard['n'], 457)
        self.assertEqual(hard['fpr_event_macro'], 1 / 8)
        self.assertEqual(hard['fpr_pooled'], 58 / 457)
        event = result['paired_source']['flood']['evaluations']['pair/native']['events']['0']
        self.assertEqual(event['n_negative'], 46)
        self.assertEqual(event['specificity'], 45 / 46)
        self.assertEqual(event['fpr'], 0.)
        self.assertEqual(event['parse_failures'], 1)
        self.assertIsNone(result['paired_source']['landslide']['contrasts']['pair_minus_full']['ci95_delta'])

    def test_exact_schedule_steps_finalbatch_and_exposure_with_tamper_rejection(self):
        schedule, original = step_fixture()
        self.assertEqual([len(epoch) for epoch in schedule], [530, 530, 530])
        self.assertEqual([len(epoch[-1]) for epoch in schedule], [2, 2, 2])
        result = audit.audit_steps(original, schedule)
        self.assertEqual(result['epoch_mean_batch_loss'], [1., 2., 3.])
        self.assertEqual(result['exposures'], 12702)
        mutations = {'step': lambda rows: rows[0].update(step=0),
                     'exposure': lambda rows: rows[-1].update(exposures=12701),
                     'batch': lambda rows: rows[0]['ids'].reverse(),
                     'nonfinite': lambda rows: rows[0].update(loss=float('inf')),
                     'finiteflag': lambda rows: rows[0].update(finite_loss_grad_parameters=False),
                     'time_reverse': lambda rows: rows[-1].update(elapsed_s=0),
                     'missing': lambda rows: rows.pop()}
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                rows = copy.deepcopy(original)
                mutate(rows)
                with self.assertRaises(ValueError):
                    audit.audit_steps(rows, schedule)

    def test_hash_and_json_fail_closed(self):
        for raw in ('{"x":1,"x":2}', '{"loss":NaN}'):
            with self.assertRaises(ValueError):
                audit.loads(raw)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'checkpoint.pt'
            path.write_bytes(b'initial')
            expected = audit.sha(path)
            tracked = {}
            audit.verify(root, 'checkpoint.pt', expected, tracked)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                audit.verify(root, 'checkpoint.pt', expected, tracked)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                audit.verify(root, '../checkpoint.pt', expected, tracked)

    def test_twelve_completed_models_common_init_and_schedule_linkage(self):
        items, predictions = self.prediction_fixture
        indexed, _ = audit.index_predictions(items, predictions)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'initial_states').mkdir()
            (root / 'models').mkdir()

            def write(relative, value, jsonl=False):
                path = root / relative
                if jsonl:
                    path.write_text(''.join(json.dumps(row) + '\n' for row in value))
                else:
                    path.write_text(json.dumps(value))
                return audit.sha(path)

            manifest = {'files_sha256': {'ordered_ids.json': 'd' * 64, 'batches.json': 'e' * 64}}
            init_records, outcomes, batches = {}, [], {}
            for seed in audit.SEEDS:
                initial_file = root / f'initial_states/seed{seed}.pt'
                initial_file.write_bytes(f'CPU-fixture-initial-{seed}'.encode())
                init = {'seed': seed, 'file_sha256': audit.sha(initial_file), 'tensor_sha256': str(seed) * 64, 'parameter_count': 1}
                init_records[str(seed)] = init
                write(f'initial_states/seed{seed}.json', init)
                batches[str(seed)] = audit.expected_schedule([f'train_{i}' for i in range(4234)], seed)
                records, exposures = [], 0
                for epoch, epoch_batches in enumerate(batches[str(seed)]):
                    for batch_index, ids in enumerate(epoch_batches):
                        exposures += len(ids)
                        records.append({'step': len(records) + 1, 'epoch': epoch, 'batch_index': batch_index,
                            'ids': ids, 'batch_size': len(ids), 'exposures': exposures, 'loss': float(epoch + 1),
                            'elapsed_s': (len(records) + 1) / 10, 'finite_loss_grad_parameters': True})
                for arm in audit.ARMS:
                    relative = f'models/seed{seed}_{arm}'
                    directory = root / relative
                    directory.mkdir()
                    (directory / 'initial.pt').write_bytes(initial_file.read_bytes())
                    (directory / 'projector.pt').write_bytes(f'CPU-fixture-final-{seed}-{arm}'.encode())
                    contract = {'seed': seed, 'model_arm': arm, 'initial_file_sha256': init['file_sha256'],
                        'initial_tensor_sha256': init['tensor_sha256'], 'epochs': 3, 'batch_size': 8,
                        'expected_updates': 1590, 'expected_exposures': 12702,
                        'ordered_ids_sha256': 'd' * 64, 'batches_sha256': 'e' * 64,
                        'optimizer_groups': [{'lr': .0001, 'weight_decay': .01, 'betas': [.9, .999], 'eps': 1e-8, 'maximize': False}]}
                    write(relative + '/training_contract.json', contract)
                    training = {'seed': seed, 'model_arm': arm, 'updates': 1590, 'exposures': 12702,
                        'initial_file_sha256': init['file_sha256'], 'initial_tensor_sha256': init['tensor_sha256'],
                        'checkpoint_sha256': audit.sha(directory / 'projector.pt'), 'checkpoint_tensor_sha256': 'a' * 64,
                        'steps_sha256': write(relative + '/steps.jsonl', records, True),
                        'epoch_mean_batch_loss': [1., 2., 3.], 'train_s': 180.}
                    write(relative + '/training_completed.json', training)
                    answers, parse = {}, {}
                    for evaluation in ('native', 'full_no_delta') if arm == 'full' else ('native',):
                        selected = [row for row in predictions if row['seed'] == seed and row['model_arm'] == arm and row['eval_arm'] == evaluation]
                        answers[evaluation] = write(relative + f'/answers_{evaluation}.jsonl', selected, True)
                        parse[evaluation] = {phen: {'n': n, 'unparsed': 0, 'rate': 0.}
                                             for phen, n in (('flood', 1371), ('landslide', 384))}
                    write(relative + '/parse_audit.json', parse)
                    outcome = {**training, 'eval_s': 40., 'model_elapsed_s': 230., 'answers_sha256': answers, 'parse': parse}
                    outcomes.append(outcome)
                    write(relative + '/completed.json', outcome)
            write('initial_states/manifest.json', {'all_generated_before_any_training': True, 'seeds': init_records, 'at': 'synthetic'})
            write('partial_models.json', outcomes)
            actual = audit.audit_training(root, manifest, batches, indexed, {}, check_tensors=False)
            self.assertEqual(actual, outcomes)
            (root / 'models/seed2_pair/initial.pt').write_bytes(b'different condition-specific initial state')
            with self.assertRaisesRegex(ValueError, 'SHA differs'):
                audit.audit_training(root, manifest, batches, indexed, {}, check_tensors=False)

    def test_input_audit_all_3756_cache_pins_and_pool_reproduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'parent_snapshot').mkdir()
            items = [{'cache_path': f'/synthetic/cache_{index % 3756}.npy', 'indices': [0, 1]} for index in range(5989)]
            caches = {}
            for index, item in enumerate(items):
                caches.setdefault(item['cache_path'], {'sha256': 'a' * 64,
                    'before': {'size': 128, 'mtime_ns': 123}, 'after': {'size': 128, 'mtime_ns': 123},
                    'item_rows': [], 'indices_used': [0, 1], 'finite_scope': 'all referenced frames'})['item_rows'].append(index)
            c0 = {'cache_sha256': {path: value['sha256'] for path, value in caches.items()},
                  'cache_stat': {path: value['before'] for path, value in caches.items()}}
            plan = {'pool_reproduction': {'items_sha256': 'b' * 64, 'pairs_sha256': 'c' * 64}}
            sets = {'primary_same_prompt': list(range(902)), 'all_test': list(range(1755))}
            record = {'no_exclusions': True, 'n_items': 5989, 'n_unique_caches': 3756,
                'eval_set_counts': {key: len(ids) for key, ids in sets.items()}, 'cache_sources': caches,
                'pool_reproduction': {'expected_items': 209, 'matched_items': 209, 'bit_exact': True,
                                     'items_sha256': 'b' * 64, 'pairs_sha256': 'c' * 64}}
            (root / 'parent_snapshot/c0_manifest.json').write_text(json.dumps(c0))
            path = root / 'input_audit.json'
            path.write_text(json.dumps(record))
            audit.validate_input_audit(root, plan, items, sets)
            for mutation in ('pin', 'pool', 'exclusion'):
                with self.subTest(mutation=mutation):
                    changed = copy.deepcopy(record)
                    if mutation == 'pin':
                        changed['cache_sources']['/synthetic/cache_0.npy']['sha256'] = '0' * 64
                    elif mutation == 'pool':
                        changed['pool_reproduction']['matched_items'] = 208
                    else:
                        changed['no_exclusions'] = False
                    path.write_text(json.dumps(changed))
                    with self.assertRaises(ValueError):
                        audit.validate_input_audit(root, plan, items, sets)

    def test_optional_cpu_tensor_hash_is_filename_independent_and_finite(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional CPU torch is unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = {'gain': torch.tensor(1., dtype=torch.float32),
                     'weight': torch.arange(6, dtype=torch.float32).reshape(2, 3)}
            torch.save(state, root / 'first.pt')
            torch.save(state, root / 'second.pt')
            first = audit.tensor_hash(root / 'first.pt')
            second = audit.tensor_hash(root / 'second.pt')
            self.assertEqual(first, second)
            self.assertEqual(first[1], 7)
            state['gain'] = torch.tensor(float('nan'))
            torch.save(state, root / 'bad.pt')
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                audit.tensor_hash(root / 'bad.pt')


if __name__ == '__main__':
    unittest.main()
