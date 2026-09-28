"""Pure E5 tests using invented items and responses; no real E5 outputs."""
import copy
import unittest

import numpy as np

import e5_scoring_v0 as e5


def synthetic():
    items = []

    def append(partition, phen, tile, event, kind, special_date=False):
        dates = ['2020-01-01', '2020-01-13'] if kind != 'neg' else ['2019-12-20', '2020-01-01']
        if special_date:
            dates = ['2021-01-01', '2021-01-13']
        item = {'id': tile + '_' + kind, 'tile': tile, 'phen': phen, 'cluster': event,
                'partition': partition, 'kind': kind, 'answer': 'yes' if kind == 'pos' else 'no', 'dates': dates}
        if phen == 'flood':
            item.update(event=event, slots=['pre_1', 'pre_2'] if kind == 'neg' else ['pre_2', 'post'])
        items.append(item)

    for phen, n_pairs, events in (('flood', 1066, 27), ('landslide', 518, 7)):
        for index in range(n_pairs):
            for kind in ('pos', 'neg'):
                append('train', phen, f'train_{phen}_{index}', f'train_{index % events}', kind)
            if phen == 'flood':
                append('train', phen, f'train_hard_{index}', f'train_{index % events}', 'hard_neg')
    for event, n_pairs in enumerate((56, 56, 56, 56, 56, 55, 55, 55, 6, 6)):
        for index in range(n_pairs):
            for kind in ('pos', 'neg'):
                append('test', 'flood', f'flood_{event}_{index}', str(event), kind,
                       special_date=event == 0 and index == 0 and kind == 'pos')
    for event, n_hard in enumerate((57, 57, 57, 57, 57, 57, 57, 58)):
        for index in range(n_hard):
            append('test', 'flood', f'hard_{event}_{index}', str(event), 'hard_neg',
                   special_date=event == 0 and index == 0)
    for event, n_pairs in enumerate((186, 6)):
        for index in range(n_pairs):
            for kind in ('pos', 'neg'):
                append('test', 'landslide', f'landslide_{event}_{index}', str(event), kind)
    test = [item for item in items if item['partition'] == 'test']
    eval_sets = {
        'all_test': [item['id'] for item in test],
        'primary_same_prompt': [item['id'] for item in test if item['phen'] == 'flood'
                               and item['kind'] in ('pos', 'hard_neg') and int(item['cluster']) < 8],
        'paired_flood': [item['id'] for item in test if item['phen'] == 'flood' and item['kind'] != 'hard_neg'],
        'hard_negative_flood': [item['id'] for item in test if item['kind'] == 'hard_neg'],
        'landslide': [item['id'] for item in test if item['phen'] == 'landslide'],
        'e3_subset': [item['id'] for item in test[:209]],
    }
    rows = []
    indices = {item['id']: index for index, item in enumerate(items)}
    for seed in e5.SEEDS:
        for arm, evaluation in e5.EVALUATIONS:
            for item in test:
                rows.append({'seed': seed, 'model_arm': arm, 'eval_arm': evaluation,
                             **{field: item[field] for field in ('id', 'tile', 'cluster', 'phen', 'kind')},
                             'source_gold': item['answer'], 'transformed_gold': None,
                             'parsed': item['answer'], 'answer_raw': item['answer'].upper() + '.',
                             'pair_index': indices[item['id']]})
    return items, rows, eval_sets


def change(row, value):
    row['parsed'] = value
    row['answer_raw'] = value if value is not None else 'uncertain'


class TransformTests(unittest.TestCase):
    def test_four_formats_have_no_delta_leak_and_do_not_change_input(self):
        pair = np.random.default_rng(7).normal(size=(2, 64, 768)).astype(np.float32)
        before = pair.copy()
        zero = np.zeros_like(pair[0])
        expected = {'full': (pair[0], pair[1], pair[1] - pair[0]),
                    'pair': (pair[0], pair[1], zero), 'later': (zero, pair[1], zero),
                    'delta': (zero, zero, pair[1] - pair[0])}
        for arm, blocks in expected.items():
            tokens, types = e5.transform_pair(pair, arm)
            np.testing.assert_array_equal(tokens, np.concatenate(blocks))
            np.testing.assert_array_equal(types, [0] * 64 + [1] * 64 + [3] * 64)
            self.assertEqual(tokens.dtype, np.float32)
            self.assertEqual(types.dtype, np.int64)
            self.assertFalse(np.shares_memory(tokens, pair))
            tokens[:] = 999
            types[:] = 999
            np.testing.assert_array_equal(pair, before)

    def test_shape_dtype_nonfinite_unknown_and_overflow_rejected(self):
        pair = np.zeros((2, 64, 768), np.float32)
        for wrong in (pair[0], pair.astype(np.float16), pair.astype(np.float64), pair.tolist()):
            with self.assertRaises(ValueError):
                e5.transform_pair(wrong, 'full')
        with self.assertRaises(ValueError):
            e5.transform_pair(pair, 'full_no_delta')
        pair[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            e5.transform_pair(pair, 'pair')
        pair[0], pair[1] = -np.finfo(np.float32).max, np.finfo(np.float32).max
        for arm in ('full', 'delta'):
            with self.assertRaises(ValueError):
                e5.transform_pair(pair, arm)
        tokens, _ = e5.transform_pair(np.zeros_like(pair), 'delta')
        self.assertFalse(tokens.any())


class ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = synthetic()

    def test_exact_full_coverage_and_stratum_then_event_weighting(self):
        items, rows, sets = copy.deepcopy(self.fixture)
        for row in rows:
            if row['model_arm'] == 'pair' and row['id'] in ('flood_0_0_pos', 'hard_0_0_hard_neg'):
                change(row, 'no' if row['kind'] == 'pos' else 'yes')
        result = e5.score_run(rows, items, sets)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['coverage']['received'], 26325)
        primary = result['metrics']['1']['primary_same_prompt']['evaluations']['pair/native']
        self.assertEqual(primary['n_items'], 902)
        self.assertEqual(primary['n_strata'], 9)
        self.assertEqual(primary['events']['0']['ba'], .5)
        self.assertEqual(primary['macro_ba'], 7.5 / 8)
        paired = result['metrics']['1']['paired_source']
        self.assertEqual(paired['flood']['evaluations']['pair/native']['n_items'], 914)
        self.assertEqual(paired['flood']['evaluations']['pair/native']['n_events'], 10)
        self.assertIsNone(paired['landslide']['contrasts']['pair_minus_full']['ci95_delta'])

    def test_primary_same_seed_two_of_three_and_recovery_stays_descriptive(self):
        items, rows, sets = copy.deepcopy(self.fixture)
        for row in rows:
            if row['model_arm'] in ('later', 'delta') or row['eval_arm'] == 'full_no_delta' or (row['model_arm'] == 'pair' and row['seed'] == 3):
                change(row, 'no')
        result = e5.score_run(rows, items, sets)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['pair_preserves_seeds'], [1, 2])
        self.assertEqual(result['explicit_difference_helps_seeds'], [3])
        self.assertEqual(result['verdict'], 'pair_preserves_source_discrimination_at_equal_budget')
        recovery = result['metrics']['1']['primary_same_prompt']['contrasts']['pair_minus_full_no_delta']
        self.assertEqual(recovery['delta'], .5)
        self.assertEqual(recovery['ci95_delta'], [.5, .5])

    def test_full_strength_gate_and_explicit_difference_verdict(self):
        items, rows, sets = copy.deepcopy(self.fixture)
        for row in rows:
            if row['model_arm'] == 'pair':
                change(row, 'no')
        result = e5.score_run(rows, items, sets)
        self.assertEqual(result['verdict'], 'explicit_difference_helps_at_this_budget')
        self.assertEqual(result['explicit_difference_helps_seeds'], [1, 2, 3])
        for row in rows:
            if row['model_arm'] == 'full' and row['eval_arm'] == 'native':
                change(row, 'no')
        result = e5.score_run(rows, items, sets)
        self.assertTrue(result['valid'], result)
        self.assertEqual(result['verdict'], 'mixed_or_inconclusive')
        self.assertTrue(all(not decision['eligible'] for decision in result['seed_decisions'].values()))

    def test_null_negative_counts_incorrect_not_as_true_negative(self):
        items, rows, sets = copy.deepcopy(self.fixture)
        target = next(row for row in rows if row['seed'] == 1 and row['model_arm'] == 'pair' and row['id'] == 'hard_0_0_hard_neg')
        change(target, None)
        result = e5.score_run(rows, items, sets)
        self.assertTrue(result['valid'], result)
        groups = result['metrics']['1']['primary_same_prompt']['evaluations']['pair/native']['events']['0']['strata']
        tiny = next(group for group in groups if group['n_pos'] == 1)
        self.assertEqual(tiny['fpr'], 0.)
        self.assertEqual(tiny['specificity'], 0.)
        self.assertEqual(tiny['ba'], .5)
        self.assertEqual(tiny['parse_failures'], 1)
        hard = result['metrics']['1']['hard_negative']['pair/native']
        self.assertEqual(hard['fpr_pooled'], 0.)
        self.assertEqual(hard['parse_failures'], 1)
        # Four unparsed landslide items fail 4/384 > .01 even with all flood parsed.
        n = 0
        for row in rows:
            if row['seed'] == 1 and row['model_arm'] == 'pair' and row['phen'] == 'landslide' and n < 4:
                change(row, None)
                n += 1
        self.assertFalse(e5.score_run(rows, items, sets)['valid'])

    def test_coverage_and_identity_fail_without_exclusion(self):
        items, original, sets = self.fixture
        changes = {
            'missing': lambda rows: rows.pop(), 'duplicate': lambda rows: rows.append(copy.deepcopy(rows[0])),
            'raw_parse': lambda rows: rows[0].update(answer_raw='yes'),
            'wrong_source_gold': lambda rows: rows[0].update(source_gold='yes'),
            'physical_gold': lambda rows: rows[0].update(transformed_gold='no'),
            'missing_null': lambda rows: rows[0].pop('transformed_gold'),
            'train_id': lambda rows: rows[0].update(id=items[0]['id']),
            'index': lambda rows: rows[0].update(pair_index=0),
            'bool_index': lambda rows: rows[0].update(pair_index=True),
            'wrong_model_eval': lambda rows: rows[0].update(model_arm='later', eval_arm='full_no_delta'),
        }
        # First generated row is a positive example; choose mismatches explicitly.
        changes['raw_parse'] = lambda rows: rows[0].update(answer_raw='no' if rows[0]['parsed'] == 'yes' else 'yes')
        changes['wrong_source_gold'] = lambda rows: rows[0].update(source_gold='no' if rows[0]['source_gold'] == 'yes' else 'yes')
        for name, mutate in changes.items():
            with self.subTest(name=name):
                rows = copy.deepcopy(original)
                mutate(rows)
                self.assertFalse(e5.score_run(rows, items, sets)['valid'])

    def test_membership_and_class_support_not_repaired_posthoc(self):
        items, rows, sets = self.fixture
        wrong = copy.deepcopy(sets)
        wrong['all_test'][0] = items[0]['id']
        self.assertFalse(e5.score_run(rows, items, wrong)['valid'])
        wrong = copy.deepcopy(sets)
        wrong['paired_flood'].pop()
        self.assertFalse(e5.score_run(rows, items, wrong)['valid'])
        wrong_items = copy.deepcopy(items)
        target = next(item for item in wrong_items if item['id'] == 'hard_0_0_hard_neg')
        target['dates'] = ['2099-01-01', '2099-01-13']
        result = e5.score_run(rows, wrong_items, sets)
        self.assertFalse(result['valid'])
        self.assertIn('stratum lacks', result['invalid_reason'])

    def test_hard_fpr_uses_all_457_and_event_macro_is_separate(self):
        items, rows, sets = copy.deepcopy(self.fixture)
        for row in rows:
            if row['model_arm'] == 'later' and row['kind'] == 'hard_neg' and row['cluster'] == '7':
                change(row, 'yes')
        result = e5.score_run(rows, items, sets)
        self.assertTrue(result['valid'], result)
        hard = result['metrics']['1']['hard_negative']['later/native']
        self.assertEqual(hard['n'], 457)
        self.assertEqual(hard['fpr_pooled'], 58 / 457)
        self.assertEqual(hard['fpr_event_macro'], 1 / 8)


if __name__ == '__main__':
    unittest.main()
