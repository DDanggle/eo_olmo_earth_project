import copy
import unittest

import numpy as np

import e4_delta_probe_v0 as e4


def synthetic(tiles_per_event=1):
    items = []
    for phen, events in (('flood', 10), ('landslide', 2)):
        for event in range(events):
            for tile in range(tiles_per_event if phen == 'flood' else 1):
                for kind in ('pos', 'neg'):
                    items.append({'id': f'{phen}_{event}_{tile}_{kind}', 'tile': f'{phen}_{event}_{tile}',
                                  'cluster': str(event), 'phen': phen, 'kind': kind,
                                  'answer': 'yes' if kind == 'pos' else 'no'})
    items.append({'id': 'hard', 'tile': 'hard_tile', 'cluster': '0', 'phen': 'flood', 'kind': 'hard_neg', 'answer': 'no'})
    rows = []
    for seed in e4.SEEDS:
        for arm in e4.ARMS:
            for item in items:
                rows.append({'seed': seed, 'arm': arm, **{key: item[key] for key in ('id', 'tile', 'cluster', 'phen', 'kind')},
                             'source_gold': item['answer'], 'transformed_gold': None,
                             'parsed': item['answer'], 'answer_raw': item['answer']})
    refs = {name: {str(seed): {item['id']: item['answer'] for item in items} for seed in e4.SEEDS}
            for name in ('e2_real', 'e3_real')}
    return items, rows, refs


class DeltaTransformTests(unittest.TestCase):
    def setUp(self):
        self.pair = np.random.default_rng(7).normal(size=(2, 64, 768)).astype(np.float32)

    def test_delta_only_retains_both_observations_difference_without_raw_blocks(self):
        tokens, types = e4.transform_pair(self.pair, 'delta_only')
        self.assertEqual(tokens.shape, (192, 768))
        self.assertFalse(tokens[:128].any())
        np.testing.assert_array_equal(tokens[128:], self.pair[1] - self.pair[0])
        np.testing.assert_array_equal(types, [0] * 64 + [1] * 64 + [3] * 64)

    def test_sign_flip_does_not_swap_original_observations(self):
        tokens, _ = e4.transform_pair(self.pair, 'delta_sign_flip')
        np.testing.assert_array_equal(tokens[:64], self.pair[0])
        np.testing.assert_array_equal(tokens[64:128], self.pair[1])
        np.testing.assert_array_equal(tokens[128:], -(self.pair[1] - self.pair[0]))

    def test_fixed_permutation_preserves_token_distribution_and_norm(self):
        perm = e4.feature_permutation()
        self.assertEqual(set(perm), set(range(768)))
        self.assertTrue(np.all(perm != np.arange(768)))
        delta = (self.pair[1] - self.pair[0]).astype(np.float64)
        tokens, _ = e4.transform_pair(self.pair, 'delta_feature_permute')
        changed = tokens[128:].astype(np.float64)
        np.testing.assert_array_equal(tokens[:128], self.pair.reshape(128, 768))
        np.testing.assert_array_equal(np.sort(changed, axis=1), np.sort(delta, axis=1))
        for statistic in (lambda x: x.mean(axis=1), lambda x: x.var(axis=1), lambda x: np.linalg.norm(x, axis=1)):
            np.testing.assert_allclose(statistic(changed), statistic(delta), rtol=1e-6, atol=1e-6)
        self.assertFalse(np.array_equal(changed, delta))

    def test_every_arm_is_immutable_and_rejects_nonfinite_or_wrong_shape(self):
        before = self.pair.copy()
        for arm in e4.ARMS:
            tokens, _ = e4.transform_pair(self.pair, arm)
            self.assertEqual(tokens.dtype, self.pair.dtype)
            self.assertFalse(np.shares_memory(tokens, self.pair))
            tokens[:] = 999
            np.testing.assert_array_equal(self.pair, before)
        for pair, arm in ((self.pair[0], 'real'), (self.pair, 'unknown')):
            with self.assertRaises(ValueError):
                e4.transform_pair(pair, arm)
        bad = self.pair.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            e4.transform_pair(bad, 'delta_only')
        bad = np.empty_like(self.pair)
        bad[0], bad[1] = -np.finfo(np.float32).max, np.finfo(np.float32).max
        with self.assertRaises(ValueError):
            e4.transform_pair(bad, 'real')


class DeltaScoringTests(unittest.TestCase):
    def test_two_seed_sufficiency_and_hard_negatives_remain_separate(self):
        items, rows, references = synthetic()
        for row in rows:
            if row['arm'] == 'delta_only' and row['kind'] == 'hard_neg':
                row['parsed'] = row['answer_raw'] = 'yes'
            if row['seed'] == 3 and row['arm'] == 'delta_only' and row['kind'] != 'hard_neg':
                row['parsed'] = row['answer_raw'] = 'no'
        result = e4.score_run(rows, items, references)
        self.assertTrue(result['valid'])
        self.assertEqual(result['sufficient_seeds'], [1, 2])
        self.assertEqual(result['verdict'], 'difference_block_sufficient_under_intervention')
        self.assertEqual(result['metrics']['1']['flood']['hard_negative']['delta_only']['fpr_pooled'], 1.)
        self.assertEqual(result['metrics']['1']['flood']['paired']['delta_only']['macro_ba'], 1.)
        self.assertIsNone(result['metrics']['1']['landslide']['contrasts']['delta_only']['ci95_delta'])

    def test_delta_only_degradation_has_preregistered_point_and_interval(self):
        items, rows, references = synthetic()
        for row in rows:
            if row['arm'] == 'delta_only':
                row['parsed'] = row['answer_raw'] = 'no'
        result = e4.score_run(rows, items, references)
        self.assertTrue(result['valid'])
        self.assertEqual(result['verdict'], 'delta_only_degrades_source_agreement')
        contrast = result['metrics']['1']['flood']['contrasts']['delta_only']
        self.assertEqual(contrast['delta'], -.5)
        self.assertEqual(contrast['ci95_delta'], [-.5, -.5])

    def test_unparsed_output_is_not_counted_as_agreement(self):
        items, rows, references = synthetic(tiles_per_event=10)
        for row in rows:
            if row['seed'] == 1 and row['arm'] == 'delta_only' and row['id'] == 'flood_0_0_pos':
                row['parsed'], row['answer_raw'] = None, 'uncertain'
        result = e4.score_run(rows, items, references)
        self.assertTrue(result['valid'])
        agreement = result['metrics']['1']['flood']['agreement_with_real']['delta_only']
        self.assertEqual(agreement['agreement_all_items'], 200 / 201)
        self.assertEqual(agreement['agreement_both_parsed'], 1.)
        self.assertEqual(agreement['matched_valid_decisions'], 200)

    def test_reference_coverage_reproduction_and_pseudogold_fail_closed(self):
        items, rows, references = synthetic()
        for change in ('coverage', 'reproduction', 'pseudo_gold', 'missing_output'):
            altered_rows, altered_refs = copy.deepcopy(rows), copy.deepcopy(references)
            if change == 'coverage': altered_refs['e3_real']['1'].pop('hard')
            if change == 'reproduction':
                for row in altered_rows:
                    if row['seed'] == 1 and row['arm'] == 'real' and row['id'] == 'flood_0_0_pos':
                        row['parsed'] = row['answer_raw'] = 'no'
            if change == 'pseudo_gold': altered_rows[0]['transformed_gold'] = 'no'
            if change == 'missing_output': altered_rows.pop()
            with self.subTest(change=change):
                result = e4.score_run(altered_rows, items, altered_refs)
                self.assertFalse(result['valid'])
                self.assertEqual(result['verdict'], 'invalid')


if __name__ == '__main__':
    unittest.main()
