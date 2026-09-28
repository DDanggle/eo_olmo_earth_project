"""Synthetic independent-auditor tests; never load experimental result files."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import audit_e4_delta_results_v0 as audit


def synthetic():
    items = []
    flood_sizes = (3, 8, 8, 5, 8, 8, 1, 8, 2, 6)
    for phen, sizes in (('flood', flood_sizes), ('landslide', (16, 6))):
        for event, size in enumerate(sizes):
            for tile in range(size):
                for kind in ('pos', 'neg'):
                    items.append({'id': f'{phen}_{event}_{tile}_{kind}', 'tile': f'{phen}_{event}_{tile}',
                                  'cluster': str(event), 'phen': phen, 'kind': kind,
                                  'answer': 'yes' if kind == 'pos' else 'no', 'pair_key': 'p' + str(len(items)),
                                  'allowed_arms': ['old_E3_provenance_is_ignored']})
    for event, count in ((1, 8), (2, 8), (3, 2), (4, 8), (5, 8), (6, 7), (7, 8), (9, 2)):
        for tile in range(count):
            items.append({'id': f'hard_{event}_{tile}', 'tile': f'hard_{event}_{tile}', 'cluster': str(event),
                          'phen': 'flood', 'kind': 'hard_neg', 'answer': 'no', 'pair_key': 'p' + str(len(items)),
                          'allowed_arms': ['real']})
    rows = []
    for seed in audit.SEEDS:
        for arm in audit.ARMS:
            for item in items:
                rows.append({'seed': seed, 'arm': arm,
                             **{field: item[field] for field in ('id', 'tile', 'cluster', 'phen', 'kind', 'pair_key')},
                             'source_gold': item['answer'], 'transformed_gold': None,
                             'parsed': item['answer'], 'answer_raw': '  ' + item['answer'].upper() + '. '})
    references = {name: {str(seed): {item['id']: item['answer'] for item in items} for seed in audit.SEEDS}
                  for name in audit.REFERENCE_NAMES}
    return items, rows, references


def set_prediction(row, prediction):
    row['parsed'] = prediction
    row['answer_raw'] = prediction if prediction is not None else 'uncertain'


class IndependentMetricsTests(unittest.TestCase):
    def test_exact_209_by_four_by_three_and_equal_event_not_tile_weight(self):
        items, rows, refs = synthetic()
        audit.validate_items(items, frozen_population=True)
        for row in rows:
            if row['seed'] == 1 and row['arm'] == 'delta_sign_flip' and row['phen'] == 'flood':
                if row['cluster'] == '0':
                    set_prediction(row, 'no' if row['kind'] == 'pos' else 'yes')
                elif row['kind'] == 'hard_neg' and row['cluster'] == '1':
                    set_prediction(row, 'yes')
        result = audit.recompute(items, rows, refs)
        self.assertEqual(result['coverage'], {'expected': 2508, 'received': 2508})
        flood = result['metrics']['1']['flood']
        self.assertAlmostEqual(flood['paired']['delta_sign_flip']['macro_ba'], .9)
        self.assertNotAlmostEqual(flood['paired']['delta_sign_flip']['macro_ba'], 54 / 57)
        hard = flood['hard_negative']['delta_sign_flip']
        self.assertEqual(hard['n'], 51)
        self.assertAlmostEqual(hard['fpr_event_macro'], 1 / 8)
        self.assertAlmostEqual(hard['fpr_pooled'], 8 / 51)
        self.assertIsNone(result['metrics']['1']['landslide']['contrasts']['delta_only']['ci95_delta'])

    def test_same_seed_sufficiency_ignores_secondary_arm_failure(self):
        items, rows, refs = synthetic()
        for row in rows:
            if row['arm'] in ('delta_sign_flip', 'delta_feature_permute') or (row['seed'] == 3 and row['arm'] == 'delta_only'):
                set_prediction(row, 'no')
        result = audit.recompute(items, rows, refs)
        self.assertEqual(result['sufficient_seeds'], [1, 2])
        self.assertEqual(result['degraded_seeds'], [3])
        self.assertEqual(result['verdict'], 'difference_block_sufficient_under_intervention')
        self.assertEqual(result['metrics']['3']['flood']['contrasts']['delta_only']['ci95_delta'], [-.5, -.5])
        for row in rows:
            if row['arm'] == 'delta_only':
                set_prediction(row, 'no')
        result = audit.recompute(items, rows, refs)
        self.assertEqual(result['verdict'], 'delta_only_degrades_source_agreement')
        self.assertEqual(result['degraded_seeds'], [1, 2, 3])

    def test_null_null_not_agreement_and_parse_threshold_is_per_phenomenon(self):
        items, rows, refs = synthetic()
        for row in rows:
            if row['seed'] == 1 and row['arm'] in ('real', 'delta_only') and row['id'] == 'flood_0_0_pos':
                set_prediction(row, None)
        result = audit.recompute(items, rows, refs)
        agreement = result['metrics']['1']['flood']['agreement_with_real']['delta_only']
        self.assertEqual(agreement['matched_valid_decisions'], 164)
        self.assertAlmostEqual(agreement['agreement_all_items'], 164 / 165)
        self.assertEqual(agreement['agreement_both_parsed'], 1.)
        self.assertAlmostEqual(result['reproduction']['1']['flood']['e2_real']['reproduction_rate'], 164 / 165)
        target = next(row for row in rows if row['seed'] == 1 and row['arm'] == 'delta_only' and row['id'] == 'landslide_0_0_pos')
        set_prediction(target, None)
        with self.assertRaisesRegex(ValueError, 'Parse-failure'):
            audit.recompute(items, rows, refs)

    def test_coverage_raw_parsing_gold_and_metadata_fail_closed(self):
        items, rows, refs = synthetic()
        mutations = {
            'missing': lambda changed: changed.pop(),
            'duplicate': lambda changed: changed.append(copy.deepcopy(changed[0])),
            'unknown': lambda changed: changed[0].update(arm='reverse'),
            'raw': lambda changed: changed[0].update(answer_raw='NO, yes appears only later'),
            'physical_gold': lambda changed: changed[0].update(transformed_gold='yes'),
            'missing_null': lambda changed: changed[0].pop('transformed_gold'),
            'source_gold': lambda changed: changed[0].update(source_gold='no'),
            'pair_key': lambda changed: changed[0].update(pair_key='wrong'),
            'bool_seed': lambda changed: changed[0].update(seed=True),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                changed = copy.deepcopy(rows)
                mutate(changed)
                with self.assertRaises(ValueError):
                    audit.recompute(items, changed, refs)

    def test_reference_reproduction_and_support_are_checked_separately(self):
        items, rows, refs = synthetic()
        for name in audit.REFERENCE_NAMES:
            changed = copy.deepcopy(refs)
            changed[name]['2']['landslide_0_0_pos'] = 'no'
            with self.assertRaisesRegex(ValueError, 'reproduction'):
                audit.recompute(items, rows, changed)
        changed = copy.deepcopy(refs)
        changed['e3_real']['1'].pop(items[0]['id'])
        with self.assertRaisesRegex(ValueError, 'support'):
            audit.recompute(items, rows, changed)
        altered_items = copy.deepcopy(items)
        altered_items[1]['cluster'] = 'elsewhere'
        with self.assertRaisesRegex(ValueError, 'paired tile'):
            audit.validate_items(altered_items)

    def test_bootstrap_paired_constant_and_hand_calculated_two_event_extremes(self):
        self.assertEqual(audit.event_interval([-.1] * 10), [-.1, -.1])
        self.assertEqual(audit.event_interval([-.5, .5]), [-.5, .5])
        first = audit.event_interval([-.5, -.25, 0, .25, .5])
        self.assertEqual(first, audit.event_interval([-.5, -.25, 0, .25, .5]))
        self.assertLessEqual(first[0], 0)
        self.assertGreaterEqual(first[1], 0)


class FrozenFileTests(unittest.TestCase):
    def test_pair_archive_shape_nonfinite_difference_overflow_and_missing_key(self):
        item = {'id': 'x', 'pair_key': 'p0'}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pairs.npz'
            good = np.zeros((2, 64, 768), np.float32)
            good[1, :, :] = np.arange(768)
            np.savez_compressed(path, p0=good)
            audit.validate_pair_archive(path, [item])
            bad_nan = good.copy()
            bad_nan[0, 0, 0] = np.nan
            overflow = good.copy()
            overflow[0, :, :] = -np.finfo(np.float32).max
            overflow[1, :, :] = np.finfo(np.float32).max
            for array in (good[0], good.astype(np.float64), bad_nan, overflow):
                np.savez_compressed(path, p0=array)
                with self.assertRaises(ValueError):
                    audit.validate_pair_archive(path, [item])
            np.savez_compressed(path, wrong=good)
            with self.assertRaisesRegex(ValueError, 'coverage'):
                audit.validate_pair_archive(path, [item])

    def test_sha_tampering_duplicate_json_and_strict_comparison_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'sample.json'
            path.write_text('{"x":1}\n')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            tracked = {}
            audit.verify_file(root, 'sample.json', expected, tracked)
            self.assertEqual(tracked[str(path)], expected)
            path.write_text('{"x":2}\n')
            with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                audit.verify_file(root, 'sample.json', expected, tracked)
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                audit.verify_file(root, '../sample.json', expected, tracked)
        for text in ('{"same":1,"same":2}', '{"x":NaN}'):
            with self.assertRaises(ValueError):
                audit.loads(text)
        with self.assertRaises(ValueError):
            audit.compare({'n': True}, {'n': 1})
        with self.assertRaises(ValueError):
            audit.compare({'a': 1.}, {'a': 1., 'missing': 0})


if __name__ == '__main__':
    unittest.main()
