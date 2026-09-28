import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import c1_flood_label_quality_audit_v0 as c1


def item(kind, stats):
    return {'id': 'synthetic_' + kind, 'kind': kind, 'flood_frac': round(stats['flood_frac'], 4)}


class LabelQualityTests(unittest.TestCase):
    def test_e1_labelled_denominator_excludes_valid_but_unlabelled(self):
        mask = np.ones(c1.SHAPE, dtype=np.uint8)
        valid = np.ones(c1.SHAPE, dtype=np.uint8)
        mask[:96] = 0
        mask[96:98] = 3
        valid[191] = 0
        stats = c1.mask_stats(mask, valid)
        self.assertEqual(stats['labelled_px'], 95 * 192)
        self.assertEqual(stats['flood_px'], 2 * 192)
        self.assertEqual(stats['valid_frac'], 95 / 192)
        self.assertEqual(stats['flood_frac'], 2 / 95)
        self.assertEqual(stats['valid_but_unlabelled_px'], 96 * 192)
        self.assertEqual(sum(stats['class_valid_counts'].values()), 192 * 192)
        self.assertEqual(sum(stats['position']['labelled_rows_px']), stats['labelled_px'])
        self.assertEqual(sum(stats['position']['flood_columns_px']), stats['flood_px'])
        self.assertFalse(c1.check_source_label(item('pos', stats), stats))

    def test_positive_and_hard_quality_threshold_boundaries(self):
        mask = np.ones(c1.SHAPE, dtype=np.uint8)
        valid = np.ones(c1.SHAPE, dtype=np.uint8)
        for count, accepted in ((737, False), (738, True)):
            with self.subTest(flood_px=count):
                mask[:] = 1
                mask.flat[:count] = 3
                stats = c1.mask_stats(mask, valid)
                if accepted:
                    self.assertTrue(c1.check_source_label(item('pos', stats), stats))
                else:
                    with self.assertRaises(ValueError):
                        c1.check_source_label(item('pos', stats), stats)
        for count, accepted in ((33177, False), (33178, True)):
            with self.subTest(valid_px=count):
                mask[:] = 1
                valid[:] = 0
                valid.flat[:count] = 1
                stats = c1.mask_stats(mask, valid)
                if accepted:
                    self.assertTrue(c1.check_source_label(item('hard_neg', stats), stats))
                else:
                    with self.assertRaises(ValueError):
                        c1.check_source_label(item('hard_neg', stats), stats)
        bad = item('hard_neg', stats)
        bad['flood_frac'] = .001
        with self.assertRaises(ValueError):
            c1.check_source_label(bad, stats)

    def test_invalid_shape_dtype_and_values_fail_closed(self):
        good = np.ones(c1.SHAPE, dtype=np.uint8)
        cases = [(good.astype(float), good), (good[:1], good),
                 (np.full(c1.SHAPE, 4, dtype=np.uint8), good),
                 (good, np.full(c1.SHAPE, 2, dtype=np.uint8))]
        for mask, valid in cases:
            with self.subTest(shape=mask.shape, dtype=mask.dtype), self.assertRaises(ValueError):
                c1.mask_stats(mask, valid)

    def test_missing_mask_fails_and_source_hashes_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                c1.read_tile(root, 'ks_00001')
            for name in ('mask', 'valid'):
                directory = root / 'kurosiwo_s1_cache' / (name + '_u8')
                directory.mkdir(parents=True)
                np.save(directory / 'ks_00001.npy', np.ones(c1.SHAPE, dtype=np.uint8))
            stats = c1.read_tile(root, 'ks_00001')
            self.assertEqual(stats['valid_frac'], 1.)
            self.assertEqual(stats['flood_px'], 0)
            for name in ('mask', 'valid'):
                self.assertEqual(c1.sha(root / stats[name + '_path']), stats[name + '_sha256'])
            # An absent frozen C0 source must preserve failure artifacts, and a
            # retry must not overwrite the existing attempted-run directory.
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'schema': 'c1-flood-label-quality-plan-v0', 'no_model_scores': True,
                                         'thresholds': {'flood_min': .02, 'valid_min': .90,
                                                        'rounded_qa_abs_tolerance': 5e-5},
                                         'expected_selected_counts': {'pos': 457, 'hard_neg': 457},
                                         'expected_c0': {'manifest_sha256': '0' * 64}}))
            args = SimpleNamespace(root=root, out='audit', prereg=plan)
            with self.assertRaises(FileNotFoundError):
                c1.audit(args)
            status = root / 'audit/status.json'
            self.assertEqual(json.loads(status.read_text())['status'], 'invalid')
            self.assertTrue((root / 'audit/error.json').exists())
            old_status_hash = c1.sha(status)
            with self.assertRaises(FileExistsError):
                c1.audit(args)
            self.assertEqual(c1.sha(status), old_status_hash)

    def test_fixed_population_ids_labels_and_slots(self):
        items = []
        for n in range(457):
            for kind in ('pos', 'neg', 'hard_neg'):
                tile = f'ks_{n + (1000 if kind == "hard_neg" else 0):05d}'
                suffix = 'hard' if kind == 'hard_neg' else kind
                items.append({'id': tile + '_q1_' + suffix, 'tile': tile, 'kind': kind,
                              'answer': 'yes' if kind == 'pos' else 'no', 'partition': 'test',
                              'phen': 'flood', 'fold': 'test', 'type': 'Q1', 'event': 1, 'cluster': '1',
                              'slots': ['pre_1', 'pre_2'] if kind == 'neg' else ['pre_2', 'post'],
                              'dates': ['2020-01-01', '2020-01-13'], 'flood_frac': .1 if kind == 'pos' else 0.})
        self.assertEqual(len(c1.selected_items(items)), 914)
        for mutation in ('duplicate', 'missing', 'wrong_label', 'wrong_slots', 'wrong_id'):
            bad = copy.deepcopy(items)
            if mutation == 'duplicate': bad.append(bad[0])
            if mutation == 'missing': bad.pop(0)
            if mutation == 'wrong_label': bad[0]['answer'] = 'no'
            if mutation == 'wrong_slots': bad[0]['slots'] = ['pre_1', 'post']
            if mutation == 'wrong_id': bad[0]['id'] = 'wrong_id'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                c1.selected_items(bad)


if __name__ == '__main__':
    unittest.main()
