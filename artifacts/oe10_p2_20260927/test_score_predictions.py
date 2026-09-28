"""Synthetic scorer tests; these fixtures are not EO model results."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from score_predictions import (AGGREGATION, KS, ScoringError, cohort_hash, file_sha,
                               hierarchical_macro, public_expected, read_jsonl,
                               score_core)


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.episodes, self.scoring, self.predictions, self.gold = {}, {}, {}, {}
        # Three present and three absent bases; class1 has two present queries,
        # class2 only one. This catches class-macro vs episode-average mistakes.
        specs = [('q1', 1, 2, True), ('q2', 1, 2, True), ('q3', 2, 1, True),
                 ('q1', 2, 1, False), ('q2', 2, 1, False), ('q3', 1, 2, False)]
        for i, (query, target_cls, counter_cls, present) in enumerate(specs):
            base = f'synthetic-base-{i}'
            valid = np.zeros((128, 128), bool)
            valid.flat[:1000] = True
            target = np.zeros_like(valid)
            counter = np.zeros_like(valid)
            if present:
                target.flat[:100] = True
            else:
                counter.flat[:100] = True
            pred_path = self.root / f'probability_{i}.npz'
            np.savez_compressed(pred_path, probability=target.astype(np.float32))
            for k in KS:
                eid = f'{base}:k{k}'
                pair = f'synthetic-pair-{target_cls}-{counter_cls}'
                self.episodes[eid] = {
                    'episode_id': eid, 'base_id': base, 'k_pairs': k, 'split': 'development',
                    'query_patch_id': query, 'query_parent_tile': 'synthetic-parent', 'pair_id': pair,
                }
                self.scoring[eid] = {
                    'episode_id': eid, 'base_id': base, 'k_pairs': k, 'query_patch_id': query,
                    'pair_id': pair, 'target_class': target_cls, 'counter_class': counter_cls,
                    'target_pixels': int(target.sum()), 'counter_pixels': int(counter.sum()),
                    'label_valid_pixels': int(valid.sum()), 'target_present': present,
                    'counter_present': not present, 'query_label_npz': f'labels/{query}.npz',
                    'query_label_sha256': 'a'*64, 'expected_query_label_sha256': 'a'*64,
                    'k8_auc_cohort': True,
                }
                self.predictions[eid] = {'episode_id': eid, 'base_id': base, 'k': k,
                    'npz_path': pred_path.name, 'sha256': file_sha(pred_path)}
                self.gold[eid] = {'target_mask': target.copy(), 'counter_mask': counter.copy(),
                    'label_valid': valid.copy(), 'audit': {'purpose': 'evaluation', 'episode_id': eid,
                        'query_patch_id': query, 'label_sha256': 'a'*64}}
        self.config = {'p2': {'ks': list(KS), 'auc_denominator': 7, 'aggregation': AGGREGATION,
            'absence_guardrails': {'absent_case_positive_if_predicted_valid_area_fraction_gt': .001},
            'cohort': {'base_count': 6, 'query_patch_count': 3,
                       'target_present_base_count': 3, 'target_absent_base_count': 3,
                       'target_classes': [1, 2], 'development_parent_ids': ['synthetic-parent'],
                       'cohort_base_ids_sha256': cohort_hash([f'synthetic-base-{i}' for i in range(6)]),
                       'source_scoring_sha256': 'b'*64}}}
        self.accessed = []

    def accessor(self, eid):
        self.accessed.append(eid)
        return copy.deepcopy(self.gold[eid])

    def score(self):
        return score_core(self.episodes, self.scoring, self.predictions, self.root,
                          self.accessor, self.config)

    def replace_base_probability(self, base_number, probability, **extra):
        path = self.root / f'probability_{base_number}.npz'
        np.savez_compressed(path, probability=probability, **extra)
        for row in self.predictions.values():
            if row['base_id'] == f'synthetic-base-{base_number}':
                row['sha256'] = file_sha(path)


class IndependentScorerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.f = Fixture(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_perfect_present_zero_absent_and_gold_read_once_per_base(self):
        r = self.f.score()
        self.assertEqual(r['target_iou_auc'], 1)
        self.assertEqual(r['target_iou_by_k'], {str(k): 1 for k in KS})
        self.assertEqual(r['absent_fp_area_by_k'], {str(k): 0 for k in KS})
        self.assertEqual(len(self.f.accessed), 6)
        self.assertFalse(r['training_adequacy_assertions_generated'])
        self.assertFalse(r['fairness_assertions_generated'])

    def test_target_class_macro_not_episode_weighted_mean(self):
        self.f.replace_base_probability(2, np.zeros((128, 128), np.float32))
        self.assertEqual(self.f.score()['target_iou_auc'], .5)

    def test_query_before_class_macro_and_parent_macro(self):
        rows = [
            {'parent': 'p0', 'target_class': 1, 'query_patch_id': 'q1', 'value': 1},
            {'parent': 'p0', 'target_class': 1, 'query_patch_id': 'q1', 'value': 1},
            {'parent': 'p0', 'target_class': 1, 'query_patch_id': 'q2', 'value': 0},
            {'parent': 'p0', 'target_class': 2, 'query_patch_id': 'q3', 'value': 0},
            {'parent': 'p1', 'target_class': 1, 'query_patch_id': 'q4', 'value': 1},
        ]
        value, _ = hierarchical_macro(rows, 'value')
        self.assertEqual(value, .625)  # p0=(.5+0)/2=.25, p1=1, parent macro=.625.

    def test_all_void_predictions_do_not_create_false_positive(self):
        probability = np.zeros((128, 128), np.float32)
        probability.flat[1000:] = 1
        self.f.replace_base_probability(3, probability)
        self.assertEqual(self.f.score()['absent_fp_area_by_k']['1'], 0)

    def test_absent_empty_mask_not_added_as_perfect_target_iou(self):
        for i in range(3): self.f.replace_base_probability(i, np.zeros((128, 128), np.float32))
        r = self.f.score()
        self.assertEqual(r['target_iou_auc'], 0)
        self.assertEqual(sum(x['target_iou'] is None for x in r['per_episode_scoring_only']), 12)

    def test_strict_probability_threshold(self):
        probability = np.full((128, 128), .5, np.float32)
        self.f.replace_base_probability(0, probability)
        self.assertEqual(self.f.score()['target_iou_by_k']['1'], .75)

    def test_absent_case_threshold_is_strictly_greater_than(self):
        probability = np.zeros((128, 128), np.float32)
        probability.flat[0] = 1  # 1/1000 == .001: not an FP case.
        self.f.replace_base_probability(3, probability)
        r = self.f.score()
        self.assertEqual(r['absent_fp_case_rate_by_k']['1'], 0)
        probability.flat[1] = 1
        self.f.replace_base_probability(3, probability)
        r = self.f.score()
        self.assertEqual(r['absent_fp_case_rate_by_k']['1'], .25)

    def test_nonuniform_K_auc_not_simple_average(self):
        for row in self.f.predictions.values():
            if row['k'] != 8 and int(row['base_id'].split('-')[-1]) < 3:
                path = self.f.root / 'zero.npz'
                if not path.exists(): np.savez_compressed(path, probability=np.zeros((128, 128), np.float32))
                row['npz_path'], row['sha256'] = path.name, file_sha(path)
        self.assertAlmostEqual(self.f.score()['target_iou_auc'], 2/7)

    def test_missing_prediction_is_rejected_before_gold_access(self):
        self.f.predictions.pop(next(iter(self.f.predictions)))
        with self.assertRaises(ScoringError): self.f.score()
        self.assertFalse(self.f.accessed)

    def test_extra_lower_K_prediction_is_rejected(self):
        self.f.predictions['extra'] = copy.deepcopy(next(iter(self.f.predictions.values())))
        self.f.predictions['extra']['episode_id'] = 'extra'
        with self.assertRaises(ScoringError): self.f.score()

    def test_duplicate_JSONL_episode_is_rejected(self):
        row = next(iter(self.f.predictions.values()))
        path = self.f.root / 'dup.jsonl'
        path.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
        with self.assertRaises(ScoringError): read_jsonl(path, 'episode_id')

    def test_base_or_K_join_mismatch_is_rejected(self):
        row = next(iter(self.f.predictions.values()))
        row['k'] = 8
        with self.assertRaises(ScoringError): self.f.score()

    def test_changed_cohort_hash_is_rejected(self):
        self.f.config['p2']['cohort']['cohort_base_ids_sha256'] = '0'*64
        with self.assertRaises(ScoringError): self.f.score()

    def test_prediction_hash_mismatch_is_rejected(self):
        next(iter(self.f.predictions.values()))['sha256'] = '0'*64
        with self.assertRaises(ScoringError): self.f.score()

    def test_nonfinite_and_out_of_range_are_rejected(self):
        for bad in (np.nan, np.inf, -0.1, 1.1):
            probability = np.zeros((128, 128), np.float32)
            probability.flat[0] = bad
            self.f.replace_base_probability(0, probability)
            with self.assertRaises(ScoringError): self.f.score()

    def test_wrong_shape_or_integer_mask_is_rejected(self):
        for probability in (np.zeros((32, 32), np.float32), np.zeros((128, 128), np.uint8)):
            self.f.replace_base_probability(0, probability)
            with self.assertRaises(ScoringError): self.f.score()

    def test_extra_npz_key_is_rejected(self):
        self.f.replace_base_probability(0, np.zeros((128, 128), np.float32), semantic=np.zeros((128,128)))
        with self.assertRaises(ScoringError): self.f.score()

    def test_path_escape_is_rejected(self):
        next(iter(self.f.predictions.values()))['npz_path'] = '../probability_0.npz'
        with self.assertRaises(ScoringError): self.f.score()

    def test_scoring_gold_count_identity_mismatch_is_rejected(self):
        for eid in self.f.gold:
            if eid.startswith('synthetic-base-0:'):
                self.f.gold[eid]['target_mask'].flat[0] = False
        with self.assertRaises(ScoringError): self.f.score()

    def test_gold_audit_mismatch_is_rejected(self):
        for eid in self.f.gold:
            if eid.startswith('synthetic-base-0:'):
                self.f.gold[eid]['audit']['purpose'] = 'training'
        with self.assertRaises(ScoringError): self.f.score()

    def test_gold_class_identity_cannot_change_between_K(self):
        self.f.scoring['synthetic-base-0:k8']['target_class'] = 3
        with self.assertRaises(ScoringError): self.f.score()

    def test_no_fallback_when_absent_stratum_missing(self):
        with self.assertRaises(ScoringError): hierarchical_macro([{'parent':'p','target_class':1,'query_patch_id':'q','x':None}], 'x')


if __name__ == '__main__':
    unittest.main(verbosity=2)
