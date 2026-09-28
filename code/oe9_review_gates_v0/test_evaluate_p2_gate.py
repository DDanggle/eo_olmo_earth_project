"""Synthetic boundary fixtures only. No values here are model measurements."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from evaluate_p2_gate import InvalidInput, evaluate

ROOT = Path(__file__).resolve().parent
CONFIG_BYTES = (ROOT / 'review_execution_gates.json').read_bytes()
CONFIG = json.loads(CONFIG_BYTES)
CONFIG_SHA = hashlib.sha256(CONFIG_BYTES).hexdigest()


def synthetic_summary(deltas=(0.03, 0.03, 0.03)):
    p = CONFIG['p2']
    result = {
        'schema_version': 'oe9_p2_scorer_summary_v1', 'gate_config_sha256': CONFIG_SHA,
        'protocol_locked_before_results': True, 'scorer_source_sha256': 'a'*64,
        'predictions_manifest_sha256': 'b'*64, 'aggregation': p['aggregation'],
        'absent_case_area_threshold': .001,
        'cohort': copy.deepcopy(p['cohort']), 'comparison_axis': 'matched_data_exposure',
        'fairness_checks': {key: True for key in p['fairness_checks_required']},
        'paired_seed_ids': p['paired_seed_ids'].copy(), 'seed_ids_locked_before_results': True, 'runs': [],
        'SYNTHETIC_FIXTURE_NOT_MODEL_MEASUREMENT': True,
    }
    for seed, delta in zip(p['paired_seed_ids'], deltas):
        for arm, score in (('B0', .5), ('B2', .5 + delta)):
            checks = {key: True for key in p['training_adequacy']['required_checks']}
            checks.update(completed_updates=300, predeclared_minimum_updates=200,
                          predeclared_eval_interval_updates=50,
                          budget_cap_reached_while_improving=False,
                          last_evaluation_windows=[{'step': step, 'auc': score}
                                                   for step in (200, 250, 300)])
            result['runs'].append({
                'arm': arm, 'seed': seed, 'checkpoint_sha256': 'c'*64, 'run_receipt_sha256': 'd'*64,
                'cohort_base_ids_sha256': p['cohort']['cohort_base_ids_sha256'],
                'target_iou_by_k': {str(k): score for k in p['ks']},
                'absent_fp_area_by_k': {str(k): .02 for k in p['ks']},
                'absent_fp_case_rate_by_k': {str(k): .2 for k in p['ks']},
                'training_adequacy': checks,
                'costs': {'total_gpu_seconds_including_feature_precompute': 100,
                          'training_observation_exposures': 1000, 'inference_gpu_seconds': 10,
                          'inference_distinct_query_observations': 4,
                          'inference_support_observations': 16},
            })
    return result


class P2GateBoundaryTests(unittest.TestCase):
    def run_gate(self, summary):
        return evaluate(summary, CONFIG, CONFIG_SHA)

    def test_exact_positive_margin_all_seeds_is_development_gain_not_novelty(self):
        r = self.run_gate(synthetic_summary((.02, .02, .02)))
        self.assertEqual(r['decision'], 'development_repeated_practical_gain')
        self.assertFalse(r['cross_geography_claim_promotion_allowed'])
        self.assertFalse(r['formal_equivalence_test_performed'])
        self.assertFalse(r['CVPR_novelty_or_acceptance_established'])

    def test_inside_band_is_descriptive_tie_and_not_global_eo_impossibility(self):
        r = self.run_gate(synthetic_summary((.019, -.019, 0)))
        self.assertEqual(r['decision'], 'development_observed_practical_tie')
        self.assertFalse(r['geographic_equivalence_proven'])
        self.assertFalse(r['all_EO_encoder_learning_ruled_out'])

    def test_positive_average_with_one_negative_seed_is_inconclusive(self):
        r = self.run_gate(synthetic_summary((.06, .06, -.01)))
        self.assertEqual(r['decision'], 'inconclusive_development_result')

    def test_exact_negative_margin_is_practical_harm(self):
        self.assertEqual(self.run_gate(synthetic_summary((-.02, -.02, -.02)))['decision'],
                         'development_repeated_practical_harm')

    def test_one_seed_gain_others_tie_is_not_repeated_gain(self):
        self.assertEqual(self.run_gate(synthetic_summary((.03, .01, .01)))['decision'],
                         'inconclusive_development_result')

    def test_absent_area_at_limit_allowed_and_just_over_blocked(self):
        s = synthetic_summary()
        r = next(r for r in s['runs'] if r['arm'] == 'B2')
        r['absent_fp_area_by_k']['8'] = .025
        self.assertEqual(self.run_gate(s)['decision'], 'development_repeated_practical_gain')
        r['absent_fp_area_by_k']['8'] += 1e-6
        self.assertEqual(self.run_gate(s)['decision'], 'blocked_absent_false_positive_regression')

    def test_absent_case_regression_blocks_even_when_area_does_not(self):
        s = synthetic_summary()
        next(r for r in s['runs'] if r['arm'] == 'B2')['absent_fp_case_rate_by_k']['1'] = .23
        self.assertEqual(self.run_gate(s)['decision'], 'blocked_absent_false_positive_regression')

    def test_undertrained_baseline_blocks_apparent_gain(self):
        s = synthetic_summary()
        s['runs'][0]['training_adequacy']['budget_cap_reached_while_improving'] = True
        self.assertEqual(self.run_gate(s)['decision'], 'blocked_inadequate_or_unfair_comparison')

    def test_unstable_last_windows_are_not_convergence(self):
        s = synthetic_summary()
        s['runs'][0]['training_adequacy']['last_evaluation_windows'][0]['auc'] = .48
        self.assertEqual(self.run_gate(s)['decision'], 'blocked_inadequate_or_unfair_comparison')

    def test_same_support_guard_false_blocks(self):
        s = synthetic_summary()
        s['fairness_checks']['same_support_draws_and_prefixes'] = False
        self.assertEqual(self.run_gate(s)['decision'], 'blocked_inadequate_or_unfair_comparison')

    def test_full192_cannot_replace_common96(self):
        s = synthetic_summary()
        s['cohort']['base_count'] = 192
        with self.assertRaises(InvalidInput): self.run_gate(s)

    def test_nan_is_rejected(self):
        s = synthetic_summary()
        s['runs'][0]['target_iou_by_k']['1'] = float('nan')
        with self.assertRaises(InvalidInput): self.run_gate(s)

    def test_missing_seed_cannot_be_called_three_seed_replication(self):
        s = synthetic_summary()
        s['runs'].pop()
        with self.assertRaises(InvalidInput): self.run_gate(s)

    def test_duplicate_arm_seed_is_rejected(self):
        s = synthetic_summary()
        s['runs'][1] = copy.deepcopy(s['runs'][0])
        with self.assertRaises(InvalidInput): self.run_gate(s)

    def test_wrong_config_hash_is_rejected(self):
        s = synthetic_summary()
        s['gate_config_sha256'] = '0'*64
        with self.assertRaises(InvalidInput): self.run_gate(s)

    def test_changed_absent_case_definition_is_rejected(self):
        s = synthetic_summary()
        s['absent_case_area_threshold'] = .01
        with self.assertRaises(InvalidInput): self.run_gate(s)

    def test_old_plateau_cannot_hide_unassessed_later_training(self):
        s = synthetic_summary()
        s['runs'][0]['training_adequacy']['completed_updates'] = 400
        with self.assertRaises(InvalidInput): self.run_gate(s)

    def test_nonuniform_k_spacing_uses_trapezoid_not_mean(self):
        s = synthetic_summary((0, 0, 0))
        for r in s['runs']:
            r['target_iou_by_k'] = {'1': 0., '2': 0., '4': 0., '8': 1.}
        self.assertAlmostEqual(self.run_gate(s)['seed_results'][0]['B0_auc'], 2/7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
