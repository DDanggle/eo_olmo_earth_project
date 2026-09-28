"""Optional E4 display API with E2/E3 regression tests; synthetic data only."""
import copy
import importlib.util
from pathlib import Path
import unittest

import test_eo_e3_reader_case_api as previous

MODULE = Path(__file__).resolve().parents[1]/'code/eo_evidence_search_v0.py'
spec = importlib.util.spec_from_file_location('e4_reader_case_api_under_test', MODULE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
ARMS = ('real', 'delta_only', 'delta_sign_flip', 'delta_feature_permute')


class E4ReaderCaseApiTests(previous.E3ReaderCaseApiTests):
    # All sixteen inherited E2/E3 tests run against the new E4-capable API.
    def read(self, tile=None):
        return api.read_reader_case(self.out, self.tile if tile is None else tile, self.catalog_ids)

    def attach_e4(self):
        case = self.payload['cases'][self.tile]
        case['e4_control'] = {'run_id': 'E4-D-v0', 'question_id': case['question_id'],
            'source_role': 'historical_model_output', 'source_gold': case['reference_label'],
            'predictions': {arm: {'1': 'yes', '2': None, '3': 'no'} for arm in ARMS},
            'source_sha256': {'saved_answers': 'a' * 64, 'independent_audit': '9' * 64}}
        return case['e4_control']

    def assert_invalid(self):
        self.save()
        result, status = self.read()
        self.assertEqual(status, 503)
        self.assertFalse(result['available'])
        self.assertIsNone(result.get('case'))

    def test_valid_e4_coexists_with_e2_e3_and_unchanged_source_label(self):
        e3 = copy.deepcopy(self.attach_e3())
        e4 = self.attach_e4()
        e4['predictions']['delta_only']['1'] = 'no'
        self.save()
        result, status = self.read()
        self.assertEqual(status, 200)
        self.assertTrue(result['available'])
        case = result['case']
        self.assertEqual(case['reference_label'], 'yes')
        self.assertEqual(case['reader_predictions']['1'], 'no')
        self.assertEqual(case['e3_control'], e3)
        self.assertEqual(case['e4_control'], e4)
        self.assertEqual(case['e4_control']['predictions']['delta_only']['1'], 'no')
        self.assertIsNone(case['e4_control']['predictions']['delta_sign_flip']['2'])

    def test_hard_negative_e4_requires_all_four_arms(self):
        case = self.payload['cases'][self.tile]
        case['reference_label'] = 'no'
        case['question_id'] = self.tile + '_q1_hard'
        e3 = self.attach_e3()
        e3['predictions'] = {arm: e3['predictions'][arm] for arm in previous.HARD_ARMS}
        e4 = self.attach_e4()
        self.save()
        result, status = self.read()
        self.assertEqual(status, 200)
        self.assertEqual(set(result['case']['e4_control']['predictions']), set(ARMS))
        self.assertEqual(len(result['case']['e3_control']['predictions']), 3)
        self.assertEqual(result['case']['reference_label'], 'no')
        e4['predictions'].pop('delta_feature_permute')
        self.assert_invalid()

    def test_optional_e4_null_is_invalid(self):
        self.payload['cases'][self.tile]['e4_control'] = None
        self.assert_invalid()

    def test_unknown_or_missing_e4_arms_are_invalid(self):
        self.attach_e4()
        original = copy.deepcopy(self.payload)
        for change in ('unknown', 'missing_real', 'missing_delta_only', 'missing_sign', 'missing_permutation'):
            with self.subTest(change=change):
                self.payload = copy.deepcopy(original)
                predictions = self.payload['cases'][self.tile]['e4_control']['predictions']
                if change == 'unknown':
                    predictions['reverse'] = predictions['real']
                else:
                    key = {'missing_real': 'real', 'missing_delta_only': 'delta_only',
                           'missing_sign': 'delta_sign_flip', 'missing_permutation': 'delta_feature_permute'}[change]
                    predictions.pop(key)
                self.assert_invalid()

    def test_e4_seed_maps_reject_unknown_missing_and_malformed_values(self):
        self.attach_e4()
        original = copy.deepcopy(self.payload)
        for values in ({'1': 'yes', '2': 'no', '4': None}, {'1': 'yes', '2': 'no'},
                       {'1': 'yes', '2': 'no', '3': None, '0': 'yes'}, ['yes', 'no', None], None):
            with self.subTest(values=values):
                self.payload = copy.deepcopy(original)
                self.payload['cases'][self.tile]['e4_control']['predictions']['real'] = values
                self.assert_invalid()

    def test_e4_prediction_classes_are_yes_no_or_null_only(self):
        self.attach_e4()
        original = copy.deepcopy(self.payload)
        for value in ('maybe', 'YES', True, 1, 0, '<script>yes</script>', {}):
            with self.subTest(value=value):
                self.payload = copy.deepcopy(original)
                self.payload['cases'][self.tile]['e4_control']['predictions']['delta_only']['1'] = value
                self.assert_invalid()

    def test_e4_identity_role_and_source_gold_match_the_case(self):
        self.attach_e4()
        original = copy.deepcopy(self.payload)
        for key, value in (('source_gold', 'no'), ('source_gold', None), ('question_id', 'another_question'),
                           ('source_role', 'physical_counterfactual_gold'), ('run_id', 'E3-PD-v1')):
            with self.subTest(key=key, value=value):
                self.payload = copy.deepcopy(original)
                self.payload['cases'][self.tile]['e4_control'][key] = value
                self.assert_invalid()

    def test_e4_hashes_require_nonempty_lowercase_sha256_mapping(self):
        self.attach_e4()
        original = copy.deepcopy(self.payload)
        for hashes in ({}, {'source': 'g' * 64}, {'source': 'A' * 64}, {'source': 'a' * 63},
                       {'source': None}, [], None):
            with self.subTest(hashes=hashes):
                self.payload = copy.deepcopy(original)
                self.payload['cases'][self.tile]['e4_control']['source_sha256'] = hashes
                self.assert_invalid()


if __name__ == '__main__':
    unittest.main()
