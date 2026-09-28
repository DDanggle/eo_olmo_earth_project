"""E2 compatibility and optional E3 display API validation; synthetic JSON only."""
import copy
import importlib.util
from pathlib import Path
import unittest

import test_eo_reader_case_api as baseline

MODULE = Path(__file__).resolve().parents[1] / 'code/eo_evidence_search_v0.py'
spec = importlib.util.spec_from_file_location('e3_reader_case_api_under_test', MODULE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
ARMS = ('real', 'earlier_only', 'later_only', 'repeat_earlier', 'repeat_later', 'no_delta', 'reverse')
HARD_ARMS = ('real', 'later_only', 'repeat_later')


class E3ReaderCaseApiTests(baseline.ReaderCaseApiTests):
    # Inherited tests exercise all original eight E2 behaviours with no E3 block.
    def read(self, tile=None):
        return api.read_reader_case(self.out, self.tile if tile is None else tile, self.catalog_ids)

    def attach_e3(self):
        case = self.payload['cases'][self.tile]
        case['e3_control'] = {'run_id': 'E3-PD-v1', 'question_id': case['question_id'],
            'source_role': 'historical_model_output', 'source_gold': case['reference_label'],
            'predictions': {arm: {'1': 'yes', '2': None, '3': 'no'} for arm in ARMS},
            'source_sha256': {'saved_answers': 'a' * 64}}
        return case['e3_control']

    def test_valid_optional_e3_does_not_replace_e2_or_reference(self):
        extension = self.attach_e3()
        self.save()
        result, status = self.read()
        self.assertEqual(status, 200)
        self.assertTrue(result['available'])
        self.assertEqual(result['case']['e3_control'], extension)
        self.assertEqual(result['case']['reference_label'], 'yes')
        self.assertEqual(result['case']['reader_predictions']['1'], 'no')
        self.assertEqual(result['case']['e3_control']['predictions']['real']['1'], 'yes')
        self.assertIsNone(result['case']['e3_control']['predictions']['reverse']['2'])

    def test_null_optional_e3_is_invalid(self):
        self.payload['cases'][self.tile]['e3_control'] = None
        self.save()
        result, status = self.read()
        self.assertEqual(status, 503)
        self.assertFalse(result['available'])

    def test_unknown_or_missing_e3_arms_are_invalid(self):
        self.attach_e3()
        original = copy.deepcopy(self.payload)
        for change in ('unknown', 'missing_real', 'missing_reverse'):
            with self.subTest(change=change):
                self.payload = copy.deepcopy(original)
                predictions = self.payload['cases'][self.tile]['e3_control']['predictions']
                if change == 'unknown': predictions['unknown_arm'] = predictions['real']
                elif change == 'missing_real': predictions.pop('real')
                else: predictions.pop('reverse')
                self.save()
                result, status = self.read()
                self.assertEqual(status, 503)
                self.assertFalse(result['available'])

    def test_unknown_missing_or_malformed_e3_seeds_are_invalid(self):
        self.attach_e3()
        original = copy.deepcopy(self.payload)
        for values in ({'1': 'yes', '2': 'no', '4': None}, {'1': 'yes', '2': 'no'}, ['yes', 'no', None]):
            with self.subTest(values=values):
                self.payload = copy.deepcopy(original)
                self.payload['cases'][self.tile]['e3_control']['predictions']['real'] = values
                self.save()
                result, status = self.read()
                self.assertEqual(status, 503)
                self.assertFalse(result['available'])

    def test_unrecognized_e3_prediction_values_are_invalid(self):
        self.attach_e3()
        original = copy.deepcopy(self.payload)
        for value in ('maybe', True, 1, '<script>yes</script>'):
            with self.subTest(value=value):
                self.payload = copy.deepcopy(original)
                self.payload['cases'][self.tile]['e3_control']['predictions']['real']['1'] = value
                self.save()
                result, status = self.read()
                self.assertEqual(status, 503)
                self.assertFalse(result['available'])

    def test_e3_label_question_role_and_run_must_match_contract(self):
        self.attach_e3()
        original = copy.deepcopy(self.payload)
        for key, value in (('source_gold', 'no'), ('question_id', 'another_question'),
                           ('source_role', 'physical_counterfactual_gold'), ('run_id', 'E2')):
            with self.subTest(key=key):
                self.payload = copy.deepcopy(original)
                self.payload['cases'][self.tile]['e3_control'][key] = value
                self.save()
                result, status = self.read()
                self.assertEqual(status, 503)
                self.assertFalse(result['available'])

    def test_e3_hashes_must_be_full_hex_sha256(self):
        self.attach_e3()
        original = copy.deepcopy(self.payload)
        for hashes in ({'source': 'g' * 64}, {'source': 'a' * 63}, {'source': None}, []):
            with self.subTest(hashes=hashes):
                self.payload = copy.deepcopy(original)
                self.payload['cases'][self.tile]['e3_control']['source_sha256'] = hashes
                self.save()
                result, status = self.read()
                self.assertEqual(status, 503)
                self.assertFalse(result['available'])

    def test_hard_negative_e3_uses_only_three_registered_arms(self):
        case = self.payload['cases'][self.tile]
        case['reference_label'] = 'no'
        case['question_id'] = self.tile + '_q1_hard'
        extension = self.attach_e3()
        extension['predictions'] = {arm: extension['predictions'][arm] for arm in HARD_ARMS}
        self.save()
        result, status = self.read()
        self.assertEqual(status, 200)
        self.assertEqual(set(result['case']['e3_control']['predictions']), set(HARD_ARMS))
        extension['predictions']['earlier_only'] = {'1': 'yes', '2': 'no', '3': None}
        self.save()
        result, status = self.read()
        self.assertEqual(status, 503)
        self.assertFalse(result['available'])


if __name__ == '__main__':
    unittest.main()
