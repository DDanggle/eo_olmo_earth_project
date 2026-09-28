"""Small isolated reader-case API tests; only temporary JSON is opened."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

CODE_DIR = Path(__file__).resolve().parents[1] / 'code'
sys.path.insert(0, str(CODE_DIR))
MODULE_PATH = CODE_DIR / 'eo_evidence_search_v0.py'
spec = importlib.util.spec_from_file_location('reader_case_api_under_test', MODULE_PATH)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class ReaderCaseApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.tile = 'ks_00001'
        self.catalog_ids = {self.tile}
        self.payload = {
            'schema_version': 'eo_reader_cases_v1',
            'checked_at': '2026-09-25T15:00:00+09:00',
            'cases': {self.tile: {
                'tile': self.tile, 'run_id': 'E2', 'source_role': 'historical_model_output',
                'question_id': self.tile + '_q1_pos', 'dates': ['2022-08-29', '2022-09-10'],
                'slots': ['pre_2', 'post'], 'reference_label': 'yes',
                'reader_predictions': {'1': 'no', '2': 'yes', '3': None},
                'blind_predictions': {'1': 'no', '2': 'no', '3': 'no'},
                'source_sha256': {}, 'limitations': ['Synthetic timestamps'],
                'c1_membership': {'quality_eligible': True, 'supported_stratum': False},
            }},
        }

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, payload=None):
        (self.out / 'reader_cases.json').write_text(json.dumps(self.payload if payload is None else payload))

    def read(self, tile=None):
        return api.read_reader_case(self.out, self.tile if tile is None else tile, self.catalog_ids)

    def test_missing_file_or_entry_is_unavailable(self):
        result, status = self.read()
        self.assertEqual(status, 200)
        self.assertFalse(result['available'])
        payload = copy.deepcopy(self.payload)
        payload['cases'] = {}
        self.save(payload)
        result, status = self.read()
        self.assertEqual(status, 200)
        self.assertFalse(result['available'])

    def test_unknown_catalog_tile_does_not_return_json_entry(self):
        payload = copy.deepcopy(self.payload)
        payload['cases']['ks_99999'] = dict(payload['cases'][self.tile], tile='ks_99999')
        self.save(payload)
        result, status = self.read('ks_99999')
        self.assertEqual(status, 200)
        self.assertFalse(result['available'])
        self.assertIsNone(result.get('case'))

    def test_valid_record_preserves_source_label_and_distinct_model_answer(self):
        self.save()
        result, status = self.read()
        self.assertEqual(status, 200)
        self.assertTrue(result['available'])
        self.assertEqual(result['tile'], self.tile)
        self.assertEqual(result['checked_at'], self.payload['checked_at'])
        self.assertEqual(result['case'], self.payload['cases'][self.tile])
        self.assertEqual(result['case']['reference_label'], 'yes')
        self.assertEqual(result['case']['reader_predictions']['1'], 'no')
        self.assertIsNone(result['case']['reader_predictions']['3'])

    def test_case_tile_mismatch_is_invalid(self):
        self.payload['cases'][self.tile]['tile'] = 'ks_00002'
        self.save()
        result, status = self.read()
        self.assertEqual(status, 503)
        self.assertFalse(result['available'])

    def test_malformed_json_and_wrong_schema_are_invalid(self):
        for value in ('invalid json', '[]', '{}', json.dumps(dict(self.payload, schema_version='other'))):
            with self.subTest(value=value):
                (self.out / 'reader_cases.json').write_text(value)
                result, status = self.read()
                self.assertEqual(status, 503)
                self.assertFalse(result['available'])
                self.assertNotIn(self.tmp.name, json.dumps(result))
                self.assertNotIn('Traceback', json.dumps(result))

    def test_wrong_run_or_source_role_is_invalid(self):
        for key, value in (('run_id', 'E3'), ('source_role', 'reference_gold')):
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload['cases'][self.tile][key] = value
                self.save(payload)
                result, status = self.read()
                self.assertEqual(status, 503)
                self.assertFalse(result['available'])

    def test_malformed_case_fields_are_rejected(self):
        for key, value in (('dates', ['one_date']), ('limitations', 'not a list'),
                           ('reader_predictions', {'1': '<script>not a class</script>'}),
                           ('source_sha256', [])):
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload['cases'][self.tile][key] = value
                self.save(payload)
                result, status = self.read()
                self.assertEqual(status, 503)
                self.assertFalse(result['available'])

    def test_rereads_current_snapshot_without_server_restart(self):
        self.save()
        first, status = self.read()
        self.assertEqual(status, 200)
        self.payload['checked_at'] = '2026-09-25T16:00:00+09:00'
        self.payload['cases'][self.tile]['reader_predictions']['1'] = 'yes'
        self.save()
        updated, status = self.read()
        self.assertEqual(status, 200)
        self.assertNotEqual(first['checked_at'], updated['checked_at'])
        self.assertEqual(first['case']['reader_predictions']['1'], 'no')
        self.assertEqual(updated['case']['reader_predictions']['1'], 'yes')


if __name__ == '__main__':
    unittest.main()
