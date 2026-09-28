"""Synthetic frozen-artifact tests for the local historical reader-case builder."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
import build_eo_reader_cases_v0 as builder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def write_rows(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in values))


class ReaderCaseBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.repo, self.c1, self.quality = base / 'repo', base / 'c1', base / 'quality'
        for path in (self.repo, self.c1, self.quality):
            path.mkdir()
        self.catalog = base / 'catalog.json'
        self.items_rel = 'artifacts/c0_linear_view_probe_v1_20260925/items.jsonl'
        self.items = []
        for index in range(457):
            for kind in ('pos', 'neg', 'hard_neg'):
                tile = f'ks_{index + (1000 if kind == "hard_neg" else 0):05d}'
                suffix = 'hard' if kind == 'hard_neg' else kind
                self.items.append({'id': tile + '_q1_' + suffix, 'tile': tile, 'kind': kind,
                                   'answer': 'yes' if kind == 'pos' else 'no', 'partition': 'test',
                                   'phen': 'flood', 'fold': 'test', 'type': 'Q1', 'event': 1, 'cluster': '1',
                                   'sensor': 'sentinel1', 'slots': ['pre_1', 'pre_2'] if kind == 'neg' else ['pre_2', 'post'],
                                   'dates': ['2022-08-17', '2022-08-29'] if kind == 'neg' else ['2022-08-29', '2022-09-10'],
                                   'flood_frac': (round(4000 / (30000 if index == 0 else 36864), 4)
                                                  if kind in ('pos', 'neg') else 0.)})
        for index in range(384):
            kind = 'pos' if index % 2 else 'neg'
            self.items.append({'id': f'land_{index}', 'tile': f'landtile_{index}', 'phen': 'landslide',
                               'partition': 'test', 'fold': 'test', 'kind': kind,
                               'answer': 'yes' if kind == 'pos' else 'no', 'dates': ['2020-01-01', '2020-01-13']})
        write_rows(self.repo / self.items_rel, self.items)
        self.code_paths = []
        for name in ('e2_multi_reader_v0.py', 'e1_flood_qa_v0.py', 'extract_kurosiwo_s1_cache.py'):
            path = self.repo / 'code' / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('# Synthetic source fixture\n')
            self.code_paths.append(path)
        c0 = (self.repo / self.items_rel).parent
        (c0 / 'source.py').write_text('# Synthetic C0 source\n')
        write(c0 / 'prereg.json', {'scope': 'synthetic'})
        write(c0 / 'manifest.json', {'schema': 'c0-prepared-v1'})
        c0_hashes = {key: digest(c0 / filename) for filename, key in
                    [('manifest.json', 'manifest_sha256'), ('source.py', 'code_sha256'),
                     ('prereg.json', 'prereg_sha256'), ('items.jsonl', 'items_sha256')]}
        self.targets = [x for x in self.items if x['phen'] == 'flood' and x['kind'] in ('pos', 'hard_neg')]
        self.predictions = {}
        for arm in ('reader', 'blind'):
            for seed in (1, 2, 3):
                path = self.repo / f'artifacts/e2_multi_reader_v0/{arm}_seed{seed}/answers_real_all.jsonl'
                rows = []
                for item in self.items:
                    rows.append({key: item[key] for key in ('id', 'tile', 'fold', 'phen', 'kind')} | {
                        'emb_item': item['id'], 'text_gold': item['answer'], 'emb_gold': item['answer'],
                        'parsed': item['answer'] if arm == 'reader' and seed != 1 else 'no',
                        'answer_raw': item['answer'] if arm == 'reader' and seed != 1 else 'no'})
                write_rows(path, rows)
                self.predictions[(arm, seed)] = path
        self.quality_rows = []
        for item in self.targets:
            eligible = item['id'] != 'ks_00000_q1_pos'
            denominator = 36864 if eligible else 30000
            self.quality_rows.append({'id': item['id'], 'tile': item['tile'], 'event': str(item['event']),
                                      'dates': item['dates'], 'slots': item['slots'], 'kind': item['kind'],
                                      'source_answer': item['answer'], 'eligible_symmetric_quality': eligible,
                                      'source_qa_flood_frac': item['flood_frac'],
                                      'valid_frac': denominator / 36864,
                                      'flood_frac': 4000 / denominator if item['kind'] == 'pos' else 0.,
                                      'flood_px': 4000 if item['kind'] == 'pos' else 0,
                                      'mask_sha256': 'a' * 64, 'valid_sha256': 'b' * 64,
                                      'mask_path': f"kurosiwo_s1_cache/mask_u8/{item['tile']}.npy",
                                      'valid_path': f"kurosiwo_s1_cache/valid_u8/{item['tile']}.npy"})
        write_rows(self.quality / 'quality.jsonl', self.quality_rows)
        write(self.quality / 'status.json', {'status': 'complete', 'valid': True, 'n': 914, 'no_model_scores': True})
        (self.quality / 'source.py').write_text('# Synthetic quality source fixture\n')
        write(self.quality / 'prereg.json', {'schema': 'c1-flood-label-quality-plan-v0', 'no_model_scores': True,
              'expected_selected_counts': {'pos': 457, 'hard_neg': 457}, 'expected_c0': c0_hashes,
              'thresholds': {'flood_min': .02, 'valid_min': .90, 'rounded_qa_abs_tolerance': .00005}})
        write(self.quality / 'summary.json', {'n': 914, 'n_unique_tiles': 914, 'no_model_scores': True,
                                            'counts': {'pos': 457, 'hard_neg': 457}, 'symmetric_n': 913,
                                            'symmetric_counts': {'pos': 456, 'hard_neg': 457},
                                            'all_source_labels_verified': True})
        selected_ids = sorted(x['id'] for x in self.quality_rows if x['eligible_symmetric_quality'])
        write(self.quality / 'symmetric_quality_ids.json', {'ids': selected_ids,
              'excluded_ids': ['ks_00000_q1_pos'], 'no_model_scores': True,
              'by_kind': {kind: sorted(x['id'] for x in self.quality_rows if x['kind'] == kind and x['eligible_symmetric_quality'])
                          for kind in ('pos', 'hard_neg')},
              'source_items_sha256': digest(self.repo / self.items_rel)})
        qm = {'schema': 'c1-flood-label-quality-v0', 'no_model_scores': True,
              'items_sha256': digest(self.repo / self.items_rel), 'quality_sha256': digest(self.quality / 'quality.jsonl'),
              'code_sha256': digest(self.quality / 'source.py'), 'prereg_sha256': digest(self.quality / 'prereg.json'),
              'summary_sha256': digest(self.quality / 'summary.json'),
              'symmetric_ids_sha256': digest(self.quality / 'symmetric_quality_ids.json'),
              'n': 914, 'selected_ids_sha256': hashlib.sha256(json.dumps(sorted(x['id'] for x in self.targets), separators=(',', ':')).encode()).hexdigest(),
              'c0_manifest_sha256': c0_hashes['manifest_sha256'], 'c0_code_sha256': c0_hashes['code_sha256'],
              'c0_prereg_sha256': c0_hashes['prereg_sha256'],
              'source_files': {'c0_linear_view_probe_v1/' + filename: digest(c0 / filename)
                               for filename in ('manifest.json', 'source.py', 'prereg.json', 'items.jsonl')}}
        for row in self.quality_rows:
            qm['source_files'][row['mask_path']] = row['mask_sha256']
            qm['source_files'][row['valid_path']] = row['valid_sha256']
        write(self.quality / 'manifest.json', qm)
        write(self.catalog, {'records': [{'id': item['tile'], 'dataset': 'kurosiwo', 'split': 'test',
                                         'aoi_id': '1', 'event_date': '2022-09-10'} for item in self.targets]})
        write(self.c1 / 'status.json', {'status': 'complete', 'valid': True})
        (self.c1 / 'source.py').write_text('# Synthetic same-prompt source fixture\n')
        self.selected = {'quality_symmetric': selected_ids,
                         'all_original_targets': sorted(x['id'] for x in self.targets)}
        write(self.c1 / 'selected_ids.json', self.selected)
        self.plan = {'schema': 'c1-same-prompt-flood-analysis-plan-v0',
                     'items_path': self.items_rel, 'inputs_sha256': {}, 'scope': 'synthetic only'}
        self.result = {'schema': 'c1-same-prompt-flood-diagnostic-v0', 'valid': True,
                       'subsets': {}, 'provenance': {}, 'quality_excluded_ids': ['ks_00000_q1_pos'],
                       'blind_consistency': {str(seed): {'consistent': True, 'violations': [],
                                                        'n_reconstructed_prompts': 1} for seed in (1, 2, 3)}}
        for subset, ids in self.selected.items():
            positive_ids = sorted(i for i in ids if i.endswith('_pos'))
            negative_ids = sorted(i for i in ids if i.endswith('_hard'))
            group = {'event': '1', 'pos_ids': positive_ids, 'hard_neg_ids': negative_ids,
                     'n_pos': len(positive_ids), 'n_hard_neg': len(negative_ids), 'slots': ['pre_2', 'post']}
            text = ('These are 2 Sentinel-1 observations of the same area in chronological order, '
                    'taken on 2022-08-29, 2022-09-10: <EO> Did a flood occur between the two observations? '
                    'Answer with yes or no.')
            group.update(reconstructed_prompt=text, prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
            self.result['subsets'][subset] = {'candidate_count': len(ids), 'n_events': 1, 'events': ['1'], 'unsupported_strata': [],
                'supported_n_pos': len(positive_ids), 'supported_n_hard_neg': len(negative_ids),
                'per_seed': {str(seed): {'arms': {arm: {'1': {'strata': [copy.deepcopy(group)]}}
                                             for arm in ('reader', 'blind')}} for seed in (1, 2, 3)}}
        self.reseal_c1()

    def tearDown(self):
        self.tmp.cleanup()

    def reseal_c1(self):
        paths = [self.repo / self.items_rel] + list(self.predictions.values()) + self.code_paths
        self.plan['inputs_sha256'] = {str(path.relative_to(self.repo)): digest(path) for path in paths}
        write(self.c1 / 'analysis_plan.json', self.plan)
        self.result['provenance'] = {'plan_sha256': digest(self.c1 / 'analysis_plan.json'),
            'code_sha256': digest(self.c1 / 'source.py'), 'quality_manifest_sha256': digest(self.quality / 'manifest.json'),
            'quality_sha256': digest(self.quality / 'quality.jsonl'),
            'selected_ids_sha256': digest(self.c1 / 'selected_ids.json'), 'frozen_input_sha256': self.plan['inputs_sha256']}
        write(self.c1 / 'results.json', self.result)

    def build(self):
        return builder.build_cases(self.repo, self.c1, self.quality, self.catalog)

    def test_valid_914_cases_keep_source_answer_and_quality_membership_distinct(self):
        result = self.build()
        self.assertEqual(result['schema_version'], 'eo_reader_cases_v1')
        self.assertEqual(len(result['cases']), 914)
        first = result['cases']['ks_00000']
        self.assertEqual(first['reference_label'], 'yes')
        self.assertEqual(first['reader_predictions']['1'], 'no')
        self.assertEqual(first['run_id'], 'E2')
        self.assertEqual(first['source_role'], 'historical_model_output')
        self.assertFalse(first['c1_membership']['quality_eligible'])
        self.assertFalse(first['c1_membership']['supported_stratum'])
        second = result['cases']['ks_00001']
        self.assertTrue(second['c1_membership']['quality_eligible'])
        self.assertTrue(second['c1_membership']['supported_stratum'])
        self.assertTrue(all(c['question_id'].endswith(('_pos', '_hard')) for c in result['cases'].values()))

    def test_incomplete_or_invalid_c1_is_rejected(self):
        for status, valid in (('running', True), ('complete', False)):
            with self.subTest(status=status, valid=valid):
                write(self.c1 / 'status.json', {'status': status, 'valid': valid})
                with self.assertRaises((ValueError, RuntimeError)):
                    self.build()

    def test_frozen_prediction_bytes_cannot_change(self):
        path = self.predictions[('reader', 1)]
        path.write_text(path.read_text() + '\n')
        with self.assertRaises((ValueError, RuntimeError)):
            self.build()

    def test_missing_duplicate_and_unknown_prediction_ids_are_rejected(self):
        path = self.predictions[('reader', 1)]
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for mutation in ('missing', 'duplicate', 'unknown'):
            with self.subTest(mutation=mutation):
                rows = copy.deepcopy(original)
                if mutation == 'missing': rows.pop()
                if mutation == 'duplicate': rows[-1] = rows[0]
                if mutation == 'unknown': rows[-1]['id'] = 'unknown'
                write_rows(path, rows)
                self.reseal_c1()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.build()

    def test_source_metadata_or_gold_mismatch_is_rejected_even_when_resealed(self):
        path = self.predictions[('reader', 1)]
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for key, value in (('tile', 'ks_99999'), ('text_gold', 'no'), ('emb_item', 'other_id')):
            with self.subTest(key=key):
                rows = copy.deepcopy(original)
                rows[0][key] = value
                write_rows(path, rows)
                self.reseal_c1()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.build()

    def test_unknown_catalog_tile_is_rejected(self):
        catalog = json.loads(self.catalog.read_text())
        catalog['records'] = catalog['records'][1:]
        write(self.catalog, catalog)
        with self.assertRaises((ValueError, RuntimeError)):
            self.build()

    def test_quality_artifact_tamper_is_rejected(self):
        path = self.quality / 'quality.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['eligible_symmetric_quality'] = True
        write_rows(path, rows)
        with self.assertRaises((ValueError, RuntimeError)):
            self.build()

    def test_supported_stratum_membership_tamper_is_rejected(self):
        self.result['subsets']['quality_symmetric']['per_seed']['1']['arms']['reader']['1']['strata'][0]['pos_ids'].pop()
        self.reseal_c1()
        with self.assertRaises((ValueError, RuntimeError)):
            self.build()

    def test_existing_output_cannot_be_overwritten(self):
        out = self.repo / 'existing.json'
        out.write_text('original bytes')
        with self.assertRaises(FileExistsError):
            builder.write_cases({'new': 'payload'}, out)
        self.assertEqual(out.read_text(), 'original bytes')


if __name__ == '__main__':
    unittest.main()
