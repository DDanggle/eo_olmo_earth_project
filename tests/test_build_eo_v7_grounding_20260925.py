"""Independent v7 builder tests using tiny synthetic source exports only.

Frozen repository metadata and the existing v6 payload are read-only inputs.
The fake TIFF/NPZ/PNG bytes test provenance contracts, not raster decoding.
"""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / 'code/build_eo_v7_grounding_20260925.py'
spec = importlib.util.spec_from_file_location('v7_grounding_builder_under_test', MODULE)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'artifacts/eo_evidence_search_v6_20260925'
PREPARED = REPO / 'artifacts/e5_prepared_review_20260925'


def write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')


def polygon(x, y, radius):
    return {'type': 'Polygon', 'coordinates': [[[x - radius, y - radius], [x + radius, y - radius],
              [x + radius, y + radius], [x - radius, y + radius], [x - radius, y - radius]]]}


class GroundingBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.export = self.root / 'export'
        self.export.mkdir()
        self.out = self.root / 'v7'
        self.base = BASE
        self.plan = builder.read(REPO / 'code/e5_bundle_v0/e5_equal_budget_prereg_v0.json')
        frozen = REPO / 'code/e5_bundle_v0/frozen_inputs'
        self.c0 = builder.read(frozen / 'c0_manifest.json')
        self.quality = {row['tile']: row for row in builder.rows(frozen / 'c1_quality.jsonl')}
        self.e3 = builder.rows(frozen / 'e3_items.jsonl')
        self.catalog = builder.read(BASE / 'catalog.json')
        entries = []
        for sid in builder.CASE_IDS:
            directory = self.export / sid
            directory.mkdir()
            row = next(row for row in self.catalog['records'] if row['id'] == sid)
            quality = self.quality[sid]
            x, y = row['point']
            assets = []
            for index, role in builder.ASSET_ROLES.items():
                path = directory / f'asset_{index}.tif'
                path.write_bytes(f'SYNTHETIC TIFF BYTES {sid} {index}'.encode())
                digest = builder.sha(path)
                assets.append({'index': index, 'role': role, 'file': path.name, 'sha256': digest,
                               'original_embedded_bytes': {'sha256': digest}})
            (directory / 'evidence_arrays.npz').write_bytes(b'SYNTHETIC NPZ CONTRACT FIXTURE')
            (directory / 'preview.png').write_bytes(b'\x89PNG\r\n\x1a\nSYNTHETIC PREVIEW CONTRACT FIXTURE')
            counts = quality['class_valid_counts']
            total, known, flood = quality['total_px'], quality['labelled_px'], quality['flood_px']
            summary = {'total_px': total, 'labelled_px': known, 'flood_px': flood, 'unknown_px': total - known,
                'labelled_fraction': known / total, 'flood_fraction_all_crop': flood / total,
                'flood_fraction_labelled': flood / known,
                'raw_class_counts': {str(raw): sum(counts[f'mask_{remap}|valid_{v}'] for v in (0, 1))
                                     for raw, remap in enumerate((1, 2, 3, 0))},
                'source_valid_px': sum(counts[f'mask_{k}|valid_1'] for k in range(4))}
            source = {'id': sid, 'event_id': 562, 'aoi_id': '13', 'split': 'test', 'event_date': '2022-01-29',
                'predictions_included': False, 'npys_equal_original_tiff_values': True, 'c1_mask_valid_hashes_verified': True,
                'acquisition_dates': {'pre_1': None, 'pre_2': None, 'post': None},
                'model_crop': {'rows': [16, 208], 'columns': [16, 208], 'shape': [192, 192]},
                'geometry': {'whole_tile': polygon(x, y, .02), 'model_crop': polygon(x, y, .01)},
                'centroid_wkt_from_source': f'POINT ({x} {y})',
                'source_row': {'pflood': row['flood_fraction_pct'], 'pwater': row['permanent_water_fraction_pct']},
                'qa_items': [item for item in self.e3 if item['tile'] == sid],
                'eligible_C1_symmetric_quality': quality['eligible_symmetric_quality'], 'reference_summary': summary,
                'model_embedding_sha256': self.c0['cache_sha256'][str(Path(self.c0['root']) / f'kurosiwo_s1_cache/single_fp16/{sid}.npy')],
                'assets': assets, 'arrays_sha256': builder.sha(directory / 'evidence_arrays.npz'),
                'preview_file': 'preview.png', 'preview_sha256': builder.sha(directory / 'preview.png')}
            write(directory / 'source_manifest.json', source)
            entries.append({'id': sid, 'manifest': f'{sid}/source_manifest.json',
                            'manifest_sha256': builder.sha(directory / 'source_manifest.json'),
                            'preview_file': f'{sid}/preview.png',
                            'eligible_C1_symmetric_quality': quality['eligible_symmetric_quality']})
        self.index = {'schema': 'eo-two-fixed-source-cases-v0', 'status': 'exported', 'case_ids': list(builder.CASE_IDS),
                      'predictions_included': False, 'no_new_model_inference': True,
                      'metadata_pins': {'e3_items': self.plan['parents']['e3_items.jsonl'],
                                        'quality': self.plan['parents']['c1_quality.jsonl'],
                                        'c0_manifest': self.plan['parents']['c0_manifest.json']}, 'cases': entries}
        write(self.export / 'case_index.json', self.index)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, *, base=None, prepared=None, out=None):
        return builder.build(REPO, base or self.base, self.export, prepared or PREPARED, out or self.out)

    def change_source(self, function, sid='ks_06770'):
        path = self.export / sid / 'source_manifest.json'
        source = builder.read(path)
        function(source)
        write(path, source)
        index = builder.read(self.export / 'case_index.json')
        next(entry for entry in index['cases'] if entry['id'] == sid)['manifest_sha256'] = builder.sha(path)
        write(self.export / 'case_index.json', index)

    def test_success_preserves_v6_predictions_and_adds_only_two_grounded_previews(self):
        before = {str(path.relative_to(BASE)): builder.sha(path) for path in BASE.rglob('*') if path.is_file()}
        report = self.build()
        self.assertEqual(report['case_ids'], ['ks_06770', 'ks_05265'])
        self.assertTrue(report['reader_cases_byte_identical'])
        self.assertFalse(report['e5_outcomes_included'])
        allowed = {'catalog.json', 'meta.json', 'research_runs.json', 'research_sources.json', 'index.html'}
        for relative, digest in before.items():
            self.assertEqual(builder.sha(BASE / relative), digest, relative)
            if relative not in allowed:
                self.assertEqual(builder.sha(self.out / relative), digest, relative)
        updated = builder.read(self.out / 'catalog.json')
        source_records = {row['id']: row for row in self.catalog['records']}
        for row in updated['records']:
            if row['id'] not in builder.CASE_IDS:
                self.assertEqual(row, source_records[row['id']])
                continue
            expected = copy.deepcopy(row)
            expected['evidence'] = []
            self.assertEqual(expected, source_records[row['id']])
            self.assertEqual(len(row['evidence']), 1)
            evidence = row['evidence'][0]
            self.assertEqual(evidence['source_role'], 'original_observations_and_reference_labels')
            self.assertIn('source valid=1', evidence['caption'])
            self.assertIn('라벨 확인 가능', evidence['caption'])
            self.assertIn('실제 취득일은 미검증', evidence['caption'])
            self.assertEqual((self.out / evidence['image_url'].lstrip('/')).read_bytes(),
                             (self.export / row['id'] / 'preview.png').read_bytes())
        old_research = builder.read(BASE / 'research_runs.json')
        research = builder.read(self.out / 'research_runs.json')
        self.assertEqual(research['runs'][:-1], old_research['runs'])
        pending = research['runs'][-1]
        self.assertEqual(pending['id'], 'E5-EB')
        self.assertNotIn('metrics', pending)
        self.assertIn('결과 없음', pending['status'])
        self.assertTrue(any(builder.SNAPSHOT_AT in finding for finding in pending['findings']))
        self.assertEqual(builder.read(self.out / 'meta.json')['preview_count'], builder.read(BASE / 'meta.json')['preview_count'] + 2)

    def test_source_sidecar_keeps_old_keys_and_links_every_verified_input(self):
        report = self.build()
        old = builder.read(BASE / 'research_sources.json')
        current = builder.read(self.out / 'research_sources.json')
        self.assertEqual({key: current[key] for key in old}, old)
        added = current['source_grounding_v7']
        self.assertEqual(added['e5_status_as_of'], '2026-09-25T06:14:07Z')
        self.assertFalse(added['model_outcomes_included'])
        self.assertEqual(added['sources'], report['source_sha256'])
        for path, digest in added['sources'].items():
            self.assertEqual(builder.sha(path), digest)
        for sid in builder.CASE_IDS:
            self.assertEqual(builder.sha(self.out / f'source_grounding_v7/{sid}/source_manifest.json'),
                             builder.sha(self.export / sid / 'source_manifest.json'))
        self.assertEqual(builder.sha(self.out / 'source_grounding_v7/e5_prepared_review/manifest.json'), builder.PREPARED_SHA)

    def test_disk_and_real_api_function_readback_accepts_the_new_snapshot(self):
        path = Path('/private/tmp/verify_eo_v7_grounding_20260925.py')
        spec = importlib.util.spec_from_file_location('v7_readback_under_test', path)
        readback = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(readback)
        self.build()
        report = readback.verify(REPO, BASE, self.out, self.export)
        self.assertTrue(report['valid'])
        self.assertTrue(report['api_function_readback'])
        self.assertFalse(report['http_or_browser_checked'])
        self.assertFalse(report['e5_outcomes_included'])

    def test_missing_failed_or_wrong_case_export_is_rejected_before_copy(self):
        original = (self.export / 'case_index.json').read_bytes()
        for kind in ('missing', 'failed', 'wrong_id', 'prediction_contamination'):
            with self.subTest(kind=kind):
                (self.export / 'case_index.json').write_bytes(original)
                failure = self.export / 'failure.json'
                if failure.exists():
                    failure.unlink()
                if kind == 'missing':
                    (self.export / 'case_index.json').unlink()
                elif kind == 'failed':
                    failure.write_text('{}')
                else:
                    index = builder.read(self.export / 'case_index.json')
                    if kind == 'wrong_id':
                        index['cases'][0]['id'] = 'ks_99999'
                    else:
                        index['predictions_included'] = True
                    write(self.export / 'case_index.json', index)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.build()
                self.assertFalse(self.out.exists())

    def test_manifest_and_each_export_asset_hash_tampering_is_rejected(self):
        directory = self.export / 'ks_06770'
        for name in ('source_manifest.json', 'asset_0.tif', 'asset_1.tif', 'asset_2.tif', 'asset_4.tif',
                     'asset_5.tif', 'evidence_arrays.npz', 'preview.png'):
            with self.subTest(name=name):
                path = directory / name
                original = path.read_bytes()
                path.write_bytes(original + b'changed')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    self.build()
                self.assertFalse(self.out.exists())
                path.write_bytes(original)

    def test_rehashed_wrong_png_scope_embedding_dates_and_quality_are_rejected(self):
        original_manifest = (self.export / 'ks_06770/source_manifest.json').read_bytes()
        original_index = (self.export / 'case_index.json').read_bytes()
        changes = {
            'id': lambda value: value.update(id='ks_05265'),
            'split': lambda value: value.update(split='train'),
            'model_answer': lambda value: value.update(predictions_included=True),
            'embedding': lambda value: value.update(model_embedding_sha256='0' * 64),
            'dates': lambda value: value.update(acquisition_dates={'pre_1': '2022-01-01', 'pre_2': None, 'post': None}),
            'quality': lambda value: value.update(eligible_C1_symmetric_quality=True),
            'qa_items': lambda value: value.update(qa_items=[]),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name):
                (self.export / 'ks_06770/source_manifest.json').write_bytes(original_manifest)
                (self.export / 'case_index.json').write_bytes(original_index)
                self.change_source(mutate)
                with self.assertRaises(ValueError):
                    self.build()
        (self.export / 'ks_06770/source_manifest.json').write_bytes(original_manifest)
        (self.export / 'case_index.json').write_bytes(original_index)
        png = self.export / 'ks_06770/preview.png'
        png.write_bytes(b'not a PNG')
        self.change_source(lambda value: value.update(preview_sha256=builder.sha(png)))
        with self.assertRaisesRegex(ValueError, 'Expected PNG'):
            self.build()

    def test_rehashed_reference_stats_and_geometry_disagreement_are_rejected(self):
        original_manifest = (self.export / 'ks_06770/source_manifest.json').read_bytes()
        original_index = (self.export / 'case_index.json').read_bytes()
        changes = {
            'pixel': lambda value: value['reference_summary'].update(flood_px=value['reference_summary']['flood_px'] + 1),
            'fraction': lambda value: value['reference_summary'].update(flood_fraction_labelled=.999),
            'raw_class': lambda value: value['reference_summary']['raw_class_counts'].update({'0': value['reference_summary']['raw_class_counts']['0'] + 1}),
            'source_valid': lambda value: value['reference_summary'].update(source_valid_px=0),
            'catalog_fraction': lambda value: value['source_row'].update(pflood=-1),
            'centroid': lambda value: value.update(centroid_wkt_from_source='POINT (0 0)'),
            'crop_contains': lambda value: value['geometry'].update(model_crop=copy.deepcopy(value['geometry']['whole_tile'])),
            'closed': lambda value: value['geometry']['model_crop']['coordinates'][0].pop(),
        }
        for name, mutate in changes.items():
            with self.subTest(name=name):
                (self.export / 'ks_06770/source_manifest.json').write_bytes(original_manifest)
                (self.export / 'case_index.json').write_bytes(original_index)
                self.change_source(mutate)
                with self.assertRaises(ValueError):
                    self.build()
                self.assertFalse(self.out.exists())

    def test_wrong_prepared_manifest_and_unsafe_asset_path_are_rejected(self):
        prepared = self.root / 'prepared'
        prepared.mkdir()
        (prepared / 'manifest.json').write_bytes((PREPARED / 'manifest.json').read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.build(prepared=prepared)
        self.change_source(lambda value: value['assets'][0].update(file='../outside.tif'))
        with self.assertRaisesRegex(ValueError, 'Asset file differs'):
            self.build()
        self.assertFalse(self.out.exists())

    def test_existing_output_and_existing_evidence_are_never_overwritten(self):
        self.out.mkdir()
        marker = self.out / 'keep.txt'
        marker.write_text('do not overwrite')
        with self.assertRaisesRegex(ValueError, 'exclusive output'):
            self.build()
        self.assertEqual(marker.read_text(), 'do not overwrite')
        base = self.root / 'base'
        shutil.copytree(BASE, base)
        catalog = builder.read(base / 'catalog.json')
        next(row for row in catalog['records'] if row['id'] == 'ks_06770')['evidence'] = [{'caption': 'preexisting'}]
        write(base / 'catalog.json', catalog)
        with self.assertRaisesRegex(ValueError, 'must not be replaced'):
            self.build(base=base, out=self.root / 'another_v7')
        self.assertFalse((self.root / 'another_v7').exists())


if __name__ == '__main__':
    unittest.main()
