"""Synthetic end-to-end tests of immutable local rendering; no real case data."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import export_two_eo_cases_20260925 as exporter
import render_two_eo_cases_20260925 as renderer


def fixture(root):
    root.mkdir()
    cases = []
    for sid in exporter.CASE_IDS:
        destination = root / sid
        destination.mkdir()
        assets = []
        for index, role in exporter.ASSETS.items():
            path = destination / f'asset_{index}.tif'
            # The exporter already verifies TIFF decoding; the local helper must
            # preserve, hash-check, and never decode/re-encode the original bytes.
            path.write_bytes(b'SYNTHETIC_TIFF_BYTES_NOT_REAL_' + sid.encode() + bytes([index]))
            assets.append({'index': index, 'role': role, 'file': path.name,
                           'sha256': exporter.sha(path),
                           'original_embedded_bytes': {'sha256': exporter.sha(path)}})
        cube = np.full((2, 3, 192, 192), .01, dtype='float32')
        mask = np.zeros((192, 192), dtype='uint8')
        valid = np.ones_like(mask)
        if sid == 'ks_06770':
            mask[:, 96:] = 3
            mask[20:40, 30:50] = 2
        np.savez_compressed(destination / 'evidence_arrays.npz', raw_linear_vv_vh=cube,
            reference_mask_raw=mask, source_valid=valid,
            labelled_valid=((valid == 1) & (mask != 3)).astype('uint8'))
        manifest = {'id': sid, 'split': 'test', 'event_id': 562, 'aoi_id': '13',
                    'event_date': '2022-01-29', 'predictions_included': False,
                    'eligible_C1_symmetric_quality': sid == 'ks_05265',
                    'acquisition_dates': {'pre_1': None, 'pre_2': None, 'post': None},
                    'assets': assets, 'arrays_sha256': exporter.sha(destination / 'evidence_arrays.npz'),
                    'reference_summary': exporter.mask_summary(mask, valid)}
        exporter.write(destination / 'source_manifest.json', manifest)
        cases.append({'id': sid, 'manifest': f'{sid}/source_manifest.json',
                      'manifest_sha256': exporter.sha(destination / 'source_manifest.json'),
                      'preview_file': None, 'eligible_C1_symmetric_quality': sid == 'ks_05265'})
    index = {'schema': 'eo-two-fixed-source-cases-v0', 'status': 'exported',
             'case_ids': list(exporter.CASE_IDS), 'cases': cases,
             'predictions_included': False, 'no_new_model_inference': True,
             'metadata_pins': exporter.PINS}
    exporter.write(root / 'case_index.json', index)
    exporter.write(root / 'selection.json', {'ids': list(exporter.CASE_IDS),
                                           'frozen_metadata_sha256': exporter.PINS})
    return index


def alter_manifest(root, callback):
    index = renderer.read(root / 'case_index.json')
    entry = index['cases'][0]
    path = root / entry['manifest']
    manifest = renderer.read(path)
    callback(manifest)
    exporter.write(path, manifest)
    entry['manifest_sha256'] = exporter.sha(path)
    exporter.write(root / 'case_index.json', index)


class ImmutableRenderingTests(unittest.TestCase):
    def test_complete_new_copy_original_unchanged_and_parent_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, output = Path(tmp) / 'original', Path(tmp) / 'rendered'
            original_index = fixture(root)
            _, _, before = renderer.validate_export(root)
            self.assertEqual(len(before), 16)
            result = renderer.render_copy(root, output)
            self.assertEqual(renderer.validate_export(root)[2], before)
            self.assertEqual(renderer.read(root / 'case_index.json'), original_index)
            self.assertEqual((result['schema'], result['status']), ('eo-two-fixed-source-cases-v0', 'exported'))
            self.assertEqual(result['case_ids'], list(exporter.CASE_IDS))
            self.assertTrue(result['rendered_copy'])
            provenance = result['rendering_provenance']
            self.assertEqual(provenance['source_file_sha256'], before)
            self.assertEqual(provenance['source_case_index_sha256'], before['case_index.json'])
            self.assertEqual(provenance['renderer_module_sha256'], exporter.sha(exporter.__file__))
            self.assertEqual(provenance['helper_sha256'], exporter.sha(renderer.__file__))
            self.assertFalse((output / 'failure.json').exists())
            for entry in result['cases']:
                sid = entry['id']
                manifest_path = output / entry['manifest']
                manifest = renderer.read(manifest_path)
                self.assertEqual(entry['manifest_sha256'], exporter.sha(manifest_path))
                self.assertEqual(entry['parent_source_manifest_sha256'], before[entry['manifest']])
                self.assertNotEqual(entry['manifest_sha256'], before[entry['manifest']])
                self.assertEqual(entry['preview_file'], f'{sid}/preview.png')
                self.assertEqual(manifest['preview_file'], 'preview.png')
                self.assertEqual(entry['preview_sha256'], exporter.sha(output / entry['preview_file']))
                self.assertEqual(manifest['preview_sha256'], entry['preview_sha256'])
                with Image.open(output / entry['preview_file']) as image:
                    self.assertEqual(image.size, (1840, 1440))
                self.assertEqual(manifest['rendering_provenance'], provenance)
                self.assertIs(manifest['eligible_C1_symmetric_quality'], sid == 'ks_05265')
                for rel, digest in before.items():
                    if rel.startswith(sid + '/') and not rel.endswith('source_manifest.json'):
                        self.assertEqual(exporter.sha(output / rel), digest)

    def test_existing_or_overlapping_output_rejected_without_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'original'
            fixture(root)
            before = renderer.validate_export(root)[2]
            existing = Path(tmp) / 'existing'
            existing.mkdir()
            (existing / 'sentinel').write_text('unchanged')
            for output in [root, root / 'inside', Path(tmp), existing]:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    renderer.render_copy(root, output)
            self.assertEqual(renderer.validate_export(root)[2], before)
            self.assertEqual(list(existing.iterdir()), [existing / 'sentinel'])

    def test_failed_source_and_extra_file_rejected_before_destination_created(self):
        for name in ['failure.json', 'unexpected.txt', 'ks_06770/failure.json']:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root, output = Path(tmp) / 'original', Path(tmp) / 'new'
                fixture(root)
                (root / name).write_text('{}')
                with self.assertRaises(ValueError):
                    renderer.render_copy(root, output)
                self.assertFalse(output.exists())

    def test_paths_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'original'
            fixture(root)
            outside = Path(tmp) / 'outside'
            outside.write_text('unchanged')
            for name in ['../outside', str(outside), 'missing']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    renderer.bounded_file(root, name)
            (root / 'link').symlink_to(outside)
            with self.assertRaises(ValueError):
                renderer.validate_export(root)
            with self.assertRaises(ValueError):
                renderer.bounded_file(root, 'link')
            self.assertEqual(outside.read_text(), 'unchanged')

    def test_array_and_tiff_hash_corruption_rejected(self):
        for filename in ['evidence_arrays.npz', 'asset_0.tif', 'source_manifest.json']:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                root, output = Path(tmp) / 'original', Path(tmp) / 'new'
                fixture(root)
                path = root / exporter.CASE_IDS[0] / filename
                path.write_bytes(path.read_bytes() + b'changed')
                with self.assertRaises(ValueError):
                    renderer.render_copy(root, output)
                self.assertFalse(output.exists())

    def test_rehashed_but_invalid_semantics_rejected(self):
        callbacks = [lambda m: m.update(event_id=123),
                     lambda m: m.update(eligible_C1_symmetric_quality=True),
                     lambda m: m['acquisition_dates'].update(pre_1='2022-01-17'),
                     lambda m: m['reference_summary'].update(flood_px=999),
                     lambda m: m['assets'][0].update(role='post_event')]
        for callback in callbacks:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'original'
                fixture(root)
                alter_manifest(root, callback)
                with self.assertRaises(ValueError):
                    renderer.validate_export(root)

    def test_missing_case_order_or_index_quality_cannot_silently_change(self):
        for mutate in [lambda i: i['cases'].pop(), lambda i: i['cases'].reverse(),
                       lambda i: i['cases'][0].update(eligible_C1_symmetric_quality=True),
                       lambda i: i.update(case_ids=['ks_05265', 'ks_06770'])]:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / 'original'
                fixture(root)
                index = renderer.read(root / 'case_index.json')
                mutate(index)
                exporter.write(root / 'case_index.json', index)
                with self.assertRaises(ValueError):
                    renderer.validate_export(root)

    def test_inconsistent_labelled_support_rejected_even_with_fresh_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'original'
            fixture(root)
            path = root / exporter.CASE_IDS[0] / 'evidence_arrays.npz'
            with np.load(path, allow_pickle=False) as source:
                arrays = {key: source[key] for key in source.files}
            arrays['labelled_valid'][:] = 1
            np.savez_compressed(path, **arrays)
            alter_manifest(root, lambda m: m.update(arrays_sha256=exporter.sha(path)))
            with self.assertRaisesRegex(ValueError, 'Known-support'):
                renderer.validate_export(root)


if __name__ == '__main__':
    unittest.main()
