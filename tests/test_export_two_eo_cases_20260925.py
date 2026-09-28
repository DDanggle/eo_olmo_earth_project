"""Bounded synthetic tests only; no server, real-case export, download or model."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import export_two_eo_cases_20260925 as exporter


class EmbeddedBytesTests(unittest.TestCase):
    def test_exact_original_bytes_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.tortilla'
            source.write_bytes(b'PREFIX' + b'II*\x00TEST_TIFF_BYTES' + b'END')
            for offset, length in [(0, 1), (6, 18), (source.stat().st_size - 1, 1), (0, source.stat().st_size)]:
                data, record = exporter.embedded_bytes(f'/vsisubfile/{offset}_{length},{source}', source)
                expected = source.read_bytes()[offset:offset + length]
                self.assertEqual(data, expected)
                self.assertEqual(record['sha256'], hashlib.sha256(expected).hexdigest())
                self.assertEqual((record['offset'], record['length']), (offset, length))

    def test_outside_zero_negative_truncated_or_oversized_slice_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.tortilla'
            source.write_bytes(b'12345678')
            for extent in ['0_0', '8_1', '9_1', '0_9', '-1_1', '0_-1', '0_16777216']:
                with self.subTest(extent=extent), self.assertRaises(ValueError):
                    exporter.embedded_bytes(f'/vsisubfile/{extent},{source}', source)

    def test_network_another_file_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'data' / 'source.tortilla'
            source.parent.mkdir()
            source.write_bytes(b'12345678')
            other = root / 'other.tortilla'
            other.write_bytes(b'abcdefgh')
            for locator in ['https://example.invalid/data.tif', '/vsicurl/https://example.invalid/data.tif',
                            f'/vsisubfile/0_4,{other}', f'/vsisubfile/0_4,{source.parent}/../other.tortilla']:
                with self.subTest(locator=locator), self.assertRaises(ValueError):
                    exporter.embedded_bytes(locator, source)
            self.assertEqual(source.read_bytes(), b'12345678')
            self.assertEqual(other.read_bytes(), b'abcdefgh')

    def test_symlink_to_different_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, other, link = root/'source', root/'other', root/'link'
            source.write_bytes(b'1234')
            other.write_bytes(b'abcd')
            link.symlink_to(other)
            with self.assertRaises(ValueError):
                exporter.embedded_bytes(f'/vsisubfile/0_4,{link}', source)


class MaskAndCacheTests(unittest.TestCase):
    def test_raw_zero_is_semantic_nowater_not_tiff_nodata(self):
        raw, valid = np.zeros((192, 192), dtype='uint8'), np.ones((192, 192), dtype='uint8')
        result = exporter.mask_summary(raw, valid)
        self.assertEqual(result['labelled_px'], 36864)
        self.assertEqual(result['unknown_px'], 0)
        self.assertEqual(result['flood_px'], 0)
        self.assertEqual(result['flood_fraction_labelled'], 0.)

    def test_unknown_valid_and_permanent_water_are_distinct(self):
        raw, valid = np.zeros((192, 192), dtype='uint8'), np.ones((192, 192), dtype='uint8')
        raw[0, :5] = [0, 1, 2, 3, 2]
        valid[0, 4] = 0
        before = raw.copy(), valid.copy()
        result = exporter.mask_summary(raw, valid)
        self.assertEqual(result['source_valid_px'], 36863)
        self.assertEqual(result['labelled_px'], 36862)
        self.assertEqual(result['unknown_px'], 2)
        self.assertEqual(result['flood_px'], 1)
        self.assertEqual(result['raw_class_counts']['2'], 2)
        self.assertAlmostEqual(result['flood_fraction_labelled'], 1/36862)
        self.assertAlmostEqual(result['flood_fraction_all_crop'], 1/36864)
        self.assertTrue(np.array_equal(before[0], raw) and np.array_equal(before[1], valid))

    def test_all_unknown_reports_null_labelled_fraction(self):
        raw, valid = np.full((192, 192), 3, dtype='uint8'), np.ones((192, 192), dtype='uint8')
        result = exporter.mask_summary(raw, valid)
        self.assertEqual(result['labelled_px'], 0)
        self.assertEqual(result['unknown_px'], 36864)
        self.assertIsNone(result['flood_fraction_labelled'])

    def test_wrong_shape_class_or_validity_rejected(self):
        raw, valid = np.zeros((192, 192), dtype='uint8'), np.ones((192, 192), dtype='uint8')
        with self.assertRaises(ValueError):
            exporter.mask_summary(raw[:191], valid[:191])
        raw[0, 0] = 4
        with self.assertRaises(ValueError):
            exporter.mask_summary(raw, valid)
        raw[0, 0], valid[0, 0] = 0, 2
        with self.assertRaises(ValueError):
            exporter.mask_summary(raw, valid)

    def test_c1_exact_hash_remap_counts_and_failure_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask, valid = np.zeros((192, 192), dtype='uint8'), np.ones((192, 192), dtype='uint8')
            mask[0, :4] = [0, 1, 2, 3]
            np.save(root/'mask.npy', np.array([1, 2, 3, 0], dtype='uint8')[mask])
            np.save(root/'valid.npy', valid)
            quality = {'mask_path': 'mask.npy', 'valid_path': 'valid.npy',
                       'mask_sha256': exporter.sha(root/'mask.npy'), 'valid_sha256': exporter.sha(root/'valid.npy'),
                       'flood_px': 1, 'labelled_px': 36863, 'total_px': 36864}
            self.assertEqual(exporter.validate_c1_cache(root, quality, mask, valid)['flood_px'], 1)
            changed = copy.deepcopy(quality)
            changed['labelled_px'] = 36864
            with self.assertRaisesRegex(ValueError, 'coverage'):
                exporter.validate_c1_cache(root, changed, mask, valid)
            np.save(root/'mask.npy', np.zeros((192, 192), dtype='uint8'))
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                exporter.validate_c1_cache(root, quality, mask, valid)
            changed = copy.deepcopy(quality)
            changed['mask_sha256'] = exporter.sha(root/'mask.npy')
            with self.assertRaisesRegex(ValueError, 'remap differs'):
                exporter.validate_c1_cache(root, changed, mask, valid)
            changed['mask_path'] = '../outside.npy'
            with self.assertRaisesRegex(ValueError, 'outside root'):
                exporter.validate_c1_cache(root, changed, mask, valid)


class FixedCasesTests(unittest.TestCase):
    def test_exact_two_case_policy_including_low_coverage(self):
        self.assertEqual(exporter.CASE_IDS, ('ks_06770', 'ks_05265'))
        row = {'tortilla:data_split': 'test', 'actid': 562, 'aoiid': '13', 'flood_date': '2022-01-29 08:00:00'}
        for sid in exporter.CASE_IDS:
            exporter.check_case_policy(sid, row)
        with self.assertRaises(ValueError):
            exporter.check_case_policy('ks_00001', row)
        for key, value in [('tortilla:data_split', 'train'), ('actid', 123), ('aoiid', '14'), ('flood_date', '2022-02-01')]:
            changed = dict(row, **{key: value})
            with self.assertRaises(ValueError):
                exporter.check_case_policy('ks_06770', changed)
        self.assertIn('do not replace', exporter.SELECTION)

    def test_quality_no_replacement_no_duplicate_no_silent_filter(self):
        records = [{'tile': 'ks_06770', 'eligible_symmetric_quality': False},
                   {'tile': 'ks_05265', 'eligible_symmetric_quality': True},
                   {'tile': 'some_other', 'eligible_symmetric_quality': True}]
        selected = exporter.fixed_quality_rows(records)
        self.assertEqual(set(selected), set(exporter.CASE_IDS))
        self.assertFalse(selected['ks_06770']['eligible_symmetric_quality'])
        for bad in [records[1:], records + [records[0]],
                    [{'tile': 'ks_06770', 'eligible_symmetric_quality': True}, records[1]]]:
            with self.assertRaises(ValueError):
                exporter.fixed_quality_rows(bad)


class RendererTests(unittest.TestCase):
    @staticmethod
    def fixture(all_unknown=False):
        cube = np.full((2, 3, 192, 192), .01, dtype='float32')
        cube[0, 0, 0, 0] = 0
        cube[0, 0, 0, 1] = np.nan
        cube[1, 2, 1, 1] = -1
        mask = np.zeros((192, 192), dtype='uint8')
        mask[:, 96:] = 3
        mask[30:70, 20:60] = 2
        mask[90:130, 30:60] = 1
        if all_unknown:
            mask[:] = 3
        valid = np.ones_like(mask)
        valid[160:180, 10:30] = 0
        meta = {'id': 'SYNTHETIC_FIXTURE_NOT_REAL_CASE', 'event_id': 'FIXTURE', 'event_date': '2000-01-01'}
        return cube, mask, valid, meta

    def test_real_png_generation_legends_outside_pixels_and_values_unmodified(self):
        cube, mask, valid, meta = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'synthetic.png'
            summary = exporter.mask_summary(mask, valid)
            layout = exporter.render(path, cube, mask, valid, meta, summary)
            image = Image.open(path)
            self.assertEqual(image.size, (1840, 1440))
            self.assertEqual(image.mode, 'RGB')
            self.assertEqual(len(layout['data_rectangles']), 8)
            for legend in layout['legend_rectangles']:
                for rect in layout['data_rectangles'].values():
                    intersect = min(legend[2], rect[2]) > max(legend[0], rect[0]) and min(legend[3], rect[3]) > max(legend[1], rect[1])
                    self.assertFalse(intersect, (legend, rect))
            pixels = np.array(image)
            x, y, right, bottom = layout['data_rectangles']['reference_mask']
            palette = np.array([[217,224,229], [53,106,160], [231,120,50], [255,211,78], [48,52,59]], dtype='uint8')
            classes = mask.copy()
            classes[valid != 1] = 4
            expected = np.repeat(np.repeat(palette[classes], 2, 0), 2, 1)
            self.assertTrue(np.array_equal(pixels[y:bottom, x:right], expected), 'Legend/text obscured original mask pixels')
            x, y, right, bottom = layout['data_rectangles']['VV_0']
            self.assertEqual(pixels[y, x].tolist(), [187, 67, 163])
            self.assertEqual(pixels[y, x+2].tolist(), [187, 67, 163])
            self.assertEqual(pixels[y+6, x+6].tolist(), [51, 51, 51])
            self.assertEqual(cube[0, 0, 0, 0], 0)
            self.assertTrue(np.isnan(cube[0, 0, 0, 1]))
            # Retain one clearly synthetic file for local visual QA by reviewer.
            preview = Path('/private/tmp/eo_two_case_export_synthetic_preview.png')
            preview.write_bytes(path.read_bytes())
            Path('/private/tmp/eo_two_case_export_synthetic_layout.json').write_text(json.dumps(layout, indent=2)+'\n')

    def test_all_unknown_renderer_and_incorrect_summary(self):
        cube, mask, valid, meta = self.fixture(all_unknown=True)
        with tempfile.TemporaryDirectory() as tmp:
            summary = exporter.mask_summary(mask, valid)
            path = Path(tmp)/'unknown.png'
            exporter.render(path, cube, mask, valid, meta, summary)
            self.assertTrue(path.is_file())
            changed = dict(summary, unknown_px=1)
            with self.assertRaisesRegex(ValueError, 'summary differs'):
                exporter.render(path, cube, mask, valid, meta, changed)


if __name__ == '__main__':
    unittest.main()
