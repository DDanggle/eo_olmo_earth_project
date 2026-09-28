"""Synthetic-only tests: no tacoreader installation, image, or server needed."""
from contextlib import ExitStack
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('t0_exporter', '/private/tmp/export_kuro_metadata_t0_20260925.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Table:
    def __init__(self, rows):
        self.rows = rows
        self.columns = list(rows[0])
        self.dtypes = {key: 'synthetic object' for key in self.columns}

    def iterrows(self):
        yield from enumerate(self.rows)

    def read(self, *args, **kwargs):
        raise AssertionError('Child/image reads are forbidden')


def fixture_rows():
    rows, refs = [], []
    for i, split in enumerate(('train', 'validation', 'test')):
        x, y = -1120 + i * 2240, 1120
        lon = math.degrees((x+1120) / 6378137.0)
        center = f'POINT ({lon:.6f} 0.000000)'
        row = {'tortilla:id': str(i), 'tortilla:data_split': split,
               'stac:crs': 'EPSG:3857', 'stac:geotransform': [x, 10, 0, y, 0, -10],
               'stac:raster_shape': [224, 224], 'stac:centroid': center,
               'stac:time_start': '1643414400.0', 'stac:time_end': '1643414400.0',
               'actid': '562', 'aoiid': '013', 'flood_date': '2022-01-29 07:00:00',
               'pwater': '6.25', 'pflood': '23.125', 'unselected:raw': 'keep\nall; λ strings'}
        refs.append({'id': f'ks_{i:05d}', 'split': split, 'actid': 562, 'aoiid': '13',
                     'flood_date': '2022-01-29', 'pflood': 23.125,
                     'pwater': 6.25, 'centroid': center})
        rows.append(row)
    return rows, refs


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='kuro-t0-tests-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'geobench2/kurosiwo/kurosiwo/geobench_kuro_siwo.tortilla'
        self.source.parent.mkdir(parents=True)
        self.footer = b'PAR1synthetic metadata onlyPAR1'
        header = b'#y' + (300).to_bytes(8, 'little') + len(self.footer).to_bytes(8, 'little')
        self.container = header + b'B' * (300-len(header)) + self.footer
        self.source.write_bytes(self.container)
        self.rows, self.refs = fixture_rows()
        self.meta = self.root / 'kurosiwo_s1_cache/meta.jsonl'
        self.meta.parent.mkdir()
        self.meta.write_bytes(b''.join(module.encode(r) for r in self.refs))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.object(module, 'N', 3))
        self.stack.enter_context(mock.patch.object(module, 'SPLITS', {'train': 1, 'validation': 1, 'test': 1}))
        self.stack.enter_context(mock.patch.object(module, 'META_SHA', module.file_sha(self.meta)))
        self.reference = module.validate_reference(self.refs)

    def captured(self, row=None):
        return module.capture_row(row or self.rows[0], 0)

    def test_normalization_scientific_gdal_order_full_precision_and_closed_bbox(self):
        row = copy.deepcopy(self.rows[0])
        row['stac:geotransform'] = '[-1.120e+3 1e1 0 1.120e3\n0 -10]'
        row['stac:raster_shape'] = '[224, 224]'
        got = module.normalize(self.captured(row), self.reference)
        self.assertEqual(got['gdal_geotransform'], [-1120, 10, 0, 1120, 0, -10])
        self.assertEqual(got['bbox_epsg3857'], [-1120, -1120, 1120, 1120])
        self.assertEqual(got['corners_epsg3857'], [[-1120, 1120], [1120, 1120],
                                                  [1120, -1120], [-1120, -1120], [-1120, 1120]])
        self.assertEqual((got['source_id'], got['aoi_id']), (0, 13))
        self.assertEqual(got['source_row']['aoiid'], '013')
        self.assertEqual(got['source_row']['unselected:raw'], self.rows[0]['unselected:raw'])
        self.assertIsNone(got['acquisition_dates'])
        # Numeric array/list representation is preserved independently from str.
        captured = self.captured(self.rows[0])
        self.assertEqual(captured['source_numeric_vectors']['stac:geotransform'],
                         ['-1120', '10', '0', '1120', '0', '-10'])

    def test_geometry_nonfinite_and_partial_string_parsing_rejected(self):
        cases = [
            ('stac:crs', 'EPSG:4326'), ('stac:raster_shape', '[192 192]'),
            ('stac:geotransform', '[-1120 10 1 1120 0 -10]'),
            ('stac:geotransform', '[-1120 20 0 1120 0 -10]'),
            ('stac:geotransform', '[-1120 10 0 1120 0 10]'),
            ('stac:geotransform', '[-1120 10 0 1120 0 -10] ignored'),
            ('stac:geotransform', '[-1120 10 0 1120 0 NaN]'),
            ('stac:geotransform', '[nan 10 0 1120 0 -10]'),
            ('stac:geotransform', '[1e999 10 0 1120 0 -10]'),
            ('stac:geotransform', '[-1110 10 0 1120 0 -10]'),
            ('stac:time_end', '1'), ('stac:centroid', 'POINT Z (0 0 1)'),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                row = copy.deepcopy(self.rows[0]); row[field] = value
                with self.assertRaises(ValueError):
                    module.normalize(self.captured(row), self.reference)

    def test_identity_date_and_label_metadata_tamper_rejected(self):
        cases = [('tortilla:id', '3'), ('tortilla:id', '0.0'), ('tortilla:data_split', 'test'),
                 ('actid', '563'), ('aoiid', '14'), ('pflood', '23.2'), ('pwater', '-1'),
                 ('flood_date', '2022-01-30 07:00:00'), ('flood_date', '2022-02-30'),
                 ('stac:centroid', 'POINT (0.00001 0)')]
        for field, value in cases:
            with self.subTest(field=field):
                row = copy.deepcopy(self.rows[0]); row[field] = value
                with self.assertRaises(ValueError):
                    module.normalize(self.captured(row), self.reference)
        refs = copy.deepcopy(self.refs); refs[-1]['id'] = refs[0]['id']
        with self.assertRaisesRegex(ValueError, 'Duplicate reference ID'):
            module.validate_reference(refs)

    def test_header_footer_bounded_exact_bytes_and_bad_ranges(self):
        header, footer, info = module.top_metadata_bytes(self.source)
        self.assertEqual((len(header), footer, info['offset']), (18, self.footer, 300))
        self.assertEqual(info['sha256'], hashlib.sha256(self.footer).hexdigest())
        for offset, length in ((0, 8), (300, 1000), (300, module.MAX_FOOTER_BYTES+1), (300, 4)):
            with self.subTest(offset=offset, length=length):
                self.source.write_bytes(b'#y'+offset.to_bytes(8,'little')+length.to_bytes(8,'little')+b'X'*400)
                with self.assertRaises(ValueError):
                    module.top_metadata_bytes(self.source)

    def test_success_preserves_all_rows_and_hashes_without_child_reads(self):
        out = self.root / 'output'
        summary = module.export(self.root, out, loader=lambda path: Table(self.rows))
        self.assertTrue(summary['valid'])
        self.assertEqual(summary['n_rows'], 3)
        self.assertEqual(self.source.read_bytes(), self.container)
        manifest = json.loads((out/'manifest.json').read_text())
        for filename, digest in manifest['files_sha256'].items():
            self.assertEqual(module.file_sha(out/filename), digest)
        self.assertIsNone(manifest['container_full_sha256'])
        self.assertEqual((out/'footer.parquet').read_bytes(), self.footer)
        raw = module.read_jsonl_bytes((out/'raw_rows.jsonl').read_bytes())
        normalized = module.read_jsonl_bytes((out/'metadata.jsonl').read_bytes())
        self.assertEqual(len(raw), len(normalized))
        self.assertEqual(raw[2]['source_row'], normalized[2]['source_row'])
        self.assertEqual(raw[2]['source_row']['unselected:raw'], self.rows[2]['unselected:raw'])
        self.assertEqual(json.loads((out/'status.json').read_text())['manifest_sha256'], module.file_sha(out/'manifest.json'))

    def test_any_row_or_coverage_mismatch_invalid_preserving_all_raw_rows(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['stac:geotransform'][2] = 1
        rows[-1]['tortilla:id'] = '1'
        out = self.root / 'invalid'
        with self.assertRaisesRegex(ValueError, 'T0 invalid'):
            module.export(self.root, out, loader=lambda path: Table(rows))
        self.assertEqual(len(module.read_jsonl_bytes((out/'raw_rows.jsonl').read_bytes())), 3)
        self.assertEqual(len(module.read_jsonl_bytes((out/'metadata.jsonl').read_bytes())), 3)
        self.assertFalse(json.loads((out/'manifest.json').read_text())['valid'])
        self.assertFalse(json.loads((out/'status.json').read_text())['valid'])
        self.assertTrue((out/'failure.json').is_file())
        mismatches = json.loads((out/'mismatches.json').read_text())
        self.assertTrue(any(x.get('missing') == [2] and x.get('duplicates') == {'1': 2} for x in mismatches))

    def test_changed_reference_and_existing_output_fail_without_load_or_overwrite(self):
        self.meta.write_bytes(self.meta.read_bytes()+b'\n')
        out = self.root / 'bad_reference'
        loader = mock.Mock()
        with self.assertRaisesRegex(ValueError, 'Pinned cache metadata SHA mismatch'):
            module.export(self.root, out, loader=loader)
        loader.assert_not_called()
        before = (out/'failure.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Output exists'):
            module.export(self.root, out, loader=loader)
        self.assertEqual((out/'failure.json').read_bytes(), before)

    def test_source_mutation_fails_and_preserves_raw_export(self):
        def mutated(path):
            self.source.write_bytes(self.container+b'changed')
            return Table(self.rows)
        out = self.root / 'mutated'
        with self.assertRaisesRegex(ValueError, 'stat changed'):
            module.export(self.root, out, loader=mutated)
        self.assertEqual(len(module.read_jsonl_bytes((out/'raw_rows.jsonl').read_bytes())), 3)
        self.assertEqual(json.loads((out/'failure.json').read_text())['phase'], 'final_source_verification')
        self.assertFalse((out/'manifest.json').exists())

    def test_time_limit_raises_and_restores_prior_handler(self):
        old = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(TimeoutError):
            with module.time_limit(5):
                signal.raise_signal(signal.SIGALRM)
        self.assertEqual(signal.getsignal(signal.SIGALRM), old)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
