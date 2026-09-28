"""Synthetic-only checks for the bounded Kuro metadata data-opcode reader.

Does not load either downloaded pickle or execute pickle deserialization.
pickle.dumps creates inert builtin-container fixtures only.
"""
import gzip
import importlib.util
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SOURCE = Path('/private/tmp/kuro_date_metadata_20260925/inspect_grid_metadata.py')
spec = importlib.util.spec_from_file_location('kuro_safe_data_reader', SOURCE)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


class DataOnlyTests(unittest.TestCase):
    def parse(self, raw):
        with tempfile.TemporaryDirectory(prefix='kuro-data-reader-test-') as td:
            path = Path(td) / 'synthetic.gz'
            path.write_bytes(gzip.compress(raw))
            return reader.data_only(path)

    def test_builtin_roundtrip_and_long_memo_reference(self):
        shared = {'date': '2022-02-03', 'weight': 100.0}
        value = {
            'items': [f'item_{i}' for i in range(300)] + [shared, shared],
            'numbers': [0, 255, 256, 65535, 65536, -5, 1.25, -2.5],
            'flags': [True, False],
            'empty': [],
            'empty_mapping': {},
        }
        for protocol in (4, 5):
            with self.subTest(protocol=protocol):
                raw = pickle.dumps(value, protocol=protocol)
                self.assertIn('LONG_BINGET', [op.name for op, _, _ in reader.pickletools.genops(raw)])
                got = self.parse(raw)
                self.assertEqual(got, value)
                self.assertIs(got['items'][-1], got['items'][-2])
                self.assertIs(got['flags'][0], True)
                self.assertIs(got['flags'][1], False)

    def test_callable_and_external_reference_opcodes_rejected(self):
        # Each opcode is rejected before any stack lookup or callable use.
        cases = {
            'GLOBAL': b'cos\nsystem\n.',
            'REDUCE': b'R.',
            'BUILD': b'b.',
            'STACK_GLOBAL': b'\x93.',
            'NEWOBJ': b'\x81.',
            'INST': b'ibuiltins\nlist\n.',
            'PERSID': b'Px\n.',
            'BINPERSID': b'Q.',
        }
        for name, raw in cases.items():
            with self.subTest(opcode=name):
                with self.assertRaisesRegex(ValueError, 'Unsupported opcode ' + name):
                    self.parse(raw)

    def test_unsupported_data_types_rejected(self):
        # Narrow format support is intentional, not a general pickle reader.
        for value in (None, (1, 2), b'bytes', 2**80, 'x' * 300):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ValueError, 'Unsupported opcode'):
                    self.parse(pickle.dumps(value, protocol=4))

    def test_truncated_and_malformed_inputs_fail_closed(self):
        valid = pickle.dumps({'a': [1, 2], 'b': 3.5}, protocol=4)
        for end in (0, 1, 5, len(valid) - 1):
            with self.subTest(truncated_at=end):
                with self.assertRaises((ValueError, IndexError, KeyError)):
                    self.parse(valid[:end])
        for raw in (b'}].', b'a.', b'\x94.', b'h\x00.', b'\x80\x04}(K\x01u.'):
            with self.subTest(malformed=repr(raw)):
                with self.assertRaises((ValueError, IndexError, KeyError, TypeError)):
                    self.parse(raw)
        with tempfile.TemporaryDirectory(prefix='kuro-bad-gzip-') as td:
            path = Path(td) / 'not_gzip.gz'
            path.write_bytes(b'not gzip')
            with self.assertRaises((OSError, EOFError)):
                reader.data_only(path)

    def test_decompression_cap_before_opcode_processing(self):
        # Exercise the cap without allocating/decompressing 200 MB.
        class OverCap(bytes):
            def __len__(self):
                return 200_000_001

        fake = mock.MagicMock()
        fake.__enter__.return_value = fake
        fake.read.return_value = OverCap()
        with mock.patch.object(reader.gzip, 'open', return_value=fake):
            with mock.patch.object(reader.pickletools, 'genops') as genops:
                with self.assertRaisesRegex(ValueError, 'metadata decompression cap'):
                    reader.data_only(Path('unused.gz'))
                fake.read.assert_called_once_with(200_000_001)
                genops.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
