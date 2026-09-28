"""Identity faults, including actual CLI refusal before any torch/model import."""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from input_identity import (REQUIRED_SOURCE_FILES, read_pinned_json, sha256,
                            verify_episode_tree, verify_source_tree)

ROOT = Path(__file__).absolute().parent


def freeze(root, name):
    files = [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size,
              'sha256': sha256(p)} for p in sorted(root.rglob('*'))
             if p.is_file() and p.name != name and '__pycache__' not in p.parts]
    (root / name).write_text(json.dumps({'files': files}, sort_keys=True))
    return sha256(root / name)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp' if Path('/private/tmp').exists() else None)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / 'episodes'
        (self.root / 'scoring').mkdir(parents=True)
        self.payloads = {'episode_contract.json': b'{"initial_candidate_positions":[2,5]}',
                         'episodes_train.jsonl': b'{"episode_id":"train:k1"}\n',
                         'scoring/scoring_train.jsonl': b'{"target":"synthetic"}\n',
                         'support_masks/example.npz': b'synthetic mask; not decoded'}
        for name, content in self.payloads.items():
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        self.pin = freeze(self.root, 'export_manifest.json')

    def test_all_files_verified_without_semantic_parsing(self):
        receipt = verify_episode_tree(self.root, self.pin)
        self.assertEqual(receipt['payload_files_verified'], 4)
        self.assertFalse(receipt['semantic_data_read'])

    def test_payload_edit_refused(self):
        p = self.root / 'scoring/scoring_train.jsonl'
        p.write_bytes(b'x' * p.stat().st_size)
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            verify_episode_tree(self.root, self.pin)

    def test_catalog_contract_scoring_and_manifest_cotamper_refused(self):
        for name in self.payloads:
            (self.root / name).write_bytes(b'jointly rewritten')
        freeze(self.root, 'export_manifest.json')
        with self.assertRaisesRegex(ValueError, 'External manifest SHA256 mismatch'):
            verify_episode_tree(self.root, self.pin)

    def test_missing_file_refused(self):
        (self.root / 'scoring/scoring_train.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing regular file'):
            verify_episode_tree(self.root, self.pin)

    def test_omitted_required_entry_refused_even_with_new_pin(self):
        (self.root / 'scoring/scoring_train.jsonl').unlink()
        new_pin = freeze(self.root, 'export_manifest.json')
        with self.assertRaisesRegex(ValueError, 'Required identity files omitted'):
            verify_episode_tree(self.root, new_pin)

    def test_unlisted_file_refused(self):
        (self.root / 'extra.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            verify_episode_tree(self.root, self.pin)

    def test_payload_and_parent_and_root_symlinks_refused(self):
        payload = self.root / 'scoring/scoring_train.jsonl'
        original = payload.read_bytes()
        outside = self.root.parent / 'outside.jsonl'
        outside.write_bytes(original)
        payload.unlink()
        payload.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Symlink forbidden'):
            verify_episode_tree(self.root, self.pin)
        payload.unlink()
        payload.write_bytes(original)
        target = self.root.parent / 'scoring_real'
        (self.root / 'scoring').rename(target)
        (self.root / 'scoring').symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink forbidden'):
            verify_episode_tree(self.root, self.pin)
        (self.root / 'scoring').unlink()
        target.rename(self.root / 'scoring')
        alias = self.root.parent / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink forbidden'):
            verify_episode_tree(alias, self.pin)

    def test_traversal_absolute_noncanonical_and_duplicate_paths_refused(self):
        original = json.loads((self.root / 'export_manifest.json').read_text())
        for name in ('../outside', '/etc/hosts', 'scoring/../episode_contract.json',
                     './episode_contract.json', 'scoring//scoring_train.jsonl', 'x\\y'):
            value = json.loads(json.dumps(original))
            value['files'][0]['path'] = name
            manifest = self.root / 'export_manifest.json'
            manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'path escape/noncanonical'):
                verify_episode_tree(self.root, sha256(manifest))
        value = json.loads(json.dumps(original))
        value['files'].append(dict(value['files'][0]))
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'Duplicate/self'):
            verify_episode_tree(self.root, sha256(manifest))

    def test_bytes_mismatch_refused(self):
        manifest = self.root / 'export_manifest.json'
        value = json.loads(manifest.read_text())
        value['files'][0]['bytes'] += 1
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'size mismatch'):
            verify_episode_tree(self.root, sha256(manifest))

    def test_external_pin_missing_and_json_duplicate_refused(self):
        manifest = self.root / 'export_manifest.json'
        with self.assertRaisesRegex(ValueError, 'External expected'):
            read_pinned_json(manifest, None)
        manifest.write_text('{"files": [], "files": []}')
        with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
            read_pinned_json(manifest, sha256(manifest))

    def test_source_helper_tamper_and_unlisted_source_refused(self):
        source = self.root.parent / 'source'
        for name in REQUIRED_SOURCE_FILES:
            p = source / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('# synthetic source\n')
        pin = freeze(source, 'source_manifest.json')
        self.assertEqual(verify_source_tree(source, pin)['payload_files_verified'], len(REQUIRED_SOURCE_FILES))
        (source / 'input_identity.py').write_text('# changed\n')
        with self.assertRaises(ValueError):
            verify_source_tree(source, pin)
        (source / 'input_identity.py').write_text('# synthetic source\n')
        (source / 'unexpected.py').write_text('# extra source\n')
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            verify_source_tree(source, pin)

    def test_actual_cli_identity_only_passes_without_torch(self):
        identity = self.root.parent / 'identity.json'
        identity.write_text(json.dumps({'episodes_export_manifest_sha256': self.pin}))
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'check_connection.py'),
            '--identity-only', '--source-manifest-sha256', sha256(ROOT / 'source_manifest.json'),
            '--identity-manifest', str(identity), '--identity-manifest-sha256', sha256(identity),
            '--episodes', str(self.root)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)['gpu_started'])

    def test_actual_execute_refuses_changed_identity_before_torch(self):
        identity = self.root.parent / 'identity.json'
        identity.write_text(json.dumps({'episodes_export_manifest_sha256': self.pin}))
        identity_pin = sha256(identity)
        identity.write_text(json.dumps({'episodes_export_manifest_sha256': '0' * 64}))
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'check_connection.py'),
            '--execute', '--source-manifest-sha256', sha256(ROOT / 'source_manifest.json'),
            '--identity-manifest', str(identity), '--identity-manifest-sha256', identity_pin,
            '--episodes', str(self.root)], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('External manifest SHA256 mismatch', result.stderr)
        self.assertNotIn("No module named 'torch'", result.stderr)


if __name__ == '__main__':
    unittest.main()
