"""Synthetic full-NPZ CPU regression tests. No actual EO data or models."""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from loader_adapter import make_loader_class, PINNED_LOADER_SHA256
from smoke_packets import run as run_smoke

REPO = Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
BASE_LOADER = Path(os.environ.get('OE11_BASE_LOADER', str(REPO / 'code/oe10_text_mask_v2/base_snapshot/episode_loader.py')))
BASE_BUILDER = Path(os.environ.get('OE8_BASE_BUILDER', str(REPO / 'code/oe8_episode_prepare_v0/episode_builder.py')))
CATALOG_CODE = Path(os.environ.get('OE11_CATALOG_CODE', '/private/tmp/oe11_episode_catalog'))
sys.path.insert(0, str(CATALOG_CODE))
import episode_catalog as builder
from test_episode_catalog import fixture as catalog_fixture
Loader = make_loader_class(BASE_LOADER)


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def fixture(root):
    rows = catalog_fixture(root)
    # Builder tests need fewer packet fields. Complete the actual loader
    # contract in this isolated fixture before hashes/catalogs are generated.
    for row in rows:
        row['role'] = 'development' if row['training_partition'] == 'calibration' else 'train'
        path = root / row['npz_path']
        with np.load(path, allow_pickle=False) as z:
            packet = {key: z[key] for key in z.files}
        packet['band_observed'] = np.array([True] * 10 + [False] * 2, dtype=bool)
        packet['nodata_observed'] = packet['raw_selected_s2'] == -10000
        np.savez_compressed(path, **packet)
        row['npz_sha256'] = builder.sha(path)
    jsonl(root / 'manifest.jsonl', rows)
    return rows


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='oe11_loader_test_')
        self.root = Path(self.tmp.name)
        self.prepared = self.root / 'prepared'
        self.rows = fixture(self.prepared)
        self.catalog = self.root / 'catalog'
        builder.build(self.prepared, self.catalog, BASE_BUILDER, 'pooled')

    def tearDown(self): self.tmp.cleanup()

    def loader(self, split='development', root=None, mode='pooled'):
        return Loader(self.prepared, root or self.catalog, split, support_mode=mode)

    def change_catalog(self, split, mutate):
        path = self.catalog / f'episodes_{split}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows[0]); jsonl(path, rows)
        cp = self.catalog / 'episode_contract.json'; contract = json.loads(cp.read_text())
        contract['public_catalog_sha256'][split] = builder.sha(path)
        cp.write_text(json.dumps(contract))

    def test_original_manifest_and_catalog_unchanged_and_memory_not_aliased(self):
        paths = [self.prepared / 'manifest.jsonl', self.catalog / 'episodes_development.jsonl']
        before = [p.read_bytes() for p in paths]
        loader = self.loader()
        self.assertTrue(all(m['training_partition'] == 'calibration' for m in loader._manifest.values()
                            if m['role'] == 'development'))
        episode = next(iter(loader.episodes.values()))
        self.assertIn('/source_bank/', episode['support_pairs'][0]['positive']['mask_npz'])
        loader.load(episode)
        self.assertEqual(before, [p.read_bytes() for p in paths])
        public = loader.episodes; public[episode['episode_id']]['query_label_supplied'] = True
        self.assertFalse(loader.episodes[episode['episode_id']]['query_label_supplied'])

    def test_load_never_opens_query_gold_or_scoring_and_acquires_only_requested_dates(self):
        original_open = Path.open
        accessed = []
        def guard(path, *args, **kwargs):
            accessed.append(str(path))
            if 'labels' in path.parts or 'scoring' in path.parts:
                raise AssertionError('Implicit query gold/scoring read: ' + str(path))
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', guard):
            loader = self.loader()
            for k in (1, 8):
                episode = next(e for e in loader.episodes.values() if e['k_pairs'] == k)
                for positions in ([2, 5], [7, 0, 5, 2], list(range(8))):
                    value = loader.load(episode, acquired_positions=positions)
                    q = value['model_input']['query']
                    self.assertEqual(q['s2'].shape, (128, 128, len(positions), 12))
                    self.assertIsNone(q['s2'].base)
                    self.assertEqual(q['dates_yyyymmdd'].tolist(), [20190101 + i for i in sorted(positions)])
                    self.assertNotIn('mask', q)
                    self.assertEqual(set(value['model_input']), {'prompt', 'query', 'support_pairs'})
                    self.assertEqual(value['audit']['returned_support_observation_instances'], 16 * k)
                    self.assertFalse(value['audit']['query_gold_read'])
                    self.assertIsNone(loader._scoring)
            with self.assertRaises(ValueError): loader.load(episode, acquired_positions=[0, 1])
        self.assertTrue(accessed)

    def test_targets_are_explicit_partitioned_and_hash_verified(self):
        train = self.loader('train'); dev = self.loader()
        te, de = next(iter(train.episodes)), next(iter(dev.episodes))
        with self.assertRaises(ValueError): train.load_target(te, purpose='training')
        with self.assertRaises(ValueError): dev.load_target(de, purpose='training', training=True)
        with self.assertRaises(ValueError): train.load_target(te, purpose='evaluation')
        target = train.load_target(te, purpose='training', training=True)
        self.assertEqual(target['target_mask'].shape, (128, 128))
        self.assertEqual(dev.load_target(de, purpose='evaluation')['audit']['purpose'], 'evaluation')
        with (self.catalog / 'scoring/scoring_development.jsonl').open('a') as f: f.write('\n')
        # Inference does not depend on scoring; explicit access fails even when
        # an earlier validated scoring table exists in memory.
        dev.load(de)
        with self.assertRaisesRegex(ValueError, 'scoring file hash'):
            dev.load_target(de, purpose='evaluation')

    def test_base_guards_preserved_and_manifest_restored_after_validation_failure(self):
        loader = self.loader()
        original = copy.deepcopy(loader._manifest)
        episode = next(iter(loader.episodes.values())); episode['query_label_supplied'] = True
        with self.assertRaisesRegex(ValueError, 'query gold'):
            loader._validate_episode(episode)
        self.assertEqual(loader._manifest, original)
        episode = next(iter(loader.episodes.values()))
        episode['support_pairs'][0]['positive']['mask_npz'] = 'support_masks/calibration/fake.npz'
        with self.assertRaisesRegex(ValueError, 'mask lineage'): loader._validate_episode(episode)
        self.assertEqual(loader._manifest, original)
        episode = next(iter(loader.episodes.values()))
        episode['support_pairs'][0]['positive']['patch_id'] = episode['query_patch_id']
        with self.assertRaises(ValueError): loader._validate_episode(episode)

    def test_cross_parent_train_constraint_and_common_development_bank(self):
        cross = self.root / 'cross'; builder.build(self.prepared, cross, BASE_BUILDER, 'cross_parent')
        train = self.loader('train', cross, 'cross_parent')
        self.assertTrue(all(s['parent_tile'] != e['query_parent_tile'] for e in train.episodes.values()
                            for pair in e['support_pairs'] for s in pair.values()))
        dev = self.loader('development', cross, 'cross_parent')
        self.assertTrue(any(s['parent_tile'] == e['query_parent_tile'] for e in dev.episodes.values()
                            for pair in e['support_pairs'] for s in pair.values()))
        self.assertEqual(dev.episodes, self.loader().episodes)
        with self.assertRaisesRegex(ValueError, 'mode mismatch'):
            self.loader('train', cross, 'pooled')
        # A pooled TRAIN catalog labelled cross-parent must fail if it includes
        # same-parent support; updating only the declaration cannot bypass it.
        cp = self.catalog / 'episode_contract.json'; contract = json.loads(cp.read_text())
        contract['train_support_parent_policy'] = 'cross_parent'; cp.write_text(json.dumps(contract))
        with self.assertRaisesRegex(ValueError, 'cross-parent'):
            self.loader('train', mode='cross_parent')

    def test_original_hash_reference_roots_and_unseen_parent_guard(self):
        with (self.prepared / 'manifest.jsonl').open('a') as f: f.write('\n')
        with self.assertRaisesRegex(ValueError, 'manifest hash'): self.loader()
        jsonl(self.prepared / 'manifest.jsonl', self.rows)
        cp = self.catalog / 'episode_contract.json'; original = json.loads(cp.read_text())
        changed = copy.deepcopy(original); changed['support_mask_reference_root'] = str(self.root / 'elsewhere')
        cp.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'mask reference root'): self.loader()
        cp.write_text(json.dumps(original))
        self.rows[0]['parent_tile'] = 't30uxv'; jsonl(self.prepared / 'manifest.jsonl', self.rows)
        original['prepared_manifest_sha256'] = builder.sha(self.prepared / 'manifest.jsonl')
        cp.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, 'parent'): self.loader()

    def test_base_source_sha_is_required(self):
        copied = self.root / 'base.py'; copied.write_bytes(BASE_LOADER.read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'SHA'): make_loader_class(copied)

    def test_bounded_cli_logic_all_modes_splits_k_and_acquisitions(self):
        cross = self.root / 'cross'; builder.build(self.prepared, cross, BASE_BUILDER, 'cross_parent')
        report = run_smoke(self.prepared, {'pooled': self.catalog, 'cross_parent': cross}, BASE_LOADER)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['loads_checked'], 24)
        self.assertTrue(report['manifest_unchanged'])
        self.assertFalse(report['query_gold_access_requested'])
        self.assertTrue(all(not c['scoring_loaded'] for c in report['catalogs']))


if __name__ == '__main__': unittest.main(verbosity=2)
