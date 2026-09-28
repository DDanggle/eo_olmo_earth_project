"""Scientific/data-separation regression checks on synthetic CPU packets only."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import episode_catalog as adapter

BASE = Path(os.environ.get('OE8_BASE_BUILDER',
    '/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/oe8_episode_prepare_v0/episode_builder.py'))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def labels(path, classes=(1, 3, 8, 14, 2), copies=2):
    semantic = np.zeros((128, 128), dtype=np.int64)
    instances = np.zeros_like(semantic)
    for i, cls in enumerate(classes):
        for j in range(copies):
            sl = np.s_[2 + 12 * i:10 + 12 * i, 2 + 12 * j:10 + 12 * j]
            semantic[sl] = cls
            instances[sl] = 100 * cls + j + 1
    np.savez_compressed(path, semantic=semantic, instances=instances,
                        label_valid=semantic != 19, crop_label_valid=(semantic > 0) & (semantic < 19))


def fixture(root, *, bank_copies=2):
    root.mkdir()
    (root / 'inputs').mkdir()
    (root / 'labels').mkdir()
    dates = [20190101 + i for i in range(8)]
    template = root / 'template.npz'
    np.savez_compressed(template, timestamps=np.array([[i + 1, 0, 2019] for i in range(8)], dtype=np.int64),
        observation_valid=np.ones((8, 128, 128), dtype=bool),
        raw_selected_s2=np.zeros((8, 10, 128, 128), dtype=np.int16),
        normalized_s2=np.zeros((128, 128, 8, 12), dtype=np.float32))
    rows = []
    for parent in sorted(adapter.SOURCE_PARENTS):
        for partition, count in [('train_pool', 2), ('source_bank', 2), ('calibration', 1)]:
            for i in range(count):
                pid = f'{parent}_{partition}_{i}'
                input_ref, label_ref = f'inputs/{pid}.npz', f'labels/{pid}.npz'
                shutil.copyfile(template, root / input_ref)
                labels(root / label_ref, copies=bank_copies if partition == 'source_bank' else 2)
                rows.append(dict(patch_id=pid, parent_tile=parent, training_partition=partition,
                    npz_path=input_ref, label_path=label_ref, npz_sha256=adapter.sha(root / input_ref),
                    label_sha256=adapter.sha(root / label_ref), selected_dates=dates,
                    strict_no_missing_input_eligible=True, supervised_training_allowed=partition == 'train_pool',
                    clean_training_eligible=partition == 'train_pool'))
    template.unlink()
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (root / 'selection_policy.json').write_text(json.dumps(dict(source_parents=sorted(adapter.SOURCE_PARENTS),
        target_classes=list(adapter.CLASSES), caps_per_parent=dict(train_pool=64, source_bank=16, calibration=16),
        observations='OE8 fixed 8 dates, initial indices2/5; additional0/7')))
    return rows


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='oe11_catalog_test_')
        self.root = Path(self.tmp.name)
        self.prepared = self.root / 'prepared'
        self.rows = fixture(self.prepared)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, name='catalog', policy='pooled'):
        out = self.root / name
        return out, adapter.build(self.prepared, out, BASE, policy)

    def test_arbitrary_counts_all_three_parents_are_sources_and_calibration(self):
        rows, roles = adapter.adapt_rows(self.rows)
        self.assertEqual(len(rows), 15)
        self.assertEqual(len(roles), 15)
        self.assertEqual(set(roles.values()), {'train_pool', 'source_bank', 'dev_query'})
        self.assertEqual({r['parent_tile'] for r in rows if roles[r['patch_id']] == 'train_pool'}, adapter.SOURCE_PARENTS)
        self.assertEqual({r['parent_tile'] for r in rows if roles[r['patch_id']] == 'dev_query'}, adapter.SOURCE_PARENTS)

    def test_scope_and_partition_guards(self):
        for field, value in [('parent_tile', 't30uxv'), ('training_partition', 'final_holdout'),
                             ('supervised_training_allowed', False), ('patch_id', '../outside')]:
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                adapter.adapt_rows(rows)
        with self.assertRaises(ValueError):
            adapter.adapt_rows(self.rows + [self.rows[0]])
        with self.assertRaises(ValueError):
            adapter.adapt_rows([r for r in self.rows if not
                (r['parent_tile'] == 't31tfm' and r['training_partition'] == 'train_pool')])

    def test_base_source_hash_guard_precedes_import(self):
        bad = self.root / 'wrong_builder.py'
        bad.write_text("raise RuntimeError('Must never import')\n")
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            adapter.load_base(bad)

    def test_full_catalog_fixed_pairs_nested_support_and_split_isolation(self):
        out, contract = self.build()
        self.assertEqual(contract['status'], 'valid_catalog_complete_coverage')
        self.assertEqual(contract['common_ks_all_queries_all_pairs_both_splits'], [1, 2, 4, 8])
        self.assertFalse(contract['training_ready'])
        self.assertFalse(contract['calibration_is_unseen_geographic_region'])
        self.assertTrue(contract['former_development_parent_t31tfm_promoted_to_source'])
        self.assertFalse(contract['all_policy_partition_caps_filled'])
        pair_map = json.loads((out / 'pair_catalog.json').read_text())
        self.assertEqual(pair_map['selected_classes'], [1, 3, 8, 14])
        self.assertEqual(len(pair_map['directed_pairs']), 12)
        source_objects = read_jsonl(out / 'source_objects.jsonl')
        self.assertEqual({o['class_id'] for o in source_objects}, set(adapter.CLASSES))
        self.assertFalse(any('calibration' in o['mask_npz'] for o in source_objects))
        source_roles = {o['object_key']: o['episode_role'] for o in source_objects}
        for split, count, role in [('train', 288, 'train_pool'), ('development', 144, 'source_bank')]:
            episodes = read_jsonl(out / f'episodes_{split}.jsonl')
            scoring = read_jsonl(out / 'scoring' / f'scoring_{split}.jsonl')
            self.assertEqual(len(episodes), count)
            self.assertEqual(len(scoring), count)
            groups = defaultdict(dict)
            by_query_pair_k = {}
            class_by_id = {p['pair_id']: (p['target_class'], p['counter_class']) for p in pair_map['directed_pairs']}
            for episode in episodes:
                groups[episode['base_id']][episode['k_pairs']] = episode
                a, b = class_by_id[episode['pair_id']]
                by_query_pair_k[episode['query_patch_id'], a, b, episode['k_pairs']] = episode
                self.assertNotIn('target_class', episode)
                self.assertNotIn('query_label_npz', episode)
                self.assertEqual(episode['initial_observation_ids'],
                    [f"{episode['query_patch_id']}:obs:{i}" for i in (2, 5)])
                for pair in episode['support_pairs']:
                    for obj in pair.values():
                        self.assertNotEqual(obj['patch_id'], episode['query_patch_id'])
                        self.assertEqual(source_roles[obj['object_key']], role)
                        self.assertEqual(adapter.sha(out / obj['mask_npz']), obj['mask_sha256'])
            for group in groups.values():
                self.assertEqual(set(group), set(adapter.KS))
                for k in adapter.KS:
                    self.assertEqual(group[k]['support_pairs'], group[8]['support_pairs'][:k])
            for (qid, a, b, k), episode in by_query_pair_k.items():
                reverse = by_query_pair_k[qid, b, a, k]
                self.assertEqual([p['positive'] for p in episode['support_pairs']],
                                 [p['counterexample'] for p in reverse['support_pairs']])
        for ref, digest in contract['frozen_artifacts_sha256'].items():
            self.assertEqual(adapter.sha(out / ref), digest)

    def test_calibration_gold_opens_after_freeze_and_cannot_change_public_tasks(self):
        base = adapter.load_base(BASE)
        old_load = base.load_labels
        out = self.root / 'catalog'
        opened = []
        def guarded_load(path, expected_sha):
            if 'calibration' in path.name:
                self.assertTrue((out / 'frozen_artifacts_before_scoring.json').is_file())
                frozen = json.loads((out / 'frozen_artifacts_before_scoring.json').read_text())
                self.assertTrue(all(adapter.sha(out / ref) == digest for ref, digest in frozen.items()))
                opened.append(path)
            return old_load(path, expected_sha)
        with patch.object(adapter, 'load_base', return_value=base), patch.object(base, 'load_labels', guarded_load):
            first = adapter.build(self.prepared, out, BASE)
        self.assertEqual(len(opened), 3)
        for row in self.rows:
            if row['training_partition'] == 'calibration':
                labels(self.prepared / row['label_path'], classes=(2,), copies=1)
                row['label_sha256'] = adapter.sha(self.prepared / row['label_path'])
                row['class_statistics'] = {'must_not_select_tasks_from_this': 123456}
        (self.prepared / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        second_out, second = self.build('changed_calibration_gold')
        self.assertEqual(first['public_catalog_sha256'], second['public_catalog_sha256'])
        self.assertEqual(first['frozen_artifacts_sha256'], second['frozen_artifacts_sha256'])
        self.assertNotEqual(first['scoring_file_sha256']['development'], second['scoring_file_sha256']['development'])
        self.assertEqual(second['scoring_counts']['development']['target_absent_episodes'], 144)

    def test_shortage_keeps_lower_ks_without_changing_class_or_pair_scope(self):
        for row in self.rows:
            if row['training_partition'] == 'source_bank':
                labels(self.prepared / row['label_path'], copies=1)
                row['label_sha256'] = adapter.sha(self.prepared / row['label_path'])
        (self.prepared / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        out, contract = self.build()
        self.assertEqual(contract['status'], 'valid_catalog_with_support_shortages')
        self.assertFalse(contract['intended_subset_training_ready'])
        self.assertFalse(contract['training_ready'])
        self.assertEqual(contract['common_ks_all_queries_all_pairs_both_splits'], [1, 2, 4])
        self.assertEqual(contract['selected_classes'], [1, 3, 8, 14])
        coverage = json.loads((out / 'coverage.json').read_text())['development']
        self.assertEqual(len(coverage['all_directed_pairs']), 12)
        self.assertEqual(len(coverage['support_shortages']), 36)
        self.assertEqual(coverage['episodes_by_k'], {'1': 36, '2': 36, '4': 36, '8': 0})

    def test_cross_parent_supports_and_same_development_tasks(self):
        pooled_out, pooled = self.build('pooled')
        cross_out, cross = self.build('cross', 'cross_parent')
        self.assertEqual(pooled['public_catalog_sha256']['development'], cross['public_catalog_sha256']['development'])
        self.assertEqual(cross['common_ks_all_queries_all_pairs_both_splits'], [1, 2, 4, 8])
        pooled_episodes = read_jsonl(pooled_out / 'episodes_train.jsonl')
        cross_episodes = read_jsonl(cross_out / 'episodes_train.jsonl')
        self.assertEqual({r['episode_id'] for r in pooled_episodes}, {r['episode_id'] for r in cross_episodes})
        for row in cross_episodes:
            for pair in row['support_pairs']:
                for obj in pair.values():
                    self.assertNotEqual(row['query_parent_tile'], obj['parent_tile'])
        self.assertEqual(len(cross_episodes), 288)

    def test_policy_conflicts_and_existing_output_are_rejected(self):
        policy_path = self.prepared / 'selection_policy.json'
        original = json.loads(policy_path.read_text())
        for key, value in [('initial_candidate_positions', [0, 1]), ('target_classes', [1, 2, 3, 4]),
                           ('source_parents', ['t31tfj', 't32ulu']),
                           ('caps_per_parent', dict(train_pool=1, source_bank=1, calibration=1))]:
            policy_path.write_text(json.dumps({**original, key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.build()
        policy_path.write_text(json.dumps(original))
        out, _ = self.build()
        before = adapter.sha(out / 'episode_contract.json')
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(before, adapter.sha(out / 'episode_contract.json'))


if __name__ == '__main__':
    unittest.main()
