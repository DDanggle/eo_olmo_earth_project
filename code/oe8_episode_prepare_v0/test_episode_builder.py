"""Synthetic integrity checks only; no server, network, GPU, or actual dataset.

Run after permission to execute preparation code:
  python -m unittest discover -s <stage> -p test_episode_builder.py
"""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import episode_builder as b


def fixture():
    rows, roles, inputs, train, bank = [], {}, {}, [], []
    groups = [('train_pool', 6), ('source_bank', 4), ('dev_query', 2)]
    for role, count in groups:
        for i in range(count):
            pid = f'{role}_{i}'
            parent = 't31tfm' if role == 'dev_query' else ('t32ulu' if i % 2 else 't31tfj')
            clean = i != 0
            rows.append({'patch_id': pid, 'parent_tile': parent,
                'npz_path': f'inputs/{pid}.npz', 'label_path': f'labels/{pid}.npz',
                'label_sha256': 'a' * 64, 'training_partition': role,
                'strict_no_missing_input_eligible': clean,
                'supervised_training_allowed': role == 'train_pool',
                'clean_training_eligible': clean and role == 'train_pool',
                'class_pixel_counts': {'not_for_inference': 999}})
            roles[pid] = role
            inputs[pid] = {'dates': [20190101 + d for d in range(8)], 'input_sha256': 'b' * 64}
            if role == 'dev_query':
                continue
            for cls in (1, 2):
                for objidx in range(3):
                    inst = cls * 100 + objidx
                    obj = {'object_key': f'{pid}:{inst}', 'patch_id': pid,
                           'parent_tile': parent, 'class_id': cls,
                           'mask_npz': f'support_masks/{pid}_{inst}.npz',
                           'mask_sha256': 'c' * 64}
                    (train if role == 'train_pool' else bank).append(obj)
    return rows, roles, inputs, train, bank


class EpisodePolicyTest(unittest.TestCase):
    def test_role_partition_is_label_blind_and_permutation_invariant(self):
        rows = [{'patch_id': f'{p}_{i}', 'parent_tile': p, 'role': 'train'}
                for p in b.SOURCE_PARENTS for i in range(32)]
        rows += [{'patch_id': f'dev_{i}', 'parent_tile': b.DEV_PARENT, 'role': 'development'}
                 for i in range(16)]
        roles = b.assign_roles(rows)
        self.assertEqual(roles, b.assign_roles(list(reversed(rows))))
        self.assertEqual(Counter(roles.values()), {'train_pool': 48, 'source_bank': 16, 'dev_query': 16})
        for parent in b.SOURCE_PARENTS:
            self.assertEqual(sum(roles[r['patch_id']] == 'source_bank'
                                 for r in rows if r['parent_tile'] == parent), 8)

    def test_nonclean_train_and_development_queries_are_retained(self):
        rows, roles, inputs, train, bank = fixture()
        for split, role in [('train', 'train_pool'), ('development', 'dev_query')]:
            episodes, _, coverage = b.build_catalog(rows, roles, inputs, train, bank, [1, 2], split)
            self.assertEqual(set(e['query_patch_id'] for e in episodes),
                             set(pid for pid in roles if roles[pid] == role))
            self.assertEqual(coverage['query_candidates'], sum(v == role for v in roles.values()))
            sample = next(e for e in episodes if e['query_patch_id'] == f'{role}_0')
            self.assertTrue(sample['missing_aware_adapter_required'])
            self.assertEqual(sample['initial_observation_ids'], [f'{role}_0:obs:2', f'{role}_0:obs:5'])

    def test_public_catalog_is_unchanged_by_development_gold_metadata(self):
        rows, roles, inputs, train, bank = fixture()
        before, _, _ = b.build_catalog(rows, roles, inputs, train, bank, [1, 2], 'development')
        changed = copy.deepcopy(rows)
        for row in changed:
            if roles[row['patch_id']] == 'dev_query':
                row['label_path'] = '/nonexistent/poisoned_gold.npz'
                row['label_sha256'] = 'changed'
                row['class_pixel_counts'] = {'19': 16384}
        after, private, _ = b.build_catalog(changed, roles, inputs, train, bank, [1, 2], 'development')
        self.assertEqual(before, after)
        forbidden = ('query_label', 'target_class', 'counter_class', 'target_present', 'target_pixels', 'class_pixel_counts')
        # query_label_supplied=False is an explicit contract, not a gold payload.
        for episode in before:
            self.assertFalse(episode['query_label_supplied'])
            public_copy = {k: v for k, v in episode.items() if k != 'query_label_supplied'}
            for token in forbidden:
                self.assertNotIn(token, json.dumps(public_copy))
        self.assertTrue(all(row['query_label_npz'] == '/nonexistent/poisoned_gold.npz' for row in private))

    def test_support_prefixes_reverse_tasks_and_per_patch_cap(self):
        rows, roles, inputs, train, bank = fixture()
        episodes, scoring, _ = b.build_catalog(rows, roles, inputs, train, bank, [1, 2], 'development')
        targets = {r['episode_id']: r for r in scoring}
        query = next(e['query_patch_id'] for e in episodes)
        forward = {e['k_pairs']: e for e in episodes if e['query_patch_id'] == query
                   and targets[e['episode_id']]['target_class'] == 1}
        reverse = {e['k_pairs']: e for e in episodes if e['query_patch_id'] == query
                   and targets[e['episode_id']]['target_class'] == 2}
        self.assertEqual(set(forward), {1, 2, 4, 8})
        for k in b.KS:
            self.assertEqual(forward[k]['support_pairs'], forward[8]['support_pairs'][:k])
            for a, z in zip(forward[k]['support_pairs'], reverse[k]['support_pairs']):
                self.assertEqual(a['positive'], z['counterexample'])
                self.assertEqual(a['counterexample'], z['positive'])
        for kind in ('positive', 'counterexample'):
            counts = Counter(p[kind]['patch_id'] for p in forward[8]['support_pairs'])
            self.assertEqual(max(counts.values()), 2)
            self.assertTrue(all(roles[pid] == 'source_bank' for pid in counts))

    def test_train_support_excludes_query_and_bank(self):
        rows, roles, inputs, train, bank = fixture()
        episodes, _, _ = b.build_catalog(rows, roles, inputs, train, bank, [1, 2], 'train')
        for e in episodes:
            support = [obj for pair in e['support_pairs'] for obj in pair.values()]
            keys = [obj['object_key'] for obj in support]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertTrue(all(roles[obj['patch_id']] == 'train_pool' for obj in support))
            self.assertTrue(all(obj['patch_id'] != e['query_patch_id'] for obj in support))

    def test_insufficient_bank_is_not_padded_or_promoted_to_k8(self):
        rows, roles, inputs, train, bank = fixture()
        bank = [o for o in bank if o['patch_id'] != 'source_bank_3']
        episodes, _, coverage = b.build_catalog(rows, roles, inputs, train, bank, [1, 2], 'development')
        self.assertEqual(set(e['k_pairs'] for e in episodes), {1, 2, 4})
        self.assertEqual(coverage['k8_auc_base_count'], 0)
        self.assertEqual(len(coverage['support_shortages']), 4)
        self.assertEqual(len({e['base_id'] for e in episodes if e['k_pairs'] == 1}),
                         len({e['base_id'] for e in episodes if e['k_pairs'] == 4}))

    def test_class_selection_requires_four_training_patches_and_tie_breaks(self):
        train = [{'class_id': cls, 'patch_id': f'p{i}'}
                 for cls, count in [(9, 7), (3, 4), (2, 4), (8, 3), (6, 4), (7, 4)]
                 for i in range(count)]
        classes, coverage = b.select_classes(train)
        self.assertEqual(classes, [9, 2, 3, 6])
        self.assertEqual(coverage['8']['eligible_patches'], 3)

    def test_object_purity_size_and_whole_instance_validity(self):
        semantic = np.zeros((128, 128), dtype=np.int64)
        instances = np.zeros_like(semantic)
        instances[10:20, 10:20] = 1
        semantic[instances == 1] = 3
        valid = np.ones((8, 128, 128), dtype=np.bool_)
        objects, _ = b.eligible_objects(semantic, instances, semantic != 19, valid, 'p', 't')
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]['object_key'], 'p:1')
        semantic[10, 10:15] = 4  # 95% pure: allowed, mask excludes five other-class pixels.
        objects, _ = b.eligible_objects(semantic, instances, semantic != 19, valid, 'p', 't')
        self.assertEqual(objects[0]['class_pixel_count'], 95)
        valid[7, 10, 10] = False  # Invalid excluded minority pixel still rejects whole instance.
        objects, reasons = b.eligible_objects(semantic, instances, semantic != 19, valid, 'p', 't')
        self.assertFalse(objects)
        self.assertEqual(reasons['instance_not_valid_in_all_eight_observations'], 1)
        valid[:] = True
        semantic[10, 15] = 4
        objects, reasons = b.eligible_objects(semantic, instances, semantic != 19, valid, 'p', 't')
        self.assertFalse(objects)
        self.assertEqual(reasons['class_purity_below_threshold'], 1)

    def test_manifest_dates_match_zero_based_month_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'example.npz'
            dates = [20190101 + i for i in range(8)]
            values = dict(raw_selected_s2=np.zeros((8, 10, 128, 128), dtype=np.int16),
                normalized_s2=np.zeros((128, 128, 8, 12), dtype=np.float32),
                timestamps=np.asarray([[i + 1, 0, 2019] for i in range(8)], dtype=np.int64),
                observation_valid=np.ones((8, 128, 128), dtype=np.bool_))
            np.savez_compressed(path, **values)
            row = {'npz_path': path.name, 'npz_sha256': b.file_sha(path),
                   'selected_dates': dates, 'strict_no_missing_input_eligible': True}
            self.assertEqual(b.inspect_input(root, row)['dates'], dates)
            row['selected_dates'] = [20190201 + i for i in range(8)]
            with self.assertRaisesRegex(ValueError, 'timestamps'):
                b.inspect_input(root, row)


if __name__ == '__main__':
    unittest.main()
