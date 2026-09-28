"""Actual metadata schedule/cohort tests, not GPU or cold-resume numeric tests."""
import collections
import json
from pathlib import Path
import unittest

from p2_worker import episode_order


BUNDLE = Path('/private/tmp/oe8_pastis_prepare_20260927/download/review_bundle_v0')


class WorkerDataContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train={r['episode_id']:r for r in map(json.loads,(BUNDLE/'episodes/episodes_train.jsonl').read_text().splitlines())}
        cls.dev={r['episode_id']:r for r in map(json.loads,(BUNDLE/'episodes/episodes_development.jsonl').read_text().splitlines())}
        cls.manifest={r['patch_id']:r for r in map(json.loads,(BUNDLE/'manifest.jsonl').read_text().splitlines())}

    def test_paired_schedule_stable_prefix_and_seed_changes(self):
        b0=episode_order(self.train,270927,128)
        b2=episode_order(self.train,270927,128)
        self.assertEqual(b0,b2)
        self.assertEqual(b0,episode_order(self.train,270927,2400)[:128])
        self.assertNotEqual(b0,episode_order(self.train,270928,128))
        self.assertEqual(len(set(b0)),128)

    def test_full_cycle_balanced_covers_all_training_and_no_bank_or_dev(self):
        order=episode_order(self.train,270927,len(self.train))
        self.assertEqual(len(order),2304)
        self.assertEqual(set(order),set(self.train))
        self.assertEqual(collections.Counter(self.train[e]['k_pairs'] for e in order),{1:576,2:576,4:576,8:576})
        for start in range(0,len(order),4):
            self.assertEqual({self.train[e]['k_pairs'] for e in order[start:start+4]},{1,2,4,8})
        for eid in order:
            e=self.train[eid]
            for pid in [e['query_patch_id']]+[s['patch_id'] for p in e['support_pairs'] for s in p.values()]:
                self.assertEqual(self.manifest[pid]['training_partition'],'train_pool')

    def test_common96_cohort_has_all_four_k_and_only_development_queries(self):
        bases={e['base_id'] for e in self.dev.values() if e['k_pairs']==8}
        selected=[e for e in self.dev.values() if e['base_id'] in bases]
        self.assertEqual(len(bases),96)
        self.assertEqual(len(selected),384)
        bybase=collections.defaultdict(set)
        for e in selected:
            bybase[e['base_id']].add(e['k_pairs'])
            self.assertEqual(self.manifest[e['query_patch_id']]['training_partition'],'dev_query')
            for pair in e['support_pairs']:
                for s in pair.values():self.assertEqual(self.manifest[s['patch_id']]['training_partition'],'source_bank')
        self.assertTrue(all(ks=={1,2,4,8} for ks in bybase.values()))


if __name__ == '__main__': unittest.main(verbosity=2)
