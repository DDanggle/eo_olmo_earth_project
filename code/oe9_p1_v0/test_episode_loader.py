"""Safety failures and three actual OE8 packets; no GPU or model numeric claim."""
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from episode_loader import ContractError, EpisodeLoader, PUBLIC_PROMPT, digest, safe_path

REAL = Path('/private/tmp/oe8_pastis_prepare_20260927/download/review_bundle_v0')


def write_jsonl(path, records):
    path.write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in records))


def fixture(root):
    prep, eps = root / 'prepared', root / 'episodes'
    for p in (prep / 'inputs', prep / 'labels', eps / 'support_masks', eps / 'scoring'):
        p.mkdir(parents=True, exist_ok=True)
    dates = [20200101 + i for i in range(8)]
    partitions = {'t0': 'train_pool', 't1': 'train_pool', 't2': 'train_pool',
                  'b0': 'source_bank', 'b1': 'source_bank', 'd0': 'dev_query'}
    manifest = []
    semantic = np.zeros((128, 128), dtype=np.int64)
    semantic[:32] = 1
    semantic[32:64] = 3
    semantic[-1] = 19
    for pid, part in partitions.items():
        raw = np.broadcast_to(np.arange(8, dtype=np.int16)[:, None, None, None],
                              (8, 10, 128, 128)).copy()
        norm = np.broadcast_to(np.arange(8, dtype=np.float32)[None, None, :, None],
                               (128, 128, 8, 12)).copy()
        valid = np.ones((8, 128, 128), dtype=bool)
        # Acquired date has real missingness: loader must preserve it, not filter gold.
        raw[2, :, 0, 0] = -10000
        norm[0, 0, 2] = 0
        valid[2, 0, 0] = False
        np.savez_compressed(prep / 'inputs' / f'{pid}.npz', raw_selected_s2=raw,
                            normalized_s2=norm, observation_valid=valid,
                            nodata_observed=raw == -10000,
                            timestamps=np.array([[i+1, 0, 2020] for i in range(8)], dtype=np.int64),
                            band_observed=np.array([True]*10+[False]*2))
        np.savez_compressed(prep / 'labels' / f'{pid}.npz', semantic=semantic,
                            instances=np.zeros((128, 128), np.int64), label_valid=semantic != 19,
                            crop_label_valid=(semantic >= 1) & (semantic <= 18))
        manifest.append({'patch_id': pid, 'parent_tile': 'tile_dev' if part == 'dev_query' else 'tile_train',
                         'role': 'development' if part == 'dev_query' else 'train',
                         'training_partition': part, 'supervised_training_allowed': part == 'train_pool',
                         'npz_path': f'inputs/{pid}.npz', 'npz_sha256': digest(prep/'inputs'/f'{pid}.npz'),
                         'label_path': f'labels/{pid}.npz', 'label_sha256': digest(prep/'labels'/f'{pid}.npz'),
                         'selected_dates': dates, 'class_pixel_counts': [999999] * 20})
    bypid = {m['patch_id']: m for m in manifest}
    for split, qid, supports in [('train', 't0', ('t1','t2')), ('development', 'd0', ('b0','b1'))]:
        q = bypid[qid]
        pair = {}
        for i, (kind, pid) in enumerate(zip(('positive', 'counterexample'), supports), 1):
            m = bypid[pid]
            mask = semantic == (1 if kind == 'positive' else 3)
            mpath = eps / 'support_masks' / f'{pid}_{i}.npz'
            np.savez_compressed(mpath, mask=mask)
            pair[kind] = {'patch_id': pid, 'parent_tile': m['parent_tile'], 'object_key': f'{pid}:{i}',
                          'input_npz': m['npz_path'], 'input_sha256': m['npz_sha256'],
                          'mask_npz': f'support_masks/{pid}_{i}.npz', 'mask_sha256': digest(mpath),
                          'dates_yyyymmdd': dates, 'observation_ids': [f'{pid}:obs:{j}' for j in range(8)]}
        e = {'episode_id': f'{split}:k1', 'base_id': split, 'pair_id': 'pair', 'split': split,
             'query_patch_id': qid, 'query_parent_tile': q['parent_tile'], 'query_input_npz': q['npz_path'],
             'query_input_sha256': q['npz_sha256'], 'query_label_supplied': False,
             'supervised_training_allowed': split == 'train', 'k_pairs': 1, 'support_pairs': [pair],
             'prompt': PUBLIC_PROMPT, 'initial_observation_ids': [f'{qid}:obs:{i}' for i in (2,5)],
             'query_observations': [{'observation_id': f'{qid}:obs:{i}', 'date_yyyymmdd': d}
                                    for i, d in enumerate(dates)]}
        s = {k:e[k] for k in ('episode_id','base_id','pair_id','query_patch_id','k_pairs')}
        s.update(query_label_npz=q['label_path'], query_label_sha256=q['label_sha256'],
                 expected_query_label_sha256=q['label_sha256'], target_class=1, counter_class=3,
                 target_pixels=int((semantic == 1).sum()), counter_pixels=int((semantic == 3).sum()),
                 label_valid_pixels=int((semantic != 19).sum()))
        write_jsonl(eps / f'episodes_{split}.jsonl', [e])
        write_jsonl(eps / 'scoring' / f'scoring_{split}.jsonl', [s])
    write_jsonl(prep/'manifest.jsonl', manifest)
    refresh_hashes(prep, eps)
    return prep, eps


def refresh_hashes(prep, eps):
    contract = {'prepared_manifest_sha256': digest(prep/'manifest.jsonl'),
                'public_catalog_sha256': {s:digest(eps/f'episodes_{s}.jsonl') for s in ('train','development')},
                'initial_candidate_positions': [2,5],
                'all_eight_support_observations_must_be_counted_in_cost_ledger': True}
    (eps/'episode_contract.json').write_text(json.dumps(contract))


class LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.TemporaryDirectory()
        fixture(Path(cls.base.name))

    @classmethod
    def tearDownClass(cls):
        cls.base.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        shutil.copytree(self.base.name, self.root, dirs_exist_ok=True)
        self.prep, self.eps = self.root/'prepared', self.root/'episodes'
        self.loader = EpisodeLoader(self.prep, self.eps, 'train')

    def tearDown(self):
        self.temp.cleanup()

    def test_acquisition_and_full_resolution_no_backdoor_views(self):
        loaded = self.loader.load('train:k1')
        q = loaded['model_input']['query']
        self.assertEqual(q['s2'].shape, (128,128,2,12))
        self.assertEqual(q['raw_s2'].shape, (2,10,128,128))
        self.assertEqual(q['dates_yyyymmdd'].tolist(), [20200103,20200106])
        self.assertEqual(q['s2'][1,1,:,0].tolist(), [2,5])
        self.assertFalse(q['observation_valid'][0,0,0])
        for a in q.values():
            self.assertIsNone(a.base)
        for s in loaded['model_input']['support_pairs'][0].values():
            self.assertEqual(s['s2'].shape, (128,128,8,12))
        self.assertEqual(loaded['audit']['cpu_decoded_observations'], 24)
        self.assertEqual(loaded['audit']['returned_support_observation_instances'], 16)

    def test_explicit_extra_acquisition_chronological_and_no_implicit_changes(self):
        q = self.loader.load('train:k1', acquired_positions=[7,5,0,2])['model_input']['query']
        self.assertEqual(q['s2'][1,1,:,0].tolist(), [0,2,5,7])
        self.assertEqual(self.loader.load('train:k1')['model_input']['query']['s2'].shape[2], 2)
        for invalid in ([2], [2,5,5], [2,5,8], [2,5,-1], [2,5,True]):
            with self.assertRaises(ContractError): self.loader.load('train:k1', invalid)

    def test_load_never_reads_gold_and_model_contains_only_whitelist(self):
        shutil.rmtree(self.prep/'labels')
        shutil.rmtree(self.eps/'scoring')
        actual_load = np.load
        def guard(path, *args, **kwargs):
            self.assertNotIn('labels', Path(path).parts)
            self.assertNotIn('scoring', Path(path).parts)
            return actual_load(path,*args,**kwargs)
        with patch('episode_loader.np.load', side_effect=guard):
            x = self.loader.load('train:k1')['model_input']
        self.assertIsNone(self.loader._scoring)
        self.assertEqual(set(x), {'prompt','query','support_pairs'})
        allowed = {'s2','raw_s2','timestamps','dates_yyyymmdd','observation_valid','band_observed'}
        self.assertEqual(set(x['query']), allowed)
        for s in x['support_pairs'][0].values(): self.assertEqual(set(s), allowed | {'mask'})
        self.assertNotIn('t0', x['prompt'])

    def test_targets_role_separation_and_no_all_date_validity_filter(self):
        with self.assertRaises(ContractError): self.loader.training_target('train:k1')
        with self.assertRaises(ContractError): self.loader.evaluation_target('train:k1')
        t = self.loader.training_target('train:k1', training=True)
        self.assertEqual(t['target_mask'].dtype, bool)
        self.assertTrue(t['target_mask'][0,0])  # Input nodata at this pixel does not censor label gold.
        dev = EpisodeLoader(self.prep,self.eps,'development')
        with self.assertRaises(ContractError): dev.training_target('development:k1',training=True)
        self.assertEqual(dev.evaluation_target('development:k1')['target_mask'].shape,(128,128))
        with self.assertRaises(ContractError): self.loader.load('development:k1')

    def test_dictionary_tamper_and_catalog_hash_rejected(self):
        e = self.loader.episodes['train:k1']
        e['query_patch_id'] = 'd0'
        with self.assertRaises(ContractError): self.loader.load(e)
        self.assertEqual(self.loader.episodes['train:k1']['query_patch_id'], 't0')
        with (self.eps/'episodes_train.jsonl').open('a') as f: f.write('\n')
        with self.assertRaises(ContractError): EpisodeLoader(self.prep,self.eps,'train')

    def test_resealed_manifest_still_cannot_make_bank_supervised_query(self):
        path = self.prep/'manifest.jsonl'
        m = [json.loads(s) for s in path.read_text().splitlines()]
        m[0]['training_partition'] = 'source_bank'
        write_jsonl(path,m); refresh_hashes(self.prep,self.eps)
        with self.assertRaises(ContractError): EpisodeLoader(self.prep,self.eps,'train')

    def test_wrong_support_role_duplicate_or_query_support_rejected(self):
        path = self.eps/'episodes_train.jsonl'
        original = json.loads(path.read_text())
        for pid in ('b0','t0'):
            e = copy.deepcopy(original)
            e['support_pairs'][0]['positive']['patch_id'] = pid
            write_jsonl(path,[e]); refresh_hashes(self.prep,self.eps)
            with self.assertRaises(ContractError): EpisodeLoader(self.prep,self.eps,'train')
        e = copy.deepcopy(original)
        e['support_pairs'][0]['counterexample'] = copy.deepcopy(e['support_pairs'][0]['positive'])
        write_jsonl(path,[e]); refresh_hashes(self.prep,self.eps)
        with self.assertRaises(ContractError): EpisodeLoader(self.prep,self.eps,'train')

    def test_input_corruption_and_path_escape_rejected(self):
        with (self.prep/'inputs/t0.npz').open('ab') as f: f.write(b'corruption')
        with self.assertRaises(ContractError): self.loader.load('train:k1')
        for path in ('../labels/t0.npz','labels/t0.npz','/inputs/t0.npz','inputs/../labels/t0.npz'):
            with self.assertRaises(ContractError): safe_path(self.prep,path,'inputs')
        (self.prep/'inputs/evil.npz').symlink_to(self.prep/'labels/t0.npz')
        with self.assertRaises(ContractError): safe_path(self.prep,'inputs/evil.npz','inputs')

    def test_resized_input_rejected_even_after_hash_refresh(self):
        path = self.prep/'inputs/t0.npz'
        with np.load(path) as z: a = {k:z[k] for k in z.files}
        a['normalized_s2'] = a['normalized_s2'][::2,::2].copy()
        np.savez_compressed(path,**a)
        self.loader._manifest['t0']['npz_sha256'] = digest(path)
        with self.assertRaisesRegex(ContractError,'shape/dtype'): self.loader.load('train:k1')

    def test_scoring_misjoin_or_count_corruption_rejected(self):
        path = self.eps/'scoring/scoring_train.jsonl'
        row = json.loads(path.read_text())
        row['target_pixels'] += 1
        write_jsonl(path,[row])
        with self.assertRaisesRegex(ContractError,'pixel counts'):
            self.loader.training_target('train:k1',training=True)


@unittest.skipUnless(REAL.exists(), 'actual review bundle absent')
class ActualReceiptTests(unittest.TestCase):
    def test_all_actual_metadata_and_three_available_packets(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            shutil.copy(REAL/'manifest.jsonl',p/'manifest.jsonl')
            (p/'inputs').mkdir(); (p/'labels').mkdir()
            for kind in ('inputs','labels'):
                for f in (REAL/'sample_packets'/kind).glob('*.npz'):
                    os.link(f,p/kind/f.name)
            train = EpisodeLoader(p,REAL/'episodes','train')
            dev = EpisodeLoader(p,REAL/'episodes','development')
            self.assertEqual(len(train.episodes),2304)
            self.assertEqual(len(dev.episodes),672)
            self.assertEqual(len(train._manifest),80)
            for pid in ('40411','40139','30031'):
                a, ledger = train._read_packet(pid)
                acquired = train._model_packet(a,[2,5],train._manifest[pid]['selected_dates'])
                self.assertEqual(acquired['s2'].shape,(128,128,2,12))
                self.assertEqual(ledger['cpu_decoded_observations'],8)
            for loader,pid in ((train,'40411'),(dev,'30031')):
                e = next(e for e in loader.episodes.values() if e['query_patch_id'] == pid)
                t = loader.training_target(e,training=True) if loader.split == 'train' else loader.evaluation_target(e)
                self.assertEqual(t['target_mask'].shape,(128,128))


if __name__ == '__main__': unittest.main(verbosity=2)
