#!/usr/bin/env python3
"""Independent OE8 PASTIS input/source verifier; no model/GPU or cloud certification.

Run --self-test for deliberately corrupted local fixtures. Production source
verification additionally needs h5py and --source-root; no extractor is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import traceback
import unittest

import numpy as np

INPUT_BANDS = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
NATIVE_BANDS = ['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09']
CHANNEL_MAP = [0,1,2,6,3,4,5,7,8,9,0,7]
SHARD = 'geobench_pastis.0000.part.tortilla'
SHARD_SHA = '56b1490c6dc7345fdff79e94d9132753ee28d8504bb061d8db39d19e888f7ca3'
COMPUTED_SHA = '774e461babd2bd4cbe3591b46457d39054f8ac529cb57a052547501076ebaef7'
CANDIDATES_SHA = 'd0bfc58d11cbac0c6b270c5739a3f6b5c34f1f5de2510ab9635fd1bab7d1aef0'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def resolved_child(root, relative):
    root = Path(root).resolve()
    child = (root / relative).resolve()
    require(root in child.parents, 'prepared path escapes bundle')
    return child


def verify_inventory(rows, candidates):
    require(len(rows) == 80 and len(candidates) == 80, 'expected all 80 candidates')
    by_id = {str(r['patch_id']): r for r in rows}
    expected = {str(r['patch_id']): r for r in candidates}
    require(len(by_id) == len(expected) == 80, 'duplicate patch_id')
    require(set(by_id) == set(expected), 'candidate membership changed')
    require(Counter(r['role'] for r in rows) == {'train':64,'development':16}, 'role counts')
    bank_ids = set()
    for parent in ('t32ulu', 't31tfj'):
        ids = [str(r['patch_id']) for r in candidates if r['parent_tile'] == parent]
        require(len(ids) == 32, 'expected 32 train candidates per parent')
        ids.sort(key=lambda v: hashlib.sha256(('oe8-bank-v0:' + v).encode()).hexdigest())
        bank_ids.update(ids[:8])
    for patch_id, row in by_id.items():
        candidate = expected[patch_id]
        for key in ('parent_tile','role','all_dates_yyyymmdd'):
            require(row[key] == candidate[key], f'{patch_id}: candidate {key} changed')
        train = row['role'] == 'train'
        require(row['parent_tile'] in ({'t32ulu','t31tfj'} if train else {'t31tfm'}),
                f'{patch_id}: geographic role mismatch')
        partition = ('source_bank' if patch_id in bank_ids else 'train_pool') if train else 'dev_query'
        require(row['training_partition'] == partition, f'{patch_id}: bank/train leakage or hash mismatch')
        require(row['npz_path'] == f'inputs/{patch_id}.npz', 'input path convention')
        require(row['label_path'] == f'labels/{patch_id}.npz', 'label path convention')
    require(Counter(r['training_partition'] for r in rows) ==
            {'source_bank':16,'train_pool':48,'dev_query':16}, 'partition counts')
    return {'candidate_count':80, 'role_counts':{'train':64,'development':16},
            'partition_counts':{'source_bank':16,'train_pool':48,'dev_query':16},
            'source_bank_disjoint_from_supervised_train_pool':True}


def verify_arrays(row, inputs, labels, stats, source=None, hw=128):
    require(not ({'semantic','instances','target_mask','label_valid','crop_label_valid'} & set(inputs)),
            'query gold leaked into model input NPZ')
    raw = inputs['raw_selected_s2']
    require(raw.shape == (8,10,hw,hw) and raw.dtype == np.int16, 'raw shape/dtype')
    require(np.isfinite(raw).all(), 'raw nonfinite')
    dates = [datetime.strptime(str(int(d)), '%Y%m%d') for d in row['all_dates_yyyymmdd']]
    require(len(dates) >= 8 and dates == sorted(dates) and len(set(dates)) == len(dates),
            'source date count/order/duplicates')
    indices = [i * (len(dates)-1) // 7 for i in range(8)]
    require(row['selected_indices'] == indices, 'inclusive uniform frame indices')
    chosen = [row['all_dates_yyyymmdd'][i] for i in indices]
    require(row['selected_dates'] == chosen, 'selected dates mismatch')
    wanted_ts = np.array([[dates[i].day,dates[i].month-1,dates[i].year] for i in indices], dtype=np.int64)
    require(inputs['timestamps'].dtype == np.int64 and
            np.array_equal(inputs['timestamps'], wanted_ts), 'native timestamp mismatch')
    band_observed = inputs['band_observed']
    require(band_observed.dtype == np.bool_ and
            np.array_equal(band_observed, [True]*10 + [False]*2), 'imputed band marked observed')
    nodata = raw == -10000
    valid = ~nodata.any(axis=1)
    require(inputs['nodata_observed'].dtype == np.bool_ and
            np.array_equal(inputs['nodata_observed'], nodata), 'nodata mask mismatch')
    require(inputs['observation_valid'].dtype == np.bool_ and
            np.array_equal(inputs['observation_valid'], valid), 'observation valid mismatch')
    strict = bool(valid.all())
    supervised = row['training_partition'] == 'train_pool'
    for field, wanted in [('strict_no_missing_input_eligible',strict),
                          ('supervised_training_allowed',supervised),
                          ('clean_training_eligible',strict and supervised)]:
        require(isinstance(row[field],bool) and row[field] == wanted,
                f'{field}: input eligibility or supervised-bank isolation mismatch')
    native = np.transpose(raw[:, CHANNEL_MAP].astype(np.float32), (2,3,0,1))
    mean = np.array([stats[b]['mean'] for b in NATIVE_BANDS], dtype=np.float64)
    std = np.array([stats[b]['std'] for b in NATIVE_BANDS], dtype=np.float64)
    require(np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all(), 'invalid statistics')
    expected = ((native - (mean - 2*std)) / (4*std)).astype(np.float32)
    expected[np.transpose(~valid, (1,2,0))] = 0.0
    normalized = inputs['normalized_s2']
    require(normalized.dtype == np.float32 and normalized.shape == (hw,hw,8,12), 'normalized shape/dtype')
    require(np.isfinite(normalized).all(), 'normalized nonfinite')
    require(np.array_equal(normalized, expected), 'normalization/band-order/invalid-fill mismatch')
    semantic, instances = labels['semantic'], labels['instances']
    require(semantic.shape == instances.shape == (hw,hw), 'label shape')
    require(semantic.dtype == instances.dtype == np.int64, 'label dtype')
    require(np.isin(semantic, np.arange(20)).all(), 'semantic class out of range')
    require((instances >= 0).all(), 'negative instance IDs')
    require(labels['label_valid'].dtype == np.bool_ and
            np.array_equal(labels['label_valid'], semantic != 19), 'void mask mismatch')
    require(labels['crop_label_valid'].dtype == np.bool_ and
            np.array_equal(labels['crop_label_valid'], (semantic >= 1) & (semantic <= 18)),
            'background/void counted as crop labels')
    if source is not None:
        full_raw = source['s2']
        require(full_raw.shape == (len(dates),10,hw,hw) and full_raw.dtype == np.int16,
                'source S2 shape/date count/dtype')
        require(np.array_equal(raw, full_raw[indices]), 'raw pixels do not match source frame indices')
        full_semantic, full_instances = source['semantic'], source['instance']
        require(full_semantic.ndim == 3 and full_semantic.shape[1:] == (hw,hw), 'source semantic plane shape')
        require(np.isfinite(full_instances).all() and
                np.equal(full_instances, full_instances.astype(np.int64)).all(), 'source instance not integer')
        require(np.array_equal(semantic, full_semantic[0].astype(np.int64)), 'semantic source mismatch')
        require(np.array_equal(instances, full_instances.astype(np.int64)), 'instance source mismatch')
    return {'patch_id':str(row['patch_id']), 'passed':True,
            'normalization_max_delta':0.0, 'source_arrays_reread':source is not None,
            'nodata_observed_values':int(nodata.sum()), 'invalid_pixel_dates':int((~valid).sum()),
            'negative_non_nodata_values':int(((raw < 0) & ~nodata).sum()),
            'strict_no_missing_input_eligible':strict,
            'supervised_training_allowed':supervised,
            'clean_training_eligible':strict and supervised, 'cloud_quality_certified':False}


def source_arrays(row, source_root):
    import h5py  # Needed only for actual independent source reread.
    result = {}
    root = Path(source_root).resolve()
    for name in ('s2','semantic','instance'):
        payload = row['source_payloads'][name]
        match = re.fullmatch(r'/vsisubfile/(\d+)_(\d+),(.+)', payload['reference'])
        require(match is not None, 'malformed packed reference')
        offset, length = int(match[1]), int(match[2])
        original = Path(match[3])
        require(original.name == SHARD, 'attempt to read unverified shard')
        path = (root / original.name).resolve()
        require(path.parent == root and path.name == SHARD, 'source path escapes root')
        require(0 < length <= (128 << 20) and offset + length <= path.stat().st_size,
                'payload outside bounded source')
        with path.open('rb') as stream:
            stream.seek(offset)
            raw = stream.read(length)
        require(len(raw) == length and hashlib.sha256(raw).hexdigest() == payload['payload_sha256'],
                'source payload hash mismatch')
        with h5py.File(io.BytesIO(raw), 'r') as h5:
            result[name] = np.asarray(h5['data'])
    return result


def verify_catalog_binding(row, candidate, catalog_record, references, nested_dates):
    """Bind payload pointers to a frozen candidate via independently read catalog."""
    require(str(catalog_record['patch_id']) == str(candidate['patch_id']) == str(row['patch_id']),
            'source catalog patch mismatch')
    require(str(catalog_record['tile']) == candidate['parent_tile'] == row['parent_tile'],
            'source catalog parent mismatch')
    require(catalog_record['tortilla:data_split'] == 'train', 'source benchmark split changed')
    require(str(catalog_record['internal:subfile']) == candidate['source_packed_reference'],
            'source outer packed reference mismatch')
    source_dates = [int(d) for d in catalog_record['dates']]
    require(source_dates == candidate['all_dates_yyyymmdd'] == row['all_dates_yyyymmdd'],
            'source catalog dates mismatch')
    require([int(d) for d in nested_dates] == source_dates, 'source nested dates mismatch')
    for name in ('s2','semantic','instance'):
        require(references[name] == row['source_payloads'][name]['reference'],
                f'{name}: payload reference does not belong to source patch')


def verify_source_catalog(rows, candidates, source_root):
    import tacoreader  # Independent reader, never the extraction script.
    catalog = tacoreader.load([str(Path(source_root)/SHARD)])
    train = catalog[catalog['tortilla:data_split'] == 'train'].reset_index(drop=True)
    require(len(train) == 911, 'verified partial source catalog changed')
    by_id = {str(row['patch_id']): row for row in rows}
    for candidate in candidates:
        index = candidate['partial_catalog_index']
        require(isinstance(index,int) and 0 <= index < len(train), 'source catalog index')
        outer = train.iloc[index]
        nested = train.read(index)
        ids = list(nested['tortilla:id'])
        require(len(ids) == len(set(ids)), 'duplicate nested modality')
        require(all(name in ids for name in ('s2','semantic','instance')), 'missing nested modality')
        for _, metadata in nested.iterrows():
            require(str(metadata['patch_id']) == str(candidate['patch_id']) and
                    str(metadata['tile']) == candidate['parent_tile'], 'nested source identity mismatch')
        references = {name:str(nested.read(ids.index(name))) for name in ('s2','semantic','instance')}
        nested_dates = nested['dates'].iloc[ids.index('s2')].tolist()
        verify_catalog_binding(by_id[str(candidate['patch_id'])], candidate, outer,
                               references, nested_dates)
    return True


def verify_contract(contract, computed):
    require(str(contract.get('status','')).startswith('prepared') and
            'fail' not in str(contract.get('status','')).lower(), 'contract not successfully prepared')
    require(contract.get('source_shards_unchanged') is True, 'source stability not certified')
    require(contract.get('verified_source_stat_continuity') is True, 'source audit stat continuity missing')
    require(contract.get('failures') == [], 'extraction failures reported')
    require(contract['source_stat_before'] == contract['source_stat_after'], 'source stats changed')
    require(contract['input_bands'] == INPUT_BANDS, 'contract source band order')
    require(contract['native_bands'] == NATIVE_BANDS, 'contract native band order')
    require(contract['imputation'] == {'B01':'B02','B09':'B8A'}, 'contract imputation')
    require(contract.get('std_multiplier', contract.get('normalization_std_multiplier')) == 2,
            'contract std multiplier')
    actual = sha(computed)
    require(contract['normalization_config_sha256'] == actual == COMPUTED_SHA,
            'normalization config identity changed')
    shards = contract['source_shards']
    if isinstance(shards, dict):
        shards = [dict(value, name=name) if isinstance(value, dict) else
                  {'name':name,'sha256':value} for name,value in shards.items()]
    require(isinstance(shards, list), 'contract shard inventory')
    allowed = [s for s in shards if s.get('name') in {SHARD,'0000'}]
    require(len(allowed) == 1, 'contract missing or duplicate verified0000')
    identity = allowed[0].get('actual_sha256', allowed[0].get('sha256'))
    require(identity == SHARD_SHA and allowed[0].get('valid', True), 'contract0000 identity')
    return allowed[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--computed', type=Path)
    parser.add_argument('--candidates', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FailureFixtures)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    require(all((args.prepared,args.computed,args.candidates,args.out)), 'required CLI arguments missing')
    receipt = {'status':'verifying', 'model_executed':False, 'gpu_used':False,
               'source_reread_requested':args.source_root is not None,
               'source_arrays_reread':False, 'source_catalog_lineage_verified':False,
               'cloud_quality_certified':False, 'exact_footprint_verified':False,
               'comparative_training_ready':False, 'checks':[],
               'script_sha256':sha(__file__), 'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        contract = json.loads((args.prepared/'contract.json').read_text())
        source_record = verify_contract(contract, args.computed)
        rows = read_jsonl(args.prepared/'manifest.jsonl')
        candidates = read_jsonl(args.candidates)
        receipt['inventory'] = verify_inventory(rows, candidates)
        receipt['candidate_manifest_sha256'] = sha(args.candidates)
        require(contract['candidate_manifest_sha256'] == sha(args.candidates) == CANDIDATES_SHA,
                'frozen candidate manifest hash mismatch')
        require(sha(args.prepared/'frozen_candidates.jsonl') == CANDIDATES_SHA,
                'copied frozen candidate manifest changed')
        require(sha(args.prepared/'quality_policy.json') == contract['quality_policy_sha256'],
                'quality policy hash mismatch')
        require(json.loads((args.prepared/'partition.json').read_text()) ==
                {str(r['patch_id']):r['training_partition'] for r in rows}, 'partition artifact mismatch')
        if args.source_root:
            before = (args.source_root/SHARD).stat()
            require(before.st_size == 19990546644, 'verified shard size changed')
            require(contract['source_stat_before'] == {'bytes':before.st_size,'mtime_ns':before.st_mtime_ns},
                    'prepared/source current stat mismatch')
            if 'mtime_ns' in source_record:
                require(before.st_mtime_ns == source_record['mtime_ns'], 'verified shard mtime changed')
            receipt['source_catalog_lineage_verified'] = verify_source_catalog(
                rows, candidates, args.source_root)
        stats = json.loads(args.computed.read_text())['sentinel2_l2a']
        for row in rows:
            input_path, label_path = (resolved_child(args.prepared,row[k])
                                      for k in ('npz_path','label_path'))
            require(sha(input_path) == row['npz_sha256'], 'input NPZ hash mismatch')
            require(sha(label_path) == row['label_sha256'], 'label NPZ hash mismatch')
            source = source_arrays(row,args.source_root) if args.source_root else None
            with np.load(input_path,allow_pickle=False) as inputs, np.load(label_path,allow_pickle=False) as labels:
                receipt['checks'].append(verify_arrays(row,inputs,labels,stats,source))
        if args.source_root:
            after = (args.source_root/SHARD).stat()
            require((before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns),
                    'source changed during verification')
            receipt['source_arrays_reread'] = True
        receipt['status'] = ('passed_independent_source_and_input_checks_not_model_performance'
                             if args.source_root else 'passed_internal_checks_source_reread_not_done')
        receipt['clean_training_eligible_count'] = sum(r['clean_training_eligible'] for r in receipt['checks'])
        strict_count = sum(r['strict_no_missing_input_eligible'] for r in receipt['checks'])
        receipt['strict_no_missing_input_count'] = strict_count
        receipt['adapter_required_count'] = len(rows) - strict_count
        require(contract['strict_no_missing_cases'] == strict_count and
                contract['cases_requiring_missing_aware_adapter'] == len(rows)-strict_count,
                'contract missing-data counts mismatch')
        require(contract['records'] == 80 and contract['roles'] == receipt['inventory']['role_counts'] and
                contract['partitions'] == receipt['inventory']['partition_counts'], 'contract inventory counts')
    except Exception as error:
        receipt['status'] = 'failed_input_verification'
        receipt['error'] = str(error)
        receipt['traceback'] = traceback.format_exc()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    receipt['verified_case_count'] = len(receipt['checks'])
    receipt['limitations'] = ['No model execution, cloud certification, or exact footprint validation',
                             'Original acquisition/calibration not independently reacquired',
                             'Public annual crop labels are not expert correction or change labels']
    args.out.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('checks','traceback')}))
    return 1 if receipt['status'].startswith('failed') else 0


class FailureFixtures(unittest.TestCase):
    def setUp(self):
        self.stats = {b:{'mean':1000.0+i*11,'std':100.0+i} for i,b in enumerate(NATIVE_BANDS)}
        full = (np.arange(10*10*4*4).reshape(10,10,4,4) + 100).astype(np.int16)
        full[0,0,0,0] = -10000
        full[0,1,0,1] = -37  # Legitimate negative value must not be silently dropped.
        dates = [20200101+i for i in range(10)]
        indices = [i*9//7 for i in range(8)]
        raw = full[indices].copy()
        valid = ~(raw == -10000).any(axis=1)
        native = np.transpose(raw[:,CHANNEL_MAP].astype(np.float32),(2,3,0,1))
        mean=np.array([self.stats[b]['mean'] for b in NATIVE_BANDS])
        std=np.array([self.stats[b]['std'] for b in NATIVE_BANDS])
        normalized=((native-(mean-2*std))/(4*std)).astype(np.float32)
        normalized[np.transpose(~valid,(1,2,0))]=0
        self.row={'patch_id':'fixture','all_dates_yyyymmdd':dates,'selected_indices':indices,
                  'selected_dates':[dates[i] for i in indices],'clean_training_eligible':False,
                  'training_partition':'train_pool','strict_no_missing_input_eligible':False,
                  'supervised_training_allowed':True}
        self.inputs={'raw_selected_s2':raw,'normalized_s2':normalized,
                     'timestamps':np.array([[i+1,0,2020] for i in indices],dtype=np.int64),
                     'band_observed':np.array([True]*10+[False]*2),
                     'nodata_observed':raw == -10000,'observation_valid':valid}
        semantic=(np.arange(16).reshape(4,4)%5).astype(np.int64); semantic[0,0]=19
        instances=np.arange(16).reshape(4,4).astype(np.int64)
        self.labels={'semantic':semantic,'instances':instances,'label_valid':semantic!=19,
                     'crop_label_valid':(semantic>=1)&(semantic<=18)}
        self.source={'s2':full,'semantic':np.stack([semantic,np.zeros_like(semantic),np.zeros_like(semantic)]),
                     'instance':instances.copy()}

    def check(self):
        return verify_arrays(self.row,self.inputs,self.labels,self.stats,self.source,hw=4)

    def test_valid_fixture_preserves_negative_and_marks_nodata(self):
        result=self.check()
        self.assertEqual(result['negative_non_nodata_values'],1)
        self.assertEqual(result['invalid_pixel_dates'],1)
        self.assertFalse(result['cloud_quality_certified'])

    def test_wrong_dates_and_frames_detected(self):
        self.row['selected_dates'][0]=20200102
        with self.assertRaisesRegex(ValueError,'selected dates'): self.check()
        self.setUp(); self.inputs['raw_selected_s2'][1,0,1,1]+=1
        # Bypass the transform check by modifying source-independent expected output
        # nowhere: this still catches corrupt raw, even before the source comparison.
        with self.assertRaises(ValueError): self.check()
        self.setUp(); self.source['s2'][self.row['selected_indices'][1],0,1,1]+=1
        with self.assertRaisesRegex(ValueError,'source frame'): self.check()

    def test_double_normalization_or_band_swap_detected(self):
        self.inputs['normalized_s2']/=1000
        with self.assertRaisesRegex(ValueError,'normalization'): self.check()
        self.setUp(); self.inputs['normalized_s2']=self.inputs['normalized_s2'][...,[1,0]+list(range(2,12))]
        with self.assertRaisesRegex(ValueError,'normalization'): self.check()

    def test_wrong_invalid_fill_and_availability_detected(self):
        self.inputs['normalized_s2'][0,0,0,0]=1
        with self.assertRaisesRegex(ValueError,'invalid-fill'): self.check()
        self.setUp(); self.inputs['band_observed'][-1]=True
        with self.assertRaisesRegex(ValueError,'imputed'): self.check()

    def test_nodata_and_strict_eligibility_detected(self):
        self.inputs['observation_valid'][0,0,0]=True
        with self.assertRaisesRegex(ValueError,'observation valid'): self.check()
        self.setUp(); self.row['clean_training_eligible']=True
        with self.assertRaisesRegex(ValueError,'eligibility'): self.check()
        self.setUp(); self.row['training_partition']='source_bank'
        with self.assertRaisesRegex(ValueError,'bank isolation'): self.check()
        self.row['supervised_training_allowed']=False
        self.assertFalse(self.check()['clean_training_eligible'])

    def test_label_void_source_and_leakage_detected(self):
        self.labels['label_valid'][0,0]=True
        with self.assertRaisesRegex(ValueError,'void'): self.check()
        self.setUp(); self.source['instance'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'instance source'): self.check()
        self.setUp(); self.inputs['semantic']=self.labels['semantic']
        with self.assertRaisesRegex(ValueError,'gold leaked'): self.check()

    def test_partition_hash_count_and_candidate_checks(self):
        import copy
        candidates=[]
        for parent,count,role in [('t32ulu',32,'train'),('t31tfj',32,'train'),('t31tfm',16,'development')]:
            for i in range(count):
                candidates.append({'patch_id':str(len(candidates)), 'parent_tile':parent,
                                   'role':role,'all_dates_yyyymmdd':[20200101]})
        rows=copy.deepcopy(candidates)
        banks=set()
        for parent in ('t32ulu','t31tfj'):
            ids=[r['patch_id'] for r in candidates if r['parent_tile']==parent]
            ids.sort(key=lambda v:hashlib.sha256(('oe8-bank-v0:'+v).encode()).hexdigest())
            banks.update(ids[:8])
        for row in rows:
            pid=row['patch_id']
            row.update(training_partition='dev_query' if row['role']=='development' else
                       ('source_bank' if pid in banks else 'train_pool'),
                       npz_path=f'inputs/{pid}.npz',label_path=f'labels/{pid}.npz')
        self.assertEqual(verify_inventory(rows,candidates)['candidate_count'],80)
        bad=copy.deepcopy(rows); bad[0]['training_partition']='dev_query'
        with self.assertRaisesRegex(ValueError,'leakage'): verify_inventory(bad,candidates)
        bad=copy.deepcopy(rows); bad[0]['all_dates_yyyymmdd']=[20200202]
        with self.assertRaisesRegex(ValueError,'candidate'): verify_inventory(bad,candidates)
        with self.assertRaises(ValueError): verify_inventory(rows[:-1],candidates)

    def test_source_catalog_detects_coordinated_wrong_payload_reference(self):
        import copy
        refs={name:f'/vsisubfile/{i}_20,/data/{SHARD}'
              for i,name in enumerate(('s2','semantic','instance'))}
        candidate={'patch_id':'8','parent_tile':'t32ulu','source_packed_reference':'outer:8',
                   'all_dates_yyyymmdd':[20200101,20200102]}
        row=dict(candidate, source_payloads={name:{'reference':ref} for name,ref in refs.items()})
        outer={'patch_id':'8','tile':'t32ulu','internal:subfile':'outer:8',
               'tortilla:data_split':'train','dates':[20200101,20200102]}
        verify_catalog_binding(row,candidate,outer,refs,[20200101,20200102])
        bad=copy.deepcopy(row); bad['source_payloads']['s2']['reference']='another-valid-payload'
        with self.assertRaisesRegex(ValueError,'does not belong'):
            verify_catalog_binding(bad,candidate,outer,refs,[20200101,20200102])
        for key,value in [('patch_id','9'),('tile','t31tfm'),('internal:subfile','outer:9'),
                          ('dates',[20200101,20200103])]:
            bad_outer=dict(outer); bad_outer[key]=value
            with self.assertRaises(ValueError):
                verify_catalog_binding(row,candidate,bad_outer,refs,[20200101,20200102])


if __name__ == '__main__':
    sys.exit(main())
