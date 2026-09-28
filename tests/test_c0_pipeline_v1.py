"""Synthetic CPU-only end-to-end C0 contract checks; no research data/models."""
import collections
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
sys.path.insert(0, str(Path(__file__).parent))
import c0_linear_view_probe_v1 as c0

class Pipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='c0_synthetic_')
        self.root = Path(self.temp.name); self.out = self.root / 'out'; self.out.mkdir()
        self.original_shapes = dict(c0.SHAPES)
        c0.SHAPES.update(flood=(3, 768, 2, 2), landslide=(12, 768, 2, 2))
        config_path = next(p for p in (Path(__file__).with_name('c0_linear_view_prereg_v1.json'), Path(__file__).resolve().parents[1] / 'config/c0_linear_view_prereg_v1.json') if p.is_file())
        self.cfg = json.loads(config_path.read_text())
        sources = {k: [] for k in c0.SOURCES}; self.expected = []; contracts = []
        for split in ('train', 'test'):
            for phen in ('flood', 'landslide'):
                tile = phen + '_' + split; fold = split if phen == 'flood' else split + '_region'
                if phen == 'landslide':
                    dates = [f'2020-01-{d:02d}' for d in range(1, 16)]
                    contracts.append(dict(sample_id=tile, scl_clear_fraction=[1.] * 15, times=dates))
                parent = self.root / ('kurosiwo_s1_cache/single_fp16' if phen == 'flood' else 'olmo_streaming_dev/single_fp16'); parent.mkdir(parents=True, exist_ok=True)
                tensor = np.full(c0.SHAPES[phen], -1, dtype=np.float16); tensor[2] = 1; np.save(parent / (tile + '.npy'), tensor)
                for kind, answer, ix in [('pos','yes',(1,2)),('neg','no',(0,1))]:
                    it = dict(id=tile+'_'+kind, tile=tile, type='Q1', fold=fold, kind=kind, answer=answer)
                    if phen == 'flood': it.update(event=111 if split == 'train' else 222, slots=[['pre_1','pre_2','post'][i] for i in ix], dates=['imputed','imputed'])
                    else: it.update(dates=[dates[i] for i in ix])
                    sources['flood' if phen == 'flood' else 'landslide_'+split].append(it)
                    self.expected.append(dict(it, phen=phen, partition=split))
                if phen == 'flood':
                    htile=tile+'_dry'; np.save(parent/(htile+'.npy'), np.full(c0.SHAPES[phen],-1,dtype=np.float16))
                    h=dict(id=htile, tile=htile, type='Q1', fold=split, kind='hard_neg',answer='no',event=111 if split=='train' else 222,slots=['pre_2','post'],dates=['imputed','imputed'])
                    sources['flood'].append(h);self.expected.append(dict(h,phen=phen,partition=split))
        sources['contract']=contracts
        for name, rows in sources.items():
            p=self.root/c0.SOURCES[name];p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        references=[{k:it[k] for k in ('id','tile','phen','kind','fold')} | {'text_gold':it['answer']} for it in self.expected if it['partition']=='test']
        self.cfg['expected_source_sha256']={}
        for seed in (1,2,3):
            rel=f'e2_multi_reader_v0/reader_seed{seed}/answers_real_all.jsonl';p=self.root/rel;p.parent.mkdir(parents=True);p.write_text(''.join(json.dumps(r)+'\n' for r in references));self.cfg['expected_source_sha256'][rel]=c0.sha(p)
        self.cfg['expected_counts']=dict(collections.Counter(f"{i['partition']}|{i['phen']}|{i['answer']}" for i in self.expected))
        config=self.root/'config.json';config.write_text(json.dumps(self.cfg));self.a=SimpleNamespace(root=str(self.root),config=str(config))
    def tearDown(self):
        c0.SHAPES.clear();c0.SHAPES.update(self.original_shapes);self.temp.cleanup()
    def test_prepare_then_six_fits(self):
        with contextlib.redirect_stdout(io.StringIO()):
            c0.prepare(self.a,self.out)
            self.assertTrue(c0.run(self.a,self.out))
        manifest=c0.load(self.out/'manifest.json');result=c0.load(self.out/'results.json')
        self.assertEqual(len(result['fits']),6);self.assertEqual(len(result['model_sha256']),6)
        self.assertEqual(result['metrics']['landslide']['later_minus_pair']['ci95_delta'],None)
        self.assertEqual(result['metrics']['flood']['pair']['hard_negative']['n'],1)
        self.assertEqual(len(c0.lines(self.out/'predictions.jsonl')),15)
        self.assertEqual(len(manifest['cache_sha256']),6)
        self.assertTrue(manifest['tuple_counts']);self.assertEqual(c0.load(self.out/'status.json')['status'],'complete')
        self.assertTrue(all(x['converged'] for x in result['fits'].values()))
    def add_missing(self, *, contract_missing=False):
        rel=c0.SOURCES['landslide_train' if contract_missing else 'flood']
        item=dict(id='excluded_qa',tile='excluded_tile',type='Q1',fold='train',kind='pos',answer='yes',event=111,slots=['INVALID_SLOT'],dates=['bad_date'])
        with (self.root/rel).open('a') as f:f.write(json.dumps(item)+'\n')
        cache=self.root/('olmo_streaming_dev/single_fp16' if contract_missing else 'kurosiwo_s1_cache/single_fp16')/'excluded_tile.npy'
        if contract_missing:np.save(cache,np.zeros(c0.SHAPES['landslide'],dtype=np.float16))
        return cache
    def test_missing_extra_has_frozen_explicit_exclusion_before_indices(self):
        self.add_missing()
        with contextlib.redirect_stdout(io.StringIO()):c0.prepare(self.a,self.out)
        audit=c0.load(self.out/'eligibility_audit.json');excluded=audit['excluded']
        self.assertEqual(len(excluded),1);self.assertEqual(excluded[0]['id'],'excluded_qa')
        self.assertEqual(excluded[0]['partition'],'train');self.assertEqual(excluded[0]['phen'],'flood')
        self.assertEqual(excluded[0]['reasons'],['missing_cache'])
        self.assertNotIn('excluded_qa',[i['id'] for i in c0.lines(self.out/'items.jsonl')])
        self.assertEqual(c0.load(self.out/'manifest.json')['eligibility_sha256'],c0.sha(self.out/'eligibility_audit.json'))
    def test_missing_contract_excluded_before_bad_date(self):
        self.add_missing(contract_missing=True)
        with contextlib.redirect_stdout(io.StringIO()):c0.prepare(self.a,self.out)
        excluded=c0.load(self.out/'eligibility_audit.json')['excluded']
        self.assertEqual(excluded[0]['reasons'],['missing_contract'])
    def test_expected_population_missing_cache_rejected_with_audit(self):
        (self.root/'kurosiwo_s1_cache/single_fp16/flood_train_dry.npy').unlink()
        with self.assertRaisesRegex(ValueError,'population counts'):c0.population(self.root,self.cfg,self.out/'eligibility_audit.json')
        self.assertEqual(c0.load(self.out/'eligibility_audit.json')['excluded'][0]['id'],'flood_train_dry')
    def test_test_id_mismatch_rejected_even_if_counts_modified(self):
        (self.root/'kurosiwo_s1_cache/single_fp16/flood_test_dry.npy').unlink()
        self.cfg['expected_counts']['test|flood|no']-=1
        with self.assertRaisesRegex(ValueError,'exact test ID'):c0.population(self.root,self.cfg,self.out/'eligibility_audit.json')
    def test_excluded_cache_appearing_after_freeze_rejected(self):
        cache=self.add_missing()
        with contextlib.redirect_stdout(io.StringIO()):c0.prepare(self.a,self.out)
        np.save(cache,np.zeros(c0.SHAPES['flood'],dtype=np.float16))
        with self.assertRaisesRegex(ValueError,'eligibility changed'):c0.run(self.a,self.out)
        self.assertFalse((self.out/'partial_fits.json').exists())
    def test_eligible_cache_disappearing_after_freeze_rejected(self):
        with contextlib.redirect_stdout(io.StringIO()):c0.prepare(self.a,self.out)
        (self.root/'kurosiwo_s1_cache/single_fp16/flood_train_dry.npy').unlink()
        with self.assertRaisesRegex(ValueError,'eligibility changed'):c0.run(self.a,self.out)
    def test_eligibility_change_during_extraction_rejected(self):
        cache=self.add_missing();original=c0.spatial_mean
        def changed(path,shape):
            result=original(path,shape)
            if not cache.exists():np.save(cache,np.zeros(c0.SHAPES['flood'],dtype=np.float16))
            return result
        with mock.patch.object(c0,'spatial_mean',side_effect=changed),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'eligibility changed'):c0.prepare(self.a,self.out)
    def test_already_read_cache_changed_later_in_prepare_rejected(self):
        original=c0.spatial_mean;seen=[]
        def changed(path,shape):
            result=original(path,shape)
            if len(seen)==1:
                a=np.load(seen[0]);a[0,0,0,0]=5;np.save(seen[0],a)
            seen.append(path)
            return result
        with mock.patch.object(c0,'spatial_mean',side_effect=changed),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'Used cache changed during prepare'):c0.prepare(self.a,self.out)
    def test_nonfinite_eligible_is_failure_not_exclusion(self):
        path=self.root/'kurosiwo_s1_cache/single_fp16/flood_train_dry.npy'
        a=np.load(path);a[0,0,0,0]=np.nan;np.save(path,a)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError,'Nonfinite cache'):c0.prepare(self.a,self.out)
        self.assertEqual(c0.load(self.out/'eligibility_audit.json')['excluded'],[])
    def test_source_mutation_rejected_before_fitting(self):
        with contextlib.redirect_stdout(io.StringIO()):c0.prepare(self.a,self.out)
        with (self.root/c0.SOURCES['flood']).open('a') as f:f.write('\n')
        with self.assertRaisesRegex(ValueError,'Frozen source changed'):c0.run(self.a,self.out)
        self.assertFalse((self.out/'partial_fits.json').exists())
    def test_changed_config_solver_rejected(self):
        self.cfg['fit']['lambda']=.1;Path(self.a.config).write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError,'Config differs'):c0.prepare(self.a,self.out)

if __name__=='__main__':unittest.main()
