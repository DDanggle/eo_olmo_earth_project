"""E5 metadata and bounded CPU pooling tests; never prepare the full 2.19GiB array."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import numpy as np

STAGE=Path(__file__).resolve().parents[1];INPUTS=STAGE/'frozen_inputs'
spec=importlib.util.spec_from_file_location('e5_prepare_tests_module',STAGE/'e5_prepare_v0.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)


def read_fixture_parents(directory):
    return {name:prep.rows(directory/name)if Path(name).suffix=='.jsonl'else prep.read(directory/name)
            for name in sorted(prep.REQUIRED_PARENTS)}


class FrozenMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items=prep.rows(INPUTS/'items.jsonl');cls.c0=prep.read(INPUTS/'c0_manifest.json')
        cls.parents=read_fixture_parents(INPUTS)
    def test_fixture_ignores_appledouble_and_unpinned_sidecars(self):
        with tempfile.TemporaryDirectory(prefix='e5_parent_sidecars_')as tmp:
            directory=Path(tmp)
            for name in prep.REQUIRED_PARENTS:
                shutil.copyfile(INPUTS/name,directory/name)
                (directory/('._'+name)).write_bytes(b'\x00\x05\x16\x07\xff\xfe')
            (directory/'unregistered.json').write_text('not JSON')
            self.assertEqual(read_fixture_parents(directory),self.parents)
            self.assertEqual(set(read_fixture_parents(directory)),prep.REQUIRED_PARENTS)
    def test_actual_population_order_disjointness_and_cache_coverage(self):
        before=prep.sha(INPUTS/'items.jsonl');ordered,audit=prep.validate_population(self.items,self.c0)
        self.assertEqual([len(ordered[p])for p in ('train','test')],[4234,1755])
        for part in ('train','test'):self.assertEqual(ordered[part],[x['id']for x in self.items if x['partition']==part])
        self.assertEqual([len(audit[p]['flood_events'])for p in ('train','test')],[27,10])
        self.assertEqual([len(audit[p]['landslide_regions'])for p in ('train','test')],[7,2])
        self.assertEqual(prep.sha(INPUTS/'items.jsonl'),before)
    def test_exclusion_and_duplicate_items_rejected(self):
        for xs in [self.items[1:],self.items+[self.items[0]]]:
            with self.subTest(n=len(xs)),self.assertRaises(ValueError):prep.validate_population(xs,self.c0)
    def test_train_test_tile_leakage_rejected(self):
        items=copy.deepcopy(self.items)
        train=next(x for x in items if x['partition']=='train');test=next(x for x in items if x['partition']=='test')
        train['tile']=test['tile']
        with self.assertRaisesRegex(ValueError,'leakage: tiles'):prep.validate_population(items,self.c0)
    def test_missing_cache_or_out_of_range_index_rejected(self):
        c0=copy.deepcopy(self.c0);c0['cache_sha256'].pop(next(iter(c0['cache_sha256'])))
        with self.assertRaisesRegex(ValueError,'cache coverage'):prep.validate_population(self.items,c0)
        items=copy.deepcopy(self.items);items[0]['indices']=[0,99]
        with self.assertRaisesRegex(ValueError,'frame indices'):prep.validate_population(items,self.c0)
    def test_exact_primary_support_and_all_evaluation_ids(self):
        sets=prep.make_eval_sets(self.items,self.parents)
        self.assertEqual({k:len(v)for k,v in sets.items()},{'primary_same_prompt':902,'all_test':1755,
            'paired_flood':914,'hard_negative_flood':457,'landslide':384,'e3_subset':209})
        by_id={x['id']:x for x in self.items};primary=[by_id[i]for i in sets['primary_same_prompt']]
        self.assertEqual(sum(x['kind']=='pos'for x in primary),445)
        self.assertEqual(sum(x['kind']=='hard_neg'for x in primary),457)
        self.assertTrue(all(by_id[i]['partition']=='test'for ids in sets.values()for i in ids))
        selected=set(sets['primary_same_prompt'])
        self.assertEqual(sets['primary_same_prompt'],[x['id']for x in self.items if x['id']in selected])
    def test_quality_metadata_and_saved_support_tamper_rejected(self):
        parents=copy.deepcopy(self.parents);parents['c1_quality.jsonl'][0]['dates'][0]='1900-01-01'
        with self.assertRaisesRegex(ValueError,'quality metadata'):prep.make_eval_sets(self.items,parents)
        parents=copy.deepcopy(self.parents);event=next(iter(parents['c1_results.json']['subsets']['quality_symmetric']['per_seed']['1']['arms']['reader'].values()))
        event['strata'][0]['pos_ids'].pop()
        with self.assertRaisesRegex(ValueError,'stratum membership'):prep.make_eval_sets(self.items,parents)
    def test_batch_rng_is_continuous_per_seed_with_equal_budget(self):
        train=[x['id']for x in self.items if x['partition']=='train'];result=prep.make_batches(train)
        self.assertEqual(set(result),{'1','2','3'})
        for seed in (1,2,3):
            rng=np.random.default_rng(seed);epochs=result[str(seed)]
            self.assertEqual(sum(len(e)for e in epochs),1590)
            self.assertEqual(sum(len(b)for e in epochs for b in e),12702)
            for batches in epochs:
                self.assertEqual(len(batches),530);self.assertEqual(len(batches[-1]),2)
                actual=[i for batch in batches for i in batch]
                self.assertEqual(actual,[train[int(i)]for i in rng.permutation(4234)])
                self.assertEqual(set(actual),set(train));self.assertEqual(len(actual),len(set(actual)))
        self.assertEqual(result,prep.make_batches(train));self.assertNotEqual(result['1'][0],result['1'][1])
    def test_batch_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):prep.make_batches(['a','a'])


class StreamingPoolsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='e5_pool_test_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.original=Path('/original/frozen/root')
        self.c0={'root':str(self.original),'cache_sha256':{},'cache_stat':{}}
        self.paths={}
        for phen,folder,tile,shape,pool in [('flood','kurosiwo_s1_cache','ks_test',(3,768,48,48),6),
                                           ('landslide','olmo_streaming_dev','region_s2_test',(12,768,32,32),4)]:
            rel=Path(folder)/'single_fp16'/(tile+'.npy');path=self.root/rel;path.parent.mkdir(parents=True)
            blocks=np.arange(64,dtype=np.float16).reshape(8,8).repeat(pool,0).repeat(pool,1)
            values=np.arange(shape[0],dtype=np.float16)[:,None,None,None]*10+((np.arange(768)%2)*.5).astype(np.float16)[None,:,None,None]+blocks[None,None,:,:]
            np.save(path,values.astype(np.float16));key=str(self.original/rel);self.paths[phen]=(path,key)
            self.c0['cache_sha256'][key]=prep.sha(path);self.c0['cache_stat'][key]=prep.stat(path)
        self.items=[{'id':'land','phen':'landslide','cache_path':self.paths['landslide'][1],'indices':[0,3]},
                    {'id':'flood_post','phen':'flood','cache_path':self.paths['flood'][1],'indices':[1,2]},
                    {'id':'flood_pre','phen':'flood','cache_path':self.paths['flood'][1],'indices':[0,1]}]
        self.out=self.root/'pairs.npy'
    def update_pin(self,phen):
        path,key=self.paths[phen];self.c0['cache_sha256'][key]=prep.sha(path);self.c0['cache_stat'][key]=prep.stat(path)
    def build(self):return prep.build_pairs(self.items,self.c0,self.out,self.root)
    def test_sensor_pooling_and_global_item_row_order(self):
        audits=self.build();actual=np.load(self.out,mmap_mode='r')
        self.assertEqual(actual.shape,(3,2,64,768));self.assertEqual(actual.dtype,np.float32)
        basis=np.arange(64,dtype=np.float32)[:,None]+((np.arange(768)%2)*.5).astype(np.float32)[None,:]
        for row,item in enumerate(self.items):
            for t,index in enumerate(item['indices']):np.testing.assert_array_equal(actual[row,t],basis+10*index)
        self.assertEqual(audits[self.paths['flood'][1]]['item_rows'],[1,2])
        self.assertEqual(audits[self.paths['flood'][1]]['indices_used'],[0,1,2])
    def test_wrong_cache_hash_and_missing_file_rejected(self):
        self.c0['cache_sha256'][self.paths['flood'][1]]='0'*64
        with self.assertRaisesRegex(ValueError,'Frozen cache changed'):self.build()
        self.update_pin('flood');self.paths['flood'][0].unlink()
        with self.assertRaises(FileNotFoundError):self.build()
    def test_wrong_shape_rejected_even_when_hash_matches(self):
        path,_=self.paths['flood'];np.save(path,np.zeros((2,768,48,48),dtype=np.float16));self.update_pin('flood')
        with self.assertRaisesRegex(ValueError,'shape/dtype'):self.build()
    def test_nonfinite_used_source_rejected_even_when_hash_matches(self):
        path,_=self.paths['flood'];array=np.load(path);array[1,0,0,0]=np.nan;np.save(path,array);self.update_pin('flood')
        with self.assertRaisesRegex(ValueError,'Nonfinite referenced'):self.build()
    def test_cache_stat_mutation_during_pooling_rejected(self):
        import torch.nn.functional as F
        original=F.avg_pool2d;path,_=self.paths['flood'];old=path.stat()
        def mutate(*args,**kwargs):
            out=original(*args,**kwargs);os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns+1000));return out
        with mock.patch.object(F,'avg_pool2d',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'mutated during pooling'):self.build()
    def test_no_cache_manifest_exclusion_allowed(self):
        self.c0['cache_sha256'].pop(self.paths['landslide'][1])
        with self.assertRaisesRegex(ValueError,'coverage mismatch'):self.build()
    def test_pool_reference_uses_ids_not_reference_order_and_rejects_changed_values(self):
        self.build();array=np.load(self.out)
        # Additional fields mirror the parent item contract while row order differs.
        items=[dict(item,tile=item['id'],kind='pos',dates=['2020-01-01','2020-01-02'],answer='yes',cluster='event')for item in self.items]
        refs=[dict(items[2],pair_key='third'),dict(items[0],pair_key='first')]
        archive=self.root/'ref.npz';np.savez(archive,third=array[2],first=array[0])
        self.assertEqual(prep.compare_pool_rows(items,self.out,refs,archive),2)
        changed=array[2].copy();changed[0,0,0]+=1;np.savez(archive,third=changed,first=array[0])
        with self.assertRaisesRegex(ValueError,'Pooled values differ'):prep.compare_pool_rows(items,self.out,refs,archive)
    def test_pool_reference_rejects_metadata_change(self):
        self.build();array=np.load(self.out);items=[dict(item,tile=item['id'],kind='pos',dates=['a','b'],answer='yes',cluster='event')for item in self.items]
        ref=dict(items[0],pair_key='p',dates=['wrong','b']);archive=self.root/'ref.npz';np.savez(archive,p=array[0])
        with self.assertRaisesRegex(ValueError,'metadata mismatch'):prep.compare_pool_rows(items,self.out,[ref],archive)


class PreparationFailureTests(unittest.TestCase):
    def test_bad_bundled_hash_preserves_failure_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory(prefix='e5_prepare_failure_')as tmp:
            root=Path(tmp).resolve();bundle=root/'bundle';bundle.mkdir()
            shutil.copyfile(STAGE/'e5_prepare_v0.py',bundle/'e5_prepare_v0.py')
            for name in prep.SOURCE_FILES[1:]:(bundle/name).write_text('# unused preparation-test peer\n')
            module_spec=importlib.util.spec_from_file_location('e5_prepare_failure_module',bundle/'e5_prepare_v0.py');module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
            cfg=prep.read(STAGE/'e5_equal_budget_prereg_v0.json');cfg['parents']['c0_manifest.json']='0'*64
            config=root/'plan.json';prep.write(config,cfg);out=root/'e5_equal_budget_v0'
            with self.assertRaisesRegex(ValueError,'Frozen bundled input changed'):module.prepare(config,INPUTS,out,bundle,root)
            self.assertEqual(prep.read(out/'status.json')['status'],'invalid_preparation')
            self.assertEqual(prep.read(out/'failure.json')['phase'],'freezing_inputs');self.assertFalse((out/'pairs.npy').exists())
            old=(out/'failure.json').read_bytes()
            with self.assertRaisesRegex(ValueError,'never resume'):module.prepare(config,INPUTS,out,bundle,root)
            self.assertEqual((out/'failure.json').read_bytes(),old)


if __name__=='__main__':unittest.main()
