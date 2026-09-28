import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import native_replay as n


ACTUAL = Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/oe4_gpu1_retry_20260927/snapshot_02/runs/native_development_gpu1_v2/data_manifest.json')


class FakeH5(dict):
    def __enter__(self): return self
    def __exit__(self,*args): pass


class ReplayTests(unittest.TestCase):
    def test_actual_manifest_and_full_cycle_exclude_every_dev_row(self):
        rows=n.load_train_manifest(ACTUAL)
        self.assertEqual(len(rows),64)
        cycle=[r for step in range(32) for r in n.batch_rows(rows,step)]
        self.assertEqual(cycle,rows)
        self.assertEqual(n.batch_rows(rows,32),rows[:2])
        dev={r['file'] for r in json.loads(ACTUAL.read_text())['selected'] if r['split']=='dev_diagnostic'}
        self.assertFalse({r['file'] for r in cycle}&dev)
        for bad in (-1,True,1.5):
            with self.assertRaises(ValueError):n.batch_rows(rows,bad)

    def test_manifest_mutation_and_dev_injection_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';p.write_text(ACTUAL.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'pinned'):n.load_train_manifest(p)
        rows=n.load_train_manifest(ACTUAL);rows[0]['split']='dev_diagnostic'
        with self.assertRaisesRegex(ValueError,'train64'):n.batch_rows(rows,0)

    def fixture(self,root):
        p=root/'sample.h5';p.write_bytes(b'fixture')
        raw=np.ones((32,32,3,12),np.float32)*100
        wc=np.ones((32,32,1,1),np.float32)*10
        ts=np.array([[1,0,2020],[2,0,2020],[3,0,2020],[4,0,2020]],np.int64)
        h=FakeH5(sentinel2_l2a=raw,worldcover=wc,timestamps=ts,latlon=np.array([35.,128.]))
        h['missing_timesteps_masks/sentinel2_l2a']=np.array([0,1,1,1])
        row={'split':'train_diagnostic','file':str(p),'file_bytes':7,'source_s2_shape':list(raw.shape),
             'source_worldcover_shape':list(wc.shape),'compact_timesteps':[0,1],'timestamp_indices':[1,2],
             'crop_yxhw':[0,0,32,32],'timestamps_day_month0_year':ts[[1,2]].tolist(),'latlon':[35.,128.]}
        row['raw_crop_sha256']=hashlib.sha256(raw[:,:,:2].tobytes()+wc.tobytes()+ts[[1,2]].tobytes()).hexdigest()
        seen=[]
        norm=SimpleNamespace(normalize=lambda mod,x:(seen.append(mod) or x/100))
        reader=SimpleNamespace(File=lambda *args:h)
        return row,h,reader,norm,SimpleNamespace(get=lambda x:x),seen

    def test_crop_normalizes_once_and_ignores_unused_dates(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(n,'PINNED_DATA_ROOT',Path(tmp)):
            row,h,reader,norm,mod,seen=self.fixture(Path(tmp))
            h['sentinel2_l2a'][:,:,2]=999  # Unselected date cannot affect pinned crop.
            sample=n.read_native_sample(row,reader,norm,mod,-10000)
            self.assertEqual(sample['sentinel2_l2a'].shape,(32,32,2,12))
            self.assertTrue((sample['sentinel2_l2a']==1).all())
            self.assertEqual(seen,['sentinel2_l2a','worldcover'])
            self.assertEqual(sample['timestamps'].tolist(),row['timestamps_day_month0_year'])

    def test_raw_mutation_date_mapping_and_dev_reader_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(n,'PINNED_DATA_ROOT',Path(tmp)):
            row,h,reader,norm,mod,seen=self.fixture(Path(tmp))
            h['sentinel2_l2a'][0,0,0,0]+=1
            with self.assertRaisesRegex(ValueError,'raw crop hash'):n.read_native_sample(row,reader,norm,mod,-10000)
            h['sentinel2_l2a'][0,0,0,0]-=1
            wrong=copy.deepcopy(row);wrong['timestamp_indices']=[0,1]
            with self.assertRaisesRegex(ValueError,'index mismatch'):n.read_native_sample(wrong,reader,norm,mod,-10000)
            row['split']='dev_diagnostic'
            with self.assertRaisesRegex(ValueError,'dev row'):n.read_native_sample(row,reader,norm,mod,-10000)
            self.assertEqual(seen,[])

    def test_bridge_requires_exact_recipe_config_and_preserves_source_dict(self):
        recipe='''ONLY_DECODE_MODALITIES = [Modality.SENTINEL2_L2A.name]
def build_train_module_config():
    return ContrastiveLatentMIMTrainModuleConfig(
        loss_config=LossConfig(loss_config={"type":"MSE","decode_modalities":ONLY_DECODE_MODALITIES}),
        contrastive_config=LossConfig(loss_config={"type":"InfoNCE","temperature":0.1}))
'''
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'scripts/official/v1_2/base.py';p.parent.mkdir(parents=True);p.write_text(recipe)
            config={'loss_config':{'loss_config':{'decode_modalities':['sentinel2_l2a']}},
                    'contrastive_config':{'loss_config':{'temperature':.1}}}
            mod=SimpleNamespace(SENTINEL2_L2A=SimpleNamespace(name='sentinel2_l2a'))
            with patch.object(n,'PINNED_V12_RECIPE_SHA256',n.sha256(p)):
                bridged,report=n.bridge_public_loss_registry_keys(config,n.PINNED_PUBLIC_CONFIG_SHA256,root,mod)
                self.assertEqual(bridged['loss_config']['loss_config']['type'],'MSE')
                self.assertEqual(len(report['changes']),2)
                self.assertNotIn('type',config['loss_config']['loss_config'])
                with self.assertRaisesRegex(ValueError,'unreviewed'):
                    n.bridge_public_loss_registry_keys(config,'unreviewed',root,mod)
                config['contrastive_config']['loss_config']['temperature']=.2
                with self.assertRaisesRegex(ValueError,'parameters differ'):
                    n.bridge_public_loss_registry_keys(config,n.PINNED_PUBLIC_CONFIG_SHA256,root,mod)


if __name__=='__main__':unittest.main(verbosity=2)
