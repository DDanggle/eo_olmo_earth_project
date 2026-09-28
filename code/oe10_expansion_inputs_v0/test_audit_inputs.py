import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from audit_inputs import permit_source,safe,packet_check,object_check,capacity

ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'artifacts/oe8_pastis_prepare_20260927/review_bundle_v0'
class Boundaries(unittest.TestCase):
    def test_no_dev_or_unopened_parent(self):
        for part,parent in [('dev_query','t31tfm'),('train_pool','t30uxv')]:
            with self.assertRaises(ValueError):permit_source({'training_partition':part,'parent_tile':parent})
    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'inputs').mkdir();(root/'labels').mkdir();(root/'labels/x').write_text('x')
            with self.assertRaises(ValueError):safe(root,'labels/x','inputs')
    def test_real_packet_and_normalization_corruption(self):
        row=next(json.loads(x) for x in (BUNDLE/'manifest.jsonl').read_text().splitlines() if json.loads(x)['patch_id']=='40411')
        stats=json.loads((BUNDLE/'computed.json').read_text())['sentinel2_l2a']
        with np.load(BUNDLE/'sample_packets/inputs/40411.npz') as x: inp={k:x[k] for k in x.files}
        with np.load(BUNDLE/'sample_packets/labels/40411.npz') as x: lab={k:x[k] for k in x.files}
        packet_check(row,inp,lab,stats)
        inp['normalized_s2'][0,0,0,0]+=1
        with self.assertRaises(ValueError):packet_check(row,inp,lab,stats)
    def test_query_gold_field_rejected(self):
        with self.assertRaisesRegex(ValueError,'labels in input'):packet_check({'training_partition':'train_pool','parent_tile':'t31tfj'},{'semantic':np.zeros(1)},{},{})
    def test_mask_binding_and_missing_observation(self):
        sem=np.zeros((128,128),int);ins=sem.copy();sem[:8,:8]=8;ins[:8,:8]=1
        mask=sem==8;valid=np.ones((8,128,128),bool)
        obj={'instance_id':1,'class_id':8,'class_pixel_count':64,'instance_pixel_count':64,'class_purity':1,'touches_image_border':True,'bbox_xyxy_exclusive':[0,0,8,8]}
        object_check(obj,sem,ins,valid,mask)
        with self.assertRaisesRegex(ValueError,'border flag'):object_check(dict(obj,touches_image_border=False),sem,ins,valid,mask)
        invalid=mask.copy();invalid[0,0]=False
        with self.assertRaises(ValueError):object_check(obj,sem,ins,valid,invalid)
        valid[7,0,0]=False
        with self.assertRaises(ValueError):object_check(obj,sem,ins,valid,mask)
    def test_query_exclusion_and_patch_cap(self):
        objects=[{'episode_role':'train_pool','class_id':8,'patch_id':p,'touches_image_border':False} for p in ('a','a','a','b','c')]
        self.assertEqual(capacity(objects,'train_pool',8,'a'),{'objects':2,'patches':2,'max2_per_patch_capacity':2})
        self.assertEqual(capacity(objects,'train_pool',8)['max2_per_patch_capacity'],4)
if __name__=='__main__':unittest.main()

