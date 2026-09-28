import copy,unittest
from pathlib import Path
import build_catalog as b
ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'artifacts/oe8_pastis_prepare_20260927/review_bundle_v0'
class TrainingBoundary(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.builder=b.module(ROOT/'code/oe8_episode_prepare_v0/episode_builder.py','frozen_builder_tests')
  cls.manifest=b.rows(BUNDLE/'manifest.jsonl')
  cls.objects=b.rows(BUNDLE/'episodes/source_objects.jsonl')
  cls.public,cls.private,_=b.make(cls.builder,cls.manifest,cls.objects)
 def test_permutation_and_gold_independence(self):
  changed=copy.deepcopy(self.manifest)
  for r in changed:r['class_pixel_counts']=[999]*20;r['label_path']='labels/poison.npz'
  public,_,_=b.make(self.builder,list(reversed(changed)),list(reversed(self.objects)))
  self.assertEqual(public,self.public)
 def test_public_gold_rejected(self):
  public=copy.deepcopy(self.public);public[0]['target_class']=8
  with self.assertRaisesRegex(ValueError,'query gold'):b.validate(public,self.private,self.manifest,self.objects)
 def test_source_bank_object_rejected(self):
  public=copy.deepcopy(self.public);bank=next(o for o in self.objects if o['class_id']==8 and o['episode_role']=='source_bank')
  first=next(e for e in public if next(p for p in self.private if p['episode_id']==e['episode_id'])['target_class']==8)
  first['support_pairs'][0]['positive']['object_key']=bank['object_key']
  with self.assertRaisesRegex(ValueError,'class/role'):b.validate(public,self.private,self.manifest,self.objects)
 def test_query_patch_support_rejected(self):
  public=copy.deepcopy(self.public);public[0]['support_pairs'][0]['positive']['patch_id']=public[0]['query_patch_id']
  with self.assertRaisesRegex(ValueError,'query/bank'):b.validate(public,self.private,self.manifest,self.objects)
 def test_k_prefix_corruption_rejected(self):
  public=copy.deepcopy(self.public);one=next(e for e in public if e['k_pairs']==1)
  two=next(e for e in public if e['base_id']==one['base_id'] and e['k_pairs']==2)
  one['support_pairs'][0]=copy.deepcopy(two['support_pairs'][1])
  with self.assertRaisesRegex(ValueError,'prefix'):b.validate(public,self.private,self.manifest,self.objects)
if __name__=='__main__':unittest.main()

