import copy,json,unittest
from pathlib import Path
from build_contexts import prepare,render,CONDITIONS
ROOT=Path(__file__).resolve().parents[2]
CAT=ROOT/"artifacts/oe10_expansion_catalog_20260928/oe10_expansion_catalog_v0/runtime_v2/episodes_train.jsonl"
OBJ=ROOT/"artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/episodes/source_objects.jsonl"

def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
class Boundaries(unittest.TestCase):
    def setUp(self):
        self.episodes=rows(CAT);self.objects=rows(OBJ)
        self.cards={"expert_reviewed":False,"patch_level_expert_annotations":0,"sources":[{"id":"test_only"}],
          "cards":[{"class_id":c,"card_id":str(c),"concept_name":"concept "+str(c),"claim_text":"synthetic test fact "+str(c),"transfer_limit":"test only", "source_ids":["test_only"]} for c in (8,14)]}
    def test_full_catalog_and_projection(self):
        r=prepare(self.episodes,self.objects,self.cards)
        self.assertEqual(len(r),384);self.assertEqual(len({x['audit_only']['query_patch_id_audit_only'] for x in r}),48)
        self.assertTrue(all(set(c)=={'instruction','positive','counterexample'} for x in r for c in x['model_context_by_condition'].values()))
    def test_query_gold_rejected(self):
        self.episodes[0]['target_pixels']=99
        with self.assertRaises(AssertionError):prepare(self.episodes,self.objects,self.cards)
    def test_dev_rejected(self):
        self.episodes[0]['split']='dev'
        with self.assertRaises(AssertionError):prepare(self.episodes,self.objects,self.cards)
    def test_bank_support_rejected(self):
        key=self.episodes[0]['support_pairs'][0]['positive']['object_key']
        next(o for o in self.objects if o['object_key']==key)['episode_role']='source_bank'
        with self.assertRaises(KeyError):prepare(self.episodes,self.objects,self.cards)
    def test_expert_and_missing_source_rejected(self):
        self.cards['expert_reviewed']=True
        with self.assertRaises(ValueError):prepare(self.episodes,self.objects,self.cards)
        self.cards['expert_reviewed']=False;self.cards['sources']=[]
        with self.assertRaises(ValueError):prepare(self.episodes,self.objects,self.cards)
    def test_swap_preserves_names_removal_is_exact(self):
        r=prepare(self.episodes[:1],self.objects,self.cards)[0]['model_context_by_condition']
        self.assertEqual(r['names_only'],r['removed_knowledge_diagnostic'])
        for role in ('positive','counterexample'):
            name=r['names_only'][role]
            self.assertTrue(r['matched_knowledge'][role].startswith(name+'.'))
            self.assertTrue(r['swapped_knowledge_diagnostic'][role].startswith(name+'.'))
            self.assertNotEqual(r['matched_knowledge'][role],r['swapped_knowledge_diagnostic'][role])
    def test_irrelevant_query_metadata_does_not_change_projection(self):
        r=prepare(self.episodes[:1],self.objects,self.cards)[0]['model_context_by_condition']
        self.episodes[0]['query_parent_tile']='dummy_audit_only'
        self.episodes[0]['class_pixel_counts_audit_only']=[999]
        self.assertEqual(r,prepare(self.episodes[:1],self.objects,self.cards)[0]['model_context_by_condition'])

if __name__=='__main__':unittest.main()
