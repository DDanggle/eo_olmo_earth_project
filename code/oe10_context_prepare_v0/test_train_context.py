"""Information-boundary regression checks for the future context input draft."""
import copy,json,unittest
from pathlib import Path
import build_train_context as b
ROOT=Path(__file__).resolve().parents[2]
EP=ROOT/'artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/episodes'

class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e=b.rows(EP/'episodes_train.jsonl')[0]
        cls.objects={r['object_key']:r for r in b.rows(EP/'source_objects.jsonl') if r['episode_role']=='train_pool'}
        k=json.loads((ROOT/'config/oe10_agronomy_context_cards_20260928.json').read_text())
        cls.cards={r['class_id']:r for r in k['cards']};cls.byid={r['card_id']:r for r in cls.cards.values()}
    def test_query_gold_rejected(self):
        for field in ('target_class','target_present','target_pixels','target_mask'):
            e=copy.deepcopy(self.e);e[field]=2
            with self.assertRaises(AssertionError):b.support_annotation(e,self.objects,self.cards)
    def test_development_rejected(self):
        e=copy.deepcopy(self.e);e['split']='development'
        with self.assertRaises(AssertionError):b.support_annotation(e,self.objects,self.cards)
    def test_bank_support_rejected(self):
        objects=copy.deepcopy(self.objects);key=self.e['support_pairs'][0]['positive']['object_key'];objects[key]['episode_role']='source_bank'
        with self.assertRaises(AssertionError):b.support_annotation(self.e,objects,self.cards)
    def test_mask_lineage_mismatch_rejected(self):
        e=copy.deepcopy(self.e);e['support_pairs'][0]['positive']['mask_sha256']='0'*64
        with self.assertRaises(AssertionError):b.support_annotation(e,self.objects,self.cards)
    def test_control_preserves_named_roles(self):
        a=b.support_annotation(self.e,self.objects,self.cards)
        name=b.render_text(a,self.byid,'names_only');matched=b.render_text(a,self.byid,'matched_knowledge');swap=b.render_text(a,self.byid,'swapped_knowledge_diagnostic')
        self.assertTrue(matched.startswith(name));self.assertTrue(swap.startswith(name));self.assertNotEqual(matched,swap)
        generic=b.render_text(a,self.byid,'generic')
        for card in self.cards.values():self.assertNotIn(card['concept_name'],generic)
    def test_artifact_cohort_and_no_new_labels(self):
        out=ROOT/'artifacts/oe10_context_prepare_20260928_v1'
        r=json.loads((out/'preparation_receipt.json').read_text());q=json.loads((out/'review_queue_20.json').read_text())
        self.assertEqual(r['attachments'],2304);self.assertEqual(r['unique_train_queries'],48)
        self.assertEqual(len({n['query_patch_id_audit_only'] for n in q['cases']}),20)
        self.assertEqual(r['expert_responses'],0);self.assertFalse(r['model_integration_verified']);self.assertFalse(r['query_gold_read'])
        for f in json.loads((out/'file_manifest.json').read_text())['files']:self.assertEqual(b.sha(out/f['path']),f['sha256'])

if __name__=='__main__':unittest.main()
