from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import build_human_pilot as b


def options(n=48):
    return [{'patch_id':str(i),'parent_tile':'r0' if i%2==0 else 'r1',
             'reference_category':c,'instance_id':1,'episode_id':f'{i}:{c}',
             'canonical_hash':str(i)} for i in range(n) for c in b.QUOTAS]


def fixture(root):
    prepared=root/'prepared';episodes=root/'episodes'
    (prepared/'inputs').mkdir(parents=True);(prepared/'labels').mkdir()
    (episodes/'support_masks').mkdir(parents=True);(episodes/'scoring').mkdir()
    raw=np.full((8,10,128,128),1000,dtype=np.int16)
    np.savez_compressed(prepared/'inputs/shared.npz',raw_selected_s2=raw,observation_valid=np.ones((8,128,128),dtype=np.bool_))
    inst=np.zeros((128,128),dtype=np.int64);inst[24:40,45:61]=1
    sem=np.zeros_like(inst);sem[inst==1]=1
    inst[70:78,70:78]=2;sem[70:78,70:78]=19  # Void candidate must never be selected.
    np.savez_compressed(prepared/'labels/shared.npz',semantic=sem,instances=inst)
    np.savez_compressed(episodes/'support_masks/shared.npz',mask=inst==1)
    ih=b.sha(prepared/'inputs/shared.npz');lh=b.sha(prepared/'labels/shared.npz');mh=b.sha(episodes/'support_masks/shared.npz')
    rows=[];eps=[];scores=[]
    for i in range(48):
        pid=f'SOURCE_PRIVATE_{i:02d}'
        rows.append({'patch_id':pid,'parent_tile':'r0' if i%2==0 else 'r1','training_partition':'train_pool',
                     'npz_path':'inputs/shared.npz','npz_sha256':ih,'label_path':'labels/shared.npz',
                     'label_sha256':lh,'selected_dates':[20190101+j for j in range(8)]})
        support={kind:{'patch_id':f'SOURCE_PRIVATE_{(i+delta)%48:02d}','input_npz':'inputs/shared.npz',
            'input_sha256':ih,'mask_npz':'support_masks/shared.npz','mask_sha256':mh,'object_key':f'PRIVATE_OBJ_{i}_{delta}',
            'dates_yyyymmdd':rows[-1]['selected_dates']} for kind,delta in [('positive',1),('counterexample',2)]}
        for j,(target,counter) in enumerate([(1,2),(2,1),(2,3)]):
            eid=f'PRIVATE_EPISODE_{i}_{j}'
            eps.append({'query_patch_id':pid,'episode_id':eid,'pair_id':f'PRIVATE_PAIR_{j}','k_pairs':1,'support_pairs':[support]})
            scores.append({'episode_id':eid,'target_class':target,'counter_class':counter})
    for path,items in [(prepared/'manifest.jsonl',rows),(episodes/'episodes_train.jsonl',eps),(episodes/'scoring/scoring_train.jsonl',scores)]:
        path.write_text(''.join(json.dumps(x)+'\n' for x in items))
    return prepared,episodes,rows


class SamplingV2Test(unittest.TestCase):
    def test_original_queries_preserved_unique_deterministic_and_anonymous_order(self):
        opts=options();original=[str(i) for i in range(20)]
        first,meta=b.select_stratified(opts,original,[str(i) for i in range(48)])
        second,_=b.select_stratified(list(reversed(opts)),list(reversed(original)),[str(i) for i in range(48)])
        self.assertEqual(first,second)
        self.assertEqual({o['patch_id'] for o in first},set(original))
        self.assertTrue(meta['same_v1_query_ids_retained'])
        self.assertTrue(meta['balanced_per_stratum_parent'])
        self.assertEqual(Counter(o['reference_category'] for o in first),b.QUOTAS)
        self.assertEqual(Counter(o['parent_tile'] for o in first),{'r0':10,'r1':10})
        self.assertEqual([o['patch_id'] for o in first],sorted(original,key=lambda pid:b.rank(b.VERSION+':anonymous-order',pid)))
        self.assertNotEqual([o['reference_category'] for o in first],['target']*8+['counterexample']*8+['neither']*4)

    def test_fallback_only_when_needed_and_no_duplicate_padding(self):
        opts=[o for o in options() if not (int(o['patch_id'])<20 and o['reference_category']=='counterexample')]
        selected,meta=b.select_stratified(opts,[str(i) for i in range(20)],[str(i) for i in range(48)])
        self.assertEqual(meta['chosen_cohort'],'all_train_pool48')
        self.assertEqual(len({o['patch_id'] for o in selected}),20)
        with self.assertRaisesRegex(ValueError,'No 20-unique-query'):
            b.select_stratified(options(19),[str(i) for i in range(19)],[str(i) for i in range(19)])

    def test_inventory_purity_fixed_coordinates_and_reference_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared,episodes,rows=fixture(Path(tmp))
            eps=b.read_jsonl(episodes/'episodes_train.jsonl');k1={r['patch_id']:[e for e in eps if e['query_patch_id']==r['patch_id']] for r in rows}
            scores={s['episode_id']:s for s in b.read_jsonl(episodes/'scoring/scoring_train.jsonl')}
            inventory,exclusions=b.canonical_inventory(prepared,rows,k1,scores)
            self.assertEqual(len(inventory),48*3)
            self.assertEqual({o['instance_id'] for o in inventory},{1})
            self.assertEqual(exclusions['void_background_or_mixed_candidate'],48)
            labels=b.load_verified(prepared,rows[0]['label_path'],rows[0]['label_sha256'])
            mask=labels['instances']==1
            raw=np.full((10,128,128),1000,dtype=np.int16);valid=np.ones((128,128),dtype=np.bool_)
            target=Path(tmp)/'outline.png';b.render(raw,valid,mask,target,[2,1,0])
            image=np.asarray(Image.open(target))
            self.assertTrue((image[24*3,45*3]==255).all())
            self.assertFalse((image[45*3,24*3]==255).all())  # Coordinates were not transposed.
            for option in inventory:
                score=scores[option['episode_id']]
                reference=b.private_reference(labels['semantic'],mask,score['target_class'],score['counter_class'])
                self.assertEqual(reference['reference_category'],option['reference_category'])

    def test_generated_public_bundle_hides_labels_quotas_and_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared,episodes,rows=fixture(root)
            result=b.build(prepared,episodes,root/'output',synthetic_fixture=True)
            public=Path(result['public']);private=Path(result['private'])
            mapping=json.loads((private/'reference_mapping.json').read_text())
            self.assertEqual(Counter(r['reference_category'] for r in mapping['cases']),b.QUOTAS)
            self.assertEqual(len({r['patch_id'] for r in mapping['cases']}),20)
            original={r['patch_id'] for r in b.select_queries(rows)}
            self.assertEqual({r['patch_id'] for r in mapping['cases']},original)
            for name in ('manifest.json','pilot_data.js','index.html','review_app.js','PROTOCOL.md'):
                value=(public/name).read_text()
                for forbidden in ['SOURCE_PRIVATE_','PRIVATE_PAIR_','PRIVATE_EPISODE_','target_class','counter_class','reference_category','source_label_reference_quotas','8/8/4']:
                    self.assertNotIn(forbidden,value)
            self.assertTrue((private/'PRIVATE_PROTOCOL.md').is_file())
            self.assertFalse((public/'PRIVATE_PROTOCOL.md').exists())

if __name__=='__main__':unittest.main()
