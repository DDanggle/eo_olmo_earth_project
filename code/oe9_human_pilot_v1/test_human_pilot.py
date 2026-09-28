from __future__ import annotations
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

import build_human_pilot as build
import score_human_pilot as scoring


def minimal_manifest():
    cases = [{'case_id':f'P{i+1:03d}','candidate_available':True,
              'frames':{'query':[{'frame_id':f'P{i+1:03d}-q-01','date':'20190101'}]}}
             for i in range(20)]
    manifest={'package_id':'synthetic_unit_package','schema_version':build.VERSION,'cases':cases}
    refs={'package_id':manifest['package_id'],'cases':[{'case_id':c['case_id'],
        'reference_category':'target' if i<5 else 'counterexample' if i<10 else 'neither'} for i,c in enumerate(cases)]}
    return manifest, refs


def response(manifest, refs, role, constant=None):
    base=datetime(2026,9,27,0,0,tzinfo=timezone.utc)
    records=[]
    for i,c in enumerate(manifest['cases']):
        start=(base+timedelta(seconds=120*i)).isoformat()
        end=(base+timedelta(seconds=120*i+100)).isoformat()
        records.append({'case_id':c['case_id'],'status':'complete',
            'decision':constant or refs['cases'][i]['reference_category'],
            'evidence_frame_ids':[c['frames']['query'][0]['frame_id']],
            'reason':'SYNTHETIC SOFTWARE TEST RESPONSE, NOT HUMAN', 'issues':['shape'],
            'confidence':'medium','active_seconds':100,'first_started_at':start,'completed_at':end,
            'intervals':[{'started_at':start,'ended_at':end,'elapsed_seconds':100,'active_seconds':100,'stop_reason':'completed'}]})
    return {'schema_version':build.VERSION,'package_id':manifest['package_id'],
            'annotator_id':'synthetic_'+role,'assignment':role,'synthetic_fixture':True,
            'exported_at':(base+timedelta(hours=1)).isoformat(),'records':records}


def create_synthetic_source(root):
    prepared=root/'prepared';episodes=root/'episodes'
    (prepared/'inputs').mkdir(parents=True);(prepared/'labels').mkdir()
    (episodes/'support_masks').mkdir(parents=True);(episodes/'scoring').mkdir()
    raw=np.zeros((8,10,128,128),dtype=np.int16)
    yy,xx=np.indices((128,128))
    for t in range(8):
        for b in range(10):raw[t,b]=((xx*8+yy*2+t*80+b*60)%3000).astype(np.int16)
    valid=np.ones((8,128,128),dtype=np.bool_)
    np.savez_compressed(prepared/'inputs/shared.npz',raw_selected_s2=raw,observation_valid=valid)
    instances=np.zeros((128,128),dtype=np.int64);instances[30:70,40:80]=1
    semantic=np.zeros_like(instances);semantic[instances==1]=1
    np.savez_compressed(prepared/'labels/shared.npz',semantic=semantic,instances=instances)
    np.savez_compressed(episodes/'support_masks/shared.npz',mask=instances==1)
    ih=build.sha(prepared/'inputs/shared.npz');lh=build.sha(prepared/'labels/shared.npz');mh=build.sha(episodes/'support_masks/shared.npz')
    rows=[];eps=[];targets=[]
    dates=[20190101+i for i in range(8)]
    for i in range(48):
        pid=f'SOURCE_PRIVATE_{i:02d}'
        rows.append({'patch_id':pid,'training_partition':'train_pool','npz_path':'inputs/shared.npz',
                     'npz_sha256':ih,'label_path':'labels/shared.npz','label_sha256':lh,'selected_dates':dates})
        pair={}
        for kind,delta in [('positive',1),('counterexample',2)]:
            spid=f'SOURCE_PRIVATE_{(i+delta)%48:02d}'
            pair[kind]={'patch_id':spid,'input_npz':'inputs/shared.npz','input_sha256':ih,
                'mask_npz':'support_masks/shared.npz','mask_sha256':mh,'object_key':spid+':1','dates_yyyymmdd':dates}
        eps.append({'query_patch_id':pid,'k_pairs':1,'pair_id':'PRIVATE_PAIR','episode_id':pid+':k1','support_pairs':[pair]})
        targets.append({'episode_id':pid+':k1','target_class':1,'counter_class':2})
    for path,data in [(prepared/'manifest.jsonl',rows),(episodes/'episodes_train.jsonl',eps),(episodes/'scoring/scoring_train.jsonl',targets)]:
        path.write_text(''.join(json.dumps(r)+'\n' for r in data))
    return prepared,episodes


class PilotTests(unittest.TestCase):
    def test_queries_are_selected_without_gold_and_ignore_other_partitions(self):
        rows=[{'patch_id':str(i),'training_partition':'train_pool','class_pixel_counts':[i]} for i in range(48)]
        rows.extend([{'patch_id':'bank','training_partition':'source_bank'}, {'patch_id':'dev','training_partition':'dev_query'}])
        before=[r['patch_id'] for r in build.select_queries(rows)]
        for r in rows:r['class_pixel_counts']=['CHANGED']
        after=[r['patch_id'] for r in build.select_queries(list(reversed(rows)))]
        self.assertEqual(before,after);self.assertEqual(len(before),20)
        self.assertFalse({'bank','dev'}&set(before))

    def test_candidate_geometry_threshold_and_stability(self):
        inst=np.zeros((128,128),dtype=np.int64);inst[0:8,0:8]=1;inst[20:27,20:29]=2
        self.assertEqual(build.select_candidate(inst,'test'),1)
        self.assertIsNone(build.select_candidate(np.zeros_like(inst),'test'))

    def test_missing_reviewers_remain_zero(self):
        manifest,refs=minimal_manifest();report=scoring.score(manifest,refs)
        self.assertEqual(report['actual_human_complete_records'],0)
        self.assertEqual(report['status'],'awaiting_two_complete_independent_reviews')
        self.assertIsNone(report['overall_agreement']['exact_agreement'])

    def test_insufficient_informative_coverage_is_design_issue_before_humans(self):
        manifest,refs=minimal_manifest()
        for r in refs['cases']:r['reference_category']='neither'
        report=scoring.score(manifest,refs)
        self.assertEqual(report['status'],'pilot_design_insufficient_informative_cases')
        self.assertEqual(report['human_collection_status'],'not_collected')

    def test_synthetic_exports_cannot_pass_human_gate(self):
        manifest,refs=minimal_manifest();a=response(manifest,refs,'A');b=response(manifest,refs,'B')
        with self.assertRaisesRegex(ValueError,'Synthetic fixtures'):scoring.score(manifest,refs,a,b)
        report=scoring.score(manifest,refs,a,b,True)
        self.assertEqual(report['overall_agreement']['exact_agreement'],1)
        self.assertEqual(report['actual_human_complete_records'],0)
        self.assertFalse(report['operational_gate']['passed'])

    def test_constant_neither_agreement_cannot_pass_informative_gate(self):
        manifest,refs=minimal_manifest()
        report=scoring.score(manifest,refs,response(manifest,refs,'A','neither'),response(manifest,refs,'B','neither'),True)
        self.assertEqual(report['overall_agreement']['exact_agreement'],1)
        self.assertFalse(report['operational_gate']['informative_decisions_sufficient'])
        self.assertFalse(report['operational_gate']['nondegenerate_decision_counts'])
        self.assertIsNone(report['overall_agreement']['cohen_kappa'])

    def test_same_reviewer_and_timing_fabrication_are_rejected(self):
        manifest,refs=minimal_manifest();a=response(manifest,refs,'A');b=response(manifest,refs,'B')
        b['annotator_id']=a['annotator_id']
        with self.assertRaisesRegex(ValueError,'distinct reviewers'):scoring.score(manifest,refs,a,b,True)
        a['records'][0]['active_seconds']=999
        with self.assertRaisesRegex(ValueError,'total'):scoring.validate(a,manifest,'A',True)

    def test_answer_needs_query_evidence_and_unobservable_reason(self):
        manifest,refs=minimal_manifest();a=response(manifest,refs,'A')
        a['records'][0]['evidence_frame_ids']=[]
        with self.assertRaisesRegex(ValueError,'evidence'):scoring.validate(a,manifest,'A',True)
        a=response(manifest,refs,'A');a['records'][0]['decision']='unobservable'
        with self.assertRaisesRegex(ValueError,'limitation'):scoring.validate(a,manifest,'A',True)

    def test_untouched_cases_are_not_filled_and_partial_counts_are_explicit(self):
        manifest,refs=minimal_manifest();a=response(manifest,refs,'A')
        r=a['records'][0];r.update(status='not_started',decision=None,evidence_frame_ids=[],reason='',issues=[],confidence=None,intervals=[],active_seconds=0,first_started_at=None,completed_at=None)
        validated=scoring.validate(a,manifest,'A',True)
        self.assertEqual(sum(r['status']=='complete' for r in validated.values()),19)
        r['decision']='target'
        with self.assertRaisesRegex(ValueError,'Untouched'):scoring.validate(a,manifest,'A',True)

    def test_cpu_generator_keeps_private_answers_out_of_distributed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prepared,episodes=create_synthetic_source(root)
            result=build.build(prepared,episodes,root/'output',synthetic_fixture=True)
            public=Path(result['public']);manifest=json.loads((public/'manifest.json').read_text())
            self.assertTrue(manifest['synthetic_fixture_package'])
            self.assertEqual(len(manifest['cases']),20)
            self.assertEqual(len(list((public/'assets').glob('*.png'))),960)
            for name in ('manifest.json','pilot_data.js','index.html','review_app.js'):
                text=(public/name).read_text()
                for forbidden in ('SOURCE_PRIVATE_', 'PRIVATE_PAIR', 'target_class', 'counter_class', 'dominant_class', 'reference_category'):
                    self.assertNotIn(forbidden,text)
            self.assertTrue((Path(result['private'])/'reference_mapping.json').is_file())
            self.assertEqual(set(manifest['orders']['A']),set(manifest['orders']['B']))
            self.assertNotEqual(manifest['orders']['A'],manifest['orders']['B'])

if __name__=='__main__':unittest.main()
