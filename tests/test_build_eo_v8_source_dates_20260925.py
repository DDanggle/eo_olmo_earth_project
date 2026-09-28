"""Synthetic preservation/provenance tests only; no real results/server/build."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import build_eo_v8_source_dates_20260925 as v8


PINS={key:'a'*64 for key in ['join_manifest_sha256','joined_rows_sha256','independent_review_sha256','upstream_package_manifest_sha256']}


def fixture_row(i=0,partition='test',event=3):
    sid=f'ks_{i:05d}'
    catalog={'id':sid,'dataset':'kurosiwo','aoi_id':str(event),'region_id':'13','split':partition,
        'event_date':'2022-01-29','flood_fraction_pct':25.,'permanent_water_fraction_pct':5.,
        'evidence':[{'image_url':'/previews/synthetic.png','acquisition_dates':{'pre_1':None,'pre_2':None,'post':None}}],
        'acquisition_dates':None,'synthetic_model_dates':['2022-01-17','2022-01-29']}
    fields={k:{'current':str(value),'upstream':value,'equal':True} for k,value in [('actid',event),('aoiid',13),('pcovered',100.),('pflood',25.),('pwater',5.)]}
    fields['flood_date']={'current':'2022-01-29 07:00:00','upstream':'2022-01-29 07:00:00','equal':True}
    fields['event_date']={'current':'2022-01-29','upstream':'2022-01-29','equal':True}
    joined={'id':sid,'source_id':i,'table_row_index':i,'event_id':event,'aoi_id':13,'partition':partition,
        'status':'matched_metadata','original_export_valid':True,'reasons':[],'candidate_count':1,
        'candidate_grid_ids':[f'synthetic-grid-{i}'],'grid_id':f'synthetic-grid-{i}','field_checks':fields,
        'temporal':{'source_date_precision':'day','historical_raster_to_published_slot_lineage':'not_verified',
            'published_slot_dates':{'pre_1':'2021-08-07','pre_2':'2021-08-19','post':'2022-02-03'},
            'old_synthetic_date_only':{'pre_1':'2022-01-05','pre_2':'2022-01-17','post':'2022-01-29'},
            'raw_sources':{k:{'source_date':d,'s1_ids':['synthetic-source-'+k]} for k,d in [('SL2','2021-08-07'),('SL1','2021-08-19'),('MS1','2022-02-03')]},
            'intervals_days':{'pre1_to_pre2_days':12,'pre2_to_post_days':168,'pre1_to_post_days':180,'event_to_post_days':5,
                'pre_1_source_minus_synthetic_days':-151,'pre_2_source_minus_synthetic_days':-151,'post_source_minus_synthetic_days':5},
            'strictly_increasing_published_dates':True}}
    return catalog,joined


def minimal_html():
    return '<html><script>function select(r){let html="";if(r.evidence?.length)html+=r.id;}</script></html>'


def make_build_fixture(root):
    base,t0=root/'base',root/'t0';base.mkdir();t0.mkdir()
    catalog={'schema':'eo-evidence-catalog-v0.2','records':[],'original_metadata':'preserve'};joined=[]
    for i in range(7000):
        partition,event=('train',1) if i<4000 else ('validation',2) if i<5000 else ('test',3)
        row,j=fixture_row(i,partition,event);catalog['records'].append(row);joined.append(j)
    catalog['records'].append({'id':'synthetic_sn7','dataset':'spacenet7','original':'unchanged'})
    v8.write(base/'catalog.json',catalog);v8.write(base/'meta.json',{'title':'SYNTHETIC META','acquisition_dates':None})
    v8.write(base/'reader_cases.json',{'cases':{'synthetic':{'dates':['2022-01-17','2022-01-29'],'predictions':'SYNTHETIC ONLY'}}})
    v8.write(base/'research_runs.json',{'schema_version':'eo_research_snapshot_v1','checked_at':'2026-09-25T06:30:11Z',
        'runs':[{'id':'E2','status':'SYNTHETIC inherited'},
                {'id':'E5-EB','status':'학습 진행 스냅샷 · 결과 없음','findings':['고정시각 '+v8.E5_STATUS_AS_OF]}]})
    v8.write(base/'research_sources.json',{'existing':'preserve','source_grounding_v7':{'e5_status_as_of':v8.E5_STATUS_AS_OF}})
    (base/'index.html').write_text(minimal_html());(base/'previews').mkdir()
    (base/'previews/synthetic.png').write_bytes(b'\x89PNG\r\n\x1a\nSYNTHETIC_PRESERVATION_BYTES')
    files={str(p.relative_to(base)):v8.sha(p) for p in base.rglob('*') if p.is_file()}
    v8.write(base/'v7_build_manifest.json',{'output_files_sha256':files})
    for name in v8.V7_EXTRA-{'v7_build_manifest.json'}:v8.write(base/name,{'historical_check':'synthetic'})
    (t0/'joined_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in joined))
    groups={}
    for key,n in [('all',7000),('split:train',4000),('split:validation',1000),('split:test',2000)]:
        groups[key]={'source_rows':n,'interval_distributions':{'pre2_to_post_days':{'median':168.,'max':168}}}
    summary={'valid':True,'global_errors':[],'summary':groups};v8.write(t0/'summary.json',summary)
    v8.write(t0/'manifest.json',{'schema':'kuro-t0-exact-source-date-join-v0','valid':True,'expected_rows':7000,
        'expected_splits':{'train':4000,'validation':1000,'test':2000},'upstream_package_manifest_sha256':v8.UPSTREAM_SHA,
        'files_sha256':{name:v8.sha(t0/name) for name in ['joined_rows.jsonl','summary.json']}})
    audit=root/'audit.json';v8.write(audit,{'schema':'t0-independent-result-review-v0','consistent':True,
        'n_rows':7000,'n_unique_grid_matches':7000,'all_strictly_chronological_count':7000,
        'verified_full_summary':groups,'source_hashes_verified':{str(t0/name):v8.sha(t0/name) for name in ['manifest.json','joined_rows.jsonl','summary.json']}})
    return base,t0,audit


class PureTests(unittest.TestCase):
    def test_only_new_field_added_and_other_dataset_images_dates_unchanged(self):
        row,joined=fixture_row();before={'schema':'eo-evidence-catalog-v0.2','records':[row,{'id':'sn7','dataset':'spacenet7','date':'unchanged'}]}
        source=copy.deepcopy(before);jcopy=copy.deepcopy(joined)
        result=v8.attach_dates(before,[joined],PINS,expected_count=1)
        p=result['records'][0].pop('source_date_provenance')
        self.assertEqual(result,source);self.assertEqual(before,source);self.assertEqual(joined,jcopy)
        self.assertFalse(p['actual_acquisition_verified']);self.assertTrue(all(d is None for d in p['sensor_utc_acquisition_timestamps'].values()))
        self.assertEqual(p['historical_raster_to_published_slot_lineage'],'not_verified')
        self.assertEqual(p['published_slot_dates']['pre_2'],'2021-08-19')
        self.assertEqual(p['model_assumed_slot_dates']['pre_2'],'2022-01-17')

    def test_false_verified_lineage_invalid_dates_and_field_claim_rejected(self):
        mutations=[lambda j:j['temporal'].update(historical_raster_to_published_slot_lineage='verified'),
            lambda j:j['temporal']['published_slot_dates'].update(post='2022-02-03T00:00:00Z'),
            lambda j:j['field_checks']['pflood'].update(equal=False),
            lambda j:j['temporal']['intervals_days'].update(pre2_to_post_days=12),
            lambda j:j.update(status='no_exact_match')]
        for mutate in mutations:
            row,j=fixture_row();mutate(j)
            with self.assertRaises(ValueError):v8.provenance_for(row,j,PINS)

    def test_id_region_event_fraction_and_synthetic_date_mismatch_rejected(self):
        for key,value in [('id','ks_99999'),('aoi_id','4'),('region_id','14'),('event_date','2022-01-30'),('flood_fraction_pct',26.)]:
            row,j=fixture_row();row[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):v8.provenance_for(row,j,PINS)
        row,j=fixture_row();j['temporal']['old_synthetic_date_only']['pre_1']='2022-01-06'
        with self.assertRaisesRegex(ValueError,'synthetic'):v8.provenance_for(row,j,PINS)

    def test_missing_duplicate_or_existing_provenance_rejected(self):
        row,j=fixture_row();catalog={'schema':'eo-evidence-catalog-v0.2','records':[row]}
        for records in [[],[j,j]]:
            with self.assertRaises(ValueError):v8.attach_dates(catalog,records,PINS,expected_count=1)
        row['source_date_provenance']={'old':'must not overwrite'}
        with self.assertRaisesRegex(ValueError,'already exists'):v8.attach_dates(catalog,[j],PINS,expected_count=1)

    def test_ui_insertion_preserves_original_script_and_rejects_second_patch(self):
        original=minimal_html();result=v8.patch_html(original)
        self.assertIn('html+=sourceDateDetail(r);',result)
        self.assertIn('실제 UTC 취득시각',result);self.assertIn('검색 기간은 이 사건일 기준',result)
        self.assertEqual(result.replace(v8.DATE_UI+'\n','').replace('html+=sourceDateDetail(r);\n',''),original)
        with self.assertRaises(ValueError):v8.patch_html(result)


class BuildTests(unittest.TestCase):
    def test_complete_synthetic_build_preserves_every_unmodified_file_and_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();base,t0,audit=make_build_fixture(root)
            hashes={str(p.relative_to(base)):v8.sha(p) for p in base.rglob('*') if p.is_file()}
            original_research=v8.read(base/'research_runs.json');original_catalog=v8.read(base/'catalog.json')
            with mock.patch.multiple(v8,V7_SHA=v8.sha(base/'v7_build_manifest.json'),T0_SHA=v8.sha(t0/'manifest.json'),AUDIT_SHA=v8.sha(audit)):
                result=v8.build(base,t0,audit,root/'out')
            self.assertEqual(result['n_source_dates_added'],7000)
            for name,digest in hashes.items():
                self.assertEqual(v8.sha(base/name),digest)
                if name not in v8.CHANGED:self.assertEqual(v8.sha(root/'out'/name),digest)
            research=v8.read(root/'out/research_runs.json')
            self.assertEqual(research['runs'][:-1],original_research['runs']);self.assertEqual(research['runs'][-1]['id'],'T0-DATE')
            catalog=v8.read(root/'out/catalog.json')
            for row in catalog['records']:row.pop('source_date_provenance',None)
            self.assertEqual(catalog,original_catalog)
            self.assertFalse((root/'out/v8_failure.json').exists())
            with self.assertRaisesRegex(ValueError,'no overwrite'):v8.build(base,t0,audit,root/'out')

    def test_hash_tamper_rejected_before_output_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();base,t0,audit=make_build_fixture(root)
            pins={'V7_SHA':v8.sha(base/'v7_build_manifest.json'),'T0_SHA':v8.sha(t0/'manifest.json'),'AUDIT_SHA':v8.sha(audit)}
            for p in [base/'catalog.json',t0/'joined_rows.jsonl',audit]:
                before=p.read_bytes();p.write_bytes(before+b' ')
                with mock.patch.multiple(v8,**pins),self.assertRaises(ValueError):v8.build(base,t0,audit,root/'out')
                self.assertFalse((root/'out').exists());p.write_bytes(before)


if __name__=='__main__':unittest.main()
