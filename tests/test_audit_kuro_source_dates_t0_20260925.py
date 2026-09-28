"""Independent synthetic-only tests for T0 exact metadata joins.

No original EO metadata, images, E5 artifacts, or server are read. The production
join module is imported; every data row/package in these tests is constructed.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import audit_kuro_source_dates_t0_20260925 as audit


def row_and_record(i, partition='test', event=3, x=None):
    x = 10000 + 2240*i if x is None else x
    y = 20000
    corners = [[x,y],[x+2240,y],[x+2240,y-2240],[x,y-2240],[x,y]]
    geom = 'POLYGON (('+', '.join(f'{a} {b}' for a,b in corners)+'))'
    key = f'{i:032x}'
    grid = f'{key[:8]}-{key[8:12]}-{key[12:16]}-{key[16:20]}-{key[20:]}'
    source = {'tortilla:id':str(i),'tortilla:data_split':partition,'actid':str(event),'aoiid':'13',
              'pcovered':'100.0','pwater':'5.0','pflood':'25.0','flood_date':'2022-01-29 07:00:00'}
    row = {'id':f'ks_{i:05d}','source_id':i,'table_row_index':i,'partition':partition,'event_id':event,
           'aoi_id':13,'pwater':5.,'pflood':25.,'event_date':'2022-01-29','valid':True,'mismatches':[],
           'crs':'EPSG:3857','raster_shape':[224,224],'gdal_geotransform':[x,10,0,y,0,-10],
           'corners_epsg3857':corners,'bbox_epsg3857':[x,y-2240,x+2240,y],
           'source_row':source,'source_numeric_vectors':{
               'stac:geotransform':[x,10,0,y,0,-10],'stac:raster_shape':[224,224]}}
    info = {'grid_id':grid,'actid':event,'aoiid':13,'geom':geom,'pcovered':100.,'pwater':5.,'pflood':25.,
            'flood_date':'2022-01-29 07:00:00','sources':{
                'SL2':{'source_date':'2022-01-06','s1_ids':['SYNTHETIC-SL2']},
                'SL1':{'source_date':'2022-01-18','s1_ids':['SYNTHETIC-SL1']},
                'MS1':{'source_date':'2022-02-01','s1_ids':['SYNTHETIC-MS1']}}}
    return row,key,{'path':f'{event}/13/{key}','info':info}


def small(n=2):
    values=[row_and_record(i) for i in range(n)]
    return [v[0] for v in values],{v[1]:v[2] for v in values}


def join(rows, upstream, n=None):
    n=len(rows) if n is None else n
    return audit.audit_rows(rows,upstream,expected_count=n,expected_splits={'test':n},published_splits={3:'test'})


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def package(root):
    """Minimal hash-valid synthetic package; normal load stops at empty row input.

    Its parser is a tiny safe stub that returns fabricated support counts. This
    exercises load_inputs' file/manifest/raw-row guards without opening data.
    """
    upstream=root/'upstream';export=root/'export'
    upstream.mkdir();export.mkdir()
    write(upstream/'published_geobench_slot_contract.json',{'published_mapping':{
        k:{'upstream_key':v} for k,v in [('pre_event_1','SL2'),('pre_event_2','SL1'),('post_event','MS1')]}})
    (upstream/'geobench2_kuro_generate_20260925.py').write_text('train_acts=[1]\nval_acts=[2]\ntest_acts=[3]\n')
    parser = '''def data_only(path):
    n=67490 if path.name=="KuroV2_grid_dict_test_0_100.gz" else 31707
    result={str(i):{"synthetic":i} for i in range(n)}
    first="e2f98639b3ca5c438d3b79c11b55bc5c"
    result.pop("0");result[first]={"synthetic_fixed":True}
    if n==67490:
        result.pop("1");result["d762ee26ef585723b9f7d9d18f419bd8"]={"synthetic_fixed":False}
    return result
'''
    (upstream/'inspect_grid_metadata.py').write_text(parser)
    (upstream/audit.PRIMARY).write_bytes(b'SYNTHETIC_PRIMARY_NOT_REAL_GZIP')
    (upstream/audit.SECONDARY).write_bytes(b'SYNTHETIC_SECONDARY_NOT_REAL_GZIP')
    write(upstream/'MANIFEST.json',{'files':{p.name:{'sha256':audit.sha(p)} for p in upstream.iterdir()}})
    rows,_=small(1)
    raw=[{k:r[k] for k in ('table_row_index','source_row','source_numeric_vectors')} for r in rows]
    for name,values in [('metadata.jsonl',rows),('raw_rows.jsonl',raw)]:
        (export/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
    for name in ('selected_columns.jsonl','cache_meta.jsonl'): (export/name).write_text('{}\n')
    (export/'source.py').write_text('# SYNTHETIC export metadata fixture; not production source\n')
    (export/'footer.parquet').write_bytes(b'SYNTHETIC_FOOTER')
    (export/'header18.bin').write_bytes(b'SYNTHETIC_HEADER!!!')
    pins={p.name:audit.sha(p) for p in export.iterdir()}
    manifest={'schema':'kuro-t0-top-metadata-manifest-v0','valid':True,'files_sha256':pins,
        'metadata_sha256':pins['metadata.jsonl'],'raw_rows_sha256':pins['raw_rows.jsonl'],
        'selected_columns_sha256':pins['selected_columns.jsonl'],'cache_meta_sha256':pins['cache_meta.jsonl'],
        'code_sha256':pins['source.py'],'top_metadata':{'sha256':pins['footer.parquet'],'header_sha256':pins['header18.bin']},
        'source_stat_before':{'synthetic':1},'source_stat_after':{'synthetic':1}}
    write(export/'manifest.json',manifest)
    digest=audit.sha(export/'manifest.json')
    write(export/'status.json',{'status':'complete','valid':True,'manifest_sha256':digest})
    return export,upstream,digest,audit.sha(upstream/'MANIFEST.json')


class ExactJoinTests(unittest.TestCase):
    def test_complete_7000_population_and_all_three_splits(self):
        rows=[];upstream={}
        for i in range(7000):
            partition,event=('train',1) if i<4000 else ('validation',2) if i<5000 else ('test',3)
            row,key,record=row_and_record(i,partition,event)
            rows.append(row);upstream[key]=record
        result=audit.audit_rows(rows,upstream,published_splits={1:'train',2:'validation',3:'test'})
        self.assertTrue(result['valid']);self.assertEqual(len(result['rows']),7000)
        self.assertEqual(result['summary']['all']['status_counts'],{'matched_metadata':7000})
        for key,n in audit.EXPECTED_SPLITS.items():self.assertEqual(result['summary']['split:'+key]['source_rows'],n)
        missing=audit.audit_rows(rows[:-1],upstream,published_splits={1:'train',2:'validation',3:'test'})
        self.assertFalse(missing['valid']);self.assertEqual(len(missing['rows']),6999)
        self.assertEqual(set(missing['global_errors']),{'row_count','source_id_coverage','split_counts'})

    def test_exact_geometry_not_nearest_or_fraction_fallback(self):
        rows,upstream=small(1)
        self.assertTrue(join(rows,upstream)['valid'])
        key=next(iter(upstream));info=upstream[key]['info']
        original=copy.deepcopy(info)
        # One-centimeter displacement has identical event/AOI and label ratios.
        info['geom']=row_and_record(0,x=10000.01)[2]['info']['geom']
        result=join(rows,upstream)
        self.assertEqual(result['rows'][0]['status'],'no_exact_match')
        self.assertNotIn('temporal',result['rows'][0])
        info.update(original);info['actid']=4;upstream[key]['path']=f'4/13/{key}'
        self.assertEqual(join(rows,upstream)['rows'][0]['status'],'no_exact_match')

    def test_corner_order_closed_or_open_ring_is_exact_geometry(self):
        rows,upstream=small(1)
        rows[0]['corners_epsg3857']=list(reversed(rows[0]['corners_epsg3857'][:-1]))
        info=next(iter(upstream.values()))['info']
        corners=rows[0]['corners_epsg3857'];ring=corners+[corners[0]]
        info['geom']='POLYGON (('+', '.join(f'{x} {y}' for x,y in ring)+'))'
        self.assertTrue(join(rows,upstream)['valid'])
        rows[0]['bbox_epsg3857'][0]+=.01
        self.assertEqual(join(rows,upstream)['rows'][0]['status'],'invalid_source')

    def test_ambiguous_upstream_grid_has_no_assigned_date(self):
        rows,upstream=small(1)
        _,key,other=row_and_record(1,x=10000)
        upstream[key]=other
        result=join(rows,upstream)
        self.assertEqual(result['rows'][0]['status'],'ambiguous_exact_match')
        self.assertEqual(result['rows'][0]['candidate_count'],2)
        self.assertNotIn('temporal',result['rows'][0]);self.assertFalse(result['valid'])

    def test_multiple_current_ids_same_grid_preserved_and_excluded_from_dates_summary(self):
        rows,upstream=small(1)
        other=copy.deepcopy(rows[0]);other.update(id='ks_00001',source_id=1,table_row_index=1)
        other['source_row']['tortilla:id']='1';rows.append(other)
        result=join(rows,upstream)
        self.assertFalse(result['valid']);self.assertEqual(len(result['rows']),2)
        self.assertEqual({r['status'] for r in result['rows']},{'duplicate_current_grid'})
        self.assertEqual(result['summary']['all']['date_triplets'],{})
        self.assertEqual(len(result['duplicate_current_grid_groups']),1)

    def test_missing_or_invalid_dates_preserved_without_synthetic_replacement(self):
        for change in ('missing_slot','bad_precision','missing_ids'):
            rows,upstream=small(1);sources=next(iter(upstream.values()))['info']['sources']
            if change=='missing_slot':sources.pop('SL2')
            elif change=='bad_precision':sources['SL2']['source_date']='2022-01-06T00:00:00Z'
            else:sources['SL2']['s1_ids']=[]
            result=join(rows,upstream)
            with self.subTest(change=change):
                self.assertFalse(result['valid']);self.assertEqual(len(result['rows']),1)
                self.assertEqual(result['rows'][0]['status'],'invalid_upstream_dates')
                self.assertNotIn('temporal',result['rows'][0])

    def test_field_mismatch_invalid_export_and_published_split_preserved(self):
        rows,upstream=small(3)
        upstream[f'{0:032x}']['info']['pcovered']=99.
        rows[1]['valid']=False;rows[1]['mismatches']=['synthetic_export_failure']
        rows[2]['partition']='train';rows[2]['source_row']['tortilla:data_split']='train'
        result=join(rows,upstream)
        self.assertEqual([r['status'] for r in result['rows']],['metadata_field_mismatch','invalid_source','invalid_source'])
        self.assertTrue(all('temporal' not in r for r in result['rows']))
        self.assertEqual(result['rows'][1]['original_export_mismatches'],['synthetic_export_failure'])
        self.assertIn('published event split mismatch',result['rows'][2]['reasons'])

    def test_candidate_dates_are_not_verified_acquisitions_and_old_dates_remain_separate(self):
        rows,upstream=small(1);result=join(rows,upstream)
        temporal=result['rows'][0]['temporal']
        self.assertEqual(temporal['published_slot_dates'],{'pre_1':'2022-01-06','pre_2':'2022-01-18','post':'2022-02-01'})
        self.assertEqual(temporal['old_synthetic_date_only'],{'pre_1':'2022-01-05','pre_2':'2022-01-17','post':'2022-01-29'})
        self.assertEqual(temporal['historical_raster_to_published_slot_lineage'],'not_verified')
        self.assertEqual(temporal['source_date_precision'],'day')
        self.assertEqual(temporal['intervals_days'],{'pre1_to_pre2_days':12,'pre2_to_post_days':14,'pre1_to_post_days':26,'event_to_post_days':3,
            'pre_1_source_minus_synthetic_days':1,'pre_2_source_minus_synthetic_days':1,'post_source_minus_synthetic_days':3})
        self.assertNotIn('actual_acquisition_dates',temporal)

    def test_nonchronological_published_dates_are_kept_as_diagnostic(self):
        rows,upstream=small(1)
        next(iter(upstream.values()))['info']['sources']['MS1']['source_date']='2022-01-01'
        result=join(rows,upstream)
        self.assertEqual(result['rows'][0]['status'],'matched_metadata')
        self.assertFalse(result['rows'][0]['temporal']['strictly_increasing_published_dates'])
        self.assertEqual(result['summary']['all']['strict_chronological_count'],0)
        self.assertEqual(result['summary']['all']['interval_distributions']['pre2_to_post_days']['negative_count'],1)

    def test_duplicate_current_ids_and_malformed_rectangle_rejected(self):
        rows,upstream=small(2);rows[1]['id']=rows[0]['id']
        result=join(rows,upstream)
        self.assertIn('duplicate_current_id',result['global_errors']);self.assertFalse(result['valid'])
        for wkt in ('POLYGON ((0 0, 1 1, 0 1, 1 0, 0 0))','POLYGON ((0 0, 1 0, 1 1, 0 1, 2 2))',
                    'POLYGON ((nan 0, 1 0, 1 1, nan 1, nan 0))'):
            with self.subTest(wkt=wkt),self.assertRaises(ValueError):audit.rectangle(wkt)

    def test_invalid_normalization_keeps_table_row_and_raw_identity(self):
        rows,upstream=small(1)
        for key in ('id','source_id','partition','event_id','aoi_id'):rows[0].pop(key)
        rows[0]['valid']=False;rows[0]['mismatches']=['synthetic_normalization_error']
        result=join(rows,upstream)
        entry=result['rows'][0]
        self.assertFalse(result['valid']);self.assertEqual(entry['status'],'invalid_source')
        self.assertEqual(entry['table_row_index'],0)
        self.assertEqual(entry['source_identity_raw'],{'tortilla:id':'0','tortilla:data_split':'test','actid':'3','aoiid':'13'})
        self.assertNotIn('temporal',entry)


class InputPinTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.export,self.upstream,self.expected,self.up_pin=package(self.root)
        self.patch=mock.patch.object(audit,'PIN_MANIFEST_SHA',self.up_pin);self.patch.start()
    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()
    def load(self):return audit.load_inputs(self.export,self.upstream,self.expected)

    def test_synthetic_hash_valid_package_loads_without_real_data(self):
        rows,primary,splits,pins=self.load()
        self.assertEqual(len(rows),1);self.assertEqual(len(primary),67490)
        self.assertEqual(splits,{1:'train',2:'validation',3:'test'})
        self.assertTrue(pins['no_nearest_neighbor_or_other_metadata_fallback'])
        self.assertEqual(pins['fixed_two_case_crosscheck']['ks_05265']['secondary_present'],False)

    def test_manifest_and_payload_tampering_rejected(self):
        for path in [self.upstream/'MANIFEST.json',self.export/'manifest.json',self.export/'metadata.jsonl']:
            original=path.read_bytes();path.write_bytes(original+b' ')
            with self.subTest(path=path.name),self.assertRaises(ValueError):self.load()
            path.write_bytes(original)

    def test_status_manifest_mismatch_and_unfinished_export_rejected(self):
        write(self.export/'status.json',{'status':'complete','manifest_sha256':'0'*64})
        with self.assertRaisesRegex(ValueError,'status/manifest'):self.load()
        write(self.export/'status.json',{'status':'running','manifest_sha256':self.expected})
        with self.assertRaisesRegex(ValueError,'unfinished'):self.load()

    def test_safe_relative_and_pinned_payload_reject_escape(self):
        for name in ('/tmp/other','../other',''):
            with self.subTest(name=name),self.assertRaises(ValueError):audit.safe_relative(self.export,name)
        outside=self.root/'outside';outside.write_bytes(b'SYNTHETIC')
        (self.export/'link').symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'escapes'):audit.safe_relative(self.export,'link')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):audit.verify_files(self.export,{'metadata.jsonl':'0'*64})

    def test_invalid_or_false_valid_export_status_cannot_be_promoted(self):
        original_join=audit.audit_rows
        def one_row_join(rows,upstream,**kwargs):
            return original_join(rows,upstream,expected_count=1,expected_splits={'test':1},**kwargs)
        for index,(status,valid,expected) in enumerate([('complete',True,True),('invalid',True,False),('complete',False,False)]):
            write(self.export/'status.json',{'status':status,'valid':valid,'manifest_sha256':self.expected})
            rows,_,splits,pins=self.load()
            _,upstream=small(1)
            with mock.patch.object(audit,'load_inputs',return_value=(rows,upstream,splits,pins)),mock.patch.object(audit,'audit_rows',side_effect=one_row_join):
                result=audit.run(self.export,self.upstream,self.root/f'out{index}',self.expected)
            with self.subTest(status=status,valid=valid):
                self.assertIs(result['valid'],expected)
                self.assertEqual(result['matched'],1)
                if not expected:self.assertIn('source_export_not_complete',result['global_errors'])

    def test_status_changed_during_join_rejected_and_existing_output_untouched(self):
        rows,_,splits,pins=self.load();_,upstream=small(1)
        original_join=audit.audit_rows
        def change_status(rows,upstream,**kwargs):
            result=original_join(rows,upstream,expected_count=1,expected_splits={'test':1},**kwargs)
            (self.export/'status.json').write_text('{}\n')
            return result
        out=self.root/'changed_during_join'
        with mock.patch.object(audit,'load_inputs',return_value=(rows,upstream,splits,pins)),mock.patch.object(audit,'audit_rows',side_effect=change_status):
            with self.assertRaisesRegex(ValueError,'status changed'):audit.run(self.export,self.upstream,out,self.expected)
        self.assertFalse(out.exists())
        out.mkdir();(out/'sentinel').write_text('do not replace')
        with self.assertRaises(FileExistsError):audit.run(self.export,self.upstream,out,self.expected)
        self.assertEqual((out/'sentinel').read_text(),'do not replace')


if __name__=='__main__':unittest.main()
