"""Independent synthetic coverage and provenance checks for saved E5 case export."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
for directory in (HERE, HERE.parent):
    if (directory / 'build_eo_v11_e5_cases_20260925.py').is_file():
        sys.path.insert(0, str(directory))
        break


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows))


def synthetic_population():
    items = []
    def append(partition, phen, tile, event, kind):
        item = {'id': tile + '_q1_' + ('hard' if kind == 'hard_neg' else kind),
                'tile': tile, 'phen': phen, 'cluster': event, 'partition': partition,
                'kind': kind, 'answer': 'yes' if kind == 'pos' else 'no',
                'dates': ['2020-01-01', '2020-01-13'] if kind != 'neg' else ['2019-12-20', '2020-01-01']}
        if phen == 'flood':
            item.update(event=event, slots=['pre_1', 'pre_2'] if kind == 'neg' else ['pre_2', 'post'])
        items.append(item)
    for phen, n, events in [('flood',1066,27),('landslide',518,7)]:
        for index in range(n):
            for kind in ('pos','neg'):
                append('train',phen,f'train_{phen}_{index}',f'train_{index % events}',kind)
            if phen == 'flood':
                append('train',phen,f'train_hard_{index}',f'train_{index % events}','hard_neg')
    count = 0
    for event, n in enumerate((56,56,56,56,56,55,55,55,6,6)):
        for _ in range(n):
            tile = f'ks_{count:05d}'; count += 1
            for kind in ('pos','neg'):
                append('test','flood',tile,str(event),kind)
    hard_count = 1000
    for event, n in enumerate((57,57,57,57,57,57,57,58)):
        for _ in range(n):
            append('test','flood',f'ks_{hard_count:05d}',str(event),'hard_neg'); hard_count += 1
    for event, n in enumerate((186,6)):
        for index in range(n):
            for kind in ('pos','neg'):
                append('test','landslide',f'land_{event}_{index}',str(event),kind)
    test = [x for x in items if x['partition']=='test']
    target = [x for x in test if x['phen']=='flood' and x['kind'] in ('pos','hard_neg')]
    primary = [x['id'] for x in target if int(x['cluster']) < 8]
    unsupported = [x['id'] for x in target if int(x['cluster']) >= 8]
    eligible = set(primary + unsupported[:5])
    sets = {'all_test':[x['id']for x in test], 'primary_same_prompt':primary,
            'paired_flood':[x['id']for x in test if x['phen']=='flood' and x['kind'] in ('pos','neg')],
            'hard_negative_flood':[x['id']for x in target if x['kind']=='hard_neg'],
            'landslide':[x['id']for x in test if x['phen']=='landslide'],
            'e3_subset':[x['id']for x in test[:209]]}
    rows = []
    conditions = [('full','native'),('pair','native'),('later','native'),('delta','native'),('full','full_no_delta')]
    indices = {x['id']: i for i,x in enumerate(items)}
    for seed in (1,2,3):
        for arm, condition in conditions:
            for item in test:
                parsed = ('yes' if seed == 1 else 'no') if arm in ('pair','later') else item['answer']
                if condition == 'full_no_delta': parsed='no'
                rows.append({'seed':seed,'model_arm':arm,'eval_arm':condition,
                             **{k:item[k]for k in ('id','tile','cluster','phen','kind')},
                             'source_gold':item['answer'],'transformed_gold':None,'pair_index':indices[item['id']],
                             'answer_raw':parsed.upper()+'.','parsed':parsed})
    assert (len(items),len(test),len(target),len(primary),len(eligible),len(rows)) == (5989,1755,914,902,907,26325)
    return items, rows, sets, target, eligible

import build_eo_v11_e5_cases_20260925 as builder


class Fixture:
    def __init__(self, root):
        self.root = root.resolve(); self.base = self.root/'v10'; self.e5 = self.root/'e5_replica'
        self.audit_path = self.root/'audit.json'; self.out = self.root/'v11'
        self.base.mkdir(); self.e5.mkdir()
        self.items, self.rows, self.sets, self.target, self.eligible = synthetic_population()
        primary = set(self.sets['primary_same_prompt'])
        self.quality = [{'id':x['id'], **{k:x[k]for k in ('tile','dates','slots','kind','event')},
                         'source_answer':x['answer'],'total_px':100,'labelled_px':100 if x['id']in self.eligible else 80,
                         'valid_frac':1. if x['id']in self.eligible else .8,
                         'eligible_symmetric_quality':x['id']in self.eligible}for x in self.target]
        self.selected = {'quality_symmetric':sorted(self.eligible),'all_original_targets':sorted(x['id']for x in self.target)}
        # One genuine stored unknown, while missing rows must remain fatal.
        for row in self.rows:
            if row['seed']==3 and row['model_arm']=='delta' and row['id']==self.target[0]['id']:
                row.update(parsed=None,answer_raw='uncertain')
        write_rows(self.e5/'items.jsonl',self.items);write_rows(self.e5/'predictions.jsonl',self.rows)
        self.prompts=[{'id':x['id'],'pair_index':i,'source_gold':x['answer'],'n_eo_tokens':192,
                       'user_text':('These are 2 Sentinel-1 observations of the same area in chronological order, '
                         f"taken on {', '.join(x['dates'])}: <EO> Did a flood occur between the two observations? Answer with yes or no.")}
                      for i,x in enumerate(self.items)]
        write_rows(self.e5/'prompts.jsonl',self.prompts)
        write(self.e5/'eval_sets.json',self.sets);write_rows(self.e5/'parent_snapshot/c1_quality.jsonl',self.quality)
        write(self.e5/'parent_snapshot/c1_selected_ids.json',self.selected)
        self.original = Path('/synthetic/e5_equal_budget_v0')
        self.plan = {'root':str(self.original.parent),'output_directory':self.original.name,'parents':{}}
        self.manifest = {'schema':'e5-equal-budget-prepared-v0','files_sha256':{},'parent_snapshot_sha256':{}}
        self.status = {'status':'completed','n_models':12,'n_rows':26325,'verdict':'mixed_or_inconclusive',
                       'at':'2026-09-25T08:00:00Z'}
        primaries = {s:{'synthetic_primary_checked':True}for s in builder.SEEDS}
        self.scores = {'schema':'e5-equal-budget-scores-v0','valid':True,'verdict':'mixed_or_inconclusive',
                       'coverage':{'expected':26325,'received':26325,'n_items':5989,'n_train':4234,'n_test':1755,'n_primary':902},
                       'metrics':{s:{'primary_same_prompt':p}for s,p in primaries.items()},'seed_decisions':{s:{}for s in builder.SEEDS}}
        self.audit = {'schema':'e5-independent-result-audit-v0','consistent':True,'checkpoint_tensors_loaded_and_checked_on_cpu':True,
                      'audit_code_sha256':builder.AUDITOR_SHA,'n_models':12,'n_steps':19080,'n_training_exposures':152424,
                      'n_answers':26325,'primary_n':902,'primary_events':8,'checked_at':'2026-09-25T08:10:00Z',
                      'artifact':str(self.original),'verdict':'mixed_or_inconclusive','primary_metrics':primaries,
                      'seed_decisions':{s:{}for s in builder.SEEDS}}
        self.reseal_e5()
        cases = {}
        for item in self.target:
            prompt = ('These are 2 Sentinel-1 observations of the same area in chronological order, '
                      "taken on 2020-01-01, 2020-01-13: <EO> Did a flood occur between the two observations? Answer with yes or no.")
            cases[item['tile']] = {'tile':item['tile'],'question_id':item['id'],'dates':item['dates'],'slots':item['slots'],
              'reference_label':item['answer'],'reference_label_role':'source_mask_derived_qa_label','run_id':'E2',
              'source_role':'historical_model_output','prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
              'reader_predictions':{'1':'no','2':'yes','3':None},'blind_predictions':{'1':'no','2':'no','3':'no'},
              'source_sha256':{'items':digest(self.e5/'items.jsonl'),'quality':digest(self.e5/'parent_snapshot/c1_quality.jsonl')},
              'c1_membership':{'quality_eligible':item['id']in self.eligible,'supported_stratum':item['id']in primary,
                               'scope':'quality_symmetric_primary_subset'},
              'e3_control':{'frozen_test_sentinel':['E3',1,None]},'e4_control':{'frozen_test_sentinel':['E4',0,'no']}}
        self.reader = {'schema_version':'eo_reader_cases_v1','checked_at':'2026-09-25T05:00:00Z',
                       'run_id':'E2','source_role':'historical_model_output','summary':{'case_count':914,'unchanged':True},
                       'cases':cases,'provenance':{'old':'keep verbatim'}}
        catalog = {'records':[{'id':x['tile'],'dataset':'kurosiwo','split':'test','aoi_id':x['event'],
                              'event_date':'2020-01-13','source_dates_v8':{'unchanged':True}}for x in self.target]}
        write(self.base/'reader_cases.json',self.reader);write(self.base/'catalog.json',catalog)
        write(self.base/'research_sources.json',{'existing_date_provenance':'preserve'})
        write(self.base/'research_runs.json',{'historical_E5_scores':'do not modify'})
        write(self.base/'meta.json',{'unchanged':'yes'})
        (self.base/'index.html').write_text('<p>Existing UI</p>\n')
        (self.base/'previews').mkdir();(self.base/'previews/example.png').write_bytes(b'\x89PNG\r\n\x1a\nSYNTHETIC')
        for name in builder.EXTRAS-{'v10_build_manifest.json'}:
            write(self.base/name,{'old_snapshot_validation':True})
        self.reseal_base()

    def reseal_e5(self):
        self.plan['parents']={name:digest(self.e5/'parent_snapshot'/name)for name in ('c1_quality.jsonl','c1_selected_ids.json')}
        write(self.e5/'prereg.json',self.plan);self.plan_sha=digest(self.e5/'prereg.json')
        self.manifest['files_sha256']={name:digest(self.e5/name)for name in ('prereg.json','items.jsonl','eval_sets.json')}
        self.manifest['parent_snapshot_sha256']=copy.deepcopy(self.plan['parents'])
        write(self.e5/'manifest.json',self.manifest);self.parent_sha=digest(self.e5/'manifest.json')
        self.scores.update(manifest_sha256=self.parent_sha,predictions_sha256=digest(self.e5/'predictions.jsonl'),
                           prompts_sha256=digest(self.e5/'prompts.jsonl'))
        write(self.e5/'scores.json',self.scores);write(self.e5/'status.json',self.status)
        self.audit['hashes_verified']={str(self.original/name):digest(self.e5/name)for name in builder.SOURCE_NAMES}
        write(self.audit_path,self.audit)

    def reseal_base(self):
        files={str(p.relative_to(self.base)):digest(p)for p in self.base.rglob('*')if p.is_file() and str(p.relative_to(self.base))not in builder.EXTRAS}
        write(self.base/'v10_build_manifest.json',{'schema':'eo-v10-delta-wording-v0','output_files_sha256':files})
        self.base_hashes={str(p.relative_to(self.base)):digest(p)for p in self.base.rglob('*')if p.is_file()}

    def build(self,html=None):
        with mock.patch.multiple(builder,V10_SHA=digest(self.base/'v10_build_manifest.json'),PLAN_SHA=self.plan_sha,PARENT_SHA=self.parent_sha):
            return builder.build(self.base,self.e5,self.audit_path,self.out,html)

    def loaded(self):
        return {name:([builder.decode(line)for line in (self.e5/name).read_text().splitlines()if line.strip()]
                      if name.endswith('.jsonl')else builder.read(self.e5/name))for name in builder.SOURCE_NAMES}


class E5CaseBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=synthetic_population()
        cls.items,cls.rows,cls.sets,cls.target,cls.eligible=cls.data
        cls.source={x['id']:x for x in cls.items};cls.test={k:x for k,x in cls.source.items()if x['partition']=='test'}

    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        return Fixture(Path(temp.name))

    def index(self,rows):
        return builder.index_predictions(rows,self.items,self.source,self.test)

    def test_exact_join_preserves_earlier_reader_fields_and_all_other_assets(self):
        f=self.fixture();report=f.build();got=builder.read(f.out/'reader_cases.json')
        self.assertEqual((report['n_cases'],report['n_saved_answers_attached'],report['primary_membership_count']),(914,13710,902))
        self.assertEqual(report['changed_existing_files'],['reader_cases.json','research_sources.json'])
        self.assertEqual(got['checked_at'],f.reader['checked_at']);self.assertEqual(got['summary'],f.reader['summary'])
        indexed={(str(r['seed']),r['model_arm']+'/'+r['eval_arm'],r['id']):r for r in f.rows}
        for tile,case in got['cases'].items():
            control=case.pop('e5_control');old=f.reader['cases'][tile]
            self.assertEqual(case,old)
            self.assertEqual(control['question_id'],old['question_id']);self.assertEqual(control['source_gold'],old['reference_label'])
            self.assertEqual(control['primary_membership'],old['c1_membership']['supported_stratum'])
            self.assertEqual(control['dates'],old['dates']);self.assertEqual(control['slots'],old['slots'])
            self.assertEqual(set(control['predictions']),set(builder.EVALUATIONS))
            for arm in builder.EVALUATIONS:
                self.assertEqual(control['predictions'][arm],{s:indexed[(s,arm,old['question_id'])]['parsed']for s in builder.SEEDS})
        first=builder.read(f.out/'reader_cases.json')['cases'][f.target[0]['tile']]
        self.assertEqual(first['reference_label'],'yes');self.assertEqual(first['e5_control']['predictions']['pair/native']['2'],'no')
        self.assertIsNone(first['e5_control']['predictions']['delta/native']['3'])
        for name,before in f.base_hashes.items():
            self.assertEqual(digest(f.base/name),before)
            if name not in report['changed_existing_files']:
                self.assertEqual(digest(f.out/name),before)
        for name in builder.SOURCE_NAMES:
            self.assertEqual(digest(f.out/'e5_cases_v11'/name),digest(f.e5/name))
        joins=builder.read(f.out/'e5_cases_v11/case_join_index.json')['records']
        for record in joins:
            self.assertEqual(f.prompts[record['prompt_record_number_1_based']-1]['id'],record['question_id'])
            for arm,seed_map in record['prediction_record_numbers_1_based'].items():
                for seed,n in seed_map.items():
                    row=f.rows[n-1]
                    self.assertEqual((row['id'],str(row['seed']),row['model_arm']+'/'+row['eval_arm']),
                                     (record['question_id'],seed,arm))
        self.assertEqual(set(got['cases']),{x['tile']for x in f.target})

    def test_optional_html_changes_only_explicitly_allowed_extra_file(self):
        f=self.fixture();html=f.root/'reviewed.html';html.write_text('<p>Reviewed E5 display</p>')
        report=f.build(html)
        self.assertEqual(report['changed_existing_files'],['index.html','reader_cases.json','research_sources.json'])
        self.assertEqual((f.out/'index.html').read_bytes(),html.read_bytes())
        self.assertEqual(digest(f.out/'catalog.json'),f.base_hashes['catalog.json'])
        self.assertEqual(digest(f.out/'research_runs.json'),f.base_hashes['research_runs.json'])

    def test_missing_duplicate_unknown_id_and_training_id_rejected(self):
        self.index(self.rows)
        for action in ('missing','duplicate','unknown','training'):
            with self.subTest(action=action):
                rows=list(self.rows)
                if action=='missing':rows.pop()
                elif action=='duplicate':rows[-1]=dict(rows[0])
                else:
                    rows[0]=dict(rows[0]);rows[0]['id']='unknown'if action=='unknown'else self.items[0]['id']
                with self.assertRaises(ValueError):self.index(rows)

    def test_wrong_seed_condition_and_noninteger_seed_rejected(self):
        for field,value in [('seed',4),('seed','1'),('seed',True),('model_arm','donor'),('eval_arm','retrained_no_delta')]:
            with self.subTest(field=field,value=value):
                rows=list(self.rows);rows[0]={**rows[0],field:value}
                with self.assertRaises(ValueError):self.index(rows)

    def test_donor_metadata_gold_index_and_invented_transformed_gold_rejected(self):
        for field,value in [('tile','ks_01000'),('cluster','different_event'),('phen','landslide'),('kind','hard_neg'),
                            ('source_gold','no'),('transformed_gold','no'),('pair_index',0),('pair_index',True)]:
            with self.subTest(field=field):
                rows=list(self.rows);rows[0]={**rows[0],field:value}
                with self.assertRaises(ValueError):self.index(rows)

    def test_raw_parse_consistency_and_explicit_unknown(self):
        rows=list(self.rows);rows[0]={**rows[0],'answer_raw':'uncertain','parsed':None}
        indexed,_=self.index(rows);self.assertIsNone(indexed[('1','full/native',rows[0]['id'])]['parsed'])
        for change in ({'answer_raw':'yes','parsed':'no'},{'parsed':'unknown'},{'answer_raw':None}):
            bad=list(self.rows);bad[0]={**bad[0],**change}
            with self.assertRaises(ValueError):self.index(bad)

    def test_primary_membership_and_quality_eligibility_are_recomputed(self):
        f=self.fixture();loaded=f.loaded();builder.population(loaded)
        first=loaded['eval_sets.json']['primary_same_prompt'][0]
        unsupported=next(x['id']for x in f.target if x['id']in f.eligible and x['id']not in f.sets['primary_same_prompt'])
        changed=copy.deepcopy(loaded)
        selected=set(changed['eval_sets.json']['primary_same_prompt']);selected.remove(first);selected.add(unsupported)
        changed['eval_sets.json']['primary_same_prompt']=[x['id']for x in f.items if x['id']in selected]
        with self.assertRaisesRegex(ValueError,'supported quality strata'):builder.population(changed)
        changed=copy.deepcopy(loaded);changed['parent_snapshot/c1_quality.jsonl'][0]['eligible_symmetric_quality']=False
        with self.assertRaisesRegex(ValueError,'Quality eligibility'):builder.population(changed)
        changed=copy.deepcopy(loaded);changed['parent_snapshot/c1_selected_ids.json']['quality_symmetric'].pop()
        with self.assertRaisesRegex(ValueError,'quality selection'):builder.population(changed)

    def test_existing_case_gold_dates_prompt_and_membership_cannot_change(self):
        f=self.fixture();loaded=f.loaded();pins={name:digest(f.e5/name)for name in builder.SOURCE_NAMES}
        tile=f.target[0]['tile'];catalog=builder.read(f.base/'catalog.json')
        mutations=[lambda c:c.update(reference_label='no'),lambda c:c.update(question_id='other'),
                   lambda c:c.update(dates=['2020-01-02','2020-01-13']),lambda c:c.update(prompt='Another question'),
                   lambda c:c['c1_membership'].update(supported_stratum=False),lambda c:c['source_sha256'].update(items='a'*64)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                reader=copy.deepcopy(f.reader);mutate(reader['cases'][tile])
                with self.assertRaises(ValueError):builder.attach_cases(reader,catalog,loaded,f.audit,pins,digest(f.audit_path))

    def test_catalog_training_unknown_or_wrong_event_rejected(self):
        f=self.fixture();loaded=f.loaded();pins={name:digest(f.e5/name)for name in builder.SOURCE_NAMES}
        original=builder.read(f.base/'catalog.json')
        for field,value in [('split','train'),('id','unknown'),('aoi_id','different')]:
            with self.subTest(field=field):
                catalog=copy.deepcopy(original);catalog['records'][0][field]=value
                with self.assertRaises(ValueError):builder.attach_cases(f.reader,catalog,loaded,f.audit,pins,digest(f.audit_path))

    def test_saved_prompt_order_gold_index_and_actual_text_mismatch(self):
        f=self.fixture();loaded=f.loaded();pins={name:digest(f.e5/name)for name in builder.SOURCE_NAMES}
        catalog=builder.read(f.base/'catalog.json')
        changed=copy.deepcopy(loaded)
        changed['prompts.jsonl'][0],changed['prompts.jsonl'][1]=changed['prompts.jsonl'][1],changed['prompts.jsonl'][0]
        with self.assertRaisesRegex(ValueError,'prompt order/coverage'):builder.population(changed)
        for field,value in [('source_gold','no'),('pair_index',1),('n_eo_tokens',191)]:
            changed=copy.deepcopy(loaded);changed['prompts.jsonl'][0][field]=value
            with self.assertRaisesRegex(ValueError,'prompt index/label/token'):builder.population(changed)
        changed=copy.deepcopy(loaded);index=next(i for i,x in enumerate(f.items)if x['id']==f.target[0]['id'])
        changed['prompts.jsonl'][index]['user_text']='A different original question'
        with self.assertRaisesRegex(ValueError,'actual saved E5 source question'):
            builder.attach_cases(f.reader,catalog,changed,f.audit,pins,digest(f.audit_path))

    def test_hash_gate_audit_invalid_and_overwrite_refusal(self):
        f=self.fixture();write(f.audit_path,{**f.audit,'checkpoint_tensors_loaded_and_checked_on_cpu':False})
        with self.assertRaisesRegex(ValueError,'CPU tensor audit'):f.build()
        write(f.audit_path,f.audit)
        with (f.e5/'predictions.jsonl').open('a')as stream:stream.write('\n')
        with self.assertRaisesRegex(ValueError,'Source hash mismatch'):f.build()
        write_rows(f.e5/'predictions.jsonl',f.rows);f.build()
        before=digest(f.out/'reader_cases.json')
        with self.assertRaisesRegex(ValueError,'no overwrite'):f.build()
        self.assertEqual(digest(f.out/'reader_cases.json'),before)

    def test_midcopy_tamper_retains_invalid_partial_result(self):
        f=self.fixture();original=builder.shutil.copyfile
        def altered(source,dest,**kwargs):
            value=original(source,dest,**kwargs)
            if Path(source)==f.e5/'predictions.jsonl':Path(dest).write_text('tampered\n')
            return value
        with mock.patch.object(builder.shutil,'copyfile',side_effect=altered):
            with self.assertRaisesRegex(ValueError,'E5 source copy changed'):f.build()
        self.assertTrue(builder.read(f.out/'v11_failure.json')['partial_copy_not_valid'])
        self.assertFalse((f.out/'v11_build_manifest.json').exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
