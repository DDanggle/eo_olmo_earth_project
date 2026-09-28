"""Invented E6 populations/answers only; no model results or filesystem data."""
import copy
import math
from pathlib import Path
import random
import sys
import unittest

for directory in (Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent):
    if (directory/'e6_scoring_v0.py').is_file():
        sys.path.insert(0, str(directory))
        break
import e6_scoring_v0 as e6


def change_head(row, value):
    row['logit'] = 2.0 if value == 'yes' else -2.0
    row['probability'] = 1/(1+math.exp(-row['logit']))
    row['prediction'] = value


def change_reference(row, value):
    row['parsed'] = value
    row['answer_raw'] = value if value is not None else 'uncertain'


def synthetic():
    items = []

    def append(partition, phen, tile, event, kind, special_date=False):
        dates = ['2020-01-01', '2020-01-13'] if kind != 'neg' else ['2019-12-20', '2020-01-01']
        if special_date:
            dates = ['2021-01-01', '2021-01-13']
        item = {'id': tile + '_' + kind, 'tile': tile, 'phen': phen, 'cluster': event,
                'partition': partition, 'kind': kind, 'answer': 'yes' if kind == 'pos' else 'no', 'dates': dates}
        if phen == 'flood':
            item.update(event=event, slots=['pre_1', 'pre_2'] if kind == 'neg' else ['pre_2', 'post'])
        items.append(item)

    for phen, n_pairs, events in (('flood', 1066, 27), ('landslide', 518, 7)):
        for index in range(n_pairs):
            for kind in ('pos', 'neg'):
                append('train', phen, f'train_{phen}_{index}', f'train_{index % events}', kind)
            if phen == 'flood':
                append('train', phen, f'train_hard_{index}', f'train_{index % events}', 'hard_neg')
    for event, n_pairs in enumerate((56, 56, 56, 56, 56, 55, 55, 55, 6, 6)):
        for index in range(n_pairs):
            for kind in ('pos', 'neg'):
                append('test', 'flood', f'flood_{event}_{index}', str(event), kind,
                       special_date=event == 0 and index == 0 and kind == 'pos')
    for event, n_hard in enumerate((57, 57, 57, 57, 57, 57, 57, 58)):
        for index in range(n_hard):
            append('test', 'flood', f'hard_{event}_{index}', str(event), 'hard_neg',
                   special_date=event == 0 and index == 0)
    for event, n_pairs in enumerate((186, 6)):
        for index in range(n_pairs):
            for kind in ('pos', 'neg'):
                append('test', 'landslide', f'landslide_{event}_{index}', str(event), kind)
    test = [item for item in items if item['partition'] == 'test']
    eval_sets = {
        'all_test': [item['id'] for item in test],
        'primary_same_prompt': [item['id'] for item in test if item['phen'] == 'flood'
                               and item['kind'] in ('pos', 'hard_neg') and int(item['cluster']) < 8],
        'paired_flood': [item['id'] for item in test if item['phen'] == 'flood' and item['kind'] != 'hard_neg'],
        'hard_negative_flood': [item['id'] for item in test if item['kind'] == 'hard_neg'],
        'landslide': [item['id'] for item in test if item['phen'] == 'landslide'],
        'e3_subset': [item['id'] for item in test[:209]],
    }
    rows = []
    indices = {item['id']: index for index, item in enumerate(items)}
    for seed in (1, 2, 3):
        for arm, evaluation in (('full', 'native'),):
            for item in test:
                rows.append({'seed': seed, 'model_arm': arm, 'eval_arm': evaluation,
                             **{field: item[field] for field in ('id', 'tile', 'cluster', 'phen', 'kind')},
                             'source_gold': item['answer'], 'transformed_gold': None,
                             'parsed': item['answer'], 'answer_raw': item['answer'].upper() + '.',
                             'pair_index': indices[item['id']]})
    heads = []
    for row in rows:
        head = {k:v for k,v in row.items() if k not in ('parsed','answer_raw')}
        head['model_arm'] = 'full_head'
        change_head(head, row['source_gold'])
        heads.append(head)
    return items, heads, rows, eval_sets

class E6ScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = synthetic()

    def check(self, data=None):
        items, head, reference, sets = data or self.fixture
        return e6.score_run(head, reference, items, sets)

    def test_perfect_complete_fixed_reference_descriptive_only(self):
        result=self.check()
        self.assertTrue(result['valid'],result)
        self.assertEqual(result['schema'],'e6-no-llm-scores-v0')
        self.assertEqual(result['verdict'],'descriptive_system_comparison')
        self.assertEqual(result['coverage']['received_head'],5265)
        self.assertEqual(result['coverage']['received_reference'],5265)
        self.assertEqual(set(result['metrics']),{'1','2','3'})
        self.assertNotIn('seed_decisions',result)
        for seed,metrics in result['metrics'].items():
            primary=metrics['primary_same_prompt']
            self.assertEqual(set(primary['evaluations']),{'full/native','full_head/native'})
            self.assertEqual(primary['evaluations']['full_head/native']['n_items'],902)
            self.assertEqual(primary['evaluations']['full_head/native']['n_events'],8)
            self.assertEqual(primary['evaluations']['full_head/native']['macro_ba'],1.)
            self.assertEqual(primary['contrasts']['full_minus_head']['delta'],0.)
            self.assertEqual(primary['contrasts']['full_minus_head']['ci95_delta'],[0.,0.])

    def test_equal_strata_then_equal_events_and_paired_event_bootstrap(self):
        data=copy.deepcopy(self.fixture)
        for row in data[1]:
            if row['id'] in ('flood_0_0_pos','hard_0_0_hard_neg'):
                change_head(row,'no' if row['source_gold']=='yes' else 'yes')
        result=self.check(data);self.assertTrue(result['valid'],result)
        p=result['metrics']['1']['primary_same_prompt'];head=p['evaluations']['full_head/native']
        self.assertEqual(head['n_strata'],9)
        self.assertEqual(head['events']['0']['ba'],.5)
        self.assertEqual(head['macro_ba'],7.5/8)
        self.assertEqual(p['contrasts']['full_minus_head']['delta'],.5/8)
        self.assertEqual(p['contrasts']['full_minus_head']['ci95_delta'],[0.,.1875])

    def test_negative_contrast_direction_no_success_cutoff(self):
        data=copy.deepcopy(self.fixture)
        for row in data[2]:change_reference(row,'no')
        result=self.check(data);self.assertTrue(result['valid'],result)
        c=result['metrics']['1']['primary_same_prompt']['contrasts']['full_minus_head']
        self.assertEqual(c['delta'],-.5)
        self.assertEqual(c['ci95_delta'],[-.5,-.5])
        self.assertEqual(result['verdict'],'descriptive_system_comparison')

    def test_seeds_separate_and_no_averaged_primary_interval(self):
        data=copy.deepcopy(self.fixture)
        for row in data[1]:
            if row['seed']==3:change_head(row,'no')
        result=self.check(data);self.assertTrue(result['valid'],result)
        cs=[result['metrics'][str(s)]['primary_same_prompt']['contrasts']['full_minus_head']['delta'] for s in (1,2,3)]
        self.assertEqual(cs,[0.,0.,.5])
        self.assertNotIn('mean',result['metrics'])
        self.assertEqual(result['verdict'],'descriptive_system_comparison')

    def test_zero_logit_tie_is_yes(self):
        data=copy.deepcopy(self.fixture);r=data[1][0]
        r.update(logit=-0.0,probability=.5,prediction='yes')
        self.assertTrue(self.check(data)['valid'])
        r['prediction']='no'
        self.assertFalse(self.check(data)['valid'])

    def test_sigmoid_extreme_logits_do_not_overflow(self):
        data=copy.deepcopy(self.fixture)
        data[1][0].update(logit=1000.,probability=1.,prediction='yes')
        data[1][1].update(logit=-1000.,probability=0.,prediction='no')
        self.assertTrue(self.check(data)['valid'])

    def test_probability_consistency_and_finite_contract(self):
        for field,value in [('logit',float('nan')),('logit',float('inf')),('probability',float('nan')),
                            ('probability',-1e-10),('probability',1.1),('logit',True),('probability',False),
                            ('logit','2.0')]:
            with self.subTest(field=field,value=value):
                data=copy.deepcopy(self.fixture);data[1][0][field]=value
                self.assertFalse(self.check(data)['valid'])
        data=copy.deepcopy(self.fixture);r=data[1][0];r.update(logit=0.,prediction='yes',probability=.5000005)
        self.assertTrue(self.check(data)['valid'])
        r['probability']=.5001;self.assertFalse(self.check(data)['valid'])

    def test_head_missing_prediction_is_invalid(self):
        data=copy.deepcopy(self.fixture);data[1][0]['prediction']=None
        self.assertFalse(self.check(data)['valid'])

    def test_reference_unparsed_negative_is_incorrect_not_true_negative(self):
        data=copy.deepcopy(self.fixture)
        row=next(r for r in data[2] if r['seed']==1 and r['id']=='hard_0_0_hard_neg')
        change_reference(row,None)
        result=self.check(data);self.assertTrue(result['valid'],result)
        primary=result['metrics']['1']['primary_same_prompt']['evaluations']['full/native']
        tiny=next(s for s in primary['events']['0']['strata'] if s['n_pos']==1)
        self.assertEqual(tiny['specificity'],0.)
        self.assertEqual(tiny['fpr'],0.)
        self.assertEqual(tiny['ba'],.5)
        self.assertEqual(primary['macro_ba'],7.75/8)
        self.assertEqual(result['metrics']['1']['hard_negative']['full/native']['parse_failures'],1)

    def test_reference_parse_rate_is_per_seed_and_phenomenon(self):
        data=copy.deepcopy(self.fixture)
        land=[r for r in data[2] if r['seed']==1 and r['phen']=='landslide']
        for row in land[:3]:change_reference(row,None)
        self.assertTrue(self.check(data)['valid'])
        change_reference(land[3],None)
        result=self.check(data);self.assertFalse(result['valid']);self.assertIn('parse failure',result['invalid_reason'])

    def test_reference_raw_parse_contract(self):
        data=copy.deepcopy(self.fixture)
        data[2][0]['answer_raw']='NO.' if data[2][0]['parsed']=='yes' else 'YES.'
        self.assertFalse(self.check(data)['valid'])

    def test_other_reference_arm_or_intervention_is_rejected(self):
        for arm,mode in [('pair','native'),('later','native'),('delta','native'),('full','full_no_delta'),('full_head','native')]:
            with self.subTest(arm=arm,mode=mode):
                data=copy.deepcopy(self.fixture);data[2][0].update(model_arm=arm,eval_arm=mode)
                self.assertFalse(self.check(data)['valid'])

    def test_missing_duplicate_rows_or_seeds_invalid_without_repair(self):
        for system in (1,2):
            for change in ('missing','extra','replace_duplicate','missing_seed'):
                with self.subTest(system=system,change=change):
                    data=copy.deepcopy(self.fixture);rows=data[system]
                    if change=='missing':rows.pop()
                    if change=='extra':rows.append(copy.deepcopy(rows[0]))
                    if change=='replace_duplicate':rows[-1]=copy.deepcopy(rows[0])
                    if change=='missing_seed':
                        for row in rows:
                            if row['seed']==3:row['seed']=2
                    self.assertFalse(self.check(data)['valid'])

    def test_original_metadata_and_global_pair_index_are_required(self):
        changes={'tile':'wrong','cluster':'wrong','phen':'wrong','kind':'wrong','pair_index':0,'seed':True,
                 'source_gold':'no','transformed_gold':'yes','id':'train_flood_0_pos'}
        for system in (1,2):
            for key,value in changes.items():
                with self.subTest(system=system,key=key):
                    data=copy.deepcopy(self.fixture);row=data[system][0]
                    if key=='source_gold':value='no' if row['source_gold']=='yes' else 'yes'
                    row[key]=value
                    self.assertFalse(self.check(data)['valid'])
            data=copy.deepcopy(self.fixture);data[system][0].pop('transformed_gold')
            self.assertFalse(self.check(data)['valid'])

    def test_frozen_evaluation_membership_and_order(self):
        data=copy.deepcopy(self.fixture);data[3]['all_test'][0]=data[0][0]['id']
        self.assertFalse(self.check(data)['valid'])
        data=copy.deepcopy(self.fixture);data[3]['paired_flood'].pop();self.assertFalse(self.check(data)['valid'])
        data=copy.deepcopy(self.fixture);ids=data[3]['all_test'];ids[0],ids[1]=ids[1],ids[0]
        self.assertFalse(self.check(data)['valid'])
        data=copy.deepcopy(self.fixture);data[3]['primary_same_prompt'][0]=data[3]['paired_flood'][-2]
        self.assertFalse(self.check(data)['valid'])

    def test_missing_primary_class_stratum_is_not_excluded(self):
        data=copy.deepcopy(self.fixture)
        next(i for i in data[0] if i['id']=='hard_0_0_hard_neg')['dates']=['2099-01-01','2099-01-13']
        result=self.check(data);self.assertFalse(result['valid']);self.assertIn('stratum lacks',result['invalid_reason'])

    def test_fixed_source_counts_and_cross_partition_event_leakage(self):
        data=copy.deepcopy(self.fixture);data[0].pop();self.assertFalse(self.check(data)['valid'])
        data=copy.deepcopy(self.fixture)
        for item in data[0]:
            if item['phen']=='flood' and item['partition']=='train' and item['cluster']=='train_0':
                item['cluster']='0';item['event']='0'
        self.assertFalse(self.check(data)['valid'])

    def test_landslide_macro_is_region_balanced_and_secondary_has_no_ci(self):
        data=copy.deepcopy(self.fixture)
        for row in data[1]:
            if row['phen']=='landslide' and row['cluster']=='1':change_head(row,'no' if row['source_gold']=='yes' else 'yes')
        result=self.check(data);self.assertTrue(result['valid'],result)
        paired=result['metrics']['1']['paired_source']
        land=paired['landslide']['evaluations']['full_head/native']
        self.assertEqual(land['n_items'],384)
        self.assertEqual(land['macro_ba'],.5)
        self.assertEqual(land['events']['0']['ba'],1.)
        self.assertEqual(land['events']['1']['ba'],0.)
        for phen in ['flood','landslide']:
            self.assertIsNone(paired[phen]['contrasts']['full_minus_head']['ci95_delta'])
        flood=paired['flood']['evaluations']['full_head/native']
        self.assertEqual((flood['n_items'],flood['n_events']),(914,10))

    def test_hard_negative_fpr_pooled_and_event_macro_are_separate(self):
        data=copy.deepcopy(self.fixture)
        for row in data[1]:
            if row['kind']=='hard_neg' and row['cluster']=='7':change_head(row,'yes')
        result=self.check(data);self.assertTrue(result['valid'],result)
        hard=result['metrics']['1']['hard_negative']['full_head/native']
        self.assertEqual(hard['n'],457)
        self.assertEqual(hard['fpr_pooled'],58/457)
        self.assertEqual(hard['fpr_event_macro'],1/8)

    def test_row_arrival_order_invariant_and_inputs_unchanged(self):
        data=copy.deepcopy(self.fixture);before=copy.deepcopy(data)
        first=self.check(data)
        self.assertEqual(data,before)
        random.Random(5).shuffle(data[1]);random.Random(6).shuffle(data[2])
        second=self.check(data)
        self.assertEqual(first,second)

if __name__=='__main__':unittest.main()
