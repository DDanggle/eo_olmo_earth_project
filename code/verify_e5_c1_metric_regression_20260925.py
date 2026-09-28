#!/usr/bin/env python3
"""Regression only: repeat historical E2 answers across a synthetic E5-shaped fixture.

No E5 training, model inference, or E5 output file is read or created. The scorer's
synthetic decision is intentionally not reported as an experimental outcome.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]
def require(ok,message):
    if not ok:raise ValueError(message)
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def verify(bundle,repo):
    config_path=bundle/'e5_equal_budget_prereg_v0.json';cfg=read(config_path);inputs=bundle/'frozen_inputs'
    hashes={str(config_path):sha(config_path)};parents={}
    for name,digest in cfg['parents'].items():
        path=inputs/name;hashes[str(path)]=sha(path);require(hashes[str(path)]==digest,'Frozen metadata SHA mismatch: '+name)
        parents[name]=rows(path)if name.endswith('.jsonl')else read(path)
    prep=module('e5_regression_prepare',bundle/'e5_prepare_v0.py')
    scorer=module('e5_regression_scorer',bundle/'e5_scoring_v0.py')
    hashes[str(bundle/'e5_prepare_v0.py')]=sha(bundle/'e5_prepare_v0.py')
    hashes[str(bundle/'e5_scoring_v0.py')]=sha(bundle/'e5_scoring_v0.py')
    items=parents['items.jsonl'];by_id={item['id']:item for item in items};indices={item['id']:i for i,item in enumerate(items)}
    ordered,split=prep.validate_population(items,parents['c0_manifest.json']);eval_sets=prep.make_eval_sets(items,parents)
    test_ids=set(ordered['test']);historical={}
    for seed in (1,2,3):
        path=repo/f'artifacts/e2_multi_reader_v0/reader_seed{seed}/answers_real_all.jsonl'
        digest=sha(path);hashes[str(path)]=digest
        expected=parents['c0_manifest.json']['source_sha256'][f'e2_multi_reader_v0/reader_seed{seed}/answers_real_all.jsonl']
        require(digest==expected,'Historical E2 SHA differs from frozen C0 lineage')
        data=rows(path);mapping={row['id']:row for row in data}
        require(len(data)==len(mapping)==1755 and set(mapping)==test_ids,'Historical E2 exact test ID coverage mismatch')
        for key,row in mapping.items():
            item=by_id[key]
            require(all(row[k]==item[k]for k in ('tile','phen','kind','fold')) and row['emb_item']==key and
                    row['text_gold']==row['emb_gold']==item['answer'] and row['parsed'] in ('yes','no'),
                    'Historical E2 metadata mismatch: '+key)
        historical[seed]=mapping
    fixture=[]
    evaluations=[('full','native'),('pair','native'),('later','native'),('delta','native'),('full','full_no_delta')]
    for seed in (1,2,3):
        for arm,evaluation in evaluations:
            for key in ordered['test']:
                item=by_id[key];old=historical[seed][key]
                fixture.append({'seed':seed,'model_arm':arm,'eval_arm':evaluation,'id':key,'tile':item['tile'],
                    'cluster':item['cluster'],'phen':item['phen'],'kind':item['kind'],'pair_index':indices[key],
                    'source_gold':item['answer'],'transformed_gold':None,'answer_raw':old['answer_raw'],'parsed':old['parsed'],
                    'fixture_role':'synthetic_historical_E2_duplication_NOT_E5_outcome'})
    fixture_sha=hashlib.sha256(json.dumps(fixture,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    scores=scorer.score_run(fixture,items,eval_sets)
    require(scores['valid']is True,'Synthetic full-coverage fixture rejected: '+str(scores.get('invalid_reason')))
    c1=parents['c1_results.json']['subsets']['quality_symmetric'];primary=set(eval_sets['primary_same_prompt'])
    comparison={};max_error=0.;bitwise=True;stratum_checks=0;event_checks=0
    for seed in (1,2,3):
        source=c1['per_seed'][str(seed)];c1_events=source['arms']['reader']
        target=scores['metrics'][str(seed)]['primary_same_prompt']['evaluations']['full/native']
        require(set(target['events'])==set(c1_events),'C1/E5 event set mismatch')
        errors=[]
        for event,old in c1_events.items():
            new=target['events'][event]
            old_groups={}
            for group in old['strata']:
                ids=group['pos_ids']+group['hard_neg_ids'];require(set(ids)<=primary,'C1 stratum outside frozen primary')
                keys={(str(by_id[key]['event']),tuple(by_id[key]['dates']),tuple(by_id[key]['slots']))for key in ids}
                require(len(keys)==1 and next(iter(keys))[0]==event,'C1 stratum metadata mismatch')
                group_key=next(iter(keys))[1:];require(group_key not in old_groups,'Duplicate C1 stratum')
                old_groups[group_key]=group
            new_groups={(tuple(group['dates']),tuple(group['slots'])):group for group in new['strata']}
            require(set(new_groups)==set(old_groups),'C1/E5 stratum set mismatch')
            for group_key,old_group in old_groups.items():
                new_group=new_groups[group_key]
                require(new_group['n_pos']==old_group['n_pos'] and new_group['n_negative']==old_group['n_hard_neg'],
                        'C1/E5 stratum class counts differ')
                for key in ('ba','recall','fpr'):
                    error=abs(new_group[key]-old_group[key]);errors.append(error);bitwise &= new_group[key]==old_group[key]
                    require(error<=1e-15,'C1/E5 stratum statistic mismatch: '+key)
                stratum_checks+=1
            for key in ('ba','recall','fpr'):
                error=abs(new[key]-old[key]);errors.append(error);bitwise &= new[key]==old[key]
                require(error<=1e-15,'C1/E5 event statistic mismatch: '+key)
            event_checks+=1
        expected=source['reader_macro_ba'];observed=target['macro_ba'];error=abs(expected-observed);errors.append(error)
        require(error<=1e-15,'C1/E5 macro mean mismatch')
        bitwise &= expected==observed;max_error=max(max_error,max(errors))
        for arm,evaluation in evaluations:
            require(scores['metrics'][str(seed)]['primary_same_prompt']['evaluations'][arm+'/'+evaluation]==target,
                    'Duplicated historical fixture differs between synthetic conditions')
        comparison[str(seed)]={'c1_reader_macro_ba':expected,'e5_scorer_on_synthetic_fixture_macro_ba':observed,
                               'macro_exact_float_equal':expected==observed,'macro_abs_error':error,
                               'max_event_or_stratum_abs_error':max(errors),'n_events':len(c1_events)}
    for path,digest in hashes.items():require(sha(path)==digest,'Input changed during regression check: '+path)
    return {'schema':'e5-c1-metric-regression-v0','checked_at':datetime.now(timezone.utc).isoformat(),'valid':True,
            'fixture_role':'Synthetic duplication of historical E2 reader answers; NOT E5 model outcomes',
            'no_training_or_inference':True,'e5_outcome_files_read_or_created':False,
            'fixture_rows':len(fixture),'fixture_sha256':fixture_sha,'synthetic_scorer_verdict_intentionally_omitted':True,
            'frozen_metadata_files_verified':len(cfg['parents']),'primary_membership_count':len(primary),
            'primary_counts':{'pos':445,'hard_neg':457,'events':8},'event_comparisons':event_checks,'stratum_comparisons':stratum_checks,
            'per_seed':comparison,'all_float_statistics_exactly_equal':bitwise,'max_absolute_difference':max_error,
            'numeric_tolerance':1e-15,'source_sha256':hashes,'script_sha256':sha(__file__),
            'environment':{'python':sys.version,'numpy':np.__version__},
            'interpretation':'This checks metric and membership regression only. Identical replicated conditions contain no evidence about E5 training or arm performance.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    require(not a.out.exists(),'Refusing existing regression report')
    report=verify(a.bundle.resolve(),a.repo.resolve())
    with a.out.open('x')as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('source_sha256','environment')},indent=2))
