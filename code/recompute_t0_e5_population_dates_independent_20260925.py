#!/usr/bin/env python3
"""Independent descriptive date audit. No predictions, model scores or C1 results read."""
import collections
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
OUT=Path('/private/tmp/T0_E5_POPULATION_DATE_DIAGNOSTIC_INDEPENDENT_20260925.json')
PLAN_SHA='fc7866feebbf1e9fcd4e93de5154bf9ffa141f434e70dc4731390cf18dfd4696'
MAPPING={'pre_1':'SL2','pre_2':'SL1','post':'MS1'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def stats(values):return {'min':min(values),'median':statistics.median(values),'max':max(values),'mean':statistics.mean(values)}
def index_unique(rows,key):
    result={r[key]:r for r in rows};assert len(rows)==len(result);return result

def describe(group):
    events={e:[r for r in group if r['event']==e] for e in sorted({r['event'] for r in group},key=int)}
    gap=lambda r:r['documented_gap_days']
    per_event={e:{'n_items':len(rows),'kind_counts':dict(collections.Counter(r['kind'] for r in rows)),
                  'gap_set_days':sorted({gap(r) for r in rows}), 'mean_gap_days':statistics.mean(map(gap,rows)),
                  'both_dates_equal_count':sum(r['both_dates_equal'] for r in rows),
                  'gap_equal_count':sum(r['gap_equal'] for r in rows)} for e,rows in events.items()}
    return {'n_items':len(group),'n_tiles':len({r['tile'] for r in group}),'n_events':len(events),
            'documented_gap_days':stats(list(map(gap,group))),
            'both_dates_equal_to_current':sum(r['both_dates_equal'] for r in group),
            'individual_date_occurrences_equal':sum(r['individual_dates_equal'] for r in group),
            'gap_equal_to_current':sum(r['gap_equal'] for r in group),
            'event_gap_sets_days':{e:d['gap_set_days'] for e,d in per_event.items()},'per_event':per_event,
            'tile_or_item_weighted_gap_equal_fraction':statistics.mean(r['gap_equal'] for r in group),
            'tile_or_item_weighted_both_dates_equal_fraction':statistics.mean(r['both_dates_equal'] for r in group),
            'event_balanced_mean_gap_days':statistics.mean(d['mean_gap_days'] for d in per_event.values()),
            'event_balanced_gap_equal_fraction':statistics.mean(d['gap_equal_count']/d['n_items'] for d in per_event.values()),
            'event_balanced_both_dates_equal_fraction':statistics.mean(d['both_dates_equal_count']/d['n_items'] for d in per_event.values()),
            'events_with_any_non12_day_gap':sum(any(g!=12 for g in d['gap_set_days']) for d in per_event.values()),
            'events_with_all_non12_day_gaps':sum(all(g!=12 for g in d['gap_set_days']) for d in per_event.values())}

def main():
    assert not OUT.exists()
    planpath=REPO/'config/e5_equal_budget_prereg_v0.json';assert sha(planpath)==PLAN_SHA;plan=read(planpath)
    frozen=REPO/'code/e5_bundle_v0/frozen_inputs'
    names=['items.jsonl','c1_quality.jsonl','c1_quality_manifest.json','c1_selected_ids.json']
    hashes={str(planpath):sha(planpath)}
    for name in names:
        p=frozen/name;assert sha(p)==plan['parents'][name];hashes[str(p)]=sha(p)
    items=lines(frozen/'items.jsonl');by_id=index_unique(items,'id');assert len(items)==5989
    assert collections.Counter(i['partition'] for i in items)=={'train':4234,'test':1755}
    quality=lines(frozen/'c1_quality.jsonl');quality_by_id=index_unique(quality,'id');assert len(quality)==914
    cutoff=plan['population']['primary_same_prompt']['symmetric_valid_fraction_min'];assert cutoff==.9
    selected=read(frozen/'c1_selected_ids.json')
    assert set(quality_by_id)==set(selected['all_original_targets'])
    quality_ids=set()
    for q in quality:
        i=by_id[q['id']];assert i['phen']=='flood' and i['partition']=='test'
        assert i['kind']==q['kind'] and i['tile']==q['tile'] and str(i['event'])==str(q['event']) and i['dates']==q['dates'] and i['slots']==q['slots']
        assert q['total_px']>0 and q['valid_frac']==q['labelled_px']/q['total_px']
        good=q['labelled_px']/q['total_px']>=cutoff;assert good==q['eligible_symmetric_quality']
        if good:quality_ids.add(i['id'])
    assert quality_ids==set(selected['quality_symmetric']) and len(quality_ids)==907
    strata=collections.defaultdict(lambda:collections.defaultdict(list))
    for sid in quality_ids:
        i=by_id[sid];strata[(i['event'],tuple(i['dates']),tuple(i['slots']))][i['kind']].append(sid)
    primary=set();excluded=[]
    for key,kinds in strata.items():
        assert set(kinds)<= {'pos','hard_neg'}
        if kinds.get('pos') and kinds.get('hard_neg'):
            for ids in kinds.values():primary.update(ids)
        else:excluded.extend(sid for ids in kinds.values() for sid in ids)
    assert len(primary)==902 and collections.Counter(by_id[s]['kind'] for s in primary)=={'pos':445,'hard_neg':457}
    assert len({by_id[s]['event'] for s in primary})==8
    t0=REPO/'artifacts/t0_source_date_join_v0_20260925';manifest=read(t0/'manifest.json');assert manifest['valid'] is True
    hashes[str(t0/'manifest.json')]=sha(t0/'manifest.json')
    for name,digest in manifest['files_sha256'].items():assert sha(t0/name)==digest
    joined=lines(t0/'joined_rows.jsonl');by_tile=index_unique(joined,'id');assert len(joined)==7000 and len({r['grid_id'] for r in joined})==7000
    assert collections.Counter(r['partition'] for r in joined)=={'train':4000,'validation':1000,'test':2000}
    hashes[str(t0/'joined_rows.jsonl')]=sha(t0/'joined_rows.jsonl')
    recomputed={};catalog_rows=[]
    for r in joined:
        assert r['status']=='matched_metadata' and all(v['equal'] for v in r['field_checks'].values())
        # Independently use upstream raw source_date, not stored gap statistics.
        dates={s:date.fromisoformat(r['temporal']['raw_sources'][k]['source_date']) for s,k in MAPPING.items()}
        event=datetime.fromisoformat(r['field_checks']['flood_date']['current']).date()
        synthetic={'pre_1':event-timedelta(days=24),'pre_2':event-timedelta(days=12),'post':event}
        assert {s:d.isoformat() for s,d in dates.items()}==r['temporal']['published_slot_dates']
        assert {s:d.isoformat() for s,d in synthetic.items()}==r['temporal']['old_synthetic_date_only']
        assert dates['pre_1']<dates['pre_2']<dates['post']
        recomputed[r['id']]={'dates':dates,'synthetic':synthetic}
        a,b=dates['pre_2'],dates['post'];old_a,old_b=synthetic['pre_2'],synthetic['post']
        catalog_rows.append({'tile':r['id'],'event':str(r['event_id']),'partition':r['partition'],'kind':'catalog_pre2_post',
                             'documented_gap_days':(b-a).days,'both_dates_equal':[a,b]==[old_a,old_b],
                             'individual_dates_equal':sum(x==y for x,y in zip([a,b],[old_a,old_b])),
                             'gap_equal':(b-a)==(old_b-old_a)})
    diagnostics=[]
    for item in items:
        if item['phen']!='flood':continue
        r=by_tile[item['tile']]; t=recomputed[item['tile']]
        assert item['partition']==r['partition'] and str(item['event'])==str(r['event_id'])
        days=[t['dates'][s] for s in item['slots']];old=[t['synthetic'][s] for s in item['slots']]
        assert [d.isoformat() for d in old]==item['dates'] and len(days)==2
        assert item['slots']==(['pre_1','pre_2'] if item['kind']=='neg' else ['pre_2','post'])
        diagnostics.append({'id':item['id'],'tile':item['tile'],'partition':item['partition'],'kind':item['kind'],
                            'event':str(item['event']),'primary':item['id'] in primary,
                            'documented_dates':[d.isoformat() for d in days],
                            'documented_gap_days':(days[1]-days[0]).days,'imputed_gap_days':(old[1]-old[0]).days,
                            'both_dates_equal':days==old,'individual_dates_equal':sum(a==b for a,b in zip(days,old)),
                            'gap_equal':days[1]-days[0]==old[1]-old[0]})
    assert len(diagnostics)==4569
    groups={'C1_primary_902':[r for r in diagnostics if r['primary']]}
    groups.update({part+'/'+kind:[r for r in diagnostics if r['partition']==part and r['kind']==kind]
                   for part in ['train','test'] for kind in ['pos','neg','hard_neg']})
    summaries={k:describe(v) for k,v in groups.items()}
    catalog={'all':describe(catalog_rows)}
    catalog.update({part:describe([r for r in catalog_rows if r['partition']==part]) for part in ['train','validation','test']})
    target_path=REPO/'artifacts/T0_E5_POPULATION_DATE_DIAGNOSTIC_20260925.json';target=read(target_path)
    hashes[str(target_path)]=sha(target_path)
    target_by_id=index_unique(target['rows'],'id');new_by_id=index_unique(diagnostics,'id')
    assert set(target_by_id)==set(new_by_id)
    differences=[]
    for sid,old in target_by_id.items():
        for key,value in old.items():
            if new_by_id[sid][key]!=value:differences.append([sid,key,value,new_by_id[sid][key]])
    for name,stored in target['summary'].items():
        for key,value in stored.items():
            rebuilt=summaries[name][key]
            if key=='documented_gap_days':rebuilt={k:rebuilt[k] for k in value}
            if rebuilt!=value:differences.append([name,key,value,rebuilt])
    independent=Path('/private/tmp/T0_INDEPENDENT_RESULT_REVIEW_20260925.json');basis=read(independent)
    assert basis['consistent'] and basis['n_rows']==7000 and basis['n_unique_grid_matches']==7000
    hashes[str(independent)]=sha(independent)
    final={'schema':'t0-e5-date-population-independent-review-v0','checked_at':datetime.now(timezone.utc).isoformat(),
           'consistent':not differences,'differences':differences,'all_reported_summary_fields_compared':True,
           'all_flood_item_rows_compared':len(diagnostics), 'primary_ids_rebuilt_from_quality_only':sorted(primary),
           'primary_quality_eligible':907,'primary_quality_eligible_but_unsupported_ids':sorted(excluded),
           'c1_results_or_prediction_files_read':False,'model_output_files_read':False,
           'source_scope':'raw source_date in verified T0 join + independently rebuilt frozen quality memberships; no target script imports',
           'population_summary':summaries,'catalog_pre2_post_summary':catalog,'source_sha256':hashes,
           'script_sha256':sha(Path(__file__)),'basis_full_7000_independent_review':{'consistent':basis['consistent'],'four_corner_matches_checked':basis['four_corner_matches_checked'],'raw_six_field_comparisons':basis['raw_six_field_comparisons'],'published_slot_dates_checked':basis['published_slot_dates_checked']}}
    with OUT.open('x') as f:json.dump(final,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'consistent':final['consistent'],'differences':differences,'primary':summaries['C1_primary_902'],
                      'catalog':{k:{m:v[m] for m in ['n_items','n_events','documented_gap_days','both_dates_equal_to_current','gap_equal_to_current','event_balanced_mean_gap_days']} for k,v in catalog.items()},'output':str(OUT),'sha256':sha(OUT)},indent=2))
if __name__=='__main__':main()
