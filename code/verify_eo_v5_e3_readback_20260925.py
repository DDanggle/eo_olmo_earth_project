#!/usr/bin/env python3
"""Independent read-only EO v5 case/source/HTTP audit; no model or score execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ARMS = ('real','earlier_only','later_only','repeat_earlier','repeat_later','no_delta','reverse')
HARD_ARMS = ('real','later_only','repeat_later')
SEEDS = ('1','2','3')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def require(value, message):
    if not value:
        raise ValueError(message)


def unique(data):
    result = {}
    for row in data:
        require(row['id'] not in result, 'Duplicate source ID: ' + row['id'])
        result[row['id']] = row
    return result


def audit(repo, base_url):
    artifact = repo/'artifacts'
    output = artifact/'eo_evidence_search_v5_20260925'
    e3 = artifact/'e3_pair_dependence_v1_20260925'
    audit_path = artifact/'e3_independent_audit_20260925.json'
    snapshot_path = output/'reader_cases.json'
    inputs = {}
    def track(path):
        inputs[str(path)] = sha(path)
        return inputs[str(path)]
    for path in [snapshot_path, e3/'manifest.json', e3/'scores.json', e3/'status.json',
                 e3/'items.jsonl', e3/'saved_e2_real.json', audit_path]:
        track(path)
    snapshot, manifest, scores = read(snapshot_path), read(e3/'manifest.json'), read(e3/'scores.json')
    require(read(e3/'status.json')['status'] == 'completed' and scores['valid'] is True,
            'E3 is not completed and valid')
    require(read(audit_path)['consistent'] is True, 'Independent scientific audit did not pass')
    require(scores['manifest_sha256'] == track(e3/'manifest.json') and
            manifest['items_sha256'] == track(e3/'items.jsonl') and
            manifest['saved_e2_real_sha256'] == track(e3/'saved_e2_real.json'), 'Frozen E3 input SHA mismatch')
    require(snapshot['schema_version'] == 'eo_reader_cases_v1' and len(snapshot['cases']) == 914,
            'Unexpected v5 case snapshot schema/population')
    items = unique(rows(e3/'items.jsonl'))
    require(len(items) == manifest['n_items'] == 209, 'Expected 209 E3 source items')
    selected = {key:item for key,item in items.items() if item['phen']=='flood' and item['kind'] in ('pos','hard_neg')}
    require(Counter(it['kind'] for it in selected.values()) == {'pos':57,'hard_neg':51}, 'Expected 57 pos + 51 hard cases')
    require(len({it['tile'] for it in selected.values()}) == 108, 'Source selected tiles are not unique')
    answers, answer_hashes = {}, {}
    answer_rows = 0
    for seed in SEEDS:
        for arm in ARMS:
            path = e3/f'answers_seed{seed}_{arm}.jsonl'
            answer_hashes[arm+'_seed'+seed] = track(path)
            records = unique(rows(path))
            expected = {key for key,item in items.items() if arm in item['allowed_arms']}
            require(set(records) == expected, f'E3 answer ID coverage: {seed}/{arm}')
            for key,row in records.items():
                item = items[key]
                require(row['seed']==int(seed) and row['arm']==arm and
                        all(row[k]==item[k] for k in ('tile','cluster','phen','kind','pair_key')) and
                        row['source_gold']==item['answer'] and row['transformed_gold'] is None and
                        row['parsed'] in ('yes','no',None), f'E3 source row mismatch: {key}/{seed}/{arm}')
            answer_rows += len(records)
            answers[(seed,arm)] = records
    require(answer_rows == manifest['n_generations'] == 3777, 'Expected 3777 source answer rows')
    e2 = {}
    e2_ids = None
    for role in ('reader','blind'):
        for seed in SEEDS:
            path = artifact/f'e2_multi_reader_v0/{role}_seed{seed}/answers_real_all.jsonl'
            track(path)
            data = unique(rows(path))
            require(len(data)==1755, 'Expected 1755 original E2 source IDs')
            if e2_ids is None:e2_ids=set(data)
            require(set(data)==e2_ids, 'Original E2 source ID sets differ')
            e2[(role,seed)] = data
    controls = {}
    e2_case_values = 0
    for tile,case in snapshot['cases'].items():
        key=case['question_id']
        require(case['tile']==tile and case['run_id']=='E2' and case['source_role']=='historical_model_output',
                'Base E2 identity/role mismatch: '+tile)
        for role in ('reader','blind'):
            for seed in SEEDS:
                source=e2[(role,seed)][key]
                source_path=artifact/f'e2_multi_reader_v0/{role}_seed{seed}/answers_real_all.jsonl'
                require(source['tile']==tile and source['fold']=='test' and source['phen']=='flood'
                        and source['kind'] in ('pos','hard_neg') and source['emb_item']==key
                        and source['emb_gold']==source['text_gold']==case['reference_label']
                        and case[role+'_predictions'][seed]==source['parsed']
                        and case['source_sha256'][role+'_seed'+seed]==inputs[str(source_path)],
                        'Original E2 case or source hash mismatch: '+key+'/'+role+'/'+seed)
                e2_case_values+=1
        if 'e3_control' in case:controls[tile]=case
    require(set(controls)=={it['tile'] for it in selected.values()}, 'Linked tile set differs from source post-question selection')
    base_hashes={'manifest':track(e3/'manifest.json'),'scores':track(e3/'scores.json'),
                 'independent_audit':track(audit_path),'items':track(e3/'items.jsonl')}
    compared=0
    for tile,case in controls.items():
        key=case['question_id'];item=selected[key];control=case['e3_control']
        allowed=ARMS if item['kind']=='pos' else HARD_ARMS
        require(case['dates']==item['dates'] and case['slots']==item['slots']==['pre_2','post'],
                'Question/date/slot mismatch: '+tile)
        require(control['run_id']=='E3-PD-v1' and control['source_role']=='historical_model_output'
                and control['question_id']==key and control['source_gold']==item['answer']==case['reference_label'],
                'E3 control identity/label mismatch: '+tile)
        require(set(control['predictions'])==set(allowed), 'E3 control arm coverage mismatch: '+tile)
        expected_hashes={**base_hashes, **{arm+'_seed'+seed:answer_hashes[arm+'_seed'+seed] for arm in allowed for seed in SEEDS}}
        require(control['source_sha256']==expected_hashes,'E3 exact source hash mapping mismatch: '+tile)
        for arm in allowed:
            require(set(control['predictions'][arm])==set(SEEDS),'Seed coverage mismatch: '+tile+'/'+arm)
            for seed in SEEDS:
                require(control['predictions'][arm][seed]==answers[(seed,arm)][key]['parsed'],
                        'E3 displayed value differs from source: '+tile+'/'+seed+'/'+arm)
                compared+=1
        for seed in SEEDS:
            require(control['predictions']['real'][seed]==case['reader_predictions'][seed]
                    ==e2[('reader',seed)][key]['parsed'], 'E2/E3 real mismatch: '+tile+'/'+seed)
    provenance=snapshot['e3_control_provenance']
    require(provenance['linked_cases']==108 and provenance['question_ids']==sorted(selected)
            and provenance['source_role']=='historical_model_output'
            and provenance['audit_sha256']==track(audit_path), 'Top-level E3 linkage provenance mismatch')
    require('ks_05229' in snapshot['cases'] and 'e3_control' not in snapshot['cases']['ks_05229'], 'E2-only exemplar changed')
    require('ks_00276' not in snapshot['cases'], 'Training tile received a saved case')
    parsed=urllib.parse.urlsplit(base_url)
    require(parsed.scheme=='http' and parsed.hostname in ('127.0.0.1','localhost'), 'Only local HTTP readback is permitted')
    pos=sorted(it['tile'] for it in selected.values() if it['kind']=='pos')[0]
    hard=sorted(it['tile'] for it in selected.values() if it['kind']=='hard_neg')[0]
    http=[]
    for kind,tile,expected_arms in [('pos',pos,7),('hard_neg',hard,3),('E2_only','ks_05229',0),('train','ks_00276',0)]:
        url=base_url.rstrip('/')+'/api/reader-case?'+urllib.parse.urlencode({'tile':tile})
        with urllib.request.urlopen(url,timeout=10) as response:
            status=response.status; data=json.load(response)
        require(status==200 and data['tile']==tile,'HTTP status/tile mismatch: '+tile)
        require(data['checked_at']==snapshot['checked_at'], 'HTTP is not serving the verified v5 snapshot')
        require(data['available'] is (kind!='train'),'HTTP availability mismatch: '+tile)
        if kind!='train':
            require(data['case']==snapshot['cases'][tile], 'HTTP case is not byte-equivalent JSON to v5 snapshot: '+tile)
            require(len(data['case'].get('e3_control',{}).get('predictions',{}))==expected_arms, 'HTTP arm count mismatch: '+tile)
        else:require(data.get('case') is None and data['reason']=='no_saved_case','Training tile did not remain unknown')
        http.append({'kind':kind,'tile':tile,'status':status,'available':data['available'],
                     'e3_arm_count':expected_arms,'exact_case_match':kind!='train','reason':data.get('reason')})
    for path,digest in inputs.items():
        require(sha(path)==digest,'Source changed during audit: '+path)
    return {'schema':'eo-v5-e3-case-readback-audit-v0','checked_at':datetime.now(timezone.utc).isoformat(),
            'valid':True,'discrepancies':[], 'snapshot':str(snapshot_path),
            'case_count':914,'e3_control_count':108,'e3_kind_counts':{'pos':57,'hard_neg':51},
            'e3_source_items_checked':209,'e3_source_answer_rows_checked':answer_rows,
            'e3_linked_prediction_values_checked':compared,'e2_original_case_values_checked':e2_case_values,
            'e2_real_reproduction_values_checked':108*3,'e2_source_files_checked':6,
            'http_readback':http,'source_sha256':inputs,'script_sha256':sha(__file__),
            'scope':'Source/readback equality only; no new model inference, label generation, scoring, or scientific threshold changes.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project'))
    p.add_argument('--base-url',default='http://127.0.0.1:8774')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); require(not args.out.exists(),'Report already exists')
    try:result=audit(args.repo.resolve(),args.base_url)
    except Exception as exc:
        result={'schema':'eo-v5-e3-case-readback-audit-v0','checked_at':datetime.now(timezone.utc).isoformat(),
                'valid':False,'discrepancies':[{'type':type(exc).__name__,'message':str(exc)}],
                'script_sha256':sha(__file__)}
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256',)},ensure_ascii=False))
    return 0 if result['valid'] else 1


if __name__=='__main__':raise SystemExit(main())
