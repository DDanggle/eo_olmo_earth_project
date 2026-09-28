#!/usr/bin/env python3
"""Independent E4 source/readback and E2/E3 preservation checks; no inference/scoring."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ARMS=('real','delta_only','delta_sign_flip','delta_feature_permute')
SEEDS=('1','2','3')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]
def require(ok,message):
    if not ok:raise ValueError(message)


def unique(data):
    by_id={}
    for row in data:
        require(row['id'] not in by_id,'Duplicate source ID: '+row['id'])
        by_id[row['id']]=row
    return by_id


def audit(repo,base_url):
    artifacts=repo/'artifacts';e4=artifacts/'e4_delta_probe_v0_20260925'
    audit_path=artifacts/'e4_independent_audit_20260925.json'
    v5_path=artifacts/'eo_evidence_search_v5_20260925/reader_cases.json'
    v6_path=artifacts/'eo_evidence_search_v6_20260925/reader_cases.json'
    sources={}
    def track(path):
        sources[str(path)]=sha(path)
        return sources[str(path)]
    for path in (v5_path,v6_path,audit_path,e4/'status.json',e4/'manifest.json',e4/'scores.json',
                 e4/'items.jsonl',e4/'references.json',e4/'prereg.json'):
        track(path)
    old,new=read(v5_path),read(v6_path)
    require(new['schema_version']=='eo_reader_cases_v1' and len(new['cases'])==914,'Unexpected v6 snapshot population')
    # Compare the entire historical payload, not just displayed prediction arrays.
    reduced={k:v for k,v in new.items()if k not in ('checked_at','e4_control_provenance')}
    reduced['cases']={tile:{k:v for k,v in case.items()if k!='e4_control'}for tile,case in new['cases'].items()}
    baseline={k:v for k,v in old.items()if k!='checked_at'}
    require(reduced==baseline,'v6 altered prior E2/E3/reference/provenance payload')
    status,manifest,scores=read(e4/'status.json'),read(e4/'manifest.json'),read(e4/'scores.json')
    require(status['status']=='completed' and scores['valid'] is True,'E4 not completed/valid')
    require(read(audit_path)['consistent'] is True,'Independent E4 audit is not consistent')
    require(scores['manifest_sha256']==track(e4/'manifest.json'),'E4 score manifest mismatch')
    for filename,key in [('items.jsonl','items_sha256'),('references.json','references_sha256'),('prereg.json','prereg_sha256')]:
        require(track(e4/filename)==manifest[key],'Frozen E4 input changed: '+filename)
    require(manifest['items_sha256']==sha(artifacts/'e3_pair_dependence_v1_20260925/items.jsonl'),
            'E4 item bytes differ from E3 parent')
    track(artifacts/'e3_pair_dependence_v1_20260925/items.jsonl')
    items=unique(rows(e4/'items.jsonl'))
    require(len(items)==manifest['n_items']==209,'Expected 209 E4 items')
    selected={key:item for key,item in items.items()if item['phen']=='flood' and item['kind'] in ('pos','hard_neg')}
    require(Counter(item['kind']for item in selected.values())=={'pos':57,'hard_neg':51},'Wrong post-question source population')
    require(len({item['tile']for item in selected.values()})==108,'Source post-question tiles are not unique')
    references=read(e4/'references.json')
    require(set(references)=={'e2_real','e3_real'},'Missing E2/E3 frozen real reference')
    for reference in references.values():
        require(set(reference)==set(SEEDS) and all(set(mapping)==set(items)for mapping in reference.values()),'Frozen real-reference coverage mismatch')
    answers,answer_hashes={},{}
    source_count=0
    for seed in SEEDS:
        for arm in ARMS:
            path=e4/f'answers_seed{seed}_{arm}.jsonl'
            answer_hashes[arm+'_seed'+seed]=track(path)
            data=unique(rows(path))
            require(set(data)==set(items),'E4 source answer coverage mismatch: '+seed+'/'+arm)
            for key,row in data.items():
                item=items[key]
                require(row['seed']==int(seed) and row['arm']==arm and
                        all(row[k]==item[k]for k in ('tile','cluster','phen','kind','pair_key')) and
                        row['source_gold']==item['answer'] and row['transformed_gold'] is None and
                        row['parsed'] in ('yes','no',None),'E4 answer metadata/value mismatch: '+key+'/'+seed+'/'+arm)
            source_count+=len(data);answers[(seed,arm)]=data
    require(source_count==manifest['n_generations']==2508,'Expected 2508 source answer rows')
    controls={tile:case for tile,case in new['cases'].items()if 'e4_control' in case}
    require(set(controls)=={item['tile']for item in selected.values()},'E4 displayed tile set differs from source selection')
    hashes={'manifest':track(e4/'manifest.json'),'scores':track(e4/'scores.json'),
            'independent_audit':track(audit_path),'items':track(e4/'items.jsonl'),**answer_hashes}
    compared,real_matches=0,0
    for tile,case in controls.items():
        key=case['question_id'];item=selected[key];control=case['e4_control']
        require(case['tile']==tile==item['tile'] and case['dates']==item['dates'] and
                case['slots']==item['slots']==['pre_2','post'],'E4 source question/date/slots mismatch: '+tile)
        require(control['run_id']=='E4-D-v0' and control['source_role']=='historical_model_output' and
                control['question_id']==key and control['source_gold']==item['answer']==case['reference_label'],
                'E4 displayed identity/source label mismatch: '+tile)
        require(set(control['predictions'])==set(ARMS),'E4 displayed arm coverage mismatch: '+tile)
        require(control['source_sha256']==hashes,'E4 source hash map mismatch: '+tile)
        for arm in ARMS:
            require(set(control['predictions'][arm])==set(SEEDS),'E4 displayed seed coverage mismatch')
            for seed in SEEDS:
                require(control['predictions'][arm][seed]==answers[(seed,arm)][key]['parsed'],
                        'E4 displayed parsed value differs from source: '+tile+'/'+seed+'/'+arm)
                compared+=1
        for seed in SEEDS:
            value=control['predictions']['real'][seed]
            require(value==case['reader_predictions'][seed]==case['e3_control']['predictions']['real'][seed]
                    ==references['e2_real'][seed][key]==references['e3_real'][seed][key],
                    'Displayed E4 real differs from preserved E2/E3 real: '+tile+'/'+seed)
            real_matches+=1
    require(compared==1296 and real_matches==324,'Displayed prediction coverage mismatch')
    provenance=new['e4_control_provenance']
    require(provenance['linked_cases']==108 and provenance['question_ids']==sorted(selected) and
            provenance['source_role']=='historical_model_output' and provenance['audit_sha256']==track(audit_path),
            'E4 top-level provenance mismatch')
    require('ks_05229'in new['cases'] and all(k not in new['cases']['ks_05229']for k in ('e3_control','e4_control')),
            'E2-only exemplar no longer E2-only')
    require('ks_00276'not in new['cases'],'Training tile received saved model outputs')
    urlparts=urllib.parse.urlsplit(base_url)
    require(urlparts.scheme=='http' and urlparts.hostname in ('127.0.0.1','localhost'),'Only local HTTP readback permitted')
    pos=sorted(item['tile']for item in selected.values()if item['kind']=='pos')[0]
    hard=sorted(item['tile']for item in selected.values()if item['kind']=='hard_neg')[0]
    http=[]
    for kind,tile,e3_count,e4_count in [('pos',pos,7,4),('hard_neg',hard,3,4),('E2_only','ks_05229',0,0),('train','ks_00276',0,0)]:
        url=base_url.rstrip('/')+'/api/reader-case?'+urllib.parse.urlencode({'tile':tile})
        with urllib.request.urlopen(url,timeout=10)as response:code=response.status;payload=json.load(response)
        require(code==200 and payload['tile']==tile and payload['checked_at']==new['checked_at'],
                'HTTP does not identify the verified v6 snapshot: '+tile)
        require(payload['available'] is (kind!='train'),'HTTP availability mismatch: '+tile)
        if kind!='train':
            require(payload['case']==new['cases'][tile],'HTTP case differs from v6 snapshot: '+tile)
            for field,count in [('e3_control',e3_count),('e4_control',e4_count)]:
                require(len(payload['case'].get(field,{}).get('predictions',{}))==count,'HTTP control coverage mismatch: '+tile+'/'+field)
        else:
            require(payload.get('case') is None and payload['reason']=='no_saved_case','Training tile should remain unknown')
        http.append({'kind':kind,'tile':tile,'status':code,'available':payload['available'],
                     'e3_arm_count':e3_count,'e4_arm_count':e4_count,'exact_case_match':kind!='train','reason':payload.get('reason')})
    for path,digest in sources.items():require(sha(path)==digest,'Input changed during verification: '+path)
    return {'schema':'eo-v6-e4-case-readback-audit-v0','checked_at':datetime.now(timezone.utc).isoformat(),
            'valid':True,'discrepancies':[],'case_count':914,'e4_control_count':108,
            'e4_kind_counts':{'pos':57,'hard_neg':51},'e4_source_items_checked':209,
            'e4_source_answer_rows_checked':source_count,'e4_linked_prediction_values_checked':compared,
            'e4_real_e2_e3_values_matched':real_matches,'v5_e2_e3_reference_and_provenance_unchanged':True,
            'http_readback':http,'source_sha256':sources,'script_sha256':sha(__file__),
            'scope':'Source/readback equality and payload preservation only; no new inference, scoring, labels, or interpretation threshold.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project'))
    p.add_argument('--base-url',default='http://127.0.0.1:8774');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();require(not args.out.exists(),'Refusing existing report')
    try:result=audit(args.repo.resolve(),args.base_url)
    except Exception as exc:result={'schema':'eo-v6-e4-case-readback-audit-v0','checked_at':datetime.now(timezone.utc).isoformat(),
        'valid':False,'discrepancies':[{'type':type(exc).__name__,'message':str(exc)}],'script_sha256':sha(__file__)}
    with args.out.open('x')as stream:json.dump(result,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='source_sha256'},ensure_ascii=False))
    return 0 if result['valid']else 1


if __name__=='__main__':raise SystemExit(main())
