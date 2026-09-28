#!/usr/bin/env python3
"""Freeze train-only engineering cases by labels and hashes, never model scores."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def need(ok, message):
    if not ok:raise ValueError(message)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();need(not a.out.exists(),'Output already exists')
    contract=json.loads((a.catalog/'episode_contract.json').read_text())
    need(contract['train_support_parent_policy']=='pooled','Expected common pooled pilot')
    need(contract['selected_classes']==[1,3,8,14],'Fixed four classes')
    mp=a.prepared/'manifest.jsonl';cp=a.catalog/'episodes_train.jsonl';sp=a.catalog/'scoring/scoring_train.jsonl'
    need(sha(mp)==contract['prepared_manifest_sha256'],'Manifest hash')
    need(sha(cp)==contract['public_catalog_sha256']['train'],'Catalog hash')
    need(sha(sp)==contract['scoring_file_sha256']['train'],'Scoring hash')
    manifest={r['patch_id']:r for r in rows(mp)}
    catalog={r['episode_id']:r for r in rows(cp)}
    scoring={r['episode_id']:r for r in rows(sp)}
    lookup={(r['query_patch_id'],r['pair_id'],r['k_pairs']):r for r in catalog.values()}
    k1=[s for eid,s in scoring.items() if catalog[eid]['k_pairs']==1]
    candidates={};missing=[]
    for cls in (1,3,8,14):
        for kind in ('target_present','confuser_only','neither'):
            eligible=[]
            for s in k1:
                if s['target_class']!=cls:continue
                e=catalog[s['episode_id']];m=manifest[e['query_patch_id']]
                need(m['training_partition']=='train_pool' and m['supervised_training_allowed'],'Train only')
                if s['label_valid_pixels']<int(.95*128*128):continue
                t,c=s['target_pixels'],s['counter_pixels']
                fits=(kind=='target_present' and t>=64) or (kind=='confuser_only' and t==0 and c>=64) or (kind=='neither' and t==c==0)
                if fits:eligible.append(s)
            candidates[(cls,kind)]=eligible
            if not eligible:missing.append({'target_class':cls,'kind':kind})
    selected=[];used_queries=set();parent_counts=Counter()
    for kind in ('target_present','confuser_only','neither'):
        for cls in (1,3,8,14):
            options=candidates[(cls,kind)]
            if not options:continue
            def rank(s):
                e=catalog[s['episode_id']]
                return (e['query_patch_id'] in used_queries,parent_counts[e['query_parent_tile']],
                        hashlib.sha256(('oe11-train-engineering-v0:'+s['episode_id']).encode()).hexdigest())
            s=min(options,key=rank);e=catalog[s['episode_id']];m=manifest[e['query_patch_id']]
            k8=lookup[(e['query_patch_id'],e['pair_id'],8)]
            need(sha(a.prepared/m['npz_path'])==m['npz_sha256'],'Actual input hash')
            need(sha(a.prepared/m['label_path'])==m['label_sha256'],'Actual train label hash')
            used_queries.add(e['query_patch_id']);parent_counts[e['query_parent_tile']]+=1
            selected.append({'episode_id':e['episode_id'],'k8_episode_id':k8['episode_id'],
                'query_patch_id':e['query_patch_id'],'query_parent':e['query_parent_tile'],
                'pair_id':e['pair_id'],'target_class':cls,'counter_class':s['counter_class'],
                'kind':kind,'target_pixels':s['target_pixels'],'counter_pixels':s['counter_pixels'],
                'label_valid_pixels':s['label_valid_pixels'],'eligible_count':len(options)})
    fit_episode_ids=[s['episode_id'] for s in selected]
    episode_ids=[eid for s in selected for eid in (s['episode_id'],s['k8_episode_id'])]
    random_order=random.Random(290929);order=[]
    while len(order)<48 and episode_ids:
        cycle=list(episode_ids);random_order.shuffle(cycle);order.extend(cycle)
    result={'schema':'oe11_training_pilot_cases_v1','revision_reason':'Include K8 optimizer updates: K1 softmax attention selection has zero gradient; no GPU/model results used for this revision','status':'ready' if not missing else 'blocked_missing_stratum',
        'split':'train','seed':290929,'total_steps':48,'split_step':24,'episode_ids':episode_ids,'order':order[:48],'fit_episode_ids':fit_episode_ids,
        'cases':selected,'missing_strata':missing,'unique_query_count':len(used_queries),
        'parent_case_counts':dict(parent_counts),'k8_separate_probe_required':False,'fit_diagnostic_k':1,
        'selection':'Four classes x present/confuser-only/neither; >=64 positive pixels and >=95% valid labels; prefer unused query then least represented parent then hash. Training labels only, no model score.',
        'input_hashes':{str(x):sha(x) for x in (mp,cp,sp,a.catalog/'episode_contract.json')},
        'script_sha256':sha(Path(__file__)),'query_positions':[2,5],'support_positions':list(range(8)),
        'development_or_final_labels_read':False,'source_labels_are_not_human_corrections':True,
        'training_metrics_not_generalization':True,'loss_improvement_is_diagnostic_not_pass_gate':True}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','unique_query_count','parent_case_counts','missing_strata','episode_ids')}))
    if missing:raise SystemExit(2)

if __name__=='__main__':main()
