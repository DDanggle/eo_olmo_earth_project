"""Metadata-only expansion feasibility and labelled post-hoc score decomposition.
No development gold is used for expansion candidate selection. Frozen scores,
cohort, training and gates remain unchanged. No new inference is performed.
"""
import argparse,hashlib,json,statistics
from collections import Counter,defaultdict
from pathlib import Path

CLASSES=['background','meadow','soft_winter_wheat','corn','winter_barley','winter_rapeseed','spring_barley','sunflower','grapevine','beet','winter_triticale','winter_durum_wheat','fruits_vegetables_flowers','potatoes','leguminous_fodder','soybeans','orchard','mixed_cereal','sorghum']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def macro(rows):
    query=defaultdict(list)
    for r in rows:
        if r['target_present']:query[(r['parent'],r['target_class'],r['query_patch_id'])].append(r['target_iou'])
    classes=defaultdict(list)
    for (parent,c,_),v in query.items():classes[parent,c].append(statistics.mean(v))
    parents=defaultdict(list)
    for (parent,_),v in classes.items():parents[parent].append(statistics.mean(v))
    return statistics.mean(statistics.mean(v) for v in parents.values())

def main():
    p=argparse.ArgumentParser();p.add_argument('--objects',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    objects=[json.loads(x) for x in a.objects.read_text().splitlines() if x.strip()]
    coverage=[]
    for c in range(1,19):
        row={'class_id':c,'name':CLASSES[c],'already_in_training':c in (1,2,3,4)}
        for role in ('train_pool','source_bank'):
            selected=[o for o in objects if o['class_id']==c and o['episode_role']==role]
            counts=Counter(o['patch_id'] for o in selected)
            row[role]={'eligible_objects':len(selected),'unique_patches':len(counts),'capped_objects_max2_per_patch':sum(min(n,2) for n in counts.values())}
        row['train_and_bank_k8_count_screen']=row['train_pool']['unique_patches']>=4 and row['source_bank']['capped_objects_max2_per_patch']>=8
        coverage.append(row)
    candidates=sorted([r for r in coverage if not r['already_in_training'] and r['train_and_bank_k8_count_screen']],key=lambda r:(-r['train_pool']['unique_patches'],r['class_id']))
    scores={};hashes={str(a.objects):sha(a.objects)}
    for arm in ('B0','B2'):
        path=a.bundle/f'training_v0/{arm}_270927_train/score_step_002304.json';hashes[str(path)]=sha(path);s=json.loads(path.read_text())
        result={}
        for label,keep in [('original_all',lambda r:True),('posthoc_target_without_wheat_counter_may_be_wheat',lambda r:r['target_class']!=2),('posthoc_neither_role_wheat',lambda r:r['target_class']!=2 and r['counter_class']!=2)]:
            curve={str(k):macro([r for r in s['per_episode_scoring_only'] if r['k']==k and keep(r)]) for k in (1,2,4,8)}
            ks=(1,2,4,8);auc=sum((curve[str(x)]+curve[str(y)])*(y-x)/2 for x,y in zip(ks,ks[1:]))/7
            result[label]={'curve':curve,'auc':auc,'posthoc':label!='original_all'}
        assert abs(result['original_all']['auc']-s['target_iou_auc'])<1e-12
        scores[arm]=result
    report={'schema_version':'oe10_class_expansion_audit_v1','scope':'Preparation only; no new task performance. Candidate screen uses train and source-bank support inventory only, never dev scores.',
      'source_hashes':hashes,'coverage':coverage,'candidates_ranked_by_training_patch_coverage':candidates,
      'count_screen_is_not_full_eligibility':True,'unverified':['new class development coverage','second development region','cloud observability','parcel independence','task-specific label quality','training query-exclusion support capacity'],
      'posthoc_first_seed_score_decomposition':scores,'frozen_primary_replaced':False,'new_gpu_runs':0,
      'interpretation':'Removing a class or replacing the evaluation set changes the estimand, not the model. Expanded evaluation must compare both arms on the same prospectively fixed cohort.'}
    with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'candidates':candidates,'score_decomposition':scores}))

if __name__=='__main__':main()
