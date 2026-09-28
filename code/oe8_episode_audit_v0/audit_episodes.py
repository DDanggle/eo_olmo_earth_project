#!/usr/bin/env python3
"""Independent audit of exported OE8 catalogs, masks and available local samples.

Does not import the episode builder or run a model. Full source/raw80 validation
is a separate receipt; only locally exported sample labels are reread here.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import traceback
import unittest
import numpy as np

KS=(1,2,4,8)
PROMPT='Find regions matching the positive examples and exclude the counterexamples. Return a mask reference and the observation IDs you used.'
VISIBLE={'fixed generic prompt','acquired query observation pixels','acquired observation dates and validity',
         'selected support images and object masks','support observation dates and validity'}
FORBIDDEN={'class_id','class_ids','target_class','counter_class','query_label_npz','label_path',
           'semantic','instances','target_mask','target_pixels','counter_pixels','target_present','counter_present'}

def need(ok,message):
    if not ok: raise ValueError(message)

def digest(text): return hashlib.sha256(text.encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def obj(path): return json.loads(Path(path).read_text())

def child(root,ref):
    path=(root/ref).resolve()
    need(root.resolve() in path.parents,'path escapes bundle')
    return path

def no_gold(value):
    if isinstance(value,dict):
        need(not (set(value)&FORBIDDEN),'public episode includes class or query gold')
        for item in value.values(): no_gold(item)
    elif isinstance(value,list):
        for item in value: no_gold(item)

def role_map(manifest):
    need(len(manifest)==80 and len({r['patch_id'] for r in manifest})==80,'manifest count/duplicates')
    result={}
    for parent in ('t32ulu','t31tfj'):
        group=[r for r in manifest if r['parent_tile']==parent]
        need(len(group)==32 and all(r['role']=='train' for r in group),'source parent count')
        group.sort(key=lambda r:digest('oe8-bank-v0:'+r['patch_id']))
        result.update({r['patch_id']:('source_bank' if n<8 else 'train_pool') for n,r in enumerate(group)})
    dev=[r for r in manifest if r['parent_tile']=='t31tfm']
    need(len(dev)==16 and all(r['role']=='development' for r in dev),'dev parent count')
    result.update({r['patch_id']:'dev_query' for r in dev})
    for r in manifest: need(result[r['patch_id']]==r['training_partition'],'partition mismatch')
    return result

def select(objects,cls,excluded,namespace):
    choices=[o for o in objects if o['class_id']==cls and o['patch_id']!=excluded]
    choices.sort(key=lambda o:digest(namespace+':'+o['object_key']))
    count=Counter(); result=[]
    for o in choices:
        if count[o['patch_id']]<2:
            result.append(o); count[o['patch_id']]+=1
        if len(result)==8: break
    return result

def audit(bundle):
    ep_root=bundle/'episodes'
    result={'status':'auditing','model_executed':False,'runtime_inference_guard_verified':False,
            'all_raw80_reread_in_this_audit':False,'cross_patch_parcel_disjointness_verified':False}
    export=obj(bundle/'export_manifest.json')
    need(len({r['path'] for r in export['files']})==len(export['files']),'export duplicate paths')
    for r in export['files']:
        p=child(bundle,r['path'])
        need(p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'export file hash mismatch: '+r['path'])
    result['export_file_hashes_verified']=len(export['files'])
    manifest=rows(bundle/'manifest.jsonl'); rm={r['patch_id']:r for r in manifest}
    roles=role_map(manifest)
    need(roles==obj(bundle/'partition.json'),'partition file differs')
    role_rows=rows(ep_root/'roles.jsonl')
    need(len(role_rows)==80 and len({r['patch_id'] for r in role_rows})==80,'roles count')
    for r in role_rows:
        pid=r['patch_id']
        need(r['episode_role']==roles[pid] and r['parent_tile']==rm[pid]['parent_tile'] and
             r['original_role']==rm[pid]['role'] and r['bank_partition_hash']==digest('oe8-bank-v0:'+pid),'roles artifact mismatch')
    contract=obj(ep_root/'episode_contract.json')
    need(contract['prepared_manifest_sha256']==sha(bundle/'manifest.jsonl'),'prepared manifest identity')
    need(contract['quality_policy_sha256']==sha(bundle/'quality_policy.json'),'quality policy identity')
    need(set(contract['model_visible_allowlist'])==VISIBLE,'model metadata whitelist unexpectedly changed')
    hidden=set(contract['retrieval_scoring_only_never_model_prompt'])
    need({'episode_id','base_id','pair_id','patch IDs','parent tile IDs','object_key','file paths','hashes','class mapping',
          'all-candidate aggregate eligibility flags'}<=hidden,'routing identities not excluded from model prompt')
    need(contract['bank_labels_used_for_encoder_training'] is False and
         contract['parcel_disjoint_verified'] is False and contract['scientific_training_ready'] is False,
         'unsupported training/geometry readiness claim')
    need(contract['all_eight_candidate_arrays_in_npz_require_loader_acquisition_guard'] is True and
         contract['unacquired_query_pixels_features_and_per_date_quality_forbidden'] is True,'acquisition guard contract absent')
    sources=rows(ep_root/'source_objects.jsonl'); sm={o['object_key']:o for o in sources}
    need(len(sm)==len(sources),'duplicate source objects')
    masks={}; local_labels={}; local_inputs={}
    for pid in export['local_sample_packets']:
        lp=bundle/'sample_packets'/rm[pid]['label_path']; ip=bundle/'sample_packets'/rm[pid]['npz_path']
        need(sha(lp)==rm[pid]['label_sha256'] and sha(ip)==rm[pid]['npz_sha256'],'sample packet hash mismatch')
        with np.load(lp,allow_pickle=False) as z: local_labels[pid]={k:z[k] for k in z.files}
        with np.load(ip,allow_pickle=False) as z: local_inputs[pid]={'observation_valid':z['observation_valid']}
    local_object_checks=0
    for o in sources:
        pid=o['patch_id']; role=roles[pid]
        need(role in ('train_pool','source_bank') and o['episode_role']==role,'source object belongs to dev')
        need(o['object_key']==f'{pid}:{o["instance_id"]}' and o['parent_tile']==rm[pid]['parent_tile'],'object identity')
        need(1<=o['class_id']<=18 and o['instance_id']>0,'object class/instance range')
        need(o['source_label_sha256']==rm[pid]['label_sha256'],'object source label hash')
        path=child(ep_root,o['mask_npz'])
        need(o['mask_npz']==f'support_masks/{pid}_{o["instance_id"]}.npz' and sha(path)==o['mask_sha256'],'mask file identity')
        with np.load(path,allow_pickle=False) as z:
            need(set(z.files)=={'mask'},'support mask packet leaks extra data')
            mask=z['mask']
        need(mask.shape==(128,128) and mask.dtype==np.bool_,'mask shape/type')
        count=int(mask.sum()); ys,xs=np.where(mask)
        need(count==o['class_pixel_count'] and count>=64,'mask class pixel count')
        need(o['instance_pixel_count']>=count and o['class_purity']>=.95 and
             abs(o['class_purity']-count/o['instance_pixel_count'])<1e-12,'purity mismatch')
        need(o['bbox_xyxy_exclusive']==[int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1],'mask bounding box')
        masks[o['object_key']]=mask
        if pid in local_labels:
            z=local_labels[pid]; body=z['instances']==o['instance_id']
            expected=body & (z['semantic']==o['class_id']) & z['label_valid']
            need(np.array_equal(mask,expected) and int(body.sum())==o['instance_pixel_count'],'local source mask/instance mismatch')
            need(local_inputs[pid]['observation_valid'][:,body].all(),'support body missing in local sample')
            border=bool(body[0].any() or body[-1].any() or body[:,0].any() or body[:,-1].any())
            need(o['touches_image_border']==border,'local object border flag mismatch')
            local_object_checks+=1
    pools={role:[o for o in sources if o['episode_role']==role] for role in ('train_pool','source_bank')}
    pairs=obj(ep_root/'pair_catalog.json'); declared=pairs['selected_classes']
    patches=defaultdict(set); counts=Counter()
    for o in pools['train_pool']: patches[o['class_id']].add(o['patch_id']); counts[o['class_id']]+=1
    expected=sorted([c for c in range(1,19) if len(patches[c])>=4],key=lambda c:(-len(patches[c]),c))[:4]
    need(declared==expected==contract['selected_classes'],'classes not selected from train-only ranking')
    expected_coverage={str(c):{'eligible_objects':counts[c],'eligible_patches':len(patches[c])} for c in range(1,19)}
    need(expected_coverage==pairs['training_class_coverage'],'class coverage mismatch')
    need(pairs['source_bank_used_for_class_selection'] is False and pairs['development_labels_used_for_class_selection'] is False,'class-selection leakage declaration')
    expected_pairs={digest(f'oe8-directed-pair-v0:{a}:{b}')[:20]:(a,b) for a,b in itertools.permutations(expected,2)}
    need(len(pairs['directed_pairs'])==len(expected_pairs) and
         {p['pair_id']:(p['target_class'],p['counter_class']) for p in pairs['directed_pairs']}==expected_pairs,'pair catalog mismatch')
    coverage=obj(ep_root/'coverage.json'); frozen_hashes=obj(ep_root/'public_catalog_hashes_before_scoring.json')
    summaries={}; sample_scoring_checks=0
    for split in ('train','development'):
        episode_path=ep_root/f'episodes_{split}.jsonl'
        need(sha(episode_path)==frozen_hashes[split]==contract['public_catalog_sha256'][split],'public catalog changed after freeze')
        public=rows(episode_path); scoring=rows(ep_root/'scoring'/f'scoring_{split}.jsonl')
        by_id={e['episode_id']:e for e in public}; scores={s['episode_id']:s for s in scoring}
        need(len(by_id)==len(public)==len(scores)==len(scoring) and set(by_id)==set(scores),'episode/scoring duplicates or mismatch')
        query_role='train_pool' if split=='train' else 'dev_query'
        support_role='train_pool' if split=='train' else 'source_bank'
        query_ids=sorted(pid for pid,role in roles.items() if role==query_role)
        expected_ids=set(); cohort=[]; shortages=[]; by_k=Counter(); by_pair={}; positives=Counter(); complete_positives=Counter()
        for pid in query_ids:
            namespace=f'oe8-support-v0:{split}:{pid if split=="train" else "shared"}'
            seq={c:select(pools[support_role],c,pid if split=='train' else None,namespace+f':class={c}') for c in expected}
            for pair_id,(target,counter) in expected_pairs.items():
                base=digest(f'oe8-episode-base-v0:{split}:{pid}:{pair_id}')[:24]
                capacity=min(len(seq[target]),len(seq[counter])); available=[k for k in KS if k<=capacity]
                entry={'query_patch_id':pid,'pair_id':pair_id,'base_id':base,'available_ks':available,'k8_auc_eligible':capacity>=8}
                cohort.append(entry)
                if capacity<8: shortages.append(dict(entry,target_support_capacity=len(seq[target]),counter_support_capacity=len(seq[counter])))
                by_pair[pair_id]={'target_class':target,'counter_class':counter,'available_ks':available}
                previous=[]
                for k in available:
                    eid=f'{base}:k{k}'; expected_ids.add(eid); need(eid in by_id,'expected episode missing')
                    e=by_id[eid]; s=scores[eid]; row=rm[pid]
                    no_gold(e)
                    need(e['episode_id']==eid and e['base_id']==base and e['pair_id']==pair_id and e['split']==split,'episode identity')
                    need(e['query_patch_id']==pid and e['query_parent_tile']==row['parent_tile'],'query role/parent')
                    need(e['query_input_npz']==row['npz_path'] and e['query_input_sha256']==row['npz_sha256'],'query input identity')
                    need(e['prompt']==PROMPT and e['definition_mode']=='unnamed_exemplar_target' and e['query_label_supplied'] is False,'prompt/query-gold contract')
                    obs=[{'observation_id':f'{pid}:obs:{i}','date_yyyymmdd':d} for i,d in enumerate(row['selected_dates'])]
                    need(e['query_observations']==obs and e['initial_observation_ids']==[f'{pid}:obs:2',f'{pid}:obs:5'],'query observation policy')
                    for field in ('strict_no_missing_input_eligible','supervised_training_allowed','clean_training_eligible'):
                        need(e[field]==row[field],'query eligibility changed cohort')
                    need(e['missing_aware_adapter_required']==(not row['strict_no_missing_input_eligible']) and
                         e['eligibility_flags_are_not_selector_features'] is True,'hidden quality metadata policy')
                    need(e['k_pairs']==k and len(e['support_pairs'])==k,'K pair count')
                    keys=[]
                    for index,pair in enumerate(e['support_pairs']):
                        need(set(pair)=={'positive','counterexample'},'support polarity keys')
                        for polarity,cls in [('positive',target),('counterexample',counter)]:
                            support=pair[polarity]; source=seq[cls][index]; spid=source['patch_id']; sr=rm[spid]
                            need(support['object_key']==source['object_key'] and roles[spid]==support_role and spid!=pid,'support sequence/role/query overlap')
                            for field in ('patch_id','parent_tile','mask_npz','mask_sha256'):
                                need(support[field]==source[field],'support source metadata mismatch')
                            need(support['input_npz']==sr['npz_path'] and support['input_sha256']==sr['npz_sha256'],'support input identity')
                            need(support['observation_ids']==[f'{spid}:obs:{i}' for i in range(8)] and
                                 support['dates_yyyymmdd']==sr['selected_dates'],'support date mapping')
                            keys.append(support['object_key'])
                    need(len(keys)==len(set(keys)),'duplicate support object')
                    need(e['support_pairs'][:len(previous)]==previous,'K prefix inconsistency')
                    previous=e['support_pairs']
                    for polarity in ('positive','counterexample'):
                        need(max(Counter(p[polarity]['patch_id'] for p in previous).values())<=2,'patch/class cap exceeded')
                    need(s['base_id']==base and s['pair_id']==pair_id and s['query_patch_id']==pid and
                         s['target_class']==target and s['counter_class']==counter and s['k_pairs']==k and
                         s['k8_auc_cohort']==(capacity>=8),'scoring mapping mismatch')
                    need(s['query_label_npz']==row['label_path'] and
                         s['query_label_sha256']==s['expected_query_label_sha256']==row['label_sha256'],'scoring label identity')
                    tc,cc=row['class_pixel_counts'][target],row['class_pixel_counts'][counter]
                    need(s['target_pixels']==tc and s['counter_pixels']==cc and s['target_present']==(tc>0) and
                         s['counter_present']==(cc>0) and s['label_valid_pixels']==16384-row['class_pixel_counts'][19],
                         'scoring counts differ from preparation histograms')
                    need(s['all_eight_observation_valid_filter_applied_to_query_gold'] is False and
                         s['observability_not_inferred_from_labels'] is True,'query scoring filtered by missingness')
                    if pid in local_labels:
                        z=local_labels[pid]
                        need(tc==int(((z['semantic']==target)&z['label_valid']).sum()) and
                             cc==int(((z['semantic']==counter)&z['label_valid']).sum()),'local scoring mask count mismatch')
                        sample_scoring_checks+=1
                    by_k[k]+=1; positives[(k,tc>0)]+=1
                    if capacity>=8: complete_positives[(k,tc>0)]+=1
        need(set(by_id)==expected_ids,'extra catalog episodes or changed query/class population')
        cov=coverage[split]
        key=lambda r:r['base_id']
        need(sorted(cov['query_pair_cohort'],key=key)==sorted(cohort,key=key),'available-K population mismatch')
        need(sorted(cov['support_shortages'],key=key)==sorted(shortages,key=key),'shortage population mismatch')
        counts_k={str(k):by_k[k] for k in KS}; complete=sum(c['k8_auc_eligible'] for c in cohort)
        need(cov['episodes_by_k']==counts_k and cov['episodes']==len(public) and
             cov['query_candidates']==len(query_ids) and cov['k8_auc_base_count']==complete,'coverage aggregate mismatch')
        need(contract['scoring_counts'][split]=={
             'target_present_episodes':sum(positives[(k,True)] for k in KS),
             'target_absent_episodes':sum(positives[(k,False)] for k in KS)},'scoring contract aggregate mismatch')
        summaries[split]={'episodes':len(public),'episodes_by_k':counts_k,'query_patches':len(query_ids),
            'all_query_pair_bases':len(cohort),'complete_k1_2_4_8_bases':complete,
            'complete_fraction':complete/len(cohort) if cohort else None,
            'target_present_by_k':{str(k):positives[(k,True)] for k in KS},
            'target_absent_by_k':{str(k):positives[(k,False)] for k in KS},
            'complete_cohort_target_present_by_k':{str(k):complete_positives[(k,True)] for k in KS},
            'complete_cohort_target_absent_by_k':{str(k):complete_positives[(k,False)] for k in KS},
            'pair_support_availability':list(by_pair.values())}
    separate=obj(bundle/'source_verification_v0.json')
    need(separate['status']=='passed_independent_source_and_input_checks_not_model_performance' and
         separate['verified_case_count']==80 and separate['source_catalog_lineage_verified'] is True,
         'separate full raw verification receipt not passed')
    bank_caps={}
    for cls in expected:
        bank_objects=[o for o in pools['source_bank'] if o['class_id']==cls]
        bank_caps[str(cls)]={'objects':len(bank_objects),'patches':len({o['patch_id'] for o in bank_objects}),
                            'capped_capacity':len(select(bank_objects,cls,None,
                                f'oe8-support-v0:development:shared:class={cls}'))}
    result.update(status='passed_catalog_and_exported_mask_audit_not_model_or_runtime_guard',
        selected_classes=expected,source_object_masks_checked=len(sources),
        source_objects_by_partition={k:len(v) for k,v in pools.items()},
        local_raw_and_label_packet_ids=export['local_sample_packets'],local_original_support_mask_checks=local_object_checks,
        local_original_scoring_rows_checked=sample_scoring_checks,split_results=summaries,
        separate_full_raw80_receipt_sha256=sha(bundle/'source_verification_v0.json'),
        model_visible_whitelist_contract_checked=True,scientific_training_ready=False,
        development_bank_capacity_by_class=bank_caps,
        curve_reporting_requirement='Report all-pair K1/2/4 and fixed complete-cohort K1/2/4/8 separately; do not splice populations',
        limitations=['Runtime inference acquisition/gold/ID stripping guard not implemented or tested here',
                     'Full raw80/source arrays were verified by the separate receipt, not reread locally',
                     'Support source masks rechecked against original labels only for exported sample packets',
                     'Cross-patch parcel/footprint disjointness and clouds remain unverified',
                     'K8 complete-cohort curve must use the same query/class cohort at all K; missing K8 is not zero'])
    return result

class AuditFixtures(unittest.TestCase):
    def test_recursive_public_gold_detection(self):
        no_gold({'support_pairs':[{'positive':{'object_key':'1:2'}}]})
        with self.assertRaises(ValueError): no_gold({'support_pairs':[{'positive':{'class_id':4}}]})
        with self.assertRaises(ValueError): no_gold({'query_label_npz':'labels/x.npz'})

    def test_patch_cap_reduces_eight_objects_to_seven(self):
        objects=[]
        for patch,count in [('a',3),('b',2),('c',1),('d',1),('e',1)]:
            for i in range(count): objects.append({'patch_id':patch,'class_id':4,'object_key':f'{patch}:{i}'})
        selected=select(objects,4,None,'fixture')
        self.assertEqual(len(selected),7)
        self.assertLessEqual(max(Counter(o['patch_id'] for o in selected).values()),2)
        self.assertEqual(select(objects,4,'a','fixture'),[o for o in selected if o['patch_id']!='a'])

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--bundle',type=Path); p.add_argument('--out',type=Path)
    p.add_argument('--self-test',action='store_true'); a=p.parse_args()
    if a.self_test:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(AuditFixtures)
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    need(a.bundle is not None and a.out is not None,'required --bundle and --out')
    try: result=audit(a.bundle); code=0
    except Exception as e: result={'status':'failed_episode_audit','error':str(e),'traceback':traceback.format_exc()}; code=1
    result['script_sha256']=sha(__file__)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return code

if __name__=='__main__': raise SystemExit(main())
