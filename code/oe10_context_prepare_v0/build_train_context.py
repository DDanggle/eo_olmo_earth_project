"""Prepare named-support agronomy context; never load query gold or change OE10.

This is a separate, untrained input draft. Class names are EXTRA support-label
supervision, shared between name-only and knowledge conditions. Only render_text
returns model text; raw attachment records contain audit IDs and must not be fed
as model input. No image-conditioned explanation is generated here.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib,json
from pathlib import Path

PINNED_TRAIN_SHA='fadaf5edc3625a4a420babfe5b048e5752f0c824a8a44c005e5b13a6f6910293'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def support_annotation(e,objects,cards):
    forbidden={'target_class','counter_class','target_mask','target_present','target_pixels','class_pixel_counts','query_label_sha256'}
    assert not (set(e)&forbidden), 'query gold must not enter preparation'
    assert e['split']=='train' and e['supervised_training_allowed'] is True and e['query_label_supplied'] is False
    role_classes={};lineage={}
    for role in ('positive','counterexample'):
        classes=set();used=[]
        for pair in e['support_pairs']:
            s=pair[role];o=objects[s['object_key']]
            assert o['episode_role']=='train_pool' and o['patch_id']==s['patch_id']!=e['query_patch_id']
            assert o['mask_sha256']==s['mask_sha256'] and o['mask_npz']==s['mask_npz']
            classes.add(o['class_id']);used.append(o['object_key'])
        assert len(classes)==1
        c=classes.pop();assert c in cards
        role_classes[role]=c;lineage[role]=used
    assert role_classes['positive']!=role_classes['counterexample']
    acquired={o['observation_id']:o['date_yyyymmdd'] for o in e['query_observations']}
    dates=[acquired[x] for x in e['initial_observation_ids']]
    assert len(dates)==2
    return {'episode_id':e['episode_id'],'query_patch_id_audit_only':e['query_patch_id'],
            'k_pairs':e['k_pairs'],'split':'train','support_class_ids_audit_only':role_classes,
            'support_object_lineage_audit_only':lineage,
            'role_card_ids':{r:cards[c]['card_id'] for r,c in role_classes.items()},
            'acquired_dates':dates,'annotation_origin':'public_support_label_semantics_not_human_correction'}

def render_text(a,card_by_id,condition):
    assert condition in ('generic','names_only','matched_knowledge','swapped_knowledge_diagnostic')
    dates=[f'{str(d)[:4]}-{str(d)[4:6]}-{str(d)[6:]}' for d in a['acquired_dates']]
    base='Find the target using the positive and counterexample images. Region: France. Acquired query dates: '+', '.join(dates)+'. General background is not evidence that a target is present.'
    if condition=='generic':return base
    p=card_by_id[a['role_card_ids']['positive']];n=card_by_id[a['role_card_ids']['counterexample']]
    base+=' Positive support concept: '+p['concept_name']+'. Counterexample support concept: '+n['concept_name']+'.'
    if condition=='names_only':return base
    # The diagnostic swaps only explanation assignment; role names remain fixed.
    explanations=(n,p) if condition=='swapped_knowledge_diagnostic' else (p,n)
    for role,c in zip(('Positive','Counterexample'),explanations):
        base+=' '+role+' background: '+c['claim_text']+' Scope: '+c['transfer_limit']
    return base

def main():
    p=argparse.ArgumentParser();p.add_argument('--episodes',type=Path,required=True);p.add_argument('--cards',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    train_path=a.episodes/'episodes_train.jsonl';obj_path=a.episodes/'source_objects.jsonl'
    assert sha(train_path)==PINNED_TRAIN_SHA
    catalog=rows(train_path);raw=rows(obj_path)
    objects={r['object_key']:r for r in raw if r['episode_role']=='train_pool'}
    kc=json.loads(a.cards.read_text());assert kc['expert_reviewed'] is False
    cards={r['class_id']:r for r in kc['cards']};byid={r['card_id']:r for r in cards.values()}
    notes=[support_annotation(e,objects,cards) for e in catalog]
    assert len(notes)==2304 and len({n['episode_id'] for n in notes})==2304
    # Validate text projection across all prepared records; do not export gold, paths or audit IDs in prompts.
    max_words={}
    for condition in ('generic','names_only','matched_knowledge','swapped_knowledge_diagnostic'):
        texts=[render_text(n,byid,condition) for n in notes]
        max_words[condition]=max(len(t.split()) for t in texts)
        assert all(n['episode_id'] not in t and n['query_patch_id_audit_only'] not in t for n,t in zip(notes,texts))
    # Twenty distinct TRAIN query patches, without inspecting query masks/presence or scores.
    ordered=sorted([n for n in notes if n['k_pairs']==1],key=lambda n:hashlib.sha256(n['episode_id'].encode()).hexdigest())
    selected=[];seen=set()
    for is_wheat_target in (True,False):
        count=0
        for n in ordered:
            c=n['support_class_ids_audit_only']
            if (c['positive']==2)!=is_wheat_target or (not is_wheat_target and c['counterexample']!=2):continue
            if n['query_patch_id_audit_only'] in seen:continue
            selected.append(n);seen.add(n['query_patch_id_audit_only']);count+=1
            if count==10:break
        assert count==10
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'train_context_attachments.jsonl').write_text(''.join(json.dumps(n)+'\n' for n in notes))
    dump(a.out/'knowledge_cards.json',kc)
    dump(a.out/'review_queue_20.json',{'status':'selection_only_images_not_packaged','expert_responses':0,'selection':'10 winter-wheat target and 10 winter-wheat counterexample; K1; distinct train queries; no query gold read','cases':selected})
    dump(a.out/'review_response_template.json',{'reviewer_id':None,'episode_id':None,'seconds_spent':None,'observed_target_state':None,'allowed_states':['present','absent','cannot_determine'],'evidence_observation_dates':[],'observed_difference_from_counterexample':None,'uncertainty_reason':None,'additional_observation_needed':None,'source_card_supported_or_corrected':None,'status':'blank_not_a_label'})
    sample=next(n for n in selected if n['support_class_ids_audit_only']['positive']==2)
    dump(a.out/'rendered_text_example.json',{c:render_text(sample,byid,c) for c in max_words})
    receipt={'schema_version':'oe10_agronomy_context_preparation_v0','created_utc':datetime.now(timezone.utc).isoformat(),
      'status':'prepared_metadata_only_not_model_connected','opened_inputs':{str(x):sha(x) for x in (train_path,obj_path,a.cards)},
      'code_sha256':sha(Path(__file__)),'attachments':len(notes),'unique_train_queries':len({n['query_patch_id_audit_only'] for n in notes}),
      'target_concept_episode_counts':dict(Counter(str(n['support_class_ids_audit_only']['positive']) for n in notes)),
      'review_queue_cases':len(selected),'review_queue_unique_train_queries':len(seen),'expert_responses':0,
      'query_gold_read':False,'development_episode_or_score_read':False,'cards_selected_from':'positive/counterexample support annotation only',
      'task_change':'Named support concepts and external descriptions are new input information; B/C must share the same labels. This is not the frozen unnamed-exemplar task.',
      'text_word_count_max_not_token_count':max_words,'actual_token_budget_verified':False,'model_integration_verified':False,'mask_gradient_verified':False,'gpu_runs':0,
      'required_before_training':['Separate text-to-mask path with content/gradient checks','Same architecture and support concepts for names-only vs descriptions','Matched acquired observations and declared token/compute budgets','Training protocol and stop criteria fixed before new results','Expert responses kept separate from sourced summaries']}
    dump(a.out/'preparation_receipt.json',receipt)
    dump(a.out/'file_manifest.json',{'files':[{'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(a.out.iterdir()) if p.is_file()]})
    print(json.dumps(receipt))

if __name__=='__main__':main()
