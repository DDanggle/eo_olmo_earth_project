#!/usr/bin/env python3
"""Build fixed request contexts from previously written cards; standard library only.

No query data, labels, runtime outputs, model, network, or GPU is accessed.
contexts.json maps '<public pair_id>:<condition>' to exactly the three fields
accepted by the existing v2 validate_context. Routing/provenance stays separate.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

CLASSES=(1,3,8,14)
CONDITIONS=('names_only','matched_knowledge')
PAIR_NAMESPACE='oe8-directed-pair-v0'
EXPECTED={
 'config/oe10_agronomy_context_cards_20260928.json':'ce85a8768527045d3bdd6babdfb48aa9bcef99e6b5bfda497db332f928039243',
 'config/oe10_expansion_agronomy_cards_20260928.json':'d9cfddbde5f2da5985c102a8623ea5a132ffc44533e3474a07d39141882809f8',
 'code/oe10_text_mask_v2/contracts.py':'4908c438bc7c06813bd26817a4f8ef35735ac00e4a9530a0c3b7c74be452ae06',
 'code/oe10_expansion_catalog_v0/episode_builder_reference.py':'64c6df1fe10b40da8cc076286a6184be05ec0db261a4462af0548aaba8eff6d6',
}

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def pair_id(positive,counterexample):
    require(positive in CLASSES and counterexample in CLASSES and positive!=counterexample,'Unsupported directed pair')
    return hashlib.sha256(f'{PAIR_NAMESPACE}:{positive}:{counterexample}'.encode()).hexdigest()[:20]

def load_contracts(repo):
    path=repo/'code/oe10_text_mask_v2/contracts.py'
    spec=importlib.util.spec_from_file_location('oe11_existing_v2_contracts',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def build(repo):
    for relative,expected in EXPECTED.items():
        require(sha(repo/relative)==expected,'Pinned input changed: '+relative)
    contracts=load_contracts(repo)
    cards={};source_map={};card_origin={};common_limits=[]
    for relative in list(EXPECTED)[:2]:
        document=json.loads((repo/relative).read_text())
        require(document['expert_reviewed'] is False and document['patch_level_expert_annotations']==0,'Unexpected supervision provenance')
        require(document.get('human_responses',0)==0,'Unexpected human responses')
        local_sources={s['id']:s for s in document['sources']}
        require(len(local_sources)==len(document['sources']),'Duplicate source ID')
        for card in document['cards']:
            cls=card['class_id']
            if cls not in CLASSES:continue
            require(cls not in cards,'Duplicate selected class')
            require(card['hypothesis_is_measured_effect'] is False,'Unexpected measured claim')
            require(all(type(card[k]) is str and card[k].strip() for k in ('concept_name','claim_text','inspection_hypothesis','transfer_limit')),'Missing role text')
            require(bool(card['source_ids']) and all(s in local_sources for s in card['source_ids']),'Unresolved source ID')
            for claim in card.get('claim_grounding',[]):
                require(all(s in local_sources for s in claim['source_ids']),'Unresolved claim-level source ID')
            cards[cls]=card;card_origin[cls]=relative
            for sid in card['source_ids']:
                source=local_sources[sid]
                require(sid not in source_map or source_map[sid]['source']==source,'Conflicting source identity')
                source_map[sid]={'source':source,'source_sha256':canonical_sha(source),'source_file':relative,'source_file_sha256':EXPECTED[relative]}
        common_limits.extend(document['common_limits'])
    require(set(cards)==set(CLASSES),'Class coverage differs')
    role_fields={'names_only':('concept_name',),'matched_knowledge':('concept_name','claim_text','inspection_hypothesis','transfer_limit')}
    roletexts={condition:{c:'\n'.join(cards[c][field] for field in role_fields[condition]) for c in CLASSES} for condition in CONDITIONS}
    contexts={};records=[];pairs=[]
    for positive,counter in itertools.permutations(CLASSES,2):
        pid=pair_id(positive,counter)
        pairs.append({'pair_id':pid,'positive_class_id':positive,'counterexample_class_id':counter})
        for condition in CONDITIONS:
            cid=pid+':'+condition
            require(cid not in contexts,'Context ID collision')
            context={'instruction':contracts.PROMPT,'positive':roletexts[condition][positive],'counterexample':roletexts[condition][counter]}
            contexts[cid]=contracts.validate_context(context)
            roleprovenance={}
            for role,cls in [('positive',positive),('counterexample',counter)]:
                card=cards[cls];relative=card_origin[cls]
                roleprovenance[role]={'class_id':cls,'card_id':card['card_id'],'source_ids':card['source_ids'],
                    'source_file':relative,'source_file_sha256':EXPECTED[relative],'card_sha256':canonical_sha(card),
                    'copied_fields':list(role_fields[condition]),'role_text_sha256':hashlib.sha256(context[role].encode()).hexdigest()}
            records.append({'context_id':cid,'pair_id':pid,'condition':condition,'context_sha256':contracts.context_sha(context),'roles':roleprovenance})
    require(len(pairs)==len({r['pair_id'] for r in pairs})==12,'Directed-pair count differs')
    require(len(contexts)==len(records)==len({r['context_id'] for r in records})==24,'Context count differs')
    require(len({canonical_sha(c) for c in contexts.values()})==24,'Duplicate context payload')
    swaps=0
    for positive,counter in itertools.permutations(CLASSES,2):
        for condition in CONDITIONS:
            forward=contexts[pair_id(positive,counter)+':'+condition]
            reverse=contexts[pair_id(counter,positive)+':'+condition]
            require(forward['positive']==reverse['counterexample'] and forward['counterexample']==reverse['positive'],'Reverse pair does not swap exact text')
            swaps+=1
            require(set(forward)=={'instruction','positive','counterexample'},'Metadata leaked into model context')
            for role,cls in [('positive',positive),('counterexample',counter)]:
                require(forward[role]=='\n'.join(cards[cls][field] for field in role_fields[condition]),'Card text changed')
    # Exercise the actual v2 boundary on a positive and a deliberately invalid input.
    first=next(iter(contexts.values()));bad=dict(first,source_ids=['must_not_enter_model'])
    try:contracts.validate_context(bad)
    except ValueError:pass
    else:raise ValueError('v2 contract accepted provenance in model context')
    provenance={'schema':'oe11_fixed_request_context_provenance_v0','classes':list(CLASSES),'conditions':list(CONDITIONS),
        'pair_namespace':PAIR_NAMESPACE,'pair_id_format':'sha256(namespace:positive_class:counterexample_class)[:20]',
        'context_id_format':'pair_id:condition','routing_fields_are_not_model_inputs':True,
        'pair_catalog':pairs,'records':records,'sources':source_map,'input_sha256':EXPECTED,
        'human_responses':0,'patch_level_expert_annotations':0,'expert_reviewed':False,
        'parcel_level_regional_facts':False,'query_gold_read':False,'runtime_dataset_read':False,'GPU_used':False,
        'common_limits':list(dict.fromkeys(common_limits)),
        'limitations':['Fixed four-class role texts can function as class codes; this bundle does not establish semantic or regional reasoning.',
            'Class names are public request/support semantics, not inferred from query gold.',
            'Facts were copied from existing AI-written, source-grounded cards; no new source verification or factual claims were added.',
            'No parcel-specific climate, local weather, crop stage, treatment, event, health, or acquisition-year facts are supplied.',
            'matched_knowledge must be compared with names_only under matched architecture, images, supports and training budget.',
            'Source publication dates are provenance, not evidence of conditions in 2019 observations.']}
    verification={'status':'PASS','contexts':24,'directed_pairs':12,'conditions':list(CONDITIONS),
        'actual_v2_validate_context_passes':24,'unique_context_payloads':24,'reverse_role_swap_checks':swaps,
        'verbatim_existing_card_field_copy_verified':True,'all_source_ids_resolve':True,
        'model_context_has_only_three_allowed_fields':True,'provenance_injection_rejected':True,
        'classes':list(CLASSES),'winter_wheat_or_barley_target_contexts':0,
        'runtime_or_query_label_files_read':0,'model_forwards':0,'human_responses':0,
        'scope':'Text construction and contract validation only; no tokenization, model inference, learning, or performance evaluation.'}
    return contexts,provenance,verification

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    names=('contexts.json','provenance.json','verification.json','bundle_manifest.json')
    require(not any((a.out/name).exists() for name in names),'Refusing to replace an existing context bundle')
    contexts,provenance,verification=build(a.repo)
    a.out.mkdir(parents=True,exist_ok=True)
    for name,value in [('contexts.json',contexts),('provenance.json',provenance),('verification.json',verification)]:
        (a.out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
    manifest={'generator_sha256':sha(Path(__file__)),'input_sha256':EXPECTED,
        'files':{name:{'sha256':sha(a.out/name),'bytes':(a.out/name).stat().st_size} for name in names[:-1]}}
    (a.out/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(verification))

if __name__=='__main__':main()
