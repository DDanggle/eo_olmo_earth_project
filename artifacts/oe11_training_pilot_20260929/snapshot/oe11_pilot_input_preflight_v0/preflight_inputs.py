#!/usr/bin/env python3
"""CPU-only selected training packet and pretrained-file identity audit."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def need(ok,message):
    if not ok:raise ValueError(message)

def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s)
    s.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser()
    for name in ('prepared','catalog','cases','contexts','loader-code','text-mask-code','model-identity','eo-source','eo-weights','qwen-weights','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cases-sha256',required=True);p.add_argument('--model-identity-sha256',required=True)
    p.add_argument('--contexts-sha256',required=True)
    a=p.parse_args();start=time.monotonic();need(not a.out.exists(),'Existing output')
    need(sha(a.cases)==a.cases_sha256,'Case hash');need(sha(a.model_identity)==a.model_identity_sha256,'Model identity hash')
    need(sha(a.contexts)==a.contexts_sha256,'Context hash')
    cases=json.loads(a.cases.read_text());need(cases['status']=='ready' and cases['split']=='train','Fixed train-only cases')
    need(len(cases['episode_ids'])==len(set(cases['episode_ids']))==24,'24 K1/K8 identities')
    need(Counter(cases['order'])==Counter({eid:2 for eid in cases['episode_ids']}),'Two complete cycles')
    need(cases['total_steps']==48 and cases['split_step']==24 and len(cases['fit_episode_ids'])==12,'Step scope')
    contexts=json.loads(a.contexts.read_text())
    contracts=load_module('_oe11_pilot_context_contracts',a.text_mask_code/'contracts.py')
    for context in contexts.values():contracts.validate_context(context)
    loader_module=load_module('_oe11_pilot_loader',a.loader_code/'loader_adapter.py')
    Loader=loader_module.make_loader_class(a.loader_code/'frozen_episode_loader.py')
    loader=Loader(a.prepared,a.catalog,'train',support_mode='pooled')
    details=[]
    by_id={eid:row for row in cases['cases'] for eid in (row['episode_id'],row['k8_episode_id'])}
    for eid in cases['episode_ids']:
        e=loader.episodes[eid];record=by_id[eid]
        item=loader.load(eid,acquired_positions=[2,5])
        need(item['audit']['query_gold_read'] is False,'No gold in model load')
        x=item['model_input'];need(x['query']['s2'].shape==(128,128,2,12),'Two query dates')
        need(len(x['support_pairs'])==e['k_pairs'],'Support pair count')
        for pair in x['support_pairs']:
            for obj in pair.values():need(obj['s2'].shape==(128,128,8,12),'Eight support dates')
        target=loader.load_target(eid,purpose='training',training=True)
        need(int(target['target_mask'].sum())==record['target_pixels'],'Actual target mask count')
        for condition in ('names_only','matched_knowledge'):
            contracts.validate_context(contexts[e['pair_id']+':'+condition])
        details.append({'episode_id':eid,'parent':e['query_parent_tile'],'k':e['k_pairs'],
                        'kind':record['kind'],'actual_target_pixels':int(target['target_mask'].sum()),
                        'query_gold_in_inference_load':False,'separate_training_target_read':True})
    identity=json.loads(a.model_identity.read_text());verified={};count=0;size=0
    for key,root in [('reader_files_sha256',a.qwen_weights),('eo_files_sha256',a.eo_weights),('eo_source_files_sha256',a.eo_source)]:
        verified[key]={}
        for relative,expected in identity[key].items():
            f=(root/relative).resolve();need(f.is_relative_to(root.resolve()),'Identity path escape')
            actual=sha(f);need(actual==expected,'Actual pretrained identity mismatch: '+str(f))
            verified[key][relative]=actual;count+=1;size+=f.stat().st_size
    discovered={str(f.relative_to(a.qwen_weights)) for f in a.qwen_weights.rglob('*')
                if f.is_file() and f.suffix in {'.json','.jinja','.safetensors','.txt'}}
    need(discovered==set(identity['reader_files_sha256']),'Reader asset coverage')
    result={'status':'PASS','scope':'CPU actual training-packet loading and pretrained file identity; no torch/model/GPU execution',
            'cases_sha256':a.cases_sha256,'contexts_sha256':a.contexts_sha256,'model_identity_sha256':a.model_identity_sha256,
            'prepared_manifest_sha256':sha(a.prepared/'manifest.jsonl'),
            'train_catalog_sha256':sha(a.catalog/'episodes_train.jsonl'),'case_loads':details,
            'selected_case_count':24,'fit_case_count':12,'optimizer_order_count':48,
            'pretrained_files_verified':count,'pretrained_bytes_hashed':size,
            'pretrained_verified':verified,'GPU_used':False,'model_forwards':0,
            'development_or_final_labels_read':False,'loader_audit':loader.adapter_audit(),
            'environment_imported_torch':'torch' in sys.modules,'seconds':time.monotonic()-start,
            'script_sha256':sha(Path(__file__))}
    need(not result['environment_imported_torch'],'Preflight imported torch')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','selected_case_count','pretrained_files_verified','pretrained_bytes_hashed','seconds')}))

if __name__=='__main__':main()
