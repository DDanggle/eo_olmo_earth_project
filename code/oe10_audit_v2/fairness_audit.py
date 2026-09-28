#!/usr/bin/env python3
"""CPU API-contract audit of reviewed OE10 v1/v2/v3 sources and actual OE8 inputs."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

PINNED = {
    'episode_model.py':'4997e85692b5600a034af351cd16360ff303a3a133bacb5d5756c53c89098297',
    'episode_loader.py':'536de03b77bcd5bfedf439b5456d9e0fe60a3789c5f9c1a53c5511ded05d07f6',
    'native_replay.py':'26c0fab96c328e9a5a022a574caf5f78f7ffda4e6fb01164128283f7918ff670'}
REVIEWED_WORKERS = {
    'a05a68f8c706c551d01ef009b1912692d6f1f77b02d69cf8189ef4810e40da27':'v1',
    'a5dc2d2806cd87cb4719de57cad53b149f3677544d634f18219a17751ff6f01d':'v2',
    'f1f5fb3a02e7e20f0c61c8c867d0836561b73cfa00db0631f73f51c0ad30151a':'v3'}
REVIEW = {
    'same_reader_checkpoint_and_trainability':'Worker uses the same fixed VLM path and hashes actual checkpoint files; EpisodeModel freezes FP32 Qwen for both arms; optimizer excludes Qwen. Actual cross-run hash equality is additionally required by summarize_runs.py.',
    'same_connector_and_dense_head_capacity':'Both arms instantiate the same DenseHead(D) and Connector(D,QwenHD), with 64 slots, role prototypes and identical losses. Arm branches change encoder training/cache only; actual initial encoder identity is checked by collector.',
    'same_allowed_images_and_observation_access':'Worker requests [2,5] in both arms; common loader returns copied query2/support8 packets. Native replay uses the same pinned train64 schedule for both arms; execution exposure counts are checked by collector.',
    'query_gold_access_blocked':'Development forward consumes only loader.model_input and no target. Scorer runs after probabilities are saved. Runtime audit traps target access and label-NPZ opens during inference loading; this is API/dataflow separation, not OS isolation.',
    'source_bank_excluded_from_training_supervision':'Every downstream training query/support is train_pool; development support is source_bank. Native replay accepts only its separately pinned train64, excluding its dev8. No claim of historical/geographic nonoverlap between datasets.',
    'identifiers_and_class_mapping_not_model_inputs':'Loader constructs an allowlist with fixed prompt and copied arrays/masks. Model rejects extra top-level IDs, a class-bearing prompt and extra packet fields. Routing/scoring metadata remains outside that interface.',
    'acquisition_and_missing_input_guards_verified':'Loader checks unique acquired indices including 2,5, preserves nodata flags and copies arrays. The exact pinned model validator rejects sentinel-missing acquired pixels; no cloud-quality claim.'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def require(value,message):
    if not value:raise ValueError(message)

def verified_sources(source):
    manifest=json.loads((source/'source_manifest.json').read_text())['files']
    for name,expected in PINNED.items():require(sha(source/name)==expected==manifest[name],'Unreviewed source '+name)
    worker=sha(source/'p2_worker.py')
    require(worker in REVIEWED_WORKERS and worker==manifest['p2_worker.py'],'Unreviewed worker source')
    return {**PINNED,'p2_worker.py':worker},REVIEWED_WORKERS[worker]

def rejected(call,exception=ValueError):
    try:call()
    except exception:return True
    raise ValueError('A forbidden input was accepted')

def exact_model_guards(path):
    tree=ast.parse(path.read_text());scope={'np':np}
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('PROMPT','PACKET_KEYS') for t in node.targets):
            for t in node.targets:scope[t.id]=ast.literal_eval(node.value)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpisodeModel')
    for name in ('_validate_observation','features'):
        node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
        node.decorator_list=[];exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope

def audit(source,prepared,episodes):
    proof={'status':'failed','source_hashes':{},'checks_passed':[],'evidence':{},'limitations':[
        'This is reviewed source plus executable CPU API-contract evidence, not formal OS isolation or a complete runtime trace.',
        'Same checkpoint identities, paired orders, actual training/evaluation exposures and cold resume still require completed-run collector verification.',
        'All80 source input packets are verified; representative episodes cover every query, not every support mask in every episode.',
        'Clouds, parcel disjointness and historical foundation-model/native-data overlap are not certified.'], 'gpu_used':False}
    try:
        proof['source_hashes'],revision=verified_sources(source)
        proof['reviewed_runtime_revision']=revision
        proof['optimizer_revision_note']='v3 explicitly sets AdamW eps=1e-6 for both arms; this differs from v1/v2. Do not mix revisions in paired runs. Strict cold-resume evidence remains separately required.'
        spec=importlib.util.spec_from_file_location('oe10_audited_loader',source/'episode_loader.py')
        loader_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader_module)
        loaders={s:loader_module.EpisodeLoader(prepared,episodes,s) for s in ('train','development')}
        guards=exact_model_guards(source/'episode_model.py');seen=set();queries={};counts={};example=None
        def forbid_gold(*args,**kwargs):raise AssertionError('Inference attempted target access')
        original_load=np.load
        def input_only_load(path,*args,**kwargs):
            require('labels' not in Path(path).parts,'Inference attempted label NPZ access')
            return original_load(path,*args,**kwargs)
        with patch.object(np,'load',input_only_load):
            for split,loader in loaders.items():
                loader._target=forbid_gold;public=loader.episodes;counts[split]=len(public);selected={}
                for episode in public.values():
                    if episode['k_pairs']==1:selected.setdefault(episode['query_patch_id'],episode)
                queries[split]=set(selected)
                for eid in selected.values():
                    value=loader.load(eid,acquired_positions=[2,5]);model=value['model_input'];example=model
                    require(set(model)=={'prompt','query','support_pairs'} and model['prompt']==guards['PROMPT'],'Model input allowlist')
                    guards['_validate_observation'](model['query'],2)
                    for pair in model['support_pairs']:
                        for obs in pair.values():guards['_validate_observation'](obs,8,support=True)
                    require(all(a.base is None for a in model['query'].values()),'Acquired query exposes a backing array')
                    seen.update(Path(p['path']).stem for p in value['audit']['input_file_reads'])
                require(loader._scoring is None,'Inference opened scoring labels')
            train=loaders['train'];dev=loaders['development']
            for pid in train._manifest:
                if pid not in seen:train._read_packet(pid);seen.add(pid)
        parts={k:{pid for pid,r in train._manifest.items() if r['training_partition']==k} for k in ('train_pool','source_bank','dev_query')}
        require([len(parts[k]) for k in parts]==[48,16,16] and len(seen)==80,'Expected exact frozen raw80')
        require(queries['train']==parts['train_pool'] and queries['development']==parts['dev_query'],'Query roles/coverage')
        tid=next(iter(train.episodes));did=next(iter(dev.episodes))
        for positions in ([2],[0,1],[2,2,5],[2,5,8]):rejected(lambda p=positions:train.load(tid,p),loader_module.ContractError)
        rejected(lambda:dev.training_target(did,training=True),loader_module.ContractError)
        rejected(lambda:train.training_target(tid,training=False),loader_module.ContractError)
        bad=copy.deepcopy(example['query']);bad['observation_valid'][0,0,0]=False
        rejected(lambda:guards['_validate_observation'](bad,2))
        bad=copy.deepcopy(example['query']);bad['query_class']=1
        rejected(lambda:guards['_validate_observation'](bad,2))
        rejected(lambda:guards['features'](None,{**example,'pair_id':'hidden_class_shortcut'}))
        rejected(lambda:guards['features'](None,{**example,'prompt':example['prompt']+' The class is maize.'}))
        checks=list(REVIEW)
        proof.update(status='passed',checks_passed=checks,evidence={c:{'basis':'Exact pinned source semantic review plus applicable executable loader/model guards','finding':REVIEW[c]} for c in checks},
            executable_checks={'catalog_episode_counts':counts,'query_counts':{s:len(q) for s,q in queries.items()},'unique_input_packets_verified':len(seen),
                'invalid_acquisition_requests_rejected':4,'training_target_permission_rejections':2,'model_missing_or_extra_field_rejections':2,'model_id_or_prompt_rejections':2,'query_gold_access_during_inference':False},
            prepared_manifest_sha256=sha(prepared/'manifest.jsonl'),episode_contract_sha256=sha(episodes/'episode_contract.json'),
            audit_source_sha256=sha(__file__),review_scope='OE10 '+revision+' common implementation; runtime-run claims delegated to collector')
    except (OSError,ValueError,KeyError,AssertionError,StopIteration,TypeError) as error:proof['error']=str(error)
    return proof

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-dir','prepared-root','episodes-root','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=audit(a.source_dir,a.prepared_root,a.episodes_root)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    mapping_path=None
    if result['status']=='passed':
        mapping_path=a.out.with_name(a.out.stem+'_evidence.json')
        mapping={k:{'path':str(a.out.resolve()),'sha256':sha(a.out)} for k in result['checks_passed']}
        with mapping_path.open('x') as f:json.dump(mapping,f,indent=2);f.write('\n')
    print(json.dumps({'status':result['status'],'checks_passed':result['checks_passed'],'error':result.get('error'),'collector_evidence_json':str(mapping_path) if mapping_path else None}))
    raise SystemExit(0 if result['status']=='passed' else 1)
