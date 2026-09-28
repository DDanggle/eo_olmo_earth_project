#!/usr/bin/env python3
"""Independent CPU audit of completed E6 artifacts; no producer/model imports.

Reads completed files only when explicitly invoked. Recomputes all source-label
metrics and event intervals; checks saved lineage, metadata and CPU tensors.
Does not replay optimization/inference or independently observe past chronology.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

import numpy as np

PLAN_SHA = '66f820db8bc6a3d03c07d66676adcbe6170c012649a824e83b02a94d03194840'
E5_MANIFEST_SHA = 'e22fb584c95f2bac1916fd8d3c49234e993a156f80d38ac481140e12c3e7819f'
E5_AUDITOR_SHA = 'a9b1ed423b54288f6de486d76f23da28ffbfb2bc67547628dd8e52fbe0b63825'
SOURCE_FILES = {'e6_prepare_v0.py', 'e6_head_model_v0.py', 'e6_train_v0.py',
                'e6_scoring_v0.py', 'run_e6_when_idle_v0.py'}
SETS = {'all_test':1755, 'primary_same_prompt':902, 'paired_flood':914,
        'hard_negative_flood':457, 'landslide':384, 'e3_subset':209}
SEEDS = (1, 2, 3)


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        ensure(key not in result, 'Duplicate JSON key: '+key)
        result[key] = value
    return result


def decode(text):
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def safe_file(root, name):
    relative = Path(name)
    ensure(not relative.is_absolute() and relative.parts and '..' not in relative.parts,
           'Unsafe relative file: '+str(name))
    result = (root/relative).resolve()
    ensure(result.is_relative_to(root.resolve()) and result.is_file(), 'Missing/escaping file: '+str(name))
    return result


class Files:
    def __init__(self):
        self.hashes = {}
    def track(self, path, expected=None):
        path = Path(path).resolve()
        got = digest(path)
        ensure(expected is None or got == expected, 'File SHA mismatch: '+str(path))
        ensure(str(path) not in self.hashes or self.hashes[str(path)] == got, 'File changed: '+str(path))
        self.hashes[str(path)] = got
        return path
    def obj(self, path, expected=None):
        return decode(self.track(path, expected).read_text())
    def rows(self, path, expected=None):
        return [decode(line) for line in self.track(path, expected).read_text().splitlines() if line.strip()]
    def recheck(self):
        for path, expected in self.hashes.items():
            ensure(digest(path) == expected, 'File changed during audit: '+path)


def compare(expected, actual, where='value'):
    """Strict key/list structure; float tolerance only for recomputed arithmetic."""
    if isinstance(expected, dict):
        ensure(isinstance(actual, dict) and set(expected) == set(actual), 'Key coverage differs at '+where)
        for key in expected:
            compare(expected[key], actual[key], where+'.'+str(key))
    elif isinstance(expected, list):
        ensure(isinstance(actual, list) and len(expected) == len(actual), 'List coverage differs at '+where)
        for j, (a,b) in enumerate(zip(expected, actual)):
            compare(a,b,where+'['+str(j)+']')
    elif type(expected) is float:
        ensure(type(actual) in (float,int) and math.isfinite(actual)
               and math.isclose(expected,actual,rel_tol=1e-11,abs_tol=1e-12), 'Numeric value differs at '+where)
    else:
        ensure(type(expected) is type(actual) and expected == actual, 'Value/type differs at '+where)


def metadata_values(items):
    output = []
    for item in items:
        ensure(item['phen'] in ('flood','landslide'), 'Unknown phenomenon')
        dates = item['dates']
        ensure(isinstance(dates,list) and len(dates)==2 and all(type(x) is str and re.fullmatch(r'\d{4}-\d{2}-\d{2}',x) for x in dates), 'Date schema')
        first,last = [date.fromisoformat(x) for x in dates]
        ensure(first < last, 'Date chronology')
        row = [1.0 if item['phen']=='flood' else 0.0, 1.0 if item['phen']=='landslide' else 0.0]
        for day in (first,last):
            row += [(day.year-2000)/100.0, (day.month-1)/11.0, (day.day-1)/30.0]
        output.append(row)
    return np.asarray(output,dtype=np.float32)


def population(items, sets, ordered, batches):
    ensure(len(items)==5989 and len({x['id'] for x in items})==5989, 'Item coverage')
    index = {x['id']:j for j,x in enumerate(items)}
    ensure(all(type(x['id']) is str for x in items), 'Non-string item ID')
    ensure(set(ordered)=={'train','test'}, 'Ordered partition keys')
    for part,n in [('train',4234),('test',1755)]:
        ensure(ordered[part]==[x['id'] for x in items if x['partition']==part]
               and len(ordered[part])==n, 'Ordered IDs differ: '+part)
    expected = {('train','flood','pos'):1066,('train','flood','neg'):1066,('train','flood','hard_neg'):1066,
                ('train','landslide','pos'):518,('train','landslide','neg'):518,
                ('test','flood','pos'):457,('test','flood','neg'):457,('test','flood','hard_neg'):457,
                ('test','landslide','pos'):192,('test','landslide','neg'):192}
    ensure(Counter((x['partition'],x['phen'],x['kind']) for x in items)==expected, 'Category counts differ')
    tiles = defaultdict(list)
    for item in items:
        ensure(item['answer']==('yes' if item['kind']=='pos' else 'no'), 'Source gold/kind differs')
        ensure(type(item['tile']) is str and type(item['cluster']) in (str,int), 'Invalid tile/cluster')
        if item['phen']=='flood':
            slots=['pre_1','pre_2'] if item['kind']=='neg' else ['pre_2','post']
            ensure(item['slots']==slots and str(item['event'])==str(item['cluster']), 'Flood slots/event differs')
        tiles[(item['phen'],item['tile'])].append(item)
    for members in tiles.values():
        ensure(len({x['partition'] for x in members})==1, 'Train/test tile overlap')
        if any(x['kind']=='hard_neg' for x in members):
            ensure(len(members)==1, 'Hard-negative tile overlap')
        else:
            ensure(len(members)==2 and {x['kind'] for x in members}=={'pos','neg'}
                   and len({str(x['cluster']) for x in members})==1, 'Same-tile source pair differs')
    for phen,ntrain,ntest in [('flood',27,10),('landslide',7,2)]:
        groups=[{str(x['cluster']) for x in items if x['phen']==phen and x['partition']==p} for p in ('train','test')]
        ensure(tuple(map(len,groups))==(ntrain,ntest) and groups[0].isdisjoint(groups[1]), 'Event support/partition overlap')
    ensure(set(sets)==set(SETS), 'Evaluation set keys')
    test=[x for x in items if x['partition']=='test']
    for name,ids in sets.items():
        ensure(isinstance(ids,list) and len(ids)==len(set(ids))==SETS[name], 'Set coverage: '+name)
        ensure(ids==[x['id'] for x in test if x['id'] in set(ids)], 'Set order/train leakage: '+name)
    compare(ordered['test'],sets['all_test'],'all_test')
    for name,selection in [('paired_flood',lambda x:x['phen']=='flood' and x['kind']!='hard_neg'),
                           ('hard_negative_flood',lambda x:x['phen']=='flood' and x['kind']=='hard_neg'),
                           ('landslide',lambda x:x['phen']=='landslide')]:
        compare([x['id'] for x in test if selection(x)],sets[name],name)
    selected=[items[index[key]] for key in sets['primary_same_prompt']]
    ensure(Counter((x['phen'],x['kind']) for x in selected)=={('flood','pos'):445,('flood','hard_neg'):457}, 'Primary counts')
    strata=defaultdict(list)
    for x in selected:
        strata[(str(x['cluster']),tuple(x['dates']),tuple(x['slots']))].append(x)
    ensure(len({k[0] for k in strata})==8 and all({x['kind'] for x in group}=={'pos','hard_neg'} for group in strata.values()), 'Primary supported strata')
    ensure(set(batches)=={'1','2','3'}, 'Batch seed keys')
    for seed in SEEDS:
        generator=np.random.default_rng(seed)
        epochs=[]
        for epoch in range(3):
            shuffled=[ordered['train'][int(j)] for j in generator.permutation(4234)]
            epochs.append([shuffled[j:j+8] for j in range(0,4234,8)])
        compare(epochs,batches[str(seed)],'batches seed '+str(seed))
    return index,test,strata


def sigmoid(z):
    return 1/(1+math.exp(-z)) if z>=0 else math.exp(z)/(1+math.exp(z))


def predictions(rows, items, is_head):
    lookup={x['id']:(j,x) for j,x in enumerate(items)}
    expected={(seed,x['id']) for seed in SEEDS for x in items if x['partition']=='test'}
    ensure(len(rows)==len(expected)==5265,'Prediction count')
    answers={}
    for row in rows:
        ensure(type(row.get('seed')) is int and (row['seed'],row.get('id')) in expected,'Prediction identity')
        key=(row['seed'],row['id']);ensure(key not in answers,'Duplicate prediction')
        j,item=lookup[row['id']]
        ensure(row['model_arm']==('full_head' if is_head else 'full') and row['eval_arm']=='native','Wrong system')
        ensure(all(row[k]==item[k] for k in ('tile','phen','kind')) and str(row['cluster'])==str(item['cluster']),'Prediction metadata')
        ensure(type(row['pair_index']) is int and row['pair_index']==j,'Global pair index')
        ensure(row['source_gold']==item['answer'] and 'transformed_gold' in row and row['transformed_gold'] is None,'Gold changed')
        if is_head:
            z,p=row['logit'],row['probability']
            ensure(type(z) in (float,int) and type(p) in (float,int) and math.isfinite(z) and math.isfinite(p),'Nonfinite head')
            ensure(0<=p<=1 and abs(p-sigmoid(z))<=1e-6,'Sigmoid mismatch')
            answer='yes' if z>=0 else 'no'
            ensure(row['prediction']==answer,'Fixed threshold mismatch')
        else:
            ensure(type(row['answer_raw']) is str,'Raw answer schema')
            matches=re.findall(r'\b(?:yes|no)\b',row['answer_raw'].strip().lower())
            answer=matches[0] if matches else None
            ensure('parsed' in row and row['parsed']==answer,'Raw parse mismatch')
        answers[key]=answer
    ensure(set(answers)==expected,'Incomplete predictions')
    rates={}
    for seed in SEEDS:
        for phen in ('flood','landslide'):
            group=[x for x in items if x['partition']=='test' and x['phen']==phen]
            failed=sum(answers[(seed,x['id'])] is None for x in group)
            ensure(failed/len(group)<=.01+1e-12,'Reference parse failure limit')
            rates[f'{seed}|{phen}']={'n':len(group),'failed':failed,'rate':failed/len(group)}
    return answers,rates


def binary(members, answers, seed, neg):
    counts=Counter(x['kind'] for x in members)
    ensure(set(counts)=={'pos',neg}, 'Binary stratum has missing/extra class')
    tp=sum(answers[(seed,x['id'])]=='yes' for x in members if x['kind']=='pos')
    tn=sum(answers[(seed,x['id'])]=='no' for x in members if x['kind']==neg)
    fp=sum(answers[(seed,x['id'])]=='yes' for x in members if x['kind']==neg)
    recall=tp/counts['pos'];specificity=tn/counts[neg]
    return {'n_pos':counts['pos'],'n_negative':counts[neg],'negative_kind':neg,
            'recall':recall,'specificity':specificity,'fpr':fp/counts[neg],
            'ba':(recall+specificity)/2,'parse_failures':sum(answers[(seed,x['id'])] is None for x in members)}


def primary_metric(strata,answers,seed):
    byevent=defaultdict(list)
    for (event,dates,slots),group in sorted(strata.items()):
        byevent[event].append({'dates':list(dates),'slots':list(slots),**binary(group,answers,seed,'hard_neg')})
    events={}
    for event,groups in sorted(byevent.items()):
        values={k:math.fsum(x[k] for x in groups)/len(groups) for k in ('ba','recall','specificity','fpr')}
        values.update(n_strata=len(groups),n_pos=sum(x['n_pos'] for x in groups),
                      n_hard_neg=sum(x['n_negative'] for x in groups),parse_failures=sum(x['parse_failures'] for x in groups),strata=groups)
        events[event]=values
    return {'macro_ba':math.fsum(x['ba'] for x in events.values())/len(events),
            'n_events':len(events),'n_strata':len(strata),'n_items':sum(len(x) for x in strata.values()),'events':events}


def paired_metric(test,answers,seed,phen):
    groups=defaultdict(list)
    for x in test:
        if x['phen']==phen and x['kind'] in ('pos','neg'):groups[str(x['cluster'])].append(x)
    events={event:binary(group,answers,seed,'neg') for event,group in sorted(groups.items())}
    return {'macro_ba':math.fsum(x['ba'] for x in events.values())/len(events),'n_events':len(events),
            'n_items':sum(len(group) for group in groups.values()),'events':events}


def hard_metric(test,answers,seed):
    groups=defaultdict(list)
    for x in test:
        if x['phen']=='flood' and x['kind']=='hard_neg':groups[str(x['cluster'])].append(x)
    events={}
    for event,members in sorted(groups.items()):
        yes=sum(answers[(seed,x['id'])]=='yes' for x in members)
        events[event]={'n':len(members),'fpr':yes/len(members),'yes_count':yes,
                       'parse_failures':sum(answers[(seed,x['id'])] is None for x in members)}
    n=sum(x['n'] for x in events.values())
    return {'n':n,'n_events':len(events),'events':events,'fpr_pooled':sum(x['yes_count'] for x in events.values())/n,
            'fpr_event_macro':math.fsum(x['fpr'] for x in events.values())/len(events),
            'parse_failures':sum(x['parse_failures'] for x in events.values())}


def event_interval(deltas):
    ensure(len(deltas)==8 and all(math.isfinite(x) for x in deltas),'Bootstrap requires eight paired events')
    rng=np.random.default_rng(20260925)
    sample_indices=rng.integers(low=0,high=8,size=(5000,8))
    draws=np.asarray([math.fsum(deltas[int(j)] for j in row)/8 for row in sample_indices],dtype=np.float64)
    # Explicit linear percentile interpolation, not producer quantile function.
    ordered=np.sort(draws)
    limits=[]
    for probability in (.025,.975):
        index=(len(ordered)-1)*probability; lower=math.floor(index); fraction=index-lower
        limits.append(float(ordered[lower]*(1-fraction)+ordered[math.ceil(index)]*fraction))
    return limits


def contrast(reference,head,primary):
    ensure(set(reference['events'])==set(head['events']),'Contrast event support')
    values={k:reference['events'][k]['ba']-head['events'][k]['ba'] for k in sorted(reference['events'])}
    return {'delta':math.fsum(values.values())/len(values),'event_deltas':values,
            'ci95_delta':event_interval(list(values.values())) if primary else None,
            'left':'full/native','right':'full_head/native','direction':'E5_full_minus_E6_head'}


def recompute_metrics(test,strata,head,reference):
    output={}
    for seed in SEEDS:
        prim={name:primary_metric(strata,values,seed) for name,values in [('full/native',reference),('full_head/native',head)]}
        paired={}
        for phen in ('flood','landslide'):
            m={name:paired_metric(test,values,seed,phen) for name,values in [('full/native',reference),('full_head/native',head)]}
            paired[phen]={'evaluations':m,'contrasts':{'full_minus_head':contrast(m['full/native'],m['full_head/native'],False)}}
        output[str(seed)]={'primary_same_prompt':{'evaluations':prim,'contrasts':{'full_minus_head':contrast(prim['full/native'],prim['full_head/native'],True)}},
                          'paired_source':paired,
                          'hard_negative':{name:hard_metric(test,values,seed) for name,values in [('full/native',reference),('full_head/native',head)]}}
    return output


def checkpoint_schema():
    shapes={'input_norm.weight':(768,),'input_norm.bias':(768,),
            'input_linear.weight':(128,768),'input_linear.bias':(128,),
            'input_output_norm.weight':(128,),'input_output_norm.bias':(128,),
            'type_embedding.weight':(4,128),'cls':(128,),
            'metadata_linear.weight':(128,8),'metadata_linear.bias':(128,),
            'output_norm.weight':(128,),'output_norm.bias':(128,),
            'classifier.weight':(1,128),'classifier.bias':(1,),
            'positions':(192,128),'token_types':(192,)}
    for block in range(2):
        for layer,weight,bias in [('norm_attention',(128,),(128,)),('qkv',(384,128),(384,)),
                                  ('attention_output',(128,128),(128,)),('norm_ff',(128,),(128,)),
                                  ('ff_in',(256,128),(256,)),('ff_out',(128,256),(128,))]:
            shapes[f'blocks.{block}.{layer}.weight']=weight;shapes[f'blocks.{block}.{layer}.bias']=bias
    return shapes


def canonical_tensor_hash(state):
    """Producer's documented byte format, independently implemented from tensors."""
    import torch
    ensure(isinstance(state,dict) and state,'Empty/nonmapping state')
    accumulator=hashlib.sha256()
    for name in sorted(state):
        value=state[name]
        ensure(type(name) is str and isinstance(value,torch.Tensor) and value.device.type=='cpu','Non-CPU state tensor')
        value=value.detach().contiguous()
        ensure(bool(torch.isfinite(value).all()),'Nonfinite state tensor')
        header=json.dumps([name,str(value.dtype),list(value.shape)],separators=(',',':')).encode('utf8')
        data=value.view(torch.uint8).numpy().tobytes(order='C')
        accumulator.update(struct.pack('<Q',len(header)));accumulator.update(header)
        accumulator.update(struct.pack('<Q',len(data)));accumulator.update(data)
    return accumulator.hexdigest()


def check_checkpoint(path, expected_hash, initial=False):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True)
    schema=checkpoint_schema()
    ensure(set(state)==set(schema),'Checkpoint named state coverage')
    for name,shape in schema.items():
        ensure(tuple(state[name].shape)==shape and state[name].dtype==(torch.int64 if name=='token_types' else torch.float32),'Checkpoint shape/dtype: '+name)
    ensure(sum(state[name].numel() for name in schema if name not in ('positions','token_types'))==367361,'Parameter count')
    types=np.asarray([0]*64+[1]*64+[3]*64,dtype=np.int64)
    ensure(np.array_equal(state['token_types'].numpy(),types),'Fixed token types')
    grid=[]
    for row in range(8):
        for col in range(8):
            values=[]
            for coordinate in (row,col):
                for frequency in range(32):
                    phase=coordinate/(10000.0**(frequency/32.0));values += [math.sin(phase),math.cos(phase)]
            grid.append(values)
    ensure(np.array_equal(state['positions'].numpy(),np.tile(np.asarray(grid,dtype=np.float32),(3,1))),'Fixed spatial positions')
    if initial:ensure(bool((state['cls']==0).all()),'Initial CLS not zero')
    calculated=canonical_tensor_hash(state)
    ensure(calculated==expected_hash,'Canonical CPU tensor SHA differs')
    return calculated


def check_steps(rows,schedule,index,seed):
    ensure(len(rows)==1590,'Optimizer update count')
    offset=0;exposures=0;epoch_losses=[];last_elapsed=-1.0
    for epoch,batches in enumerate(schedule):
        loss_sum=0.;n=0
        for batch_number,ids in enumerate(batches):
            row=rows[offset];offset+=1;exposures+=len(ids)
            ensure(all(type(row.get(k)) is int for k in ('seed','step','epoch','batch_index','batch_size','exposures')),'Step integer schema')
            ensure((row['seed'],row['step'],row['epoch'],row['batch_index'],row['batch_size'],row['exposures'])
                   ==(seed,offset,epoch,batch_number,len(ids),exposures),'Step counters/budget')
            ensure(row['ids']==ids and row['pair_indices']==[index[key] for key in ids],'Step batch order/index')
            ensure(all(row.get(k) is True for k in ('loss_finite','gradients_finite','parameters_finite')),'Step finite attestation')
            ensure(type(row['loss']) in (int,float) and math.isfinite(row['loss']) and row['loss']>=0,'Step BCE nonfinite/negative')
            ensure(type(row['elapsed_s']) in (int,float) and math.isfinite(row['elapsed_s']) and row['elapsed_s']>=last_elapsed,'Nonmonotonic step timing')
            last_elapsed=row['elapsed_s'];loss_sum+=row['loss']*len(ids);n+=len(ids)
        ensure(n==4234,'Epoch exposure count');epoch_losses.append(loss_sum/n)
    ensure(exposures==12702,'Total exposure count')
    return epoch_losses


def check_optimizer(groups):
    ensure(isinstance(groups,list) and len(groups)==1,'Optimizer group count')
    group=groups[0]
    expected={'lr':1e-4,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'foreach':False,
              'fused':False,'amsgrad':False,'maximize':False}
    for key,value in expected.items():compare(value,group[key],'optimizer.'+key)


def audit(artifact, expected_manifest_sha256, reference_artifact=None):
    root=Path(artifact).resolve();files=Files()
    ensure(re.fullmatch(r'[0-9a-f]{64}',expected_manifest_sha256) is not None,'Explicit frozen E6 manifest SHA required')
    ensure(not (root/'failure.json').exists(),'Execution failed artifact')
    manifest=files.obj(root/'manifest.json',expected_manifest_sha256)
    ensure(manifest['schema']=='e6-no-llm-prepared-v0' and manifest['parent_e5_prepared_manifest_sha256']==E5_MANIFEST_SHA,'Prepared schema/parent')
    ensure((manifest['n_items'],manifest['n_train'],manifest['n_test'])==(5989,4234,1755),'Prepared counts')
    ensure(set(manifest['code_snapshot_sha256'])==SOURCE_FILES,'Exact code snapshot source membership')
    for name,pin in manifest['code_snapshot_sha256'].items():files.track(safe_file(root/'code_snapshot',name),pin)
    required={'items.jsonl','pairs.npy','ordered_ids.json','batches.json','eval_sets.json','reference_rows.jsonl',
              'reference_audit.json','e5_independent_audit.json','prereg.json','preparation_report.json',
              'reference_manifest.json','reference_prereg.json','reference_status.json','reference_scores.json',
              'reference_training_summary.json','reference_inference_completed.json'}
    ensure(set(manifest['files_sha256'])==required,'Prepared frozen-file membership')
    for name,pin in manifest['files_sha256'].items():files.track(safe_file(root,name),pin)
    cfg=files.obj(root/'prereg.json',PLAN_SHA)
    ensure(cfg['id']=='E6-HEAD-v0' and set(cfg['source_files'])==SOURCE_FILES,'Frozen plan identity')
    ensure(manifest['plan_sha256']==PLAN_SHA,'Manifest plan pin')
    for name,pin in cfg['reference']['files_sha256'].items():ensure(manifest['files_sha256'][name]==pin,'Inherited source pin')
    status=files.obj(root/'status.json');scores=files.obj(root/'scores.json');summary=files.obj(root/'training_summary.json')
    ensure(status['status']=='completed' and status['scientifically_valid'] is True
           and (status['n_models'],status['n_rows'])==(3,5265),'Completed status required')
    ensure(scores['schema']=='e6-no-llm-scores-v0' and scores['valid'] is True
           and scores['verdict']=='descriptive_system_comparison','Valid descriptive score required')
    ensure(summary['schema']=='e6-training-summary-v0' and summary['all_training_before_test'] is True
           and (summary['n_models'],summary['n_rows'],summary['updates_per_model'],summary['exposures_per_model'])==(3,5265,1590,12702),'Training summary contract')
    for key,name in [('manifest_sha256','manifest.json'),('training_summary_sha256','training_summary.json'),
                     ('predictions_sha256','predictions.jsonl'),('reference_rows_sha256','reference_rows.jsonl')]:files.track(root/name,scores[key])
    for key,name in [('manifest_sha256','manifest.json'),('initial_manifest_sha256','initial_states/manifest.json'),
                     ('metadata_vectors_sha256','metadata_vectors.npy'),('metadata_snapshot_sha256','metadata_snapshot.jsonl'),
                     ('predictions_sha256','predictions.jsonl'),('runtime_sha256','runtime_environment.json')]:files.track(root/name,summary[key])
    compare(manifest['code_snapshot_sha256'],summary['code_snapshot_sha256'],'summary source hashes')
    gate=files.obj(root/'reference_audit.json');independent=files.obj(root/'e5_independent_audit.json',gate['e5_independent_audit_sha256'])
    original_root=Path(cfg['reference']['directory'])
    ensure(gate['schema']=='e6-reference-gate-v0' and gate['valid'] is True and gate['independent_consistent'] is True
           and gate['n_rows']==5265 and gate['e5_prepared_manifest_sha256']==E5_MANIFEST_SHA,'Reference gate')
    ensure(independent['schema']=='e5-independent-result-audit-v0' and independent['consistent'] is True
           and independent['checkpoint_tensors_loaded_and_checked_on_cpu'] is True
           and independent['audit_code_sha256']==E5_AUDITOR_SHA and independent['artifact']==str(original_root),'Independent E5 gate')
    ensure(tuple(independent[k] for k in ('n_models','n_steps','n_training_exposures','n_answers','primary_n','primary_events'))==(12,19080,152424,26325,902,8),'Independent E5 budget')
    for name in ('manifest.json','prereg.json','status.json','scores.json','training_summary.json','inference_completed.json'):
        key=str(original_root/name);pin=gate['original_e5_files_sha256'][key]
        ensure(independent['hashes_verified'].get(key)==pin,'Reference source audit pin')
        files.track(root/('reference_'+name),pin)
    files.track(root/'reference_manifest.json',E5_MANIFEST_SHA)
    files.track(root/'reference_prereg.json',cfg['reference']['prereg_sha256'])
    for name,pin in cfg['reference']['files_sha256'].items():
        key=str(original_root/name)
        ensure(gate['original_e5_files_sha256'].get(key)==pin==independent['hashes_verified'].get(key),'Copied E5 input audit pin')
    ensure(gate['original_e5_files_sha256'].get(cfg['reference']['independent_audit'])==digest(root/'e5_independent_audit.json'),'Audit copy pin')
    e5status=files.obj(root/'reference_status.json');e5scores=files.obj(root/'reference_scores.json')
    ensure(e5status['status']=='completed' and e5scores['valid'] is True
           and e5scores['verdict']==e5status['verdict']==independent['verdict'],'Saved reference valid status')
    reference_rows=files.rows(root/'reference_rows.jsonl',gate['reference_rows_sha256'])
    reference_root=Path(reference_artifact).resolve() if reference_artifact else original_root
    original_predictions_path=reference_root/'predictions.jsonl'
    pin=gate['original_e5_files_sha256'][str(original_root/'predictions.jsonl')]
    ensure(independent['hashes_verified'].get(str(original_root/'predictions.jsonl'))==pin,'Original full prediction audit pin')
    original=files.rows(original_predictions_path,pin)
    ensure(len(original)==26325,'Original E5 prediction population')
    selected=[r for r in original if r.get('model_arm')=='full' and r.get('eval_arm')=='native']
    compare(selected,reference_rows,'Unmodified preselected full/native reference')
    items=files.rows(root/'items.jsonl');ordered=files.obj(root/'ordered_ids.json')
    batches=files.obj(root/'batches.json');sets=files.obj(root/'eval_sets.json')
    index,test,strata=population(items,sets,ordered,batches)
    matrices=np.load(root/'pairs.npy',mmap_mode='r',allow_pickle=False)
    ensure(matrices.shape==(5989,2,64,768) and matrices.dtype==np.float32,'Pair array shape/dtype')
    for j in range(5989):
        ensure(np.isfinite(matrices[j]).all(),'Nonfinite frozen pair')
        with np.errstate(over='ignore',invalid='ignore'):difference=matrices[j,1]-matrices[j,0]
        ensure(np.isfinite(difference).all(),'Nonfinite difference block')
    expected_meta=metadata_values(items);saved_meta=np.load(root/'metadata_vectors.npy',allow_pickle=False)
    ensure(saved_meta.dtype==np.float32 and np.array_equal(expected_meta,saved_meta),'Metadata vector bytes/shape differ')
    snapshot=files.rows(root/'metadata_snapshot.jsonl')
    ensure(len(snapshot)==5989,'Metadata snapshot count')
    for j,(row,item) in enumerate(zip(snapshot,items)):
        compare({'id':item['id'],'pair_index':j,'phen':item['phen'],'dates':item['dates'],'encoded_values':expected_meta[j].tolist()},row,'metadata row '+str(j))
    runtime=files.obj(root/'runtime_environment.json')
    ensure(runtime['manifest_sha256']==expected_manifest_sha256 and runtime['gpu_index']==0 and runtime['device']=='cuda:0'
           and runtime['deterministic_algorithms'] is True and runtime['tf32_matmul'] is False
           and runtime['tf32_cudnn'] is False and runtime['cudnn_benchmark'] is False
           and runtime['cublas_workspace_config']==':4096:8' and runtime['llm_loaded'] is False
           and runtime['no_source_date_repair'] is True,'Runtime contract')
    claim=files.obj(root/'run_claim.json');ensure(claim['manifest_sha256']==expected_manifest_sha256 and claim['resume_allowed'] is False,'Execution claim')
    initial=files.obj(root/'initial_states/manifest.json')
    ensure(initial['schema']=='e6-initial-states-v0' and initial['before_any_training'] is True
           and set(initial['seeds'])=={'1','2','3'},'Initial snapshot coverage')
    alltraining=files.obj(root/'all_training_completed.json')
    ensure((alltraining['n_models'],alltraining['updates'],alltraining['exposures'])==(3,4770,38106)
           and alltraining['test_inference_started'] is False
           and alltraining['initial_manifest_sha256']==summary['initial_manifest_sha256'],'Pre-inference training closure')
    ensure(len(summary['models'])==len(alltraining['models'])==3,'Completed model records')
    head_rows=files.rows(root/'predictions.jsonl');joined=[];model_audits=[]
    for k,seed in enumerate(SEEDS):
        directory=root/'models'/f'seed{seed}_full_head';start=initial['seeds'][str(seed)]
        ensure(start['seed']==seed and start['n_parameters']==367361,'Initial seed/architecture')
        files.track(root/'initial_states'/f'seed{seed}.pt',start['file_sha256'])
        files.track(directory/'initial.pt',start['file_sha256'])
        init_hash=check_checkpoint(directory/'initial.pt',start['tensor_sha256'],True)
        contract=files.obj(directory/'training_contract.json')
        compare({'seed':seed,'model_arm':'full_head','epochs':3,'batch_size':8,'expected_updates':1590,
                 'expected_exposures':12702,'initial':start,'initial_manifest_sha256':summary['initial_manifest_sha256'],
                 'batches_sha256':manifest['files_sha256']['batches.json']},contract,'training contract')
        training=files.obj(directory/'training_completed.json');complete=files.obj(directory/'completed.json')
        compare(training,alltraining['models'][k],'all-trained model record')
        compare(complete,summary['models'][k],'summary completed model record')
        for field,value in training.items():compare(value,complete[field],'training/completed.'+field)
        ensure((training['seed'],training['updates'],training['exposures'])==(seed,1590,12702),'Model update/exposure/seed')
        ensure(training['initial_file_sha256']==start['file_sha256'] and training['initial_tensor_sha256']==init_hash,'Initial state lineage')
        rows=files.rows(directory/'steps.jsonl',training['steps_sha256'])
        epoch_means=check_steps(rows,batches[str(seed)],index,seed)
        compare(epoch_means,training['epoch_example_mean_bce'],'epoch example-weighted mean BCE')
        check_optimizer(training['optimizer_groups'])
        files.track(directory/'head.pt',training['checkpoint_sha256'])
        final_hash=check_checkpoint(directory/'head.pt',training['checkpoint_tensor_sha256'])
        ensure(init_hash!=final_hash,'Final equals initial despite recorded updates')
        model_rows=files.rows(directory/'predictions.jsonl',complete['predictions_sha256'])
        ensure(complete['n_predictions']==len(model_rows)==1755
               and [(r['seed'],r['id']) for r in model_rows]==[(seed,key) for key in ordered['test']],'Per-model evaluation coverage/order')
        joined.extend(model_rows)
        ensure(type(training['train_s']) in (int,float) and math.isfinite(training['train_s']) and training['train_s']>=rows[-1]['elapsed_s'],'Train timing')
        ensure(type(complete['eval_s']) in (int,float) and math.isfinite(complete['eval_s']) and complete['eval_s']>=0
               and training['train_s']+complete['eval_s']<=1200,'Model runtime budget')
        model_audits.append({'seed':seed,'updates':1590,'exposures':12702,'initial_tensor_sha256':init_hash,
                             'checkpoint_tensor_sha256':final_hash,'epoch_example_mean_bce':epoch_means,'n_predictions':1755})
    compare(joined,head_rows,'Aggregate versus per-model rows')
    head,head_parse=predictions(head_rows,items,True);ref,ref_parse=predictions(reference_rows,items,False)
    metrics=recompute_metrics(test,strata,head,ref)
    compare(metrics,scores['metrics'],'All independently recomputed metrics')
    compare({'full/native':ref_parse,'full_head/native':head_parse},scores['parse_fail_rates'],'Parse rates')
    compare({'expected_per_system':5265,'received_head':5265,'received_reference':5265,'n_items':5989,
             'n_train':4234,'n_test':1755,'n_primary':902,'seeds':[1,2,3]},scores['coverage'],'Score coverage')
    compare({'scope':'primary contrast separately per seed only','unit':'paired whole event','n_events':8,
             'draws':5000,'rng_seed':20260925,'quantile_method':'linear','event_order':'sorted string event IDs'},scores['bootstrap'],'Bootstrap contract')
    ensure(type(scores['elapsed_s']) in (int,float) and math.isfinite(scores['elapsed_s']) and 0<=scores['elapsed_s']<=3600,'Overall runtime budget')
    files.track(Path(__file__))
    files.recheck()
    return {'schema':'e6-independent-result-audit-v0','consistent':True,'artifact':str(root),
            'reference_original_root':str(original_root),'reference_prediction_replica_root':str(reference_root),
            'audit_code_sha256':digest(__file__),'manifest_sha256':expected_manifest_sha256,
            'checkpoint_tensors_loaded_and_checked_on_cpu':True,'n_models':3,'n_steps':4770,
            'n_training_exposures':38106,'n_head_answers':5265,'n_reference_answers':5265,
            'primary_n':902,'primary_events':8,'verdict':'descriptive_system_comparison',
            'metrics':metrics,'model_audits':model_audits,'hashes_verified':files.hashes,
            'scope':'Saved local artifact lineage, byte hashes, finite frozen pair/difference and metadata arrays, CPU checkpoint tensor schema/canonical hashes, batch/update/exposure logs, original E5 reference selection, source-label metrics and seed-specific paired-event bootstrap.',
            'limitations':['No optimization or inference replay; logged gradient finiteness/optimizer execution and historical ordering are saved attestations, not independently observed execution.',
                           'The head architecture source is pinned but not imported; initialization reproducibility from RNG is not regenerated. Saved initial CPU tensors and clones are checked.',
                           'Independent E5 audit is a pinned upstream prerequisite, not rerun by this auditor. Original full predictions are read and subset equality verified.',
                           'Eight exposed development events; no LLM necessity, temporal reasoning, physical onset, causal damage or memory-method inference.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--expected-manifest-sha256',required=True)
    parser.add_argument('--reference-artifact',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    ensure(not args.out.exists(),'Audit output already exists; preserve previous evidence')
    try:
        report=audit(args.artifact,args.expected_manifest_sha256,args.reference_artifact)
    except Exception as exc:
        report={'schema':'e6-independent-result-audit-v0','consistent':False,'artifact':str(args.artifact.resolve()),
                'audit_code_sha256':digest(__file__),'error_type':type(exc).__name__,'reason':str(exc)}
    report['audited_at']=datetime.now(timezone.utc).isoformat()
    with args.out.open('x') as stream:json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({'consistent':report['consistent'],'out':str(args.out),'reason':report.get('reason')},ensure_ascii=False))
    return 0 if report['consistent'] else 2


if __name__=='__main__':sys.exit(main())
