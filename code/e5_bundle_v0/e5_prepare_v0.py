#!/usr/bin/env python3
"""CPU-only frozen E5 preparation. No selection, exclusions, training or GPU calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np

N_ITEMS, N_TRAIN, N_TEST, N_CACHES = 5989, 4234, 1755, 3756
KINDS = {'train|flood|pos':1066,'train|flood|neg':1066,'train|flood|hard_neg':1066,
         'train|landslide|pos':518,'train|landslide|neg':518,
         'test|flood|pos':457,'test|flood|neg':457,'test|flood|hard_neg':457,
         'test|landslide|pos':192,'test|landslide|neg':192}
REQUIRED_PARENTS = {'c0_manifest.json','items.jsonl','c1_quality.jsonl','c1_quality_manifest.json',
                    'c1_results.json','c1_selected_ids.json','e3_items.jsonl','e4_manifest.json'}
SOURCE_FILES = ['e5_prepare_v0.py','e5_train_v0.py','e5_scoring_v0.py','run_e5_when_idle_v0.py']


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def stat(path):
    s=Path(path).stat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns}


def unique(items):
    out={}
    for item in items:
        require(isinstance(item.get('id'),str) and item['id'] not in out,'Missing/duplicate item ID')
        out[item['id']]=item
    return out


def rooted(path,root,original_root=None):
    path=Path(path)
    if path.is_absolute():
        path=path.relative_to(Path(original_root) if original_root is not None else root)
    result=(root/path).resolve()
    require(result.is_relative_to(root.resolve()),'Source path escapes root')
    return result


def validate_population(items,c0):
    by_id=unique(items)
    require(len(items)==N_ITEMS and c0['schema']=='c0-prepared-v1','Unexpected C0 population/schema')
    require(Counter(f"{x['partition']}|{x['phen']}|{x['kind']}"for x in items)==KINDS,'Population or source kind counts changed')
    counts=Counter(f"{x['partition']}|{x['phen']}|{x['answer']}"for x in items)
    tuples=Counter(f"{x['partition']}|{x['phen']}|{x['cluster']}|{x['kind']}|{x['answer']}"for x in items)
    require(dict(counts)==c0['counts'] and dict(tuples)==c0['tuple_counts'],'Frozen C0 count mismatch')
    require({x['cache_path']for x in items}==set(c0['cache_sha256'])==set(c0['cache_stat'])
            and len(c0['cache_sha256'])==c0['n_unique_caches']==N_CACHES,'Frozen cache coverage mismatch')
    ordered={part:[x['id']for x in items if x['partition']==part]for part in ('train','test')}
    require(len(ordered['train'])==N_TRAIN and len(ordered['test'])==N_TEST,'Train/test count mismatch')
    sets={}
    for part in ('train','test'):
        ids=sorted(ordered[part]);h=hashlib.sha256(json.dumps(ids,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        require(h==c0['population_ids_sha256'][part],'Frozen source ID set mismatch')
        subset=[x for x in items if x['partition']==part]
        sets[part]={'ids':set(ordered[part]),'tiles':{x['tile']for x in subset},
                    'flood_events':{str(x['event'])for x in subset if x['phen']=='flood'},
                    'landslide_regions':{x['tile'].split('_s2_')[0]for x in subset if x['phen']=='landslide'},
                    'landslide_folds':{x['fold']for x in subset if x['phen']=='landslide'}}
    for field in sets['train']:require(not sets['train'][field]&sets['test'][field],'Train/test leakage: '+field)
    by_tile=defaultdict(list)
    for item in items:
        require(item['type']=='Q1' and item['answer']==('yes'if item['kind']=='pos'else'no'),'Source answer/type changed')
        indices=item['indices'];limit=3 if item['phen']=='flood'else 12
        require(isinstance(indices,list) and len(indices)==2 and all(type(i)is int and 0<=i<limit for i in indices)
                and indices[0]<indices[1],'Invalid frozen frame indices')
        require(isinstance(item['dates'],list) and len(item['dates'])==2,'Missing frozen dates')
        if item['phen']=='flood':
            slots=['pre_1','pre_2']if item['kind']=='neg'else['pre_2','post']
            require(item['slots']==slots and indices==([0,1]if item['kind']=='neg'else[1,2]),'Flood kind/slot/index mismatch')
            require(str(item['event'])==str(item['cluster']) and item['fold']==item['partition'],'Flood split/event mismatch')
        else:
            require(item['kind']in ('pos','neg') and item['fold']==item['cluster']
                    and item['fold']=='holdout_'+item['tile'].split('_s2_')[0],'Landslide region mismatch')
        by_tile[(item['phen'],item['tile'])].append(item)
    for its in by_tile.values():
        kinds={x['kind']for x in its}
        require((len(its)==2 and kinds=={'pos','neg'})or(len(its)==1 and kinds=={'hard_neg'}),'Incomplete/overlapping source pair')
    return ordered, {p:{key:sorted(value)for key,value in ss.items()if key!='ids'}for p,ss in sets.items()}


def make_batches(train_ids,seeds=(1,2,3),epochs=3,batch_size=8):
    require(len(train_ids)==len(set(train_ids)) and train_ids,'Empty/duplicate train IDs')
    batches={}
    for seed in seeds:
        rng=np.random.default_rng(seed);schedule=[]
        for epoch in range(epochs):
            shuffled=[train_ids[int(i)]for i in rng.permutation(len(train_ids))]
            split=[shuffled[i:i+batch_size]for i in range(0,len(shuffled),batch_size)]
            require([i for batch in split for i in batch]==shuffled,'Batch construction mismatch')
            schedule.append(split)
        batches[str(seed)]=schedule
    return batches


def make_eval_sets(items,parents):
    by_id=unique(items);test=[x for x in items if x['partition']=='test']
    target={x['id']:x for x in test if x['phen']=='flood'and x['kind']in ('pos','hard_neg')}
    quality=unique(parents['c1_quality.jsonl']);qm=parents['c1_quality_manifest.json'];c1=parents['c1_results.json']
    require(set(quality)==set(target) and len(target)==914,'C1 quality target coverage mismatch')
    require(c1['valid']is True,'C1 comparison is invalid')
    eligible=set();groups=defaultdict(list)
    for key,item in target.items():
        q=quality[key]
        require(all(q[k]==item[k]for k in ('tile','kind','dates','slots')) and str(q['event'])==str(item['event'])
                and q['source_answer']==item['answer'] and q['source_qa_flood_frac']==item['flood_frac'],'C1 quality metadata mismatch')
        require(type(q['eligible_symmetric_quality'])is bool and q['eligible_symmetric_quality']==(q['valid_frac']>=.9)
                and 0<=q['valid_frac']<=1 and 0<=q['flood_frac']<=1,'C1 quality eligibility mismatch')
        require((item['kind']=='pos'and item['answer']=='yes'and q['flood_frac']>=.02)or
                (item['kind']=='hard_neg'and item['answer']=='no'and q['flood_px']==0 and q['valid_frac']>=.9),'C1 source label mismatch')
        if q['eligible_symmetric_quality']:
            eligible.add(key);groups[(str(item['event']),tuple(item['dates']),tuple(item['slots']))].append(item)
    supported={x['id']for xs in groups.values()if {x['kind']for x in xs}=={'pos','hard_neg'}for x in xs}
    require(len(supported)==902 and Counter(by_id[i]['kind']for i in supported)=={'pos':445,'hard_neg':457}
            and len({str(by_id[i]['event'])for i in supported})==8,'Primary population differs from frozen C1')
    require(parents['c1_selected_ids.json']=={'quality_symmetric':sorted(eligible),'all_original_targets':sorted(target)}
            and c1['quality_excluded_ids']==sorted(set(target)-eligible),'C1 eligible ID sets changed')
    subset=c1['subsets']['quality_symmetric']
    require(subset['supported_n_pos']==445 and subset['supported_n_hard_neg']==457 and subset['n_events']==8,'C1 reported support changed')
    for seed in ('1','2','3'):
        for arm in ('reader','blind'):
            recorded={i for event in subset['per_seed'][seed]['arms'][arm].values()for s in event['strata']for k in ('pos_ids','hard_neg_ids')for i in s[k]}
            require(recorded==supported,'C1 saved stratum membership mismatch')
    e3=unique(parents['e3_items.jsonl'])
    require(len(e3)==209 and set(e3)<=set(x['id']for x in test),'E3 subset outside frozen test population')
    for key,row in e3.items():
        require(all(row[k]==by_id[key][k]for k in ('tile','phen','kind','dates','indices','answer','cluster')),'E3 item metadata mismatch')
    return {'primary_same_prompt':[x['id']for x in test if x['id']in supported],
            'all_test':[x['id']for x in test],
            'paired_flood':[x['id']for x in test if x['phen']=='flood'and x['kind']in ('pos','neg')],
            'hard_negative_flood':[x['id']for x in test if x['phen']=='flood'and x['kind']=='hard_neg'],
            'landslide':[x['id']for x in test if x['phen']=='landslide'],
            'e3_subset':[x['id']for x in test if x['id']in e3]}


def build_pairs(items,c0,out_path,root,progress=None):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(2)
    cache_rows=defaultdict(list)
    for row,item in enumerate(items):cache_rows[item['cache_path']].append((row,item))
    require(set(cache_rows)==set(c0['cache_sha256'])==set(c0['cache_stat']),'Cache manifest coverage mismatch')
    matrix=np.lib.format.open_memmap(out_path,mode='w+',dtype=np.float32,shape=(len(items),2,64,768))
    audits={}
    try:
        for count,(original,indexed)in enumerate(sorted(cache_rows.items()),1):
            path=rooted(original,root,c0['root']);before=stat(path)
            require(before==c0['cache_stat'][original],'Cache stat changed since C0: '+original)
            require(sha(path)==c0['cache_sha256'][original] and stat(path)==before,'Frozen cache changed before pooling: '+original)
            array=np.load(path,mmap_mode='r',allow_pickle=False)
            phenomena={it['phen']for _,it in indexed};require(len(phenomena)==1,'One cache assigned to multiple sensors')
            flood=next(iter(phenomena))=='flood';expected=(3,768,48,48)if flood else(12,768,32,32)
            require(array.shape==expected and array.dtype==np.float16,'Unexpected frozen cache shape/dtype: '+original)
            used=sorted({i for _,it in indexed for i in it['indices']})
            require(all(type(i)is int and 0<=i<expected[0]for i in used),'Frozen cache index out of range')
            # Pool each referenced frame once; shared source rows retain their exact order.
            frames=np.asarray(array[used],dtype=np.float32)
            require(np.isfinite(frames).all(),'Nonfinite referenced source frame: '+original)
            tensor=torch.from_numpy(frames)
            pooled=F.avg_pool2d(tensor,6 if flood else 4).flatten(2).permute(0,2,1).contiguous().numpy()
            require(pooled.dtype==np.float32 and pooled.shape==(len(used),64,768) and np.isfinite(pooled).all(),'Invalid pooled source frame')
            frame_map={index:j for j,index in enumerate(used)}
            for row,item in indexed:matrix[row]=pooled[[frame_map[i]for i in item['indices']]]
            del array,tensor,frames,pooled
            after=stat(path);observed=sha(path)
            require(after==before==stat(path) and observed==c0['cache_sha256'][original],'Cache mutated during pooling: '+original)
            audits[original]={'sha256':observed,'before':before,'after':after,'indices_used':used,
                              'item_rows':[row for row,_ in indexed],'finite_scope':'all referenced frames'}
            if progress is not None and (count%100==0 or count==len(cache_rows)):progress(count,len(cache_rows))
        matrix.flush()
    finally:
        del matrix
    for original,audit in audits.items():
        require(stat(rooted(original,root,c0['root']))==audit['after'],'Cache stat changed later in preparation: '+original)
    return audits


def compare_pool_rows(items,pairs_path,reference,pair_path):
    by_id={item['id']:(row,item)for row,item in enumerate(items)}
    require(len(by_id)==len(items) and len(unique(reference))==len(reference),'Pool comparison ID duplication')
    matrix=np.load(pairs_path,mmap_mode='r',allow_pickle=False)
    with np.load(pair_path,allow_pickle=False)as archive:
        require(set(archive.files)=={i['pair_key']for i in reference},'E4 pair archive coverage mismatch')
        for item in reference:
            require(item['id']in by_id,'E4 reproduction item missing from C0')
            row,current=by_id[item['id']]
            for key in ('tile','phen','kind','dates','indices','answer','cluster','slots'):
                require(item.get(key)==current.get(key),'E4 reproduction metadata mismatch: '+key)
            require(np.array_equal(matrix[row],archive[item['pair_key']]),'Pooled values differ from frozen E4: '+item['id'])
    del matrix
    return len(reference)


def pool_reproduction(items,pairs_path,cfg,root):
    source=rooted(cfg['directory'],root);item_path=source/'items.jsonl';pair_path=source/'pairs.npz'
    require(sha(item_path)==cfg['items_sha256'] and sha(pair_path)==cfg['pairs_sha256'],'E4 pooled source pins changed')
    reference=rows(item_path)
    require(len(unique(reference))==len(reference)==cfg['expected_items']==209,'E4 reproduction population mismatch')
    compare_pool_rows(items,pairs_path,reference,pair_path)
    require(sha(item_path)==cfg['items_sha256'] and sha(pair_path)==cfg['pairs_sha256'],'E4 sources changed during reproduction')
    return {'expected_items':209,'matched_items':209,'bit_exact':True,'items_sha256':cfg['items_sha256'],'pairs_sha256':cfg['pairs_sha256']}


def prepare(config,inputs,out,source_dir,root):
    config,inputs,out,source_dir,root=map(lambda p:Path(p).resolve(),(config,inputs,out,source_dir,root))
    require(out==root/'e5_equal_budget_v0','Only dedicated E5 output permitted')
    require(Path(__file__).resolve()==source_dir/'e5_prepare_v0.py','Execute preparation from the declared source bundle')
    require(not out.exists(),'Output exists: never resume or overwrite preparation')
    cfg=read(config);config_sha=sha(config)
    require(cfg['schema']=='e5-equal-budget-prereg-v0' and cfg['output_directory']=='e5_equal_budget_v0','Unknown E5 plan/output')
    require({k:cfg['population'][k]for k in ('n_items','n_train','n_test','n_unique_caches')}==
            {'n_items':N_ITEMS,'n_train':N_TRAIN,'n_test':N_TEST,'n_unique_caches':N_CACHES},'Preregistered population mismatch')
    require(cfg['inputs']['pair_shape']==[2,64,768] and cfg['inputs']['dtype']=='float32','Input tensor contract mismatch')
    require(cfg['source_files']==SOURCE_FILES and REQUIRED_PARENTS<=set(cfg['parents']),'Missing source/input freeze contract')
    training=cfg['training']
    for key,value in {'epochs':3,'batch_size':8,'seeds':[1,2,3],'model_arms':['full','pair','later','delta'],
                      'expected_updates_per_model':1590,'expected_exposures_per_model':12702}.items():
        require(training[key]==value,'Training budget contract mismatch: '+key)
    source_pins={name:sha(source_dir/name)for name in SOURCE_FILES}
    out.mkdir(parents=True,exist_ok=False);phase='freezing_inputs'
    try:
        frozen=out/'parent_snapshot';frozen.mkdir();parents={}
        for filename,h in cfg['parents'].items():
            require(Path(filename).name==filename,'Parent input filename must be local to bundle')
            require(sha(inputs/filename)==h,'Frozen bundled input changed: '+filename)
            shutil.copyfile(inputs/filename,frozen/filename)
            require(sha(frozen/filename)==h,'Copied bundled input changed: '+filename)
            parents[filename]=rows(frozen/filename)if filename.endswith('.jsonl')else read(frozen/filename)
        c0=parents['c0_manifest.json'];items=parents['items.jsonl'];qm=parents['c1_quality_manifest.json'];c1=parents['c1_results.json']
        require(cfg['parents']['items.jsonl']==c0['items_sha256']==qm['items_sha256'],'C0/C1 item bytes differ')
        require(cfg['parents']['c1_quality.jsonl']==qm['quality_sha256']==c1['provenance']['quality_sha256']
                and cfg['parents']['c1_quality_manifest.json']==c1['provenance']['quality_manifest_sha256']
                and cfg['parents']['c1_selected_ids.json']==c1['provenance']['selected_ids_sha256'],'C1 quality/selection hash linkage mismatch')
        ordered,split_audit=validate_population(items,c0);eval_sets=make_eval_sets(items,parents)
        batches=make_batches(ordered['train'])
        require(all(sum(len(e)for e in epochs)==1590 and sum(len(b)for e in epochs for b in e)==12702 for epochs in batches.values()),'Unequal update/exposure budget')
        phase='verifying_current_sources';qa_pins={}
        for rel,h in c0['source_sha256'].items():
            path=rooted(rel,root);require(sha(path)==h,'Current C0 QA/contract source changed: '+rel);qa_pins[str(path)]=h
        llm_original=parents['e4_manifest.json']['llm_files_sha256'];llm_pins={}
        for original,h in llm_original.items():
            path=rooted(original,root,c0['root']);require(sha(path)==h,'Frozen LLM/tokenizer changed: '+original);llm_pins[str(path)]=h
        llm_dir=root/'olmo_llm/Olmo-3-7B-Instruct'
        require(llm_pins and all(Path(p).parent==llm_dir for p in llm_pins),'Unexpected LLM source directory')
        shutil.copyfile(frozen/'items.jsonl',out/'items.jsonl');shutil.copyfile(config,out/'prereg.json')
        require(sha(out/'items.jsonl')==c0['items_sha256'] and sha(out/'prereg.json')==config_sha,'Copied plan/items changed')
        write(out/'ordered_ids.json',ordered);write(out/'batches.json',batches);write(out/'eval_sets.json',eval_sets)
        phase='pooling_frozen_caches'
        caches=build_pairs(items,c0,out/'pairs.npy',root,lambda n,total:print(f'E5 CPU caches {n}/{total}',flush=True))
        phase='reproducing_parent_pools';reproduction=pool_reproduction(items,out/'pairs.npy',cfg['pool_reproduction'],root)
        audit={'n_items':N_ITEMS,'n_unique_caches':len(caches),'cache_sources':caches,'split_disjointness':split_audit,
               'source_qa_sha256':qa_pins,'pool_reproduction':reproduction,'eval_set_counts':{k:len(v)for k,v in eval_sets.items()},
               'pooling':'original fp16 referenced frames -> float32 torch CPU avg_pool2d(S1=6,S2=4), flattened to64x768',
               'batch_index_basis':'item ID strings in train ordered_ids; one RNG per seed, consecutive epoch permutations',
               'current_source_verification':'All3756 cache SHA/stat checked during this E5 preparation; not an assertion about historical E2 extraction lineage',
               'no_exclusions':True,'device':'cpu','torch_threads':2}
        write(out/'input_audit.json',audit);snapshot=out/'code_snapshot';snapshot.mkdir()
        for name,h in source_pins.items():
            require(sha(source_dir/name)==h,'Source code changed during preparation: '+name)
            shutil.copyfile(source_dir/name,snapshot/name);require(sha(snapshot/name)==h,'Snapshot code changed')
        for path,h in {**qa_pins,**llm_pins}.items():require(sha(path)==h,'QA/model source changed during preparation: '+path)
        for filename,h in cfg['parents'].items():require(sha(inputs/filename)==h and sha(frozen/filename)==h,'Bundled source changed during preparation')
        require(sha(config)==config_sha,'Preregistration changed during preparation')
        for original,record in caches.items():
            require(stat(rooted(original,root,c0['root']))==record['after'],'Cache stat changed before final freeze: '+original)
        files=['items.jsonl','pairs.npy','ordered_ids.json','batches.json','eval_sets.json','prereg.json','input_audit.json']
        manifest={'schema':'e5-equal-budget-prepared-v0','prepared_at':now(),'files_sha256':{f:sha(out/f)for f in files},
                  'parent_snapshot_sha256':cfg['parents'],'code_snapshot_sha256':source_pins,'llm_dir':str(llm_dir),'llm_files_sha256':llm_pins,
                  'n_items':N_ITEMS,'n_train':N_TRAIN,'n_test':N_TEST,'pairs_shape':[N_ITEMS,2,64,768],'pairs_dtype':'float32',
                  'pair_row_order':'items.jsonl line order, unchanged from frozen C0-v1','batches_reference':'ordered_ids.train item IDs',
                  'expected_updates_per_model':1590,'expected_exposures_per_model':12702,
                  'environment':{'python':sys.version,'numpy':np.__version__,'torch':sys.modules['torch'].__version__,'device':'cpu'}}
        write(out/'manifest.json',manifest);write(out/'status.json',{'status':'prepared','at':now()});return manifest
    except BaseException as exc:
        write(out/'failure.json',{'phase':phase,'at':now(),'error':repr(exc),'traceback':traceback.format_exc()})
        write(out/'status.json',{'status':'invalid_preparation','at':now(),'phase':phase,'error':repr(exc)});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/home/work/data/olmoearth'))
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,default=Path(__file__).parent);parser.add_argument('--out',type=Path)
    args=parser.parse_args();destination=args.out or args.root/'e5_equal_budget_v0'
    result=prepare(args.config,args.inputs,destination,args.source_dir,args.root)
    print(json.dumps({'out':str(destination),'n_items':result['n_items'],'n_train':result['n_train'],'n_test':result['n_test']}))
