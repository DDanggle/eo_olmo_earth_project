"""Prepare additional class8/14 training references, not an evaluation or training run.
Pinned existing metadata builder; no input NPZ or query label is opened here.
"""
import argparse,copy,hashlib,importlib.util,json,shutil
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

EXPECTED={
 'builder':'64c6df1fe10b40da8cc076286a6184be05ec0db261a4462af0548aaba8eff6d6',
 'manifest':'13a8d23794d5b4a347aca975153aa41f08af6d6fae82a01868705e6cfbbcabc3',
 'objects':'c8b6ce724eaf3acc1e1f159ab2ec46943bb88eb49cde1aa0172e12181da53e11',
 'quality':'5c62127b4304eb310b83360f2ebf53c31428c5189984926061d846d445a2a6a1',
 'loader':'536de03b77bcd5bfedf439b5456d9e0fe60a3789c5f9c1a53c5511ded05d07f6'}
CLASSES=[8,14]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def write_rows(p,v):Path(p).write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in v))
def require(v,msg):
 if not v:raise ValueError(msg)
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def make(builder,manifest,objects):
 train=[r for r in manifest if r['training_partition']=='train_pool']
 require(len(train)==48 and len({r['patch_id'] for r in train})==48,'48 distinct train queries required')
 require(all(r['parent_tile'] in ('t31tfj','t32ulu') and r['supervised_training_allowed'] for r in train),'training role')
 pool=[o for o in objects if o['class_id'] in CLASSES and o['episode_role']=='train_pool']
 require(len(pool)==243 and len({o['object_key'] for o in pool})==243,'243 unique training source objects')
 validids={r['patch_id'] for r in train}
 require(all(o['patch_id'] in validids for o in pool),'sourcebank not allowed')
 inputs={r['patch_id']:{'dates':r['selected_dates'],'input_sha256':r['npz_sha256']} for r in train}
 roles={r['patch_id']:'train_pool' for r in train}
 public,private,coverage=builder.build_catalog(train,roles,inputs,pool,[],CLASSES,'train')
 require(len(public)==384 and len(private)==384,'384 generated episodes required')
 validate(public,private,manifest,objects)
 return public,private,coverage

def validate(public,private,manifest,objects):
 qmap={r['patch_id']:r for r in manifest};omap={o['object_key']:o for o in objects}
 gold={r['episode_id']:r for r in private}
 require(len(gold)==len(public)==384,'duplicate/missing plan entries')
 require(len({e['episode_id'] for e in public})==384,'duplicate episode IDs')
 grouped={}
 forbidden={'target_class','counter_class','target_present','target_pixels','query_label_npz','query_label_sha256','class_pixel_counts'}
 for e in public:
  require(e['split']=='train' and qmap[e['query_patch_id']]['training_partition']=='train_pool','query not training')
  require(not forbidden.intersection(e) and e['query_label_supplied'] is False,'query gold in public')
  p=gold[e['episode_id']];target,counter=p['target_class'],p['counter_class']
  require({target,counter}==set(CLASSES),'unexpected classes')
  require(p['query_patch_id']==e['query_patch_id'] and p['k_pairs']==e['k_pairs'],'plan identity')
  grouped[e['query_patch_id'],target,e['k_pairs']]=e
  keys=[]
  for kind,cls in [('positive',target),('counterexample',counter)]:
   counts=Counter()
   for pair in e['support_pairs']:
    s=pair[kind];o=omap[s['object_key']];keys.append(o['object_key'])
    require(o['class_id']==cls and o['episode_role']=='train_pool','wrong support class/role')
    require(s['patch_id']!=e['query_patch_id'] and qmap[s['patch_id']]['training_partition']=='train_pool','support query/bank leakage')
    require(s['mask_sha256']==o['mask_sha256'],'mask hash lineage')
    counts[s['patch_id']]+=1
   require(max(counts.values())<=2,'per-patch cap')
  require(len(keys)==len(set(keys))==2*e['k_pairs'],'duplicated support')
 for (qid,c,k),e in grouped.items():
  require(e['support_pairs']==grouped[qid,c,8]['support_pairs'][:k],'K prefix changed')
  reverse=grouped[qid,14 if c==8 else 8,k]
  require(all(a['positive']==b['counterexample'] and a['counterexample']==b['positive'] for a,b in zip(e['support_pairs'],reverse['support_pairs'])),'reverse mismatch')
 require(len({e['query_patch_id'] for e in public})==48,'queries removed')

def run(a):
 paths={'builder':a.builder,'manifest':a.bundle/'manifest.jsonl','objects':a.bundle/'episodes/source_objects.jsonl','quality':a.bundle/'quality_policy.json','loader':a.loader}
 for key,p in paths.items():require(sha(p)==EXPECTED[key],'pinned source mismatch: '+key)
 b=module(a.builder,'pinned_episode_builder');manifest=rows(paths['manifest']);objects=rows(paths['objects'])
 public,private,coverage=make(b,manifest,objects)
 # Counterfactual metadata test: query-gold content/references must not change public rows.
 poisoned=copy.deepcopy(manifest)
 for r in poisoned:
  r['class_pixel_counts']=[-999]*20;r['label_path']='labels/poisoned.npz';r['label_sha256']='0'*64
 require(make(b,poisoned,objects)[0]==public,'query gold affects public catalog')
 a.out.mkdir(parents=True,exist_ok=False);(a.out/'support_masks').mkdir()
 write_rows(a.out/'episodes_train.jsonl',public)
 publichash=sha(a.out/'episodes_train.jsonl')
 write(a.out/'public_catalog_hashes_before_training_targets.json',{'train':publichash})
 write_rows(a.out/'training_target_plan.jsonl',private)
 keys={s['object_key'] for e in public for pair in e['support_pairs'] for s in pair.values()}
 omap={o['object_key']:o for o in objects}
 for key in sorted(keys):
  o=omap[key];src=(a.bundle/'episodes'/o['mask_npz']).resolve()
  require(src.is_relative_to((a.bundle/'episodes/support_masks').resolve()),'mask path')
  require(sha(src)==o['mask_sha256'],'source mask changed')
  shutil.copyfile(src,a.out/o['mask_npz'])
 require(sha(a.out/'episodes_train.jsonl')==publichash,'public catalog mutated')
 contract={'status':'train_references_prepared_runtime_and_targets_pending','prepared_manifest_sha256':EXPECTED['manifest'],'public_catalog_sha256':{'train':publichash},'initial_candidate_positions':[2,5],'all_eight_support_observations_must_be_counted_in_cost_ledger':True,'selected_classes':CLASSES,'ks':[1,2,4,8],'query_count':48,'episodes':384,'original_p2_replaced':False,'source_bank_used_for_encoder_training':False,'query_gold_opened':False,'development_catalog_created':False,'historical_pretraining_exposure_unknown':True}
 write(a.out/'episode_contract.json',contract);write(a.out/'coverage.json',coverage)
 loader=module(a.loader,'pinned_loader')
 loaded=loader.EpisodeLoader(a.bundle,a.out,'train')
 require(len(loaded.episodes)==384,'loader metadata validation')
 receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(Path(__file__)),'pinned_sources':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},'query_count':48,'directed_class_pairs':2,'episodes':384,'unique_support_objects_copied':len(keys),'source_bank_training_references':0,'query_gold_files_opened':0,'raw_packets_opened':0,'metadata_loader_validation_pass':True,'actual_array_loader_runtime_checked':False,'training_targets_materialized':False,'new_gpu_seconds':0,'support_mask_reference_root':str(a.out.resolve()),'public_sha256':publichash,'knowledge_cards_for_classes8_14_available':False,'scope':'Additional two-class training reference pack; 384 rows are 48 query images with directed/K variants, not independent observations or human corrections. No expanded evaluation or performance result.'}
 write(a.out/'preparation_receipt.json',receipt)
 files=[{'path':str(p.relative_to(a.out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.out.rglob('*')) if p.is_file()]
 write(a.out/'export_manifest.json',{'files':files})
 print(json.dumps(receipt))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--builder',type=Path,required=True);p.add_argument('--loader',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())

