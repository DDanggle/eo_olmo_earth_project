"""Materialize only training targets and validate real arrays on CPU in a new directory."""
import argparse,hashlib,importlib.util,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def require(ok,msg):
 if not ok:raise ValueError(msg)
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def mod(p,name):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def verify_export(root):
 for f in read(root/'export_manifest.json')['files']:
  p=(root/f['path']).resolve()
  require(p.is_relative_to(root.resolve()) and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],'catalog export integrity')
def main(a):
 guard=mod(a.code/'audit_inputs_reference.py','source_protection')
 before=guard.protect(a.experiment_root)
 verify_export(a.catalog)
 receipt=read(a.catalog/'preparation_receipt.json')
 builderpath=a.code/'episode_builder_reference.py';loaderpath=a.code/'episode_loader_reference.py'
 require(sha(builderpath)==receipt['pinned_sources']['builder']['sha256'],'builder changed')
 require(sha(loaderpath)==receipt['pinned_sources']['loader']['sha256'],'loader changed')
 require(sha(a.prepared/'manifest.jsonl')==receipt['pinned_sources']['manifest']['sha256'],'manifest changed')
 plan=rows(a.catalog/'training_target_plan.jsonl');public=rows(a.catalog/'episodes_train.jsonl')
 byid={r['patch_id']:r for r in rows(a.prepared/'manifest.jsonl')}
 require(len(plan)==len(public)==384,'catalog count')
 for r in plan:
  m=byid[r['query_patch_id']]
  require(m['training_partition']=='train_pool' and m['supervised_training_allowed'],'nontrain target')
  require(r['query_label_npz']==m['label_path'] and r['expected_query_label_sha256']==m['label_sha256'],'query label lineage')
  require({r['target_class'],r['counter_class']}=={8,14},'class selection')
 for e in public:
  for pair in e['support_pairs']:
   for s in pair.values():require(byid[s['patch_id']]['training_partition']=='train_pool','bank support')
 require(not a.out.exists(),'never overwrite runtime package')
 shutil.copytree(a.catalog,a.out)
 (a.out/'export_manifest.json').unlink()
 (a.out/'scoring').mkdir()
 builder=mod(builderpath,'pinned_source_builder')
 scoring=builder.write_scoring(a.prepared,a.out/'scoring/scoring_train.jsonl',plan)
 require(sha(a.out/'episodes_train.jsonl')==receipt['public_sha256'],'public changed during targets')
 loader_mod=mod(loaderpath,'pinned_source_loader')
 loader=loader_mod.EpisodeLoader(a.prepared,a.out,'train')
 # Validate all 48 unique packet arrays once without any EO/VLM model.
 packets={}
 for e in public:
  q=e['query_patch_id']
  if q not in packets:
   packet,_=loader._read_packet(q);packets[q]=True
 require(len(packets)==48,'48 packets')
 # Exercise actual model-facing loader at K8 for one fixed query per source parent.
 smoke=[]
 for parent in ('t31tfj','t32ulu'):
  e=min((e for e in public if e['k_pairs']==8 and e['query_parent_tile']==parent),key=lambda e:e['episode_id'])
  result=loader.load(e['episode_id'])
  require(result['audit']['returned_query_observations']==2,'query dates')
  require(result['audit']['returned_support_observation_instances']==128,'support ledger')
  target=loader.training_target(e['episode_id'],training=True)
  smoke.append({'episode_id':e['episode_id'],'query_parent':parent,'query_dates':2,'support_date_instances':128,'training_target_return_type':type(target).__name__})
 shutil.copyfile(a.out/'preparation_receipt.json',a.out/'reference_preparation_receipt.json')
 receipt.update(training_targets_materialized=True,actual_array_loader_runtime_checked=True,support_mask_reference_root=str(a.out.resolve()))
 write(a.out/'preparation_receipt.json',receipt)
 contract=read(a.out/'episode_contract.json');contract['status']='train_targets_and_cpu_loader_verified_not_trained';write(a.out/'episode_contract.json',contract)
 protection=guard.protect(a.experiment_root);require(protection==before,'protected source changed')
 write(a.out/'runtime_receipt.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'status':'train_targets_and_real_cpu_loader_verified_not_trained','query_packets_validated':48,'training_target_rows':len(plan),'scoring_counts':scoring,'real_k8_smoke':smoke,'protected_before':before,'protected_after':protection,'query_gold_access':'training supervision only, after public catalog hash fixed','source_bank_training':False,'development_or_reserved_labels_opened':0,'new_gpu_seconds':0,'script_sha256':sha(Path(__file__)),'original_catalog_manifest_sha256':sha(a.catalog/'export_manifest.json')})
 write(a.out/'export_manifest.json',{'files':[{'path':str(p.relative_to(a.out)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(a.out.rglob('*')) if p.is_file()]})
 print(json.dumps({'query_packets_validated':48,'training_targets':384,'k8_cases':smoke,'scoring_counts':scoring,'new_gpu_seconds':0}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--experiment-root',type=Path,required=True);main(p.parse_args())

