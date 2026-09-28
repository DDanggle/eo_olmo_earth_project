"""Verify returned files and describe completed paired results without a P2 decision."""
import argparse,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main(a):
 m=read(a.export/'export_manifest.json');seen=set()
 for row in m['files']:
  rel=Path(row['path']);p=(a.export/rel).resolve()
  if rel.is_absolute() or '..' in rel.parts or not p.is_relative_to(a.export.resolve()) or row['path'] in seen:raise ValueError('invalid export path')
  seen.add(row['path'])
  if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('export hash mismatch: '+str(rel))
 c=read(a.export/a.collector);runs=[]
 for r in c.get('verified_runs',c.get('runs',[])):
  d=a.export/'training_v0'/f"{r['arm']}_{r['seed']}_train"
  receipt=read(d/'receipt.json');score=read(d/'score_step_002304.json')
  if sha(d/'receipt.json')!=r['run_receipt_sha256']:raise ValueError('collector/receipt mismatch')
  if receipt['status']!='training_completed' or receipt['completed_updates']!=2304:raise ValueError('not completed')
  curve=score['target_iou_by_k'];auc=sum((b-b0)*(curve[str(b0)]+curve[str(b)])/2 for b0,b in ((1,2),(2,4),(4,8)))/7
  if abs(auc-score['target_iou_auc'])>1e-12:raise ValueError('AUC mismatch')
  windows=r['training_adequacy']['last_evaluation_windows'];last=[x['auc'] for x in windows[-3:]];rng=max(last)-min(last)
  runs.append({'arm':r['arm'],'seed':r['seed'],'final_auc':auc,'last3_range':rng,'stability_pass':len(last)==3 and rng<=.005,
   'elapsed_seconds':receipt['elapsed_seconds'],'checkpoint_sha256_verified_by_server_collector':r['checkpoint_sha256'],
   'checkpoint_reloaded_or_reinferred':False,'last_windows':windows[-3:]})
 paired=[]
 for seed in sorted({r['seed'] for r in runs}):
  group={r['arm']:r for r in runs if r['seed']==seed}
  if set(group)=={'B0','B2'}:
   paired.append({'seed':seed,'B0':group['B0']['final_auc'],'B2':group['B2']['final_auc'],'delta':group['B2']['final_auc']-group['B0']['final_auc'],
   'both_stability_pass':all(r['stability_pass'] for r in group.values())})
 result={'created_utc':datetime.now(timezone.utc).isoformat(),'export':str(a.export),'code_sha256':sha(Path(__file__)),
 'files':len(m['files']),'bytes':sum(r['bytes'] for r in m['files']),'npz_files':sum(r['path'].endswith('.npz') for r in m['files']),
 'export_manifest_sha256':sha(a.export/'export_manifest.json'),'hash_errors':0,'verified_completed_runs':len(runs),'runs':runs,'paired':paired,
 'paired_mean_delta_descriptive_only':math.fsum(r['delta'] for r in paired)/len(paired) if paired else None,
 'full_p2_gate_executed':False,'original_cohort_and_failures_preserved':True,'new_gpu_seconds':0,
 'limits':['Completed checkpoint/prediction/log hashes verified by pinned server collector; checkpoints not deserialized here.',
 'No independent region, human correction or generated-language evaluation; data exposure matched but GPU time is not.',
 'Partial completed pairs do not establish the frozen three-seed decision or adequately trained superiority.']}
 with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--export',type=Path,required=True);p.add_argument('--collector',required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
