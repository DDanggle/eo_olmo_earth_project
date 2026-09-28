"""Local readback of already audited terminal evidence; no server/model access."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,math

R=Path(__file__).resolve().parents[2]
EXP=R/'artifacts/oe10_p2_20260927/review_export_terminal_20260928_0845_v0'
D=R/'artifacts/oe10_terminal_audit_20260928/terminal_audit_20260928_v0'
SERVER=Path('/home/work/data/olmoearth/oe10_p2_v0')
EXPECTED_ATTEST='ab387bddceb0600ac828fb2345a9df5ca488dbd25cea4533b76265348e39e3ea'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def require(v,m):
 if not v:raise ValueError(m)

def local(server):
 rel=Path(server).relative_to(SERVER)
 if rel.parts[0]=='code_snapshot':return R/'code'/Path(*rel.parts[1:])
 if rel.parts[0]=='audit_snapshot':return R/'code'/Path(*rel.parts[1:])
 if rel.parts[0]=='terminal_audit_20260928_v0':return D/rel.name
 return EXP/rel


def main():
 att=read(D/'terminal_attestation.json');e=read(D/'terminal_evidence.json')
 pins=read(R/'code/oe10_terminal_audit_v0/identity_pins.json')
 independent=read(R/'artifacts/oe10_p2_20260927/terminal_completed_runs_independent_review_20260928.json')
 status=read(EXP/'training_v0/status.json');collection=read(EXP/'terminal_collection_20260928_0843.json')
 require(sha(D/'terminal_attestation.json')==EXPECTED_ATTEST,'Attestation external SHA differs')
 require(att['p2_status_sha256']==sha(EXP/'training_v0/status.json')==e['p2_status']['sha256'],'Terminal status hash differs')
 require(att['p2_terminal_status']==status['status']=='failed','Terminal status is not expected failed state')
 require(att['audit_complete'] and att['requires_independent_review_before_use'],'Wrong attestation review contract')
 require(not att['full_p2_decision_made'] and not e['full_p2_decision_made'],'Unexpected scientific-gate decision')
 require(att['completed_runs']==e['completed_runs']==5 and att['expected_runs']==6 and att['p2_result']==e['p2_result']=='incomplete','Incomplete boundary differs')
 require(datetime.fromisoformat(att['audited_utc'])>datetime.fromisoformat(status['finished_utc']),'Audit before termination')
 audited_files=[]
 for rec in att['audit_files']:
  f=local(rec['path']);require(sha(f)==rec['sha256'],'Audit-file identity mismatch: '+str(f));audited_files.append(str(f.relative_to(R)))
 require(status['protected_and_snapshot_unchanged'] is True,'Protected source mismatch')
 require(status['protected_before']==status['protected_after']==pins['protected'],'Protected pinned records differ')
 require(status['source_hashes']==pins['source_hashes'],'Frozen source mapping differs')
 for name,rec in e['source_payloads'].items():
  require(rec['sha256']==pins['source_hashes'][name]==sha(local(rec['path'])),'Source payload differs')
 for name,rec in e['protected_actual'].items():
  require({k:rec[k] for k in ['bytes','mtime_ns','sha256']}==pins['protected'][name],'Protected readback differs')
 require(status['protocol_sha256']==pins['protocol_sha256']==e['protocol']['sha256'],'Protocol differs')
 require(e['own_pids_checked_inactive']==[status['controller_pid']]+[w['pid'] for w in status['workers']],'PID evidence set differs')
 require(e['no_controller_worker_command'] is True,'Own process evidence not empty')
 completed={w['name'] for w in status['workers'] if w['status']=='completed' and w.get('receipt_status')=='training_completed'}
 require(completed=={v['name'] for v in e['verified_runs']}=={f"{v['arm']}_{v['seed']}_train" for v in collection['verified_runs']},'Completed sets differ')
 record_hashes=0
 for v in e['verified_runs']:
  for key in ['receipt','train_log','final_score','final_predictions_manifest']:
   rec=v[key];f=local(rec['path']);require(sha(f)==rec['sha256'] and f.stat().st_size==rec['bytes'],'Run evidence identity mismatch');record_hashes+=1
  receipt=read(local(v['receipt']['path']));run=independent['runs'][v['name'].removesuffix('_train')]
  cr=next(x for x in collection['verified_runs'] if f"{x['arm']}_{x['seed']}_train"==v['name'])
  require(v['checkpoint']['sha256']==receipt['checkpoint_sha256']==cr['checkpoint_sha256'],'Checkpoint lineage mismatch')
  require(not v['last3_stability_passed'] and not run['stability_pass'],'Stability failure missing')
  require(math.isclose(v['last3_auc_range'],run['last3_auc_range'],rel_tol=0,abs_tol=1e-12),'Independent stability differs')
 u=e['unfinished_runs'];require(len(u)==1 and u[0]['name']=='B2_270929_train' and not u[0]['included_in_final_comparison'],'Unfinished arm boundary differs');u=u[0]
 for rec in [u['receipt_identity'],u['log_identity'],*u['score_files']]:
  f=local(rec['path']);require(sha(f)==rec['sha256'] and f.stat().st_size==rec['bytes'],'Unfinished evidence mismatch');record_hashes+=1
 require(u['receipt_completed_updates']==1872 and u['last_logged_step']==1876 and u['score_steps_present']==[384,768,1152,1536],'Unfinished update accounting differs')
 require(u['latest_checkpoint_matches_receipt'] and len(u['checkpoint_files'])==1 and u['checkpoint_files'][0]['sha256']==u['receipt_latest_checkpoint_sha256'],'Incomplete checkpoint lineage mismatch')
 require(not u['checkpoint_contents_deserialized'],'Unexpected checkpoint deserialization claim')
 require(independent['status']=='pass' and independent['all_prediction_npz_checked']==1920 and not independent['errors'],'Independent numerical/export review not passed')
 deploy=read(R/'artifacts/oe10_connection_launcher_20260928/launcher_v1_deployment_cpu_receipt.json')
 log=R/'artifacts/oe10_connection_launcher_20260928/launcher_v1_linux_cpu_tests.log'
 require(sha(log)==deploy['cpu_log_sha256'] and deploy['linux_cpu_tests_passed']==22 and deploy['linux_cpu_tests_skipped']==0,'Launcher Linux test evidence mismatch')
 require(sha(R/'code/oe10_connection_launcher_v1/source_manifest.json')==deploy['source_manifest_sha256'],'Launcher source digest mismatch')
 for folder in ['oe10_connection_launcher_v1','oe10_terminal_audit_v0','oe10_text_mask_v2']:
  base=R/'code'/folder;m=read(base/'source_manifest.json')
  for rec in m['files']:
   f=base/rec['path'];require(sha(f)==rec['sha256'] and f.stat().st_size==rec['bytes'],'Source identity changed: '+str(f))
 result={'schema_version':'oe10_terminal_attestation_independent_review_v1','created_utc':datetime.now(timezone.utc).isoformat(),'status':'pass','attestation_sha256':EXPECTED_ATTEST,'terminal_evidence_sha256':sha(D/'terminal_evidence.json'),'terminal_status_sha256':sha(EXP/'training_v0/status.json'),'collection_sha256':sha(EXP/'terminal_collection_20260928_0843.json'),'review_source_sha256':sha(Path(__file__)),'audit_file_hashes_verified':len(audited_files),'audit_files':audited_files,'local_run_evidence_hashes_verified':record_hashes,'completed_runs':5,'stability_pass_count':0,'p2_result':'incomplete','full_p2_decision_made':False,'unfinished_receipt_step':1872,'unfinished_log_step':1876,'unfinished_last_score_step':1536,'launcher_linux_cpu_tests_passed':22,'launcher_linux_cpu_tests_skipped':0,'new_gpu_seconds':0,'new_reservation':False,'scope':'External attestation identity and terminal evidence readback reviewed; usable as terminal-integrity prerequisite only, subject to fresh launcher preflight and existing reservation protocol.','limits':['Checkpoint files remain on server: this local review checks receipt/collector/helper hash agreement, not local checkpoint bytes or deserialization.','Own-process inactivity and protected file mtimes are server-helper observations, cross-linked to status and pins; no new remote inspection here.','Completed arrays/count aggregation independently reviewed in the referenced terminal export audit, not regenerated from source labels.','P2 incomplete with all five completed runs failing training stability; no superiority or full comparative fairness claim.']}
 out=R/'artifacts/oe10_terminal_audit_20260928/independent_review_20260928.json'
 require(not out.exists(),'Never overwrite existing independent review')
 out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['status','attestation_sha256','audit_file_hashes_verified','local_run_evidence_hashes_verified','completed_runs','stability_pass_count','p2_result','launcher_linux_cpu_tests_passed']},indent=2))

if __name__=='__main__':main()
