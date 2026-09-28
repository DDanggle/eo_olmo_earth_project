"""Independent local receipt/source audit of the bounded zero-update connection check.
Does not rerun models or recompute logits/gradients from tensors (not exported).
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math

R=Path(__file__).resolve().parents[2]
E=R/'artifacts/oe10_connection_launcher_20260928/review_export_connection_20260928_0856_v0'
J=E/'connection_runs_v0/connection_20260928_01'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def require(v,m):
 if not v:raise ValueError(m)

def main():
 manifest=read(E/'export_manifest.json');seen=set();total=0
 for rec in manifest['files']:
  p=E/rec['path'];require(rec['path'] not in seen,'Duplicate export file');seen.add(rec['path'])
  require(p.is_file() and p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],'Export SHA/size mismatch: '+str(p));total+=rec['bytes']
 model=read(J/'model_output/receipt.json');launch=read(J/'launcher_receipt.json');ledger=read(E/'pilot_budget_ledger_v0.json')
 reserve=read(J/'reservation.json');env=read(J/'environment.json');command=read(J/'command.json');pre=read(J/'preflight.json')
 protocol=read(E/'config/oe10_text_mask_connection_protocol_20260928.json');identity=read(E/'identity_preparation_v1/identity_manifest.json')
 require(sha(E/'config/oe10_text_mask_connection_protocol_20260928.json')=='bb721164b472cf6b66f2e3833a2e99689d59e59ee0c1c604d05b4f2894f5b778','Frozen protocol differs')
 require(sha(E/'identity_preparation_v1/identity_manifest.json')==protocol['identity']['sha256']==model['identity_manifest_sha256'],'Input identity differs')
 require(model['status']=='actual_connection_check_passed' and launch['status']=='completed' and launch['exit_code']==0,'Connection failed')
 require(model['new_training_updates']==protocol['configuration']['optimizer_updates']==0 and model['development_scoring'] is False and protocol['configuration']['generation'] is False,'Scope exceeds zero-update check')
 require(model['actual_human_responses']==0 and model['cold_resume_tested'] is False,'Human/resume scope differs')
 require(model['source_identity_gate']['manifest_sha256']==protocol['source']['manifest_sha256'],'Model source differs')
 require(model['episode_identity_gate']['manifest_sha256']==identity['episodes_export_manifest_sha256'] and model['episode_identity_gate']['payload_files_verified']==232,'Episode export differs')
 require(model['reader_identity']['files']==identity['reader_files_sha256'],'Reader identity differs')
 require(model['eo_identity']['files']==identity['eo_files_sha256'] and model['eo_identity']['source']==identity['eo_source_files_sha256'] and model['eo_identity']['prepared_manifest_sha256']==identity['prepared_manifest_sha256'],'EO identity differs')
 conditions=set(protocol['conditions']);diff=model['text_condition_logit_max_difference_vs_names'];token=model['text_token_audit']
 require(set(diff)==set(token)==conditions and len(conditions)==5,'Five frozen conditions missing')
 require(all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in diff.values()),'Invalid numerical dependency diagnostic')
 require(diff['names_only']==diff['removed_knowledge_diagnostic']==0 and diff['matched_knowledge']>0,'Text dependency/removal gate differs')
 require(all(type(v['tokens']) is int and 0<v['tokens']<=protocol['configuration']['text_max_tokens'] and not v['cache_hit'] for v in token.values()),'Token budget/cache evidence differs')
 require(token['names_only']['context_sha256']==token['removed_knowledge_diagnostic']['context_sha256'],'Removal input is not names-only')
 contexts=R/'artifacts/oe10_text_mask_prepare_20260928/contexts_v0/contexts.jsonl'
 require(sha(contexts)==identity['contexts_sha256'],'Local frozen contexts differ')
 row=next(json.loads(line) for line in contexts.read_text().splitlines() if json.loads(line)['episode_id']==protocol['sample']['episode_id'])
 require(row['audit_only']['split']=='train' and row['audit_only']['k_pairs']==1,'Input not train K1')
 for name,value in row['model_context_by_condition'].items():
  digest=hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  require(digest==token[name]['context_sha256'],'Actual context identity differs: '+name)
 gradients=model['gradient_nonzero_tensors']
 require(set(gradients)=={'encoder','head','qwen','unused_connector'} and model['mask_loss_and_all_present_gradients_finite'] is True,'Gradient receipt incomplete')
 require(gradients['encoder']>0 and gradients['head']>0 and gradients['qwen']==gradients['unused_connector']==0,'Gradient boundaries differ')
 delta=model['same_process_restore_max_difference'];require(math.isfinite(delta) and 0<=delta<=1e-6,'Restore gate failed')
 argv=command['argv'];require(command['shell'] is False and command['PYTHONPATH_removed'] and env['PYTHONPATH'] is None and env['CUDA_VISIBLE_DEVICES']=='1','Environment/shell boundaries differ')
 require(env['pid']==env['process_group']==model['pid']==launch['child_pid'],'Owned PID linkage differs')
 require(env['parent_death_guard']['signal']=='SIGKILL' and env['parent_death_guard']['expected_parent_pid']==int(argv[4]),'Parent guard differs')
 require('--execute' in argv and argv[argv.index('--episode-id')+1]==protocol['sample']['episode_id'] and argv[argv.index('--arm')+1]=='B2','Actual command sample/arm differs')
 require(argv[argv.index('--source-manifest-sha256')+1]==protocol['source']['manifest_sha256'] and argv[argv.index('--identity-manifest-sha256')+1]==protocol['identity']['sha256'],'Actual command identities differ')
 for key,path in protocol['paths'].items():require(argv[argv.index('--'+key)+1]==path,'Actual command path differs: '+key)
 require(pre['ready'] and not pre['reasons'],'Preflight not passed')
 att=read(E/'terminal_attestation_20260928_v0.json');ind=read(R/'artifacts/oe10_terminal_audit_20260928/independent_review_20260928.json')
 require(sha(E/'terminal_attestation_20260928_v0.json')==ind['attestation_sha256']==pre['evidence']['p2_terminal_audit']['attestation_sha256'],'Unreviewed terminal attestation used')
 require(att['p2_result']==reserve['p2_result']=='incomplete' and reserve['p2_finished_and_audited'] and datetime.fromisoformat(launch['reserved_utc'])>datetime.fromisoformat(att['audited_utc']),'Predecessor termination ordering differs')
 require(ledger['maximum_seconds']==7200 and ledger['jobs']==[launch],'Ledger/launcher inconsistent')
 require(0<launch['actual_elapsed_seconds']<=launch['reserved_seconds']==reserve['reserved_seconds']==1800,'Per-attempt bound violated')
 require(launch['charged_seconds']==math.ceil(launch['actual_elapsed_seconds'])==48 and ledger['charged_seconds']==48 and reserve['prior_gpu_used_seconds']==0,'Cumulative charge differs')
 require(launch['actual_elapsed_seconds']>=model['elapsed_seconds'] and ledger['charged_seconds']<=7200,'Whole-process timing bound differs')
 bootstrap=E/'config/oe10_pilot_budget_bootstrap_20260928.json';witness=read(E/'pilot_budget_ledger_v0.initialized.json')
 require(sha(bootstrap)==ledger['initial_snapshot_sha256']==witness['initial_snapshot_sha256'],'Initial budget lineage differs')
 require(read(bootstrap)['pilot_budget']['reset_each_heartbeat'] is False,'Budget reset changed')
 for folder,digest in [('oe10_text_mask_v2',protocol['source']['manifest_sha256']),('oe10_connection_launcher_v1',pre['evidence']['launcher_source']['manifest_sha256'])]:
  base=R/'code'/folder;require(sha(base/'source_manifest.json')==digest,'Frozen source manifest differs')
  for rec in read(base/'source_manifest.json')['files']:
   f=base/rec['path'];require(sha(f)==rec['sha256'] and f.stat().st_size==rec['bytes'],'Frozen source changed')
 result={'schema_version':'oe10_connection_result_independent_review_v1','created_utc':datetime.now(timezone.utc).isoformat(),'status':'pass','job_id':launch['job_id'],'review_source_sha256':sha(Path(__file__)),'export_manifest_sha256':sha(E/'export_manifest.json'),'export_files':len(seen),'export_bytes':total,'model_receipt_sha256':sha(J/'model_output/receipt.json'),'launcher_receipt_sha256':sha(J/'launcher_receipt.json'),'ledger_sha256':sha(E/'pilot_budget_ledger_v0.json'),'actual_elapsed_seconds':launch['actual_elapsed_seconds'],'charged_seconds':48,'cumulative_limit_seconds':7200,'remaining_charge_budget_seconds':7152,'all_five_contexts_match_frozen_training_input':True,'gradient_nonzero_tensors':gradients,'mask_loss_and_all_present_gradients_finite_reported':True,'logit_differences':diff,'same_process_restore_max_difference':delta,'optimizer_updates':0,'development_scoring':False,'generation':False,'actual_human_responses':0,'full_p2_decision_made':False,'scope':'Real-weight numerical connection and gradient-path receipt verification only; standard text-conditioned dense head, frozen Qwen text-only prefill, original EO encoder.','limits':['This review hashes receipts, source and exported artifacts and checks frozen conditions; no model rerun or independent reconstruction from logit/gradient tensors, which were not exported.','Checkpoint weights are excluded from this export; same-process restore is a worker-reported check, not fresh-process resume.','Numerical text sensitivity does not establish semantic knowledge use, expert benefit, training improvement, label efficiency or transfer.','GPU/process absence after completion is a separate live observation by the parent; this local readback performs no remote process check.']}
 out=R/'artifacts/oe10_connection_launcher_20260928/independent_review_connection_01.json';require(not out.exists(),'Never overwrite independent review');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['status','export_files','export_bytes','charged_seconds','gradient_nonzero_tensors','same_process_restore_max_difference']},indent=2))

if __name__=='__main__':main()
