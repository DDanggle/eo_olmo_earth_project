"""Read-only terminal evidence audit. Writes only a new output evidence directory.

No process control, GPU access, training, final-region opening or gate promotion.
An independently reviewed and externally pinned attestation is required for use.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path('/home/work/data/olmoearth/oe10_p2_v0')
HERE = Path(__file__).absolute().parent


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8 << 20),b''): h.update(part)
    return h.hexdigest()


def safe(path, root=None):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path,*path.parents)), 'Symlink rejected')
    require(root is None or path.is_relative_to(root), 'Path outside audit scope')
    require(path.is_file(), 'Required regular file missing: '+str(path))
    return path


def read(path): return json.loads(safe(path).read_text())


def identity(path, expected=None):
    p = safe(path); record = {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
    require(expected is None or record['sha256'] == expected, 'Hash mismatch: '+str(path))
    return record


def alive(pid):
    require(type(pid) is int and pid>1,'Invalid own PID')
    try: os.kill(pid,0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


def ps():
    return subprocess.run(['ps','-eo','pid=,args='],capture_output=True,text=True,check=True,timeout=15).stdout


def collection_runs(collection, controller):
    expected = {f'{arm}_{seed}_train' for seed in (270927,270928,270929) for arm in ('B0','B2')}
    workers = controller.get('workers',[])
    require(len({w['name'] for w in workers})==len(workers),'Duplicate controller workers')
    require({w['name'] for w in workers}<=expected,'Unexpected controller worker')
    completed = {w['name'] for w in workers if w.get('status')=='completed'
                 and w.get('receipt_status')=='training_completed' and w.get('exit_code')==0}
    schema = collection.get('schema_version')
    require(collection.get('p2_decision_made') is False,'Collector must not make a P2 decision')
    if schema=='oe10_partial_run_collection_v1':
        runs=collection['verified_runs']; blockers=collection['blockers']
        require(collection['status']=='partial_or_unverified' and len(runs)==5,'Expected honest five-run partial')
        require(all(b=='six completed workers/controller integrity not yet verified' or
                    b.partition(':')[0] in expected-completed for b in blockers),'Unexpected collector blocker')
        require(controller['status']=='failed','Partial collection requires terminal failed controller')
    elif schema=='oe9_p2_scorer_summary_v1':
        runs=collection['runs']; blockers=collection['collector_blockers']
        require(len(runs)==6 and controller['status']=='completed' and not blockers,'Six-run collector blocked')
        require(collection['fairness_checks'] and all(v is True for v in collection['fairness_checks'].values()),
                'Complete collector fairness not verified')
    else: raise ValueError('Unknown collector schema')
    names=[f"{r['arm']}_{r['seed']}_train" for r in runs]
    require(len(names)==len(set(names)) and set(names)==completed,'Controller/collector completed sets differ')
    require(len({r['cohort_base_ids_sha256'] for r in runs})==1,'Verified run cohorts differ')
    return runs,sorted(expected-completed),blockers


def audit(root, collection_path, collection_sha, pins, *, protocol_path,
          process_alive=alive, process_listing=ps):
    root=Path(root); runroot=root/'training_v0'; status_path=runroot/'status.json'
    status_identity=identity(status_path); controller=read(status_path)
    require(controller.get('status') in {'completed','failed'} and controller.get('finished_utc'),
            'P2 still running or not durably terminal')
    require(controller.get('stage')=='training','Expected P2 training controller')
    finished=datetime.fromisoformat(controller['finished_utc'].replace('Z','+00:00'))
    require(finished.tzinfo is not None and finished<=datetime.now(timezone.utc),'Invalid terminal timestamp')
    require(controller.get('protected_and_snapshot_unchanged') is True,'Controller integrity not verified')
    require(controller.get('protected_before')==controller.get('protected_after')==pins['protected'],
            'Protected before/after/pinned records differ')
    require(controller.get('source_hashes')==pins['source_hashes'],'Controller source identity differs')
    require(controller.get('protocol_sha256')==pins['protocol_sha256'],'Controller protocol differs')
    pids=[controller.get('controller_pid')]+[w.get('pid') for w in controller['workers']]
    require(not any(process_alive(pid) for pid in pids),'Own controller/worker PID still active')
    process_text=process_listing()
    require(not any(str(root) in row and ('run_bounded.py' in row or 'p2_worker.py' in row)
                    for row in process_text.splitlines()),'P2 controller/worker command still active')
    protected={}
    for name,expected in pins['protected'].items():
        path=root.parent/'code'/name; rec=identity(path,expected['sha256'])
        require(path.stat().st_mtime_ns==expected['mtime_ns'] and rec['bytes']==expected['bytes'],
                'Protected mtime/size changed: '+name)
        protected[name]=rec|{'mtime_ns':path.stat().st_mtime_ns}
    snapshot=root/'code_snapshot/oe10_p2_v3'
    source_manifest=identity(snapshot/'source_manifest.json',pins['source_manifest_sha256'])
    require(read(snapshot/'source_manifest.json')['files']==pins['source_hashes'],'Source manifest mapping differs')
    sources={name:identity(snapshot/name,digest) for name,digest in pins['source_hashes'].items()}
    collector_source=identity(root/'audit_snapshot/oe10_audit_v2/summarize_runs.py',pins['collector_sha256'])
    protocol_identity=identity(safe(protocol_path,root),pins['protocol_sha256']); protocol=read(protocol_path)
    require(protocol['updates_per_run']==pins['updates_per_run'] and protocol['max_sequence_seconds']==pins['maximum_sequence_seconds']
            and protocol['evaluation_interval_updates']==pins['evaluation_interval_updates'],'Frozen schedule mismatch')
    gate_identity=identity(snapshot/'gate_config.json',pins['gate_config_sha256']); gate=read(snapshot/'gate_config.json')
    fairness_index=identity(root/'fairness_audit_v3_evidence.json',pins['fairness_evidence_sha256'])
    fairness_proof=identity(root/'fairness_audit_v3.json',pins['fairness_proof_sha256']); proof=read(root/'fairness_audit_v3.json')
    require(proof['status']=='passed' and proof['source_hashes'] and
            all(pins['source_hashes'].get(k)==v for k,v in proof['source_hashes'].items()),'Fairness proof identity mismatch')
    evidence_index=read(root/'fairness_audit_v3_evidence.json')
    for check,record in evidence_index.items():
        require(check in gate['p2']['fairness_checks_required'] and check in proof['checks_passed'],
                'Fairness source-proof check absent')
        require(Path(record['path'])==root/'fairness_audit_v3.json' and record['sha256']==fairness_proof['sha256'],
                'Fairness evidence reference mismatch')
    collection_identity=identity(safe(collection_path,root),collection_sha); collection=read(collection_path)
    runs,missing,blockers=collection_runs(collection,controller)
    verified=[]
    for result in runs:
        name=f"{result['arm']}_{result['seed']}_train"; folder=runroot/name
        receipt_identity=identity(folder/'receipt.json',result['run_receipt_sha256']); receipt=read(folder/'receipt.json')
        require((receipt['status'],receipt['arm'],receipt['seed'],receipt['mode'])==('training_completed',result['arm'],result['seed'],'train'),
                'Completed receipt identity mismatch')
        require(receipt['protocol_sha256']==pins['protocol_sha256'] and receipt['source_hashes']==pins['source_hashes'],
                'Completed receipt source/protocol mismatch')
        require(receipt['completed_updates']==2304 and result['training_adequacy']['completed_updates']==2304
                and result['training_adequacy']['training_schedule_completed'] is True,'Incomplete arm in final run set')
        checkpoint=identity(folder/'final.pt',result['checkpoint_sha256'])
        require(receipt['checkpoint_sha256']==checkpoint['sha256'],'Checkpoint/receipt differs')
        log_identity=identity(folder/'train_log.jsonl')
        logs=[json.loads(line) for line in (folder/'train_log.jsonl').read_text().splitlines() if line.strip()]
        require([r['step'] for r in logs]==list(range(1,2305)),'Completed log step coverage differs')
        order=hashlib.sha256(('\n'.join(r['episode_id'] for r in logs)+'\n').encode()).hexdigest()
        require(order==receipt['episode_order_sha256'],'Completed order digest differs')
        final=receipt['evaluation_windows'][-1];require(final['step']==2304,'Final score not at frozen final step')
        score_identity=identity(folder/'score_step_002304.json',final['score_sha256']);score=read(folder/'score_step_002304.json')
        pred_identity=identity(folder/'predictions_step_002304/predictions.jsonl',score['predictions_manifest_sha256'])
        require(result['target_iou_by_k']==score['target_iou_by_k'],'Collector final curve differs')
        windows=result['training_adequacy']['last_evaluation_windows']
        last3=[x['auc'] for x in windows[-3:]];delta=max(last3)-min(last3)
        verified.append({'name':name,'receipt':receipt_identity,'checkpoint':checkpoint,'train_log':log_identity,
            'final_score':score_identity,'final_predictions_manifest':pred_identity,'last3_auc_range':delta,
            'frozen_maximum_last3_auc_range':gate['p2']['training_adequacy']['maximum_last3_auc_range'],
            'last3_stability_passed':delta<=gate['p2']['training_adequacy']['maximum_last3_auc_range'],
            'collector_training_adequacy':result['training_adequacy']})
    unfinished=[]
    for name in missing:
        folder=runroot/name; row={'name':name,'included_in_final_comparison':False}
        if (folder/'receipt.json').is_file():
            row['receipt_identity']=identity(folder/'receipt.json'); r=read(folder/'receipt.json')
            arm,seed,_=name.split('_')
            require((r.get('arm'),r.get('seed'),r.get('mode'))==(arm,int(seed),'train')
                    and r.get('protocol_sha256')==pins['protocol_sha256']
                    and r.get('source_hashes')==pins['source_hashes'],'Unfinished receipt identity mismatch')
            row.update(receipt_status=r.get('status'),receipt_completed_updates=r.get('completed_updates'),
                       receipt_latest_checkpoint_sha256=r.get('latest_checkpoint_sha256'))
        if (folder/'train_log.jsonl').is_file():
            row['log_identity']=identity(folder/'train_log.jsonl')
            lines=(folder/'train_log.jsonl').read_text().splitlines()
            row['last_logged_step']=json.loads(lines[-1])['step'] if lines else None
        row['score_steps_present']=[int(p.stem.rsplit('_',1)[1]) for p in sorted(folder.glob('score_step_*.json'))]
        row['score_files']=[identity(p) for p in sorted(folder.glob('score_step_*.json'))]
        row['checkpoint_files']=[identity(p) for p in sorted(folder.glob('*.pt'))]
        latest=next((p for p in row['checkpoint_files'] if Path(p['path']).name=='latest.pt'),None)
        row['latest_checkpoint_matches_receipt']=(latest is not None and
            latest['sha256']==row.get('receipt_latest_checkpoint_sha256'))
        row['checkpoint_contents_deserialized']=False
        row['checkpoint_receipt_mismatch_policy']='If false, preserve interruption mismatch as unverified; never include this arm in the final comparison or resume it.'
        unfinished.append(row)
    require(identity(status_path)==status_identity,'Terminal status changed during audit')
    result='complete' if len(verified)==6 and controller['status']=='completed' else 'incomplete'
    now=datetime.now(timezone.utc).isoformat()
    evidence={'schema_version':'oe10_terminal_evidence_v1','audited_utc':now,'p2_result':result,
              'completed_runs':len(verified),'expected_runs':6,'p2_status':status_identity,
              'controller_error':controller.get('error'),'controller_elapsed_seconds':controller.get('elapsed_seconds'),
              'own_pids_checked_inactive':pids,'no_controller_worker_command':True,'protocol':protocol_identity,
              'source_manifest':source_manifest,'source_payloads':sources,'protected_actual':protected,
              'collector_source':collector_source,'collection':collection_identity,'collector_blockers':blockers,
              'fairness_source_proof_checks_verified':sorted(evidence_index),'fairness_index':fairness_index,
              'fairness_proof':fairness_proof,'gate_config':gate_identity,'verified_runs':verified,'unfinished_runs':unfinished,
              'full_p2_decision_made':False,'new_gpu_seconds':0,
              'limits':['Terminal evidence audit only; no scientific-gate pass claim.',
                        'Partial collector does not evaluate full 13-check comparative fairness.',
                        'Prediction file hashes were verified by the frozen collector; array shape/finite/bounds and independent export review remain external, not repeated here.',
                        'Training stability failures are retained; unfinished arm excluded from final comparison.',
                        'One development region, synthetic corrections, no human or generated-language evaluation.']}
    audit_files=[collection_identity,collector_source,source_manifest,protocol_identity,gate_identity,fairness_index,fairness_proof]
    attestation={'schema_version':'oe10_p2_terminal_audit_attestation_v1','audit_complete':True,
                 'p2_status_sha256':status_identity['sha256'],'p2_terminal_status':controller['status'],
                 'completed_runs':len(verified),'expected_runs':6,'p2_result':result,'audited_utc':now,
                 'audit_files':[{k:r[k] for k in ('path','sha256')} for r in audit_files],
                 'full_p2_decision_made':False,'requires_independent_review_before_use':True,
                 'scope':'Terminal integrity audit complete; full P2 scientific gate is a separate result.'}
    return evidence,attestation


def main():
    p=argparse.ArgumentParser()
    for name in ('collection','protocol','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--collection-sha256',required=True);p.add_argument('--helper-manifest-sha256',required=True)
    a=p.parse_args()
    require(sys.platform.startswith('linux') and HERE.is_relative_to(ROOT/'audit_snapshot'),'Run fixed server audit snapshot')
    identity(HERE/'source_manifest.json',a.helper_manifest_sha256)
    for item in read(HERE/'source_manifest.json')['files']:
        rel=PurePosixPath(item['path']);require(not rel.is_absolute() and '..' not in rel.parts,'Helper manifest path escape')
        identity(HERE/rel,item['sha256'])
    a.out=Path(os.path.abspath(a.out))
    require(not a.out.exists() and a.out.is_relative_to(ROOT)
            and not any(q.is_symlink() for q in (a.out,*a.out.parents)),
            'New output directory under P2 root required')
    pins=read(HERE/'identity_pins.json')
    evidence,attestation=audit(ROOT,a.collection,a.collection_sha256,pins,protocol_path=a.protocol)
    a.out.mkdir()
    for name,value in [('terminal_evidence.json',evidence)]:
        with (a.out/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    item=identity(a.out/'terminal_evidence.json');attestation['audit_files'].append({k:item[k] for k in ('path','sha256')})
    with (a.out/'terminal_attestation.json').open('x') as f:json.dump(attestation,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':'terminal_integrity_audit_completed_pending_independent_review',
         'completed_runs':evidence['completed_runs'],'p2_result':evidence['p2_result'],
         'attestation_sha256':sha(a.out/'terminal_attestation.json')}))


if __name__=='__main__':main()
