"""Synthetic filesystem audits only; no server/GPU or production attestation."""
import copy
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import audit_terminal as A


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value)+'\n')
    return A.sha(path)


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/private/tmp' if Path('/private/tmp').exists() else None)
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()/'p2';self.root.mkdir()
        snap=self.root/'code_snapshot/oe10_p2_v3';snap.mkdir(parents=True)
        guard=self.root.parent/'code/protected.py';guard.parent.mkdir();guard.write_text('# protected\n')
        protected={'protected.py':{'sha256':A.sha(guard),'bytes':guard.stat().st_size,'mtime_ns':guard.stat().st_mtime_ns}}
        protocol=self.root/'protocol.json'
        ph=put(protocol,{'updates_per_run':2304,'max_sequence_seconds':43200,'evaluation_interval_updates':384})
        gate={'p2':{'fairness_checks_required':['source_guard'],'training_adequacy':{'maximum_last3_auc_range':.005}}}
        gh=put(snap/'gate_config.json',gate);(snap/'p2_worker.py').write_text('# pinned worker\n')
        sources={'gate_config.json':gh,'p2_worker.py':A.sha(snap/'p2_worker.py')}
        sm=put(snap/'source_manifest.json',{'files':sources})
        cp=self.root/'audit_snapshot/oe10_audit_v2/summarize_runs.py';cp.parent.mkdir(parents=True);cp.write_text('# collector\n')
        proof=self.root/'fairness_audit_v3.json'
        fh=put(proof,{'status':'passed','source_hashes':sources,'checks_passed':['source_guard']})
        fi=put(self.root/'fairness_audit_v3_evidence.json',{'source_guard':{'path':str(proof),'sha256':fh}})
        self.pins={'protected':protected,'protocol_sha256':ph,'source_hashes':sources,'source_manifest_sha256':sm,
                   'collector_sha256':A.sha(cp),'gate_config_sha256':gh,'fairness_proof_sha256':fh,
                   'fairness_evidence_sha256':fi,'updates_per_run':2304,'maximum_sequence_seconds':43200,
                   'evaluation_interval_updates':384}
        workers=[];runs=[]
        for seed,arm in [(270927,'B0'),(270927,'B2'),(270928,'B0'),(270928,'B2'),(270929,'B0')]:
            name=f'{arm}_{seed}_train';folder=self.root/'training_v0'/name;folder.mkdir(parents=True)
            (folder/'final.pt').write_bytes(b'fake-checkpoint')
            logs=[{'step':i,'episode_id':'training-id'} for i in range(1,2305)]
            (folder/'train_log.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in logs))
            order=hashlib.sha256(('training-id\n'*2304).encode()).hexdigest()
            pred=folder/'predictions_step_002304/predictions.jsonl';pred.parent.mkdir();pred.write_text('{}\n')
            curve={str(k):.3 for k in (1,2,4,8)}
            sh=put(folder/'score_step_002304.json',{'predictions_manifest_sha256':A.sha(pred),'target_iou_by_k':curve})
            receipt={'status':'training_completed','mode':'train','arm':arm,'seed':seed,
                     'protocol_sha256':ph,'source_hashes':sources,'completed_updates':2304,
                     'checkpoint_sha256':A.sha(folder/'final.pt'),'episode_order_sha256':order,
                     'evaluation_windows':[{'step':2304,'score_sha256':sh}]}
            rh=put(folder/'receipt.json',receipt)
            runs.append({'arm':arm,'seed':seed,'cohort_base_ids_sha256':'same-cohort','run_receipt_sha256':rh,
                         'checkpoint_sha256':A.sha(folder/'final.pt'),'target_iou_by_k':curve,
                         'training_adequacy':{'completed_updates':2304,'training_schedule_completed':True,
                             'last_evaluation_windows':[{'step':1536,'auc':.20},{'step':1920,'auc':.30},{'step':2304,'auc':.31}]}})
            workers.append({'name':name,'status':'completed','receipt_status':'training_completed','exit_code':0,'pid':200+len(workers)})
        workers.append({'name':'B2_270929_train','status':'own_worker_timeout','pid':205})
        self.status={'status':'failed','stage':'training','finished_utc':(datetime.now(timezone.utc)-timedelta(seconds=2)).isoformat(),
                     'controller_pid':100,'workers':workers,'protected_and_snapshot_unchanged':True,
                     'protected_before':protected,'protected_after':copy.deepcopy(protected),'source_hashes':sources,
                     'protocol_sha256':ph,'error':"RuntimeError('worker timeout')",'elapsed_seconds':43220}
        put(self.root/'training_v0/status.json',self.status)
        folder=self.root/'training_v0/B2_270929_train';folder.mkdir()
        put(folder/'receipt.json',{'status':'training','completed_updates':1872,'arm':'B2','seed':270929,
            'mode':'train','protocol_sha256':ph,'source_hashes':sources,
            'latest_checkpoint_sha256':hashlib.sha256(b'partial checkpoint').hexdigest()})
        (folder/'train_log.jsonl').write_text('{"step":1876}\n')
        put(folder/'score_step_001536.json',{'partial':True});(folder/'latest.pt').write_bytes(b'partial checkpoint')
        self.collection={'schema_version':'oe10_partial_run_collection_v1','status':'partial_or_unverified',
                         'verified_runs':runs,'blockers':['B2_270929_train: run unfinished/identity mismatch',
                         'six completed workers/controller integrity not yet verified'],'p2_decision_made':False}
        self.collection_path=self.root/'collection.json';put(self.collection_path,self.collection)
    def run_audit(self,**kwargs):
        return A.audit(self.root,self.collection_path,A.sha(self.collection_path),self.pins,
                       protocol_path=self.root/'protocol.json',process_alive=kwargs.get('alive',lambda pid:False),
                       process_listing=kwargs.get('ps',lambda:''))
    def test_five_verified_terminal_keeps_incomplete_and_stability_failures(self):
        evidence,att=self.run_audit()
        self.assertEqual((att['completed_runs'],att['p2_result']),(5,'incomplete'))
        self.assertFalse(att['full_p2_decision_made']);self.assertTrue(att['requires_independent_review_before_use'])
        self.assertTrue(all(not r['last3_stability_passed'] for r in evidence['verified_runs']))
        partial=evidence['unfinished_runs'][0]
        self.assertEqual(partial['receipt_completed_updates'],1872);self.assertEqual(partial['last_logged_step'],1876)
        self.assertEqual(partial['score_steps_present'],[1536]);self.assertFalse(partial['included_in_final_comparison'])
        self.assertTrue(partial['latest_checkpoint_matches_receipt'])
    def test_running_and_live_pid_and_ps_rejected(self):
        with self.assertRaisesRegex(ValueError,'PID still active'):self.run_audit(alive=lambda p:p==100)
        with self.assertRaisesRegex(ValueError,'command still active'):
            self.run_audit(ps=lambda:f'200 python {self.root}/code_snapshot/oe10_p2_v3/p2_worker.py')
        self.status['status']='running';put(self.root/'training_v0/status.json',self.status)
        with self.assertRaisesRegex(ValueError,'still running'):self.run_audit()
    def test_collector_count_duplicate_and_unexpected_error_rejected(self):
        for field in ('missing','duplicate','blocker'):
            value=copy.deepcopy(self.collection)
            if field=='missing':value['verified_runs'].pop()
            if field=='duplicate':value['verified_runs'][-1]=value['verified_runs'][0]
            if field=='blocker':value['blockers'].append('unknown audit failure')
            put(self.collection_path,value)
            with self.subTest(field=field),self.assertRaises(ValueError):self.run_audit()
    def test_wrong_collection_pin_and_source_fairness_tamper_rejected(self):
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):
            A.audit(self.root,self.collection_path,'0'*64,self.pins,protocol_path=self.root/'protocol.json',
                    process_alive=lambda p:False,process_listing=lambda:'')
        (self.root/'fairness_audit_v3.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.run_audit()
    def test_completed_checkpoint_or_log_tamper_rejected(self):
        folder=self.root/'training_v0/B0_270927_train'
        (folder/'final.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Hash mismatch'):self.run_audit()
        (folder/'final.pt').write_bytes(b'fake-checkpoint')
        (folder/'train_log.jsonl').write_text('{"step":1,"episode_id":"changed"}\n')
        with self.assertRaisesRegex(ValueError,'step coverage'):self.run_audit()
    def test_unfinished_identity_rejected_and_checkpoint_mismatch_preserved(self):
        folder=self.root/'training_v0/B2_270929_train'
        (folder/'latest.pt').write_bytes(b'written at interruption')
        evidence,_=self.run_audit()
        self.assertFalse(evidence['unfinished_runs'][0]['latest_checkpoint_matches_receipt'])
        receipt=A.read(folder/'receipt.json');receipt['seed']=0;put(folder/'receipt.json',receipt)
        with self.assertRaisesRegex(ValueError,'Unfinished receipt identity'):self.run_audit()
    def test_protected_mtime_and_false_integrity_rejected(self):
        self.status['protected_and_snapshot_unchanged']=False;put(self.root/'training_v0/status.json',self.status)
        with self.assertRaisesRegex(ValueError,'integrity not verified'):self.run_audit()
        self.status['protected_and_snapshot_unchanged']=True
        self.status['protected_after']['protected.py']['mtime_ns']+=1;put(self.root/'training_v0/status.json',self.status)
        with self.assertRaisesRegex(ValueError,'records differ'):self.run_audit()


if __name__=='__main__':unittest.main()
