"""Small bounded collector failure tests. No model-score fixtures or server work."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from summarize_runs import collect, sha


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.run = self.root/'run'; self.run.mkdir(); self.eng = self.root/'engineering'; self.eng.mkdir()
        self.src = self.root/'source'; self.src.mkdir()
        (self.src/'worker.py').write_text('synthetic fixture source only\n')
        self.sources = {'worker.py':sha(self.src/'worker.py')}
        self.protocol = self.root/'protocol.json'; self.protocol.write_text(json.dumps({'paired_seed_ids':[270927,270928,270929],'locked_before_development_results':True}))
        self.gate = self.root/'gate.json'; self.gate.write_text(json.dumps({'p2':{'paired_seed_ids':[270927,270928,270929]}}))
        self.status = {'stage':'training','status':'running','source_hashes':self.sources,'protocol_sha256':sha(self.protocol),'workers':[]}
        (self.run/'status.json').write_text(json.dumps(self.status))
        (self.eng/'status.json').write_text(json.dumps({'status':'completed','protected_and_snapshot_unchanged':True,'source_hashes':self.sources}))
        for arm in ('B0','B2'):
            for mode in ('reference','resume'):
                d = self.eng/f'{arm}_270927_{mode}'; d.mkdir()
                r = {'status':'reference_completed' if mode=='reference' else 'cold_resume_passed','source_hashes':self.sources,
                     'pass_result':True,'fresh_process_resume':True,'reference_pid':100,'current_pid':101}
                (d/'receipt.json').write_text(json.dumps(r))

    def tearDown(self): self.tmp.cleanup()

    def run_collector(self):
        return collect(self.protocol,self.run,self.eng,self.src,self.gate)

    def test_missing_protocol_is_partial_not_claimed_complete(self):
        self.protocol.unlink(); r = self.run_collector()
        self.assertEqual(r['schema_version'],'oe10_partial_run_collection_v1')
        self.assertEqual(r['verified_runs'],[]); self.assertTrue(r['blockers'])

    def test_all_six_missing_runs_are_listed_without_reading_predictions(self):
        r = self.run_collector()
        self.assertEqual(len(r['blockers']),7)
        self.assertEqual(r['verified_runs'],[]); self.assertFalse(r['p2_decision_made'])

    def test_changed_source_blocks_before_run_processing(self):
        (self.src/'worker.py').write_text('different synthetic source\n')
        r = self.run_collector()
        self.assertIn('source',r['blockers'][0]); self.assertEqual(r['verified_runs'],[])

    def test_one_collected_run_never_emits_six_run_summary(self):
        self.status['workers'] = [{'name':'B0_270927_train','status':'completed','exit_code':0}]
        (self.run/'status.json').write_text(json.dumps(self.status))
        def one(folder,*args):
            if folder.name != 'B0_270927_train': raise FileNotFoundError('not finished')
            return {'arm':'B0','seed':270927},{},[],{}
        with patch('summarize_runs.run_metrics',side_effect=one): r = self.run_collector()
        self.assertEqual(r['schema_version'],'oe10_partial_run_collection_v1')
        self.assertEqual(len(r['verified_runs']),1)

    def test_controller_not_done_blocks_even_if_six_run_artifacts_exist(self):
        self.status['workers'] = [{'name':f'{arm}_{seed}_train','status':'completed','exit_code':0}
                                  for seed in (270927,270928,270929) for arm in ('B0','B2')]
        (self.run/'status.json').write_text(json.dumps(self.status))
        with patch('summarize_runs.run_metrics',return_value=({}, {}, [], {})): r = self.run_collector()
        self.assertEqual(r['schema_version'],'oe10_partial_run_collection_v1')
        self.assertEqual(len(r['verified_runs']),6)


if __name__ == '__main__': unittest.main(verbosity=2)
