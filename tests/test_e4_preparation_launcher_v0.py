"""CPU-only E4 preparation and mocked-launcher checks against actual frozen E3 bytes."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

SOURCE=Path(__file__).resolve().parents[1]/'code/e4_delta_bundle_v0'
ACTUAL=Path(__file__).resolve().parents[1]/'artifacts/e3_pair_dependence_v1_20260925'
sys.path.insert(0,str(SOURCE))
import e4_delta_runner_v0 as runner
import run_e4_when_idle_v0 as launcher


def load(path):return json.loads(Path(path).read_text())
def save(path,obj):Path(path).write_text(json.dumps(obj)+'\n')


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='e4_prepare_test_')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.parent=self.root/'e3_pair_dependence_v1';self.parent.mkdir()
        for name in ['manifest.json','scores.json','items.jsonl','pairs.npz','saved_e2_real.json','status.json','prereg.json']+[f'answers_seed{s}_real.jsonl'for s in (1,2,3)]:
            (self.parent/name).symlink_to(ACTUAL/name)
        self.out=self.root/'e4_delta_probe_v0';self.out.mkdir()
        self.cfg=load(SOURCE/'e4_delta_probe_prereg_v0.json');self.config=self.root/'plan.json';save(self.config,self.cfg)
        self.patch=mock.patch.object(runner,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
        self.args=argparse.Namespace(config=self.config,out='e4_delta_probe_v0')
    def call(self):
        with contextlib.redirect_stdout(__import__('io').StringIO()):runner.prepare(self.args)
    def test_actual_frozen_e3_preparation_copies_unchanged_inputs(self):
        before={p.name:runner.sha(p)for p in self.parent.iterdir()}
        self.call();manifest=load(self.out/'manifest.json')
        self.assertEqual(load(self.out/'status.json')['status'],'prepared')
        self.assertEqual((manifest['n_items'],manifest['n_generations']),(209,2508))
        for name,key in [('items.jsonl','items_sha256'),('pairs.npz','pairs_sha256')]:
            self.assertEqual(runner.sha(self.out/name),self.cfg['parent'][key]);self.assertEqual(manifest[key],self.cfg['parent'][key])
        refs=load(self.out/'references.json')
        self.assertEqual(refs['e2_real'],refs['e3_real'])
        self.assertEqual(set(refs['e3_real']),{'1','2','3'})
        self.assertTrue(all(len(v)==209 for v in refs['e3_real'].values()))
        audit=load(self.out/'input_audit.json');self.assertEqual(audit['n_items'],209);self.assertTrue(all(x['all_four_finite']for x in audit['all']))
        for name,h in manifest['code_snapshot_sha256'].items():self.assertEqual(runner.sha(self.out/'code_snapshot'/name),h)
        self.assertEqual(before,{p.name:runner.sha(p)for p in self.parent.iterdir()})
        self.assertFalse(any(p.name.startswith('answers_')for p in self.out.iterdir()))
    def test_wrong_frozen_parent_hash_fails_before_copy(self):
        self.cfg['parent']['manifest_sha256']='0'*64;save(self.config,self.cfg)
        with self.assertRaisesRegex(ValueError,'Pinned E3 parent changed'):self.call()
        self.assertEqual(list(self.out.iterdir()),[])
    def test_parent_not_completed_is_rejected(self):
        p=self.parent/'status.json';p.unlink();save(p,{'status':'running'})
        with self.assertRaisesRegex(ValueError,'valid and complete'):self.call()
    def test_checkpoint_pin_change_is_rejected(self):
        self.cfg['checkpoints']['1']='0'*64;save(self.config,self.cfg)
        with self.assertRaisesRegex(ValueError,'Checkpoint pin mismatch'):self.call()
    def test_real_reference_duplicate_id_is_rejected(self):
        p=self.parent/'answers_seed1_real.jsonl';rows=runner.lines(p);rows[1]=rows[0];p.unlink();p.write_text(''.join(json.dumps(row)+'\n'for row in rows))
        self.cfg['parent']['real_answer_sha256']['1']=runner.sha(p);save(self.config,self.cfg)
        with self.assertRaisesRegex(ValueError,'Parent real coverage'):self.call()
    def test_copy_mutation_cannot_be_frozen_as_new_source(self):
        original=shutil.copyfile
        def corrupt(src,dst,*args,**kwargs):
            result=original(src,dst,*args,**kwargs)
            if Path(dst)==self.out/'items.jsonl':
                rows=runner.lines(dst);rows[0]['dates'][0]='1900-01-01';Path(dst).write_text(''.join(json.dumps(row)+'\n'for row in rows))
            return result
        with mock.patch.object(runner.shutil,'copyfile',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'(Copied|copy|prepared|source|SHA|changed|Frozen)'):self.call()
        self.assertFalse((self.out/'manifest.json').exists())


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='e4_launch_test_');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.out=self.root/'e4_delta_probe_v0';self.out.mkdir()
        self.manifest={'code_snapshot_sha256':{'run_e4_when_idle_v0.py':runner.sha(launcher.__file__)}}
        save(self.out/'manifest.json',self.manifest);save(self.out/'status.json',{'status':'prepared'})
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        for name,value in [('ROOT',self.root),('OUT',self.out),('PY',self.root/'fake-python')]:self.stack.enter_context(mock.patch.object(launcher,name,value))
        self.query=self.stack.enter_context(mock.patch.object(launcher.subprocess,'check_output',side_effect=['GPU-FAKE\n','']))
        self.call=self.stack.enter_context(mock.patch.object(launcher.subprocess,'call',return_value=0))
        self.stack.enter_context(contextlib.redirect_stdout(__import__('io').StringIO()))
    def test_inherited_locks_exact_namespace_and_clean_environment(self):
        observed={}
        def child(cmd,**kw):
            observed.update(kw);observed['cmd']=cmd
            self.assertEqual(len(kw['pass_fds']),4)
            for fd in kw['pass_fds']:os.fstat(fd)
            self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'0');self.assertNotIn('PYTHONPATH',kw['env'])
            return 0
        self.call.side_effect=child
        with mock.patch.dict(os.environ,{'PYTHONPATH':'/unrelated/project'}):self.assertEqual(launcher.main(),0)
        names={p.name for p in self.root.glob('.*.lock')}
        self.assertEqual(names,{'.eo_e3_pair_dependence.lock','.eo_reader_gpu0.lock','.eo_reader_gpu1.lock','.eo_e4_delta_probe.lock'})
        self.assertEqual(observed['cmd'][2],str(self.out/'code_snapshot/e4_delta_runner_v0.py'))
        self.assertEqual(observed['cmd'][-1],'run')
        self.assertEqual(load(self.out/'queue.json')['status'],'completed')
        for fd in observed['pass_fds']:
            with self.assertRaises(OSError):os.fstat(fd)
    def test_existing_output_rejected_before_gpu_query(self):
        (self.out/'run.log').write_text('preserve me')
        with self.assertRaisesRegex(ValueError,'Output already exists'):launcher.main()
        self.query.assert_not_called();self.call.assert_not_called();self.assertEqual((self.out/'run.log').read_text(),'preserve me')
    def test_wrong_launcher_hash_rejected_before_gpu_query(self):
        self.manifest['code_snapshot_sha256']['run_e4_when_idle_v0.py']='0'*64;save(self.out/'manifest.json',self.manifest)
        with self.assertRaisesRegex(ValueError,'Launcher not frozen'):launcher.main()
        self.query.assert_not_called();self.call.assert_not_called()
    def test_not_prepared_rejected_before_gpu_query(self):
        save(self.out/'status.json',{'status':'running'})
        with self.assertRaisesRegex(ValueError,'Already run or unprepared'):launcher.main()
        self.query.assert_not_called();self.call.assert_not_called()
    def test_busy_gpu_waits_then_launches_once_without_real_sleep(self):
        self.query.side_effect=['GPU-FAKE\n','GPU-FAKE\n','']
        with mock.patch.object(launcher.time,'sleep') as sleep:
            self.assertEqual(launcher.main(),0);sleep.assert_called_once_with(45)
        self.call.assert_called_once()
        self.assertEqual(self.query.call_args_list[0].args[0][1:3],['-i','0'])


if __name__=='__main__':unittest.main()
