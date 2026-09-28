"""Synthetic E6 gate/copy/queue/lease tests. No actual E5 files or GPU calls.

All filesystem fixtures and advisory locks are in a temporary directory. Every
subprocess/GPU query is mocked; no child process, training or server is started.
"""
import copy
from contextlib import ExitStack, redirect_stdout
import fcntl
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import e6_prepare_v0 as prep
import run_e6_when_idle_v0 as launch


def write(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')


def reference_data():
    items=[]
    for i in range(5989):
        kind='pos' if i%2 else 'neg'
        items.append({'id':f'synthetic_{i}','partition':'train' if i<4234 else 'test',
            'tile':f'synthetic_tile_{i}','cluster':'synthetic_event','phen':'flood','kind':kind,'answer':'yes' if kind=='pos' else 'no'})
    rows=[]
    for seed in (1,2,3):
        for arm,evaluation in [('full','native'),('pair','native'),('later','native'),('delta','native'),('full','full_no_delta')]:
            for i,item in enumerate(items[4234:],4234):
                rows.append({k:item[k] for k in ['id','tile','cluster','phen','kind']}|{
                    'seed':seed,'model_arm':arm,'eval_arm':evaluation,'pair_index':i,
                    'source_gold':item['answer'],'transformed_gold':None,'answer_raw':'yes','parsed':'yes'})
    return items,rows


def report(artifact):
    return {'schema':'e5-independent-result-audit-v0','consistent':True,'artifact':str(artifact.resolve()),
        'checkpoint_tensors_loaded_and_checked_on_cpu':True,'audit_code_sha256':prep.AUDITOR_SHA,
        'n_models':12,'n_steps':19080,'n_training_exposures':152424,'n_answers':26325,
        'primary_n':902,'primary_events':8,'hashes_verified':{},'verdict':'synthetic_only'}


def prepared_fixture(root,wait_hours=6):
    out=root/'e6_no_llm_v0';out.mkdir();snapshot=out/'code_snapshot';snapshot.mkdir()
    for name in prep.SOURCE_FILES:(snapshot/name).write_text('# SYNTHETIC CODE '+name+'\n')
    cfg={'source_files':list(prep.SOURCE_FILES),'compute':{'gpu_index':0,'lock_names':list(launch.LOCKS),'queue_wait_hours':wait_hours}}
    write(out/'prereg.json',cfg);write(out/'reference_audit.json',{'valid':True,'independent_consistent':True})
    manifest={'schema':'e6-no-llm-prepared-v0',
        'code_snapshot_sha256':{n:launch.sha(snapshot/n) for n in prep.SOURCE_FILES},
        'files_sha256':{n:launch.sha(out/n) for n in ['prereg.json','reference_audit.json']}}
    write(out/'manifest.json',manifest);write(out/'status.json',{'status':'prepared'})
    return out


class GateAndSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.items,cls.rows=reference_data()

    def test_exact_audit_schema_tensor_source_and_counts(self):
        artifact=Path('/private/tmp/SYNTHETIC_E5_REFERENCE')
        valid=report(artifact);prep.validate_reference_report(valid,artifact)
        mutations={'schema':'wrong','consistent':False,'artifact':'/other/reference',
            'checkpoint_tensors_loaded_and_checked_on_cpu':False,'audit_code_sha256':'0'*64,
            'n_answers':5265,'n_steps':1590,'primary_events':9,'hashes_verified':None}
        for key,value in mutations.items():
            changed=copy.deepcopy(valid);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):prep.validate_reference_report(changed,artifact)

    def test_only_preselected_full_native_all5265_in_original_order(self):
        selected=prep.select_reference_rows(self.rows,self.items)
        self.assertEqual(len(selected),5265)
        self.assertTrue(all(r['model_arm']=='full' and r['eval_arm']=='native' for r in selected))
        self.assertEqual(selected,[r for r in self.rows if r['model_arm']=='full' and r['eval_arm']=='native'])
        self.assertEqual({r['seed'] for r in selected},{1,2,3})
        self.assertEqual(len({r['id'] for r in selected}),1755)

    def test_duplicate_missing_seed_train_id_or_metadata_reference_rejected(self):
        for key,value in [('seed',4),('id','synthetic_0'),('pair_index',0),('source_gold','WRONG'),('cluster','other'),('seed',True)]:
            rows=list(self.rows);rows[0]={**rows[0],key:value}
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):prep.select_reference_rows(rows,self.items)
        rows=list(self.rows);rows[1]=rows[0]
        with self.assertRaisesRegex(ValueError,'Incomplete/duplicate'):prep.select_reference_rows(rows,self.items)
        rows=list(self.rows);rows[0]={**rows[0],'eval_arm':'full_no_delta'}
        with self.assertRaisesRegex(ValueError,'Incomplete/duplicate'):prep.select_reference_rows(rows,self.items)

    def test_source_snapshot_preserves_and_detects_mid_copy_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir()
            for n in prep.SOURCE_FILES:(bundle/n).write_text('SYNTHETIC '+n)
            pins=prep.snapshot_sources(bundle,root/'copied')
            self.assertEqual(set(pins),set(prep.SOURCE_FILES))
            self.assertTrue(all(prep.sha(root/'copied'/n)==pins[n] for n in pins))
            original_copy=prep.shutil.copyfile
            def change(source,dest):
                result=original_copy(source,dest)
                Path(source).write_text('SYNTHETIC CHANGED')
                return result
            with mock.patch.object(prep.shutil,'copyfile',side_effect=change),self.assertRaisesRegex(ValueError,'Source changed'):
                prep.snapshot_sources(bundle,root/'interrupted')
            self.assertTrue((root/'interrupted').exists())


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve();self.out=prepared_fixture(self.root)
        self.stack=ExitStack()
        self.stack.enter_context(mock.patch.object(launch,'ROOT',self.root))
        self.stack.enter_context(mock.patch.object(launch,'PLAN_SHA',launch.sha(self.out/'prereg.json')))
        self.stack.enter_context(mock.patch.object(launch,'__file__',str(self.out/'code_snapshot/run_e6_when_idle_v0.py')))
        self.stack.enter_context(mock.patch('sys.argv',['run_e6_when_idle_v0.py','--out',str(self.out)]))
    def tearDown(self):self.stack.close();self.temp.cleanup()
    def main(self):
        with redirect_stdout(io.StringIO()):return launch.main()

    def test_exact_source_membership_and_frozen_bytes(self):
        launch.validate_prepared(self.out)
        m=launch.read(self.out/'manifest.json');m['code_snapshot_sha256'].pop('e6_train_v0.py');write(self.out/'manifest.json',m)
        with self.assertRaisesRegex(ValueError,'membership'):launch.validate_prepared(self.out)
        m['code_snapshot_sha256']['e6_train_v0.py']=launch.sha(self.out/'code_snapshot/e6_train_v0.py');write(self.out/'manifest.json',m)
        (self.out/'code_snapshot/e6_train_v0.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'executable changed'):launch.validate_prepared(self.out)

    def test_launch_from_unfrozen_location_or_existing_outputs_rejected(self):
        with mock.patch.object(launch,'__file__',str(self.root/'elsewhere.py')),self.assertRaisesRegex(ValueError,'frozen launcher'):
            launch.validate_prepared(self.out)
        for name in ('run.log','run_claim.json','scores.json','failure.json','initial_states','models'):
            p=self.out/name;p.touch()
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'no resume'):launch.validate_prepared(self.out)
            p.unlink()

    def test_false_or_changed_reference_gate_rejected(self):
        write(self.out/'reference_audit.json',{'valid':False,'independent_consistent':True})
        with self.assertRaisesRegex(ValueError,'reference gate'):launch.validate_prepared(self.out)
        write(self.out/'reference_audit.json',{'valid':True,'independent_consistent':True,'changed':True})
        with self.assertRaisesRegex(ValueError,'gate changed'):launch.validate_prepared(self.out)

    def test_idle_gpu_launch_inherits_all_five_held_locks_and_exact_identity(self):
        def child(command,**kwargs):
            self.assertEqual(command,[str(self.root/'.venv-master/bin/python'),'-B',str(self.out/'code_snapshot/e6_train_v0.py'),'--out',str(self.out)])
            env=kwargs['env'];fds=json.loads(env['E6_LOCK_FDS'])
            self.assertEqual(set(fds),set(launch.LOCKS));self.assertEqual(set(kwargs['pass_fds']),set(fds.values()))
            self.assertEqual((env['CUDA_VISIBLE_DEVICES'],env['E6_GPU_INDEX'],env['E6_GPU_UUID']),('0','0','GPU-SYNTHETIC'))
            self.assertEqual(env['E6_MANIFEST_SHA256'],launch.sha(self.out/'manifest.json'))
            self.assertNotIn('PYTHONPATH',env)
            for name,fd in fds.items():
                self.assertEqual(os.fstat(fd).st_ino,(self.root/name).stat().st_ino)
                with (self.root/name).open('a+') as other:
                    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            write(self.out/'status.json',{'status':'completed'});write(self.out/'scores.json',{'valid':True});return 0
        with mock.patch.object(launch.subprocess,'check_output',side_effect=['GPU-SYNTHETIC\n','']),mock.patch.object(launch.subprocess,'call',side_effect=child) as call:
            self.assertEqual(self.main(),0);call.assert_called_once()
        self.assertEqual(launch.read(self.out/'queue.json')['status'],'completed')
        for name in launch.LOCKS:
            with (self.root/name).open('a+') as released:fcntl.flock(released,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_busy_lease_aborts_without_gpu_query_or_child(self):
        with (self.root/launch.LOCKS[0]).open('a+') as other:
            fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with mock.patch.object(launch.subprocess,'check_output') as query,mock.patch.object(launch.subprocess,'call') as child:
                with self.assertRaises(BlockingIOError):self.main()
                query.assert_not_called();child.assert_not_called()
        self.assertFalse((self.out/'run.log').exists())

    def test_gpu_becomes_free_after_deadline_never_launches(self):
        clock=[0.]
        cfg=launch.read(self.out/'prereg.json');cfg['compute']['queue_wait_hours']=1/3600
        write(self.out/'prereg.json',cfg)
        m=launch.read(self.out/'manifest.json');m['files_sha256']['prereg.json']=launch.sha(self.out/'prereg.json');write(self.out/'manifest.json',m)
        def sleep(seconds):clock[0]+=seconds
        with (mock.patch.object(launch,'PLAN_SHA',launch.sha(self.out/'prereg.json')),
              mock.patch.object(launch.time,'monotonic',side_effect=lambda:clock[0]),
              mock.patch.object(launch.time,'sleep',side_effect=sleep),
              mock.patch.object(launch.subprocess,'check_output',side_effect=['GPU-SYNTHETIC\n','GPU-SYNTHETIC\n','']),
              mock.patch.object(launch.subprocess,'call') as child):
            self.assertEqual(self.main(),3);child.assert_not_called()
        self.assertEqual(launch.read(self.out/'queue.json')['status'],'wait_expired_without_launch')
        self.assertFalse((self.out/'run.log').exists())

    def test_source_or_manifest_change_during_queue_blocks_child(self):
        for kind in ('code','manifest'):
            with self.subTest(kind=kind):
                changed=self.out/'code_snapshot/e6_train_v0.py' if kind=='code' else self.out/'manifest.json'
                before=changed.read_bytes()
                def sleep(seconds):
                    if kind=='code':changed.write_bytes(before+b'# SYNTHETIC MUTATION\n')
                    else:
                        value=launch.read(changed);value['synthetic_mutation']=True;write(changed,value)
                with (mock.patch.object(launch.time,'sleep',side_effect=sleep),
                      mock.patch.object(launch.subprocess,'check_output',side_effect=['GPU-SYNTHETIC\n','GPU-SYNTHETIC\n','']),
                      mock.patch.object(launch.subprocess,'call') as child):
                    with self.assertRaises(ValueError):self.main()
                    child.assert_not_called()
                changed.write_bytes(before)
                self.assertFalse((self.out/'run.log').exists())

    def test_nonzero_child_is_failed_and_runlog_prevents_retry(self):
        with (mock.patch.object(launch.subprocess,'check_output',side_effect=['GPU-SYNTHETIC\n','']),
              mock.patch.object(launch.subprocess,'call',return_value=7)):
            self.assertEqual(self.main(),7)
        self.assertEqual(launch.read(self.out/'queue.json')['status'],'failed')
        self.assertTrue((self.out/'run.log').is_file())
        with self.assertRaisesRegex(ValueError,'no resume'):launch.validate_prepared(self.out)


class PreparationFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
        self.ref=self.root/'e5_equal_budget_v0';self.ref.mkdir();self.bundle=self.root/'bundle';self.bundle.mkdir()
        for name in set(prep.INPUT_FILES+prep.REFERENCE_FILES):
            (self.ref/name).write_bytes(b'SYNTHETIC_PAIRS_NOT_A_NUMPY_ARRAY' if name=='pairs.npy' else b'{}\n')
        write(self.ref/'status.json',{'status':'completed','verdict':'synthetic_only'})
        write(self.ref/'scores.json',{'valid':True,'verdict':'synthetic_only'})
        for name in prep.SOURCE_FILES:(self.bundle/name).write_text('# SYNTHETIC '+name+'\n')
        self.parent_sha=prep.sha(self.ref/'manifest.json')
        a=report(self.ref);a['hashes_verified']={str(self.ref/name):prep.sha(self.ref/name) for name in set(prep.INPUT_FILES+prep.REFERENCE_FILES)}
        self.audit=self.root/'e5_independent_audit_20260925.json';write(self.audit,a)
        cfg={'source_files':list(prep.SOURCE_FILES),'root':str(self.root),'output_directory':'e6_no_llm_v0',
            'reference':{'directory':str(self.ref),'independent_audit':str(self.audit),
                'prepared_manifest_sha256':self.parent_sha,'prereg_sha256':prep.sha(self.ref/'prereg.json'),
                'files_sha256':{name:prep.sha(self.ref/name) for name in prep.INPUT_FILES}}}
        write(self.bundle/'e6_no_llm_prereg_v0.json',cfg)
        self.out=self.root/'e6_no_llm_v0';self.stack=ExitStack()
        self.stack.enter_context(mock.patch.object(prep,'ROOT',self.root))
        self.stack.enter_context(mock.patch.object(prep,'PARENT_SHA',self.parent_sha))
        self.stack.enter_context(mock.patch.object(prep,'PLAN_SHA',prep.sha(self.bundle/'e6_no_llm_prereg_v0.json')))
        # Full selection is tested above; this fixture isolates failure handling.
        self.stack.enter_context(mock.patch.object(prep,'select_reference_rows',return_value=[]))
    def tearDown(self):self.stack.close();self.temp.cleanup()

    def test_reference_bytes_must_match_independent_audit_before_output_created(self):
        (self.ref/'pairs.npy').write_bytes(b'SYNTHETIC_TAMPER')
        with self.assertRaisesRegex(ValueError,'differs from independent audit'):prep.prepare(self.out,self.bundle)
        self.assertFalse(self.out.exists())

    def test_partial_source_snapshot_failure_is_retained_and_no_retry(self):
        def fail(bundle,destination):
            destination.mkdir();(destination/'partial.txt').write_text('SYNTHETIC partial copy')
            raise RuntimeError('synthetic snapshot failure')
        with mock.patch.object(prep,'snapshot_sources',side_effect=fail),self.assertRaisesRegex(RuntimeError,'snapshot failure'):
            prep.prepare(self.out,self.bundle)
        self.assertEqual(prep.read(self.out/'status.json')['status'],'failed')
        self.assertEqual(prep.read(self.out/'failure.json')['phase'],'preparation')
        self.assertTrue((self.out/'code_snapshot/partial.txt').exists())
        self.assertFalse((self.out/'manifest.json').exists())
        with self.assertRaisesRegex(ValueError,'no silent resume'):prep.prepare(self.out,self.bundle)


if __name__=='__main__':unittest.main()
