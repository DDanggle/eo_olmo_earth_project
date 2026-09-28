import importlib.util,json,os,tempfile,unittest
from pathlib import Path
from unittest import mock

SOURCE=Path(__file__).resolve().parent/'run_e5_when_idle_v0.py'
spec=importlib.util.spec_from_file_location('e5_launcher',SOURCE);launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
class LauncherTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name).resolve();self.out=self.root/'e5_equal_budget_v0';self.snap=self.out/'code_snapshot';self.snap.mkdir(parents=True)
  self.src=self.snap/SOURCE.name;self.src.write_bytes(SOURCE.read_bytes());self.runner=self.snap/'e5_train_v0.py';self.runner.write_text('frozen dummy source')
  self.cfg={'compute':{'gpu_index':0,'lock_names':launch.LOCKS,'queue_wait_hours':6}}
  self.save('prereg.json',self.cfg);self.save('status.json',{'status':'prepared'})
  self.manifest={'files_sha256':{'prereg.json':launch.sha(self.out/'prereg.json')},'code_snapshot_sha256':{p.name:launch.sha(p) for p in self.snap.iterdir()}}
  self.save('manifest.json',self.manifest)
  for name,value in [('ROOT',self.root),('__file__',str(self.src))]:
   patch=mock.patch.object(launch,name,value);patch.start();self.addCleanup(patch.stop)
 def save(self,name,obj):
  p=self.out/name;p.write_text(json.dumps(obj));return p
 def run_launch(self):return launch.main(['--out',str(self.out)])
 def check_output(self,cmd,**kwargs):
  return 'GPU-test\n' if '-i' in cmd else ''
 def test_passes_alive_lease_fds_and_exact_environment(self):
  def child(cmd,**kw):
   self.assertEqual(cmd[-2:],['--out',str(self.out)]);self.assertNotIn('PYTHONPATH',kw['env']);self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'0')
   self.assertEqual(kw['env']['E5_MANIFEST_SHA256'],launch.sha(self.out/'manifest.json'))
   fds=json.loads(kw['env']['E5_LOCK_FDS']);self.assertEqual(list(fds),launch.LOCKS);self.assertEqual(tuple(fds.values()),kw['pass_fds'])
   for name,fd in fds.items():self.assertEqual(os.fstat(fd).st_ino,(self.root/name).stat().st_ino)
   self.save('status.json',{'status':'completed'});self.save('scores.json',{'valid':True});return 0
  with mock.patch.object(launch.subprocess,'check_output',side_effect=self.check_output),mock.patch.object(launch.subprocess,'call',side_effect=child),mock.patch.dict(os.environ,{'PYTHONPATH':'contamination'}):
   self.assertEqual(self.run_launch(),0)
  self.assertEqual(json.loads((self.out/'queue.json').read_text())['status'],'completed')
 def test_zero_exit_without_valid_science_is_failed(self):
  with mock.patch.object(launch.subprocess,'check_output',side_effect=self.check_output),mock.patch.object(launch.subprocess,'call',return_value=0):self.assertEqual(self.run_launch(),1)
  self.assertEqual(json.loads((self.out/'queue.json').read_text())['status'],'failed')
 def test_existing_output_and_nonprepared_status_stop_before_gpu_query(self):
  for status,existing in [('completed',False),('prepared',True)]:
   with self.subTest(status=status):
    self.save('status.json',{'status':status})
    if existing:(self.out/'run.log').write_text('preserve')
    with mock.patch.object(launch.subprocess,'check_output') as query,self.assertRaises(ValueError):self.run_launch()
    query.assert_not_called()
 def test_wrong_frozen_source_hash_stops_before_gpu_query(self):
  self.src.write_text('changed')
  with mock.patch.object(launch.subprocess,'check_output') as query,self.assertRaises(ValueError):self.run_launch()
  query.assert_not_called()
 def test_busy_gpu_waits_without_launching_foreign_work(self):
  self.cfg['compute']['queue_wait_hours']=0;self.save('prereg.json',self.cfg);self.manifest['files_sha256']['prereg.json']=launch.sha(self.out/'prereg.json');self.save('manifest.json',self.manifest)
  with mock.patch.object(launch.subprocess,'check_output',return_value='GPU-test\n'),mock.patch.object(launch.subprocess,'call') as child:self.assertEqual(self.run_launch(),3)
  child.assert_not_called()
  self.assertEqual(json.loads((self.out/'queue.json').read_text())['status'],'wait_expired_without_launch')
 def test_changed_manifest_while_waiting_rejected(self):
  def change(_):
   self.manifest['mutation']='after-freeze';self.save('manifest.json',self.manifest)
  with mock.patch.object(launch.subprocess,'check_output',side_effect=['GPU-test\n','GPU-test\n','']),mock.patch.object(launch.time,'sleep',side_effect=change),mock.patch.object(launch.subprocess,'call') as child,self.assertRaisesRegex(ValueError,'freeze changed'):self.run_launch()
  child.assert_not_called()
if __name__=='__main__':unittest.main()
