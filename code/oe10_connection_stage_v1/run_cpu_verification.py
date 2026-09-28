"""CPU-only staged launcher verification; no production execute/reserve."""
import argparse,hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
 root=Path('/home/work/data/olmoearth/oe10_text_mask_identity_v2')
 code=root/'code_snapshot/oe10_connection_launcher_v0'
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU verification requires empty CUDA_VISIBLE_DEVICES')
 if sha(code/'source_manifest.json')!=a.launcher_sha:raise ValueError('Launcher manifest mismatch')
 manifest=json.loads((code/'source_manifest.json').read_text())
 for row in manifest['files']:
  p=code/row['path']
  if sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('Source payload mismatch')
 a.out.mkdir(parents=True,exist_ok=False)
 def production_state():
  names=('pilot_budget_ledger_v0.json','pilot_budget_ledger_v0.lock','pilot_budget_ledger_v0.initialized.json','connection_runs_v0')
  state={}
  for name in names:
   p=root/name
   if p.is_file():state[name]={'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns}
   elif p.is_dir():state[name]={'files':{str(x.relative_to(p)):sha(x) for x in p.rglob('*') if x.is_file()}}
   else:state[name]=None
  return state
 before=production_state();env=os.environ.copy();env.pop('PYTHONPATH',None)
 # The outside-scope preflight fixture must actually be outside the production root.
 temp=root.parent/'oe10_connection_cpu_fixtures_v1';temp.mkdir(exist_ok=False);env['TMPDIR']=str(temp)
 tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(code),'-p','test_launcher.py','-v'],env=env,capture_output=True,text=True,timeout=60)
 (a.out/'cpu_tests.log').write_text(tests.stdout+tests.stderr)
 protocol=root/'config/oe10_text_mask_connection_protocol_20260928.json'
 initial=root/'config/oe10_pilot_budget_bootstrap_20260928.json'
 command=[sys.executable,'-B',str(code/'launcher.py'),'--preflight-only','--protocol',str(protocol),
 '--initial-budget',str(initial),'--initial-budget-sha','0c7a2c815fc12270a68fa5cb9e8739a36614efe52837d94dcd13bd47b094cd20',
 '--launcher-sha',a.launcher_sha,'--job-id','prospective_connection_01','--seconds','1800']
 probe=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
 (a.out/'preflight_stdout.json').write_text(probe.stdout);(a.out/'preflight_stderr.log').write_text(probe.stderr)
 report=json.loads(probe.stdout)
 after=production_state()
 checks={'tests_passed':tests.returncode==0 and 'Ran 17 tests' in tests.stderr and '\nOK\n' in tests.stderr and 'skipped=' not in tests.stderr,
 'preflight_blocked':probe.returncode==2 and report['ready'] is False,
 'live_p2_blocker_present':any('P2 still running' in x for x in report['reasons']),
 'source_identity_verified':all(k in report['evidence'] for k in ('launcher_source','model_source','episodes')),
 'production_state_unchanged':before==after,
 'production_state_absent':all(v is None for v in before.values())}
 result={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'pass' if all(checks.values()) else 'fail','checks':checks,
 'test_returncode':tests.returncode,'preflight_returncode':probe.returncode,'preflight_reasons':report['reasons'],
 'source_manifest_sha256':a.launcher_sha,'script_sha256':sha(Path(__file__)),'command':command,'production_before':before,'production_after':after,
 'actual_models_loaded':False,'new_gpu_seconds':0,'budget_reserved_seconds':0,
 'scope':'CPU boundary tests and rejection while existing P2 is running; not a successful production launch or weight forward'}
 (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
 files=[{'path':str(p.relative_to(a.out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.out.rglob('*')) if p.is_file()]
 (a.out/'export_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
 print(json.dumps(result))
 if result['status']!='pass':raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--launcher-sha',required=True);p.add_argument('--out',type=Path,required=True)
 main(p.parse_args())
