"""Read-only stage identity check; no torch, arrays, GPU, ledger writes or launch."""
import argparse,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def require(ok,msg):
 if not ok:raise ValueError(msg)
def main(a):
 require(sha(a.reference)==a.reference_sha256,'external protected reference changed')
 ref=json.loads(a.reference.read_text())['protected_before']
 got={}
 for name,row in ref['protected'].items():
  p=a.root/'code'/name
  require(p.is_file() and not p.is_symlink(),'protected file missing or symlink')
  got[name]={'sha256':sha(p),'mtime_ns':p.stat().st_mtime_ns,'bytes':p.stat().st_size}
 require(got==ref['protected'],'protected files changed')
 snapshot={}
 for name,h in ref['snapshot'].items():
  p=a.root/'oe10_p2_v0/code_snapshot/oe10_p2_v3'/name
  require(p.is_file() and not p.is_symlink(),'P2 file missing or symlink')
  snapshot[name]=sha(p)
 require(snapshot==ref['snapshot'],'P2 snapshot changed')
 protocol=sha(a.root/'oe10_p2_v0/config/oe10_p2_execution_protocol_20260927.json')
 require(protocol==ref['protocol'],'P2 protocol changed')
 output={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'protected_sources_and_protocol_verified',
 'protected':got,'snapshot':snapshot,'protocol_sha256':protocol,
 'reference_sha256':a.reference_sha256,'script_sha256':sha(Path(__file__)),
 'new_gpu_seconds':0,'actual_model_loaded':False,'budget_mutated':False}
 with a.out.open('x') as f:json.dump(output,f,indent=2);f.write('\n')
 print(json.dumps({'status':output['status'],'protected_files':len(got),'p2_snapshot_files':len(snapshot),'output_sha256':sha(a.out)}))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('root','reference','out'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--reference-sha256',required=True)
 main(p.parse_args())
