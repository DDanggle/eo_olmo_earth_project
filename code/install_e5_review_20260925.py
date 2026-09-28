"""Verify and install the small E5 result replica; do not claim a local full audit."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

repo=Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
archive=Path('/private/tmp/e5_download_20260925/e5_equal_budget_v0_review_20260925.tar.gz')
expected='b0ff71ae045e90d1144ba165c9af83084798b4fad072ccd373fc627ca499a286'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive)==expected
out=repo/'artifacts/e5_equal_budget_v0_review_20260925'
out.mkdir(exist_ok=False)
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers()
 assert sum(m.size for m in members)<512*1024**2
 for m in members:
  p=PurePosixPath(m.name)
  assert not p.is_absolute() and '..' not in p.parts and (m.isfile() or m.isdir())
  assert p.parts[0] in ('e5_equal_budget_v0','e5_independent_audit_20260925.json')
 t.extractall(out,filter='data')
root=out/'e5_equal_budget_v0'
audit=json.loads((out/'e5_independent_audit_20260925.json').read_text())
assert audit['consistent'] is True and audit['checkpoint_tensors_loaded_and_checked_on_cpu'] is True
original=PurePosixPath(audit['artifact'])
checked={};not_downloaded=[]
for key,digest in audit['hashes_verified'].items():
 p=PurePosixPath(key)
 if not p.is_relative_to(original):continue
 relative=p.relative_to(original);local=root/str(relative)
 if local.is_file():
  assert sha(local)==digest, key
  checked[str(relative)]=digest
 else:not_downloaded.append(str(relative))
assert all(name in checked for name in ('manifest.json','status.json','scores.json','prereg.json','predictions.jsonl','training_summary.json'))
report={'schema':'e5-small-replica-integrity-v0','checked_at':datetime.now(timezone.utc).isoformat(),
 'archive_sha256':expected,'archive_bytes':archive.stat().st_size,'source_audit_root':str(original),
 'local_replica':str(root),'verified_local_files':checked,'not_downloaded':not_downloaded,
 'independent_full_audit_ran_on_server':True,'local_full_audit_rerun':False,
 'scope':'Exact replica hashes for downloaded files. Server audit also verified original pairs.npy and all checkpoint tensors; these large files were intentionally not downloaded.'}
(out/'replica_integrity.json').write_text(json.dumps(report,indent=2)+'\n')
s=json.loads((root/'scores.json').read_text())
compact={'verdict':s['verdict'],'valid':s['valid'],'seed_decisions':s['seed_decisions'],'primary':{},'secondary':{}}
for seed in ('1','2','3'):
 p=s['metrics'][seed]['primary_same_prompt']
 compact['primary'][seed]={'ba':{k:v['macro_ba'] for k,v in p['evaluations'].items()},'contrasts':p['contrasts']}
 compact['secondary'][seed]={
  'paired_flood':{k:v['macro_ba'] for k,v in s['metrics'][seed]['paired_source']['flood']['evaluations'].items()},
  'hard_fpr_pooled':{k:v['fpr_pooled'] for k,v in s['metrics'][seed]['hard_negative'].items()}}
print(json.dumps({'replica_files_verified':len(checked),'large_files_not_downloaded':len(not_downloaded),'results':compact},indent=2))
