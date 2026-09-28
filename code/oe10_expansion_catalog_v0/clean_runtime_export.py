"""Portable runtime bundle: verify source manifest, exclude only AppleDouble metadata."""
import argparse,hashlib,importlib.util,json,shutil
from pathlib import Path
from datetime import datetime,timezone
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def sidecar(s):return any(x.startswith('._') for x in Path(s).parts)
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--guard',type=Path,required=True);p.add_argument('--experiment-root',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('source_guard',a.guard);guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
before=guard.protect(a.experiment_root);m=read(a.source/'export_manifest.json')
for item in m['files']:
 f=(a.source/item['path']).resolve()
 assert f.is_relative_to(a.source.resolve()) and f.stat().st_size==item['bytes'] and sha(f)==item['sha256']
assert not a.out.exists()
shutil.copytree(a.source,a.out,ignore=shutil.ignore_patterns('._*'))
(a.out/'export_manifest.json').unlink()
prep=read(a.out/'preparation_receipt.json')
assert prep['query_gold_files_opened']==prep['raw_packets_opened']==48
prep['support_mask_reference_root']=str(a.out.resolve());write(a.out/'preparation_receipt.json',prep)
omitted=[x for x in m['files'] if sidecar(x['path'])]
for item in m['files']:
 if not sidecar(item['path']) and item['path']!='preparation_receipt.json':
  assert sha(a.out/item['path'])==item['sha256']
after=guard.protect(a.experiment_root);assert before==after
write(a.out/'portable_export_receipt.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'source_runtime':str(a.source),'source_manifest_sha256':sha(a.source/'export_manifest.json'),'script_sha256':sha(Path(__file__)),'omitted_appledouble_metadata':omitted,'research_data_modified':False,'metadata_change':'support reference root points to new portable bundle','protected_before':before,'protected_after':after,'new_gpu_seconds':0})
files=[{'path':str(f.relative_to(a.out)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(a.out.rglob('*')) if f.is_file()]
assert not any(sidecar(x['path']) for x in files)
write(a.out/'export_manifest.json',{'files':files,'appledouble_included':False})
print(json.dumps({'research_files':len(files),'bytes':sum(x['bytes'] for x in files),'omitted_appledouble':len(omitted),'new_gpu_seconds':0}))

