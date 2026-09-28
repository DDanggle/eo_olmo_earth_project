"""Freeze only separate preparation code/text artifacts into a portable transfer tar."""
import argparse,hashlib,json,tarfile
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(a):
 base=a.repo.resolve(); own=base/'code/oe10_expansion_context_v0';model=base/'code/oe10_text_mask_v0'
 if not (model/'source_manifest.json').is_file():raise ValueError('wait for model source freeze')
 for name,h in json.loads((model/'source_manifest.json').read_text())['files'].items():
  if sha(model/name)!=h:raise ValueError('model code changed')
 ownfiles={p.relative_to(own).as_posix():sha(p) for p in sorted(own.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='source_manifest.json' and not p.name.startswith('._')}
 (own/'source_manifest.json').write_text(json.dumps({'created_utc':datetime.now(timezone.utc).isoformat(),'files':ownfiles},indent=2)+'\n')
 mapping=[(own,'code_snapshot/oe10_expansion_context_v0'),(model,'code_snapshot/oe10_text_mask_v0'),(base/'artifacts/oe10_text_mask_prepare_20260928/contexts_v0','contexts_v0')]
 a.out.parent.mkdir(parents=True,exist_ok=True)
 records=[]
 with tarfile.open(a.out,'x:gz',format=tarfile.USTAR_FORMAT) as tar:
  for folder,prefix in mapping:
   for p in sorted(folder.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name.startswith('._'):continue
    name=prefix+'/'+p.relative_to(folder).as_posix()
    info=tar.gettarinfo(str(p),arcname=name);info.uid=0;info.gid=0;info.uname='';info.gname='';info.mtime=0
    with p.open('rb') as f:tar.addfile(info,f)
    records.append({'path':name,'sha256':sha(p),'bytes':p.stat().st_size})
 receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'bundle_sha256':sha(a.out),'bundle_bytes':a.out.stat().st_size,'files':records,'new_gpu_seconds':0}
 a.out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'files':len(records),'bytes':receipt['bundle_bytes'],'sha256':receipt['bundle_sha256']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
