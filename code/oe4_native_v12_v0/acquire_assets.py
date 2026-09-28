#!/usr/bin/env python3
"""Pinned public assets for a bounded development run; no GPU usage."""
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

ROOT=Path('/home/work/data/olmoearth/oe4_native_v12_v0')
REV='2e99a734a30e9aeb993aaa39946c6dcf554739e0'
ASSETS=[
 ('official_subset_1k', 'https://storage.googleapis.com/ai2-olmoearth-projects-public-data/pretraining_data/subset_1k/subset_1k_h5py_data.tar', 'downloads/subset_1k_h5py_data.tar', 17855037440),
 ('base_config',f'https://huggingface.co/allenai/OlmoEarth-v1_2-Base/resolve/{REV}/config.json','models/OlmoEarth-v1_2-Base/config.json',None),
 ('base_weights',f'https://huggingface.co/allenai/OlmoEarth-v1_2-Base/resolve/{REV}/weights.pth','models/OlmoEarth-v1_2-Base/weights.pth',None),
]

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()

def download(asset):
 name,url,rel,expected=asset
 path=ROOT/rel;path.parent.mkdir(parents=True,exist_ok=True)
 receipt=path.with_suffix(path.suffix+'.receipt.json')
 if path.exists():
  old=json.loads(receipt.read_text())
  if sha(path)!=old['sha256']:raise RuntimeError(f'Existing asset hash mismatch: {path}')
  return old
 part=path.with_suffix(path.suffix+'.part')
 if part.exists():raise RuntimeError(f'Incomplete previous download needs explicit review: {part}')
 h=hashlib.sha256();n=0;t0=last=time.monotonic()
 req=urllib.request.Request(url,headers={'User-Agent':'OlmoEarth-research-OE4/1.0'})
 with urllib.request.urlopen(req,timeout=60) as r,part.open('xb') as f:
  length=r.headers.get('Content-Length');etag=r.headers.get('ETag')
  while block:=r.read(8<<20):
   f.write(block);h.update(block);n+=len(block)
   if time.monotonic()-last>20:
    print(json.dumps({'asset':name,'bytes':n,'seconds':round(time.monotonic()-t0,1)}),flush=True);last=time.monotonic()
 if expected is not None and n!=expected:raise RuntimeError(f'Expected {expected} bytes, got {n}')
 if length and n!=int(length):raise RuntimeError(f'HTTP size mismatch {name}')
 value={'name':name,'url':url,'relative_path':rel,'bytes':n,'sha256':h.hexdigest(),'etag':etag,'download_seconds':time.monotonic()-t0,'completed_utc':datetime.now(timezone.utc).isoformat()}
 part.rename(path);receipt.write_text(json.dumps(value,indent=2)+'\n')
 print(json.dumps({'download_complete':name,'bytes':n,'sha256':value['sha256']}),flush=True)
 return value

def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  receipts=list(pool.map(download,ASSETS))
 target=ROOT/'data/official_subset_1k'
 if target.exists():raise FileExistsError(target)
 target.mkdir(parents=True)
 members=[]
 with tarfile.open(ROOT/'downloads/subset_1k_h5py_data.tar','r|') as tf:
  for member in tf:
   dest=(target/member.name).resolve()
   if not dest.is_relative_to(target.resolve()):raise RuntimeError('Archive path outside target')
   if member.isdir():dest.mkdir(parents=True,exist_ok=True)
   elif member.isfile():
    dest.parent.mkdir(parents=True,exist_ok=True)
    src=tf.extractfile(member)
    with dest.open('xb') as out:shutil.copyfileobj(src,out,8<<20)
    members.append({'name':member.name,'bytes':member.size})
   else:raise RuntimeError(f'Unexpected archive member type: {member.name}')
 result={'status':'assets_acquired_not_model_validation','model_revision':REV,'assets':receipts,'members':members,'h5_count':sum(x['name'].endswith(('.h5','.hdf5')) for x in members),'completed_utc':datetime.now(timezone.utc).isoformat()}
 (ROOT/'asset_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'complete':True,'h5_count':result['h5_count'],'members':len(members)}),flush=True)

if __name__=='__main__':main()
