#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('/home/work/data/olmoearth/oe4_native_v12_v0')
sys.path[:0]=[str(ROOT/'deps'),str(ROOT/'source')]

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 out={'checked_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'imports':{},'packages':{},'protected':{}}
 for p in ['torch','h5py','hdf5plugin','einops','class_registry','upath','huggingface_hub','tacoreader','transformers','yaml']:
  try:
   m=importlib.import_module(p);out['imports'][p]={'ok':True,'file':getattr(m,'__file__',None),'version':getattr(m,'__version__',None)}
  except Exception as e:out['imports'][p]={'ok':False,'error':repr(e)}
 for p in ['torch','olmoearth-pretrain-minimal','hdf5plugin','class-registry','tacoreader','universal-pathlib']:
  try:out['packages'][p]=importlib.metadata.version(p)
  except importlib.metadata.PackageNotFoundError:out['packages'][p]=None
 for p in ['pilot_sen12_gp_heads.py','sen12_official_baselines.py','extract_sen12_fold_cache.py','audit_sen12_fold_cache.py']:
  f=ROOT.parent/'code'/p;st=f.stat();out['protected'][p]={'sha256':sha(f),'mtime_ns':st.st_mtime_ns,'bytes':st.st_size}
 try:
  from olmoearth_pretrain.model_loader import load_model_from_path
  from olmoearth_pretrain.train.loss import LossConfig
  from olmoearth_pretrain.train.masking import MaskingConfig
  out['official_imports']={'ok':True}
 except Exception as e:
  import traceback
  out['official_imports']={'ok':False,'error':repr(e),'traceback':traceback.format_exc()}
 model=ROOT/'models/OlmoEarth-v1_2-Base/config.json'
 if model.exists():out['model_config']=json.loads(model.read_text())
 out['pastis_files']=[{'name':p.name,'bytes':p.stat().st_size} for p in sorted((ROOT.parent/'geobench2/pastis').glob('*')) if p.is_file()]
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,default=str)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['model_config','protected']},default=str))

if __name__=='__main__':main()
