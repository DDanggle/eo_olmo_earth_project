#!/usr/bin/env python3
"""Bounded, metadata-only readiness inventory. No model/data load or training."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path('/home/work/data/olmoearth')
PROTECTED=('pilot_sen12_gp_heads.py','sen12_official_baselines.py','extract_sen12_fold_cache.py','audit_sen12_fold_cache.py')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def protected():
    return {n:{'sha256':sha(ROOT/'code'/n),'size':(ROOT/'code'/n).stat().st_size,
               'mtime_ns':(ROOT/'code'/n).stat().st_mtime_ns} for n in PROTECTED}

def describe(p):
    item={'path':str(p),'exists':p.exists()}
    if not p.exists():return item
    item['is_directory']=p.is_dir()
    if p.is_dir():
        children=sorted(p.iterdir(),key=lambda x:x.name)
        item['direct_entry_count']=len(children)
        item['first_24_entries']=[{'name':x.name,'directory':x.is_dir(),'bytes':x.stat().st_size if x.is_file() else None} for x in children[:24]]
    else:item['bytes']=p.stat().st_size
    return item

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    before=protected()
    names=['geobench','geobench2','research_benchmarks','floods','sen1floods11','Sen1Floods11',
           'mados','pastis','pastis_r','olmoearth_pretrain_dataset','pretrain','pretraining',
           'oe1_bentxt_v0/data','oe1_bentxt_v0/models/OlmoEarth-v1-Tiny','models',
           'olmoearth_pretrain','olmoearth_pretrain_minimal']
    related=[p for p in ROOT.iterdir() if any(k in p.name.lower() for k in ('pretrain','geobench','flood','pastis','mados','cashew','olmoearth'))]
    paths=sorted({ROOT/n for n in names}|set(related))
    distributions={}
    for package in ('torch','olmoearth-pretrain','olmoearth-pretrain-minimal','ai2-olmo-core','geobench','h5py','rasterio','transformers'):
        try:distributions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:distributions[package]=None
    model=ROOT/'oe1_bentxt_v0/models/OlmoEarth-v1-Tiny'
    model_pins={n:sha(model/n) for n in ('config.json','weights.pth') if (model/n).is_file()}
    after=protected()
    if before!=after:raise RuntimeError('Protected existing code changed')
    value={'status':'metadata_inventory_only_not_dataset_readiness_or_baseline_result',
           'checked_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,
           'source_sha256':sha(Path(__file__)),'paths':[describe(p) for p in paths],
           'installed_distributions':distributions,'existing_tiny_files_sha256':model_pins,
           'protected_before':before,'protected_after':after,'protected_unchanged':True,
           'limitations':['Directory existence does not establish completeness, correct version, georeferencing, split or original training-target availability.',
                          'No recursive exhaustive search; unlisted locations may contain additional assets.',
                          'No import/forward/backward compatibility test; distribution presence is metadata only.',
                          'No download, model inference, optimizer update, API call or remote original-file modification.']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({'status':value['status'],'out':str(a.out),'paths_checked':len(paths),'protected_unchanged':True}))

if __name__=='__main__':main()
