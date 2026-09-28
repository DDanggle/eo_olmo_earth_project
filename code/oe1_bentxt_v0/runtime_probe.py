#!/usr/bin/env python3
"""Read-only runtime inventory and explicitly requested public model acquisition for OE1."""
import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path('/home/work/data/olmoearth')
DEPS = ROOT / 'region_language_contract_20260921/deps'
if DEPS.is_dir():
    sys.path.insert(0, str(DEPS))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--download-model', action='store_true')
    ap.add_argument('--out', type=Path, default=ROOT/'oe1_bentxt_v0/runtime.json')
    a=ap.parse_args()
    out={'python':sys.version, 'executable':sys.executable, 'packages':{}, 'files':{}}
    for name in ['torch','transformers','olmoearth-pretrain-minimal','rslearn','rasterio','pyarrow','zstandard','huggingface-hub','safetensors']:
        try: out['packages'][name]=md.version(name)
        except md.PackageNotFoundError: out['packages'][name]=None
    import torch
    import olmoearth_pretrain_minimal
    out['minimal_path']=olmoearth_pretrain_minimal.__file__
    out['cuda_available']=torch.cuda.is_available()
    out['devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    for name in ['region_language_contract_20260921/BigEarthNet.txt.parquet','geobench2/benv2/geobench_benv2.tortilla','olmo_llm/Olmo-3-7B-Instruct/config.json']:
        p=ROOT/name
        out['files'][name]={'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None}
    from huggingface_hub import HfApi, snapshot_download
    if a.download_model:
        repo='allenai/OlmoEarth-v1-Tiny'
        revision=HfApi().model_info(repo).sha
        directory=ROOT/'oe1_bentxt_v0/models/OlmoEarth-v1-Tiny'
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(directory),
                          allow_patterns=['config.json','model.safetensors','*.json','*.pt','*.pth'])
        out['model']={'repo':repo,'revision':revision,'path':str(directory),'files':{}}
        for p in sorted(directory.iterdir()):
            if p.is_file():
                out['model']['files'][p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
