#!/usr/bin/env python3
"""Bounded public-data audit and optional frozen OlmoEarth smoke. Not an accuracy experiment."""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import time
import urllib.request

ROOT = Path('/home/work/data/olmoearth/region_language_contract_20260921')
sys.path.insert(0, str(ROOT / 'deps'))  # isolated optional readers; never mutate shared venv
TEXT_URL = 'https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt/resolve/main/BigEarthNet.txt.parquet'
IMAGE_URL = 'https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1'
BANDS = ('B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09')

def save(name, value):
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')

def download(url, path, limit):
    if path.exists():
        return
    tmp = path.with_suffix(path.suffix + '.partial')
    n = 0
    with urllib.request.urlopen(url, timeout=90) as src, tmp.open('wb') as dst:
        while chunk := src.read(1 << 20):
            n += len(chunk)
            if n > limit:
                raise RuntimeError(f'download exceeds {limit} bytes')
            dst.write(chunk)
    tmp.rename(path)
    print('downloaded', path.name, n, flush=True)

def sample_images(limit=8):
    import zstandard
    class LimitedReader:
        def __init__(self, raw): self.raw, self.n = raw, 0
        def read(self, size=-1):
            chunk = self.raw.read(min(size if size >= 0 else 131072, 131072))
            self.n += len(chunk)
            if self.n > (32 << 20): raise RuntimeError('compressed image prefix exceeded 32 MiB')
            return chunk
    done = {}
    downloaded = 0
    status = {'url': IMAGE_URL, 'sampling': 'archive prefix, non-representative', 'limit': limit}
    try:
        with urllib.request.urlopen(IMAGE_URL, timeout=90) as raw:
            source = LimitedReader(raw)
            with zstandard.ZstdDecompressor().stream_reader(source) as decompressed:
                with tarfile.open(fileobj=decompressed, mode='r|') as archive:
                    for member in archive:
                        if not member.isfile(): continue
                        base = Path(member.name).name
                        match = re.fullmatch(r'(S2[AB]_MSIL2A_.*_\d+_\d+)_(B(?:\d\d|8A))\.tif', base)
                        if not match or match[2] not in BANDS: continue
                        patch, band = match.groups()
                        if patch not in done and len(done) >= limit: continue
                        target = ROOT / 'images' / patch / base
                        target.parent.mkdir(parents=True, exist_ok=True)
                        stream = archive.extractfile(member)
                        assert stream is not None
                        content = stream.read()
                        target.write_bytes(content)
                        done.setdefault(patch, set()).add(band)
                        if len(done) >= limit and all(len(b) == 12 for b in done.values()): break
            downloaded = source.n
        status['status'] = 'complete' if len(done) == limit and all(len(v)==12 for v in done.values()) else 'partial'
    except Exception as exc:
        status.update(status='failed_or_partial', error=repr(exc))
    status.update(compressed_bytes_read=downloaded, patches={k: sorted(v) for k,v in done.items()})
    save('image_sampling.json', status)
    print(json.dumps(status), flush=True)
    return set(k for k,v in done.items() if len(v)==12)

def audit():
    import pyarrow.parquet as pq
    ROOT.mkdir(parents=True, exist_ok=True)
    text_path = ROOT / 'BigEarthNet.txt.parquet'
    download(TEXT_URL, text_path, 650000000)
    patches = sample_images()
    file = pq.ParquetFile(text_path)
    counts, per_type, per_category = collections.Counter(), collections.Counter(), collections.Counter()
    unique = collections.defaultdict(set)
    lookup = collections.defaultdict(lambda: collections.Counter())
    task_examples, image_text = {}, []
    rows = 0
    for batch in file.iter_batches(batch_size=32768):
        for row in batch.to_pylist():
            rows += 1
            split, patch, kind = row['split'], row['patch_id'], row['type']
            category = str(row['category'])
            counts[split] += 1; per_type[kind] += 1; per_category[category] += 1
            unique[split].add(patch)
            if patch in patches: image_text.append(row)
            if split == 'train' and (kind,category) not in task_examples:
                task_examples[(kind,category)] = row
            field = {'country':'country','season':'season','climate zone':'climate_zone'}.get(category)
            if kind == 'mcq' and field:
                c = lookup[category]; c['n'] += 1
                options = re.findall(r'\b([a-d])\)\s*(.*?)(?=\s*,?\s*\b[a-d]\)\s*|$)', row['input'])
                norm = lambda text: re.sub(r'\s+', ' ', str(text).strip(' .,:;').lower())
                match = [letter for letter, value in options if norm(value) == norm(row[field])]
                if len(match)==1:
                    c['parsed'] += 1
                    c['correct'] += int(match[0] == norm(row['output']))
        if rows % 327680 == 0: print('text rows', rows, flush=True)
    splits = sorted(unique)
    summary = {'source': TEXT_URL, 'file_bytes': text_path.stat().st_size,
        'schema': str(file.schema_arrow), 'rows':rows,
        'annotations_by_split': dict(counts), 'unique_patches_by_split': {s:len(v) for s,v in unique.items()},
        'unique_patches_total': len(set().union(*unique.values())), 'types':dict(per_type), 'categories':dict(per_category),
        'split_patch_overlap': {a+'__'+b:len(unique[a]&unique[b]) for i,a in enumerate(splits) for b in splits[i+1:]},
        'metadata_mcq_lookup': dict(lookup), 'sampled_complete_images':len(patches),
        'sampled_images_with_exact_text_join':len({r['patch_id'] for r in image_text}),
        'joined_text_rows':len(image_text),
        'limitations':['archive-prefix image sample; not a test set','lookup baseline is not visual understanding','no model training']}
    save('audit_summary.json',summary)
    save('task_examples.json',list(task_examples.values()))
    save('sample_image_text.json',image_text)
    print(json.dumps(summary,indent=2),flush=True)

def gpu():
    from datetime import datetime, timezone
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    import torch
    from PIL import Image, ImageDraw
    from olmoearth_pretrain_minimal import ModelID
    from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1': raise RuntimeError('GPU1 only')
    joined = json.loads((ROOT/'sample_image_text.json').read_text())
    ids = sorted({r['patch_id'] for r in joined})[:8]
    if not ids: raise RuntimeError('No exact image-text joins; no GPU smoke')
    start = time.perf_counter()
    wrapper = OlmoEarth(patch_size=4, model_id=ModelID.OLMOEARTH_V1_BASE,
        token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype='bfloat16').cuda().eval()
    rows=[]; thumbnails=[]
    for patch in ids:
        planes=[]; band_shapes={}; band_bounds={}
        for band in BANDS:
            with rasterio.open(ROOT/'images'/patch/f'{patch}_{band}.tif') as ds:
                band_shapes[band]=list(ds.shape); band_bounds[band]=list(ds.bounds)
                planes.append(ds.read(1,out_shape=(120,120),resampling=Resampling.bilinear).astype('float32'))
                crs = str(ds.crs)
        if any(not np.allclose(band_bounds[BANDS[0]],v) for v in band_bounds.values()):
            raise RuntimeError('Mismatched band bounds')
        cube=np.stack(planes)[:,None]
        rgb=np.stack([planes[BANDS.index(b)] for b in ('B04','B03','B02')],axis=-1)
        lo,hi=np.percentile(rgb,[2,98])
        rgb=np.clip((rgb-lo)/max(hi-lo,1),0,1)
        thumbnails.append((patch,Image.fromarray((rgb*255).astype('uint8')).resize((360,360),Image.Resampling.NEAREST)))
        timestamp=datetime.strptime(patch.split('_')[2],'%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
        context={'sentinel2_l2a':RasterImage(image=torch.from_numpy(cube).cuda(),timestamps=[(timestamp,timestamp)])}
        wrapper.normalizer(context,{})
        sample,present,_=wrapper._prepare_modality_inputs(ModelContext(inputs=[context],metadatas=[]))
        # All 12 bands exist; do NOT apply the missing B01/B09 mask from Sen12 scripts.
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.bfloat16):
            tm=wrapper.model(sample,fast_pass=False,patch_size=4)['tokens_and_masks']
            mask=(tm.sentinel2_l2a_mask != MaskValue.MISSING.value).unsqueeze(-1)
            feats=(tm.sentinel2_l2a*mask).sum(dim=(3,4))/mask.sum(dim=(3,4)).clamp(min=1)
        f=feats[0].float().cpu().numpy()
        (ROOT/'features').mkdir(exist_ok=True)
        np.save(ROOT/'features'/f'{patch}.npy',f.astype('float16'))
        row={'patch_id':patch,'bands':list(BANDS),'original_shapes':band_shapes,'timestamp':timestamp.isoformat(),
             'crs':crs,'bounds':band_bounds[BANDS[0]],'feature_shape':list(f.shape),'finite':bool(np.isfinite(f).all()),
             'feature_std':float(f.std()),'text_rows':sum(r['patch_id']==patch for r in joined)}
        rows.append(row); print(json.dumps(row),flush=True)
    torch.cuda.synchronize()
    summary={'device':torch.cuda.get_device_name(),'model_id':str(ModelID.OLMOEARTH_V1_BASE),
        'seconds_including_load':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'n':len(rows),'all_finite':all(r['finite'] for r in rows),'rows':rows,
        'claim':'native raw-band plus exact text-ID execution contract only; not language alignment or accuracy'}
    save('gpu_smoke.json',summary)
    if thumbnails:
        sheet=Image.new('RGB',(4*360,2*400),'white'); draw=ImageDraw.Draw(sheet)
        for i,(patch,thumb) in enumerate(thumbnails):
            x,y=(i%4)*360,(i//4)*400
            sheet.paste(thumb,(x,y)); draw.text((x+3,y+362),patch.split('_')[2]+' / '+'_'.join(patch.split('_')[-2:]),fill='black')
        sheet.save(ROOT/'rgb_contact_sheet.png')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=['inspect','audit','gpu'],default='inspect'); args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    if args.mode=='inspect':
        print({m:bool(importlib.util.find_spec(m)) for m in ['torch','pyarrow','rasterio','zstandard','rslearn','olmoearth_pretrain_minimal','transformers']})
    elif args.mode=='audit': audit()
    else: gpu()
