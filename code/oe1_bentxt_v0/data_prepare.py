#!/usr/bin/env python3
"""Prepare a small, local-only BigEarthNet.txt / GEO-Bench-2 S2 development set.

No download, model inference, or test/bench materialization. Requires numpy,
pyarrow, rasterio and Pillow. Reads only selected S2 members from a local
tortilla. Official text train/validation and GEO-Bench splits must agree.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

SOURCE_BANDS = ('B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B11','B12')
OLMO_BANDS = ('B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09')
CLASSES = ('Urban fabric','Industrial or commercial units','Arable land','Permanent crops',
    'Pastures','Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas','Broad-leaved forest','Coniferous forest','Mixed forest',
    'Natural grassland and sparsely vegetated areas','Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland, shrub','Beaches, dunes, sands','Inland wetlands','Coastal wetlands',
    'Inland waters','Marine waters')
GEO_SHA256 = '821c2f429c3e85c158c758bbb215bf61170a2451a11284efaf0f89cef97e468a'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def rank(value, seed):
    return digest(f'{seed}:{value}'.encode())


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def read_index(fp, offset=0, container_length=None):
    import pyarrow.parquet as pq
    fp.seek(offset)
    header = fp.read(18)
    if len(header) != 18 or header[:2] not in (b'#y', b'WX'):
        raise ValueError(f'Invalid tortilla header at {offset}')
    footer_offset = int.from_bytes(header[2:10], 'little')
    footer_length = int.from_bytes(header[10:18], 'little')
    if footer_length > 50_000_000 or footer_offset < 18:
        raise ValueError('Unexpected tortilla metadata dimensions')
    if container_length is not None and footer_offset + footer_length > container_length:
        raise ValueError('Nested footer extends beyond parent member')
    fp.seek(offset + footer_offset)
    body = fp.read(footer_length)
    if len(body) != footer_length:
        raise ValueError('Truncated tortilla index')
    return pq.read_table(io.BytesIO(body)).to_pylist(), digest(body)


def parse_patch(patch_id):
    m = re.fullmatch(r'S2[AB]_MSIL2A_(\d{8}T\d{6})_N\d+_R\d+_(T\d{2}[A-Z]{3})_\d+_\d+', patch_id)
    if not m:
        raise ValueError(f'Unexpected v2 patch ID: {patch_id}')
    date = datetime.strptime(m[1], '%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    return date.isoformat(), m[2]


def parse_query_class(question):
    # Conservatively use exact canonical substrings; no invented synonym labels.
    q = question.casefold()
    hits = [name for name in CLASSES if name.casefold() in q]
    return hits[0] if len(hits) == 1 else None


def read_presence_rows(text_path, index_by_id):
    import pyarrow.dataset as ds
    expression = ((ds.field('split').isin(['train','validation'])) &
                  (ds.field('type') == 'binary') & (ds.field('category') == 'presence'))
    scanner = ds.dataset(text_path, format='parquet').scanner(
        columns=['ID','patch_id','input','output','type','category','split'],
        filter=expression, batch_size=32768)
    joined = defaultdict(list)
    mismatches = []
    ids = set()
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            patch = row['patch_id']
            if patch not in index_by_id:
                continue
            if row['split'] != index_by_id[patch]['tortilla:data_split']:
                mismatches.append((patch,row['split'],index_by_id[patch]['tortilla:data_split']))
                continue
            if str(row['output']).strip().lower() not in ('yes','no'):
                raise ValueError('Nonbinary answer in binary presence task')
            if row['ID'] in ids:
                raise ValueError('Duplicate original question ID among candidate rows')
            ids.add(row['ID'])
            joined[patch].append(row)
    if mismatches:
        raise ValueError(f'GEO-Bench and official text splits disagree: {len(mismatches)} rows; first={mismatches[0]}')
    # Two original questions per patch, one for each answer: identical exposure.
    eligible = {p: rows for p,rows in joined.items()
                if {str(r['output']).strip().lower() for r in rows} == {'yes','no'}}
    return eligible, {'joined_presence_patches':len(joined),
                      'eligible_yes_and_no_patches':len(eligible), 'split_mismatch_rows':0}


def round_robin_pick(rows, count, seed):
    # Rotate through countries and MGRS tiles before taking a second patch/group.
    groups = defaultdict(list)
    for row in rows:
        groups[(row['country'], row['mgrs'])].append(row)
    ordered = sorted(groups, key=lambda k: rank(':'.join(k), seed))
    pools = [deque(sorted(groups[k], key=lambda r: rank(r['patch_id'], seed))) for k in ordered]
    result = []
    while len(result) < count:
        before = len(result)
        for pool in pools:
            if pool:
                result.append(pool.popleft())
                if len(result) == count:
                    break
        if len(result) == before:
            raise ValueError(f'Insufficient eligible patches: need {count}, have {len(result)}')
    return result


def minimum_km(rows, reference):
    import numpy as np
    a = np.radians([[r['lat'],r['lon']] for r in rows])
    b = np.radians([[r['lat'],r['lon']] for r in reference])
    lat_delta = a[:,None,0] - b[None,:,0]
    lon_delta = a[:,None,1] - b[None,:,1]
    h = np.sin(lat_delta/2)**2 + np.cos(a[:,None,0])*np.cos(b[None,:,0])*np.sin(lon_delta/2)**2
    return (2*6371.0088*np.arcsin(np.sqrt(np.clip(h,0,1)))).min(axis=1)


def select_patches(index_by_id, eligible, n_train, n_dev, seed, separation_km):
    rows = [index_by_id[p] for p in eligible]
    available_dev = [r for r in rows if r['tortilla:data_split']=='validation']
    available_train = [r for r in rows if r['tortilla:data_split']=='train']
    all_tiles = sorted({r['mgrs'] for r in available_dev}, key=lambda m: rank(m,seed))
    # Approximately one fifth of validation tiles; extend only for sample counts.
    n_tiles = max(2, (len(all_tiles)+4)//5)
    while n_tiles <= len(all_tiles):
        dev_tiles = set(all_tiles[:n_tiles])
        dev_pool = [r for r in available_dev if r['mgrs'] in dev_tiles]
        train_pool = [r for r in available_train if r['mgrs'] not in dev_tiles]
        if len(dev_pool) >= n_dev and len(train_pool) >= n_train:
            dev = round_robin_pick(dev_pool,n_dev,seed)
            distances = minimum_km(train_pool,dev)
            train_pool = [r for r,d in zip(train_pool,distances) if d >= separation_km]
            if len(train_pool) >= n_train:
                train = round_robin_pick(train_pool,n_train,seed)
                return train,dev,float(minimum_km(train,dev).min())
        n_tiles += 1
    raise ValueError('Cannot obtain disjoint MGRS development groups at requested sizes')


def extract_s2(fp, outer, output, artifact_name):
    import numpy as np
    import rasterio
    from rasterio.io import MemoryFile
    offset = int(outer['tortilla:offset'])
    nested,_ = read_index(fp,offset,int(outer['tortilla:length']))
    s2_rows = [r for r in nested if r['tortilla:id']=='s2' and r['tortilla:file_format']=='GTiff']
    if len(s2_rows) != 1:
        raise ValueError('S2 member missing or duplicated')
    inner = s2_rows[0]
    for key in ('patch_id','tortilla:data_split'):
        if inner[key] != outer[key]:
            raise ValueError(f'Inner/outer mismatch: {key}')
    start = int(inner['tortilla:offset'])
    length = int(inner['tortilla:length'])
    if start < 18 or start + length > int(outer['tortilla:length']):
        raise ValueError('S2 member extends beyond parent')
    fp.seek(offset+start)
    raw = fp.read(length)
    if len(raw) != length:
        raise ValueError('Truncated TIFF')
    with MemoryFile(raw) as memory:
        with memory.open() as src:
            cube = src.read()
            transform = list(src.transform.to_gdal())
            crs = str(src.crs)
            descriptions = list(src.descriptions)
    if cube.shape != (12,120,120):
        raise ValueError(f'Unexpected S2 cube shape: {cube.shape}')
    if not np.isfinite(cube).all():
        raise ValueError('Nonfinite S2 values')
    outer_transform = outer['stac:geotransform']
    if not np.allclose(outer_transform[1:3],[10,0]) or not np.allclose(outer_transform[4:6],[0,-10]):
        raise ValueError('Unexpected outer 10m grid; review before proceeding')
    if crs != outer['stac:crs']:
        raise ValueError('Nested and outer CRS disagree')
    if any(x is not None for x in descriptions) and tuple(descriptions) != SOURCE_BANDS:
        raise ValueError(f'Band descriptions disagree with official loader: {descriptions}')
    # GEO-Bench bundles bands resampled to 120²; retain DN values, only reorder.
    cube = np.ascontiguousarray(cube[[SOURCE_BANDS.index(b) for b in OLMO_BANDS]])
    if float(cube.max()) < 2:
        raise ValueError('Unexpected reflectance scale, expected raw scaled S2 DN')
    dest = output / artifact_name
    dest.parent.mkdir(parents=True,exist_ok=True)
    np.save(dest,cube,allow_pickle=False)
    return cube, {
        'image_path':artifact_name,'image_array_sha256':digest(cube.tobytes()),
        'source_tiff_sha256':digest(raw),'shape':list(cube.shape),'dtype':str(cube.dtype),
        'minimum':float(cube.min()),'maximum':float(cube.max()),
        'source_tiff_geotransform':transform,
        'source_tiff_transform_matches_outer':bool(np.allclose(transform,outer_transform)),
        'source_band_descriptions':descriptions,
        'bands':list(OLMO_BANDS),'radiometry':'raw scaled S2 L2A DN, unnormalized',
        'resampling':'upstream GEO-Bench-2 resampling to 120x120; method not independently audited',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tortilla',type=Path,required=True)
    ap.add_argument('--text-parquet',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--deps-dir',type=Path,help='Existing isolated optional dependency directory')
    ap.add_argument('--train-patches',type=int,default=512)
    ap.add_argument('--dev-patches',type=int,default=128)
    ap.add_argument('--seed',type=int,default=20260926)
    ap.add_argument('--separation-km',type=float,default=2.0)
    ap.add_argument('--metadata-only',action='store_true')
    args=ap.parse_args()
    if args.deps_dir:
        if not args.deps_dir.is_dir():
            ap.error('--deps-dir must be an existing directory')
        sys.path.insert(0,str(args.deps_dir))
    if not 1 <= args.train_patches <= 2048 or not 1 <= args.dev_patches <= 256:
        ap.error('Pilot bounded to 1..2048 train and 1..256 dev patches')
    if args.separation_km < 2:
        ap.error('Cross-split center separation must be at least 2km')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.json').exists():
        raise RuntimeError('Output already contains a completed manifest; choose a fresh output')
    with args.tortilla.open('rb') as fp:
        index,index_hash=read_index(fp,container_length=args.tortilla.stat().st_size)
        index_by_id={}
        for row in index:
            if row['tortilla:data_split'] not in ('train','validation'):
                continue
            if row['contains_cloud_or_shadow'] or row['contains_seasonal_snow']:
                continue
            p=row['patch_id']
            if p in index_by_id: raise ValueError('Duplicate patch in GEO-Bench index')
            row['timestamp_utc'],row['mgrs']=parse_patch(p)
            index_day=datetime.fromtimestamp(row['stac:time_start'],timezone.utc).date()
            if index_day != datetime.fromisoformat(row['timestamp_utc']).date():
                raise ValueError('Patch ID and metadata acquisition day disagree')
            index_by_id[p]=row
        eligible,join_audit=read_presence_rows(args.text_parquet,index_by_id)
        train,dev,min_km=select_patches(index_by_id,eligible,args.train_patches,args.dev_patches,args.seed,args.separation_km)
        assert not ({r['mgrs'] for r in train}&{r['mgrs'] for r in dev})
        manifest={'status':'metadata_only' if args.metadata_only else 'complete',
            'purpose':'OE1 development feasibility pilot, not final heldout evaluation',
            'created_utc':datetime.now(timezone.utc).isoformat(),'seed':args.seed,
            'sources':{'tortilla':str(args.tortilla),'tortilla_size':args.tortilla.stat().st_size,
                'tortilla_index_sha256':index_hash,'official_tortilla_sha256_expected_not_recomputed':GEO_SHA256,
                'text_parquet':str(args.text_parquet),'text_parquet_size':args.text_parquet.stat().st_size,
                'geobench_repo':'https://huggingface.co/datasets/aialliance/benv2',
                'geobench_revision':'20ac91314d2959d585faef744e07a2beb219759d',
                'text_repo':'https://huggingface.co/datasets/BIFOLD-BigEarthNetv2-0/BigEarthNet.txt',
                'note':'Local files used; full content hashes must be matched to prior download receipts or separately verified.'},
            'join_audit':join_audit,'bands':list(OLMO_BANDS),
            'questions_per_patch':2,'original_question_wording_preserved':True,
            'text_test_or_bench_rows_materialized':False,'cross_split_minimum_center_km':min_km,
            'spatial_split':'disjoint MGRS groups plus >=2km center separation',
            'limitations':['development only; unseen public-pretraining exposure not excluded',
                'class-presence questions do not test change, numeric grounding, or uncertainty',
                'CLC2018-derived labels are not fresh expert judgments of every observation',
                'GEO-Bench S2 TIFF transforms may retain 60m scaling after 120x120 resampling; use outer10m footprint',
                'native spectral bands retained, native per-band resolution already resampled upstream'],
            'splits':{},'patches':[]}
        hashes={}; previews=[]
        for split,chosen in [('train',train),('dev',dev)]:
            qa=[]
            for outer in chosen:
                p=outer['patch_id']; metadata={
                    'patch_id':p,'split':split,'official_split':outer['tortilla:data_split'],
                    'geobench_split':outer['tortilla:data_split'],'timestamp_utc':outer['timestamp_utc'],
                    'timestamp_source':'v2 patch ID sensing timestamp, day crosschecked with outer STAC',
                    'mgrs':outer['mgrs'],'country':outer['country'],'latitude':outer['lat'],'longitude':outer['lon'],
                    'labels':outer['labels'],'crs':outer['stac:crs'],
                    'geotransform':outer['stac:geotransform'],'footprint_source':'outer GEO-Bench index 120x120, 10m grid'}
                if not args.metadata_only:
                    cube,img=extract_s2(fp,outer,args.output,f'images/{p}.npy')
                    metadata.update(img)
                    h=img['image_array_sha256']
                    if h in hashes: raise ValueError(f'Duplicate pixel arrays: {p}, {hashes[h]}')
                    hashes[h]=p
                    if split=='dev' and len(previews)<4: previews.append((p,cube))
                manifest['patches'].append(metadata)
                for answer in ('yes','no'):
                    candidates=[r for r in eligible[p] if str(r['output']).strip().lower()==answer]
                    original=min(candidates,key=lambda r:rank(str(r['ID']),args.seed))
                    qa.append({'id':str(original['ID']),'question_id':str(original['ID']),'patch_id':p,
                        'image_path':f'images/{p}.npy','input':original['input'],'output':answer,
                        'question':original['input'],'answer':answer,'date':outer['timestamp_utc'],
                        'query_class':parse_query_class(original['input']),
                        'split':split,'official_split':original['split'],
                        'timestamp_utc':outer['timestamp_utc'],'mgrs':outer['mgrs'],
                        'bands':list(OLMO_BANDS),'task':'binary_presence'})
            with (args.output/f'{split}.jsonl').open('w') as target:
                for row in qa: target.write(json.dumps(row,ensure_ascii=False)+'\n')
            manifest['splits'][split]={'patches':len(chosen),'questions':len(qa),
                'answers':dict(Counter(q['answer'] for q in qa)),
                'query_classes':dict(Counter(q['query_class'] or 'unparsed' for q in qa)),
                'countries':dict(Counter(r['country'] for r in chosen)),
                'mgrs':dict(Counter(r['mgrs'] for r in chosen)),
                'question_file_sha256':digest((args.output/f'{split}.jsonl').read_bytes())}
        if previews:
            import numpy as np
            from PIL import Image,ImageDraw
            sheet=Image.new('RGB',(960,270),'white'); draw=ImageDraw.Draw(sheet)
            for i,(p,cube) in enumerate(previews):
                rgb=np.moveaxis(cube[[OLMO_BANDS.index(b) for b in ('B04','B03','B02')]].astype('float32'),0,-1)
                lo,hi=np.percentile(rgb,[2,98]); rgb=np.clip((rgb-lo)/max(hi-lo,1),0,1)
                im=Image.fromarray((rgb*255).astype('uint8')).resize((240,240))
                sheet.paste(im,(i*240,0)); draw.text((i*240+3,244),p.split('_')[5]+' '+p.split('_')[2],fill='black')
            sheet.save(args.output/'dev_preview.png')
        manifest['image_array_hashes_unique']=not args.metadata_only
        manifest['tiff_transform_mismatches']=sum(p.get('source_tiff_transform_matches_outer') is False for p in manifest['patches'])
        write_json(args.output/'manifest.json',manifest)
    print(json.dumps({'status':manifest['status'],'output':str(args.output),
                      'splits':manifest['splits'],'join_audit':join_audit,
                      'min_center_km':min_km,'tiff_transform_mismatches':manifest['tiff_transform_mismatches']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
