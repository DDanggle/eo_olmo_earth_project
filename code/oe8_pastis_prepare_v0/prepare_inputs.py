#!/usr/bin/env python3
"""Bounded CPU extraction of frozen PASTIS candidates. No model execution."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
import traceback
import numpy as np

INPUT = ['B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12']
NATIVE = ['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09']
MAPPING = [0,1,2,6,3,4,5,7,8,9,0,7]
CLASSES = ['background','meadow','soft_winter_wheat','corn','winter_barley','winter_rapeseed','spring_barley','sunflower','grapevine','beet','winter_triticale','winter_durum_wheat','fruits_vegetables_flowers','potatoes','leguminous_fodder','soybeans','orchard','mixed_cereal','sorghum','void_label']
SHARD = 'geobench_pastis.0000.part.tortilla'
SHARD_SHA = '56b1490c6dc7345fdff79e94d9132753ee28d8504bb061d8db39d19e888f7ca3'
COMPUTED_SHA = '774e461babd2bd4cbe3591b46457d39054f8ac529cb57a052547501076ebaef7'
CANDIDATE_SHA = 'd0bfc58d11cbac0c6b270c5739a3f6b5c34f1f5de2510ab9635fd1bab7d1aef0'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n')

def frame_indices(n):
    if n<8:raise ValueError('Eight distinct candidate dates required; no padding')
    return [i*(n-1)//7 for i in range(8)]

def partition(rows):
    bank=set()
    for parent in ['t32ulu','t31tfj']:
        r=sorted([r for r in rows if r['parent_tile']==parent],key=lambda r:hashlib.sha256(('oe8-bank-v0:'+r['patch_id']).encode()).hexdigest())
        assert len(r)==32
        bank.update(x['patch_id'] for x in r[:8])
    return {r['patch_id']:('dev_query' if r['role']=='development' else 'source_bank' if r['patch_id'] in bank else 'train_pool') for r in rows}

def read_payload(reference, allowed):
    import h5py
    match=re.fullmatch(r'/vsisubfile/(\d+)_(\d+),(.+)',str(reference))
    if not match:raise ValueError('Invalid packed reference')
    offset,size,path=int(match[1]),int(match[2]),Path(match[3]).resolve()
    if path not in allowed or size<=0 or size>(128<<20) or offset+size>path.stat().st_size:
        raise ValueError('Unverified or out-of-bounds source payload')
    with path.open('rb') as f:
        f.seek(offset);payload=f.read(size)
    if len(payload)!=size:raise ValueError('Short payload')
    with h5py.File(io.BytesIO(payload),'r') as f:
        arr=np.asarray(f['data'])
    return arr,{'reference':str(reference),'payload_sha256':hashlib.sha256(payload).hexdigest(),'bytes':size,'offset':offset,'shape':list(arr.shape),'dtype':str(arr.dtype)}

def transform(raw, dates, semantic_raw, instance_raw, stats):
    if raw.dtype!=np.int16 or raw.ndim!=4 or raw.shape[1:]!=(10,128,128):raise ValueError('Unexpected S2 shape/dtype')
    if not np.isfinite(raw).all():raise ValueError('Nonfinite raw input')
    if len(dates)!=len(raw):raise ValueError('Frame/date count mismatch')
    parsed=[datetime.strptime(str(int(d)),'%Y%m%d') for d in dates]
    if parsed!=sorted(parsed) or len(set(parsed))!=len(parsed):raise ValueError('Non-monotonic/duplicate dates')
    if semantic_raw.shape!=(3,128,128):raise ValueError('Unexpected original semantic stack')
    semantic=semantic_raw[0].astype(np.int64)
    if not np.array_equal(semantic,semantic_raw[0]) or not set(np.unique(semantic))<=set(range(20)):raise ValueError('Invalid semantic class')
    instances=instance_raw.astype(np.int64)
    if instances.shape!=(128,128) or not np.array_equal(instances,instance_raw) or (instances<0).any():raise ValueError('Invalid instance grid')
    idx=frame_indices(len(raw))
    selected=raw[idx]
    nodata=(selected==-10000)
    observed_valid=~nodata.any(axis=1)
    native=np.transpose(selected[:,MAPPING].astype(np.float32),(2,3,0,1))
    mu=np.array([stats[b]['mean'] for b in NATIVE])
    sd=np.array([stats[b]['std'] for b in NATIVE])
    norm=((native-(mu-2*sd))/(4*sd)).astype(np.float32)
    norm[~observed_valid.transpose(1,2,0)]=0.0
    timestamps=np.asarray([[parsed[i].day,parsed[i].month-1,parsed[i].year] for i in idx],dtype=np.int64)
    inputs={'raw_selected_s2':selected,'normalized_s2':norm,'timestamps':timestamps,
            'band_observed':np.asarray([True]*10+[False]*2),'nodata_observed':nodata,'observation_valid':observed_valid}
    labels={'semantic':semantic,'instances':instances,'label_valid':semantic!=19,'crop_label_valid':(semantic>0)&(semantic<19)}
    return idx,inputs,labels

def render_page(rows, output, number):
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(128*9,156*len(rows)), 'white');draw=ImageDraw.Draw(canvas)
    palette=np.random.default_rng(0).integers(40,230,size=(20,3),dtype=np.uint8);palette[0]=0;palette[19]=160
    for j,row in enumerate(rows):
        with np.load(output/row['npz_path']) as z:
            raw=z['raw_selected_s2']
            valid=z['observation_valid']
            for k in range(8):
                rgb=np.round(np.clip(raw[k,[2,1,0]].transpose(1,2,0)/3000,0,1)*255).astype(np.uint8)
                rgb[~valid[k]]=[255,0,255]
                canvas.paste(Image.fromarray(rgb),(k*128,j*156+25))
                draw.text((k*128+2,j*156+12),str(row['selected_dates'][k]),fill='black')
        with np.load(output/row['label_path']) as z:canvas.paste(Image.fromarray(palette[z['semantic']]),(1024,j*156+25))
        draw.text((2,j*156),f"{row['patch_id']} {row['training_partition']} missing={row['quality']['invalid_pixel_dates']}",fill='black')
        draw.text((1026,j*156+12),'Annual crop mask',fill='black')
    canvas.save(output/'qa'/f'page_{number:02d}.png')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--candidates',type=Path,required=True)
    ap.add_argument('--computed',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    for sub in ['inputs','labels','qa']: (a.out/sub).mkdir()
    start=time.monotonic()
    assert sha(a.candidates)==CANDIDATE_SHA and sha(a.computed)==COMPUTED_SHA
    audit=json.loads(a.audit.read_text());assert audit['source_shards_unchanged'] and 'partial_read_error' not in audit
    s=next(s for s in audit['shards'] if s['name']==SHARD)
    assert s['valid'] and s['stable_during_hash'] and s['actual_sha256']==SHARD_SHA
    source=(a.root/SHARD).resolve();before=source.stat()
    assert (before.st_size,before.st_mtime_ns)==(s['bytes'],s['mtime_ns']),'Prior verified source changed; re-audit required'
    candidates=[json.loads(l) for l in a.candidates.read_text().splitlines()]
    assert len(candidates)==80 and len({r['patch_id'] for r in candidates})==80
    assert Counter(r['role'] for r in candidates)=={'train':64,'development':16}
    assert Counter(r['parent_tile'] for r in candidates)=={'t32ulu':32,'t31tfj':32,'t31tfm':16}
    parts=partition(candidates)
    policy={'revision':'oe8_quality_v0','created_utc':datetime.now(timezone.utc).isoformat(),
        'candidate_count':8,'index_formula':'floor(i*(T-1)/7), i=0..7, require T>=8',
        'initial_candidate_positions':[2,5],'fixed_additional_candidate_positions':[0,7],
        'date_selection_uses_labels_predictions_or_visual_clear_selection':False,
        'nodata_rule':'raw==-10000 per band; operational missing-value convention, source semantics pending independent satellite calibration',
        'negative_values_other_than_minus10000':'preserved_not_automatically_missing',
        'normalization':'(native-(mean-2*std))/(4*std), cast float32 once; no clipping',
        'invalid_pixel_rule':'any of 10 measured bands equals sentinel; normalized all12 bands set0 and validity stored',
        'label_valid':'semantic!=19, independent of observation quality',
        'cloud_mask':'unavailable_unknown; no cloud-based rejection',
        'strict_no_missing_input_eligibility':'all observed pixels valid in all8 frames; cloud clearness not certified',
        'missing_input_path':'prepared but requires missing-aware training adapter/declared policy before use',
        'bank_assignment':'sha256(oe8-bank-v0:+patch_id) first8 per train parent before label extraction',
        'source_bank_supervised_encoder_training_allowed':False,'reserved_region_opened':False}
    write_json(a.out/'quality_policy.json',policy)
    shutil.copyfile(a.candidates,a.out/'frozen_candidates.jsonl');shutil.copyfile(a.computed,a.out/'computed.json')
    write_json(a.out/'partition.json',parts)
    import tacoreader
    frame=tacoreader.load([str(source)])
    frame=frame[frame['tortilla:data_split']=='train'].reset_index(drop=True)
    assert len(frame)==911
    stats=json.loads(a.computed.read_text())['sentinel2_l2a']
    records=[];failures=[]
    for n,c in enumerate(candidates):
        try:
            i=c['partial_catalog_index'];outer=frame.iloc[i]
            assert str(outer['patch_id'])==c['patch_id'] and str(outer['tile'])==c['parent_tile']
            assert str(outer['internal:subfile'])==c['source_packed_reference']
            assert [int(v) for v in outer['dates']]==c['all_dates_yyyymmdd']
            assert np.allclose([float(outer['lon']),float(outer['lat'])],c['centroid_lon_lat'],atol=1e-10,rtol=0)
            nested=frame.read(i);names=list(nested['tortilla:id'])
            assert names==['s2','s1a','s1d','semantic','instance']
            for _,meta in nested.iterrows():
                assert str(meta['patch_id'])==c['patch_id'] and str(meta['tile'])==c['parent_tile']
            dates=[int(x) for x in nested.iloc[0]['dates']];assert dates==c['all_dates_yyyymmdd']
            arrays={};provenance={}
            for key in ['s2','semantic','instance']:
                arrays[key],provenance[key]=read_payload(nested.read(names.index(key)),{source})
            idx,inputs,labels=transform(arrays['s2'],dates,arrays['semantic'],arrays['instance'],stats)
            ip=f"inputs/{c['patch_id']}.npz";lp=f"labels/{c['patch_id']}.npz"
            np.savez_compressed(a.out/ip,**inputs);np.savez_compressed(a.out/lp,**labels)
            raw=inputs['raw_selected_s2'];valid=inputs['observation_valid']
            counts=np.bincount(labels['semantic'].ravel(),minlength=20)
            r=dict(c,training_partition=parts[c['patch_id']],npz_path=ip,npz_sha256=sha(a.out/ip),
                label_path=lp,label_sha256=sha(a.out/lp),selected_indices=idx,selected_dates=[dates[j] for j in idx],
                source_payloads=provenance,raw_full_shape=list(arrays['s2'].shape),normalized_shape=list(inputs['normalized_s2'].shape),
                strict_no_missing_input_eligible=bool(valid.all()),
                supervised_training_allowed=parts[c['patch_id']]=='train_pool',
                clean_training_eligible=bool(valid.all()) and parts[c['patch_id']]=='train_pool',
                class_pixel_counts=counts.tolist(),
                exact_footprint_available=False,cloud_quality_certified=False,
                quality={'invalid_pixel_dates':int((~valid).sum()),'per_frame_valid_fraction':valid.mean((1,2)).tolist(),
                    'raw_min':int(raw.min()),'raw_max':int(raw.max()),'negative_non_sentinel_count':int(((raw<0)&(raw!=-10000)).sum()),
                    'all_zero_pixel_dates':int((raw==0).all(1).sum()),'void_pixels':int((labels['semantic']==19).sum())})
            records.append(r)
            with (a.out/'manifest.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            print(json.dumps({'prepared':len(records),'of':80,'patch_id':c['patch_id'],'partition':r['training_partition'],'invalid_pixel_dates':r['quality']['invalid_pixel_dates']}),flush=True)
        except Exception:
            failures.append({'patch_id':c['patch_id'],'error':traceback.format_exc()})
            write_json(a.out/'failures.json',failures)
        write_json(a.out/'progress.json',{'processed':n+1,'prepared':len(records),'failed':len(failures),'wall_seconds':time.monotonic()-start})
    for page,i in enumerate(range(0,len(records),8)):render_page(records[i:i+8],a.out,page)
    after=source.stat();unchanged=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    contract={'status':'prepared_requires_independent_verification' if not failures and unchanged else 'incomplete_or_source_changed',
        'records':len(records),'roles':dict(Counter(r['role'] for r in records)),
        'partitions':dict(Counter(r['training_partition'] for r in records)),
        'input_bands':INPUT,'native_bands':NATIVE,'classes':CLASSES,'imputation':{'B01':'B02','B09':'B8A'},
        'normalization_config_sha256':COMPUTED_SHA,'normalization_std_multiplier':2,
        'candidate_manifest_sha256':CANDIDATE_SHA,'audit_sha256':sha(a.audit),
        'quality_policy_sha256':sha(a.out/'quality_policy.json'),'script_sha256':sha(__file__),
        'source_shards':[s],'source_shards_unchanged':unchanged,'verified_source_stat_continuity':unchanged,
        'source_stat_before':{'bytes':before.st_size,'mtime_ns':before.st_mtime_ns},
        'source_stat_after':{'bytes':after.st_size,'mtime_ns':after.st_mtime_ns},
        'fresh_whole_shard_hash_this_run':False,'source_audit_reused_with_stat_check':True,
        'strict_no_missing_cases':sum(r['strict_no_missing_input_eligible'] for r in records),
        'cases_requiring_missing_aware_adapter':sum(not r['strict_no_missing_input_eligible'] for r in records),
        'cloud_quality_certified':False,'exact_footprint_verified':False,'parcel_disjoint_verified':False,
        'GPU_used':False,'comparative_training_ready':False,'new_dataset_downloaded':False,
        'original_source_modified':False,'failures':failures,'wall_seconds':time.monotonic()-start}
    write_json(a.out/'contract.json',contract)
    print(json.dumps(contract,default=str),flush=True)
    if failures or not unchanged:raise SystemExit(2)

if __name__=='__main__':main()
