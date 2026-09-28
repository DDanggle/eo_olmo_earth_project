#!/usr/bin/env python3
"""Independent packet verification plus small CPU review export."""
import argparse,hashlib,json,math,shutil
from pathlib import Path
from datetime import datetime
from collections import Counter
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dist(a,b):
    lon1,lat1=np.radians(a);lon2,lat2=np.radians(b)
    h=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return float(2*6371008.8*np.arcsin(np.sqrt(np.clip(h,0,1))))
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--code',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.root/'receipt.json').read_text());assert r['status']=='inputs_prepared_requires_independent_verification'
    code_manifest=json.loads((a.code/'source_manifest.json').read_text())
    for path,h in code_manifest.items():assert sha(a.code/path)==h,path
    policy=json.loads((a.code/'selection_policy.json').read_text());assert sha(a.code/'selection_policy.json')==r['policy_sha256']
    assert sha(a.code/'prepare_regions.py')==r['script_sha256'] and sha(a.code/'oe8_transform.py')==r['transform_sha256']
    rows=[json.loads(x) for x in (a.root/'manifest.jsonl').read_text().splitlines()]
    candidates=[json.loads(x) for x in (a.root/'selected_candidates.jsonl').read_text().splitlines()]
    assert sha(a.root/'selected_candidates.jsonl')==r['candidate_sha256']
    assert [x['patch_id'] for x in rows]==[x['patch_id'] for x in candidates]
    legacy={json.loads(x)['patch_id'] for x in (a.code/'legacy80.jsonl').read_text().splitlines()}
    assert len(rows)==len({x['patch_id'] for x in rows})==r['prepared']
    assert all(x['parent_tile'] in policy['source_parents'] and x['patch_id'] not in legacy for x in rows)
    bands=['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09'];mapping=[0,1,2,6,3,4,5,7,8,9,0,7]
    stats=json.loads((a.code/'computed.json').read_text())['sentinel2_l2a']
    mu=np.array([stats[b]['mean'] for b in bands]);sd=np.array([stats[b]['std'] for b in bands])
    for i,x in enumerate(rows):
        ip=a.root/x['npz_path'];lp=a.root/x['label_path'];assert sha(ip)==x['npz_sha256'] and sha(lp)==x['label_sha256']
        with np.load(ip,allow_pickle=False) as z:
            assert not {'semantic','instances','target_mask'}&set(z.files)
            raw=z['raw_selected_s2'];valid=z['observation_valid'];norm=z['normalized_s2']
            assert raw.shape==(8,10,128,128) and norm.shape==(128,128,8,12)
            assert raw.dtype==np.int16 and norm.dtype==np.float32 and valid.dtype==np.bool_
            assert np.array_equal(valid,~(raw==-10000).any(1))
            expected=((raw[:,mapping].transpose(2,3,0,1).astype(np.float64)-(mu-2*sd))/(4*sd)).astype(np.float32)
            expected[~valid.transpose(1,2,0)]=0
            assert np.array_equal(norm,expected)
            parsed=[datetime.strptime(str(d),'%Y%m%d') for d in x['selected_dates']]
            assert np.array_equal(z['timestamps'],[[d.day,d.month-1,d.year] for d in parsed])
            assert np.array_equal(z['band_observed'],[True]*10+[False]*2)
            assert bool(valid.all())==x['strict_no_missing_input_eligible']
        with np.load(lp,allow_pickle=False) as z:
            sem=z['semantic'];ins=z['instances']
            assert sem.shape==ins.shape==(128,128) and sem.dtype==ins.dtype==np.int64
            assert (ins>=0).all() and set(np.unique(sem))<=set(range(20))
            assert np.array_equal(z['label_valid'],sem!=19) and np.array_equal(z['crop_label_valid'],(sem>0)&(sem<19))
            assert np.bincount(sem.ravel(),minlength=20).tolist()==x['class_pixel_counts']
        if i%60==0:print(json.dumps({'verified_packets':i+1,'total':len(rows)}),flush=True)
    minimum=min(dist(x['centroid_lon_lat'],y['centroid_lon_lat']) for i,x in enumerate(rows) for y in rows[i+1:] if x['training_partition']!=y['training_partition'])
    assert minimum>=policy['minimum_cross_partition_centroid_m']
    a.out.mkdir(parents=True,exist_ok=False)
    for name in ['receipt.json','availability_summary.json','before_label_split.json','selection_policy.json','manifest.jsonl','selected_candidates.jsonl','availability.jsonl','source_metadata.jsonl']:
        shutil.copyfile(a.root/name,a.out/name)
    shutil.copytree(a.root/'qa',a.out/'qa')
    (a.out/'sample_packets'/'inputs').mkdir(parents=True);(a.out/'sample_packets'/'labels').mkdir()
    for parent in policy['source_parents']:
        x=next(x for x in rows if x['parent_tile']==parent and x['training_partition']=='train_pool')
        for field in ['npz_path','label_path']:shutil.copyfile(a.root/x[field],a.out/'sample_packets'/x[field])
    verification={'status':'PASS','packets_verified':len(rows),'all_input_and_label_hashes_verified':True,'normalization_exact_recomputed':True,'spatial_guard_minimum_m':minimum,'legacy_overlap':0,'reserved_parent_packets':0,'partitions':dict(Counter(x['training_partition'] for x in rows)),'cloud_certified':False,'exact_parcel_independence_verified':False,'verifier_sha256':sha(__file__),'manifest_sha256':sha(a.root/'manifest.jsonl')}
    (a.out/'independent_packet_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    files=[x for x in a.out.rglob('*') if x.is_file()]
    manifest={str(x.relative_to(a.out)):{'sha256':sha(x),'bytes':x.stat().st_size} for x in files}
    (a.out/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(verification),flush=True)
if __name__=='__main__':main()
