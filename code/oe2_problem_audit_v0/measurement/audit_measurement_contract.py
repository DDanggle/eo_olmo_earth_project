#!/usr/bin/env python3
"""Audit the saved OE1 metadata only. Does not infer or modify pixel labels."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json
from measurement_contract import assess_measurement_contract,resampled_affine,pixel_area_coordinate_units


def audit(root):
    p=Path(root)/'artifacts/oe1_bentxt_v0_20260926/data/manifest.json'
    original=p.read_bytes()
    manifest=json.loads(original)
    rows=[]
    for meta in manifest['patches']:
        r=assess_measurement_contract(meta,{'scope':'scene_presence'})
        rows.append({'patch_id':meta['patch_id'],'split':meta['split'],**r})
    assert p.read_bytes()==original
    return {
        'status':'retrospective_metadata_audit_no_new_model_result',
        'source':str(p),'source_sha256':hashlib.sha256(original).hexdigest(),
        'patch_count':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),
        'blocker_counts':dict(Counter(b for r in rows for b in r['blockers'])),
        'area_ratio_counts':dict(Counter(str(r['inner_to_outer_pixel_area_ratio']) for r in rows)),
        'measurement_eligible_count':sum(r['measurement_eligible'] for r in rows),
        'classification_results_invalidated':False,
        'historical_evidence':'The existing OE1 source contract already records the affine discrepancy; this audit quantifies coverage and introduces an explicit future measurement gate.',
        'synthetic_resampling_reproduction':{
            'source_shape':[20,20],'output_shape':[120,120],
            'copied_affine':[0,60,0,0,0,-60],
            'footprint_preserving_affine':resampled_affine([0,60,0,0,0,-60],20,20,120,120),
            'wrong_area_m2_under_declared_metre_crs':120*120*60*60,
            'correct_area_m2_under_declared_metre_crs':120*120*10*10,
            'is_real_image_independent_georeferencing':False,
        },
        'rows':rows,
    }


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    data=audit(a.repo)
    with out.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in data.items() if k!='rows'},ensure_ascii=False))
