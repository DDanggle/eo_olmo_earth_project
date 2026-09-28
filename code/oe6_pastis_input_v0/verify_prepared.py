"""Independent array and source checks; no model accuracy evaluation."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--prepared',type=Path,required=True)
ap.add_argument('--computed',type=Path,required=True)
ap.add_argument('--candidates',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args()
c=json.loads((a.prepared/'contract.json').read_text())
assert hashlib.sha256(a.computed.read_bytes()).hexdigest()==c['normalization_config_sha256']
stats=json.loads(a.computed.read_text())['sentinel2_l2a']
bands=c['native_bands']
mu=np.array([stats[b]['mean'] for b in bands])
sd=np.array([stats[b]['std'] for b in bands])
checks=[]
for r in c['records']:
    z=np.load(a.prepared/r['path'],allow_pickle=False)
    expected_dates=[r['all_dates_yyyymmdd'][i] for i in r['selected_indices']]
    assert r['selected_dates']==expected_dates
    parsed=[datetime.strptime(str(d),'%Y%m%d') for d in expected_dates]
    assert np.array_equal(z['timestamps'],[[d.day,d.month-1,d.year] for d in parsed])
    mapping=[0,1,2,6,3,4,5,7,8,9,0,7]
    expected=np.transpose(z['raw_s2'][r['selected_indices']][:,mapping],(2,3,0,1)).astype(np.float32)
    assert np.array_equal(z['raw_native_selected'],expected)
    norm=((expected-(mu-2*sd))/(4*sd)).astype(np.float32)
    assert np.array_equal(norm,z['normalized_s2'])
    assert np.array_equal(z['band_observed'],[True]*10+[False]*2)
    assert np.array_equal(z['label_valid'],z['semantic']!=19)
    assert np.array_equal(z['crop_label_valid'],(z['semantic']>0)&(z['semantic']!=19))
    assert np.array_equal(z['target_mask'],z['semantic']==r['target_class_id'])
    assert int(z['target_mask'].sum())==r['target_pixel_count']
    assert z['instances'].shape==z['semantic'].shape==(128,128)
    checks.append({'patch_id':r['patch_id'],'passed':True,'normalization_max_delta':0.0,
        'selected_dates':expected_dates,'target_pixels':r['target_pixel_count'],
        'selected_all_bands_minus10000':int((expected==-10000).all(-1).sum()),
        'raw_min':float(expected.min()),'raw_max':float(expected.max()),
        'cloud_quality_certified':False,'native_file_sha256':hashlib.sha256((a.prepared/r['path']).read_bytes()).hexdigest()})
rows=[json.loads(l) for l in (a.candidates/'candidates.jsonl').read_text().splitlines()]
assert len(rows)==80 and len({r['patch_id'] for r in rows})==80
assert sum(r['role']=='train' for r in rows)==64
assert sum(r['role']=='development' for r in rows)==16
assert {r['parent_tile'] for r in rows if r['role']=='train'}.isdisjoint({r['parent_tile'] for r in rows if r['role']=='development'})
assert not any(r['parent_tile']=='t30uxv' for r in rows)
out={'status':'passed_input_arithmetic_and_metadata_checks_not_model_performance','cases':checks,
    'candidate_split_disjoint':True,'candidate_counts':{'train':64,'development':16},
    'actual_prepared_native_cases':2,'candidate_raw_prepared':False,
    'test_opened_this_preparation':False,'upstream_date_mismatch_fixture':c['date_tests'],
    'limits':['source calibration and geometry not independently reacquired from original satellite products',
        'annual label does not prove two selected dates identify the class','no cloud ground truth',
        'B01 and B09 imputed and explicitly marked','no VLM accuracy or checkpoint parity run']}
a.out.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['limits','upstream_date_mismatch_fixture']}))
