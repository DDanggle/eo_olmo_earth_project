#!/usr/bin/env python3
"""Prepare <=2 PASTIS train engineering inputs from SHA-verified shards.

No model run, no geographic transfer claim, no cloud/biophysical ground truth.
The two missing S2 bands use the official PASTIS imputation recipe and are
explicitly marked as imputed; per-band availability is a sidecar, not an encoder mask.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import textwrap


def indices_and_dates(n, dates, count):
    if n != len(dates) or not 1 <= count <= n:
        raise ValueError('Each frame needs its own date; no silent padding')
    parsed = [datetime.strptime(str(int(d)), '%Y%m%d') for d in dates]
    if parsed != sorted(parsed) or len(set(parsed)) != n:
        raise ValueError('Dates must be ordered and unique for this contract')
    indices = [i*n//count for i in range(count)]
    return indices, [[parsed[i].day, parsed[i].month-1, parsed[i].year] for i in indices]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--visual-selection', type=Path)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.source))
    import numpy as np
    import h5py
    import tacoreader
    import torch
    from PIL import Image, ImageDraw
    from torchgeo.datasets import PASTIS
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    audit=json.loads(a.audit.read_text())
    assert audit['status'].startswith('completed_')
    assert audit['source_shards_unchanged'] is True and 'partial_read_error' not in audit
    assert len(audit['inspected_train_records']) == 2
    root=Path('/home/work/data/olmoearth/geobench2/pastis')
    verified=[root/s['name'] for s in audit['shards'] if s['valid'] and s['stable_during_hash']]
    assert verified
    for s in audit['shards']:
        if s['valid']:
            st=(root/s['name']).stat()
            assert (st.st_size,st.st_mtime_ns)==(s['bytes'],s['mtime_ns'])
    df=tacoreader.load([str(p) for p in verified])
    train=df[df['tortilla:data_split']=='train'].reset_index(drop=True)
    raw_dir=a.audit.parent
    records=[]
    classes=list(PASTIS.classes)
    assert len(classes)==20
    input_bands=list(audit['expected']['band_default_order']['s2'])
    native_bands=list(Modality.SENTINEL2_L2A.band_order)
    assert native_bands == ['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09']
    aliases={'B01':'B02','B09':'B8A'}
    channel_indices=[input_bands.index(aliases.get(b,b)) for b in native_bands]
    normalizer=Normalizer(Strategy.COMPUTED)
    visual_selection=json.loads(a.visual_selection.read_text()) if a.visual_selection else None
    result={'status':'preparing', 'dataset_complete':audit['complete_dataset_verified'],
            'native_input_cases':0,'scientific_training_ready':False,'model_executed':False,
            'classes':classes,'input_bands':input_bands,'native_bands':native_bands,
            'imputation':aliases,'imputation_source':'pinned official OlmoEarth evals/datasets/pastis_processor.py impute',
            'imputed_bands_observed':False,'date_tests':[],
            'audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),
            'partial_train_parent_counts':train['tile'].value_counts().to_dict()}
    # Execute the installed loader's actual _load_image function on an indexed fixture.
    tree=ast.parse((raw_dir/'installed_pastis.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GeoBenchPASTIS')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_load_image')
    source=ast.unparse(method)
    namespace={'torch':torch,'h5py':h5py,'Tensor':torch.Tensor}
    exec(compile(source,'installed_pastis._load_image','exec'),namespace)
    fixture=io.BytesIO()
    with h5py.File(fixture,'w') as f:
        f.create_dataset('data',data=np.arange(5,dtype=np.float32).reshape(5,1,1,1))
    payload=fixture.getvalue()
    class LoaderFixture:
        num_time_steps=2
        temporal_aggregation=None
        def _return_byte_stream(self,path):return io.BytesIO(payload)
    observed=namespace['_load_image'](LoaderFixture(),'fixture').flatten().tolist()
    fixture_dates=[20190101,20190102,20190103,20190104,20190105]
    fixed_indices,fixed_ts=indices_and_dates(5,fixture_dates,2)
    getitem=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
    date_branch=next(n for n in getitem.body if isinstance(n,ast.If) and ast.unparse(n.test)=='len(dates) < self.num_time_steps')
    date_code=compile(ast.fix_missing_locations(ast.Module(body=[date_branch],type_ignores=[])), 'installed_pastis.__getitem__.date_branch', 'exec')
    date_ns={'dates':np.asarray(fixture_dates),'self':LoaderFixture()}
    exec(date_code,date_ns)
    tail_indices=[fixture_dates.index(int(d)) for d in date_ns['sample_dates']]
    result['date_tests'].append({'kind':'actual_image_method_and_date_branch_synthetic_fixture','image_frame_indices':observed,
        'getitem_tail_date_indices':tail_indices,'correct_joint_indices':fixed_indices,'fixed_timestamps':fixed_ts,
        'mismatch_reproduced':observed==[0.,2.] and tail_indices==[3,4] and fixed_indices==[0,2]})
    assert result['date_tests'][0]['mismatch_reproduced']
    # Boundary tests for the independent common-index routine.
    assert indices_and_dates(5,fixture_dates,1)[0]==[0]
    assert indices_and_dates(5,fixture_dates,5)[0]==list(range(5))
    for n,dates,k in [(5,fixture_dates[:-1],2),(5,fixture_dates,6),(5,list(reversed(fixture_dates)),2)]:
        try:indices_and_dates(n,dates,k)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    canvas=Image.new('RGB',(128*3,160*2),'white')
    draw=ImageDraw.Draw(canvas)
    rng=np.random.default_rng(0)
    palette=rng.integers(40,230,size=(20,3),dtype=np.uint8)
    palette[0]=0;palette[19]=[170,170,170]
    for idx in range(2):
        item=train.iloc[idx]
        nested=train.read(idx)
        dates=nested['dates'].iloc[0].tolist()
        assert list(nested['tortilla:id'])==['s2','s1a','s1d','semantic','instance']
        provenance=json.loads((raw_dir/f'inspection_record_{idx}.json').read_text())
        old=provenance['catalog_record']
        assert str(item['patch_id'])==str(old['patch_id']) and str(item['tile'])==str(old['tile'])
        assert item['tortilla:data_split']==old['tortilla:data_split']=='train'
        assert [int(v) for v in re.findall(r'\d{8}',old['dates'])]==[int(v) for v in dates]
        assert str(item['internal:subfile'])==old['internal:subfile']
        raw_path=raw_dir/f'inspection_raw_{idx}.npz'
        raw_sha=hashlib.sha256(raw_path.read_bytes()).hexdigest()
        raw=np.load(raw_path,allow_pickle=False)
        s2=raw['s2_data']
        semantic=raw['semantic_data'][0]
        instances=raw['instance_data']
        assert s2.shape[1:]==(10,128,128) and semantic.shape==(128,128)
        assert instances.shape==(128,128) and np.isfinite(instances).all()
        assert np.equal(instances,instances.astype(np.int64)).all()
        assert np.isfinite(s2).all() and np.isfinite(semantic).all()
        assert np.equal(semantic,semantic.astype(np.int64)).all()
        assert set(np.unique(semantic)).issubset(set(range(20)))
        selected,timestamps=indices_and_dates(len(s2),dates,2)
        if visual_selection:
            selection=visual_selection['cases'][str(item['patch_id'])]
            selected=selection['indices']
            assert len(selected)==2 and selected==sorted(set(selected)) and 0<=min(selected)<=max(selected)<len(s2)
            _,all_timestamps=indices_and_dates(len(s2),dates,len(s2))
            timestamps=[all_timestamps[i] for i in selected]
        native=np.transpose(s2[selected][:,channel_indices].astype(np.float32),(2,3,0,1))
        normalized=normalizer.normalize(Modality.SENTINEL2_L2A,native).astype(np.float32)
        assert np.isfinite(normalized).all()
        valid=(semantic!=19)
        crop_valid=(semantic>0)&valid
        labels,counts=np.unique(semantic[crop_valid],return_counts=True)
        assert len(labels)>0, 'No valid crop pixels for this engineering target'
        target=int(labels[np.argmax(counts)])
        target_mask=(semantic==target)&valid
        avail=np.asarray([b not in aliases for b in native_bands],dtype=np.bool_)
        np.savez_compressed(a.out/f'case_{idx}.npz',raw_s2=s2,raw_native_selected=native,
            normalized_s2=normalized,timestamps=np.asarray(timestamps,dtype=np.int64),
            semantic=semantic.astype(np.int64),label_valid=valid,crop_label_valid=crop_valid,
            target_mask=target_mask,instances=instances,band_observed=avail)
        actual_date_ns={'dates':np.asarray(dates),'self':LoaderFixture()}
        exec(date_code,actual_date_ns)
        legacy_dates=[int(d) for d in actual_date_ns['sample_dates']]
        chosen_dates=[int(dates[i]) for i in selected]
        row={'case':idx,'path':f'case_{idx}.npz','source_partial_train_index':idx,
            'source_raw_npz_sha256':raw_sha,'source_record':provenance,
            'patch_id':str(item['patch_id']),'parent_tile':str(item['tile']),
            'centroid_lon_lat':[float(item['lon']),float(item['lat'])],
            'exact_footprint_available':False,'all_dates_yyyymmdd':[int(d) for d in dates],
            'selected_indices':selected,'selected_dates':chosen_dates,'native_timestamps':timestamps,
            'selection':visual_selection if visual_selection else 'uniform engineering diagnostic',
            'legacy_loader_date_labels':legacy_dates,'legacy_dates_match':legacy_dates==chosen_dates,
            'raw_s2_shape':list(s2.shape),'normalized_s2_shape':list(normalized.shape),
            'target_class_id':target,'target_class_name':classes[target],
            'question':f'Locate the pixels labeled as {classes[target]} in this annual crop reference.',
            'answer_kind':'source_annual_crop_mask_not_monthly_health_or_change',
            'target_pixel_count':int(target_mask.sum()),'void_pixel_count':int((~valid).sum()),
            'normalization':'official COMPUTED once; no GeoBench ZScoreNormalizer',
            'band_availability':'sidecar only; encoder per-band mask not implemented',
            'cloud_validity':'not available; not inferred from crop labels',
            'local_coordinate_contract':'same 128x128 pixel grid for source, mask, RGB; no geodesic area claim'}
        records.append(row)
        for col,t in enumerate(selected):
            rgb=np.transpose(s2[t,[2,1,0]],(1,2,0))
            rgb=np.round(np.clip(rgb/3000,0,1)*255).astype(np.uint8)
            canvas.paste(Image.fromarray(rgb),(128*col,160*idx+24))
            draw.text((128*col+3,160*idx+5),str(int(dates[t])),fill='black')
        canvas.paste(Image.fromarray(palette[semantic.astype(np.int64)]),(256,160*idx+24))
        draw.text((259,160*idx+5),f'Annual mask {item["patch_id"]}',fill='black')
    canvas.save(a.out/'preview.png')
    (a.out/'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n')
    result.update(status='two_native_engineering_inputs_prepared_with_declared_imputation',native_input_cases=len(records),
        remaining=['full dataset missing/corrupt shards','original footprint not stored in this package',
                   'cloud/observation quality not labeled','independent parent train/dev/test manifest',
                   'real VLM trainer and checkpoint save/reload','two-date crop labels do not guarantee visual identifiability'],
        records=records,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        normalization_source_sha256=hashlib.sha256((a.source/'olmoearth_pretrain/data/normalize.py').read_bytes()).hexdigest(),
        normalization_config_sha256=hashlib.sha256((a.source/'olmoearth_pretrain/data/norm_configs/computed.json').read_bytes()).hexdigest(),
        normalization_std_multiplier=normalizer.std_multiplier,
        imputation_source_sha256=hashlib.sha256((a.source/'olmoearth_pretrain/evals/datasets/pastis_processor.py').read_bytes()).hexdigest())
    (a.out/'contract.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'cases':len(records),'legacy_dates_match':[r['legacy_dates_match'] for r in records]}))


if __name__=='__main__':main()
