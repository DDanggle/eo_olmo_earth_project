#!/usr/bin/env python3
"""Real official-train S2 input, pretrained Tiny forward/backward execution check only."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path('/home/work/data/olmoearth')
sys.path.insert(0,str(ROOT/'region_language_contract_20260921/deps'))

def main():
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    import torch
    from eo_model import EOImageEncoder, S2_BANDS
    from datetime import datetime, timezone
    a=argparse.ArgumentParser()
    a.add_argument('--model-dir',type=Path,required=True)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--data-dir',type=Path)
    args=a.parse_args()
    started=time.monotonic()
    if args.data_dir:
        manifest=json.loads((args.data_dir/'manifest.json').read_text())
        if manifest['status']!='complete': raise RuntimeError('Data preparation incomplete')
        row=json.loads((args.data_dir/'train.jsonl').read_text().splitlines()[0])
        if row['official_split']!='train': raise RuntimeError('Smoke must use official train')
        patch,date=row['patch_id'],row['date']
        array=np.load(args.data_dir/row['image_path'],allow_pickle=False)
        array=array[[row['bands'].index(b) for b in S2_BANDS]]
    else:
        old=ROOT/'region_language_contract_20260921'
        rows=json.loads((old/'sample_image_text.json').read_text())
        ids=sorted({r['patch_id'] for r in rows if r['split']=='train'})
        if not ids: raise RuntimeError('No official train images available for smoke')
        patch=ids[0]
        planes=[]
        for b in S2_BANDS:
            with rasterio.open(old/'images'/patch/f'{patch}_{b}.tif') as ds:
                planes.append(ds.read(1,out_shape=(120,120),resampling=Resampling.bilinear))
        array=np.stack(planes)
        date=datetime.strptime(patch.split('_')[2],'%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc).isoformat()
    raw=torch.from_numpy(array.astype('float32'))[None].cuda()
    reports={}
    for mode in ['frozen','joint']:
        model=EOImageEncoder.from_pretrained(args.model_dir,trainable=mode=='joint').cuda()
        before=model.capture_parameter_state()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            f=model(raw,[date],band_names=S2_BANDS)
            loss=f.float().square().mean()
        if mode=='joint':
            opt=torch.optim.AdamW(model.parameters(),lr=1e-5)
            loss.backward()
            opt.step()
        report=model.audit_after_step(before)
        if mode=='joint' and not report['changed_parameter_tensors']: raise RuntimeError('No encoder update')
        reports[mode]={'features_shape':list(f.shape),'loss':float(loss.detach()),'requires_grad':f.requires_grad,'audit':report}
        del model,f,loss
        if mode=='joint': del opt
        torch.cuda.empty_cache()
    output={'kind':'execution_smoke_not_accuracy','patch_id':patch,'split':'train','date':date,
            'device':torch.cuda.get_device_name(),'seconds':time.monotonic()-started,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reports':reports}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))

if __name__=='__main__': main()
