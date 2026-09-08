#!/usr/bin/env python3
"""Cross-sensor streaming material: for each Sen12 tile in the T1 manifests, embed Sentinel-1 ascending SINGLE acquisitions with OlmoEarth
(modality 'sentinel1', bands [vv, vh], dB as stored), aligned to the 12 selected S2 dates by nearest S1 acquisition date.
Writes olmo_streaming_dev/single_s1_fp16/<sid>.npy (12,768,32,32) and single_s1_dates/<sid>.json (S1 dates used, |gap| days to the S2 date).
Same crop contract as the S2 cache (4x64 px crops, patch 4). S1 MASK/DEM ignored. Values with VV==0 (nodata, 0.1%) left as is (0 dB) and noted."""
import argparse, json, os, sys, time, numpy as np, torch, xarray as xr
from pathlib import Path
from datetime import datetime
ap=argparse.ArgumentParser(); ap.add_argument("--ids-file",default="sen12_gp_contract/t1_ids_all.txt"); ap.add_argument("--probe",action="store_true"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"olmo_streaming_dev"; EXT=Path("/home/work/data/sen12landslides/extracted"); dev=torch.device("cuda"); PATCH=4
(OUT/"single_s1_fp16").mkdir(parents=True,exist_ok=True); (OUT/"single_s1_dates").mkdir(exist_ok=True)
REC={}
for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl"):
    if l.strip(): r=json.loads(l); REC[r["sample_id"]]=r
def s2_dates(sid):
    r=REC[sid]; q=r["scl_clear_fraction"]; idx=sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:12]); return [datetime.fromisoformat(str(r["times"][i])[:19]) for i in idx]
w=OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
def embed_singles(cube,ts):   # cube (2,T,128,128) dB -> (T,768,32,32): all T singles x 4 crops in one forward
    inputs=[]
    for t in range(cube.shape[1]):
        for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
            image=torch.from_numpy(np.ascontiguousarray(cube[:,t:t+1,y0:y0+64,x0:x0+64])).to(dev); inp={"sentinel1":RasterImage(image=image,timestamps=[(ts[t],ts[t])])}; w.normalizer(inp,{}); inputs.append(inp)
    sample,present,_=w._prepare_modality_inputs(ModelContext(inputs=inputs,metadatas=[])); assert present==["sentinel1"],present
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel1_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        p=((tm.sentinel1*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1)).permute(0,3,1,2).float().cpu()   # (T*4,768,16,16)
    out=torch.empty((cube.shape[1],768,32,32))
    for t in range(cube.shape[1]):
        for j,(y0,x0) in enumerate(((0,0),(0,64),(64,0),(64,64))): out[t,:,y0//4:(y0+64)//4,x0//4:(x0+64)//4]=p[t*4+j]
    return out
ids=[s for s in Path(ROOT/a.ids_file).read_text().split() if s]
done=0; skipped=[]; t0=time.perf_counter(); gaps_all=[]
for sid in (ids[:1] if a.probe else ids):
    o=OUT/"single_s1_fp16"/f"{sid}.npy"
    if o.exists():
        try:
            if np.load(o,mmap_mode="r").shape==(12,768,32,32): done+=1; continue
        except Exception: pass
    region,num=sid.rsplit("_s2_",1); p=EXT/f"{region}_s1asc_{num}.nc"
    if not p.exists(): skipped.append({"id":sid,"err":"no s1asc file"}); continue
    try:
        ds=xr.open_dataset(p,decode_times=True); s1t=[datetime.fromisoformat(str(t)[:19]) for t in ds.time.values]; vv=ds["VV"].values.astype("float32"); vh=ds["VH"].values.astype("float32"); ds.close()
        sel=[]; gaps=[]
        for d in s2_dates(sid):
            k=int(np.argmin([abs((t-d).days) for t in s1t])); sel.append(k); gaps.append((s1t[k]-d).days)
        cube=np.stack([vv[sel],vh[sel]],0)            # (2,12,128,128) dB
        cube=np.nan_to_num(cube,nan=-30.0,posinf=0.0,neginf=-30.0)
        emb=embed_singles(cube,[s1t[k] for k in sel])
        if a.probe: print("probe",emb.shape,"gaps(days)",gaps,f"{time.perf_counter()-t0:.1f}s"); sys.exit(0)
        np.save(o,emb.numpy().astype("float16")); (OUT/"single_s1_dates"/f"{sid}.json").write_text(json.dumps({"s1_dates":[str(s1t[k])[:10] for k in sel],"gap_days":gaps})); gaps_all+= [abs(g) for g in gaps]; done+=1
    except Exception as ex:
        skipped.append({"id":sid,"err":str(ex)[:160]})
        if len(skipped)<3: import traceback; traceback.print_exc()
    if done%500==0 and done: print(done,"tiles",f"{time.perf_counter()-t0:.0f}s",flush=True)
rep={"n_ids":len(ids),"done":done,"skipped":len(skipped),"skipped_list":skipped[:20],"gap_days_mean":float(np.mean(gaps_all)) if gaps_all else None,"gap_days_p90":float(np.percentile(gaps_all,90)) if gaps_all else None,"elapsed_s":time.perf_counter()-t0}
(OUT/"single_s1_audit.json").write_text(json.dumps(rep,indent=1)); print(json.dumps({k:v for k,v in rep.items() if k!="skipped_list"})); print("S1 SINGLES DONE")
