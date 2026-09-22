#!/usr/bin/env python3
"""KuroSiwo (GEO-Bench-2) OlmoEarth Sentinel-1 streaming cache. Per tile (224x224 @10 m -> center 192x192 = 3x3 crops of 64 px, patch 4 -> 48x48 tokens):
  teacher3_fp16/<id>.npy  (768,48,48)   window [pre_1, pre_2, post]
  stale2_fp16/<id>.npy    (768,48,48)   window [pre_1, pre_2]
  single_fp16/<id>.npy    (3,768,48,48) single-date windows pre_1 / pre_2 / post
  mask_u8/<id>.npy        (192,192)     0 no-data, 1 no water, 2 permanent water, 3 flood   (GEO-Bench-2 remap of the on-disk 0/1/2/3=nodata coding)
  valid_u8/<id>.npy       (192,192)     invalid_data==1 (valid)
Units: on-disk linear backscatter -> 10*log10(clip(x,1e-6)); zeros (0.9% of pixels) are no-data -> set to the OlmoEarth band mean after normalisation (i.e. neutral) via mask? Simpler and documented: filled with -30 dB.
Timestamps: post = flood_date, pre_2 = -12 d, pre_1 = -24 d (declared approximation). Modality key 'sentinel1' bands [vv, vh]."""
import argparse, os, sys, json, time, numpy as np, torch
from pathlib import Path
from datetime import timedelta
import pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument("--out",default="kurosiwo_s1_cache"); ap.add_argument("--probe",action="store_true"); ap.add_argument("--split",default="all"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/olmoearth"); NPY=ROOT/"kurosiwo_npy"; OUT=ROOT/a.out; dev=torch.device("cuda"); PATCH=4; OFF=16; SIZE=192
for d in ("teacher3_fp16","stale2_fp16","single_fp16","mask_u8","valid_u8"): (OUT/d).mkdir(parents=True,exist_ok=True)
w=OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
META=[json.loads(l) for l in (NPY/"meta.jsonl").read_text().splitlines() if l]; meta=[]
def db(x): return 10*np.log10(np.clip(x,1e-6,None))
def pooled_batch(crops,ts):   # crops: list of (2,T,64,64) arrays -> list of (768,16,16)
    inputs=[]
    for c in crops:
        image=torch.from_numpy(c).to(dev); inp={"sentinel1":RasterImage(image=image,timestamps=[(x,x) for x in ts])}; w.normalizer(inp,{}); inputs.append(inp)
    sample,present,_=w._prepare_modality_inputs(ModelContext(inputs=inputs,metadatas=[])); assert present==["sentinel1"],present
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel1_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        return ((tm.sentinel1*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1)).permute(0,3,1,2).float().cpu()
def window(cube,ts,idx):   # cube (2,3,192,192) dB, idx = timestep indices
    crops=[np.ascontiguousarray(cube[:,idx,y:y+64,x:x+64]) for y in (0,64,128) for x in (0,64,128)]; p=pooled_batch(crops,[ts[i] for i in idx]); feat=torch.empty((768,48,48))
    k=0
    for y in (0,64,128):
        for x in (0,64,128): feat[:,y//4:(y+64)//4,x//4:(x+64)//4]=p[k]; k+=1
    return feat
rows=[(i,r) for i,r in enumerate(META) if (a.split=="all" or r["split"]==a.split) and (NPY/"raw_f32"/f"{r['id']}.npy").exists()]
done=0; t0=time.perf_counter()
for i,r in (rows[:1] if a.probe else rows):
    sid=r["id"]; fp=OUT/"single_fp16"/f"{sid}.npy"
    if fp.exists() and (OUT/"teacher3_fp16"/f"{sid}.npy").exists(): done+=1; continue
    raw=np.load(NPY/"raw_f32"/f"{sid}.npy"); imgs=[raw[:,k,OFF:OFF+SIZE,OFF:OFF+SIZE] for k in (0,1,2)]; mask=np.load(NPY/"mask_raw_u8"/f"{sid}.npy")[OFF:OFF+SIZE,OFF:OFF+SIZE]; valid=np.load(NPY/"valid_u8"/f"{sid}.npy")[OFF:OFF+SIZE,OFF:OFF+SIZE]
    cube=np.stack([db(im.astype("float64")) for im in imgs],1).astype("float32"); cube[np.stack([im==0 for im in imgs],1)]=-30.0   # (2,3,192,192)
    fd=pd.Timestamp(r["flood_date"]).to_pydatetime().replace(tzinfo=None); ts=[fd-timedelta(days=24),fd-timedelta(days=12),fd]
    te=window(cube,ts,[0,1,2]); st=window(cube,ts,[0,1]); si=torch.stack([window(cube,ts,[k]) for k in range(3)])
    remap={3:0,0:1,1:2,2:3}; mk=np.vectorize(remap.get)(mask).astype("uint8")
    if a.probe: print("probe",te.shape,st.shape,si.shape,mk.shape,"classes",np.unique(mk).tolist(),"cos(te,st)",float(torch.nn.functional.cosine_similarity(te.flatten(),st.flatten(),dim=0)),f"{time.perf_counter()-t0:.1f}s"); sys.exit(0)
    np.save(OUT/"teacher3_fp16"/f"{sid}.npy",te.numpy().astype("float16")); np.save(OUT/"stale2_fp16"/f"{sid}.npy",st.numpy().astype("float16")); np.save(fp,si.numpy().astype("float16")); np.save(OUT/"mask_u8"/f"{sid}.npy",mk); np.save(OUT/"valid_u8"/f"{sid}.npy",(valid==1).astype("uint8"))
    meta.append(dict(r)); done+=1
    if done%500==0: print(done,"tiles",f"{time.perf_counter()-t0:.0f}s",flush=True)
mp=OUT/"meta.jsonl"
with open(mp,"a") as f:
    for m_ in meta: f.write(json.dumps(m_)+"\n")
print(json.dumps({"done":done,"elapsed_s":time.perf_counter()-t0})); print("KUROSIWO S1 CACHE DONE")
