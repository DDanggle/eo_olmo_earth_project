#!/usr/bin/env python3
"""OlmoEarth v1 variants on the Sen12 tiles for the architecture axes (addendum_v1b): --size nano|tiny|base, --depth-frac (truncate encoder blocks).
Same crop/pool contract as the sealed extractor (4x64 px crops, patch 4, token_pooling, B01/B09 zero + band-set 2 MISSING). Deviation: timestamps synthesised as day 15 of the cached month (year 2020) instead of the NetCDF dates."""
import argparse, os, sys, json
from pathlib import Path
from datetime import datetime
import numpy as np, torch, torch.nn as nn
ap=argparse.ArgumentParser(); ap.add_argument("--size",default="base",choices=["nano","tiny","base"]); ap.add_argument("--depth-frac",type=float,default=1.0); ap.add_argument("--out",required=True); ap.add_argument("--probe",action="store_true"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"sen12_pilot/holdout_chimanimani"; OUT=ROOT/a.out; dev=torch.device("cuda")
(OUT/"emb_fp16").mkdir(parents=True,exist_ok=True)
for d in ("raw_u16","mask_u8"):
    if not (OUT/d).exists(): os.symlink(SRC/d,OUT/d)
for f in ("months.jsonl","cache_audit.json"):
    if not (OUT/f).exists(): os.symlink(SRC/f,OUT/f)
MID={"nano":ModelID.OLMOEARTH_V1_NANO,"tiny":ModelID.OLMOEARTH_V1_TINY,"base":ModelID.OLMOEARTH_V1_BASE}[a.size]
w=OlmoEarth(patch_size=4, model_id=MID, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
enc=w.model; nb=len(enc.blocks)
if a.depth_frac<1.0: k=max(1,int(round(nb*a.depth_frac))); enc.blocks=nn.ModuleList(list(enc.blocks)[:k]); print("olmo depth",k,"/",nb,flush=True)
months={json.loads(l)["sample_id"]:json.loads(l)["months_0_11"] for l in open(SRC/"months.jsonl") if l.strip()}
ids=sorted(p.stem for p in (SRC/"emb_fp16").glob("*.npy")); done=0; skipped=[]
@torch.no_grad()
def embed_crop(crop,ts):
    image=torch.from_numpy(crop).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts])}; w.normalizer(inp,{})
    sample,present,_=w._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=4)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        pooled=((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1))[0].permute(2,0,1).float().cpu()
    return pooled
def embed(sid):
    raw=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32"); T=raw.shape[1]; cube=np.zeros((12,T,128,128),dtype="float32"); cube[:10]=raw
    ts=[datetime(2020,int(m)+1,15) for m in months.get(sid,[0]*T)[:T]]; feat=None
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        f=embed_crop(np.ascontiguousarray(cube[:,:,y0:y0+64,x0:x0+64]),ts)
        if feat is None: feat=torch.empty((f.shape[0],32,32))
        feat[:,y0//4:(y0+64)//4,x0//4:(x0+64)//4]=f
    if a.probe: print("feat",tuple(feat.shape),flush=True); return None
    return feat.numpy().astype("float16")
for sid in (ids[:2] if a.probe else ids):
    o=OUT/"emb_fp16"/f"{sid}.npy"
    if o.exists(): done+=1; continue
    try:
        e=embed(sid)
        if e is not None: np.save(o,e); done+=1
    except Exception as ex:
        skipped.append({"id":sid,"err":str(ex)[:160]})
        if a.probe or len(skipped)<3: import traceback; traceback.print_exc()
    if done%1000==0 and done: print(done,"tiles",flush=True)
if a.probe: sys.exit(0)
fs=sorted((OUT/"emb_fp16").glob("*.npy")); arr=np.load(fs[0],mmap_mode="r")
audit={"schema":"olmo-variant-cache-audit-v1","size":a.size,"depth_frac":a.depth_frac,"blocks_total":nb,"shape":list(arr.shape),"n_tiles":len(fs),"expected":len(ids),"n_skipped":len(skipped),"skipped":skipped[:20],"all_gates_pass":len(fs)==len(ids) and tuple(arr.shape[1:])==(32,32),"deviation":"synthetic timestamps (day 15 of cached month, 2020)"}
(OUT/"olmo_variant_audit.json").write_text(json.dumps(audit,indent=1)); print(json.dumps({k:audit[k] for k in ("all_gates_pass","n_tiles","n_skipped","shape")})); print("OLMO VARIANT CACHE DONE")
