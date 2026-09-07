#!/usr/bin/env python3
"""OlmoEarth v1 variants on the Sen12 tiles for the architecture axes (addendum_v1b): --size nano|tiny|base, --depth-frac (truncate encoder blocks).
Same crop/pool contract as the sealed extractor (4x64 px crops, patch 4, token_pooling, B01/B09 zero + band-set 2 MISSING). Deviation: timestamps synthesised as day 15 of the cached month (year 2020) instead of the NetCDF dates."""
import argparse, os, sys, json, time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np, torch, torch.nn as nn
ap=argparse.ArgumentParser(); ap.add_argument("--size",default="base",choices=["nano","tiny","base"]); ap.add_argument("--depth-frac",type=float,default=1.0); ap.add_argument("--patch",type=int,default=4,help="OlmoEarth patch size (4 -> 40 m tokens, 2 -> 20 m tokens)"); ap.add_argument("--out",required=True); ap.add_argument("--probe",action="store_true"); ap.add_argument("--src",default="sen12_pilot/holdout_chimanimani"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
from cache_grid_controls import expected_olmo_shape
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/a.src; OUT=ROOT/a.out; dev=torch.device("cuda")
started_perf=time.perf_counter()
(OUT/"emb_fp16").mkdir(parents=True,exist_ok=True)
for d in ("raw_u16","mask_u8"):
    if not (OUT/d).exists(): os.symlink(SRC/d,OUT/d)
for f in ("months.jsonl",):
    if not (OUT/f).exists(): os.symlink(SRC/f,OUT/f)
MID={"nano":ModelID.OLMOEARTH_V1_NANO,"tiny":ModelID.OLMOEARTH_V1_TINY,"base":ModelID.OLMOEARTH_V1_BASE}[a.size]
w=OlmoEarth(patch_size=a.patch, model_id=MID, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
enc=w.model; nb=len(enc.blocks)
try: EXPECTED_SHAPE=expected_olmo_shape(a.size,a.patch)
except ValueError as exc: raise SystemExit(str(exc)) from exc
if a.depth_frac<1.0: k=max(1,int(round(nb*a.depth_frac))); enc.blocks=nn.ModuleList(list(enc.blocks)[:k]); print("olmo depth",k,"/",nb,flush=True)
months={json.loads(l)["sample_id"]:json.loads(l)["months_0_11"] for l in open(SRC/"months.jsonl") if l.strip()}
ids=sorted(p.stem for p in ((SRC/"emb_fp16") if (SRC/"emb_fp16").exists() and any((SRC/"emb_fp16").glob("*.npy")) else (SRC/"raw_u16")).glob("*.npy")); done=0; skipped=[]
def embed_crop(crop,ts):
    image=torch.from_numpy(crop).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts])}; w.normalizer(inp,{})
    sample,present,_=w._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=a.patch)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        pooled=((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1))[0].permute(2,0,1).float().cpu()
    return pooled
@torch.no_grad()
def embed(sid):
    raw=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32"); T=raw.shape[1]; cube=np.zeros((12,T,128,128),dtype="float32"); cube[:10]=raw
    ts=[datetime(2020,int(m)+1,1)+timedelta(days=1+i) for i,m in enumerate(months.get(sid,[0]*T)[:T])]; feat=None
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        f=embed_crop(np.ascontiguousarray(cube[:,:,y0:y0+64,x0:x0+64]),ts)
        if feat is None: feat=torch.empty((f.shape[0],128//a.patch,128//a.patch))
        feat[:,y0//a.patch:(y0+64)//a.patch,x0//a.patch:(x0+64)//a.patch]=f
    if a.probe: print("feat",tuple(feat.shape),flush=True); return None
    return feat.detach().cpu().numpy().astype("float16")
def valid_cached_embedding(path):
    """Return True only for a complete, readable cache entry.

    Existence alone is not sufficient: a killed ``np.save`` can leave a truncated
    file that a later run would otherwise skip forever.
    """
    try:
        arr=np.load(path,mmap_mode="r",allow_pickle=False)
        return arr.dtype==np.float16 and tuple(arr.shape)==EXPECTED_SHAPE
    except (OSError,ValueError,EOFError):
        return False

def atomic_save(path,array):
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(tmp,array,allow_pickle=False)
        os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()

for sid in (ids[:2] if a.probe else ids):
    o=OUT/"emb_fp16"/f"{sid}.npy"
    if o.exists() and valid_cached_embedding(o): done+=1; continue
    try:
        e=embed(sid)
        if e is not None: atomic_save(o,e); done+=1
    except Exception as ex:
        skipped.append({"id":sid,"err":str(ex)[:160]})
        if a.probe or len(skipped)<3: import traceback; traceback.print_exc()
    if done%1000==0 and done: print(done,"tiles",flush=True)
if a.probe: sys.exit(0)
fs=sorted((OUT/"emb_fp16").glob("*.npy")); id_set=set(ids); file_ids={p.stem for p in fs}
valid=[]; invalid=[]; shapes=set()
for path in fs:
    try:
        arr=np.load(path,mmap_mode="r",allow_pickle=False)
        shapes.add((str(arr.dtype),tuple(arr.shape)))
        (valid if valid_cached_embedding(path) else invalid).append(path.stem)
    except (OSError,ValueError,EOFError):
        invalid.append(path.stem)
audit={"schema":"olmo-variant-cache-audit-v3","size":a.size,"depth_frac":a.depth_frac,"patch":a.patch,"blocks_total":nb,"expected_shape":list(EXPECTED_SHAPE),"shapes":[[dtype,list(shape)] for dtype,shape in sorted(shapes)],"n_tiles":len(fs),"n_valid":len(valid),"expected":len(ids),"missing_ids":sorted(id_set-file_ids)[:20],"unexpected_ids":sorted(file_ids-id_set)[:20],"n_invalid":len(invalid),"invalid_ids":invalid[:20],"n_skipped":len(skipped),"skipped":skipped[:20],"cache_bytes":sum(p.stat().st_size for p in fs),"elapsed_s_this_invocation":time.perf_counter()-started_perf,"all_gates_pass":file_ids==id_set and len(valid)==len(ids) and not invalid and not skipped and shapes=={("float16",EXPECTED_SHAPE)},"deviation":"synthetic unique timestamps (cached month, day 2+i, 2020)"}
(OUT/"olmo_variant_audit.json").write_text(json.dumps(audit,indent=1)); print(json.dumps({k:audit[k] for k in ("all_gates_pass","n_tiles","n_valid","n_invalid","n_skipped")})); print("OLMO VARIANT CACHE DONE")
