#!/usr/bin/env python3
"""Frozen-sensitivity diagnostic extractor (prereg config/frozen_sensitivity_prereg_v0.json).
Same contract as the sealed P4 cache (OlmoEarth v1 base, 4x64 px crops, patch 4, token_pooling, B01/B09 zero + band-set 2 MISSING, real dates),
plus one perturbation: a pixel offset of the crop grid (tile viewed at (dy,dx), far edge replicated) or a month-label perturbation of the timestamps.
Writes emb_fp16 for the fold's TEST tiles only. Never touches sen12_pilot/."""
import argparse, os, sys, json, time
from pathlib import Path
from datetime import datetime
import numpy as np, torch
ap=argparse.ArgumentParser(); ap.add_argument("--fold",required=True); ap.add_argument("--arm",required=True); ap.add_argument("--offset",default="0,0"); ap.add_argument("--time",default="real",choices=["real","synth_month","plus1","shuffle","year_plus1","dup_month"]); ap.add_argument("--drop",default="none",choices=["none","rand3","rand6","contig3","contig6","pre3","post3","span3","late3","postkeep1","postkeep2","postkeep3","postlast1","postnone","sm_dropY","sm_swap"]); ap.add_argument("--cloud",type=int,default=0,help="replace the k clearest kept timesteps with the k cloudiest dropped ones (reads NetCDF)"); ap.add_argument("--data-root",default="/home/work/data/sen12landslides/extracted"); ap.add_argument("--outroot",default="frozen_sensitivity_v0"); ap.add_argument("--probe",action="store_true"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"sen12_pilot"/a.fold; OUT=ROOT/a.outroot/a.fold/a.arm; dev=torch.device("cuda"); PATCH=4
DY,DX=(int(v) for v in a.offset.split(",")); assert 0<=DY<=4 and 0<=DX<=4
MODEL_BANDS=["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
FOLDS=json.loads((ROOT/"sen12_gp_contract/loco_folds.json").read_text()); fold=next(f for f in FOLDS["folds"] if f["fold"]==a.fold)
REC={}
for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl"):
    if l.strip(): r=json.loads(l); REC[r["sample_id"]]=r
ids=sorted(s for s,r in REC.items() if r["region"]==fold["test_region"] and not r.get("error") and r.get("s15_eligible",True) and (SRC/"mask_u8"/f"{s}.npy").exists())
def real_timestamps(sid,T):
    r=REC[sid]; q=r["scl_clear_fraction"]; idx=sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:T])
    return [datetime.fromisoformat(str(r["times"][i])[:19]) for i in idx]
def perturb_times(sid,ts):
    if a.time=="real": return ts
    if a.time=="synth_month": return [datetime(2020,t.month,min(2+i,28)) for i,t in enumerate(ts)]
    if a.time=="plus1": return [t.replace(year=t.year+(t.month==12),month=t.month%12+1,day=min(t.day,28)) for t in ts]
    if a.time=="year_plus1": return [t.replace(year=t.year+1,day=min(t.day,28)) for t in ts]
    if a.time=="dup_month": return [ts[0]]+[ts[0].replace(day=min(ts[0].day%27+1,28))]+ts[2:]  # slot1 gets slot0 month, different day
    if a.time=="shuffle":
        rng=np.random.default_rng(int.from_bytes(sid.encode()[-8:],"little")%(2**32)); perm=rng.permutation(len(ts))
        return [datetime(ts[j].year,ts[j].month,min(2+i,28)) for i,j in enumerate(perm)]  # unique day per slot: wrapper rejects duplicate timestamps
(OUT/"emb_fp16").mkdir(parents=True,exist_ok=True)
if not (OUT/"mask_u8").exists(): os.symlink(SRC/"mask_u8",OUT/"mask_u8")
w=OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
def embed_crop(crop,ts):
    image=torch.from_numpy(crop).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts])}; w.normalizer(inp,{})
    sample,present,_=w._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        pooled=((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1))[0].permute(2,0,1).float().cpu()
    return pooled
@torch.no_grad()
def embed(sid):
    raw=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32"); T=raw.shape[1]; cube=np.zeros((12,T,128,128),dtype="float32"); cube[:10]=raw
    ts=real_timestamps(sid,T)
    if a.cloud:
        import xarray as xr; from datetime import datetime as _dt
        r=REC[sid]; q=[float(v) for v in r["scl_clear_fraction"]]; order=sorted(range(15),key=lambda i:(-q[i],i)); kept=sorted(order[:T]); dropped=order[T:]
        cloudiest=sorted(dropped,key=lambda i:q[i])[:a.cloud]; clearest=sorted(kept,key=lambda i:-q[i])[:a.cloud]; idx=sorted((set(kept)-set(clearest))|set(cloudiest))
        with xr.open_dataset(Path(a.data_root)/r["file"],decode_times=True,cache=False) as ds:
            cube=np.stack([np.asarray(ds[b].values[idx],dtype="float32") if b in ds else np.zeros((len(idx),128,128),dtype="float32") for b in MODEL_BANDS],0)
            ts=[_dt.fromisoformat(str(np.datetime_as_string(t,unit="s"))) for t in np.asarray(ds["time"].values)[idx]]
    if a.drop!="none":
        rng=np.random.default_rng(int.from_bytes(sid.encode()[-8:],"little")%(2**32)); k=int(a.drop[-1]) if a.drop[-1].isdigit() else 0; s0=int(rng.integers(0,T-k+1))
        if a.drop in ("pre3","post3","span3","late3"):
            r=REC[sid]; q=[float(v) for v in r["scl_clear_fraction"]]; kept=sorted(sorted(range(15),key=lambda i:(-q[i],i))[:T]); post=r.get("post_index")
            fp=next((j for j,i in enumerate(kept) if post is not None and i>=post),None)
            if fp is None or fp<3 or fp>T-3: s0=T//2-1; POSFLAG.append({"id":sid,"fallback":True,"first_post_slot":fp})
            else: s0={"pre3":fp-3,"post3":fp,"span3":fp-1,"late3":T-3}[a.drop]; POSFLAG.append({"id":sid,"fallback":False,"first_post_slot":fp,"s0":s0})
        if a.drop in ("sm_dropY","sm_swap"):
            # same-calendar-month pair among the kept 12 (first pair by time); Y = later of the pair
            pair=next(((i,i+1) for i in range(T-1) if ts[i].year==ts[i+1].year and ts[i].month==ts[i+1].month),None)
            if pair is None: POSFLAG.append({"id":sid,"fallback":True}); keep=list(range(T))
            else:
                x,y=pair; POSFLAG.append({"id":sid,"fallback":False,"pair":[x,y],"dates":[ts[x].isoformat(),ts[y].isoformat()]})
                if a.drop=="sm_dropY": keep=[i for i in range(T) if i!=y]
                else:
                    keep=list(range(T)); keep[x],keep[y]=y,x   # swap images; timestamps stay in original order below
            if a.drop=="sm_swap" and pair is not None: cube=cube[:,keep]; keep=list(range(T))
        elif a.drop.startswith("post") and a.drop not in ("post3",):
            r=REC[sid]; q=[float(v) for v in r["scl_clear_fraction"]]; kept=sorted(sorted(range(15),key=lambda i:(-q[i],i))[:T]); post=r.get("post_index")
            postslots=[j for j,i in enumerate(kept) if post is not None and i>=post]; preslots=[j for j in range(T) if j not in postslots]
            if not postslots: POSFLAG.append({"id":sid,"fallback":True,"n_post":0}); keep=list(range(T))
            else:
                sel={"postkeep1":postslots[:1],"postkeep2":postslots[:2],"postkeep3":postslots[:3],"postlast1":postslots[-1:],"postnone":[]}[a.drop]
                keep=sorted(preslots+sel); POSFLAG.append({"id":sid,"fallback":False,"n_post":len(postslots),"n_post_kept":len(sel),"first_post_slot":postslots[0]})
        else:
            keep=sorted(set(range(T))-set(rng.choice(T,k,replace=False).tolist())) if a.drop.startswith("rand") else [i for i in range(T) if not s0<=i<s0+k]
        cube=cube[:,keep]; ts=[ts[i] for i in keep]
    if DY or DX: cube=np.pad(cube,((0,0),(0,0),(0,DY),(0,DX)),mode="edge")[:,:,DY:DY+128,DX:DX+128]
    ts=perturb_times(sid,ts); feat=torch.empty((768,32,32))
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        feat[:,y0//PATCH:(y0+64)//PATCH,x0//PATCH:(x0+64)//PATCH]=embed_crop(np.ascontiguousarray(cube[:,:,y0:y0+64,x0:x0+64]),ts)
    return feat.numpy().astype("float16")
def valid(path):
    try: arr=np.load(path,mmap_mode="r",allow_pickle=False); return arr.dtype==np.float16 and tuple(arr.shape)==(768,32,32)
    except (OSError,ValueError,EOFError): return False
t0=time.perf_counter(); done=0; skipped=[]; POSFLAG=[]
for sid in (ids[:2] if a.probe else ids):
    o=OUT/"emb_fp16"/f"{sid}.npy"
    if o.exists() and valid(o): done+=1; continue
    try:
        e=embed(sid); tmp=o.with_name(f".{o.name}.{os.getpid()}.tmp.npy"); np.save(tmp,e,allow_pickle=False); os.replace(tmp,o); done+=1
    except Exception as ex:
        skipped.append({"id":sid,"err":str(ex)[:160]})
        if len(skipped)<3: import traceback; traceback.print_exc()
    if done%200==0 and done: print(done,"tiles",f"{time.perf_counter()-t0:.0f}s",flush=True)
if a.probe: print("probe ok",done); sys.exit(0)
n_valid=sum(valid(OUT/"emb_fp16"/f"{s}.npy") for s in ids)
audit={"schema":"frozen-sensitivity-extract-v0","fold":a.fold,"arm":a.arm,"offset_px":[DY,DX],"time":a.time,"drop":a.drop,"cloud":a.cloud,"n_ids":len(ids),"n_valid":n_valid,"n_skipped":len(skipped),"skipped":skipped[:20],"elapsed_s":time.perf_counter()-t0,"all_gates_pass":n_valid==len(ids) and not skipped}
(OUT/"extract_audit.json").write_text(json.dumps(audit,indent=1))
if POSFLAG: (OUT/"drop_positions.jsonl").write_text("\n".join(json.dumps(x) for x in POSFLAG)+"\n"); print(json.dumps({k:audit[k] for k in ("arm","n_ids","n_valid","n_skipped","all_gates_pass")})); print("PERTURB EXTRACT DONE")
