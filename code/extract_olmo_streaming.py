#!/usr/bin/env python3
"""T1 streaming-cache development extraction. Same OlmoEarth contract as the sealed extractor (4x64 crops, patch 4, token_pooling, B01/B09 zero+MISSING,
synthetic month timestamps). For each tile with T=12 timesteps writes:
  teacher_fp16/<sid>.npy : (5,768,32,32)  full-window pooled embedding for cutoffs c in CUTOFFS=(4,6,8,10,12) using timesteps [0:c]  (c=12 == sealed cache)
  single_fp16/<sid>.npy  : (12,768,32,32) single-acquisition embedding for each timestep t using window [t:t+1]
Cost bookkeeping: timestep-units encoded per tile = sum(CUTOFFS)+12 = 52 (vs 12 for the sealed cache)."""
import argparse, os, sys, json, time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np, torch
ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--src",default="sen12_pilot/holdout_chimanimani"); ap.add_argument("--ids-file",required=True); ap.add_argument("--probe",action="store_true"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/a.src; OUT=ROOT/a.out; dev=torch.device("cuda"); PATCH=4; CUTOFFS=(4,6,8,10,12)

# Real acquisition timestamps from the sealed contract (select_timestep_indices: keep the 12 clearest of 15 by SCL, ordered). Synthetic month
# timestamps change the embedding (cos .989 vs sealed on a probe tile, 2026-09-07); real times reproduce the sealed cache to 2e-3.
_REC={}
for _l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl"):
    if _l.strip(): _r=json.loads(_l); _REC[_r["sample_id"]]=_r
def real_timestamps(sid,T):
    r=_REC[sid]; q=r["scl_clear_fraction"]; idx=sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:T])
    return [datetime.fromisoformat(str(r["times"][i])[:19]) for i in idx]
for d in ("teacher_fp16","single_fp16"): (OUT/d).mkdir(parents=True,exist_ok=True)
for d in ("raw_u16","mask_u8","emb_fp16"):
    if not (OUT/d).exists(): os.symlink(SRC/d,OUT/d)
w=OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
months={json.loads(l)["sample_id"]:json.loads(l)["months_0_11"] for l in open(SRC/"months.jsonl") if l.strip()}
ids=[i for i in Path(a.ids_file).read_text().split() if (SRC/"raw_u16"/f"{i}.npy").exists()]
def pooled(crop,ts):
    image=torch.from_numpy(crop).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts])}; w.normalizer(inp,{})
    sample,present,_=w._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        return ((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1))[0].permute(2,0,1).float().cpu()   # (768,32/2,32/2) for a 64px crop
def window(cube,ts,t0,t1):
    feat=torch.empty((768,32,32))
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        feat[:,y0//PATCH:(y0+64)//PATCH,x0//PATCH:(x0+64)//PATCH]=pooled(np.ascontiguousarray(cube[:,t0:t1,y0:y0+64,x0:x0+64]),ts[t0:t1])
    return feat
@torch.no_grad()
def embed(sid):
    raw=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32"); T=raw.shape[1]; assert T==12,(sid,T)
    cube=np.zeros((12,T,128,128),dtype="float32"); cube[:10]=raw; ts=real_timestamps(sid,T)
    teacher=torch.stack([window(cube,ts,0,c) for c in CUTOFFS]); single=torch.stack([window(cube,ts,t,t+1) for t in range(T)])
    return teacher.numpy().astype("float16"), single.numpy().astype("float16")
def ok(p,n):
    try: arr=np.load(p,mmap_mode="r",allow_pickle=False); return arr.dtype==np.float16 and arr.shape==(n,768,32,32)
    except Exception: return False
def save(p,arr):
    tmp=p.with_name(f".{p.name}.{os.getpid()}.tmp.npy"); np.save(tmp,arr,allow_pickle=False); os.replace(tmp,p)
done=0; skipped=[]; audit=[]; t0=time.perf_counter()
for sid in (ids[:1] if a.probe else ids):
    tp,sp=OUT/"teacher_fp16"/f"{sid}.npy",OUT/"single_fp16"/f"{sid}.npy"
    if ok(tp,5) and ok(sp,12): done+=1; continue
    try:
        te,si=embed(sid)
        if len(audit)<30: ref=np.load(SRC/"emb_fp16"/f"{sid}.npy").astype("float32"); audit.append({"id":sid,"max_abs_diff_c12_vs_sealed":float(np.abs(te[-1].astype("float32")-ref).max())})
        if a.probe: print("probe",te.shape,si.shape,audit[-1],f"{time.perf_counter()-t0:.1f}s"); sys.exit(0)
        save(tp,te); save(sp,si); done+=1
    except Exception as ex:
        skipped.append({"id":sid,"err":str(ex)[:160]})
        if len(skipped)<3: import traceback; traceback.print_exc()
    if done%200==0 and done: print(done,"tiles",f"{time.perf_counter()-t0:.0f}s",flush=True)
n_valid=sum(1 for i in ids if ok(OUT/"teacher_fp16"/f"{i}.npy",5) and ok(OUT/"single_fp16"/f"{i}.npy",12))
rep={"schema":"olmo-streaming-dev-audit-v0","cutoffs":CUTOFFS,"n_ids":len(ids),"n_valid":n_valid,"n_skipped":len(skipped),"skipped":skipped[:20],"audit":audit,"audit_max":max((x["max_abs_diff_c12_vs_sealed"] for x in audit),default=None),"all_gates_pass":n_valid==len(ids) and not skipped,"elapsed_s":time.perf_counter()-t0,"timestep_units_per_tile":sum(CUTOFFS)+12}
(OUT/"olmo_streaming_audit.json").write_text(json.dumps(rep,indent=1)); print(json.dumps({k:v for k,v in rep.items() if k not in ("audit","skipped")})); print("OLMO STREAMING DEV DONE")
