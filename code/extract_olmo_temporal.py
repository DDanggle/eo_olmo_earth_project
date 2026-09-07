#!/usr/bin/env python3
"""Temporal-evidence cache (T0 screen): OlmoEarth v1 Base on the Sen12 tiles, SAME contract as the sealed extractor
(4x64 px crops, patch 4, token_pooling, B01/B09 zero + band-set 2 MISSING, synthetic month timestamps as extract_olmo_variants.py),
but pooled over band groups ONLY, keeping the time axis: emb_time_fp16/<sid>.npy with shape (T, 768, 32, 32).
Audit: mean over T must reproduce the sealed mean cache emb_fp16 (max |diff| recorded on --audit-n tiles; joint (T,G) masked mean
equals mean-over-T of the G-masked mean only when the missing pattern is constant over time, which holds for the B01/B09 contract)."""
import argparse, os, sys, json, time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np, torch
ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--src",default="sen12_pilot/holdout_chimanimani"); ap.add_argument("--probe",action="store_true"); ap.add_argument("--audit-n",type=int,default=50); ap.add_argument("--ids-file",default=None,help="optional newline list of sample ids to restrict extraction"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/a.src; OUT=ROOT/a.out; dev=torch.device("cuda"); PATCH=4

# Real acquisition timestamps from the sealed contract (select_timestep_indices: keep the 12 clearest of 15 by SCL, ordered). Synthetic month
# timestamps change the embedding (cos .989 vs sealed on a probe tile, 2026-09-07); real times reproduce the sealed cache to 2e-3.
_REC={}
for _l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl"):
    if _l.strip(): _r=json.loads(_l); _REC[_r["sample_id"]]=_r
def real_timestamps(sid,T):
    r=_REC[sid]; q=r["scl_clear_fraction"]; idx=sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:T])
    return [datetime.fromisoformat(str(r["times"][i])[:19]) for i in idx]
(OUT/"emb_time_fp16").mkdir(parents=True,exist_ok=True)
for d in ("raw_u16","mask_u8","emb_fp16"):
    if not (OUT/d).exists(): os.symlink(SRC/d,OUT/d)
if not (OUT/"months.jsonl").exists(): os.symlink(SRC/"months.jsonl",OUT/"months.jsonl")
w=OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
months={json.loads(l)["sample_id"]:json.loads(l)["months_0_11"] for l in open(SRC/"months.jsonl") if l.strip()}
ids=sorted(p.stem for p in (SRC/"emb_fp16").glob("*.npy"))
if a.ids_file: keep=set(Path(a.ids_file).read_text().split()); ids=[i for i in ids if i in keep]
def embed_crop(crop,ts):
    image=torch.from_numpy(crop).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts])}; w.normalizer(inp,{})
    sample,present,_=w._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        pooled=((tm.sentinel2_l2a*m).sum(dim=4)/m.sum(dim=4).clamp(min=1))[0]   # (H,W,T,D): band-group pooled, time kept
    return pooled.permute(2,3,0,1).float().cpu()                                  # (T,D,H,W)
@torch.no_grad()
def embed(sid):
    raw=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32"); T=raw.shape[1]; cube=np.zeros((12,T,128,128),dtype="float32"); cube[:10]=raw
    ts=real_timestamps(sid,T); feat=None
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        f=embed_crop(np.ascontiguousarray(cube[:,:,y0:y0+64,x0:x0+64]),ts)
        if feat is None: feat=torch.empty((f.shape[0],f.shape[1],128//PATCH,128//PATCH))
        feat[:,:,y0//PATCH:(y0+64)//PATCH,x0//PATCH:(x0+64)//PATCH]=f
    return feat.numpy().astype("float16")
def valid(path,T):
    try: arr=np.load(path,mmap_mode="r",allow_pickle=False); return arr.dtype==np.float16 and arr.shape[1:]==(768,32,32) and arr.shape[0]==T
    except (OSError,ValueError,EOFError): return False
def atomic_save(path,array):
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try: np.save(tmp,array,allow_pickle=False); os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()
done=0; skipped=[]; audit=[]; t0=time.perf_counter()
for sid in (ids[:2] if a.probe else ids):
    o=OUT/"emb_time_fp16"/f"{sid}.npy"; T=np.load(SRC/"raw_u16"/f"{sid}.npy",mmap_mode="r").shape[1]
    if o.exists() and valid(o,T): done+=1; continue
    try:
        e=embed(sid)
        if len(audit)<a.audit_n:
            ref=np.load(SRC/"emb_fp16"/f"{sid}.npy").astype("float32"); audit.append({"id":sid,"max_abs_diff_mean_vs_sealed":float(np.abs(e.astype("float32").mean(0)-ref).max()),"ref_abs_max":float(np.abs(ref).max())})
        if a.probe: print("feat",e.shape,audit[-1],flush=True); continue
        atomic_save(o,e); done+=1
    except Exception as ex:
        skipped.append({"id":sid,"err":str(ex)[:160]})
        if len(skipped)<3: import traceback; traceback.print_exc()
    if done%500==0 and done: print(done,"tiles",f"{time.perf_counter()-t0:.0f}s",flush=True)
if a.probe: sys.exit(0)
n_valid=sum(1 for sid in ids if valid(OUT/"emb_time_fp16"/f"{sid}.npy",np.load(SRC/"raw_u16"/f"{sid}.npy",mmap_mode="r").shape[1]))
rep={"schema":"olmo-temporal-cache-audit-v0","n_ids":len(ids),"n_valid":n_valid,"n_skipped":len(skipped),"skipped":skipped[:20],"audit_mean_vs_sealed":audit,"audit_max":max((x["max_abs_diff_mean_vs_sealed"] for x in audit),default=None),"all_gates_pass":n_valid==len(ids) and not skipped,"elapsed_s":time.perf_counter()-t0}
(OUT/"olmo_temporal_audit.json").write_text(json.dumps(rep,indent=1)); print(json.dumps({k:v for k,v in rep.items() if k not in ("audit_mean_vs_sealed","skipped")})); print("OLMO TEMPORAL CACHE DONE")
