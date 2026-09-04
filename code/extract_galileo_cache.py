#!/usr/bin/env python3
"""Galileo (nasaharvest, ICML 2025) frozen cache on the Sen12 tiles. S2-only (10 bands), T=12 timesteps, months from cache, patch_size 4 (10 m px -> 40 m token, 32x32 grid = OlmoEarth contract).
Output per tile: mean over T and over seen S2 band-group tokens -> (D,32,32) fp16. --probe prints token shapes on 2 tiles and exits."""
import argparse, os, sys, json
from pathlib import Path
import numpy as np, torch
ap=argparse.ArgumentParser(); ap.add_argument("--size",default="base"); ap.add_argument("--patch",type=int,default=4); ap.add_argument("--out",default="galileo_cache"); ap.add_argument("--probe",action="store_true"); a=ap.parse_args()
sys.path.insert(0,"/home/work/data/olmoearth/third_party/pydeps"); sys.path.insert(0,"/home/work/data/olmoearth/third_party/galileo")
from single_file_galileo import Encoder
from src.data.utils import construct_galileo_input
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"sen12_pilot/holdout_chimanimani"; OUT=ROOT/a.out; dev=torch.device("cuda")
(OUT/"emb_fp16").mkdir(parents=True,exist_ok=True)
for d in ("raw_u16","mask_u8"):
    if not (OUT/d).exists(): os.symlink(SRC/d,OUT/d)
for f in ("months.jsonl","cache_audit.json"):
    if not (OUT/f).exists(): os.symlink(SRC/f,OUT/f)
enc=Encoder.load_from_folder(Path("/home/work/data/olmoearth/third_party/galileo/data/models")/a.size, device=dev).to(dev).eval()
RAW=["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12"]; GAL=["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]; perm=[RAW.index(b) for b in GAL]
months={}
for l in (SRC/"months.jsonl").read_text().splitlines():
    if l: r=json.loads(l); months[r["sample_id"]]=r["months_0_11"]
ids=sorted(p.stem for p in (SRC/"emb_fp16").glob("*.npy")); done=0; skipped=[]
@torch.no_grad()
def encode_crop(x,mo):  # x (10,T,64,64) -> (D,16,16); same 4x64px tiling as the OlmoEarth extractor
    T=x.shape[1]; s2=torch.from_numpy(np.ascontiguousarray(x)).permute(2,3,1,0).contiguous()
    m=construct_galileo_input(s2=s2,months=mo,normalize=True)
    args=[t.unsqueeze(0).to(dev).float() for t in (m.space_time_x,m.space_x,m.time_x,m.static_x)]+[t.unsqueeze(0).to(dev) for t in (m.space_time_mask,m.space_mask,m.time_mask,m.static_mask)]
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=enc(*args, m.months.unsqueeze(0).to(dev).long(), patch_size=a.patch)
    s_t_x,s_t_m=out[0].float(),out[4]
    if a.probe: print("space_time_x",tuple(s_t_x.shape),"mask",tuple(s_t_m.shape),"mask uniq",torch.unique(s_t_m).tolist(),"space_x",tuple(out[1].shape),"time_x",tuple(out[2].shape),"static",tuple(out[3].shape),flush=True)
    keep=(s_t_m==0).unsqueeze(-1).float()
    tok=(s_t_x*keep).sum(dim=(3,4))/keep.sum(dim=(3,4)).clamp(min=1)
    return tok[0].permute(2,0,1)
def embed(sid):
    x=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32")[perm]; T=x.shape[1]; mo=torch.tensor(months.get(sid,list(range(T)))[:T],dtype=torch.long)
    g=64//a.patch; feat=None
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        f=encode_crop(x[:,:,y0:y0+64,x0:x0+64],mo)
        if feat is None: feat=torch.empty((f.shape[0],128//a.patch,128//a.patch))
        feat[:,y0//a.patch:y0//a.patch+g,x0//a.patch:x0//a.patch+g]=f.cpu()
    if a.probe: return None
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
    if done%500==0 and done: print(done,"tiles",flush=True)
if a.probe: sys.exit(0)
fs=sorted((OUT/"emb_fp16").glob("*.npy")); arr=np.load(fs[0],mmap_mode="r")
audit={"schema":"galileo-cache-audit-v1","model":f"Galileo {a.size}","patch":a.patch,"shape":list(arr.shape),"n_tiles":len(fs),"expected":len(ids),"n_skipped":len(skipped),"skipped":skipped[:30],"all_gates_pass":len(fs)==len(ids) and tuple(arr.shape[1:])==(128//a.patch,128//a.patch),
       "contract":"S2 10 bands (Galileo order), T=12, months from cache, normalize=True (pretraining stats), other modalities absent/masked; mean over T and seen band-group tokens"}
(OUT/"galileo_audit.json").write_text(json.dumps(audit,indent=1)); print(json.dumps({k:audit[k] for k in ("all_gates_pass","n_tiles","n_skipped","shape")})); print("GALILEO CACHE DONE")
