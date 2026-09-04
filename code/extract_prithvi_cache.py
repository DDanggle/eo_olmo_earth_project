#!/usr/bin/env python3
"""Prithvi-EO-2.0-300M (IBM/NASA, temporal 3D-patch ViT) frozen cache on the Sen12 tiles.
Contract shift is intrinsic (HLS 6 bands @30 m vs S2 10 m): we feed 10 m 128 px tiles, 6 S2 bands mapped to the HLS order [B02,B03,B04,B8A,B11,B12] -> ["B02","B03","B04","B05","B06","B07"],
T=4 evenly spaced of the 12 cached timesteps (indices 1,4,7,10; registered), patch (1,16,16) -> 8x8 tokens (160 m support), mean over T -> (1024,8,8) fp16. mask_ratio 0."""
import argparse, os, sys, json, inspect
from pathlib import Path
import numpy as np, torch
ap=argparse.ArgumentParser(); ap.add_argument("--out",default="prithvi_cache"); ap.add_argument("--probe",action="store_true"); ap.add_argument("--tsel",default="1,4,7,10"); a=ap.parse_args()
sys.path.insert(0,"/home/work/data/olmoearth/third_party/pydeps"); sys.path.insert(0,"/home/work/data/olmoearth/third_party/prithvi_eo2_300m")
from prithvi_mae import PrithviMAE
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"sen12_pilot/holdout_chimanimani"; OUT=ROOT/a.out; dev=torch.device("cuda"); MD=Path("/home/work/data/olmoearth/third_party/prithvi_eo2_300m")
(OUT/"emb_fp16").mkdir(parents=True,exist_ok=True)
for d in ("raw_u16","mask_u8"):
    if not (OUT/d).exists(): os.symlink(SRC/d,OUT/d)
for f in ("months.jsonl","cache_audit.json"):
    if not (OUT/f).exists(): os.symlink(SRC/f,OUT/f)
cfg=json.loads((MD/"config.json").read_text())["pretrained_cfg"]; TSEL=[int(t) for t in a.tsel.split(",")]
sig=inspect.signature(PrithviMAE.__init__).parameters; kw={k:v for k,v in cfg.items() if k in sig}; kw.update(img_size=128,num_frames=len(TSEL),in_chans=6,coords_encoding=[])
model=PrithviMAE(**kw); sd=torch.load(MD/"Prithvi_EO_V2_300M.pt",map_location="cpu",weights_only=True)
sd={k:v for k,v in sd.items() if "pos_embed" not in k}; miss=model.load_state_dict(sd,strict=False); model=model.to(dev).eval()
print("missing",len(miss.missing_keys),"unexpected",len(miss.unexpected_keys),[k for k in miss.missing_keys if "pos_embed" not in k][:5],flush=True)
RAW=["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12"]; perm=[RAW.index(b) for b in ("B02","B03","B04","B8A","B11","B12")]
mean=torch.tensor(cfg["mean"]).view(1,6,1,1,1).to(dev); std=torch.tensor(cfg["std"]).view(1,6,1,1,1).to(dev)
ids=sorted(p.stem for p in (SRC/"emb_fp16").glob("*.npy")); done=0; skipped=[]
@torch.no_grad()
def embed(sid):
    x=np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32")[perm][:,TSEL]          # (6,4,128,128) DN ~ reflectance*1e4
    x=(torch.from_numpy(x).unsqueeze(0).to(dev)-mean)/std
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tok,_,_=model.encoder(x,mask_ratio=0.0)                                       # (1, 1+T*64, D)
    tok=tok[0,1:,:].float(); T=len(TSEL); g=128//16; tok=tok.reshape(T,g,g,-1).mean(0).permute(2,0,1)  # (D,8,8)
    if a.probe: print("tokens",tuple(tok.shape),flush=True); return None
    return tok.cpu().numpy().astype("float16")
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
audit={"schema":"prithvi-cache-audit-v1","model":"Prithvi-EO-2.0-300M","patch":[1,16,16],"shape":list(arr.shape),"n_tiles":len(fs),"expected":len(ids),"n_skipped":len(skipped),"skipped":skipped[:30],"all_gates_pass":len(fs)==len(ids) and tuple(arr.shape[1:])==(8,8),
       "contract":f"S2 6 bands mapped to HLS order, 10 m input (contract shift acknowledged), T={len(TSEL)} indices {TSEL}, mean over T, mask_ratio 0, coords encodings off"}
(OUT/"prithvi_audit.json").write_text(json.dumps(audit,indent=1)); print(json.dumps({k:audit[k] for k in ("all_gates_pass","n_tiles","n_skipped","shape")})); print("PRITHVI CACHE DONE")
