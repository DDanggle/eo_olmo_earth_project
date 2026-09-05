#!/usr/bin/env python3
"""두 번째 FM 캐시 — Clay v1.5 (config/second_fm_cache_prereg_v0.json). GPU1.
Sen12 S12q 타일(raw_u16 10밴드×12시점×128², 캐시와 동일 표본)을 Clay 인코더로 시점별 임베딩 → 시점 평균 → 32×32 격자(bilinear) → fp16.
Clay 계약: 밴드 순서 blue,green,red,rededge1,rededge2,rededge3,nir,nir08,swir16,swir22 = B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12 (raw 순서 B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12 를 재배열),
metadata.yaml 의 mean/std 정규화, waves = 파장(µm), gsd=10, time=[sin/cos(week), sin/cos(hour)], latlon=[sin/cos(lat), sin/cos(lon)] (Clay inference 튜토리얼 규약).
근사(등록): 시각은 월 인덱스(months.jsonl)에서 주=월*4.35+2, 시각=10.5h(S2 강하 시각). 타일 중심은 nc 속성 center(UTM)+crs 를 WGS84 로 변환.
출력: /home/work/data/olmoearth/clay_cache/emb_fp16/<sid>.npy (D,32,32) + cache_audit.json + 소스 캐시의 mask_u8/raw_u16 는 심볼릭 링크."""
import argparse, os, sys, json, math, glob
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, yaml
if os.environ.get("CUDA_VISIBLE_DEVICES")!="1": raise SystemExit("CUDA_VISIBLE_DEVICES must be 1")
sys.path.insert(0,"/home/work/data/olmoearth/third_party/pydeps"); sys.path.insert(0,"/home/work/data/olmoearth/third_party/clay")
from claymodel.module import ClayMAEModule
_ap=argparse.ArgumentParser(); _ap.add_argument("--grid",default="up32",choices=["up32","native16","in256"]); _ap.add_argument("--temporal",default="mean",choices=["mean","last"]); _ap.add_argument("--out",default=None); _ap.add_argument("--depth-frac",type=float,default=1.0,help="keep the first frac of transformer layers (depth sensitivity)"); _a=_ap.parse_args()
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"sen12_pilot/holdout_chimanimani"; OUT=(ROOT/_a.out) if _a.out else ROOT/"clay_cache"; NC=Path("/home/work/data/sen12landslides/extracted")
CK="/home/work/data/clay/clay-v1.5.ckpt"; META="/home/work/data/olmoearth/third_party/clay/configs/metadata.yaml"
(OUT/"emb_fp16").mkdir(parents=True,exist_ok=True)
for d in ("mask_u8","raw_u16"):
    if not (OUT/d).exists(): os.symlink(SRC/d, OUT/d)
for f in ("months.jsonl",):
    if not (OUT/f).exists(): os.symlink(SRC/f, OUT/f)
meta=yaml.safe_load(open(META))["sentinel-2-l2a"]; order=meta["band_order"]
mean=torch.tensor([meta["bands"]["mean"][b] for b in order]).view(1,-1,1,1); std=torch.tensor([meta["bands"]["std"][b] for b in order]).view(1,-1,1,1)
waves=torch.tensor([meta["bands"]["wavelength"][b] for b in order])
RAW=["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12"]; CLAY=["B02","B03","B04","B05","B06","B07","B08","B8A","B11","B12"]; perm=[RAW.index(b) for b in CLAY]
dev=torch.device("cuda")
model=ClayMAEModule.load_from_checkpoint(CK, metadata_path=META, mask_ratio=0.0, shuffle=False, map_location="cpu").to(dev).eval()
enc=model.model.encoder; D=enc.dim; P=enc.patch_size; print("clay dim",D,"patch",P,flush=True)
if _a.depth_frac<1.0:
    import torch.nn as _nn; L=enc.transformer.layers; k=max(1,int(round(len(L)*_a.depth_frac))); enc.transformer.layers=_nn.ModuleList(list(L)[:k]); print("clay depth",k,"/",len(L),flush=True)
months={json.loads(l)["sample_id"]:json.loads(l)["months_0_11"] for l in open(SRC/"months.jsonl") if l.strip()}
import xarray as xr
from pyproj import Transformer
def latlon_of(sid):
    f=NC/f"{sid}.nc"
    with xr.open_dataset(f) as ds: at=ds.attrs; cx=float(at.get("center_lon")); cy=float(at.get("center_lat")); crs=str(at.get("crs"))
    lon,lat=Transformer.from_crs(crs,"EPSG:4326",always_xy=True).transform(cx,cy); return lat,lon
def enc_feats(t):
    lat,lon=t["latlon"]; la,lo=math.radians(lat),math.radians(lon)
    latlon=torch.tensor([[math.sin(la),math.cos(la),math.sin(lo),math.cos(lo)]],dtype=torch.float32)
    return latlon
ids=sorted(p.stem for p in (SRC/"raw_u16").glob("*.npy")); done=0; skipped=[]
@torch.no_grad()
def embed_tile(sid):
    x=torch.from_numpy(np.load(SRC/"raw_u16"/f"{sid}.npy").astype("float32"))[perm]   # (10,T,128,128) Clay 순서
    T=x.shape[1]; lat,lon=latlon_of(sid); la,lo=math.radians(lat),math.radians(lon)
    latlon=torch.tensor([[math.sin(la),math.cos(la),math.sin(lo),math.cos(lo)]]).repeat(T,1)
    mo=months.get(sid,[0]*T)[:T]; week=[(m*4.35+2)/52.0 for m in mo]; hour=10.5/24.0
    time=torch.tensor([[math.sin(2*math.pi*w),math.cos(2*math.pi*w),math.sin(2*math.pi*hour),math.cos(2*math.pi*hour)] for w in week],dtype=torch.float32)
    pix=((x.permute(1,0,2,3)-mean)/std).to(dev)                                       # (T,10,128,128)
    if _a.grid=="in256": pix=F.interpolate(pix,size=(256,256),mode="bilinear",align_corners=False); 

    cube={"pixels":pix,"time":time.to(dev),"latlon":latlon.to(dev),"gsd":torch.tensor(10.0,device=dev),"waves":waves.to(dev)}
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=enc(cube)[0]                                                               # (T, 1+L, D)
    tok=out[:,1:,:].float(); g=(256 if _a.grid=="in256" else 128)//P; tok=tok.reshape(T,g,g,D).permute(0,3,1,2)      # (T,D,g,g)
    if _a.grid=="up32": tok=F.interpolate(tok,size=(32,32),mode="bilinear",align_corners=False)   # v0 interpolated adapter
    tok=tok[-1] if _a.temporal=="last" else tok.mean(0)                                              # (D,g,g)
    return tok.cpu().numpy().astype("float16")
for sid in ids:
    o=OUT/"emb_fp16"/f"{sid}.npy"
    if o.exists(): done+=1; continue
    try: np.save(o, embed_tile(sid)); done+=1
    except Exception as e: skipped.append({"id":sid,"err":str(e)[:150]})
    if done%500==0: print(done,"tiles",flush=True)
fs=sorted((OUT/"emb_fp16").glob("*.npy")); a=np.load(fs[0],mmap_mode="r")
audit={"schema":"clay-cache-audit-v1","all_gates_pass":len(fs)==len(ids) and tuple(a.shape)==(D,32,32),"n_tiles":len(fs),"expected":len(ids),"shape":list(a.shape),"dtype":str(a.dtype),"model":"Clay v1.5","patch":P,"dim":D,
       "grid":_a.grid,"depth_frac":_a.depth_frac,"temporal":_a.temporal,"contract":"per-timestep encode, mean over 12 timesteps, 16x16->32x32 bilinear; band reorder to Clay S2 order; metadata mean/std; week from month index, hour 10.5; latlon from nc center+crs","skipped":skipped[:30],"n_skipped":len(skipped)}
(OUT/"cache_audit.json").write_text(json.dumps(audit,indent=1)); print(json.dumps({k:audit[k] for k in ("all_gates_pass","n_tiles","n_skipped","dim","patch")})); print("CLAY CACHE DONE")
