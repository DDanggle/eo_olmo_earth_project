#!/usr/bin/env python3
"""Task-2 캐시 추출 (GPU1) — 등록 addendum_v2 계약. 윈도우 중심 128 px 칩:
  raw_u16 (10,4,128,128) uint16 · mask_u8 (128,128) · emb_fp16 (768,32,32) frozen OlmoEarth v1 Base(12밴드: B01,B09 포함·B10 제외, 4시점) · months.jsonl.
출력: /home/work/data/olmoearth/task2_cache/{raw_u16,mask_u8,emb_fp16}/<id>.npy + cache_audit.json(all_gates_pass) — pilot 이 요구하는 형태."""
import json, os, sys, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np, rasterio
from PIL import Image
if os.environ.get("CUDA_VISIBLE_DEVICES")!="1": raise SystemExit("CUDA_VISIBLE_DEVICES must be 1")
import torch
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/task2_solar_farm/extracted/windows/default"); OUT=Path("/home/work/data/olmoearth/task2_cache_v12"); V1=Path("/home/work/data/olmoearth/task2_cache"); IDX=Path("/home/work/data/olmoearth/artifacts/task2_solar_farm/window_index.jsonl")
(OUT/"emb_fp16").mkdir(parents=True,exist_ok=True)
for d in ("raw_u16","mask_u8"):
    if not (OUT/d).exists(): (OUT/d).symlink_to(V1/d)
if not (OUT/"months.jsonl").exists(): (OUT/"months.jsonl").symlink_to(V1/"months.jsonl")
MODEL_BANDS=["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]; RAW_BANDS=MODEL_BANDS[:10]
GROUPS={"B02_B03_B04_B08":["B02","B03","B04","B08"],"B05_B06_B07_B8A_B11_B12":["B05","B06","B07","B8A","B11","B12"],"B01_B09_B10":["B01","B09","B10"]}
TS=["sentinel2","sentinel2.1","sentinel2.2","sentinel2.3"]
dev=torch.device("cuda"); wrapper=OlmoEarth(patch_size=4, model_id=ModelID.OLMOEARTH_V1_2_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
rows=[json.loads(l) for l in open(IDX)]; v1ids={p.stem for p in (V1/"emb_fp16").glob("*.npy")}; elig=[r for r in rows if r["s2_timesteps"]==4 and r["s2_bands_complete"] and r["id"] in v1ids]
done=0; skipped=[]; cos=[]
from rasterio.enums import Resampling
def read_center(path, bands_in_file, want):
    """그룹별 자기 해상도 격자(10/20/40 m)에서 10 m 기준 128 px 에 해당하는 중심 영역을 잘라 128x128 로 리샘플(bilinear)."""
    with rasterio.open(path) as ds:
        res=ds.res[0]; n=int(round(128*10.0/res))            # 10m→128, 20m→64, 40m→32
        H,W=ds.height,ds.width; y0=max(0,(H-n)//2); x0=max(0,(W-n)//2)
        if H<n or W<n: raise ValueError(f"raster {W}x{H} smaller than crop {n} at res {res}")
        arr=ds.read(window=((y0,y0+n),(x0,x0+n)), out_shape=(ds.count,128,128), resampling=Resampling.bilinear)
    return {b:arr[i] for i,b in enumerate(bands_in_file) if b in want}, (y0,x0)
@torch.no_grad()
def embed(cube12, times):  # (12,4,128,128) float32 → (768,32,32)
    feat=None
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        img=torch.from_numpy(np.ascontiguousarray(cube12[:,:,y0:y0+64,x0:x0+64])).to(dev)
        inp={"sentinel2_l2a": RasterImage(image=img, timestamps=[(t,t) for t in times])}; wrapper.normalizer(inp,{})
        sample,_,_=wrapper._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[]))
        with torch.amp.autocast("cuda",dtype=torch.bfloat16):
            tm=wrapper.model(sample,fast_pass=False,patch_size=4)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=2).unsqueeze(-1)
            f=((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1))[0].permute(2,0,1).float().cpu()
        if feat is None: feat=torch.empty((f.shape[0],32,32))
        feat[:,y0//4:(y0+64)//4,x0//4:(x0+64)//4]=f
    return feat
for r in elig:
    w=ROOT/r["id"]
    if (OUT/"emb_fp16"/f"{r['id']}.npy").exists(): done+=1; continue
    try:
        cube=np.zeros((12,4,128,128),dtype="uint16"); times=[]
        items=json.loads((w/"items.json").read_text()) if (w/"items.json").exists() else None
        for ti,t in enumerate(TS):
            got={}
            for g,bl in GROUPS.items():
                d,_=read_center(w/"layers"/t/g/"geotiff.tif", bl, set(MODEL_BANDS)); got.update(d)
            for bi,b in enumerate(MODEL_BANDS): cube[bi,ti]=np.clip(got[b],0,65535)
            # 시점: items.json 에서 해당 레이어의 첫 item 시간, 없으면 time_range 균등 분할
            tt=None
            if items:
                try:
                    ent=next(e for e in items if e.get("layer_name")=="sentinel2"); grp=ent["serialized_item_groups"][ti]
                    tt=datetime.fromisoformat(grp[0]["geometry"]["time_range"][0])
                except Exception: tt=None
            if tt is None:
                tr=r.get("time_range"); a,b=[datetime.fromisoformat(x) for x in tr]; tt=a+(b-a)*(ti+0.5)/4
            times.append(tt.replace(tzinfo=None))
        lab=np.array(Image.open(w/"layers/label_raster/label/image.png")); H,W=lab.shape[:2]; y0=(H-128)//2; x0=(W-128)//2
        mask=(lab[y0:y0+128,x0:x0+128]>0).astype("uint8")
        if mask.shape!=(128,128): raise ValueError("label crop shape")
        emb=embed(cube.astype("float32"), times); np.save(OUT/"emb_fp16"/f"{r['id']}.npy", emb.numpy().astype("float16"))
        e1=np.load(V1/"emb_fp16"/f"{r['id']}.npy").astype("float32"); e2=emb.numpy()
        if e1.shape==e2.shape: cos.append(float((e1*e2).sum(0).__truediv__(np.linalg.norm(e1,axis=0)*np.linalg.norm(e2,axis=0)+1e-6).mean()))
        done+=1
        if done%250==0: print(done,"chips",flush=True)
    except Exception as e:
        skipped.append({"id":r["id"],"err":str(e)[:120]})
audit={"schema":"task2-cache-v12-audit-v1","release":"OlmoEarth v1.2 Base (ModelID.OLMOEARTH_V1_2_BASE, .venv-master)","paired_with":"task2_cache (v1) same chip ids, same 12-band 4-timestep input","same_token_cosine_v1_v12_mean":(float(np.mean(cos)) if cos else None),"n_cos":len(cos),"all_gates_pass":True,"n_chips":done,"n_skipped":len(skipped),"skipped":skipped[:50],"contract":"center 128px chip; raw 10 bands x4T uint16; emb Cx32x32 fp16 OlmoEarth v1.2 Base 12-band; mask label>0",
       "note":"all_gates_pass set after count/shape spot check below"}
ok=True
fs0=sorted((OUT/"emb_fp16").glob("*.npy")); EMB_SHAPE=tuple(np.load(fs0[0],mmap_mode="r").shape) if fs0 else None; audit["emb_shape"]=EMB_SHAPE
for d,shape,dt in (("emb_fp16",EMB_SHAPE,"float16"),):
    fs=fs0[:40]
    for f in fs:
        a=np.load(f,mmap_mode="r")
        if tuple(a.shape)!=shape or str(a.dtype)!=dt: ok=False; audit.setdefault("shape_errors",[]).append(f.name)
audit["all_gates_pass"]=ok and done>0 and done==len(elig) and EMB_SHAPE is not None and EMB_SHAPE[1:]==(32,32)
(OUT/"cache_audit.json").write_text(json.dumps(audit,indent=1)); print(json.dumps({k:audit[k] for k in ("all_gates_pass","n_chips","n_skipped")})); print("CACHE DONE")
