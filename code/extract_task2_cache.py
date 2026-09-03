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
ROOT=Path("/home/work/data/task2_solar_farm/extracted/windows/default"); OUT=Path("/home/work/data/olmoearth/task2_cache"); IDX=Path("/home/work/data/olmoearth/artifacts/task2_solar_farm/window_index.jsonl")
for d in ("raw_u16","mask_u8","emb_fp16"): (OUT/d).mkdir(parents=True,exist_ok=True)
MODEL_BANDS=["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]; RAW_BANDS=MODEL_BANDS[:10]
GROUPS={"B02_B03_B04_B08":["B02","B03","B04","B08"],"B05_B06_B07_B8A_B11_B12":["B05","B06","B07","B8A","B11","B12"],"B01_B09_B10":["B01","B09","B10"]}
TS=["sentinel2","sentinel2.1","sentinel2.2","sentinel2.3"]
dev=torch.device("cuda"); wrapper=OlmoEarth(patch_size=4, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
rows=[json.loads(l) for l in open(IDX)]; elig=[r for r in rows if r["s2_timesteps"]==4 and r["s2_bands_complete"]]
months=open(OUT/"months.jsonl","w"); done=0; skipped=[]
def read_center(path, bands_in_file, want):
    with rasterio.open(path) as ds:
        H,W=ds.height,ds.width; y0=(H-128)//2; x0=(W-128)//2
        arr=ds.read(window=((y0,y0+128),(x0,x0+128)))  # (nb,128,128)
    return {b:arr[i] for i,b in enumerate(bands_in_file) if b in want}, (y0,x0)
@torch.no_grad()
def embed(cube12, times):  # (12,4,128,128) float32 → (768,32,32)
    feat=torch.empty((768,32,32))
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)):
        img=torch.from_numpy(np.ascontiguousarray(cube12[:,:,y0:y0+64,x0:x0+64])).to(dev)
        inp={"sentinel2_l2a": RasterImage(image=img, timestamps=[(t,t) for t in times])}; wrapper.normalizer(inp,{})
        sample,_,_=wrapper._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[]))
        with torch.amp.autocast("cuda",dtype=torch.bfloat16):
            tm=wrapper.model(sample,fast_pass=False,patch_size=4)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=2).unsqueeze(-1)
            f=((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1))[0].permute(2,0,1).float().cpu()
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
                    lay=items.get(t) or items.get(t.split(".")[0]); it=(lay[0]["items"][0] if isinstance(lay,list) else None)
                    tt=datetime.fromisoformat(it["geometry"]["time_range"][0]) if it else None
                except Exception: tt=None
            if tt is None:
                tr=r.get("time_range"); a,b=[datetime.fromisoformat(x) for x in tr]; tt=a+(b-a)*(ti+0.5)/4
            times.append(tt.replace(tzinfo=None))
        lab=np.array(Image.open(w/"layers/label_raster/label/image.png")); H,W=lab.shape[:2]; y0=(H-128)//2; x0=(W-128)//2
        mask=(lab[y0:y0+128,x0:x0+128]>0).astype("uint8")
        if mask.shape!=(128,128): raise ValueError("label crop shape")
        np.save(OUT/"raw_u16"/f"{r['id']}.npy", cube[:10]); np.save(OUT/"mask_u8"/f"{r['id']}.npy", mask)
        emb=embed(cube.astype("float32"), times); np.save(OUT/"emb_fp16"/f"{r['id']}.npy", emb.numpy().astype("float16"))
        months.write(json.dumps({"sample_id":r["id"],"months_0_11":[t.month-1 for t in times]})+"\n"); done+=1
        if done%200==0: print(done,"chips",flush=True)
    except Exception as e:
        skipped.append({"id":r["id"],"err":str(e)[:120]})
months.close()
audit={"schema":"task2-cache-audit-v1","all_gates_pass":True,"n_chips":done,"n_skipped":len(skipped),"skipped":skipped[:50],"contract":"center 128px chip; raw 10 bands x4T uint16; emb 768x32x32 fp16 OlmoEarth v1 Base 12-band; mask label>0",
       "note":"all_gates_pass set after count/shape spot check below"}
ok=True
for d,shape,dt in (("raw_u16",(10,4,128,128),"uint16"),("mask_u8",(128,128),"uint8"),("emb_fp16",(768,32,32),"float16")):
    fs=sorted((OUT/d).glob("*.npy"))[:40]
    for f in fs:
        a=np.load(f,mmap_mode="r")
        if tuple(a.shape)!=shape or str(a.dtype)!=dt: ok=False; audit.setdefault("shape_errors",[]).append(f.name)
audit["all_gates_pass"]=ok and done>0
(OUT/"cache_audit.json").write_text(json.dumps(audit,indent=1)); print(json.dumps({k:audit[k] for k in ("all_gates_pass","n_chips","n_skipped")})); print("CACHE DONE")
