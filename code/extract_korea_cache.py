#!/usr/bin/env python3
"""Korea (AI-Hub 71363) OlmoEarth cache: one embedding per 128-px chip from the tile's full date series (real acquisition dates), primary 10-band view
(B01/B09 zeroed and marked MISSING, as in Sen12) with the sealed crop/pool contract (4x64 px crops, patch 4, token_pooling) -> emb_fp16/<chip_id>.npy (768,32,32).
Also writes single-acquisition embeddings per date -> single_fp16/<chip_id>.npy (T,768,32,32) when --singles is set (for later streaming arms).
Reads only cube arrays + chip manifest; no label file is touched."""
import argparse, json, os, sys, time, numpy as np, torch
from pathlib import Path
from datetime import datetime
ap=argparse.ArgumentParser(); ap.add_argument("--out",default="korea_cache_v1"); ap.add_argument("--singles",action="store_true"); ap.add_argument("--probe",action="store_true"); ap.add_argument("--split",default="all"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/olmoearth"); A=ROOT/"aihub/s2_12band_v2/arrays"; OUT=ROOT/a.out; dev=torch.device("cuda"); PATCH=4
(OUT/"emb_fp16").mkdir(parents=True,exist_ok=True); (OUT/"single_fp16").mkdir(exist_ok=True)
chips=[json.loads(l) for l in (ROOT/"aihub/korea_chip_manifest.jsonl").read_text().splitlines() if l.strip()]
if a.split!="all": chips=[c for c in chips if c["split"]==a.split]
w=OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
# manifest band order == OlmoEarth bandset order: B02,B03,B04,B08 | B05,B06,B07,B8A,B11,B12 | B01,B09
def pooled_batch(crops,ts):
    inputs=[]
    for c in crops:
        image=torch.from_numpy(c).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts])}; w.normalizer(inp,{}); inputs.append(inp)
    sample,present,_=w._prepare_modality_inputs(ModelContext(inputs=inputs,metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        return ((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1)).permute(0,3,1,2).float().cpu()   # (B,768,16,16)
def chip_embed(cube,ts):   # cube (12,T,128,128) float32 -> (768,32,32); 4 crops in one call
    crops=[np.ascontiguousarray(cube[:,:,y:y+64,x:x+64]) for y in (0,64) for x in (0,64)]; p=pooled_batch(crops,ts); f=torch.empty((768,32,32)); k=0
    for y in (0,64):
        for x in (0,64): f[:,y//4:(y+64)//4,x//4:(x+64)//4]=p[k]; k+=1
    return f
tiles={}
for c in chips: tiles.setdefault(c["tile_id"],[]).append(c)
done=0; skipped=[]; t0=time.perf_counter()
for tile,cs in sorted(tiles.items()):
    todo=[c for c in cs if not (OUT/"emb_fp16"/f"{c['chip_id']}.npy").exists() or (a.singles and not (OUT/"single_fp16"/f"{c['chip_id']}.npy").exists())]
    if not todo: done+=len(cs); continue
    keys=cs[0]["keys"]; ts=[datetime.strptime(k.split("_")[1],"%Y%m%d") for k in keys]
    try:
        arr=np.stack([np.load(A/f"{k}.npy").astype("float32") for k in keys],1)   # (12,T,1024,1024)
        arr[10:12]=0.0                                                              # primary 10-band view: B01/B09 absent
        for c in todo:
            cube=arr[:,:,c["y0"]:c["y0"]+128,c["x0"]:c["x0"]+128]
            e=chip_embed(cube,ts); np.save(OUT/"emb_fp16"/f"{c['chip_id']}.npy",e.numpy().astype("float16"))
            if a.singles: np.save(OUT/"single_fp16"/f"{c['chip_id']}.npy",torch.stack([chip_embed(cube[:,t:t+1],ts[t:t+1]) for t in range(cube.shape[1])]).numpy().astype("float16"))
            done+=1
            if a.probe: print("probe",tile,c["chip_id"],"T",cube.shape[1],e.shape,float(e.abs().max()),f"{time.perf_counter()-t0:.1f}s"); sys.exit(0)
    except Exception as ex:
        skipped.append({"tile":tile,"err":str(ex)[:160]})
        if len(skipped)<3: import traceback; traceback.print_exc()
    if done%1024==0 and done: print(done,"chips",f"{time.perf_counter()-t0:.0f}s",flush=True)
n_valid=sum(1 for c in chips if (OUT/"emb_fp16"/f"{c['chip_id']}.npy").exists())
rep={"schema":"korea-olmo-cache-audit-v1","n_chips":len(chips),"n_valid":n_valid,"n_skipped_tiles":len(skipped),"skipped":skipped[:20],"view":"10-band primary (B01/B09 zero+MISSING)","dates":"real acquisition dates from cube keys","all_gates_pass":n_valid==len(chips) and not skipped,"elapsed_s":time.perf_counter()-t0}
(OUT/"korea_cache_audit.json").write_text(json.dumps(rep,indent=1)); print(json.dumps({k:v for k,v in rep.items() if k!="skipped"})); print("KOREA CACHE DONE")
