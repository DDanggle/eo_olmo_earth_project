#!/usr/bin/env python3
"""P1 data prep (config/p1_arrival_extract_prereg_v0.json): all 15 acquisitions in arrival order for hiroshima, singles + causal prefix states."""
import argparse, os, sys, json, time
from pathlib import Path
from datetime import datetime
import numpy as np, torch, xarray as xr
ap=argparse.ArgumentParser(); ap.add_argument("--out",default="arrival_v0/hiroshima"); ap.add_argument("--probe",action="store_true"); ap.add_argument("--data-root",default="/home/work/data/sen12landslides/extracted"); a=ap.parse_args()
from olmoearth_pretrain_minimal import ModelID
from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
from rslearn.train.model_context import ModelContext, RasterImage
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/a.out; dev=torch.device("cuda"); PATCH=4
MODEL_BANDS=["B02","B03","B04","B08","B05","B06","B07","B8A","B11","B12","B01","B09"]
REC={json.loads(l)["sample_id"]:json.loads(l) for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl") if l.strip()}
ids=sorted(s for s,r in REC.items() if r["region"]=="hiroshima" and not r.get("error"))
for d in ("single_fp16","prefix_fp16","meta"): (OUT/d).mkdir(parents=True,exist_ok=True)
w=OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False, normalize=True, autocast_dtype="bfloat16").to(dev).eval()
def embed_crop(crop,ts):
    image=torch.from_numpy(crop).to(dev); inp={"sentinel2_l2a":RasterImage(image=image,timestamps=[(t,t) for t in ts])}; w.normalizer(inp,{})
    sample,present,_=w._prepare_modality_inputs(ModelContext(inputs=[inp],metadatas=[])); sample.sentinel2_l2a_mask[...,2]=MaskValue.MISSING.value
    with torch.autocast("cuda",dtype=torch.bfloat16):
        tm=w.model(sample,fast_pass=False,patch_size=PATCH)["tokens_and_masks"]; m=(tm.sentinel2_l2a_mask!=MaskValue.MISSING.value).unsqueeze(-1)
        return ((tm.sentinel2_l2a*m).sum(dim=(3,4))/m.sum(dim=(3,4)).clamp(min=1))[0].permute(2,0,1).float().cpu()
def window(cube,ts,a,b):
    feat=torch.empty((768,32,32))
    for y0,x0 in ((0,0),(0,64),(64,0),(64,64)): feat[:,y0//PATCH:(y0+64)//PATCH,x0//PATCH:(x0+64)//PATCH]=embed_crop(np.ascontiguousarray(cube[:,a:b,y0:y0+64,x0:x0+64]),ts[a:b])
    return feat
t0=time.perf_counter(); done=0; skipped=[]
with torch.no_grad():
    for sid in (ids[:2] if a.probe else ids):
        o=OUT/"single_fp16"/f"{sid}.npy"
        if o.exists() and (OUT/"prefix_fp16"/f"{sid}.npy").exists(): done+=1; continue
        try:
            r=REC[sid]
            with xr.open_dataset(Path(a.data_root)/r["file"],decode_times=True,cache=False) as ds:
                cube=np.stack([np.asarray(ds[b].values,dtype="float32") if b in ds else np.zeros((15,128,128),dtype="float32") for b in MODEL_BANDS],0)
                ts=[datetime.fromisoformat(str(np.datetime_as_string(t,unit="s"))) for t in np.asarray(ds["time"].values)]
            T=cube.shape[1]; single=torch.stack([window(cube,ts,i,i+1) for i in range(T)]); prefix=torch.stack([window(cube,ts,max(0,c-12),c) for c in range(1,T+1)])
            np.save(o,single.numpy().astype("float16"),allow_pickle=False); np.save(OUT/"prefix_fp16"/f"{sid}.npy",prefix.numpy().astype("float16"),allow_pickle=False)
            (OUT/"meta"/f"{sid}.json").write_text(json.dumps({"dates":[t.isoformat() for t in ts],"clear":[float(v) for v in r["scl_clear_fraction"]],"post_index":r.get("post_index"),"event_date":r.get("event_date"),"n_obs":T}))
            done+=1
        except Exception as ex:
            skipped.append({"id":sid,"err":str(ex)[:160]})
            if len(skipped)<3: import traceback; traceback.print_exc()
        if done%50==0 and done: print(done,"tiles",f"{time.perf_counter()-t0:.0f}s",flush=True)
if a.probe: print("probe ok",done); sys.exit(0)
(OUT/"extract_audit.json").write_text(json.dumps({"schema":"arrival-extract-v0","n_ids":len(ids),"n_done":done,"n_skipped":len(skipped),"skipped":skipped[:20],"elapsed_s":time.perf_counter()-t0,"all_gates_pass":done==len(ids) and not skipped},indent=1)); print("ARRIVAL EXTRACT DONE")
