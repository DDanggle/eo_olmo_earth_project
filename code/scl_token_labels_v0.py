#!/usr/bin/env python3
"""Per-token 'do not trust' labels from Sentinel-2 SCL for the trust head: fraction of untrustworthy pixels {0 nodata,1 saturated,3 cloud shadow,8 cloud medium,9 cloud high,10 thin cirrus} per 4x4-pixel token, per acquisition. Writes arrival_v0/<region>/scl_token_fp16/<tile>.npy (15,32,32)."""
import json, sys, numpy as np, xarray as xr
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); DATA=Path("/home/work/data/sen12landslides/extracted"); REGIONS=sys.argv[1].split(",") if len(sys.argv)>1 else ["hiroshima","thrissur","itogon","hokkaido"]
REC={json.loads(l)["sample_id"]:json.loads(l) for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl") if l.strip()}
BAD={0,1,3,8,9,10}
for reg in REGIONS:
    D=ROOT/"arrival_v0"/reg; (D/"scl_token_fp16").mkdir(exist_ok=True); n=0; miss=0
    for p in sorted((D/"meta").glob("*.json")):
        s=p.stem; o=D/"scl_token_fp16"/f"{s}.npy"
        if o.exists(): n+=1; continue
        try:
            with xr.open_dataset(DATA/REC[s]["file"],decode_times=True,cache=False) as ds: scl=np.asarray(ds["SCL"].values).astype("int16")  # T,128,128
            bad=np.isin(scl,list(BAD)).astype("float32"); tok=bad.reshape(bad.shape[0],32,4,32,4).mean(axis=(2,4)); np.save(o,tok.astype("float16")); n+=1
        except Exception as e: miss+=1; print("skip",s,str(e)[:80])
    print(reg,"tiles",n,"skipped",miss,flush=True)
print("SCL TOKEN LABELS DONE")
