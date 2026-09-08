#!/usr/bin/env python3
"""Export KuroSiwo tortilla to plain npy (geobench venv; tacoreader/rasterio not in .venv-master). Per tile: raw_f32/<id>.npy (2,3,224,224) linear [vv,vh] x [pre_1,pre_2,post], mask_raw_u8, valid_u8; meta.jsonl."""
import json, numpy as np, tacoreader, rasterio
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"kurosiwo_npy"; [ (OUT/d).mkdir(parents=True,exist_ok=True) for d in ("raw_f32","mask_raw_u8","valid_u8")]
t=tacoreader.load(str(ROOT/"geobench2/kurosiwo/kurosiwo/geobench_kuro_siwo.tortilla"))
def read(s,k):
    with rasterio.open(s.read(k)) as r: return r.read()
with open(OUT/"meta.jsonl","w") as f:
    for i in range(len(t)):
        r=t.iloc[i]; sid=f"ks_{int(r['tortilla:id']):05d}"; p=OUT/"raw_f32"/f"{sid}.npy"
        if not p.exists():
            s=t.read(i); cube=np.stack([read(s,k) for k in (0,1,2)],1).astype("float32"); np.save(p,cube); np.save(OUT/"mask_raw_u8"/f"{sid}.npy",read(s,4)[0].astype("uint8")); np.save(OUT/"valid_u8"/f"{sid}.npy",(read(s,5)[0]==1).astype("uint8"))
        f.write(json.dumps({"id":sid,"split":r["tortilla:data_split"],"actid":int(r["actid"]),"aoiid":str(r["aoiid"]),"flood_date":str(r["flood_date"])[:10],"pflood":float(r["pflood"]),"pwater":float(r["pwater"]),"centroid":str(r["stac:centroid"])})+"\n")
        if i%1000==0: print(i,flush=True)
print("EXPORT DONE")
