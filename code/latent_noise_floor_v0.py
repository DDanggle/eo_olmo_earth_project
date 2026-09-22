#!/usr/bin/env python3
"""Repeatability ceiling of single-observation embeddings (config/latent_noise_floor_prereg_v0.json)."""
import json, numpy as np, torch
from pathlib import Path
from datetime import datetime
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"olmo_streaming_dev/single_fp16"; OUT=ROOT/"latent_noise_floor_v0"; OUT.mkdir(exist_ok=True)
REC={json.loads(l)["sample_id"]:json.loads(l) for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl") if l.strip()}
def dates(sid): r=REC[sid]; q=r["scl_clear_fraction"]; idx=sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:12]); return [datetime.fromisoformat(str(r["times"][i])[:19]) for i in idx]
ids=sorted(p.stem for p in SRC.glob("*.npy")); rng=np.random.default_rng(0); ids=sorted(rng.choice(ids,min(1500,len(ids)),replace=False).tolist())
bins=[("<=5",0,5),("6-10",6,10),("11-20",11,20),("21-40",21,40),("41-70",41,70),("71-120",71,120),(">120",121,10**6)]
rows=[]
for sid in ids:
    S=torch.from_numpy(np.load(SRC/f"{sid}.npy").astype("float32")).cuda(); d=dates(sid)
    if S.shape[0]!=12: continue
    for j in range(11):
        c=float(torch.nn.functional.cosine_similarity(S[j].flatten(1),S[j+1].flatten(1),dim=0).mean()); g=(d[j+1]-d[j]).days
        rows.append({"id":sid,"region":sid.split("_s2_")[0],"gap":g,"same_month":d[j].month==d[j+1].month and d[j].year==d[j+1].year,"cos":c})
def agg(sel): v=np.array([r["cos"] for r in sel]); return {"n":len(v),"cos_mean":float(v.mean()) if len(v) else None,"cos_p10":float(np.quantile(v,.1)) if len(v) else None,"cos_p90":float(np.quantile(v,.9)) if len(v) else None}
res={"schema":"latent-noise-floor-v0","n_tiles":len(ids),"n_pairs":len(rows),"by_gap":{lab:agg([r for r in rows if lo<=r["gap"]<=hi]) for lab,lo,hi in bins},"same_month":agg([r for r in rows if r["same_month"]]),
     "ceiling_gap_le10":agg([r for r in rows if r["gap"]<=10]),"by_region_gap_le10":{g:agg([r for r in rows if r["region"]==g and r["gap"]<=10]) for g in sorted({r["region"] for r in rows})}}
ceil=res["ceiling_gap_le10"]["cos_mean"]; lin=0.8681; model=0.8495; lin_test=0.8205  # MS-126 linear (dev sample); MS-130 resmlp mean and its linear on test regions
res["normalised_gain"]={"ceiling":ceil,"resmlp_test_vs_linear_test":(model-lin_test)/(ceil-lin_test) if ceil and ceil>lin_test else None,"note":"gain/(ceiling-linear); ceiling from dev sample, model/linear from test regions (MS-130) — descriptive"}
(OUT/"summary.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res,indent=1)); print("NOISE FLOOR DONE")
