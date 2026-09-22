#!/usr/bin/env python3
"""G-T2 latent interpolation benchmark v0 (config/latent_interp_prereg_v0.json). Baselines only."""
import json, sys
from pathlib import Path
from datetime import datetime
import numpy as np, torch
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"olmo_streaming_dev"/"single_fp16"; OUT=ROOT/"latent_interp_v0"; OUT.mkdir(exist_ok=True)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
REC={}
for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl"):
    if l.strip(): r=json.loads(l); REC[r["sample_id"]]=r
def dates(sid):
    r=REC[sid]; q=r["scl_clear_fraction"]; idx=sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:12])
    return [datetime.fromisoformat(str(r["times"][i])[:19]) for i in idx]
ids=sorted(p.stem for p in SRC.glob("*.npy")); rng=np.random.default_rng(0); ids=sorted(rng.choice(ids,min(1500,len(ids)),replace=False).tolist())
def tokcos(a,b): return float(torch.nn.functional.cosine_similarity(a.flatten(1),b.flatten(1),dim=0).mean())
rows=[]
for n,sid in enumerate(ids):
    S=torch.from_numpy(np.load(SRC/f"{sid}.npy").astype("float32")).to(dev); d=dates(sid)
    if S.shape[0]!=12: continue
    mean_all=S.mean(0,keepdim=True)
    for j in range(12):
        tgt=S[j]; others=torch.cat([S[:j],S[j+1:]]); m_all=others.mean(0)
        denom=float(((tgt-m_all)**2).sum())+1e-9
        row={"id":sid,"region":sid.split("_s2_")[0],"j":j}
        if 1<=j<=10:
            t0,t1,tj=d[j-1],d[j+1],d[j]; gap=(t1-t0).days; w=(tj-t0).days/max(gap,1)
            preds={"previous":S[j-1],"nearest":S[j-1] if (tj-t0)<=(t1-tj) else S[j+1],"linear":(1-w)*S[j-1]+w*S[j+1],"mean_all":m_all}
            row.update({"gap_days":gap,"neighbour_cos":tokcos(S[j-1],S[j+1])})
        else:
            preds={"previous":S[j-1] if j==11 else S[j+1],"mean_all":m_all}; row["gap_days"]=None
        for k,p in preds.items(): row[f"cos_{k}"]=tokcos(p,tgt); row[f"rmse_{k}"]=float(((p-tgt)**2).sum())/denom
        rows.append(row)
    if n%300==0: print(n,"tiles",flush=True)
(OUT/"rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
def agg(sel,keys):
    o={}
    for k in keys:
        v=np.array([r[f"cos_{k}"] for r in sel if f"cos_{k}" in r]); o[k]={"n":len(v),"cos_mean":float(v.mean()) if len(v) else None,"cos_p10":float(np.quantile(v,.1)) if len(v) else None}
    return o
interior=[r for r in rows if 1<=r["j"]<=10]; keys=["nearest","previous","linear","mean_all"]
bins=[("<=20",0,20),("21-40",21,40),("41-70",41,70),("71-120",71,120),(">120",121,10**6)]
summary={"schema":"latent-interp-v0-summary","n_tiles":len(ids),"n_interior_rows":len(interior),"overall_interior":agg(interior,keys),"neighbour_cos_mean":float(np.mean([r["neighbour_cos"] for r in interior])),
 "by_gap":{lab:agg([r for r in interior if lo<=r["gap_days"]<=hi],keys) for lab,lo,hi in bins},
 "by_position":{str(j):agg([r for r in interior if r["j"]==j],keys) for j in range(1,11)},
 "by_region":{reg:agg([r for r in interior if r["region"]==reg],keys) for reg in sorted({r["region"] for r in interior})},
 "extrapolation_edges":agg([r for r in rows if r["j"] in (0,11)],["previous","mean_all"]),
 "headroom_1_minus_cos_linear":1-agg(interior,["linear"])["linear"]["cos_mean"]}
(OUT/"summary.json").write_text(json.dumps(summary,indent=1)); print(json.dumps({k:summary[k] for k in ("n_tiles","overall_interior","neighbour_cos_mean","headroom_1_minus_cos_linear")},indent=1)); print("LATENT INTERP V0 DONE")
