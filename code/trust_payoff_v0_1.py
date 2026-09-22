#!/usr/bin/env python3
"""Trust payoff v0.1 rescoring (trust_head_prereg_v0 amendment_1): soft trust weighting and a token-level-SCL normal set; reuses the MS-147 LOEO heads. No training."""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from datetime import datetime
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"trust_head_v0"; dev=torch.device("cuda"); REGIONS=["hiroshima","thrissur","itogon","hokkaido"]
GAPS=[("<=10",0,10),("11-20",11,20),("21-40",21,40),("41-70",41,70),("71-120",71,120),(">120",121,10**6)]
gapbin=lambda g: next(l for l,a,b in GAPS if a<=g<=b); season=lambda m: "DJF" if m in (12,1,2) else "MAM" if m in (3,4,5) else "JJA" if m in (6,7,8) else "SON"
class Head(nn.Module):
    def __init__(s,c=768): super().__init__(); s.f=nn.Sequential(nn.Conv2d(c,128,1),nn.ReLU(inplace=True),nn.Conv2d(128,1,1))
    def forward(s,x): return s.f(x)
def pairs(reg,head):
    D=ROOT/"arrival_v0"/reg; rows=[]
    with torch.no_grad():
        for p in sorted((D/"single_fp16").glob("*.npy")):
            s=p.stem
            if not (D/"scl_token_fp16"/f"{s}.npy").exists(): continue
            U=torch.from_numpy(np.load(p).astype("float32")).to(dev); L=np.load(D/"scl_token_fp16"/f"{s}.npy").astype("float32"); m=json.loads((D/"meta"/f"{s}.json").read_text()); d=[datetime.fromisoformat(x) for x in m["dates"]]; clear=m["clear"]; post=m["post_index"]
            pr=torch.sigmoid(head(U)).squeeze(1).cpu().numpy()
            for i in range(U.shape[0]-1):
                if post is not None and i<post<=i+1: continue
                c=(1-F.cosine_similarity(U[i],U[i+1],dim=0)).cpu().numpy(); w=(1-pr[i])*(1-pr[i+1]); hard=(pr[i]<0.5)&(pr[i+1]<0.5)
                kind="cloudy" if (clear[i]<0.5 or clear[i+1]<0.5) else ("normal_v01" if (L[i].mean()<0.1 and L[i+1].mean()<0.1) else "other")
                rows.append({"gap":(d[i+1]-d[i]).days,"season":season(d[i+1].month),"kind":kind,"plain":float(c.mean()),"soft":float((w*c).sum()/max(w.sum(),1e-6)),"soft_cov":float(w.sum()),"hard":float(c[hard].mean()) if hard.sum()>=64 else None,"hard_cov":int(hard.sum())})
    return rows
def table(rows,key):
    t={}
    for gb,_,_ in GAPS:
        for se in ("DJF","MAM","JJA","SON"):
            v=np.array([r[key] for r in rows if r["kind"]=="normal_v01" and gapbin(r["gap"])==gb and r["season"]==se and r[key] is not None])
            if len(v)>=20: t[f"{gb}|{se}"]={"mu":float(v.mean()),"sigma":float(v.std()+1e-6)}
    mu=np.mean([x["mu"] for x in t.values()]); sg=np.mean([x["sigma"] for x in t.values()]); return t,mu,sg
def zf(r,key,T,mu0,sg0):
    k=f"{gapbin(r['gap'])}|{r['season']}"; mu,sg=(T[k]["mu"],T[k]["sigma"]) if k in T else (mu0,sg0); return (r[key]-mu)/sg
res={"schema":"trust-payoff-v0.1","folds":{}}
cache={}
for held in REGIONS:
    h=Head(); h.load_state_dict(torch.load(OUT/f"trust_heldout_{held}.pt",map_location="cpu")); h=h.to(dev).eval()
    others=[]
    for r in REGIONS:
        if r==held: continue
        others+=pairs(r,h)  # trust from the held-out fold's head applied to training regions (consistent weighting for the table)
    test=pairs(held,h)
    out={}
    for key in ("plain","soft","hard"):
        T,mu0,sg0=table(others,key); cl=[r for r in test if r["kind"]=="cloudy" and r[key] is not None]; no=[r for r in test if r["kind"]=="normal_v01" and r[key] is not None]
        out[key]={"cloudy_n":len(cl),"cloudy_z2":float(np.mean([zf(r,key,T,mu0,sg0)>=2 for r in cl])) if cl else None,"normal_n":len(no),"normal_z2":float(np.mean([zf(r,key,T,mu0,sg0)>=2 for r in no])) if no else None}
    n_cl=sum(1 for r in test if r["kind"]=="cloudy"); out["coverage"]={"cloudy_pairs":n_cl,"soft_scorable_frac":float(np.mean([r["soft_cov"]>=64 for r in test if r["kind"]=="cloudy"])) if n_cl else None,"hard_scorable_frac":float(np.mean([r["hard"] is not None for r in test if r["kind"]=="cloudy"])) if n_cl else None}
    res["folds"][held]=out; print(held,json.dumps(out),flush=True)
ok=[h for h,f in res["folds"].items() if f["soft"]["cloudy_z2"] is not None and f["soft"]["normal_z2"] and f["soft"]["cloudy_z2"]<=2*f["soft"]["normal_z2"] and f["coverage"]["soft_scorable_frac"]>=0.8]
res["v0_1_reading"]={"folds_meeting_soft_criterion":ok,"count":len(ok)}
(OUT/"payoff_v0_1.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res["v0_1_reading"])); print("PAYOFF V01 DONE")
