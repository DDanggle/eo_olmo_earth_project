#!/usr/bin/env python3
"""Precompute LOEO-consistent per-token signal maps for the trust-aware reader (trust_aware_reader_prereg_v0): for region R, token z (change vs previous CLEAR acquisition, gap/season table fitted on the other three regions' normal clear-to-clear spans, tokens subsampled) and token trust p(untrusted) from trust_heldout_R.pt. Writes arrival_v0/R/signal_fp16/<tile>.npy with shape (15,2,32,32): [:,0]=z (clipped to [-5,10]; step 0 or no previous clear -> 0), [:,1]=p_untrusted."""
import json, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from datetime import datetime
ROOT=Path("/home/work/data/olmoearth"); dev=torch.device("cuda"); REGIONS=["hiroshima","thrissur","itogon","hokkaido"]; TARGETS=sys.argv[1].split(",") if len(sys.argv)>1 else REGIONS
GAPS=[("<=10",0,10),("11-20",11,20),("21-40",21,40),("41-70",41,70),("71-120",71,120),(">120",121,10**6)]
gapbin=lambda g: next(l for l,a,b in GAPS if a<=g<=b); season=lambda m: "DJF" if m in (12,1,2) else "MAM" if m in (3,4,5) else "JJA" if m in (6,7,8) else "SON"
class Head(nn.Module):
    def __init__(s,c=768): super().__init__(); s.f=nn.Sequential(nn.Conv2d(c,128,1),nn.ReLU(inplace=True),nn.Conv2d(128,1,1))
    def forward(s,x): return s.f(x)
rng=np.random.default_rng(0)
def table_for(exclude):
    acc={}
    for reg in REGIONS:
        if reg==exclude: continue
        D=ROOT/"arrival_v0"/reg
        for p in sorted((D/"single_fp16").glob("*.npy")):
            s=p.stem; m=json.loads((D/"meta"/f"{s}.json").read_text()); d=[datetime.fromisoformat(x) for x in m["dates"]]; clear=m["clear"]; post=m["post_index"]
            S=torch.from_numpy(np.load(p).astype("float32")).to(dev); cl=[i for i in range(S.shape[0]) if clear[i]>=0.5]
            for a_,b_ in zip(cl[:-1],cl[1:]):
                if post is None or (a_<post and b_<post) or (a_>=post and b_>=post):
                    c=(1-F.cosine_similarity(S[a_],S[b_],dim=0)).cpu().numpy(); acc.setdefault(f"{gapbin((d[b_]-d[a_]).days)}|{season(d[b_].month)}",[]).append(c[rng.random((32,32))<0.15])
    T={k:{"mu":float(np.concatenate(v).mean()),"sigma":float(np.concatenate(v).std()+1e-6)} for k,v in acc.items() if sum(len(x) for x in v)>=2000}
    mu=np.mean([x["mu"] for x in T.values()]); sg=np.mean([x["sigma"] for x in T.values()]); return T,mu,sg
for R in TARGETS:
    hp=ROOT/"trust_head_v0"/f"trust_heldout_{R}.pt"
    if not hp.exists(): print(R,"trust head not yet"); continue
    T,mu0,sg0=table_for(R); h=Head(); h.load_state_dict(torch.load(hp,map_location="cpu")); h=h.to(dev).eval()
    D=ROOT/"arrival_v0"/R; (D/"signal_fp16").mkdir(exist_ok=True); n=0
    with torch.no_grad():
        for p in sorted((D/"single_fp16").glob("*.npy")):
            s=p.stem; m=json.loads((D/"meta"/f"{s}.json").read_text()); d=[datetime.fromisoformat(x) for x in m["dates"]]; clear=m["clear"]
            S=torch.from_numpy(np.load(p).astype("float32")).to(dev); tr=torch.sigmoid(h(S)).squeeze(1).cpu().numpy(); out=np.zeros((S.shape[0],2,32,32),dtype="float32"); out[:,1]=tr
            prev=None
            for i in range(S.shape[0]):
                if prev is not None:
                    k=f"{gapbin((d[i]-d[prev]).days)}|{season(d[i].month)}"; mu,sg=(T[k]["mu"],T[k]["sigma"]) if k in T else (mu0,sg0)
                    z=((1-F.cosine_similarity(S[i],S[prev],dim=0)).cpu().numpy()-mu)/sg; out[i,0]=np.clip(z,-5,10)
                if clear[i]>=0.5: prev=i
            np.save(D/"signal_fp16"/f"{s}.npy",out.astype("float16")); n+=1
    print(R,"signal maps",n,"table cells",len(T),flush=True)
print("SIGNAL MAPS DONE")
