#!/usr/bin/env python3
"""Seasonal baseline v0 (config/seasonal_baseline_prereg_v0.json): in-region gap x season / gap x month tables fitted on tile-half A, normal z>=2 and event token AUC evaluated on half B; LOEO table as reference."""
import json, sys, hashlib, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from datetime import datetime
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"seasonal_baseline_v0_1"; OUT.mkdir(exist_ok=True); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); rng=np.random.default_rng(0)
REGIONS=["hiroshima","thrissur","itogon","hokkaido"]
GAPS=[("<=10",0,10),("11-20",11,20),("21-40",21,40),("41-70",41,70),("71-120",71,120),(">120",121,10**6)]
gapbin=lambda g: next(l for l,a,b in GAPS if a<=g<=b); season=lambda m: "DJF" if m in (12,1,2) else "MAM" if m in (3,4,5) else "JJA" if m in (6,7,8) else "SON"
half=lambda s: 1-int(hashlib.sha256(s.encode()).hexdigest(),16)%2  # v0.1: swapped — odd = A(fit)=0, even = B(eval)=1
def auc(pos,neg):
    if len(pos)==0 or len(neg)==0: return None
    allv=np.concatenate([pos,neg]); ranks=allv.argsort().argsort()+1; rp=ranks[:len(pos)].sum(); return float((rp-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
normal=[]; events=[]  # normal: dict(region,half,gap,season,month,c); events: dict(region,half,gap,season,month,change,tokmask)
for reg in REGIONS:
    D=ROOT/"arrival_v0"/reg
    for p in sorted((D/"single_fp16").glob("*.npy")):
        s=p.stem; m=json.loads((D/"meta"/f"{s}.json").read_text()); d=[datetime.fromisoformat(x) for x in m["dates"]]; clear=m["clear"]; post=m["post_index"]; h=half(s)
        mp=ROOT/f"sen12_pilot/holdout_{reg}/mask_u8"/f"{s}.npy"; mask=(np.load(mp)>0) if mp.exists() else None; positive=bool(mask.any()) if mask is not None else False
        S=torch.from_numpy(np.load(p).astype("float32")).to(dev); cl=[i for i in range(S.shape[0]) if clear[i]>=0.5]
        ch=lambda i,j:(1-F.cosine_similarity(S[i],S[j],dim=0)).cpu().numpy()
        for a_,b_ in zip(cl[:-1],cl[1:]):
            if post is None or (a_<post and b_<post) or (a_>=post and b_>=post):
                c=ch(a_,b_); normal.append(dict(region=reg,half=h,gap=gapbin((d[b_]-d[a_]).days),season=season(d[b_].month),month=d[b_].month,c=c[rng.random((32,32))<0.25]))
        if post is not None and positive and mask is not None:
            pre=[i for i in cl if i<post]; pst=[i for i in cl if i>=post]
            if pre and pst:
                i,j=pre[-1],pst[0]; events.append(dict(region=reg,half=h,gap=gapbin((d[j]-d[i]).days),season=season(d[j].month),month=d[j].month,change=ch(i,j),tokmask=mask.reshape(32,4,32,4).mean(axis=(1,3))>=0.25))
def fit(rows,axis):
    acc={}
    for r in rows: acc.setdefault((r["gap"],r[axis]),[]).append(r["c"])
    return {k:(float(np.concatenate(v).mean()),float(np.concatenate(v).std()+1e-6)) for k,v in acc.items() if sum(len(x) for x in v)>=2000}
class Table:
    def __init__(s,rows,axis):
        s.axis=axis; s.T=fit(rows,axis); s.Ts=fit(rows,"season") if axis=="month" else s.T; s.mu=np.mean([v[0] for v in s.Ts.values()]); s.sg=np.mean([v[1] for v in s.Ts.values()])
    def z(s,r,x):
        k=(r["gap"],r[s.axis]); ks=(r["gap"],r["season"])
        mu,sg=s.T.get(k) or s.Ts.get(ks) or (s.mu,s.sg); return (x-mu)/sg
def evaluate(T,reg):
    N=[T.z(r,r["c"]) for r in normal if r["region"]==reg and r["half"]==1]; Nz=np.concatenate(N) if N else np.zeros(0)
    pos=[];neg=[]
    for e in events:
        if e["region"]!=reg or e["half"]!=1: continue
        z=T.z(e,e["change"]); pos.append(z[e["tokmask"]]); neg.append(z[~e["tokmask"]])
    P=np.concatenate(pos) if pos else np.zeros(0); Ng=np.concatenate(neg) if neg else np.zeros(0)
    return {"normal_pairs":len(N),"normal_z2":float((Nz>=2).mean()) if len(Nz) else None,"normal_z_mean":float(Nz.mean()) if len(Nz) else None,"n_events":len(pos),"event_auc":auc(P,Ng),"event_tpr_z2":float((P>=2).mean()) if len(P) else None,"event_fpr_z2":float((Ng>=2).mean()) if len(Ng) else None,"cells":len(T.T)}
res={"schema":"seasonal-baseline-v0.1","regions":{}}
for reg in REGIONS:
    loeo=Table([r for r in normal if r["region"]!=reg],"season"); A=[r for r in normal if r["region"]==reg and r["half"]==0]
    res["regions"][reg]={"loeo":evaluate(loeo,reg),"inregion_season":evaluate(Table(A,"season"),reg),"inregion_month":evaluate(Table(A,"month"),reg),"fit_half_pairs":len(A)}
    print(reg,json.dumps(res["regions"][reg]),flush=True)
h=res["regions"]["hokkaido"]; ok=lambda a: a["normal_z2"] is not None and a["normal_z2"]<=.10 and a["event_auc"] is not None and a["event_auc"]>=(h["loeo"]["event_auc"] or 0)-.03
res["gate"]={"hokkaido_pass":bool(ok(h["inregion_season"])),"controls_ok":all((res["regions"][r]["inregion_season"]["normal_z2"] or 0)<=(res["regions"][r]["loeo"]["normal_z2"] or 0)+.02 for r in REGIONS if r!="hokkaido")}
(OUT/"summary.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res["gate"])); print("SEASONAL DONE")
