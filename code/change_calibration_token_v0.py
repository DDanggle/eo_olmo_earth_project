#!/usr/bin/env python3
"""Token-level change calibration v0.2 (config/change_calibration_prereg_v0.json amendment_2)."""
import json, sys, numpy as np, torch
from pathlib import Path
from datetime import datetime
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"change_calibration_v0"; SIZE=sys.argv[1] if len(sys.argv)>1 else "base"; REGIONS=sys.argv[2].split(",") if len(sys.argv)>2 else ["hiroshima","thrissur","itogon","hokkaido"]
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); rng=np.random.default_rng(0)
GAPS=[("<=10",0,10),("11-20",11,20),("21-40",21,40),("41-70",41,70),("71-120",71,120),(">120",121,10**6)]
gapbin=lambda g: next(l for l,a,b in GAPS if a<=g<=b); season=lambda m: "DJF" if m in (12,1,2) else "MAM" if m in (3,4,5) else "JJA" if m in (6,7,8) else "SON"
def auc(pos,neg):
    pos,neg=np.asarray(pos),np.asarray(neg)
    if len(pos)==0 or len(neg)==0: return None
    allv=np.concatenate([pos,neg]); ranks=allv.argsort().argsort()+1; rp=ranks[:len(pos)].sum(); return float((rp-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
normal={}; events=[]  # normal: (region,pre_flag) -> list of (gapbin,season,changes-array)
for reg in REGIONS:
    D=ROOT/"arrival_v0"/reg if SIZE=="base" else ROOT/f"arrival_v0_{SIZE}"/reg
    for p in sorted((D/"single_fp16").glob("*.npy")):
        s=p.stem; m=json.loads((ROOT/"arrival_v0"/reg/"meta"/f"{s}.json").read_text()); d=[datetime.fromisoformat(x) for x in m["dates"]]; clear=m["clear"]; post=m["post_index"]
        mp=ROOT/f"sen12_pilot/holdout_{reg}/mask_u8"/f"{s}.npy"; mask=(np.load(mp)>0) if mp.exists() else None; positive=bool(mask.any()) if mask is not None else False
        S=torch.from_numpy(np.load(p).astype("float32")).to(dev); cl=[i for i in range(S.shape[0]) if clear[i]>=0.5]
        def ch(i,j): return (1-torch.nn.functional.cosine_similarity(S[i],S[j],dim=0)).cpu().numpy()  # 32x32
        for a_,b_ in zip(cl[:-1],cl[1:]):
            if post is None or (a_<post and b_<post) or (a_>=post and b_>=post):
                c=ch(a_,b_); pre=bool(post is None or b_<post); normal.setdefault((reg,pre),[]).append((gapbin((d[b_]-d[a_]).days),season(d[b_].month),c[rng.random((32,32))<0.25]))
        if post is not None and positive and mask is not None:
            pre=[i for i in cl if i<post]; pst=[i for i in cl if i>=post]
            if pre and pst:
                i,j=pre[-1],pst[0]; tokmask=mask.reshape(32,4,32,4).mean(axis=(1,3))>=0.25; events.append({"region":reg,"tile":s,"gapbin":gapbin((d[j]-d[i]).days),"season":season(d[j].month),"change":ch(i,j),"tokmask":tokmask})
def table(keys):
    acc={}
    for k in keys:
        for gb,se,c in normal.get(k,[]): acc.setdefault(f"{gb}|{se}",[]).append(c)
    return {k:{"mu":float(np.concatenate(v).mean()),"sigma":float(np.concatenate(v).std()+1e-6),"n":int(sum(len(x) for x in v))} for k,v in acc.items() if sum(len(x) for x in v)>=2000}
def zmap(e,T):
    k=f"{e['gapbin']}|{e['season']}"
    if k not in T: mu=np.mean([x["mu"] for x in T.values()]); sg=np.mean([x["sigma"] for x in T.values()])
    else: mu,sg=T[k]["mu"],T[k]["sigma"]
    return (e["change"]-mu)/sg
def evaluate(E,T,factor=1.0):
    pos=[];neg=[];aucs=[]
    for e in E:
        z=zmap(e,T)/factor; pm=z[e["tokmask"]]; nm=z[~e["tokmask"]]
        if len(pm) and len(nm): pos.append(pm); neg.append(nm); aucs.append(auc(pm,nm))
    if not pos: return None
    P=np.concatenate(pos); N=np.concatenate(neg); return {"n_events":len(aucs),"auc_per_event_median":float(np.median(aucs)),"auc_pooled":auc(P,N),"tpr_z2":float((P>=2).mean()),"fpr_z2":float((N>=2).mean()),"mask_tokens":int(len(P)),"nonmask_tokens":int(len(N))}
Tall=table(list(normal.keys())); res={"schema":"change-calibration-token-v0.2","size":SIZE,"n_event_pairs":len(events),"table_cells":len(Tall),"pooled":evaluate(events,Tall),"loeo":{}}
def spread(keys): v=[c for k in keys for _,_,c in normal.get(k,[])]; return float(np.concatenate(v).std()) if v else None
sp_pool=spread([k for k in normal if k[1]])
for held in REGIONS:
    Tt=table([k for k in normal if k[0]!=held]); sr=spread([(held,True)]); f=(sr/sp_pool) if (sr and sp_pool) else 1.0
    E=[e for e in events if e["region"]==held]; res["loeo"][held]={"n_events":len(E),"factor":round(f,3),"plain":evaluate(E,Tt),"with_factor":evaluate(E,Tt,f)}
(OUT/f"summary_token_{SIZE}.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res,indent=1)); print("TOKEN CALIBRATION DONE")
