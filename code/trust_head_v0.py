#!/usr/bin/env python3
"""Trust head v0 (config/trust_head_prereg_v0.json): per-token untrustworthy probability from frozen single-observation tokens; LOEO; payoff on cloudy-pair z inflation."""
import json, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from datetime import datetime
ROOT=Path("/home/work/data/olmoearth"); OUT=ROOT/"trust_head_v0"; OUT.mkdir(exist_ok=True); dev=torch.device("cuda"); torch.manual_seed(1); np.random.seed(1)
REGIONS=["hiroshima","thrissur","itogon","hokkaido"]; PROBE="--probe" in sys.argv
GAPS=[("<=10",0,10),("11-20",11,20),("21-40",21,40),("41-70",41,70),("71-120",71,120),(">120",121,10**6)]
gapbin=lambda g: next(l for l,a,b in GAPS if a<=g<=b); season=lambda m: "DJF" if m in (12,1,2) else "MAM" if m in (3,4,5) else "JJA" if m in (6,7,8) else "SON"
def tiles(reg):
    D=ROOT/"arrival_v0"/reg; ids=sorted(p.stem for p in (D/"scl_token_fp16").glob("*.npy") if (D/"single_fp16"/f"{p.stem}.npy").exists()); return ids[:20] if PROBE else ids
def load(reg,s):
    D=ROOT/"arrival_v0"/reg; return np.load(D/"single_fp16"/f"{s}.npy").astype("float32"), np.load(D/"scl_token_fp16"/f"{s}.npy").astype("float32"), json.loads((D/"meta"/f"{s}.json").read_text())
class Head(nn.Module):
    def __init__(s,c=768): super().__init__(); s.f=nn.Sequential(nn.Conv2d(c,128,1),nn.ReLU(inplace=True),nn.Conv2d(128,1,1))
    def forward(s,x): return s.f(x)
def auc(pos,neg):
    pos,neg=np.asarray(pos),np.asarray(neg)
    if len(pos)==0 or len(neg)==0: return None
    allv=np.concatenate([pos,neg]); ranks=allv.argsort().argsort()+1; rp=ranks[:len(pos)].sum(); return float((rp-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
res={"schema":"trust-head-v0","folds":{}}
for held in REGIONS:
    t0=time.perf_counter(); train=[(r,s) for r in REGIONS if r!=held for s in tiles(r)]; test=[(held,s) for s in tiles(held)]
    h=Head().to(dev); opt=torch.optim.AdamW(h.parameters(),1e-3,weight_decay=1e-4); EP=1 if PROBE else 10
    for ep in range(EP):
        order=np.random.permutation(len(train)); tot=0
        for j in order:
            r,s=train[j]; U,L,_=load(r,s); X=torch.from_numpy(U).to(dev); Y=(torch.from_numpy(L)>=0.5).float().to(dev)[:,None]
            pw=((Y.numel()-Y.sum())/Y.sum().clamp(min=1)).clamp(max=20); loss=F.binary_cross_entropy_with_logits(h(X),Y,pos_weight=pw); opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss)
        print(f"[{held}] epoch {ep} loss {tot/len(order):.4f} {time.perf_counter()-t0:.0f}s",flush=True)
    h.eval(); torch.save(h.state_dict(),OUT/f"trust_heldout_{held}.pt")
    # token AUC + acquisition-level agreement + payoff on cloudy-pair z inflation
    P=[];N=[]; acq_pred=[]; acq_clear=[]; cache={}
    with torch.no_grad():
        for r,s in test:
            U,L,m=load(r,s); pr=torch.sigmoid(h(torch.from_numpy(U).to(dev))).squeeze(1).cpu().numpy(); cache[s]=(U,pr,m)
            lab=L>=0.5; sub=np.random.random(lab.shape)<0.2; P+=pr[lab&sub].tolist(); N+=pr[(~lab)&sub].tolist(); acq_pred+=pr.mean((1,2)).tolist(); acq_clear+=m["clear"]
    from scipy.stats import spearmanr
    rho=float(spearmanr(acq_pred,acq_clear).correlation)
    # gap/season table from the OTHER regions' normal consecutive clear pairs (tile-level 1-cos), then z on the held-out region's cloudy/normal pairs, with and without trust masking
    def pairs(regs,mask_fn=None):
        rows=[]
        for r in regs:
            for s in tiles(r):
                U,L,m=load(r,s) if s not in cache else (cache[s][0],None,cache[s][2]); d=[datetime.fromisoformat(x) for x in m["dates"]]; clear=m["clear"]; post=m["post_index"]
                Ut=torch.from_numpy(U).to(dev); Fm=Ut.flatten(2)
                for i in range(U.shape[0]-1):
                    crosses=(post is not None and i<post<=i+1)
                    if crosses: continue
                    kind="cloudy" if (clear[i]<0.5 or clear[i+1]<0.5) else "normal"
                    c=(1-F.cosine_similarity(Ut[i],Ut[i+1],dim=0)).cpu().numpy()  # 32x32
                    if mask_fn is not None:
                        keep=mask_fn(s,i)&mask_fn(s,i+1); ch=float(c[keep].mean()) if keep.sum()>=64 else None
                    else: ch=float(c.mean())
                    if ch is not None: rows.append({"gap":(d[i+1]-d[i]).days,"season":season(d[i+1].month),"change":ch,"kind":kind})
        return rows
    tr_rows=pairs([r for r in REGIONS if r!=held]); T={}
    for gb,_,_ in GAPS:
        for se in ("DJF","MAM","JJA","SON"):
            v=np.array([x["change"] for x in tr_rows if x["kind"]=="normal" and gapbin(x["gap"])==gb and x["season"]==se])
            if len(v)>=20: T[f"{gb}|{se}"]={"mu":float(v.mean()),"sigma":float(v.std()+1e-6)}
    def z(x):
        k=f"{gapbin(x['gap'])}|{x['season']}"; mu,sg=(T[k]["mu"],T[k]["sigma"]) if k in T else (np.mean([v["mu"] for v in T.values()]),np.mean([v["sigma"] for v in T.values()])); return (x["change"]-mu)/sg
    before=pairs([held]); after=pairs([held],mask_fn=lambda s,i: cache[s][1][i]<0.5)
    def frac(rows,kind): v=[z(x)>=2 for x in rows if x["kind"]==kind]; return (float(np.mean(v)) if v else None, len(v))
    res["folds"][held]={"n_train_tiles":len(train),"n_test_tiles":len(test),"token_auc":auc(P,N),"acq_spearman_pred_vs_clear":rho,"cloudy_z2_before":frac(before,"cloudy"),"cloudy_z2_after_mask":frac(after,"cloudy"),"normal_z2_before":frac(before,"normal"),"normal_z2_after_mask":frac(after,"normal")}
    print(held,json.dumps(res["folds"][held]),flush=True)
ok=[h for h,f in res["folds"].items() if f["token_auc"] and f["token_auc"]>=0.90]
drop=[h for h,f in res["folds"].items() if f["cloudy_z2_before"][0] and f["cloudy_z2_after_mask"][0] is not None and f["cloudy_z2_after_mask"][0]<=0.5*f["cloudy_z2_before"][0] and abs((f["normal_z2_after_mask"][0] or 0)-(f["normal_z2_before"][0] or 0))<0.02]
res["decision"]={"auc_ge_090_folds":ok,"payoff_folds":drop,"trust_usable":len(ok)>=3 and len(drop)>=3}
(OUT/"summary.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res["decision"])); print("TRUST HEAD DONE")
