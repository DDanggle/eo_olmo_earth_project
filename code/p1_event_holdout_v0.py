#!/usr/bin/env python3
"""P1 event-level holdout v0 (config/p1_event_holdout_prereg_v0.json): apply hiroshima-trained v0.3 updaters and matched heads to unseen events; thresholds from hiroshima validation (v0.3 summary). Evaluation only."""
import argparse, json, sys
from pathlib import Path
from datetime import datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, ROOT
ap=argparse.ArgumentParser(); ap.add_argument("--src",default="p1_causal_v0_3"); ap.add_argument("--out",default="p1_event_holdout_v0"); ap.add_argument("--events",default="thrissur,itogon,hokkaido"); ap.add_argument("--probe",action="store_true"); a=ap.parse_args()
SRC=ROOT/a.src; OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda"); W=slice(8,120)
summ=json.loads((SRC/"summary_seed1.json").read_text()); THR={arm:{fb:summ["arms"][arm][fb]["thr_area_px"] for fb in ("far05","far10")} for arm in summ["arms"] if arm!="sealed12"}
# hiroshima normalisation stats (as used in v0.3) and frozen readout
mean,sd=emb_stats_from_cache(ROOT/"sen12_pilot/holdout_hiroshima","holdout_hiroshima"); mean,sd=mean.to(dev),sd.to(dev)
ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control/holdout_hiroshima_seed1_best.pt",map_location="cpu"); dec=EmbDecoder(ck["cin"]).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval()
# module classes re-declared below, identical to p1_causal_update_v0_3.py (checkpoint key-compatible)
class EMA(nn.Module):
    def __init__(s): super().__init__(); s.a=nn.Parameter(torch.tensor(-1.0))
    def forward(s,m,u,dt,q): a=torch.sigmoid(s.a); return (1-a)*m+a*u
class GRU(nn.Module):
    def __init__(s,c=768,extra=0): super().__init__(); s.zr=nn.Conv2d(2*c+extra,2*c,1); s.h=nn.Conv2d(2*c+extra,c,1); s.extra=extra
    def forward(s,m,u,dt,q,x=None):
        inp=torch.cat([m,u]+([x] if x is not None else []),1); z,r=torch.sigmoid(s.zr(inp)).chunk(2,1); n=torch.tanh(s.h(torch.cat([r*m,u]+([x] if x is not None else []),1))); return (1-z)*m+z*n
class GRUdt(GRU):
    def __init__(s,c=768): super().__init__(c,16); s.emb=nn.Linear(1,16)
    def forward(s,m,u,dt,q): x=s.emb(dt.view(-1,1)).view(-1,16,1,1).expand(-1,16,m.shape[2],m.shape[3]); return super().forward(m,u,dt,q,x)
class ResidQ(nn.Module):
    def __init__(s,c=768,h=768): super().__init__(); s.P=nn.Conv2d(c,c,1); s.f=nn.Sequential(nn.Conv2d(3*c,h,1),nn.ReLU(inplace=True),nn.Conv2d(h,c,1)); s.g=nn.Conv2d(3*c+2,c,1)
    def forward(s,m,u,dt,q):
        d=u-s.P(m); x=torch.cat([m,u,d],1); cond=torch.stack([q,dt],1).view(-1,2,1,1).expand(-1,2,m.shape[2],m.shape[3]); return m+torch.sigmoid(s.g(torch.cat([x,cond],1)))*s.f(x)
class Head(nn.Module):
    def __init__(s,c=768): super().__init__(); s.f=nn.Sequential(nn.Conv2d(c,128,1),nn.ReLU(inplace=True),nn.Conv2d(128,1,1))
    def forward(s,x): return F.interpolate(s.f(x),size=(128,128),mode="bilinear",align_corners=False)
def load_sd(m,name): m.load_state_dict(torch.load(SRC/name,map_location="cpu")); return m.to(dev).eval()
UPD={"gru_dt":load_sd(GRUdt(),"gru_dt_seed1.pt"),"resid_q":load_sd(ResidQ(),"resid_q_seed1.pt")}
HEADS={k:load_sd(Head(),f"head_{k}_seed1.pt") for k in ("last","gru_dt","prefix","resid_q")}
def rollout(model,U,DT,Q):
    m=U[0:1]; out=[m]
    for i in range(1,U.shape[0]): m=model(m,U[i:i+1],DT[i:i+1],Q[i:i+1]); out.append(m)
    return torch.cat(out)
def feats(meta):
    d=[datetime.fromisoformat(x) for x in meta["dates"]]; dt=[0.0]+[(d[i]-d[i-1]).days/30.0 for i in range(1,len(d))]; return torch.tensor(dt),torch.tensor([float(v) for v in meta["clear"]]),d
ARMS=["last","latest_head","gru_dt","gru_dt+head","prefix","prefix+head","resid_q+head"]
@torch.no_grad()
def states(arm,U,T,DT,Q):
    base=arm[:-5] if arm.endswith("+head") else ("last" if arm=="latest_head" else arm)
    if base=="last": return U
    if base=="prefix": return T
    return rollout(UPD[base],U,DT,Q)
@torch.no_grad()
def predict(arm,S):
    X=(S-mean)/sd; net=HEADS["last"] if arm=="latest_head" else HEADS[arm[:-5]] if arm.endswith("+head") else dec
    return torch.sigmoid(net(X).float()).squeeze(1).cpu().numpy()[:,W,W]>0.5
res={"schema":"p1-event-holdout-v0","source":a.src,"events":{}}
for ev in a.events.split(","):
    D=ROOT/"arrival_v0"/ev; MD=ROOT/f"sen12_pilot/holdout_{ev}/mask_u8"
    ids=sorted(p.stem for p in (D/"meta").glob("*.json") if (D/"single_fp16"/f"{p.stem}.npy").exists() and (MD/f"{p.stem}.npy").exists())
    if a.probe: ids=ids[:8]
    rows={arm:[] for arm in ARMS}
    for s in ids:
        meta=json.loads((D/"meta"/f"{s}.json").read_text()); U=torch.from_numpy(np.load(D/"single_fp16"/f"{s}.npy").astype("float32")).to(dev); T=torch.from_numpy(np.load(D/"prefix_fp16"/f"{s}.npy").astype("float32")).to(dev); DT,Q,d=feats(meta); DT,Q=DT.to(dev),Q.to(dev)
        Y=np.load(MD/f"{s}.npy")[W,W]>0; post=meta["post_index"]; clear=[float(v) for v in meta["clear"]]
        fvp=next((i for i in range(post,len(d)) if clear[i]>=0.5),None) if post is not None else None
        evd=datetime.fromisoformat(meta["event_date"]) if meta.get("event_date") else (d[post] if post is not None else None)
        for arm in ARMS:
            P=predict(arm,states(arm,U,T,DT,Q)); area=P.sum((1,2)); iou=[float((p&Y).sum()/max((p|Y).sum(),1)) for p in P]
            rows[arm].append({"id":s,"area":area.tolist(),"iou":iou,"post":post,"fvp":fvp,"dates":[x.date().isoformat() for x in d],"event":evd.date().isoformat() if evd else None,"positive":bool(Y.any())})
    out={}
    for arm in ARMS:
        R=rows[arm]; out[arm]={}
        for fb in ("far05","far10"):
            thr=THR[arm][fb]; far=float(np.mean([max(x["area"][:x["post"]])>=thr for x in R if x["post"]])) if any(x["post"] for x in R) else None
            pos=[x for x in R if x["positive"] and x["post"] is not None]; latA=[];latB=[];conf=0;piou=[]
            for x in pos:
                hit=[i for i in range(x["post"],len(x["area"])) if x["area"][i]>=thr]; piou+=x["iou"][x["post"]:]
                if hit: conf+=1; dh=datetime.fromisoformat(x["dates"][hit[0]]); latA.append((dh-datetime.fromisoformat(x["event"])).days); latB.append((dh-datetime.fromisoformat(x["dates"][x["fvp"]])).days if x["fvp"] is not None else None)
            latB=[v for v in latB if v is not None]; budget=0.05 if fb=="far05" else 0.10
            out[arm][fb]={"thr":thr,"realised_pre_event_far":far,"valid":(far is not None and far<3*budget),"n_positive":len(pos),"confirmed":conf,"never":len(pos)-conf,"confirm_rate":conf/max(len(pos),1),"latA_median":float(np.median(latA)) if latA else None,"latB_median":float(np.median(latB)) if latB else None,"latB_p25_p75":[float(np.quantile(latB,.25)),float(np.quantile(latB,.75))] if latB else None,"post_iou":float(np.mean(piou)) if piou else None}
    res["events"][ev]={"n_tiles":len(ids),"arms":out}; (OUT/f"rows_{ev}.json").write_text(json.dumps(rows))
lh=[res["events"][e]["arms"]["latest_head"]["far10"] for e in res["events"]]
res["event_level"]={"n_events":len(lh),"latest_head_valid_and_conf70":sum(1 for x in lh if x["valid"] and x["confirm_rate"]>=0.70),"generalises":sum(1 for x in lh if x["valid"] and x["confirm_rate"]>=0.70)>=2}
(OUT/"summary.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res["event_level"])); print("EVENT HOLDOUT DONE")
