#!/usr/bin/env python3
"""P1 v0 causal update on arrival-order hiroshima (config/p1_causal_update_prereg_v0.json)."""
import argparse, json, hashlib, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, ROOT
ap=argparse.ArgumentParser(); ap.add_argument("--data",default="arrival_v0/hiroshima"); ap.add_argument("--out",required=True); ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--probe",action="store_true"); ap.add_argument("--far",type=float,default=0.05); a=ap.parse_args()
D=ROOT/a.data; OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda"); torch.manual_seed(a.seed); np.random.seed(a.seed); W=slice(8,120)
SEALED=ROOT/"sen12_pilot/holdout_hiroshima"
ids=sorted(p.stem for p in (D/"meta").glob("*.json") if (D/"single_fp16"/f"{p.stem}.npy").exists() and (D/"prefix_fp16"/f"{p.stem}.npy").exists() and (SEALED/"mask_u8"/f"{p.stem}.npy").exists())
blk=lambda s:int(hashlib.sha256(s.encode()).hexdigest(),16)%5; split={"train":[s for s in ids if blk(s) in (0,1,2)],"val":[s for s in ids if blk(s)==3],"test":[s for s in ids if blk(s)==4]}
if a.probe: split={k:v[:6] for k,v in split.items()}; a.epochs=1
print({k:len(v) for k,v in split.items()},flush=True)
META={s:json.loads((D/"meta"/f"{s}.json").read_text()) for s in ids}
mean,sd=emb_stats_from_cache(SEALED,"holdout_hiroshima"); mean,sd=mean.to(dev),sd.to(dev)
ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control/holdout_hiroshima_seed1_best.pt",map_location="cpu"); dec=EmbDecoder(ck["cin"]).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval()
MASK={s:(np.load(SEALED/"mask_u8"/f"{s}.npy")[W,W]>0) for s in ids}
def load(s): return torch.from_numpy(np.load(D/"single_fp16"/f"{s}.npy").astype("float32")), torch.from_numpy(np.load(D/"prefix_fp16"/f"{s}.npy").astype("float32"))
def feats(s):
    m=META[s]; d=[datetime.fromisoformat(x) for x in m["dates"]]; dt=[0.0]+[(d[i]-d[i-1]).days/30.0 for i in range(1,len(d))]; q=[float(v) for v in m["clear"]]; return torch.tensor(dt),torch.tensor(q)
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
    """proposed: innovation d=u-P(m); gate conditioned on [m,u,d,q,dt]; correction f([m,u,d])"""
    def __init__(s,c=768,h=768): super().__init__(); s.P=nn.Conv2d(c,c,1); s.f=nn.Sequential(nn.Conv2d(3*c,h,1),nn.ReLU(inplace=True),nn.Conv2d(h,c,1)); s.g=nn.Conv2d(3*c+2,c,1)
    def forward(s,m,u,dt,q):
        d=u-s.P(m); x=torch.cat([m,u,d],1); cond=torch.stack([q,dt],1).view(-1,2,1,1).expand(-1,2,m.shape[2],m.shape[3]); return m+torch.sigmoid(s.g(torch.cat([x,cond],1)))*s.f(x)
LEARNED={"ema":EMA,"gru":GRU,"gru_dt":GRUdt,"resid_q":ResidQ}
def rollout(model,U,DT,Q):  # U: T,768,32,32 (one tile) -> states T,768,32,32
    m=U[0:1]; out=[m]
    for i in range(1,U.shape[0]): m=model(m,U[i:i+1],DT[i:i+1],Q[i:i+1]); out.append(m)
    return torch.cat(out)
def states_for(arm,model,s):
    U,T=load(s); U,T=U.to(dev),T.to(dev); DT,Q=feats(s); DT,Q=DT.to(dev),Q.to(dev)
    if arm=="last": return U
    if arm=="mean": return torch.cumsum(U,0)/torch.arange(1,U.shape[0]+1,device=dev).view(-1,1,1,1)
    if arm=="prefix": return T
    with torch.no_grad(): return rollout(model,U,DT,Q)
models={}; t0=time.perf_counter()
for arm,cls in LEARNED.items():
    model=cls().to(dev); opt=torch.optim.AdamW(model.parameters(),1e-3,weight_decay=1e-4)
    for ep in range(a.epochs):
        order=np.random.permutation(len(split["train"])); tot=0
        for i in range(0,len(order),8):
            loss=0
            for j in order[i:i+8]:
                s=split["train"][j]; U,T=load(s); U,T=U.to(dev),T.to(dev); DT,Q=feats(s); DT,Q=DT.to(dev),Q.to(dev); S=rollout(model,U,DT,Q); loss=loss+F.mse_loss(S[1:],T[1:])
            loss=loss/max(1,len(order[i:i+8])); opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss)
        print(f"{arm} epoch {ep} loss {tot/max(1,int(np.ceil(len(order)/8))):.4f} {time.perf_counter()-t0:.0f}s",flush=True)
    model.eval(); models[arm]=model; torch.save(model.state_dict(),OUT/f"{arm}_seed{a.seed}.pt")
ARMS=["last","mean","ema","gru","gru_dt","resid_q","prefix"]
@torch.no_grad()
def readout_area_iou(S,s):
    X=(S-mean)/sd; P=torch.sigmoid(dec(X).float()).squeeze(1).cpu().numpy()[:,W,W]>0.5; Y=MASK[s]
    area=P.sum((1,2)); iou=[float((p&Y).sum()/max((p|Y).sum(),1)) for p in P]; return area,iou
def per_tile(arm,s):
    S=states_for(arm,models.get(arm),s); area,iou=readout_area_iou(S,s); m=META[s]; post=m["post_index"]; d=[datetime.fromisoformat(x) for x in m["dates"]]
    ev=datetime.fromisoformat(m["event_date"]) if m.get("event_date") else (d[post] if post is not None else None)
    return {"id":s,"area":area.tolist(),"iou":iou,"post":post,"dates":[x.date().isoformat() for x in d],"event":ev.date().isoformat() if ev else None,"positive":bool(MASK[s].any())}
res={"schema":"p1-causal-update-v0","split":{k:len(v) for k,v in split.items()},"far_target":a.far,"arms":{}}
rows={arm:{sp:[per_tile(arm,s) for s in split[sp]] for sp in ("val","test")} for arm in ARMS}
def pre_areas(R): return np.array([x["area"][i] for x in R if x["post"] is not None for i in range(x["post"])])
for arm in ARMS:
    pa=pre_areas(rows[arm]["val"]); thr=float(np.quantile(pa,1-a.far)) if len(pa) else 0.0
    T=rows[arm]["test"]; pre=pre_areas(T); far=float((pre>=thr).mean()) if len(pre) else None; lat=[]; conf=0; npos=0; post_iou=[]
    for x in T:
        if not x["positive"] or x["post"] is None: continue
        npos+=1; ev=datetime.fromisoformat(x["event"]); hit=[i for i in range(x["post"],len(x["area"])) if x["area"][i]>=thr]
        post_iou+= x["iou"][x["post"]:]
        if hit: conf+=1; lat.append((datetime.fromisoformat(x["dates"][hit[0]])-ev).days)
    res["arms"][arm]={"thr_area_px":thr,"test_pre_event_far":far,"n_positive":npos,"confirmed":conf,"confirm_rate":conf/max(npos,1),"latency_days_median":float(np.median(lat)) if lat else None,"latency_days_p25_p75":[float(np.quantile(lat,.25)),float(np.quantile(lat,.75))] if lat else None,"post_event_mean_iou":float(np.mean(post_iou)) if post_iou else None,"encode_units_per_arrival":"min(c,12)" if arm=="prefix" else 1}
    (OUT/f"rows_{arm}_seed{a.seed}.json").write_text(json.dumps(rows[arm]))
# paired bootstrap on latency (tiles confirmed by both) resid_q vs gru_dt and vs last
def lat_map(arm):
    thr=res["arms"][arm]["thr_area_px"]; out={}
    for x in rows[arm]["test"]:
        if not x["positive"] or x["post"] is None: continue
        hit=[i for i in range(x["post"],len(x["area"])) if x["area"][i]>=thr]; out[x["id"]]=(datetime.fromisoformat(x["dates"][hit[0]])-datetime.fromisoformat(x["event"])).days if hit else None
    return out
rng=np.random.default_rng(0); res["paired"]={}
for other in ("gru_dt","last"):
    A,B=lat_map("resid_q"),lat_map(other); common=[k for k in A if A[k] is not None and B[k] is not None]; d=np.array([A[k]-B[k] for k in common])
    bs=[rng.choice(d,len(d)).mean() for _ in range(2000)] if len(d)>5 else []
    res["paired"][f"resid_q_minus_{other}_latency_days"]={"n":len(common),"mean":float(d.mean()) if len(d) else None,"ci95":[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))] if bs else None,"resid_q_only_confirmed":sum(1 for k in A if A[k] is not None and B.get(k) is None),"other_only_confirmed":sum(1 for k in B if B[k] is not None and A.get(k) is None)}
(OUT/f"summary_seed{a.seed}.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res,indent=1)); print("P1 CAUSAL DONE")
