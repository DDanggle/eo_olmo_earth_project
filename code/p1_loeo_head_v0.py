#!/usr/bin/env python3
"""LOEO arrival-time head (config/p1_loeo_head_prereg_v0.json)."""
import argparse, json, hashlib, sys, time
from pathlib import Path
from datetime import datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, ROOT
ap=argparse.ArgumentParser(); ap.add_argument("--out",default="p1_loeo_head_v0"); ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--probe",action="store_true"); a=ap.parse_args()
EVENTS=["hiroshima","thrissur","itogon","hokkaido"]; OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda"); W=slice(8,120); torch.manual_seed(a.seed); np.random.seed(a.seed)
mean,sd=emb_stats_from_cache(ROOT/"sen12_pilot/holdout_hiroshima","holdout_hiroshima"); mean,sd=mean.to(dev),sd.to(dev)
class Head(nn.Module):
    def __init__(s,c=768): super().__init__(); s.f=nn.Sequential(nn.Conv2d(c,128,1),nn.ReLU(inplace=True),nn.Conv2d(128,1,1))
    def forward(s,x): return F.interpolate(s.f(x),size=(128,128),mode="bilinear",align_corners=False)
def tiles(ev):
    D=ROOT/"arrival_v0"/ev; MD=ROOT/f"sen12_pilot/holdout_{ev}/mask_u8"
    ids=sorted(p.stem for p in (D/"meta").glob("*.json") if (D/"single_fp16"/f"{p.stem}.npy").exists() and (D/"prefix_fp16"/f"{p.stem}.npy").exists() and (MD/f"{p.stem}.npy").exists())
    return [(ev,s) for s in (ids[:12] if a.probe else ids)]
def load(ev,s):
    D=ROOT/"arrival_v0"/ev; m=json.loads((D/"meta"/f"{s}.json").read_text()); U=torch.from_numpy(np.load(D/"single_fp16"/f"{s}.npy").astype("float32")); T=torch.from_numpy(np.load(D/"prefix_fp16"/f"{s}.npy").astype("float32"))
    mask=np.load(ROOT/f"sen12_pilot/holdout_{ev}/mask_u8"/f"{s}.npy")>0; return U,T,m,mask
def train_head(train_tiles):
    h=Head().to(dev); opt=torch.optim.AdamW(h.parameters(),1e-3,weight_decay=1e-4); t0=time.perf_counter()
    for ep in range(a.epochs):
        order=np.random.permutation(len(train_tiles)); tot=0
        for j in order:
            ev,s=train_tiles[j]; U,_,m,mask=load(ev,s); post=m["post_index"]; X=(U.to(dev)-mean)/sd; Y=torch.from_numpy(mask.astype("float32")).to(dev)[None,None].expand(U.shape[0],1,128,128).clone()
            if post is not None: Y[:post]=0
            else: Y[:]=0
            Wt=torch.ones(U.shape[0],1,1,1,device=dev)
            if post is not None and 0<post<U.shape[0] and mask.any(): Wt[:post]=(U.shape[0]-post)/post
            loss=(F.binary_cross_entropy_with_logits(h(X),Y,reduction="none")*Wt).mean(); opt.zero_grad(); loss.backward(); opt.step(); tot+=float(loss)
        if ep%5==0 or ep==a.epochs-1: print(f"  epoch {ep} loss {tot/len(order):.4f} {time.perf_counter()-t0:.0f}s",flush=True)
    return h.eval()
@torch.no_grad()
def rows_for(arm,net,tl):
    out=[]
    for ev,s in tl:
        U,T,m,mask=load(ev,s); S=(U if arm in ("head","last") else T).to(dev); P=torch.sigmoid(net((S-mean)/sd).float()).squeeze(1).cpu().numpy()[:,W,W]>0.5; Y=mask[W,W]
        d=[datetime.fromisoformat(x) for x in m["dates"]]; post=m["post_index"]; clear=[float(v) for v in m["clear"]]; fvp=next((i for i in range(post,len(d)) if clear[i]>=0.5),None) if post is not None else None
        evd=datetime.fromisoformat(m["event_date"]) if m.get("event_date") else (d[post] if post is not None else None)
        out.append({"id":s,"area":[int(p.sum()) for p in P],"iou":[float((p&Y).sum()/max((p|Y).sum(),1)) for p in P],"post":post,"fvp":fvp,"dates":[x.date().isoformat() for x in d],"event":evd.date().isoformat() if evd else None,"positive":bool(Y.any())})
    return out
def evaluate(R_val,R_test,far):
    pa=np.array([max(x["area"][:x["post"]]) for x in R_val if x["post"]]); thr=float(np.quantile(pa,1-far)) if len(pa) else 0.0
    pre=np.array([max(x["area"][:x["post"]]) for x in R_test if x["post"]]); realised=float((pre>=thr).mean()) if len(pre) else None
    pos=[x for x in R_test if x["positive"] and x["post"] is not None]; latA=[];latB=[];conf=0;piou=[]
    for x in pos:
        hit=[i for i in range(x["post"],len(x["area"])) if x["area"][i]>=thr]; piou+=x["iou"][x["post"]:]
        if hit: conf+=1; dh=datetime.fromisoformat(x["dates"][hit[0]]); latA.append((dh-datetime.fromisoformat(x["event"])).days); latB.append((dh-datetime.fromisoformat(x["dates"][x["fvp"]])).days if x["fvp"] is not None else None)
    latB=[v for v in latB if v is not None]
    return {"thr":thr,"realised_far":realised,"valid":(realised is not None and realised<3*far),"n_positive":len(pos),"confirmed":conf,"never":len(pos)-conf,"confirm_rate":conf/max(len(pos),1),"latA_median":float(np.median(latA)) if latA else None,"latB_median":float(np.median(latB)) if latB else None,"post_iou":float(np.mean(piou)) if piou else None}
res={"schema":"p1-loeo-head-v0","folds":{}}
for held in EVENTS:
    print("== fold held-out:",held,flush=True)
    tr_all=[t for ev in EVENTS if ev!=held for t in tiles(ev)]; blk=lambda s:int(hashlib.sha256(s.encode()).hexdigest(),16)%10; tr=[t for t in tr_all if blk(t[1])!=0]; va=[t for t in tr_all if blk(t[1])==0]; te=tiles(held)
    h=train_head(tr); torch.save(h.state_dict(),OUT/f"head_heldout_{held}.pt")
    ck=torch.load(ROOT/"resolution_contract_v2/p4_native_control"/f"holdout_{held}_seed1_best.pt",map_location="cpu"); dec=EmbDecoder(ck["cin"]).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval()
    arms={"loeo_head":("head",h),"prefix_frozen":("prefix",dec),"last_frozen":("last",dec)}; fold={}
    for name,(arm,net) in arms.items():
        Rv=rows_for(arm,net,va); Rt=rows_for(arm,net,te); fold[name]={fb:evaluate(Rv,Rt,f) for fb,f in (("far05",0.05),("far10",0.10))}
        e=fold[name]["far10"]; print(f"  {name:13s} FAR={e['realised_far']:.3f} valid={e['valid']} conf={e['confirmed']}/{e['n_positive']} latB={e['latB_median']} IoU={e['post_iou']:.3f}" if e['post_iou'] is not None else f"  {name} n/a",flush=True)
    res["folds"][held]={"n_train":len(tr),"n_val":len(va),"n_test":len(te),"arms":fold}
ok=[h for h,f in res["folds"].items() if f["arms"]["loeo_head"]["far10"]["valid"] and f["arms"]["loeo_head"]["far10"]["confirm_rate"]>=0.70]
res["event_level"]={"folds_valid_conf70":len(ok),"which":ok,"verdict":"generalises" if len(ok)>=3 else "partial" if len(ok)==2 else "fails"}
(OUT/"summary.json").write_text(json.dumps(res,indent=1)); print(json.dumps(res["event_level"])); print("LOEO HEAD DONE")
