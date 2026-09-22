#!/usr/bin/env python3
"""KuroSiwo streaming pipeline on the OlmoEarth S1 cache (config/kurosiwo_streaming_prereg_v0.json).
Stage 'decoder': train EmbDecoder(768) on TEACHER embeddings (train split), select on val, freeze. Target: flood (class 3) vs rest, no-data/invalid pixels ignored.
Stage 'update' : train a one-step updater on (stale2, single_post) -> teacher3 (train split, feature MSE), select on val.
Stage 'eval'   : apply the frozen decoder (--dec-seed) to teacher3 / stale2 / singles-mean / updater states on TEST; exact AP of flood, IoU@0.5, false-alarm on no-water; per-activation macro."""
import argparse, json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0","1"): raise SystemExit("set CUDA_VISIBLE_DEVICES to 0 or 1")
ROOT=Path("/home/work/data/olmoearth"); C=ROOT/"kurosiwo_s1_cache"; dev=torch.device("cuda")
ap=argparse.ArgumentParser(); ap.add_argument("--stage",required=True,choices=["decoder","update","eval"]); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--module",default="gru",choices=["gru","gru_noobs","ema"]); ap.add_argument("--dec-seed",type=int,default=1); ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--out",default="artifacts/kurosiwo"); a=ap.parse_args()
OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); torch.manual_seed(a.seed); np.random.seed(a.seed)
META=[json.loads(l) for l in (C/"meta.jsonl").read_text().splitlines() if l]; ids={"train":[],"val":[],"test":[]}
for m in META: ids[{"validation":"val"}.get(m["split"],m["split"])].append(m["id"])
for k in ids: ids[k]=sorted(set(ids[k]))
def _finite(s): return all(np.isfinite(np.load(C/f"{k}_fp16"/f"{s}.npy").astype("float32")).all() for k in ("stale2","single","teacher3"))
_drop={k:[s for s in v if not _finite(s)] for k,v in ids.items()}
if any(_drop.values()):
    print("DROPPING non-finite cache tiles:",{k:len(v) for k,v in _drop.items()},flush=True); (OUT/"dropped_nonfinite.json").write_text(json.dumps(_drop))
    ids={k:[s for s in v if s not in set(_drop[k])] for k,v in ids.items()}
act={m["id"]:m["actid"] for m in META}
def L(kind,s): return np.load(C/f"{kind}_fp16"/f"{s}.npy").astype("float32")
def labels(split):
    Y=np.stack([np.load(C/"mask_u8"/f"{s}.npy") for s in ids[split]]); V=np.stack([np.load(C/"valid_u8"/f"{s}.npy") for s in ids[split]])
    return torch.from_numpy((Y==3).astype("float32")), torch.from_numpy((Y==0)|(V==0)), torch.from_numpy(Y==1)
def conv_bn(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class EmbDecoder(nn.Module):
    def __init__(s,cin,base=128):
        super().__init__(); s.proj=nn.Sequential(nn.Conv2d(cin,base,1),nn.BatchNorm2d(base),nn.ReLU(inplace=True)); s.u1,s.u2=conv_bn(base,base//2),conv_bn(base//2,base//4); s.head=nn.Conv2d(base//4,1,1)
    def forward(s,x):
        x=s.proj(x); x=s.u1(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); x=s.u2(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); return F.interpolate(s.head(x),size=(192,192),mode="bilinear",align_corners=False)
class GRU(nn.Module):
    def __init__(s,c=768): super().__init__(); s.zr=nn.Conv2d(2*c,2*c,1); s.h=nn.Conv2d(2*c,c,1)
    def forward(s,m,u):
        z,r=torch.sigmoid(s.zr(torch.cat([m,u],1))).chunk(2,1); n=torch.tanh(s.h(torch.cat([r*m,u],1))); return (1-z)*m+z*n
class EMA(nn.Module):
    def __init__(s): super().__init__(); s.a=nn.Parameter(torch.tensor(-1.0))
    def forward(s,m,u): a=torch.sigmoid(s.a); return (1-a)*m+a*u
MODULES={"gru":GRU,"gru_noobs":GRU,"ema":EMA}
def exact_ap(scores,labels):
    o=np.argsort(-scores,kind="mergesort"); s=scores[o]; l=labels[o]; P=l.sum()
    if P==0: return None
    b=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]; tp=np.cumsum(l)[b]; fp=(b+1)-tp; prec=tp/np.maximum(tp+fp,1); rec=tp/P; return float(np.sum(np.diff(np.r_[0,rec])*prec))
if a.stage=="decoder":
    Xtr=torch.from_numpy(np.stack([L("teacher3",s) for s in ids["train"]])); Ytr,Itr,_=labels("train"); Xva=torch.from_numpy(np.stack([L("teacher3",s) for s in ids["val"]])); Yva,Iva,_=labels("val")
    mu=Xtr.mean(dim=(0,2,3),keepdim=True); sd=Xtr.std(dim=(0,2,3),keepdim=True).clamp(min=1e-3); Xtr=(Xtr-mu)/sd; Xva=(Xva-mu)/sd; model=EmbDecoder(768).to(dev)
    pos=float(Ytr[~Itr].sum()); neg=float((~Itr).sum()-pos); pw=min(neg/max(pos,1),50.0); lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=dev),reduction="none")
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs); g=torch.Generator().manual_seed(a.seed); best=(-1.0,None,0)
    def val_ap():
        model.eval(); out=[]
        with torch.no_grad():
            for i in range(0,len(Xva),32):
                with torch.autocast("cuda",dtype=torch.bfloat16): out.append(torch.sigmoid(model(Xva[i:i+32].to(dev)).float()).cpu())
        P=torch.cat(out).squeeze(1); keep=~Iva; return exact_ap(P[keep].numpy().astype("float64"),Yva[keep].numpy().astype("uint8")) or 0.0
    for ep in range(1,a.epochs+1):
        model.train(); perm=torch.randperm(len(Xtr),generator=g)
        for i in range(0,len(perm),16):
            idx=perm[i:i+16]; x=Xtr[idx].to(dev); y=Ytr[idx].to(dev); keep=(~Itr[idx]).to(dev)
            with torch.autocast("cuda",dtype=torch.bfloat16): l=lossf(model(x).squeeze(1).float(),y); l=(l*keep).sum()/keep.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True); l.backward(); opt.step()
        sched.step(); v=val_ap()
        if v>best[0]: best=(v,{k:t.detach().cpu().clone() for k,t in model.state_dict().items()},ep)
        print(f"epoch {ep} val flood AP {v:.4f} best {best[0]:.4f}@{best[2]}",flush=True)
    torch.save({"model_state":best[1],"mu":mu,"sd":sd,"val_ap":best[0],"epoch":best[2],"pos_weight":pw},OUT/f"decoder_seed{a.seed}.pt"); print("DONE decoder",best[0])
elif a.stage=="update":
    def S(split): return (torch.from_numpy(np.stack([L("stale2",s) for s in ids[split]])),torch.from_numpy(np.stack([L("single",s)[2] for s in ids[split]])),torch.from_numpy(np.stack([L("teacher3",s) for s in ids[split]])))
    Mtr,Utr,Ttr=S("train"); Mva,Uva,Tva=S("val"); sc=float(Ttr[::40].float().std()); model=MODULES[a.module]().to(dev)
    print("finite check",bool(torch.isfinite(Mtr).all()),bool(torch.isfinite(Utr).all()),bool(torch.isfinite(Ttr).all()),"sc",sc,flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4 if a.module!="ema" else 1e-1,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs); g=torch.Generator().manual_seed(a.seed); best=(1e9,None,0)
    def step(M,U,idx):   # fp32 on purpose: the bf16 path produced non-finite gradients on this cache (2026-09-08)
        u=U[idx].to(dev); u=torch.zeros_like(u) if a.module=="gru_noobs" else u
        return model(M[idx].to(dev),u).float()
    def vloss():
        model.eval(); tot=0.0
        with torch.no_grad():
            for i in range(0,len(Mva),16): idx=torch.arange(i,min(i+16,len(Mva))); tot+=float(F.mse_loss(step(Mva,Uva,idx),Tva[idx].to(dev))/sc**2)*len(idx)
        return tot/len(Mva)
    skipped=0
    for ep in range(1,a.epochs+1):
        model.train(); perm=torch.randperm(len(Mtr),generator=g)
        for i in range(0,len(perm),16):
            idx=perm[i:i+16]; l=F.mse_loss(step(Mtr,Utr,idx),Ttr[idx].to(dev))/sc**2
            if not torch.isfinite(l): skipped+=1; continue
            opt.zero_grad(set_to_none=True); l.backward(); gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            if not torch.isfinite(gn): skipped+=1; opt.zero_grad(set_to_none=True); continue
            opt.step()
            if i==0 and ep==1: print("first batch loss",float(l),"grad norm",float(gn),flush=True)
        sched.step(); v=vloss()
        if np.isfinite(v) and v<best[0]: best=(v,{k:t.detach().cpu().clone() for k,t in model.state_dict().items()},ep)
        print(f"epoch {ep} val {v:.4f} best {best[0]:.4f}@{best[2]}",flush=True)
    assert best[1] is not None, "no finite validation epoch"
    torch.save({"model_state":best[1],"module":a.module,"val":best[0],"skipped_batches":skipped,"lr":5e-4,"grad_clip":1.0},OUT/f"updater_{a.module}_seed{a.seed}.pt"); print("DONE update",best[0],"skipped",skipped)
else:
    ck=torch.load(OUT/f"decoder_seed{a.dec_seed}.pt",map_location="cpu"); dec=EmbDecoder(768).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval(); mu,sd=ck["mu"],ck["sd"]
    T=torch.from_numpy(np.stack([L("teacher3",s) for s in ids["test"]])); M=torch.from_numpy(np.stack([L("stale2",s) for s in ids["test"]])); Sg=torch.from_numpy(np.stack([L("single",s) for s in ids["test"]])); Y,I,NW=labels("test"); acts=np.array([act[s] for s in ids["test"]])
    def probs(X):
        out=[]
        with torch.no_grad():
            for i in range(0,len(X),32):
                with torch.autocast("cuda",dtype=torch.bfloat16): out.append(torch.sigmoid(dec(((X[i:i+32]-mu)/sd).to(dev)).float()).cpu())
        return torch.cat(out).squeeze(1)
    def metr(P):
        keep=~I; ap_=exact_ap(P[keep].numpy().astype("float64"),Y[keep].numpy().astype("uint8")); pred=(P>0.5)&keep; tp=float((pred&(Y>0)).sum()); fp=float((pred&(Y==0)).sum()); fn=float(((~pred)&(Y>0)&keep).sum())
        far=float((pred&NW).sum()/max(float(NW.sum()),1)); per={}
        for A in np.unique(acts):
            sel=torch.from_numpy(acts==A); k=keep[sel]; per[int(A)]=exact_ap(P[sel][k].numpy().astype("float64"),Y[sel][k].numpy().astype("uint8"))
        return {"flood_ap":ap_,"flood_iou":tp/max(tp+fp+fn,1),"false_alarm_no_water":far,"per_activation_ap":per,"activation_macro_ap":float(np.mean([v for v in per.values() if v is not None]))}
    res={"decoder_seed":a.dec_seed,"n_test":len(ids["test"]),"arms":{"teacher3":metr(probs(T)),"stale2":metr(probs(M)),"singles_mean":metr(probs(Sg.mean(1)))}}
    for mod in ("gru","gru_noobs","ema"):
        for s in (1,2,3):
            p=OUT/f"updater_{mod}_seed{s}.pt"
            if not p.exists(): continue
            st=torch.load(p,map_location="cpu"); g=MODULES[mod]().to(dev); g.load_state_dict(st["model_state"]); g.eval(); outs=[]
            with torch.no_grad():
                for i in range(0,len(M),16):
                    u=Sg[i:i+16,2].to(dev); u=torch.zeros_like(u) if mod=="gru_noobs" else u
                    with torch.autocast("cuda",dtype=torch.bfloat16): outs.append(g(M[i:i+16].to(dev),u).float().cpu())
            res["arms"][f"{mod}_seed{s}"]=metr(probs(torch.cat(outs)))
    t_,f_=res["arms"]["teacher3"]["flood_ap"],res["arms"]["stale2"]["flood_ap"]
    for k,v in res["arms"].items(): v["recovery"]=(v["flood_ap"]-f_)/max(t_-f_,1e-9)
    (OUT/f"eval_dec{a.dec_seed}.json").write_text(json.dumps(res,indent=1)); print(json.dumps({k:(round(v["flood_ap"],4),round(v["recovery"],3)) for k,v in res["arms"].items()})); print("DONE eval")
