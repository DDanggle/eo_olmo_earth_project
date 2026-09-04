#!/usr/bin/env python3
"""B-v1 generic frozen-cache decoder trainer (NOT the sealed pilot; a re-implementation of its P4 recipe for caches of any C x G x G, G in {16,32}).
Recipe (pilot_sen12_gp_heads.py, sealed): AdamW lr 1e-3 wd 1e-4 cosine(T=epochs), batch 16, 40 epochs, BCEWithLogits pos_weight = train neg/pos (exact, all masks) capped 50,
best epoch by val pixel IoU@0.5; test: micro pixel IoU@0.5, positive-patch macro IoU@0.5, exact AP. Embedding normalisation: per-channel mean/std from 400 evenly spaced train tiles.
Decoder = EmbDecoder(cin): 1x1 proj->128, BN, ReLU; two conv_bn upsample stages (x2, x2); 1x1 head; bilinear to 128. Identical parameter count for G=16 and G=32.
Calibration gate: --cache sen12_pilot/holdout_chimanimani (OlmoEarth 32x32) on a confirmatory fold must land within seed noise of the sealed pilot's P4 (reported, not asserted)."""
import json, os, sys, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
if os.environ.get("CUDA_VISIBLE_DEVICES")!="1": raise SystemExit("CUDA_VISIBLE_DEVICES must be 1")
ROOT=Path("/home/work/data/olmoearth")
ap=argparse.ArgumentParser(); ap.add_argument("--cache",required=True); ap.add_argument("--fold",required=True); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--epochs",type=int,default=40)
ap.add_argument("--folds",default=str(ROOT/"sen12_gp_contract/loco_folds.json")); ap.add_argument("--contract",default=str(ROOT/"sen12_gp_contract/sample_contract.jsonl")); ap.add_argument("--out",required=True)
a=ap.parse_args(); CACHE=ROOT/a.cache; OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda")
torch.manual_seed(a.seed); np.random.seed(a.seed)
FOLDS=json.loads(Path(a.folds).read_text()); fold=next(f for f in FOLDS["folds"] if f["fold"]==a.fold)
recs=[json.loads(l) for l in Path(a.contract).read_text().splitlines() if l]
def members(split):
    regions=fold["train_regions"] if split=="train" else [fold["val_region"]] if split=="val" else [fold["test_region"]]
    return sorted(r["sample_id"] for r in recs if r["region"] in regions and not r.get("error") and r.get("s15_eligible",True) and (CACHE/"mask_u8"/f"{r['sample_id']}.npy").exists())
ids={s:members(s) for s in ("train","val","test")}
def emb_stats(train_ids,sample=400):
    idx=np.linspace(0,len(train_ids)-1,min(sample,len(train_ids))).astype(int); acc=acc2=None; n=0
    for j in idx:
        x=np.load(CACHE/"emb_fp16"/f"{train_ids[j]}.npy").astype("float32"); m=x.mean(axis=(1,2)); m2=(x**2).mean(axis=(1,2)); acc=m if acc is None else acc+m; acc2=m2 if acc2 is None else acc2+m2; n+=1
    mean=acc/n; sd=np.sqrt(np.maximum(acc2/n-mean**2,1e-6)); return torch.tensor(mean).view(-1,1,1).float(), torch.tensor(sd).view(-1,1,1).float()
st=emb_stats(ids["train"])
def load(split):
    X=torch.from_numpy(np.stack([np.load(CACHE/"emb_fp16"/f"{s}.npy").astype("float32") for s in ids[split]])); X=(X-st[0])/st[1]
    Y=torch.from_numpy(np.stack([np.load(CACHE/"mask_u8"/f"{s}.npy") for s in ids[split]]).astype("float32")).unsqueeze(1); return X,Y
Xtr,Ytr=load("train"); Xva,Yva=load("val"); Xte,Yte=load("test"); C,G=Xtr.shape[1],Xtr.shape[2]
def conv_bn(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class EmbDecoder(nn.Module):
    def __init__(s,cin,base=128):
        super().__init__(); s.proj=nn.Sequential(nn.Conv2d(cin,base,1),nn.BatchNorm2d(base),nn.ReLU(inplace=True)); s.u1,s.u2=conv_bn(base,base//2),conv_bn(base//2,base//4); s.head=nn.Conv2d(base//4,1,1)
    def forward(s,x):
        x=s.proj(x); x=s.u1(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); x=s.u2(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); return F.interpolate(s.head(x),size=(128,128),mode="bilinear",align_corners=False)
model=EmbDecoder(C).to(dev); npar=sum(p.numel() for p in model.parameters())
pos=float(Ytr.sum()); neg=float(Ytr.numel()-pos); pw=min(neg/max(pos,1.0),50.0)
lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=dev)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
@torch.no_grad()
def predict(X,bs=32):
    model.eval(); out=[]
    for i in range(0,len(X),bs):
        with torch.autocast("cuda",dtype=torch.bfloat16): out.append(torch.sigmoid(model(X[i:i+bs].to(dev)).float()).cpu())
    return torch.cat(out).squeeze(1)
def exact_ap(scores,labels):
    o=np.argsort(-scores,kind="mergesort"); s=scores[o]; l=labels[o]; P=l.sum()
    if P==0: return None
    b=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]; tp=np.cumsum(l)[b]; fp=(b+1)-tp; prec=tp/np.maximum(tp+fp,1); rec=tp/P; return float(np.sum(np.diff(np.r_[0,rec])*prec))
def metrics(P,Y):
    pred=(P>0.5).float(); tp=(pred*Y).flatten(1).sum(1); fp=(pred*(1-Y)).flatten(1).sum(1); fn=((1-pred)*Y).flatten(1).sum(1); mp=Y.flatten(1).sum(1)
    den=tp+fp+fn; piou=torch.where(den>0,tp/den.clamp(min=1e-9),torch.ones_like(den)); posm=float(piou[mp>0].mean()) if (mp>0).any() else None
    return {"iou":float(tp.sum()/max(float((tp+fp+fn).sum()),1e-9)),"positive_patch_macro_iou":posm,"positive_patch_n":int((mp>0).sum()),"auprc_exact":exact_ap(P.numpy().ravel().astype("float64"),Y.numpy().ravel().astype("uint8"))}
g=torch.Generator().manual_seed(a.seed); best={"val_iou":-1,"epoch":0,"state":None}; hist=[]; t0=time.perf_counter()
for ep in range(1,a.epochs+1):
    model.train(); perm=torch.randperm(len(Xtr),generator=g); tot=0.0
    for i in range(0,len(perm),16):
        idx=perm[i:i+16]; x=Xtr[idx].to(dev); y=Ytr[idx].squeeze(1).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16): loss=lossf(model(x).squeeze(1).float(),y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot+=float(loss)*len(idx)
    sched.step(); mv=metrics(predict(Xva),Yva.squeeze(1)); hist.append({"epoch":ep,"loss":tot/len(perm),"val_iou":mv["iou"]})
    if mv["iou"]>best["val_iou"]: best={"val_iou":mv["iou"],"epoch":ep,"state":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    print(f"epoch {ep}/{a.epochs} loss {tot/len(perm):.4f} val_iou {mv['iou']:.4f} (best {best['val_iou']:.4f}@{best['epoch']}) {time.perf_counter()-t0:.0f}s",flush=True)
model.load_state_dict(best["state"]); mt=metrics(predict(Xte),Yte.squeeze(1)); mv=metrics(predict(Xva),Yva.squeeze(1))
rep={"schema":"cache-decoder-train-v1","cache":a.cache,"fold":a.fold,"seed":a.seed,"emb_shape":[C,G,G],"trainable_params":npar,"pos_weight":pw,"epochs":a.epochs,"best_val_epoch":best["epoch"],"best_val_iou":best["val_iou"],"val":mv,"test":mt,"split":{k:len(v) for k,v in ids.items()},"train_s":time.perf_counter()-t0,"history":hist,
     "recipe":"AdamW 1e-3 wd 1e-4 cosine, batch 16, BCE pos_weight exact cap 50, best val pixel IoU@0.5; re-implementation of the sealed pilot P4 recipe for G in {16,32}"}
(OUT/f"{a.fold}_seed{a.seed}.json").write_text(json.dumps(rep,indent=1)); torch.save({"model_state":best["state"],"cin":C,"grid":G},OUT/f"{a.fold}_seed{a.seed}_best.pt")
print("TEST",json.dumps(mt)); print("DONE")
