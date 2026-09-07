#!/usr/bin/env python3
"""T0 screen trainer: does keeping temporal evidence in the cache help a frozen-cache decoder? Same recipe as cache_decoder_train.py
(AdamW 1e-3 wd 1e-4 cosine, batch 16, 40 epochs, BCE pos_weight exact cap 50, best val pixel IoU@0.5), same EmbDecoder, same folds/contract.
Readouts (input to the identical decoder; all time features are first projected per timestep by a source-fit PCA-64 so RAM and bytes are bounded):
  mean     : sealed mean cache only (768 ch)                                  -> baseline
  diffpca  : mean + PCA-32 of (late-half mean - early-half mean) fit on source  -> training-free temporal sketch, +32 ch
  sketch   : mean + learned 1x1 conv over the (T*64) PCA time stack -> 32 ch     -> source-trained temporal sketch, +32 ch (byte-matched to diffpca)
  full     : learned 1x1 conv over (T*64) PCA time stack -> 128 ch, no mean     -> reference readout over all timesteps (not an upper bound)
Reports positive-patch macro IoU@0.5 and exact AP on the fold's test region."""
import json, os, time, argparse
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
if os.environ.get("CUDA_VISIBLE_DEVICES")!="1": raise SystemExit("CUDA_VISIBLE_DEVICES must be 1")
ROOT=Path("/home/work/data/olmoearth")
ap=argparse.ArgumentParser(); ap.add_argument("--cache",required=True,help="dir with emb_fp16, emb_time_fp16, mask_u8"); ap.add_argument("--fold",required=True); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--epochs",type=int,default=40)
ap.add_argument("--readout",required=True,choices=["mean","diffpca","sketch","full"]); ap.add_argument("--pca-dim",type=int,default=64); ap.add_argument("--sketch-dim",type=int,default=32)
ap.add_argument("--folds",default=str(ROOT/"sen12_gp_contract/loco_folds.json")); ap.add_argument("--contract",default=str(ROOT/"sen12_gp_contract/sample_contract.jsonl")); ap.add_argument("--out",required=True)
a=ap.parse_args(); CACHE=ROOT/a.cache; OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda"); torch.manual_seed(a.seed); np.random.seed(a.seed)
FOLDS=json.loads(Path(a.folds).read_text()); fold=next(f for f in FOLDS["folds"] if f["fold"]==a.fold); recs=[json.loads(l) for l in Path(a.contract).read_text().splitlines() if l]
def members(split):
    regions=fold["train_regions"] if split=="train" else [fold["val_region"]] if split=="val" else [fold["test_region"]]
    return sorted(r["sample_id"] for r in recs if r["region"] in regions and not r.get("error") and r.get("s15_eligible",True) and (CACHE/"mask_u8"/f"{r['sample_id']}.npy").exists() and (CACHE/"emb_time_fp16"/f"{r['sample_id']}.npy").exists())
ids={s:members(s) for s in ("train","val","test")}; assert all(ids.values()), {k:len(v) for k,v in ids.items()}
def load_time(s): return np.load(CACHE/"emb_time_fp16"/f"{s}.npy").astype("float32")      # (T,768,32,32)
def load_mean(s): return np.load(CACHE/"emb_fp16"/f"{s}.npy").astype("float32")           # (768,32,32)
# ---- source-fit statistics (train split only) ----
fit_idx=np.linspace(0,len(ids["train"])-1,min(400,len(ids["train"]))).astype(int); fit_ids=[ids["train"][j] for j in fit_idx]
def pca_fit(vecs,k):
    mu=vecs.mean(0); X=vecs-mu; U,S,Vt=np.linalg.svd(X,full_matrices=False); comp=Vt[:k]; proj=X@comp.T; return mu.astype("float32"),comp.astype("float32"),proj.std(0).astype("float32")+1e-6
Tref=load_time(fit_ids[0]).shape[0]
if a.readout in ("sketch","full"):
    samp=np.concatenate([load_time(s).transpose(0,2,3,1).reshape(-1,768)[::16] for s in fit_ids[:100]]); pmu,pcomp,psd=pca_fit(samp,a.pca_dim); del samp
if a.readout=="diffpca":
    def diff(s):
        t=load_time(s); h=t.shape[0]//2; return t[h:].mean(0)-t[:h].mean(0)                 # (768,32,32)
    samp=np.concatenate([diff(s).transpose(1,2,0).reshape(-1,768)[::8] for s in fit_ids[:200]]); dmu,dcomp,dsd=pca_fit(samp,a.sketch_dim); del samp
def features(s):
    m=load_mean(s)
    if a.readout=="mean": return m
    if a.readout=="diffpca":
        d=diff(s).transpose(1,2,0); z=((d-dmu)@dcomp.T)/dsd; return np.concatenate([m,z.transpose(2,0,1)],0)      # (768+32,32,32)
    t=load_time(s); assert t.shape[0]==Tref, (s,t.shape); z=((t.transpose(0,2,3,1)-pmu)@pcomp.T)/psd                   # (T,32,32,64)
    z=z.transpose(0,3,1,2).reshape(-1,32,32)                                                                           # (T*64,32,32)
    return np.concatenate([m,z],0) if a.readout=="sketch" else z
def stack(split):
    X=torch.from_numpy(np.stack([features(s) for s in ids[split]]).astype("float32")); Y=torch.from_numpy(np.stack([np.load(CACHE/"mask_u8"/f"{s}.npy") for s in ids[split]]).astype("float32")).unsqueeze(1); return X,Y
Xtr,Ytr=stack("train"); Xva,Yva=stack("val"); Xte,Yte=stack("test")
mean_=Xtr.mean(dim=(0,2,3),keepdim=True); sd_=Xtr.std(dim=(0,2,3),keepdim=True).clamp(min=1e-3); Xtr=(Xtr-mean_)/sd_; Xva=(Xva-mean_)/sd_; Xte=(Xte-mean_)/sd_
def conv_bn(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class EmbDecoder(nn.Module):
    def __init__(s,cin,base=128):
        super().__init__(); s.proj=nn.Sequential(nn.Conv2d(cin,base,1),nn.BatchNorm2d(base),nn.ReLU(inplace=True)); s.u1,s.u2=conv_bn(base,base//2),conv_bn(base//2,base//4); s.head=nn.Conv2d(base//4,1,1)
    def forward(s,x):
        x=s.proj(x); x=s.u1(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); x=s.u2(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); return F.interpolate(s.head(x),size=(128,128),mode="bilinear",align_corners=False)
class Readout(nn.Module):
    """sketch: mean(768) ++ learned 1x1 over time stack -> sketch_dim; full: learned 1x1 over time stack -> 128 (replaces mean)."""
    def __init__(s,kind,T,pd,sk):
        super().__init__(); s.kind=kind; s.T=T
        if kind=="sketch": s.tconv=nn.Conv2d(T*pd,sk,1); s.dec=EmbDecoder(768+sk)
        elif kind=="full": s.tconv=nn.Conv2d(T*pd,128,1); s.dec=EmbDecoder(128)
        else: s.dec=EmbDecoder(768 + (sk if kind=="diffpca" else 0))
    def forward(s,x):
        if s.kind=="sketch": return s.dec(torch.cat([x[:,:768],s.tconv(x[:,768:])],1))
        if s.kind=="full": return s.dec(s.tconv(x))
        return s.dec(x)
model=Readout(a.readout,Tref,a.pca_dim,a.sketch_dim).to(dev); npar=sum(p.numel() for p in model.parameters())
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
extra_bytes_per_tile={"mean":0,"diffpca":a.sketch_dim*32*32*2,"sketch":a.sketch_dim*32*32*2,"full":Tref*a.pca_dim*32*32*2}[a.readout]
rep={"schema":"temporal-readout-train-v0","cache":a.cache,"fold":a.fold,"seed":a.seed,"readout":a.readout,"T":Tref,"pca_dim":a.pca_dim,"sketch_dim":a.sketch_dim,"input_channels":int(Xtr.shape[1]),"trainable_params":npar,"extra_cache_bytes_per_tile_fp16":extra_bytes_per_tile,"pos_weight":pw,"epochs":a.epochs,"best_val_epoch":best["epoch"],"best_val_iou":best["val_iou"],"val":mv,"test":mt,"split":{k:len(v) for k,v in ids.items()},"train_s":time.perf_counter()-t0,"history":hist,
     "note":"sketch/full operate on a source-fit PCA-64 per-timestep projection, so 'full' is a bounded reference readout, not an upper bound over raw temporal tokens"}
(OUT/f"{a.fold}_{a.readout}_seed{a.seed}.json").write_text(json.dumps(rep,indent=1)); print("TEST",json.dumps(mt)); print("DONE")
