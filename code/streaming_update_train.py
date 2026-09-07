#!/usr/bin/env python3
"""T1: learn to keep a materialized OlmoEarth cache alive without re-encoding the full window.
State m starts as the full-window embedding at cutoff 4 (teacher e_4). For each later cutoff c in (6,8,10,12) the module sees ONLY
the previous state and the two new single-acquisition embeddings u_{c-2},u_{c-1} (mean-pooled to u) and must predict the full-window
teacher e_c. Closed-loop rollout (student feeds its own state). Modules (all per-pixel, 1x1 convs, channel 768):
  ema      : m = (1-a) m + a u, a = sigmoid(scalar)                                      (training-free-ish reference, 1 parameter)
  gru      : ConvGRU cell 1x1 (hidden 768) on input u                                    (budget-matched recurrent baseline)
  residual : m = m + g * f([m, u, u - P(m)]), g = sigmoid gate; P = 1x1; f = 1x1-ReLU-1x1 (ours: predicted-vs-observed residual)
Loss: MSE to the teacher at every step. Evaluation on the fold's test region at c=12: (i) cosine / relative MSE vs teacher e_12,
(ii) DOWNSTREAM: the frozen mean-cache decoder (resolution_contract_v2/p4_native_control checkpoint, trained on e_12) applied to the
student state -> AP and positive-patch macro IoU, compared with the same decoder on the teacher e_12 and on naive baselines
(freeze m_4; mean of singles). (iii) Cost: encoded timestep-units per tile: full re-encode at each step = 6+8+10+12 = 36; streaming = 4 + 8 = 12."""
import json, os, time, argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
if os.environ.get("CUDA_VISIBLE_DEVICES")!="1": raise SystemExit("CUDA_VISIBLE_DEVICES must be 1")
ROOT=Path("/home/work/data/olmoearth"); sys.path.insert(0,str(ROOT/"code"))
ap=argparse.ArgumentParser(); ap.add_argument("--data",default="olmo_streaming_dev"); ap.add_argument("--fold",required=True); ap.add_argument("--module",required=True,choices=["ema","gru","residual"]); ap.add_argument("--seed",type=int,default=1)
ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--decoder-dir",default="resolution_contract_v2/p4_native_control"); ap.add_argument("--sealed-cache",default="sen12_pilot/holdout_chimanimani"); ap.add_argument("--out",required=True); a=ap.parse_args()
D=ROOT/a.data; OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda"); torch.manual_seed(a.seed); np.random.seed(a.seed)
man=json.loads((ROOT/"sen12_gp_contract/t1_manifest.json").read_text())[a.fold]; CUT=(4,6,8,10,12)
def ok(sid): return (D/"teacher_fp16"/f"{sid}.npy").exists() and (D/"single_fp16"/f"{sid}.npy").exists()
ids={k:[s for s in v if ok(s)] for k,v in man.items()}; print({k:len(v) for k,v in ids.items()},flush=True)
def load(split):
    T=torch.from_numpy(np.stack([np.load(D/"teacher_fp16"/f"{s}.npy") for s in ids[split]]).astype("float32"))   # (N,5,768,32,32)
    S=torch.from_numpy(np.stack([np.load(D/"single_fp16"/f"{s}.npy") for s in ids[split]]).astype("float32"))    # (N,12,768,32,32)
    U=torch.stack([S[:,c-2:c].mean(1) for c in CUT[1:]],1)                                                          # (N,4,768,32,32) new evidence per step
    Y=torch.from_numpy(np.stack([np.load(D/"mask_u8"/f"{s}.npy") for s in ids[split]]).astype("float32"))
    return T,U,S,Y
Ttr,Utr,_,_=load("train"); Tva,Uva,_,_=load("val"); Tte,Ute,Ste,Yte=load("test")
sc=Ttr[:,-1].std().item()  # global scale for relative losses
class EMA(nn.Module):
    def __init__(s): super().__init__(); s.a=nn.Parameter(torch.tensor(-1.0))
    def forward(s,m,u): a=torch.sigmoid(s.a); return (1-a)*m+a*u
class GRU(nn.Module):
    def __init__(s,c=768): super().__init__(); s.zr=nn.Conv2d(2*c,2*c,1); s.h=nn.Conv2d(2*c,c,1)
    def forward(s,m,u):
        z,r=torch.sigmoid(s.zr(torch.cat([m,u],1))).chunk(2,1); n=torch.tanh(s.h(torch.cat([r*m,u],1))); return (1-z)*m+z*n
class Residual(nn.Module):
    def __init__(s,c=768,h=768): super().__init__(); s.P=nn.Conv2d(c,c,1); s.f=nn.Sequential(nn.Conv2d(3*c,h,1),nn.ReLU(inplace=True),nn.Conv2d(h,c,1)); s.g=nn.Conv2d(3*c,c,1)
    def forward(s,m,u): d=u-s.P(m); x=torch.cat([m,u,d],1); return m+torch.sigmoid(s.g(x))*s.f(x)
model={"ema":EMA,"gru":GRU,"residual":Residual}[a.module]().to(dev); npar=sum(p.numel() for p in model.parameters())
def rollout(T,U,bs_idx):
    m=T[bs_idx,0].to(dev); loss=0.0; states=[]
    for k in range(4):
        with torch.autocast("cuda",dtype=torch.bfloat16): m=model(m,U[bs_idx,k].to(dev)).float()
        loss=loss+F.mse_loss(m,T[bs_idx,k+1].to(dev))/sc**2; states.append(m)
    return loss/4, states[-1]
opt=torch.optim.AdamW(model.parameters(),lr=1e-3 if a.module!="ema" else 1e-1,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
g=torch.Generator().manual_seed(a.seed); best={"val":1e9,"epoch":0,"state":None}; hist=[]; t0=time.perf_counter()
@torch.no_grad()
def eval_loss(T,U):
    model.eval(); tot=0.0; n=0
    for i in range(0,len(T),16): idx=torch.arange(i,min(i+16,len(T))); l,_=rollout(T,U,idx); tot+=float(l)*len(idx); n+=len(idx)
    return tot/n
for ep in range(1,a.epochs+1):
    model.train(); perm=torch.randperm(len(Ttr),generator=g); tot=0.0
    for i in range(0,len(perm),16):
        idx=perm[i:i+16]; loss,_=rollout(Ttr,Utr,idx); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot+=float(loss)*len(idx)
    sched.step(); v=eval_loss(Tva,Uva); hist.append({"epoch":ep,"train":tot/len(perm),"val":v})
    if v<best["val"]: best={"val":v,"epoch":ep,"state":{k:x.detach().cpu().clone() for k,x in model.state_dict().items()}}
    print(f"epoch {ep}/{a.epochs} train {tot/len(perm):.4f} val {v:.4f} (best {best['val']:.4f}@{best['epoch']}) {time.perf_counter()-t0:.0f}s",flush=True)
model.load_state_dict(best["state"]); model.eval()
# ---- test: student state at c=12 vs teacher, plus downstream with the frozen decoder ----
from cache_decoder_train_lib import EmbDecoder, emb_stats_from_cache, metrics   # small lib extracted for reuse
ck=torch.load(ROOT/a.decoder_dir/f"{a.fold}_seed1_best.pt",map_location="cpu"); dec=EmbDecoder(ck["cin"]).to(dev); dec.load_state_dict(ck["model_state"]); dec.eval()
mu,sd=emb_stats_from_cache(ROOT/a.sealed_cache, a.fold)   # identical normalisation to the decoder's training
@torch.no_grad()
def downstream(X):
    out=[]
    for i in range(0,len(X),32):
        with torch.autocast("cuda",dtype=torch.bfloat16): out.append(torch.sigmoid(dec(((X[i:i+32]-mu)/sd).to(dev)).float()).cpu())
    return metrics(torch.cat(out).squeeze(1),Yte)
with torch.no_grad():
    student=torch.cat([rollout(Tte,Ute,torch.arange(i,min(i+16,len(Tte))))[1].cpu() for i in range(0,len(Tte),16)])
teacher=Tte[:,-1]; frozen=Tte[:,0]; singles_mean=Ste.mean(1)
def agree(X): return {"cosine":float(F.cosine_similarity(X.flatten(1),teacher.flatten(1),dim=1).mean()),"rel_mse":float(((X-teacher)**2).mean()/(teacher**2).mean())}
rep={"schema":"streaming-update-train-v0","fold":a.fold,"module":a.module,"seed":a.seed,"params":npar,"n":{k:len(v) for k,v in ids.items()},"best_val_epoch":best["epoch"],"best_val_loss":best["val"],"history":hist,"train_s":time.perf_counter()-t0,
     "agreement_c12":{"student":agree(student),"frozen_m4":agree(frozen),"singles_mean":agree(singles_mean)},
     "downstream_c12":{"teacher_full_reencode":downstream(teacher),"student":downstream(student),"frozen_m4":downstream(frozen),"singles_mean":downstream(singles_mean)},
     "cost_timestep_units":{"full_reencode_per_step_total":36,"streaming_total":12,"initial_encode":4}}
(OUT/f"{a.fold}_{a.module}_seed{a.seed}.json").write_text(json.dumps(rep,indent=1)); print("TEST",json.dumps({k:rep[k] for k in ("agreement_c12","downstream_c12")})); print("DONE")
