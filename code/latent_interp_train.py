#!/usr/bin/env python3
"""G-T4: learned continuous-time latent interpolator on OlmoEarth single-observation embeddings (config/latent_interp_train_prereg_v0.json)."""
import argparse, json, math, time, sys
from pathlib import Path
from datetime import datetime
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
ap=argparse.ArgumentParser(); ap.add_argument("--model",required=True,choices=["resmlp","ctxattn"]); ap.add_argument("--seed",type=int,default=1); ap.add_argument("--steps",type=int,default=20000); ap.add_argument("--out",required=True); ap.add_argument("--probe",action="store_true"); a=ap.parse_args()
ROOT=Path("/home/work/data/olmoearth"); SRC=ROOT/"olmo_streaming_dev"/"single_fp16"; OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda")
torch.manual_seed(a.seed); np.random.seed(a.seed); TEST={"hokkaido","kyrgyzstan1","newzealand"}
REC={}
for l in open(ROOT/"sen12_gp_contract/sample_contract.jsonl"):
    if l.strip(): r=json.loads(l); REC[r["sample_id"]]=r
def dates(sid):
    r=REC[sid]; q=r["scl_clear_fraction"]; idx=sorted(sorted(range(len(q)),key=lambda i:(-float(q[i]),i))[:12])
    return [datetime.fromisoformat(str(r["times"][i])[:19]) for i in idx]
ids=sorted(p.stem for p in SRC.glob("*.npy")); reg=lambda s:s.split("_s2_")[0]
test=[s for s in ids if reg(s) in TEST]; trainall=[s for s in ids if reg(s) not in TEST]; rng=np.random.default_rng(a.seed); rng.shuffle(trainall); nval=len(trainall)//10; val=trainall[:nval]; train=trainall[nval:]
if a.probe: train=train[:16]; val=val[:8]; test=test[:8]; a.steps=20
def tfeat(d,ref):  # continuous time features of date d relative to ref date
    doy=(d.timetuple().tm_yday-1)/365.0*2*math.pi; return [math.sin(doy),math.cos(doy),(d-ref).days/365.0]
class ResMLP(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(nn.Linear(768*3+4,512),nn.GELU(),nn.Linear(512,768))
    def forward(s,ctx,ctx_t,tgt_t,jm,jp,w):  # ctx: B,11,768 (others), ctx_t: B,11,3, tgt_t: B,3; jm/jp: indices in ctx of prev/next
        B=ctx.shape[0]; prev=ctx[torch.arange(B),jm]; nxt=ctx[torch.arange(B),jp]; lin=(1-w)[:,None]*prev+w[:,None]*nxt
        x=torch.cat([prev,nxt,lin,w[:,None],tgt_t[:,2:3]*0+ (ctx_t[torch.arange(B),jp,2]-ctx_t[torch.arange(B),jm,2])[:,None],tgt_t[:,0:2]],1); return lin+s.net(x)
class CtxAttn(nn.Module):
    def __init__(s,d=256):
        super().__init__(); s.inp=nn.Linear(768+3,d); s.q=nn.Linear(3,d); enc=nn.TransformerEncoderLayer(d,4,512,batch_first=True); s.tr=nn.TransformerEncoder(enc,2); s.out=nn.Linear(d,768)
    def forward(s,ctx,ctx_t,tgt_t,jm,jp,w):
        B=ctx.shape[0]; lin=(1-w)[:,None]*ctx[torch.arange(B),jm]+w[:,None]*ctx[torch.arange(B),jp]
        x=torch.cat([s.q(tgt_t)[:,None],s.inp(torch.cat([ctx,ctx_t],2))],1); h=s.tr(x)[:,0]; return lin+s.out(h)
model=(ResMLP() if a.model=="resmlp" else CtxAttn()).to(dev); opt=torch.optim.AdamW(model.parameters(),1e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.steps)
cache={}
def load(sid):
    if sid not in cache:
        if len(cache)>600: cache.pop(next(iter(cache)))
        cache[sid]=(torch.from_numpy(np.load(SRC/f"{sid}.npy").astype("float32")),dates(sid))
    return cache[sid]
def make_batch(sids,js=None,ntok=128,all_tokens=False):
    C=[];CT=[];TT=[];JM=[];JP=[];W=[];Y=[]
    for k,sid in enumerate(sids):
        S,d=load(sid); j=int(np.random.randint(1,11)) if js is None else js[k]; others=[i for i in range(12) if i!=j]; ref=d[j]
        St=S.reshape(12,768,-1); tok=np.arange(St.shape[2]) if all_tokens else np.random.choice(St.shape[2],ntok,replace=False)
        ctx=St[others][:,:,tok].permute(2,0,1); Y.append(St[j][:,tok].T); C.append(ctx); CT.append(torch.tensor([tfeat(d[i],ref) for i in others]).expand(len(tok),11,3)); TT.append(torch.tensor(tfeat(ref,ref)).expand(len(tok),3))
        jm=others.index(j-1); jp=others.index(j+1); gap=max((d[j+1]-d[j-1]).days,1); w=(d[j]-d[j-1]).days/gap
        JM+= [jm]*len(tok); JP+=[jp]*len(tok); W+=[w]*len(tok)
    return torch.cat(C).to(dev),torch.cat(CT).to(dev),torch.cat(TT).to(dev),torch.tensor(JM,device=dev),torch.tensor(JP,device=dev),torch.tensor(W,device=dev,dtype=torch.float32),torch.cat(Y).to(dev)
def loss_fn(p,y): return F.mse_loss(p,y)+ (1-F.cosine_similarity(p,y,dim=1)).mean()
t0=time.perf_counter(); hist=[]
for step in range(1,a.steps+1):
    model.train(); sids=[train[i] for i in np.random.randint(0,len(train),8)]; b=make_batch(sids); p=model(*b[:6]); loss=loss_fn(p,b[6]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step%500==0 or step==a.steps:
        model.eval(); 
        with torch.no_grad():
            vs=[val[i] for i in np.random.randint(0,len(val),16)]; b=make_batch(vs); p=model(*b[:6]); B=b[0].shape[0]; lin=(1-b[5])[:,None]*b[0][torch.arange(B),b[3]]+b[5][:,None]*b[0][torch.arange(B),b[4]]
            vc=float(F.cosine_similarity(p,b[6],dim=1).mean()); lc=float(F.cosine_similarity(lin,b[6],dim=1).mean())
        hist.append({"step":step,"loss":float(loss),"val_cos":vc,"val_linear_cos":lc}); print(f"step {step} loss {loss:.4f} val_cos {vc:.4f} linear {lc:.4f} {time.perf_counter()-t0:.0f}s",flush=True)
# test: all interior slots, all tokens, per-tile mean cos for model and linear, by gap bin
model.eval(); rows=[]
with torch.no_grad():
    for sid in test:
        S,d=load(sid)
        for j in range(1,11):
            b=make_batch([sid],js=[j],all_tokens=True); p=model(*b[:6]); B=b[0].shape[0]; lin=(1-b[5])[:,None]*b[0][torch.arange(B),b[3]]+b[5][:,None]*b[0][torch.arange(B),b[4]]
            rows.append({"id":sid,"region":reg(sid),"j":j,"gap_days":(d[j+1]-d[j-1]).days,"cos_model":float(F.cosine_similarity(p,b[6],dim=1).mean()),"cos_linear":float(F.cosine_similarity(lin,b[6],dim=1).mean())})
(OUT/f"{a.model}_seed{a.seed}_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
tiles=sorted({r["id"] for r in rows}); per={t:(np.mean([r["cos_model"] for r in rows if r["id"]==t]),np.mean([r["cos_linear"] for r in rows if r["id"]==t])) for t in tiles}
dm=np.array([per[t][0]-per[t][1] for t in tiles]); rng2=np.random.default_rng(0); bs=[rng2.choice(dm,len(dm)).mean() for _ in range(2000)]
bins=[("<=20",0,20),("21-40",21,40),("41-70",41,70),("71-120",71,120),(">120",121,10**6)]
rep={"schema":"latent-interp-train-v0","model":a.model,"seed":a.seed,"steps":a.steps,"n_train":len(train),"n_val":len(val),"n_test":len(test),"test_regions":sorted(TEST),
 "test_cos_model":float(np.mean([per[t][0] for t in tiles])),"test_cos_linear":float(np.mean([per[t][1] for t in tiles])),"paired_diff_mean":float(dm.mean()),"paired_diff_ci95":[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],
 "by_gap":{lab:{"n":sum(1 for r in rows if lo<=r["gap_days"]<=hi),"cos_model":float(np.mean([r["cos_model"] for r in rows if lo<=r["gap_days"]<=hi])) if any(lo<=r["gap_days"]<=hi for r in rows) else None,"cos_linear":float(np.mean([r["cos_linear"] for r in rows if lo<=r["gap_days"]<=hi])) if any(lo<=r["gap_days"]<=hi for r in rows) else None} for lab,lo,hi in bins},
 "by_region":{g:{"cos_model":float(np.mean([r["cos_model"] for r in rows if r["region"]==g])),"cos_linear":float(np.mean([r["cos_linear"] for r in rows if r["region"]==g]))} for g in sorted({r["region"] for r in rows})},
 "history":hist,"train_s":time.perf_counter()-t0,"params":sum(p.numel() for p in model.parameters())}
(OUT/f"{a.model}_seed{a.seed}.json").write_text(json.dumps(rep,indent=1)); torch.save(model.state_dict(),OUT/f"{a.model}_seed{a.seed}.pt")
print("TEST",json.dumps({k:rep[k] for k in ("test_cos_model","test_cos_linear","paired_diff_mean","paired_diff_ci95")})); print("LATENT INTERP TRAIN DONE")
