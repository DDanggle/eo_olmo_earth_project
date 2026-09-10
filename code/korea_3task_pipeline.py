#!/usr/bin/env python3
"""Korea (AI-Hub 71363) shared-cache 3-task pipeline — prereg config/korea_shared_cache_3task_prereg_v1_amendment.json (Q1 shared asset, Q2 label efficiency, Q4 cost).
Tasks : T1 land_cover  = codes {10,20,30,40,50,60,100} 7-class (70/80 pixels ignored, index 255)
        T2 logged      = code 70 binary ;  T3 landslide = code 80 binary ("산사태및토석류피해지")
Target: label raster of the LAST labeled acquisition of each chip (the state at the end of the window the cache embedding encodes). Recorded as a design choice, not tuned.
Arms  : FULL_CACHE (all train chips, frozen cache + head) ; CACHE_K / RAW_K (K in {5,20}, seeds 1-3, draw random|posaware) ; FULL_RAW optional (--full-raw, budgeted steps)
Cache : korea_cache_v1/emb_fp16/<chip>.npy (768,32,32) fp16 ; Raw: aihub/s2_12band_v2/arrays/<tile>_<date>.npy (12,1024,1024) uint16 -> chip (12,T,128,128)
Eval  : 7 test clusters. T1: cluster-macro mIoU + per-class AP. T2/T3: all-cluster exact AP, IoU@0.5, positive-query-cluster IoU (conditional). val split is reported, NOT used for selection (val has ~0 positives for T2/T3).
Cost  : cache bytes read, raw bytes read, GPU seconds per arm, peak memory."""
import argparse, json, time, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument("--tasks",default="land_cover,logged,landslide"); ap.add_argument("--arms",default="FULL_CACHE,CACHE_K,RAW_K"); ap.add_argument("--K",default="5,20"); ap.add_argument("--seeds",default="1,2,3")
ap.add_argument("--draws",default="random,posaware"); ap.add_argument("--full-steps",type=int,default=4000); ap.add_argument("--k-steps",type=int,default=300); ap.add_argument("--full-raw",action="store_true"); ap.add_argument("--out",default="artifacts/korea_3task/run_v1"); ap.add_argument("--probe",action="store_true"); ap.add_argument("--full-bs",type=int,default=16,help="same batch size for FULL_CACHE and FULL_RAW (equal sample exposure)"); ap.add_argument("--allow-partial-window",action="store_true",help="keep chips whose last readable label predates the last cube date (default: drop; cache would see imagery the label/raw do not)"); a=ap.parse_args()
ROOT=Path("/home/work/data/olmoearth"); AI=ROOT/"aihub"; EMB=ROOT/"korea_cache_v1/emb_fp16"; MASK=AI/"labels_v1/mask_u8"; RAWA=AI/"s2_12band_v2/arrays"; OUT=ROOT/a.out; OUT.mkdir(parents=True,exist_ok=True); dev=torch.device("cuda")
LC=[10,20,30,40,50,60,100]; LUT=np.full(256,255,np.uint8)
for i,c in enumerate(LC): LUT[c]=i
TASK={"land_cover":("multi",7),"logged":("binary",70),"landslide":("binary",80)}
chips=[json.loads(l) for l in (AI/"korea_chip_manifest.jsonl").read_text().splitlines() if l.strip()]
def last_label_date(c):
    for d in reversed(c["dates"]):
        if (MASK/f"{c['chip_id']}__{d}.npy").exists(): return d
    return None
for c in chips: c["dates"]=[d.replace("-","") for d in c["dates"]]; c["ldate"]=last_label_date(c)   # manifest dates are ISO; mask/array files use yyyymmdd
chips=[c for c in chips if c["ldate"]]
n_partial=sum(1 for c in chips if c["ldate"]!=c["dates"][-1])
if not a.allow_partial_window: chips=[c for c in chips if c["ldate"]==c["dates"][-1]]   # v2 fix: cache encodes all dates; label/raw must cover the same window
print("chips with last label before last cube date:",n_partial,"dropped" if not a.allow_partial_window else "kept",flush=True); by_split=collections.defaultdict(list)
for c in chips: by_split[c["split"]].append(c)
if a.probe: by_split={s:v[:200] for s,v in by_split.items()}
print({s:len(v) for s,v in by_split.items()},flush=True)
def load_mask(c): return np.load(MASK/f"{c['chip_id']}__{c['ldate']}.npy")
def target(m,task):
    kind,code=TASK[task]; return LUT[m].astype(np.int64) if kind=="multi" else (m==code).astype(np.float32)
def load_emb_batch(cs,stats): X=torch.from_numpy(np.stack([np.load(EMB/f"{c['chip_id']}.npy").astype("float32") for c in cs])); return (X-stats[0])/stats[1], sum((EMB/f"{c['chip_id']}.npy").stat().st_size for c in cs)
_tile_cache={}
def load_raw_chip(c):
    xs=[]; nb=0
    for d in c["dates"]:
        if d>c["ldate"]: break
        p=RAWA/f"{c['tile_id']}_{d}.npy"
        if p not in _tile_cache:
            if len(_tile_cache)>24: _tile_cache.clear()
            _tile_cache[p]=np.load(p,mmap_mode="r")
        arr=_tile_cache[p]; xs.append(np.asarray(arr[:,c["y0"]:c["y0"]+128,c["x0"]:c["x0"]+128]).astype("float32")); nb+=12*128*128*2
    x=np.stack(xs,1)/10000.0; return np.clip(x,0,1.5), nb   # (12,T,128,128)
def emb_stats(cs,sample=400):
    idx=np.linspace(0,len(cs)-1,min(sample,len(cs))).astype(int); acc=np.zeros(768); acc2=np.zeros(768)
    for j in idx: x=np.load(EMB/f"{cs[j]['chip_id']}.npy").astype("float32"); acc+=x.mean((1,2)); acc2+=(x**2).mean((1,2))
    m=acc/len(idx); s=np.sqrt(np.maximum(acc2/len(idx)-m**2,1e-6)); return torch.tensor(m,dtype=torch.float32).view(-1,1,1),torch.tensor(s,dtype=torch.float32).view(-1,1,1)
def conv_bn(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class EmbDecoder(nn.Module):   # same family as Sen12/KuroSiwo cache heads (proj 1x1 -> 2 upsample blocks -> head), nout classes
    def __init__(s,cin=768,base=128,nout=1):
        super().__init__(); s.proj=nn.Sequential(nn.Conv2d(cin,base,1),nn.BatchNorm2d(base),nn.ReLU(inplace=True)); s.u1,s.u2=conv_bn(base,base//2),conv_bn(base//2,base//4); s.head=nn.Conv2d(base//4,nout,1)
    def forward(s,x):
        x=s.proj(x); x=s.u1(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); x=s.u2(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); return F.interpolate(s.head(x),size=(128,128),mode="bilinear",align_corners=False)
class RawUNetT(nn.Module):
    """Raw baseline for variable-length series (T=1..8): shared 2D encoder per date -> temporal max+mean pooling of features -> 2D U-Net decoder. Not the official Sen12 3D U-Net."""
    def __init__(s,cin=12,base=32,nout=1):
        super().__init__(); s.e1,s.e2,s.b=conv_bn(cin,base),conv_bn(base,base*2),conv_bn(base*2,base*4); s.d2=conv_bn(base*8+base*2*2,base*2); s.d1=conv_bn(base*2+base*2,base); s.head=nn.Conv2d(base,nout,1)
    def forward(s,x):   # B,C,T,H,W
        B,C,T,H,W=x.shape; z=x.permute(0,2,1,3,4).reshape(B*T,C,H,W); e1=s.e1(z); e2=s.e2(F.max_pool2d(e1,2)); b=s.b(F.max_pool2d(e2,2))
        def tp(f): f=f.view(B,T,*f.shape[1:]); return torch.cat([f.max(1)[0],f.mean(1)],1)
        e1,e2,b=tp(e1),tp(e2),tp(b); d2=s.d2(torch.cat([F.interpolate(b,scale_factor=2,mode="nearest"),e2],1)); d1=s.d1(torch.cat([F.interpolate(d2,scale_factor=2,mode="nearest"),e1],1)); return s.head(d1)
def exact_ap(scores,labels):
    o=np.argsort(-scores,kind="mergesort"); s=scores[o]; l=labels[o]; P=l.sum()
    if P==0: return None
    b=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]; tp=np.cumsum(l)[b]; fp=(b+1)-tp; prec=tp/np.maximum(tp+fp,1); rec=tp/P; return float(np.sum(np.diff(np.r_[0,rec])*prec))
def loss_fn(task,Y):
    if TASK[task][0]=="multi": return nn.CrossEntropyLoss(ignore_index=255)
    pos=float(Y.sum()); neg=float(Y.numel()-pos); return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(neg/max(pos,1.0),50.0),device=dev))
def train(model,task,cs,steps,bs,seed,raw,stats,lr=1e-3):
    torch.manual_seed(seed); g=np.random.default_rng(seed); model.train(); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    Yall=np.stack([target(load_mask(c),task) for c in cs]); lf=loss_fn(task,None if TASK[task][0]=="multi" else torch.from_numpy(Yall)); nbytes=0; torch.cuda.synchronize(); t0=time.perf_counter()
    small=len(cs)<=64
    if small: Xs=[load_raw_chip(c)[0] for c in cs] if raw else load_emb_batch(cs,stats)[0]
    for it in range(steps):
        idx=g.integers(0,len(cs),min(bs,len(cs)))
        if raw:
            if small: xb=[Xs[i] for i in idx]
            else:
                xb=[]
                for i in idx: x,nb=load_raw_chip(cs[i]); xb.append(x); nbytes+=nb
            T=max(x.shape[1] for x in xb); X=torch.from_numpy(np.stack([np.pad(x,((0,0),(0,T-x.shape[1]),(0,0),(0,0)),mode="edge") for x in xb]))
        else:
            if small: X=Xs[idx]
            else: X,nb=load_emb_batch([cs[i] for i in idx],stats); nbytes+=nb
        Y=torch.from_numpy(Yall[idx]); out=model(X.to(dev)); loss=lf(out,Y.to(dev)) if TASK[task][0]=="multi" else lf(out[:,0],Y.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        if it%500==0: print(f"  step {it} loss {loss.item():.4f}",flush=True)
    torch.cuda.synchronize(); model.eval(); return {"steps":steps,"gpu_s":time.perf_counter()-t0,"final_loss":float(loss.item()),"train_bytes_read":nbytes,"peak_mem_mb":torch.cuda.max_memory_allocated()/2**20}
@torch.no_grad()
def predict(model,task,cs,raw,stats,bs=32):
    P=[]; nbytes=0
    for i in range(0,len(cs),bs):
        b=cs[i:i+bs]
        if raw:
            xb=[];
            for c in b: x,nb=load_raw_chip(c); xb.append(x); nbytes+=nb
            T=max(x.shape[1] for x in xb); X=torch.from_numpy(np.stack([np.pad(x,((0,0),(0,T-x.shape[1]),(0,0),(0,0)),mode="edge") for x in xb]))
        else: X,nb=load_emb_batch(b,stats); nbytes+=nb
        o=model(X.to(dev)); P.append((torch.softmax(o,1) if TASK[task][0]=="multi" else torch.sigmoid(o[:,0])).cpu().numpy().astype("float32"))
    return np.concatenate(P), nbytes
def evaluate(task,P,cs):
    Y=np.stack([target(load_mask(c),task) for c in cs]); clus=np.array([c["cluster"] for c in cs]); res={"n_chips":len(cs),"clusters":{}}
    if TASK[task][0]=="multi":
        pred=P.argmax(1); valid=Y!=255; per_class_ap={}
        for i,code in enumerate(LC):
            yi=(Y==i)[valid]; per_class_ap[str(code)]=exact_ap(P[:,i][valid].astype("float64"),yi.astype("uint8")) if yi.any() else None
        def miou(sel):
            ious=[]
            for i in range(7):
                t=(Y[sel]==i)&valid[sel]; p=(pred[sel]==i)&valid[sel]; u=(t|p).sum()
                if t.sum()>0: ious.append((t&p).sum()/u)
            return float(np.mean(ious)) if ious else None
        for k in sorted(set(clus)): s=clus==k; res["clusters"][k]={"miou":miou(s),"n":int(s.sum())}
        v=[r["miou"] for r in res["clusters"].values() if r["miou"] is not None]; res.update({"cluster_macro_miou":float(np.mean(v)) if v else None,"pooled_miou":miou(np.ones(len(cs),bool)),"pixel_acc":float((pred[valid]==Y[valid]).mean()),"per_class_ap":per_class_ap})
    else:
        Yb=Y.astype(bool); pb=P>=0.5
        def iou(s): t=Yb[s]; p=pb[s]; u=(t|p).sum(); return float((t&p).sum()/u) if u else None
        for k in sorted(set(clus)):
            s=clus==k; npos=int(Yb[s].sum()); res["clusters"][k]={"n":int(s.sum()),"pos_px":npos,"ap":exact_ap(P[s].ravel().astype("float64"),Yb[s].ravel().astype("uint8")) if npos else None,"iou":iou(s) if npos else None}
        pc=[r["iou"] for r in res["clusters"].values() if r["pos_px"]>0 and r["iou"] is not None]
        res.update({"all_cluster_ap":exact_ap(P.ravel().astype("float64"),Yb.ravel().astype("uint8")),"iou_0p5":iou(np.ones(len(cs),bool)),"positive_cluster_macro_iou":float(np.mean(pc)) if pc else None,"positive_clusters":len(pc),"test_pos_px":int(Yb.sum())})
    return res
def draw_support(task,cs,K,seed,draw):
    g=np.random.default_rng(1000*seed+K)
    if draw=="random" or TASK[task][0]=="multi": pick=g.choice(len(cs),K,replace=False)
    else:
        code=TASK[task][1]; pos=[i for i,c in enumerate(cs) if (load_mask(c)==code).any()] if not hasattr(draw_support,"_pos") or task not in draw_support._pos else draw_support._pos[task]
        draw_support._pos=getattr(draw_support,"_pos",{}); draw_support._pos[task]=pos; npos=min(len(pos),max(1,K//2)); pick=np.concatenate([g.choice(pos,npos,replace=False),g.choice([i for i in range(len(cs)) if i not in set(pos)],K-npos,replace=False)])
    pick=np.asarray(pick).astype(int); sup=[cs[i] for i in pick]; code=TASK[task][1]; return sup,{"support_pos_chips":int(sum((load_mask(c)==code).any() for c in sup)) if TASK[task][0]=="binary" else None,"support_total_chips":len(sup),"support_ids":[c["chip_id"] for c in sup]}
train_cs,val_cs,test_cs=by_split["train"],by_split["val"],by_split["test"]; stats=emb_stats(train_cs); torch.save(stats,OUT/"emb_stats.pt")
report={"schema":"korea-3task-v1","prereg":"config/korea_shared_cache_3task_prereg_v1_amendment.json","target_rule":"last labeled acquisition per chip","n":{s:len(v) for s,v in by_split.items()},"tasks":{},"raw_model":"RawUNetT (per-date 2D encoder, temporal max+mean pooling, 2D U-Net decoder; base 32)","cache_head":"EmbDecoder(768->128, 2 upsample blocks)"}
rp=OUT/"report.json"
def save(): rp.write_text(json.dumps(report,indent=1,ensure_ascii=False))
for task in a.tasks.split(","):
    kind,nout=TASK[task][0],(7 if TASK[task][0]=="multi" else 1); R=report["tasks"].setdefault(task,{}); print("== task",task,flush=True)
    if "FULL_CACHE" in a.arms:
        torch.cuda.reset_peak_memory_stats(); m=EmbDecoder(nout=nout).to(dev); tr=train(m,task,train_cs,a.full_steps if not a.probe else 20,a.full_bs,1,False,stats); P,nb=predict(m,task,test_cs,False,stats); ev=evaluate(task,P,test_cs); Pv,_=predict(m,task,val_cs,False,stats)
        R["FULL_CACHE"]={"train":tr,"test":ev,"val":evaluate(task,Pv,val_cs),"test_bytes_read":nb}; torch.save(m.state_dict(),OUT/f"{task}_FULL_CACHE.pt"); print(task,"FULL_CACHE",json.dumps({k:v for k,v in ev.items() if k!="clusters"})[:400],flush=True); save()
    if a.full_raw and "FULL_RAW" in a.arms:
        torch.cuda.reset_peak_memory_stats(); m=RawUNetT(nout=nout).to(dev); tr=train(m,task,train_cs,a.full_steps if not a.probe else 20,a.full_bs,1,True,stats); P,nb=predict(m,task,test_cs,True,stats); ev=evaluate(task,P,test_cs)
        R["FULL_RAW"]={"train":tr,"test":ev,"test_bytes_read":nb}; torch.save(m.state_dict(),OUT/f"{task}_FULL_RAW.pt"); print(task,"FULL_RAW",json.dumps({k:v for k,v in ev.items() if k!="clusters"})[:400],flush=True); save()
    for K in [int(k) for k in a.K.split(",")]:
        for draw in a.draws.split(","):
            if draw=="posaware" and kind=="multi": continue
            for seed in [int(s) for s in a.seeds.split(",")]:
                sup,sinfo=draw_support(task,train_cs,K,seed,draw)
                for arm in [x for x in ("CACHE_K","RAW_K") if x in a.arms]:
                    key=f"{arm}_K{K}_{draw}_s{seed}"
                    if key in R: continue
                    torch.cuda.reset_peak_memory_stats(); raw=arm=="RAW_K"; m=(RawUNetT(nout=nout) if raw else EmbDecoder(nout=nout)).to(dev)
                    tr=train(m,task,sup,a.k_steps if not a.probe else 10,min(K,8),seed,raw,stats); P,nb=predict(m,task,test_cs,raw,stats); ev=evaluate(task,P,test_cs)
                    R[key]={"K":K,"draw":draw,"seed":seed,"support":sinfo,"train":tr,"test":ev,"test_bytes_read":nb}; print(key,json.dumps({k:v for k,v in ev.items() if k not in ("clusters","per_class_ap")})[:300],flush=True); save()
report["done_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); save(); print("KOREA 3TASK DONE")
