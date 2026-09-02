#!/usr/bin/env python3
"""A1 (cache + decoder 적응) vs A4 (raw UNet3D 재학습) — target few-shot, 사전 등록 config/fewshot_a1_vs_a4_prereg_v0.json.
arms: A0, A1, A4s(scratch, floor), A4w(source-init P2, headline). --arms 로 선택. --exposure fixed 이면 updates=300*K/5.
A1/A0 는 cachetune_pt1 과 동일 정의(재계산). raw 입력 계약은 pilot 과 동일: raw_u16/10000 clamp 1.5 + 월 채널(월/11) → 11채널.
GPU1 only. 측정: FP 매칭 IoU(primary), IoU@0.5, tie-AP, trainable, GPU s, raw bytes read."""
import json, os, sys, time, hashlib, argparse, importlib.util
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
if os.environ.get("CUDA_VISIBLE_DEVICES")!="1": raise SystemExit("CUDA_VISIBLE_DEVICES must be 1")
ROOT=Path("/home/work/data/olmoearth"); CACHE=ROOT/"sen12_pilot/holdout_chimanimani"
SRC4=ROOT/"cachetune_source_p4_v1"; SRC2=ROOT/"cachetune_source_p2_v1"; PT0=ROOT/"artifacts/cachetune_pt0"
ap=argparse.ArgumentParser(); ap.add_argument("--arms",default="A0,A1,A4s"); ap.add_argument("--exposure",default="fixed_update",choices=["fixed_update","fixed_exposure"])
ap.add_argument("--out",default=str(ROOT/"artifacts/fewshot_a1_a4")); ap.add_argument("--support",default="stratified",choices=["stratified","random"],help="random: pool 에서 K 타일 균등 추출(seed 100+s), 양성 강제 없음")
ap.add_argument("--confirmatory",action="store_true",help="8 확증 지역: confirmatory/holdout_<r>/P{4,2}_seed<s>/checkpoints 사용, manifest=fewshot_confirmatory_manifests"); a=ap.parse_args()
OUT=Path(a.out); OUT.mkdir(parents=True,exist_ok=True); ARMS=a.arms.split(",")
spec=importlib.util.spec_from_file_location("ob", ROOT/"sen12_official_baselines.py"); ob=importlib.util.module_from_spec(spec); spec.loader.exec_module(ob)
sys.path.insert(0,str(ROOT/"code")); 
dev=torch.device("cuda"); EPS=1e-7; BASE_STEPS=300
FOLDS=json.loads((ROOT/"sen12_gp_contract/loco_folds.json").read_text()); CONTRACT=ROOT/"sen12_gp_contract/sample_contract.jsonl"
MONTHS={json.loads(l)["sample_id"]:json.loads(l)["months_0_11"] for l in (CACHE/"months.jsonl").read_text().splitlines() if l}
def conv_bn(i,o,k=3):
    return nn.Sequential(nn.Conv2d(i,o,k,padding=k//2),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.Conv2d(o,o,k,padding=k//2),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class EmbDecoder(nn.Module):
    def __init__(self,cin=768,base=128):
        super().__init__(); self.proj=nn.Sequential(nn.Conv2d(cin,base,1),nn.BatchNorm2d(base),nn.ReLU(inplace=True)); self.u1,self.u2=conv_bn(base,base//2),conv_bn(base//2,base//4); self.head=nn.Conv2d(base//4,1,1)
    def forward(self,x):
        x=self.proj(x); x=self.u1(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); x=self.u2(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); return F.interpolate(self.head(x),size=(128,128),mode="bilinear",align_corners=False)
def members(fold, split):
    recs=[json.loads(l) for l in CONTRACT.read_text().splitlines() if l]; regions=(fold["train_regions"] if split=="train" else [fold["val_region"]] if split=="val" else [fold["test_region"]])
    return sorted(r["sample_id"] for r in recs if r["region"] in regions and not r.get("error") and r.get("s15_eligible",True) and (CACHE/"mask_u8"/f"{r['sample_id']}.npy").exists())
def emb_stats(train_ids, sample=400):
    idx=np.linspace(0,len(train_ids)-1,min(sample,len(train_ids))).astype(int); acc=np.zeros(768); acc2=np.zeros(768); n=0
    for j in idx: x=np.load(CACHE/"emb_fp16"/f"{train_ids[j]}.npy").astype("float32"); acc+=x.mean(axis=(1,2)); acc2+=(x**2).mean(axis=(1,2)); n+=1
    mean=acc/n; var=np.maximum(acc2/n-mean**2,1e-6); return torch.tensor(mean,dtype=torch.float32).view(-1,1,1), torch.tensor(np.sqrt(var),dtype=torch.float32).view(-1,1,1)
def load_emb(ids,stats): X=np.stack([np.load(CACHE/"emb_fp16"/f"{s}.npy").astype("float32") for s in ids]); return (torch.from_numpy(X)-stats[0])/stats[1]
def load_raw(ids):
    xs=[]; nbytes=0
    for s in ids:
        p=CACHE/"raw_u16"/f"{s}.npy"; nbytes+=p.stat().st_size; x=torch.from_numpy(np.load(p).astype("float32"))/10000.0; x=x.clamp(0,1.5)  # 10,T,H,W
        m=MONTHS.get(s); T=x.shape[1]; mt=torch.tensor((m[:T] if m else [0]*T),dtype=torch.float32)
        if mt.numel()<T: mt=F.pad(mt,(0,T-mt.numel()))
        ch=(mt/11.0).view(1,T,1,1).expand(1,T,x.shape[2],x.shape[3]); xs.append(torch.cat([x,ch],0))
    return torch.stack(xs), nbytes
def load_masks(ids): return torch.from_numpy(np.stack([np.load(CACHE/"mask_u8"/f"{s}.npy") for s in ids]).astype("float32")).unsqueeze(1)
def sha_state(m): h=hashlib.sha256(); [h.update(p.detach().cpu().numpy().tobytes()) for p in m.parameters()]; return h.hexdigest()[:16]
@torch.no_grad()
def probs(model,X,bs=16):
    out=[]
    for i in range(0,len(X),bs): out.append(torch.sigmoid(model(X[i:i+bs].to(dev))).cpu())
    return torch.cat(out).squeeze(1).numpy()
def tile_iou(p,y,thr): b=p>=thr; u=(b|y).sum(); return float((b&y).sum()/u) if u else 0.0
def pos_macro_iou(P,Y,thr): v=[tile_iou(P[i],Y[i],thr) for i in range(len(Y)) if Y[i].any()]; return float(np.mean(v)) if v else None
def empty_fp(P,Y,thr): return int(sum(int(((P[i]>=thr)&~Y[i]).sum()) for i in range(len(Y)) if not Y[i].any()))
def thr_for_budget(P,Y,budget):
    lo,hi=0.001,0.999
    for _ in range(40):
        mid=(lo+hi)/2
        if empty_fp(P,Y,mid)>budget: lo=mid
        else: hi=mid
    return hi
def tie_ap(P,Y):
    s=P.ravel(); t=Y.ravel().astype(np.float64); o=np.argsort(-s,kind="mergesort"); s=s[o]; t=t[o]; Pn=t.sum()
    if Pn==0: return None
    b=np.flatnonzero(np.diff(s)!=0)+1; st=np.concatenate([[0],b]); en=np.concatenate([b,[len(s)]]); ap_=0.0; tpb=0.0
    for a_,b_ in zip(st,en):
        k=b_-a_; tpi=t[a_:b_].sum()
        if tpi>0: xs=np.linspace(1/(2*k),1-1/(2*k),int(k)) if k>1 else np.array([0.5]); ap_+=tpi*float(((tpb+tpi*xs)/(a_+k*xs)).mean())
        tpb+=tpi
    return float(ap_/Pn)
def evaluate(P,Y,budget):
    Yb=Y.astype(bool); thr=thr_for_budget(P,Yb,budget); return {"iou_fp_matched":pos_macro_iou(P,Yb,thr),"thr_fp_matched":float(thr),"iou_05":pos_macro_iou(P,Yb,0.5),"empty_fp_05":empty_fp(P,Yb,0.5),"tie_ap":tie_ap(P,Yb)}
def train(model, Xs, Ys, steps, lr, wd, seed, bn_train):
    torch.manual_seed(seed); model.train() if bn_train else model.eval()
    if not bn_train:
        for m in model.modules():
            if isinstance(m,(nn.BatchNorm2d,nn.BatchNorm3d)): m.eval()
    params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.Adam(params,lr=lr,weight_decay=wd)
    pos=float(Ys.sum()); neg=float(Ys.numel()-pos); pw=min(neg/max(pos,1.0),50.0); lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=dev))
    K=len(Xs); bs=min(K,8); g=torch.Generator().manual_seed(seed); torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(steps):
        idx=torch.randint(0,K,(bs,),generator=g); z=Xs[idx].to(dev); y=Ys[idx].to(dev); loss=lossf(model(z),y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize(); model.eval(); return {"trainable_params":sum(p.numel() for p in params),"pos_weight":pw,"gpu_s":time.perf_counter()-t0,"final_loss":float(loss.item()),"steps":steps}
rep={"schema":"fewshot-a1-a4-v2","support":a.support,"preregistration":"config/fewshot_a1_vs_a4_prereg_v0.json","exposure":a.exposure,"arms":ARMS,"runs":[]}
outfile=OUT/f"report_{a.exposure}.json"
REGIONS=("hiroshima","hokkaido","indonesia","itogon","kyrgyzstan1","kyrgyzstan2","newzealand","thrissur") if a.confirmatory else ("china","chimanimani")
MANDIR=ROOT/"artifacts/fewshot_confirmatory_manifests" if a.confirmatory else PT0
def ck_path(region,arm,seed):
    if a.confirmatory: return ROOT/"confirmatory"/f"holdout_{region}"/f"{arm}_seed{seed}"/"checkpoints"/f"holdout_{region}"/f"{arm}_best.pt"
    return (SRC4 if arm=="P4" else SRC2)/f"holdout_{region}_seed{seed}/checkpoints/holdout_{region}/{arm}_best.pt"
for region in REGIONS:
    fold=next(f for f in FOLDS["folds"] if f["fold"]==f"holdout_{region}"); stats=emb_stats(members(fold,"train"))
    man=json.loads((MANDIR/f"{region}_manifest.json").read_text()); qids=man["query"]["ids"]; Yq=load_masks(qids); Yq_np=Yq.squeeze(1).numpy()
    Xq_emb=load_emb(qids,stats); Xq_raw,raw_q_bytes=(load_raw(qids) if any(x.startswith("A4") for x in ARMS) else (None,0))
    for seed in (1,2,3):
        ck4=torch.load(ck_path(region,"P4",seed),map_location="cpu")["model_state"]
        dec0=EmbDecoder().to(dev); dec0.load_state_dict(ck4,strict=True); dec0.eval(); P0=probs(dec0,Xq_emb); budget=empty_fp(P0,Yq_np.astype(bool),0.5)
        if "A0" in ARMS:
            rep["runs"].append({"region":region,"seed":seed,"K":None,"arm":"A0","eval":evaluate(P0,Yq_np,budget),"train":{"trainable_params":0,"raw_bytes_read":0},"fp_budget":budget}); print(region,seed,"A0",round(rep["runs"][-1]["eval"]["iou_fp_matched"],4),flush=True)
        for K in (5,20):
            if a.support=="stratified":
                draw=next(x for x in man["draws"][str(K)]["draws"] if x["seed"]==seed); sids=draw["support_ids"]
            else:
                import random as _r; rng=_r.Random(100+seed+K*7); sids=sorted(rng.sample(man["support_pool"]["ids"], K))
            Ys=load_masks(sids)
            steps=BASE_STEPS if a.exposure=="fixed_update" else BASE_STEPS*K//5
            for arm in [x for x in ARMS if x!="A0" and not (x=="A4w0" and K!=5)]:
                if arm=="A1":
                    m=EmbDecoder().to(dev); m.load_state_dict(ck4,strict=True); Xs=load_emb(sids,stats); tr=train(m,Xs,Ys,steps,1e-4,1e-4,seed*1000+K,bn_train=False); P=probs(m,Xq_emb); tr["raw_bytes_read"]=0
                elif arm=="A4s":
                    m=ob.OfficialUNet3D(in_channels=11).to(dev); Xs,nb=load_raw(sids); tr=train(m,Xs,Ys,steps,1e-3,1e-4,seed*1000+K,bn_train=True); P=probs(m,Xq_raw,bs=8); tr["raw_bytes_read"]=nb+raw_q_bytes
                elif arm=="A4w0":
                    ck2=torch.load(ck_path(region,"P2",seed),map_location="cpu")["model_state"]
                    m=ob.OfficialUNet3D(in_channels=11).to(dev); m.load_state_dict(ck2,strict=True); m.eval(); P=probs(m,Xq_raw,bs=8); tr={"trainable_params":0,"gpu_s":0.0,"steps":0,"raw_bytes_read":raw_q_bytes,"pos_weight":None,"final_loss":None}
                elif arm in ("A4h","A4p"):
                    ck2=torch.load(ck_path(region,"P2",seed),map_location="cpu")["model_state"]
                    m=ob.OfficialUNet3D(in_channels=11).to(dev); m.load_state_dict(ck2,strict=True)
                    for prm in m.parameters(): prm.requires_grad_(False)
                    for prm in list(m.dec[-1].parameters())+list(m.head.parameters()): prm.requires_grad_(True)
                    if arm=="A4p":
                        for prm in m.up[-1].parameters(): prm.requires_grad_(True)
                    Xs,nb=load_raw(sids); tr=train(m,Xs,Ys,steps,1e-4,1e-4,seed*1000+K,bn_train=False); P=probs(m,Xq_raw,bs=8); tr["raw_bytes_read"]=nb+raw_q_bytes
                elif arm=="A4w":
                    ck2=torch.load(ck_path(region,"P2",seed),map_location="cpu")["model_state"]
                    m=ob.OfficialUNet3D(in_channels=11).to(dev); m.load_state_dict(ck2,strict=True); Xs,nb=load_raw(sids); tr=train(m,Xs,Ys,steps,1e-4,1e-4,seed*1000+K,bn_train=False); P=probs(m,Xq_raw,bs=8); tr["raw_bytes_read"]=nb+raw_q_bytes
                else: continue
                ev=evaluate(P,Yq_np,budget); rep["runs"].append({"region":region,"seed":seed,"K":K,"arm":arm,"eval":ev,"train":tr,"fp_budget":budget})
                print(f"{region} s{seed} K={K} {arm} iou_fpm={ev['iou_fp_matched']:.4f} iou05={ev['iou_05']:.4f} ap={ev['tie_ap']:.4f} params={tr['trainable_params']} gpu={tr['gpu_s']:.1f}s steps={steps}",flush=True)
                outfile.write_text(json.dumps(rep,indent=1)); del m; torch.cuda.empty_cache()
import collections; agg=collections.defaultdict(list)
for r in rep["runs"]: agg[(r["region"],r["K"],r["arm"])].append(r["eval"]["iou_fp_matched"])
rep["summary"]={f"{k[0]}|{k[1]}|{k[2]}":{"mean":float(np.mean(v)),"sd":float(np.std(v)),"n":len(v)} for k,v in agg.items()}
outfile.write_text(json.dumps(rep,indent=1))
for k,v in sorted(rep["summary"].items()): print(k,"%.4f ± %.4f"%(v["mean"],v["sd"]))
print("DONE")
