#!/usr/bin/env python3
"""CacheTune PT-1 — exposed-region strict method gate (GPU1). 사전 등록: config/cachetune_pt0_preregistration_v0.json.

arm (모두 raw 입력 없음, cache 유효):
  A0  source decoder 동결, adapter 없음 (기준; 학습 없음)
  A1  source-초기화 decoder 전체를 support 로 적응
  A2s CacheTune-strict: decoder 동결, z' = z + U(GELU(DWConv3x3(V(LN(z))))) + scalar calibration (a·logit+b). U 는 0 초기화 → step0 = A0
  A2n 파라미터 정합 non-spatial control: DWConv3x3 → 1x1 conv(16→16)
구현 고정(결과 보기 전):
  - 표준화 z 는 source fold train 채널통계(pilot 과 동일한 400표본 linspace 근사)로 계산; adapter 는 표준화 z 위에서 동작.
  - LN: 채널축 LayerNorm, affine 없음. rank r=16.
  - 학습: 고정 300 update, batch=min(K,8) 복원추출, BCEWithLogits(pos_weight = support neg/pos, cap 50). A1 lr 1e-4 wd 1e-4 / A2 lr 1e-3 wd 0 (Adam).
    BN 통계는 모든 target arm 에서 동결(K≤20). target validation 없음, epoch 선택 없음.
  - support draw seed s ↔ source decoder seed s 짝.
  - 지표: query positive-tile macro IoU — primary 는 FP 매칭 작동점(예산 = A0 의 query empty-tile FP@0.5; 각 arm 임계를 query empty tile 에서 맞춤,
    MS-91/92 와 같은 분석 관행), 보조 IoU@0.5, tie-correct AP, empty FP, trainable params, adapter bytes, GPU s.
  - 불변식: A2 초기 logit == A0 logit (max|Δ|<1e-4), 동결 decoder 파라미터 해시 전후 동일, trainable 파라미터 수 기록.
"""
import json, hashlib, time, math, sys, os
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
if os.environ.get("CUDA_VISIBLE_DEVICES")!="1": raise SystemExit("CUDA_VISIBLE_DEVICES must be 1")
ROOT=Path("/home/work/data/olmoearth"); CACHE=ROOT/"sen12_pilot/holdout_chimanimani"; SRC=ROOT/"cachetune_source_p4_v1"
PT0=ROOT/"artifacts/cachetune_pt0"; OUT=ROOT/"artifacts/cachetune_pt1"; OUT.mkdir(parents=True,exist_ok=True)
FOLDS=json.loads((ROOT/"sen12_gp_contract/loco_folds.json").read_text()); CONTRACT=ROOT/"sen12_gp_contract/sample_contract.jsonl"
STEPS, RANK, EPS = 300, 16, 1e-7
dev=torch.device("cuda"); torch.manual_seed(0)
def sha_state(m): h=hashlib.sha256(); [h.update(p.detach().cpu().numpy().tobytes()) for p in m.parameters()]; return h.hexdigest()[:16]
def conv_bn(i,o,k=3):
    return nn.Sequential(nn.Conv2d(i,o,k,padding=k//2),nn.BatchNorm2d(o),nn.ReLU(inplace=True),nn.Conv2d(o,o,k,padding=k//2),nn.BatchNorm2d(o),nn.ReLU(inplace=True))
class EmbDecoder(nn.Module):  # pilot 과 동일 구조 (state_dict strict 로드로 검증)
    def __init__(self,cin=768,base=128):
        super().__init__(); self.proj=nn.Sequential(nn.Conv2d(cin,base,1),nn.BatchNorm2d(base),nn.ReLU(inplace=True))
        self.u1,self.u2=conv_bn(base,base//2),conv_bn(base//2,base//4); self.head=nn.Conv2d(base//4,1,1)
    def forward(self,x):
        x=self.proj(x); x=self.u1(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False)); x=self.u2(F.interpolate(x,scale_factor=2,mode="bilinear",align_corners=False))
        return F.interpolate(self.head(x),size=(128,128),mode="bilinear",align_corners=False)
class CacheAdapter(nn.Module):
    def __init__(self,c=768,r=RANK,spatial=True):
        super().__init__(); self.ln=nn.GroupNorm(1,c,affine=False)  # 채널축 정규화(위치별) — LN 대용, affine 없음
        self.V=nn.Conv2d(c,r,1); self.mid=nn.Conv2d(r,r,3,padding=1,groups=r) if spatial else nn.Conv2d(r,r,1); self.U=nn.Conv2d(r,c,1)
        nn.init.zeros_(self.U.weight); nn.init.zeros_(self.U.bias)
        self.a=nn.Parameter(torch.ones(1)); self.b=nn.Parameter(torch.zeros(1))
    def forward(self,z): return z+self.U(F.gelu(self.mid(self.V(self.ln(z)))))
    def calibrate(self,logit): return self.a*logit+self.b
def members(fold, split):
    recs=[json.loads(l) for l in CONTRACT.read_text().splitlines() if l]
    regions=(fold["train_regions"] if split=="train" else [fold["val_region"]] if split=="val" else [fold["test_region"]])
    return sorted(r["sample_id"] for r in recs if r["region"] in regions and not r.get("error") and r.get("s15_eligible",True) and (CACHE/"mask_u8"/f"{r['sample_id']}.npy").exists())
def emb_stats(train_ids, sample=400):
    idx=np.linspace(0,len(train_ids)-1,min(sample,len(train_ids))).astype(int); acc=np.zeros(768); acc2=np.zeros(768); n=0
    for j in idx:
        a=np.load(CACHE/"emb_fp16"/f"{train_ids[j]}.npy").astype("float32"); acc+=a.mean(axis=(1,2)); acc2+=(a**2).mean(axis=(1,2)); n+=1
    mean=acc/n; var=np.maximum(acc2/n-mean**2,1e-6)
    return torch.tensor(mean,dtype=torch.float32).view(-1,1,1), torch.tensor(np.sqrt(var),dtype=torch.float32).view(-1,1,1)
def load_tiles(ids, stats):
    X=np.stack([np.load(CACHE/"emb_fp16"/f"{s}.npy").astype("float32") for s in ids]); Y=np.stack([np.load(CACHE/"mask_u8"/f"{s}.npy") for s in ids]).astype("float32")
    X=(torch.from_numpy(X)-stats[0])/stats[1]; return X, torch.from_numpy(Y).unsqueeze(1)
@torch.no_grad()
def logits_of(dec, X, adapter=None, bs=32):
    out=[]
    for i in range(0,len(X),bs):
        z=X[i:i+bs].to(dev); z=adapter(z) if adapter is not None else z; l=dec(z); l=adapter.calibrate(l) if adapter is not None else l; out.append(l.cpu())
    return torch.cat(out)
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
    b=np.flatnonzero(np.diff(s)!=0)+1; st=np.concatenate([[0],b]); en=np.concatenate([b,[len(s)]]); ap=0.0; tpb=0.0
    for a_,b_ in zip(st,en):
        k=b_-a_; tpi=t[a_:b_].sum()
        if tpi>0:
            xs=np.linspace(1/(2*k),1-1/(2*k),int(k)) if k>1 else np.array([0.5]); ap+=tpi*float(((tpb+tpi*xs)/(a_+k*xs)).mean())
        tpb+=tpi
    return float(ap/Pn)
def train_arm(arm, dec_state, Xs, Ys, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dec=EmbDecoder().to(dev); dec.load_state_dict(dec_state, strict=True); dec.eval()   # BN 통계 동결(모든 arm)
    adapter=None
    if arm=="A1":
        for p in dec.parameters(): p.requires_grad_(True)
        params=list(dec.parameters()); opt=torch.optim.Adam(params,lr=1e-4,weight_decay=1e-4)
    else:
        for p in dec.parameters(): p.requires_grad_(False)
        adapter=CacheAdapter(spatial=(arm=="A2s")).to(dev); params=list(adapter.parameters()); opt=torch.optim.Adam(params,lr=1e-3,weight_decay=0.0)
    n_tr=sum(p.numel() for p in params if p.requires_grad)
    pos=float(Ys.sum()); neg=float(Ys.numel()-pos); pw=min(neg/max(pos,1.0),50.0); lossf=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw,device=dev))
    dec_hash0=sha_state(dec); K=len(Xs); bs=min(K,8); g=torch.Generator().manual_seed(seed)
    t0=time.perf_counter(); torch.cuda.synchronize()
    for step in range(STEPS):
        idx=torch.randint(0,K,(bs,),generator=g); z=Xs[idx].to(dev); y=Ys[idx].to(dev)
        z=adapter(z) if adapter is not None else z; l=dec(z); l=adapter.calibrate(l) if adapter is not None else l
        loss=lossf(l,y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize(); gpu_s=time.perf_counter()-t0
    if adapter is not None: adapter.eval(); assert sha_state(dec)==dec_hash0, "frozen decoder changed"
    ad_bytes=sum(p.numel()*4 for p in adapter.parameters()) if adapter is not None else None
    return dec, adapter, {"trainable_params":n_tr,"pos_weight":pw,"gpu_s":gpu_s,"adapter_bytes":ad_bytes,"final_loss":float(loss.item())}
def evaluate(P, Y, budget):
    Yb=Y.astype(bool); thr=thr_for_budget(P,Yb,budget)
    return {"iou_fp_matched":pos_macro_iou(P,Yb,thr),"thr_fp_matched":float(thr),"iou_05":pos_macro_iou(P,Yb,0.5),"empty_fp_05":empty_fp(P,Yb,0.5),"tie_ap":tie_ap(P,Yb)}
rep={"schema":"cachetune-pt1-v1","preregistration":"config/cachetune_pt0_preregistration_v0.json","steps":STEPS,"rank":RANK,"runs":[]}
for region in ("china","chimanimani"):
    fold=next(f for f in FOLDS["folds"] if f["fold"]==f"holdout_{region}")
    stats=emb_stats(members(fold,"train")); man=json.loads((PT0/f"{region}_manifest.json").read_text())
    Xq,Yq=load_tiles(man["query"]["ids"],stats); Yq_np=Yq.squeeze(1).numpy()
    for seed in (1,2,3):
        ck=torch.load(SRC/f"holdout_{region}_seed{seed}/checkpoints/holdout_{region}/P4_best.pt",map_location="cpu")
        dec_state={k:v for k,v in ck["model_state"].items()}
        dec0=EmbDecoder().to(dev); dec0.load_state_dict(dec_state,strict=True); dec0.eval()
        P0=torch.sigmoid(logits_of(dec0,Xq)).squeeze(1).numpy(); budget=empty_fp(P0,Yq_np.astype(bool),0.5)
        # 불변식: A2 zero-init == A0
        ad=CacheAdapter().to(dev).eval(); d=(logits_of(dec0,Xq[:8],ad)-logits_of(dec0,Xq[:8])).abs().max().item(); assert d<1e-4, f"zero-init mismatch {d}"
        r0={"region":region,"seed":seed,"K":None,"arm":"A0","eval":evaluate(P0,Yq_np,budget),"train":{"trainable_params":0},"fp_budget":budget,"zero_init_max_abs_diff":d,"source_ckpt_val_iou":ck.get("best_val_iou")}
        rep["runs"].append(r0); print(f"{region} s{seed} A0 iou_fpm={r0['eval']['iou_fp_matched']:.4f} iou05={r0['eval']['iou_05']:.4f} budget={budget}",flush=True)
        for K in (5,20):
            draw=next(x for x in man["draws"][str(K)]["draws"] if x["seed"]==seed); Xs,Ys=load_tiles(draw["support_ids"],stats)
            for arm in ("A1","A2s","A2n"):
                dec,adapter,tr=train_arm(arm,dec_state,Xs,Ys,seed*1000+K)
                P=torch.sigmoid(logits_of(dec,Xq,adapter)).squeeze(1).numpy(); ev=evaluate(P,Yq_np,budget)
                rep["runs"].append({"region":region,"seed":seed,"K":K,"arm":arm,"eval":ev,"train":tr,"fp_budget":budget,"support_pos_tiles":draw["support_pos_tiles"]})
                print(f"{region} s{seed} K={K} {arm} iou_fpm={ev['iou_fp_matched']:.4f} iou05={ev['iou_05']:.4f} ap={ev['tie_ap']:.4f} params={tr['trainable_params']} gpu={tr['gpu_s']:.1f}s",flush=True)
                (OUT/"report.json").write_text(json.dumps(rep,indent=1))
# 집계 + method gate
import collections
agg=collections.defaultdict(list)
for r in rep["runs"]: agg[(r["region"],r["K"],r["arm"])].append(r["eval"]["iou_fp_matched"])
summary={}
for (region,K,arm),v in agg.items(): summary[f"{region}|{K}|{arm}"]={"iou_fpm_mean":float(np.mean(v)),"iou_fpm_sd":float(np.std(v)),"n":len(v)}
rep["summary"]=summary
gate={}
for K in (5,20):
    diffs={reg: summary[f"{reg}|{K}|A2s"]["iou_fpm_mean"]-summary[f"{reg}|{K}|A1"]["iou_fpm_mean"] for reg in ("china","chimanimani")}
    pa=[r["train"]["trainable_params"] for r in rep["runs"] if r["arm"]=="A1"][0]; pb=[r["train"]["trainable_params"] for r in rep["runs"] if r["arm"]=="A2s"][0]
    rule1=all(dv>=0.01 for dv in diffs.values()); rule2=all(dv>=-0.01 for dv in diffs.values()) and pa>=5*pb
    gate[str(K)]={"A2s_minus_A1":diffs,"rule1_plus001_both_regions":rule1,"rule2_within001_and_5x_fewer_params":rule2,"params_A1":pa,"params_A2s":pb,"pass":bool(rule1 or rule2)}
rep["method_gate"]=gate
(OUT/"report.json").write_text(json.dumps(rep,indent=1))
for k,v in summary.items(): print(k, round(v["iou_fpm_mean"],4), "±", round(v["iou_fpm_sd"],4))
print("METHOD GATE:", json.dumps(gate)); print("PT1 DONE")
