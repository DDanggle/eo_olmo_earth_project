#!/usr/bin/env python3
"""GeoContextGate v1 — 타깃 지역 라벨 없이 "이 타일은 embedding arm(P4)을 믿을까, raw arm(P2)을 믿을까"를 예측.

사전 등록: config/geocontextgate_impl_v1.json (커밋 50185d1), 승급 기준: config/geocontextgate_promotion_gate.json.
- leave-one-region-out: 타깃 R 판정 시 나머지 7지역 타일로만 학습 (라벨도 source 만).
- 특징 51개: 확률 통계 12 + 일치도 6 + geo-context 33(OlmoEarth 768-d 공간평균의 source-fit PCA 32 + 공간 std 1).
- primary logistic regression(GroupKFold로 C 선택), secondary MLP.
- 변형: hard(타일별 arm 선택) / soft(alpha blend).
- 지표: positive-tile macro IoU@0.5, empty-tile FP 를 P4 seed-median 에 맞춘 작동점 IoU, tie-correct AP.
재학습 없음 — 봉인된 확률맵·임베딩 캐시만 읽는다. CPU 로 충분(GPU1 은 C1b 가 사용 중).
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np
ROOT=Path("/home/work/data/olmoearth/confirmatory"); CACHE=Path("/home/work/data/olmoearth/sen12_pilot")
OUT=Path("/home/work/data/olmoearth/artifacts/geocontextgate_v1"); OUT.mkdir(parents=True, exist_ok=True)
REGIONS=sorted(d.name.replace("holdout_","") for d in ROOT.glob("holdout_*"))
EPS=1e-6; RNG=np.random.default_rng(20260902)

def load_probs(region, arm, seed, split="test"):
    d=ROOT/f"holdout_{region}"/f"{arm}_seed{seed}"/"prob_maps"/f"holdout_{region}"
    idx=json.load(open(d/f"{arm}_{split}_probs_index.json"))
    return idx["sample_ids"], np.load(d/f"{arm}_{split}_probs_u8.npy").astype("float32")/255.0
def load_masks(region, ids):
    m=CACHE/f"holdout_{region}"/"mask_u8"
    return np.stack([np.load(m/f"{s}.npy") for s in ids]).astype(bool)
def load_emb_summary(region, ids):
    """타일별 (768 공간평균, 공간 std 채널평균). fp16 캐시를 mmap 으로 읽는다."""
    e=CACHE/f"holdout_{region}"/"emb_fp16"
    mu=np.zeros((len(ids),768),dtype="float32"); sd=np.zeros(len(ids),dtype="float32")
    for i,s in enumerate(ids):
        a=np.load(e/f"{s}.npy", mmap_mode="r")
        arr=np.asarray(a,dtype="float32")
        mu[i]=arr.mean(axis=(1,2)); sd[i]=float(arr.std(axis=(1,2)).mean())
    return mu, sd
def tile_iou(p, y, thr=0.5):
    b=p>=thr; u=(b|y).sum(); return float((b&y).sum()/u) if u else 0.0
def prob_feats(p):
    ent=-(np.clip(p,EPS,1-EPS)*np.log(np.clip(p,EPS,1-EPS))+(1-np.clip(p,EPS,1-EPS))*np.log(1-np.clip(p,EPS,1-EPS)))
    return [p.mean(), p.std(), p.max(), np.quantile(p,0.9), (p>0.25).mean(), (p>0.5).mean(), ent.mean()]
def pair_feats(p2,p4):
    a=(p2>=0.5); b=(p4>=0.5); u=(a|b).sum()
    c=np.corrcoef(p2.ravel(),p4.ravel())[0,1] if p2.std()>0 and p4.std()>0 else 0.0
    return [np.abs(p2-p4).mean(), float(np.nan_to_num(c)), float((a&b).sum()/u) if u else 0.0,
            float((a&b).mean()), float((a&~b).mean()), float((~a&b).mean())]
def build_region(region, seed):
    ids,p2=load_probs(region,"P2",seed); ids4,p4=load_probs(region,"P4",seed); assert ids==ids4
    y=load_masks(region,ids); mu,sd=load_emb_summary(region,ids)
    F=[]; iou2=[]; iou4=[]; pos=[]
    for i in range(len(ids)):
        F.append(prob_feats(p2[i])+prob_feats(p4[i])+pair_feats(p2[i],p4[i])+[sd[i]])
        iou2.append(tile_iou(p2[i],y[i])); iou4.append(tile_iou(p4[i],y[i])); pos.append(bool(y[i].any()))
    return dict(ids=ids,p2=p2,p4=p4,y=y,mu=mu,F=np.asarray(F,dtype="float32"),
                iou2=np.asarray(iou2),iou4=np.asarray(iou4),pos=np.asarray(pos))
def pca_fit(X,k=32):
    m=X.mean(0); Xc=X-m
    U,S,Vt=np.linalg.svd(Xc, full_matrices=False)
    return m, Vt[:k].T
def logreg(X,y,C,iters=400):
    """L2 로지스틱 회귀 (뉴턴 없이 경사하강 + 표준화는 호출자 책임)."""
    n,d=X.shape; w=np.zeros(d); b=0.0; lr=0.5
    for _ in range(iters):
        z=X@w+b; p=1/(1+np.exp(-z)); g=(p-y)
        gw=X.T@g/n + w/(C*n); gb=g.mean()
        w-=lr*gw; b-=lr*gb
    return w,b
def predict(X,w,b): return 1/(1+np.exp(-(X@w+b)))
def macro(vals): return float(np.mean([v for v in vals if v is not None]))
def pos_tile_iou_from_probs(P, Y, thr=0.5):
    v=[tile_iou(P[i],Y[i],thr) for i in range(len(Y)) if Y[i].any()]
    return float(np.mean(v)) if v else None
def empty_fp(P, Y, thr): return int(sum(int(((P[i]>=thr)&~Y[i]).sum()) for i in range(len(Y)) if not Y[i].any()))
def thr_for_fp_budget(P, Y, budget):
    lo,hi=0.01,0.999
    for _ in range(40):
        mid=(lo+hi)/2
        if empty_fp(P,Y,mid)>budget: lo=mid
        else: hi=mid
    return hi

data={}
for r in REGIONS:
    for s in (1,2,3):
        data[(r,s)]=build_region(r,s); print("built",r,s,flush=True)
rep={"schema":"geocontextgate-v1","preregistration":"config/geocontextgate_impl_v1.json (commit 50185d1)",
     "promotion_gate":"config/geocontextgate_promotion_gate.json","oracle_headroom_reference":0.023753,
     "per_seed":{}, "regions":{}}
for s in (1,2,3):
    res={}
    for target in REGIONS:
        src=[r for r in REGIONS if r!=target]
        # source: positive tiles only for training
        Xs=[]; ys=[]; grp=[]; MUs=[]
        for r in src:
            d=data[(r,s)]; m=d["pos"]
            Xs.append(d["F"][m]); MUs.append(d["mu"][m]); ys.append((d["iou4"][m]>d["iou2"][m]).astype("float64"))
            grp += [r]*int(m.sum())
        Xs=np.concatenate(Xs); MUs=np.concatenate(MUs); ys=np.concatenate(ys); grp=np.asarray(grp)
        pm, pv = pca_fit(MUs, 32)                       # source-fit PCA
        Xs=np.hstack([Xs, (MUs-pm)@pv])
        mu_x=Xs.mean(0); sd_x=Xs.std(0)+1e-6; Xs=(Xs-mu_x)/sd_x
        # GroupKFold(region)로 C 선택
        best=(None,-1)
        for C in (0.03,0.1,0.3,1.0,3.0):
            acc=[]
            for held in src:
                tr=grp!=held; va=~tr
                w,b=logreg(Xs[tr],ys[tr],C)
                acc.append(float((( predict(Xs[va],w,b)>=0.5)==(ys[va]>0.5)).mean()))
            a=float(np.mean(acc))
            if a>best[1]: best=(C,a)
        C=best[0]; w,b=logreg(Xs,ys,C)
        d=data[(target,s)]
        Xt=np.hstack([d["F"], (d["mu"]-pm)@pv]); Xt=(Xt-mu_x)/sd_x
        alpha=predict(Xt,w,b)
        P_hard=np.where(alpha[:,None,None]>=0.5, d["p4"], d["p2"])
        P_soft=alpha[:,None,None]*d["p4"]+(1-alpha[:,None,None])*d["p2"]
        P_avg=(d["p2"]+d["p4"])/2
        arms={"P2":d["p2"],"P4":d["p4"],"average":P_avg,"gate_hard":P_hard,"gate_soft":P_soft}
        budget=empty_fp(d["p4"], d["y"], 0.5)
        row={"C":C,"cv_acc":best[1],"alpha_mean":float(alpha.mean()),"fp_budget_p4":budget}
        for name,P in arms.items():
            thr=thr_for_fp_budget(P,d["y"],budget)
            row[name]={"iou_05":pos_tile_iou_from_probs(P,d["y"],0.5),
                       "iou_fp_matched":pos_tile_iou_from_probs(P,d["y"],thr),"thr_fp_matched":float(thr)}
        # 상한: 타일 oracle
        Po=np.where((d["iou4"]>d["iou2"])[:,None,None], d["p4"], d["p2"])
        row["oracle_tile"]={"iou_05":pos_tile_iou_from_probs(Po,d["y"],0.5)}
        res[target]=row
        print(f"seed{s} {target} C={C} acc={best[1]:.3f} P4={row['P4']['iou_05']:.4f} avg={row['average']['iou_05']:.4f} hard={row['gate_hard']['iou_05']:.4f} soft={row['gate_soft']['iou_05']:.4f} oracle={row['oracle_tile']['iou_05']:.4f}",flush=True)
    rep["per_seed"][str(s)]=res
    for k in ("P2","P4","average","gate_hard","gate_soft"):
        rep["per_seed"][str(s)]["macro_"+k]=macro([res[r][k]["iou_05"] for r in REGIONS])
        rep["per_seed"][str(s)]["macro_fpm_"+k]=macro([res[r][k]["iou_fp_matched"] for r in REGIONS])
    rep["per_seed"][str(s)]["macro_oracle"]=macro([res[r]["oracle_tile"]["iou_05"] for r in REGIONS])
    print(f"== seed{s} macro P4={rep['per_seed'][str(s)]['macro_P4']:.4f} avg={rep['per_seed'][str(s)]['macro_average']:.4f} hard={rep['per_seed'][str(s)]['macro_gate_hard']:.4f} soft={rep['per_seed'][str(s)]['macro_gate_soft']:.4f} oracle={rep['per_seed'][str(s)]['macro_oracle']:.4f}",flush=True)
    (OUT/"report.json").write_text(json.dumps(rep,indent=1))
# 승급 규칙 판정
ms={k: float(np.mean([rep["per_seed"][str(s)]["macro_"+k] for s in (1,2,3)])) for k in ("P2","P4","average","gate_hard","gate_soft")}
msf={k: float(np.mean([rep["per_seed"][str(s)]["macro_fpm_"+k] for s in (1,2,3)])) for k in ("P2","P4","average","gate_hard","gate_soft")}
best_single=max(ms["P2"],ms["P4"]); best_naive=ms["average"]
rep["macro_seedmean"]=ms; rep["macro_seedmean_fp_matched"]=msf
for v in ("gate_hard","gate_soft"):
    beats_all_seeds=all(rep["per_seed"][str(s)]["macro_"+v] > rep["per_seed"][str(s)]["macro_average"] for s in (1,2,3))
    gain=ms[v]-best_single
    rep[f"verdict_{v}"]={"beats_naive_all_3_seeds":bool(beats_all_seeds),"gain_vs_best_single":gain,
        "realized_headroom_rule":bool(gain>=0.01 or gain>=0.5*0.023753),
        "fp_matched_gain_vs_best_single":msf[v]-max(msf["P2"],msf["P4"]),
        "fp_matched_beats_naive":bool(msf[v]>msf["average"]),
        "promoted":bool(beats_all_seeds and (gain>=0.01 or gain>=0.5*0.023753) and msf[v]>msf["average"] and (msf[v]-max(msf["P2"],msf["P4"]))>=0.01)}
(OUT/"report.json").write_text(json.dumps(rep,indent=1))
print("macro seed-mean:", {k:round(v,4) for k,v in ms.items()})
print("fp-matched:", {k:round(v,4) for k,v in msf.items()})
for v in ("gate_hard","gate_soft"): print(v, rep[f"verdict_{v}"])
