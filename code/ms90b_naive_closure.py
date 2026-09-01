#!/usr/bin/env python3
"""MS-90B — 등록된 naive fusion 완결 (CPU, 재학습 없음). GeoContextGate 승급 게이트의 선행 조건.

등록 항목(config/geocontextgate_promotion_gate.json · RESTART_HERE.md):
  naive 4종: average=(p2+p4)/2 · AND=min(p2,p4) · OR=max(p2,p4) · logit-mean=σ((logit p2 + logit p4)/2)
  validation calibration: 온도 스케일링(arm별, val NLL 최소화 격자탐색)을 적용한 뒤 같은 4종 재평가
  지표: positive-tile macro IoU@0.5, tie-correct AP(동점 평균 순위 기반 pixel AP)
  입력 prob/mask 파일 SHA-256 봉인. 판정 서술은 "고정 규칙 전부/일부 불충분"까지만 — 학습 gate 필요성 증명 아님.
seed 짝(1↔1,2↔2,3↔3) → 지역값은 seed 평균. 기준선: best single arm (P2, P4 중 큰 쪽, 지표별).
"""
import json, hashlib
import numpy as np
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth/confirmatory"); CACHE=Path("/home/work/data/olmoearth/sen12_pilot")
OUT=Path("/home/work/data/olmoearth/artifacts/ms90b_naive_closure"); OUT.mkdir(parents=True,exist_ok=True)
REGIONS=sorted(d.name.replace("holdout_","") for d in ROOT.glob("holdout_*"))
EPS=1e-6
def sha(p): h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def load(region, arm, seed, split):
    d=ROOT/f"holdout_{region}"/f"{arm}_seed{seed}"/"prob_maps"/f"holdout_{region}"
    idx=json.load(open(d/f"{arm}_{split}_probs_index.json")); a=np.load(d/f"{arm}_{split}_probs_u8.npy")
    return idx["sample_ids"], a.astype("float32")/255.0, [str(d/f"{arm}_{split}_probs_index.json"), str(d/f"{arm}_{split}_probs_u8.npy")]
def masks(region, ids):
    m=CACHE/f"holdout_{region}"/"mask_u8"
    return np.stack([np.load(m/f"{s}.npy") for s in ids]).astype(bool), [str(m/f"{ids[0]}.npy"), str(m/f"{ids[-1]}.npy")]
def logit(p): p=np.clip(p,EPS,1-EPS); return np.log(p/(1-p))
def sigmoid(x): return 1/(1+np.exp(-x))
def pos_tile_macro_iou(prob, y):
    ious=[]
    for i in range(len(y)):
        if not y[i].any(): continue
        p=prob[i]>=0.5; u=(p|y[i]).sum(); ious.append((p&y[i]).sum()/u if u else 0.0)
    return float(np.mean(ious)) if ious else None
def tie_correct_ap(prob, y):
    """동점을 평균 순위로 처리한 pixel AP (tie-correct): 동점 블록 내에서는 정밀도를 블록 평균으로."""
    s=prob.ravel(); t=y.ravel().astype(np.float64)
    order=np.argsort(-s, kind="mergesort"); s=s[order]; t=t[order]
    P=t.sum()
    if P==0: return None
    # 동점 블록 경계
    bounds=np.flatnonzero(np.diff(s)!=0)+1; starts=np.concatenate([[0],bounds]); ends=np.concatenate([bounds,[len(s)]])
    ap=0.0; tp_before=0.0
    for a,b in zip(starts,ends):
        k=b-a; tp_in=t[a:b].sum()
        if tp_in>0:
            # 블록 내 균등 분포 가정: i번째 양성의 기대 정밀도 적분(사다리꼴 근사)
            # precision(x) = (tp_before + tp_in * x) / (a + k * x), x∈(0,1]; 양성 밀도 균일
            xs=np.linspace(1/(2*k), 1-1/(2*k), int(k)) if k>1 else np.array([0.5])
            prec=(tp_before + tp_in*xs)/(a + k*xs)
            ap += tp_in*float(prec.mean())
        tp_before+=tp_in
    return float(ap/P)
def fit_temperature(prob, y):
    """val NLL 최소 온도(logit/T), T 격자 0.25–4.0."""
    z=logit(prob).ravel(); t=y.ravel().astype(np.float64)
    best_T,best_nll=1.0,np.inf
    for T in np.geomspace(0.25,4.0,25):
        p=np.clip(sigmoid(z/T),EPS,1-EPS)
        nll=float(-(t*np.log(p)+(1-t)*np.log(1-p)).mean())
        if nll<best_nll: best_nll,best_T=nll,float(T)
    return best_T
FUSIONS={"average":lambda a,b:(a+b)/2,"and_min":np.minimum,"or_max":np.maximum,
         "logit_mean":lambda a,b:sigmoid((logit(a)+logit(b))/2)}
rep={"schema":"ms90b-naive-closure-v1","preregistered":"registered naive set average/AND/OR/logit-mean, with and without per-arm validation temperature calibration; metrics positive-tile macro IoU@0.5 and tie-correct pixel AP; verdict limited to sufficiency of fixed rules; no claim that a learned gate is necessary or will succeed",
     "input_sha256":{}, "regions":{}}
for region in REGIONS:
    per_seed=[]
    for seed in (1,2,3):
        ids2,p2,f2=load(region,"P2",seed,"test"); ids4,p4,f4=load(region,"P4",seed,"test"); assert ids2==ids4
        y,fm=masks(region,ids2)
        v2i,v2,_=load(region,"P2",seed,"val"); v4i,v4,_=load(region,"P4",seed,"val"); assert v2i==v4i
        yv,_=masks(region,v2i)
        for f in f2+f4+fm: rep["input_sha256"].setdefault(f, sha(f))
        T2=fit_temperature(v2,yv); T4=fit_temperature(v4,yv)
        p2c=sigmoid(logit(p2)/T2); p4c=sigmoid(logit(p4)/T4)
        row={"T2":T2,"T4":T4}
        arms={"P2":p2,"P4":p4}
        for name,fn in FUSIONS.items():
            arms[name]=fn(p2,p4); arms[name+"_cal"]=fn(p2c,p4c)
        for name,pr in arms.items():
            row[name]={"pos_tile_macro_iou":pos_tile_macro_iou(pr,y),"tie_ap":tie_correct_ap(pr,y)}
        per_seed.append(row)
    agg={"T2":[s["T2"] for s in per_seed],"T4":[s["T4"] for s in per_seed]}
    keys=[k for k in per_seed[0] if isinstance(per_seed[0][k],dict)]
    for k in keys:
        agg[k]={m: float(np.mean([s[k][m] for s in per_seed])) for m in ("pos_tile_macro_iou","tie_ap")}
    best=max(agg["P2"]["pos_tile_macro_iou"], agg["P4"]["pos_tile_macro_iou"])
    for k in keys:
        if k not in ("P2","P4"): agg[k]["gain_iou_vs_best_single"]=agg[k]["pos_tile_macro_iou"]-best
    rep["regions"][region]=agg
    print(region, {k:round(agg[k]["pos_tile_macro_iou"],4) for k in keys}, flush=True)
for k in ["P2","P4"]+[n for n in FUSIONS]+[n+"_cal" for n in FUSIONS]:
    rep["macro_iou_"+k]=float(np.mean([r[k]["pos_tile_macro_iou"] for r in rep["regions"].values()]))
    rep["macro_tieap_"+k]=float(np.mean([r[k]["tie_ap"] for r in rep["regions"].values()]))
for name in list(FUSIONS)+[n+"_cal" for n in FUSIONS]:
    rep["pass_regions_"+name]=sum(1 for r in rep["regions"].values() if r[name].get("gain_iou_vs_best_single",-1)>=0.01)
(OUT/"report.json").write_text(json.dumps(rep,indent=1))
print("macro IoU:", {k.replace("macro_iou_",""):round(v,4) for k,v in rep.items() if k.startswith("macro_iou_")})
print("pass(>= +0.01, /8):", {k.replace("pass_regions_",""):v for k,v in rep.items() if k.startswith("pass_regions_")})
