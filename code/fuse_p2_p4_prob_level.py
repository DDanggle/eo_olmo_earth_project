#!/usr/bin/env python3
"""M90 — P2(raw UNet3D)×P4(frozen OlmoEarth) 확률 수준 융합, 재학습 없음 (CPU).

M86 메커니즘 감사의 후속: tile-oracle headroom(+0.0238 macro IoU)이 "정답을 보고 arm을 고르는" 상한이었다.
여기서는 정답을 보지 않는 융합이 그 일부를 회수하는지 잰다. 봉인된 prob_maps(u8)와 mask_u8만 읽는다.

사전 등록(결과 보기 전):
  primary  : F_mean = (P2+P4)/2, 판정지표 = positive-tile macro IoU@0.5 (oracle headroom과 동일 지표).
             F_mean ≥ best(single P2,P4) + 0.01 인 지역 수 ≥ 5/8 → "융합이 headroom 일부 회수".
  secondary: F_max = max(P2,P4); F_val-α = val 확률로 α∈{0,.1,…,1} 를 골라 test 적용(αP4+(1-α)P2).
  부지표   : pooled pixel AP(AUPRC) per region.
seed 는 같은 seed 끼리 융합(1↔1,2↔2,3↔3), 지역값은 seed 평균.
"""
import json, numpy as np
from pathlib import Path
ROOT=Path("/home/work/data/olmoearth/confirmatory"); CACHE=Path("/home/work/data/olmoearth/sen12_pilot")
OUT=Path("/home/work/data/olmoearth/artifacts/fusion_prob_level"); OUT.mkdir(parents=True,exist_ok=True)
REGIONS=sorted(d.name.replace("holdout_","") for d in ROOT.glob("holdout_*"))
def load(region, arm, seed, split):
    d=ROOT/f"holdout_{region}"/f"{arm}_seed{seed}"/"prob_maps"/f"holdout_{region}"
    idx=json.load(open(d/f"{arm}_{split}_probs_index.json")); a=np.load(d/f"{arm}_{split}_probs_u8.npy")
    return idx["sample_ids"], a.astype("float32")/255.0
def masks(region, ids):
    m=CACHE/f"holdout_{region}"/"mask_u8"
    return np.stack([np.load(m/f"{s}.npy") for s in ids]).astype(bool)
def pos_tile_macro_iou(prob, y):
    pos=[i for i in range(len(y)) if y[i].any()]
    ious=[]
    for i in pos:
        p=prob[i]>=0.5; t=y[i]; u=(p|t).sum(); ious.append((p&t).sum()/u if u else 0.0)
    return float(np.mean(ious)) if ious else None
def ap(prob, y):
    s=prob.ravel(); t=y.ravel()
    o=np.argsort(-s); t=t[o]; tp=np.cumsum(t); prec=tp/np.arange(1,len(t)+1)
    return float((prec*t).sum()/max(1,t.sum()))
rep={"schema":"p2p4-prob-fusion-v1","measurement_id":"M90",
     "preregistered":"primary F_mean pos-tile macro IoU@0.5 >= best single +0.01 in >=5/8 regions; secondary F_max, val-tuned alpha; seeds matched then averaged","regions":{}}
for region in REGIONS:
    per_seed=[]
    for seed in (1,2,3):
        ids2,p2=load(region,"P2",seed,"test"); ids4,p4=load(region,"P4",seed,"test")
        assert ids2==ids4, region
        y=masks(region, ids2)
        v2i,v2=load(region,"P2",seed,"val"); v4i,v4=load(region,"P4",seed,"val"); assert v2i==v4i
        yv=masks(region, v2i)
        # val 에서 α 선택 (macro IoU 기준)
        best_a,best_v=0.5,-1
        for a_ in np.arange(0,1.01,0.1):
            v=pos_tile_macro_iou(a_*v4+(1-a_)*v2, yv)
            if v is not None and v>best_v: best_v, best_a = v, float(a_)
        row={}
        for name,pr in (("P2",p2),("P4",p4),("F_mean",(p2+p4)/2),("F_max",np.maximum(p2,p4)),("F_alpha",best_a*p4+(1-best_a)*p2)):
            row[name]={"pos_tile_macro_iou":pos_tile_macro_iou(pr,y),"pixel_ap":ap(pr,y)}
        row["alpha_from_val"]=best_a; per_seed.append(row)
    agg={}
    for name in ("P2","P4","F_mean","F_max","F_alpha"):
        agg[name]={k: float(np.mean([s[name][k] for s in per_seed])) for k in ("pos_tile_macro_iou","pixel_ap")}
    agg["alpha_from_val_per_seed"]=[s["alpha_from_val"] for s in per_seed]
    best_single=max(agg["P2"]["pos_tile_macro_iou"], agg["P4"]["pos_tile_macro_iou"])
    agg["gain_F_mean_vs_best_single"]=agg["F_mean"]["pos_tile_macro_iou"]-best_single
    rep["regions"][region]=agg
    print(region, "P2 %.4f P4 %.4f F_mean %.4f F_max %.4f F_α %.4f (α=%s) gain=%+.4f"%(
        agg["P2"]["pos_tile_macro_iou"],agg["P4"]["pos_tile_macro_iou"],agg["F_mean"]["pos_tile_macro_iou"],
        agg["F_max"]["pos_tile_macro_iou"],agg["F_alpha"]["pos_tile_macro_iou"],agg["alpha_from_val_per_seed"],agg["gain_F_mean_vs_best_single"]), flush=True)
wins=sum(1 for r in rep["regions"].values() if r["gain_F_mean_vs_best_single"]>=0.01)
rep["primary_pass_regions"]=wins; rep["primary_pass"]=wins>=5
for name in ("P2","P4","F_mean","F_max","F_alpha"):
    rep[f"macro_{name}"]=float(np.mean([r[name]["pos_tile_macro_iou"] for r in rep["regions"].values()]))
(OUT/"report.json").write_text(json.dumps(rep,indent=1))
print("macro:", {k:round(rep[k],4) for k in rep if k.startswith("macro_")}, "primary pass", rep["primary_pass"], f"({wins}/8)")
