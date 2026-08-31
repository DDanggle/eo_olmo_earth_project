#!/usr/bin/env python3
"""C1 사전 sanity 스크린 — CPU 전용, 봉인 실험 아님.

질문: full cache 완성 전에, 이미 추출된 Presto 픽셀 임베딩이 산사태 픽셀을
raw 밴드 통계보다 잘 분리하는가(선형 프로브 AUPRC). 결과는 개발 판단용이며
C1 본판정(동일 decoder 3-seed)에 쓰지 않는다.

계약: train region 타일만 사용(개봉된 test 재사용 금지), 타일 단위 분리
(프로브 학습 타일과 평가 타일을 겹치지 않게).
"""
import json, random
from pathlib import Path
import numpy as np

EMB = Path("/home/work/data/olmoearth/presto_c1/holdout_chimanimani/emb_fp16")
MASK = Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani/mask_u8")
RAW = Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani/raw_u16")
FOLDS = Path("/home/work/data/olmoearth/sen12_gp_contract/loco_folds.json")
CONTRACT = Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl")
OUT = Path("/home/work/data/olmoearth/presto_c1/sanity_probe.json")
rng = random.Random(20260901)

folds = json.loads(FOLDS.read_text())
fold = next(f for f in folds["folds"] if f["fold"] == "holdout_chimanimani")
train_regions = set(fold["train_regions"])
recs = {}
for line in CONTRACT.read_text().splitlines():
    if line:
        r = json.loads(line)
        recs[r["sample_id"]] = r

avail = [p.stem for p in EMB.glob("*.npy")
         if p.stem in recs and recs[p.stem]["region"] in train_regions
         and (MASK / p.name).exists() and (RAW / p.name).exists()]
pos_tiles, neg_tiles = [], []
for sid in avail:
    m = np.load(MASK / f"{sid}.npy")
    (pos_tiles if m.any() else neg_tiles).append(sid)
rng.shuffle(pos_tiles); rng.shuffle(neg_tiles)
sel_pos, sel_neg = pos_tiles[:60], neg_tiles[:60]
half = len(sel_pos) // 2
fit_tiles = sel_pos[:half] + sel_neg[:half]
ev_tiles = sel_pos[half:2*half] + sel_neg[half:2*half]

def features(sid):
    emb = np.load(EMB / f"{sid}.npy").astype("float32")        # 128,128,128
    raw = np.load(RAW / f"{sid}.npy").astype("float32") / 1e4  # 10,12,128,128
    m = np.load(MASK / f"{sid}.npy").astype("uint8")
    fe = emb.reshape(128, -1).T                                # N,128
    fr = np.concatenate([raw.mean(1), raw.std(1)], 0).reshape(20, -1).T  # N,20
    y = (m.reshape(-1) > 0).astype("int8")
    idx = rng.sample(range(len(y)), 1500)
    return fe[idx], fr[idx], y[idx]

def collect(tiles):
    E, R, Y = [], [], []
    for s in tiles:
        e, r, y = features(s)
        E.append(e); R.append(r); Y.append(y)
    return np.concatenate(E), np.concatenate(R), np.concatenate(Y)

Ef, Rf, Yf = collect(fit_tiles)
Ee, Re, Ye = collect(ev_tiles)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

def probe(Xf, Xe):
    sc = StandardScaler().fit(Xf)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xf), Yf)
    p = clf.predict_proba(sc.transform(Xe))[:, 1]
    return {"auprc": round(float(average_precision_score(Ye, p)), 4),
            "auroc": round(float(roc_auc_score(Ye, p)), 4)}

res = {
    "schema": "presto-c1-sanity-probe-v1",
    "claim_boundary": [
        "Development sanity screen only; not the sealed C1 comparison.",
        "Pixel-level linear probe on train-region tiles; tile-disjoint fit/eval.",
        "Not comparable to positive-tile macro IoU numbers.",
    ],
    "tiles": {"fit": len(fit_tiles), "eval": len(ev_tiles),
              "pos_avail": len(pos_tiles), "neg_avail": len(neg_tiles)},
    "pixels": {"fit": int(len(Yf)), "eval": int(len(Ye)),
               "eval_pos_rate": round(float(Ye.mean()), 4)},
    "presto_emb_128d": probe(Ef, Ee),
    "raw_band_stats_20d": probe(Rf, Re),
    "prevalence_baseline_auprc": round(float(Ye.mean()), 4),
}
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(res, ensure_ascii=False, indent=2))
