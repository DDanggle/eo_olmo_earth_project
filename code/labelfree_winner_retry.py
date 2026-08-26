#!/usr/bin/env python3
"""M40 재시도 — disagreement 특징 + 공간 블록 CV로 라벨 없는 승자 예측. CPU 전용.

M40의 실패 조건 두 개를 고친다:
  (1) 특징이 빈약했음(mean_prob·pred_px 뿐) → 이제 확률맵이 있으므로
      **arm 간 불일치(disagreement)** 를 쓸 수 있다. 앙상블 문헌에서 가장 강한
      label-free 신호다.
  (2) in-sample 임계값 탐색이었음 → **5.12 km 공간 블록 5-fold CV**로 바꾼다.
      규칙은 train 블록에서만 학습하고 held-out 블록에서 평가한다.

"label-free"의 뜻: 배포 시점에 새 라벨이 필요 없다는 것. 개발 라벨로 예측기를
학습하는 것은 정당하다(source/development label 사용).

사전 등록 판정: held-out 정확도가 다수결(prevalence max)을 **+5%p 이상** 넘고
5-fold 전부에서 다수결 이상이어야 "예측 가능" 신호로 인정. 아니면 M40 실패 유지.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

BASE = pathlib.Path("/home/work/data/olmoearth")
PROB = BASE / "probmaps_eval"
MASK = BASE / "sen12_pilot/holdout_chimanimani/mask_u8"
COORD = BASE / "gp_official_bundle/tile_coords.json"
OUT = BASE / "gp_official_bundle/labelfree_winner_retry.json"
BLOCK_M, NFOLD, SEED = 5120.0, 5, 20260826


def load(arm):
    d = PROB / arm / "prob_maps" / "holdout_chimanimani"
    idx = json.loads((d / f"{arm}_test_probs_index.json").read_text())
    return idx["sample_ids"], np.load(d / f"{arm}_test_probs_u8.npy", mmap_mode="r")


def main():
    sA, pA = load("P2")
    sB, pB = load("P4c")
    assert sA == sB
    coords = json.loads(COORD.read_text())

    rows = []
    for i, s in enumerate(sA):
        m = np.load(MASK / f"{s}.npy") > 0
        a = np.asarray(pA[i], dtype="float32") / 255.0
        b = np.asarray(pB[i], dtype="float32") / 255.0
        da, db = a >= 0.5, b >= 0.5
        ia = (da & m).sum() / max((da | m).sum(), 1)
        ib = (db & m).sum() / max((db | m).sum(), 1)
        if not (m.any() or da.any() or db.any()) or ia == ib:
            continue  # 승자 정의 불가/동률
        inter = (da & db).sum(); union = (da | db).sum()
        rows.append({
            "sid": s, "y": 1 if ib > ia else 0,     # 1 = P4c 승
            # ── 특징 (전부 라벨 무관) ──
            "f_disagree_iou": 1.0 - (inter / max(union, 1)),
            "f_disagree_px": float(np.mean(da != db)),
            "f_prob_l1": float(np.mean(np.abs(a - b))),
            "f_meanp_a": float(a.mean()), "f_meanp_b": float(b.mean()),
            "f_predpx_a": float(da.mean()), "f_predpx_b": float(db.mean()),
            "f_maxp_a": float(a.max()), "f_maxp_b": float(b.max()),
            "f_conf_a": float(np.mean(np.abs(a - 0.5))),
            "f_conf_b": float(np.mean(np.abs(b - 0.5))),
            "x": coords[s][0], "y_coord": coords[s][1]})
    feats = [k for k in rows[0] if k.startswith("f_")]
    X = np.array([[r[k] for k in feats] for r in rows])
    y = np.array([r["y"] for r in rows])
    key = (np.floor(np.array([r["x"] for r in rows]) / BLOCK_M).astype(np.int64) * 1_000_003
           + np.floor(np.array([r["y_coord"] for r in rows]) / BLOCK_M).astype(np.int64))
    uniq = np.unique(key)
    rng = np.random.default_rng(SEED)
    fold_of_block = {b: i % NFOLD for i, b in enumerate(rng.permutation(uniq))}
    fold = np.array([fold_of_block[k] for k in key])

    def logreg(Xtr, ytr, Xte, iters=3000, lr=0.1):
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        w = np.zeros(Xtr.shape[1]); b = 0.0
        for _ in range(iters):
            z = Xtr @ w + b; p = 1 / (1 + np.exp(-z))
            g = Xtr.T @ (p - ytr) / len(ytr); gb = float(np.mean(p - ytr))
            w -= lr * (g + 1e-3 * w); b -= lr * gb
        return (1 / (1 + np.exp(-(Xte @ w + b))) > 0.5).astype(int), w

    accs, maj_accs, weights = [], [], []
    for f in range(NFOLD):
        tr, te = fold != f, fold == f
        if te.sum() == 0:
            continue
        pred, w = logreg(X[tr], y[tr], X[te])
        accs.append(float((pred == y[te]).mean()))
        maj = int(round(y[tr].mean()))                 # train 다수 클래스
        maj_accs.append(float((y[te] == maj).mean()))
        weights.append(w)

    wbar = np.mean(weights, axis=0)
    res = {"schema": "labelfree-winner-retry-v1",
           "evidence_status": "development_only_not_confirmatory",
           "pair": ["P2 (seed1)", "P4c (seed1)"],
           "n_decidable": int(len(y)), "prevalence_p4c_wins": round(float(y.mean()), 4),
           "cv": {"folds": NFOLD, "block_km": BLOCK_M / 1000,
                  "acc_per_fold": [round(a, 4) for a in accs],
                  "majority_per_fold": [round(a, 4) for a in maj_accs],
                  "acc_mean": round(float(np.mean(accs)), 4),
                  "majority_mean": round(float(np.mean(maj_accs)), 4),
                  "lift": round(float(np.mean(accs) - np.mean(maj_accs)), 4),
                  "beats_majority_all_folds": bool(all(a >= m for a, m in zip(accs, maj_accs)))},
           "feature_weights_mean": {k: round(float(v), 3) for k, v in zip(feats, wbar)},
           "preregistered_pass_rule": "lift >= +0.05 이고 전 fold에서 다수결 이상",
           "verdict": None}
    res["verdict"] = ("예측 가능 신호 있음"
                      if res["cv"]["lift"] >= 0.05 and res["cv"]["beats_majority_all_folds"]
                      else "M40 실패 유지 — 이 특징으로도 부족")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
