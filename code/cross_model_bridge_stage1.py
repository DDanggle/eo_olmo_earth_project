"""교차모델 브리지 1단계 (CPU 전용): Clay 토큰을 OlmoEarth 토큰 공간으로 옮길 수 있는가?

배경: MS-100에서 같은 모델의 릴리스 사이(v1->v1.2)는 라벨 없는 선형 사상으로 AP 97%가
복구됐다. 여기서 묻는 것은 **다른 모델 사이**에도 되는가다. 되면 Clay/Galileo 캐시를
살려 쓰는 모델 무관 방법이 되고, 안 되면 "호환성은 모델 계열 안에서만"이라는 경계가 된다.

이 단계는 디코더를 학습하지 않는다(GPU 0). 사상이 정보를 실어 나르는지만 본다.
통과해야만 2단계(디코더 학습, GPU)로 간다.

=== 사전 등록 판정 기준 (결과 보기 전에 고정) ===
holdout 폴드에서, 학습 폴드로만 적합한 ridge 사상 W에 대해:
  P1 재구성   : 예측 토큰과 실제 OlmoEarth 토큰의 중위 코사인 >= 0.50
  P2 설명력   : 토큰 차원 평균 R^2 >= 0.30
  P3 구조보존 : 예측 공간의 이웃 순위가 실제와 Spearman >= 0.50
  P4 비자명성 : 위 셋이 "타일 평균만 맞춘 것"(상수 예측) 대비 명확히 높아야 한다.
                구체적으로 코사인이 상수-예측 기준선보다 +0.10 이상 높아야 한다.
판정: P1·P2·P3 전부 통과 AND P4 통과 -> 2단계 진행.
      P4만 실패 -> "사상은 평균만 옮긴다" 로 기록하고 중단.
      P1~P3 중 하나라도 실패 -> 교차모델 브리지 실패로 기록하고 중단. 기준을 낮추지 않는다.
"""
import json, sys
import numpy as np
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth")
SRC = sys.argv[1] if len(sys.argv) > 1 else "clay_cache_native16"
TGT = sys.argv[2] if len(sys.argv) > 2 else "olmo_cache_pool16"
HOLDOUT = sys.argv[3] if len(sys.argv) > 3 else "hiroshima"
MAX_TILES = int(sys.argv[4]) if len(sys.argv) > 4 else 400   # CPU 예산
RNG = np.random.default_rng(0)

GATES = {"P1_cosine_min": 0.50, "P2_r2_min": 0.30,
         "P3_spearman_min": 0.50, "P4_margin_over_constant": 0.10}


def load(cache):
    d = ROOT / cache / "emb_fp16"
    files = sorted(d.glob("*.npy"))
    if not files:
        raise SystemExit(f"no .npy in {d}")
    return {f.stem: f for f in files}


def fold_of(tile_id, folds):
    for f in folds:
        if f in tile_id:
            return f
    return None


src_files, tgt_files = load(SRC), load(TGT)
common = sorted(set(src_files) & set(tgt_files))
if not common:
    raise SystemExit(f"타일 교집합 0 — src {len(src_files)} tgt {len(tgt_files)}")

FOLDS = ["hiroshima", "hokkaido", "indonesia", "itogon",
         "kyrgyzstan1", "kyrgyzstan2", "newzealand", "thrissur"]
test_ids = [t for t in common if fold_of(t, FOLDS) == HOLDOUT]
train_ids = [t for t in common if fold_of(t, FOLDS) not in (HOLDOUT, None)]
if not test_ids or not train_ids:
    raise SystemExit(f"폴드 분리 실패: train {len(train_ids)} test {len(test_ids)} "
                     f"(타일명 예: {common[:3]})")
RNG.shuffle(train_ids); RNG.shuffle(test_ids)
train_ids, test_ids = train_ids[:MAX_TILES], test_ids[:max(40, MAX_TILES // 4)]


def tokens(ids, files, per_tile=48):
    """타일당 토큰 몇 개를 무작위로 뽑아 (N, C) 로 편다."""
    out = []
    for t in ids:
        a = np.load(files[t]).astype(np.float32)      # (C, H, W)
        c = a.shape[0]
        flat = a.reshape(c, -1).T                     # (H*W, C)
        idx = RNG.choice(flat.shape[0], size=min(per_tile, flat.shape[0]), replace=False)
        out.append(flat[idx])
    return np.concatenate(out, 0)


# 같은 타일·같은 토큰 위치를 짝지어야 하므로 시드를 고정해 두 번 뽑는다
def paired(ids, per_tile=48):
    xs, ys = [], []
    for t in ids:
        a = np.load(src_files[t]).astype(np.float32)
        b = np.load(tgt_files[t]).astype(np.float32)
        if a.shape[1:] != b.shape[1:]:
            return None, None, f"격자 불일치 {a.shape} vs {b.shape}"
        n = a.shape[1] * a.shape[2]
        idx = RNG.choice(n, size=min(per_tile, n), replace=False)
        xs.append(a.reshape(a.shape[0], -1).T[idx])
        ys.append(b.reshape(b.shape[0], -1).T[idx])
    return np.concatenate(xs, 0), np.concatenate(ys, 0), None


Xtr, Ytr, err = paired(train_ids)
if err:
    print(json.dumps({"status": "ABORT", "reason": err}, ensure_ascii=False)); raise SystemExit(1)
Xte, Yte, _ = paired(test_ids)
print(f"train {Xtr.shape} -> {Ytr.shape} | test {Xte.shape} -> {Yte.shape}", flush=True)

mx, my = Xtr.mean(0), Ytr.mean(0)
Xc, Yc = Xtr - mx, Ytr - my
lam = 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1]
W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
P = (Xte - mx) @ W + my                     # 예측
C = np.broadcast_to(my, Yte.shape)          # 상수(평균) 기준선


def cos(a, b):
    na = np.linalg.norm(a, axis=1) + 1e-8; nb = np.linalg.norm(b, axis=1) + 1e-8
    return (a * b).sum(1) / (na * nb)


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0) + 1e-8
    return float(np.mean(1 - ss_res / ss_tot))


def rank_spearman(pred, true, n=300):
    i = RNG.choice(len(pred), size=min(n, len(pred)), replace=False)
    def pdist(z):
        z = z[i]; g = z @ z.T
        d = np.diag(g); return (d[:, None] + d[None, :] - 2 * g)[np.triu_indices(len(i), 1)]
    a, b = pdist(pred), pdist(true)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


res = {
    "src": SRC, "tgt": TGT, "holdout": HOLDOUT,
    "n_train_tiles": len(train_ids), "n_test_tiles": len(test_ids),
    "src_dim": int(Xtr.shape[1]), "tgt_dim": int(Ytr.shape[1]),
    "gates": GATES,
    "median_cosine": float(np.median(cos(P, Yte))),
    "median_cosine_constant_baseline": float(np.median(cos(C, Yte))),
    "mean_r2": r2(P, Yte),
    "mean_r2_constant_baseline": r2(C, Yte),
    "neighbor_spearman": rank_spearman(P, Yte),
}
res["P1_pass"] = res["median_cosine"] >= GATES["P1_cosine_min"]
res["P2_pass"] = res["mean_r2"] >= GATES["P2_r2_min"]
res["P3_pass"] = res["neighbor_spearman"] >= GATES["P3_spearman_min"]
res["margin_over_constant"] = res["median_cosine"] - res["median_cosine_constant_baseline"]
res["P4_pass"] = res["margin_over_constant"] >= GATES["P4_margin_over_constant"]
allp = all(res[k] for k in ("P1_pass", "P2_pass", "P3_pass", "P4_pass"))
res["verdict"] = ("PROCEED_TO_STAGE2" if allp else
                  ("MEAN_ONLY" if (res["P1_pass"] and not res["P4_pass"]) else "STOP_BRIDGE_FAILS"))

out = ROOT / f"artifacts/cross_model_bridge_{SRC}__{TGT}__{HOLDOUT}.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
