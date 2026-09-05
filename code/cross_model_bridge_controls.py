"""교차모델 브리지 1단계의 빠진 대조군.

1단계는 사전등록 기준으로 STOP(P2 실패)이었다. 이 스크립트는 그 판정을 바꾸지 않고,
**그 신호가 사상 덕분인지 아니면 사상 없이도 이미 있었는지**를 가른다.

핵심 대조: 사상을 전혀 적용하지 않은 Clay 원본 토큰의 이웃 구조가 이미 OlmoEarth와
얼마나 일치하는가(identity control). 이미 높으면 사상은 한 일이 없다.

=== 2차 사전 등록 (1차 STOP 이후, 결과 보기 전 고정) ===
  Q1 사상 기여   : spearman(사상) - spearman(Clay 원본) >= 0.10  → 사상이 구조를 실었다
  Q2 코사인 기여 : cos(사상) - cos(무작위 사상) >= 0.10          → 선형사상 자체가 아니라 학습된 것
  Q3 상한 확인   : spearman(Clay 원본)이 이미 >= 0.85 이면, 사상 여부와 무관하게
                   "두 모델의 토큰 기하가 원래 비슷하다"가 결론이며 브리지는 불필요.
판정:
  Q3 발동 → GEOMETRY_ALREADY_ALIGNED (브리지 불필요; 그러면 왜 성능이 다른가가 새 질문)
  Q1·Q2 통과 → MAP_CARRIES_STRUCTURE (1차 P2 게이트 재설계 근거)
  그 외 → STOP 확정
"""
import json, sys
import numpy as np
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth")
SRC = sys.argv[1] if len(sys.argv) > 1 else "clay_cache_native16"
TGT = sys.argv[2] if len(sys.argv) > 2 else "olmo_cache_pool16"
HOLDOUT = sys.argv[3] if len(sys.argv) > 3 else "hiroshima"
MAX_TILES = int(sys.argv[4]) if len(sys.argv) > 4 else 300
RNG = np.random.default_rng(0)
GATES = {"Q1_map_gain_min": 0.10, "Q2_vs_random_min": 0.10, "Q3_already_aligned": 0.85}

FOLDS = ["hiroshima", "hokkaido", "indonesia", "itogon",
         "kyrgyzstan1", "kyrgyzstan2", "newzealand", "thrissur"]
sf = {f.stem: f for f in sorted((ROOT / SRC / "emb_fp16").glob("*.npy"))}
tf = {f.stem: f for f in sorted((ROOT / TGT / "emb_fp16").glob("*.npy"))}
common = sorted(set(sf) & set(tf))
fold_of = lambda t: next((f for f in FOLDS if f in t), None)
tr = [t for t in common if fold_of(t) not in (HOLDOUT, None)]
te = [t for t in common if fold_of(t) == HOLDOUT]
RNG.shuffle(tr); RNG.shuffle(te)
tr, te = tr[:MAX_TILES], te[:max(40, MAX_TILES // 4)]

def paired(ids, per_tile=48):
    xs, ys = [], []
    for t in ids:
        a = np.load(sf[t]).astype(np.float32); b = np.load(tf[t]).astype(np.float32)
        n = a.shape[1] * a.shape[2]
        idx = RNG.choice(n, size=min(per_tile, n), replace=False)
        xs.append(a.reshape(a.shape[0], -1).T[idx]); ys.append(b.reshape(b.shape[0], -1).T[idx])
    return np.concatenate(xs, 0), np.concatenate(ys, 0)

Xtr, Ytr = paired(tr); Xte, Yte = paired(te)
mx, my = Xtr.mean(0), Ytr.mean(0)
Xc, Yc = Xtr - mx, Ytr - my
lam = 1e-2 * np.trace(Xc.T @ Xc) / Xc.shape[1]
W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
P = (Xte - mx) @ W + my
Wr = RNG.normal(0, 1 / np.sqrt(Xc.shape[1]), size=W.shape).astype(np.float32)
Pr = (Xte - mx) @ Wr * (np.linalg.norm(W) / (np.linalg.norm(Wr) + 1e-8)) + my

def cos(a, b):
    return float(np.median((a * b).sum(1) /
                 ((np.linalg.norm(a, axis=1) + 1e-8) * (np.linalg.norm(b, axis=1) + 1e-8))))

idx = RNG.choice(len(Yte), size=min(400, len(Yte)), replace=False)
def pd(z):
    z = z[idx]; g = z @ z.T; d = np.diag(g)
    return (d[:, None] + d[None, :] - 2 * g)[np.triu_indices(len(idx), 1)]
def sp(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])

d_true = pd(Yte)
res = {
    "src": SRC, "tgt": TGT, "holdout": HOLDOUT, "gates": GATES,
    "spearman_mapped":   sp(pd(P), d_true),
    "spearman_identity": sp(pd(Xte), d_true),      # ← 핵심 대조: 사상 없이 Clay 원본
    "spearman_random":   sp(pd(Pr), d_true),
    "cosine_mapped": cos(P, Yte), "cosine_random": cos(Pr, Yte),
}
res["Q1_map_gain"] = res["spearman_mapped"] - res["spearman_identity"]
res["Q2_vs_random"] = res["cosine_mapped"] - res["cosine_random"]
res["Q1_pass"] = res["Q1_map_gain"] >= GATES["Q1_map_gain_min"]
res["Q2_pass"] = res["Q2_vs_random"] >= GATES["Q2_vs_random_min"]
res["Q3_triggered"] = res["spearman_identity"] >= GATES["Q3_already_aligned"]
res["verdict"] = ("GEOMETRY_ALREADY_ALIGNED" if res["Q3_triggered"] else
                  ("MAP_CARRIES_STRUCTURE" if (res["Q1_pass"] and res["Q2_pass"]) else "STOP_CONFIRMED"))
(ROOT / f"artifacts/cross_model_bridge_controls_{HOLDOUT}.json").write_text(
    json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
