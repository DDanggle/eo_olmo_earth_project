"""라벨 없는 캐시 프로브 — "이 임베딩 캐시가 downstream에서 쓸모 있을지"를 라벨 0으로 예측한다.

동기: 유사도 지표(CKA·거리상관·이웃순위)가 기능적 교환가능성을 예측하지 못한다는 것은
ML에서 이미 확립됐다(ICLR 2023 CKA reliability 외). 따라서 "지표가 틀렸다"는 기여가 아니다.
기여가 되려면 **대신 무엇이 예측하는가**를 내놔야 한다. 이 스크립트가 그 후보를 잰다.

프로브는 전부 사람 라벨 0이다. 물리 프로브는 원본 픽셀에서 공짜로 나오는 지수를 쓴다.

밴드 계약 (code/extract_sen12_fold_cache.py:33 에서 확인):
    raw_u16 = (10 bands, 12 timesteps, 128, 128)
    idx  0=B02(blue) 1=B03(green) 2=B04(red) 3=B08(NIR)
         4=B05 5=B06 6=B07 7=B8A 8=B11(SWIR1) 9=B12(SWIR2)
    B08은 인덱스 3이다 (6이 아니다).

공정성: 캐시마다 채널수가 128~3840으로 다르다. 채널이 많으면 ridge가 유리하므로
물리 프로브는 **PCA로 동일 차원(PCA_DIM)으로 줄인 뒤** 적합한다. 원차원 결과도 함께 남긴다.
"""
from __future__ import annotations
import json, sys
import numpy as np
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth")
PCA_DIM = 96
RNG_SEED = 0

# --- 밴드 인덱스 (검증됨) ---
B_BLUE, B_GREEN, B_RED, B_NIR, B_SWIR1 = 0, 1, 2, 3, 8


def physical_indices(raw: np.ndarray) -> np.ndarray:
    """raw (B,T,H,W) uint16 -> (4,H,W) float32: NDVI, NDWI, NDBI, brightness.

    시간축은 중앙값으로 집계한다(구름 완화). 임베딩도 시간 pooled 이므로 대응된다.
    """
    x = raw.astype(np.float32)
    m = np.median(x, axis=1)                      # (B,H,W)
    eps = 1e-6
    ndvi = (m[B_NIR] - m[B_RED]) / (m[B_NIR] + m[B_RED] + eps)
    ndwi = (m[B_GREEN] - m[B_NIR]) / (m[B_GREEN] + m[B_NIR] + eps)
    ndbi = (m[B_SWIR1] - m[B_NIR]) / (m[B_SWIR1] + m[B_NIR] + eps)
    bright = m.mean(0) / 10000.0
    return np.stack([ndvi, ndwi, ndbi, bright], 0).astype(np.float32)


def block_pool(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """(C,H0,W0) -> (C,h,w) 평균 풀링. H0가 h의 배수가 아니면 자른다."""
    c, H, W = a.shape
    fh, fw = H // h, W // w
    if fh < 1 or fw < 1:
        raise ValueError(f"cannot pool {a.shape} to {(h, w)}")
    a = a[:, : fh * h, : fw * w]
    return a.reshape(c, h, fh, w, fw).mean(axis=(2, 4))


# ---------------- 기하 프로브 (임베딩만) ----------------

def effective_rank(tok: np.ndarray) -> float:
    """공분산 고윳값 분포의 엔트로피 지수. 등방 가우시안이면 ~차원, rank-1이면 ~1."""
    x = tok - tok.mean(0)
    ev = np.linalg.eigvalsh(np.cov(x, rowvar=False) + 1e-12 * np.eye(x.shape[1]))
    ev = np.clip(ev, 1e-12, None); p = ev / ev.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def participation_ratio(tok: np.ndarray) -> float:
    x = tok - tok.mean(0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(x, rowvar=False)), 0, None)
    return float(ev.sum() ** 2 / (np.square(ev).sum() + 1e-12))


def anisotropy(tok: np.ndarray, rng, n: int = 4000) -> float:
    """무작위 토큰쌍 평균 코사인. 0에 가까울수록 방향이 고르게 퍼져 있다."""
    i = rng.integers(0, len(tok), n); j = rng.integers(0, len(tok), n)
    a, b = tok[i], tok[j]
    na = np.linalg.norm(a, axis=1) + 1e-8; nb = np.linalg.norm(b, axis=1) + 1e-8
    return float(np.mean((a * b).sum(1) / (na * nb)))


def dead_channel_frac(tok: np.ndarray, rel: float = 1e-3) -> float:
    s = tok.std(0); med = np.median(s) + 1e-12
    return float(np.mean(s < rel * med))


def spatial_gap(emb: np.ndarray, rng, n: int = 2000) -> float:
    """인접 토큰 코사인 - 무작위 토큰 코사인. 공간 구조가 살아 있으면 양수."""
    c, h, w = emb.shape
    f = emb.reshape(c, -1).T
    def cs(a, b):
        na = np.linalg.norm(a, axis=1) + 1e-8; nb = np.linalg.norm(b, axis=1) + 1e-8
        return float(np.mean((a * b).sum(1) / (na * nb)))
    yi = rng.integers(0, h - 1, n); xi = rng.integers(0, w, n)
    adj = cs(f[yi * w + xi], f[(yi + 1) * w + xi])
    return float(adj - anisotropy(f, rng, n))


# ---------------- 물리 프로브 (임베딩 -> 무료 지수) ----------------

def fit_pca(X: np.ndarray, dim: int):
    mu = X.mean(0)
    Xc = X - mu
    # 표본이 많으므로 공분산 고유분해가 SVD보다 싸다
    C = np.cov(Xc, rowvar=False)
    ev, V = np.linalg.eigh(C)
    V = V[:, np.argsort(ev)[::-1][:dim]]
    return mu, V


def ridge_r2(Xtr, Ytr, Xte, Yte, alpha_rel=1e-2) -> list[float]:
    """각 목표 차원의 held-out R^2. 기준은 **학습 평균**(테스트 평균이 아니라)."""
    mx, my = Xtr.mean(0), Ytr.mean(0)
    Xc, Yc = Xtr - mx, Ytr - my
    lam = alpha_rel * np.trace(Xc.T @ Xc) / Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Yc)
    P = (Xte - mx) @ W + my
    ss_res = ((Yte - P) ** 2).sum(0)
    ss_tot = ((Yte - my) ** 2).sum(0) + 1e-12      # 학습 평균 기준선
    return [float(v) for v in (1 - ss_res / ss_tot)]


def probe_cache(cache: str, n_train: int = 120, n_test: int = 40,
                per_tile: int = 64, seed: int = RNG_SEED) -> dict:
    rng = np.random.default_rng(seed)
    edir, rdir = ROOT / cache / "emb_fp16", ROOT / cache / "raw_u16"
    ids = sorted(p.stem for p in edir.glob("*.npy"))
    if not ids:
        return {"cache": cache, "error": "no embeddings"}
    rng.shuffle(ids)
    tr_ids, te_ids = ids[:n_train], ids[n_train:n_train + n_test]

    def collect(id_list):
        E, Y = [], []
        for t in id_list:
            e = np.load(edir / f"{t}.npy").astype(np.float32)          # (C,h,w)
            r = np.load(rdir / f"{t}.npy")                             # (B,T,H,W)
            phys = physical_indices(r)                                 # (4,H,W)
            ph = block_pool(phys, e.shape[1], e.shape[2])              # (4,h,w)
            n = e.shape[1] * e.shape[2]
            idx = rng.choice(n, size=min(per_tile, n), replace=False)
            E.append(e.reshape(e.shape[0], -1).T[idx])
            Y.append(ph.reshape(4, -1).T[idx])
        return np.concatenate(E, 0), np.concatenate(Y, 0)

    Etr, Ytr = collect(tr_ids)
    Ete, Yte = collect(te_ids)
    tok = Etr

    out = {
        "cache": cache, "n_train_tiles": len(tr_ids), "n_test_tiles": len(te_ids),
        "emb_dim": int(tok.shape[1]), "n_train_tokens": int(tok.shape[0]),
        "effective_rank": effective_rank(tok),
        "participation_ratio": participation_ratio(tok),
        "anisotropy": anisotropy(tok, rng),
        "dead_channel_frac": dead_channel_frac(tok),
        "spatial_gap": spatial_gap(np.load(edir / f"{tr_ids[0]}.npy").astype(np.float32), rng),
    }
    out["effective_rank_frac"] = out["effective_rank"] / out["emb_dim"]

    names = ["ndvi", "ndwi", "ndbi", "brightness"]
    r2_full = ridge_r2(Etr, Ytr, Ete, Yte)
    out["phys_r2_fulldim"] = dict(zip(names, r2_full))
    out["phys_r2_fulldim_mean"] = float(np.mean(r2_full))

    # 채널수 교란 제거: 전부 같은 차원으로 줄여서 다시
    d = min(PCA_DIM, tok.shape[1])
    mu, V = fit_pca(Etr, d)
    r2_pca = ridge_r2((Etr - mu) @ V, Ytr, (Ete - mu) @ V, Yte)
    out["pca_dim"] = d
    out["phys_r2_pca"] = dict(zip(names, r2_pca))
    out["phys_r2_pca_mean"] = float(np.mean(r2_pca))
    return out


if __name__ == "__main__":
    caches = sys.argv[1:] or ["olmo_cache_pool16"]
    res = []
    for c in caches:
        try:
            r = probe_cache(c)
        except Exception as e:                       # 한 캐시 실패가 전체를 죽이지 않게
            r = {"cache": c, "error": f"{type(e).__name__}: {e}"}
        res.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    outp = ROOT / "artifacts/cache_probes.json"
    outp.parent.mkdir(exist_ok=True)
    prev = json.loads(outp.read_text()) if outp.exists() else []
    merged = {x["cache"]: x for x in prev}
    merged.update({x["cache"]: x for x in res})
    outp.write_text(json.dumps(list(merged.values()), indent=1, ensure_ascii=False))
    print(f"wrote {outp} ({len(merged)} caches)")
