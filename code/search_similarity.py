"""임베딩 유사도 검색 — 이방성 교정(v1 vs v2) 비교 + ESA WorldCover 정량 평가.

Korea Earth Search의 핵심 실험 (2026-08-20). 세션에서 즉석 실행했던 코드를 파일로 복원.

핵심 발견: 원시 임베딩은 이방성 때문에 모든 쌍의 cosine이 ~0.7에 몰려 검색이 안 된다.
전체 평균 벡터를 빼고 재정규화하면(mean-centering) 판별력이 살아난다.
평가는 WorldCover 클래스별 무작위 쿼리 5개, precision@2000 / 기저율 = 리프트.

실측 결과:
  완도  — built ×26.0, cropland ×8.3, tree ×2.6, water ×1.6
  제주  — cropland ×14.8, built ×18.3, grass ×14.1, tree ×3.6, water ×1.5

사용법 (서버):
    python search_similarity.py wando   # 그룹 접두사
    python search_similarity.py jeju
"""

import glob
import sys

import numpy as np
import rasterio

BASE = "/home/work/data/olmoearth/embed_search/dataset/windows/default"
OUT = "/home/work/data/olmoearth/embed_search"
GROUP = sys.argv[1] if len(sys.argv) > 1 else "wando"
BBOX = {
    "wando": [126.60, 34.20, 126.90, 34.45],
    "jeju": [126.10, 33.15, 127.00, 33.60],
    "jiri": [127.55, 35.25, 127.85, 35.45],
}[GROUP]
K = 2000
N_QUERIES = 5

# ---------- 임베딩 로드 ----------
tifs = sorted(glob.glob(f"{BASE}/{GROUP}*/layers/embeddings/*/geotiff.tif"))
srcs = [rasterio.open(t) for t in tifs]
print(f"{GROUP}: {len(srcs)} windows", flush=True)

vecs, transforms = [], []
for s in srcs:
    vecs.append(s.read().astype(np.float32))  # (768, 256, 256)
    transforms.append(s.transform)
C = vecs[0].shape[0]
flat = np.concatenate([v.reshape(C, -1) for v in vecs], axis=1)  # (768, N)
print("vectors:", flat.shape, flush=True)


def l2(x):
    n = np.linalg.norm(x, axis=0, keepdims=True)
    n[n == 0] = 1
    return x / n


raw = l2(flat)                        # v1: 원시 임베딩 정규화만
centered = l2(flat - flat.mean(axis=1, keepdims=True))  # v2: 이방성 교정

# ---------- WorldCover 정답지 ----------
import planetary_computer as pc
from pystac_client import Client
from rasterio.warp import Resampling, reproject

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)
items = list(
    catalog.search(
        collections=["esa-worldcover"],
        bbox=BBOX,
        query={"esa_worldcover:product_version": {"eq": "2.0.0"}},
    ).items()
)
labels = []
for s, tr in zip(srcs, transforms):
    wc = np.zeros((s.height, s.width), np.uint8)
    for it in items:
        with rasterio.open(it.assets["map"].href) as src:
            reproject(
                rasterio.band(src, 1), wc,
                dst_transform=tr, dst_crs=s.crs, resampling=Resampling.nearest,
                src_transform=src.transform, src_crs=src.crs,
            )
    labels.append(wc.ravel())
L = np.concatenate(labels)
print("worldcover classes:", dict(zip(*[x.tolist() for x in np.unique(L, return_counts=True)])), flush=True)

# ---------- precision@K 평가 ----------
NAMES = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}
rng = np.random.default_rng(0)
print(f"\n{'space':10s}{'class':9s}{'P@'+str(K):>9s}{'base':>8s}{'lift':>8s}")
for space_name, space in (("raw(v1)", raw), ("centered(v2)", centered)):
    for cls, name in NAMES.items():
        idx = np.where(L == cls)[0]
        if idx.size < 1000:
            continue
        precs = []
        for q in rng.choice(idx, N_QUERIES, replace=False):
            sim = space.T @ space[:, q]
            order = np.argsort(sim)[::-1][1 : K + 1]
            precs.append((L[order] == cls).mean())
        base = (L == cls).mean()
        mp = float(np.mean(precs))
        print(f"{space_name:10s}{name:9s}{mp:9.3f}{base:8.3f}{mp/base:8.1f}x", flush=True)
print("\nDONE")
