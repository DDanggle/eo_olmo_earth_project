"""양식장 프로토타입 검색 — few-shot 몽타주 쿼리 + 교차지역 전이 평가.

단일 픽셀 쿼리는 "바다 일반"에만 반응해 의미 특이성이 없었다(제주 데모 3번).
해결: 실제 양식장 여러 곳의 임베딩 평균 = "양식장 벡터"(프로토타입).

실측 결과 (2026-08-21):
  완도 held-out 20곳  — 유사도 백분위 중앙값 100.0 (75%가 상위 1% 내)
  제주 교차지역 9곳   — 백분위 96.0~99.8 (9/9), 해상 김양식 → 육상 수조 유형 전이 성립
  한계: 제주 히트맵이 "해안 일반"에도 반응 → 특이도 불완전. OSM 라벨 커버리지 편향 →
        어장정보도 전수 폴리곤으로 재평가 필요.

사용법 (서버, osm_aqua_*.json을 embed_search/에 먼저 업로드):
    python farm_prototype.py
"""

import glob
import json

import numpy as np
import rasterio
from pyproj import Transformer

BASE = "/home/work/data/olmoearth/embed_search/dataset/windows/default"
OUT = "/home/work/data/olmoearth/embed_search"
SEED = 42

# ---------- 임베딩 로드 + 전역 mean-centering (단일 검색 공간) ----------
wins = []
for t in sorted(glob.glob(f"{BASE}/*/layers/embeddings/*/geotiff.tif")):
    group = t.split("/windows/default/")[1].split("_")[0]
    with rasterio.open(t) as s:
        wins.append({"grp": group, "arr": s.read().astype(np.float32), "tr": s.transform, "crs": s.crs})
print("windows:", len(wins), "| groups:", sorted({w["grp"] for w in wins}), flush=True)

mu = np.mean([w["arr"].mean(axis=(1, 2)) for w in wins], axis=0)[:, None, None]
for w in wins:
    a = w["arr"] - mu
    n = np.linalg.norm(a, axis=0, keepdims=True)
    n[n == 0] = 1
    w["arr"] = a / n

_TR: dict[str, Transformer] = {}


def locate(lon: float, lat: float):
    """좌표 → (윈도우 인덱스, row, col). 못 찾으면 None."""
    for i, w in enumerate(wins):
        key = str(w["crs"])
        if key not in _TR:
            _TR[key] = Transformer.from_crs("EPSG:4326", w["crs"], always_xy=True)
        x, y = _TR[key].transform(lon, lat)
        col, row = ~w["tr"] * (x, y)
        if 0 <= row < w["arr"].shape[1] and 0 <= col < w["arr"].shape[2]:
            return i, int(row), int(col)
    return None


def dedup_locate(points: list[tuple[float, float]]):
    """(lat,lon) 목록 → 중복 제거된 픽셀 위치 목록."""
    out, seen = [], set()
    for lat, lon in points:
        loc = locate(lon, lat)
        if loc and loc not in seen:
            seen.add(loc)
            out.append(loc)
    return out


wando_pts = json.load(open(f"{OUT}/osm_aqua_wando.json"))
jeju_pts = json.load(open(f"{OUT}/osm_aqua_jeju.json"))
wando_locs = dedup_locate(wando_pts)
jeju_locs = dedup_locate(jeju_pts)
print(f"wando farm pixels: {len(wando_locs)} | jeju: {len(jeju_locs)}", flush=True)

# ---------- 프로토타입: 완도 절반으로 만들고 나머지는 평가용 ----------
rng = np.random.default_rng(SEED)
rng.shuffle(wando_locs)
half = len(wando_locs) // 2
proto_locs, hold_locs = wando_locs[:half], wando_locs[half:]
proto = np.mean([wins[i]["arr"][:, r, c] for i, r, c in proto_locs], axis=0)
proto /= np.linalg.norm(proto)
print(f"prototype from {len(proto_locs)} farms | held-out {len(hold_locs)}", flush=True)


def region_sims(group: str):
    idxs = [i for i, w in enumerate(wins) if w["grp"] == group]
    sims = {i: np.einsum("chw,c->hw", wins[i]["arr"], proto) for i in idxs}
    allv = np.concatenate([sims[i].ravel() for i in idxs])
    return sims, allv


def percentiles(sims, allv, locs):
    return [float((allv < sims[i][r, c]).mean() * 100) for i, r, c in locs if i in sims]


sims_w, all_w = region_sims("wando")
pr_hold = percentiles(sims_w, all_w, hold_locs)
print(
    f"WANDO held-out ({len(pr_hold)}): median pct {np.median(pr_hold):.2f} | >=99th {np.mean(np.array(pr_hold) >= 99):.2f}",
    flush=True,
)

sims_j, all_j = region_sims("jeju")
pr_j = percentiles(sims_j, all_j, jeju_locs)
print(f"JEJU cross-region ({len(pr_j)}): {[round(p, 1) for p in pr_j]}", flush=True)

# ---------- 지도 ----------
PX = 40.0


def stitch(sims):
    ws = [wins[i] for i in sims]
    x0, y0 = min(w["tr"].c for w in ws), max(w["tr"].f for w in ws)
    W = int((max(w["tr"].c for w in ws) - x0) / PX) + 256
    H = int((y0 - min(w["tr"].f for w in ws)) / PX) + 256
    cv = np.full((H, W), np.nan, np.float32)
    for i, w in zip(sims, ws):
        cx, cy = int((w["tr"].c - x0) / PX), int((y0 - w["tr"].f) / PX)
        cv[cy : cy + 256, cx : cx + 256] = sims[i]
    return cv, (x0, y0, PX)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, (sims, locs, title) in zip(
    axes,
    [(sims_w, hold_locs, "Wando (held-out farms)"), (sims_j, jeju_locs, "Jeju cross-region (OSM farms)")],
):
    cv, geo = stitch(sims)
    im = ax.imshow(cv, cmap="magma", vmin=-0.1, vmax=0.95)
    for i, r, c in locs:
        if i not in sims:
            continue
        x, y = wins[i]["tr"] * (c, r)
        ax.plot((x - geo[0]) / geo[2], (geo[1] - y) / geo[2], "o", ms=7, mfc="none", mec="cyan", mew=1.5)
    ax.set_title(f"aquaculture prototype -> {title}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.03)
plt.tight_layout()
plt.savefig(f"{OUT}/farm_query.png", dpi=100, bbox_inches="tight")
print("saved farm_query.png\nDONE")
