"""제주 4개년 변화 탐지 v2 — 계단형(step) 변화 탐지 + 토지피복 층화.

v1의 실패: "연도 간 코사인 거리"로 재니 바다가 상위권을 독식했다.
바다는 파도·햇빛 반사 때문에 실제 변화가 없어도 매년 지문이 달라진다.

v2의 세 가지 강화:
  (1) 계단 검출  — 4개년(23/24/25/26)을 앞/뒤 두 그룹으로 자르는 3가지 분할에 대해
                   "그룹 간 차이 - 그룹 내 차이"를 계산. 한 번 바뀌고 유지되는 변화만
                   높은 점수를 받고, 매년 요동치는 잡음은 그룹 내 차이도 커서 상쇄된다.
  (2) 층화 z점수 — ESA WorldCover 클래스별로 점수를 표준화. 바다는 바다끼리,
                   숲은 숲끼리 경쟁시켜 "그 지역 유형에서 이례적인가"를 본다.
  (3) 마스킹     — nodata/무효 픽셀 제외.

출력: 계단 점수 지도, 클래스별 z점수 지도, Top 후보 목록(JSON, 분할 시점 포함).
"""

import os

if __name__ == "__main__" and os.environ.get("ALLOW_HISTORICAL_INVALID_JEJU_4TS") != "1":
    raise SystemExit(
        "REFUSED: this historical four-period path is season-confounded and the "
        "2025/rolling-2026 windows overlap. It cannot emit annual-change candidates. "
        "Set ALLOW_HISTORICAL_INVALID_JEJU_4TS=1 only to reproduce the preserved failure."
    )

import glob
import json

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter

BASE = "/home/work/data/olmoearth/embed_search/dataset/windows/default"
OUT = "/home/work/data/olmoearth/embed_search"
YEARS = {"2023": "jeju23_", "2024": "jeju_", "2025": "jeju25_", "2026": "jeju26r_"}
YEAR_ORDER = ["2023", "2024", "2025", "2026"]
PX = 40.0

# ---------- 1. 임베딩 로드 ----------
data: dict[str, dict] = {}
for year, prefix in YEARS.items():
    d = {}
    for path in glob.glob(f"{BASE}/{prefix}*/layers/embeddings/*/geotiff.tif"):
        window = path.split("/windows/default/")[1].split("/")[0]
        with rasterio.open(path) as src:
            d[window.replace(prefix, "")] = {
                "arr": src.read().astype(np.float32),
                "tr": src.transform,
                "crs": src.crs,
            }
    data[year] = d
    print(year, len(d), "windows", flush=True)

keys = sorted(set.intersection(*[set(d) for d in data.values()]))
print("matched locations:", len(keys), flush=True)

# ---------- 2. 연도별 mean-centering (그 해 공통 효과 제거) ----------
for year, d in data.items():
    mu = np.mean([w["arr"].mean(axis=(1, 2)) for w in d.values()], axis=0)[:, None, None]
    for w in d.values():
        a = w["arr"] - mu
        n = np.linalg.norm(a, axis=0, keepdims=True)
        n[n == 0] = 1
        w["arr"] = a / n
print("centered", flush=True)

# ---------- 3. WorldCover 라벨 (층화용) ----------
import planetary_computer as pc
from pystac_client import Client

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)
wc_items = list(
    catalog.search(
        collections=["esa-worldcover"],
        bbox=[126.10, 33.15, 127.00, 33.60],
        query={"esa_worldcover:product_version": {"eq": "2.0.0"}},
    ).items()
)
print("worldcover items:", len(wc_items), flush=True)

wc_by_key = {}
for k in keys:
    w = data["2024"][k]
    wc = np.zeros((256, 256), dtype=np.uint8)
    for item in wc_items:
        with rasterio.open(item.assets["map"].href) as src:
            reproject(
                rasterio.band(src, 1),
                wc,
                dst_transform=w["tr"],
                dst_crs=w["crs"],
                resampling=Resampling.nearest,
                src_transform=src.transform,
                src_crs=src.crs,
            )
    wc_by_key[k] = wc
print("worldcover rasterized", flush=True)

# ---------- 4. 계단 변화 점수 ----------
# 분할 s: [0..s]는 "이전", [s+1..3]은 "이후" (s = 0,1,2 → 2023/24/25 이후 변화)
SPLITS = [0, 1, 2]
SPLIT_LABEL = {0: "2023->2024", 1: "2024->2025", 2: "2025->2026"}


def pair_dist(a, b):
    """두 임베딩 맵 사이의 픽셀별 코사인 거리."""
    return 1.0 - np.einsum("chw,chw->hw", a, b)


step_score = {}
best_split = {}
for k in keys:
    embs = [data[y][k]["arr"] for y in YEAR_ORDER]
    scores = []
    for s in SPLITS:
        before, after = list(range(0, s + 1)), list(range(s + 1, 4))
        cross = np.mean([pair_dist(embs[i], embs[j]) for i in before for j in after], axis=0)
        within = []
        for grp in (before, after):
            for x in range(len(grp)):
                for y in range(x + 1, len(grp)):
                    within.append(pair_dist(embs[grp[x]], embs[grp[y]]))
        within_mean = np.mean(within, axis=0) if within else np.zeros_like(cross)
        # 계단 점수: 그룹 간 차이가 그룹 내 요동보다 얼마나 큰가
        scores.append(cross - within_mean)
    stack = np.stack(scores)  # (3, 256, 256)
    step_score[k] = stack.max(axis=0)
    best_split[k] = stack.argmax(axis=0).astype(np.uint8)
print("step scores computed", flush=True)

# ---------- 5. 토지피복 클래스별 z점수 ----------
all_scores = np.concatenate([step_score[k].ravel() for k in keys])
all_wc = np.concatenate([wc_by_key[k].ravel() for k in keys])
stats = {}
for cls in np.unique(all_wc):
    v = all_scores[all_wc == cls]
    if v.size >= 1000:
        stats[int(cls)] = (float(v.mean()), float(v.std() + 1e-9), int(v.size))
print("class stats (mean, std, n):", {k: (round(v[0], 4), round(v[1], 4), v[2]) for k, v in stats.items()}, flush=True)

z_by_key = {}
for k in keys:
    z = np.zeros_like(step_score[k])
    wc = wc_by_key[k]
    for cls, (m, s, _) in stats.items():
        sel = wc == cls
        if sel.any():
            z[sel] = (step_score[k][sel] - m) / s
    z_by_key[k] = z

# ---------- 6. 모자이크 + 지도 ----------
def stitch(vals):
    ws = [data["2024"][k] for k in keys]
    xs = [w["tr"].c for w in ws]
    ys = [w["tr"].f for w in ws]
    x0, y0 = min(xs), max(ys)
    W = int((max(xs) - x0) / PX) + 256
    H = int((y0 - min(ys)) / PX) + 256
    canvas = np.full((H, W), np.nan, np.float32)
    for k in keys:
        w = data["2024"][k]
        cx = int((w["tr"].c - x0) / PX)
        cy = int((y0 - w["tr"].f) / PX)
        canvas[cy : cy + 256, cx : cx + 256] = vals[k]
    return canvas, (x0, y0, PX)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw_map, geo = stitch(step_score)
z_map, _ = stitch(z_by_key)
split_map, _ = stitch({k: best_split[k].astype(np.float32) for k in keys})
wc_map, _ = stitch({k: wc_by_key[k].astype(np.float32) for k in keys})

# Top 후보: z점수 기준, 물(80)·습지(90) 제외, 3x3 평균으로 단일픽셀 잡음 억제
z_smooth = uniform_filter(np.nan_to_num(z_map, nan=-99), size=3)
land = ~np.isin(np.nan_to_num(wc_map, nan=0).astype(np.uint8), [80, 90])
z_land = np.where(land, z_smooth, -99)

tr_back = Transformer.from_crs(data["2024"][keys[0]]["crs"], "EPSG:4326", always_xy=True)
WC_NAME = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}

tops = []
seen = []
for f in np.argsort(z_land.ravel())[::-1][:20000]:
    r, c = divmod(int(f), z_land.shape[1])
    if any(abs(r - rr) < 15 and abs(c - cc) < 15 for rr, cc in seen):
        continue
    seen.append((r, c))
    x = geo[0] + c * geo[2]
    y = geo[1] - r * geo[2]
    lon, lat = tr_back.transform(x, y)
    sp = int(np.nan_to_num(split_map, nan=0)[r, c])
    cls = int(np.nan_to_num(wc_map, nan=0)[r, c])
    tops.append(
        {
            "z": round(float(z_land[r, c]), 2),
            "raw": round(float(np.nan_to_num(raw_map, nan=0)[r, c]), 3),
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "when": SPLIT_LABEL[sp],
            "landcover": WC_NAME.get(cls, str(cls)),
        }
    )
    if len(tops) >= 30:
        break

fig, axes = plt.subplots(3, 1, figsize=(15, 17))
im0 = axes[0].imshow(raw_map, cmap="inferno", vmin=0, vmax=0.4)
axes[0].set_title("v2-a: step-change score (cross-group - within-group distance)")
plt.colorbar(im0, ax=axes[0], fraction=0.02)

im1 = axes[1].imshow(z_map, cmap="inferno", vmin=-1, vmax=6)
axes[1].set_title("v2-b: same score, z-scored within each WorldCover class")
plt.colorbar(im1, ax=axes[1], fraction=0.02)

im2 = axes[2].imshow(z_land, cmap="inferno", vmin=0, vmax=6)
axes[2].set_title("v2-c: land only + Top-30 candidates (cyan)")
plt.colorbar(im2, ax=axes[2], fraction=0.02)
for r, c in seen[: len(tops)]:
    axes[2].plot(c, r, "o", ms=10, mfc="none", mec="cyan", mew=1.5)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/jeju_change_v2.png", dpi=95, bbox_inches="tight")

json.dump(
    {"class_stats": {str(k): v for k, v in stats.items()}, "top": tops},
    open(f"{OUT}/jeju_change_v2_top.json", "w"),
    indent=1,
)
for t in tops[:12]:
    print(t, flush=True)
print("DONE", flush=True)
