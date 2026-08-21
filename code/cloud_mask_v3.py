"""1번 대책: 구름 오염도 지도를 만들고, 깨끗한 픽셀에서만 변화를 다시 순위 매긴다.

배경: v2 계단형 탐지의 Top 후보가 육안 검증에서 전부 구름으로 판명됐다.
     (2023년 첫 모자이크가 서귀포 일대에서 구름에 먹힘)

이 스크립트가 하는 일:
  1) 각 지점·연도마다 "12장의 모자이크 중 구름에 가린 비율"을 파랑밴드 밝기로 추정
     → 40m 격자의 구름 오염도 지도 (재사용 가능한 품질 마스크, 3번 대책의 부품)
  2) 4개년 모두 깨끗한 픽셀만 남기고 계단형 변화 점수를 다시 순위 매김
  3) 마스크 전/후 Top-30 비교 + 살아남은 픽셀 비율 리포트

구름 판정: Sentinel-2 L2A 파랑밴드(B02) 반사도. 구름은 가시광 전체에서 밝다.
DN 1800 (반사도 0.18) 이상을 구름/두꺼운 헤이즈로 본다. 0(nodata)도 무효 처리.
"""

import glob
import json

import numpy as np
import rasterio

BASE = "/home/work/data/olmoearth/embed_search/dataset/windows/default"
OUT = "/home/work/data/olmoearth/embed_search"
YEARS = {"2023": "jeju23_", "2024": "jeju_", "2025": "jeju25_", "2026": "jeju26r_"}
YEAR_ORDER = ["2023", "2024", "2025", "2026"]
B02_INDEX = 2  # config.json 밴드 순서 B01,B02,... → 1-based
CLOUD_DN = 1800
BLOCK = 4  # 10m → 40m (임베딩 격자와 맞춤)
CLEAN_MAX = 0.20  # 연도별 구름 비율 상한 (이보다 높으면 오염으로 판정)


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """(H,W) → (H/k, W/k) 블록 평균."""
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


# ---------- 1. 구름 오염도 지도 ----------
cloud = {}  # cloud[year][window_key] = (256,256) float, 구름 비율 0~1
for year, prefix in YEARS.items():
    cloud[year] = {}
    for wdir in sorted(glob.glob(f"{BASE}/{prefix}*")):
        key = wdir.split("/")[-1].replace(prefix, "")
        layers = sorted(glob.glob(f"{wdir}/layers/sentinel2_l2a*"))
        fracs = []
        for layer in layers:
            tifs = glob.glob(f"{layer}/*/geotiff.tif")
            if not tifs:
                continue
            with rasterio.open(tifs[0]) as src:
                b02 = src.read(B02_INDEX)
            bad = (b02 > CLOUD_DN) | (b02 == 0)
            fracs.append(block_mean(bad.astype(np.float32), BLOCK))
        if fracs:
            cloud[year][key] = np.mean(fracs, axis=0)
    n = len(cloud[year])
    mean_frac = float(np.mean([v.mean() for v in cloud[year].values()])) if n else float("nan")
    print(f"{year}: {n} windows, mean cloud fraction {mean_frac:.3f}", flush=True)

np.savez_compressed(
    f"{OUT}/cloud_fraction.npz",
    **{f"{y}__{k}": v for y, d in cloud.items() for k, v in d.items()},
)
print("saved cloud_fraction.npz", flush=True)

# ---------- 2. 임베딩 로드 + 연도별 centering ----------
data = {}
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
keys = sorted(set.intersection(*[set(d) for d in data.values()]) & set.intersection(*[set(cloud[y]) for y in YEAR_ORDER]))
print("matched locations:", len(keys), flush=True)

for year, d in data.items():
    mu = np.mean([w["arr"].mean(axis=(1, 2)) for w in d.values()], axis=0)[:, None, None]
    for w in d.values():
        a = w["arr"] - mu
        n = np.linalg.norm(a, axis=0, keepdims=True)
        n[n == 0] = 1
        w["arr"] = a / n

# ---------- 3. 계단형 점수 + 깨끗한 픽셀 마스크 ----------
SPLIT_LABEL = {0: "2023->2024", 1: "2024->2025", 2: "2025->2026"}


def pair_dist(a, b):
    return 1.0 - np.einsum("chw,chw->hw", a, b)


step, split_idx, clean = {}, {}, {}
for k in keys:
    embs = [data[y][k]["arr"] for y in YEAR_ORDER]
    scores = []
    for s in (0, 1, 2):
        before, after = list(range(0, s + 1)), list(range(s + 1, 4))
        cross = np.mean([pair_dist(embs[i], embs[j]) for i in before for j in after], axis=0)
        within = [
            pair_dist(embs[g[x]], embs[g[y]])
            for g in (before, after)
            for x in range(len(g))
            for y in range(x + 1, len(g))
        ]
        scores.append(cross - (np.mean(within, axis=0) if within else 0.0))
    stack = np.stack(scores)
    step[k] = stack.max(axis=0)
    split_idx[k] = stack.argmax(axis=0).astype(np.uint8)
    clean[k] = np.stack([cloud[y][k] for y in YEAR_ORDER]).max(axis=0) <= CLEAN_MAX

clean_ratio = float(np.mean([c.mean() for c in clean.values()]))
print(f"clean pixels (all 4 years <= {CLEAN_MAX} cloud): {clean_ratio:.1%}", flush=True)

# ---------- 4. WorldCover 층화 z점수 (깨끗한 픽셀 통계로) ----------
import planetary_computer as pc
from pystac_client import Client
from rasterio.warp import Resampling, reproject

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
wc_items = list(
    catalog.search(
        collections=["esa-worldcover"],
        bbox=[126.10, 33.15, 127.00, 33.60],
        query={"esa_worldcover:product_version": {"eq": "2.0.0"}},
    ).items()
)
wc_by_key = {}
for k in keys:
    w = data["2024"][k]
    wc = np.zeros((256, 256), dtype=np.uint8)
    for item in wc_items:
        with rasterio.open(item.assets["map"].href) as src:
            reproject(
                rasterio.band(src, 1), wc,
                dst_transform=w["tr"], dst_crs=w["crs"], resampling=Resampling.nearest,
                src_transform=src.transform, src_crs=src.crs,
            )
    wc_by_key[k] = wc

sel_scores = np.concatenate([step[k][clean[k]].ravel() for k in keys])
sel_wc = np.concatenate([wc_by_key[k][clean[k]].ravel() for k in keys])
stats = {}
for cls in np.unique(sel_wc):
    v = sel_scores[sel_wc == cls]
    if v.size >= 1000:
        stats[int(cls)] = (float(v.mean()), float(v.std() + 1e-9), int(v.size))
print("class stats on clean pixels:", {k: (round(v[0], 4), round(v[1], 4), v[2]) for k, v in stats.items()}, flush=True)

z = {}
for k in keys:
    zz = np.full_like(step[k], -99.0)
    for cls, (m, s, _) in stats.items():
        sel = (wc_by_key[k] == cls) & clean[k]
        if sel.any():
            zz[sel] = (step[k][sel] - m) / s
    z[k] = zz

# ---------- 5. 모자이크 + Top-30 ----------
from pyproj import Transformer
from scipy.ndimage import uniform_filter

PX = 40.0
ws = [data["2024"][k] for k in keys]
x0, y0 = min(w["tr"].c for w in ws), max(w["tr"].f for w in ws)
W = int((max(w["tr"].c for w in ws) - x0) / PX) + 256
H = int((y0 - min(w["tr"].f for w in ws)) / PX) + 256


def stitch(vals, fill=np.nan):
    cv = np.full((H, W), fill, np.float32)
    for k in keys:
        w = data["2024"][k]
        cx, cy = int((w["tr"].c - x0) / PX), int((y0 - w["tr"].f) / PX)
        cv[cy : cy + 256, cx : cx + 256] = vals[k]
    return cv


z_map = stitch(z, fill=-99)
cloud_map = stitch({k: np.stack([cloud[y][k] for y in YEAR_ORDER]).max(axis=0) for k in keys})
split_map = stitch({k: split_idx[k].astype(np.float32) for k in keys}, fill=0)
wc_map = stitch({k: wc_by_key[k].astype(np.float32) for k in keys}, fill=0)

z_s = uniform_filter(np.where(z_map > -50, z_map, -99), size=3)
land = ~np.isin(np.nan_to_num(wc_map, nan=0).astype(np.uint8), [80, 90])
z_land = np.where(land & (z_map > -50), z_s, -99)

tr_back = Transformer.from_crs(data["2024"][keys[0]]["crs"], "EPSG:4326", always_xy=True)
WC_NAME = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}
tops, seen = [], []
for f in np.argsort(z_land.ravel())[::-1][:60000]:
    r, c = divmod(int(f), W)
    if z_land[r, c] < -50:
        break
    if any(abs(r - rr) < 15 and abs(c - cc) < 15 for rr, cc in seen):
        continue
    seen.append((r, c))
    lon, lat = tr_back.transform(x0 + c * PX, y0 - r * PX)
    tops.append({
        "z": round(float(z_land[r, c]), 2),
        "lat": round(lat, 4), "lon": round(lon, 4),
        "when": SPLIT_LABEL[int(split_map[r, c])],
        "landcover": WC_NAME.get(int(wc_map[r, c]), str(int(wc_map[r, c]))),
        "cloud_max": round(float(np.nan_to_num(cloud_map, nan=1)[r, c]), 3),
    })
    if len(tops) >= 30:
        break

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(15, 12))
im0 = axes[0].imshow(cloud_map, cmap="Blues", vmin=0, vmax=0.5)
axes[0].set_title("cloud contamination (max over 4 years, fraction of cloudy mosaics)")
plt.colorbar(im0, ax=axes[0], fraction=0.02)
im1 = axes[1].imshow(np.where(z_land > -50, z_land, np.nan), cmap="inferno", vmin=0, vmax=6)
axes[1].set_title(f"v3: step-change z-score on CLEAN land pixels only + Top-{len(tops)}")
plt.colorbar(im1, ax=axes[1], fraction=0.02)
for r, c in seen[: len(tops)]:
    axes[1].plot(c, r, "o", ms=10, mfc="none", mec="cyan", mew=1.5)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/jeju_change_v3.png", dpi=95, bbox_inches="tight")

json.dump(
    {"clean_ratio": clean_ratio, "cloud_threshold": CLEAN_MAX,
     "class_stats": {str(k): v for k, v in stats.items()}, "top": tops},
    open(f"{OUT}/jeju_change_v3_top.json", "w"), indent=1,
)
for t in tops[:12]:
    print(t, flush=True)
print("DONE", flush=True)
