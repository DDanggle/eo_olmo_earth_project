"""v4: 구름 마스크를 '평균'에서 '최악의 한 장(max)' 기준으로 교정.

v3의 결함: 연중 12장의 구름 비율을 평균했더니, 1장이 완전히 구름이어도 평균은
1/12 ≈ 0.08 → 기준(0.20)을 통과했다. 그러나 모델은 12장 전부로 지문을 만들기 때문에
그 1장이 연도 임베딩을 오염시킨다. 육안 검증에서 5곳 중 3곳이 이 경로로 통과했다.

v4: 픽셀별로 `max_over_mosaics(구름 비율)`을 쓴다. 한 장만 심하게 가려도 즉시 탈락.
평균과 최댓값을 모두 저장해 두 기준을 비교할 수 있게 한다.
"""

import glob
import json
import sys

import numpy as np
import rasterio

BASE = "/home/work/data/olmoearth/embed_search/dataset/windows/default"
OUT = "/home/work/data/olmoearth/embed_search"
YEARS = {"2023": "jeju23_", "2024": "jeju_", "2025": "jeju25_", "2026": "jeju26r_"}
YEAR_ORDER = ["2023", "2024", "2025", "2026"]
B02_INDEX = 2
CLOUD_DN = 1800
BLOCK = 4
MAX_THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35  # 최악 한 장의 허용 구름 비율
TAG = sys.argv[2] if len(sys.argv) > 2 else "v4"


def block_mean(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


# ---------- 1. 픽셀별 구름 통계 (평균 + 최댓값) ----------
cmean, cmax = {}, {}
for year, prefix in YEARS.items():
    cmean[year], cmax[year] = {}, {}
    for wdir in sorted(glob.glob(f"{BASE}/{prefix}*")):
        key = wdir.split("/")[-1].replace(prefix, "")
        fracs = []
        for layer in sorted(glob.glob(f"{wdir}/layers/sentinel2_l2a*")):
            tifs = glob.glob(f"{layer}/*/geotiff.tif")
            if not tifs:
                continue
            with rasterio.open(tifs[0]) as src:
                b02 = src.read(B02_INDEX)
            fracs.append(block_mean(((b02 > CLOUD_DN) | (b02 == 0)).astype(np.float32), BLOCK))
        if fracs:
            stack = np.stack(fracs)
            cmean[year][key] = stack.mean(axis=0)
            cmax[year][key] = stack.max(axis=0)
    print(
        f"{year}: {len(cmax[year])} windows | mean-of-mean {np.mean([v.mean() for v in cmean[year].values()]):.3f}"
        f" | mean-of-max {np.mean([v.mean() for v in cmax[year].values()]):.3f}",
        flush=True,
    )

np.savez_compressed(
    f"{OUT}/cloud_stats.npz",
    **{f"mean__{y}__{k}": v for y, d in cmean.items() for k, v in d.items()},
    **{f"max__{y}__{k}": v for y, d in cmax.items() for k, v in d.items()},
)
print("saved cloud_stats.npz", flush=True)

# ---------- 2. 임베딩 + centering ----------
data = {}
for year, prefix in YEARS.items():
    d = {}
    for path in glob.glob(f"{BASE}/{prefix}*/layers/embeddings/*/geotiff.tif"):
        window = path.split("/windows/default/")[1].split("/")[0]
        with rasterio.open(path) as src:
            d[window.replace(prefix, "")] = {"arr": src.read().astype(np.float32), "tr": src.transform, "crs": src.crs}
    data[year] = d
keys = sorted(set.intersection(*[set(d) for d in data.values()]) & set.intersection(*[set(cmax[y]) for y in YEAR_ORDER]))
print("matched:", len(keys), flush=True)

for d in data.values():
    mu = np.mean([w["arr"].mean(axis=(1, 2)) for w in d.values()], axis=0)[:, None, None]
    for w in d.values():
        a = w["arr"] - mu
        n = np.linalg.norm(a, axis=0, keepdims=True)
        n[n == 0] = 1
        w["arr"] = a / n

# ---------- 3. 계단형 점수 + 엄격한 마스크 ----------
SPLIT_LABEL = {0: "2023->2024", 1: "2024->2025", 2: "2025->2026"}


def pd_(a, b):
    return 1.0 - np.einsum("chw,chw->hw", a, b)


step, split_idx, clean = {}, {}, {}
for k in keys:
    e = [data[y][k]["arr"] for y in YEAR_ORDER]
    sc = []
    for s in (0, 1, 2):
        bef, aft = list(range(0, s + 1)), list(range(s + 1, 4))
        cross = np.mean([pd_(e[i], e[j]) for i in bef for j in aft], axis=0)
        within = [pd_(e[g[x]], e[g[y]]) for g in (bef, aft) for x in range(len(g)) for y in range(x + 1, len(g))]
        sc.append(cross - (np.mean(within, axis=0) if within else 0.0))
    st = np.stack(sc)
    step[k] = st.max(axis=0)
    split_idx[k] = st.argmax(axis=0).astype(np.uint8)
    clean[k] = np.stack([cmax[y][k] for y in YEAR_ORDER]).max(axis=0) <= MAX_THRESH

ratio = float(np.mean([c.mean() for c in clean.values()]))
print(f"clean pixels (worst mosaic <= {MAX_THRESH} in all 4 years): {ratio:.1%}", flush=True)

# ---------- 4. WorldCover 층화 ----------
import planetary_computer as pc
from pystac_client import Client
from rasterio.warp import Resampling, reproject

cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
items = list(cat.search(collections=["esa-worldcover"], bbox=[126.10, 33.15, 127.00, 33.60],
                        query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
wc = {}
for k in keys:
    w = data["2024"][k]
    arr = np.zeros((256, 256), np.uint8)
    for it in items:
        with rasterio.open(it.assets["map"].href) as src:
            reproject(rasterio.band(src, 1), arr, dst_transform=w["tr"], dst_crs=w["crs"],
                      resampling=Resampling.nearest, src_transform=src.transform, src_crs=src.crs)
    wc[k] = arr

sv = np.concatenate([step[k][clean[k]].ravel() for k in keys])
sw = np.concatenate([wc[k][clean[k]].ravel() for k in keys])
stats = {int(c): (float(sv[sw == c].mean()), float(sv[sw == c].std() + 1e-9), int((sw == c).sum()))
         for c in np.unique(sw) if (sw == c).sum() >= 1000}
print("class stats:", {k: (round(v[0], 4), round(v[1], 4), v[2]) for k, v in stats.items()}, flush=True)

z = {}
for k in keys:
    zz = np.full_like(step[k], -99.0)
    for c, (m, s, _) in stats.items():
        sel = (wc[k] == c) & clean[k]
        if sel.any():
            zz[sel] = (step[k][sel] - m) / s
    z[k] = zz

# ---------- 5. 지도 + Top ----------
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
        cv[cy:cy + 256, cx:cx + 256] = vals[k]
    return cv


z_map = stitch(z, fill=-99)
worst = stitch({k: np.stack([cmax[y][k] for y in YEAR_ORDER]).max(axis=0) for k in keys})
split_map = stitch({k: split_idx[k].astype(np.float32) for k in keys}, fill=0)
wc_map = stitch({k: wc[k].astype(np.float32) for k in keys}, fill=0)
z_s = uniform_filter(np.where(z_map > -50, z_map, -99), size=3)
land = ~np.isin(np.nan_to_num(wc_map, nan=0).astype(np.uint8), [80, 90])
z_land = np.where(land & (z_map > -50), z_s, -99)

trb = Transformer.from_crs(data["2024"][keys[0]]["crs"], "EPSG:4326", always_xy=True)
NAME = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}
tops, seen = [], []
for f in np.argsort(z_land.ravel())[::-1][:80000]:
    r, c = divmod(int(f), W)
    if z_land[r, c] < -50:
        break
    if any(abs(r - rr) < 15 and abs(c - cc) < 15 for rr, cc in seen):
        continue
    seen.append((r, c))
    lon, lat = trb.transform(x0 + c * PX, y0 - r * PX)
    tops.append({"z": round(float(z_land[r, c]), 2), "lat": round(lat, 4), "lon": round(lon, 4),
                 "when": SPLIT_LABEL[int(split_map[r, c])], "landcover": NAME.get(int(wc_map[r, c]), str(int(wc_map[r, c]))),
                 "worst_mosaic_cloud": round(float(np.nan_to_num(worst, nan=1)[r, c]), 3)})
    if len(tops) >= 30:
        break

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(15, 12))
im0 = axes[0].imshow(worst, cmap="Blues", vmin=0, vmax=1)
axes[0].set_title("worst-mosaic cloud fraction (max over 12 mosaics, max over 4 years)")
plt.colorbar(im0, ax=axes[0], fraction=0.02)
im1 = axes[1].imshow(np.where(z_land > -50, z_land, np.nan), cmap="inferno", vmin=0, vmax=6)
axes[1].set_title(f"{TAG}: step-change z on strictly clean land (<= {MAX_THRESH}) + Top-{len(tops)}")
plt.colorbar(im1, ax=axes[1], fraction=0.02)
for r, c in seen[: len(tops)]:
    axes[1].plot(c, r, "o", ms=10, mfc="none", mec="cyan", mew=1.5)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/jeju_change_{TAG}.png", dpi=95, bbox_inches="tight")
json.dump({"clean_ratio": ratio, "max_thresh": MAX_THRESH,
           "class_stats": {str(k): v for k, v in stats.items()}, "top": tops},
          open(f"{OUT}/jeju_change_{TAG}_top.json", "w"), indent=1)
for t in tops[:12]:
    print(t, flush=True)
print("DONE", flush=True)
