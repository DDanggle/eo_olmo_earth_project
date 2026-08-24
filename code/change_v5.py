"""Historical v5 analysis — retained as a failed equivalence experiment.

2026-08-22 audit: under rslearn 0.1.13, MOSAIC+period_duration and deprecated
PER_PERIOD_MOSAIC+period_duration use the same handler. The 2,592 ordered source
groups, exhaustive B02 quality metrics, and deterministic pixel samples matched.
Do not interpret this script's output as a cloud-compositing improvement.

배경 (GOAL.md 실패 계보):
  v1 연도간 거리 → 바다 독식 / v2 계단형+층화 → 육안검증 5/5 구름
  v3 구름 평균 마스킹 → 3/5 구름 잔존 / v4 최악모자이크 마스킹 → 생존 1.2%(사후 마스킹 불가)
  ⇒ 당시에는 합성 레시피 차이라고 가정했으나, 후속 감사에서 설정 별칭으로 기각됨.

이 스크립트가 하는 일:
  1) v2 데이터셋(embed_jeju_v2)으로 계단형 변화 점수 + WorldCover 층화 z점수 → Top-30
  2) 같은 좌표의 구름 통계를 v1/v2 두 데이터셋에서 계산해 비교 (합성 레시피 효과 정량화)
  3) v1 데이터의 Top-30과 v2의 Top-30이 얼마나 겹치는지 (Jaccard) → "입력 스키마가 결론을 바꾼다"의 증거
"""

import os

if __name__ == "__main__" and os.environ.get("ALLOW_HISTORICAL_INVALID_JEJU_4TS") != "1":
    raise SystemExit(
        "REFUSED: this historical four-period path is season-confounded and the "
        "2025/rolling-2026 windows overlap. It is retained only for failure reproduction. "
        "Set ALLOW_HISTORICAL_INVALID_JEJU_4TS=1 to run it explicitly."
    )

import glob
import json

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter

V1 = "/home/work/data/olmoearth/embed_search/dataset/windows/default"
V2 = "/home/work/data/olmoearth/embed_jeju_v2/dataset/windows/default"
OUT = "/home/work/data/olmoearth/embed_jeju_v2"
# v1은 그룹명이 jeju_/jeju23_/..., v2는 jeju24_/jeju23_/...
PREFIX = {"v1": {"2023": "jeju23_", "2024": "jeju_", "2025": "jeju25_", "2026": "jeju26r_"},
          "v2": {"2023": "jeju23_", "2024": "jeju24_", "2025": "jeju25_", "2026": "jeju26r_"}}
YEARS = ["2023", "2024", "2025", "2026"]
SPLIT_LABEL = {0: "2023->2024", 1: "2024->2025", 2: "2025->2026"}
PX, B02, CLOUD_DN, BLOCK = 40.0, 2, 1800, 4


def block_mean(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def load_embeddings(base, prefix_map):
    data = {}
    for year, pref in prefix_map.items():
        d = {}
        for p in glob.glob(f"{base}/{pref}*/layers/embeddings/*/geotiff.tif"):
            win = p.split("/windows/default/")[1].split("/")[0]
            with rasterio.open(p) as s:
                d[win.replace(pref, "")] = {"arr": s.read().astype(np.float32), "tr": s.transform, "crs": s.crs}
        data[year] = d
    return data


def center(data):
    for d in data.values():
        mu = np.mean([w["arr"].mean(axis=(1, 2)) for w in d.values()], axis=0)[:, None, None]
        for w in d.values():
            a = w["arr"] - mu
            n = np.linalg.norm(a, axis=0, keepdims=True)
            n[n == 0] = 1
            w["arr"] = a / n


def cloud_stats(base, prefix_map, keys):
    """픽셀별 (평균, 최댓값) 구름 비율 — 합성 레시피 효과 비교용."""
    mean_, max_ = {}, {}
    for year, pref in prefix_map.items():
        mean_[year], max_[year] = {}, {}
        for k in keys:
            wdir = f"{base}/{pref}{k}"
            fr = []
            for layer in sorted(glob.glob(f"{wdir}/layers/sentinel2_l2a*")):
                tifs = glob.glob(f"{layer}/*/geotiff.tif")
                if not tifs:
                    continue
                with rasterio.open(tifs[0]) as s:
                    b = s.read(B02)
                fr.append(block_mean(((b > CLOUD_DN) | (b == 0)).astype(np.float32), BLOCK))
            if fr:
                st = np.stack(fr)
                mean_[year][k], max_[year][k] = st.mean(axis=0), st.max(axis=0)
    return mean_, max_


def step_scores(data, keys):
    step, split = {}, {}
    for k in keys:
        e = [data[y][k]["arr"] for y in YEARS]
        sc = []
        for s in (0, 1, 2):
            bef, aft = list(range(0, s + 1)), list(range(s + 1, 4))
            cross = np.mean([1 - np.einsum("chw,chw->hw", e[i], e[j]) for i in bef for j in aft], axis=0)
            wi = [1 - np.einsum("chw,chw->hw", e[g[x]], e[g[y]])
                  for g in (bef, aft) for x in range(len(g)) for y in range(x + 1, len(g))]
            sc.append(cross - (np.mean(wi, axis=0) if wi else 0.0))
        st = np.stack(sc)
        step[k] = st.max(axis=0)
        split[k] = st.argmax(axis=0).astype(np.uint8)
    return step, split


def worldcover(data, keys):
    import planetary_computer as pc
    from pystac_client import Client
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(cat.search(collections=["esa-worldcover"], bbox=[126.10, 33.15, 127.00, 33.60],
                            query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    out = {}
    for k in keys:
        w = data["2024"][k]
        arr = np.zeros(w["arr"].shape[1:], np.uint8)
        for it in items:
            with rasterio.open(it.assets["map"].href) as s:
                reproject(rasterio.band(s, 1), arr, dst_transform=w["tr"], dst_crs=w["crs"],
                          resampling=Resampling.nearest, src_transform=s.transform, src_crs=s.crs)
        out[k] = arr
    return out


NAME = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}


def rank_top(data, keys, step, split, wc, n=30):
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

    # 클래스별 z점수
    sv = np.concatenate([step[k].ravel() for k in keys])
    sw = np.concatenate([wc[k].ravel() for k in keys])
    stats = {int(c): (float(sv[sw == c].mean()), float(sv[sw == c].std() + 1e-9))
             for c in np.unique(sw) if (sw == c).sum() >= 1000}
    z = {}
    for k in keys:
        zz = np.zeros_like(step[k])
        for c, (m, s) in stats.items():
            sel = wc[k] == c
            if sel.any():
                zz[sel] = (step[k][sel] - m) / s
        z[k] = zz

    z_map, wc_map, sp_map = stitch(z, -99), stitch({k: wc[k].astype(np.float32) for k in keys}, 0), stitch({k: split[k].astype(np.float32) for k in keys}, 0)
    z_s = uniform_filter(np.where(z_map > -50, z_map, -99), size=3)
    land = ~np.isin(np.nan_to_num(wc_map, nan=0).astype(np.uint8), [80, 90])
    z_land = np.where(land & (z_map > -50), z_s, -99)

    trb = Transformer.from_crs(data["2024"][keys[0]]["crs"], "EPSG:4326", always_xy=True)
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
                     "when": SPLIT_LABEL[int(sp_map[r, c])],
                     "landcover": NAME.get(int(wc_map[r, c]), str(int(wc_map[r, c])))})
        if len(tops) >= n:
            break
    return tops, z_land, seen, stats, (x0, y0)


# ================= 실행 =================
print("=== v2 데이터 로드 (PER_PERIOD_MOSAIC) ===", flush=True)
d2 = load_embeddings(V2, PREFIX["v2"])
for y in YEARS:
    print(f"  {y}: {len(d2[y])} windows", flush=True)
keys2 = sorted(set.intersection(*[set(d2[y]) for y in YEARS]))
print("  matched:", len(keys2), flush=True)
center(d2)
step2, split2 = step_scores(d2, keys2)
wc2 = worldcover(d2, keys2)
top2, zland2, seen2, stats2, geo2 = rank_top(d2, keys2, step2, split2, wc2)
print("  class stats:", {k: (round(v[0], 4), round(v[1], 4)) for k, v in stats2.items()}, flush=True)
print("  Top-10:", flush=True)
for t in top2[:10]:
    print("   ", t, flush=True)
import collections
print("  when:", dict(collections.Counter(t["when"] for t in top2)),
      "| landcover:", dict(collections.Counter(t["landcover"] for t in top2)), flush=True)

print("\n=== 통제 실험: 합성 레시피가 구름 오염을 얼마나 줄였나 ===", flush=True)
cm1, cx1 = cloud_stats(V1, PREFIX["v1"], keys2)
cm2, cx2 = cloud_stats(V2, PREFIX["v2"], keys2)
rows = []
for y in YEARS:
    if y in cm1 and cm1[y] and y in cm2 and cm2[y]:
        m1 = float(np.mean([v.mean() for v in cm1[y].values()]))
        M1 = float(np.mean([v.mean() for v in cx1[y].values()]))
        m2 = float(np.mean([v.mean() for v in cm2[y].values()]))
        M2 = float(np.mean([v.mean() for v in cx2[y].values()]))
        rows.append({"year": y, "v1_mean": round(m1, 3), "v2_mean": round(m2, 3),
                     "v1_worst": round(M1, 3), "v2_worst": round(M2, 3)})
        print(f"  {y}: mean {m1:.3f} -> {m2:.3f} | worst-mosaic {M1:.3f} -> {M2:.3f}", flush=True)

print("\n=== 통제 실험: Top-30 목록이 얼마나 뒤바뀌나 ===", flush=True)
d1 = load_embeddings(V1, PREFIX["v1"])
keys1 = sorted(set.intersection(*[set(d1[y]) for y in YEARS]) & set(keys2))
center(d1)
step1, split1 = step_scores(d1, keys1)
wc1 = worldcover(d1, keys1)
top1, _, _, _, _ = rank_top(d1, keys1, step1, split1, wc1)


def near(a, b, tol=0.01):
    return abs(a["lat"] - b["lat"]) < tol and abs(a["lon"] - b["lon"]) < tol


inter = sum(1 for a in top1 if any(near(a, b) for b in top2))
jac = inter / (len(top1) + len(top2) - inter) if (top1 or top2) else 0
print(f"  v1 Top-30 ∩ v2 Top-30 = {inter}곳 | Jaccard = {jac:.3f}", flush=True)
print(f"  v1 when: {dict(collections.Counter(t['when'] for t in top1))}", flush=True)
print(f"  v2 when: {dict(collections.Counter(t['when'] for t in top2))}", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(15, 7))
im = ax.imshow(np.where(zland2 > -50, zland2, np.nan), cmap="inferno", vmin=0, vmax=6)
for r, c in seen2[:len(top2)]:
    ax.plot(c, r, "o", ms=10, mfc="none", mec="cyan", mew=1.5)
ax.set_title("v5: step-change z (PER_PERIOD_MOSAIC data, no post-hoc masking) + Top-30")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.02)
plt.tight_layout()
plt.savefig(f"{OUT}/jeju_change_v5.png", dpi=95, bbox_inches="tight")

json.dump({"top": top2, "class_stats": {str(k): v for k, v in stats2.items()},
           "cloud_comparison": rows,
           "control": {"intersection": inter, "jaccard": round(jac, 3),
                       "v1_top": top1[:10], "v2_top": top2[:10]}},
          open(f"{OUT}/jeju_change_v5_top.json", "w"), indent=1)
print("\nDONE", flush=True)
