"""v6: 12타임스텝 임베딩으로 변화탐지 + 4 vs 12 통제실험.

v1~v5 뒤 확인한 새 요인: 기존 모델 설정은 12개 모자이크 중 저장 순서상 앞 4개만
사용했다. ``items.json`` 실측상 이 순서는 역시간순이며 달력연도는 대체로 9~12월,
rolling-2026은 대체로 3~6월이다. 따라서 4 vs 12 차이는 타임스텝 수·계절 범위의 효과를
보여주지만, 4개년 변화 자체는 계절 정렬 전까지 확정하지 않는다.

이 스크립트:
  1) `embeddings_t12`(12타임스텝)로 계단형 변화 점수 + WorldCover 층화 z → Top-30
  2) `embeddings`(4타임스텝)와 Top-30 목록 비교 (Jaccard, 시점 분포)
     → "입력 타임스텝 수가 변화탐지 결론을 얼마나 바꾸는가"의 정량
  3) 두 지도를 나란히 그려 저장
"""

import os

if __name__ == "__main__" and os.environ.get("ALLOW_HISTORICAL_INVALID_JEJU_V6") != "1":
    raise SystemExit(
        "REFUSED: v6 mixes a season-confounded four-period diagnostic with a "
        "2025/rolling-2026 pair that overlaps by 184 days; even the 12-period month "
        "sets are not fully aligned. Use this file only to reproduce the preserved "
        "failure by setting ALLOW_HISTORICAL_INVALID_JEJU_V6=1. Build a new "
        "non-overlapping, season-aligned manifest before candidate generation."
    )

import glob, json, collections
import numpy as np, rasterio
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter

BASE = "/home/work/data/olmoearth/embed_jeju_v2/dataset/windows/default"
OUT = "/home/work/data/olmoearth/embed_jeju_v2"
PREFIX = {"2023": "jeju23_", "2024": "jeju24_", "2025": "jeju25_", "2026": "jeju26r_"}
YEARS = ["2023", "2024", "2025", "2026"]
SPLIT = {0: "2023->2024", 1: "2024->2025", 2: "2025->2026"}
NAME = {10: "tree", 20: "shrub", 30: "grass", 40: "crop", 50: "built", 60: "bare", 80: "water", 90: "wetland"}
PX = 40.0

def load(layer):
    data = {}
    for y, pref in PREFIX.items():
        d = {}
        for p in glob.glob(f"{BASE}/{pref}*/layers/{layer}/*/geotiff.tif"):
            w = p.split("/windows/default/")[1].split("/")[0]
            with rasterio.open(p) as s:
                d[w.replace(pref, "")] = {"arr": s.read().astype(np.float32), "tr": s.transform, "crs": s.crs}
        data[y] = d
    keys = sorted(set.intersection(*[set(data[y]) for y in YEARS]))
    for d in data.values():
        mu = np.mean([w["arr"].mean(axis=(1, 2)) for w in d.values()], axis=0)[:, None, None]
        for w in d.values():
            a = w["arr"] - mu
            n = np.linalg.norm(a, axis=0, keepdims=True); n[n == 0] = 1
            w["arr"] = a / n
    return data, keys

def analyze(data, keys, wc):
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
        st = np.stack(sc); step[k] = st.max(axis=0); split[k] = st.argmax(axis=0).astype(np.uint8)
    ws = [data["2024"][k] for k in keys]
    x0, y0 = min(w["tr"].c for w in ws), max(w["tr"].f for w in ws)
    W = int((max(w["tr"].c for w in ws) - x0) / PX) + 256
    H = int((y0 - min(w["tr"].f for w in ws)) / PX) + 256
    def stitch(v, fill=np.nan):
        cv = np.full((H, W), fill, np.float32)
        for k in keys:
            w = data["2024"][k]
            cx, cy = int((w["tr"].c - x0) / PX), int((y0 - w["tr"].f) / PX)
            cv[cy:cy+256, cx:cx+256] = v[k]
        return cv
    sv = np.concatenate([step[k].ravel() for k in keys]); sw = np.concatenate([wc[k].ravel() for k in keys])
    stats = {int(c): (float(sv[sw==c].mean()), float(sv[sw==c].std()+1e-9)) for c in np.unique(sw) if (sw==c).sum()>=1000}
    z = {}
    for k in keys:
        zz = np.zeros_like(step[k])
        for c,(m,s) in stats.items():
            sel = wc[k]==c
            if sel.any(): zz[sel] = (step[k][sel]-m)/s
        z[k] = zz
    zm, wm, sm = stitch(z,-99), stitch({k:wc[k].astype(np.float32) for k in keys},0), stitch({k:split[k].astype(np.float32) for k in keys},0)
    zs = uniform_filter(np.where(zm>-50, zm, -99), size=3)
    land = ~np.isin(np.nan_to_num(wm,nan=0).astype(np.uint8),[80,90])
    zl = np.where(land & (zm>-50), zs, -99)
    trb = Transformer.from_crs(data["2024"][keys[0]]["crs"], "EPSG:4326", always_xy=True)
    tops, seen = [], []
    for f in np.argsort(zl.ravel())[::-1][:80000]:
        r,c = divmod(int(f), W)
        if zl[r,c] < -50: break
        if any(abs(r-rr)<15 and abs(c-cc)<15 for rr,cc in seen): continue
        seen.append((r,c))
        lon,lat = trb.transform(x0+c*PX, y0-r*PX)
        tops.append({"z":round(float(zl[r,c]),2),"lat":round(lat,4),"lon":round(lon,4),
                     "when":SPLIT[int(sm[r,c])],"landcover":NAME.get(int(wm[r,c]),str(int(wm[r,c])))})
        if len(tops)>=30: break
    return tops, zl, seen, stats

# WorldCover (한 번만)
def worldcover(data, keys):
    import planetary_computer as pc
    from pystac_client import Client
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)
    items = list(cat.search(collections=["esa-worldcover"], bbox=[126.10,33.15,127.00,33.60],
                            query={"esa_worldcover:product_version":{"eq":"2.0.0"}}).items())
    out = {}
    for k in keys:
        w = data["2024"][k]; arr = np.zeros(w["arr"].shape[1:], np.uint8)
        for it in items:
            with rasterio.open(it.assets["map"].href) as s:
                reproject(rasterio.band(s,1), arr, dst_transform=w["tr"], dst_crs=w["crs"],
                          resampling=Resampling.nearest, src_transform=s.transform, src_crs=s.crs)
        out[k] = arr
    return out

print("=== 12타임스텝 ===", flush=True)
d12, keys = load("embeddings_t12")
print(f"  windows: {len(keys)}", flush=True)
wc = worldcover(d12, keys)
top12, zl12, seen12, st12 = analyze(d12, keys, wc)
print("  class stats:", {k:(round(v[0],4),round(v[1],4)) for k,v in st12.items()}, flush=True)
for t in top12[:10]: print("   ", t, flush=True)
print("  when:", dict(collections.Counter(t["when"] for t in top12)),
      "| landcover:", dict(collections.Counter(t["landcover"] for t in top12)), flush=True)
del d12

print("\n=== 4타임스텝 (기존) ===", flush=True)
d4, keys4 = load("embeddings")
keys4 = [k for k in keys4 if k in keys]
top4, zl4, seen4, st4 = analyze(d4, keys4, wc)
print("  when:", dict(collections.Counter(t["when"] for t in top4)),
      "| landcover:", dict(collections.Counter(t["landcover"] for t in top4)), flush=True)
del d4

def near(a,b,tol=0.01): return abs(a["lat"]-b["lat"])<tol and abs(a["lon"]-b["lon"])<tol
inter = sum(1 for a in top4 if any(near(a,b) for b in top12))
jac = inter/(len(top4)+len(top12)-inter) if (top4 or top12) else 0
print(f"\n=== 통제실험: 타임스텝 수만 바꿨을 때 ===", flush=True)
print(f"  Top-30 교집합 {inter}곳 | Jaccard {jac:.3f}", flush=True)
print(f"  4ts z범위 {top4[0]['z']}~{top4[-1]['z']} | 12ts z범위 {top12[0]['z']}~{top12[-1]['z']}", flush=True)
print("  WARNING: 4ts seasons are not aligned across all four year windows; see time-axis audit", flush=True)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,1,figsize=(15,13))
for ax,(zl,seen,tops,title) in zip(axes,[(zl4,seen4,top4,"first 4 stored periods — season-confounded"),
                                          (zl12,seen12,top12,"12 stored periods — full-window coverage")]):
    im = ax.imshow(np.where(zl>-50, zl, np.nan), cmap="inferno", vmin=0, vmax=6)
    for r,c in seen[:len(tops)]: ax.plot(c,r,"o",ms=9,mfc="none",mec="cyan",mew=1.4)
    ax.set_title(f"step-change z, land only, Top-30 — {title}"); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.02)
plt.tight_layout(); plt.savefig(f"{OUT}/jeju_change_v6_4vs12.png", dpi=95, bbox_inches="tight")
json.dump({"top_12ts":top12,"top_4ts":top4,"control":{"intersection":inter,"jaccard":round(jac,3)},
           "class_stats_12ts":{str(k):v for k,v in st12.items()},
           "limitations":["4ts month sequences differ for rolling-2026",
                          "12ts order is reverse chronological and phase-shifted for rolling-2026",
                          "Top-k lists require RGB and probability-sample validation"]},
          open(f"{OUT}/jeju_change_v6_top.json","w"), indent=1)
print("\nDONE", flush=True)
