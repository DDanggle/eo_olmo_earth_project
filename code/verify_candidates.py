"""변화 후보 육안 검증 — 후보 지점의 연도별 Sentinel-2 RGB 칩을 나란히 뽑는다.

v2 계단형 탐지가 내놓은 Top 후보가 진짜 변화인지, 아니면 특정 연도 모자이크의
품질 문제(구름/안개)인지 판별하는 결정적 테스트.

각 후보마다 4개년 × 2장(연중 다른 시기 모자이크)을 보여준다. 같은 연도의 두 모자이크가
서로 딴판이면 그 연도 데이터 품질이 나쁘다는 뜻 → 가짜 변화 의심.
"""

import glob
import json

import numpy as np
import rasterio
from pyproj import Transformer

BASE = "/home/work/data/olmoearth/embed_search/dataset/windows/default"
OUT = "/home/work/data/olmoearth/embed_search"
YEARS = {"2023": "jeju23_", "2024": "jeju_", "2025": "jeju25_", "2026": "jeju26r_"}
YEAR_ORDER = ["2023", "2024", "2025", "2026"]
# config.json의 밴드 순서: B01,B02,B03,B04,... → RGB = 4,3,2 (1-based)
RGB_IDX = [4, 3, 2]
CHIP = 60  # 10m 픽셀 기준 반경 → 1.2km 폭
# 연중 두 시점: 첫 모자이크와 중간 모자이크
LAYER_SUFFIX = ["", ".6"]

import sys

TOP_JSON = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/jeju_change_v2_top.json"
SAVE_AS = sys.argv[2] if len(sys.argv) > 2 else f"{OUT}/verify_candidates.png"
top = json.load(open(TOP_JSON))["top"]
# 공간적으로 흩어진 후보를 고르기 위해 위치 기준으로 추린다
picks = []
for t in top:
    if all(abs(t["lat"] - p["lat"]) > 0.02 or abs(t["lon"] - p["lon"]) > 0.02 for p in picks):
        picks.append(t)
    if len(picks) >= 5:
        break
print("verifying:", [(p["lat"], p["lon"], p["when"]) for p in picks], flush=True)


def find_window(year, lat, lon):
    """해당 좌표를 포함하는 윈도우의 S2 레이어 경로와 픽셀 좌표를 찾는다."""
    prefix = YEARS[year]
    for wdir in glob.glob(f"{BASE}/{prefix}*"):
        tifs = sorted(glob.glob(f"{wdir}/layers/sentinel2_l2a/*/geotiff.tif"))
        if not tifs:
            continue
        with rasterio.open(tifs[0]) as src:
            tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = tr.transform(lon, lat)
            col, row = ~src.transform * (x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                return wdir, int(row), int(col)
    return None, None, None


def read_chip(wdir, suffix, row, col):
    layer = f"sentinel2_l2a{suffix}"
    tifs = sorted(glob.glob(f"{wdir}/layers/{layer}/*/geotiff.tif"))
    if not tifs:
        return None
    with rasterio.open(tifs[0]) as src:
        r0, c0 = max(0, row - CHIP), max(0, col - CHIP)
        r1, c1 = min(src.height, row + CHIP), min(src.width, col + CHIP)
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        arr = np.stack([src.read(b, window=win) for b in RGB_IDX]).astype(np.float32)
    # 2~98 퍼센타일 스트레치
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi <= lo:
        return np.zeros(arr.shape[1:] + (3,), np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1).transpose(1, 2, 0)


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

n_rows = len(picks)
n_cols = len(YEAR_ORDER) * len(LAYER_SUFFIX)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows))
for i, p in enumerate(picks):
    for j, year in enumerate(YEAR_ORDER):
        wdir, row, col = find_window(year, p["lat"], p["lon"])
        for k, suffix in enumerate(LAYER_SUFFIX):
            ax = axes[i, j * len(LAYER_SUFFIX) + k]
            ax.axis("off")
            chip = read_chip(wdir, suffix, row, col) if wdir else None
            if chip is not None:
                ax.imshow(chip)
            if i == 0:
                ax.set_title(f"{year}{'a' if k == 0 else 'b'}", fontsize=10)
            if j == 0 and k == 0:
                ax.text(
                    -0.15, 0.5, f"z={p['z']}\n{p['when']}\n{p['landcover']}",
                    transform=ax.transAxes, ha="right", va="center", fontsize=8,
                )
plt.suptitle("candidate verification - S2 RGB by year (a: early mosaic, b: mid-year mosaic)", fontsize=12)
plt.tight_layout()
plt.savefig(SAVE_AS, dpi=100, bbox_inches="tight")
print("DONE", flush=True)
