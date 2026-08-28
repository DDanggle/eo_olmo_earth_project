#!/usr/bin/env python3
"""8/27 S2B의 AOI 단위 관측성 — 타일 구름률(78.47%)을 앵커별 실측으로 대체함.

주의(정직성): 우리 큐브에는 SCL이 없음(12 반사도 밴드만). 따라서 이것은 공식
구름 마스크가 아니라 **밝기 휴리스틱**임 — B02(청색) > BRIGHT_DN 픽셀 비율을
"밝음(구름 또는 눈)"으로 셈. 히말라야 고지대라 눈/구름이 섞이며, 그 한계를 그대로 표기함.
zero = nodata(스와스 밖)도 따로 셈.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1] / "artifacts/external_data/nepal_olmo_live_v1/materialized/s2_live/dataset/windows/nepal"
BAND_DIR = "B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12"
BRIGHT_DN = 2600   # L2A DN(0-10000), 대략 반사도 0.26 — 사전 고정
OUT = Path(__file__).resolve().parents[1] / "artifacts/aoi_observability_20260827.json"

def latest_layer(anchor: Path) -> Path | None:
    # 8/27이 든 레이어 = items.json 그룹 0 (최신). 레이어 디렉터리는 무접미사.
    p = anchor / "layers/sentinel2_l2a" / BAND_DIR / "geotiff.tif"
    return p if p.exists() else None

res = {"schema": "aoi-observability-20260827-v1",
       "method": f"B02 > {BRIGHT_DN} DN → bright(cloud OR snow); zero-all-bands → nodata. SCL 없음 — 휴리스틱임",
       "anchors": {}}
for a in sorted(ROOT.iterdir()):
    if not a.is_dir():
        continue
    tif = latest_layer(a)
    if tif is None:
        res["anchors"][a.name] = None
        continue
    with rasterio.open(tif) as src:
        arr = src.read().astype(np.float32)   # (12*T? or 12, H, W)
    # 마지막 기간의 12밴드만: 파일은 기간별 분리 저장이므로 이 tif가 곧 최신 모자이크
    b02 = arr[1] if arr.shape[0] >= 12 else arr[0]
    nodata = (arr.sum(axis=0) == 0)
    valid = ~nodata
    bright = (b02 > BRIGHT_DN) & valid
    res["anchors"][a.name] = {
        "pixels": int(valid.size),
        "nodata_frac": round(float(nodata.mean()), 4),
        "bright_frac_of_valid": round(float(bright.sum() / max(1, valid.sum())), 4),
        "clear_dark_frac_of_valid": round(float(1 - bright.sum() / max(1, valid.sum())), 4),
    }
OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(res, ensure_ascii=False, indent=1))
