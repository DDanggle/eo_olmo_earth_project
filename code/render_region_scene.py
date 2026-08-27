#!/usr/bin/env python3
"""지역 대표 장면 렌더 — 갤러리용. CPU 전용. 예측 없이 캐시(원본+정답)만 씀."""
import json, pathlib, sys
import numpy as np
from PIL import Image

FOLD = sys.argv[1]
CACHE = pathlib.Path("/home/work/data/olmoearth/sen12_pilot") / FOLD
OUT = pathlib.Path("/home/work/data/olmoearth/gallery_png"); OUT.mkdir(exist_ok=True)
REGION = FOLD.replace("holdout_", "")


def stretch(a):
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a.astype("float32") - lo) / max(hi - lo, 1e-6), 0, 1)


# 그 지역 이름이 붙은 타일 중 양성 화소 최대 타일 = 대표 장면
best = None
for p in sorted((CACHE / "mask_u8").glob(f"{REGION}_*.npy")):
    n = int((np.load(p) > 0).sum())
    if best is None or n > best[0]:
        best = (n, p.stem)
n, sid = best
cube = np.load(CACHE / "raw_u16" / f"{sid}.npy")
m = np.load(CACHE / "mask_u8" / f"{sid}.npy") > 0
t = cube.shape[1] // 2
rgb = np.stack([stretch(cube[k, t]) for k in (2, 1, 0)], axis=-1)
Image.fromarray((rgb*255).astype("uint8")).resize((340,340), Image.NEAREST).save(OUT/f"{REGION}_rgb.png")
ov = rgb.copy(); ov[m] = ov[m]*0.35 + np.array([1.0,0.15,0.15])*0.65
Image.fromarray((ov*255).astype("uint8")).resize((340,340), Image.NEAREST).save(OUT/f"{REGION}_truth.png")
print(json.dumps({"region": REGION, "sid": sid, "mask_px": n}))
