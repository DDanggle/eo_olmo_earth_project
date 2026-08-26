#!/usr/bin/env python3
"""설명용 실제 위성사진 렌더 (L5 육안 검증 겸용).

- AI-Hub 타일의 12밴드 물질화 결과에서 true-color(B04/B03/B02)와 false-color(B08/B04/B03)
- Sen12Landslides 샘플의 raw 큐브 + MASK 오버레이
percentile stretch 2~98%만 쓰고 다른 보정은 하지 않는다. 그림으로 속이지 않기 위해서다.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from PIL import Image

OUT = pathlib.Path("/home/work/data/olmoearth/explainer_png")
OUT.mkdir(parents=True, exist_ok=True)


def stretch(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a.astype("float32") - lo) / max(hi - lo, 1e-6), 0, 1)


def rgb(cube, idx, size=520):
    im = np.stack([stretch(cube[i]) for i in idx], axis=-1)
    return Image.fromarray((im * 255).astype("uint8")).resize((size, size), Image.BILINEAR)


def natural(cube, idx=(2, 1, 0), size=520, scale=3000.0, gamma=1 / 1.6):
    """자연색. 밴드별 stretch를 쓰지 않는다 — 밴드마다 다르게 늘리면 색이 거짓이 된다.
    L2A 반사도를 세 밴드 **공통** 스케일로 나누고 gamma만 적용한다."""
    im = np.stack([np.clip(cube[i].astype("float32") / scale, 0, 1) for i in idx], axis=-1)
    im = np.power(im, gamma)
    return Image.fromarray((im * 255).astype("uint8")).resize((size, size), Image.BILINEAR)


# ── AI-Hub 12밴드 (오늘 물질화한 것) ──
d = pathlib.Path("/home/work/data/olmoearth/aihub/s2_12band")
rows = [json.loads(l) for l in (d / "manifest.jsonl").read_text().splitlines() if l]
picked = []
for r in rows:
    if r["key"] not in {x["key"] for x in picked}:
        picked.append(r)
    if len(picked) >= 3:
        break
meta = []
for r in picked:
    c = np.load(d / "arrays" / (r["key"] + ".npy"))
    natural(c).save(OUT / f'aihub_{r["key"]}_true.png')
    rgb(c, (3, 2, 1)).save(OUT / f'aihub_{r["key"]}_false.png')
    meta.append({"key": r["key"], "date": r["date"], "cc": r["cloud_cover"],
                 "item": r["item_id"], "mgrs": r["mgrs"]})

# ── Sen12Landslides (실제 실험에 쓴 데이터) ──
p = pathlib.Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani")
sen = []
if (p / "raw_u16").exists():
    files = sorted((p / "raw_u16").glob("*.npy"))
    scored = []
    for f in files[:400]:
        m = np.load(p / "mask_u8" / f.name)
        scored.append((float(m.mean()), f))
    scored.sort(reverse=True)
    for frac, f in scored[:3]:
        cube = np.load(f)              # 10,T,128,128
        m = np.load(p / "mask_u8" / f.name)
        t = cube.shape[1] // 2
        img = np.stack([stretch(cube[i, t]) for i in (2, 1, 0)], axis=-1)
        base = Image.fromarray((img * 255).astype("uint8")).resize((320, 320), Image.NEAREST)
        ov = img.copy()
        mm = np.array(Image.fromarray(m * 255).resize((128, 128), Image.NEAREST)) > 127
        ov[mm] = ov[mm] * 0.35 + np.array([1.0, 0.15, 0.15]) * 0.65
        over = Image.fromarray((ov * 255).astype("uint8")).resize((320, 320), Image.NEAREST)
        base.save(OUT / f"sen12_{f.stem}_rgb.png")
        over.save(OUT / f"sen12_{f.stem}_mask.png")
        sen.append({"sample_id": f.stem, "positive_frac": round(frac, 5),
                    "timesteps": int(cube.shape[1])})

(OUT / "index.json").write_text(json.dumps(
    {"aihub": meta, "sen12": sen}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"aihub": meta, "sen12": sen}, ensure_ascii=False, indent=2))
print("files:", sorted(x.name for x in OUT.glob("*.png")))
