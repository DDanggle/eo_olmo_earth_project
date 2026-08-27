#!/usr/bin/env python3
"""확증 지역 타일 렌더(L5 육안 검증). CPU 전용 — GPU 확증 실행에 개입하지 않음.

두 종류를 고름:
  gap   양성 타일 중 reuse−raw_strong IoU 격차 최대 2개
  fp    산사태 없는 타일 중 raw_strong 오경보 최대 2개 (승리의 실제 출처 확인용)
"""
import json, pathlib, sys
import numpy as np
from PIL import Image

FOLD = sys.argv[1]
BASE = pathlib.Path("/home/work/data/olmoearth")
CACHE = BASE / "sen12_pilot" / FOLD
CONF = BASE / "confirmatory" / FOLD
OUT = BASE / f"explainer_png_{FOLD}"
OUT.mkdir(parents=True, exist_ok=True)


def stretch(a):
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a.astype("float32") - lo) / max(hi - lo, 1e-6), 0, 1)


def probs(arm):
    d = CONF / f"{arm}_seed1" / "prob_maps" / FOLD
    idx = json.loads((d / f"{arm}_test_probs_index.json").read_text())
    return idx["sample_ids"], np.load(d / f"{arm}_test_probs_u8.npy", mmap_mode="r")


sids, pA = probs("P4")
_, pB = probs("P2")

gap_cand, fp_cand = [], []
for i, s in enumerate(sids):
    m = np.load(CACHE / "mask_u8" / f"{s}.npy") > 0
    a = np.asarray(pA[i]) >= 128
    b = np.asarray(pB[i]) >= 128
    if m.sum() >= 300:
        ia = (a & m).sum() / max((a | m).sum(), 1)
        ib = (b & m).sum() / max((b | m).sum(), 1)
        gap_cand.append((ia - ib, s, i, int(m.sum())))
    elif m.sum() == 0:
        fp_cand.append((int(b.sum()) - int(a.sum()), s, i, int(a.sum()), int(b.sum())))
gap_cand.sort(reverse=True)
fp_cand.sort(reverse=True)

meta = {"gap": [], "fp": []}
def render(s, i, kind):
    cube = np.load(CACHE / "raw_u16" / f"{s}.npy")
    m = np.load(CACHE / "mask_u8" / f"{s}.npy") > 0
    t = cube.shape[1] // 2
    rgb = np.stack([stretch(cube[k, t]) for k in (2, 1, 0)], axis=-1)
    Image.fromarray((rgb*255).astype("uint8")).resize((340,340), Image.NEAREST).save(OUT/f"{s}_rgb.png")
    ov = rgb.copy(); ov[m] = ov[m]*0.35 + np.array([1.0,0.15,0.15])*0.65
    Image.fromarray((ov*255).astype("uint8")).resize((340,340), Image.NEAREST).save(OUT/f"{s}_truth.png")
    for arm, arr in (("P4", pA), ("P2", pB)):
        pr = np.asarray(arr[i]) >= 128
        o = rgb.copy(); o[pr] = o[pr]*0.35 + np.array([0.15,0.55,1.0])*0.65
        Image.fromarray((o*255).astype("uint8")).resize((340,340), Image.NEAREST).save(OUT/f"{s}_{arm}.png")

for gap, s, i, npx in gap_cand[:2]:
    render(s, i, "gap"); meta["gap"].append({"sid": s, "mask_px": npx, "iou_gap": round(float(gap),4)})
for d, s, i, fa, fb in fp_cand[:2]:
    render(s, i, "fp"); meta["fp"].append({"sid": s, "P4_fp_px": fa, "P2_fp_px": fb})
(OUT/"index.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False, indent=2))
