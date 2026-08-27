#!/usr/bin/env python3
"""thrissur 확증 지역의 실제 타일을 렌더함. 두 방식의 예측 차이를 눈으로 확인(L5)."""
import json, pathlib
import numpy as np
from PIL import Image

BASE = pathlib.Path("/home/work/data/olmoearth")
CACHE = BASE / "sen12_pilot/holdout_thrissur"
CONF = BASE / "confirmatory/holdout_thrissur"
OUT = BASE / "explainer_png_thrissur"
OUT.mkdir(parents=True, exist_ok=True)


def stretch(a):
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    return np.clip((a.astype("float32") - lo) / max(hi - lo, 1e-6), 0, 1)


def probs(arm):
    d = CONF / f"{arm}_seed1" / "prob_maps" / "holdout_thrissur"
    idx = json.loads((d / f"{arm}_test_probs_index.json").read_text())
    return idx["sample_ids"], np.load(d / f"{arm}_test_probs_u8.npy", mmap_mode="r")


sids, pA = probs("P4")
_, pB = probs("P2")
rows = [json.loads(l) for l in
        (CONF / "P4_seed1/per_sample/holdout_thrissur/P4_test.jsonl").read_text().splitlines() if l]
info = {r["sample_id"]: r for r in rows}

# 양성 화소가 많고 두 방식 차이가 큰 타일을 고름
cand = []
for i, s in enumerate(sids):
    m = np.load(CACHE / "mask_u8" / f"{s}.npy") > 0
    if m.sum() < 300:
        continue
    a = np.asarray(pA[i]) >= 128
    b = np.asarray(pB[i]) >= 128
    ia = (a & m).sum() / max((a | m).sum(), 1)
    ib = (b & m).sum() / max((b | m).sum(), 1)
    cand.append((ia - ib, s, i, int(m.sum())))
cand.sort(reverse=True)
picked = cand[:3]

meta = []
for gap, s, i, npx in picked:
    cube = np.load(CACHE / "raw_u16" / f"{s}.npy")      # 10,T,H,W
    m = np.load(CACHE / "mask_u8" / f"{s}.npy") > 0
    t = cube.shape[1] // 2
    rgb = np.stack([stretch(cube[k, t]) for k in (2, 1, 0)], axis=-1)
    Image.fromarray((rgb * 255).astype("uint8")).resize((340, 340), Image.NEAREST)\
        .save(OUT / f"{s}_rgb.png")
    ov = rgb.copy(); ov[m] = ov[m] * 0.35 + np.array([1.0, 0.15, 0.15]) * 0.65
    Image.fromarray((ov * 255).astype("uint8")).resize((340, 340), Image.NEAREST)\
        .save(OUT / f"{s}_truth.png")
    for arm, arr in (("P4", pA), ("P2", pB)):
        pr = np.asarray(arr[i]) >= 128
        o = rgb.copy(); o[pr] = o[pr] * 0.35 + np.array([0.15, 0.55, 1.0]) * 0.65
        Image.fromarray((o * 255).astype("uint8")).resize((340, 340), Image.NEAREST)\
            .save(OUT / f"{s}_{arm}.png")
    meta.append({"sample_id": s, "mask_px": npx, "iou_gap": round(float(gap), 4)})
(OUT / "index.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(meta, ensure_ascii=False, indent=2))
print(sorted(p.name for p in OUT.glob("*.png")))
