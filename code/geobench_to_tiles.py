"""GEO-Bench 과업을 OlmoEarth 추출기용 raw_u16 타일 포맷으로 변환한다.

왜: geobench 는 z-score 정규화된 float 를 준다. OlmoEarth 추출기는 raw DN(uint16)에 자기
정규화기를 적용한다. z-score 를 또 넣으면 이중 정규화(오염) → 이번 세션 내내 잡아온 함정.
해결: data_normalizer=Passthrough 로 raw DN 을 꺼낸다(확인: DEN min 1 max 5891 mean 1492 = 정상).

출력(과업당): <out>/raw_u16/<id>.npy  (B, T, 128, 128) uint16
             <out>/mask_u8/<id>.npy  (128, 128) uint8   (dense seg 만)
             <out>/months.jsonl      (id 당 months_0_11)
             <out>/chip_manifest.json (원본 샘플→칩 매핑, split, 검증 수치)

계약 고정(사전등록 후보): 128px 칩, 원본 512/256px 을 128 격자로 non-overlap 절단.
image 와 mask 를 동일 슬라이스로 잘라 정렬 보장. 밴드 순서 = 우리 10밴드(B02,B03,B04,B08,...).
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

ROOT = Path("/home/work/data/olmoearth")
OUR_S2 = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
CHIP = 128

# (module, class, s2_key, mask_key, planet_key)  head_type: seg | classification | regression
TASKS = {
    "dynamic_earthnet": ("geobench_v2.datasets.dynamic_earthnet", "GeoBenchDynamicEarthNet",
                         "image_s2", "mask", "seg"),
    "benv2": ("geobench_v2.datasets.benv2", "GeoBenchBENV2", "image_s2", "label", "classification"),
}


class Passthrough(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, x, *a, **k): return x
    def __call__(self, x, *a, **k): return x


def load_dataset(task, split):
    import importlib
    mod_name, cls_name, s2k, labk, head = TASKS[task]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    bdo = getattr(cls, "band_default_order", {})
    bo = {}
    if isinstance(bdo, dict):
        for mod, bands in bdo.items():
            bo[mod] = [b for b in OUR_S2 if b in list(bands)] if mod == "s2" else list(bands)
    ds = cls(root=str(ROOT / "geobench2" / task), split=split, band_order=bo,
             data_normalizer=Passthrough, download=False)
    return ds, s2k, labk, head


def chip_grid(h, w, size=CHIP):
    ys = list(range(0, h - size + 1, size))
    xs = list(range(0, w - size + 1, size))
    return [(y, x) for y in ys for x in xs]


def materialize(task, split, limit=None):
    ds, s2k, labk, head = load_dataset(task, split)
    out = ROOT / "geobench_tiles" / task
    (out / "raw_u16").mkdir(parents=True, exist_ok=True)
    if head == "seg":
        (out / "mask_u8").mkdir(parents=True, exist_ok=True)
    months_f = open(out / "months.jsonl", "a")
    manifest = []
    labels = {}
    n = len(ds) if limit is None else min(limit, len(ds))
    checks = {"dn_min": 1e9, "dn_max": -1e9, "n_chips": 0, "mask_pos_frac": []}
    for i in range(n):
        s = ds[i]
        img = s[s2k].numpy()                       # (B, H, W) raw DN  (single timestep)
        if img.ndim == 4:                          # (B, T, H, W)
            img_bt = img
        else:
            img_bt = img[:, None, :, :]            # add T=1
        B, T, H, W = img_bt.shape
        checks["dn_min"] = min(checks["dn_min"], float(img.min()))
        checks["dn_max"] = max(checks["dn_max"], float(img.max()))
        if head == "seg":
            mask = s[labk].numpy()
            if mask.ndim == 3:
                mask = mask[0]
        for (y, x) in chip_grid(H, W):
            cid = f"{split}_{i:05d}_{y}_{x}"
            chip = img_bt[:, :, y:y + CHIP, x:x + CHIP]
            np.save(out / "raw_u16" / f"{cid}.npy", np.clip(chip, 0, 65535).astype(np.uint16))
            months_f.write(json.dumps({"sample_id": cid, "months_0_11": [6] * T}) + "\n")
            if head == "seg":
                mc = mask[y:y + CHIP, x:x + CHIP].astype(np.uint8)
                np.save(out / "mask_u8" / f"{cid}.npy", mc)
                checks["mask_pos_frac"].append(float((mc > 0).mean()))
            else:
                labels[cid] = s[labk].numpy().astype(np.int64).tolist()
            manifest.append({"chip_id": cid, "src_index": i, "y": y, "x": x, "split": split})
            checks["n_chips"] += 1
    months_f.close()
    if head == "classification":
        (out / "labels.json").write_text(json.dumps(labels))
    m = out / "chip_manifest.json"
    prev = json.loads(m.read_text()) if m.exists() else {"chips": [], "checks": {}}
    prev["chips"].extend(manifest)
    prev["head_type"] = head
    prev["checks"] = {"dn_min": checks["dn_min"], "dn_max": checks["dn_max"],
                      "n_chips_total": len(prev["chips"]), "chip_px": CHIP,
                      "band_order": OUR_S2,
                      "mask_pos_frac_mean": (float(np.mean(checks["mask_pos_frac"]))
                                             if checks["mask_pos_frac"] else None)}
    m.write_text(json.dumps(prev, indent=1))
    print(json.dumps({"task": task, "split": split, "head": head, "n_src": n,
                      "n_chips_added": len(manifest), **prev["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=sorted(TASKS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    materialize(a.task, a.split, a.limit)
