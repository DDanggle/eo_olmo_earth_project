#!/usr/bin/env python3
"""A1b step 3 (resolution): OlmoEarth S1 change maps at patch size P for KuroSiwo tiles, without storing embeddings.

Prereg: config/a1b_scenario_prereg_v0.json. Preprocessing is identical to code/extract_kurosiwo_s1_cache.py
(10*log10 linear backscatter, zeros -> -30 dB, centre 192x192, timestamps flood_date -24/-12/0 d, OlmoEarth v1 Base,
token pooling, 3x3 crops of 64 px); only patch_size differs (default 2 -> 96x96 tokens of 20 m).
Output per tile: a1b/olmo_p{P}/<id>.npz with d_post = 1-cos(post, pre_2) and d_pre = 1-cos(pre_2, pre_1), float16,
plus per-tile GPU seconds. Storing embeddings at patch 2 would need ~42 MB/tile.

  CUDA_VISIBLE_DEVICES=1 python -B code/a1b_extract_s1_change_v0.py --ids a1b/tiles_ids.txt --patch 2
"""
import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/work/data/olmoearth")
NPY = ROOT / "kurosiwo_npy"
OFF, SIZE = 16, 192


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--patch", type=int, default=2)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    from olmoearth_pretrain_minimal import ModelID
    from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage

    P = a.patch
    out = ROOT / "a1b" / f"olmo_p{P}"
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    w = OlmoEarth(patch_size=P, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False,
                  normalize=True, autocast_dtype="bfloat16").to(dev).eval()
    meta = {json.loads(l)["id"]: json.loads(l) for l in (NPY / "meta.jsonl").read_text().splitlines() if l.strip()}
    ids = [i for i in (ROOT / a.ids).read_text().split() if i]
    g = 64 // P  # tokens per crop side

    def pooled_batch(crops, ts):
        inputs = []
        for c in crops:
            inp = {"sentinel1": RasterImage(image=torch.from_numpy(c).to(dev), timestamps=[(x, x) for x in ts])}
            w.normalizer(inp, {})
            inputs.append(inp)
        sample, present, _ = w._prepare_modality_inputs(ModelContext(inputs=inputs, metadatas=[]))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tm = w.model(sample, fast_pass=False, patch_size=P)["tokens_and_masks"]
            m = (tm.sentinel1_mask != MaskValue.MISSING.value).unsqueeze(-1)
            return ((tm.sentinel1 * m).sum(dim=(3, 4)) / m.sum(dim=(3, 4)).clamp(min=1)).permute(0, 3, 1, 2).float()

    def window(cube, ts, k):
        crops = [np.ascontiguousarray(cube[:, [k], y:y + 64, x:x + 64]) for y in (0, 64, 128) for x in (0, 64, 128)]
        p = pooled_batch(crops, [ts[k]])
        feat = torch.empty((768, 3 * g, 3 * g), device=dev)
        n = 0
        for y in range(3):
            for x in range(3):
                feat[:, y * g:(y + 1) * g, x * g:(x + 1) * g] = p[n]
                n += 1
        return feat

    def cosd(a_, b_):
        return (1 - torch.nn.functional.cosine_similarity(a_, b_, dim=0)).cpu().numpy().astype(np.float16)

    log, t_all = [], time.perf_counter()
    for sid in (ids[:1] if a.probe else ids):
        f = out / f"{sid}.npz"
        if f.exists():
            continue
        t0 = time.perf_counter()
        raw = np.load(NPY / "raw_f32" / f"{sid}.npy")
        imgs = [raw[:, k, OFF:OFF + SIZE, OFF:OFF + SIZE] for k in (0, 1, 2)]
        cube = np.stack([10 * np.log10(np.clip(im.astype("float64"), 1e-6, None)) for im in imgs], 1).astype("float32")
        cube[np.stack([im == 0 for im in imgs], 1)] = -30.0
        fd = pd.Timestamp(meta[sid]["flood_date"]).to_pydatetime().replace(tzinfo=None)
        ts = [fd - timedelta(days=24), fd - timedelta(days=12), fd]
        e = [window(cube, ts, k) for k in range(3)]
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        np.savez(f, d_post=cosd(e[2], e[1]), d_pre=cosd(e[1], e[0]))
        log.append({"id": sid, "seconds": secs})
        if a.probe:
            print("probe", sid, np.load(f)["d_post"].shape, f"{secs:.2f}s")
            return
        if len(log) % 200 == 0:
            print(len(log), "tiles", f"{time.perf_counter() - t_all:.0f}s", flush=True)
    (out / f"extract_log_{int(time.time())}.json").write_text(json.dumps({"patch": P, "n": len(log), "total_s": time.perf_counter() - t_all,
                                                                         "median_s": float(np.median([x['seconds'] for x in log])) if log else None}))
    print("A1b EXTRACT DONE", P, len(log))


if __name__ == "__main__":
    main()
