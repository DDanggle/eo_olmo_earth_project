#!/usr/bin/env python3
"""N1 embeddings: OlmoEarth v1 Base single-date S1 embeddings for the multi-year PC RTC chips (n1_s1/chips).

Prereg: config/n1_learned_normal_prereg_v0.json. Same model call as code/extract_kurosiwo_s1_cache.py (patch 4, token
pooling, 3x3 crops of 64 px, modality 'sentinel1' [vv, vh] in dB with -30 fill), timestamp = acquisition date.
Per tile writes n1_s1/emb/<id>.npz with
  dates (T,), roles (T,), emb24 (T,768,24,24) float16  [2x2-pooled 48x48 tokens = 80 m]
  pre48, post48 (768,48,48) float16 for the pre/post acquisitions (40 m) if present.

  CUDA_VISIBLE_DEVICES=0 python -B code/n1_embed_s1_v0.py
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/work/data/olmoearth")
SRC = ROOT / "n1_s1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    from olmoearth_pretrain_minimal import ModelID
    from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage

    dev = torch.device("cuda")
    w = OlmoEarth(patch_size=4, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True, use_legacy_timestamps=False,
                  normalize=True, autocast_dtype="bfloat16").to(dev).eval()
    out = SRC / "emb"
    out.mkdir(exist_ok=True)
    manifests = sorted((SRC / "chips").glob("*.json"))

    def embed(cube, ts):
        """cube (2,192,192) dB -> (768,48,48)."""
        crops = [np.ascontiguousarray(cube[:, None, y:y + 64, x:x + 64]) for y in (0, 64, 128) for x in (0, 64, 128)]
        inputs = []
        for c in crops:
            inp = {"sentinel1": RasterImage(image=torch.from_numpy(c).to(dev), timestamps=[(ts, ts)])}
            w.normalizer(inp, {})
            inputs.append(inp)
        sample, _, _ = w._prepare_modality_inputs(ModelContext(inputs=inputs, metadatas=[]))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tm = w.model(sample, fast_pass=False, patch_size=4)["tokens_and_masks"]
            m = (tm.sentinel1_mask != MaskValue.MISSING.value).unsqueeze(-1)
            p = ((tm.sentinel1 * m).sum(dim=(3, 4)) / m.sum(dim=(3, 4)).clamp(min=1)).permute(0, 3, 1, 2).float()
        feat = torch.empty((768, 48, 48), device=dev)
        k = 0
        for y in range(3):
            for x in range(3):
                feat[:, y * 16:(y + 1) * 16, x * 16:(x + 1) * 16] = p[k]
                k += 1
        return feat

    t0, n = time.perf_counter(), 0
    for mf in manifests:
        rec = json.loads(mf.read_text())
        f = out / f"{rec['id']}.npz"
        if rec.get("status") != "ok" or f.exists():
            continue
        dates, roles, e24, pre48, post48 = [], [], [], None, None
        for c in rec["chips"]:
            p = SRC / "chips" / f"{rec['id']}_{c['date']}.npy"
            if not p.exists():
                continue
            cube = np.load(p).astype(np.float32)
            e = embed(cube, datetime.fromisoformat(c["date"]))
            e24.append(F.avg_pool2d(e[None], 2)[0].cpu().numpy().astype(np.float16))
            dates.append(c["date"])
            roles.append(c["role"])
            if c["role"] == "pre":
                pre48 = e.cpu().numpy().astype(np.float16)
            if c["role"] == "post":
                post48 = e.cpu().numpy().astype(np.float16)
        np.savez(f, dates=np.array(dates), roles=np.array(roles), emb24=np.stack(e24),
                 **({"pre48": pre48} if pre48 is not None else {}), **({"post48": post48} if post48 is not None else {}))
        n += 1
        if a.probe:
            print("probe", rec["id"], np.stack(e24).shape, f"{time.perf_counter() - t0:.1f}s")
            return
        if n % 100 == 0:
            print(n, "tiles", f"{time.perf_counter() - t0:.0f}s", flush=True)
    print("N1 EMBED DONE", n, f"{time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
