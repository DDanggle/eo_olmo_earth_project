#!/usr/bin/env python3
"""CPU-only read-only KuroSiwo numerical diagnostic. No test labels, training, or cache writes.

Default uses <=64 train/val tiles per split. --full-train-std reproduces ONLY the
trainer's CPU scale reduction, requiring <=29 GB and >=96 GB available host RAM.
No CUDA device is initialized; stdout is the sole output artifact.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=Path("/home/work/data/olmoearth/kurosiwo_s1_cache"))
    p.add_argument("--full-train-std", action="store_true")
    p.add_argument("--scan-all-inputs", action="store_true", help="Finite/bounds scan of train/val teacher, stale, post only; no labels or model execution")
    p.add_argument("--inspect-validation-id", help="Inspect known invalid validation tile, including its validity/NoData mask, never test labels")
    args = p.parse_args()
    torch.set_num_threads(4)
    meta = [json.loads(l) for l in (args.cache/"meta.jsonl").read_text().splitlines() if l]
    out = {"torch": torch.__version__, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "sampled": {}}
    ids = {sp: sorted({r["id"] for r in meta if r["split"] == sp}) for sp in ("train", "validation")}
    if args.inspect_validation_id:
        sid = args.inspect_validation_id
        if sid not in ids["validation"]:
            raise SystemExit("Only a validation ID can be inspected")
        mask = np.load(args.cache/"mask_u8"/f"{sid}.npy", allow_pickle=False)
        valid = np.load(args.cache/"valid_u8"/f"{sid}.npy", allow_pickle=False)
        keep = (mask != 0)&(valid != 0)
        token_keep = keep.reshape(48,4,48,4).any(axis=(1,3))
        result = {"id":sid,"split":"validation", "valid_pixels":int(keep.sum()), "total_pixels":int(keep.size),
            "mask_values":dict(zip(*[x.tolist() for x in np.unique(mask, return_counts=True)])), "features":{}}
        for kind in ("teacher3","stale2","single"):
            arr = np.load(args.cache/f"{kind}_fp16"/f"{sid}.npy", allow_pickle=False)
            bad = ~np.isfinite(arr)
            bad_tokens = bad.any(axis=0) if arr.ndim==3 else bad.any(axis=(0,1))
            result["features"][kind] = {"nonfinite_values":int(bad.sum()),"total_values":int(arr.size),
                "bad_spatial_tokens":int(bad_tokens.sum()),"bad_tokens_with_any_valid_label":int((bad_tokens&token_keep).sum())}
        raw = np.load(args.cache.parent/"kurosiwo_npy/raw_f32"/f"{sid}.npy",allow_pickle=False)
        result["raw"] = {"shape":list(raw.shape), "nonfinite_count":int((~np.isfinite(raw)).sum()),
            "zeros":int((raw==0).sum()),"nan_by_time":[int(np.isnan(raw[:,i]).sum()) for i in range(raw.shape[1])]}
        print(json.dumps({"validation_tile":result},allow_nan=False),flush=True)
        return
    for sp, pool in ids.items():
        if args.scan_all_inputs:
            for kind in ("teacher3", "stale2", "single"):
                bad, lo, hi = [], float("inf"), float("-inf")
                for i, sid in enumerate(pool):
                    a = np.load(args.cache/f"{kind}_fp16"/f"{sid}.npy", mmap_mode="r", allow_pickle=False)
                    a = np.array(a[2] if kind == "single" else a, dtype=np.float32)
                    if not np.isfinite(a).all():
                        bad.append(sid)
                    else:
                        lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
                print(json.dumps({"complete_input_scan": {"split": sp, "kind": kind,
                    "n": len(pool), "nonfinite_ids": bad, "min": lo if math.isfinite(lo) else None,
                    "max": hi if math.isfinite(hi) else None}}, allow_nan=False), flush=True)
            continue
        selection = [pool[i] for i in np.linspace(0, len(pool)-1, min(64, len(pool)), dtype=int)]
        for kind in ("teacher3", "stale2", "single"):
            bad, bounds, stds = [], [], []
            for sid in selection:
                a = np.load(args.cache/f"{kind}_fp16"/f"{sid}.npy", mmap_mode="r", allow_pickle=False)
                a = a[2] if kind == "single" else a
                if not np.isfinite(a).all():
                    bad.append(sid)
                else:
                    bounds.append([float(a.min()), float(a.max())])
                    stds.append(float(np.std(a, dtype=np.float64)))
            out["sampled"][f"{sp}/{kind}"] = {"n": len(selection), "nonfinite_ids": bad,
                "min": min((b[0] for b in bounds), default=None), "max": max((b[1] for b in bounds), default=None),
                "mean_tile_std_f64": float(np.mean(stds)) if stds else None}
    print(json.dumps({"sample_probe": out}, allow_nan=False), flush=True)
    if args.full_train_std:
        n = len(ids["train"])
        shape = tuple(np.load(args.cache/"teacher3_fp16"/f"{ids['train'][0]}.npy", mmap_mode="r").shape)
        size = n*math.prod(shape)*4
        meminfo = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
        avail = int(meminfo["MemAvailable"].split()[0])*1024
        if size > 29_000_000_000 or avail < max(96*1024**3, 3*size):
            raise SystemExit("Refuse diagnostic allocation: memory bound/available memory")
        start = time.perf_counter()
        x = torch.empty((n, *shape), dtype=torch.float32, device="cpu")
        cnt, mean, m2 = 0, 0.0, 0.0
        for i, sid in enumerate(ids["train"]):
            a = np.load(args.cache/"teacher3_fp16"/f"{sid}.npy", allow_pickle=False).astype("float32")
            if not np.isfinite(a).all():
                raise ValueError(f"Nonfinite teacher tile {sid}")
            x[i].copy_(torch.from_numpy(a))
            bn, bm, bv = a.size, float(a.mean(dtype=np.float64)), float(a.var(dtype=np.float64))
            delta, total = bm-mean, cnt+bn
            m2 += bv*bn + delta*delta*cnt*bn/total
            mean += delta*bn/total
            cnt = total
            if (i+1) % 1000 == 0:
                print(json.dumps({"scale_scan": i+1, "of": n}), flush=True)
        exact = float(x.std().item())
        chunked = math.sqrt(m2/(cnt-1))
        result = {"numel": cnt, "allocated_bytes": size, "all_train_teacher_finite": True,
            "trainer_torch_std": exact if math.isfinite(exact) else None,
            "trainer_torch_std_nonfinite": not math.isfinite(exact), "chunked_float64_std": chunked,
            "elapsed_s": time.perf_counter()-start,
            "scope": "CPU std expression only, no updater train/forward; missing input scans are sampled"}
        print(json.dumps({"full_train_scale": result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
