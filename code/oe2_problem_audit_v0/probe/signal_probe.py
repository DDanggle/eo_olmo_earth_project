#!/usr/bin/env python3
"""Bounded frozen-encoder sensor-signal audit; no semantic/VLM/bottleneck claim."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path

# Set before NumPy / torch imports. Affects only this audit process.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"
import numpy as np

BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09")
SALT = "oe2-signal-probe-fivefold-v0|20260926|"
LIMITS = [
    "20 previously used OE1 train scenes only; this is a development audit, not independent semantic evaluation.",
    "DN-derived B08/B04 normalized difference is a sensor-signal target, not calibrated geophysical NDVI or vegetation health gold.",
    "Native bandsets are captured but the primary comparison averages them identically; bandset-averaging loss is NOT isolated.",
    "Nearest-restored coarse features and a pointwise linear probe impose a restricted decoder; errors do not prove irreversible information loss.",
    "Distinct MGRS does not guarantee geographically independent sites; no geographic buffer is imposed.",
    "No cloud mask, dense semantic ground truth, VLM evaluation, encoder update or novel learning objective is used.",
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def assign_folds(cases):
    ids = [x["case_id"] for x in cases]
    groups = [x["original_patch_metadata"]["mgrs"] for x in cases]
    if len(ids) != 20 or len(set(ids)) != 20 or len(set(groups)) != 20:
        raise ValueError("Require exactly 20 distinct scenes and distinct MGRS groups")
    order = sorted(ids, key=lambda x: (hashlib.sha256((SALT + x).encode()).hexdigest(), x))
    return {x: i % 5 for i, x in enumerate(order)}


def signal_target(cube, grid):
    """Average valid per-pixel DN ratios inside each native patch; all pixels required."""
    red, nir = cube[2].astype(np.float64), cube[3].astype(np.float64)
    valid = np.isfinite(red) & np.isfinite(nir) & (red >= 0) & (nir >= 0)
    valid &= (red < 65535) & (nir < 65535) & ((red + nir) > 0)
    ratio = np.zeros_like(red)
    np.divide(nir - red, nir + red, out=ratio, where=valid)
    gh, gw = grid
    h, w = red.shape
    if h % gh or w % gw:
        raise ValueError("Native token grid does not divide raw image")
    ph, pw = h // gh, w // gw
    blocks = ratio.reshape(gh, ph, gw, pw)
    mask = valid.reshape(gh, ph, gw, pw)
    # Do not make varying partial-pixel targets: keep only fully valid native cells.
    cell_valid = mask.all(axis=(1, 3))
    target = blocks.mean(axis=(1, 3))
    return target, cell_valid, {"pixel_count": h * w, "valid_pixels": int(valid.sum()),
                               "native_cells": gh * gw, "valid_native_cells": int(cell_valid.sum()),
                               "patch_shape": [ph, pw]}


def ridge_fit(xs, ys, alpha=1.0):
    """Equal-scene MSE + alpha*||w||²; train-only weighted standardization; free intercept."""
    if not xs or len(xs) != len(ys):
        raise ValueError("Need matching train scenes")
    x = np.concatenate(xs).astype(np.float64)
    y = np.concatenate(ys).astype(np.float64)
    if len(x) != len(y) or not all(len(a) for a in xs):
        raise ValueError("Invalid train scene shapes")
    weights = np.concatenate([np.full(len(a), 1.0 / (len(xs) * len(a))) for a in xs])
    mean = weights @ x
    std = np.sqrt(weights @ ((x - mean) ** 2))
    std = np.where(std < 1e-12, 1.0, std)
    z = (x - mean) / std
    ym = float(weights @ y)
    gram = z.T @ (weights[:, None] * z)
    coef = np.linalg.solve(gram + alpha * np.eye(z.shape[1]), z.T @ (weights * (y - ym)))
    return {"mean": mean, "std": std, "coef": coef, "intercept": ym}


def predict(model, x):
    return ((x - model["mean"]) / model["std"]) @ model["coef"] + model["intercept"]


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    err = pred - y
    sst = float(np.square(y - y.mean()).sum())
    return {"valid_cells": len(y), "mse": float(np.mean(err ** 2)), "mae": float(np.mean(abs(err))),
            "signed_bias": float(err.mean()), "r2_within_scene": None if sst < 1e-12 else 1 - float((err ** 2).sum()) / sst}


def evaluate(scenes, folds, out_dir):
    methods = ("full", "pool4_nearest", "pool8_nearest", "coordinates", "train_mean")
    rows, split_rows, model_receipts = [], [], []
    for fold in range(5):
        train = [s for s in scenes if folds[s["case_id"]] != fold]
        test = [s for s in scenes if folds[s["case_id"]] == fold]
        assert len(train) == 16 and len(test) == 4
        split_rows.append({"fold": fold, "train": [s["case_id"] for s in train], "held_out": [s["case_id"] for s in test]})
        for name in methods:
            baseline = float(np.mean([s["target"].mean() for s in train]))
            model = None if name == "train_mean" else ridge_fit([s[name] for s in train], [s["target"] for s in train])
            if model is not None:
                model_receipts.append({"fold": fold, "method": name, "feature_dim": len(model["coef"]),
                                       "train_mean_sha256": hashlib.sha256(model["mean"].tobytes()).hexdigest(),
                                       "train_std_sha256": hashlib.sha256(model["std"].tobytes()).hexdigest(),
                                       "coefficient_sha256": hashlib.sha256(model["coef"].tobytes()).hexdigest(),
                                       "intercept": model["intercept"]})
            for scene in test:
                pred = np.full_like(scene["target"], baseline) if model is None else predict(model, scene[name])
                rows.append({"case_id": scene["case_id"], "mgrs": scene["mgrs"], "fold": fold, "method": name,
                             **metrics(scene["target"], pred)})
    summary = {}
    for name in methods:
        r = [x for x in rows if x["method"] == name]
        summary[name] = {"scene_count": len(r), "mean_scene_mse": float(np.mean([x["mse"] for x in r])),
                         "mean_scene_mae": float(np.mean([x["mae"] for x in r])),
                         "median_scene_mae": float(np.median([x["mae"] for x in r])),
                         "mean_scene_bias": float(np.mean([x["signed_bias"] for x in r])),
                         "mean_within_scene_r2": float(np.mean([x["r2_within_scene"] for x in r if x["r2_within_scene"] is not None])) if any(x["r2_within_scene"] is not None for x in r) else None}
    save_json(out_dir / "folds.json", split_rows)
    save_json(out_dir / "probe_receipts.json", model_receipts)
    save_json(out_dir / "scene_metrics.json", rows)
    return summary


def selftest():
    rng = np.random.default_rng(17)
    xs = [rng.normal(size=(9, 3)) for _ in range(4)]
    ys = [x @ np.array([1., -2., .5]) + 3 for x in xs]
    model = ridge_fit(xs, ys)
    x, y = np.concatenate(xs), np.concatenate(ys)
    z = (x - model["mean"]) / model["std"]
    expected = np.linalg.solve(z.T @ z / len(z) + np.eye(3), z.T @ (y - y.mean()) / len(z))
    np.testing.assert_allclose(model["coef"], expected, atol=1e-12)
    held_out = rng.normal(size=(7, 3))
    original = {k: np.array(v, copy=True) for k, v in model.items()}
    predict(model, held_out * 1e12)
    for k in model:
        np.testing.assert_array_equal(model[k], original[k])
    cube = np.ones((12, 8, 8), dtype=np.uint16) * 100
    cube[3] = 300
    target, valid, _ = signal_target(cube, (2, 2))
    np.testing.assert_allclose(target, .5)
    assert valid.all()
    cube[2, 0, 0] = 65535
    assert signal_target(cube, (2, 2))[1].sum() == 3
    cases = [{"case_id": str(i), "original_patch_metadata": {"mgrs": str(i)}} for i in range(20)]
    folds = assign_folds(cases)
    assert folds == assign_folds(list(reversed(cases)))
    assert [list(folds.values()).count(i) for i in range(5)] == [4] * 5
    print("selftest passed: ridge objective, prediction isolation, target mask, scene folds", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--data-dir", type=Path)
    ap.add_argument("--eo-ckpt", type=Path)
    ap.add_argument("--eo-module", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--wall-seconds", type=int, default=540)
    ap.add_argument("--save-features", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if any(getattr(args, x) is None for x in ("manifest", "data_dir", "eo_ckpt", "eo_module", "out")):
        ap.error("manifest/data-dir/eo-ckpt/eo-module/out required")
    if not 30 <= args.wall_seconds <= 540:
        ap.error("wall-seconds must be 30..540")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    done = threading.Event()
    def watchdog():
        if not done.wait(args.wall_seconds):
            save_json(args.out / "status.json", {"status": "hard_wall_timeout", "wall_seconds": args.wall_seconds})
            os._exit(124)
    threading.Thread(target=watchdog, daemon=True).start()
    save_json(args.out / "status.json", {"status": "running"})
    try:
        run(args, started)
        done.set()
    except BaseException as exc:
        done.set()
        save_json(args.out / "failure.json", {"exception": repr(exc), "traceback": traceback.format_exc()})
        save_json(args.out / "status.json", {"status": "failed", "elapsed_seconds": time.monotonic() - started})
        raise


def run(args, started):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = importlib.util.spec_from_file_location("oe1_signal_source", args.eo_module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    manifest = json.loads(args.manifest.read_text())
    cases = manifest["candidates"]
    folds = assign_folds(cases)
    # Validate ALL raw files before any model execution. Historical ready=false is not bypassed:
    # this runner separately establishes present file/array identity, not semantic readiness.
    raw, raw_receipts = [], []
    for c in cases:
        meta, receipt = c["original_patch_metadata"], c["raw_image"]
        if any(meta.get(k) != "train" for k in ("split", "official_split", "geobench_split")) or tuple(meta["bands"]) != BANDS:
            raise ValueError("Only pinned official-train native-band data allowed")
        relative = Path(meta["image_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe relative raw path")
        path = (args.data_dir / relative).resolve(strict=True)
        file_sha = sha(path)
        if file_sha != receipt["expected_npy_file_sha256"]:
            raise ValueError(f"Raw file SHA mismatch: {c['case_id']}")
        cube = np.load(path, allow_pickle=False)
        array_sha = hashlib.sha256(cube.tobytes()).hexdigest()
        if array_sha != receipt["expected_array_sha256"] or list(cube.shape) != [12,120,120] or cube.dtype != np.uint16:
            raise ValueError(f"Raw array contract mismatch: {c['case_id']}")
        raw.append(cube)
        raw_receipts.append({"case_id": c["case_id"], "path": str(path), "historical_server_path": receipt["server_path_from_historical_receipt"],
                             "file_sha256": file_sha, "array_sha256": array_sha, "shape": list(cube.shape)})
    save_json(args.out / "raw_receipts.json", raw_receipts)
    model = mod.EOImageEncoder.from_pretrained(args.eo_ckpt, trainable=False, pooling="spatial").to(args.device).eval()
    if tuple(mod.S2_BANDS) != BANDS:
        raise ValueError("EO module band contract differs")
    before = model.capture_parameter_state()
    captured = []
    def hook(module, inputs, output):
        captured.append(output["tokens_and_masks"].sentinel2_l2a.detach().clone())
    handle = model.encoder.register_forward_hook(hook)
    scenes, token_receipts, native_arrays = [], [], {}
    for i, (c, cube) in enumerate(zip(cases, raw)):
        with torch.no_grad():
            normal = model(torch.from_numpy(cube.astype(np.float32)).unsqueeze(0).to(args.device),
                           [c["original_patch_metadata"]["timestamp_utc"]], band_names=BANDS)
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one native encoder output")
        native = captured.pop()
        if native.ndim != 6 or native.shape[0] != 1 or native.shape[3] != 1 or not torch.isfinite(native).all():
            raise RuntimeError("Invalid native token contract")
        fmap = native.mean(dim=(3,4)).permute(0,3,1,2).float()
        torch.testing.assert_close(normal.float(), fmap.flatten(2).transpose(1,2), rtol=0, atol=0)
        gh, gw = fmap.shape[-2:]
        target, valid, info = signal_target(cube, (gh, gw))
        if not valid.any():
            raise ValueError("Scene has no valid signal target cells")
        features = {"full": fmap}
        for width in (4,8):
            if min(gh,gw) < width:
                raise ValueError("Native grid too small")
            features[f"pool{width}_nearest"] = F.interpolate(F.adaptive_avg_pool2d(fmap, (width,width)), size=(gh,gw), mode="nearest")
        yy, xx = np.meshgrid((np.arange(gh)+.5)/gh*2-1, (np.arange(gw)+.5)/gw*2-1, indexing="ij")
        scene = {"case_id": c["case_id"], "mgrs": c["original_patch_metadata"]["mgrs"],
                 "target": target[valid], "coordinates": np.stack([yy,xx], -1)[valid]}
        for name, tensor in features.items():
            arr = tensor[0].permute(1,2,0).cpu().numpy()
            scene[name] = arr[valid].astype(np.float64)
        scenes.append(scene)
        arr = native.cpu().numpy()
        token_receipts.append({"case_id": c["case_id"], "native_shape": list(arr.shape), "native_dtype": str(arr.dtype),
                               "native_sha256": hashlib.sha256(arr.tobytes()).hexdigest(), "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                               "valid_mask_sha256": hashlib.sha256(valid.tobytes()).hexdigest(), **info})
        if args.save_features:
            native_arrays[f"native_{i:02d}"] = arr
            native_arrays[f"target_{i:02d}"] = target
            native_arrays[f"valid_{i:02d}"] = valid
        print(json.dumps({"event":"extracted", "scene":i+1, "native_shape":list(arr.shape), "elapsed":time.monotonic()-started}), flush=True)
    handle.remove()
    if before != model.capture_parameter_state() or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen encoder state/gradients changed")
    summary = evaluate(scenes, folds, args.out)
    if args.save_features:
        np.savez_compressed(args.out / "native_tokens_and_targets.npz", **native_arrays)
    save_json(args.out / "token_receipts.json", token_receipts)
    receipt = {"status":"completed_development_diagnostic", "limitations":LIMITS,
               "arguments":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               "source_sha256":{str(Path(__file__).resolve()):sha(__file__),str(args.eo_module):sha(args.eo_module),str(args.manifest):sha(args.manifest)},
               "model_provenance":model.provenance, "encoder_unchanged":True, "encoder_gradients_absent":True,
               "environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"torch":torch.__version__,
                              "cuda":torch.version.cuda,"device":str(args.device),"device_name":torch.cuda.get_device_name(torch.device(args.device)) if str(args.device).startswith("cuda") else "cpu"},
               "ridge":{"alpha":1.0,"loss":"mean_over_train_scenes(mean_valid_cells((y-pred)^2)) + alpha*||w||_2^2; intercept unpenalized",
                        "standardization":"equal-scene weighted mean/std fitted on training scenes only", "hyperparameter_selection":"none",
                        "train_scenes_per_fold":16,"held_out_scenes_per_fold":4,"fold_count":5,"fold_salt":SALT},
               "target":"mean per-pixel DN ratio (B08-B04)/(B08+B04) in each native cell; all constituent pixels finite,nonnegative,<65535,denominator>0",
               "spatial_restore":"torch adaptive_avg_pool2d to4/8 then interpolate nearest to native grid; full and compressed probes have identical D",
               "elapsed_seconds":time.monotonic()-started,"summary":summary}
    save_json(args.out / "summary.json", receipt)
    save_json(args.out / "status.json", {"status":"completed", "elapsed_seconds":time.monotonic()-started})
    print(json.dumps({"event":"completed","summary":summary}), flush=True)


if __name__ == "__main__":
    main()
