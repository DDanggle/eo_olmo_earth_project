#!/usr/bin/env python3
"""Bounded real-H5 OlmoEarth v1.2 objective engineering run; not CPT reproduction.

Uses the official full-source model loader, normalizer, masking and losses. No
olmo-core trainer, distributed run, augmentation, scheduler or downstream test.
The file-disjoint development loss is an engineering diagnostic, not an
independent geographic evaluation or an estimate of generalization.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback


PINNED_PUBLIC_CONFIG_SHA256 = "0d531a67ad3e477e7011efabcceb01ed80f430aa0a0a3d344fe18cec0f229b8a"
PINNED_V12_RECIPE_SHA256 = "d88b23daca3d8a5657ede3ccbe98ce7e2ea6cdbf0e7b026b24fd15e61205f9ee"


def official_recipe_loss_configs(recipe_path, modality_cls):
    """Read exact loss dictionaries from the pinned official recipe without executing it."""
    tree = ast.parse(recipe_path.read_text())
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}

    def literal(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [literal(child) for child in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.Dict):
            return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Name) and node.id == "ONLY_DECODE_MODALITIES":
            return literal(assignments[node.id])
        if (isinstance(node, ast.Attribute) and node.attr == "name" and
                isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name) and
                node.value.value.id == "Modality"):
            return getattr(modality_cls, node.value.attr).name
        raise ValueError(f"Unsupported official recipe expression: {ast.dump(node)}")

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_train_module_config"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one official build_train_module_config")
    constructors = [node.value for node in functions[0].body if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "ContrastiveLatentMIMTrainModuleConfig"]
    if len(constructors) != 1:
        raise ValueError("Expected exact official train module config constructor")
    result = {}
    for keyword in constructors[0].keywords:
        if keyword.arg not in ["loss_config", "contrastive_config"]:
            continue
        call = keyword.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "LossConfig":
            raise ValueError("Expected official LossConfig constructor")
        payloads = [k.value for k in call.keywords if k.arg == "loss_config"]
        if len(payloads) != 1:
            raise ValueError("Expected one official loss_config payload")
        result[keyword.arg] = literal(payloads[0])
    if set(result) != {"loss_config", "contrastive_config"}:
        raise ValueError("Incomplete official loss recipe")
    return result


def bridge_public_loss_registry_keys(train_config, config_sha256, source_root, modality_cls):
    """Restore only serialized-away registry keys with an exact pinned-source match.

    Official LossConfig.build mutates its input with pop('type'), and the official
    experiment writes config after constructing the train module. _CLASS_ names
    the Config dataclass, not the missing registered loss type.
    """
    recipe_path = source_root / "scripts/official/v1_2/base.py"
    recipe_hash = sha256(recipe_path)
    if recipe_hash != PINNED_V12_RECIPE_SHA256:
        raise ValueError("Public loss bridge requires the reviewed pinned v1.2 recipe SHA")
    expected = official_recipe_loss_configs(recipe_path, modality_cls)
    normalized = copy.deepcopy(train_config)
    changes = []
    for role in ["loss_config", "contrastive_config"]:
        observed = normalized[role]["loss_config"]
        if "type" not in observed:
            if config_sha256 != PINNED_PUBLIC_CONFIG_SHA256:
                raise ValueError("Missing loss registry key in an unreviewed checkpoint config; refusing inference")
            expected_parameters = {k: v for k, v in expected[role].items() if k != "type"}
            if observed != expected_parameters:
                raise ValueError(f"Public {role} parameters differ from pinned official recipe")
            observed["type"] = expected[role]["type"]
            changes.append({"path": f"train_module.{role}.loss_config.type", "from": "absent",
                            "to": expected[role]["type"], "evidence": str(recipe_path)})
        elif observed != expected[role]:
            raise ValueError(f"Explicit {role} differs from this pinned v1.2 objective probe")
    return normalized, {"source_recipe_sha256": recipe_hash, "checkpoint_config_sha256": config_sha256,
                        "changes": changes, "official_recipe_loss_configs": expected,
                        "scope": "Memory-only restore of registry keys; public config and weights are not edited",
                        "cause": "Official LossConfig.build pops type before experiment serializes the built train configuration"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--deps-root", type=Path)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--h5-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-files", type=int, default=32)
    p.add_argument("--dev-files", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop-size", type=int, default=32)
    p.add_argument("--timesteps", type=int, default=2)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=270927)
    p.add_argument("--max-wall-seconds", type=int, default=900)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--save-weights", action="store_true")
    a = p.parse_args()
    if not (2 <= a.batch_size <= 8 and 1 <= a.steps <= 64):
        p.error("Require batch size 2..8 and steps 1..64.")
    if not (a.batch_size <= a.train_files <= 128 and a.batch_size <= a.dev_files <= 32):
        p.error("Require train files batch_size..128 and dev files batch_size..32.")
    if a.train_files % a.batch_size or a.dev_files % a.batch_size:
        p.error("File counts must be multiples of batch size.")
    if not (16 <= a.crop_size <= 64 and a.crop_size % a.patch_size == 0 and 2 <= a.timesteps <= 4):
        p.error("Require crop16..64 divisible by patch size and timesteps2..4.")
    return a


def main():
    a = parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "initializing", "started_utc": datetime.now(timezone.utc).isoformat(),
               "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
               "scope": "real-H5 bounded objective engineering run; not official CPT reproduction",
               "development_split_limit": "file-disjoint only; neighboring subtiles may share parent region; no external-test claim",
               "declared_model_revision": "2e99a734a30e9aeb993aaa39946c6dcf554739e0"}
    output = a.out_dir / "receipt.json"
    write_json(output, receipt)
    start = time.monotonic()
    try:
        if a.deps_root:
            sys.path.insert(0, str(a.deps_root.resolve()))
        sys.path.insert(0, str(a.source_root.resolve()))
        import numpy as np
        import torch
        import h5py
        import hdf5plugin  # noqa: F401 -- registers official H5 Zstd filter.
        import olmoearth_pretrain
        from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
        from olmoearth_pretrain.data.normalize import Normalizer, Strategy
        from olmoearth_pretrain.datatypes import OlmoEarthSample, MaskValue
        from olmoearth_pretrain.model_loader import load_model_from_path
        from olmoearth_pretrain.train.loss import LossConfig
        from olmoearth_pretrain.train.masking import MaskingConfig

        package_file = Path(olmoearth_pretrain.__file__).resolve()
        if not package_file.is_relative_to(a.source_root.resolve()):
            raise RuntimeError(f"Unexpected model package imported: {package_file}")
        receipt["package_file"] = str(package_file)
        receipt["versions"] = {}
        for pkg in ["torch", "numpy", "einops", "h5py", "hdf5plugin", "class-registry", "universal-pathlib"]:
            try:
                receipt["versions"][pkg] = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                receipt["versions"][pkg] = None
        receipt["torch_declared_dependency"] = ">=2.9,<2.10; other installed versions require this compatibility probe"
        receipt["source_file_hashes"] = {p: sha256(a.source_root / p) for p in [
            "olmoearth_pretrain/model_loader.py", "olmoearth_pretrain/train/masking.py",
            "olmoearth_pretrain/train/loss.py", "scripts/official/v1_2/base.py"]}
        receipt["checkpoint_hashes"] = {p: sha256(a.checkpoint_dir / p) for p in ["config.json", "weights.pth"]}
        receipt["runtime_script_sha256"] = sha256(__file__)
        config = json.loads((a.checkpoint_dir / "config.json").read_text())
        train_config, loss_bridge = bridge_public_loss_registry_keys(
            config["train_module"], receipt["checkpoint_hashes"]["config.json"], a.source_root, Modality)
        receipt["public_loss_registry_bridge"] = loss_bridge
        if train_config.get("ema_decay") not in ([1.0, 1.0], (1.0, 1.0)):
            raise RuntimeError("This minimal loop only implements the official fixed projection target, EMA=(1,1).")
        if train_config.get("mae_loss_config") or train_config.get("regularizer_config"):
            raise RuntimeError("Checkpoint requires an additional objective not implemented in this bounded loop.")
        if any(v != 0 for v in train_config["token_exit_cfg"].values()):
            raise RuntimeError("Expected all target exits zero for v1.2 recipe.")
        # Build official loss objects before any data scan or model/GPU allocation.
        # In particular --prepare-only now validates the actual constructor path.
        base_loss = LossConfig(loss_config=copy.deepcopy(train_config["loss_config"]["loss_config"])).build()
        contrastive = LossConfig(loss_config=copy.deepcopy(train_config["contrastive_config"]["loss_config"])).build()
        mask_constructor_probe = MaskingConfig(strategy_config=copy.deepcopy(train_config["masking_config"]["strategy_config"])).build()
        receipt["cpu_objective_constructor_validation"] = {
            "base_loss_class": f"{type(base_loss).__module__}.{type(base_loss).__qualname__}",
            "contrastive_loss_class": f"{type(contrastive).__module__}.{type(contrastive).__qualname__}",
            "mask_strategy_class": f"{type(mask_constructor_probe).__module__}.{type(mask_constructor_probe).__qualname__}",
            "passed": True, "tokenization_mask_shapes": "validated after loading the actual model"}
        del mask_constructor_probe
        write_json(output, receipt)
        computed = Normalizer(Strategy.COMPUTED)
        predefined = Normalizer(Strategy.PREDEFINED)

        def normalize(name, raw):
            missing = raw == MISSING_VALUE
            try:
                norm = computed.normalize(Modality.get(name), raw)
                strategy = "computed"
            except (KeyError, ValueError):
                norm = predefined.normalize(Modality.get(name), raw)
                strategy = "predefined"
            norm = np.where(missing, MISSING_VALUE, norm).astype(np.float32)
            if not np.isfinite(norm).all():
                raise ValueError(f"Nonfinite normalized {name}")
            return norm, strategy

        def load_sample(path):
            with h5py.File(path, "r") as f:
                for key in ["sentinel2_l2a", "worldcover", "timestamps"]:
                    if key not in f:
                        raise ValueError(f"Missing key {key}")
                s2 = f["sentinel2_l2a"]
                if s2.ndim != 4 or s2.shape[-1] != 12:
                    raise ValueError(f"Expected S2 [H,W,T,12], got {s2.shape}")
                timestamps = np.asarray(f["timestamps"][()])
                if timestamps.ndim != 2 or timestamps.shape[1] != 3:
                    raise ValueError(f"Expected timestamps [T,3], got {timestamps.shape}")
                if not np.isfinite(timestamps).all() or not np.equal(timestamps, np.round(timestamps)).all():
                    raise ValueError("Nonintegral or nonfinite timestamp")
                if not ((timestamps[:, 0] >= 1).all() and (timestamps[:, 0] <= 31).all()
                        and (timestamps[:, 1] >= 0).all() and (timestamps[:, 1] <= 11).all()):
                    raise ValueError("Timestamp contract requires [day1..31,month0..11,year]")
                mask_key = "missing_timesteps_masks/sentinel2_l2a"
                if mask_key in f:
                    present = np.asarray(f[mask_key][()]).reshape(-1)
                    if len(present) != len(timestamps) or not np.isin(present, [0, 1]).all():
                        raise ValueError("Invalid S2 timestamp presence mask")
                    indices = np.flatnonzero(present)
                    if len(indices) != s2.shape[2]:
                        raise ValueError("Compact S2 timesteps do not match number of True presence flags")
                elif s2.shape[2] == len(timestamps):
                    indices = np.arange(len(timestamps))
                else:
                    raise ValueError("Missing presence mask for compact S2 timestamps")
                if len(indices) < a.timesteps:
                    raise ValueError("Too few observed S2 timesteps")
                h, w = s2.shape[:2]
                if min(h, w) < a.crop_size:
                    raise ValueError("Image smaller than crop")
                y, x = (h - a.crop_size) // 2, (w - a.crop_size) // 2
                raw_s2 = np.asarray(s2[y:y+a.crop_size, x:x+a.crop_size, :a.timesteps, :], dtype=np.float32)
                wc = f["worldcover"]
                if wc.shape[:2] != (h, w):
                    raise ValueError("WorldCover/S2 spatial shape mismatch; no silent resampling")
                raw_wc = np.asarray(wc[y:y+a.crop_size, x:x+a.crop_size], dtype=np.float32)
                if raw_wc.ndim == 3 and raw_wc.shape[-1] == 1:
                    raw_wc = raw_wc[:, :, None, :]
                if raw_wc.shape != (a.crop_size, a.crop_size, 1, 1):
                    raise ValueError(f"Unexpected static WorldCover shape {raw_wc.shape}")
                if not np.isfinite(raw_s2).all() or not np.isfinite(raw_wc).all():
                    raise ValueError("Nonfinite raw data")
                if np.any(raw_s2 == MISSING_VALUE) or np.any(raw_wc == MISSING_VALUE):
                    raise ValueError("Smoke requires fully observed crop; missing-pixel support is a later loader test")
                if not np.any(raw_s2 != 0):
                    raise ValueError("All-zero S2 crop")
                s2_norm, s2_strategy = normalize("sentinel2_l2a", raw_s2)
                wc_norm, wc_strategy = normalize("worldcover", raw_wc)
                selected_timestamps = timestamps[indices[:a.timesteps]].astype(np.int64)
                raw_hash = hashlib.sha256(raw_s2.tobytes() + raw_wc.tobytes() + selected_timestamps.tobytes()).hexdigest()
                row = {"file": str(path), "file_bytes": path.stat().st_size,
                       "crop_yxhw": [y, x, a.crop_size, a.crop_size], "raw_crop_sha256": raw_hash,
                       "compact_timesteps": list(range(a.timesteps)), "timestamp_indices": indices[:a.timesteps].tolist(),
                       "timestamps_day_month0_year": selected_timestamps.tolist(),
                       "source_s2_shape": list(s2.shape), "source_worldcover_shape": list(wc.shape),
                       "worldcover_unique_raw_values": np.unique(raw_wc).tolist(),
                       "normalization": {"sentinel2_l2a": s2_strategy, "worldcover": wc_strategy},
                       "latlon": np.asarray(f["latlon"][()]).tolist() if "latlon" in f else None}
                return {"sentinel2_l2a": s2_norm, "worldcover": wc_norm, "timestamps": selected_timestamps}, row

        paths = list(a.h5_root.rglob("*.h5")) + list(a.h5_root.rglob("*.hdf5"))
        # Choose the whole development subset before any model loss is observed.
        paths = sorted(set(paths), key=lambda p: hashlib.sha256(f"{a.seed}:{p.relative_to(a.h5_root)}".encode()).hexdigest())
        samples, manifest, rejected = [], [], []
        for path in paths:
            try:
                sample, row = load_sample(path)
                # Reject exact raw-crop duplicates across both diagnostic subsets.
                if row["raw_crop_sha256"] in {r["raw_crop_sha256"] for r in manifest}:
                    raise ValueError("Duplicate selected crop")
                samples.append(sample)
                manifest.append(row)
            except (ValueError, OSError, KeyError) as e:
                rejected.append({"file": str(path), "reason": str(e)})
            if len(samples) == a.train_files + a.dev_files:
                break
        if len(samples) != a.train_files + a.dev_files:
            raise RuntimeError(f"Need {a.train_files+a.dev_files} accepted files; got {len(samples)}; rejected={rejected[:8]}")
        for i, row in enumerate(manifest):
            row["split"] = "train_diagnostic" if i < a.train_files else "dev_diagnostic"
        data_manifest = {"selected": manifest, "rejected": rejected,
                         "selection": "seed-hash ordered filenames; center crop; first present S2 times; no loss-guided selection",
                         "no_independent_geography_claim": True}
        write_json(a.out_dir / "data_manifest.json", data_manifest)
        receipt["data_manifest_sha256"] = sha256(a.out_dir / "data_manifest.json")
        receipt["data_summary"] = {"train_files": a.train_files, "dev_files": a.dev_files, "rejected": len(rejected)}
        receipt["objective_configs"] = {k: train_config.get(k) for k in ["masking_config", "loss_config", "contrastive_config", "token_exit_cfg", "ema_decay"]}
        receipt["status"] = "data_prepared_not_gpu_validated"
        write_json(output, receipt)
        if a.prepare_only:
            print(json.dumps({"status": receipt["status"], "receipt": str(output)}), flush=True)
            return
        device = torch.device(a.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        random.seed(a.seed)
        model = load_model_from_path(str(a.checkpoint_dir)).to(device)
        # Public weights load unchanged. SDPA is supported by the official model;
        # disable the optional external flash-attn backend for this portable probe.
        backend_changes = []
        for name, module in model.named_modules():
            if getattr(module, "use_flash_attn", False):
                module.use_flash_attn = False
                backend_changes.append(name)
        receipt["flash_attn_disabled_modules"] = backend_changes
        receipt["precision"] = "fp32 parameters/forward/loss; no autocast; no FSDP"
        # Official trainer explicitly enables this only on the online encoder;
        # model.train() alone does not activate the configured band dropout.
        model.encoder.enable_band_dropout()
        receipt["band_dropout"] = {
            "online_effective_max_rate": float(model.encoder.patch_embeddings.band_dropout_rate),
            "online_random_rate": bool(model.encoder.patch_embeddings.random_band_dropout),
            "target_effective_max_rate": float(model.target_encoder.patch_embeddings.band_dropout_rate),
            "enable_scope": "online encoder only, matching official trainer initializer; eval mode disables it",
        }
        if receipt["band_dropout"]["target_effective_max_rate"] != 0:
            raise RuntimeError("Expected frozen target band dropout disabled")
        if model.encoder.tokenization_config.get_num_bandsets("sentinel2_l2a") != 1:
            raise RuntimeError("Expected v1.2 S2 single bandset")
        if any(p.requires_grad for p in model.target_encoder.parameters()):
            raise RuntimeError("Official target must be frozen")
        receipt["parameters"] = {"encoder": sum(p.numel() for p in model.encoder.parameters()),
                                  "all_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        masker = MaskingConfig(strategy_config=copy.deepcopy(train_config["masking_config"]["strategy_config"]),
                               tokenization_config=model.encoder.tokenization_config).build()
        train_ids = list(range(a.train_files))
        dev_ids = list(range(a.train_files, a.train_files + a.dev_files))

        def batch(ids):
            return OlmoEarthSample(**{k: torch.from_numpy(np.stack([samples[i][k] for i in ids])).to(device)
                                      for k in ["sentinel2_l2a", "worldcover", "timestamps"]})

        def objective(ids, seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            b = batch(ids)
            masked = [masker.apply_mask(b, patch_size=a.patch_size) for _ in range(2)]
            losses, pooled, counts = [], [], []
            for v in masked:
                count = {}
                for name in ["sentinel2_l2a", "worldcover"]:
                    mask = getattr(v, name + "_mask")
                    count[name] = {str(int(k)): int(n) for k, n in zip(*torch.unique(mask, return_counts=True))}
                if int((v.sentinel2_l2a_mask == MaskValue.ONLINE_ENCODER.value).sum()) == 0:
                    raise RuntimeError("Masker produced no online S2 tokens")
                if int((v.worldcover_mask == MaskValue.ONLINE_ENCODER.value).sum()) != 0:
                    raise RuntimeError("Decode-only WorldCover appeared in online input")
                _, decoded, projected, _, _ = model(v, a.patch_size)
                with torch.no_grad():
                    target = model.target_encoder(v.unmask(), patch_size=a.patch_size,
                                                  token_exit_cfg=train_config["token_exit_cfg"])["tokens_and_masks"]
                with torch.autocast(device_type=device.type, enabled=False):
                    losses.append(base_loss.compute(decoded, target))
                pooled.append(projected)
                counts.append(count)
            with torch.autocast(device_type=device.type, enabled=False):
                con = contrastive.compute(pooled[0], pooled[1])
                base = (losses[0] + losses[1]) / 2
                total = base + con
            if not bool(torch.isfinite(total)):
                raise RuntimeError("Nonfinite loss")
            return total, {"total": float(total.detach()), "base": float(base.detach()), "contrastive": float(con.detach()), "mask_counts": counts}

        def evaluate():
            model.eval()
            rows = []
            with torch.no_grad():
                for k in range(0, len(dev_ids), a.batch_size):
                    _, row = objective(dev_ids[k:k+a.batch_size], a.seed + 100000 + k)
                    rows.append(row)
            return {"mean_total": float(np.mean([r["total"] for r in rows])),
                    "mean_base": float(np.mean([r["base"] for r in rows])),
                    "mean_contrastive": float(np.mean([r["contrastive"] for r in rows])),
                    "base_positive_batches": sum(r["base"] > 0 for r in rows),
                    "contrastive_positive_batches": sum(r["contrastive"] > 0 for r in rows), "batches": rows}

        def encoder_probe():
            """Retain one full S2 token tensor for an exact same-input reload check."""
            model.eval()
            torch.manual_seed(a.seed + 200000)
            np.random.seed(a.seed + 200000)
            random.seed(a.seed + 200000)
            with torch.no_grad():
                masked = masker.apply_mask(batch(dev_ids[:a.batch_size]), patch_size=a.patch_size)
                tokens = model.encoder(masked, patch_size=a.patch_size)["tokens_and_masks"]
                return tokens.sentinel2_l2a.detach().cpu().clone()

        receipt["dev_before"] = evaluate()
        receipt["status"] = "official_forward_and_objective_passed"
        write_json(output, receipt)
        print(json.dumps({"status": receipt["status"], "dev_before": receipt["dev_before"]["mean_total"]}), flush=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
        tracked = {}
        logs = []
        optimizer_start = time.monotonic()
        for step in range(a.steps):
            if time.monotonic() - start >= a.max_wall_seconds:
                raise RuntimeError("Wall budget reached; partial log retained")
            offset = (step * a.batch_size) % len(train_ids)
            ids = train_ids[offset:offset+a.batch_size]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            t0 = time.monotonic()
            loss, row = objective(ids, a.seed + step + 1)
            loss.backward()
            if any(p.grad is not None for p in model.target_encoder.parameters()):
                raise RuntimeError("Frozen target received a gradient")
            encoder_grad_names = [n for n, p in model.encoder.named_parameters()
                                  if p.grad is not None and bool(torch.count_nonzero(p.grad))]
            if not encoder_grad_names:
                raise RuntimeError("No nonzero encoder gradients")
            if step == 0:
                wanted = encoder_grad_names[:4] + [n for n in encoder_grad_names if "attn" in n][:4]
                for n, p in model.encoder.named_parameters():
                    if n in wanted:
                        tracked[n] = p.detach().clone()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row.update({"step": step+1, "sample_indices": ids, "grad_norm_before_clip": float(grad_norm),
                        "encoder_nonzero_grad_parameter_tensors": len(encoder_grad_names), "seconds": time.monotonic()-t0})
            logs.append(row)
            with (a.out_dir / "train_log.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "mask_counts"}), flush=True)
        receipt["optimizer_seconds"] = time.monotonic() - optimizer_start
        receipt["completed_steps"] = len(logs)
        receipt["base_loss_positive_steps"] = sum(r["base"] > 0 for r in logs)
        receipt["contrastive_loss_positive_steps"] = sum(r["contrastive"] > 0 for r in logs)
        receipt["base_loss_activity_note"] = (
            "Flat WorldCover patches can have no valid contrastive negatives; zero native patch loss is retained and reported, not repaired by changing the official loss."
        )
        receipt["objective_activity_scope"] = (
            "Positive native patch and InfoNCE losses were observed; this does not prove full multimodal objective coverage."
            if receipt["base_loss_positive_steps"] > 0 and receipt["contrastive_loss_positive_steps"] > 0
            else "One or more loss components were inactive; encoder-update pass must not be called a two-objective training pass."
        )
        receipt["train_samples_exposed"] = sorted({i for row in logs for i in row["sample_indices"]})
        receipt["encoder_tracked_parameter_max_abs_delta"] = {
            n: float((p.detach()-tracked[n]).abs().max()) for n, p in model.encoder.named_parameters() if n in tracked}
        if not any(v > 0 for v in receipt["encoder_tracked_parameter_max_abs_delta"].values()):
            raise RuntimeError("Optimizer produced no detected encoder parameter change")
        receipt["dev_after"] = evaluate()
        receipt["dev_total_delta_after_minus_before"] = receipt["dev_after"]["mean_total"] - receipt["dev_before"]["mean_total"]
        receipt["dev_metric_interpretation"] = "Paired same-mask, eval-mode file-disjoint objective diagnostic; sign is not a pass criterion or downstream result."
        receipt["gpu_max_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        if a.save_weights:
            probe_before_reload = encoder_probe()
            target_dir = a.out_dir / "checkpoint_after"
            target_dir.mkdir()
            (target_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, target_dir / "weights.pth")
            receipt["saved_checkpoint"] = {"path": str(target_dir), "weights_sha256": sha256(target_dir / "weights.pth"),
                                           "optimizer_state_saved": False, "scope": "inference/loadable model snapshot, not full trainer resume"}
            # Free the optimizer and trained GPU model before loading the saved
            # snapshot through the real official loader; no fabricated module.
            del optimizer
            model.cpu()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            model = load_model_from_path(str(target_dir)).to(device)
            for module in model.modules():
                if getattr(module, "use_flash_attn", False):
                    module.use_flash_attn = False
            probe_after_reload = encoder_probe()
            if probe_before_reload.shape != probe_after_reload.shape:
                raise RuntimeError("Saved/reloaded encoder token shape changed")
            delta = (probe_before_reload - probe_after_reload).abs()
            parity = {"encoder_token_shape": list(probe_before_reload.shape),
                      "encoder_max_abs_delta": float(delta.max()), "encoder_mean_abs_delta": float(delta.mean()),
                      "encoder_allclose_atol": 1e-6, "encoder_allclose_rtol": 1e-5,
                      "encoder_allclose": bool(torch.allclose(probe_before_reload, probe_after_reload, atol=1e-6, rtol=1e-5)),
                      "dev_objective_before_reload": receipt["dev_after"]["mean_total"]}
            reloaded_dev = evaluate()
            parity["dev_objective_after_reload"] = reloaded_dev["mean_total"]
            parity["dev_objective_abs_delta"] = abs(parity["dev_objective_after_reload"]-parity["dev_objective_before_reload"])
            # Paired evaluation uses identical actual observations and RNG masks;
            # batchwise component checks are stricter than comparing averages.
            parity["dev_components_allclose"] = all(
                np.isclose(before[k], after[k], atol=1e-6, rtol=1e-5)
                for before, after in zip(receipt["dev_after"]["batches"], reloaded_dev["batches"])
                for k in ["total", "base", "contrastive"]
            )
            receipt["saved_checkpoint_reload_parity"] = parity
            write_json(output, receipt)
            if not parity["encoder_allclose"] or not parity["dev_components_allclose"]:
                raise RuntimeError("Official loader saved-checkpoint parity failed")
        receipt["status"] = "PASS_bounded_encoder_update"
        receipt["wall_seconds"] = time.monotonic() - start
        receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output, receipt)
        print(json.dumps({"status": receipt["status"], "receipt": str(output), "dev_delta": receipt["dev_total_delta_after_minus_before"]}), flush=True)
    except Exception as e:
        receipt.update({"status": "FAILED", "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc(), "wall_seconds": time.monotonic()-start})
        write_json(output, receipt)
        raise


if __name__ == "__main__":
    main()
