#!/usr/bin/env python3
"""GPU1에서 Sen12 S12q → frozen OlmoEarth v1 spatial cache의 실행 계약을 잰다.

성능 probe가 아니라 G-P 직전 smoke다. 10개 task-eligible region의 positive/negative를
결정적으로 층화하고, 128x128 patch를 공식 crop size와 같은 64x64 네 조각으로 처리한다.
Sen12에 없는 B01/B09는 숫자 0으로 채우되 v1 band-set 2 전체를 MISSING으로 표시하므로
encoder/pooling에 기여하지 않는다. 출력은 네 16x16 feature map을 32x32로 붙인 768-d grid다.
OlmoEarth v1 time embedding table은 최대 12시점이므로 Sen12의 15시점 중 SCL clear fraction이
높은 12개를 label-independent하게 고른 뒤 다시 시간순으로 정렬한다. 모든 baseline도 같은 12개를 쓴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from build_sen12_gp_contract import BANDS, HEADLINE_REGIONS


MODEL_BANDS = (
    "B02", "B03", "B04", "B08",  # v1 band-set 0
    "B05", "B06", "B07", "B8A", "B11", "B12",  # band-set 1
    "B01", "B09",  # band-set 2: Sen12 missing
)
CROP = 64
PATCH = 4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_stratified(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """region×label bucket을 round-robin해 큰 지역이 smoke를 독식하지 못하게 한다."""
    buckets: list[list[dict[str, Any]]] = []
    for region in HEADLINE_REGIONS:
        for positive in (True, False):
            bucket = sorted(
                (
                    r for r in records
                    if r.get("region") == region
                    and r.get("s15_eligible") is True
                    and r.get("label_positive") is positive
                ),
                key=lambda r: r["sample_id"],
            )
            if not bucket:
                raise ValueError(f"empty smoke stratum: {region}, positive={positive}")
            buckets.append(bucket)

    selected = []
    depth = 0
    while len(selected) < limit:
        added = False
        for bucket in buckets:
            if depth < len(bucket):
                selected.append(bucket[depth])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    if len(selected) != limit:
        raise ValueError(f"requested {limit} samples, selected {len(selected)}")
    return selected


def select_timestep_indices(record: dict[str, Any], keep: int = 12) -> list[int]:
    quality = record.get("scl_clear_fraction")
    if not isinstance(quality, list) or len(quality) != 15:
        raise ValueError(f"invalid SCL quality vector: {record.get('sample_id')}")
    chosen = sorted(range(len(quality)), key=lambda i: (-float(quality[i]), i))[:keep]
    return sorted(chosen)


def load_cube(path: Path, indices: list[int]):
    import numpy as np
    import xarray as xr

    with xr.open_dataset(path, decode_times=True, cache=False) as ds:
        arrays = []
        for band in MODEL_BANDS:
            if band in ds:
                arrays.append(np.asarray(ds[band].values[indices], dtype="float32"))
            else:
                arrays.append(np.zeros((len(indices), 128, 128), dtype="float32"))
        cube = np.stack(arrays, axis=0)  # C,T,H,W
        times = [
            datetime.fromisoformat(str(np.datetime_as_string(t, unit="s")))
            for t in np.asarray(ds["time"].values)[indices]
        ]
    return cube, times


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl"))
    p.add_argument("--data-root", type=Path,
                   default=Path("/home/work/data/sen12landslides/extracted"))
    p.add_argument("--out", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_olmo_v1_smoke"))
    p.add_argument("--limit", type=int, default=64)
    return p.parse_args()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("refusing to run: CUDA_VISIBLE_DEVICES must be exactly 1")

    import numpy as np
    import torch
    from olmoearth_pretrain_minimal import ModelID
    from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "cache_fp16"
    cache_dir.mkdir(exist_ok=True)
    records = [
        json.loads(line) for line in args.contract.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected = select_stratified(records, args.limit)

    device = torch.device("cuda")
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=1이므로 physical GPU1이다.
    torch.manual_seed(20260825)
    load_start = time.perf_counter()
    wrapper = OlmoEarth(
        patch_size=PATCH,
        model_id=ModelID.OLMOEARTH_V1_BASE,
        token_pooling=True,
        use_legacy_timestamps=False,
        normalize=True,
        autocast_dtype="bfloat16",
    ).to(device).eval()
    model_load_s = time.perf_counter() - load_start
    torch.cuda.reset_peak_memory_stats(device)

    def embed_crop(crop, timestamps):
        image = torch.from_numpy(crop).to(device)
        ranges = [(t, t) for t in timestamps]
        input_dict = {"sentinel2_l2a": RasterImage(image=image, timestamps=ranges)}
        # official wrapper와 같은 normalization을 먼저 적용한다.
        wrapper.normalizer(input_dict, {})
        context = ModelContext(inputs=[input_dict], metadatas=[])
        sample, present, _ = wrapper._prepare_modality_inputs(context)
        assert present == ["sentinel2_l2a"]
        # Sen12의 B01/B09 부재를 v1의 독립 band-set missing으로 표현한다.
        sample.sentinel2_l2a_mask[..., 2] = MaskValue.MISSING.value
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = wrapper.model(sample, fast_pass=False, patch_size=PATCH)
            tm = output["tokens_and_masks"]
            tokens = tm.sentinel2_l2a
            mask = (tm.sentinel2_l2a_mask != MaskValue.MISSING.value).unsqueeze(-1)
            count = mask.sum(dim=(3, 4)).clamp(min=1)
            pooled = (tokens * mask).sum(dim=(3, 4)) / count  # B,H,W,C
            feature = pooled[0].permute(2, 0, 1).contiguous()  # C,H,W
        return feature

    def embed_sample(record):
        indices = select_timestep_indices(record)
        cube, timestamps = load_cube(args.data_root / record["file"], indices)
        output = torch.empty((768, 32, 32), dtype=torch.float32, device="cpu")
        for y0, x0 in ((0, 0), (0, 64), (64, 0), (64, 64)):
            feature = embed_crop(cube[:, :, y0:y0 + CROP, x0:x0 + CROP], timestamps)
            output[:, y0 // PATCH:(y0 + CROP) // PATCH,
                   x0 // PATCH:(x0 + CROP) // PATCH] = feature.float().cpu()
        return output, indices

    run_start = time.perf_counter()
    outputs = []
    replay_reference = None
    for i, record in enumerate(selected, 1):
        t0 = time.perf_counter()
        feature, timestep_indices = embed_sample(record)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if i == 1:
            replay_reference = feature.clone()
        target = cache_dir / f"{record['sample_id']}.npy"
        np.save(target, feature.numpy().astype("float16"), allow_pickle=False)
        outputs.append({
            "sample_id": record["sample_id"],
            "region": record["region"],
            "label_positive": record["label_positive"],
            "seconds": round(elapsed, 6),
            "shape": list(feature.shape),
            "finite": bool(torch.isfinite(feature).all()),
            "mean": float(feature.mean()),
            "std": float(feature.std()),
            "cache_bytes": target.stat().st_size,
            "cache_sha256": sha256_file(target),
            "timestep_indices": timestep_indices,
        })
        print(f"[{i}/{len(selected)}] {record['sample_id']} {elapsed:.3f}s", flush=True)
    torch.cuda.synchronize()
    inference_s = time.perf_counter() - run_start

    replay, _ = embed_sample(selected[0])
    torch.cuda.synchronize()
    replay_max_abs = float((replay - replay_reference).abs().max())
    peak = int(torch.cuda.max_memory_allocated(device))
    summary = {
        "schema": "sen12-olmo-v1-embedding-smoke-v1",
        "input_contract_sha256": sha256_file(args.contract),
        "model_id": "OLMOEARTH_V1_BASE",
        "missing_contract": "B01/B09 zeros + band-set 2 MaskValue.MISSING",
        "crop_size": CROP,
        "patch_size": PATCH,
        "source_timesteps": 15,
        "model_timesteps": 12,
        "timestep_policy": "top-12 SCL clear fraction, then chronological; label-independent",
        "samples": len(outputs),
        "crops": len(outputs) * 4,
        "model_load_s": round(model_load_s, 6),
        "inference_and_write_s": round(inference_s, 6),
        "samples_per_s": round(len(outputs) / inference_s, 6),
        "peak_cuda_bytes": peak,
        "cache_total_bytes": sum(r["cache_bytes"] for r in outputs),
        "cache_bytes_per_sample": outputs[0]["cache_bytes"] if outputs else None,
        "replay_max_abs_diff": replay_max_abs,
        "outputs": outputs,
        "gates": {
            "sample_count": len(outputs) == args.limit,
            "shape_768x32x32": all(r["shape"] == [768, 32, 32] for r in outputs),
            "all_finite": all(r["finite"] for r in outputs),
            "deterministic_replay": replay_max_abs == 0.0,
            "all_10_regions": {r["region"] for r in outputs} == set(HEADLINE_REGIONS),
            "both_labels": {r["label_positive"] for r in outputs} == {True, False},
        },
    }
    summary["all_gates_pass"] = all(summary["gates"].values())
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "outputs"},
                     ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
