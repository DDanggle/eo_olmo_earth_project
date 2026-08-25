#!/usr/bin/env python3
"""G-P pilot 1단계 — LOCO fold 하나의 S12q 캐시를 만든다. GPU1 전용.

한 번의 NetCDF 읽기로 **두 캐시를 동시에** 쓴다. 같은 파일을 두 번 열지 않는다.

  emb_fp16/<id>.npy    frozen OlmoEarth v1 spatial cache  768x32x32  fp16   (P4 입력)
  raw_u16/<id>.npy     S12q raw 큐브 (실관측 10밴드)      10x12x128x128 uint16 (P1/P2 입력)
  mask_u8/<id>.npy     MASK (시간 불변이므로 t=0)          128x128 uint8

S12q 계약(동결):
  - 15 timestep 중 SCL clear fraction 상위 12개를 **라벨을 보지 않고** 고르고 시간순 복원
  - 모든 arm(P1/P2/P4)이 **같은 12 index**를 쓴다. 그래야 모델 차이와 입력 정보량 차이가 섞이지 않는다
  - Sen12는 B01/B09가 없으므로 v1의 band-set 2를 `MaskValue.MISSING`으로 표시한다
  - 128x128을 64 crop 4장으로 나눠 임베딩하고 32x32 토큰 격자로 이어붙인다 (smoke와 동일)

raw 큐브를 uint16으로 저장하는 이유: 원본 반사도가 int16 [0,10000] 이므로 무손실이고,
fp16과 같은 2바이트다. 정규화는 학습 직전에 arm별로 명시적으로 한다.
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

# 정규화기는 `num_timesteps = image.shape[0] // len(band_names)` 로 계산하므로
# **12밴드 전부**를 dim0에 두어야 한다 (B01/B09는 0으로 채움). 처음에 10개만 넣어
# 12로 나눈 몫이 0이 되어 정규화가 한 번도 돌지 않았고 norm.py:93에서 거부됐다.
MODEL_BANDS = ["B02", "B03", "B04", "B08",                       # v1 band-set 0
               "B05", "B06", "B07", "B8A", "B11", "B12",         # band-set 1
               "B01", "B09"]                                     # band-set 2: Sen12 부재
REAL_BANDS = 10          # raw 캐시에는 실제 관측 10밴드만 저장한다 (B01/B09는 0이므로 무의미)
PATCH, CROP = 4, 64


def select_timestep_indices(record: dict[str, Any], keep: int = 12) -> list[int]:
    quality = record.get("scl_clear_fraction")
    if not isinstance(quality, list) or len(quality) != 15:
        raise ValueError(f"invalid SCL quality vector: {record.get('sample_id')}")
    chosen = sorted(range(len(quality)), key=lambda i: (-float(quality[i]), i))[:keep]
    return sorted(chosen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl"))
    p.add_argument("--folds", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/loco_folds.json"))
    p.add_argument("--fold", type=str, default="holdout_chimanimani")
    p.add_argument("--data-root", type=Path,
                   default=Path("/home/work/data/sen12landslides/extracted"))
    p.add_argument("--out", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_pilot"))
    p.add_argument("--limit", type=int, default=0, help="0이면 전체")
    return p.parse_args()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("refusing to run: CUDA_VISIBLE_DEVICES must be exactly 1")

    import numpy as np
    import torch
    import xarray as xr
    from olmoearth_pretrain_minimal import ModelID
    from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage

    args = parse_args()
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fold = next((f for f in folds["folds"] if f["fold"] == args.fold), None)
    if fold is None:
        raise SystemExit(f"fold 없음: {args.fold} (가능: "
                         f"{[f['fold'] for f in folds['folds']][:3]}…)")

    records = {}
    for line in args.contract.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            records[r["sample_id"]] = r

    # fold의 split 멤버십. 봉인된 해시와 대조한다.
    split_ids: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        regions = (fold["train_regions"] if split == "train"
                   else [fold["val_region"]] if split == "val"
                   else [fold["test_region"]])
        # 봉인과 **같은 필터**를 써야 해시가 맞는다 (build_sen12_gp_contract.build_loco_folds):
        #   region in HEADLINE_REGIONS · not error · s15_eligible
        # 처음에 이 필터를 빼서 hiroshima label anomaly 2건이 섞여 해시가 어긋났다.
        ids = sorted(sid for sid, r in records.items()
                     if r["region"] in regions and not r.get("error")
                     and r.get("s15_eligible", True))
        got = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        want = fold["sample_sha256"][split]
        if got != want:
            raise SystemExit(f"{split} split 해시 불일치\n  기대 {want}\n  실제 {got}")
        split_ids[split] = ids
    print(f"fold {args.fold} 해시 3/3 일치: "
          + " ".join(f"{k}={len(v)}" for k, v in split_ids.items()), flush=True)

    ordered = [sid for s in ("train", "val", "test") for sid in split_ids[s]]
    if args.limit:
        ordered = ordered[:args.limit]

    out = args.out / args.fold
    dirs = {k: out / k for k in ("emb_fp16", "raw_u16", "mask_u8")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=1 이므로 physical GPU1
    load0 = time.perf_counter()
    wrapper = OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE,
                        token_pooling=True, use_legacy_timestamps=False,
                        normalize=True, autocast_dtype="bfloat16").to(device).eval()
    model_load_s = time.perf_counter() - load0
    torch.cuda.reset_peak_memory_stats(device)

    def embed_crop(crop, timestamps):
        image = torch.from_numpy(crop).to(device)
        input_dict = {"sentinel2_l2a": RasterImage(image=image,
                                                   timestamps=[(t, t) for t in timestamps])}
        wrapper.normalizer(input_dict, {})
        context = ModelContext(inputs=[input_dict], metadatas=[])
        sample, present, _ = wrapper._prepare_modality_inputs(context)
        assert present == ["sentinel2_l2a"]
        sample.sentinel2_l2a_mask[..., 2] = MaskValue.MISSING.value   # B01/B09 부재
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = wrapper.model(sample, fast_pass=False, patch_size=PATCH)
            tm = output["tokens_and_masks"]
            m = (tm.sentinel2_l2a_mask != MaskValue.MISSING.value).unsqueeze(-1)
            pooled = (tm.sentinel2_l2a * m).sum(dim=(3, 4)) / m.sum(dim=(3, 4)).clamp(min=1)
            return pooled[0].permute(2, 0, 1).contiguous()

    done = {k: 0 for k in dirs}
    t_start = time.perf_counter()
    per_sample = []
    for i, sid in enumerate(ordered, 1):
        rec = records[sid]
        e_p, r_p, m_p = (dirs["emb_fp16"] / f"{sid}.npy",
                         dirs["raw_u16"] / f"{sid}.npy", dirs["mask_u8"] / f"{sid}.npy")
        if e_p.exists() and r_p.exists() and m_p.exists():
            for k in dirs:
                done[k] += 1
            continue
        idx = select_timestep_indices(rec)
        t0 = time.perf_counter()
        with xr.open_dataset(args.data_root / rec["file"], decode_times=True,
                            cache=False) as ds:
            bands = []
            for b in MODEL_BANDS:
                if b in ds:
                    bands.append(np.asarray(ds[b].values[idx], dtype="float32"))
                else:
                    bands.append(np.zeros((len(idx), 128, 128), dtype="float32"))
            cube = np.stack(bands, axis=0)                      # C,T,H,W
            times = [datetime.fromisoformat(str(np.datetime_as_string(t, unit="s")))
                     for t in np.asarray(ds["time"].values)[idx]]
            mask = np.asarray(ds["MASK"].values[idx[0]], dtype="uint8")
        io_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        feat = torch.empty((768, 32, 32), dtype=torch.float32, device="cpu")
        for y0, x0 in ((0, 0), (0, 64), (64, 0), (64, 64)):
            f = embed_crop(cube[:, :, y0:y0 + CROP, x0:x0 + CROP], times)
            feat[:, y0 // PATCH:(y0 + CROP) // PATCH,
                 x0 // PATCH:(x0 + CROP) // PATCH] = f.float().cpu()
        torch.cuda.synchronize()
        gpu_s = time.perf_counter() - t1

        np.save(e_p, feat.numpy().astype("float16"), allow_pickle=False)
        # 원본 반사도는 int16 [0,10000] 이므로 uint16 저장이 무손실이다.
        # raw 캐시는 실제 관측 10밴드만. B01/B09는 0이므로 저장하지 않는다.
        np.save(r_p, np.clip(cube[:REAL_BANDS], 0, 65535).astype("uint16"),
                allow_pickle=False)
        np.save(m_p, mask, allow_pickle=False)
        for k in dirs:
            done[k] += 1
        per_sample.append({"sample_id": sid, "io_s": round(io_s, 4),
                           "gpu_s": round(gpu_s, 4)})
        if i % 250 == 0 or i == len(ordered):
            el = time.perf_counter() - t_start
            print(f"  [{i}/{len(ordered)}] {el/60:.1f}분 경과 · "
                  f"{i/max(el,1e-9):.2f} sample/s · 남은 {(len(ordered)-i)/max(i/el,1e-9)/60:.1f}분",
                  flush=True)

    total_s = time.perf_counter() - t_start
    sizes = {k: sum(p.stat().st_size for p in d.glob("*.npy")) for k, d in dirs.items()}
    summary = {
        "schema": "sen12-fold-cache-v1",
        "fold": args.fold,
        "split_counts": {k: len(v) for k, v in split_ids.items()},
        "split_sha256_verified": True,
        "samples_written": len(per_sample),
        "samples_total": len(ordered),
        "model_load_s": round(model_load_s, 3),
        "total_s": round(total_s, 2),
        "samples_per_s": round(len(per_sample) / total_s, 4) if per_sample else None,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cache_bytes": sizes,
        "cache_bytes_total": sum(sizes.values()),
        "timestep_policy": "top-12 SCL clear fraction, then chronological; label-independent",
        "missing_contract": "B01/B09 zeros + band-set 2 MaskValue.MISSING",
        "raw_dtype": "uint16 (lossless for int16 reflectance)",
        "input_contract_sha256": folds.get("sample_contract_sha256"),
    }
    (out / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
