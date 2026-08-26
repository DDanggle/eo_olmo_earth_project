#!/usr/bin/env python3
"""E1 — 128x128을 **한 번에** 인코딩한 캐시를 만든다. GPU1 전용.

기존 캐시(M30의 P4)는 128을 64 네 조각으로 **독립** 인코딩해 이어붙였다.
그러면 조각 사이 self-attention 문맥이 없고 x/y=64에 seam이 생길 수 있다.
UNet3D(P2)는 128 전체를 한 번에 본다 — 이 차이가 열세의 원인인지 분리한다.

M32에서 해상도 가설은 이미 기각됐다(토큰 격자 천장 0.607 vs P4 실제 0.131).
그래서 이 실험이 남은 첫 용의자다.

기존 캐시와 **토큰 격자·dtype·밴드 계약을 동일하게** 유지한다. 바꾸는 것은 crop 하나뿐이다.
"""
from __future__ import annotations
import argparse, json, os, time
from datetime import datetime
from pathlib import Path

MODEL_BANDS = ["B02", "B03", "B04", "B08",
               "B05", "B06", "B07", "B8A", "B11", "B12",
               "B01", "B09"]
PATCH = 4
FULL = 128


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl"))
    p.add_argument("--folds", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/loco_folds.json"))
    p.add_argument("--fold", default="holdout_chimanimani")
    p.add_argument("--data-root", type=Path,
                   default=Path("/home/work/data/sen12landslides/extracted"))
    p.add_argument("--out", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_pilot_full128"))
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("refusing to run: CUDA_VISIBLE_DEVICES must be exactly 1")

    import numpy as np, torch, xarray as xr
    from olmoearth_pretrain_minimal import ModelID
    from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "efc", Path(__file__).parent / "extract_sen12_fold_cache.py")
    efc = importlib.util.module_from_spec(spec); spec.loader.exec_module(efc)

    args = parse_args()
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fold = next(f for f in folds["folds"] if f["fold"] == args.fold)
    records = {}
    for line in args.contract.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line); records[r["sample_id"]] = r

    import hashlib
    ordered = []
    for split in ("train", "val", "test"):
        regions = (fold["train_regions"] if split == "train"
                   else [fold["val_region"]] if split == "val" else [fold["test_region"]])
        ids = sorted(sid for sid, r in records.items()
                     if r["region"] in regions and not r.get("error")
                     and r.get("s15_eligible", True))
        got = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if got != fold["sample_sha256"][split]:
            raise SystemExit(f"{split} split 해시 불일치")
        ordered += ids
    print(f"해시 3/3 일치 · {len(ordered)} samples", flush=True)
    if args.limit:
        ordered = ordered[:args.limit]

    out = args.out / args.fold
    emb_dir = out / "emb_fp16"; emb_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.cuda.set_device(0)
    wrapper = OlmoEarth(patch_size=PATCH, model_id=ModelID.OLMOEARTH_V1_BASE,
                        token_pooling=True, use_legacy_timestamps=False,
                        normalize=True, autocast_dtype="bfloat16").to(device).eval()
    torch.cuda.reset_peak_memory_stats(device)

    def embed_full(cube, timestamps):
        image = torch.from_numpy(cube).to(device)
        input_dict = {"sentinel2_l2a": RasterImage(
            image=image, timestamps=[(t, t) for t in timestamps])}
        wrapper.normalizer(input_dict, {})
        context = ModelContext(inputs=[input_dict], metadatas=[])
        sample, present, _ = wrapper._prepare_modality_inputs(context)
        assert present == ["sentinel2_l2a"]
        sample.sentinel2_l2a_mask[..., 2] = MaskValue.MISSING.value
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = wrapper.model(sample, fast_pass=False, patch_size=PATCH)
            tm = o["tokens_and_masks"]
            m = (tm.sentinel2_l2a_mask != MaskValue.MISSING.value).unsqueeze(-1)
            pooled = (tm.sentinel2_l2a * m).sum(dim=(3, 4)) / m.sum(dim=(3, 4)).clamp(min=1)
            return pooled[0].permute(2, 0, 1).contiguous()

    t0 = time.perf_counter(); written = 0
    for i, sid in enumerate(ordered, 1):
        target = emb_dir / f"{sid}.npy"
        if target.exists():
            continue
        rec = records[sid]
        idx = efc.select_timestep_indices(rec)
        with xr.open_dataset(args.data_root / rec["file"], decode_times=True,
                             cache=False) as ds:
            bands = [np.asarray(ds[b].values[idx], dtype="float32") if b in ds
                     else np.zeros((len(idx), FULL, FULL), dtype="float32")
                     for b in MODEL_BANDS]
            cube = np.stack(bands, axis=0)
            times = [datetime.fromisoformat(str(np.datetime_as_string(t, unit="s")))
                     for t in np.asarray(ds["time"].values)[idx]]
        feat = embed_full(cube, times)
        np.save(target, feat.float().cpu().numpy().astype("float16"), allow_pickle=False)
        written += 1
        if i % 250 == 0 or i == len(ordered):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(ordered)}] {el/60:.1f}분 · "
                  f"남은 {(len(ordered)-i)/max(i/el,1e-9)/60:.1f}분", flush=True)

    total = time.perf_counter() - t0
    summary = {
        "schema": "sen12-cache-full128-v1", "fold": args.fold,
        "difference_from_baseline": "128x128 단일 패스 인코딩 (기존은 64 crop 4장 독립 인코딩)",
        "unchanged": ["patch_size 4", "token grid 32x32", "fp16", "MODEL_BANDS 12",
                      "band-set 2 MISSING", "동일 12 timestep"],
        "samples_written": written, "samples_total": len(ordered),
        "total_s": round(total, 2),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cache_bytes": sum(p.stat().st_size for p in emb_dir.glob("*.npy")),
    }
    (out / "cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)); print("DONE")


if __name__ == "__main__":
    main()
