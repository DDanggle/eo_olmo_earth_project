#!/usr/bin/env python3
"""C1 — frozen Presto per-pixel cache 추출기. GPU1 전용.

config/presto_c1_contract.json (schema presto-c1-input-contract-v2)의 구현임.

  emb_fp16/<sid>.npy   frozen Presto per-pixel embedding  128 x 128 x 128  fp16

계약 요점 (extract_sen12_fold_cache.py 와 동일한 S12q 골격):
  - 같은 contract jsonl · loco_folds.json · 봉인 split 해시 대조 · top-12 SCL index
  - 정규화: DN / 10000, shift 0 (봉인 upstream 소스와 동일)
  - 월: 시점별 **실제 달력 월** [B, 12] 2-D 텐서. 스칼라 시작월 금지 (M79 v1 오류 재발 방지)
  - 좌표: NetCDF CRS + x/y → WGS84 픽셀별 lat/lon. center_lat/lon attr 사용 금지 (투영값 발견 이력)
  - S1/ERA5/SRTM/DynamicWorld는 mask=1 (결측), NDVI는 B04/B08 파생 (S2-only 위반 아님)
  - 코드·가중치 sha256 을 계약과 대조하고 불일치 시 실행 거부

smoke 모드 (--smoke N):
  N타일만 처리하며 픽셀 배치 1024/4096/8192 각각의 처리율·peak 메모리 측정,
  같은 타일 2회 인코딩 비트 동일성, finite, 월/좌표 실측값 기록, 출력 content sha256 기록.
  smoke 산출물은 본 캐시 디렉터리가 아니라 smoke/ 하위에 격리한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_here = Path(__file__).resolve().parent
CONTRACT_JSON = next((c for c in (
    _here / "presto_c1_contract.json",
    _here / "config" / "presto_c1_contract.json",
    _here.parent / "config" / "presto_c1_contract.json") if c.exists()), None)
if CONTRACT_JSON is None:
    raise SystemExit("presto_c1_contract.json 을 찾지 못함")
REAL_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"]
# single_file_presto 17채널 인덱스 (probe P-2 실측 + 계약 presto_mapping)
S2_IDX = {"B02": 2, "B03": 3, "B04": 4, "B05": 5, "B06": 6, "B07": 7,
          "B08": 8, "B8A": 9, "B11": 10, "B12": 11}
NDVI_IDX = 16
N_CH = 17
H = W = 128
EMB_DIM = 128


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl"))
    p.add_argument("--folds", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/loco_folds.json"))
    p.add_argument("--fold", type=str, default="holdout_chimanimani")
    p.add_argument("--data-root", type=Path,
                   default=Path("/home/work/data/sen12landslides/extracted"))
    p.add_argument("--presto-src", type=Path,
                   default=Path("/home/work/data/olmoearth/models/presto/src"))
    p.add_argument("--out", type=Path, default=Path("/home/work/data/olmoearth/presto_c1"))
    p.add_argument("--pixel-batch", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0, help="0이면 전체")
    p.add_argument("--smoke", type=int, default=0,
                   help=">0 이면 smoke: 해당 타일 수만 처리, 배치 스윕·결정성·계약 실측 기록")
    return p.parse_args()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("refusing to run: CUDA_VISIBLE_DEVICES must be exactly 1")

    import numpy as np
    import torch
    import xarray as xr
    from pyproj import Transformer

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    args = parse_args()
    contract = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))

    # ── 코드·가중치 봉인 대조 (불일치 = 즉시 거부) ──
    single = next(args.presto_src.rglob("single_file_presto.py"), None)
    if single is None:
        raise SystemExit(f"single_file_presto.py 없음: {args.presto_src}")
    got_code = sha256_file(single)
    want_code = contract["upstream"]["single_file_presto_sha256"]
    if got_code != want_code:
        raise SystemExit(f"single_file_presto.py sha 불일치\n  기대 {want_code}\n  실제 {got_code}")
    wt = next((p for p in args.presto_src.rglob("default_model.pt")
               if p.stat().st_size > 1_000_000), None)
    if wt is None:
        raise SystemExit("default_model.pt 실물 없음")
    got_wt = sha256_file(wt)
    want_wt = contract["upstream"]["default_model_pt_sha256"]
    if got_wt != want_wt:
        raise SystemExit(f"default_model.pt sha 불일치\n  기대 {want_wt}\n  실제 {got_wt}")
    print(f"sha 대조 2/2 일치: code={got_code[:12]} weights={got_wt[:12]}", flush=True)

    sys.path.insert(0, str(single.parent))
    import single_file_presto as pm

    # ── fold·split 봉인 대조 (extract_sen12_fold_cache.py 와 동일 로직) ──
    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fold = next((f for f in folds["folds"] if f["fold"] == args.fold), None)
    if fold is None:
        raise SystemExit(f"fold 없음: {args.fold}")
    records = {}
    for line in args.contract.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            records[r["sample_id"]] = r
    split_ids: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        regions = (fold["train_regions"] if split == "train"
                   else [fold["val_region"]] if split == "val"
                   else [fold["test_region"]])
        ids = sorted(sid for sid, r in records.items()
                     if r["region"] in regions and not r.get("error")
                     and r.get("s15_eligible", True))
        got = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        if got != fold["sample_sha256"][split]:
            raise SystemExit(f"{split} split 해시 불일치")
        split_ids[split] = ids
    print("split 해시 3/3 일치: "
          + " ".join(f"{k}={len(v)}" for k, v in split_ids.items()), flush=True)

    ordered = [sid for s in ("train", "val", "test") for sid in split_ids[s]]
    if args.smoke:
        ordered = ordered[:args.smoke]
    elif args.limit:
        ordered = ordered[:args.limit]

    out = args.out / args.fold / ("smoke" if args.smoke else "")
    emb_dir = out / "emb_fp16"
    emb_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=1 → physical GPU1
    load0 = time.perf_counter()
    model = pm.Presto.construct()
    sd = torch.load(wt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model = model.to(device).eval()
    model_load_s = time.perf_counter() - load0
    # 월 임베딩 크기 확인 — 0-indexed 월이 범위 안인지 (계약 month_encoding 검증)
    month_table = None
    for name, par in model.encoder.named_parameters():
        if "month" in name and par.dim() == 2:
            month_table = (name, tuple(par.shape))
    print(f"model params={sum(p.numel() for p in model.parameters()):,} "
          f"month_table={month_table}", flush=True)

    def load_tile(sid: str):
        rec = records[sid]
        quality = rec["scl_clear_fraction"]
        idx = sorted(sorted(range(len(quality)),
                            key=lambda i: (-float(quality[i]), i))[:12])
        with xr.open_dataset(args.data_root / rec["file"], decode_times=True,
                             cache=False) as ds:
            cube = np.stack([np.asarray(ds[b].values[idx], dtype="float32")
                             for b in REAL_BANDS], axis=0)          # 10,12,H,W
            times = [datetime.fromisoformat(str(np.datetime_as_string(t, unit="s")))
                     for t in np.asarray(ds["time"].values)[idx]]
            # WGS84 픽셀 좌표 — CRS+x/y에서 유도 (attr 직접 사용 금지)
            crs = None
            for key in ("spatial_ref", "crs"):
                if key in ds:
                    crs = ds[key].attrs.get("crs_wkt") or ds[key].attrs.get("spatial_ref")
                    if crs:
                        break
            if crs is None:
                crs = ds.attrs.get("crs") or ds.attrs.get("spatial_ref")
            if crs is None:
                raise ValueError(f"CRS 없음: {sid}")
            xs = np.asarray(ds["x"].values, dtype="float64")
            ys = np.asarray(ds["y"].values, dtype="float64")
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        gx, gy = np.meshgrid(xs, ys)                                 # H,W
        lon, lat = tr.transform(gx, gy)
        return cube, times, np.stack([lat, lon], axis=-1)            # H,W,2

    def encode_tile(cube, times, latlon, pixel_batch: int):
        # 정규화: /10000, shift 0 — 계약 문구 그대로
        px = (cube / 10000.0).reshape(len(REAL_BANDS), 12, -1).transpose(2, 1, 0)  # N,12,10
        n = px.shape[0]
        x = torch.zeros(n, 12, N_CH)
        mask = torch.ones(n, 12, N_CH)
        for i, b in enumerate(REAL_BANDS):
            x[:, :, S2_IDX[b]] = torch.from_numpy(px[:, :, i])
            mask[:, :, S2_IDX[b]] = 0
        b4, b8 = px[:, :, 2], px[:, :, 3]
        x[:, :, NDVI_IDX] = torch.from_numpy(
            ((b8 - b4) / np.clip(b8 + b4, 1e-6, None)).astype("float32"))
        mask[:, :, NDVI_IDX] = 0
        months = torch.tensor([[t.month - 1 for t in times]] * n, dtype=torch.long)  # N,12
        if month_table is not None:
            assert int(months.max()) < month_table[1][0] or int(months.max()) < month_table[1][1], \
                f"월 인덱스 {int(months.max())} 가 임베딩 테이블 {month_table} 범위 밖"
        dw = torch.full((n, 12), 9, dtype=torch.long)
        ll = torch.from_numpy(latlon.reshape(-1, 2).astype("float32"))
        embs = torch.empty(n, EMB_DIM, dtype=torch.float32)
        with torch.no_grad():
            for s in range(0, n, pixel_batch):
                e = s + pixel_batch
                emb = model.encoder(x[s:e].to(device),
                                    dynamic_world=dw[s:e].to(device),
                                    latlons=ll[s:e].to(device),
                                    mask=mask[s:e].to(device),
                                    month=months[s:e].to(device))
                embs[s:e] = emb.float().cpu()
        torch.cuda.synchronize()
        return embs.reshape(H, W, EMB_DIM).permute(2, 0, 1).contiguous()  # 128,H,W

    report: dict = {"schema": "presto-c1-cache-v1", "fold": args.fold,
                    "mode": "smoke" if args.smoke else "full",
                    "pixel_batch": args.pixel_batch,
                    "code_sha256": got_code, "weights_sha256": got_wt,
                    "model_load_s": round(model_load_s, 3),
                    "split_counts": {k: len(v) for k, v in split_ids.items()},
                    "normalization": "DN/10000 shift 0",
                    "month_encoding": "per-timestep calendar month - 1, shape [N,12]",
                    "month_table": month_table}

    if args.smoke:
        sid0 = ordered[0]
        cube, times, latlon = load_tile(sid0)
        report["contract_probe"] = {
            "sample_id": sid0,
            "months_0idx": [t.month - 1 for t in times],
            "dates": [t.strftime("%Y-%m-%d") for t in times],
            "latlon_corner_00": [round(float(latlon[0, 0, 0]), 6),
                                 round(float(latlon[0, 0, 1]), 6)],
            "latlon_corner_hw": [round(float(latlon[-1, -1, 0]), 6),
                                 round(float(latlon[-1, -1, 1]), 6)],
            "latlon_plausible": bool(abs(latlon[..., 0]).max() <= 90
                                     and abs(latlon[..., 1]).max() <= 180),
            "raw_dn_p50": float(np.median(cube)),
        }
        # 결정성: 같은 타일 2회
        e1 = encode_tile(cube, times, latlon, args.pixel_batch)
        e2 = encode_tile(cube, times, latlon, args.pixel_batch)
        report["deterministic"] = bool(torch.equal(e1, e2))
        report["finite"] = bool(torch.isfinite(e1).all())
        report["emb_std"] = round(float(e1.std()), 6)
        # 배치 스윕
        sweep = []
        for pb in (1024, 4096, 8192):
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.perf_counter()
            encode_tile(cube, times, latlon, pb)
            dt = time.perf_counter() - t0
            sweep.append({"pixel_batch": pb, "tile_s": round(dt, 3),
                          "peak_cuda_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1)})
        report["batch_sweep"] = sweep

    torch.cuda.reset_peak_memory_stats(device)
    t_start = time.perf_counter()
    n_done = 0
    hasher = hashlib.sha256()
    for i, sid in enumerate(ordered, 1):
        p = emb_dir / f"{sid}.npy"
        if p.exists() and not args.smoke:
            n_done += 1
            continue
        cube, times, latlon = load_tile(sid)
        feat = encode_tile(cube, times, latlon, args.pixel_batch)
        arr = feat.numpy().astype("float16")
        np.save(p, arr, allow_pickle=False)
        hasher.update(arr.tobytes())
        n_done += 1
        if i % 100 == 0 or i == len(ordered):
            el = time.perf_counter() - t_start
            print(f"  [{i}/{len(ordered)}] {el/60:.1f}분 · {i/max(el,1e-9):.3f} tile/s "
                  f"· 남은 {(len(ordered)-i)/max(i/el,1e-9)/60:.1f}분", flush=True)

    total_s = time.perf_counter() - t_start
    best = min(report.get("batch_sweep", [{"tile_s": None}]),
               key=lambda d: d["tile_s"] or 1e9) if args.smoke else None
    report.update({
        "tiles_written": n_done, "tiles_total": len(ordered),
        "total_s": round(total_s, 2),
        "tiles_per_s": round(n_done / total_s, 4) if total_s > 0 else None,
        "peak_cuda_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "content_sha256": hasher.hexdigest(),
        "bytes_per_tile": EMB_DIM * H * W * 2,
        "projected_full_6834_gib": round(EMB_DIM * H * W * 2 * 6834 / 2**30, 2),
    })
    if args.smoke and best and best["tile_s"]:
        report["projected_full_6834_hours"] = round(6834 * best["tile_s"] / 3600, 2)
    (out / ("smoke_report.json" if args.smoke else "cache_summary.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
