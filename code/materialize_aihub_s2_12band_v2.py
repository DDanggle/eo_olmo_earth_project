#!/usr/bin/env python3
"""AI-Hub 71363용 Sentinel-2 12-band cube v2 materializer.

v1은 첫 STAC item 하나를 boundless read해 target 밖을 0으로 채웠다. v2는 같은 날짜·플랫폼의
모든 후보를 정확한 target grid로 warp한 뒤 source validity mask로 결정론적 mosaic를 만든다.
12밴드 공통 coverage가 사전 기준보다 낮으면 파일을 만들지 않고 fail-closed한다.

계약: docs/AIHUB_CUBE_V2_CONTRACT.md. v1 경로를 절대 덮어쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


BANDS = [
    "B02", "B03", "B04", "B08",
    "B05", "B06", "B07", "B8A", "B11", "B12",
    "B01", "B09",
]
CLOUD_MAX = 60.0
MIN_COMMON_COVERAGE = 0.999
RESAMPLING = "nearest"
TARGET_CRS = "EPSG:32652"
TILE_PX = 1024
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", type=Path,
        default=Path("/home/work/data/olmoearth/aihub/inventory/inventory.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("/home/work/data/olmoearth/aihub/s2_12band_v2"),
    )
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--keys-file", type=Path, help="pilot용 key 한 줄 하나")
    return parser.parse_args()


def normalize_platform(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.upper().replace("SENTINEL", "S").replace("-", "").replace("_", "")
    if normalized in {"S2A", "S2B"}:
        return normalized
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_jsonl(stream, row: dict) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def main() -> None:
    import numpy as np
    import planetary_computer as planetary_computer
    import rasterio
    from pystac_client import Client
    from rasterio.enums import Resampling
    from rasterio.transform import from_bounds
    from rasterio.vrt import WarpedVRT

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    array_dir = args.out / "arrays"
    validity_dir = args.out / "validity"
    array_dir.mkdir(exist_ok=True)
    validity_dir.mkdir(exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    excluded_path = args.out / "excluded.jsonl"

    done = set()
    for path in (manifest_path, excluded_path):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line:
                    done.add(json.loads(line)["key"])

    requested_keys = None
    if args.keys_file:
        requested_keys = {
            line.strip() for line in args.keys_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    records = [
        json.loads(line) for line in args.inventory.read_text(encoding="utf-8").splitlines()
        if line
    ]
    records.sort(key=lambda record: record["key"])
    records = records[args.start:]
    if requested_keys is not None:
        records = [record for record in records if record["key"] in requested_keys]
    todo = [record for record in records if record["key"] not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"대상 {len(todo)} / 선택 {len(records)} (이미 처리 {len(done)})", flush=True)
    client = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    resampling = getattr(Resampling, RESAMPLING)
    started = time.perf_counter()
    materialized = excluded = failed = 0

    with manifest_path.open("a", encoding="utf-8") as manifest_stream, \
            excluded_path.open("a", encoding="utf-8") as excluded_stream:
        for index, record in enumerate(todo, 1):
            key, raw_date = record["key"], record["date"]
            iso_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
            expected_platform = normalize_platform(record.get("platform"))
            try:
                search = client.search(
                    collections=["sentinel-2-l2a"],
                    bbox=record["wgs84_bbox"],
                    datetime=f"{iso_date}T00:00:00Z/{iso_date}T23:59:59Z",
                    limit=100,
                )
                all_items = sorted(search.item_collection(), key=lambda item: item.id)
                platform_items = [
                    item for item in all_items
                    if normalize_platform(item.properties.get("platform")) == expected_platform
                ]
                cloud_items = [
                    item for item in platform_items
                    if item.properties.get("eo:cloud_cover") is not None
                    and float(item.properties["eo:cloud_cover"]) <= CLOUD_MAX
                ]
                if not all_items:
                    append_jsonl(excluded_stream, {
                        "key": key, "date": iso_date, "reason": "no_stac_item",
                    })
                    excluded += 1
                    continue
                if expected_platform is None or not platform_items:
                    append_jsonl(excluded_stream, {
                        "key": key, "date": iso_date, "reason": "no_platform_match",
                        "platform_meta": record.get("platform"),
                        "candidate_platforms": sorted({
                            str(item.properties.get("platform")) for item in all_items
                        }),
                        "candidate_ids": [item.id for item in all_items],
                    })
                    excluded += 1
                    continue
                if not cloud_items:
                    append_jsonl(excluded_stream, {
                        "key": key, "date": iso_date, "reason": "no_candidate_under_cloud_max",
                        "cloud_max": CLOUD_MAX,
                        "candidate_cloud_cover": {
                            item.id: item.properties.get("eo:cloud_cover") for item in platform_items
                        },
                    })
                    excluded += 1
                    continue

                west, south, east, north = record["utm52n_bbox"]
                target_transform = from_bounds(west, south, east, north, TILE_PX, TILE_PX)
                cube = np.zeros((len(BANDS), TILE_PX, TILE_PX), dtype=np.uint16)
                band_valid = np.zeros((len(BANDS), TILE_PX, TILE_PX), dtype=bool)
                item_contribution: dict[str, list[str]] = {band: [] for band in BANDS}
                missing_assets: dict[str, list[str]] = {}

                for item in cloud_items:
                    for band_index, band in enumerate(BANDS):
                        asset = item.assets.get(band)
                        if asset is None:
                            missing_assets.setdefault(item.id, []).append(band)
                            continue
                        with rasterio.open(asset.href) as source:
                            vrt_options = {
                                "crs": TARGET_CRS,
                                "transform": target_transform,
                                "width": TILE_PX,
                                "height": TILE_PX,
                                "resampling": resampling,
                                # 별도 alpha를 만들어 실제 값 0과 target 밖 invalid를 구분한다.
                                "add_alpha": True,
                            }
                            if source.nodata is not None:
                                vrt_options["src_nodata"] = source.nodata
                            with WarpedVRT(source, **vrt_options) as vrt:
                                values = vrt.read(1)
                                source_valid = vrt.read(vrt.count) > 0
                        fill = source_valid & ~band_valid[band_index]
                        if bool(fill.any()):
                            cube[band_index, fill] = np.clip(values[fill], 0, 65535).astype(np.uint16)
                            band_valid[band_index, fill] = True
                            item_contribution[band].append(item.id)

                band_coverage = band_valid.mean(axis=(1, 2))
                common_valid = band_valid.all(axis=0)
                common_coverage = float(common_valid.mean())
                coverage_record = {
                    "key": key,
                    "date": iso_date,
                    "platform": expected_platform,
                    "candidate_ids": [item.id for item in cloud_items],
                    "candidate_cloud_cover": {
                        item.id: item.properties.get("eo:cloud_cover") for item in cloud_items
                    },
                    "item_contribution_by_band": item_contribution,
                    "missing_assets_by_item": missing_assets,
                    "band_coverage": {
                        band: round(float(coverage), 8)
                        for band, coverage in zip(BANDS, band_coverage)
                    },
                    "common_12band_coverage": round(common_coverage, 8),
                    "min_common_coverage": MIN_COMMON_COVERAGE,
                }
                if common_coverage < MIN_COMMON_COVERAGE:
                    append_jsonl(excluded_stream, {
                        **coverage_record, "reason": "insufficient_common_coverage",
                    })
                    excluded += 1
                    continue

                target = array_dir / f"{key}.npy"
                validity_target = validity_dir / f"{key}.npy"
                target_tmp = array_dir / f"{key}.tmp.npy"
                validity_tmp = validity_dir / f"{key}.tmp.npy"
                np.save(target_tmp, cube, allow_pickle=False)
                np.save(validity_tmp, common_valid.astype(np.uint8), allow_pickle=False)
                target_tmp.replace(target)
                validity_tmp.replace(validity_target)
                append_jsonl(manifest_stream, {
                    **coverage_record,
                    "status": "coverage_valid",
                    "target_crs": TARGET_CRS,
                    "target_transform": list(target_transform)[:6],
                    "target_shape": [len(BANDS), TILE_PX, TILE_PX],
                    "bands": BANDS,
                    "resampling": RESAMPLING,
                    "dtype": "uint16",
                    "array_bytes": target.stat().st_size,
                    "array_file_sha256": sha256_file(target),
                    "validity_file_sha256": sha256_file(validity_target),
                })
                materialized += 1
            except Exception as error:  # noqa: BLE001
                append_jsonl(excluded_stream, {
                    "key": key, "date": iso_date, "reason": "error",
                    "error": repr(error)[:500],
                })
                failed += 1

            if index % 25 == 0 or index == len(todo):
                elapsed = time.perf_counter() - started
                rate = index / max(elapsed, 1e-9)
                print(
                    f"[{index}/{len(todo)}] materialized={materialized} excluded={excluded} "
                    f"failed={failed} {rate:.2f}/s",
                    flush=True,
                )

    summary = {
        "schema": "aihub-s2-12band-materialize-v2",
        "contract": "docs/AIHUB_CUBE_V2_CONTRACT.md",
        "materialized": materialized,
        "excluded": excluded,
        "failed": failed,
        "selection": {
            "same_date": True,
            "platform_match_required": True,
            "cloud_max": CLOUD_MAX,
            "candidate_order": "item_id ascending; first valid pixel wins",
        },
        "grid": {
            "crs": TARGET_CRS,
            "shape": [TILE_PX, TILE_PX],
            "resampling": RESAMPLING,
            "min_common_12band_coverage": MIN_COMMON_COVERAGE,
        },
        "warning": "materialized is not experiment_eligible until full health and selection-bias audits pass",
    }
    summary_path = args.out / "materialize_summary.json"
    summary_tmp = args.out / "materialize_summary.tmp.json"
    summary_tmp.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_tmp.replace(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
