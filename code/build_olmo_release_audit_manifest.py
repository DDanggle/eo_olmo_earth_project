#!/usr/bin/env python3
"""Build immutable input and smoke-selection manifests for a paired release audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


BAND_DIRECTORY = "B01_B02_B03_B04_B05_B06_B07_B08_B8A_B09_B11_B12"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def input_layer_name(period_index: int) -> str:
    return "sentinel2_l2a" if period_index == 0 else f"sentinel2_l2a.{period_index}"


def inventory_file(path: Path, hash_policy: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
    }
    if hash_policy == "sha256":
        result["sha256"] = file_sha256(path)
    return result


def select_smoke(
    records: list[dict[str, Any]],
    years: list[int],
    clear_per_year: int,
    contaminated_per_year: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for year in years:
        available = sorted(
            (record for record in records if record["year"] == year),
            key=lambda value: (value["bad_proxy_mean"], value["sample_id"]),
        )
        required = clear_per_year + contaminated_per_year
        if len(available) < required:
            raise ValueError(f"year {year} has {len(available)} records, expected at least {required}")
        clear = available[:clear_per_year]
        clear_ids = {record["sample_id"] for record in clear}
        contaminated = [record for record in reversed(available) if record["sample_id"] not in clear_ids][
            :contaminated_per_year
        ]
        selected.extend({**record, "smoke_stratum": "clear_proxy"} for record in clear)
        selected.extend(
            {**record, "smoke_stratum": "contaminated_proxy"} for record in contaminated
        )
    return sorted(selected, key=lambda value: value["sample_id"])


def build_manifest(
    config: dict[str, Any],
    dataset_root: Path,
    time_axis_rows: list[dict[str, str]],
    quality_rows: list[dict[str, str]],
    hash_policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped_time: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in time_axis_rows:
        if row["dataset"] != "v1":
            continue
        grouped_time[(int(row["year_label"]), row["key"])].append(row)
    quality = {
        (int(row["year"]), row["key"]): float(row["bad_proxy_mean"])
        for row in quality_rows
        if row["dataset"] == "v1" and row["scope"] == config["smoke_selection"]["quality_scope"]
    }
    expected_site_years = len(config["year_prefixes"]) * 54
    if len(grouped_time) != expected_site_years:
        raise ValueError(f"expected {expected_site_years} v1 site-years, found {len(grouped_time)}")

    records: list[dict[str, Any]] = []
    for (year, key), rows in sorted(grouped_time.items()):
        rows = sorted(rows, key=lambda value: int(value["period_index"]))
        if [int(value["period_index"]) for value in rows] != list(range(config["period_count"])):
            raise ValueError(f"incomplete periods for {year}/{key}")
        prefix = config["year_prefixes"][str(year)]
        window_name = f"{prefix}_{key}"
        window_dir = dataset_root / "windows/default" / window_name
        input_layers = []
        for period_index in range(config["period_count"]):
            layer_name = input_layer_name(period_index)
            layer_dir = window_dir / "layers" / layer_name / BAND_DIRECTORY
            input_layers.append(
                {
                    "period_index": period_index,
                    "layer_name": layer_name,
                    "geotiff": inventory_file(layer_dir / "geotiff.tif", hash_policy),
                    "metadata": inventory_file(layer_dir / "metadata.json", hash_policy),
                    "ordered_item_name_sha256": rows[period_index]["ordered_item_name_sha256"],
                    "group_start": rows[period_index]["group_start"],
                    "group_end": rows[period_index]["group_end"],
                }
            )
        identity_payload = {
            "window_name": window_name,
            "input_layers": input_layers,
            "items_json": inventory_file(window_dir / "items.json", hash_policy),
            "window_metadata": inventory_file(window_dir / "metadata.json", hash_policy),
        }
        records.append(
            {
                "schema": "olmoearth-release-input-v1",
                "sample_id": f"legacy_mosaic_12_period__{year}__{key}",
                "recipe_id": "legacy_mosaic_12_period",
                "year": year,
                "spatial_key": key,
                "spatial_cluster_id": f"jeju-window-{key}",
                "window_name": window_name,
                "window_dir": window_dir.as_posix(),
                "bad_proxy_mean": quality[(year, key)],
                "hash_policy": hash_policy,
                "input_bundle_identity": hashlib.sha256(canonical_bytes(identity_payload)).hexdigest(),
                **identity_payload,
            }
        )
    smoke_config = config["smoke_selection"]
    smoke = select_smoke(
        records,
        smoke_config["years"],
        smoke_config["clear_per_year"],
        smoke_config["contaminated_per_year"],
    )
    return records, smoke


def write_outputs(
    output_dir: Path,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    smoke: list[dict[str, Any]],
    hash_policy: str,
    source_files: list[Path],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "site_year_inputs.jsonl").open("w", encoding="utf-8") as output:
        for record in records:
            output.write(canonical_bytes(record).decode("utf-8"))
    smoke_payload = {
        "schema": "olmoearth-release-smoke-selection-v1",
        "selection_uses_human_labels": False,
        "selection_uses_public_evidence_labels": False,
        "records": smoke,
    }
    (output_dir / "smoke_inputs.json").write_bytes(canonical_bytes(smoke_payload))
    with (output_dir / "legacy_release_matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        fields = [
            "sample_id",
            "release_id",
            "timestamp_track",
            "input_bundle_identity",
            "status",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for model in config["models"]:
                writer.writerow(
                    {
                        "sample_id": record["sample_id"],
                        "release_id": model["release_id"],
                        "timestamp_track": "legacy_timestamps",
                        "input_bundle_identity": record["input_bundle_identity"],
                        "status": "not_run",
                    }
                )
    summary = {
        "schema": "olmoearth-release-input-summary-v1",
        "audit_id": config["audit_id"],
        "hash_policy": hash_policy,
        "exact_tensor_file_pairing_ready": hash_policy == "sha256",
        "site_years": len(records),
        "spatial_windows": len({record["spatial_key"] for record in records}),
        "years": sorted({record["year"] for record in records}),
        "adjacent_year_events": len({record["spatial_key"] for record in records}) * 3,
        "legacy_primary_release_outputs_planned": len(records) * len(config["models"]),
        "full_two_recipe_primary_outputs_planned": len(records)
        * len(config["models"])
        * len(config["input_recipes"]),
        "smoke_site_years": len(smoke),
        "smoke_release_outputs_planned": len(smoke) * len(config["models"]),
        "source_files": [
            {"path": path.as_posix(), "sha256": file_sha256(path)} for path in source_files
        ],
        "limitations": [
            "This manifest covers the legacy materialization recipe only.",
            "The SCL BestClear recipe remains blocked until 216 site-years are materialized.",
            "No accuracy or negative-transfer metric is available without independent labels.",
        ],
    }
    (output_dir / "run_summary.json").write_bytes(canonical_bytes(summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--time-axis-csv", type=Path, required=True)
    parser.add_argument("--quality-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hash-policy", choices=("metadata", "sha256"), default="metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    dataset_root = args.dataset_root or Path(config["source_dataset_root"])
    records, smoke = build_manifest(
        config,
        dataset_root,
        read_csv(args.time_axis_csv),
        read_csv(args.quality_csv),
        args.hash_policy,
    )
    summary = write_outputs(
        args.output_dir,
        config,
        records,
        smoke,
        args.hash_policy,
        [args.config, args.time_axis_csv, args.quality_csv],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
