#!/usr/bin/env python3
"""Build a timestamp and source-item manifest for Jeju v1/v5 inputs.

The rslearn window names say "year", but the actual model semantics depend on
the ordered item groups in each ``items.json``. This script records only source
item names, timestamps, counts, and hashes (never signed asset URLs), then tests
whether the first four model inputs are seasonally aligned across years.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


YEAR_PREFIX = {
    "2023": {"v1": "jeju23_", "v5": "jeju23_"},
    "2024": {"v1": "jeju_", "v5": "jeju24_"},
    "2025": {"v1": "jeju25_", "v5": "jeju25_"},
    "2026": {"v1": "jeju26r_", "v5": "jeju26r_"},
}
YEARS = list(YEAR_PREFIX)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v1-root",
        default="/home/work/data/olmoearth/embed_search/dataset/windows/default",
    )
    parser.add_argument(
        "--v5-root",
        default="/home/work/data/olmoearth/embed_jeju_v2/dataset/windows/default",
    )
    parser.add_argument(
        "--out",
        default="/home/work/data/olmoearth/embed_jeju_v2/audit_v1_vs_v5",
    )
    return parser.parse_args()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def source_groups(window_dir: Path) -> list[list[dict]]:
    payload = json.loads((window_dir / "items.json").read_text(encoding="utf-8"))
    layer = next(item for item in payload if item["layer_name"] == "sentinel2_l2a")
    return layer["serialized_item_groups"]


def group_record(
    dataset: str,
    year: str,
    key: str,
    window_range: list[str],
    index: int,
    group: list[dict],
) -> dict:
    names = [item["name"] for item in group]
    timestamps = [
        parse_time(item["geometry"]["time_range"][0])
        for item in group
    ]
    ordered_hash = hashlib.sha256("\n".join(names).encode()).hexdigest()
    return {
        "dataset": dataset,
        "year_label": year,
        "key": key,
        "window_start": window_range[0],
        "window_end": window_range[1],
        "period_index": index,
        "group_start": min(timestamps).isoformat(),
        "group_end": max(timestamps).isoformat(),
        "representative_month": int(np.median([time.month for time in timestamps])),
        "item_count": len(group),
        "ordered_item_name_sha256": ordered_hash,
    }


def collect(dataset: str, root: Path) -> list[dict]:
    rows: list[dict] = []
    for year in YEARS:
        prefix = YEAR_PREFIX[year][dataset]
        windows = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
        for window_dir in windows:
            key = window_dir.name.removeprefix(prefix)
            metadata = json.loads(
                (window_dir / "metadata.json").read_text(encoding="utf-8")
            )
            groups = source_groups(window_dir)
            for index, group in enumerate(groups):
                if not group:
                    continue
                rows.append(
                    group_record(
                        dataset,
                        year,
                        key,
                        metadata["time_range"],
                        index,
                        group,
                    )
                )
        print(f"{dataset} {year}: {len(windows)} windows", flush=True)
    return rows


def mode_sequence(rows: list[dict], dataset: str, year: str) -> list[int]:
    selected = [
        row for row in rows if row["dataset"] == dataset and row["year_label"] == year
    ]
    sequence = []
    for index in range(12):
        months = [row["representative_month"] for row in selected if row["period_index"] == index]
        sequence.append(Counter(months).most_common(1)[0][0])
    return sequence


def summarize(rows: list[dict]) -> dict:
    output: dict = {}
    for dataset in ("v1", "v5"):
        output[dataset] = {}
        for year in YEARS:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["year_label"] == year
            ]
            keys = sorted({row["key"] for row in selected})
            counts = Counter(row["period_index"] for row in selected)
            sequence = mode_sequence(rows, dataset, year)
            per_window_counts = [
                sum(row["item_count"] for row in selected if row["key"] == key)
                for key in keys
            ]
            output[dataset][year] = {
                "windows": len(keys),
                "period_indexes": sorted(counts),
                "windows_per_period": sorted(set(counts.values())),
                "mode_month_sequence": sequence,
                "model_first4_mode_months": sequence[:4],
                "all12_month_set": sorted(set(sequence)),
                "direction": (
                    "reverse_chronological"
                    if all((sequence[i] - sequence[i + 1]) % 12 == 1 for i in range(11))
                    else "other"
                ),
                "source_items_per_window_min": min(per_window_counts),
                "source_items_per_window_median": float(np.median(per_window_counts)),
                "source_items_per_window_max": max(per_window_counts),
            }

    v5_first4 = [output["v5"][year]["model_first4_mode_months"] for year in YEARS]
    v5_all12_sets = [output["v5"][year]["all12_month_set"] for year in YEARS]
    output["gates"] = {
        "all_years_54_windows_12_periods": all(
            output[dataset][year]["windows"] == 54
            and output[dataset][year]["period_indexes"] == list(range(12))
            and output[dataset][year]["windows_per_period"] == [54]
            for dataset in ("v1", "v5")
            for year in YEARS
        ),
        "model_first4_season_aligned_across_years": all(
            sequence == v5_first4[0] for sequence in v5_first4[1:]
        ),
        "all12_cover_same_calendar_month_set": all(
            month_set == v5_all12_sets[0] for month_set in v5_all12_sets[1:]
        ),
    }
    output["interpretation"] = {
        "model_config": "embeddings uses ordered period indexes 0..3",
        "t12_config": "embeddings_t12 uses ordered period indexes 0..11",
        "warning": (
            "A year label is not a time-axis guarantee. Compare the month sequences and "
            "ordered item hashes before interpreting embedding differences as world change."
        ),
    }
    return output


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect("v1", Path(args.v1_root)) + collect("v5", Path(args.v5_root))
    fields = list(rows[0])
    with (out / "time_axis_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "time_axis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["gates"], indent=2), flush=True)
    for dataset in ("v1", "v5"):
        for year in YEARS:
            print(
                dataset,
                year,
                summary[dataset][year]["mode_month_sequence"],
                flush=True,
            )
    print(f"TIME_AXIS_AUDIT_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
