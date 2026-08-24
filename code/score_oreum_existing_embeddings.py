#!/usr/bin/env python3
"""Screen geolocated Jeju oreum against existing 4/12-period embeddings.

This is deliberately a *screen*, not causal attribution. It reuses the existing
v6 embedding products and compares point-level step scores from the 4-period and
12-period model inputs. A candidate is called ``high_stable`` only when both
variants are in the top decile and agree on the most likely year split.

Known limitation: the underlying v6 inputs remain season/time-axis confounded and
precede the island-wide SCL compositor validation. Every output therefore has
evidence grade M (model screen), never A/B official evidence.
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __name__ == "__main__" and os.environ.get("ALLOW_HISTORICAL_INVALID_JEJU_CANDIDATES") != "1":
    raise SystemExit(
        "REFUSED: the existing four/twelve-period embeddings fail the Jeju time "
        "contract and cannot create new oreum change candidates. Set "
        "ALLOW_HISTORICAL_INVALID_JEJU_CANDIDATES=1 only to reproduce the "
        "preserved historical screen."
    )

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window


YEARS = ["2023", "2024", "2025", "2026"]
PREFIX = {
    "2023": "jeju23_",
    "2024": "jeju24_",
    "2025": "jeju25_",
    "2026": "jeju26r_",
}
LAYERS = ["embeddings", "embeddings_t12"]
SPLITS = {0: "2023->2024", 1: "2024->2025", 2: "2025->2026"}


def files_for(root: Path, layer: str) -> dict[str, dict[str, Path]]:
    output: dict[str, dict[str, Path]] = {}
    for year, prefix in PREFIX.items():
        paths = {}
        pattern = str(root / f"{prefix}*" / "layers" / layer / "*" / "geotiff.tif")
        for value in glob.glob(pattern):
            path = Path(value)
            window_name = path.parts[path.parts.index("default") + 1]
            spatial_key = window_name.removeprefix(prefix)
            paths[spatial_key] = path
        output[year] = paths
    keys = set.intersection(*(set(output[year]) for year in YEARS))
    if len(keys) != 54:
        raise ValueError(f"{layer}: expected 54 shared spatial windows, got {len(keys)}")
    return output


def build_spatial_index(
    paths: dict[str, dict[str, Path]], records: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, tuple[float, float]], str]:
    reference = paths["2024"]
    first_path = next(iter(reference.values()))
    with rasterio.open(first_path) as src:
        crs = str(src.crs)
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    projected = {}
    for record in records:
        location = record["location"]
        if location.get("lat") is None:
            continue
        projected[record["official_record_no"]] = transformer.transform(
            float(location["lon"]), float(location["lat"])
        )
    assignment = {}
    for spatial_key, path in reference.items():
        with rasterio.open(path) as src:
            bounds = src.bounds
        for record_no, (x, y) in projected.items():
            if bounds.left <= x < bounds.right and bounds.bottom < y <= bounds.top:
                assignment[record_no] = spatial_key
    return assignment, projected, crs


def sample_year_vectors(
    year_paths: dict[str, Path],
    assignment: dict[str, str],
    projected: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Read only 3x3 point chips, grouped so each GeoTIFF opens once."""
    grouped: dict[str, list[str]] = {}
    for record_no, spatial_key in assignment.items():
        grouped.setdefault(spatial_key, []).append(record_no)
    raw: dict[str, np.ndarray] = {}
    for idx, (spatial_key, record_nos) in enumerate(sorted(grouped.items()), 1):
        path = year_paths[spatial_key]
        with rasterio.open(path) as src:
            for record_no in record_nos:
                x, y = projected[record_no]
                row, col = src.index(x, y)
                row = min(max(row, 1), src.height - 2)
                col = min(max(col, 1), src.width - 2)
                array = src.read(window=Window(col - 1, row - 1, 3, 3)).astype(np.float32)
                raw[record_no] = np.mean(array, axis=(1, 2))
        if idx % 18 == 0:
            print(f"point windows: {idx}/{len(grouped)}", flush=True)
    if not raw:
        return {}
    center = np.mean(np.stack(list(raw.values())), axis=0)
    output = {}
    for record_no, vector in raw.items():
        vector = vector - center
        norm = np.linalg.norm(vector)
        output[record_no] = vector / norm if norm else vector
    return output


def step_score(vectors: list[np.ndarray]) -> tuple[float, str, list[float]]:
    values = []
    for split in (0, 1, 2):
        before = list(range(0, split + 1))
        after = list(range(split + 1, 4))
        cross = [
            1 - float(np.dot(vectors[i], vectors[j]))
            for i in before
            for j in after
        ]
        within = [
            1 - float(np.dot(vectors[group[i]], vectors[group[j]]))
            for group in (before, after)
            for i in range(len(group))
            for j in range(i + 1, len(group))
        ]
        values.append(float(np.mean(cross) - (np.mean(within) if within else 0.0)))
    best = int(np.argmax(values))
    return values[best], SPLITS[best], values


def percentile(value: float, sorted_values: list[float]) -> float:
    return round(100 * bisect.bisect_right(sorted_values, value) / len(sorted_values), 1)


def classify(p4: float, p12: float, split4: str, split12: str) -> str:
    same = split4 == split12
    if p4 >= 90 and p12 >= 90 and same:
        return "high_stable"
    if max(p4, p12) >= 90:
        return "high_unstable"
    if p4 >= 75 and p12 >= 75 and same:
        return "moderate_stable"
    return "low_or_unstable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    records = registry["records"]
    layer_results: dict[str, dict[str, dict[str, Any]]] = {}
    spatial_assignment = None
    projected = None
    crs = None
    for layer in LAYERS:
        print(f"=== {layer} ===", flush=True)
        paths = files_for(args.dataset_root, layer)
        if spatial_assignment is None or projected is None:
            spatial_assignment, projected, crs = build_spatial_index(paths, records)
        vectors_by_year = {}
        for year in YEARS:
            print(f"sample {layer} {year}", flush=True)
            vectors_by_year[year] = sample_year_vectors(
                paths[year], spatial_assignment, projected
            )
        shared_years = set.intersection(
            *(set(vectors_by_year[year]) for year in YEARS)
        )
        results = {}
        for idx, record_no in enumerate(sorted(shared_years), 1):
            spatial_key = spatial_assignment[record_no]
            vectors = [vectors_by_year[year][record_no] for year in YEARS]
            score, split, all_splits = step_score(vectors)
            results[record_no] = {
                "score": score,
                "split": split,
                "split_scores": [round(value, 6) for value in all_splits],
                "spatial_window": spatial_key,
            }
            if idx % 50 == 0:
                print(f"scored points: {idx}/{len(shared_years)}", flush=True)
        layer_results[layer] = results
        print(f"screened {layer}: {len(results)}", flush=True)

    shared = sorted(set(layer_results["embeddings"]) & set(layer_results["embeddings_t12"]))
    values4 = sorted(layer_results["embeddings"][key]["score"] for key in shared)
    values12 = sorted(layer_results["embeddings_t12"][key]["score"] for key in shared)
    output_records = []
    for record_no in shared:
        result4 = layer_results["embeddings"][record_no]
        result12 = layer_results["embeddings_t12"][record_no]
        p4 = percentile(result4["score"], values4)
        p12 = percentile(result12["score"], values12)
        screen_class = classify(p4, p12, result4["split"], result12["split"])
        output_records.append(
            {
                "official_record_no": record_no,
                "status": "screen_complete_existing_v6",
                "grade": "M",
                "screen_class": screen_class,
                "percentile_4": p4,
                "percentile_12": p12,
                "score_4": round(result4["score"], 6),
                "score_12": round(result12["score"], 6),
                "split_4": result4["split"],
                "split_12": result12["split"],
                "spatial_window": result4["spatial_window"],
                "warning": (
                    "model screening only; existing v6 is season/time-axis confounded and "
                    "predates island-wide SCL validation"
                ),
            }
        )
    classes = {
        key: sum(1 for row in output_records if row["screen_class"] == key)
        for key in ("high_stable", "high_unstable", "moderate_stable", "low_or_unstable")
    }
    payload = {
        "schema": "kearth-oreum-existing-embedding-screen-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(args.registry),
        "dataset_root": str(args.dataset_root),
        "projection": crs,
        "method": {
            "spatial_sample": "3x3 embedding pixels around current offline OSM peak",
            "centering": "per-year mean of all sampled oreum raw vectors",
            "stability_rule": "both 4/12-period percentiles >=90 and same step split",
        },
        "summary": {
            "official_inventory": len(records),
            "osm_geolocated": sum(1 for row in records if row["location"].get("lat") is not None),
            "within_existing_grid_and_screened": len(output_records),
            "outside_grid_or_unresolved": len(records) - len(output_records),
            "classes": classes,
        },
        "limitations": [
            "OSM point is not an official oreum boundary.",
            "Existing v6 four-period inputs are not season aligned.",
            "Existing v6 twelve-period inputs include rolling-2026 phase shift.",
            "The full-island v7 SCL compositor has not passed its multi-window gate.",
            "Percentiles rank only the geolocated oreum subset and are not prevalence estimates.",
        ],
        "records": output_records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
