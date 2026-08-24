#!/usr/bin/env python3
"""Determine whether Jeju v5 is semantically different from v1.

The quality audit found zero aggregate difference. This script locates the
equivalence boundary with three independent checks:

1. all 2,592 ordered source-item groups,
2. the runtime SpaceMode handler and normalized query configuration,
3. deterministic full-raster and embedding-window pixel samples.

It never claims full pixel equality from a sample alone; the source manifest and
runtime handler checks provide the causal explanation for the sample result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rslearn.config import SpaceMode
from rslearn.data_sources.utils import space_mode_handlers


YEAR_PREFIX = {
    "2023": {"v1": "jeju23_", "v5": "jeju23_"},
    "2024": {"v1": "jeju_", "v5": "jeju24_"},
    "2025": {"v1": "jeju25_", "v5": "jeju25_"},
    "2026": {"v1": "jeju26r_", "v5": "jeju26r_"},
}
YEARS = list(YEAR_PREFIX)
RNG_SEED = 20260822
RASTER_SAMPLE_COUNT = 24
EMBEDDING_SAMPLE_COUNT = 24


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
        default="/home/work/data/olmoearth/embed_jeju_v2/audit_v1_vs_v5/v5_equivalence.json",
    )
    return parser.parse_args()


def layer_index(path: Path) -> int:
    match = re.fullmatch(r"sentinel2_l2a(?:\.(\d+))?", path.name)
    return int(match.group(1) or 0) if match else 10_000


def s2_paths(window_dir: Path) -> list[Path]:
    result = []
    for layer in sorted(
        (path for path in (window_dir / "layers").glob("sentinel2_l2a*") if path.is_dir()),
        key=layer_index,
    ):
        matches = sorted(layer.glob("*/geotiff.tif"))
        if matches:
            result.append(matches[0])
    return result


def embedding_path(window_dir: Path) -> Path:
    matches = sorted((window_dir / "layers" / "embeddings").glob("*/geotiff.tif"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one embedding in {window_dir}, got {len(matches)}")
    return matches[0]


def discover(root: Path, dataset: str) -> dict[tuple[str, str], Path]:
    output = {}
    for year in YEARS:
        prefix = YEAR_PREFIX[year][dataset]
        for window_dir in sorted(path for path in root.glob(f"{prefix}*") if path.is_dir()):
            output[(year, window_dir.name.removeprefix(prefix))] = window_dir
    return output


def item_groups(window_dir: Path) -> list[list[str]]:
    payload = json.loads((window_dir / "items.json").read_text(encoding="utf-8"))
    layer = next(item for item in payload if item["layer_name"] == "sentinel2_l2a")
    return [
        [serialized["name"] for serialized in group]
        for group in layer["serialized_item_groups"]
    ]


def group_hash(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def query_config(root: Path) -> dict:
    dataset_root = root.parents[1]
    config = json.loads((dataset_root / "config.json").read_text(encoding="utf-8"))
    return config["layers"]["sentinel2_l2a"]["data_source"]["query_config"]


def normalized_query(query: dict) -> dict:
    result = copy.deepcopy(query)
    if result.get("space_mode") == "PER_PERIOD_MOSAIC":
        result["space_mode"] = "MOSAIC"
    return result


def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if np.issubdtype(left.dtype, np.floating):
        return float(np.nanmax(np.abs(left - right)))
    return float(np.max(np.abs(left.astype(np.int64) - right.astype(np.int64))))


def compare_full_rasters(
    candidates: list[tuple[str, str, int, Path, Path]], rng: np.random.Generator
) -> list[dict]:
    indexes = rng.choice(len(candidates), size=RASTER_SAMPLE_COUNT, replace=False)
    records = []
    for sample_index, candidate_index in enumerate(indexes, start=1):
        year, key, period, left_path, right_path = candidates[int(candidate_index)]
        with rasterio.open(left_path) as left_src, rasterio.open(right_path) as right_src:
            left = left_src.read()
            right = right_src.read()
        equal = bool(np.array_equal(left, right, equal_nan=True))
        records.append(
            {
                "year": year,
                "key": key,
                "period_index": period,
                "shape_equal": left.shape == right.shape,
                "array_equal": equal,
                "max_abs_difference": max_abs_difference(left, right),
            }
        )
        print(f"raster sample {sample_index}/{RASTER_SAMPLE_COUNT}: equal={equal}", flush=True)
    return records


def compare_embedding_windows(
    candidates: list[tuple[str, str, Path, Path]], rng: np.random.Generator
) -> list[dict]:
    indexes = rng.choice(len(candidates), size=EMBEDDING_SAMPLE_COUNT, replace=False)
    records = []
    for sample_index, candidate_index in enumerate(indexes, start=1):
        year, key, left_path, right_path = candidates[int(candidate_index)]
        with rasterio.open(left_path) as left_src, rasterio.open(right_path) as right_src:
            height, width = left_src.height, left_src.width
            row = int(rng.integers(0, height - 32 + 1))
            col = int(rng.integers(0, width - 32 + 1))
            window = Window(col, row, 32, 32)
            left = left_src.read(window=window)
            right = right_src.read(window=window)
        equal = bool(np.array_equal(left, right, equal_nan=True))
        records.append(
            {
                "year": year,
                "key": key,
                "row": row,
                "col": col,
                "shape_equal": left.shape == right.shape,
                "array_equal": equal,
                "max_abs_difference": max_abs_difference(left, right),
            }
        )
        print(
            f"embedding sample {sample_index}/{EMBEDDING_SAMPLE_COUNT}: equal={equal}",
            flush=True,
        )
    return records


def main() -> None:
    args = parse_args()
    roots = {"v1": Path(args.v1_root), "v5": Path(args.v5_root)}
    windows = {dataset: discover(root, dataset) for dataset, root in roots.items()}
    common = sorted(set(windows["v1"]) & set(windows["v5"]))
    if len(common) != 216:
        raise RuntimeError(f"expected 216 paired windows, got {len(common)}")

    group_total = 0
    group_hash_matches = 0
    group_count_matches = 0
    raster_candidates = []
    embedding_candidates = []
    for year_key in common:
        left_window = windows["v1"][year_key]
        right_window = windows["v5"][year_key]
        left_groups = item_groups(left_window)
        right_groups = item_groups(right_window)
        if len(left_groups) != 12 or len(right_groups) != 12:
            raise RuntimeError(f"expected 12 periods in {year_key}")
        left_paths = s2_paths(left_window)
        right_paths = s2_paths(right_window)
        if len(left_paths) != 12 or len(right_paths) != 12:
            raise RuntimeError(f"expected 12 materialized rasters in {year_key}")
        for period, (left_group, right_group, left_path, right_path) in enumerate(
            zip(left_groups, right_groups, left_paths, right_paths)
        ):
            group_total += 1
            group_count_matches += int(len(left_group) == len(right_group))
            group_hash_matches += int(group_hash(left_group) == group_hash(right_group))
            raster_candidates.append(
                (year_key[0], year_key[1], period, left_path, right_path)
            )
        embedding_candidates.append(
            (
                year_key[0],
                year_key[1],
                embedding_path(left_window),
                embedding_path(right_window),
            )
        )

    queries = {dataset: query_config(root) for dataset, root in roots.items()}
    same_runtime_handler = bool(
        space_mode_handlers[SpaceMode.MOSAIC]
        is space_mode_handlers[SpaceMode.PER_PERIOD_MOSAIC]
    )
    normalized_queries_equal = normalized_query(queries["v1"]) == normalized_query(
        queries["v5"]
    )

    rng = np.random.default_rng(RNG_SEED)
    raster_samples = compare_full_rasters(raster_candidates, rng)
    embedding_samples = compare_embedding_windows(embedding_candidates, rng)
    all_raster_samples_equal = all(sample["array_equal"] for sample in raster_samples)
    all_embedding_samples_equal = all(
        sample["array_equal"] for sample in embedding_samples
    )
    source_groups_all_equal = group_hash_matches == group_total

    equivalent = all(
        (
            same_runtime_handler,
            normalized_queries_equal,
            source_groups_all_equal,
            all_raster_samples_equal,
            all_embedding_samples_equal,
        )
    )
    rslearn_distribution = distribution("rslearn")
    direct_url = json.loads(rslearn_distribution.read_text("direct_url.json") or "{}")
    output = {
        "status": "equivalent_duplicate" if equivalent else "difference_found",
        "provenance": {
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rng_seed": RNG_SEED,
            "full_raster_samples": RASTER_SAMPLE_COUNT,
            "embedding_32x32_samples": EMBEDDING_SAMPLE_COUNT,
            "rslearn_version": rslearn_distribution.version,
            "rslearn_direct_url": direct_url,
        },
        "config": {
            "v1_query": queries["v1"],
            "v5_query": queries["v5"],
            "normalized_queries_equal": normalized_queries_equal,
            "mosaic_and_per_period_use_same_runtime_handler": same_runtime_handler,
            "runtime_handler_name": space_mode_handlers[SpaceMode.MOSAIC].__name__,
        },
        "source_groups": {
            "period_groups_total": group_total,
            "item_count_matches": group_count_matches,
            "ordered_item_hash_matches": group_hash_matches,
            "all_equal": source_groups_all_equal,
        },
        "full_raster_samples": raster_samples,
        "embedding_samples": embedding_samples,
        "gates": {
            "all_source_groups_equal": source_groups_all_equal,
            "all_full_raster_samples_equal": all_raster_samples_equal,
            "all_embedding_samples_equal": all_embedding_samples_equal,
            "v5_is_semantic_duplicate_under_runtime": equivalent,
        },
        "interpretation": (
            "With period_duration set in both configs, the installed rslearn maps MOSAIC "
            "and deprecated PER_PERIOD_MOSAIC to the same mosaic handler. The configs "
            "therefore produce the same ordered groups and sampled arrays. This does not "
            "test a new cloud-compositing recipe."
        ),
        "limitations": [
            "Full-raster and embedding equality are deterministic samples, not all-file hashes.",
            "All B02 cloud/zero metrics were already compared exhaustively in the quality audit.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "config": output["config"],
                "source_groups": output["source_groups"],
                "gates": output["gates"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"EQUIVALENCE_DIAGNOSIS_DONE out={out}", flush=True)


if __name__ == "__main__":
    main()
