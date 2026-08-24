#!/usr/bin/env python3
"""Audit paired OlmoEarth release embeddings without making performance claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_RELEASES = ("olmoearth_v1_base", "olmoearth_v1_2_base")
NEIGHBOR_K_VALUES = (1, 2)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _center(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("metrics require a 2D matrix with at least two observations")
    return value - value.mean(axis=0, keepdims=True)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear centered-kernel alignment for the same ordered observations."""
    left_centered, right_centered = _center(left), _center(right)
    if left_centered.shape[0] != right_centered.shape[0]:
        raise ValueError("CKA inputs must contain the same paired observations")

    observations = left_centered.shape[0]
    if observations <= min(left_centered.shape[1], right_centered.shape[1]):
        left_gram = left_centered @ left_centered.T
        right_gram = right_centered @ right_centered.T
        numerator = float(np.sum(left_gram * right_gram))
        denominator = float(
            np.sqrt(np.sum(left_gram**2) * np.sum(right_gram**2))
        )
    else:
        cross = left_centered.T @ right_centered
        left_self = left_centered.T @ left_centered
        right_self = right_centered.T @ right_centered
        numerator = float(np.linalg.norm(cross, ord="fro") ** 2)
        denominator = float(
            np.linalg.norm(left_self, ord="fro")
            * np.linalg.norm(right_self, ord="fro")
        )
    if denominator <= 0:
        raise ValueError("CKA is undefined for a constant representation")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def row_l2_normalized_cka(left: np.ndarray, right: np.ndarray) -> float:
    values = []
    for matrix in (left, right):
        matrix = np.asarray(matrix, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise ValueError("row-normalized CKA requires non-zero token vectors")
        values.append(matrix / norms)
    return linear_cka(values[0], values[1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy rankdata(method='average')."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.asarray(left).shape != np.asarray(right).shape:
        raise ValueError("Spearman inputs must have the same shape")
    left_rank, right_rank = _rankdata(np.ravel(left)), _rankdata(np.ravel(right))
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    if denominator <= 0:
        raise ValueError("Spearman correlation is undefined for constant ranks")
    return float(np.clip(np.dot(left_rank, right_rank) / denominator, -1.0, 1.0))


def _euclidean_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("distance inputs must be a finite 2D matrix")
    differences = value[:, None, :] - value[None, :, :]
    return np.sqrt(np.sum(differences**2, axis=2))


def _geometry_metrics(
    left: np.ndarray, right: np.ndarray, neighbor_ks: tuple[int, ...]
) -> dict[str, Any]:
    left_value, right_value = np.asarray(left, dtype=np.float64), np.asarray(
        right, dtype=np.float64
    )
    if left_value.ndim != 2 or right_value.ndim != 2:
        raise ValueError("representations must be two-dimensional")
    if left_value.shape[0] != right_value.shape[0] or left_value.shape[0] < 3:
        raise ValueError("representations need at least three paired identities")
    if any(not 1 <= value < left_value.shape[0] for value in neighbor_ks):
        raise ValueError("neighbor k must be between one and n-1")

    left_distance = _euclidean_distance_matrix(left_value)
    right_distance = _euclidean_distance_matrix(right_value)
    triangle = np.triu_indices(left_value.shape[0], k=1)
    distance_ties = {
        "left": int(left_distance[triangle].size - np.unique(left_distance[triangle]).size),
        "right": int(right_distance[triangle].size - np.unique(right_distance[triangle]).size),
    }
    np.fill_diagonal(left_distance, np.inf)
    np.fill_diagonal(right_distance, np.inf)
    left_order = np.argsort(left_distance, axis=1, kind="mergesort")
    right_order = np.argsort(right_distance, axis=1, kind="mergesort")

    overlaps: dict[str, Any] = {}
    for neighbor_k in neighbor_ks:
        per_identity = []
        for left_row, right_row in zip(left_order, right_order, strict=True):
            intersection = set(left_row[:neighbor_k]) & set(right_row[:neighbor_k])
            per_identity.append(len(intersection) / neighbor_k)
        observed = float(np.mean(per_identity))
        chance = neighbor_k / (left_value.shape[0] - 1)
        corrected = (observed - chance) / (1.0 - chance)
        overlaps[str(neighbor_k)] = {
            "mean_fraction": observed,
            "per_identity_fraction": [float(value) for value in per_identity],
            "random_expectation": float(chance),
            "chance_corrected": float(corrected),
        }

    return {
        "site_years": int(left_value.shape[0]),
        "embedding_dimensions": {
            "left": int(left_value.shape[1]),
            "right": int(right_value.shape[1]),
        },
        "pairwise_euclidean_distance_spearman": spearman_correlation(
            left_distance[triangle], right_distance[triangle]
        ),
        "pooled_linear_cka": linear_cka(left_value, right_value),
        "neighbor_overlap": overlaps,
        "distance_tie_count": distance_ties,
        "tie_rule": "stable sample_id order via mergesort",
    }


def representation_metrics(
    left: np.ndarray,
    right: np.ndarray,
    spatial_cluster_ids: list[str] | None = None,
    neighbor_ks: tuple[int, ...] = NEIGHBOR_K_VALUES,
) -> dict[str, Any]:
    """Within-release geometry preservation over paired site-year identities."""
    result = _geometry_metrics(left, right, neighbor_ks)
    if spatial_cluster_ids is None:
        return result
    if len(spatial_cluster_ids) != np.asarray(left).shape[0]:
        raise ValueError("one spatial cluster ID is required per site-year")

    leave_one_out = []
    for cluster_id in sorted(set(spatial_cluster_ids)):
        keep = np.array([value != cluster_id for value in spatial_cluster_ids])
        if int(keep.sum()) < max(neighbor_ks) + 1:
            raise ValueError("too few site-years after leaving out a spatial cluster")
        metrics = _geometry_metrics(np.asarray(left)[keep], np.asarray(right)[keep], neighbor_ks)
        leave_one_out.append(
            {
                "left_out_spatial_cluster_id": cluster_id,
                "remaining_site_years": int(keep.sum()),
                "pairwise_euclidean_distance_spearman": metrics[
                    "pairwise_euclidean_distance_spearman"
                ],
                "neighbor_overlap": {
                    key: value["mean_fraction"]
                    for key, value in metrics["neighbor_overlap"].items()
                },
            }
        )
    result["spatial_clusters"] = len(set(spatial_cluster_ids))
    result["leave_one_spatial_cluster_out"] = leave_one_out
    result["leave_one_spatial_cluster_out_range"] = {
        "pairwise_euclidean_distance_spearman": [
            min(value["pairwise_euclidean_distance_spearman"] for value in leave_one_out),
            max(value["pairwise_euclidean_distance_spearman"] for value in leave_one_out),
        ],
        "neighbor_overlap": {
            str(neighbor_k): [
                min(value["neighbor_overlap"][str(neighbor_k)] for value in leave_one_out),
                max(value["neighbor_overlap"][str(neighbor_k)] for value in leave_one_out),
            ]
            for neighbor_k in neighbor_ks
        },
    }
    return result


def paired_valid_tokens(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Require identical spatial validity masks; never compact each release separately."""
    left_value, right_value = np.asarray(left), np.asarray(right)
    if left_value.ndim != 2 or right_value.ndim != 2:
        raise ValueError("paired spatial embeddings must be two-dimensional")
    if left_value.shape[0] != right_value.shape[0]:
        raise ValueError("paired spatial embeddings must have the same token grid")
    left_valid = np.isfinite(left_value).all(axis=1)
    right_valid = np.isfinite(right_value).all(axis=1)
    if not np.array_equal(left_valid, right_valid):
        raise ValueError("release outputs have different valid/nodata masks")
    if int(left_valid.sum()) < 2:
        raise ValueError("at least two valid paired spatial tokens are required")
    return left_value[left_valid], right_value[right_valid]


def _intersection_valid_tokens(
    left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    if int(valid.sum()) < 2:
        raise ValueError("shift control has fewer than two jointly valid tokens")
    return left[valid], right[valid]


def _sample_paired_tokens(
    left: np.ndarray, right: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    if left.shape[0] != right.shape[0]:
        raise ValueError("paired representations must have the same observation count")
    count = min(maximum, left.shape[0])
    indices = np.linspace(0, left.shape[0] - 1, num=count, dtype=np.int64)
    return left[indices], right[indices]


def _fixed_toroidal_shifts(height: int, width: int) -> list[tuple[int, int]]:
    candidates = [
        (height // 4, 0),
        (0, width // 4),
        (height // 4, width // 4),
        (height // 2, 0),
        (0, width // 2),
        (height // 2, width // 2),
    ]
    result = []
    for value in candidates:
        normalized = (value[0] % height, value[1] % width)
        if normalized != (0, 0) and normalized not in result:
            result.append(normalized)
    if not result:
        raise ValueError("output grid is too small for a toroidal-shift control")
    return result


def spatial_cka_metrics(
    left_grid: np.ndarray, right_grid: np.ndarray, maximum_tokens: int
) -> dict[str, Any]:
    if left_grid.ndim != 3 or right_grid.ndim != 3:
        raise ValueError("spatial CKA expects height×width×features arrays")
    if left_grid.shape[:2] != right_grid.shape[:2]:
        raise ValueError("release output grids have different height/width")
    height, width = left_grid.shape[:2]
    left, right = paired_valid_tokens(
        left_grid.reshape(-1, left_grid.shape[-1]),
        right_grid.reshape(-1, right_grid.shape[-1]),
    )
    left_sample, right_sample = _sample_paired_tokens(left, right, maximum_tokens)
    observed = linear_cka(left_sample, right_sample)
    normalized = row_l2_normalized_cka(left_sample, right_sample)

    shift_values = []
    for row_shift, column_shift in _fixed_toroidal_shifts(height, width):
        shifted = np.roll(right_grid, shift=(row_shift, column_shift), axis=(0, 1))
        null_left, null_right = _intersection_valid_tokens(
            left_grid.reshape(-1, left_grid.shape[-1]),
            shifted.reshape(-1, shifted.shape[-1]),
        )
        null_left, null_right = _sample_paired_tokens(
            null_left, null_right, maximum_tokens
        )
        shift_values.append(
            {
                "row_shift": row_shift,
                "column_shift": column_shift,
                "linear_cka": linear_cka(null_left, null_right),
            }
        )
    null_median = float(np.median([value["linear_cka"] for value in shift_values]))
    return {
        "valid_tokens": int(left.shape[0]),
        "sampled_tokens": int(left_sample.shape[0]),
        "linear_cka": observed,
        "row_l2_normalized_linear_cka": normalized,
        "toroidal_shift_null": shift_values,
        "toroidal_shift_null_median": null_median,
        "excess_over_shift_null_median": observed - null_median,
    }


def _read_embedding(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - server integration dependency
        raise RuntimeError("rasterio is required to read embedding GeoTIFF outputs") from exc

    with rasterio.open(path) as dataset:
        array = dataset.read(masked=True).astype(np.float64)
        metadata = {
            "shape": [dataset.height, dataset.width],
            "bands": dataset.count,
            "crs": dataset.crs.to_string() if dataset.crs else None,
            "transform": list(dataset.transform)[:6],
            "bounds": list(dataset.bounds),
            "pixel_size": [abs(dataset.transform.a), abs(dataset.transform.e)],
            "nodata": dataset.nodata,
        }
    return np.moveaxis(np.ma.filled(array, np.nan), 0, -1), metadata


def _read_expected_grid(record: dict[str, Any]) -> dict[str, Any]:
    inventory = record["window_metadata"]
    path = Path(inventory["path"])
    if path.stat().st_size != inventory["bytes"] or file_sha256(path) != inventory["sha256"]:
        raise ValueError(f"window metadata drift detected: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    projection, bounds = metadata["projection"], metadata["bounds"]
    x_values = [bounds[0] * projection["x_resolution"], bounds[2] * projection["x_resolution"]]
    y_values = [bounds[1] * projection["y_resolution"], bounds[3] * projection["y_resolution"]]
    return {
        "crs": projection["crs"],
        "bounds": [min(x_values), min(y_values), max(x_values), max(y_values)],
    }


def _validated_exact_inputs(
    exact_inputs: Path, run_payload: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if file_sha256(exact_inputs) != run_payload["input_pairing"]["exact_inputs_sha256"]:
        raise ValueError("exact input manifest SHA does not match the release run")
    payload = json.loads(exact_inputs.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if len(records) != 8 or not payload.get("exact_tensor_file_pairing_ready"):
        raise ValueError("analysis requires eight content-hashed exact input records")
    by_sample = {record["sample_id"]: record for record in records}
    if len(by_sample) != 8:
        raise ValueError("exact input sample IDs are not unique")
    return by_sample


def _validated_runs(
    run_summary: Path, complete_marker: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(run_summary.read_text(encoding="utf-8"))
    complete = json.loads(complete_marker.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError("release smoke run is not complete")
    if complete.get("run_summary_sha256") != file_sha256(run_summary):
        raise ValueError("release smoke COMPLETE marker does not match run_summary.json")
    if not payload.get("input_pairing", {}).get("same_manifest_for_both_releases"):
        raise ValueError("releases were not run from one exact input manifest")
    by_release = {run["release_id"]: run for run in payload["runs"]}
    if set(by_release) != set(EXPECTED_RELEASES):
        raise ValueError("run summary does not contain the exact v1/v1.2 pair")
    for run in by_release.values():
        if len(run.get("outputs", [])) != 8:
            raise ValueError("each release must contain exactly eight output records")
        started_epoch = datetime.fromisoformat(run["started_at"]).timestamp()
        sample_ids = set()
        for output in run["outputs"]:
            if output["sample_id"] in sample_ids:
                raise ValueError("duplicate output sample ID within a release")
            sample_ids.add(output["sample_id"])
            path = Path(output["path"])
            if path.stat().st_size != output["bytes"] or file_sha256(path) != output["sha256"]:
                raise ValueError(f"output drift detected: {path}")
            if output["mtime_ns"] / 1_000_000_000 + 1.0 < started_epoch:
                raise ValueError(f"stale output predates its release run: {path}")
    return payload, by_release


def analyze(
    run_summary: Path,
    complete_marker: Path,
    exact_inputs: Path,
    maximum_tokens_per_window: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload, by_release = _validated_runs(run_summary, complete_marker)
    inputs = _validated_exact_inputs(exact_inputs, payload)
    outputs = {
        release: {item["sample_id"]: item for item in run["outputs"]}
        for release, run in by_release.items()
    }
    if any(set(value) != set(inputs) for value in outputs.values()):
        raise ValueError("release outputs do not form the exact eight sample pairs")

    pooled: dict[str, list[np.ndarray]] = {release: [] for release in EXPECTED_RELEASES}
    cluster_ids, rows = [], []
    for sample_id in sorted(inputs):
        input_record = inputs[sample_id]
        pair = [outputs[release][sample_id] for release in EXPECTED_RELEASES]
        for output in pair:
            if output["window"] != input_record["window_name"]:
                raise ValueError(f"window identity mismatch for {sample_id}")
            if output["input_bundle_identity"] != input_record["input_bundle_identity"]:
                raise ValueError(f"input bundle identity mismatch for {sample_id}")
            if output["spatial_cluster_id"] != input_record["spatial_cluster_id"]:
                raise ValueError(f"spatial cluster identity mismatch for {sample_id}")

        left_grid, left_meta = _read_embedding(Path(pair[0]["path"]))
        right_grid, right_meta = _read_embedding(Path(pair[1]["path"]))
        for key in ("shape", "crs", "transform", "bounds", "nodata"):
            if left_meta[key] != right_meta[key]:
                raise ValueError(f"spatial alignment mismatch for {sample_id}: {key}")
        expected = _read_expected_grid(input_record)
        if left_meta["crs"] != expected["crs"] or not np.allclose(
            left_meta["bounds"], expected["bounds"], rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"release outputs do not match the expected window grid: {sample_id}")

        spatial = spatial_cka_metrics(
            left_grid, right_grid, maximum_tokens_per_window
        )
        left_tokens, right_tokens = paired_valid_tokens(
            left_grid.reshape(-1, left_grid.shape[-1]),
            right_grid.reshape(-1, right_grid.shape[-1]),
        )
        pooled[EXPECTED_RELEASES[0]].append(left_tokens.mean(axis=0))
        pooled[EXPECTED_RELEASES[1]].append(right_tokens.mean(axis=0))
        cluster_ids.append(input_record["spatial_cluster_id"])
        rows.append(
            {
                "sample_id": sample_id,
                "window": input_record["window_name"],
                "spatial_cluster_id": input_record["spatial_cluster_id"],
                "smoke_stratum": input_record["smoke_stratum"],
                "height": left_meta["shape"][0],
                "width": left_meta["shape"][1],
                "v1_embedding_bands": left_meta["bands"],
                "v1_2_embedding_bands": right_meta["bands"],
                "valid_tokens": spatial["valid_tokens"],
                "sampled_tokens": spatial["sampled_tokens"],
                "linear_cka": spatial["linear_cka"],
                "row_l2_normalized_linear_cka": spatial[
                    "row_l2_normalized_linear_cka"
                ],
                "shift_null_median": spatial["toroidal_shift_null_median"],
                "excess_cka_over_shift_null": spatial[
                    "excess_over_shift_null_median"
                ],
            }
        )

    pooled_metrics = representation_metrics(
        np.stack(pooled[EXPECTED_RELEASES[0]]),
        np.stack(pooled[EXPECTED_RELEASES[1]]),
        spatial_cluster_ids=cluster_ids,
    )
    summary = {
        "schema": "olmoearth-release-smoke-analysis-v1",
        "status": "complete",
        "run_summary_sha256": file_sha256(run_summary),
        "exact_inputs_sha256": file_sha256(exact_inputs),
        "releases": list(EXPECTED_RELEASES),
        "sample_contract": {
            "n_records": len(inputs),
            "n_spatial_clusters": len(set(cluster_ids)),
            "selected_extreme_smoke_set": True,
            "labels": 0,
            "population_inference_allowed": False,
            "input_recipes": 1,
        },
        "maximum_tokens_per_window": maximum_tokens_per_window,
        "per_window_spatial_cka": {
            key: {
                "mean": float(np.mean([row[key] for row in rows])),
                "minimum": float(np.min([row[key] for row in rows])),
                "maximum": float(np.max([row[key] for row in rows])),
            }
            for key in (
                "linear_cka",
                "row_l2_normalized_linear_cka",
                "shift_null_median",
                "excess_cka_over_shift_null",
            )
        },
        "pooled_site_year_geometry": pooled_metrics,
        "claims_allowed": [
            "descriptive_representation_continuity_on_the_eight_prespecified_exact_inputs"
        ],
        "claims_forbidden": [
            "accuracy_improvement",
            "negative_transfer_reduction",
            "cloud_robustness",
            "korea_or_jeju_generalization",
            "backward_compatible_cache",
            "input_effect_vs_release_effect",
            "confidence_interval_or_population_significance",
        ],
        "limitations": [
            "Eight selected label-free site-years span only seven spatial clusters.",
            "Pixels and the 28 pairwise distances are not independent replicates.",
            "No raw cross-version cosine or learned alignment is reported.",
            "CKA, neighbor overlap, and rank preservation do not measure task utility.",
            "Only the legacy input recipe exists, so input-by-release contrasts are unavailable.",
        ],
    }
    return summary, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--complete-marker", type=Path, required=True)
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-tokens-per-window", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.maximum_tokens_per_window < 2:
        raise SystemExit("--maximum-tokens-per-window must be at least two")
    summary, rows = analyze(
        args.run_summary,
        args.complete_marker,
        args.exact_inputs,
        args.maximum_tokens_per_window,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "analysis_summary.json"
    summary_path.write_bytes(canonical_bytes(summary))
    csv_path = args.output_dir / "per_window_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = {
        "schema": "olmoearth-release-smoke-analysis-completion-v1",
        "analysis_summary_sha256": file_sha256(summary_path),
        "per_window_metrics_sha256": file_sha256(csv_path),
    }
    (args.output_dir / "ANALYSIS_COMPLETE.json").write_bytes(canonical_bytes(complete))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
