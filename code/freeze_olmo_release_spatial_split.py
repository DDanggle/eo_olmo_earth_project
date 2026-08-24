#!/usr/bin/env python3
"""Freeze a leakage-safe spatial split before inspecting full release outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CALIBRATION_X = {22528, 23552, 24576, 25600, 26624}
EMBARGO_X = {27648}
EAST_X = {28672, 29696, 30720}
EXPECTED_YEARS = {2023, 2024, 2025, 2026}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def atomic_create(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace different evidence: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def spatial_x(spatial_key: str) -> int:
    try:
        x_text, _ = spatial_key.split("_", maxsplit=1)
        return int(x_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid spatial key: {spatial_key!r}") from exc


def build_split(
    exact_inputs_path: Path,
    disclosed_inputs_path: Path,
) -> dict[str, Any]:
    exact_payload = read_json(exact_inputs_path)
    disclosed_payload = read_json(disclosed_inputs_path)
    records = exact_payload.get("records", [])
    if not exact_payload.get("exact_tensor_file_pairing_ready") or len(records) != 216:
        raise ValueError("full split requires 216 content-hashed exact records")
    if any(record.get("hash_policy") != "sha256" for record in records):
        raise ValueError("full split received a non-SHA input record")

    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_cluster[record["spatial_key"]].append(record)
    if len(by_cluster) != 54:
        raise ValueError(f"expected 54 spatial clusters, found {len(by_cluster)}")
    for key, values in by_cluster.items():
        years = [int(value["year"]) for value in values]
        if len(values) != 4 or set(years) != EXPECTED_YEARS or len(set(years)) != 4:
            raise ValueError(f"cluster is not a complete one-per-year panel: {key}/{years}")

    disclosed_clusters = {
        record["spatial_key"]
        for record in disclosed_payload.get("records", [])
        if spatial_x(record["spatial_key"]) in EAST_X
    }
    expected_disclosed = {"28672_-372736", "29696_-367616"}
    if disclosed_clusters != expected_disclosed:
        raise ValueError(
            f"unexpected disclosed east-test clusters: {sorted(disclosed_clusters)}"
        )

    assignments = []
    for spatial_key in sorted(by_cluster):
        x_coordinate = spatial_x(spatial_key)
        if x_coordinate in CALIBRATION_X:
            split = "calibration"
        elif x_coordinate in EMBARGO_X:
            split = "embargo"
        elif x_coordinate in EAST_X and spatial_key in disclosed_clusters:
            split = "disclosed_audit"
        elif x_coordinate in EAST_X:
            split = "sealed_test"
        else:
            raise ValueError(f"spatial cluster is outside the preregistered grid: {spatial_key}")
        assignments.append(
            {
                "spatial_key": spatial_key,
                "spatial_cluster_id": by_cluster[spatial_key][0]["spatial_cluster_id"],
                "x_coordinate": x_coordinate,
                "split": split,
                "years": sorted(int(value["year"]) for value in by_cluster[spatial_key]),
                "sample_ids": sorted(value["sample_id"] for value in by_cluster[spatial_key]),
            }
        )

    counts = Counter(value["split"] for value in assignments)
    expected_counts = {
        "calibration": 30,
        "embargo": 6,
        "sealed_test": 16,
        "disclosed_audit": 2,
    }
    if dict(counts) != expected_counts:
        raise ValueError(f"split counts drifted: {dict(counts)}")
    return {
        "schema": "olmoearth-release-spatial-split-v1",
        "exact_inputs": {
            "path": exact_inputs_path.as_posix(),
            "sha256": file_sha256(exact_inputs_path),
        },
        "prior_disclosure_inputs": {
            "path": disclosed_inputs_path.as_posix(),
            "sha256": file_sha256(disclosed_inputs_path),
        },
        "frozen_before_full_output_inspection": True,
        "split_rule": {
            "calibration_x": sorted(CALIBRATION_X),
            "embargo_x": sorted(EMBARGO_X),
            "east_x": sorted(EAST_X),
            "disclosed_east_clusters_removed_from_test": sorted(disclosed_clusters),
        },
        "assignments": assignments,
        "counts": {
            split: {
                "spatial_clusters": count,
                "site_years": count * 4,
                "adjacent_year_events": count * 3,
            }
            for split, count in expected_counts.items()
        },
        "analysis_contract": {
            "bridge_fit": "calibration only",
            "hyperparameter_selection": "grouped inner validation within calibration only",
            "embargo": "unused buffer; no fitting or model selection",
            "sealed_test": "one final evaluation after analysis code is frozen",
            "disclosed_audit": "descriptive audit only; never test",
            "all_years_of_each_location_share_one_split": True,
        },
        "claims_forbidden": [
            "korean_population_generalization",
            "task_accuracy",
            "semantic_retrieval_correctness",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--disclosed-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_split(args.exact_inputs, args.disclosed_inputs)
    rendered = canonical_bytes(result)
    atomic_create(args.output, rendered)
    marker_path = args.output.with_name("SPLIT_COMPLETE.json")
    atomic_create(
        marker_path,
        canonical_bytes(
            {
                "schema": "olmoearth-release-spatial-split-completion-v1",
                "split_manifest_sha256": hashlib.sha256(rendered).hexdigest(),
                "frozen_before_full_output_inspection": True,
            }
        ),
    )
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
