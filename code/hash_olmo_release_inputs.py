#!/usr/bin/env python3
"""Freeze release-audit inputs with stable SHA-256 content identities.

The source may be the small JSON selection used by the P0 smoke test or the
newline-delimited 216 site-year population manifest.  The final JSON and its
completion marker are written atomically and are deterministic for the same
source path and bytes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from build_olmo_release_audit_manifest import (
    canonical_bytes,
    file_sha256,
    input_layer_name,
    inventory_file,
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read either a JSON payload containing ``records`` or a JSONL manifest."""
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            records = [json.loads(line) for line in source if line.strip()]
    else:
        payload = read_json(path)
        records = payload.get("records", [])
    if not records:
        raise ValueError(f"no input records found in {path}")
    return records


def referenced_inventories(record: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for layer in record["input_layers"]:
        result.extend((layer["geotiff"], layer["metadata"]))
    result.extend((record["items_json"], record["window_metadata"]))
    return result


def stable_inventory(path: Path, expected_bytes: int) -> dict[str, Any]:
    """Hash a file and reject a file that changed while it was being read."""
    before = path.stat()
    if before.st_size != expected_bytes:
        raise ValueError(
            f"metadata size drift before hashing: {path} "
            f"({before.st_size} != {expected_bytes})"
        )
    digest = file_sha256(path)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError(f"input changed while hashing: {path}")
    return {"path": path.as_posix(), "bytes": after.st_size, "sha256": digest}


def hash_referenced_files(
    records: list[dict[str, Any]], workers: int
) -> dict[str, dict[str, Any]]:
    """Hash each unique path once while preserving deterministic output order."""
    expected_by_path: dict[str, int] = {}
    for record in records:
        for inventory in referenced_inventories(record):
            path = Path(inventory["path"])
            canonical_path = path.as_posix()
            expected_bytes = int(inventory["bytes"])
            previous = expected_by_path.setdefault(canonical_path, expected_bytes)
            if previous != expected_bytes:
                raise ValueError(f"conflicting byte counts for {canonical_path}")

    ordered = sorted(expected_by_path.items())
    if workers == 1:
        values = [stable_inventory(Path(path), size) for path, size in ordered]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            values = list(
                executor.map(
                    lambda pair: stable_inventory(Path(pair[0]), pair[1]),
                    ordered,
                )
            )
    return {value["path"]: value for value in values}


def upgrade_record(
    record: dict[str, Any],
    inventory_by_path: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    upgraded = json.loads(json.dumps(record))
    for layer in upgraded["input_layers"]:
        for field in ("geotiff", "metadata"):
            path = Path(layer[field]["path"])
            layer[field] = (
                inventory_by_path[path.as_posix()]
                if inventory_by_path is not None
                else inventory_file(path, "sha256")
            )
    for field in ("items_json", "window_metadata"):
        path = Path(upgraded[field]["path"])
        upgraded[field] = (
            inventory_by_path[path.as_posix()]
            if inventory_by_path is not None
            else inventory_file(path, "sha256")
        )
    identity_payload = {
        "window_name": upgraded["window_name"],
        "input_layers": upgraded["input_layers"],
        "items_json": upgraded["items_json"],
        "window_metadata": upgraded["window_metadata"],
    }
    upgraded["hash_policy"] = "sha256"
    upgraded["input_bundle_identity"] = hashlib.sha256(
        canonical_bytes(identity_payload)
    ).hexdigest()
    return upgraded


def validate_contract(
    records: list[dict[str, Any]],
    expected_records: int | None,
    expected_spatial_clusters: int | None,
    expected_years: set[int] | None,
) -> None:
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(f"expected {expected_records} records, found {len(records)}")
    sample_ids = [record["sample_id"] for record in records]
    window_names = [record["window_name"] for record in records]
    if len(set(sample_ids)) != len(records):
        raise ValueError("sample IDs are not unique")
    if len(set(window_names)) != len(records):
        raise ValueError("window names are not unique")
    clusters = {record["spatial_cluster_id"] for record in records}
    if expected_spatial_clusters is not None and len(clusters) != expected_spatial_clusters:
        raise ValueError(
            f"expected {expected_spatial_clusters} spatial clusters, found {len(clusters)}"
        )
    years = {int(record["year"]) for record in records}
    if expected_years is not None and years != expected_years:
        raise ValueError(f"expected years {sorted(expected_years)}, found {sorted(years)}")
    for record in records:
        periods = [int(layer["period_index"]) for layer in record["input_layers"]]
        if periods != list(range(12)):
            raise ValueError(f"record does not contain ordered periods 0..11: {record['sample_id']}")
        layer_names = [layer["layer_name"] for layer in record["input_layers"]]
        if layer_names != [input_layer_name(index) for index in range(12)]:
            raise ValueError(f"record has an unexpected layer order: {record['sample_id']}")
    if expected_spatial_clusters is not None and expected_years is not None:
        panel = Counter(
            (record["spatial_cluster_id"], int(record["year"])) for record in records
        )
        expected_panel = {
            (cluster, year) for cluster in clusters for year in expected_years
        }
        if set(panel) != expected_panel or any(value != 1 for value in panel.values()):
            raise ValueError("spatial-cluster by year panel is not exactly one complete cross")


def write_if_absent_or_identical(path: Path, content: bytes) -> None:
    """Atomically create a result; never replace a different evidence file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace different evidence: {path}")
        return
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expected-spatial-clusters", type=int)
    parser.add_argument("--expected-years", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least one")
    source_records = read_records(args.input)
    validate_contract(
        source_records,
        args.expected_records,
        args.expected_spatial_clusters,
        set(args.expected_years) if args.expected_years else None,
    )
    inventory_by_path = hash_referenced_files(source_records, args.workers)
    records = [
        upgrade_record(record, inventory_by_path=inventory_by_path)
        for record in source_records
    ]
    result = {
        "schema": "olmoearth-release-exact-input-selection-v1",
        "selection_source": {
            "path": args.input.as_posix(),
            "sha256": file_sha256(args.input),
        },
        "exact_tensor_file_pairing_ready": True,
        "content_hash_algorithm": "sha256",
        "records": records,
    }
    rendered = canonical_bytes(result)
    write_if_absent_or_identical(args.output, rendered)
    result_sha256 = hashlib.sha256(rendered).hexdigest()
    marker = {
        "schema": "olmoearth-release-exact-input-completion-v1",
        "exact_inputs_sha256": result_sha256,
        "records": len(records),
        "spatial_clusters": len({record["spatial_cluster_id"] for record in records}),
        "years": sorted({int(record["year"]) for record in records}),
        "referenced_files": sum(len(record["input_layers"]) * 2 + 2 for record in records),
        "unique_files": len(inventory_by_path),
    }
    marker_path = args.output.with_name("EXACT_INPUTS_COMPLETE.json")
    write_if_absent_or_identical(marker_path, canonical_bytes(marker))
    print(
        json.dumps(
            {
                "records": len(records),
                "referenced_files_hashed": sum(
                    len(record["input_layers"]) * 2 + 2 for record in records
                ),
                "unique_files_hashed": len(inventory_by_path),
                "output": args.output.as_posix(),
                "output_sha256": result_sha256,
                "completion_marker": marker_path.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
