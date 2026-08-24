#!/usr/bin/env python3
"""Run one fail-closed 216-window OlmoEarth release inference on GPU0.

The two releases intentionally run in separate immutable roots.  This makes a
completed v1 run recoverable if v1.2 later fails, and prevents an old aggregate
completion marker from surviving a partial rerun.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmo_release_raster_contract import inspect_raster
from olmo_release_semantic_contract import (
    fingerprint_rslearn_runtime,
    normalize_checkpoint_manifest,
    validate_launcher_runtime_binding,
    validate_promoted_execution_contract,
    validate_resolved_config,
)
from prepare_olmo_release_audit_view import build_view
from hash_olmo_release_inputs import hash_referenced_files
from run_olmo_release_batch_gate import (
    TelemetrySampler,
    atomic_create,
    canonical_bytes,
    gpu_uuid,
    runtime_versions,
)
from run_olmo_release_smoke import (
    file_sha256,
    gpu_processes,
    output_inventory,
    validate_checkpoints,
)


RELEASES = {
    "v1": {
        "release_id": "olmoearth_v1_base",
        "repo_id": "allenai/OlmoEarth-v1-Base",
        "model_env": "OLMO_V1_MODEL_PATH",
        "output_layer": "embeddings_full_v1_legacy",
    },
    "v1_2": {
        "release_id": "olmoearth_v1_2_base",
        "repo_id": "allenai/OlmoEarth-v1_2-Base",
        "model_env": "OLMO_V1_2_MODEL_PATH",
        "output_layer": "embeddings_full_v1_2_legacy",
    },
}
EXPECTED_YEARS = {2023, 2024, 2025, 2026}
CROPS_PER_WINDOW = 961
FULL_RUNNER_CODE_SCHEMA = "olmoearth-release-full-runner-code-contract-v1"
FULL_RUNNER_CODE_OWNER_ROLE = "full_release_runner"
FULL_RUNNER_POST_CODE_SCHEMA = (
    "olmoearth-release-full-runner-post-run-code-verification-v1"
)


def stable_python_code_record(
    path: Path, *, module: str, role: str
) -> dict[str, Any]:
    """Return a stable, canonical inventory record for one Python source file."""

    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.suffix != ".py":
        raise ValueError(f"audit code is not a Python source file: {resolved}")
    before = resolved.stat()
    content = resolved.read_bytes()
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"audit code is not UTF-8 Python source: {resolved}") from exc
    if not content or b"\x00" in content:
        raise ValueError(f"audit code content is empty or binary: {resolved}")
    digest = hashlib.sha256(content).hexdigest()
    after = resolved.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(content) != after.st_size:
        raise ValueError(f"audit code changed while hashing: {resolved}")
    return {
        "role": role,
        "module": module,
        "content": "python_source_utf8",
        "path": resolved.as_posix(),
        "bytes": after.st_size,
        "sha256": digest,
    }


def build_python_code_contract(
    *,
    schema: str,
    owner_role: str,
    owner_module: str,
    owner_path: Path,
    direct_local_helpers: dict[str, Path],
) -> dict[str, Any]:
    """Build a deterministic owner/direct-helper source inventory."""

    owner = stable_python_code_record(
        owner_path, module=owner_module, role="owner"
    )
    helpers = [
        stable_python_code_record(path, module=module, role="direct_local_helper")
        for module, path in sorted(direct_local_helpers.items())
    ]
    inventory = {
        "owner_role": owner_role,
        "owner": owner,
        "direct_local_helpers": helpers,
    }
    return {
        "schema": schema,
        **inventory,
        "inventory_sha256": hashlib.sha256(canonical_bytes(inventory)).hexdigest(),
    }


def validate_python_code_contract(
    value: Any,
    *,
    schema: str,
    owner_role: str,
    owner_module: str,
    owner_path: Path,
    direct_local_helpers: dict[str, Path],
    require_live_match: bool,
) -> dict[str, Any]:
    """Validate schema, roles, paths, content records, digest, and optional live state."""

    required_top = {
        "schema",
        "owner_role",
        "owner",
        "direct_local_helpers",
        "inventory_sha256",
    }
    if not isinstance(value, dict) or set(value) != required_top:
        raise ValueError("code contract top-level fields drifted")
    if value.get("schema") != schema or value.get("owner_role") != owner_role:
        raise ValueError("code contract schema/owner role drifted")
    owner = value.get("owner")
    helpers = value.get("direct_local_helpers")
    if not isinstance(owner, dict) or not isinstance(helpers, list):
        raise ValueError("code contract lacks owner/direct-helper inventory")
    expected_paths = {
        owner_module: owner_path.resolve(strict=True),
        **{
            module: path.resolve(strict=True)
            for module, path in direct_local_helpers.items()
        },
    }
    expected_modules = sorted(direct_local_helpers)
    if owner.get("module") != owner_module or owner.get("role") != "owner":
        raise ValueError("code contract owner module/role drifted")
    if [record.get("module") for record in helpers] != expected_modules:
        raise ValueError("code contract direct-helper module inventory drifted")
    records = [owner, *helpers]
    if len({record.get("module") for record in records}) != len(records):
        raise ValueError("code contract contains duplicate module roles")
    required_record = {"role", "module", "content", "path", "bytes", "sha256"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != required_record:
            raise ValueError("code contract record fields drifted")
        module = record["module"]
        expected_role = "owner" if index == 0 else "direct_local_helper"
        if record.get("role") != expected_role:
            raise ValueError(f"code contract role drifted: {module}")
        if record.get("content") != "python_source_utf8":
            raise ValueError(f"code contract content role drifted: {module}")
        path_value = record.get("path")
        if (
            module not in expected_paths
            or not isinstance(path_value, str)
            or not Path(path_value).is_absolute()
            or Path(path_value) != expected_paths[module]
        ):
            raise ValueError(f"code contract canonical path drifted: {module}")
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            raise ValueError(f"code contract byte count is invalid: {module}")
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"code contract SHA-256 is invalid: {module}")
    inventory = {
        "owner_role": owner_role,
        "owner": dict(owner),
        "direct_local_helpers": [dict(record) for record in helpers],
    }
    inventory_sha = hashlib.sha256(canonical_bytes(inventory)).hexdigest()
    if value.get("inventory_sha256") != inventory_sha:
        raise ValueError("code contract inventory digest mismatch")
    normalized = {"schema": schema, **inventory, "inventory_sha256": inventory_sha}
    if require_live_match:
        live = build_python_code_contract(
            schema=schema,
            owner_role=owner_role,
            owner_module=owner_module,
            owner_path=owner_path,
            direct_local_helpers=direct_local_helpers,
        )
        if normalized != live:
            raise ValueError("persisted code contract differs from current live source")
    return normalized


def full_runner_helper_paths() -> dict[str, Path]:
    """Every local Python module imported directly by this full-runner module."""

    return {
        "hash_olmo_release_inputs": Path(hash_referenced_files.__code__.co_filename),
        "olmo_release_raster_contract": Path(inspect_raster.__code__.co_filename),
        "olmo_release_semantic_contract": Path(
            fingerprint_rslearn_runtime.__code__.co_filename
        ),
        "prepare_olmo_release_audit_view": Path(build_view.__code__.co_filename),
        "run_olmo_release_batch_gate": Path(atomic_create.__code__.co_filename),
        "run_olmo_release_smoke": Path(file_sha256.__code__.co_filename),
    }


def full_runner_code_contract() -> dict[str, Any]:
    return build_python_code_contract(
        schema=FULL_RUNNER_CODE_SCHEMA,
        owner_role=FULL_RUNNER_CODE_OWNER_ROLE,
        owner_module="run_olmo_release_full",
        owner_path=Path(__file__),
        direct_local_helpers=full_runner_helper_paths(),
    )


def validate_full_runner_code_contract(
    value: Any, *, require_live_match: bool = True
) -> dict[str, Any]:
    return validate_python_code_contract(
        value,
        schema=FULL_RUNNER_CODE_SCHEMA,
        owner_role=FULL_RUNNER_CODE_OWNER_ROLE,
        owner_module="run_olmo_release_full",
        owner_path=Path(__file__),
        direct_local_helpers=full_runner_helper_paths(),
        require_live_match=require_live_match,
    )


def verify_full_runner_code_stability(
    initial: dict[str, Any],
) -> dict[str, Any]:
    """Fail unless the canonical full-runner source inventory is unchanged."""

    validate_full_runner_code_contract(initial, require_live_match=False)
    live = full_runner_code_contract()
    if live != initial:
        raise ValueError(
            "full runner or directly imported local helper changed during execution"
        )
    validate_full_runner_code_contract(live, require_live_match=True)
    return live


def validate_full_release_command(
    value: Any, *, rslearn_entrypoint: Path, resolved_config: Path
) -> dict[str, Any]:
    """Accept only the fixed, absolute rslearn invocation with no extra wrapper."""

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("full release executed_command must be a string list")
    core = [
        rslearn_entrypoint.resolve().as_posix(),
        "model",
        "predict",
        "--config",
        resolved_config.resolve().as_posix(),
    ]
    if value == core:
        return {"wrapper": None, "core": core}
    raise ValueError("full release executed_command has an unexpected path or argument")


def validate_full_runner_code_evidence(
    *,
    preflight: dict[str, Any],
    run_summary: dict[str, Any],
    completion: dict[str, Any],
    result_root: Path,
    require_live_match: bool = True,
) -> dict[str, Any]:
    """Close preflight, post-run proof, summary, and completion around one code contract."""

    persisted = validate_full_runner_code_contract(
        preflight.get("full_runner_code_contract"),
        require_live_match=require_live_match,
    )
    if validate_full_runner_code_contract(
        run_summary.get("full_runner_code_contract"),
        require_live_match=require_live_match,
    ) != persisted:
        raise ValueError("run/preflight full-runner code contract drift")
    if (
        run_summary.get("post_run_full_runner_code_verified") is not True
        or run_summary.get("post_run_full_runner_code_error") is not None
    ):
        raise ValueError("release lacks successful post-run full-runner code verification")
    descriptor = run_summary.get("post_run_full_runner_code_verification")
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("full-runner code verification descriptor drift")
    expected_marker = result_root.resolve() / "POST_RUN_FULL_RUNNER_CODE_VERIFICATION.json"
    marker_path = descriptor.get("path")
    if (
        not isinstance(marker_path, str)
        or not Path(marker_path).is_absolute()
        or Path(marker_path).resolve() != expected_marker
        or not expected_marker.is_file()
        or file_sha256(expected_marker) != descriptor.get("sha256")
    ):
        raise ValueError("full-runner code verification evidence drift")
    marker = read_json(expected_marker)
    if (
        not isinstance(marker, dict)
        or set(marker)
        != {
            "schema",
            "status",
            "initial_full_runner_code_contract",
            "live_full_runner_code_contract",
            "error",
        }
        or marker.get("schema") != FULL_RUNNER_POST_CODE_SCHEMA
        or marker.get("status") != "verified"
        or marker.get("initial_full_runner_code_contract") != persisted
        or marker.get("live_full_runner_code_contract") != persisted
        or marker.get("error") is not None
    ):
        raise ValueError("full-runner code verification marker content drift")
    if (
        completion.get("post_run_full_runner_code_verified") is not True
        or completion.get("full_runner_code_contract_sha256")
        != persisted["inventory_sha256"]
        or completion.get("post_run_full_runner_code_verification_sha256")
        != descriptor["sha256"]
    ):
        raise ValueError("release completion/code-contract binding drift")
    return persisted


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def validate_exact_inputs(
    exact_inputs: Path,
    exact_complete: Path,
    source_dataset: Path,
    hash_workers: int,
) -> list[dict[str, Any]]:
    payload = read_json(exact_inputs)
    completion = read_json(exact_complete)
    if completion.get("exact_inputs_sha256") != file_sha256(exact_inputs):
        raise ValueError("exact input completion marker does not match the manifest")
    records = payload.get("records", [])
    if not payload.get("exact_tensor_file_pairing_ready") or len(records) != 216:
        raise ValueError("full release run requires exactly 216 content-hashed records")
    if completion.get("records") != 216 or completion.get("unique_files") != 5616:
        raise ValueError("exact input completion counts do not match the full contract")
    sample_ids = [record["sample_id"] for record in records]
    windows = [record["window_name"] for record in records]
    if len(set(sample_ids)) != 216 or len(set(windows)) != 216:
        raise ValueError("full input identities are not unique")
    panel = Counter(
        (record["spatial_cluster_id"], int(record["year"])) for record in records
    )
    clusters = {record["spatial_cluster_id"] for record in records}
    expected_panel = {(cluster, year) for cluster in clusters for year in EXPECTED_YEARS}
    if len(clusters) != 54 or set(panel) != expected_panel or any(value != 1 for value in panel.values()):
        raise ValueError("full inputs are not an exact 54-cluster by four-year panel")

    resolved_source = source_dataset.resolve()
    referenced_paths = set()
    for record in records:
        if record.get("hash_policy") != "sha256" or len(record["input_layers"]) != 12:
            raise ValueError(f"record is not an exact 12-period SHA input: {record['sample_id']}")
        window_dir = Path(record["window_dir"])
        if not window_dir.resolve().is_relative_to(resolved_source):
            raise ValueError(f"input window escapes the source dataset: {window_dir}")
        if [layer["period_index"] for layer in record["input_layers"]] != list(range(12)):
            raise ValueError(f"period order drift: {record['sample_id']}")
        inventories = [record["items_json"], record["window_metadata"]]
        for layer in record["input_layers"]:
            inventories.extend((layer["geotiff"], layer["metadata"]))
        for inventory in inventories:
            path = Path(inventory["path"])
            if not path.resolve().is_relative_to(resolved_source):
                raise ValueError(f"input file escapes the source dataset: {path}")
            if path.stat().st_size != inventory["bytes"] or len(inventory.get("sha256", "")) != 64:
                raise ValueError(f"input inventory size/SHA contract drift: {path}")
            referenced_paths.add(path.as_posix())
        identity_payload = {
            "window_name": record["window_name"],
            "input_layers": record["input_layers"],
            "items_json": record["items_json"],
            "window_metadata": record["window_metadata"],
        }
        if hashlib.sha256(canonical_bytes(identity_payload)).hexdigest() != record[
            "input_bundle_identity"
        ]:
            raise ValueError(f"input bundle identity drift: {record['sample_id']}")
    if len(referenced_paths) != 5616:
        raise ValueError(f"expected 5,616 unique input files, found {len(referenced_paths)}")
    live_inventories = hash_referenced_files(records, workers=hash_workers)
    if set(live_inventories) != referenced_paths:
        raise ValueError("live input hash inventory paths differ from the frozen manifest")
    for record in records:
        inventories = [record["items_json"], record["window_metadata"]]
        for layer in record["input_layers"]:
            inventories.extend((layer["geotiff"], layer["metadata"]))
        for inventory in inventories:
            live = live_inventories[inventory["path"]]
            if live["bytes"] != inventory["bytes"] or live["sha256"] != inventory["sha256"]:
                raise ValueError(f"live input SHA drift from frozen manifest: {inventory['path']}")
    return records


def validate_spatial_split(
    split_manifest: Path,
    split_complete: Path,
    exact_inputs: Path,
    exact_records: list[dict[str, Any]],
) -> dict[str, Any]:
    split = read_json(split_manifest)
    completion = read_json(split_complete)
    if completion.get("split_manifest_sha256") != file_sha256(split_manifest):
        raise ValueError("spatial split completion marker mismatch")
    if split.get("exact_inputs", {}).get("sha256") != file_sha256(exact_inputs):
        raise ValueError("spatial split was not frozen against these exact inputs")
    if split.get("schema") != "olmoearth-release-spatial-split-v1":
        raise ValueError("unrecognized spatial split schema")
    if split.get("frozen_before_full_output_inspection") is not True:
        raise ValueError("spatial split was not frozen before output inspection")
    if completion.get("frozen_before_full_output_inspection") is not True:
        raise ValueError("spatial split completion marker lacks the freeze assertion")
    expected = {
        "calibration": 30,
        "embargo": 6,
        "sealed_test": 16,
        "disclosed_audit": 2,
    }
    records_by_key: dict[str, list[dict[str, Any]]] = {}
    for record in exact_records:
        records_by_key.setdefault(record["spatial_key"], []).append(record)
    assignments = split.get("assignments", [])
    if len(assignments) != 54 or len({value.get("spatial_key") for value in assignments}) != 54:
        raise ValueError("spatial split must contain 54 unique assignments")
    if {value.get("spatial_key") for value in assignments} != set(records_by_key):
        raise ValueError("spatial split assignments do not cover the exact input population")
    recomputed = Counter()
    for assignment in assignments:
        key = assignment["spatial_key"]
        values = records_by_key[key]
        if assignment.get("split") not in expected:
            raise ValueError(f"unknown spatial split assignment: {assignment}")
        if assignment.get("spatial_cluster_id") != values[0]["spatial_cluster_id"]:
            raise ValueError(f"spatial cluster ID drift in split: {key}")
        if assignment.get("years") != sorted(int(value["year"]) for value in values):
            raise ValueError(f"year membership drift in split: {key}")
        if assignment.get("sample_ids") != sorted(value["sample_id"] for value in values):
            raise ValueError(f"sample membership drift in split: {key}")
        recomputed[assignment["split"]] += 1
    if dict(recomputed) != expected:
        raise ValueError(f"spatial split assignment counts drifted: {dict(recomputed)}")
    counts = split.get("counts", {})
    if {key: value.get("spatial_clusters") for key, value in counts.items()} != expected:
        raise ValueError("spatial split counts drifted")
    return split


def validate_batch_contract(
    path: Path,
    complete_marker: Path,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    contract = read_json(path)
    completion = read_json(complete_marker)
    if completion.get("batch_contract_sha256") != file_sha256(path):
        raise ValueError("batch promotion completion marker mismatch")
    if completion.get("status") != "promoted":
        raise ValueError("batch promotion marker is not complete")
    if contract.get("schema") != "olmoearth-release-batch-contract-v1":
        raise ValueError("unrecognized batch contract schema")
    if contract.get("status") != "promoted":
        raise ValueError("batch contract is not promoted")
    selected = contract.get("selected", {})
    if selected.get("batch_size") != batch_size or selected.get("num_workers") != workers:
        raise ValueError("requested batch/workers differ from the promoted contract")
    if contract.get("full_run_allowed") is not True:
        raise ValueError("batch contract does not allow the full run")
    if contract.get("promotion_pending") != []:
        raise ValueError("batch contract still has pending promotion checks")
    if contract.get("execution_contract", {}).get("schema") != (
        "olmoearth-release-execution-contract-v1"
    ):
        raise ValueError("batch contract lacks a promoted scientific execution contract")
    gate_checks = contract.get("gate_checks", {})
    if not gate_checks or not all(value is True for value in gate_checks.values()):
        raise ValueError("batch contract gate checks are incomplete")
    evidence = contract.get("evidence_files", [])
    if len(evidence) < 3:
        raise ValueError("batch contract requires at least three bound evidence files")
    for item in evidence:
        evidence_path = Path(item["path"])
        if file_sha256(evidence_path) != item["sha256"]:
            raise ValueError(f"batch-contract evidence drift: {evidence_path}")
    return contract


def existing_disk_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"no existing ancestor for {path}")
        candidate = candidate.parent
    return candidate


def render_config(
    template: Path,
    output: Path,
    *,
    model_env: str,
    output_layer: str,
    batch_size: int,
    workers: int,
) -> None:
    rendered = template.read_text(encoding="utf-8")
    replacements = {
        "__MODEL_PATH__": "${" + model_env + "}",
        "__OUTPUT_LAYER__": output_layer,
        "__BATCH_SIZE__": str(batch_size),
        "__NUM_WORKERS__": str(workers),
    }
    for placeholder, replacement in replacements.items():
        if rendered.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        rendered = rendered.replace(placeholder, replacement)
    if any(placeholder in rendered for placeholder in replacements):
        raise ValueError("unresolved full-run config placeholder")
    atomic_create(output, rendered.encode("utf-8"))


def validate_fresh_view(
    dataset_root: Path,
    records: list[dict[str, Any]],
    output_layer: str,
) -> None:
    windows_root = dataset_root / "windows/default"
    actual_windows = {path.name for path in windows_root.iterdir() if path.is_dir()}
    expected_windows = {record["window_name"] for record in records}
    if actual_windows != expected_windows:
        raise ValueError("full audit view does not contain exactly 216 manifest windows")
    for record in records:
        target_window = windows_root / record["window_name"]
        if (target_window / "layers" / output_layer).exists():
            raise ValueError(f"fresh view already contains output: {record['window_name']}")
        for layer in record["input_layers"]:
            view_layer = target_window / "layers" / layer["layer_name"]
            source_layer = Path(layer["geotiff"]["path"]).parents[1]
            if not view_layer.is_symlink() or view_layer.resolve() != source_layer.resolve():
                raise ValueError(f"view link drift: {view_layer}")
        for field, name in (("items_json", "items.json"), ("window_metadata", "metadata.json")):
            target = target_window / name
            inventory = record[field]
            if target.stat().st_size != inventory["bytes"] or file_sha256(target) != inventory["sha256"]:
                raise ValueError(f"view metadata is not bound to the exact manifest: {target}")


def validate_output_raster_contracts(
    outputs: list[dict[str, Any]], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records_by_sample = {record["sample_id"]: record for record in records}
    if len(outputs) != len(records_by_sample) or {
        output["sample_id"] for output in outputs
    } != set(records_by_sample):
        raise ValueError("output health scan does not cover the exact sample population")
    health = []
    for output in outputs:
        record = records_by_sample[output["sample_id"]]
        window_metadata = read_json(Path(record["window_metadata"]["path"]))
        contract, _ = inspect_raster(Path(output["path"]), window_metadata)
        health.append({"sample_id": output["sample_id"], **contract})
    return health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=sorted(RELEASES), required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--exact-complete", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-complete", type=Path, required=True)
    parser.add_argument("--batch-contract", type=Path, required=True)
    parser.add_argument("--batch-contract-complete", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--rslearn", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--minimum-free-gib", type=float, default=200.0)
    parser.add_argument("--maximum-memory-mib", type=float, default=115000.0)
    parser.add_argument("--input-hash-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_full_runner_code_contract = full_runner_code_contract()
    release = RELEASES[args.release]
    if args.gpu != "0":
        raise SystemExit("this experiment is hard-pinned to physical GPU0; GPU1 is forbidden")
    gpu_lock_path = Path("/home/work/data/.jobs/gpu0.lock")
    if args.run_root.exists():
        raise SystemExit(f"refusing existing release run root: {args.run_root}")
    if args.batch_size < 1 or not 0 <= args.num_workers <= 8:
        raise SystemExit("invalid batch/workers")
    free_gib = shutil.disk_usage(existing_disk_ancestor(args.run_root.parent)).free / 1024**3
    if free_gib < args.minimum_free_gib:
        raise SystemExit(
            f"only {free_gib:.1f} GiB free; full release run requires {args.minimum_free_gib:.1f} GiB"
        )
    if not 1 <= args.input_hash_workers <= 2:
        raise SystemExit("input hash workers must be one or two to limit shared-storage pressure")
    records = validate_exact_inputs(
        args.exact_inputs,
        args.exact_complete,
        args.source_dataset,
        args.input_hash_workers,
    )
    split = validate_spatial_split(
        args.split_manifest,
        args.split_complete,
        args.exact_inputs,
        records,
    )
    batch_contract = validate_batch_contract(
        args.batch_contract,
        args.batch_contract_complete,
        args.batch_size,
        args.num_workers,
    )
    checkpoints = validate_checkpoints(args.checkpoint_manifest)
    checkpoint_models = normalize_checkpoint_manifest(read_json(args.checkpoint_manifest))
    model = checkpoints[release["repo_id"]]
    rslearn_runtime_fingerprint = fingerprint_rslearn_runtime(args.rslearn)
    args.rslearn = Path(rslearn_runtime_fingerprint["entrypoint"]["path"])
    selected_uuid = gpu_uuid(args.gpu)
    current_runtime_versions = validate_launcher_runtime_binding(
        runtime_versions(), Path(os.sys.executable), rslearn_runtime_fingerprint
    )
    if gpu_processes(args.gpu):
        raise SystemExit("selected GPU is occupied before the full release run")

    gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
    with gpu_lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"GPU lock is held: {gpu_lock_path}") from exc
        if gpu_processes(args.gpu):
            raise SystemExit("selected GPU became occupied after acquiring its lock")
        args.run_root.mkdir(parents=True)
        dataset_root = args.run_root / "dataset"
        result_root = args.run_root / "result"
        config_path = args.run_root / "resolved_config.yaml"
        result_root.mkdir()
        view = build_view(
            args.source_dataset,
            args.exact_inputs,
            dataset_root,
            output_layers=(release["output_layer"],),
        )
        validate_fresh_view(dataset_root, records, release["output_layer"])
        render_config(
            args.template_config,
            config_path,
            model_env=release["model_env"],
            output_layer=release["output_layer"],
            batch_size=args.batch_size,
            workers=args.num_workers,
        )
        resolved_config_contract = validate_resolved_config(
            config_path,
            model_env=release["model_env"],
            output_layer=release["output_layer"],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        execution_contract_check = validate_promoted_execution_contract(
            batch_contract["execution_contract"],
            repo_id=release["repo_id"],
            resolved_config_contract=resolved_config_contract,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            runtime_versions=current_runtime_versions,
            gpu_index=args.gpu,
            gpu_uuid=selected_uuid,
            checkpoint_manifest_sha256=file_sha256(args.checkpoint_manifest),
            checkpoint_models=checkpoint_models,
            rslearn_runtime_fingerprint=rslearn_runtime_fingerprint,
        )
        preflight = {
            "schema": "olmoearth-release-full-preflight-v1",
            "status": "ready",
            "release": release,
            "records": 216,
            "spatial_clusters": 54,
            "exact_inputs_sha256": file_sha256(args.exact_inputs),
            "exact_complete_sha256": file_sha256(args.exact_complete),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "split_complete_sha256": file_sha256(args.split_complete),
            "batch_contract_sha256": file_sha256(args.batch_contract),
            "batch_contract_complete_sha256": file_sha256(args.batch_contract_complete),
            "batch_contract": batch_contract,
            "checkpoint_manifest_sha256": file_sha256(args.checkpoint_manifest),
            "template_config_sha256": file_sha256(args.template_config),
            "resolved_config_sha256": file_sha256(config_path),
            "resolved_config_contract": resolved_config_contract,
            "promoted_execution_contract_check": execution_contract_check,
            "selected_gpu": args.gpu,
            "selected_gpu_uuid": selected_uuid,
            "monitored_other_gpu": "1",
            "monitored_other_gpu_uuid": gpu_uuid("1"),
            "gpu_lock": gpu_lock_path.as_posix(),
            "free_gib_before": free_gib,
            "runtime_versions": current_runtime_versions,
            "rslearn_runtime_fingerprint": rslearn_runtime_fingerprint,
            "full_runner_code_contract": initial_full_runner_code_contract,
            "input_hash_workers": args.input_hash_workers,
            "view": view,
            "sealed_test_not_used_by_runner": split["analysis_contract"]["sealed_test"],
        }
        preflight_path = result_root / "preflight.json"
        atomic_create(preflight_path, canonical_bytes(preflight))
        if gpu_processes(args.gpu):
            raise SystemExit("selected GPU became occupied immediately before rslearn")

        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.update(
            {
                "DATASET_PATH": dataset_root.as_posix(),
                release["model_env"]: model["snapshot_path"],
                "CUDA_VISIBLE_DEVICES": selected_uuid,
            }
        )
        log_path = result_root / "rslearn.log"
        started_at = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()
        selected_telemetry = TelemetrySampler(args.gpu)
        other_gpu = "1" if args.gpu == "0" else "0"
        other_uuid = gpu_uuid(other_gpu)
        other_telemetry = TelemetrySampler(other_gpu)
        command = [
            args.rslearn.resolve().as_posix(),
            "model",
            "predict",
            "--config",
            config_path.resolve().as_posix(),
        ]
        executed_command_contract = validate_full_release_command(
            command,
            rslearn_entrypoint=args.rslearn,
            resolved_config=config_path,
        )
        process: subprocess.Popen[str] | None = None
        return_code = -999
        execution_error = None
        forwarded_signals: list[int] = []

        def forward_signal(signum: int, _frame: Any) -> None:
            forwarded_signals.append(signum)
            if process is not None and process.poll() is None:
                process.send_signal(signum)

        previous_handlers = {
            signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
        }
        for signum in previous_handlers:
            signal.signal(signum, forward_signal)
        selected_telemetry.start()
        other_telemetry.start()
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                return_code = process.wait()
        except Exception as exc:
            execution_error = f"{type(exc).__name__}: {exc}"
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            try:
                selected_telemetry.stop()
            except Exception as exc:
                execution_error = execution_error or f"selected telemetry stop: {type(exc).__name__}: {exc}"
            try:
                other_telemetry.stop()
            except Exception as exc:
                execution_error = execution_error or f"other telemetry stop: {type(exc).__name__}: {exc}"
        wall_seconds = time.monotonic() - start
        selected_telemetry_path = result_root / "gpu_selected_telemetry.json"
        other_telemetry_path = result_root / "gpu_other_telemetry.json"
        atomic_create(selected_telemetry_path, canonical_bytes(selected_telemetry.rows))
        atomic_create(other_telemetry_path, canonical_bytes(other_telemetry.rows))
        selected_summary = selected_telemetry.summary()
        failure_reasons = []
        if execution_error is not None:
            failure_reasons.append(f"execution_exception:{execution_error}")
        if return_code != 0:
            failure_reasons.append(f"rslearn_exit_{return_code}")
        if forwarded_signals:
            failure_reasons.append(f"runner_forwarded_signals:{forwarded_signals}")
        if selected_summary.get("errors"):
            failure_reasons.append("selected_gpu_telemetry_error")
        if selected_summary.get("gpu_uuid_values") != [selected_uuid]:
            failure_reasons.append("selected_gpu_uuid_mismatch")
        if selected_summary.get("maximum_sample_gap_seconds", 0.0) > 3.0:
            failure_reasons.append("selected_gpu_telemetry_gap_above_3s")
        if selected_summary.get("gpu_utilization_available_samples", 0) == 0:
            failure_reasons.append("selected_gpu_utilization_unavailable")
        peak_memory = selected_summary.get("peak_memory_used_mib")
        if peak_memory is None:
            failure_reasons.append("selected_gpu_memory_telemetry_unavailable")
        elif peak_memory > args.maximum_memory_mib:
            failure_reasons.append("selected_gpu_peak_memory_above_gate")
        other_summary = other_telemetry.summary()
        if other_summary.get("errors"):
            failure_reasons.append("other_gpu_telemetry_error")
        if other_summary.get("gpu_uuid_values") != [other_uuid]:
            failure_reasons.append("other_gpu_uuid_mismatch")
        if other_summary.get("maximum_sample_gap_seconds", 0.0) > 3.0:
            failure_reasons.append("other_gpu_telemetry_gap_above_3s")

        outputs = []
        output_health = []
        if return_code == 0:
            try:
                outputs = output_inventory(
                    dataset_root, release["output_layer"], records, 0.0
                )
            except Exception as exc:
                failure_reasons.append(
                    f"output_inventory_exception:{type(exc).__name__}:{exc}"
                )
        if len(outputs) != 216:
            failure_reasons.append("outputs_not_216_of_216")
        elif outputs:
            try:
                output_health = validate_output_raster_contracts(outputs, records)
            except Exception as exc:
                failure_reasons.append(
                    f"output_raster_contract_exception:{type(exc).__name__}:{exc}"
                )
        post_run_inputs_verified = False
        try:
            validate_exact_inputs(
                args.exact_inputs,
                args.exact_complete,
                args.source_dataset,
                args.input_hash_workers,
            )
            post_run_inputs_verified = True
        except Exception as exc:
            failure_reasons.append(
                f"post_run_input_verification_exception:{type(exc).__name__}:{exc}"
            )
        post_run_checkpoints_verified = False
        checkpoint_verification_marker = result_root / "POST_RUN_CHECKPOINTS_VERIFIED.json"
        try:
            if file_sha256(args.checkpoint_manifest) != preflight[
                "checkpoint_manifest_sha256"
            ]:
                raise ValueError("checkpoint manifest SHA changed during the full run")
            validate_checkpoints(args.checkpoint_manifest)
            live_checkpoint_models = normalize_checkpoint_manifest(
                read_json(args.checkpoint_manifest)
            )
            if live_checkpoint_models != checkpoint_models:
                raise ValueError("checkpoint revision/file hashes changed during the full run")
            atomic_create(
                checkpoint_verification_marker,
                canonical_bytes(
                    {
                        "schema": "olmoearth-release-post-run-checkpoint-verification-v1",
                        "status": "verified",
                        "checkpoint_manifest_sha256": file_sha256(
                            args.checkpoint_manifest
                        ),
                        "repo_id": release["repo_id"],
                        "revision": live_checkpoint_models[release["repo_id"]][
                            "revision"
                        ],
                        "checkpoint_files": live_checkpoint_models[
                            release["repo_id"]
                        ]["files"],
                    }
                ),
            )
            post_run_checkpoints_verified = True
        except Exception as exc:
            failure_reasons.append(
                f"post_run_checkpoint_verification_exception:{type(exc).__name__}:{exc}"
            )
        post_run_rslearn_runtime_verified = False
        rslearn_verification_marker = result_root / "POST_RUN_RSLEARN_RUNTIME_VERIFIED.json"
        try:
            live_rslearn_runtime = fingerprint_rslearn_runtime(args.rslearn)
            if live_rslearn_runtime != rslearn_runtime_fingerprint:
                raise ValueError(
                    "rslearn executable/interpreter/package source changed during the full run"
                )
            atomic_create(
                rslearn_verification_marker,
                canonical_bytes(
                    {
                        "schema": "olmoearth-release-post-run-rslearn-runtime-verification-v1",
                        "status": "verified",
                        "rslearn_runtime_fingerprint": live_rslearn_runtime,
                    }
                ),
            )
            post_run_rslearn_runtime_verified = True
        except Exception as exc:
            failure_reasons.append(
                f"post_run_rslearn_runtime_verification_exception:{type(exc).__name__}:{exc}"
            )
        post_run_full_runner_code_verified = False
        post_run_full_runner_code_error = None
        live_full_runner_code_contract: dict[str, Any] | None = None
        try:
            live_full_runner_code_contract = verify_full_runner_code_stability(
                initial_full_runner_code_contract
            )
            post_run_full_runner_code_verified = True
        except Exception as exc:
            post_run_full_runner_code_error = f"{type(exc).__name__}: {exc}"
            failure_reasons.append(
                "post_run_full_runner_code_verification_exception:"
                f"{post_run_full_runner_code_error}"
            )
        runner_code_marker = result_root / "POST_RUN_FULL_RUNNER_CODE_VERIFICATION.json"
        atomic_create(
            runner_code_marker,
            canonical_bytes(
                {
                    "schema": FULL_RUNNER_POST_CODE_SCHEMA,
                    "status": "verified"
                    if post_run_full_runner_code_verified
                    else "failed",
                    "initial_full_runner_code_contract": initial_full_runner_code_contract,
                    "live_full_runner_code_contract": live_full_runner_code_contract,
                    "error": post_run_full_runner_code_error,
                }
            ),
        )
        result = {
            "schema": "olmoearth-release-full-result-v1",
            "status": "complete" if not failure_reasons else "failed",
            "release": release,
            "repo_id": release["repo_id"],
            "revision": model["revision"],
            "checkpoint_files": model["files"],
            "records": 216,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "timestamp_track": "legacy_timestamps",
            "started_at": started_at,
            "fresh_release_root_created_by_runner": True,
            "wall_seconds": round(wall_seconds, 6),
            "expected_crops": 216 * CROPS_PER_WINDOW,
            "end_to_end_crops_per_second": 216 * CROPS_PER_WINDOW / wall_seconds,
            "preflight_sha256": file_sha256(preflight_path),
            "resolved_config_sha256": file_sha256(config_path),
            "log_path": log_path.as_posix(),
            "log_sha256": file_sha256(log_path),
            "selected_gpu_telemetry": {
                "path": selected_telemetry_path.as_posix(),
                "sha256": file_sha256(selected_telemetry_path),
                "summary": selected_summary,
            },
            "other_gpu_telemetry": {
                "path": other_telemetry_path.as_posix(),
                "sha256": file_sha256(other_telemetry_path),
                "summary": other_summary,
            },
            "executed_command": command,
            "executed_command_contract": executed_command_contract,
            "post_run_inputs_verified": post_run_inputs_verified,
            "post_run_checkpoints_verified": post_run_checkpoints_verified,
            "post_run_checkpoint_verification": {
                "path": checkpoint_verification_marker.as_posix(),
                "sha256": file_sha256(checkpoint_verification_marker),
            }
            if post_run_checkpoints_verified
            else None,
            "post_run_rslearn_runtime_verified": post_run_rslearn_runtime_verified,
            "post_run_rslearn_runtime_verification": {
                "path": rslearn_verification_marker.as_posix(),
                "sha256": file_sha256(rslearn_verification_marker),
            }
            if post_run_rslearn_runtime_verified
            else None,
            "full_runner_code_contract": initial_full_runner_code_contract,
            "post_run_full_runner_code_verified": post_run_full_runner_code_verified,
            "post_run_full_runner_code_error": post_run_full_runner_code_error,
            "post_run_full_runner_code_verification": {
                "path": runner_code_marker.resolve().as_posix(),
                "sha256": file_sha256(runner_code_marker),
            },
            "outputs": outputs,
            "output_health": output_health,
            "failure_reasons": failure_reasons,
            "claims_allowed": ["single-release full exact-input inference completion"],
            "claims_forbidden": [
                "task_accuracy",
                "negative_transfer",
                "cloud_robustness",
                "release_compatibility_before_paired_finalization",
            ],
        }
        result_path = result_root / "run_summary.json"
        atomic_create(result_path, canonical_bytes(result))
        marker_name = "RELEASE_COMPLETE.json" if not failure_reasons else "RELEASE_FAILED.json"
        atomic_create(
            result_root / marker_name,
            canonical_bytes(
                {
                    "schema": "olmoearth-release-full-completion-v1",
                    "status": result["status"],
                    "run_summary_sha256": file_sha256(result_path),
                    "post_run_full_runner_code_verified": post_run_full_runner_code_verified,
                    "full_runner_code_contract_sha256": initial_full_runner_code_contract[
                        "inventory_sha256"
                    ],
                    "post_run_full_runner_code_verification_sha256": file_sha256(
                        runner_code_marker
                    ),
                }
            ),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if failure_reasons:
            raise SystemExit(f"release failed gates: {failure_reasons}")


if __name__ == "__main__":
    main()
