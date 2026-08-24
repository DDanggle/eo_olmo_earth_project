#!/usr/bin/env python3
"""Close the paired 216-window release evidence bundle after both runs finish."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

import numpy as np

from hash_olmo_release_inputs import hash_referenced_files, stable_inventory
from olmo_release_raster_contract import inspect_raster
from run_olmo_release_batch_gate import atomic_create, canonical_bytes
from run_olmo_release_full import (
    build_python_code_contract,
    full_runner_code_contract,
    validate_full_release_command,
    validate_full_runner_code_contract,
    validate_full_runner_code_evidence,
    validate_python_code_contract,
)
from run_olmo_release_smoke import file_sha256


EXPECTED = {
    "v1": ("olmoearth_v1_base", "embeddings_full_v1_legacy"),
    "v1_2": ("olmoearth_v1_2_base", "embeddings_full_v1_2_legacy"),
}
FINALIZER_CODE_SCHEMA = "olmoearth-release-full-finalizer-code-contract-v1"
FINALIZER_CODE_OWNER_ROLE = "paired_release_finalizer"
FINALIZER_POST_CODE_SCHEMA = (
    "olmoearth-release-full-finalizer-post-run-code-verification-v1"
)


def finalizer_helper_paths() -> dict[str, Path]:
    """Every local Python module imported directly by this finalizer."""

    return {
        "hash_olmo_release_inputs": Path(hash_referenced_files.__code__.co_filename),
        "olmo_release_raster_contract": Path(inspect_raster.__code__.co_filename),
        "run_olmo_release_batch_gate": Path(atomic_create.__code__.co_filename),
        "run_olmo_release_full": Path(full_runner_code_contract.__code__.co_filename),
        "run_olmo_release_smoke": Path(file_sha256.__code__.co_filename),
    }


def finalizer_code_contract() -> dict[str, Any]:
    return build_python_code_contract(
        schema=FINALIZER_CODE_SCHEMA,
        owner_role=FINALIZER_CODE_OWNER_ROLE,
        owner_module="finalize_olmo_release_full",
        owner_path=Path(__file__),
        direct_local_helpers=finalizer_helper_paths(),
    )


def validate_finalizer_code_contract(
    value: Any, *, require_live_match: bool = True
) -> dict[str, Any]:
    return validate_python_code_contract(
        value,
        schema=FINALIZER_CODE_SCHEMA,
        owner_role=FINALIZER_CODE_OWNER_ROLE,
        owner_module="finalize_olmo_release_full",
        owner_path=Path(__file__),
        direct_local_helpers=finalizer_helper_paths(),
        require_live_match=require_live_match,
    )


def verify_finalizer_code_stability(
    initial: dict[str, Any],
) -> dict[str, Any]:
    """Fail unless the finalizer and every direct local helper are unchanged."""

    validate_finalizer_code_contract(initial, require_live_match=False)
    live = finalizer_code_contract()
    if live != initial:
        raise ValueError(
            "finalizer or directly imported local helper changed during finalization"
        )
    validate_finalizer_code_contract(live, require_live_match=True)
    return live


def validate_finalizer_code_evidence(
    *,
    evidence_summary: dict[str, Any],
    evidence_completion: dict[str, Any],
    evidence_root: Path,
    require_live_match: bool = True,
) -> dict[str, dict[str, Any]]:
    """Validate the finalizer proof and its bound full-runner source contract."""

    finalizer_contract = validate_finalizer_code_contract(
        evidence_summary.get("finalizer_code_contract"),
        require_live_match=require_live_match,
    )
    if (
        evidence_summary.get("finalizer_code_sha256")
        != finalizer_contract["owner"]["sha256"]
        or evidence_summary.get("post_run_finalizer_code_verified") is not True
    ):
        raise ValueError("paired evidence finalizer code assertion drift")
    descriptor = evidence_summary.get("post_run_finalizer_code_verification")
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("paired evidence finalizer-code descriptor drift")
    expected_marker = (
        evidence_root.resolve() / "POST_RUN_FINALIZER_CODE_VERIFICATION.json"
    )
    marker_path = descriptor.get("path")
    if (
        not isinstance(marker_path, str)
        or not Path(marker_path).is_absolute()
        or Path(marker_path).resolve() != expected_marker
        or not expected_marker.is_file()
        or file_sha256(expected_marker) != descriptor.get("sha256")
    ):
        raise ValueError("paired evidence finalizer-code marker drift")
    marker = read_json(expected_marker)
    if (
        not isinstance(marker, dict)
        or set(marker)
        != {
            "schema",
            "status",
            "initial_finalizer_code_contract",
            "live_finalizer_code_contract",
            "error",
        }
        or marker.get("schema") != FINALIZER_POST_CODE_SCHEMA
        or marker.get("status") != "verified"
        or marker.get("initial_finalizer_code_contract") != finalizer_contract
        or marker.get("live_finalizer_code_contract") != finalizer_contract
        or marker.get("error") is not None
    ):
        raise ValueError("paired evidence finalizer-code marker content drift")
    runner_contract = validate_full_runner_code_contract(
        evidence_summary.get("full_runner_code_contract"),
        require_live_match=require_live_match,
    )
    if (
        evidence_completion.get("status") != "complete"
        or evidence_completion.get("finalizer_code_contract_sha256")
        != finalizer_contract["inventory_sha256"]
        or evidence_completion.get("post_run_finalizer_code_verified") is not True
        or evidence_completion.get("post_run_finalizer_code_verification_sha256")
        != descriptor["sha256"]
        or evidence_completion.get("full_runner_code_contract_sha256")
        != runner_contract["inventory_sha256"]
    ):
        raise ValueError("paired completion/finalizer code-contract binding drift")
    return {"finalizer": finalizer_contract, "full_runner": runner_contract}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def validate_release(
    run_summary: Path,
    complete_marker: Path,
    release_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_result_root = run_summary.parent.resolve()
    if (
        run_summary.resolve() != expected_result_root / "run_summary.json"
        or complete_marker.resolve() != expected_result_root / "RELEASE_COMPLETE.json"
    ):
        raise ValueError(f"release evidence path contract mismatch: {release_key}")
    payload = read_json(run_summary)
    marker = read_json(complete_marker)
    if (
        marker.get("schema") != "olmoearth-release-full-completion-v1"
        or marker.get("status") != "complete"
        or marker.get("run_summary_sha256") != file_sha256(run_summary)
    ):
        raise ValueError(f"release marker mismatch: {release_key}")
    expected_id, expected_layer = EXPECTED[release_key]
    release = payload.get("release", {})
    if (
        payload.get("status") != "complete"
        or payload.get("records") != 216
        or release.get("release_id") != expected_id
        or release.get("output_layer") != expected_layer
    ):
        raise ValueError(f"release result contract mismatch: {release_key}")
    outputs = payload.get("outputs", [])
    if len(outputs) != 216 or len({value["sample_id"] for value in outputs}) != 216:
        raise ValueError(f"release output identity/count mismatch: {release_key}")
    result_root = run_summary.parent
    preflight_path = result_root / "preflight.json"
    if file_sha256(preflight_path) != payload["preflight_sha256"]:
        raise ValueError(f"preflight drift: {release_key}")
    preflight = read_json(preflight_path)
    if (
        preflight.get("schema") != "olmoearth-release-full-preflight-v1"
        or preflight.get("status") != "ready"
    ):
        raise ValueError(f"preflight schema/status drift: {release_key}")
    try:
        validate_full_runner_code_evidence(
            preflight=preflight,
            run_summary=payload,
            completion=marker,
            result_root=result_root,
            require_live_match=True,
        )
    except ValueError as exc:
        raise ValueError(f"full-runner code evidence drift: {release_key}: {exc}") from exc
    config_path = result_root.parent / "resolved_config.yaml"
    if file_sha256(config_path) != payload["resolved_config_sha256"]:
        raise ValueError(f"resolved config drift: {release_key}")
    if file_sha256(Path(payload["log_path"])) != payload["log_sha256"]:
        raise ValueError(f"run log drift: {release_key}")
    command_contract = validate_full_release_command(
        payload.get("executed_command"),
        rslearn_entrypoint=Path(
            preflight["rslearn_runtime_fingerprint"]["entrypoint"]["path"]
        ),
        resolved_config=config_path,
    )
    if payload.get("executed_command_contract") != command_contract:
        raise ValueError(f"full release executed-command contract drift: {release_key}")
    for telemetry_key in ("selected_gpu_telemetry", "other_gpu_telemetry"):
        telemetry = payload[telemetry_key]
        if file_sha256(Path(telemetry["path"])) != telemetry["sha256"]:
            raise ValueError(f"telemetry drift: {release_key}/{telemetry_key}")
    for verified_key, evidence_key in (
        ("post_run_checkpoints_verified", "post_run_checkpoint_verification"),
        ("post_run_rslearn_runtime_verified", "post_run_rslearn_runtime_verification"),
    ):
        if payload.get(verified_key) is not True:
            raise ValueError(f"release lacks {verified_key}: {release_key}")
        evidence = payload.get(evidence_key, {})
        if file_sha256(Path(evidence["path"])) != evidence["sha256"]:
            raise ValueError(f"post-run verification drift: {release_key}/{evidence_key}")
    runtime_marker = read_json(Path(payload["post_run_rslearn_runtime_verification"]["path"]))
    if (
        runtime_marker.get("status") != "verified"
        or runtime_marker.get("rslearn_runtime_fingerprint")
        != preflight.get("rslearn_runtime_fingerprint")
    ):
        raise ValueError(f"post-run rslearn runtime marker mismatch: {release_key}")
    return payload, preflight


def referenced_inventories(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        result.extend((record["items_json"], record["window_metadata"]))
        for layer in record["input_layers"]:
            result.extend((layer["geotiff"], layer["metadata"]))
    return result


def verify_input_content(records: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    live = hash_referenced_files(records, workers=workers)
    inventories = referenced_inventories(records)
    if len(live) != 5616 or len(inventories) != 5616:
        raise ValueError("full post-run input inventory is not exactly 5,616 files")
    for inventory in inventories:
        actual = live[inventory["path"]]
        if actual["bytes"] != inventory["bytes"] or actual["sha256"] != inventory["sha256"]:
            raise ValueError(f"post-run input content drift: {inventory['path']}")
    return {
        "files": len(live),
        "bytes": sum(value["bytes"] for value in live.values()),
        "all_sha256_match_frozen_manifest": True,
    }


def verify_output_content(
    outputs: list[dict[str, Any]], workers: int
) -> dict[str, dict[str, Any]]:
    expected = {value["path"]: value for value in outputs}
    if len(expected) != len(outputs):
        raise ValueError("duplicate output paths in paired release runs")
    ordered = sorted(expected)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        actual_values = list(
            executor.map(
                lambda path: stable_inventory(Path(path), int(expected[path]["bytes"])),
                ordered,
            )
        )
    actual = {value["path"]: value for value in actual_values}
    for path, inventory in expected.items():
        if actual[path]["sha256"] != inventory["sha256"]:
            raise ValueError(f"post-run output SHA drift: {path}")
    return actual


def validate_pairs(
    v1_outputs: list[dict[str, Any]],
    v12_outputs: list[dict[str, Any]],
    exact_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    left = {value["sample_id"]: value for value in v1_outputs}
    right = {value["sample_id"]: value for value in v12_outputs}
    if set(left) != set(right) or len(left) != 216:
        raise ValueError("release outputs do not form 216 exact pairs")
    exact = {value["sample_id"]: value for value in exact_records}
    if set(left) != set(exact) or len(exact) != 216:
        raise ValueError("paired outputs do not cover the exact input sample population")
    pairs = []
    for sample_id in sorted(left):
        left_record, right_record = left[sample_id], right[sample_id]
        for field in ("window", "input_bundle_identity", "spatial_cluster_id"):
            if left_record[field] != right_record[field]:
                raise ValueError(f"paired output identity mismatch: {sample_id}/{field}")
        expected = exact[sample_id]
        expected_values = {
            "window": expected["window_name"],
            "input_bundle_identity": expected["input_bundle_identity"],
            "spatial_cluster_id": expected["spatial_cluster_id"],
        }
        for field, expected_value in expected_values.items():
            if left_record[field] != expected_value:
                raise ValueError(f"output is not bound to exact input: {sample_id}/{field}")
        window_metadata = read_json(Path(expected["window_metadata"]["path"]))
        left_health, left_validity = inspect_raster(
            Path(left_record["path"]), window_metadata, return_validity_mask=True
        )
        right_health, right_validity = inspect_raster(
            Path(right_record["path"]), window_metadata, return_validity_mask=True
        )
        structural_fields = (
            "height",
            "width",
            "count",
            "dtypes",
            "crs",
            "transform",
            "bounds",
            "nodata",
        )
        left_contract = {key: left_health[key] for key in structural_fields}
        right_contract = {key: right_health[key] for key in structural_fields}
        if left_contract != right_contract:
            raise ValueError(f"paired output raster contract mismatch: {sample_id}")
        if left_validity is None or right_validity is None:
            raise RuntimeError("paired raster inspection did not return validity masks")
        if not np.array_equal(left_validity, right_validity):
            raise ValueError(f"paired output validity-mask mismatch: {sample_id}")
        pairs.append(
            {
                "sample_id": sample_id,
                "window": left_record["window"],
                "spatial_cluster_id": left_record["spatial_cluster_id"],
                "input_bundle_identity": left_record["input_bundle_identity"],
                "v1_output": {
                    key: left_record[key] for key in ("path", "bytes", "sha256", "mtime_ns")
                },
                "v1_2_output": {
                    key: right_record[key] for key in ("path", "bytes", "sha256", "mtime_ns")
                },
                "raster_contract": left_contract,
                "v1_value_health": {
                    key: left_health[key]
                    for key in (
                        "usable_tokens",
                        "nonzero_usable_tokens",
                        "finite_values",
                        "total_values",
                        "all_values_finite",
                    )
                },
                "v1_2_value_health": {
                    key: right_health[key]
                    for key in (
                        "usable_tokens",
                        "nonzero_usable_tokens",
                        "finite_values",
                        "total_values",
                        "all_values_finite",
                    )
                },
                "validity_masks_exact": True,
                "value_health_passed_both_releases": True,
            }
        )
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--exact-complete", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-complete", type=Path, required=True)
    parser.add_argument("--v1-summary", type=Path, required=True)
    parser.add_argument("--v1-complete", type=Path, required=True)
    parser.add_argument("--v1-2-summary", type=Path, required=True)
    parser.add_argument("--v1-2-complete", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-hash-workers", type=int, default=1)
    parser.add_argument("--output-hash-workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_finalizer_code_contract = finalizer_code_contract()
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing finalization directory: {args.output_dir}")
    if not 1 <= args.input_hash_workers <= 2 or not 1 <= args.output_hash_workers <= 2:
        raise SystemExit("hash workers must be one or two to limit shared-storage pressure")
    exact_payload = read_json(args.exact_inputs)
    exact_complete = read_json(args.exact_complete)
    if exact_complete.get("exact_inputs_sha256") != file_sha256(args.exact_inputs):
        raise ValueError("exact input completion marker mismatch")
    split_complete = read_json(args.split_complete)
    if split_complete.get("split_manifest_sha256") != file_sha256(args.split_manifest):
        raise ValueError("split completion marker mismatch")
    split = read_json(args.split_manifest)
    if split.get("exact_inputs", {}).get("sha256") != file_sha256(args.exact_inputs):
        raise ValueError("split is not bound to the exact inputs")

    v1, v1_preflight = validate_release(args.v1_summary, args.v1_complete, "v1")
    v12, v12_preflight = validate_release(args.v1_2_summary, args.v1_2_complete, "v1_2")
    for field in (
        "exact_inputs_sha256",
        "exact_complete_sha256",
        "split_manifest_sha256",
        "split_complete_sha256",
        "batch_contract_sha256",
        "batch_contract_complete_sha256",
    ):
        if v1_preflight[field] != v12_preflight[field]:
            raise ValueError(f"paired preflight drift: {field}")
    current_bindings = {
        "exact_inputs_sha256": file_sha256(args.exact_inputs),
        "exact_complete_sha256": file_sha256(args.exact_complete),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "split_complete_sha256": file_sha256(args.split_complete),
    }
    for field, expected_hash in current_bindings.items():
        if v1_preflight[field] != expected_hash or v12_preflight[field] != expected_hash:
            raise ValueError(f"release preflights are not bound to finalizer argument: {field}")
    if v1_preflight["selected_gpu_uuid"] != v12_preflight["selected_gpu_uuid"]:
        raise ValueError("releases ran on different physical GPUs")
    if (
        v1_preflight.get("rslearn_runtime_fingerprint")
        != v12_preflight.get("rslearn_runtime_fingerprint")
    ):
        raise ValueError("releases ran with different rslearn executable/runtime/source")
    v1_runner_code = validate_full_runner_code_contract(
        v1_preflight.get("full_runner_code_contract"), require_live_match=True
    )
    v12_runner_code = validate_full_runner_code_contract(
        v12_preflight.get("full_runner_code_contract"), require_live_match=True
    )
    live_runner_code = full_runner_code_contract()
    if v1_runner_code != v12_runner_code or v1_runner_code != live_runner_code:
        raise ValueError(
            "paired releases do not share the current full-runner/helper code contract"
        )

    input_closure = verify_input_content(
        exact_payload["records"], workers=args.input_hash_workers
    )
    combined_outputs = v1["outputs"] + v12["outputs"]
    live_outputs = verify_output_content(combined_outputs, workers=args.output_hash_workers)
    pairs = validate_pairs(v1["outputs"], v12["outputs"], exact_payload["records"])
    live_finalizer_code_contract = verify_finalizer_code_stability(
        initial_finalizer_code_contract
    )
    args.output_dir.mkdir(parents=True)
    finalizer_code_marker = (
        args.output_dir / "POST_RUN_FINALIZER_CODE_VERIFICATION.json"
    )
    atomic_create(
        finalizer_code_marker,
        canonical_bytes(
            {
                "schema": FINALIZER_POST_CODE_SCHEMA,
                "status": "verified",
                "initial_finalizer_code_contract": initial_finalizer_code_contract,
                "live_finalizer_code_contract": live_finalizer_code_contract,
                "error": None,
            }
        ),
    )
    pair_path = args.output_dir / "paired_outputs.jsonl"
    atomic_create(pair_path, b"".join(canonical_bytes(pair) for pair in pairs))
    summary = {
        "schema": "olmoearth-release-full-paired-evidence-v1",
        "status": "complete",
        "finalizer_code_sha256": initial_finalizer_code_contract["owner"]["sha256"],
        "finalizer_code_contract": initial_finalizer_code_contract,
        "post_run_finalizer_code_verified": True,
        "post_run_finalizer_code_verification": {
            "path": finalizer_code_marker.resolve().as_posix(),
            "sha256": file_sha256(finalizer_code_marker),
        },
        "full_runner_code_contract": v1_runner_code,
        "exact_inputs_sha256": file_sha256(args.exact_inputs),
        "exact_complete_sha256": file_sha256(args.exact_complete),
        "split_manifest_sha256": file_sha256(args.split_manifest),
        "split_complete_sha256": file_sha256(args.split_complete),
        "v1_run_summary_sha256": file_sha256(args.v1_summary),
        "v1_complete_sha256": file_sha256(args.v1_complete),
        "v1_2_run_summary_sha256": file_sha256(args.v1_2_summary),
        "v1_2_complete_sha256": file_sha256(args.v1_2_complete),
        "input_post_run_closure": input_closure,
        "output_post_run_closure": {
            "files": len(live_outputs),
            "bytes": sum(value["bytes"] for value in live_outputs.values()),
            "all_sha256_match_release_manifests": True,
        },
        "paired_outputs": 216,
        "paired_outputs_jsonl_sha256": file_sha256(pair_path),
        "raster_contracts_exact": True,
        "validity_masks_exact": True,
        "value_health_passed_all_432_outputs": True,
        "selected_gpu_uuid": v1_preflight["selected_gpu_uuid"],
        "same_exact_inputs_both_releases": True,
        "claims_allowed": [
            "paired exact-input full-release representation analysis may proceed"
        ],
        "claims_forbidden": [
            "task_accuracy",
            "negative_transfer_reduction",
            "cloud_robustness",
            "korean_population_generalization",
            "model_native_backward_compatibility_before_held_out_analysis",
        ],
    }
    summary_path = args.output_dir / "evidence_summary.json"
    atomic_create(summary_path, canonical_bytes(summary))
    completion_finalizer_code_contract = verify_finalizer_code_stability(
        initial_finalizer_code_contract
    )
    if completion_finalizer_code_contract != live_finalizer_code_contract:
        raise ValueError("finalizer code contract drifted before completion sealing")
    atomic_create(
        args.output_dir / "FULL_EVIDENCE_COMPLETE.json",
        canonical_bytes(
            {
                "schema": "olmoearth-release-full-paired-evidence-completion-v1",
                "status": "complete",
                "evidence_summary_sha256": file_sha256(summary_path),
                "paired_outputs_jsonl_sha256": file_sha256(pair_path),
                "finalizer_code_contract_sha256": initial_finalizer_code_contract[
                    "inventory_sha256"
                ],
                "post_run_finalizer_code_verified": True,
                "post_run_finalizer_code_verification_sha256": file_sha256(
                    finalizer_code_marker
                ),
                "full_runner_code_contract_sha256": v1_runner_code[
                    "inventory_sha256"
                ],
            }
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
