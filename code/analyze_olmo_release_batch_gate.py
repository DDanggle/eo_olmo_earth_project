#!/usr/bin/env python3
"""Verify numerical equivalence and select a provisional safe batch setting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_olmo_release_smoke import representation_metrics
from olmo_release_semantic_contract import (
    RELEASE_SPECS,
    fingerprint_rslearn_runtime,
    normalize_checkpoint_manifest,
    validate_launcher_runtime_binding,
    validate_physical_gpu,
    validate_resolved_config,
    validate_rslearn_runtime_fingerprint,
    validate_runtime_versions,
)
from run_olmo_release_batch_gate import (
    batch_audit_code_contract,
    stable_code_file_record,
    validate_batch_audit_code_contract,
)
from run_olmo_release_smoke import validate_checkpoints, validate_inputs


REFERENCE_CANDIDATE = "b001_w02"
ANALYZER_CODE_SCHEMA = "olmoearth-release-batch-analyzer-code-contract-v1"
ANALYZER_HELPER_MODULES = {
    "analyze_olmo_release_smoke": Path(
        representation_metrics.__code__.co_filename
    ).resolve()
}


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


def batch_analyzer_code_contract() -> dict[str, Any]:
    """Bind the analyzer and its directly imported local metric implementation."""

    analyzer = stable_code_file_record(
        Path(__file__), module="analyze_olmo_release_batch_gate"
    )
    helpers = [
        stable_code_file_record(path, module=module)
        for module, path in sorted(ANALYZER_HELPER_MODULES.items())
    ]
    inventory = {"analyzer": analyzer, "direct_local_helpers": helpers}
    return {
        "schema": ANALYZER_CODE_SCHEMA,
        **inventory,
        "inventory_sha256": hashlib.sha256(canonical_bytes(inventory)).hexdigest(),
    }


def validate_batch_analyzer_code_contract(value: Any) -> dict[str, Any]:
    """Validate and canonically normalize persisted analyzer code provenance."""

    if not isinstance(value, dict) or value.get("schema") != ANALYZER_CODE_SCHEMA:
        raise ValueError("unrecognized batch analyzer code contract")
    analyzer = value.get("analyzer")
    helpers = value.get("direct_local_helpers")
    if not isinstance(analyzer, dict) or not isinstance(helpers, list):
        raise ValueError("batch analyzer code contract lacks source inventory")
    if analyzer.get("module") != "analyze_olmo_release_batch_gate":
        raise ValueError("batch analyzer module identity drift")
    if [record.get("module") for record in helpers] != sorted(
        ANALYZER_HELPER_MODULES
    ):
        raise ValueError("batch analyzer direct-helper inventory drift")
    records = [analyzer, *helpers]
    for record in records:
        path, size, digest = (
            record.get("path"),
            record.get("bytes"),
            record.get("sha256"),
        )
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("batch analyzer code path must be absolute")
        if not isinstance(size, int) or size <= 0:
            raise ValueError("batch analyzer code byte count is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("batch analyzer code SHA-256 is invalid")
    normalized_inventory = {
        "analyzer": dict(analyzer),
        "direct_local_helpers": [dict(record) for record in helpers],
    }
    expected_sha = hashlib.sha256(canonical_bytes(normalized_inventory)).hexdigest()
    if value.get("inventory_sha256") != expected_sha:
        raise ValueError("batch analyzer code inventory digest mismatch")
    return {
        "schema": ANALYZER_CODE_SCHEMA,
        **normalized_inventory,
        "inventory_sha256": expected_sha,
    }


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def resolve_evidence_path(value: str | Path, evidence_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else evidence_root / path


def validate_execution_audit_code(
    payload: dict[str, Any],
    completion_marker: dict[str, Any],
    preflight: dict[str, Any],
    evidence_root: Path,
) -> dict[str, Any]:
    """Close preflight -> post-run -> aggregate marker audit-code provenance."""

    persisted = validate_batch_audit_code_contract(
        preflight.get("audit_code_contract")
    )
    if payload.get("audit_code_contract") != persisted:
        raise ValueError("batch run summary audit code differs from preflight")
    if payload.get("post_run_audit_code_verified") is not True:
        raise ValueError("batch run lacks post-run audit code verification")
    if completion_marker.get("post_run_audit_code_verified") is not True:
        raise ValueError("batch completion marker lacks audit code verification")
    if completion_marker.get("audit_code_contract_sha256") != persisted[
        "inventory_sha256"
    ]:
        raise ValueError("batch completion marker audit code digest mismatch")

    evidence = payload.get("post_run_audit_code_verification")
    if not isinstance(evidence, dict):
        raise ValueError("batch run lacks post-run audit code evidence")
    evidence_path_value = evidence.get("path")
    if not isinstance(evidence_path_value, str) or not evidence_path_value:
        raise ValueError("post-run audit code evidence path is invalid")
    evidence_path = resolve_evidence_path(evidence_path_value, evidence_root)
    if not evidence_path.is_file() or file_sha256(evidence_path) != evidence.get("sha256"):
        raise ValueError("post-run audit code evidence drift")
    if completion_marker.get("post_run_audit_code_verification_sha256") != evidence.get(
        "sha256"
    ):
        raise ValueError("completion marker does not bind post-run audit code evidence")
    post_run = read_json(evidence_path)
    if (
        post_run.get("schema")
        != "olmoearth-release-batch-post-run-audit-code-verification-v1"
        or post_run.get("status") != "verified"
        or post_run.get("initial_audit_code_contract") != persisted
        or post_run.get("live_audit_code_contract") != persisted
        or post_run.get("error") is not None
    ):
        raise ValueError("post-run audit code marker contract mismatch")
    live = batch_audit_code_contract()
    if live != persisted:
        raise ValueError("batch runner/direct-helper code drifted after execution")
    return persisted


def validate_execution_checkpoint_evidence(
    payload: dict[str, Any],
    completion_marker: dict[str, Any],
    preflight: dict[str, Any],
    checkpoint_manifest: Path,
    evidence_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a post-run rehash of both immutable release checkpoints."""

    if payload.get("post_run_checkpoints_verified") is not True:
        raise ValueError("batch run lacks post-run checkpoint verification")
    if completion_marker.get("post_run_checkpoints_verified") is not True:
        raise ValueError("batch completion marker lacks checkpoint verification")
    expected_manifest_sha = preflight.get("checkpoint_manifest_sha256")
    if (
        completion_marker.get("checkpoint_manifest_sha256")
        != expected_manifest_sha
    ):
        raise ValueError("batch completion checkpoint manifest digest mismatch")
    evidence = payload.get("post_run_checkpoint_verification")
    if not isinstance(evidence, dict):
        raise ValueError("batch run lacks post-run checkpoint evidence")
    evidence_path_value = evidence.get("path")
    if not isinstance(evidence_path_value, str) or not evidence_path_value:
        raise ValueError("post-run checkpoint evidence path is invalid")
    evidence_path = resolve_evidence_path(evidence_path_value, evidence_root)
    if not evidence_path.is_file() or file_sha256(evidence_path) != evidence.get("sha256"):
        raise ValueError("post-run checkpoint evidence drift")
    if completion_marker.get("post_run_checkpoint_verification_sha256") != evidence.get(
        "sha256"
    ):
        raise ValueError("completion marker does not bind post-run checkpoint evidence")
    marker = read_json(evidence_path)
    resolved_manifest = checkpoint_manifest.resolve()
    if (
        marker.get("schema")
        != "olmoearth-release-batch-post-run-checkpoint-verification-v1"
        or marker.get("status") != "verified"
        or marker.get("checkpoint_manifest_path") != resolved_manifest.as_posix()
        or marker.get("initial_checkpoint_manifest_sha256")
        != expected_manifest_sha
        or marker.get("live_checkpoint_manifest_sha256") != expected_manifest_sha
        or marker.get("initial_checkpoint_models")
        != marker.get("live_checkpoint_models")
        or marker.get("error") is not None
    ):
        raise ValueError("post-run checkpoint marker contract mismatch")
    if file_sha256(resolved_manifest) != expected_manifest_sha:
        raise ValueError("checkpoint manifest drifted after batch execution")
    validate_checkpoints(resolved_manifest)
    live_models = normalize_checkpoint_manifest(read_json(resolved_manifest))
    if live_models != marker.get("live_checkpoint_models"):
        raise ValueError("checkpoint files drifted after post-run verification")
    return live_models, {
        "path": evidence_path.as_posix(),
        "sha256": evidence["sha256"],
        "checkpoint_manifest_path": resolved_manifest.as_posix(),
        "checkpoint_manifest_sha256": expected_manifest_sha,
        "checkpoint_models": live_models,
        "status": "verified",
    }


def read_embedding(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - server integration dependency
        raise RuntimeError("rasterio is required for batch equivalence") from exc
    with rasterio.open(path) as dataset:
        # Keep the reusable eight-window reference cache in native float32
        # (~1.6 GiB rather than ~3.2 GiB). Numeric comparisons promote one
        # bounded token chunk at a time to float64 below.
        array = dataset.read(masked=True, out_dtype="float32")
        metadata = {
            "shape": [dataset.height, dataset.width],
            "bands": dataset.count,
            "dtypes": list(dataset.dtypes),
            "crs": dataset.crs.to_string() if dataset.crs else None,
            "transform": list(dataset.transform)[:6],
            "bounds": list(dataset.bounds),
            "nodata": dataset.nodata,
        }
    return np.moveaxis(np.ma.filled(array, np.nan), 0, -1), metadata


def validate_executed_command(
    candidate: dict[str, Any],
    config_path: Path,
    rslearn_runtime_fingerprint: dict[str, Any],
) -> list[str]:
    expected = [
        rslearn_runtime_fingerprint["entrypoint"]["path"],
        "model",
        "predict",
        "--config",
        config_path.as_posix(),
    ]
    actual = candidate.get("executed_command")
    if actual != expected:
        raise ValueError(
            f"candidate executed command drift: {candidate.get('candidate_id')}: "
            f"{actual!r} != {expected!r}"
        )
    return list(actual)


def chunked_numeric_comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    chunk_size: int = 2048,
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("numeric comparison requires equal token×feature matrices")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("numeric comparison received non-finite valid values")
    maximum_absolute_error = 0.0
    allclose = True
    minimum_nonzero_cosine = 1.0
    compared_nonzero_tokens = 0
    for start in range(0, reference.shape[0], chunk_size):
        left = reference[start : start + chunk_size].astype(np.float64, copy=False)
        right = candidate[start : start + chunk_size].astype(np.float64, copy=False)
        absolute_error = np.abs(left - right)
        maximum_absolute_error = max(
            maximum_absolute_error, float(np.max(absolute_error, initial=0.0))
        )
        tolerance = absolute_tolerance + relative_tolerance * np.abs(left)
        allclose = allclose and bool(np.all(absolute_error <= tolerance))
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        nonzero = (left_norm > 0) & (right_norm > 0)
        if np.any(nonzero):
            cosines = np.sum(left[nonzero] * right[nonzero], axis=1) / (
                left_norm[nonzero] * right_norm[nonzero]
            )
            minimum_nonzero_cosine = min(
                minimum_nonzero_cosine, float(np.min(cosines))
            )
            compared_nonzero_tokens += int(nonzero.sum())
    return {
        "allclose": allclose,
        "maximum_absolute_error": maximum_absolute_error,
        "minimum_nonzero_token_cosine": minimum_nonzero_cosine,
        "compared_nonzero_tokens": compared_nonzero_tokens,
    }


def validate_candidate_evidence(
    candidate: dict[str, Any],
    evidence_root: Path,
    exact_inputs: Path,
    *,
    model_env: str,
    output_layer: str,
    rslearn_runtime_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    if candidate.get("status") != "pass_execution":
        raise ValueError(f"candidate did not pass execution: {candidate['candidate_id']}")
    result_root = resolve_evidence_path(candidate["log_path"], evidence_root).parent
    summary_path = result_root / "candidate_summary.json"
    marker_path = result_root / "CANDIDATE_COMPLETE.json"
    marker = read_json(marker_path)
    if (
        marker.get("schema") != "olmoearth-release-batch-candidate-completion-v1"
        or marker.get("candidate_summary_sha256") != file_sha256(summary_path)
    ):
        raise ValueError(f"candidate completion marker mismatch: {candidate['candidate_id']}")
    persisted = read_json(summary_path)
    if persisted != candidate:
        raise ValueError(f"aggregate summary differs from candidate evidence: {candidate['candidate_id']}")
    config_path = resolve_evidence_path(candidate["config_path"], evidence_root)
    log_path = resolve_evidence_path(candidate["log_path"], evidence_root)
    telemetry_path = resolve_evidence_path(candidate["telemetry_path"], evidence_root)
    if file_sha256(config_path) != candidate["config_sha256"]:
        raise ValueError(f"candidate config drift: {candidate['candidate_id']}")
    if file_sha256(log_path) != candidate["log_sha256"]:
        raise ValueError(f"candidate log drift: {candidate['candidate_id']}")
    if file_sha256(telemetry_path) != candidate["telemetry_sha256"]:
        raise ValueError(f"candidate telemetry drift: {candidate['candidate_id']}")
    validate_executed_command(candidate, config_path, rslearn_runtime_fingerprint)
    config_contract = validate_resolved_config(
        config_path,
        model_env=model_env,
        output_layer=output_layer,
        batch_size=int(candidate["batch_size"]),
        num_workers=int(candidate["num_workers"]),
    )
    dataset_root = resolve_evidence_path(
        candidate["source_view"]["output_dataset"], evidence_root
    )
    view_manifest = dataset_root / "audit_view_manifest.json"
    view_config = dataset_root / "config.json"
    if read_json(view_manifest) != candidate["source_view"]:
        raise ValueError(f"candidate view manifest drift: {candidate['candidate_id']}")
    if not view_config.is_file():
        raise ValueError(f"candidate view config missing: {candidate['candidate_id']}")
    if candidate.get("source_view_manifest_sha256") is not None and file_sha256(
        view_manifest
    ) != candidate["source_view_manifest_sha256"]:
        raise ValueError(f"candidate view manifest SHA drift: {candidate['candidate_id']}")
    if candidate.get("source_view_config_sha256") is not None and file_sha256(
        view_config
    ) != candidate["source_view_config_sha256"]:
        raise ValueError(f"candidate view config SHA drift: {candidate['candidate_id']}")
    validate_inputs(exact_inputs, dataset_root)
    for output in candidate["outputs"]:
        path = resolve_evidence_path(output["path"], evidence_root)
        if path.stat().st_size != output["bytes"] or file_sha256(path) != output["sha256"]:
            raise ValueError(f"candidate output drift: {path}")
    return config_contract


def compare_candidate(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
    evidence_root: Path,
    reference_cache: dict[str, tuple[np.ndarray, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    reference_outputs = {value["sample_id"]: value for value in reference["outputs"]}
    candidate_outputs = {value["sample_id"]: value for value in candidate["outputs"]}
    if set(reference_outputs) != set(candidate_outputs) or len(reference_outputs) != 8:
        raise ValueError("candidate output identities differ from the eight-window reference")

    pooled_reference, pooled_candidate, windows = [], [], []
    for sample_id in sorted(reference_outputs):
        left_record, right_record = reference_outputs[sample_id], candidate_outputs[sample_id]
        for identity in ("window", "input_bundle_identity", "spatial_cluster_id"):
            if left_record[identity] != right_record[identity]:
                raise ValueError(f"candidate identity mismatch for {sample_id}/{identity}")
        if reference_cache is None:
            left_grid, left_metadata = read_embedding(
                resolve_evidence_path(left_record["path"], evidence_root)
            )
        else:
            left_grid, left_metadata = reference_cache[sample_id]
        if candidate["candidate_id"] == reference["candidate_id"]:
            right_grid, right_metadata = left_grid, left_metadata
        else:
            right_grid, right_metadata = read_embedding(
                resolve_evidence_path(right_record["path"], evidence_root)
            )
        metadata_equal = left_metadata == right_metadata
        finite_mask_equal = np.array_equal(np.isfinite(left_grid), np.isfinite(right_grid))
        if not metadata_equal or not finite_mask_equal:
            raise ValueError(
                f"grid/mask equivalence failed for {candidate['candidate_id']}/{sample_id}: "
                f"metadata={metadata_equal}, finite_mask={finite_mask_equal}"
            )
        token_valid = np.isfinite(left_grid).all(axis=2)
        left = left_grid[token_valid]
        right = right_grid[token_valid]
        if left.shape[0] < 2:
            raise ValueError(
                f"fewer than two valid tokens for {candidate['candidate_id']}/{sample_id}"
            )
        numeric = chunked_numeric_comparison(
            left,
            right,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        left_pooled = left.mean(axis=0, dtype=np.float64)
        right_pooled = right.mean(axis=0, dtype=np.float64)
        denominator = np.linalg.norm(left_pooled)
        pooled_relative_l2 = float(
            np.linalg.norm(left_pooled - right_pooled) / max(denominator, 1e-12)
        )
        pooled_reference.append(left_pooled)
        pooled_candidate.append(right_pooled)
        windows.append(
            {
                "sample_id": sample_id,
                "metadata_equal": True,
                "finite_mask_equal": True,
                "valid_tokens": int(left.shape[0]),
                **numeric,
                "pooled_relative_l2_error": pooled_relative_l2,
            }
        )
        if numeric["compared_nonzero_tokens"] < 2:
            raise ValueError(
                f"fewer than two non-zero token pairs for {candidate['candidate_id']}/{sample_id}"
            )

    geometry = None
    if len(pooled_reference) == 8:
        geometry = representation_metrics(
            np.stack(pooled_reference),
            np.stack(pooled_candidate),
            neighbor_ks=(1, 2),
        )
    gate = {
        "metadata_exact_all_windows": all(value["metadata_equal"] for value in windows),
        "finite_mask_exact_all_windows": all(value["finite_mask_equal"] for value in windows),
        "allclose_all_windows": all(value["allclose"] for value in windows),
        "maximum_absolute_error_at_most_1e_4": max(
            value["maximum_absolute_error"] for value in windows
        )
        <= 1e-4,
        "minimum_token_cosine_at_least_0_999999": min(
            value["minimum_nonzero_token_cosine"] for value in windows
        )
        >= 0.999999,
        "maximum_pooled_relative_l2_at_most_1e_5": max(
            value["pooled_relative_l2_error"] for value in windows
        )
        <= 1e-5,
        "pooled_distance_spearman_is_one": geometry is not None
        and abs(geometry["pairwise_euclidean_distance_spearman"] - 1.0) <= 1e-12,
        "pooled_neighbor_top1_top2_exact": geometry is not None
        and all(
            geometry["neighbor_overlap"][str(k)]["mean_fraction"] == 1.0
            for k in (1, 2)
        ),
        "pooled_linear_cka_at_least_0_999999": geometry is not None
        and geometry["pooled_linear_cka"] >= 0.999999,
    }
    return {
        "candidate_id": candidate["candidate_id"],
        "batch_size": candidate["batch_size"],
        "num_workers": candidate["num_workers"],
        "wall_seconds": candidate["wall_seconds"],
        "end_to_end_crops_per_second": candidate["end_to_end_crops_per_second"],
        "telemetry_summary": candidate["telemetry_summary"],
        "windows": windows,
        "pooled_geometry": geometry,
        "equivalence_gate": gate,
        "equivalent_to_batch1": all(gate.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument("--execution-complete", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer_code_contract = batch_analyzer_code_contract()
    evidence_root = args.evidence_root.resolve()
    payload = read_json(args.run_summary)
    marker = read_json(args.execution_complete)
    if (
        payload.get("schema") != "olmoearth-release-batch-gate-run-v1"
        or payload.get("status") != "execution_complete_analysis_pending"
    ):
        raise ValueError("unrecognized or incomplete batch-gate run summary")
    if (
        marker.get("schema")
        != "olmoearth-release-batch-gate-execution-completion-v1"
        or marker.get("status") != "complete"
        or marker.get("post_run_rslearn_runtime_verified") is not True
        or marker.get("post_run_checkpoints_verified") is not True
        or marker.get("post_run_audit_code_verified") is not True
        or marker.get("run_summary_sha256") != file_sha256(args.run_summary)
    ):
        raise ValueError("execution completion marker does not match the run summary")
    if payload.get("post_run_rslearn_runtime_verified") is not True:
        raise ValueError("batch run lacks post-run rslearn runtime verification")
    if payload.get("post_run_checkpoints_verified") is not True:
        raise ValueError("batch run lacks post-run checkpoint verification")
    if payload.get("post_run_audit_code_verified") is not True:
        raise ValueError("batch run lacks post-run audit code verification")
    preflight_path = args.run_summary.parent / "preflight.json"
    if file_sha256(preflight_path) != payload.get("preflight_sha256"):
        raise ValueError("batch-gate preflight drift")
    preflight = read_json(preflight_path)
    persisted_audit_code = validate_execution_audit_code(
        payload, marker, preflight, evidence_root
    )
    if preflight.get("exact_inputs_sha256") != file_sha256(args.exact_inputs):
        raise ValueError("batch-gate exact input evidence drift")
    if preflight.get("checkpoint_manifest_sha256") != file_sha256(args.checkpoint_manifest):
        raise ValueError("batch-gate checkpoint manifest drift")
    if preflight.get("template_config_sha256") != file_sha256(args.template_config):
        raise ValueError("batch-gate template config drift")
    checkpoint_models, checkpoint_post_run_verification = (
        validate_execution_checkpoint_evidence(
            payload,
            marker,
            preflight,
            args.checkpoint_manifest,
            evidence_root,
        )
    )
    repo_id = payload.get("repo_id")
    if repo_id not in RELEASE_SPECS:
        raise ValueError(f"unexpected release repository: {repo_id!r}")
    release_spec = RELEASE_SPECS[repo_id]
    if payload.get("release_id") != release_spec["release_id"]:
        raise ValueError("batch-gate release ID/repository binding drift")
    release_checkpoint = checkpoint_models[repo_id]
    if payload.get("revision") != release_checkpoint["revision"]:
        raise ValueError("batch-gate revision differs from the hashed checkpoint manifest")
    physical_gpu = validate_physical_gpu(
        preflight.get("selected_gpu"), preflight.get("selected_gpu_uuid")
    )
    if (
        str(payload.get("selected_gpu")) != physical_gpu["index"]
        or payload.get("selected_gpu_uuid") != physical_gpu["uuid"]
    ):
        raise ValueError("run summary physical GPU differs from preflight")
    runtime = validate_runtime_versions(preflight.get("runtime_versions"))
    persisted_rslearn_runtime = validate_rslearn_runtime_fingerprint(
        preflight.get("rslearn_runtime_fingerprint")
    )
    validate_launcher_runtime_binding(
        runtime,
        Path(persisted_rslearn_runtime["interpreter"]["path"]),
        persisted_rslearn_runtime,
    )
    live_rslearn_runtime = fingerprint_rslearn_runtime(
        Path(persisted_rslearn_runtime["entrypoint"]["path"])
    )
    if live_rslearn_runtime != persisted_rslearn_runtime:
        raise ValueError(
            "rslearn executable/interpreter/package source drifted after batch execution"
        )
    candidates = {value["candidate_id"]: value for value in payload["candidates"]}
    if len(candidates) != len(payload["candidates"]):
        raise ValueError("duplicate candidate IDs in batch-gate run")
    if REFERENCE_CANDIDATE not in candidates:
        raise ValueError(f"missing reference candidate {REFERENCE_CANDIDATE}")
    config_contracts = {
        candidate_id: validate_candidate_evidence(
            candidate,
            evidence_root,
            args.exact_inputs,
            model_env=release_spec["model_env"],
            output_layer=release_spec["batch_output_layer"],
            rslearn_runtime_fingerprint=persisted_rslearn_runtime,
        )
        for candidate_id, candidate in candidates.items()
    }
    semantic_cores = {
        canonical_bytes(contract["semantic_core"]) for contract in config_contracts.values()
    }
    if len(semantic_cores) != 1:
        raise ValueError("candidate resolved configs do not share one semantic core")
    reference = candidates[REFERENCE_CANDIDATE]
    reference_outputs = {value["sample_id"]: value for value in reference["outputs"]}
    if len(reference_outputs) != 8:
        raise ValueError("reference candidate does not contain exactly eight unique outputs")
    reference_cache = {
        sample_id: read_embedding(resolve_evidence_path(output["path"], evidence_root))
        for sample_id, output in sorted(reference_outputs.items())
    }
    comparisons = [
        compare_candidate(
            reference,
            candidate,
            relative_tolerance=args.relative_tolerance,
            absolute_tolerance=args.absolute_tolerance,
            evidence_root=evidence_root,
            reference_cache=reference_cache,
        )
        for candidate in payload["candidates"]
    ]
    passing = [value for value in comparisons if value["equivalent_to_batch1"]]
    if not passing:
        raise ValueError("even the batch1 self-comparison failed the equivalence contract")
    fastest = max(value["end_to_end_crops_per_second"] for value in passing)
    within_five_percent = [
        value for value in passing if value["end_to_end_crops_per_second"] >= fastest * 0.95
    ]
    provisional = min(
        within_five_percent,
        key=lambda value: (value["batch_size"], value["num_workers"]),
    )
    summary = {
        "schema": "olmoearth-release-batch-equivalence-analysis-v1",
        "status": "complete",
        "analysis_code_contract": analyzer_code_contract,
        "run_summary_sha256": file_sha256(args.run_summary),
        "execution_complete_sha256": file_sha256(args.execution_complete),
        "run_provenance": {
            "release_id": payload["release_id"],
            "repo_id": payload["repo_id"],
            "revision": payload["revision"],
            "release_checkpoint": release_checkpoint,
            "selected_gpu": physical_gpu["index"],
            "selected_gpu_uuid": physical_gpu["uuid"],
            "runtime_versions": runtime,
            "rslearn_runtime_fingerprint": persisted_rslearn_runtime,
            "batch_audit_code_contract": persisted_audit_code,
            "post_run_checkpoint_verification": checkpoint_post_run_verification,
            "exact_inputs_sha256": preflight["exact_inputs_sha256"],
            "checkpoint_manifest_sha256": preflight["checkpoint_manifest_sha256"],
            "template_config_sha256": preflight["template_config_sha256"],
            "semantic_config_core": next(iter(config_contracts.values()))[
                "semantic_core"
            ],
            "candidate_execution_configs": {
                candidate_id: contract["execution_binding"]
                for candidate_id, contract in sorted(config_contracts.items())
            },
        },
        "reference_candidate": REFERENCE_CANDIDATE,
        "relative_tolerance": args.relative_tolerance,
        "absolute_tolerance": args.absolute_tolerance,
        "comparisons": comparisons,
        "provisional_safe_candidate": {
            "candidate_id": provisional["candidate_id"],
            "batch_size": provisional["batch_size"],
            "num_workers": provisional["num_workers"],
            "end_to_end_crops_per_second": provisional["end_to_end_crops_per_second"],
        },
        "selection_rule": "smallest batch/workers within 5% of fastest numerically equivalent candidate",
        "promotion_pending": [
            "worker ladder for the provisional batch",
            "reverse-order finalist repeats",
            "final batch1 anchor",
            "both-release eight-window equivalence",
        ],
        "claims_allowed": [
            "one-shot v1.2 batch numerical equivalence",
            "one-shot end-to-end throughput comparison",
        ],
        "claims_forbidden": [
            "final throughput stability",
            "full-run batch promotion",
            "task accuracy",
            "release compatibility",
        ],
    }
    if (
        batch_analyzer_code_contract() != analyzer_code_contract
    ):
        raise ValueError("batch analyzer/helper source changed during analysis")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_dir / "analysis_summary.json"
    summary_path.write_bytes(canonical_bytes(summary))
    with (args.output_dir / "per_candidate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        fields = [
            "candidate_id",
            "batch_size",
            "num_workers",
            "wall_seconds",
            "end_to_end_crops_per_second",
            "equivalent_to_batch1",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for value in comparisons:
            writer.writerow({field: value[field] for field in fields})
    completion = {
        "schema": "olmoearth-release-batch-equivalence-completion-v1",
        "analysis_summary_sha256": file_sha256(summary_path),
        "per_candidate_csv_sha256": file_sha256(args.output_dir / "per_candidate.csv"),
        "analysis_code_contract_sha256": analyzer_code_contract[
            "inventory_sha256"
        ],
        "batch_audit_code_contract_sha256": persisted_audit_code[
            "inventory_sha256"
        ],
        "post_run_checkpoint_verification_sha256": checkpoint_post_run_verification[
            "sha256"
        ],
    }
    (args.output_dir / "ANALYSIS_COMPLETE.json").write_bytes(canonical_bytes(completion))
    print(json.dumps(summary["provisional_safe_candidate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
