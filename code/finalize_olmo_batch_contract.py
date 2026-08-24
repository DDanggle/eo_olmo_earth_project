#!/usr/bin/env python3
"""Promote one batch setting only after repeated two-release equivalence gates."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_olmo_release_batch_gate import (
    batch_analyzer_code_contract,
    validate_batch_analyzer_code_contract,
)
from olmo_release_semantic_contract import (
    RELEASE_SPECS,
    validate_physical_gpu,
    validate_rslearn_runtime_fingerprint,
)
from run_olmo_release_batch_gate import (
    atomic_create,
    batch_audit_code_contract,
    canonical_bytes,
    validate_batch_audit_code_contract,
)
from run_olmo_release_smoke import file_sha256


REFERENCE_CANDIDATE = "b001_w02"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def validate_analysis(summary_path: Path, complete_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    complete = read_json(complete_path)
    if (
        summary.get("schema") != "olmoearth-release-batch-equivalence-analysis-v1"
        or summary.get("status") != "complete"
    ):
        raise ValueError(f"invalid batch analysis: {summary_path}")
    if (
        complete.get("schema")
        != "olmoearth-release-batch-equivalence-completion-v1"
        or complete.get("analysis_summary_sha256") != file_sha256(summary_path)
    ):
        raise ValueError(f"batch analysis completion mismatch: {summary_path}")
    persisted_code = validate_batch_audit_code_contract(
        summary.get("run_provenance", {}).get("batch_audit_code_contract")
    )
    live_code = batch_audit_code_contract()
    if persisted_code != live_code:
        raise ValueError(f"batch runner/helper code drift: {summary_path}")
    analyzer_code = validate_batch_analyzer_code_contract(
        summary.get("analysis_code_contract")
    )
    live_analyzer_code = batch_analyzer_code_contract()
    if analyzer_code != live_analyzer_code:
        raise ValueError(f"batch analyzer/helper code drift: {summary_path}")
    if complete.get("analysis_code_contract_sha256") != live_analyzer_code[
        "inventory_sha256"
    ]:
        raise ValueError(f"batch completion analyzer/helper SHA mismatch: {summary_path}")
    if complete.get("batch_audit_code_contract_sha256") != live_code[
        "inventory_sha256"
    ]:
        raise ValueError(f"batch completion runner/helper SHA mismatch: {summary_path}")
    checkpoint_evidence = summary.get("run_provenance", {}).get(
        "post_run_checkpoint_verification"
    )
    if not isinstance(checkpoint_evidence, dict):
        raise ValueError(f"batch analysis lacks checkpoint closure: {summary_path}")
    checkpoint_path_value = checkpoint_evidence.get("path")
    if not isinstance(checkpoint_path_value, str) or not checkpoint_path_value:
        raise ValueError(f"batch checkpoint evidence path is invalid: {summary_path}")
    checkpoint_path = Path(checkpoint_path_value)
    if (
        not checkpoint_path.is_file()
        or file_sha256(checkpoint_path) != checkpoint_evidence.get("sha256")
        or complete.get("post_run_checkpoint_verification_sha256")
        != checkpoint_evidence.get("sha256")
    ):
        raise ValueError(f"batch checkpoint evidence drift: {summary_path}")
    checkpoint_marker = read_json(checkpoint_path)
    if (
        checkpoint_evidence.get("status") != "verified"
        or checkpoint_marker.get("schema")
        != "olmoearth-release-batch-post-run-checkpoint-verification-v1"
        or checkpoint_marker.get("status") != "verified"
        or checkpoint_marker.get("checkpoint_manifest_path")
        != checkpoint_evidence.get("checkpoint_manifest_path")
        or checkpoint_marker.get("initial_checkpoint_manifest_sha256")
        != checkpoint_evidence.get("checkpoint_manifest_sha256")
        or checkpoint_marker.get("live_checkpoint_manifest_sha256")
        != checkpoint_evidence.get("checkpoint_manifest_sha256")
        or checkpoint_marker.get("initial_checkpoint_models")
        != checkpoint_evidence.get("checkpoint_models")
        or checkpoint_marker.get("live_checkpoint_models")
        != checkpoint_evidence.get("checkpoint_models")
        or checkpoint_marker.get("error") is not None
    ):
        raise ValueError(f"batch checkpoint closure mismatch: {summary_path}")
    return summary


def validate_promotion_code_contracts(
    values: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require all five analyses to bind the same currently installed code."""

    live_batch_code = batch_audit_code_contract()
    live_analyzer_code = batch_analyzer_code_contract()
    persisted_batch_contracts = {
        canonical_bytes(
            validate_batch_audit_code_contract(
                value.get("run_provenance", {}).get("batch_audit_code_contract")
            )
        )
        for value in values
    }
    persisted_analyzers = {
        canonical_bytes(
            validate_batch_analyzer_code_contract(
                value.get("analysis_code_contract")
            )
        )
        for value in values
    }
    if len(persisted_batch_contracts) != 1:
        raise ValueError("batch promotion runs do not share one runner/helper code contract")
    if len(persisted_analyzers) != 1:
        raise ValueError("batch promotion runs do not share one analyzer/helper code contract")
    if next(iter(persisted_batch_contracts)) != canonical_bytes(live_batch_code):
        raise ValueError("persisted batch runner/helper code differs from current files")
    if next(iter(persisted_analyzers)) != canonical_bytes(live_analyzer_code):
        raise ValueError("persisted batch analyzer/helper code differs from current files")
    return {
        "batch_runner_and_direct_helpers": live_batch_code,
        "batch_analyzer": live_analyzer_code,
    }


def comparison(summary: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    values = [
        value for value in summary["comparisons"] if value["candidate_id"] == candidate_id
    ]
    if len(values) != 1:
        raise ValueError(f"expected one comparison for {candidate_id}")
    return values[0]


def evidence(role: str, path: Path) -> dict[str, Any]:
    return {"role": role, "path": path.as_posix(), "sha256": file_sha256(path)}


def validate_run_roles(
    ladder: dict[str, Any],
    worker: dict[str, Any],
    repeats: list[dict[str, Any]],
    v1: dict[str, Any],
) -> dict[str, Any]:
    v12_values = [ladder, worker, *repeats]
    if any(
        value.get("run_provenance", {}).get("release_id") != "olmoearth_v1_2_base"
        or value.get("run_provenance", {}).get("repo_id")
        != "allenai/OlmoEarth-v1_2-Base"
        for value in v12_values
    ):
        raise ValueError("ladder/worker/repeats must all be v1.2 runs")
    if (
        v1.get("run_provenance", {}).get("release_id") != "olmoearth_v1_base"
        or v1.get("run_provenance", {}).get("repo_id") != "allenai/OlmoEarth-v1-Base"
    ):
        raise ValueError("v1 analysis role is not an OlmoEarth v1 run")
    values = [*v12_values, v1]
    run_hashes = [value["run_summary_sha256"] for value in values]
    if len(set(run_hashes)) != len(run_hashes):
        raise ValueError("batch promotion requires five independent run summaries")
    exact_hashes = {
        value["run_provenance"]["exact_inputs_sha256"] for value in values
    }
    checkpoint_hashes = {
        value["run_provenance"]["checkpoint_manifest_sha256"] for value in values
    }
    gpu_uuids = {value["run_provenance"]["selected_gpu_uuid"] for value in values}
    gpu_indices = {value["run_provenance"]["selected_gpu"] for value in values}
    if len(exact_hashes) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("batch promotion runs do not share exact inputs/checkpoints")
    if gpu_indices != {"0"} or len(gpu_uuids) != 1:
        raise ValueError("batch promotion runs were not isolated to one physical GPU0")
    validate_physical_gpu(next(iter(gpu_indices)), next(iter(gpu_uuids)))

    v12_template_hashes = {
        value["run_provenance"]["template_config_sha256"] for value in v12_values
    }
    if len(v12_template_hashes) != 1:
        raise ValueError("all four v1.2 runs must use the same immutable template YAML")
    semantic_cores = {
        canonical_bytes(value["run_provenance"]["semantic_config_core"])
        for value in values
    }
    runtimes = {
        canonical_bytes(value["run_provenance"]["runtime_versions"])
        for value in values
    }
    rslearn_fingerprints = {
        canonical_bytes(
            validate_rslearn_runtime_fingerprint(
                value["run_provenance"].get("rslearn_runtime_fingerprint")
            )
        )
        for value in values
    }
    if len(semantic_cores) != 1:
        raise ValueError("batch promotion runs do not share one semantic model/input contract")
    if len(runtimes) != 1:
        raise ValueError("batch promotion runs do not share one runtime version contract")
    if len(rslearn_fingerprints) != 1:
        raise ValueError(
            "batch promotion runs do not share one rslearn executable/runtime/source"
        )
    promotion_code = validate_promotion_code_contracts(values)

    for value in values:
        provenance = value["run_provenance"]
        repo_id = provenance["repo_id"]
        checkpoint = provenance["release_checkpoint"]
        spec = RELEASE_SPECS[repo_id]
        if (
            checkpoint["repo_id"] != repo_id
            or checkpoint["release_id"] != provenance["release_id"]
            or checkpoint["revision"] != provenance["revision"]
            or checkpoint["model_path_environment"] != spec["model_env"]
        ):
            raise ValueError(f"release/checkpoint binding drift in {repo_id}")
        configs = provenance.get("candidate_execution_configs", {})
        if not configs:
            raise ValueError("batch analysis has no validated resolved-config contracts")
        for candidate_id, binding in configs.items():
            if (
                binding.get("model_path_environment") != spec["model_env"]
                or binding.get("output_layer") != spec["batch_output_layer"]
                or binding.get("dataset_path_environment") != "DATASET_PATH"
                or not isinstance(binding.get("batch_size"), int)
                or not isinstance(binding.get("num_workers"), int)
            ):
                raise ValueError(f"candidate semantic binding drift: {candidate_id}")
    return promotion_code


def build_execution_contract(
    ladder: dict[str, Any],
    worker: dict[str, Any],
    repeats: list[dict[str, Any]],
    v1: dict[str, Any],
    *,
    candidate_id: str,
    batch_size: int,
    workers: int,
    promotion_code: dict[str, Any],
) -> dict[str, Any]:
    """Build the strict contract that a 216-window runner must reproduce."""

    values = [ladder, worker, *repeats, v1]
    v12_values = [ladder, worker, *repeats]
    provenance = [value["run_provenance"] for value in values]
    repeat_orders = [
        [comparison_value["candidate_id"] for comparison_value in value["comparisons"]]
        for value in repeats
    ]
    expected_repeat_candidates = {REFERENCE_CANDIDATE, candidate_id}
    if any(
        len(order) != 2 or set(order) != expected_repeat_candidates
        for order in repeat_orders
    ):
        raise ValueError("each finalist repeat must contain exactly batch1 and the finalist")
    if repeat_orders[0] != list(reversed(repeat_orders[1])):
        raise ValueError("the two finalist repeat candidate orders were not reversed")

    for value in [worker, *repeats, v1]:
        binding = value["run_provenance"]["candidate_execution_configs"].get(candidate_id)
        if binding is None:
            raise ValueError(f"selected candidate config is absent from {value['run_summary_sha256']}")
        if (
            binding["batch_size"] != batch_size
            or binding["num_workers"] != workers
        ):
            raise ValueError("selected candidate execution tuning drifted across promotion runs")

    v12_checkpoints = {
        canonical_bytes(value["run_provenance"]["release_checkpoint"])
        for value in v12_values
    }
    if len(v12_checkpoints) != 1:
        raise ValueError("v1.2 immutable revision/checkpoint file hashes drifted across runs")
    releases = {
        value["run_provenance"]["repo_id"]: value["run_provenance"][
            "release_checkpoint"
        ]
        for value in (worker, v1)
    }
    if set(releases) != set(RELEASE_SPECS):
        raise ValueError("promoted execution contract does not bind both releases")
    first = provenance[0]
    return {
        "schema": "olmoearth-release-execution-contract-v1",
        "semantic_config_core": first["semantic_config_core"],
        "selected_tuning": {
            "candidate_id": candidate_id,
            "batch_size": batch_size,
            "num_workers": workers,
        },
        "runtime_versions": first["runtime_versions"],
        "rslearn_runtime_fingerprint": first["rslearn_runtime_fingerprint"],
        "batch_audit_code": promotion_code,
        "physical_gpu": {
            "index": first["selected_gpu"],
            "uuid": first["selected_gpu_uuid"],
        },
        "exact_smoke_inputs_sha256": first["exact_inputs_sha256"],
        "checkpoint_manifest_sha256": first["checkpoint_manifest_sha256"],
        "releases": {repo_id: releases[repo_id] for repo_id in sorted(releases)},
        "batch_template_sha256": {
            "v1_2": v12_values[0]["run_provenance"]["template_config_sha256"],
            "v1": v1["run_provenance"]["template_config_sha256"],
        },
        "validation": {
            "five_independent_runs": True,
            "four_v1_2_runs_same_template": True,
            "two_finalist_repeat_orders_reversed": True,
            "actual_resolved_yaml_validated": True,
            "five_runs_same_batch_runner_and_helpers": True,
            "five_runs_same_batch_analyzer_and_helpers": True,
            "five_runs_post_run_checkpoints_verified": True,
            "promotion_time_batch_code_reverified": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-verification", type=Path, required=True)
    parser.add_argument("--p0-verification-complete", type=Path, required=True)
    parser.add_argument("--ladder-analysis", type=Path, required=True)
    parser.add_argument("--ladder-complete", type=Path, required=True)
    parser.add_argument("--worker-analysis", type=Path, required=True)
    parser.add_argument("--worker-complete", type=Path, required=True)
    parser.add_argument("--repeat-analysis", type=Path, action="append", required=True)
    parser.add_argument("--repeat-complete", type=Path, action="append", required=True)
    parser.add_argument("--v1-analysis", type=Path, required=True)
    parser.add_argument("--v1-complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.repeat_analysis) != 2 or len(args.repeat_complete) != 2:
        raise SystemExit("exactly two independent v1.2 repeat analyses are required")
    p0 = read_json(args.p0_verification)
    p0_complete = read_json(args.p0_verification_complete)
    if (
        p0.get("status") != "FULL_EVIDENCE_VERIFIED"
        or p0.get("checks_passed") != p0.get("checks_total")
        or p0_complete.get("verification_sha256") != file_sha256(args.p0_verification)
    ):
        raise ValueError("P0 paired batch1 verification is not closed")

    ladder = validate_analysis(args.ladder_analysis, args.ladder_complete)
    worker = validate_analysis(args.worker_analysis, args.worker_complete)
    repeats = [
        validate_analysis(summary, complete)
        for summary, complete in zip(
            args.repeat_analysis, args.repeat_complete, strict=True
        )
    ]
    v1 = validate_analysis(args.v1_analysis, args.v1_complete)
    promotion_code = validate_run_roles(ladder, worker, repeats, v1)
    selected = worker["provisional_safe_candidate"]
    candidate_id = selected["candidate_id"]
    batch_size = int(selected["batch_size"])
    workers = int(selected["num_workers"])
    if candidate_id == "b001_w02":
        raise ValueError("optimized promotion is unnecessary; use the verified batch1 fallback")
    execution_contract = build_execution_contract(
        ladder,
        worker,
        repeats,
        v1,
        candidate_id=candidate_id,
        batch_size=batch_size,
        workers=workers,
        promotion_code=promotion_code,
    )

    selected_measurements = [comparison(worker, candidate_id)] + [
        comparison(value, candidate_id) for value in repeats
    ]
    v1_selected = comparison(v1, candidate_id)
    all_selected = [*selected_measurements, v1_selected]
    equivalence_all = all(value["equivalent_to_batch1"] for value in all_selected)
    v12_speeds = [value["end_to_end_crops_per_second"] for value in selected_measurements]
    finalist_cv = statistics.pstdev(v12_speeds) / statistics.mean(v12_speeds)

    anchor_summaries = [ladder, worker, *repeats]
    anchor_speeds = [
        comparison(value, "b001_w02")["end_to_end_crops_per_second"]
        for value in anchor_summaries
    ]
    anchor_first_last_drift = abs(anchor_speeds[-1] - anchor_speeds[0]) / anchor_speeds[0]
    anchor_range_fraction = (max(anchor_speeds) - min(anchor_speeds)) / statistics.mean(
        anchor_speeds
    )
    ladder_selected = ladder["provisional_safe_candidate"]
    selected_ladder_id = ladder_selected["candidate_id"]
    worker_selection_consistent = batch_size == int(ladder_selected["batch_size"])
    gate_checks = {
        "p0_batch1_two_release_bundle_verified": True,
        "five_runs_same_current_batch_runner_helper_analyzer_code": True,
        "ladder_selected_candidate_equivalent": comparison(ladder, selected_ladder_id)[
            "equivalent_to_batch1"
        ],
        "worker_ladder_selected_candidate_equivalent": comparison(worker, candidate_id)[
            "equivalent_to_batch1"
        ],
        "worker_selection_consistent_with_ladder_candidates": worker_selection_consistent,
        "two_reverse_repeat_candidates_equivalent": all(
            comparison(value, candidate_id)["equivalent_to_batch1"] for value in repeats
        ),
        "v1_candidate_equivalent": v1_selected["equivalent_to_batch1"],
        "both_release_numeric_gates_true": equivalence_all,
        "finalist_repeat_cv_at_most_0_05": finalist_cv <= 0.05,
        "anchor_first_last_drift_at_most_0_10": anchor_first_last_drift <= 0.10,
        "anchor_range_fraction_at_most_0_10": anchor_range_fraction <= 0.10,
    }
    promoted = all(gate_checks.values())
    evidence_files = [
        evidence("p0_verification", args.p0_verification),
        evidence("p0_verification_complete", args.p0_verification_complete),
        evidence("ladder_analysis", args.ladder_analysis),
        evidence("ladder_analysis_complete", args.ladder_complete),
        evidence("worker_analysis", args.worker_analysis),
        evidence("worker_analysis_complete", args.worker_complete),
        evidence("v1_analysis", args.v1_analysis),
        evidence("v1_analysis_complete", args.v1_complete),
    ]
    for index, (summary, complete) in enumerate(
        zip(args.repeat_analysis, args.repeat_complete, strict=True), start=1
    ):
        evidence_files.extend(
            (
                evidence(f"v1_2_repeat_{index}_analysis", summary),
                evidence(f"v1_2_repeat_{index}_complete", complete),
            )
        )
    result = {
        "schema": "olmoearth-release-batch-contract-v1",
        "finalizer_code_sha256": file_sha256(Path(__file__).resolve()),
        "status": "promoted" if promoted else "rejected",
        "selected": {
            "candidate_id": candidate_id,
            "batch_size": batch_size,
            "num_workers": workers,
        },
        "selection_mode": "optimized_two_release_numerical_equivalence",
        "execution_contract": execution_contract,
        "full_run_allowed": promoted,
        "promotion_pending": [] if promoted else [
            key for key, value in gate_checks.items() if not value
        ],
        "gate_checks": gate_checks,
        "measurements": {
            "v1_2_finalist_crops_per_second": v12_speeds,
            "v1_2_finalist_cv": finalist_cv,
            "v1_2_batch1_anchor_crops_per_second": anchor_speeds,
            "batch1_anchor_first_last_drift": anchor_first_last_drift,
            "batch1_anchor_range_fraction": anchor_range_fraction,
            "v1_finalist_crops_per_second": v1_selected[
                "end_to_end_crops_per_second"
            ],
        },
        "evidence_files": evidence_files,
        "claims_allowed": [
            "selected batch is numerically equivalent to batch1 on the exact eight-window gate"
        ]
        if promoted
        else [],
        "claims_forbidden": [
            "task_accuracy",
            "release_compatibility",
            "full_population_result_before_216_execution",
        ],
    }
    rendered = canonical_bytes(result)
    atomic_create(args.output, rendered)
    marker = args.output.with_name("BATCH_CONTRACT_COMPLETE.json")
    atomic_create(
        marker,
        canonical_bytes(
            {
                "schema": "olmoearth-release-batch-contract-completion-v1",
                "status": result["status"],
                "batch_contract_sha256": file_sha256(args.output),
            }
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not promoted:
        raise SystemExit(f"batch promotion rejected: {result['promotion_pending']}")


if __name__ == "__main__":
    main()
