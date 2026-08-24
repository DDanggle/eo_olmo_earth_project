#!/usr/bin/env python3
"""Close the OlmoEarth release-smoke evidence chain without adding new claims."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_RELEASES = ("olmoearth_v1_base", "olmoearth_v1_2_base")
EXPECTED_ANALYSIS_CLAIM = (
    "descriptive_representation_continuity_on_the_eight_prespecified_exact_inputs"
)


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


def parse_first_json(text: str) -> dict[str, Any]:
    value, _ = json.JSONDecoder().raw_decode(text.lstrip())
    if not isinstance(value, dict):
        raise ValueError("the first launcher payload is not a JSON object")
    return value


class EvidenceChecks:
    def __init__(self, require_raw: bool) -> None:
        self.require_raw = require_raw
        self.checks: dict[str, bool] = {}
        self.failures: list[str] = []
        self.missing_raw: list[str] = []
        self.raw_files_checked = 0
        self.raw_bytes_checked = 0

    def require(self, label: str, condition: bool) -> None:
        passed = bool(condition)
        self.checks[label] = passed
        if not passed:
            self.failures.append(label)

    def inventory(
        self,
        label: str,
        path: Path,
        expected: dict[str, Any],
        *,
        raw: bool,
    ) -> None:
        exists = path.is_file()
        self.checks[f"{label}.exists"] = exists
        if not exists:
            if raw:
                self.missing_raw.append(path.as_posix())
                if self.require_raw:
                    self.failures.append(f"{label}.exists")
            else:
                self.failures.append(f"{label}.exists")
            return
        size = path.stat().st_size
        digest = file_sha256(path)
        self.raw_files_checked += int(raw)
        self.raw_bytes_checked += size if raw else 0
        if "bytes" in expected:
            self.require(f"{label}.bytes", size == int(expected["bytes"]))
        self.require(f"{label}.sha256", digest == expected["sha256"])


def _identity(record: dict[str, Any]) -> dict[str, str]:
    return {
        "sample_id": record["sample_id"],
        "window_name": record.get("window_name", record.get("window", "")),
        "spatial_cluster_id": record["spatial_cluster_id"],
        "input_bundle_identity": record["input_bundle_identity"],
    }


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def verify_bundle(project_root: Path, bundle_root: Path, require_raw: bool) -> dict[str, Any]:
    release_root = bundle_root / "release_audit_p0"
    results_root = release_root / "results"
    analysis_root = results_root / "analysis"
    paths = {
        "preflight": release_root / "preflight.json",
        "checkpoints": release_root / "checkpoints.json",
        "exact_inputs": release_root / "smoke_inputs_exact.json",
        "run_summary": results_root / "run_summary.json",
        "complete": results_root / "COMPLETE.json",
        "launcher_log": results_root / "launcher.log",
        "analysis_summary": analysis_root / "analysis_summary.json",
        "analysis_complete": analysis_root / "ANALYSIS_COMPLETE.json",
        "per_window_metrics": analysis_root / "per_window_metrics.csv",
    }
    checks = EvidenceChecks(require_raw=require_raw)
    for label, path in paths.items():
        checks.require(f"core.{label}.exists", path.is_file())
    if checks.failures:
        return _report(checks, require_raw, 0, 0)

    preflight = read_json(paths["preflight"])
    checkpoints = read_json(paths["checkpoints"])
    exact_inputs = read_json(paths["exact_inputs"])
    run_summary = read_json(paths["run_summary"])
    complete = read_json(paths["complete"])
    analysis = read_json(paths["analysis_summary"])
    analysis_complete = read_json(paths["analysis_complete"])
    launcher_preflight = parse_first_json(paths["launcher_log"].read_text(encoding="utf-8"))

    exact_sha = file_sha256(paths["exact_inputs"])
    checkpoint_sha = file_sha256(paths["checkpoints"])
    run_sha = file_sha256(paths["run_summary"])
    analysis_sha = file_sha256(paths["analysis_summary"])
    metrics_sha = file_sha256(paths["per_window_metrics"])

    checks.require("preflight.schema", preflight.get("schema") == "olmoearth-release-smoke-preflight-v1")
    checks.require("preflight.ready", preflight.get("ready") is True)
    checks.require("preflight.execute_requested", preflight.get("execute_requested") is True)
    checks.require("preflight.selected_gpu_0", preflight.get("selected_gpu") == "0")
    checks.require("preflight.selected_gpu_was_idle", preflight.get("gpu_processes") == [])
    checks.require("preflight.records_8", preflight.get("records") == 8)
    checks.require("preflight.release_runs_2", preflight.get("release_runs") == 2)
    checks.require("preflight.launcher_copy_equal", preflight == launcher_preflight)
    checks.require("preflight.exact_inputs_sha", preflight.get("exact_inputs_sha256") == exact_sha)
    checks.require(
        "preflight.checkpoint_manifest_sha",
        preflight.get("checkpoint_manifest_sha256") == checkpoint_sha,
    )

    checks.require("complete.run_summary_sha", complete.get("run_summary_sha256") == run_sha)
    checks.require("run.status_complete", run_summary.get("status") == "complete")
    checks.require(
        "run.exact_inputs_sha",
        run_summary.get("input_pairing", {}).get("exact_inputs_sha256") == exact_sha,
    )
    checks.require(
        "run.same_manifest_for_both_releases",
        run_summary.get("input_pairing", {}).get("same_manifest_for_both_releases") is True,
    )

    exact_records = exact_inputs.get("records", [])
    exact_identities = sorted((_identity(record) for record in exact_records), key=lambda value: value["sample_id"])
    paired_identities = sorted(
        (_identity(record) for record in run_summary.get("input_pairing", {}).get("samples", [])),
        key=lambda value: value["sample_id"],
    )
    checks.require("identity.exact_records_8", len(exact_identities) == 8)
    checks.require("identity.paired_records_8", len(paired_identities) == 8)
    checks.require("identity.pairing_equal", exact_identities == paired_identities)

    checkpoint_by_repo = {record["repo_id"]: record for record in checkpoints.get("models", [])}
    runs = run_summary.get("runs", [])
    checks.require("run.two_releases", tuple(record.get("release_id") for record in runs) == EXPECTED_RELEASES)
    output_identity_sets: list[list[dict[str, str]]] = []
    for run_index, run in enumerate(runs):
        prefix = f"run.{run_index}.{run.get('release_id', 'unknown')}"
        checkpoint = checkpoint_by_repo.get(run.get("repo_id"))
        checks.require(f"{prefix}.checkpoint_repo", checkpoint is not None)
        if checkpoint is not None:
            checks.require(f"{prefix}.checkpoint_revision", run.get("revision") == checkpoint.get("revision"))
            checks.require(f"{prefix}.checkpoint_inventory", run.get("checkpoint_files") == checkpoint.get("files"))
            for file_index, expected in enumerate(checkpoint.get("files", [])):
                raw_path = Path(checkpoint["snapshot_path"]) / expected["name"]
                checks.inventory(f"{prefix}.checkpoint_file_{file_index}", raw_path, expected, raw=True)

        config_path = project_root / run["config_path"]
        checks.inventory(
            f"{prefix}.config",
            config_path,
            {"sha256": run["config_sha256"]},
            raw=False,
        )
        log_path = bundle_root / run["log_path"]
        checks.inventory(
            f"{prefix}.log",
            log_path,
            {"sha256": run["log_sha256"]},
            raw=False,
        )
        if log_path.is_file():
            checks.require(
                f"{prefix}.cuda_visible_devices_0",
                "CUDA_VISIBLE_DEVICES: [0]" in log_path.read_text(encoding="utf-8"),
            )
        checks.require(f"{prefix}.legacy_timestamps", run.get("timestamp_track") == "legacy_timestamps")
        checks.require(f"{prefix}.eight_outputs", len(run.get("outputs", [])) == 8)
        output_identities = sorted(
            (_identity(record) for record in run.get("outputs", [])),
            key=lambda value: value["sample_id"],
        )
        output_identity_sets.append(output_identities)
        checks.require(f"{prefix}.output_identity_equal", output_identities == exact_identities)
        for output_index, output in enumerate(run.get("outputs", [])):
            checks.inventory(
                f"{prefix}.output_{output_index}",
                bundle_root / output["path"],
                output,
                raw=True,
            )

    checks.require(
        "identity.release_output_sets_equal",
        len(output_identity_sets) == 2 and output_identity_sets[0] == output_identity_sets[1],
    )

    for record_index, record in enumerate(exact_records):
        for layer_index, layer in enumerate(record.get("input_layers", [])):
            checks.inventory(
                f"input.{record_index}.layer_{layer_index}.geotiff",
                Path(layer["geotiff"]["path"]),
                layer["geotiff"],
                raw=True,
            )
            checks.inventory(
                f"input.{record_index}.layer_{layer_index}.metadata",
                Path(layer["metadata"]["path"]),
                layer["metadata"],
                raw=True,
            )
        checks.inventory(
            f"input.{record_index}.items_json",
            Path(record["items_json"]["path"]),
            record["items_json"],
            raw=True,
        )
        checks.inventory(
            f"input.{record_index}.window_metadata",
            Path(record["window_metadata"]["path"]),
            record["window_metadata"],
            raw=True,
        )

    checks.require("analysis.summary_sha", analysis_complete.get("analysis_summary_sha256") == analysis_sha)
    checks.require("analysis.metrics_sha", analysis_complete.get("per_window_metrics_sha256") == metrics_sha)
    checks.require("analysis.run_summary_sha", analysis.get("run_summary_sha256") == run_sha)
    checks.require("analysis.exact_inputs_sha", analysis.get("exact_inputs_sha256") == exact_sha)
    checks.require("analysis.status_complete", analysis.get("status") == "complete")
    checks.require("analysis.only_descriptive_claim", analysis.get("claims_allowed") == [EXPECTED_ANALYSIS_CLAIM])
    sample_contract = analysis.get("sample_contract", {})
    checks.require("analysis.records_8", sample_contract.get("n_records") == 8)
    checks.require("analysis.spatial_clusters_7", sample_contract.get("n_spatial_clusters") == 7)
    checks.require("analysis.labels_0", sample_contract.get("labels") == 0)
    checks.require("analysis.no_population_inference", sample_contract.get("population_inference_allowed") is False)

    with paths["per_window_metrics"].open(encoding="utf-8", newline="") as source:
        metric_rows = list(csv.DictReader(source))
    checks.require("analysis.metric_rows_8", len(metric_rows) == 8)
    checks.require(
        "analysis.metric_sample_ids_equal",
        sorted(row["sample_id"] for row in metric_rows)
        == sorted(record["sample_id"] for record in exact_records),
    )
    for field, summary_name in (
        ("linear_cka", "linear_cka"),
        ("row_l2_normalized_linear_cka", "row_l2_normalized_linear_cka"),
        ("shift_null_median", "shift_null_median"),
        ("excess_cka_over_shift_null", "excess_cka_over_shift_null"),
    ):
        values = [float(row[field]) for row in metric_rows]
        expected = analysis["per_window_spatial_cka"][summary_name]
        checks.require(f"analysis.{field}.minimum", _close(min(values), expected["minimum"]))
        checks.require(f"analysis.{field}.mean", _close(sum(values) / len(values), expected["mean"]))
        checks.require(f"analysis.{field}.maximum", _close(max(values), expected["maximum"]))

    return _report(checks, require_raw, len(exact_identities), len(runs))


def _report(
    checks: EvidenceChecks,
    require_raw: bool,
    identities: int,
    releases: int,
) -> dict[str, Any]:
    if checks.failures:
        status = "FAILED"
    elif checks.missing_raw:
        status = "PARTIAL_VERIFIED"
    else:
        status = "FULL_EVIDENCE_VERIFIED"
    return {
        "schema": "olmoearth-release-evidence-verification-v1",
        "status": status,
        "raw_verification_required": require_raw,
        "verified_identities": identities,
        "verified_releases": releases,
        "raw_files_checked": checks.raw_files_checked,
        "raw_bytes_checked": checks.raw_bytes_checked,
        "checks_total": len(checks.checks),
        "checks_passed": sum(checks.checks.values()),
        "failures": checks.failures,
        "missing_raw": checks.missing_raw,
        "claims_allowed": [
            "execution_and_evidence_chain_integrity",
            EXPECTED_ANALYSIS_CLAIM,
        ],
        "claims_forbidden": [
            "accuracy_improvement",
            "negative_transfer_reduction",
            "cloud_robustness",
            "korea_or_jeju_generalization",
            "backward_compatible_cache",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--require-raw", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--complete-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.complete_output.exists():
        raise FileExistsError("verification outputs already exist; refuse stale overwrite")
    report = verify_bundle(
        project_root=args.project_root,
        bundle_root=args.bundle_root,
        require_raw=args.require_raw,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report))
    if report["status"] != "FAILED":
        marker = {
            "schema": "olmoearth-release-evidence-verification-completion-v1",
            "status": report["status"],
            "verification_sha256": file_sha256(args.output),
        }
        args.complete_output.write_bytes(canonical_bytes(marker))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
