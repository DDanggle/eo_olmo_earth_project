from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from analyze_olmo_release_batch_gate import (  # noqa: E402
    batch_analyzer_code_contract,
    chunked_numeric_comparison,
    validate_execution_audit_code,
    validate_execution_checkpoint_evidence,
    validate_executed_command,
)
from finalize_olmo_batch_contract import (  # noqa: E402
    validate_analysis,
    validate_promotion_code_contracts,
)
from run_olmo_release_batch_gate import (  # noqa: E402
    batch_audit_code_contract,
    canonical_bytes,
    optional_nvidia_float,
    parse_candidate,
    percentile,
    validate_batch_audit_code_contract,
)


class BatchGateTests(unittest.TestCase):
    def test_batch_audit_code_contract_is_canonical_and_detects_drift(self) -> None:
        contract = batch_audit_code_contract()
        self.assertEqual(validate_batch_audit_code_contract(contract), contract)
        self.assertEqual(contract, batch_audit_code_contract())
        drifted = copy.deepcopy(contract)
        drifted["direct_local_helpers"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "inventory digest mismatch"):
            validate_batch_audit_code_contract(drifted)

    def test_execution_code_closes_preflight_post_run_and_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            contract = batch_audit_code_contract()
            post_path = root / "POST_RUN_AUDIT_CODE_VERIFICATION.json"
            post_path.write_bytes(
                canonical_bytes(
                    {
                        "schema": "olmoearth-release-batch-post-run-audit-code-verification-v1",
                        "status": "verified",
                        "initial_audit_code_contract": contract,
                        "live_audit_code_contract": contract,
                        "error": None,
                    }
                )
            )
            post_sha = hashlib.sha256(post_path.read_bytes()).hexdigest()
            payload = {
                "audit_code_contract": contract,
                "post_run_audit_code_verified": True,
                "post_run_audit_code_verification": {
                    "path": post_path.as_posix(),
                    "sha256": post_sha,
                },
            }
            completion = {
                "post_run_audit_code_verified": True,
                "audit_code_contract_sha256": contract["inventory_sha256"],
                "post_run_audit_code_verification_sha256": post_sha,
            }
            self.assertEqual(
                validate_execution_audit_code(
                    payload,
                    completion,
                    {"audit_code_contract": contract},
                    root,
                ),
                contract,
            )
            post_path.write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence drift"):
                validate_execution_audit_code(
                    payload,
                    completion,
                    {"audit_code_contract": contract},
                    root,
                )

    def test_checkpoint_post_run_marker_is_required_and_drift_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            manifest = root / "checkpoints.json"
            manifest.write_bytes(canonical_bytes({"models": []}))
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            models = {"release": {"revision": "1" * 40, "files": []}}
            marker_path = root / "POST_RUN_CHECKPOINTS_VERIFICATION.json"
            marker_bytes = canonical_bytes(
                {
                    "schema": "olmoearth-release-batch-post-run-checkpoint-verification-v1",
                    "status": "verified",
                    "checkpoint_manifest_path": manifest.resolve().as_posix(),
                    "initial_checkpoint_manifest_sha256": manifest_sha,
                    "live_checkpoint_manifest_sha256": manifest_sha,
                    "initial_checkpoint_models": models,
                    "live_checkpoint_models": models,
                    "error": None,
                }
            )
            marker_path.write_bytes(marker_bytes)
            marker_sha = hashlib.sha256(marker_path.read_bytes()).hexdigest()
            payload = {
                "post_run_checkpoints_verified": True,
                "post_run_checkpoint_verification": {
                    "path": marker_path.as_posix(),
                    "sha256": marker_sha,
                },
            }
            completion = {
                "post_run_checkpoints_verified": True,
                "checkpoint_manifest_sha256": manifest_sha,
                "post_run_checkpoint_verification_sha256": marker_sha,
            }
            with mock.patch(
                "analyze_olmo_release_batch_gate.validate_checkpoints"
            ), mock.patch(
                "analyze_olmo_release_batch_gate.normalize_checkpoint_manifest",
                return_value=models,
            ):
                live, proof = validate_execution_checkpoint_evidence(
                    payload,
                    completion,
                    {"checkpoint_manifest_sha256": manifest_sha},
                    manifest,
                    root,
                )
            self.assertEqual(live, models)
            self.assertEqual(proof["status"], "verified")
            marker_path.write_text("drift\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence drift"):
                validate_execution_checkpoint_evidence(
                    payload,
                    completion,
                    {"checkpoint_manifest_sha256": manifest_sha},
                    manifest,
                    root,
                )
            marker_path.write_bytes(marker_bytes)
            marker_path.unlink()
            with self.assertRaisesRegex(ValueError, "evidence drift"):
                validate_execution_checkpoint_evidence(
                    payload,
                    completion,
                    {"checkpoint_manifest_sha256": manifest_sha},
                    manifest,
                    root,
                )

    def test_finalizer_requires_five_analyses_to_match_current_code(self) -> None:
        contract = batch_audit_code_contract()
        analyzer = batch_analyzer_code_contract()
        values = [
            {
                "run_provenance": {
                    "batch_audit_code_contract": copy.deepcopy(contract)
                },
                "analysis_code_contract": copy.deepcopy(analyzer),
            }
            for _ in range(5)
        ]
        validated = validate_promotion_code_contracts(values)
        self.assertEqual(validated["batch_runner_and_direct_helpers"], contract)
        drifted = copy.deepcopy(values)
        drifted_contract = drifted[-1]["analysis_code_contract"]
        drifted_contract["direct_local_helpers"][0]["sha256"] = "0" * 64
        drifted_inventory = {
            "analyzer": drifted_contract["analyzer"],
            "direct_local_helpers": drifted_contract["direct_local_helpers"],
        }
        drifted_contract["inventory_sha256"] = hashlib.sha256(
            canonical_bytes(drifted_inventory)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "one analyzer/helper code contract"):
            validate_promotion_code_contracts(drifted)

    def test_analysis_completion_is_bound_to_current_runner_and_analyzer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            contract = batch_audit_code_contract()
            analyzer = batch_analyzer_code_contract()
            models = {"release": {"revision": "1" * 40, "files": []}}
            manifest_sha = "3" * 64
            checkpoint_path = root / "POST_RUN_CHECKPOINTS_VERIFICATION.json"
            checkpoint_path.write_bytes(
                canonical_bytes(
                    {
                        "schema": "olmoearth-release-batch-post-run-checkpoint-verification-v1",
                        "status": "verified",
                        "checkpoint_manifest_path": "/evidence/checkpoints.json",
                        "initial_checkpoint_manifest_sha256": manifest_sha,
                        "live_checkpoint_manifest_sha256": manifest_sha,
                        "initial_checkpoint_models": models,
                        "live_checkpoint_models": models,
                        "error": None,
                    }
                )
            )
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            checkpoint_proof = {
                "path": checkpoint_path.as_posix(),
                "sha256": checkpoint_sha,
                "checkpoint_manifest_path": "/evidence/checkpoints.json",
                "checkpoint_manifest_sha256": manifest_sha,
                "checkpoint_models": models,
                "status": "verified",
            }
            summary_path = root / "analysis_summary.json"
            complete_path = root / "ANALYSIS_COMPLETE.json"
            summary = {
                "schema": "olmoearth-release-batch-equivalence-analysis-v1",
                "status": "complete",
                "analysis_code_contract": analyzer,
                "run_provenance": {
                    "batch_audit_code_contract": contract,
                    "post_run_checkpoint_verification": checkpoint_proof,
                },
            }
            summary_path.write_bytes(canonical_bytes(summary))
            complete_path.write_bytes(
                canonical_bytes(
                    {
                        "schema": "olmoearth-release-batch-equivalence-completion-v1",
                        "analysis_summary_sha256": hashlib.sha256(
                            summary_path.read_bytes()
                        ).hexdigest(),
                        "analysis_code_contract_sha256": analyzer[
                            "inventory_sha256"
                        ],
                        "batch_audit_code_contract_sha256": contract[
                            "inventory_sha256"
                        ],
                        "post_run_checkpoint_verification_sha256": checkpoint_sha,
                    }
                )
            )
            self.assertEqual(validate_analysis(summary_path, complete_path), summary)
            legacy = json.loads(summary_path.read_text(encoding="utf-8"))
            legacy.pop("analysis_code_contract")
            summary_path.write_bytes(canonical_bytes(legacy))
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            complete["analysis_summary_sha256"] = hashlib.sha256(
                summary_path.read_bytes()
            ).hexdigest()
            complete_path.write_bytes(canonical_bytes(complete))
            with self.assertRaisesRegex(ValueError, "analyzer code contract"):
                validate_analysis(summary_path, complete_path)

    def test_candidate_parser_rejects_unsafe_worker_count(self) -> None:
        self.assertEqual(parse_candidate("4:6"), (4, 6))
        with self.assertRaises(Exception):
            parse_candidate("4:16")

    def test_percentile_interpolates_deterministically(self) -> None:
        self.assertEqual(percentile([0.0, 10.0], 0.5), 5.0)

    def test_unavailable_nvidia_field_is_not_a_telemetry_error(self) -> None:
        self.assertIsNone(optional_nvidia_float("[Not Found]"))
        self.assertEqual(optional_nvidia_float("41"), 41.0)

    def test_numeric_gate_detects_small_and_large_drift(self) -> None:
        reference = np.arange(24, dtype=np.float32).reshape(4, 6) + 1.0
        close = reference + 1e-7
        far = reference.copy()
        far[0, 0] += 1e-2
        close_result = chunked_numeric_comparison(
            reference, close, relative_tolerance=1e-4, absolute_tolerance=1e-5
        )
        far_result = chunked_numeric_comparison(
            reference, far, relative_tolerance=1e-4, absolute_tolerance=1e-5
        )
        self.assertTrue(close_result["allclose"])
        self.assertFalse(far_result["allclose"])

    def test_candidate_command_is_bound_to_promoted_entrypoint_and_config(self) -> None:
        config = Path("/evidence/candidate/resolved_config.yaml")
        runtime = {"entrypoint": {"path": "/runtime/bin/rslearn"}}
        candidate = {
            "candidate_id": "b004_w02",
            "executed_command": [
                "/runtime/bin/rslearn",
                "model",
                "predict",
                "--config",
                config.as_posix(),
            ],
        }
        self.assertEqual(
            validate_executed_command(candidate, config, runtime),
            candidate["executed_command"],
        )
        candidate["executed_command"][0] = "/other/bin/rslearn"
        with self.assertRaisesRegex(ValueError, "executed command drift"):
            validate_executed_command(candidate, config, runtime)


if __name__ == "__main__":
    unittest.main()
