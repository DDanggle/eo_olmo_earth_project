from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from analyze_olmo_release_full import (  # noqa: E402
    ANALYZER_POST_CODE_SCHEMA,
    analyzer_code_contract,
    validate_analyzer_code_contract,
    validate_analyzer_code_evidence,
    verify_analyzer_code_stability,
)
from finalize_olmo_release_full import (  # noqa: E402
    FINALIZER_POST_CODE_SCHEMA,
    finalizer_code_contract,
    validate_finalizer_code_contract,
    validate_finalizer_code_evidence,
    verify_finalizer_code_stability,
)
from run_olmo_release_full import (  # noqa: E402
    FULL_RUNNER_POST_CODE_SCHEMA,
    canonical_bytes,
    full_runner_code_contract,
    validate_full_release_command,
    validate_full_runner_code_contract,
    validate_full_runner_code_evidence,
    verify_full_runner_code_stability,
)


def rehash_contract(value: dict[str, object]) -> None:
    inventory = {
        "owner_role": value["owner_role"],
        "owner": value["owner"],
        "direct_local_helpers": value["direct_local_helpers"],
    }
    value["inventory_sha256"] = hashlib.sha256(canonical_bytes(inventory)).hexdigest()


class FullReleaseCodeChainTests(unittest.TestCase):
    def test_full_runner_inventory_is_canonical_and_live(self) -> None:
        contract = full_runner_code_contract()
        self.assertEqual(validate_full_runner_code_contract(contract), contract)
        self.assertEqual(contract["owner_role"], "full_release_runner")
        self.assertEqual(contract["owner"]["role"], "owner")
        self.assertEqual(contract["owner"]["content"], "python_source_utf8")
        self.assertEqual(
            [value["module"] for value in contract["direct_local_helpers"]],
            [
                "hash_olmo_release_inputs",
                "olmo_release_raster_contract",
                "olmo_release_semantic_contract",
                "prepare_olmo_release_audit_view",
                "run_olmo_release_batch_gate",
                "run_olmo_release_smoke",
            ],
        )

    def test_runner_contract_rejects_missing_tampered_and_live_drift(self) -> None:
        contract = full_runner_code_contract()
        missing = copy.deepcopy(contract)
        missing["direct_local_helpers"].pop()
        rehash_contract(missing)
        with self.assertRaisesRegex(ValueError, "helper module inventory drifted"):
            validate_full_runner_code_contract(missing, require_live_match=False)

        for field, value, message in (
            ("role", "owner", "role drifted"),
            ("content", "binary_blob", "content role drifted"),
            ("path", "/tmp/not-the-runner.py", "canonical path drifted"),
        ):
            malformed = copy.deepcopy(contract)
            malformed["direct_local_helpers"][0][field] = value
            rehash_contract(malformed)
            with self.assertRaisesRegex(ValueError, message):
                validate_full_runner_code_contract(
                    malformed, require_live_match=False
                )

        tampered = copy.deepcopy(contract)
        tampered["owner"]["sha256"] = "0" * 64
        rehash_contract(tampered)
        self.assertEqual(
            validate_full_runner_code_contract(tampered, require_live_match=False),
            tampered,
        )
        with self.assertRaisesRegex(ValueError, "differs from current live source"):
            validate_full_runner_code_contract(tampered, require_live_match=True)

        live_drift = copy.deepcopy(contract)
        live_drift["owner"]["sha256"] = "1" * 64
        rehash_contract(live_drift)
        with mock.patch(
            "run_olmo_release_full.full_runner_code_contract",
            return_value=live_drift,
        ):
            with self.assertRaisesRegex(ValueError, "changed during execution"):
                verify_full_runner_code_stability(contract)

    def test_finalizer_contract_rejects_start_end_drift(self) -> None:
        contract = finalizer_code_contract()
        self.assertEqual(validate_finalizer_code_contract(contract), contract)
        drifted = copy.deepcopy(contract)
        drifted["direct_local_helpers"][0]["sha256"] = "2" * 64
        rehash_contract(drifted)
        with mock.patch(
            "finalize_olmo_release_full.finalizer_code_contract",
            return_value=drifted,
        ):
            with self.assertRaisesRegex(ValueError, "changed during finalization"):
                verify_finalizer_code_stability(contract)

    def test_full_runner_marker_content_and_sha_are_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            result_root = Path(temporary_name)
            contract = full_runner_code_contract()
            marker_path = result_root / "POST_RUN_FULL_RUNNER_CODE_VERIFICATION.json"
            marker = {
                "schema": FULL_RUNNER_POST_CODE_SCHEMA,
                "status": "verified",
                "initial_full_runner_code_contract": contract,
                "live_full_runner_code_contract": contract,
                "error": None,
            }
            marker_path.write_bytes(canonical_bytes(marker))
            marker_sha = hashlib.sha256(marker_path.read_bytes()).hexdigest()
            preflight = {"full_runner_code_contract": contract}
            run = {
                "full_runner_code_contract": contract,
                "post_run_full_runner_code_verified": True,
                "post_run_full_runner_code_error": None,
                "post_run_full_runner_code_verification": {
                    "path": marker_path.resolve().as_posix(),
                    "sha256": marker_sha,
                },
            }
            completion = {
                "post_run_full_runner_code_verified": True,
                "full_runner_code_contract_sha256": contract["inventory_sha256"],
                "post_run_full_runner_code_verification_sha256": marker_sha,
            }
            self.assertEqual(
                validate_full_runner_code_evidence(
                    preflight=preflight,
                    run_summary=run,
                    completion=completion,
                    result_root=result_root,
                ),
                contract,
            )

            marker["status"] = "failed"
            marker_path.write_bytes(canonical_bytes(marker))
            marker_sha = hashlib.sha256(marker_path.read_bytes()).hexdigest()
            run["post_run_full_runner_code_verification"]["sha256"] = marker_sha
            completion["post_run_full_runner_code_verification_sha256"] = marker_sha
            with self.assertRaisesRegex(ValueError, "marker content drift"):
                validate_full_runner_code_evidence(
                    preflight=preflight,
                    run_summary=run,
                    completion=completion,
                    result_root=result_root,
                )

            marker_path.unlink()
            with self.assertRaisesRegex(ValueError, "evidence drift"):
                validate_full_runner_code_evidence(
                    preflight=preflight,
                    run_summary=run,
                    completion=completion,
                    result_root=result_root,
                )

    def test_finalizer_evidence_rejects_marker_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            finalizer = finalizer_code_contract()
            runner = full_runner_code_contract()
            marker_path = root / "POST_RUN_FINALIZER_CODE_VERIFICATION.json"
            marker = {
                "schema": FINALIZER_POST_CODE_SCHEMA,
                "status": "verified",
                "initial_finalizer_code_contract": finalizer,
                "live_finalizer_code_contract": finalizer,
                "error": None,
            }
            marker_path.write_bytes(canonical_bytes(marker))
            marker_sha = hashlib.sha256(marker_path.read_bytes()).hexdigest()
            summary = {
                "finalizer_code_sha256": finalizer["owner"]["sha256"],
                "finalizer_code_contract": finalizer,
                "post_run_finalizer_code_verified": True,
                "post_run_finalizer_code_verification": {
                    "path": marker_path.resolve().as_posix(),
                    "sha256": marker_sha,
                },
                "full_runner_code_contract": runner,
            }
            completion = {
                "status": "complete",
                "finalizer_code_contract_sha256": finalizer["inventory_sha256"],
                "post_run_finalizer_code_verified": True,
                "post_run_finalizer_code_verification_sha256": marker_sha,
                "full_runner_code_contract_sha256": runner["inventory_sha256"],
            }
            validated = validate_finalizer_code_evidence(
                evidence_summary=summary,
                evidence_completion=completion,
                evidence_root=root,
            )
            self.assertEqual(validated["finalizer"], finalizer)
            self.assertEqual(validated["full_runner"], runner)

            marker["error"] = "tampered"
            marker_path.write_bytes(canonical_bytes(marker))
            marker_sha = hashlib.sha256(marker_path.read_bytes()).hexdigest()
            summary["post_run_finalizer_code_verification"]["sha256"] = marker_sha
            completion["post_run_finalizer_code_verification_sha256"] = marker_sha
            with self.assertRaisesRegex(ValueError, "marker content drift"):
                validate_finalizer_code_evidence(
                    evidence_summary=summary,
                    evidence_completion=completion,
                    evidence_root=root,
                )

    def test_analyzer_contract_and_lock_summary_completion_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            analyzer = analyzer_code_contract()
            runner = full_runner_code_contract()
            finalizer = finalizer_code_contract()
            self.assertEqual(validate_analyzer_code_contract(analyzer), analyzer)
            self.assertEqual(
                [item["module"] for item in analyzer["direct_local_helpers"]],
                ["finalize_olmo_release_full", "run_olmo_release_full"],
            )
            marker_path = root / "POST_ANALYSIS_CODE_VERIFICATION.json"
            marker = {
                "schema": ANALYZER_POST_CODE_SCHEMA,
                "status": "verified",
                "initial_analyzer_code_contract": analyzer,
                "live_analyzer_code_contract": analyzer,
                "error": None,
            }
            marker_path.write_bytes(canonical_bytes(marker))
            marker_sha = hashlib.sha256(marker_path.read_bytes()).hexdigest()
            lock = {
                "schema": "olmoearth-release-full-preanalysis-lock-v1",
                "status": "sealed_before_any_output_raster_read",
                "analyzer": {
                    "path": analyzer["owner"]["path"],
                    "sha256": analyzer["owner"]["sha256"],
                },
                "analyzer_code_contract": analyzer,
                "upstream_code_contracts": {
                    "full_runner": runner,
                    "finalizer": finalizer,
                },
            }
            summary = {
                "analysis_code_contract": analyzer,
                "analysis_code_sha256": analyzer["owner"]["sha256"],
                "post_analysis_code_verified": True,
                "post_analysis_code_verification": {
                    "path": marker_path.resolve().as_posix(),
                    "sha256": marker_sha,
                },
                "upstream_code_contracts": lock["upstream_code_contracts"],
            }
            completion = {
                "post_analysis_code_verified": True,
                "analysis_code_contract_sha256": analyzer["inventory_sha256"],
                "analysis_code_sha256": analyzer["owner"]["sha256"],
                "post_analysis_code_verification_sha256": marker_sha,
                "full_runner_code_contract_sha256": runner["inventory_sha256"],
                "finalizer_code_contract_sha256": finalizer["inventory_sha256"],
            }
            validated = validate_analyzer_code_evidence(
                lock_payload=lock,
                analysis_summary=summary,
                analysis_root=root,
                completion=completion,
            )
            self.assertEqual(validated["analyzer"], analyzer)

            marker["live_analyzer_code_contract"] = runner
            marker_path.write_bytes(canonical_bytes(marker))
            marker_sha = hashlib.sha256(marker_path.read_bytes()).hexdigest()
            summary["post_analysis_code_verification"]["sha256"] = marker_sha
            completion["post_analysis_code_verification_sha256"] = marker_sha
            with self.assertRaisesRegex(ValueError, "marker content drift"):
                validate_analyzer_code_evidence(
                    lock_payload=lock,
                    analysis_summary=summary,
                    analysis_root=root,
                    completion=completion,
                )

    def test_analyzer_start_end_drift_fails_closed(self) -> None:
        contract = analyzer_code_contract()
        drifted = copy.deepcopy(contract)
        drifted["owner"]["sha256"] = "3" * 64
        rehash_contract(drifted)
        with mock.patch(
            "analyze_olmo_release_full.analyzer_code_contract",
            return_value=drifted,
        ):
            with self.assertRaisesRegex(ValueError, "changed during analysis"):
                verify_analyzer_code_stability(contract)

    def test_full_release_command_rejects_extra_or_reordered_arguments(self) -> None:
        rslearn = Path("/opt/venv/bin/rslearn")
        config = Path("/evidence/run/resolved_config.yaml")
        core = [
            rslearn.as_posix(),
            "model",
            "predict",
            "--config",
            config.as_posix(),
        ]
        self.assertIsNone(
            validate_full_release_command(
                core, rslearn_entrypoint=rslearn, resolved_config=config
            )["wrapper"]
        )
        for bad in (
            [*core, "--extra"],
            [core[0], "predict", "model", *core[3:]],
            ["ionice", "-c", "2", "-n", "7", *core],
            ["ionice", "-c", "3", "-n", "7", *core],
        ):
            with self.assertRaisesRegex(ValueError, "unexpected path or argument"):
                validate_full_release_command(
                    bad, rslearn_entrypoint=rslearn, resolved_config=config
                )


if __name__ == "__main__":
    unittest.main()
