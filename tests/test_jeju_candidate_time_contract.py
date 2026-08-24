import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "audit_jeju_candidate_time_contract.py"
SPEC = importlib.util.spec_from_file_location("jeju_candidate_time_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class JejuCandidateTimeContractTest(unittest.TestCase):
    def test_historical_overlap_is_184_days(self):
        self.assertEqual(
            MODULE.overlap_days(MODULE.WINDOWS["2025"], MODULE.WINDOWS["2026"]),
            184,
        )

    def test_repository_evidence_reproduces_nine_of_fourteen(self):
        time_axis = json.loads(
            (ROOT / "artifacts/results/jeju_time_axis_summary.json").read_text()
        )
        candidates = json.loads(
            (ROOT / "artifacts/human_review_v1/manifest.json").read_text()
        )
        result = MODULE.build_audit(time_axis, candidates)
        self.assertEqual(
            result["candidate_summary"],
            {
                "records": 14,
                "overlap_transition_records": 5,
                "legacy_four_period_source_records": 5,
                "union_contract_exposed_records": 9,
                "not_exposed_by_these_two_checks": 5,
            },
        )
        self.assertFalse(
            result["time_axis_gates"]["model_first4_season_aligned_across_years"]
        )
        self.assertFalse(
            result["time_axis_gates"]["all12_cover_same_calendar_month_set"]
        )

    def test_legacy_candidate_scripts_refuse_default_execution(self):
        for name in (
            "change_v2_step.py",
            "change_v5.py",
            "change_v6_t12.py",
            "build_jeju_human_review.py",
            "score_oreum_existing_embeddings.py",
        ):
            proc = subprocess.run(
                [sys.executable, str(ROOT / "code" / name)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0, name)
            self.assertIn("REFUSED", proc.stderr + proc.stdout, name)

    def test_legacy_window_shell_scripts_refuse_default_execution(self):
        for name in ("setup_jeju_v2.sh", "extract_jeju_t12.sh"):
            proc = subprocess.run(
                ["bash", str(ROOT / "code" / name)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0, name)
            self.assertIn("REFUSED", proc.stderr + proc.stdout, name)


if __name__ == "__main__":
    unittest.main()
