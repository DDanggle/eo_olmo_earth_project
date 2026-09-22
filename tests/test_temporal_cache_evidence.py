import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("cache_evidence", Path(__file__).parents[1] / "code/validate_temporal_cache_evidence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EvidenceTests(unittest.TestCase):
    def report(self):
        return {"schema": "olmo-streaming-dev-audit-v0", "n_ids": 3, "n_valid": 3,
                "n_skipped": 0, "all_gates_pass": True, "audit_max": 0.0,
                "audit": [{"id": f"s{i}", "max_abs_diff_c12_vs_sealed": 0.0} for i in range(3)]}

    def test_exact_zero_is_not_missing(self):
        self.assertEqual(module.validate(self.report(), 3), [])
        self.assertFalse((0.0 or 1) < 0.05)  # reproduces the old false failure

    def test_small_nonzero(self):
        report = self.report()
        report["audit_max"] = report["audit"][0]["max_abs_diff_c12_vs_sealed"] = 0.003
        self.assertEqual(module.validate(report, 3), [])

    def test_reject_null_nan_inf_negative_bool(self):
        for value in (None, float("nan"), float("inf"), -1, False):
            with self.subTest(value=value):
                report = self.report(); report["audit_max"] = value
                self.assertTrue(module.validate(report, 3))

    def test_strict_tolerance(self):
        report = self.report()
        report["audit_max"] = report["audit"][0]["max_abs_diff_c12_vs_sealed"] = 0.05
        self.assertTrue(module.validate(report, 3))

    def test_insufficient_or_missing_samples(self):
        report = self.report()
        self.assertTrue(module.validate(report, 4))
        report["audit"] = []
        self.assertTrue(module.validate(report, 1))

    def test_incomplete_extraction(self):
        for key, value in (("n_ids", 0), ("n_valid", 2), ("n_skipped", 1), ("all_gates_pass", False)):
            report = self.report(); report[key] = value
            self.assertTrue(module.validate(report, 3))

    def test_duplicate_and_inconsistent_summary(self):
        report = self.report(); report["audit"][1] = copy.deepcopy(report["audit"][0])
        self.assertTrue(module.validate(report, 3))
        report = self.report(); report["audit_max"] = 0.01
        self.assertTrue(module.validate(report, 3))

    def test_t0_schema(self):
        report = self.report(); report["schema"] = "olmo-temporal-cache-audit-v0"
        report["audit_mean_vs_sealed"] = [{"id": r["id"], "max_abs_diff_mean_vs_sealed": 0} for r in report.pop("audit")]
        self.assertEqual(module.validate(report, 3), [])


if __name__ == "__main__":
    unittest.main()
