import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).parents[1] / "code" / "geobench_action_headroom.py"
SPEC = importlib.util.spec_from_file_location("geobench_action_headroom", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(task, episode, action, seed, score, gpu=1.0, anchor=None):
    r = {
        "episode_id": episode, "task": task, "action": action, "seed": seed,
        "score": score, "higher_is_better": True,
        "gpu_seconds": gpu, "raw_bytes": 0, "cache_bytes": 1,
    }
    if anchor is not None:
        r["lower_anchor"], r["upper_anchor"] = anchor
    return r


class HeadroomTest(unittest.TestCase):
    def test_g0a_heterogeneity_and_g0b_value_pass_with_anchors(self):
        rows = []
        # Four independent task groups split 2:2 between A and B, three seeds each,
        # winners robust on every seed. Anchors fixed at (0, 1).
        for task, a, b in (("t1", 0.9, 0.5), ("t2", 0.8, 0.4), ("t3", 0.4, 0.8), ("t4", 0.5, 0.9)):
            for seed in (1, 2, 3):
                rows.append(row(task, f"{task}-r", "A", seed, a, anchor=(0.0, 1.0)))
                rows.append(row(task, f"{task}-r", "B", seed, b, anchor=(0.0, 1.0)))
        report = MODULE.analyze(rows and {"rows": rows})
        self.assertTrue(report["G0A_action_heterogeneity"]["pass"])
        self.assertEqual(report["G0A_action_heterogeneity"]["distinct_robust_winners"], ["A", "B"])
        self.assertTrue(report["G0B_operational_value"]["pass"])
        self.assertTrue(report["G0_pass"])
        # oracle = mean(max per task) = mean(0.9,0.8,0.8,0.9)=0.85; best static tie 0.65
        self.assertAlmostEqual(report["budgets"][0]["oracle_headroom"], 0.20)

    def test_dominant_action_fails_g0a_and_g0b(self):
        rows = []
        for task in ("t1", "t2", "t3"):
            rows.append(row(task, task, "always_best", 1, 0.9, anchor=(0.0, 1.0)))
            rows.append(row(task, task, "other", 1, 0.5, anchor=(0.0, 1.0)))
        report = MODULE.analyze({"rows": rows})
        self.assertFalse(report["G0_pass"])
        self.assertEqual(report["budgets"][0]["oracle_headroom"], 0.0)
        self.assertEqual(report["G0A_action_heterogeneity"]["distinct_robust_winners"], ["always_best"])

    def test_headroom_unavailable_without_anchors(self):
        rows = [row("t1", "e1", "A", 1, 0.6), row("t1", "e1", "B", 1, 0.4)]
        report = MODULE.analyze({"rows": rows})
        self.assertIsNone(report["budgets"][0]["oracle_headroom"])
        self.assertFalse(report["G0B_operational_value"]["computable"])
        self.assertFalse(report["G0_pass"])

    def test_iia_dominated_action_does_not_change_result(self):
        # Two tasks, opposite winners, anchored. Injecting a dominated action must not move anything.
        payload = {"rows": [
            row("t1", "e1", "cache", 1, 0.3, anchor=(0.0, 1.0)),
            row("t1", "e1", "adapt", 1, 0.6, anchor=(0.0, 1.0)),
            row("t2", "e2", "cache", 1, 0.7, anchor=(0.0, 1.0)),
            row("t2", "e2", "adapt", 1, 0.5, anchor=(0.0, 1.0)),
        ]}
        check = MODULE.check_iia(payload)
        self.assertTrue(check["iia_holds"])
        self.assertEqual(check["base"]["budgets"][0]["oracle_headroom"],
                         check["perturbed"]["budgets"][0]["oracle_headroom"])

    def test_budget_removes_expensive_action_without_renormalizing(self):
        rows = [
            row("t1", "e1", "cache", 1, 0.6, gpu=1, anchor=(0.0, 1.0)),
            row("t1", "e1", "reembed", 1, 0.9, gpu=20, anchor=(0.0, 1.0)),
            row("t2", "e2", "cache", 1, 0.7, gpu=1, anchor=(0.0, 1.0)),
            row("t2", "e2", "reembed", 1, 0.8, gpu=20, anchor=(0.0, 1.0)),
        ]
        report = MODULE.analyze({"rows": rows, "budgets": [{"name": "cheap", "max_gpu_seconds": 5}]})
        budget = report["budgets"][0]
        self.assertEqual(budget["common_static_actions"], ["cache"])
        # cache normalized values are its native anchored scores, unchanged by removing reembed
        self.assertAlmostEqual(budget["oracle_headroom"], 0.0)

    def test_support_label_budget_filters(self):
        rows = [
            row("t1", "e1", "cache", 1, 0.6, anchor=(0.0, 1.0)),
            row("t2", "e2", "cache", 1, 0.7, anchor=(0.0, 1.0)),
        ]
        rows[0]["support_label_count"] = 0
        rows[1]["support_label_count"] = 0
        adapt = [
            {**row("t1", "e1", "adapt", 1, 0.9, anchor=(0.0, 1.0)), "support_label_count": 20},
            {**row("t2", "e2", "adapt", 1, 0.9, anchor=(0.0, 1.0)), "support_label_count": 20},
        ]
        report = MODULE.analyze({"rows": rows + adapt,
                                 "budgets": [{"name": "zeroshot", "max_support_labels": 0}]})
        self.assertEqual(report["budgets"][0]["common_static_actions"], ["cache"])

    def test_duplicate_seed_outcome_rejected(self):
        d = row("t1", "e1", "A", 1, 0.8)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.analyze({"rows": [d, dict(d)]})

    def test_bad_anchor_rejected(self):
        with self.assertRaisesRegex(ValueError, "upper_anchor must exceed"):
            MODULE.analyze({"rows": [row("t1", "e1", "A", 1, 0.5, anchor=(0.9, 0.9))]})

    def test_negative_regression_score_allowed(self):
        rows = [row("reg", "reg-1", "A", 1, -0.2, anchor=(-1.0, 0.0)),
                row("reg", "reg-1", "B", 1, -0.5, anchor=(-1.0, 0.0))]
        report = MODULE.analyze({"rows": rows})
        self.assertEqual(report["budgets"][0]["episodes"][0]["native_best_action"], "A")

    def test_declared_incomplete_action_matrix_fails_closed(self):
        rows = [
            row("t1", "e1", "cache", 1, 0.6, anchor=(0.0, 1.0)),
            row("t1", "e1", "adapt", 1, 0.8, anchor=(0.0, 1.0)),
            row("t2", "e2", "cache", 1, 0.8, anchor=(0.0, 1.0)),
            # adapt was never measured for e2; this is not action ineligibility.
        ]
        report = MODULE.analyze({
            "rows": rows,
            "required_actions": ["cache", "adapt"],
            "required_seed_count": 1,
        })
        self.assertEqual(report["status"], "INCOMPLETE_ACTION_MATRIX_DIAGNOSTIC_ONLY")
        self.assertFalse(report["matrix_contract"]["complete"])
        self.assertEqual(report["matrix_contract"]["missing_actions_by_episode"],
                         {"e2": ["adapt"]})
        self.assertFalse(report["G0A_action_heterogeneity"]["pass"])
        self.assertFalse(report["G0B_operational_value"]["pass"])
        self.assertFalse(report["G0_pass"])

    def test_declared_seed_count_is_enforced(self):
        rows = []
        for seed in (1, 2, 3):
            rows.append(row("t1", "e1", "cache", seed, 0.6, anchor=(0.0, 1.0)))
        rows.append(row("t1", "e1", "adapt", 1, 0.8, anchor=(0.0, 1.0)))
        report = MODULE.analyze({
            "rows": rows,
            "required_actions": ["cache", "adapt"],
            "required_seed_count": 3,
        })
        self.assertFalse(report["matrix_contract"]["complete"])
        self.assertEqual(report["matrix_contract"]["insufficient_seed_counts_by_episode"],
                         {"e1": {"adapt": 1}})
        self.assertFalse(report["G0_pass"])


if __name__ == "__main__":
    unittest.main()
