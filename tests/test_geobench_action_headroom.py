import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).parents[1] / "code" / "geobench_action_headroom.py"
SPEC = importlib.util.spec_from_file_location("geobench_action_headroom", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(task, episode, action, seed, score, gpu=1.0):
    return {
        "episode_id": episode,
        "task": task,
        "action": action,
        "seed": seed,
        "score": score,
        "higher_is_better": True,
        "gpu_seconds": gpu,
        "raw_bytes": 0,
        "cache_bytes": 1,
    }


class HeadroomTest(unittest.TestCase):
    def test_rank_reversal_and_headroom_pass(self):
        rows = []
        # Four independent task groups split 2:2 between A and B. No static
        # action can match the oracle; folds and seeds are not counted as tasks.
        for task, a_score, b_score in (
            ("t1", 0.9, 0.5),
            ("t2", 0.8, 0.4),
            ("t3", 0.4, 0.8),
            ("t4", 0.5, 0.9),
        ):
            for seed in (1, 2, 3):
                rows.extend(
                    [
                        row(task, f"{task}-region", "A", seed, a_score),
                        row(task, f"{task}-region", "B", seed, b_score),
                    ]
                )
        report = MODULE.analyze({"rows": rows})
        self.assertTrue(report["G0_pass"])
        self.assertEqual(report["budgets"][0]["n_tasks"], 4)
        self.assertAlmostEqual(report["budgets"][0]["oracle_headroom"], 0.5)

    def test_dominant_action_has_no_selector_headroom(self):
        rows = []
        for task in ("t1", "t2", "t3"):
            rows.extend(
                [
                    row(task, task, "always_best", 1, 0.9),
                    row(task, task, "other", 1, 0.5),
                ]
            )
        report = MODULE.analyze({"rows": rows})
        self.assertFalse(report["G0_pass"])
        self.assertEqual(report["budgets"][0]["oracle_headroom"], 0)
        self.assertEqual(report["budgets"][0]["n_unique_task_best_actions"], 1)

    def test_budget_can_remove_expensive_action_without_zero_filling(self):
        rows = [
            row("t1", "e1", "cache", 1, 0.6, gpu=1),
            row("t1", "e1", "reembed", 1, 0.9, gpu=20),
            row("t2", "e2", "cache", 1, 0.7, gpu=1),
            row("t2", "e2", "reembed", 1, 0.8, gpu=20),
        ]
        report = MODULE.analyze(
            {"rows": rows, "budgets": [{"name": "cheap", "max_gpu_seconds": 5}]}
        )
        budget = report["budgets"][0]
        self.assertEqual(budget["common_static_actions"], ["cache"])
        self.assertTrue(all(item["eligible_actions"] == ["cache"] for item in budget["episodes"]))

    def test_duplicate_seed_outcome_rejected(self):
        duplicated = row("t1", "e1", "A", 1, 0.8)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.analyze({"rows": [duplicated, dict(duplicated)]})

    def test_negative_regression_score_is_allowed(self):
        rows = [
            row("reg", "reg-1", "A", 1, -0.2),
            row("reg", "reg-1", "B", 1, -0.5),
        ]
        report = MODULE.analyze({"rows": rows})
        self.assertEqual(report["budgets"][0]["best_static_action"], "A")


if __name__ == "__main__":
    unittest.main()
