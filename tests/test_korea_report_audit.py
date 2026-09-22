import ast
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kr_audit", ROOT / "code/audit_korea_3task_results.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class KoreaAuditTests(unittest.TestCase):
    def test_hypergeometric(self):
        self.assertAlmostEqual(audit.zero_positive_probability(100, 10, 5), math.comb(90, 5) / math.comb(100, 5))

    def test_no_positive_population(self):
        self.assertEqual(audit.zero_positive_probability(100, 0, 20), 1)

    def test_all_positive_population(self):
        self.assertEqual(audit.zero_positive_probability(100, 100, 5), 0)

    def test_invalid_population(self):
        with self.assertRaises(ValueError):
            audit.zero_positive_probability(100, 101, 5)

    def test_same_steps_different_full_exposure(self):
        self.assertEqual((4000 * 32) / (4000 * 16), 2)

    def test_rounding_loses_real_win(self):
        self.assertGreater(0.000149, 0.000101)
        self.assertEqual(round(0.000149, 4), round(0.000101, 4))

    def test_edge_padding_changes_temporal_mean(self):
        self.assertEqual(np.mean([1., 3.]), 2.)
        self.assertEqual(np.mean(np.pad([1., 3.], (0, 2), mode="edge")), 2.5)

    def test_actual_ap_function_ties_and_no_positives(self):
        # Load only the pure function, never the GPU-bound trainer at import time.
        tree = ast.parse((ROOT / "code/korea_3task_pipeline.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "exact_ap")
        env = {"np": np}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "exact_ap_from_production", "exec"), env)
        self.assertEqual(env["exact_ap"](np.array([0.5,0.5]), np.array([1,0])), 0.5)
        self.assertIsNone(env["exact_ap"](np.array([0.5,0.5]), np.array([0,0])))

    def test_actual_iou_function_omits_false_positive_only_class(self):
        tree = ast.parse((ROOT / "code/korea_3task_pipeline.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
        # Two valid pixels of class 0, one predicted class 1. Class 1 IoU=0 should
        # contribute under union-present mIoU, but the current function omits it.
        env = {"np": np, "TASK": {"land_cover": ("multi",7)}, "LC": [10,20,30,40,50,60,100],
               "load_mask": lambda c: np.array([[0,0]]), "target": lambda m,t:m,
               "exact_ap": lambda p,y: 0.5}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "evaluate_from_production", "exec"), env)
        p = np.zeros((1,7,1,2)); p[0,0,0,0] = 1; p[0,1,0,1] = 1
        r = env["evaluate"]("land_cover", p, [{"cluster":"C0"}])
        self.assertEqual(r["cluster_macro_miou"], 0.5)
        self.assertEqual(np.mean([0.5,0.0]), 0.25)


if __name__ == "__main__":
    unittest.main()
