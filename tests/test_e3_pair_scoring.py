import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from e3_pair_scoring import ARMS, HARD_ARMS, score_run


def items(events=5, per_event=2, hard=1, landslide=False):
    out = []
    sizes = [per_event] * events if isinstance(per_event, int) else per_event
    for event in range(events):
        for tile in range(sizes[event]):
            for kind, answer in (("pos", "yes"), ("neg", "no")):
                out.append({"id": f"f{event}_{tile}_{kind}", "tile": f"f{event}_{tile}", "cluster": f"e{event}",
                            "phen": "flood", "kind": kind, "answer": answer})
        for tile in range(hard):
            out.append({"id": f"h{event}_{tile}", "tile": f"h{event}_{tile}", "cluster": f"e{event}",
                        "phen": "flood", "kind": "hard_neg", "answer": "no"})
    if landslide:
        for region in ("hiroshima", "indonesia"):
            for kind, answer in (("pos", "yes"), ("neg", "no")):
                out.append({"id": f"{region}_{kind}", "tile": region, "cluster": region,
                            "phen": "landslide", "kind": kind, "answer": answer})
    return out


def outputs(expected, fn=None, seeds=(1, 2, 3)):
    fn = fn or (lambda seed, arm, item: item["answer"])
    return [{"seed": seed, "arm": arm, "id": item["id"], "tile": item["tile"],
             "cluster": item["cluster"], "phen": item["phen"], "kind": item["kind"],
             "source_gold": item["answer"], "parsed": fn(seed, arm, item)}
            for seed in seeds for item in expected for arm in (HARD_ARMS if item["kind"] == "hard_neg" else ARMS)]


class E3ScoringTests(unittest.TestCase):
    def test_perfect_source_agreement_sufficient(self):
        expected = items(landslide=True)
        result = score_run(outputs(expected), expected)
        self.assertTrue(result["valid"])
        self.assertEqual(result["verdict"], "single_second_view_sufficient_under_intervention")
        self.assertEqual(result["sufficient_seeds"], [1, 2, 3])
        self.assertEqual(result["coverage"]["expected_rows"], 3 * (24 * 7 + 5 * 3))
        self.assertEqual(result["metrics"]["1"]["flood"]["contrasts"]["later_only"]["ci95_delta"], [0, 0])

    def test_both_later_arms_drop_is_history_sensitive(self):
        expected = items()
        rows = outputs(expected, lambda seed, arm, item: "no" if arm in ("later_only", "repeat_later") else item["answer"])
        result = score_run(rows, expected)
        self.assertEqual(result["verdict"], "history_sensitive_under_intervention")
        contrast = result["metrics"]["1"]["flood"]["contrasts"]["later_only"]
        self.assertEqual(contrast["delta"], -0.5)
        self.assertEqual(contrast["ci95_delta"], [-0.5, -0.5])
        behaviour = result["metrics"]["1"]["flood"]["behaviour"]["later_only"]
        self.assertEqual(behaviour["decision_flip"], 0.5)
        self.assertEqual(behaviour["yes_persistence_on_real_yes_positives"], 0)

    def test_two_arms_must_pass_in_same_two_seeds(self):
        expected = items()
        def predict(seed, arm, item):
            if arm == "later_only" and seed == 3:
                return "no"
            if arm == "repeat_later" and seed == 1:
                return "no"
            return item["answer"]
        result = score_run(outputs(expected, predict), expected)
        self.assertEqual(result["verdict"], "mixed_or_inconclusive")
        self.assertEqual(result["sufficient_seeds"], [2])

    def test_event_macro_is_not_tile_weighted(self):
        expected = items(per_event=[1, 1, 1, 1, 10], hard=0)
        rows = outputs(expected, lambda seed, arm, item: item["answer"] if item["cluster"] == "e4" else "no", seeds=(1,))
        result = score_run(rows, expected, seeds=(1,))
        metric = result["metrics"]["1"]["flood"]["paired"]["real"]
        self.assertAlmostEqual(metric["macro_ba"], 0.6)
        self.assertEqual(metric["event_count"], 5)
        self.assertEqual(metric["events"]["e4"]["n_tiles"], 10)
        self.assertEqual(metric["events"]["e0"]["ba"], 0.5)

    def test_hard_negatives_are_separate_descriptive_fpr(self):
        expected = items(hard=3)
        rows = outputs(expected, lambda seed, arm, item: "yes" if item["kind"] == "hard_neg" else item["answer"])
        result = score_run(rows, expected)
        flood = result["metrics"]["1"]["flood"]
        self.assertEqual(result["verdict"], "single_second_view_sufficient_under_intervention")
        self.assertEqual(flood["paired"]["real"]["macro_ba"], 1)
        self.assertEqual(flood["hard_neg"]["real"]["fpr_pooled"], 1)
        self.assertEqual(flood["hard_neg"]["real"]["events"]["e1"]["n_items"], 3)
        self.assertEqual(set(flood["hard_neg"]), set(HARD_ARMS))

    def test_fewer_than_five_events_inconclusive(self):
        expected = items(events=4)
        result = score_run(outputs(expected), expected)
        self.assertTrue(result["valid"])
        self.assertEqual(result["verdict"], "mixed_or_inconclusive")
        self.assertFalse(result["seed_decisions"]["1"]["eligible"])

    def test_real_baseline_under_point_six_inconclusive(self):
        expected = items()
        result = score_run(outputs(expected, lambda seed, arm, item: "no"), expected)
        self.assertEqual(result["verdict"], "mixed_or_inconclusive")
        self.assertEqual(result["seed_decisions"]["1"]["real_macro_ba"], 0.5)

    def test_noninferiority_boundary_point_zero_five(self):
        expected = items(per_event=10)
        def predict(seed, arm, item):
            if arm in ("later_only", "repeat_later") and item["kind"] == "pos" and item["tile"].endswith("_0"):
                return "no"
            return item["answer"]
        result = score_run(outputs(expected, predict), expected)
        self.assertEqual(result["verdict"], "single_second_view_sufficient_under_intervention")
        self.assertAlmostEqual(result["metrics"]["1"]["flood"]["contrasts"]["later_only"]["delta"], -0.05)

    def test_intervention_ba_must_itself_reach_point_six(self):
        expected = items(per_event=10)
        def predict(seed, arm, item):
            tile = int(item["tile"].split("_")[-1])
            if item["kind"] == "pos":
                return "yes" if tile < (1 if arm in ("later_only", "repeat_later") else 2) else "no"
            return "no"
        result = score_run(outputs(expected, predict), expected)
        self.assertAlmostEqual(result["metrics"]["1"]["flood"]["paired"]["real"]["macro_ba"], 0.6)
        self.assertEqual(result["verdict"], "mixed_or_inconclusive")

    def test_landslide_region_results_have_no_confidence_interval(self):
        expected = items(landslide=True)
        result = score_run(outputs(expected), expected)
        land = result["metrics"]["1"]["landslide"]
        self.assertEqual(set(land["paired"]["real"]["events"]), {"hiroshima", "indonesia"})
        self.assertIsNone(land["contrasts"]["later_only"]["ci95_delta"])
        self.assertEqual(land["contrasts"]["later_only"]["uncertainty_unit"], "descriptive_regions_only")

    def test_bootstrap_is_reproducible_under_row_order(self):
        expected = items()
        def predict(seed, arm, item):
            if arm == "later_only" and item["cluster"] in ("e1", "e2"):
                return "no"
            return item["answer"]
        rows = outputs(expected, predict, seeds=(1,))
        a = score_run(rows, expected, seeds=(1,))
        b = score_run(rows[::-1], expected[::-1], seeds=(1,))
        self.assertEqual(a["metrics"], b["metrics"])
        c = a["metrics"]["1"]["flood"]["contrasts"]["later_only"]
        self.assertEqual(c["bootstrap_draws"], 5000)
        self.assertEqual(c["bootstrap_seed"], 20260925)
        self.assertAlmostEqual(c["delta"], -0.2)

    def test_missing_duplicate_and_unexpected_rows_invalid(self):
        expected = items(); base = outputs(expected)
        extra = dict(base[0]); extra["id"] = "invented"
        for rows in (base[:-1], base + [base[0]], base + [extra]):
            with self.subTest(n=len(rows)):
                result = score_run(rows, expected)
                self.assertFalse(result["valid"])
                self.assertEqual(result["verdict"], "invalid")
                self.assertEqual(result["metrics"], {})

    def test_hard_negative_unregistered_arm_is_invalid(self):
        expected = items(); rows = outputs(expected)
        extra = dict(next(r for r in rows if r["kind"] == "hard_neg")); extra["arm"] = "reverse"
        result = score_run(rows + [extra], expected)
        self.assertFalse(result["valid"])
        self.assertEqual(result["coverage"]["unexpected_count"], 1)

    def test_mismatched_metadata_or_source_label_invalid(self):
        expected = items(); base = outputs(expected)
        for field, value in (("tile", "wrong"), ("cluster", "wrong"), ("phen", "landslide"), ("source_gold", "no"), ("kind", "neg"), ("parsed", "maybe")):
            rows = copy.deepcopy(base); rows[0][field] = value
            with self.subTest(field=field):
                self.assertFalse(score_run(rows, expected)["valid"])

    def test_parse_failure_above_one_percent_invalid(self):
        expected = items(); rows = outputs(expected)
        rows[0]["parsed"] = None
        result = score_run(rows, expected)
        self.assertFalse(result["valid"])
        self.assertTrue(any("parse failure >1%" in x for x in result["invalid_reasons"]))

    def test_exact_one_percent_parse_fail_is_allowed_and_counted_as_wrong(self):
        expected = items(per_event=10, hard=0); rows = outputs(expected, seeds=(1,))
        next(r for r in rows if r["arm"] == "later_only")["parsed"] = None
        result = score_run(rows, expected, seeds=(1,))
        self.assertTrue(result["valid"])
        metric = result["metrics"]["1"]["flood"]["paired"]["later_only"]
        self.assertAlmostEqual(metric["macro_ba"], 0.99)
        self.assertEqual(metric["parse_fail_count"], 1)
        behaviour = result["metrics"]["1"]["flood"]["behaviour"]["later_only"]
        self.assertEqual(behaviour["n_excluded_parse"], 1)

    def test_expected_pair_must_have_one_yes_and_one_no(self):
        expected = items()
        expected = [x for x in expected if x["id"] != "f0_0_neg"]
        result = score_run(outputs(expected), expected)
        self.assertFalse(result["valid"])
        self.assertTrue(any("invalid within-tile" in x for x in result["invalid_reasons"]))

    def test_duplicate_expected_ids_invalid(self):
        expected = items(); expected.append(copy.deepcopy(expected[0]))
        self.assertFalse(score_run([], expected)["valid"])

    def test_no_real_yes_has_null_persistence_not_divide_by_zero(self):
        expected = items()
        result = score_run(outputs(expected, lambda seed, arm, item: "no"), expected)
        b = result["metrics"]["1"]["flood"]["behaviour"]["later_only"]
        self.assertEqual(b["n_original_positives_answered_yes_in_real"], 0)
        self.assertIsNone(b["yes_persistence_on_real_yes_positives"])

    def test_empty_expected_invalid(self):
        self.assertFalse(score_run([], [])["valid"])


if __name__ == "__main__":
    unittest.main()
