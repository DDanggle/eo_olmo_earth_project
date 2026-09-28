import ast
import copy
import math
from pathlib import Path
import struct
import unittest

from yes_no_scoring import analyze_historical_record, binary_decision, parse_first_token, score_answers

IDS = {"yes": 9891, "no": 2201}


def record(**updates):
    value = {"raw_token": "no", "parsed": "no", "eval_gold": "yes", "argmax_token_id": 2201,
             "yes_logit": 27.0, "no_logit": 27.0,
             "constrained_yes_probability": .5, "constrained_prediction": "yes"}
    value.update(updates)
    return value


class ScoringPolicyTests(unittest.TestCase):
    def test_actual_token_ids_tie_choose_no(self):
        result = binary_decision(27., 27., IDS)
        self.assertEqual(result["prediction"], "no")
        self.assertTrue(result["exact_logit_tie"])

    def test_token_id_order_not_class_name_determines_tie(self):
        self.assertEqual(binary_decision(0., 0., {"yes": 3, "no": 7})["prediction"], "yes")

    def test_non_ties_ignore_token_id_order(self):
        self.assertEqual(binary_decision(3., 2., IDS)["prediction"], "yes")
        self.assertEqual(binary_decision(2., 3., {"yes": 1, "no": 8})["prediction"], "no")

    def test_small_positive_difference_probability_rounds_to_half(self):
        result = binary_decision(1e-20, 0., IDS)
        self.assertEqual(result["yes_probability"], .5)
        self.assertFalse(result["exact_logit_tie"])
        self.assertEqual(result["prediction"], "yes")

    def test_small_negative_difference_probability_rounds_to_half(self):
        result = binary_decision(-1e-20, 0., {"yes": 1, "no": 8})
        self.assertEqual(result["yes_probability"], .5)
        self.assertFalse(result["exact_logit_tie"])
        self.assertEqual(result["prediction"], "no")

    def test_saved_float32_probability_half_is_not_logit_tie(self):
        calculated = binary_decision(-1e-8, 0., IDS)["yes_probability"]
        rounded = struct.unpack("f", struct.pack("f", calculated))[0]
        self.assertEqual(rounded, .5)
        result = analyze_historical_record(record(yes_logit=-1e-8, no_logit=0., constrained_yes_probability=rounded), IDS)
        self.assertFalse(result["exact_logit_tie"])
        self.assertTrue(result["saved_probability_half_without_logit_tie"])
        self.assertTrue(result["constrained_label_changes"])

    def test_representational_tie_does_not_claim_lost_precision_recovery(self):
        self.assertEqual(1e16 + 1, 1e16)
        self.assertTrue(binary_decision(1e16 + 1, 1e16, IDS)["exact_logit_tie"])

    def test_large_common_offset_near_tie_remains_distinguishable(self):
        large = 1e308
        smaller = math.nextafter(large, 0.)
        self.assertGreater(large, smaller)
        result = binary_decision(smaller, large, {"yes": 1, "no": 8})
        self.assertEqual(result["prediction"], "no")
        self.assertFalse(result["exact_logit_tie"])

    def test_extreme_finite_logits_no_exp_overflow(self):
        self.assertEqual(binary_decision(1e308, -1e308, IDS)["yes_probability"], 1.)
        self.assertEqual(binary_decision(-1e308, 1e308, IDS)["yes_probability"], 0.)

    def test_signed_zero_is_exact_tie(self):
        self.assertTrue(binary_decision(0., -0., IDS)["exact_logit_tie"])

    def test_nonfinite_and_wrong_numeric_types_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                binary_decision(value, 0., IDS)

    def test_invalid_canonical_token_ids_rejected(self):
        for ids in ({"yes": 1, "no": 1}, {"yes": -1, "no": 2}, {"yes": True, "no": 2}, {"yes": 1}, {"yes": 1, "no": 2, "maybe": 3}):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                binary_decision(0., 0., ids)

    def test_whole_first_token_parser(self):
        self.assertEqual(parse_first_token(" No\n"), "no")
        self.assertEqual(parse_first_token("YES"), "yes")
        for text in ("yes.", "I think no", "yesterday", "**", "yes no", ""):
            self.assertIsNone(parse_first_token(text))

    def test_invalid_negative_stays_wrong_in_denominator(self):
        result = score_answers(["yes", "no"], ["yes", None])
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertEqual(result["recall_by_label"]["no"], 0.)
        self.assertEqual(result["invalid_count"], 1)

    def test_empty_or_missing_class_ba_unavailable(self):
        self.assertIsNone(score_answers([], [])["balanced_accuracy"])
        self.assertIsNone(score_answers(["yes"], ["yes"])["balanced_accuracy"])

    def test_invalid_metric_input_rejected(self):
        for golds, predictions in ((["yes"], []), (["unknown"], ["yes"]), (["no"], ["NO"])):
            with self.assertRaises(ValueError):
                score_answers(golds, predictions)

    def test_actual_legacy_tie_reproduced_without_mutating_record(self):
        original = record(); saved = copy.deepcopy(original)
        result = analyze_historical_record(original, IDS)
        self.assertEqual(original, saved)
        self.assertEqual(result["primary_unchanged"], "no")
        self.assertEqual(result["historical_constrained"], "yes")
        self.assertEqual(result["prospective_constrained"], "no")

    def test_alternate_case_token_can_legitimately_disagree(self):
        value = record(raw_token="No", argmax_token_id=999, yes_logit=2., no_logit=1.,
                       constrained_yes_probability=1/(1+math.exp(-1)))
        result = analyze_historical_record(value, IDS)
        self.assertFalse(result["exact_logit_tie"])
        self.assertFalse(result["full_vocabulary_argmax_is_canonical"])
        self.assertTrue(result["prospective_binary_differs_from_primary"])

    def test_invalid_full_vocabulary_answer_is_not_replaced(self):
        result = analyze_historical_record(record(raw_token="**", parsed=None, argmax_token_id=999), IDS)
        self.assertIsNone(result["primary_unchanged"])
        self.assertEqual(result["prospective_constrained"], "no")

    def test_corrupted_saved_record_rejected(self):
        for updates in ({"parsed": "yes"}, {"constrained_yes_probability": .8},
                        {"constrained_prediction": "no"}, {"argmax_token_id": 9891}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                analyze_historical_record(record(**updates), IDS)

    def test_staged_trainer_uses_helper_and_preserves_primary(self):
        source = Path(__file__).with_name("train_pilot_scoring_v1_1.py").read_text()
        tree = ast.parse(source)
        record_dicts = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                        and any(isinstance(key, ast.Constant) and key.value == "constrained_prediction" for key in node.keys)]
        self.assertEqual(len(record_dicts), 1)
        pairs = {key.value: value for key, value in zip(record_dicts[0].keys, record_dicts[0].values) if isinstance(key, ast.Constant)}
        self.assertIn("binary_decision", ast.unparse(pairs["constrained_prediction"]))
        self.assertEqual(ast.unparse(pairs["parsed"]), "normalized if normalized in ('yes', 'no') else None")
        self.assertIn('winners = logits.argmax(dim=-1).tolist()', source)
        self.assertIn('out.mkdir(parents=True, exist_ok=False)', source)
        self.assertIn('"output_revision": "scoring_v1_1"', source)
        self.assertIn('Path(__file__).with_name("yes_no_scoring.py")', source)


if __name__ == "__main__":
    unittest.main()
