import copy
import unittest

from audit_prepared_receipts import expected_donors, score
from audit_results import check_prediction


class ReceiptAuditTests(unittest.TestCase):
    def test_invalid_predictions_stay_in_denominator(self):
        result = score(["yes", "yes", "no", "no"], ["yes", None, "yes", "no"])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["balanced_accuracy"], 0.5)
        self.assertEqual(result["invalid_count"], 1)
        self.assertEqual(result["support"], {"yes": 2, "no": 2})

    def test_single_class_has_no_balanced_accuracy(self):
        self.assertIsNone(score(["yes", "yes"], ["yes", None])["balanced_accuracy"])

    def donor_fixture(self):
        source = {"id": "a", "input": "Forest?", "output": "yes", "patch_id": "patch-a", "date": "2018-01-01", "mgrs": "tile-a", "query_class": "forest"}
        donor = dict(source, id="b", output="no", patch_id="patch-b", date="2018-02-01", mgrs="tile-b")
        record = {"arm": "joint", "control": "observation_swap", "id": "a", "patch_id": "patch-a", "presented_id": "b", "presented_patch_id": "patch-b", "presented_date": "2018-02-01", "source_gold": "yes", "eval_gold": "no", "raw_token": " No", "parsed": "no", "yes_logit": 0.0, "no_logit": 0.0, "constrained_yes_probability": 0.5, "constrained_prediction": "yes", "image_tokens_used": True}
        return {"a": source, "b": donor}, record

    def test_swap_uses_donor_gold_and_exact_question(self):
        items, record = self.donor_fixture()
        donors = expected_donors(list(items.values()))
        checked = check_prediction(record, items, donors)
        self.assertEqual(score([checked["eval_gold"]], [checked["parsed"]])["accuracy"], 1.0)
        altered = copy.deepcopy(record)
        altered["eval_gold"] = "yes"
        with self.assertRaisesRegex(ValueError, "source/donor gold"):
            check_prediction(altered, items, donors)
        items["b"]["input"] = "Water?"
        self.assertEqual(expected_donors(list(items.values())), {})

    def test_raw_token_is_authoritative(self):
        items, record = self.donor_fixture()
        record["raw_token"] = "maybe"
        with self.assertRaisesRegex(ValueError, "raw token"):
            check_prediction(record, items, expected_donors(list(items.values())))


if __name__ == "__main__":
    unittest.main()
