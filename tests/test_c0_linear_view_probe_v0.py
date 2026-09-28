"""Independent synthetic tests for C0; never loads the real EO cache."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import c0_linear_view_probe_v0 as c0


def metric_fixture():
    items, pred = [], []
    for cluster, count, correct in (("large", 9, True), ("small", 1, False)):
        for kind, yes in (("pos", True), ("neg", False)):
            for i in range(count):
                items.append({"id": f"{cluster}_{kind}_{i}", "tile": f"{cluster}_{i}",
                              "cluster": cluster, "phen": "flood", "kind": kind,
                              "answer": "yes" if yes else "no"})
                pred.append(yes if correct else not yes)
    for cluster, count, predicted in (("large", 3, True), ("small", 1, False)):
        for i in range(count):
            items.append({"id": f"{cluster}_hard_{i}", "tile": f"{cluster}_hard_{i}",
                          "cluster": cluster, "phen": "flood", "kind": "hard_neg", "answer": "no"})
            pred.append(predicted)
    return items, np.asarray(pred, dtype=bool)


class SpatialMeanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "tokens.npy"

    def tearDown(self):
        self.tmp.cleanup()

    def test_float32_spatial_mean_and_order(self):
        values = np.arange(2 * 768 * 2 * 3, dtype=np.float32).reshape(2, 768, 2, 3) / 100
        values = values.astype(np.float16)
        np.save(self.path, values)
        out = c0.spatial_mean(self.path, values.shape)
        expected = values.astype(np.float32).mean(axis=(-2, -1), dtype=np.float32)
        self.assertEqual(out.shape, (2, 768))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_missing_file_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            c0.spatial_mean(self.path, (2, 768, 2, 3))

    def test_shape_mismatch_is_rejected(self):
        np.save(self.path, np.zeros((2, 768, 2, 3), dtype=np.float32))
        with self.assertRaises((ValueError, RuntimeError)):
            c0.spatial_mean(self.path, (3, 768, 2, 3))

    def test_nonfinite_is_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                values = np.ones((2, 768, 2, 3), dtype=np.float32)
                values[1, 3, 1, 2] = bad
                np.save(self.path, values)
                with self.assertRaises((ValueError, RuntimeError)):
                    c0.spatial_mean(self.path, values.shape)

    def test_channel_contract_is_checked(self):
        values = np.zeros((2, 12, 2, 3), dtype=np.float32)
        np.save(self.path, values)
        with self.assertRaises((ValueError, RuntimeError)):
            c0.spatial_mean(self.path, values.shape)


class IndexTests(unittest.TestCase):
    def test_flood_exact_slot_mapping(self):
        for slots, expected in ((["pre_1", "pre_2"], [0, 1]),
                                (["pre_2", "post"], [1, 2]),
                                (["pre_1", "post"], [0, 2])):
            with self.subTest(slots=slots):
                self.assertEqual(c0.indices({"phen": "flood", "slots": slots}, {}), expected)

    def test_unknown_flood_slot_is_rejected(self):
        with self.assertRaises((KeyError, ValueError, RuntimeError)):
            c0.indices({"phen": "flood", "slots": ["pre1", "post"]}, {})

    def test_landslide_uses_clear_top12_restored_to_time_order(self):
        # Quality ties must retain earlier source indices, then restore chronology.
        qualities = [0.4, 0.9, 0.8, 0.7, 0.4, 0.6, 0.5, 0.9, 0.8, 0.7, 0.4, 0.6, 0.5, 0.4, 0.4]
        times = [f"2020-01-{i + 1:02d}T00:00:00Z" for i in range(15)]
        rec = {"tile": {"scl_clear_fraction": qualities, "times": times}}
        item = {"phen": "landslide", "tile": "tile", "dates": ["2020-01-05", "2020-01-13"]}
        # Retained indices are 0..9,11,12; source index4 => kept index4, source12 => kept11.
        self.assertEqual(c0.indices(item, rec), [4, 11])

    def test_landslide_discarded_date_is_rejected(self):
        rec = {"tile": {"scl_clear_fraction": list(range(15)),
                        "times": [f"2020-01-{i + 1:02d}" for i in range(15)]}}
        item = {"phen": "landslide", "tile": "tile", "dates": ["2020-01-01", "2020-01-15"]}
        with self.assertRaises((ValueError, RuntimeError)):
            c0.indices(item, rec)


class StandardizeTests(unittest.TestCase):
    def test_uses_only_train_statistics(self):
        train = np.array([[1., 7.], [3., 7.], [5., 7.]])
        test = np.array([[10000., -900.]])
        ztrain, ztest, mean, scale = c0.standardize(train, test)
        np.testing.assert_allclose(mean, [3., 7.])
        np.testing.assert_allclose(ztrain.mean(axis=0), [0., 0.], atol=1e-12)
        np.testing.assert_allclose(ztrain[:, 0].std(), 1.)
        self.assertEqual(scale[1], 1.)
        np.testing.assert_allclose(ztest, (test - mean) / scale)
        _, _, mean2, scale2 = c0.standardize(train, -test)
        np.testing.assert_array_equal(mean, mean2)
        np.testing.assert_array_equal(scale, scale2)

    def test_does_not_modify_inputs(self):
        train = np.array([[1., 2.], [3., 5.]])
        test = np.array([[8., 13.]])
        saved = (train.copy(), test.copy())
        c0.standardize(train, test)
        np.testing.assert_array_equal(train, saved[0])
        np.testing.assert_array_equal(test, saved[1])

    def test_nonfinite_is_rejected(self):
        for train, test in (([[np.nan], [1.]], [[1.]]), ([[0.], [1.]], [[np.inf]])):
            with self.subTest(train=train, test=test), self.assertRaises((ValueError, RuntimeError)):
                c0.standardize(np.asarray(train), np.asarray(test))


class LogisticTests(unittest.TestCase):
    def test_single_class_is_rejected(self):
        train = np.array([[-1.], [1.]])
        with self.assertRaises((ValueError, RuntimeError)):
            c0.fit_logistic(train, np.array([1., 1.]), train)

    def test_separable_fit_is_deterministic_without_seed_search(self):
        train = np.array([[-3., -1.], [-2., -1.], [-1., -1.], [1., 1.], [2., 1.], [3., 1.]])
        labels = np.array([0., 0., 0., 1., 1., 1.])
        test = np.array([[-4., -1.], [4., 1.]])
        np.random.seed(3)
        logits1, model1, diagnostics1 = c0.fit_logistic(train, labels, test)
        np.random.seed(999)
        logits2, model2, diagnostics2 = c0.fit_logistic(train, labels, test)
        np.testing.assert_array_equal(logits1, logits2)
        self.assertEqual(logits1.shape, (2,))
        self.assertTrue(np.isfinite(logits1).all())
        np.testing.assert_array_equal(logits1 > 0, [False, True])
        self.assertTrue(diagnostics1["converged"])
        self.assertTrue(diagnostics2["converged"])
        self.assertLessEqual(diagnostics1["gradient_inf"], 1e-5)
        self.assertLessEqual(diagnostics1["final_loss"], diagnostics1["initial_loss"])
        np.testing.assert_array_equal(model1["weights"], model2["weights"])
        self.assertEqual(model1["bias"], model2["bias"])


class EventScoresTests(unittest.TestCase):
    def test_event_macro_does_not_weight_large_event_more(self):
        items, pred = metric_fixture()
        out = c0.event_scores(items, pred)
        self.assertAlmostEqual(out["event_macro_ba"], 0.5)
        self.assertAlmostEqual(out["events"]["large"]["ba"], 1.0)
        self.assertAlmostEqual(out["events"]["small"]["ba"], 0.0)
        self.assertEqual(out["events"]["large"]["n"], 18)
        self.assertEqual(out["events"]["small"]["n"], 2)

    def test_hard_negatives_have_separate_denominator(self):
        items, pred = metric_fixture()
        out = c0.event_scores(items, pred)
        hard = out["hard_negative"]
        self.assertEqual(hard["n"], 4)
        self.assertAlmostEqual(hard["pooled_fpr"], 0.75)
        self.assertAlmostEqual(hard["event_macro_fpr"], 0.5)
        # Deliberately wrong hard negatives cannot contaminate standard BA/FPR.
        self.assertEqual(out["events"]["large"]["neg"], 9)
        self.assertAlmostEqual(out["events"]["large"]["fpr"], 0.)

    def test_land_delta_has_no_invented_event_confidence_interval(self):
        items, pred = metric_fixture()
        for item in items:
            item["phen"] = "landslide"
        score = c0.event_scores(items, pred)
        result = c0.compare_later_pair(score, score, "landslide")
        self.assertEqual(result["delta"], 0.)
        self.assertIsNone(result["ci95_delta"])

    def test_hard_negative_only_event_is_retained_separately(self):
        items, pred = metric_fixture()
        items.append({"id": "dry_hard", "tile": "dry_tile", "cluster": "dry",
                      "phen": "flood", "kind": "hard_neg", "answer": "no"})
        out = c0.event_scores(items, np.append(pred, True))
        self.assertNotIn("dry", out["events"])
        self.assertIn("dry", out["hard_negative"]["events"])
        self.assertEqual(out["hard_negative"]["n"], 5)

    def test_single_class_paired_event_is_rejected(self):
        item = {"id": "pos", "tile": "tile", "cluster": "event", "phen": "flood",
                "kind": "pos", "answer": "yes"}
        with self.assertRaises((ValueError, RuntimeError)):
            c0.event_scores([item], np.array([True]))

    def test_missing_event_in_comparison_is_rejected(self):
        items, pred = metric_fixture()
        full = c0.event_scores(items, pred)
        keep = [i for i, it in enumerate(items) if it["cluster"] == "large"]
        incomplete = c0.event_scores([items[i] for i in keep], pred[keep])
        with self.assertRaises((ValueError, RuntimeError)):
            c0.compare_later_pair(full, incomplete, "flood")


if __name__ == "__main__":
    unittest.main()
