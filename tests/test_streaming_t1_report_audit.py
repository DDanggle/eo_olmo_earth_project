import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("t1_audit", Path(__file__).parents[1]/"code/audit_streaming_t1_reports.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class T1ReportAuditTests(unittest.TestCase):
    def report(self, fold="holdout_hiroshima", module="gru", seed=1):
        metric = lambda ap: {"auprc_exact": ap, "positive_patch_macro_iou": .2}
        return {"schema":"streaming-update-train-v0", "fold":fold, "module":module, "seed":seed,
            "params":3541248, "n":{"train":600,"val":290,"test":862}, "best_val_epoch":29, "train_s":100,
            "agreement_c12":{"student":{"cosine":.94,"rel_mse":.12}},
            "downstream_c12":{"teacher_full_reencode":metric(.55),"frozen_m4":metric(.01),
                               "student":metric(.52),"singles_mean":metric(.19)}}

    def test_recovery_is_gap_fraction(self):
        self.assertAlmostEqual(m.recovery(.52,.01,.55), .51/.54)

    def test_no_headroom_is_not_divided_by_epsilon(self):
        self.assertIsNone(m.recovery(.3,.4,.4))
        self.assertIsNone(m.recovery(.3,.4,.2))

    def test_two_seeds_not_complete(self):
        out=m.analyze([(str(s),self.report(seed=s),"hash") for s in (1,2)])
        self.assertEqual(out["v0_gate"]["status"],"INCOMPLETE_NO_FINAL_GATE")
        self.assertIsNone(out["v0_gate"]["residual_v0_pass"])

    def test_duplicate_seed_is_error(self):
        out=m.analyze([("a",self.report(),"hash"),("b",self.report(),"hash")])
        self.assertTrue(out["errors"])

    def test_inconsistent_reference_is_error(self):
        other=self.report(seed=2)
        other["downstream_c12"]["teacher_full_reencode"]["auprc_exact"] = .6
        self.assertTrue(m.analyze([("a",self.report(),"hash"),("b",other,"hash")])["errors"])

    def test_invalid_ap_rejected(self):
        for bad in (None, float("nan"), float("inf"), -1, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError): m.recovery(bad,.01,.55)

    def test_complete_reference_does_not_mean_ours_passed(self):
        docs=[]
        for f in m.FOLDS:
            for mod in m.MODULES:
                for s in m.SEEDS: docs.append((f"{f}-{mod}-{s}",self.report(f,mod,s),"hash"))
        gate=m.analyze(docs)["v0_gate"]
        self.assertEqual(gate["status"],"COMPLETE_DEVELOPMENT_NOT_CONFIRMATORY")
        self.assertFalse(gate["residual_v0_pass"])
        self.assertTrue(gate["utility_90pct_by_module"]["gru"])

    def test_split_count_mismatch_rejected(self):
        self.assertTrue(m.analyze([("a",self.report(),"hash")],{"holdout_hiroshima":{"train":[],"val":[],"test":[]}})["errors"])

    def test_manifest_overlap_is_error(self):
        result=m.analyze([],{"holdout_hiroshima":{"train":["same_s2_1"],"val":[],"test":["same_s2_1"]}})
        self.assertTrue(result["errors"])

    def test_completed_residual_can_fail_necessary_condition_while_others_pending(self):
        docs=[]
        for f in m.FOLDS:
            for s in m.SEEDS:
                r=self.report(f,"residual",s)
                r["downstream_c12"]["student"]["auprc_exact"] = .3
                docs.append((f"{f}-{s}",r,"hash"))
        gate=m.analyze(docs)["v0_gate"]
        self.assertFalse(gate["residual_90pct_in_both_folds_prerequisite"])
        self.assertIsNone(gate["residual_v0_pass"])


if __name__ == "__main__": unittest.main()
