import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("progress", Path(__file__).parents[1]/"code/audit_streaming_progress.py")
progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progress)


class ProgressAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/"kurosiwo_s1_cache").mkdir()
        (self.root/"kurosiwo_s1_cache/meta.jsonl").write_text("\n".join(json.dumps({"id":s,"split":s,"actid":i}) for i,s in enumerate(("train","validation","test"))))
        (self.root/"logs").mkdir()
        (self.root/"logs/ks_update_gru_s1.log").write_text("epoch 1 val nan best 1000000000@0\nDONE update 1000000000\n")

    def put(self, fold="holdout_hiroshima", arm="gru", seed=1, ap=.5, teacher=.6, stale=.1, nan=False):
        folder = "streaming_t1arch" if arm in ("gru_dt","gru_sp","xattn") else "streaming_t1v"
        path = self.root/"artifacts"/folder/f"{fold}_{arm}_{seed}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fold":fold,"module":arm,"seed":seed,"params":3541248,
            "history":[{"train":float("nan") if nan else .1,"val":.2}],"best_val_epoch":1,"best_val_loss":.2,
            "n":{"train":600,"val":100,"test":100}, "downstream_c12":{
                "teacher_full_reencode":{"auprc_exact":teacher},"frozen_m4":{"auprc_exact":stale},
                "student":{"auprc_exact":ap}}}))

    def test_partial_never_becomes_complete(self):
        self.put()
        r=progress.summarize(self.root)
        row=next(x for x in r["rows"] if x["fold"]=="holdout_hiroshima" and x["arm"]=="gru")
        self.assertFalse(row["complete"])
        self.assertIsNone(r["architecture_screen"][0]["final_gate"])

    def test_recovery_over_one_is_not_clipped(self):
        for seed in (1,2,3): self.put(seed=seed,ap=.7)
        r=progress.summarize(self.root)
        self.assertAlmostEqual(r["rows"][0]["recovery"],1.2)

    def test_nonpositive_gap_is_undefined(self):
        self.put(teacher=.1, stale=.2)
        self.assertIsNone(progress.summarize(self.root)["rows"][0]["recovery"])

    def test_nonfinite_training_is_rejected(self):
        self.put(nan=True)
        r=progress.summarize(self.root)
        self.assertEqual(r["baseline_reports"],0)
        self.assertTrue(r["errors"])

    def test_reference_mismatch_is_rejected(self):
        self.put(seed=1)
        self.put(seed=2,teacher=.7)
        self.assertTrue(progress.summarize(self.root)["errors"])

    def test_two_completed_failures_bound_remaining_wins(self):
        for fold in progress.FOLDS[:2]:
            for seed in (1,2,3):
                self.put(fold, "gru", seed)
                self.put(fold, "gru_dt", seed,ap=.51)
        g=progress.summarize(self.root)["architecture_screen"][0]
        self.assertEqual(g["max_possible_wins"],2)
        self.assertFalse(g["three_of_four_still_possible"])
        self.assertIsNone(g["final_gate"])

    def test_done_with_nan_is_invalid_not_method_negative(self):
        r=progress.summarize(self.root)["kuro_gru_seed1"]
        self.assertTrue(r["says_done"])
        self.assertEqual(r["nan_epoch_count"],1)
        self.assertFalse(r["valid_scientific_negative"])


if __name__=="__main__": unittest.main()
