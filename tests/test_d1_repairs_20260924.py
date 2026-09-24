"""Regression tests for the 2026-09-24 D1 repairs (amendment registered the same day).

Synthetic fixtures only: no real pack, model, GPU or server is touched. The
episode/annotation builders reuse the pattern of code/audit_d1_contract_20260924.py.
"""
import base64
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


CODE_DIR = pathlib.Path(__file__).parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
import sn7_visible_pack_v05 as pack  # noqa: E402
import sn7_evidence_loss_v0 as loss  # noqa: E402
import sn7_d1_reader_run_v0 as reader  # noqa: E402


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII=")


def episode(i):
    return {"id": f"ep-{i}", "aoi": f"AOI-{i // 2}", "region": ("NW", "SE")[i % 2],
            "cutoff": "2020-04", "reference_id": "F000",
            "frames": [{"id": f"F{j:03d}", "date": f"2020-{j + 1:02d}",
                        "path": f"frames/{i}-{j}.png"} for j in range(4)]}


def annotation(ep, who, outcome, first=1, seconds=None):
    states = {f["id"]: "no_visible_change" for f in ep["frames"]}
    if outcome == "change_supported":
        for f in ep["frames"][first:]:
            states[f["id"]] = "visible_change"
    elif outcome == "insufficient_evidence":
        states["F001"] = "ambiguous"
    ann = {"episode_id": ep["id"], "annotator_id": who, "status": "complete", "states": states}
    if seconds is not None:
        ann["seconds"] = seconds
    return ann


def export(pack_id, who, labels):
    return {"schema": "sn7-visible-annotations-v0.5", "pack_id": pack_id,
            "annotator_id": who, "annotations": labels}


def change_target(first="F001"):
    return {"answer": "change_supported", "first_change_id": first, "evidence_ids": ["F000", first],
            "current_state": "visible_change"}


NO_CHANGE_TARGET = {"answer": "no_visible_change", "first_change_id": None,
                    "evidence_ids": ["F000", "F001", "F002", "F003"],
                    "current_state": "no_visible_change"}
WITHHELD_TARGET = {"answer": "insufficient_evidence", "first_change_id": None, "evidence_ids": [],
                   "current_state": "unreadable"}


def pred(answer, first=None, state="no_visible_change", evidence=()):
    return {"answer": answer, "first_change_id": first,
            "evidence_ids": list(evidence), "current_state": state}


def abstain():
    return pred("insufficient_evidence", None, "unreadable", [])


def real_row(eid, target):
    return {"id": f"{eid}|real", "episode_id": eid, "aoi": "AOI", "condition": "real",
            "input": {"prompt": "p", "frames": [], "image_mode": "real"},
            "target": target, "real_image_reference_target": target,
            "allowed_for_model_run": False}


def withheld_row(eid, real_target, condition):
    return {"id": f"{eid}|{condition}", "episode_id": eid, "aoi": "AOI", "condition": condition,
            "input": {"prompt": "p", "frames": [], "image_mode": condition},
            "target": dict(WITHHELD_TARGET), "real_image_reference_target": real_target,
            "allowed_for_model_run": False}


def swap_row(source, donor, donor_target, kind=None):
    row = {"id": f"{source}|swap|{donor}", "episode_id": source, "aoi": "AOI",
           "condition": "different_outcome_swap", "donor_episode_id": donor,
           "input": {"prompt": "p", "frames": [], "image_mode": "real"},
           "target": donor_target, "real_image_reference_target": donor_target,
           "allowed_for_model_run": False}
    if kind is not None:
        row["control_pair_kind"] = kind
    return row


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def write_matrix(self, directory, rows, review):
        directory.mkdir(parents=True, exist_ok=True)
        self.write_jsonl(directory / "diagnostic_matrix.jsonl", rows)
        (directory / "review_report.json").write_text(json.dumps(review))

    def run_score(self, matrix_dir, answers_path):
        out = self.root / "score-out"
        with contextlib.redirect_stdout(io.StringIO()):
            reader.cmd_score(SimpleNamespace(matrix=str(matrix_dir), answers=str(answers_path),
                                             out=str(out)))
        return json.loads((out / f"scores_{answers_path.stem}.json").read_text())


GATE_PASSED_REVIEW = {"pack_id": "p", "diagnostic_manifest_ready": True,
                      "scientific_gate_passed": True, "h_gate_reason": None,
                      "agreement_rate": 1.0, "h_gate_pass": True}
GATE_FAILED_REVIEW = {"pack_id": "p", "diagnostic_manifest_ready": True,
                      "scientific_gate_passed": False,
                      "h_gate_reason": "agreement_rate 0.250 below the registered 0.60",
                      "agreement_rate": 0.25, "h_gate_pass": False}


class HGateComputationTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [episode(i) for i in range(12)]
        self.payload = {"pack_id": "synthetic-d1-repairs", "episodes": self.episodes}

    def review_with(self, exports):
        return pack.review_exports(self.payload, exports)

    def test_agreement_3_of_12_fails_gate(self):
        outcomes = ["change_supported", "no_visible_change", "insufficient_evidence"]
        exports = []
        for who in ("a", "b"):
            labels = [annotation(ep, who, outcomes[i] if i < 3 else
                                 ("change_supported" if who == "a" else "no_visible_change"))
                      for i, ep in enumerate(self.episodes)]
            exports.append(export(self.payload["pack_id"], who, labels))
        report = self.review_with(exports)
        self.assertEqual(report["n_agreed"], 3)
        self.assertAlmostEqual(report["agreement_rate"], 0.25)
        self.assertEqual(report["n_annotators"], 2)
        self.assertFalse(report["h_gate_pass"])
        self.assertFalse(report["scientific_gate_passed"])
        self.assertIn("0.60", report["h_gate_reason"])

    def test_agreement_8_of_12_with_three_categories_passes(self):
        agreed = (["change_supported"] * 4 + ["no_visible_change"] * 2
                  + ["insufficient_evidence"] * 2)
        exports = []
        for who in ("a", "b"):
            labels = [annotation(ep, who, agreed[i] if i < 8 else
                                 ("change_supported" if who == "a" else "no_visible_change"))
                      for i, ep in enumerate(self.episodes)]
            exports.append(export(self.payload["pack_id"], who, labels))
        report = self.review_with(exports)
        self.assertEqual(report["n_agreed"], 8)
        self.assertAlmostEqual(report["agreement_rate"], 8 / 12)
        self.assertEqual(report["agreed_answer_counts"], {
            "change_supported": 4, "no_visible_change": 2, "insufficient_evidence": 2})
        self.assertTrue(report["h_gate_pass"])
        self.assertTrue(report["scientific_gate_passed"])
        self.assertIsNone(report["h_gate_reason"])

    def test_single_annotator_cannot_pass_gate(self):
        labels = [annotation(ep, "solo", "change_supported") for ep in self.episodes]
        report = self.review_with([export(self.payload["pack_id"], "solo", labels)])
        self.assertEqual(report["n_agreed"], 0)
        self.assertEqual(report["n_annotators"], 1)
        self.assertFalse(report["h_gate_pass"])
        self.assertFalse(report["scientific_gate_passed"])
        self.assertIn("consensus", report["h_gate_reason"])


class FinalizeReportTests(TempDirTestCase):
    """AOI-0: two change episodes with different first dates -> temporal pair.
    AOI-1: change vs no_change -> content pair."""

    def setUp(self):
        super().setUp()
        self.episodes = [episode(i) for i in range(4)]
        pack_dir = self.root / "annotator_pack"
        hashes = {}
        for ep in self.episodes:
            for frame in ep["frames"]:
                target = pack_dir / frame["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(PNG_BYTES)
                hashes[frame["path"]] = pack.digest(target)
        self.payload = pack.make_public_pack(self.episodes, hashes)
        pack_dir.mkdir(exist_ok=True)
        self.pack_path = pack_dir / "pack.json"
        self.pack_path.write_text(json.dumps(self.payload, ensure_ascii=False))

    def _finalize(self, out_name, with_seconds):
        outcomes = {"ep-0": ("change_supported", 1), "ep-1": ("change_supported", 2),
                    "ep-2": ("change_supported", 1), "ep-3": ("no_visible_change", 1)}
        paths = []
        for who in ("a", "b"):
            labels = []
            for ep in self.episodes:
                outcome, first = outcomes[ep["id"]]
                seconds = None
                if with_seconds and ep["id"] == "ep-0":
                    seconds = 100 if who == "a" else 300
                labels.append(annotation(ep, who, outcome, first=first, seconds=seconds))
            path = self.root / f"ann-{with_seconds}-{who}.json"
            path.write_text(json.dumps(export(self.payload["pack_id"], who, labels)))
            paths.append(str(path))
        final_dir = self.root / out_name
        with contextlib.redirect_stdout(self.stdout):
            pack.finalize(SimpleNamespace(pack=str(self.pack_path), annotations=paths,
                                          output=str(final_dir)))
        return final_dir

    def test_finalize_writes_h_report_and_kinds_swap_rows(self):
        self.stdout = io.StringIO()
        final_dir = self._finalize("final-seconds", with_seconds=True)
        stdout = self.stdout.getvalue()
        self.assertIn('"h_gate_pass": false', stdout)

        h_report = json.loads((final_dir / "H_report.json").read_text())
        self.assertEqual(h_report["schema"], "sn7-visible-h-report-v0")
        self.assertEqual(h_report["pack_id"], self.payload["pack_id"])
        self.assertEqual(h_report["prereg"], "config/decision_experiment_d1_prereg_v0.json")
        self.assertEqual(h_report["amendment"],
                         "config/decision_experiment_d1_prereg_v0_amendment_20260924.json")
        self.assertEqual(h_report["n_total"], 4)
        self.assertEqual(h_report["n_agreed"], 4)
        self.assertEqual(h_report["agreement_rate"], 1.0)
        self.assertEqual(h_report["agreed_answer_counts"], {
            "change_supported": 3, "no_visible_change": 1})
        self.assertEqual(h_report["n_annotators"], 2)
        self.assertEqual(h_report["median_seconds_per_episode"], 200.0)
        self.assertFalse(h_report["h_gate_pass"])
        self.assertIn("category", h_report["h_gate_reason"])

        matrix = [json.loads(line) for line in
                  (final_dir / "diagnostic_matrix.jsonl").read_text().splitlines()]
        swap_rows = [r for r in matrix if r["condition"] == "different_outcome_swap"]
        self.assertEqual(len(swap_rows), 4)
        kinds = {(r["episode_id"], r["donor_episode_id"]): r["control_pair_kind"]
                 for r in swap_rows}
        self.assertEqual(kinds[("ep-0", "ep-1")], "temporal")
        self.assertEqual(kinds[("ep-1", "ep-0")], "temporal")
        self.assertEqual(kinds[("ep-2", "ep-3")], "content")
        self.assertEqual(kinds[("ep-3", "ep-2")], "content")

        review = json.loads((final_dir / "review_report.json").read_text())
        pair_kinds = {p["pair_kind"] for p in review["control_pairs"]}
        self.assertEqual(pair_kinds, {"content", "temporal"})

    def test_finalize_marks_median_seconds_null_when_absent(self):
        self.stdout = io.StringIO()
        final_dir = self._finalize("final-plain", with_seconds=False)
        h_report = json.loads((final_dir / "H_report.json").read_text())
        self.assertIsNone(h_report["median_seconds_per_episode"])


class ReaderRunGateTests(TempDirTestCase):
    def cmd_run_args(self):
        return SimpleNamespace(root=str(self.root), matrix=str(self.root / "matrix"),
                               pack=str(self.root), out=str(self.root / "run"), reader="qwen")

    def test_run_exits_before_loader_when_h_gate_failed(self):
        rows = [real_row("ep-0", change_target()), swap_row("ep-0", "ep-1", NO_CHANGE_TARGET)]
        self.write_matrix(self.root / "matrix", rows, GATE_FAILED_REVIEW)
        with mock.patch.object(reader, "load_reader",
                               side_effect=RuntimeError("MODEL_LOAD_SENTINEL")):
            with self.assertRaises(SystemExit) as cm:
                reader.cmd_run(self.cmd_run_args())
        self.assertIn("stage H gate not passed", str(cm.exception))

    def test_run_exits_with_preparation_incomplete_when_no_content_pair(self):
        rows = [swap_row("ep-0", "ep-1", change_target("F002"), kind="temporal")]
        self.write_matrix(self.root / "matrix", rows, GATE_PASSED_REVIEW)
        with mock.patch.object(reader, "load_reader",
                               side_effect=RuntimeError("MODEL_LOAD_SENTINEL")):
            with self.assertRaises(SystemExit) as cm:
                reader.cmd_run(self.cmd_run_args())
        self.assertEqual(str(cm.exception),
                         "preparation_incomplete: no content-different donor pair")

    def test_run_reaches_loader_and_writes_reader_specific_context(self):
        rows = [real_row("ep-0", change_target()),
                swap_row("ep-0", "ep-1", NO_CHANGE_TARGET, kind="content")]
        self.write_matrix(self.root / "matrix", rows, GATE_PASSED_REVIEW)
        with mock.patch.object(reader, "load_reader",
                               side_effect=RuntimeError("MODEL_LOAD_SENTINEL")):
            with self.assertRaises(RuntimeError):
                reader.cmd_run(self.cmd_run_args())
        run_dir = self.root / "run"
        ctx = json.loads((run_dir / "run_context_qwen.json").read_text())
        self.assertEqual(ctx["amendment"], reader.AMENDMENT)
        self.assertEqual(ctx["prereg"], reader.PREREG)
        self.assertEqual(ctx["n_content_pairs"], 1)
        self.assertEqual(ctx["h_agreement_rate"], 1.0)
        self.assertTrue(ctx["h_gate_pass"])
        self.assertFalse((run_dir / "run_context.json").exists())


class ScorePerfectAbstentionTests(TempDirTestCase):
    """A reader that answers every real/swap row correctly and properly
    abstains on metadata_only+blank must now read as reads_content."""

    def setUp(self):
        super().setUp()
        t0, t1 = change_target("F001"), NO_CHANGE_TARGET
        self.rows = [
            real_row("ep-0", t0), real_row("ep-1", t1),
            withheld_row("ep-0", t0, "metadata_only"), withheld_row("ep-1", t1, "metadata_only"),
            withheld_row("ep-0", t0, "blank"), withheld_row("ep-1", t1, "blank"),
            swap_row("ep-0", "ep-1", t1, kind="content"),
            swap_row("ep-1", "ep-0", t0, kind="content"),
        ]
        self.answers = [
            {"id": "ep-0|real", "pred": pred("change_supported", "F001", "visible_change", ["F000", "F001"])},
            {"id": "ep-1|real", "pred": pred("no_visible_change", None, "no_visible_change",
                                             ["F000", "F001", "F002", "F003"])},
            {"id": "ep-0|metadata_only", "pred": abstain()},
            {"id": "ep-1|metadata_only", "pred": abstain()},
            {"id": "ep-0|blank", "pred": abstain()},
            {"id": "ep-1|blank", "pred": abstain()},
            {"id": "ep-0|swap|ep-1", "pred": pred("no_visible_change", None, "no_visible_change",
                                                  ["F000", "F001", "F002", "F003"])},
            {"id": "ep-1|swap|ep-0", "pred": pred("change_supported", "F001", "visible_change", ["F000", "F001"])},
        ]

    def test_perfect_abstaining_reader_reads_content(self):
        matrix_dir = self.root / "matrix"
        self.write_matrix(matrix_dir, self.rows, GATE_PASSED_REVIEW)
        answers_path = self.root / "answers.jsonl"
        self.write_jsonl(answers_path, self.answers)

        scored = self.run_score(matrix_dir, answers_path)
        self.assertEqual(scored["schema"], "sn7-d1-reader-scores-v1")
        self.assertEqual(scored["amendment"], reader.AMENDMENT)
        self.assertEqual(scored["status"], "ok")
        self.assertEqual(scored["registered_reading"], "reads_content")
        self.assertEqual(scored["coincidental_match_rate"], 0.0)
        self.assertEqual(scored["proper_abstention_rate"], 1.0)
        self.assertTrue(scored["content_check"]["content_use_pass"])
        self.assertEqual(scored["content_check"]["n_content_rows"], 2)
        self.assertEqual(scored["reading_basis"]["real_answer_acc"], 1.0)
        self.assertEqual(scored["reading_basis"]["margin"], 1.0)
        self.assertEqual(scored["answers_sha256"], reader.sha256_file(answers_path))
        self.assertEqual(scored["matrix_sha256"],
                         reader.sha256_file(matrix_dir / "diagnostic_matrix.jsonl"))


class ScoreTemporalTests(TempDirTestCase):
    """Temporal swap rows are scored on joint accuracy and never enter
    content_check; this matrix deliberately omits control_pair_kind to exercise
    the legacy-matrix fallback classification."""

    def setUp(self):
        super().setUp()
        t_early, t_late = change_target("F001"), change_target("F002")
        t_change, t_no = change_target("F001"), NO_CHANGE_TARGET
        self.rows = [
            real_row("ep-0", t_early), real_row("ep-1", t_late),
            real_row("ep-2", t_change), real_row("ep-3", t_no),
            swap_row("ep-0", "ep-1", t_late),
            swap_row("ep-1", "ep-0", t_early),
            swap_row("ep-2", "ep-3", t_no),
            swap_row("ep-3", "ep-2", t_change),
        ]
        donor_pred = lambda t: pred(t["answer"], t["first_change_id"], t["current_state"],
                                    t["evidence_ids"])
        self.answers = ([{"id": f"ep-{i}|real", "pred": donor_pred(t)}
                         for i, t in enumerate((t_early, t_late, t_change, t_no))] +
                        [{"id": "ep-0|swap|ep-1", "pred": donor_pred(t_late)},
                         {"id": "ep-1|swap|ep-0", "pred": donor_pred(t_early)},
                         {"id": "ep-2|swap|ep-3", "pred": donor_pred(t_no)},
                         {"id": "ep-3|swap|ep-2", "pred": donor_pred(t_change)}])

    def test_temporal_pairs_use_joint_accuracy_and_stay_out_of_content_check(self):
        matrix_dir = self.root / "matrix"
        self.write_matrix(matrix_dir, self.rows, GATE_PASSED_REVIEW)
        answers_path = self.root / "answers.jsonl"
        self.write_jsonl(answers_path, self.answers)

        scored = self.run_score(matrix_dir, answers_path)
        temporal = scored["temporal_check"]
        self.assertEqual(temporal["n_temporal_rows"], 2)
        self.assertEqual(temporal["swap_vs_donor_joint"], 1.0)
        self.assertEqual(temporal["swap_vs_source_joint"], 0.0)
        self.assertEqual(temporal["diff"], 1.0)
        self.assertTrue(temporal["low_support"])  # 2 temporal rows < 4
        self.assertIn("descriptive only", temporal["note"])
        self.assertEqual(scored["content_check"]["n_content_rows"], 2)
        self.assertEqual(scored["content_check"]["diff"], 1.0)
        self.assertEqual(scored["first_acc_change_subset"], 1.0)


class ScoreCompletenessTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        t0, t1 = change_target("F001"), NO_CHANGE_TARGET
        self.rows = [real_row("ep-0", t0), real_row("ep-1", t1),
                     swap_row("ep-0", "ep-1", t1, kind="content"),
                     swap_row("ep-1", "ep-0", t0, kind="content")]
        self.answers = [
            {"id": "ep-0|real", "pred": pred("change_supported", "F001", "visible_change", ["F000", "F001"])},
            {"id": "ep-1|real", "pred": pred("no_visible_change", None, "no_visible_change",
                                             ["F000", "F001", "F002", "F003"])},
            {"id": "ep-0|swap|ep-1", "pred": pred("no_visible_change", None, "no_visible_change",
                                                  ["F000", "F001", "F002", "F003"])},
            {"id": "ep-1|swap|ep-0", "pred": pred("change_supported", "F001", "visible_change", ["F000", "F001"])},
        ]
        self.matrix_dir = self.root / "matrix"
        self.write_matrix(self.matrix_dir, self.rows, GATE_PASSED_REVIEW)
        self.answers_path = self.root / "answers.jsonl"

    def test_missing_duplicate_and_unknown_ids_are_rejected(self):
        self.write_jsonl(self.answers_path, self.answers[:3])
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                reader.cmd_score(SimpleNamespace(matrix=str(self.matrix_dir),
                                                 answers=str(self.answers_path),
                                                 out=str(self.root / "out")))
        self.assertIn("missing 1", str(cm.exception))

        self.write_jsonl(self.answers_path, self.answers + [self.answers[0]])
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                reader.cmd_score(SimpleNamespace(matrix=str(self.matrix_dir),
                                                 answers=str(self.answers_path),
                                                 out=str(self.root / "out")))
        self.assertIn("duplicate 1", str(cm.exception))

        self.write_jsonl(self.answers_path,
                         self.answers + [{"id": "ghost|real", "pred": abstain()}])
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                reader.cmd_score(SimpleNamespace(matrix=str(self.matrix_dir),
                                                 answers=str(self.answers_path),
                                                 out=str(self.root / "out")))
        self.assertIn("unknown 1", str(cm.exception))

    def test_run_context_matrix_hash_mismatch_is_rejected(self):
        self.write_jsonl(self.answers_path, self.answers)
        (self.root / "run_context_qwen.json").write_text(json.dumps(
            {"matrix_sha256": "0" * 64}))
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                reader.cmd_score(SimpleNamespace(matrix=str(self.matrix_dir),
                                                 answers=str(self.answers_path),
                                                 out=str(self.root / "out")))
        self.assertIn("different matrix", str(cm.exception))

    def test_matching_run_context_hash_is_accepted(self):
        self.write_jsonl(self.answers_path, self.answers)
        matrix_hash = reader.sha256_file(self.matrix_dir / "diagnostic_matrix.jsonl")
        (self.root / "run_context_qwen.json").write_text(json.dumps(
            {"matrix_sha256": matrix_hash}))
        scored = self.run_score(self.matrix_dir, self.answers_path)
        self.assertEqual(scored["matrix_sha256"], matrix_hash)


class EvidenceLossGateTests(TempDirTestCase):
    def _argv(self, review):
        pack_dir = self.root / "pack"
        pack_dir.mkdir(exist_ok=True)
        (pack_dir / "pack.json").write_text(json.dumps(
            {"pack_id": "p", "episodes": [episode(0), episode(1)]}))
        review_dir = self.root / "review"
        review_dir.mkdir(exist_ok=True)
        (review_dir / "review_report.json").write_text(json.dumps(review))
        return ["sn7_evidence_loss_v0.py", "--pack", str(pack_dir),
                "--review", str(review_dir), "--out", str(self.root / "out")]

    def test_loss_runner_refuses_failed_h_gate(self):
        review = {"pack_id": "p", "targets": {"ep-0": change_target()},
                  "scientific_gate_passed": False,
                  "h_gate_reason": "agreement_rate 0.000 below the registered 0.60",
                  "diagnostic_manifest_ready": True}
        with mock.patch.object(sys, "argv", self._argv(review)):
            with self.assertRaises(SystemExit) as cm:
                loss.main()
        self.assertIn("stage H gate not passed", str(cm.exception))

    def test_loss_runner_refuses_when_manifest_not_ready(self):
        review = {"pack_id": "p", "targets": {"ep-0": change_target()},
                  "scientific_gate_passed": True, "h_gate_reason": None,
                  "diagnostic_manifest_ready": False}
        with mock.patch.object(sys, "argv", self._argv(review)):
            with self.assertRaises(SystemExit) as cm:
                loss.main()
        self.assertIn("diagnostic_manifest_ready", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
