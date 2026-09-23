"""Integration tests for the v0.5 visible-evidence review pack.

The PNG fixtures below are tiny synthetic byte files.  These tests exercise
selection, integrity, annotation, and target plumbing; they do *not* establish
that any real satellite image is visually readable.
"""

import base64
import copy
import json
import pathlib
import sys
import tempfile
import types
import unittest


CODE_DIR = pathlib.Path(__file__).parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))
import sn7_visible_pack_v05 as pack  # noqa: E402


# Valid 1x1 PNG used only to exercise byte copying and hash verification.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
    "AScY42YAAAAASUVORK5CYII="
)
MONTHS = ("2020-01", "2020-02", "2020-03", "2020-04")
QUADS = ("NW", "NE", "SW", "SE")


class VisiblePackIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.source_dir = self.root / "source_frames"
        self.source_dir.mkdir()
        self.rows = self._make_source_rows()
        self.source_items = self.root / "items.jsonl"
        self._write_rows(self.rows)

    def tearDown(self):
        self.temporary.cleanup()

    def _make_source_rows(self):
        rows = []
        for aoi in ("AOI-A", "AOI-B", "AOI-C"):
            png = {}
            for month in MONTHS:
                png[month] = {}
                for quadrant in QUADS:
                    source = self.source_dir / f"{aoi}_{month}_{quadrant}.png"
                    source.write_bytes(PNG_BYTES)
                    png[month][quadrant] = str(source)
            for cutoff_index in (2, 3):
                cutoff = MONTHS[cutoff_index]
                rows.append({
                    "id": f"{aoi}|{cutoff}|Q2|NW",
                    "aoi": aoi,
                    "region": "LEGACY_REGION_MUST_BE_IGNORED",
                    "cutoff": cutoff,
                    "gold": "LEGACY_SECRET_GOLD",
                    "answer": "LEGACY_SECRET_ANSWER",
                    "conds": {
                        "full_prefix": list(MONTHS[:cutoff_index + 1]),
                        "privileged_silver": ["2099-12"],
                        "privileged_wrongcontent": ["2099-12"],
                    },
                    # Includes a source image after the earlier cutoff.  It
                    # must not enter an episode selected from that cutoff.
                    "png": copy.deepcopy(png),
                })
        return rows

    def _write_rows(self, rows=None):
        rows = self.rows if rows is None else rows
        self.source_items.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _build(self, name="built", max_episodes=4):
        output = self.root / name
        args = types.SimpleNamespace(
            source_items=str(self.source_items),
            output=str(output),
            max_episodes=max_episodes,
        )
        pack.build(args)
        pack_path = output / "annotator_pack" / "pack.json"
        return output, pack_path, pack.load_pack(pack_path)

    @staticmethod
    def _states(ep, outcome):
        states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
        if outcome == "change_supported":
            states[ep["frames"][1]["id"]] = "visible_change"
            for frame in ep["frames"][2:]:
                states[frame["id"]] = "visible_change"
        elif outcome == "insufficient_evidence":
            states[ep["frames"][1]["id"]] = "ambiguous"
        elif outcome != "no_visible_change":
            raise AssertionError(f"unknown fixture outcome: {outcome}")
        return states

    @classmethod
    def _annotation(cls, ep, annotator_id, outcome, status="complete"):
        states = cls._states(ep, outcome)
        if status == "timeout":
            states = {ep["reference_id"]: "no_visible_change"}
        return {
            "episode_id": ep["id"],
            "annotator_id": annotator_id,
            "status": status,
            "states": states,
        }

    @staticmethod
    def _export(pack_payload, annotator_id, annotations):
        return {
            "schema": "sn7-visible-annotations-v0.5",
            "pack_id": pack_payload["pack_id"],
            "annotator_id": annotator_id,
            "annotations": annotations,
        }

    def test_selection_ignores_legacy_gold_and_uses_half_max_aoi_pairs(self):
        selected, copies = pack.select_episodes(self.rows, max_episodes=4)
        altered = copy.deepcopy(self.rows)
        for row in altered:
            row["gold"] = "OPPOSITE_LEGACY_GOLD"
            row["answer"] = "OPPOSITE_LEGACY_ANSWER"
            row["region"] = "OPPOSITE_LEGACY_REGION"
            row["conds"]["privileged_silver"] = ["1900-01"]
            row["conds"]["privileged_wrongcontent"] = ["1900-02"]
        selected_after, copies_after = pack.select_episodes(altered, max_episodes=4)
        self.assertEqual(selected_after, selected)
        self.assertEqual(copies_after, copies)

        self.assertEqual(len(selected), 4)
        self.assertEqual(len({ep["aoi"] for ep in selected}), 2)
        by_aoi = {}
        for ep in selected:
            by_aoi.setdefault(ep["aoi"], []).append(ep)
            self.assertEqual(ep["cutoff"], "2020-03")
            self.assertEqual(ep["frames"][-1]["date"], ep["cutoff"])
            self.assertTrue(all(frame["date"] <= ep["cutoff"] for frame in ep["frames"]))
            self.assertNotIn("2020-04", [frame["date"] for frame in ep["frames"]])
        for episodes in by_aoi.values():
            self.assertEqual(len(episodes), 2)
            regions = {ep["region"] for ep in episodes}
            self.assertIn(regions, ({"NW", "SE"}, {"NE", "SW"}))
            self.assertEqual(len({ep["cutoff"] for ep in episodes}), 1)
        self.assertEqual(len(copies), sum(len(ep["frames"]) for ep in selected))

    def test_public_pack_has_no_legacy_answer_hint_or_future_frame(self):
        output, pack_path, payload = self._build()
        public_text = pack_path.read_text(encoding="utf-8")
        html_text = (output / "annotator_pack" / "index.html").read_text(encoding="utf-8")
        for secret in ("LEGACY_SECRET_GOLD", "LEGACY_SECRET_ANSWER", "2099-12"):
            self.assertNotIn(secret, public_text)
            self.assertNotIn(secret, html_text)

        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        exposed_keys = set(keys(payload))
        self.assertTrue({"id", "aoi", "region", "cutoff", "reference_id", "frames"}
                        <= set(payload["episodes"][0]))
        self.assertTrue({"gold", "answer", "target", "states", "conds"}.isdisjoint(exposed_keys))
        self.assertTrue(all(frame["date"] <= ep["cutoff"]
                            for ep in payload["episodes"] for frame in ep["frames"]))

    def test_pack_metadata_and_image_tampering_fail_closed(self):
        _, pack_path, payload = self._build()
        tampered = copy.deepcopy(payload)
        tampered["episodes"][0]["aoi"] = "TAMPERED-AOI"
        tampered_path = pack_path.with_name("tampered-pack.json")
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content/identifier mismatch"):
            pack.load_pack(tampered_path)

        image_path = pack_path.parent / payload["episodes"][0]["frames"][0]["path"]
        image_path.write_bytes(PNG_BYTES + b"tampered")
        with self.assertRaisesRegex(ValueError, "Image changed"):
            pack.load_pack(pack_path)

    def test_duplicate_episode_and_reviewer_fail_closed(self):
        _, pack_path, payload = self._build()
        duplicate_episodes = copy.deepcopy(payload["episodes"])
        duplicate_episodes[1]["id"] = duplicate_episodes[0]["id"]
        duplicate_payload = pack.make_public_pack(
            duplicate_episodes, payload["image_sha256"])
        duplicate_path = pack_path.with_name("duplicate-pack.json")
        duplicate_path.write_text(json.dumps(duplicate_payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate episode ID"):
            pack.load_pack(duplicate_path)

        ep = payload["episodes"][0]
        first = self._export(payload, "same-rater", [
            self._annotation(ep, "same-rater", "no_visible_change")])
        repeated = copy.deepcopy(first)
        with self.assertRaisesRegex(ValueError, "Repeated annotator/episode"):
            pack.review_exports(payload, [first, repeated])

    def test_timeout_is_incomplete_not_a_scene_answer_or_consensus_vote(self):
        _, _, payload = self._build()
        ep = payload["episodes"][0]
        timeout = self._export(payload, "timed-out", [
            self._annotation(ep, "timed-out", "no_visible_change", status="timeout")])
        complete_a = self._export(payload, "rater-a", [
            self._annotation(ep, "rater-a", "no_visible_change")])
        complete_b = self._export(payload, "rater-b", [
            self._annotation(ep, "rater-b", "no_visible_change")])
        report = pack.review_exports(payload, [timeout, complete_a, complete_b])
        self.assertEqual(report["targets"][ep["id"]]["answer"], "no_visible_change")
        self.assertEqual(report["targets"][ep["id"]]["annotator_ids"],
                         ["rater-a", "rater-b"])
        self.assertIn({"episode_id": ep["id"], "annotator_id": "timed-out",
                       "reason": "timeout"}, report["incomplete"])
        self.assertNotIn("timed-out", report["targets"][ep["id"]]["annotator_ids"])

    def test_finalize_preserves_donor_gold_and_withholds_visual_gold_from_null_inputs(self):
        output, pack_path, payload = self._build()
        by_aoi = {}
        for ep in payload["episodes"]:
            by_aoi.setdefault(ep["aoi"], []).append(ep)
        outcomes = {}
        desired_pairs = (
            ("change_supported", "no_visible_change"),
            ("insufficient_evidence", "change_supported"),
        )
        for pair_index, episodes in enumerate(by_aoi.values()):
            for ep, outcome in zip(episodes, desired_pairs[pair_index]):
                outcomes[ep["id"]] = outcome

        export_paths = []
        for annotator_id in ("rater-a", "rater-b"):
            annotations = [
                self._annotation(ep, annotator_id, outcomes[ep["id"]])
                for ep in payload["episodes"]
            ]
            export = self._export(payload, annotator_id, annotations)
            export_path = self.root / f"{annotator_id}.json"
            export_path.write_text(json.dumps(export), encoding="utf-8")
            export_paths.append(str(export_path))

        final_dir = self.root / "finalized"
        pack.finalize(types.SimpleNamespace(
            pack=str(pack_path), annotations=export_paths, output=str(final_dir)))
        report = json.loads((final_dir / "review_report.json").read_text(encoding="utf-8"))
        matrix = [json.loads(line) for line in
                  (final_dir / "diagnostic_matrix.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(report["answer_counts"], {
            "change_supported": 2,
            "no_visible_change": 1,
            "insufficient_evidence": 1,
        })
        self.assertTrue(report["diagnostic_manifest_ready"])

        base_rows = [row for row in matrix if row["condition"] != "different_outcome_swap"]
        self.assertEqual(len(base_rows), 3 * len(payload["episodes"]))
        for row in base_rows:
            own_gold = report["targets"][row["episode_id"]]
            self.assertEqual(row["real_image_reference_target"], own_gold)
            if row["condition"] == "real":
                self.assertEqual(row["target"], own_gold)
            else:
                self.assertEqual(row["target"]["answer"], "insufficient_evidence")
                self.assertIsNone(row["target"]["first_change_id"])
                self.assertIsNone(row["target"]["last_clear_no_change_id"])
                self.assertEqual(row["target"]["evidence_ids"], [])
                self.assertEqual(row["target"]["current_state"], "unreadable")
                self.assertEqual(row["target"]["reference_state"], "unreadable")
                self.assertEqual(row["target"]["raw_current_state"], "unreadable")
                self.assertEqual(row["target"]["evidence_ids_role"],
                                 "supporting_observations_not_global_or_onset_proof")
                self.assertEqual(row["target"]["label_tier"], "input_withheld_control")

        swap_rows = [row for row in matrix if row["condition"] == "different_outcome_swap"]
        self.assertEqual(len(swap_rows), 2 * len(report["control_pairs"]))
        episodes = {ep["id"]: ep for ep in payload["episodes"]}
        for row in swap_rows:
            donor_id = row["donor_episode_id"]
            self.assertEqual(row["target"], report["targets"][donor_id])
            self.assertNotEqual(row["target"], report["targets"][row["episode_id"]])
            self.assertEqual(row["input"]["frames"], episodes[donor_id]["frames"])
            self.assertIn("not a building-count threshold", row["input"]["prompt"])
            self.assertFalse(row["allowed_for_model_run"])

    def test_finalize_rejects_swap_when_prompt_and_donor_frame_ids_do_not_align(self):
        _, pack_path, payload = self._build()
        by_aoi = {}
        for ep in payload["episodes"]:
            by_aoi.setdefault(ep["aoi"], []).append(ep)
        pair = next(iter(by_aoi.values()))

        altered_episodes = copy.deepcopy(payload["episodes"])
        donor_id = pair[1]["id"]
        altered_donor = next(ep for ep in altered_episodes if ep["id"] == donor_id)
        for index, frame in enumerate(altered_donor["frames"]):
            frame["id"] = f"D{index:03d}"
        altered_donor["reference_id"] = "D000"
        altered_payload = pack.make_public_pack(altered_episodes, payload["image_sha256"])
        altered_path = pack_path.with_name("unaligned-pack.json")
        altered_path.write_text(json.dumps(altered_payload), encoding="utf-8")

        altered_pair = [
            next(ep for ep in altered_episodes if ep["id"] == pair[0]["id"]),
            altered_donor,
        ]
        export_paths = []
        for annotator_id in ("rater-a", "rater-b"):
            annotations = [
                self._annotation(altered_pair[0], annotator_id, "change_supported"),
                self._annotation(altered_pair[1], annotator_id, "no_visible_change"),
            ]
            export = self._export(altered_payload, annotator_id, annotations)
            export_path = self.root / f"unaligned-{annotator_id}.json"
            export_path.write_text(json.dumps(export), encoding="utf-8")
            export_paths.append(str(export_path))

        with self.assertRaisesRegex(ValueError, "frame IDs must align"):
            pack.finalize(types.SimpleNamespace(
                pack=str(altered_path),
                annotations=export_paths,
                output=str(self.root / "unaligned-final"),
            ))


if __name__ == "__main__":
    unittest.main()
