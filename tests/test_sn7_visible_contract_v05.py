import copy
import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).parents[1] / "code" / "sn7_visible_contract_v05.py"
SPEC = importlib.util.spec_from_file_location("sn7_visible_contract_v05", MODULE_PATH)
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def episode(episode_id="ep-source", region="NW", prefix="s"):
    return {
        "id": episode_id,
        "aoi": "AOI-1",
        "region": region,
        "cutoff": "2020-04",
        "reference_id": f"{prefix}1",
        "frames": [
            {"id": f"{prefix}1", "date": "2020-01", "path": f"frames/{prefix}1.png"},
            {"id": f"{prefix}2", "date": "2020-02", "path": f"frames/{prefix}2.png"},
            {"id": f"{prefix}3", "date": "2020-03", "path": f"frames/{prefix}3.png"},
            {"id": f"{prefix}4", "date": "2020-04", "path": f"frames/{prefix}4.png"},
        ],
    }


def annotation(ep, annotator_id="rater-a", states=None, status="complete"):
    if states is None:
        states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
    return {
        "episode_id": ep["id"],
        "annotator_id": annotator_id,
        "status": status,
        "states": states,
    }


def agreed_target(ep, states, annotators=("rater-a", "rater-b")):
    anns = [annotation(ep, annotator_id=name, states=copy.deepcopy(states)) for name in annotators]
    return contract.consensus_target(ep, anns)


class ValidateEpisodeTests(unittest.TestCase):
    def test_valid_episode(self):
        self.assertIsNone(contract.validate_episode(episode()))

    def test_required_fields_and_minimum_frames(self):
        ep = episode()
        del ep["aoi"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            contract.validate_episode(ep)

        ep = episode()
        ep["frames"] = ep["frames"][:1]
        with self.assertRaisesRegex(ValueError, "at least two"):
            contract.validate_episode(ep)

    def test_frame_ids_and_dates_are_unique_and_ordered(self):
        ep = episode()
        ep["frames"][1]["id"] = ep["frames"][0]["id"]
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            contract.validate_episode(ep)

        ep = episode()
        ep["frames"][1]["date"] = ep["frames"][0]["date"]
        with self.assertRaisesRegex(ValueError, "dates must be unique"):
            contract.validate_episode(ep)

        ep = episode()
        ep["frames"][1]["date"], ep["frames"][2]["date"] = (
            ep["frames"][2]["date"], ep["frames"][1]["date"])
        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            contract.validate_episode(ep)

    def test_iso_month_cutoff_and_reference(self):
        for bad_date in ("2020-1", "2020-13", "20-01", "2020-00"):
            with self.subTest(bad_date=bad_date):
                ep = episode()
                ep["frames"][1]["date"] = bad_date
                with self.assertRaisesRegex(ValueError, "ISO YYYY-MM"):
                    contract.validate_episode(ep)

        ep = episode()
        ep["cutoff"] = "2020-03"
        with self.assertRaisesRegex(ValueError, "after episode.cutoff"):
            contract.validate_episode(ep)

        ep = episode()
        ep["reference_id"] = "s2"
        with self.assertRaisesRegex(ValueError, "first frame"):
            contract.validate_episode(ep)

    def test_rejects_unsafe_paths(self):
        bad_paths = (
            "/tmp/frame.png",
            "../frame.png",
            "frames/../../frame.png",
            "https://example.test/frame.png",
            "//server/share/frame.png",
            "frames\\..\\frame.png",
        )
        for bad_path in bad_paths:
            with self.subTest(path=bad_path):
                ep = episode()
                ep["frames"][1]["path"] = bad_path
                with self.assertRaises(ValueError):
                    contract.validate_episode(ep)


class DeriveTargetTests(unittest.TestCase):
    def test_change_supported_uses_first_positive_and_latest_clear_before_it(self):
        ep = episode()
        states = {
            "s1": "no_visible_change",
            "s2": "ambiguous",
            "s3": "no_visible_change",
            "s4": "visible_change",
        }
        target = contract.derive_target(ep, annotation(ep, states=states))
        self.assertEqual(target["answer"], "change_supported")
        self.assertEqual((target["first_change_id"], target["first_change_date"]),
                         ("s4", "2020-04"))
        self.assertEqual((target["last_clear_no_change_id"],
                          target["last_clear_no_change_date"]),
                         ("s3", "2020-03"))
        self.assertEqual(target["evidence_ids"], ["s1", "s4"])
        self.assertEqual(target["current_state"], "visible_change")
        self.assertEqual(target["raw_current_state"], "visible_change")
        self.assertEqual(target["reference_state"], "no_visible_change")
        self.assertEqual(target["evidence_ids_role"],
                         "supporting_observations_not_global_or_onset_proof")
        self.assertEqual(target["label_tier"], "human_visible_single")
        self.assertEqual(target["source"], "annotation")

    def test_historical_change_survives_an_unreadable_latest_observation(self):
        ep = episode()
        states = {
            "s1": "no_visible_change",
            "s2": "no_visible_change",
            "s3": "visible_change",
            "s4": "unreadable",
        }
        target = contract.derive_target(ep, annotation(ep, states=states))
        self.assertEqual(target["answer"], "change_supported")
        self.assertEqual(target["first_change_id"], "s3")
        self.assertEqual(target["last_clear_no_change_id"], "s2")
        self.assertEqual(target["evidence_ids"], ["s1", "s3"])
        self.assertEqual(target["current_state"], "unreadable")
        self.assertEqual(target["raw_current_state"], "unreadable")

    def test_no_change_and_insufficient_evidence_are_distinct(self):
        ep = episode()
        no_change = contract.derive_target(ep, annotation(ep))
        self.assertEqual(no_change["answer"], "no_visible_change")
        self.assertEqual(no_change["last_clear_no_change_id"], "s4")
        self.assertIsNone(no_change["first_change_id"])
        self.assertEqual(no_change["evidence_ids"], ["s1", "s2", "s3", "s4"])
        self.assertEqual(no_change["reference_state"], "no_visible_change")
        self.assertEqual(no_change["raw_current_state"], "no_visible_change")

        states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
        states["s3"] = "unreadable"
        unknown = contract.derive_target(ep, annotation(ep, states=states))
        self.assertEqual(unknown["answer"], "insufficient_evidence")
        self.assertIsNone(unknown["first_change_id"])
        self.assertEqual(unknown["current_state"], "no_visible_change")
        self.assertEqual(unknown["raw_current_state"], "no_visible_change")
        self.assertEqual(unknown["reference_state"], "no_visible_change")
        self.assertEqual(unknown["evidence_ids"], ["s1", "s3"])

    def test_unclear_reference_forces_insufficient_evidence(self):
        ep = episode()
        for reference_state in ("unreadable", "ambiguous"):
            with self.subTest(reference_state=reference_state):
                states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
                states["s1"] = reference_state
                states["s3"] = "visible_change"
                target = contract.derive_target(ep, annotation(ep, states=states))
                self.assertEqual(target["answer"], "insufficient_evidence")
                self.assertIsNone(target["first_change_id"])
                self.assertEqual(target["current_state"], "ambiguous")
                self.assertEqual(target["raw_current_state"], "no_visible_change")
                self.assertEqual(target["reference_state"], reference_state)
                self.assertEqual(target["evidence_ids"], ["s1"])

        states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
        states["s1"] = "ambiguous"
        states["s4"] = "unreadable"
        target = contract.derive_target(ep, annotation(ep, states=states))
        self.assertEqual(target["answer"], "insufficient_evidence")
        self.assertEqual(target["current_state"], "unreadable")
        self.assertEqual(target["raw_current_state"], "unreadable")
        self.assertEqual(target["evidence_ids"], ["s1", "s4"])

    def test_annotation_contract_failures(self):
        ep = episode()
        with self.assertRaisesRegex(ValueError, "timeout"):
            contract.derive_target(ep, annotation(ep, status="timeout", states={}))

        states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
        del states["s2"]
        with self.assertRaisesRegex(ValueError, "every frame exactly once"):
            contract.derive_target(ep, annotation(ep, states=states))

        states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
        states["extra"] = "no_visible_change"
        with self.assertRaisesRegex(ValueError, "every frame exactly once"):
            contract.derive_target(ep, annotation(ep, states=states))

        states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
        states["s2"] = "maybe"
        with self.assertRaisesRegex(ValueError, "invalid values"):
            contract.derive_target(ep, annotation(ep, states=states))

        states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
        states["s1"] = "visible_change"
        with self.assertRaisesRegex(ValueError, "reference frame"):
            contract.derive_target(ep, annotation(ep, states=states))

        ann = annotation(ep)
        ann["episode_id"] = "other"
        with self.assertRaisesRegex(ValueError, "does not match"):
            contract.derive_target(ep, ann)


class ConsensusTests(unittest.TestCase):
    def test_agreement_requires_distinct_annotators(self):
        ep = episode()
        states = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
        states["s4"] = "visible_change"
        result = agreed_target(ep, states, ("z-rater", "a-rater"))
        self.assertEqual(result["label_tier"], "human_visible_agreed")
        self.assertEqual(result["annotator_ids"], ["a-rater", "z-rater"])
        self.assertNotIn("annotator_id", result)
        self.assertEqual(result["answer"], "change_supported")

        same_rater = [
            annotation(ep, "rater-a", states),
            annotation(ep, "rater-a", states),
        ]
        with self.assertRaisesRegex(ValueError, "distinct annotators"):
            contract.consensus_target(ep, same_rater)
        with self.assertRaisesRegex(ValueError, "at least two"):
            contract.consensus_target(ep, same_rater[:1])

    def test_disagreement_on_required_signature_is_rejected(self):
        ep = episode()
        first = {frame["id"]: "no_visible_change" for frame in ep["frames"]}
        second = copy.deepcopy(first)
        first["s3"] = "visible_change"
        second["s4"] = "visible_change"
        anns = [annotation(ep, "rater-a", first), annotation(ep, "rater-b", second)]
        with self.assertRaisesRegex(ValueError, "do not agree"):
            contract.consensus_target(ep, anns)

    def test_same_answer_is_not_agreement_when_provenance_or_evidence_differs(self):
        ep = episode()

        unreadable_reference = {
            frame["id"]: "no_visible_change" for frame in ep["frames"]}
        ambiguous_reference = copy.deepcopy(unreadable_reference)
        unreadable_reference["s1"] = "unreadable"
        ambiguous_reference["s1"] = "ambiguous"
        anns = [
            annotation(ep, "rater-a", unreadable_reference),
            annotation(ep, "rater-b", ambiguous_reference),
        ]
        derived = [contract.derive_target(ep, ann) for ann in anns]
        self.assertEqual(derived[0]["answer"], derived[1]["answer"])
        self.assertEqual(derived[0]["current_state"], derived[1]["current_state"])
        self.assertNotEqual(derived[0]["reference_state"], derived[1]["reference_state"])
        with self.assertRaisesRegex(ValueError, "do not agree"):
            contract.consensus_target(ep, anns)

        unknown_at_s2 = {
            frame["id"]: "no_visible_change" for frame in ep["frames"]}
        unknown_at_s3 = copy.deepcopy(unknown_at_s2)
        unknown_at_s2["s2"] = "ambiguous"
        unknown_at_s3["s3"] = "ambiguous"
        anns = [
            annotation(ep, "rater-a", unknown_at_s2),
            annotation(ep, "rater-b", unknown_at_s3),
        ]
        derived = [contract.derive_target(ep, ann) for ann in anns]
        for key in ("answer", "first_change_id", "last_clear_no_change_id",
                    "current_state", "reference_state", "raw_current_state"):
            self.assertEqual(derived[0][key], derived[1][key])
        self.assertNotEqual(derived[0]["evidence_ids"], derived[1]["evidence_ids"])
        with self.assertRaisesRegex(ValueError, "do not agree"):
            contract.consensus_target(ep, anns)


class ControlPairTests(unittest.TestCase):
    def setUp(self):
        self.source_ep = episode()
        self.donor_ep = episode("ep-donor", "SE", "d")
        source_states = {frame["id"]: "no_visible_change" for frame in self.source_ep["frames"]}
        source_states["s3"] = "visible_change"
        source_states["s4"] = "visible_change"
        donor_states = {frame["id"]: "no_visible_change" for frame in self.donor_ep["frames"]}
        self.source_target = agreed_target(self.source_ep, source_states)
        self.donor_target = agreed_target(self.donor_ep, donor_states)

    def test_builds_date_mapping_and_preserves_both_targets(self):
        result = contract.make_control_pair(
            self.source_ep, self.source_target, self.donor_ep, self.donor_target)
        self.assertEqual(result["source_episode_id"], "ep-source")
        self.assertEqual(result["donor_episode_id"], "ep-donor")
        self.assertEqual(result["frame_pairs"][0], {
            "date": "2020-01",
            "source_frame_id": "s1",
            "donor_frame_id": "d1",
        })
        self.assertEqual(len(result["frame_pairs"]), 4)
        self.assertEqual(result["source_target"], self.source_target)
        self.assertEqual(result["donor_target"], self.donor_target)
        self.assertEqual(result["source_target"]["answer"], "change_supported")
        self.assertEqual(result["donor_target"]["answer"], "no_visible_change")

        result["source_target"]["answer"] = "mutated"
        self.assertEqual(self.source_target["answer"], "change_supported")

    def test_rejects_same_region_or_mismatched_dates(self):
        donor = copy.deepcopy(self.donor_ep)
        donor["region"] = self.source_ep["region"]
        with self.assertRaisesRegex(ValueError, "different full-sequence regions"):
            contract.make_control_pair(
                self.source_ep, self.source_target, donor, self.donor_target)

        donor = copy.deepcopy(self.donor_ep)
        donor["cutoff"] = "2020-05"
        donor["frames"][-1]["date"] = "2020-05"
        donor_states = {frame["id"]: "no_visible_change" for frame in donor["frames"]}
        mismatched_target = agreed_target(donor, donor_states)
        with self.assertRaisesRegex(ValueError, "dates must match exactly"):
            contract.make_control_pair(
                self.source_ep, self.source_target, donor, mismatched_target)

    def test_rejects_reused_outcome_and_nonagreed_targets(self):
        donor_states = {frame["id"]: "no_visible_change" for frame in self.donor_ep["frames"]}
        donor_states["d3"] = "visible_change"
        donor_states["d4"] = "visible_change"
        same_outcome = agreed_target(self.donor_ep, donor_states)
        with self.assertRaisesRegex(ValueError, "outcome signatures must differ"):
            contract.make_control_pair(
                self.source_ep, self.source_target, self.donor_ep, same_outcome)

        single = contract.derive_target(
            self.source_ep,
            annotation(self.source_ep, states={
                "s1": "no_visible_change",
                "s2": "no_visible_change",
                "s3": "visible_change",
                "s4": "visible_change",
            }),
        )
        with self.assertRaisesRegex(ValueError, "human_visible_agreed"):
            contract.make_control_pair(
                self.source_ep, single, self.donor_ep, self.donor_target)


if __name__ == "__main__":
    unittest.main()
