from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from kearth_benchmark.pilot import (  # noqa: E402
    PilotInputs,
    build_pilot,
    sha256_file,
    transition_aligned_exact_events,
    write_pilot,
)


def real_inputs() -> PilotInputs:
    return PilotInputs(
        config=ROOT / "config/k_evidence_shift_jeju_pilot_v0.json",
        candidate_manifest=ROOT / "artifacts/human_review_v1/manifest.json",
        assistant_review=ROOT / "artifacts/human_review_v1/assistant_review.json",
        candidate_evidence=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/candidate_evidence.json",
        observation_context=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/observation_context.json",
        api_run_summary=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/run_summary.json",
        api_requests=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/requests.json",
        api_complete_marker=ROOT / "artifacts/external_data/kearth_api_snapshot_v3/COMPLETE.json",
    )


class PilotContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pilot = build_pilot(real_inputs())
        cls.by_id = {record["candidate_id"]: record for record in cls.pilot["records"]}

    def test_fixed_candidate_population_and_audit_only_split(self) -> None:
        self.assertEqual(len(self.pilot["records"]), 14)
        self.assertEqual({record["split"] for record in self.pilot["records"]}, {"pilot_audit_pool"})
        self.assertFalse(any(record["split_eligible_for_metrics"] for record in self.pilot["records"]))

    def test_assistant_review_never_becomes_ground_truth(self) -> None:
        self.assertFalse(
            any(record["label_axes"]["visual_change"]["eligible_as_ground_truth"] for record in self.pilot["records"])
        )

    def test_spatial_duplicate_pair_is_grouped(self) -> None:
        left = self.by_id["dev_control_v3_r02"]
        right = self.by_id["built_v6_r16"]
        self.assertEqual(left["site_group_id"], right["site_group_id"])
        self.assertEqual(len({record["site_group_id"] for record in self.pilot["records"]}), 13)

    def test_shared_materialized_window_is_a_split_guard(self) -> None:
        left = self.by_id["oreum_v6_r08"]
        right = self.by_id["oreum_v6_r10"]
        self.assertNotEqual(left["site_group_id"], right["site_group_id"])
        self.assertEqual(left["spatial_window_group_id"], right["spatial_window_group_id"])

    def test_rank_selected_pool_is_not_prevalence_estimable(self) -> None:
        for record in self.pilot["records"]:
            contract = record["selection_contract"]
            self.assertFalse(contract["eligible_for_prevalence_estimation"])
            self.assertIsNone(contract["inclusion_probability"])
            self.assertFalse(contract["selection_fields_allowed_as_model_features"])

    def test_post_t1_api_snapshot_is_not_a_prospective_input(self) -> None:
        self.assertFalse(any(record["public_evidence"]["prospective_input_eligible"] for record in self.pilot["records"]))

    def test_future_rgb_is_review_only_not_transition_input(self) -> None:
        early = self.by_id["oreum_v6_r10"]
        future = [
            value
            for value in early["observations"]
            if value["temporal_role"] == "future_after_t1_review_only"
        ]
        self.assertTrue(future)
        self.assertFalse(any(value["prospective_input_eligible"] for value in future))
        self.assertTrue(early["label_axes"]["visual_change"]["uses_post_t1_observations"])
        self.assertFalse(
            early["label_axes"]["visual_change"]["eligible_for_prospective_evaluation"]
        )

    def test_transition_alignment_is_recomputed_for_candidate_interval(self) -> None:
        evidence = {
            "parcel_pnu_values": ["pnu"],
            "same_legal_dong_building_events": [
                {"pnu": "pnu", "permit_date": "20240601"},
                {"pnu": "pnu", "permit_date": "20260706"},
                {"pnu": "other", "permit_date": "20240601"},
            ],
        }
        aligned = transition_aligned_exact_events(evidence, "2024-05-01", "2025-05-01")
        self.assertEqual([value["permit_date"] for value in aligned], ["20240601"])

    def test_no_match_is_not_a_negative_cause_label(self) -> None:
        for record in self.pilot["records"]:
            official = record["label_axes"]["official_event_supported"]
            self.assertEqual(official["value"], "not_observed")
            self.assertFalse(official["absence_is_negative_label"])
            self.assertIsNone(record["label_axes"]["cause"]["value"])

    def test_r04_post_observation_event_does_not_become_support(self) -> None:
        record = self.by_id["oreum_v6_r04"]
        self.assertEqual(record["public_evidence"]["exact_parcel_event_count_any_time"], 1)
        self.assertEqual(record["public_evidence"]["time_aligned_exact_parcel_event_count"], 0)
        self.assertEqual(record["decision"]["value"], "abstain")

    def test_cloud_field_is_explicitly_a_proxy(self) -> None:
        self.assertTrue(all(record["strata"]["cloud_measurement"] == "sentinel2_rgb_proxy_only" for record in self.pilot["records"]))
        self.assertFalse(self.pilot["gk2a_current_snapshot"]["eligible_for_historical_pairing"])

    def test_output_is_deterministic_and_hashes_verify(self) -> None:
        with tempfile.TemporaryDirectory() as left_name, tempfile.TemporaryDirectory() as right_name:
            left, right = Path(left_name), Path(right_name)
            write_pilot(self.pilot, left)
            write_pilot(self.pilot, right)
            left_files = sorted(path.name for path in left.iterdir())
            self.assertEqual(left_files, sorted(path.name for path in right.iterdir()))
            for name in left_files:
                self.assertEqual(sha256_file(left / name), sha256_file(right / name), name)
            manifest = json.loads((left / "sha256_manifest.json").read_text(encoding="utf-8"))
            for item in manifest["files"]:
                self.assertEqual(sha256_file(left / item["path"]), item["sha256"])
            complete = json.loads((left / "COMPLETE.json").read_text(encoding="utf-8"))
            self.assertEqual(
                complete["sha256_manifest_sha256"], sha256_file(left / "sha256_manifest.json")
            )

    def test_build_id_is_content_not_path_dependent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            copied_root = Path(temp_name)
            copied = {}
            for source_id, source in real_inputs().items():
                target = copied_root / source.name
                shutil.copy2(source, target)
                copied[source_id] = target
            moved = build_pilot(PilotInputs(**copied))
        self.assertEqual(self.pilot["build_id"], moved["build_id"])

    def test_unexpected_stale_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            output = Path(temp_name)
            (output / "stale.txt").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected stale files"):
                write_pilot(self.pilot, output)

    def test_promotion_gate_stays_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            result = write_pilot(self.pilot, Path(temp_name))
        self.assertFalse(result["gates"]["cvpr_experiment_ready"])
        self.assertEqual(result["summary"]["independent_human_ground_truth_labels"], 0)
        self.assertEqual(result["summary"]["official_event_supported"], 0)
        gates = result["gates"]["gates"]
        self.assertFalse(gates["sealed_probability_test"]["pass"])
        self.assertFalse(gates["frozen_paired_input_contract"]["pass"])
        self.assertEqual(gates["double_reviewed_labels"]["observed"], 0)
        self.assertEqual(gates["pinned_model_checkpoints"]["observed"], 0)
        self.assertTrue(gates["independent_spatial_groups"]["pass"])
        self.assertEqual(
            result["summary"]["unique_site_visual_preannotation_values"],
            {
                "change_preannotation": 5,
                "no_change_preannotation": 5,
                "uncertain_preannotation": 3,
            },
        )
        self.assertEqual(
            result["summary"]["high_confidence_change_preannotation_unique_sites"], 4
        )
        self.assertFalse(result["leakage"]["scene_disjoint_quality_split_possible"])


if __name__ == "__main__":
    unittest.main()
