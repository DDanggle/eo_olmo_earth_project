"""Contract checks for an offline evidence query engine, not model accuracy."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
try:
    from eo_query_core_v0 import build_kurosiwo_catalog, query_catalog
except ModuleNotFoundError:
    source = Path(__file__).with_name("eo_query_core_20260925.py")
    spec = importlib.util.spec_from_file_location("eo_query_core_v0", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_kurosiwo_catalog, query_catalog = module.build_kurosiwo_catalog, module.query_catalog


def row(sid="ks_1", *, fraction=25, water=10, point="POINT (85 25)", event="2019-09-17", split="train"):
    return {"id": sid, "split": split, "actid": 1111007, "aoiid": "02", "flood_date": event,
            "pflood": fraction, "pwater": water, "centroid": point}


def sn7(sid, status, first=None, point=None):
    return {"id": sid, "dataset": "sn7", "aoi_id": "sn7_aoi", "region_id": "NW",
            "split": "exposed_development", "event_type": "structural_change", "status": status,
            "event_date": None, "observation_start": "2019-01-01", "observation_end": "2019-12-01",
            "first_change_date": first, "point": point, "evidence": [], "limitations": []}


class CatalogQueryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "meta.jsonl"

    def build(self, rows):
        self.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return build_kurosiwo_catalog(self.path)

    def test_import_retains_reference_role_and_source_hash(self):
        cat = self.build([row()])
        self.assertEqual(cat["provenance"]["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(cat["provenance"]["path"], str(self.path.resolve()))
        self.assertIn("not_model_predictions", cat["provenance"]["source_role"])
        self.assertEqual(cat["records"][0]["flood_fraction_pct"], 25)
        self.assertEqual(cat["records"][0]["point"], [85, 25])
        self.assertEqual(cat["records"][0]["aoi_id"], "1111007")
        self.assertEqual(cat["records"][0]["source_record"]["line"], 1)
        self.assertEqual(cat["records"][0]["evidence"], [])
        self.assertNotIn("prediction_score", cat["records"][0])
        self.assertNotIn("impact_area", cat["records"][0])

    def test_reference_zero_is_not_general_no_impact(self):
        cat = self.build([row(fraction=0)])
        result = query_catalog(cat, {"status": "reference_no_flood"})
        self.assertEqual(result["records"][0]["status"], "reference_no_flood")
        self.assertIn("damage_and_cause_not_measured", result["records"][0]["limitations"])
        self.assertEqual(query_catalog(cat, {"status": "reference_flood"})["status"], "no_matches")

    def test_bbox_includes_boundary_and_excludes_outside(self):
        cat = self.build([row("boundary", point="POINT (85 25)"), row("outside", point="POINT (85.1 25)")])
        result = query_catalog(cat, {"bbox": [84, 24, 85, 25]})
        self.assertEqual([r["id"] for r in result["records"]], ["boundary"])
        self.assertIn("bbox_matches_centroids_not_footprint_intersections", result["limitations"])

    def test_antimeridian_bbox(self):
        cat = self.build([row("east", point="POINT (179 1)"), row("west", point="POINT (-179 1)"), row("middle", point="POINT (0 1)")])
        result = query_catalog(cat, {"bbox": [170, -5, -170, 5]})
        self.assertEqual({r["id"] for r in result["records"]}, {"east", "west"})

    def test_missing_geography_excluded_and_counted(self):
        cat = {"records": [sn7("s1", "no_visible_change")]}
        result = query_catalog(cat, {"dataset": "sn7", "bbox": [-180, -90, 180, 90]})
        self.assertEqual(result["status"], "no_matches")
        self.assertEqual(result["excluded_missing_geometry"], 1)
        self.assertIn("no_catalog_matches_does_not_establish_no_flood_or_no_impact", result["limitations"])

    def test_event_date_is_used_without_approximate_acquisition_fallback(self):
        cat = self.build([row()])
        record = cat["records"][0]
        record["observation_start"], record["observation_end"] = "2019-08-24", "2019-09-17"
        self.assertEqual(query_catalog(cat, {"start": "2019-08-24", "end": "2019-08-24"})["status"], "no_matches")
        result = query_catalog(cat, {"start": "2019-09-17", "end": "2019-09-17"})
        self.assertEqual(result["records"][0]["time_basis"], "event_date")
        self.assertEqual(result["records"][0]["time_interval"], ["2019-09-17", "2019-09-17"])

    def test_sn7_change_date_and_unchanged_observation_overlap(self):
        cat = {"records": [sn7("changed", "change_supported", "2019-07-01"), sn7("uncertain", "insufficient_evidence")]}
        june = query_catalog(cat, {"dataset": "sn7", "start": "2019-06-01", "end": "2019-06-30"})
        self.assertEqual([r["id"] for r in june["records"]], ["uncertain"])
        self.assertEqual(june["records"][0]["time_basis"], "observation_interval")
        july = query_catalog(cat, {"dataset": "sn7", "status": "change_supported", "start": "2019-07-01", "end": "2019-07-01"})
        self.assertEqual(july["records"][0]["time_basis"], "first_change_date")
        self.assertEqual(july["matched_count"], 1)

    def test_sn7_first_change_month_preserved_and_partially_overlaps(self):
        record = sn7("monthly", "change_supported", "2020-02")
        cat = {"records": [record]}
        result = query_catalog(cat, {"start": "2020-02-14", "end": "2020-02-20"})
        self.assertEqual(result["matched_count"], 1)
        returned = result["records"][0]
        self.assertEqual(returned["first_change_date"], "2020-02")
        self.assertEqual(returned["time_precision"], "month")
        self.assertEqual(returned["time_interval"], ["2020-02-01", "2020-02-29"])
        self.assertEqual(query_catalog(cat, {"start": "2020-03-01"})["status"], "no_matches")
        self.assertEqual(query_catalog(cat, {"end": "2020-01-31"})["status"], "no_matches")

    def test_sn7_observation_month_interval_and_nonleap_february(self):
        record = sn7("monthly", "no_visible_change")
        record.update(observation_start="2019-01", observation_end="2019-02")
        result = query_catalog({"records": [record]}, {"start": "2019-02-28", "end": "2019-03-01"})
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["records"][0]["time_interval"], ["2019-01-01", "2019-02-28"])
        self.assertEqual(result["records"][0]["time_precision"], "month")
        self.assertEqual(query_catalog({"records": [record]}, {"start": "2019-03-01"})["matched_count"], 0)

    def test_reject_invalid_record_month_and_month_precision_in_query(self):
        for invalid in ("2019-00", "2019-13", "0000-01", "2019-2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    query_catalog({"records": [sn7("bad", "change_supported", invalid)]}, {})
        with self.assertRaises(ValueError):
            query_catalog({"records": [sn7("s", "change_supported", "2019-01")]}, {"start": "2019-01"})

    def test_kurosiwo_keeps_daily_precision_and_rejects_month_event_date(self):
        cat = self.build([row()])
        self.assertEqual(query_catalog(cat, {})["records"][0]["time_precision"], "day")
        cat["records"][0]["event_date"] = "2019-09"
        with self.assertRaises(ValueError):
            query_catalog(cat, {})

    def test_missing_first_date_does_not_fallback_to_observation_span(self):
        cat = {"records": [sn7("changed", "change_supported")]}
        result = query_catalog(cat, {"start": "2019-01-01", "end": "2019-12-31"})
        self.assertEqual(result["status"], "no_matches")
        self.assertEqual(result["excluded_missing_time"], 1)

    def test_scope_is_applied_before_other_dataset_date_validation(self):
        cat = self.build([row()])
        malformed_unselected = sn7("bad", "change_supported", "not-a-date")
        cat["records"].append(malformed_unselected)
        self.assertEqual(query_catalog(cat, {"dataset": "kurosiwo", "start": "2019-01-01"})["matched_count"], 1)
        self.assertEqual(query_catalog(cat, {"dataset": "absent"})["status"], "no_matches")

    def test_all_exact_scope_filters_remain_active(self):
        cat = self.build([row("train"), row("test", split="test")])
        result = query_catalog(cat, {"dataset": "kurosiwo", "event_type": "flood", "split": "test", "aoi_id": "1111007"})
        self.assertEqual([r["id"] for r in result["records"]], ["test"])
        self.assertEqual(query_catalog(cat, {"aoi_id": "missing"})["matched_count"], 0)
        self.assertEqual(query_catalog(cat, {"event_type": "forest_loss"})["matched_count"], 0)

    def test_land_cover_request_reports_unavailable_not_negative(self):
        cat = self.build([row()])
        for cover in ("cropland", "forest"):
            with self.subTest(cover=cover):
                result = query_catalog(cat, {"land_cover": cover})
                self.assertEqual(result["status"], "needs_data")
                self.assertEqual(result["records"], [])
                self.assertEqual(result["blockers"], ["not_computed"])
                self.assertIn("land_cover_overlap_was_not_evaluated_zero_returned_records_is_not_zero_impact", result["limitations"])

    def test_missing_flood_fraction_is_not_assumed_zero(self):
        cat = self.build([row(fraction=0)])
        cat["records"].append(sn7("sn7", "no_visible_change"))
        result = query_catalog(cat, {"min_flood_pct": 0})
        self.assertEqual([r["dataset"] for r in result["records"]], ["kurosiwo"])

    def test_threshold_sort_limit_and_deterministic_ties(self):
        cat = self.build([row("low", fraction=4), row("b", fraction=30), row("a", fraction=30), row("high", fraction=90)])
        result = query_catalog(cat, {"min_flood_pct": 30, "limit": 2})
        self.assertEqual(result["matched_count"], 3)
        self.assertEqual(result["returned_count"], 2)
        self.assertEqual([r["id"] for r in result["records"]], ["high", "a"])

    def test_sn7_first_date_sort(self):
        cat = {"records": [sn7("z", "change_supported", "2019-09-01"), sn7("b", "change_supported", "2019-03-01"), sn7("a", "change_supported", "2019-03-01")]}
        self.assertEqual([r["id"] for r in query_catalog(cat, {})["records"]], ["a", "b", "z"])

    def test_does_not_mutate_catalog_or_caller_query(self):
        cat = self.build([row()]); original = copy.deepcopy(cat)
        query = {"bbox": [84, 24, 86, 26]}; original_query = copy.deepcopy(query)
        result = query_catalog(cat, query)
        result["records"][0]["limitations"].append("changed")
        self.assertEqual(cat, original)
        self.assertEqual(query, original_query)

    def test_reject_duplicate_source_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            self.build([row(), row()])

    def test_reject_malformed_source_values(self):
        cases = [("pflood", -1), ("pflood", 101), ("pflood", float("nan")), ("pwater", float("inf")), ("pflood", True), ("centroid", "POINT (181 25)"), ("centroid", "POINT (85 -91)"), ("centroid", "POINT (nan 25)"), ("centroid", "LINESTRING (1 2,3 4)"), ("flood_date", "2019-02-30"), ("flood_date", "2019-9-1"), ("split", "all"), ("id", ""), ("actid", True), ("aoiid", None)]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                record = row(); record[field] = value
                with self.assertRaises(ValueError):
                    self.build([record])

    def test_reject_unknown_or_invalid_query(self):
        cat = self.build([row()])
        cases = [{"area": "place"}, {"start": "2019-02-29"}, {"start": "2020-01-02", "end": "2020-01-01"}, {"bbox": [0, 1, 2, -1]}, {"bbox": [0, 0, 181, 1]}, {"bbox": [0, 0, float("nan"), 1]}, {"bbox": [0, 0, 1]}, {"min_flood_pct": -1}, {"min_flood_pct": True}, {"limit": 0}, {"limit": 501}, {"limit": True}, {"limit": 1.0}, {"land_cover": "urban"}, {"dataset": ""}, {"aoi_id": 3}, {"split": None}]
        for query in cases:
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    query_catalog(cat, query)

    def test_reject_duplicate_catalog_records_within_dataset(self):
        cat = self.build([row()]); cat["records"].append(copy.deepcopy(cat["records"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate catalog"):
            query_catalog(cat, {})

    def test_same_identifier_across_datasets_is_allowed(self):
        cat = self.build([row("same")]); cat["records"].append(sn7("same", "no_visible_change"))
        self.assertEqual(query_catalog(cat, {})["matched_count"], 2)


if __name__ == "__main__":
    unittest.main()
