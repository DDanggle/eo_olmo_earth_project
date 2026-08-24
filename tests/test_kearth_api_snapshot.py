from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from kearth_public.api_snapshot import (  # noqa: E402
    RequestSpec,
    data_go_items,
    eia_features,
    execute_request,
    point_in_ring,
    read_env_file,
    request_hash,
    vworld_features,
    vworld_semantic_status,
)
from collect_kearth_api_snapshot import build_candidate_evidence  # noqa: E402
from merge_kearth_api_snapshots import (  # noqa: E402
    merge_request_records,
    validate_vworld_snapshot,
    vworld_pnu_records,
)


class DotenvTests(unittest.TestCase):
    def test_parser_does_not_expand_shell_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / ".env"
            path.write_text("A='literal $HOME'\nexport B=two\n", encoding="utf-8")
            self.assertEqual(read_env_file(path), {"A": "literal $HOME", "B": "two"})


class RequestTests(unittest.TestCase):
    def test_hash_ignores_credentials(self) -> None:
        left = request_hash("source", "https://example.test/api", {"x": 1})
        right = request_hash("source", "https://example.test/api", {"x": "1"})
        self.assertEqual(left, right)

    def test_missing_key_is_structured_and_transport_is_not_called(self) -> None:
        called = False

        def transport(request, timeout):
            nonlocal called
            called = True
            raise AssertionError

        with tempfile.TemporaryDirectory() as temp_name:
            record = execute_request(
                RequestSpec(
                    "fixture",
                    "https://example.test/api",
                    {"x": 1},
                    {"key": "API_KEY"},
                ),
                output_dir=Path(temp_name),
                env={},
                transport=transport,
                retrieved_at="2026-08-22T00:00:00Z",
            )
        self.assertFalse(called)
        self.assertEqual(record["outcome"], "not_requested_missing_credential")
        self.assertNotIn("url", record)

    def test_response_and_manifest_are_secret_safe(self) -> None:
        captured_query = {}

        def transport(request, timeout):
            captured_query.update(urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query))
            return 200, {"Content-Type": "application/json"}, b'{"echo":"TOPSECRET"}'

        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            record = execute_request(
                RequestSpec(
                    "fixture",
                    "https://example.test/api",
                    {"x": 1},
                    {"key": "API_KEY"},
                    ".json",
                ),
                output_dir=root,
                env={"API_KEY": "TOPSECRET"},
                transport=transport,
                retrieved_at="2026-08-22T00:00:00Z",
            )
            raw = (root / str(record["raw_file"])).read_text(encoding="utf-8")
        self.assertEqual(captured_query["key"], ["TOPSECRET"])
        self.assertNotIn("TOPSECRET", json.dumps(record))
        self.assertNotIn("TOPSECRET", raw)
        self.assertIn("<redacted>", raw)


class EnvelopeTests(unittest.TestCase):
    def test_data_go_single_item_is_normalized(self) -> None:
        payload = {"response": {"header": {"resultCode": "00"}, "body": {"items": {"item": {"a": 1}}, "totalCount": 1}}}
        items, meta = data_go_items(payload)
        self.assertEqual(items, [{"a": 1}])
        self.assertEqual(meta["total_count"], 1)

    def test_vworld_feature_is_extracted(self) -> None:
        payload = {"response": {"status": "OK", "result": {"featureCollection": {"features": [{"properties": {"pnu": "1"}}]}}}}
        features, meta = vworld_features(payload)
        self.assertEqual(features[0]["properties"]["pnu"], "1")
        self.assertEqual(meta["status"], "OK")

    def test_vworld_not_found_is_empty_coverage_not_an_api_error(self) -> None:
        payload = {"response": {"status": "NOT_FOUND", "record": {"total": "0"}}}
        features, meta = vworld_features(payload)
        self.assertEqual(features, [])
        self.assertEqual(vworld_semantic_status(meta), "api_no_features")

    def test_vworld_ok_without_features_is_still_empty_coverage(self) -> None:
        self.assertEqual(vworld_semantic_status({"status": "OK"}, 0), "api_no_features")

    def test_vworld_credential_error_remains_an_api_error(self) -> None:
        payload = {
            "response": {
                "status": "ERROR",
                "error": {"code": "INCORRECT_KEY", "text": "redacted fixture"},
            }
        }
        _, meta = vworld_features(payload)
        self.assertEqual(vworld_semantic_status(meta), "api_error")

    def test_eia_gml_and_point_in_polygon(self) -> None:
        payload = b'''<wfs:FeatureCollection xmlns:wfs="urn:wfs" xmlns:x="urn:x" xmlns:gml="urn:gml"><gml:featureMembers><x:BSNS_AREA><x:MGTNO>A1</x:MGTNO><x:the_geom><gml:Polygon><gml:exterior><gml:LinearRing><gml:posList>126 33 127 33 127 34 126 34 126 33</gml:posList></gml:LinearRing></gml:exterior></gml:Polygon></x:the_geom></x:BSNS_AREA></gml:featureMembers></wfs:FeatureCollection>'''
        features = eia_features(payload)
        self.assertEqual(features[0]["attributes"]["MGTNO"], "A1")
        self.assertTrue(point_in_ring(126.5, 33.5, features[0]["rings_lon_lat"][0]))
        self.assertFalse(point_in_ring(128, 33.5, features[0]["rings_lon_lat"][0]))


class SnapshotMergeTests(unittest.TestCase):
    def test_new_vworld_records_replace_stale_vworld_probe(self) -> None:
        base = [
            {"source_id": "building_hub_basis", "request_hash": "building"},
            {"source_id": "vworld_cadastral", "request_hash": "stale"},
        ]
        fresh = [{"source_id": "vworld_cadastral", "request_hash": "fresh"}]
        merged = merge_request_records(base, fresh)
        self.assertEqual([record["request_hash"] for record in merged], ["building", "fresh"])

    def test_vworld_anchor_becomes_cross_source_pnu_record(self) -> None:
        records = vworld_pnu_records(
            [
                {
                    "target_id": "JJ-OREUM-001",
                    "request_hash": "hash",
                    "pnu": "5011010100100010000",
                    "address": "fixture",
                    "evidence_grade": "C",
                    "interpretation": "representative point parcel",
                },
                {"target_id": "JJ-OREUM-002", "pnu": None},
            ]
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_id"], "JJ-OREUM-001")
        self.assertEqual(records[0]["source_id"], "vworld_cadastral")

    def test_vworld_validation_rejects_semantic_api_error(self) -> None:
        records = [{"target_id": "one", "semantic_status": "api_error"}]
        anchors = [{"target_id": "one", "feature_count": 0}]
        with self.assertRaisesRegex(ValueError, "failed semantically"):
            validate_vworld_snapshot(records, anchors, ["one"])

    def test_candidate_keeps_conflicting_parcel_sources_and_abstains(self) -> None:
        target = {
            "target_id": "candidate",
            "target_kind": "olmoearth_candidate",
            "lat": 33.3,
            "lon": 126.6,
            "observation_dates": ["2024-05-01", "2025-05-01"],
        }
        vworld = {
            "target_id": "candidate",
            "source_id": "vworld_cadastral",
            "pnu": "5013025324201990000",
        }
        farmmap = {
            "target_id": "candidate",
            "source_id": "mafra_farmmap_jeju",
            "pnu": "5013025324202000000",
        }
        event = {
            "legal_dong_code": "5013025324",
            "pnu": "5013025324201990000",
            "management_id": "event",
            "permit_date": "20240601",
        }
        record = build_candidate_evidence(
            [target], [vworld, farmmap], [event], [], []
        )[0]
        self.assertEqual(record["representative_parcel"], vworld)
        self.assertEqual(record["parcel_pnu_relation"], "conflict")
        self.assertEqual(len(record["parcel_evidence"]), 2)
        self.assertEqual(record["exact_parcel_building_event_count"], 1)
        self.assertEqual(record["causal_evidence_grade"], "U")
        self.assertEqual(record["decision"], "abstain")


if __name__ == "__main__":
    unittest.main()
