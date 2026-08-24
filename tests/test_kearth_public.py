from __future__ import annotations

# Tests add the repository's script directory before importing its local package.
# ruff: noqa: E402

import csv
import importlib.util
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from kearth_public.canonical import canonical_json, file_sha256, value_sha256
from kearth_public.csv_sources import (
    audit_development_permits,
    ingest_jeju_forest_use,
)
from kearth_public.farmmap import (
    FARMMAP_COLUMNS,
    PointGrid,
    ProjectedTarget,
    TargetPoint,
    ingest_farmmap,
)
from kearth_public.models import EvidenceEdge, EvidenceGrade, SourceManifest
from kearth_public.pnu import InvalidPNU, PNU
from kearth_public.temporal import CoverageAudit, DateInterval, parse_date


class CanonicalTests(unittest.TestCase):
    def test_canonical_json_is_key_order_independent(self) -> None:
        left = {"나": [2, 1], "a": {"y": 2, "x": 1}}
        right = {"a": {"x": 1, "y": 2}, "나": [2, 1]}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(value_sha256(left), value_sha256(right))

    def test_file_sha_is_chunk_independent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "sample.bin"
            path.write_bytes(b"abc" * 1_000_000)
            self.assertEqual(len(file_sha256(path)), 64)


class PNUTests(unittest.TestCase):
    def test_parse_regular_and_mountain_pnu(self) -> None:
        regular = PNU.parse("5011025331116020001")
        mountain = PNU.parse("5011025632240190000")
        self.assertEqual(regular.legal_dong_code, "5011025331")
        self.assertFalse(regular.mountain)
        self.assertEqual(regular.main_number, 1602)
        self.assertEqual(regular.sub_number, 1)
        self.assertTrue(mountain.mountain)

    def test_rejects_bad_length_characters_and_land_code(self) -> None:
        for raw in ("", "5011", "5011025331A16020001", "5011025331916020001"):
            with self.subTest(raw=raw), self.assertRaises(InvalidPNU):
                PNU.parse(raw)


class TemporalTests(unittest.TestCase):
    def test_parse_supported_date_forms(self) -> None:
        expected = date(2025, 12, 31)
        for raw in ("20251231", "2025-12-31", "2025/12/31", "2025.12.31"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_date(raw), expected)

    def test_overlap_is_closed_at_boundaries(self) -> None:
        before = DateInterval.from_values("2023-05-01", "2024-05-01")
        event = DateInterval.from_values("2024-05-01", "2024-05-01")
        self.assertTrue(before.overlaps(event))
        self.assertEqual(before.day_gap(event), 0)

    def test_gap_and_expansion(self) -> None:
        observed = DateInterval.from_values("2024-01-01", "2024-01-31")
        event = DateInterval.from_values("2024-02-10", "2024-02-10")
        self.assertEqual(observed.day_gap(event), 10)
        self.assertTrue(observed.expanded(10).overlaps(event))

    def test_no_match_requires_every_coverage_check(self) -> None:
        complete = CoverageAudit(True, True, True, True, True)
        incomplete = CoverageAudit(True, True, False, True, True)
        self.assertTrue(complete.no_match_interpretable)
        self.assertFalse(incomplete.no_match_interpretable)
        self.assertEqual(incomplete.failed_checks(), ["event_population_complete"])


class ModelTests(unittest.TestCase):
    def manifest(self) -> SourceManifest:
        return SourceManifest(
            source_id="fixture",
            provider="fixture provider",
            catalog_url="https://example.test/catalog",
            download_url="https://example.test/file",
            snapshot_date="2025-12-31",
            retrieved_at="2026-08-22T07:22:00Z",
            raw_file_name="fixture.csv",
            raw_sha256="a" * 64,
            raw_bytes=10,
            license="test",
            access_method="test",
            data_format="CSV",
            columns=("a", "b"),
            schema_sha256="b" * 64,
            row_count=1,
            spatial_coverage="fixture",
            temporal_coverage="fixture",
        )

    def test_manifest_serializes(self) -> None:
        self.assertEqual(self.manifest().to_dict()["source_id"], "fixture")

    def test_manifest_rejects_non_https_source(self) -> None:
        values = self.manifest().to_dict()
        values["catalog_url"] = "http://example.test/catalog"
        values["columns"] = tuple(values["columns"])
        with self.assertRaises(ValueError):
            SourceManifest(**values)

    def test_grade_b_requires_spatial_and_temporal_methods(self) -> None:
        for spatial_method, temporal_method in (
            (None, "date_overlap"),
            ("point_in_polygon", None),
        ):
            with self.subTest(
                spatial_method=spatial_method,
                temporal_method=temporal_method,
            ):
                with self.assertRaises(ValueError):
                    EvidenceEdge(
                        edge_id="edge",
                        source_id="source",
                        source_record_id="record",
                        relation="test",
                        evidence_grade=EvidenceGrade.B,
                        target_id="target",
                        spatial_method=spatial_method,
                        temporal_method=temporal_method,
                    )


class CsvAdapterTests(unittest.TestCase):
    def test_forest_use_cp949_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "forest.csv"
            path.write_text(
                "해당연도,건수,면적(ha),데이터기준일자\n"
                "2023,714,230.6,2026-06-30\n"
                "2024,542,74.2,2026-06-30\n",
                encoding="cp949",
            )
            result = ingest_jeju_forest_use(
                path,
                catalog_url="https://example.test/catalog",
                download_url="https://example.test/file",
                retrieved_at="2026-08-22T07:22:00Z",
            )
            self.assertEqual(result.manifest.row_count, 2)
            self.assertEqual(result.manifest.quality["source_encoding"], "cp949")
            self.assertEqual(result.summary["2023"]["approval_count"], 714)
            self.assertFalse(result.summary["parcel_joinable"])

    def test_forest_use_rejects_year_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "forest.csv"
            path.write_text(
                "해당연도,건수,면적(ha),데이터기준일자\n"
                "2022,1,1,2026-06-30\n"
                "2024,1,1,2026-06-30\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                ingest_jeju_forest_use(
                    path,
                    catalog_url="https://example.test/catalog",
                    download_url="https://example.test/file",
                    retrieved_at="2026-08-22T07:22:00Z",
                )

    def test_permit_audit_never_treats_no_match_as_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "permits.csv"
            columns = ["PNU", "위치명", "허가일자", "개발행위명", "개발행위목적"]
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "PNU": "5011025331116020001",
                        "위치명": "제주",
                        "허가일자": "2025-01-02",
                        "개발행위명": "형질변경",
                        "개발행위목적": "fixture",
                    }
                )
            audit = audit_development_permits(path)
            self.assertEqual(audit.quality["valid_pnu_count"], 1)
            self.assertFalse(audit.quality["no_match_interpretable"])


class PointGridTests(unittest.TestCase):
    def test_bbox_query_is_inclusive_and_deterministic(self) -> None:
        points = [
            ProjectedTarget(TargetPoint("b", "test", 126.0, 33.0), 2_000, 2_000),
            ProjectedTarget(TargetPoint("a", "test", 126.0, 33.0), 1_000, 1_000),
        ]
        grid = PointGrid(points, cell_size=1_000)
        found = grid.query_bbox((1_000, 1_000, 2_000, 2_000))
        self.assertEqual([point.source.target_id for point in found], ["b", "a"])


class ArchiveSafetyTests(unittest.TestCase):
    def test_unsafe_archive_member_is_rejected(self) -> None:
        from kearth_public.farmmap import _safe_extract_components

        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("../escape.shp", b"bad")
                output.writestr("../escape.shx", b"bad")
                output.writestr("../escape.dbf", b"bad")
                output.writestr("../escape.prj", b"bad")
            with self.assertRaisesRegex(ValueError, "unsafe archive"):
                _safe_extract_components(archive, root / "out")


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("shapefile", "shapely", "pyproj")),
    "optional geospatial test dependencies are not installed",
)
class FarmMapIntegrationTests(unittest.TestCase):
    def _write_fixture_shapefile(
        self,
        directory: Path,
        name: str,
        polygon: list[list[float]],
        farm_id: str,
        pnu: str,
    ) -> None:
        import shapefile
        from pyproj import CRS

        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with shapefile.Writer(
            str(path), shapeType=shapefile.POLYGON, encoding="cp949"
        ) as writer:
            for column in FARMMAP_COLUMNS:
                if column in {"AREA", "CAD_CON_RA"}:
                    writer.field(column, "N", size=18, decimal=2)
                else:
                    writer.field(column, "C", size=120)
            values = {
                "ID": farm_id,
                "UID": farm_id,
                "CLSF_NM": "밭",
                "CLSF_CD": "02",
                "STDG_CD": pnu[:10],
                "STDG_ADDR": "제주 fixture",
                "PNU": pnu,
                "LDCG_CD": "전",
                "SB_PNU": pnu,
                "SB_LDCG_CD": "전",
                "AREA": 100.0,
                "CAD_CON_RA": 100.0,
                "SOURCE_NM": "항공정사영상",
                "SOURCE_CD": "01",
                "FLIGHT_YMD": "2024-12-31",
                "UPDT_YMD": "2025-12-31",
                "UPDT_TP_NM": "변경",
                "UPDT_TP_CD": "03",
                "CHG_RSN_NM": "개선",
                "CHG_RSN_CD": "07",
                "FL_ARMT_YN": "N",
                "O_UID": farm_id,
                "O_CLSF_NM": "밭",
            }
            writer.poly([polygon])
            writer.record(*(values[column] for column in FARMMAP_COLUMNS))
        path.with_suffix(".prj").write_text(
            CRS.from_epsg(5179).to_wkt(), encoding="utf-8"
        )
        path.with_suffix(".cpg").write_text("EUC-KR", encoding="ascii")

    def test_offline_point_polygon_and_exact_pnu_join(self) -> None:
        from pyproj import Transformer

        pnu = "5011025331116020001"
        target = TargetPoint(
            "change-1",
            "change_candidate",
            126.5747,
            33.5087,
            t_before="2024-05-01",
            t_after="2025-05-01",
        )
        oreum_target = TargetPoint(
            "JJ-OREUM-001",
            "oreum_osm_point",
            126.5747,
            33.5087,
            oreum_id="JJ-OREUM-001",
        )
        x, y = Transformer.from_crs(4326, 5179, always_xy=True).transform(
            target.lon, target.lat
        )
        near = [
            [x - 50, y - 50],
            [x - 50, y + 50],
            [x + 50, y + 50],
            [x + 50, y - 50],
            [x - 50, y - 50],
        ]
        far = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "source"
            self._write_fixture_shapefile(source / "a", "a", near, "farm-1", pnu)
            self._write_fixture_shapefile(
                source / "b", "b", far, "farm-2", "5011025331116030000"
            )
            archive = root / "farmmap.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        output.write(path, path.relative_to(source))
            result = ingest_farmmap(
                archive,
                targets=[target, oreum_target],
                permit_rows_by_pnu={
                    pnu: [
                        {
                            "허가일자": "20250102",
                            "개발행위명": "토지형질변경",
                            "개발행위목적": "fixture",
                        }
                    ]
                },
                catalog_url="https://example.test/catalog",
                download_url="https://example.test/file",
                retrieved_at="2026-08-22T07:22:00Z",
            )
            self.assertEqual(result.manifest.row_count, 2)
            self.assertEqual(len(result.evidence_edges), 2)
            grades = {
                edge.target_id: edge.evidence_grade for edge in result.evidence_edges
            }
            self.assertEqual(grades["change-1"], EvidenceGrade.B)
            self.assertEqual(grades["JJ-OREUM-001"], EvidenceGrade.C)
            change_edge = next(
                edge for edge in result.evidence_edges if edge.target_id == "change-1"
            )
            self.assertEqual(
                change_edge.temporal_method, "state_date_within_observation_interval"
            )
            self.assertEqual(change_edge.day_gap, 0)
            self.assertEqual(change_edge.attributes["state_date_basis"], "flight_date")
            self.assertEqual(change_edge.attributes["state_date_used"], "2024-12-31")
            self.assertEqual(len(result.permit_links), 1)
            self.assertFalse(result.permit_links[0]["causal_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
