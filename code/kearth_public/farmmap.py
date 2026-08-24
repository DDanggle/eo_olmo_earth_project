"""Adapter for the official annual Jeju Farm Map shapefile bundle.

Geospatial dependencies are imported lazily so CSV-only audits remain usable with
the Python standard library. Candidate coordinates are joined entirely offline.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import file_sha256, schema_sha256, value_sha256
from .models import EvidenceEdge, EvidenceGrade, SourceManifest
from .pnu import InvalidPNU, PNU
from .temporal import parse_date


FARMMAP_COLUMNS = (
    "ID",
    "UID",
    "CLSF_NM",
    "CLSF_CD",
    "STDG_CD",
    "STDG_ADDR",
    "PNU",
    "LDCG_CD",
    "SB_PNU",
    "SB_LDCG_CD",
    "AREA",
    "CAD_CON_RA",
    "SOURCE_NM",
    "SOURCE_CD",
    "FLIGHT_YMD",
    "UPDT_YMD",
    "UPDT_TP_NM",
    "UPDT_TP_CD",
    "CHG_RSN_NM",
    "CHG_RSN_CD",
    "FL_ARMT_YN",
    "O_UID",
    "O_CLSF_NM",
)
ALLOWED_ARCHIVE_SUFFIXES = {".shp", ".shx", ".dbf", ".prj", ".cpg"}


class MissingGeospatialDependency(RuntimeError):
    pass


def _load_geospatial_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import shapefile
        from pyproj import CRS, Transformer
        from shapely.geometry import Point, shape
    except ImportError as exc:
        raise MissingGeospatialDependency(
            "FarmMap ingestion requires requirements-public-data.txt"
        ) from exc
    return shapefile, CRS, Transformer, (Point, shape)


@dataclass(frozen=True, slots=True)
class TargetPoint:
    target_id: str
    target_kind: str
    lon: float
    lat: float
    oreum_id: str | None = None
    change_id: str | None = None
    t_before: str | None = None
    t_after: str | None = None

    def __post_init__(self) -> None:
        if not (-180 <= self.lon <= 180 and -90 <= self.lat <= 90):
            raise ValueError(f"invalid WGS84 coordinate for {self.target_id}")
        if bool(self.t_before) != bool(self.t_after):
            raise ValueError("t_before and t_after must be supplied together")
        if self.t_before and parse_date(self.t_before) >= parse_date(self.t_after):
            raise ValueError("t_before must precede t_after")


@dataclass(frozen=True, slots=True)
class ProjectedTarget:
    source: TargetPoint
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class FarmMapResult:
    manifest: SourceManifest
    evidence_edges: list[EvidenceEdge]
    target_summary: dict[str, Any]
    permit_links: list[dict[str, Any]]


class PointGrid:
    """Tiny deterministic grid index optimized for a few hundred query points."""

    def __init__(self, points: list[ProjectedTarget], cell_size: float = 1_000.0):
        if cell_size <= 0:
            raise ValueError("cell_size must be positive")
        self.points = points
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, point in enumerate(points):
            self.cells[self._cell(point.x, point.y)].append(index)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (math.floor(x / self.cell_size), math.floor(y / self.cell_size))

    def query_bbox(self, bbox: Iterable[float]) -> list[ProjectedTarget]:
        min_x, min_y, max_x, max_y = bbox
        start_x, start_y = self._cell(min_x, min_y)
        end_x, end_y = self._cell(max_x, max_y)
        indices: set[int] = set()
        for cell_x in range(start_x, end_x + 1):
            for cell_y in range(start_y, end_y + 1):
                indices.update(self.cells.get((cell_x, cell_y), ()))
        return [
            self.points[index]
            for index in sorted(indices)
            if min_x <= self.points[index].x <= max_x
            and min_y <= self.points[index].y <= max_y
        ]


def load_targets(
    candidate_path: Path,
    candidate_manifest_path: Path,
    oreum_registry_path: Path,
) -> list[TargetPoint]:
    candidate_data = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    oreum_data = json.loads(oreum_registry_path.read_text(encoding="utf-8"))
    manifest_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in candidate_manifest.get("candidates", [])
    }
    targets: list[TargetPoint] = []
    for site in candidate_data.get("sites", []):
        candidate = manifest_by_id.get(site["candidate_id"])
        if candidate is None:
            raise ValueError(
                f"candidate missing from frozen manifest: {site['candidate_id']}"
            )
        when = str(candidate.get("algorithm", {}).get("when") or "")
        match = re.fullmatch(r"(\d{4})->(\d{4})", when)
        if not match:
            raise ValueError(f"candidate has unsupported change interval {when!r}")
        before_year, after_year = match.groups()
        season = candidate.get("season_aligned_rgb", {})
        try:
            t_before = season[before_year]["acquisition_date"]
            t_after = season[after_year]["acquisition_date"]
        except KeyError as exc:
            raise ValueError(
                f"candidate lacks acquisition dates for {when}: {site['candidate_id']}"
            ) from exc
        targets.append(
            TargetPoint(
                target_id=site["candidate_id"],
                target_kind="change_candidate",
                lon=float(site["lon"]),
                lat=float(site["lat"]),
                change_id=site["candidate_id"],
                t_before=t_before,
                t_after=t_after,
            )
        )
    for record in oreum_data.get("records", []):
        location = record.get("location") or {}
        if location.get("status") != "resolved_offline_osm_peak":
            continue
        if location.get("lat") is None or location.get("lon") is None:
            continue
        targets.append(
            TargetPoint(
                target_id=record["oreum_id"],
                target_kind="oreum_osm_point",
                lon=float(location["lon"]),
                lat=float(location["lat"]),
                oreum_id=record["oreum_id"],
            )
        )
    target_ids = [target.target_id for target in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("candidate/oreum target IDs are not unique")
    return sorted(targets, key=lambda target: target.target_id)


def _safe_extract_components(zip_path: Path, destination: Path) -> list[Path]:
    destination = destination.resolve()
    shapefiles: list[Path] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in ALLOWED_ARCHIVE_SUFFIXES:
                continue
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"unsafe archive member path: {info.filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(chunk)
            if suffix == ".shp":
                shapefiles.append(target)
    if len(shapefiles) != 2:
        raise ValueError(
            f"expected two Jeju FarmMap shapefiles, found {len(shapefiles)}"
        )
    for shp_path in shapefiles:
        missing = [
            suffix
            for suffix in (".shx", ".dbf", ".prj")
            if not shp_path.with_suffix(suffix).exists()
        ]
        if missing:
            raise ValueError(
                f"FarmMap shapefile is missing components {missing}: {shp_path}"
            )
    return sorted(shapefiles, key=lambda path: path.as_posix())


def _edge_id(target_id: str, farm_id: str) -> str:
    digest = value_sha256(["mafra_farmmap_jeju", target_id, farm_id])[:20]
    return f"farmmap:{digest}"


def ingest_farmmap(
    zip_path: Path,
    *,
    targets: list[TargetPoint],
    permit_rows_by_pnu: dict[str, list[dict[str, str]]],
    catalog_url: str,
    download_url: str,
    retrieved_at: str,
) -> FarmMapResult:
    shapefile, CRS, Transformer, geometry = _load_geospatial_dependencies()
    Point, to_shape = geometry
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    projected = [
        ProjectedTarget(target, *transformer.transform(target.lon, target.lat))
        for target in targets
    ]
    grid = PointGrid(projected)

    class_counts: Counter[str] = Counter()
    update_type_counts: Counter[str] = Counter()
    change_reason_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    flight_dates: Counter[str] = Counter()
    update_dates: Counter[str] = Counter()
    municipality_counts: dict[str, int] = {}
    invalid_pnu_count = 0
    invalid_pnu_examples: list[dict[str, str]] = []
    pnu_present_count = 0
    row_count = 0
    evidence_edges: list[EvidenceEdge] = []
    permit_links: list[dict[str, Any]] = []
    target_hits: Counter[str] = Counter()
    matched_permit_pairs: set[tuple[str, int, str]] = set()
    observed_columns: tuple[str, ...] | None = None

    with tempfile.TemporaryDirectory(prefix="kearth-farmmap-") as temp_name:
        shapefiles = _safe_extract_components(zip_path, Path(temp_name))
        for shp_path in shapefiles:
            prj_text = shp_path.with_suffix(".prj").read_text(
                encoding="utf-8", errors="strict"
            )
            crs = CRS.from_wkt(prj_text)
            if crs.to_epsg() != 5179:
                raise ValueError(
                    f"unexpected FarmMap CRS {crs.to_string()}: {shp_path}"
                )
            with shapefile.Reader(str(shp_path), encoding="cp949") as reader:
                columns = tuple(field[0] for field in reader.fields[1:])
                if columns != FARMMAP_COLUMNS:
                    raise ValueError(
                        f"unexpected FarmMap schema for {shp_path}: {columns}"
                    )
                if observed_columns is None:
                    observed_columns = columns
                municipality_counts[shp_path.parent.name] = len(reader)
                for index, shape_record in enumerate(reader.iterShapeRecords()):
                    row_count += 1
                    record = shape_record.record.as_dict()
                    farm_id = str(record["ID"]).strip()
                    raw_pnu = str(record.get("PNU") or "").strip()
                    class_counts[str(record.get("CLSF_NM") or "unknown")] += 1
                    update_type_counts[str(record.get("UPDT_TP_NM") or "unknown")] += 1
                    change_reason_counts[
                        str(record.get("CHG_RSN_NM") or "unknown")
                    ] += 1
                    source_counts[str(record.get("SOURCE_NM") or "unknown")] += 1
                    flight_dates[str(record.get("FLIGHT_YMD") or "unknown")] += 1
                    update_dates[str(record.get("UPDT_YMD") or "unknown")] += 1
                    if raw_pnu:
                        pnu_present_count += 1
                        try:
                            PNU.parse(raw_pnu)
                        except InvalidPNU as exc:
                            invalid_pnu_count += 1
                            if len(invalid_pnu_examples) < 20:
                                invalid_pnu_examples.append(
                                    {
                                        "farm_id": farm_id,
                                        "pnu": raw_pnu,
                                        "address": str(record.get("STDG_ADDR") or ""),
                                        "reason": str(exc),
                                    }
                                )
                    for permit_index, permit in enumerate(
                        permit_rows_by_pnu.get(raw_pnu, ())
                    ):
                        pair = (raw_pnu, permit_index, farm_id)
                        if pair in matched_permit_pairs:
                            continue
                        matched_permit_pairs.add(pair)
                        permit_links.append(
                            {
                                "farm_id": farm_id,
                                "pnu": raw_pnu,
                                "farm_class": record.get("CLSF_NM") or "",
                                "farm_address": record.get("STDG_ADDR") or "",
                                "farm_flight_date": record.get("FLIGHT_YMD") or "",
                                "farm_update_date": record.get("UPDT_YMD") or "",
                                "permit_date": permit.get("허가일자") or "",
                                "permit_action": permit.get("개발행위명") or "",
                                "permit_purpose": permit.get("개발행위목적") or "",
                                "relation": "exact_pnu_cross_source_context",
                                "causal_claim_allowed": False,
                            }
                        )

                    bbox_targets = grid.query_bbox(shape_record.shape.bbox)
                    if not bbox_targets:
                        continue
                    polygon = to_shape(shape_record.shape.__geo_interface__)
                    if polygon.is_empty or not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    for target in bbox_targets:
                        if not polygon.covers(Point(target.x, target.y)):
                            continue
                        source = target.source
                        state_value = str(
                            record.get("FLIGHT_YMD") or record.get("UPDT_YMD") or ""
                        ).strip()
                        state_date_basis = (
                            "flight_date"
                            if record.get("FLIGHT_YMD")
                            else "update_date"
                            if record.get("UPDT_YMD")
                            else None
                        )
                        temporal_method = None
                        day_gap = None
                        relation = "point_inside_dated_farm_polygon"
                        grade = EvidenceGrade.C
                        if source.target_kind == "change_candidate":
                            if not state_value:
                                raise ValueError(
                                    f"FarmMap hit lacks a state date: {farm_id}"
                                )
                            state_date = parse_date(state_value)
                            before_date = parse_date(source.t_before or "")
                            after_date = parse_date(source.t_after or "")
                            if state_date <= before_date:
                                relation = "official_pre_change_state_at_point"
                                temporal_method = "state_date_on_or_before_t_before"
                                day_gap = (before_date - state_date).days
                            elif state_date <= after_date:
                                relation = (
                                    "official_state_within_change_window_at_point"
                                )
                                temporal_method = (
                                    "state_date_within_observation_interval"
                                )
                                day_gap = 0
                            else:
                                relation = "official_post_change_state_at_point"
                                temporal_method = "state_date_on_or_after_t_after"
                                day_gap = (state_date - after_date).days
                            grade = EvidenceGrade.B
                        target_hits[source.target_id] += 1
                        evidence_edges.append(
                            EvidenceEdge(
                                edge_id=_edge_id(source.target_id, farm_id),
                                source_id="mafra_farmmap_jeju",
                                source_record_id=farm_id,
                                relation=relation,
                                evidence_grade=grade,
                                target_id=source.target_id,
                                oreum_id=source.oreum_id,
                                change_id=source.change_id,
                                pnu=raw_pnu or None,
                                spatial_method="official_polygon_covers_offline_point",
                                temporal_method=temporal_method,
                                day_gap=day_gap,
                                no_match_interpretable=False,
                                attributes={
                                    "target_kind": source.target_kind,
                                    "t_before": source.t_before,
                                    "t_after": source.t_after,
                                    "farm_class": record.get("CLSF_NM") or "",
                                    "farm_address": record.get("STDG_ADDR") or "",
                                    "farm_area_m2": record.get("AREA"),
                                    "farm_source": record.get("SOURCE_NM") or "",
                                    "flight_date": record.get("FLIGHT_YMD") or "",
                                    "update_date": record.get("UPDT_YMD") or "",
                                    "state_date_used": state_value or None,
                                    "state_date_basis": state_date_basis,
                                    "update_type": record.get("UPDT_TP_NM") or "",
                                    "change_reason": record.get("CHG_RSN_NM") or "",
                                    "warning": (
                                        "dated official land-state evidence; the target is a "
                                        "point and this does not establish change cause"
                                    ),
                                },
                            )
                        )

    if observed_columns is None:
        raise ValueError("FarmMap archive contained no readable records")
    evidence_edges.sort(key=lambda edge: (edge.target_id or "", edge.source_record_id))
    permit_links.sort(
        key=lambda row: (
            row["pnu"],
            row["farm_id"],
            row["permit_date"],
            row["permit_action"],
        )
    )
    manifest = SourceManifest(
        source_id="mafra_farmmap_jeju",
        provider="농림수산식품교육문화정보원",
        catalog_url=catalog_url,
        download_url=download_url,
        snapshot_date="2025-12-31",
        retrieved_at=retrieved_at,
        raw_file_name=zip_path.name,
        raw_sha256=file_sha256(zip_path),
        raw_bytes=zip_path.stat().st_size,
        license="이용허락범위 제한 없음",
        access_method="public_portal_file_download",
        data_format="ESRI Shapefile in ZIP",
        columns=observed_columns,
        schema_sha256=schema_sha256(observed_columns),
        row_count=row_count,
        spatial_coverage="Jeju-si and Seogwipo-si farm polygons",
        temporal_coverage="source-specific FLIGHT_YMD; updated 2025-12-31",
        crs="EPSG:5179",
        geometry_type="Polygon",
        quality={
            "municipality_rows": dict(sorted(municipality_counts.items())),
            "pnu_present_count": pnu_present_count,
            "valid_pnu_count": pnu_present_count - invalid_pnu_count,
            "invalid_pnu_count": invalid_pnu_count,
            "invalid_pnu_examples": invalid_pnu_examples,
            "class_counts": dict(sorted(class_counts.items())),
            "update_type_counts": dict(sorted(update_type_counts.items())),
            "change_reason_counts": dict(sorted(change_reason_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "flight_dates": dict(sorted(flight_dates.items())),
            "update_dates": dict(sorted(update_dates.items())),
            "point_targets_queried": len(targets),
            "point_targets_with_hits": len(target_hits),
            "point_polygon_edges": len(evidence_edges),
            "point_polygon_edges_by_grade": dict(
                sorted(
                    Counter(
                        edge.evidence_grade.value for edge in evidence_edges
                    ).items()
                )
            ),
            "exact_permit_pnu_links": len(permit_links),
            "exact_permit_unique_pnu": len({row["pnu"] for row in permit_links}),
            "exact_permit_unique_farm_polygon": len(
                {row["farm_id"] for row in permit_links}
            ),
            "no_match_interpretable": False,
            "no_match_reason": (
                "FarmMap covers agricultural polygons only; a point miss does not mean "
                "no land change or no administrative event"
            ),
        },
    )
    by_kind: dict[str, dict[str, int]] = defaultdict(
        lambda: {"queried": 0, "with_hit": 0, "edges": 0}
    )
    for target in targets:
        by_kind[target.target_kind]["queried"] += 1
        if target_hits[target.target_id]:
            by_kind[target.target_kind]["with_hit"] += 1
            by_kind[target.target_kind]["edges"] += target_hits[target.target_id]
    target_summary = {
        "by_kind": dict(sorted(by_kind.items())),
        "targets_with_hits": dict(sorted(target_hits.items())),
        "interpretation": (
            "a change-coordinate hit is B-grade dated state evidence; an oreum OSM-point "
            "hit remains C-grade because the input location is not an official boundary; "
            "neither is a causal attribution"
        ),
    }
    return FarmMapResult(manifest, evidence_edges, target_summary, permit_links)
