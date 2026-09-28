"""Offline, provenance-preserving evidence catalog queries. No model inference.

KuroSiwo metadata contains reference-label percentages and event dates, not
predictions, damage estimates, acquisition dates, or georeferenced footprints.
"""
from __future__ import annotations

import copy
import calendar
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

SCHEMA = "eo-evidence-catalog-v0"
_QUERY_KEYS = {"dataset", "event_type", "aoi_id", "split", "start", "end", "bbox",
               "min_flood_pct", "status", "land_cover", "limit"}
_KURO_LIMITATIONS = [
    "reference_labels_not_model_predictions",
    "event_date_not_verified_acquisition_date",
    "centroid_only_not_footprint",
    "land_cover_overlap_unavailable",
    "damage_and_cause_not_measured",
]
_POINT_RE = re.compile(r"POINT\s*\(\s*([^\s,()]+)\s+([^\s,()]+)\s*\)", re.IGNORECASE)
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty string without surrounding whitespace")
    return value


def _date(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DATE_RE.fullmatch(value):
        raise ValueError(f"{name} must be YYYY-MM-DD")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} is not a valid calendar date") from exc
    return value


def _point(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must have [longitude, latitude]")
    return [_number(value[0], f"{name}.longitude", -180, 180),
            _number(value[1], f"{name}.latitude", -90, 90)]


def _wkt_point(value: Any) -> list[float]:
    if not isinstance(value, str) or not (match := _POINT_RE.fullmatch(value.strip())):
        raise ValueError("centroid must be two-dimensional WKT POINT (longitude latitude)")
    try:
        numbers = [float(match[1]), float(match[2])]
    except ValueError as exc:
        raise ValueError("centroid must contain numeric coordinates") from exc
    return _point(numbers, "centroid")


def build_kurosiwo_catalog(meta_path: Path) -> dict:
    """Import existing reference metadata, rejecting malformed or duplicate rows."""
    path = Path(meta_path).resolve()
    raw = path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("metadata must be UTF-8") from exc
    records, seen = [], set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
            sid = _text(row["id"], "id")
            if sid in seen:
                raise ValueError(f"duplicate id: {sid}")
            split = _text(row["split"], "split")
            if split not in {"train", "validation", "test"}:
                raise ValueError("unknown KuroSiwo split")
            activation = row["actid"]
            if isinstance(activation, bool) or not isinstance(activation, (int, str)):
                raise ValueError("actid must be an integer or string identifier")
            aoi_id = _text(str(activation), "actid")
            region_id = _text(str(row["aoiid"]), "aoiid") if isinstance(row["aoiid"], (str, int)) and not isinstance(row["aoiid"], bool) else None
            if region_id is None:
                raise ValueError("aoiid must be an integer or string identifier")
            fraction = _number(row["pflood"], "pflood", 0, 100)
            permanent = _number(row["pwater"], "pwater", 0, 100)
            records.append({
                "id": sid, "dataset": "kurosiwo", "aoi_id": aoi_id,
                "region_id": region_id, "split": split, "event_type": "flood",
                "status": "reference_flood" if fraction > 0 else "reference_no_flood",
                "event_date": _date(row["flood_date"], "flood_date"),
                "point": _wkt_point(row["centroid"]),
                "flood_fraction_pct": fraction, "permanent_water_fraction_pct": permanent,
                "evidence": [], "limitations": list(_KURO_LIMITATIONS),
                "source_record": {"path": str(path), "line": line_number},
            })
            seen.add(sid)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"Invalid KuroSiwo metadata at line {line_number}: {exc}") from exc
    return {
        "schema": SCHEMA,
        "provenance": {
            "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "dataset": "kurosiwo", "record_count": len(records),
            "source_role": "dataset_reference_metadata_not_model_predictions",
            "fraction_basis": "pflood_and_pwater_reference_percentages_0_to_100",
            "date_basis": "flood_date_event_date_not_verified_image_acquisition_date",
            "centroid_basis": "source_stac_centroid_wkt_interpreted_as_longitude_latitude",
            "crs_status": "no_explicit_crs_field_in_export",
            "geometry_basis": "centroid_only_no_georeferenced_footprint",
            "impact_basis": "flood_reference_label_does_not_establish_damage_or_cause",
        },
        "records": records,
    }


def _validate_query(query: dict) -> dict:
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = set(query) - _QUERY_KEYS
    if unknown:
        raise ValueError(f"Unknown query keys: {sorted(unknown, key=str)}")
    result = copy.deepcopy(query)
    for key in ("dataset", "event_type", "aoi_id", "split", "status"):
        if key in result:
            result[key] = _text(result[key], key)
    for key in ("start", "end"):
        if key in result:
            result[key] = _date(result[key], key)
    if "start" in result and "end" in result and result["start"] > result["end"]:
        raise ValueError("start must not be later than end")
    if "bbox" in result:
        bbox = result["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must be [west, south, east, north]")
        w = _number(bbox[0], "bbox.west", -180, 180)
        s = _number(bbox[1], "bbox.south", -90, 90)
        e = _number(bbox[2], "bbox.east", -180, 180)
        n = _number(bbox[3], "bbox.north", -90, 90)
        if s > n:
            raise ValueError("bbox.south must not exceed bbox.north")
        result["bbox"] = [w, s, e, n]
    if "min_flood_pct" in result:
        result["min_flood_pct"] = _number(result["min_flood_pct"], "min_flood_pct", 0, 100)
    land_cover = result.get("land_cover", "all")
    if land_cover not in ("all", "cropland", "forest"):
        raise ValueError("land_cover must be all, cropland, or forest")
    result["land_cover"] = land_cover
    limit = result.get("limit", 100)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
        raise ValueError("limit must be an integer between 1 and 500")
    result["limit"] = limit
    return result


def _record_date_interval(value: Any, name: str) -> tuple[str | None, str | None, str]:
    """Expand known month precision into its interval, not an invented event day."""
    if value is None:
        return None, None, "unknown"
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = map(int, value.split("-"))
        try:
            first = date(year, month, 1)
            last = date(year, month, calendar.monthrange(year, month)[1])
        except ValueError as exc:
            raise ValueError(f"{name} is not a valid calendar month") from exc
        return first.isoformat(), last.isoformat(), "month"
    checked = _date(value, name)
    return checked, checked, "day"


def _time_fields(record: dict) -> tuple[str, str | None, str | None, str]:
    # Date fields never substitute for one another: an acquisition or export
    # date cannot silently replace a flood event date or first-change label.
    if record.get("dataset") == "kurosiwo":
        value = record.get("event_date")
        checked = _date(value, "event_date") if value is not None else None
        return "event_date", checked, checked, "day" if checked else "unknown"
    if record.get("status") == "change_supported":
        start, end, precision = _record_date_interval(record.get("first_change_date"), "first_change_date")
        return "first_change_date", start, end, precision
    start, _, start_precision = _record_date_interval(record.get("observation_start"), "observation_start")
    _, end, end_precision = _record_date_interval(record.get("observation_end"), "observation_end")
    if start is not None and end is not None and start > end:
        raise ValueError("record observation_start exceeds observation_end")
    precision = "month" if "month" in (start_precision, end_precision) else "day"
    if start is None or end is None:
        precision = "unknown"
    return "observation_interval", start, end, precision


def _sort_key(record: dict) -> tuple:
    dataset, sid = record.get("dataset", ""), record["id"]
    if dataset == "kurosiwo":
        return (dataset, -record.get("flood_fraction_pct", 0), sid)
    return (dataset, record.get("first_change_date") or record.get("observation_start") or "9999-12-31", sid)


def _land_cover_assessment(record: dict, requested: str) -> tuple[dict | None, str | None]:
    """Classify measurement availability separately from observed zero overlap.

    Areas describe reference flood labels intersected with a historical land
    cover map. They do not measure damage, crop loss, or physical attribution.
    Unknown area covers the supplied footprint pixels excluded by either mask.
    """
    if record.get('dataset') != 'kurosiwo':
        return None, 'dataset_not_supported'
    measured = record.get('land_cover_overlap')
    if not isinstance(measured, dict) or measured.get('status') != 'measured':
        return None, 'not_computed'
    baseline = measured.get('baseline')
    if not isinstance(baseline, dict) or not baseline.get('reference_end'):
        return None, 'baseline_time_unavailable'
    baseline_end = _date(baseline['reference_end'], 'baseline.reference_end')
    event_date = record.get('event_date')
    if event_date is None:
        return None, 'event_date_unavailable'
    if baseline_end >= _date(event_date, 'event_date'):
        return None, 'baseline_not_strictly_pre_event'
    classes = measured.get('classes')
    if not isinstance(classes, dict):
        return None, 'requested_class_not_computed'
    class_result = classes.get(requested)
    if not isinstance(class_result, dict):
        return None, 'requested_class_not_computed'
    expected_code = {'cropland': 40, 'forest': 10}[requested]
    if class_result.get('class_code') != expected_code:
        raise ValueError(f'{requested}: expected WorldCover class {expected_code}')
    # Missing coverage metadata is unavailable, never a complete negative.
    if any(key not in measured for key in ('joint_valid_area_m2', 'unknown_area_m2')):
        return None, 'coverage_metadata_unavailable'
    valid_area = _number(measured['joint_valid_area_m2'], 'joint_valid_area_m2', 0, 1e15)
    unknown_area = _number(measured['unknown_area_m2'], 'unknown_area_m2', 0, 1e15)
    overlap = _number(class_result.get('flood_overlap_m2'), 'flood_overlap_m2', 0, valid_area)
    if valid_area <= 0:
        return None, 'no_joint_valid_pixels'
    return {
        'requested_class': requested,
        'source_class_code': expected_code,
        'source_class_name': 'Cropland' if requested == 'cropland' else 'Tree cover',
        'flood_overlap_m2': overlap,
        'joint_valid_area_m2': valid_area,
        'unknown_area_m2': unknown_area,
        'pixel_coverage': 'complete' if unknown_area == 0 else 'partial',
        'baseline': copy.deepcopy(baseline),
        'meaning': 'observed_reference_flood_land_cover_overlap_not_damage',
    }, None


def query_catalog(catalog: dict, query: dict) -> dict:
    """Apply explicit filters without inferring events, labels, or impact.

    Geographic matching uses source centroids only, including wrapped longitude
    boxes. A land-cover request is unavailable, not an empty negative finding.
    """
    q = _validate_query(query)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("records"), list):
        raise ValueError("catalog must contain a records list")
    limitations = [
        "searches_only_the_supplied_catalog_not_all_places_or_events",
        "reference_labels_are_not_model_predictions_or_damage_estimates",
        "date_filter_uses_each_datasets_declared_time_basis",
    ]
    if "bbox" in q:
        limitations.append("bbox_matches_centroids_not_footprint_intersections")
    response = {"status": "ok", "query": q, "matched_count": 0,
                "returned_count": 0, "records": [], "limitations": limitations,
                "excluded_missing_geometry": 0}
    land_filter = q['land_cover'] != 'all'
    coverage = {'scope_count': 0, 'assessed_count': 0, 'unassessed_count': 0,
                'matched_count': 0, 'observed_zero_count': 0,
                'fully_observed_zero_count': 0, 'partial_pixel_count': 0,
                'unassessed_reasons': {}}
    selected = []
    missing_time = 0
    seen = set()
    for record in catalog["records"]:
        if not isinstance(record, dict):
            raise ValueError("every catalog record must be an object")
        sid = _text(record.get("id"), "record.id")
        dataset = _text(record.get("dataset"), "record.dataset")
        key = (dataset, sid)
        if key in seen:
            raise ValueError(f"duplicate catalog record: {dataset}/{sid}")
        seen.add(key)
        # Scope is checked before date semantics: unselected datasets cannot
        # acquire matches through another dataset's dates or fallback fields.
        if any(record.get(key) != q[key] for key in ("dataset", "event_type", "aoi_id", "split", "status") if key in q):
            continue
        basis, start, end, precision = _time_fields(record)
        if "start" in q or "end" in q:
            if start is None or end is None:
                missing_time += 1
                continue
            if ("start" in q and end < q["start"]) or ("end" in q and start > q["end"]):
                continue
        if "min_flood_pct" in q:
            value = record.get("flood_fraction_pct")
            if value is None:
                continue
            if _number(value, "record.flood_fraction_pct", 0, 100) < q["min_flood_pct"]:
                continue
        if "bbox" in q:
            if record.get("point") is None:
                response["excluded_missing_geometry"] += 1
                continue
            lon, lat = _point(record["point"], "record.point")
            west, south, east, north = q["bbox"]
            lon_inside = west <= lon <= east if west <= east else (lon >= west or lon <= east)
            if not lon_inside or not south <= lat <= north:
                continue
        result = copy.deepcopy(record)
        result["time_basis"] = basis
        result["time_precision"] = precision
        result["time_interval"] = [start, end]
        if land_filter:
            coverage['scope_count'] += 1
            assessment, unavailable_reason = _land_cover_assessment(record, q['land_cover'])
            if unavailable_reason is not None:
                coverage['unassessed_count'] += 1
                reasons = coverage['unassessed_reasons']
                reasons[unavailable_reason] = reasons.get(unavailable_reason, 0) + 1
                continue
            coverage['assessed_count'] += 1
            is_partial = assessment['pixel_coverage'] == 'partial'
            coverage['partial_pixel_count'] += int(is_partial)
            if assessment['flood_overlap_m2'] == 0:
                coverage['observed_zero_count'] += 1
                coverage['fully_observed_zero_count'] += int(not is_partial)
                continue
            coverage['matched_count'] += 1
            result['land_cover_match'] = assessment
        selected.append(result)
    selected.sort(key=(lambda r: (-r['land_cover_match']['flood_overlap_m2'], _sort_key(r))) if land_filter else _sort_key)
    response["matched_count"] = len(selected)
    response["records"] = selected[:q["limit"]]
    response["returned_count"] = len(response["records"])
    response["status"] = "ok" if selected else "no_matches"
    if land_filter:
        response['coverage'] = coverage
        response['ranking_basis'] = 'observed_flood_land_cover_overlap_m2_desc'
        limitations.extend([
            'coverage_scope_is_after_other_query_filters_before_land_cover_filter_and_limit',
            'unassessed_records_are_unknown_not_zero_overlap',
            'observed_zero_applies_only_to_joint_valid_pixels',
            'tree_cover_class_is_not_a_verified_forest_inventory',
            'overlap_with_historical_land_cover_is_not_measured_damage_or_current_land_cover',
        ])
        if coverage['scope_count'] and not coverage['assessed_count']:
            response['status'] = 'needs_data'
            response['blockers'] = sorted(coverage['unassessed_reasons'])
            limitations.append('land_cover_overlap_was_not_evaluated_zero_returned_records_is_not_zero_impact')
        elif coverage['unassessed_count'] or coverage['partial_pixel_count']:
            response['status'] = 'partial_coverage'
        response['coverage_complete'] = bool(coverage['scope_count']) and not (
            coverage['unassessed_count'] or coverage['partial_pixel_count'])
    response["excluded_missing_time"] = missing_time
    if response["excluded_missing_geometry"]:
        limitations.append("matching_scope_records_without_centroids_were_excluded")
    if missing_time:
        limitations.append("matching_scope_records_without_required_dates_were_excluded")
    if not selected and response['status'] == 'no_matches':
        limitations.append("no_catalog_matches_does_not_establish_no_flood_or_no_impact")
    return response
