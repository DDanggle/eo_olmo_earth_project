#!/usr/bin/env python3
"""Build an evidence-aware registry for all 368 official Jeju oreum records.

The official inventory is the fixed denominator. A user-provided HTML table is
parsed as a second, provenance-limited source for the 210 Jeju-si records. OSM
coordinates are resolved from an offline South Korea snapshot and are explicitly
graded as current community-map point evidence, never as an official boundary.

The output separates six states that must not be collapsed into "investigated":
inventory, approximate location, satellite screening, official causal evidence,
human review, and final decision. If fewer than 10% of records have grade A/B
causal evidence, the dashboard automatically switches to selective detection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable


EARTH_RADIUS_M = 6_371_008.8
OFFICIAL_OREUM_URL = "https://www.data.go.kr/data/15043497/fileData.do"
MOLIT_PERMIT_URL = "https://www.data.go.kr/data/15021109/fileData.do"
GEOFABRIK_URL = "https://download.geofabrik.de/asia/south-korea.html"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACCESS_STATUS = PROJECT_ROOT / "config" / "kearth_public_access.json"
CADASTRAL_URL = (
    "https://www.vworld.kr/dev/v4dv_2ddataguide2_s003.do?svcIde=cadastral"
)
EIA_AREA_URL = "https://www.data.go.kr/en/data/15142907/openapi.do"
BUILDING_HUB_URL = "https://www.data.go.kr/data/15136267/openapi.do"
GK2A_URL = "https://www.data.go.kr/data/15077314/openapi.do"
AERIAL_URL = "https://www.data.go.kr/data/15059918/fileData.do"
LANDCOVER_URL = "https://aid.mcee.go.kr/api/land.do"
EVIDENCE_THRESHOLD = 0.10
JEJU_BBOX = (125.95, 33.05, 127.10, 33.65)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_access_status(path: Path = DEFAULT_ACCESS_STATUS) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "kearth-public-access-status-v1":
        raise ValueError(f"unexpected access-status schema: {payload.get('schema')!r}")
    services = payload.get("services")
    if not isinstance(services, list):
        raise ValueError("access status requires a services list")
    by_id = {service.get("id"): service for service in services}
    required = {
        "vworld_cadastral",
        "eia_area",
        "building_hub",
        "gk2a",
        "vworld_context",
        "ngii_aerial",
        "mcee_landcover",
        "ecvam_context",
    }
    missing = sorted(required - by_id.keys())
    if missing:
        raise ValueError(f"access status is missing services: {missing}")
    snapshot = payload.get("snapshot")
    snapshot_fields = {
        "request_count",
        "http_success",
        "semantic_success",
        "semantic_no_features",
        "semantic_errors",
        "building_event_rows",
        "eia_feature_rows",
        "landcover_tile_rows",
        "gk2a_current_grid_values",
        "vworld_target_points",
        "vworld_parcel_features",
        "vworld_unique_pnu",
        "vworld_candidate_anchors",
        "vworld_oreum_anchors",
        "vworld_not_found",
        "candidate_exact_pnu_event",
        "candidate_time_aligned_exact_pnu_event",
        "candidate_pnu_conflicts",
        "candidate_official_cause_ab",
    }
    if not isinstance(snapshot, dict):
        raise ValueError("access status requires snapshot metrics")
    missing_snapshot = sorted(snapshot_fields - snapshot.keys())
    if missing_snapshot:
        raise ValueError(f"access status is missing snapshot metrics: {missing_snapshot}")
    if any(not isinstance(snapshot[field], int) or snapshot[field] < 0 for field in snapshot_fields):
        raise ValueError("snapshot metrics must be non-negative integers")
    if snapshot["http_success"] > snapshot["request_count"]:
        raise ValueError("HTTP success cannot exceed request count")
    if snapshot["vworld_parcel_features"] + snapshot["vworld_not_found"] != snapshot["vworld_target_points"]:
        raise ValueError("VWorld feature/no-feature counts must exhaust target points")
    if snapshot["candidate_time_aligned_exact_pnu_event"] > snapshot["candidate_exact_pnu_event"]:
        raise ValueError("time-aligned candidate events cannot exceed exact-PNU events")
    forbidden = {"key", "token", "secret", "password"}

    def reject_secret_fields(value: Any) -> None:
        if isinstance(value, dict):
            for field, child in value.items():
                if str(field).lower() in forbidden:
                    raise ValueError("access status must not contain secret values")
                reject_secret_fields(child)
        elif isinstance(value, list):
            for child in value:
                reject_secret_fields(child)

    reject_secret_fields(payload)
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as src:
        return list(csv.DictReader(src))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_text(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", clean_text(value)).lower()


def normalize_name(value: str) -> str:
    return normalize_text(re.sub(r"\([^)]*\)", "", value or ""))


def name_stems(value: str) -> set[str]:
    normalized = normalize_name(value)
    stems = {normalized}
    for suffix in ("오름", "악", "봉"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            stems.add(normalized[: -len(suffix)])
    return {stem for stem in stems if stem}


def address_tokens(value: str) -> list[str]:
    tokens = re.findall(r"[가-힣0-9]+(?:읍|면|동|리)", value or "")
    return list(dict.fromkeys(tokens))


def most_specific_locality(value: str) -> str | None:
    tokens = address_tokens(value)
    for suffix in ("리", "동", "읍", "면"):
        matches = [token for token in tokens if token.endswith(suffix)]
        if matches:
            return matches[-1]
    return None


def number(value: str) -> float | None:
    value = clean_text(value).replace(",", "")
    try:
        return float(value)
    except ValueError:
        return None


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.row is not None and self.cell is not None:
            self.row.append(clean_text(" ".join(self.cell)))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if any(self.row):
                self.rows.append(self.row)
            self.row = None


def parse_attachment(path: Path) -> list[dict[str, str]]:
    parser = TableParser()
    parser.feed(path.read_text(encoding="utf-8"))
    if not parser.rows:
        raise ValueError("attachment did not contain a table")
    expected = ["번호", "오름명", "행정시", "소재지", "비고", "표고", "면적"]
    header_index = next(
        (
            idx
            for idx, row in enumerate(parser.rows)
            if row[: len(expected)] == expected
        ),
        None,
    )
    if header_index is None:
        raise ValueError(f"attachment header not found; first row={parser.rows[0]}")
    rows = []
    for values in parser.rows[header_index + 1 :]:
        if len(values) < len(expected) or not values[0].isdigit():
            continue
        rows.append(dict(zip(expected, values[: len(expected)])))
    if len(rows) != 210:
        raise ValueError(f"expected 210 Jeju-si attachment rows, got {len(rows)}")
    sequence = [int(row["번호"]) for row in rows]
    if sequence != list(range(1, 211)):
        raise ValueError("attachment record numbers are not the fixed sequence 1..210")
    return rows


def stripped_address(value: str) -> str:
    return normalize_text(
        re.sub(r"^(제주특별자치도\s+)?(제주시|서귀포시)\s+", "", clean_text(value))
    )


def compare_attachment(
    official: list[dict[str, str]], attachment: list[dict[str, str]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    # The attachment number is a Jeju-si-only sequence, not the master official
    # record number. Joining by it would create 206 false conflicts after row 4.
    output: dict[str, dict[str, Any]] = {
        row["연번"]: {
            "status": "not_in_attachment_scope",
            "corroborated": False,
            "field_matches": {},
        }
        for row in official
    }
    used: set[str] = set()
    unmatched: list[dict[str, str]] = []
    for other in attachment:
        candidates = [
            row
            for row in official
            if row["연번"] not in used
            and normalize_name(row["오름명"]) == normalize_name(other["오름명"])
        ]
        scored = []
        for row in candidates:
            address_match = stripped_address(row["소재지"]) == stripped_address(
                other["소재지"]
            )
            area_match = number(row["면적"]) == number(other["면적"])
            note_match = number(row["비고"]) == number(other["비고"])
            city_match = normalize_text(row["행정시"]) == normalize_text(
                other["행정시"]
            )
            score = (
                20 * address_match + 8 * area_match + 4 * note_match + 2 * city_match
            )
            scored.append((score, address_match, area_match, row))
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        if not scored:
            unmatched.append(other)
            continue
        best = scored[0]
        tied = len(scored) > 1 and scored[1][:3] == best[:3]
        if tied or not (best[1] or best[2]):
            unmatched.append(other)
            continue
        row = best[3]
        record_no = row["연번"]
        used.add(record_no)
        altitude_a, altitude_b = number(row["표고"]), number(other["표고"])
        fields = {
            "name": normalize_name(row["오름명"]) == normalize_name(other["오름명"]),
            "city": normalize_text(row["행정시"]) == normalize_text(other["행정시"]),
            "address": stripped_address(row["소재지"])
            == stripped_address(other["소재지"]),
            "note": number(row["비고"]) == number(other["비고"]),
            "altitude_within_1m": (
                altitude_a is not None
                and altitude_b is not None
                and abs(altitude_a - altitude_b) <= 1.0
            ),
            "area": number(row["면적"]) == number(other["면적"]),
        }
        core = all(fields[key] for key in ("name", "city", "address", "area"))
        output[record_no] = {
            "status": "corroborated" if core else "conflict",
            "corroborated": core,
            "match_method": "name_plus_address_or_area",
            "attachment_record_no": other["번호"],
            "field_matches": fields,
            "attachment_values": other,
        }
    audit = {
        "attachment_rows": len(attachment),
        "matched_to_official": len(used),
        "corroborated_core_fields": sum(
            1 for item in output.values() if item["status"] == "corroborated"
        ),
        "linked_but_conflicting_core_fields": sum(
            1 for item in output.values() if item["status"] == "conflict"
        ),
        "unmatched_attachment_rows": unmatched,
        "discarded_join": "attachment 번호 is a Jeju-si sequence and must not join to official 연번",
    }
    return output, audit


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def feature_center(feature: dict[str, Any]) -> tuple[float, float] | None:
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates")
    points: list[tuple[float, float]] = []

    def visit(value: Any) -> None:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[1]), float(value[0])))
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(coords)
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def extract_osm_points(
    osmium: str, source: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    with tempfile.TemporaryDirectory(prefix="kearth-oreum-osm-") as temp_name:
        temp = Path(temp_name)
        filtered = temp / "oreum-place.osm.pbf"
        sequence = temp / "oreum-place.geojsonseq"
        subprocess.run(
            [
                osmium,
                "tags-filter",
                str(source),
                "n/natural=peak",
                "n/place",
                "w/place",
                "r/place",
                "-o",
                str(filtered),
            ],
            check=True,
        )
        subprocess.run(
            [osmium, "export", str(filtered), "-f", "geojsonseq", "-o", str(sequence)],
            check=True,
        )
        peaks, places = [], []
        with sequence.open("r", encoding="utf-8") as src:
            for line in src:
                line = line.lstrip("\x1e").strip()
                if not line:
                    continue
                feature = json.loads(line)
                props = feature.get("properties") or {}
                center = feature_center(feature)
                name = props.get("name") or props.get("name:ko")
                if not center or not name:
                    continue
                min_lon, min_lat, max_lon, max_lat = JEJU_BBOX
                if not (
                    min_lat <= center[0] <= max_lat and min_lon <= center[1] <= max_lon
                ):
                    continue
                compact = {
                    "name": str(name),
                    "lat": center[0],
                    "lon": center[1],
                    "properties": {
                        key: props[key]
                        for key in ("natural", "place", "ele")
                        if key in props
                    },
                }
                if props.get("natural") == "peak":
                    peaks.append(compact)
                elif props.get("place"):
                    places.append(compact)
    return peaks, places


def nearby_place_names(
    peak: dict[str, Any], places: list[dict[str, Any]], radius_m: float = 8_000
) -> list[str]:
    matches = []
    for place in places:
        distance = haversine_m(peak["lat"], peak["lon"], place["lat"], place["lon"])
        if distance <= radius_m:
            matches.append((distance, place["name"]))
    return [name for _, name in sorted(matches)[:16]]


def resolve_osm_locations(
    official: list[dict[str, str]],
    peaks: list[dict[str, Any]],
    places: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    for peak in peaks:
        peak["nearby_places"] = nearby_place_names(peak, places)
        peak["normalized_name"] = normalize_name(peak["name"])
        peak["stems"] = sorted(name_stems(peak["name"]))
    official_name_counts = Counter(normalize_name(row["오름명"]) for row in official)
    peak_name_counts = Counter(peak["normalized_name"] for peak in peaks)
    assignments: dict[str, dict[str, Any]] = {}
    used_peaks: set[int] = set()

    candidate_rows = []
    for row in official:
        exact = [
            idx
            for idx, peak in enumerate(peaks)
            if peak["normalized_name"] == normalize_name(row["오름명"])
        ]
        stem = [
            idx
            for idx, peak in enumerate(peaks)
            if name_stems(row["오름명"]) & set(peak["stems"])
        ]
        candidates = exact or stem
        tokens = [normalize_text(token) for token in address_tokens(row["소재지"])]
        scored = []
        for idx in candidates:
            peak = peaks[idx]
            places_norm = [normalize_text(name) for name in peak["nearby_places"]]
            locality_hits = sum(token in places_norm for token in tokens)
            exact_name = peak["normalized_name"] == normalize_name(row["오름명"])
            unique_exact = (
                exact_name
                and official_name_counts[normalize_name(row["오름명"])] == 1
                and peak_name_counts[peak["normalized_name"]] == 1
            )
            score = (
                (20 if locality_hits else 0)
                + (8 if exact_name else 3)
                + (5 if unique_exact else 0)
            )
            scored.append((score, locality_hits, exact_name, idx))
        scored.sort(reverse=True)
        candidate_rows.append((row, scored))

    candidate_rows.sort(
        key=lambda item: (
            item[1][0][0] if item[1] else -1,
            item[1][0][1] if item[1] else -1,
        ),
        reverse=True,
    )
    for row, scored in candidate_rows:
        record_no = row["연번"]
        available = [item for item in scored if item[3] not in used_peaks]
        if not available:
            assignments[record_no] = {
                "status": "unresolved",
                "grade": "U",
                "reason": "no unambiguous unused offline OSM peak name match",
            }
            continue
        best = available[0]
        tied = len(available) > 1 and available[1][0:2] == best[0:2]
        score, locality_hits, exact_name, idx = best
        unique_exact = (
            exact_name
            and official_name_counts[normalize_name(row["오름명"])] == 1
            and peak_name_counts[peaks[idx]["normalized_name"]] == 1
        )
        accepted = (locality_hits > 0 and not tied) or unique_exact
        if not accepted:
            assignments[record_no] = {
                "status": "ambiguous",
                "grade": "U",
                "reason": "name match exists but locality cannot disambiguate it",
                "candidate_count": len(available),
            }
            continue
        peak = peaks[idx]
        used_peaks.add(idx)
        assignments[record_no] = {
            "status": "resolved_offline_osm_peak",
            "grade": "C",
            "method": "exact_unique_name"
            if unique_exact
            else "name_plus_nearby_locality",
            "osm_name": peak["name"],
            "lat": round(peak["lat"], 7),
            "lon": round(peak["lon"], 7),
            "nearby_places": peak["nearby_places"],
            "warning": "current community-map peak point; not an official oreum boundary",
        }
    return assignments


def permit_context(
    official: list[dict[str, str]], permits: list[dict[str, str]]
) -> dict[str, dict[str, Any]]:
    output = {}
    for row in official:
        locality = most_specific_locality(row["소재지"])
        matches = [
            permit
            for permit in permits
            if locality and locality in (permit.get("위치명") or "")
        ]
        output[row["연번"]] = {
            "status": "same_locality_context" if matches else "no_usable_context",
            "grade": "D" if matches else "U",
            "locality": locality,
            "record_count": len(matches),
            "years": dict(
                sorted(
                    Counter(
                        (item.get("허가일자") or "unknown")[:4] for item in matches
                    ).items()
                )
            ),
            "warning": (
                "same-locality rows are not parcel matches; absence is not negative evidence "
                "because Jeju 2023/2024 rows are missing from this snapshot"
            ),
        }
    return output


def candidate_links(path: Path, review_path: Path) -> dict[str, list[dict[str, Any]]]:
    context = json.loads(path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for site in context.get("sites", []):
        judgment = review.get("reviews", {}).get(site["candidate_id"], {})
        for match in site.get("official_oreum_name_matches_5km", []):
            output[str(match["official_record_no"])].append(
                {
                    "candidate_id": site["candidate_id"],
                    "distance_m": match["distance_m"],
                    "rgb_effect": judgment.get("effect"),
                    "rgb_context_confidence": judgment.get("confidence"),
                    "status": (
                        "nearby_within_500m"
                        if match["distance_m"] <= 500
                        else "context_only_over_500m"
                    ),
                    "warning": "point proximity does not establish boundary overlap or cause",
                }
            )
    return output


def load_model_scores(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["official_record_no"]): row for row in payload.get("records", [])}


def load_rgb_reviews(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for oreum_id, review in payload.get("reviews", {}).items():
        output[oreum_id] = {
            "status": "reviewed",
            "image": f"rgb_review/candidates/{oreum_id}.png",
            **review,
        }
    return output


def load_farmmap_evidence(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "kearth-evidence-edge-collection-v1":
        raise ValueError("unexpected FarmMap evidence schema")
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in payload.get("edges", []):
        if edge.get("source_id") != "mafra_farmmap_jeju":
            raise ValueError(f"unexpected evidence source: {edge.get('source_id')}")
        oreum_id = edge.get("oreum_id")
        if not oreum_id:
            continue
        if edge.get("evidence_grade") != "C":
            raise ValueError(
                f"oreum FarmMap edges must remain C-grade, got {edge.get('evidence_grade')}"
            )
        output[oreum_id].append(edge)
    for edges in output.values():
        edges.sort(key=lambda edge: edge["source_record_id"])
    return output


def build_registry(
    official: list[dict[str, str]],
    attachment_matches: dict[str, dict[str, Any]],
    locations: dict[str, dict[str, Any]],
    permits: dict[str, dict[str, Any]],
    links: dict[str, list[dict[str, Any]]],
    model_scores: dict[str, dict[str, Any]],
    rgb_reviews: dict[str, dict[str, Any]],
    farmmap_evidence: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    records = []
    for row in official:
        record_no = row["연번"]
        oreum_id = f"JJ-OREUM-{int(record_no):03d}"
        location = locations[record_no]
        nearby = sorted(links.get(record_no, []), key=lambda item: item["distance_m"])
        score = model_scores.get(record_no)
        close_human_candidate = any(item["distance_m"] <= 500 for item in nearby)
        stable_high = bool(score and score.get("screen_class") == "high_stable")
        screen_decision = (
            "investigate" if close_human_candidate or stable_high else "abstain"
        )
        visual_review = rgb_reviews.get(oreum_id)
        farmmap_edges = farmmap_evidence.get(oreum_id, [])
        if visual_review and visual_review["persistent_change"] == "no":
            decision = "abstain"
        elif visual_review and visual_review["persistent_change"] in {
            "yes",
            "uncertain",
        }:
            decision = "investigate"
        else:
            decision = screen_decision
        abstain_reasons = []
        if location.get("grade") == "U":
            abstain_reasons.append("official geometry/coordinate unavailable")
        if score is None:
            abstain_reasons.append("OlmoEarth per-oreum screen not run")
        elif score.get("screen_class") == "high_unstable":
            abstain_reasons.append("4-period and 12-period screen disagree")
        if visual_review and visual_review["persistent_change"] == "no":
            abstain_reasons.append(
                "season-aligned RGB review did not confirm persistent change"
            )
        abstain_reasons.append(
            "no parcel/EIA boundary and time-aligned official causal evidence"
        )
        records.append(
            {
                "official_record_no": record_no,
                "oreum_id": oreum_id,
                "name": row["오름명"],
                "city": row["행정시"],
                "address": row["소재지"],
                "locality": most_specific_locality(row["소재지"]),
                "note": row["비고"],
                "altitude_m": number(row["표고"]),
                "area_m2": number(row["면적"]),
                "shape": row["형태"],
                "inventory": {
                    "status": "listed_official",
                    "grade": "A",
                    "snapshot_date": row["데이터기준일자"],
                    "scope_warning": "official attribute record; no coordinate or boundary",
                },
                "attachment_corroboration": attachment_matches[record_no],
                "location": location,
                "permit_context": permits[record_no],
                "candidate_links": nearby,
                "farmmap_state": {
                    "status": (
                        "official_farm_polygon_covers_osm_peak_point"
                        if farmmap_edges
                        else "no_point_hit_unknown"
                    ),
                    "grade": "C" if farmmap_edges else "U",
                    "edges": farmmap_edges,
                    "warning": (
                        "the joined target is a current OSM point, not an official oreum "
                        "boundary; a miss is not negative evidence"
                    ),
                },
                "model_screen": score
                or {
                    "status": "pending_v7_multi_window_gate",
                    "grade": "U",
                    "warning": "inventory coverage is not satellite-screen completion",
                },
                "official_causal_evidence": {
                    "status": "unavailable",
                    "grade": "U",
                    "required": "parcel or EIA polygon overlap plus time alignment",
                },
                "screen_decision": screen_decision,
                "visual_review": visual_review
                or {
                    "status": "pending",
                    "label": "pending",
                    "persistent_change": "pending",
                    "confidence": "pending",
                    "notes": "",
                },
                "selective_decision": decision,
                "abstain_reasons": abstain_reasons,
                "human_review": "pending",
            }
        )
    return records


def summarize(
    records: list[dict[str, Any]], attachment_audit: dict[str, Any]
) -> dict[str, Any]:
    total = len(records)

    def count(predicate: Callable[[dict[str, Any]], bool]) -> int:
        return sum(1 for row in records if predicate(row))

    official_causal = count(
        lambda row: row["official_causal_evidence"]["grade"] in {"A", "B"}
    )
    rate = official_causal / total
    return {
        "official_inventory": total,
        "inventory_coverage": count(lambda row: row["inventory"]["grade"] == "A"),
        "attachment_scope": count(
            lambda row: row["attachment_corroboration"]["status"]
            != "not_in_attachment_scope"
        ),
        "attachment_corroborated": count(
            lambda row: row["attachment_corroboration"]["corroborated"]
        ),
        "attachment_link_conflicts": attachment_audit[
            "linked_but_conflicting_core_fields"
        ],
        "attachment_unmatched": len(attachment_audit["unmatched_attachment_rows"]),
        "osm_peak_resolved": count(lambda row: row["location"]["grade"] == "C"),
        "locality_permit_context": count(
            lambda row: row["permit_context"]["grade"] == "D"
        ),
        "farmmap_point_state_c": count(
            lambda row: row["farmmap_state"]["grade"] == "C"
        ),
        "model_screened": count(lambda row: row["model_screen"].get("grade") == "M"),
        "model_high_stable": count(
            lambda row: row["model_screen"].get("screen_class") == "high_stable"
        ),
        "rgb_reviewed": count(lambda row: row["visual_review"]["status"] == "reviewed"),
        "rgb_persistent_confirmed": count(
            lambda row: row["visual_review"].get("persistent_change") == "yes"
        ),
        "rgb_uncertain": count(
            lambda row: row["visual_review"].get("persistent_change") == "uncertain"
        ),
        "rgb_rejected": count(
            lambda row: row["visual_review"].get("persistent_change") == "no"
        ),
        "human_candidate_within_500m": count(
            lambda row: any(
                item["distance_m"] <= 500 for item in row["candidate_links"]
            )
        ),
        "official_causal_evidence_ab": official_causal,
        "official_causal_evidence_rate": round(rate, 6),
        "mode_threshold": EVIDENCE_THRESHOLD,
        "research_mode": "selective_change_detection"
        if rate < EVIDENCE_THRESHOLD
        else "causal_attribution",
        "decision_counts": dict(Counter(row["selective_decision"] for row in records)),
        "claim_boundary": {
            "complete": "all 368 official inventory records are represented and evidence-statused",
            "incomplete": "all 368 have not received a valid OlmoEarth screen or causal attribution",
        },
    }


def csv_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in records:
        model = row["model_screen"]
        rows.append(
            {
                "oreum_id": row["oreum_id"],
                "official_record_no": row["official_record_no"],
                "name": row["name"],
                "city": row["city"],
                "address": row["address"],
                "altitude_m": row["altitude_m"],
                "area_m2": row["area_m2"],
                "shape": row["shape"],
                "attachment_status": row["attachment_corroboration"]["status"],
                "location_status": row["location"]["status"],
                "location_grade": row["location"]["grade"],
                "lat": row["location"].get("lat"),
                "lon": row["location"].get("lon"),
                "permit_context_count": row["permit_context"]["record_count"],
                "farmmap_state_grade": row["farmmap_state"]["grade"],
                "farmmap_class": (
                    row["farmmap_state"]["edges"][0]["attributes"]["farm_class"]
                    if row["farmmap_state"]["edges"]
                    else None
                ),
                "farmmap_pnu": (
                    row["farmmap_state"]["edges"][0].get("pnu")
                    if row["farmmap_state"]["edges"]
                    else None
                ),
                "model_status": model.get("status"),
                "model_screen_class": model.get("screen_class"),
                "rgb_review_label": row["visual_review"].get("label"),
                "rgb_persistent_change": row["visual_review"].get("persistent_change"),
                "official_causal_grade": row["official_causal_evidence"]["grade"],
                "selective_decision": row["selective_decision"],
            }
        )
    return rows


def render_dashboard(
    payload: dict[str, Any], access_status: dict[str, Any] | None = None
) -> str:
    summary = payload["summary"]
    access_status = access_status or load_access_status()
    access_dashboard = access_status["dashboard"]
    snapshot = access_status["snapshot"]
    services = {service["id"]: service for service in access_status["services"]}

    def access_text(service_id: str) -> str:
        return html.escape(str(services[service_id]["display_status"]))

    embedded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>K-Earth Program · 제주 오름 368</title>
<style>
:root{{--bg:#08110f;--panel:#101d19;--panel2:#152721;--line:#29423a;--text:#ecf5f0;--muted:#9bb0a8;--mint:#54e6b1;--amber:#f2cb67;--red:#ff7f73;--blue:#70b7ff}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0,#14352b 0,transparent 30%),var(--bg);color:var(--text);font:14px/1.48 ui-sans-serif,system-ui,sans-serif}}
header{{padding:34px max(22px,calc((100vw - 1500px)/2));border-bottom:1px solid var(--line)}}
.eyebrow{{color:var(--mint);font-weight:800;letter-spacing:.12em;text-transform:uppercase}}h1{{font-size:clamp(30px,5vw,60px);line-height:1.02;margin:9px 0 14px;max-width:950px}}header p{{max-width:900px;color:var(--muted);font-size:16px}}
.mode{{display:inline-flex;gap:9px;align-items:center;border:1px solid #765f2b;background:#2c2516;padding:9px 12px;border-radius:99px;color:var(--amber);font-weight:800}}.dot{{width:8px;height:8px;background:var(--amber);border-radius:50%}}
main{{max-width:1500px;margin:auto;padding:24px;display:grid;grid-template-columns:minmax(0,1fr);gap:22px}}main>*{{min-width:0}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;min-width:0}}.metric,.panel{{background:#101d19e8;border:1px solid var(--line);border-radius:15px;min-width:0}}.metric{{padding:15px}}.metric b{{display:block;font-size:25px;color:var(--mint)}}.metric span{{color:var(--muted)}}
.panel{{padding:20px}}h2{{margin:0 0 8px;font-size:21px}}h3{{margin:0 0 7px}}.muted{{color:var(--muted)}}.flow{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;margin-top:16px}}.flow div{{padding:12px;border:1px solid var(--line);border-radius:10px;background:var(--panel2)}}.flow b{{display:block;color:var(--amber)}}
.program-head{{display:flex;justify-content:space-between;gap:20px;align-items:end}}.program-head>*{{min-width:0}}.program-head p{{max-width:760px;margin:0}}.track-nav{{display:flex;gap:8px;max-width:100%;overflow:auto;padding:13px 0 17px}}.track-nav a{{white-space:nowrap;text-decoration:none;border:1px solid var(--line);background:#0b1613;padding:7px 10px;border-radius:99px}}.track-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;min-width:0}}.track-card{{background:linear-gradient(160deg,#182c25,#0d1915);border:1px solid var(--line);border-radius:13px;padding:15px;min-width:0;min-height:230px;overflow-wrap:anywhere;display:flex;flex-direction:column}}.track-no{{font:800 12px/1 ui-monospace,SFMono-Regular,monospace;color:var(--mint);letter-spacing:.08em}}.track-card h3{{font-size:17px;margin:10px 0 5px}}.track-card p{{color:var(--muted);margin:0 0 10px}}.track-card ul{{margin:0 0 13px;padding-left:18px}}.track-card li{{margin:4px 0}}.track-card .next{{border-top:1px solid var(--line);padding-top:10px;margin-top:auto}}.pill{{display:inline-block;width:max-content;max-width:100%;border:1px solid var(--line);border-radius:99px;padding:4px 8px;font-size:12px;font-weight:800}}.pill.owned{{color:var(--mint);border-color:#397c64;background:#102d24}}.pill.partial{{color:var(--amber);border-color:#765f2b;background:#2c2516}}.pill.missing{{color:var(--red);border-color:#713b36;background:#2b1816}}.pill.hypothesis{{color:var(--blue);border-color:#315b7d;background:#112535}}
.custody-intro{{display:grid;grid-template-columns:1.1fr 1fr;gap:18px;align-items:start}}.custody-callout{{background:#0b1613;border:1px solid var(--line);border-radius:12px;padding:14px}}.custody-callout b{{color:var(--mint)}}.asset-table{{overflow:auto;margin-top:16px;border:1px solid var(--line);border-radius:10px}}.asset-table table{{min-width:920px}}.asset-table td:first-child{{font-weight:800;color:#dcebe5}}
.controls{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:14px 0}}input,select,button,textarea{{background:#0b1613;color:var(--text);border:1px solid #3a574d;border-radius:8px;padding:9px}}input{{min-width:260px;flex:1}}button{{cursor:pointer}}button.active{{background:var(--mint);color:#062016;font-weight:800}}
.table-wrap{{overflow:auto;max-height:720px;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;min-width:1120px}}th{{position:sticky;top:0;background:#172820;text-align:left;z-index:2;color:#c6d8d1}}th,td{{padding:9px 10px;border-bottom:1px solid #21352e;vertical-align:top}}tbody tr:hover{{background:#14261f}}.grade{{font-weight:900}}.A{{color:var(--mint)}}.C{{color:var(--blue)}}.D{{color:var(--amber)}}.U{{color:var(--red)}}.investigate{{color:var(--amber);font-weight:900}}.abstain{{color:var(--muted)}}
.detail{{max-width:420px;color:var(--muted);font-size:12px}}.map{{width:100%;height:330px;background:#09130f;border:1px solid var(--line);border-radius:12px}}.map text{{fill:#a9beb6;font-size:10px}}.legend{{display:flex;gap:15px;color:var(--muted);margin-top:8px}}.legend i{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}}
.protocol{{display:grid;grid-template-columns:1.1fr 1fr;gap:18px}}.protocol ol{{margin:8px 0;padding-left:20px}}code{{color:var(--mint)}}footer{{padding:0 24px 35px;max-width:1500px;margin:auto;color:var(--muted)}}a{{color:var(--mint)}}
@media(max-width:1250px){{.track-grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}@media(max-width:1050px){{.metrics{{grid-template-columns:repeat(3,minmax(0,1fr))}}.flow{{grid-template-columns:repeat(2,minmax(0,1fr))}}.protocol,.custody-intro{{grid-template-columns:minmax(0,1fr)}}}}@media(max-width:760px){{.track-grid{{grid-template-columns:minmax(0,1fr)}}.program-head{{display:block}}}}@media(max-width:620px){{h1{{font-size:36px}}.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}main{{padding:14px}}.panel{{padding:16px}}}}
</style></head><body>
<header><div class="eyebrow">K-Earth Program Board · snapshot {html.escape(str(snapshot.get("version", "")))}</div><h1>제주 오름 368에서 한국형 Earth Intelligence로</h1>
<div class="mode"><span class="dot"></span>선택적 변화탐지 모드 · A/B급 원인 근거 {summary["official_causal_evidence_rate"] * 100:.1f}% &lt; 10%</div>
<p>전수 목록을 만든 것과 전수 변화판정을 끝낸 것은 다릅니다. 공공데이터 신청, 현재 보유본,
사업 가설, 한국형 연구, EarthRoute 확장을 다섯 갈래로 고정하고 공식 368건의 근거 누락을 함께 봅니다.</p></header>
<main>
<section class="panel" aria-labelledby="program-title"><div class="program-head"><div><div class="eyebrow">Five-track scope</div><h2 id="program-title">지금은 이 다섯 가지로만 확장합니다</h2></div><p class="muted">각 카드의 상태는 신청 가능성을 확보 완료로, 모델 confidence를 공식 원인근거로, 연구 파트너를 구매자로 바꾸지 않습니다.</p></div>
<nav class="track-nav" aria-label="프로그램 다섯 갈래"><a href="#track-1">1 · 데이터 신청</a><a href="#track-2">2 · 현재 상황</a><a href="#track-3">3 · 비즈니스</a><a href="#track-4">4 · 한국 연구</a><a href="#track-5">5 · 연구 노트</a></nav>
<div class="track-grid">
<article id="track-1" class="track-card" data-track="1"><span class="track-no">TRACK 01</span><h3>공공데이터 신청·보유</h3><span class="pill partial">{html.escape(access_dashboard["badge"])}</span><p>{html.escape(access_dashboard["summary"])}</p><ul><li><a href="{CADASTRAL_URL}">VWorld 지적</a>: {access_text("vworld_cadastral")}</li><li><a href="{BUILDING_HUB_URL}">건축HUB</a>: {access_text("building_hub")}</li><li><a href="{GK2A_URL}">GK2A</a>: {access_text("gk2a")}</li><li><a href="{EIA_AREA_URL}">EIA WFS</a>: {access_text("eia_area")}</li><li><a href="{LANDCOVER_URL}">토지피복 WMS</a>: {access_text("mcee_landcover")}</li></ul><div class="next"><b>실측:</b> <a href="api_snapshot/dashboard.html">API 결합 대시보드 열기</a><br><b>다음:</b> {html.escape(access_dashboard["next"])}</div></article>
<article id="track-2" class="track-card" data-track="2"><span class="track-no">TRACK 02</span><h3>현재 데이터·실험</h3><span class="pill partial">API snapshot {html.escape(str(snapshot.get("version", "")))}</span><p>{snapshot["request_count"]}개 request와 redacted raw/hash를 한 snapshot으로 묶었습니다. VWorld는 {snapshot["vworld_parcel_features"]}/{snapshot["vworld_target_points"]} 대표 필지를 반환했습니다.</p><ul><li>공식 목록 {summary["inventory_coverage"]}/368</li><li>OlmoEarth screen {summary["model_screened"]}/368</li><li>BuildingHUB {snapshot["building_event_rows"]:,}행 · EIA {snapshot["eia_feature_rows"]} polygon</li><li>토지피복 {snapshot["landcover_tile_rows"]}장 · 최신 GK2A {snapshot["gk2a_current_grid_values"]:,}값</li><li>원인 A/B {summary["official_causal_evidence_ab"]}/368</li></ul><div class="next"><b>현재 결론:</b> 후보 14/14 보류 · exact PNU 사건 {snapshot["candidate_exact_pnu_event"]}건도 관측구간 정렬은 {snapshot["candidate_time_aligned_exact_pnu_event"]}건 · 필지 source 충돌 {snapshot["candidate_pnu_conflicts"]}건</div></article>
<article id="track-3" class="track-card" data-track="3"><span class="track-no">TRACK 03</span><h3>비즈니스 가능성</h3><span class="pill hypothesis">가설 · 유료 0건</span><p>공공데이터나 transfer model이 아니라 반복 결정의 감사 결과를 팝니다.</p><ul><li>Post-EIA Evidence Pack</li><li>GeoFM Release Audit</li><li>Local Adaptation Sprint</li></ul><div class="next"><b>Gate:</b> 검수시간 30%↓ · 오단정 0 · 같은 고객의 두 번째 유료 갱신</div></article>
<article id="track-4" class="track-card" data-track="4"><span class="track-no">TRACK 04</span><h3>한국형 연구</h3><span class="pill owned">현재 플래그십</span><p>지도를 많이 붙이는 대신 불완전 행정기록 아래 말하기와 보류를 측정합니다.</p><ul><li>time-aligned evidence coverage</li><li>risk–coverage·침묵 편향</li><li>확률표본 + PPI</li></ul><div class="next"><b>질문:</b> 어떤 자료 누락이 어느 지역의 판정을 침묵시키는가?</div></article>
<article id="track-5" class="track-card" data-track="5"><span class="track-no">TRACK 05</span><h3>8월 EarthRoute 노트</h3><span class="pill missing">후속 · oracle 없음</span><p>K-Earth와 FoldRefresh가 작동한 뒤 다음 관측·모델·근거·검수를 고릅니다.</p><ul><li><code>reuse</code></li><li><code>cheap_refresh</code></li><li><code>escalate</code></li></ul><div class="next"><b>Gate:</b> oracle 실비 30–40%↓가 없으면 router 확장 중단</div></article>
</div></section>
<section class="panel" aria-labelledby="custody-title"><div class="custody-intro"><div><div class="eyebrow">Data custody</div><h2 id="custody-title">“내가 가진 데이터”의 현재 경계</h2><p class="muted">공공 원본은 누구나 받을 수 있습니다. 차별점은 같은 장소의 여러 snapshot, 누락 label, 사람 판정, 모델 릴리스 결과를 재현 가능하게 잇는 history입니다.</p></div><div class="custody-callout"><b>보유의 최소 계약</b><br>원본/request · snapshot/valid time · SHA/schema/CRS/license · PNU/polygon join · no-match/보류 history</div></div>
<div class="asset-table"><table><thead><tr><th>자산층</th><th>현재 보유</th><th>상태</th><th>다음 확보</th><th>가치</th></tr></thead><tbody>
<tr><td>공식 raw snapshot</td><td>FarmMap ZIP·산지이용 CSV·오름/허가 snapshot · 약 112 MB</td><td><span class="pill partial">부분 보유</span></td><td>분기/연간 동일 source 재수집</td><td>행정·상태 변화 history</td></tr>
<tr><td>PNU geometry spine</td><td>VWorld 대표점 {snapshot["vworld_target_points"]}개 중 필지 {snapshot["vworld_parcel_features"]}개 · unique PNU {snapshot["vworld_unique_pnu"]}개</td><td><span class="pill owned">bounded snapshot 보유</span></td><td>대표점 필지와 실제 변화 footprint 관계 검수 · 오름 공식경계 별도 확보</td><td>서로 다른 사건표를 잇는 key</td></tr>
<tr><td>위성 시계열</td><td>2023–2026 · 54윈도우 · 12기간 · 5,184 manifest행</td><td><span class="pill partial">provenance 보유</span></td><td>scene ID·SCL·pixel hash와 raw/object-store policy</td><td>세계 변화와 입력 변화를 분리</td></tr>
<tr><td>행정사건 시계열</td><td>개발행위 240행 + BuildingHUB {snapshot["building_event_rows"]:,}행 + EIA {snapshot["eia_feature_rows"]} polygon</td><td><span class="pill partial">PNU 결합 완료 · 시간정렬 0건</span></td><td>정기 재수집 · NGII 전후 항공으로 변화시점 독립 검수</td><td>원인 후보의 공간·시간 검증</td></tr>
<tr><td>독립 고해상도 관측</td><td>FarmMap 1 snapshot + 토지피복 42 tile; 전후 항공 없음</td><td><span class="pill partial">상태지도 보유</span></td><td><a href="{AERIAL_URL}">NGII 항공</a> 사전고정 후보 수동 신청</td><td>Sentinel/model과 독립된 변화 확인</td></tr>
<tr><td>사람·파트너 label</td><td>RGB 9건; 확률표본 아님</td><td><span class="pill partial">초기</span></td><td>사전 층화 100후보·blind review·불일치</td><td>risk·PPI·검수시간의 유일한 정답축</td></tr>
<tr><td>모델 릴리스 pair</td><td>Jeju v1 중심, v1.2 paired audit 미완료</td><td><span class="pill missing">실험 필요</span></td><td>동일 input hash의 v1/v1.2 결과</td><td>release continuity와 FoldRefresh</td></tr>
</tbody></table></div></section>
<section class="metrics">
<div class="metric"><b>{summary["inventory_coverage"]}/368</b><span>공식 목록 상태화</span></div>
<div class="metric"><b>{summary["attachment_corroborated"]}/210</b><span>첨부 표 대조</span></div>
<div class="metric"><b>{summary["osm_peak_resolved"]}/368</b><span>OSM peak 위치</span></div>
<div class="metric"><b>{summary["farmmap_point_state_c"]}/368</b><span>팜맵 C급 point 상태</span></div>
<div class="metric"><b>{summary["model_screened"]}/368</b><span>OlmoEarth 점별 screen</span></div>
<div class="metric"><b>{summary["official_causal_evidence_ab"]}/368</b><span>A/B급 원인 근거</span></div>
<div class="metric"><b>{summary["decision_counts"].get("investigate", 0)}</b><span>RGB 후 조사 우선</span></div>
</section>
<section class="panel"><h2>판정 사슬</h2><p class="muted">앞 단계가 비어 있으면 뒤 단계는 자동으로 보류합니다. OSM과 같은 리의 허가는 원인이 아니라 탐색 문맥입니다.</p>
<div class="flow"><div><b>1 · 목록</b>공식 속성 A</div><div><b>2 · 위치</b>OSM point C</div><div><b>3 · 위성</b>4/12기간 screen M</div><div><b>4 · 행정근거</b>필지·EIA A/B</div><div><b>5 · 사람 검수</b>독립 판정</div><div><b>6 · 결정</b>판정/조사/보류</div></div></section>
<section class="panel"><h2>전수 공간 레지스트리</h2><p class="muted">점은 공식 경계가 아니라 offline OSM에서 이름·인근 지명으로 보수적으로 연결한 현재 peak입니다.</p><svg id="map" class="map" viewBox="0 0 1000 330" aria-label="resolved oreum points"></svg><div class="legend"><span><i style="background:#54e6b1"></i>조사 우선</span><span><i style="background:#70b7ff"></i>보류·위치 있음</span><span><i style="background:#ff7f73"></i>위치 미해결</span></div></section>
<section class="panel"><h2>368개 증거 레지스트리</h2><p class="muted">모델 안정 후보 {summary["model_high_stable"]}개를 포함한 9개 RGB 검수에서 지속 변화 확정 {summary["rgb_persistent_confirmed"]}개, 오염/기각 {summary["rgb_rejected"]}개, 불확실 {summary["rgb_uncertain"]}개였습니다. <a href="rgb_review/dashboard.html">9개 RGB 검수 화면 열기</a></p>
<div class="controls"><input id="search" placeholder="오름명·주소·ID 검색"><button class="active" data-filter="all">전체</button><button data-filter="investigate">조사 우선</button><button data-filter="abstain">보류</button><button data-filter="unresolved">위치 미해결</button><button id="export">사람 판정 내보내기</button><span id="count" class="muted"></span></div>
<div class="table-wrap"><table><thead><tr><th>ID</th><th>오름</th><th>공식 목록</th><th>위치</th><th>첨부 대조</th><th>허가 문맥</th><th>팜맵 상태</th><th>OlmoEarth</th><th>선택 판정</th><th>직접 검수</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="panel protocol"><div><h2>연구 프로토콜</h2><ol><li>368개 공식 목록을 사전에 고정한다.</li><li>4기간·12기간 점수가 모두 높은 경우만 모델상 안정 후보로 둔다.</li><li>필지·환경영향평가 경계와 시점이 없으면 원인을 말하지 않는다.</li><li>층화 확률표본을 사람이 두 번 판정하고 불일치를 조정한다.</li><li>Top-k는 조사 우선순위, 전체 비율은 PPI와 신뢰구간으로 분리 보고한다.</li></ol></div>
<div><h2>현재 중단 게이트</h2><p><b class="U">원인 규명 금지:</b> A/B급 공식 원인 근거가 10% 미만입니다.</p><p><b class="C">허용되는 주장:</b> 368건 전수 레지스트리, 위치·모델·행정자료 누락률, 조사 우선순위와 보류 이유.</p><p><b class="U">금지되는 주장:</b> 오름 훼손 전수 확인, 무허가 개발, 시설과 변화의 인과효과.</p></div></section>
</main><footer>공식 오름: <a href="{OFFICIAL_OREUM_URL}">제주특별자치도 오름현황</a> · 개발행위: <a href="{MOLIT_PERMIT_URL}">국토교통부 토지이음</a> · 위치 문맥: <a href="{GEOFABRIK_URL}">© OpenStreetMap contributors / Geofabrik</a>. 생성 시각 {html.escape(payload["generated_at"])}</footer>
<script>
const data={embedded};const key='kearth-oreum-human-review-v1';let overrides=JSON.parse(localStorage.getItem(key)||'{{}}');let filter='all';
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function modelText(r){{const m=r.model_screen;const v=r.visual_review;const visual=v.status==='reviewed'?`<div class="detail"><a href="${{v.image}}">RGB: ${{esc(v.label)}}</a> · ${{esc(v.persistent_change)}}</div>`:'';if(m.grade!=='M')return '<span class="grade U">U</span> 미실행'+visual;return `<span class="grade C">M</span> ${{esc(m.screen_class)}}<div class="detail">4기 ${{m.percentile_4??'-'}} · 12기 ${{m.percentile_12??'-'}}</div>${{visual}}`}}
function farmText(r){{const f=r.farmmap_state;if(f.grade!=='C')return '<span class="grade U">U</span> point hit 없음<div class="detail">농경지 전용 지도이므로 음성 근거 아님</div>';const e=f.edges[0],a=e.attributes;return `<span class="grade C">C</span> ${{esc(a.farm_class)}}<div class="detail">PNU ${{esc(e.pnu)}} · 항공 ${{esc(a.flight_date||'미상')}} · OSM point 기준</div>`}}
function rows(){{const q=document.getElementById('search').value.toLowerCase();const list=data.records.filter(r=>{{const text=`${{r.oreum_id}} ${{r.name}} ${{r.address}}`.toLowerCase();const f=filter==='all'||(filter==='unresolved'?r.location.grade==='U':r.selective_decision===filter);return text.includes(q)&&f}});document.getElementById('count').textContent=`${{list.length}} / 368`;
document.getElementById('rows').innerHTML=list.map(r=>{{const p=r.permit_context;const loc=r.location;const a=r.attachment_corroboration;const current=overrides[r.oreum_id]||{{decision:r.selective_decision,notes:r.visual_review.notes||''}};return `<tr><td>${{r.oreum_id}}</td><td><b>${{esc(r.name)}}</b><div class="detail">${{esc(r.address)}} · ${{r.altitude_m??'-'}}m</div></td><td><span class="grade A">A</span> 등재</td><td><span class="grade ${{loc.grade}}">${{loc.grade}}</span> ${{esc(loc.status)}}<div class="detail">${{loc.lat?`${{loc.lat}}, ${{loc.lon}}`:esc(loc.reason||'')}}</div></td><td><span class="grade ${{a.corroborated?'D':'U'}}">${{a.corroborated?'D':'U'}}</span> ${{esc(a.status)}}</td><td><span class="grade ${{p.grade}}">${{p.grade}}</span> ${{p.record_count}}건<div class="detail">같은 ${{esc(p.locality||'지역')}} · 필지 일치 아님</div></td><td>${{farmText(r)}}</td><td>${{modelText(r)}}</td><td class="${{r.selective_decision}}">${{r.selective_decision==='investigate'?'조사 우선':'보류'}}<div class="detail">${{esc(r.abstain_reasons.join(' · '))}}</div></td><td><select data-id="${{r.oreum_id}}"><option value="abstain" ${{current.decision==='abstain'?'selected':''}}>보류</option><option value="investigate" ${{current.decision==='investigate'?'selected':''}}>조사 우선</option><option value="reviewed_no_change" ${{current.decision==='reviewed_no_change'?'selected':''}}>검수: 변화 아님</option><option value="decision_ready" ${{current.decision==='decision_ready'?'selected':''}}>판정 가능</option></select><textarea data-note="${{r.oreum_id}}" placeholder="검수 메모">${{esc(current.notes)}}</textarea></td></tr>`}}).join('');document.querySelectorAll('select[data-id]').forEach(el=>el.onchange=save);document.querySelectorAll('textarea[data-note]').forEach(el=>el.oninput=save)}}
function save(){{document.querySelectorAll('select[data-id]').forEach(el=>{{const id=el.dataset.id;overrides[id]=overrides[id]||{{}};overrides[id].decision=el.value}});document.querySelectorAll('textarea[data-note]').forEach(el=>{{const id=el.dataset.note;overrides[id]=overrides[id]||{{}};overrides[id].notes=el.value}});localStorage.setItem(key,JSON.stringify(overrides))}}
document.getElementById('search').oninput=rows;document.querySelectorAll('button[data-filter]').forEach(btn=>btn.onclick=()=>{{document.querySelectorAll('button[data-filter]').forEach(x=>x.classList.remove('active'));btn.classList.add('active');filter=btn.dataset.filter;rows()}});
document.getElementById('export').onclick=()=>{{save();const blob=new Blob([JSON.stringify({{schema:'kearth-oreum-human-review-v1',registry_sha256:data.registry_sha256,reviews:overrides}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='kearth_oreum_human_review.json';a.click();URL.revokeObjectURL(a.href)}};
function drawMap(){{const svg=document.getElementById('map');const pts=data.records.filter(r=>r.location.lat);const minLon=126.12,maxLon=126.98,minLat=33.18,maxLat=33.58;const x=lon=>45+(lon-minLon)/(maxLon-minLon)*910;const y=lat=>300-(lat-minLat)/(maxLat-minLat)*270;svg.innerHTML='<path d="M45 300H955M45 30V300" stroke="#29423a" fill="none"/><text x="48" y="22">offline OSM peak point distribution · not official boundaries</text>'+pts.map(r=>`<circle cx="${{x(r.location.lon).toFixed(1)}}" cy="${{y(r.location.lat).toFixed(1)}}" r="${{r.selective_decision==='investigate'?5:2.5}}" fill="${{r.selective_decision==='investigate'?'#54e6b1':'#70b7ff'}}"><title>${{esc(r.name)}} · ${{r.selective_decision}}</title></circle>`).join('')}}drawMap();rows();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-csv", type=Path, required=True)
    parser.add_argument("--attachment-html", type=Path, required=True)
    parser.add_argument("--permit-csv", type=Path, required=True)
    parser.add_argument("--candidate-context", type=Path, required=True)
    parser.add_argument("--candidate-review", type=Path, required=True)
    parser.add_argument("--osm-pbf", type=Path, required=True)
    parser.add_argument("--model-scores", type=Path)
    parser.add_argument("--rgb-review", type=Path)
    parser.add_argument("--farmmap-evidence", type=Path)
    parser.add_argument("--farmmap-manifest", type=Path)
    parser.add_argument("--access-status", type=Path, default=DEFAULT_ACCESS_STATUS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--osmium", default="osmium")
    args = parser.parse_args()
    if bool(args.farmmap_evidence) != bool(args.farmmap_manifest):
        parser.error(
            "--farmmap-evidence and --farmmap-manifest must be supplied together"
        )

    official = read_csv(args.official_csv)
    if len(official) != 368:
        raise ValueError(
            f"expected fixed official denominator 368, got {len(official)}"
        )
    attachment = parse_attachment(args.attachment_html)
    attachment_matches, attachment_audit = compare_attachment(official, attachment)
    peaks, places = extract_osm_points(args.osmium, args.osm_pbf)
    locations = resolve_osm_locations(official, peaks, places)
    permits = permit_context(official, read_csv(args.permit_csv))
    links = candidate_links(args.candidate_context, args.candidate_review)
    scores = load_model_scores(args.model_scores)
    rgb_reviews = load_rgb_reviews(args.rgb_review)
    farmmap_evidence = load_farmmap_evidence(args.farmmap_evidence)
    records = build_registry(
        official,
        attachment_matches,
        locations,
        permits,
        links,
        scores,
        rgb_reviews,
        farmmap_evidence,
    )
    summary = summarize(records, attachment_audit)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    attachment_out = args.out_dir / "attachment_jeju_city_210.csv"
    registry_csv = args.out_dir / "oreum_registry_368.csv"
    registry_json = args.out_dir / "oreum_evidence_registry.json"
    coverage_json = args.out_dir / "evidence_coverage.json"
    dashboard = args.out_dir / "dashboard.html"
    write_csv(attachment_out, attachment)
    write_csv(registry_csv, csv_rows(records))
    provenance = {
        "schema": (
            "kearth-oreum-evidence-registry-v2"
            if args.farmmap_manifest
            else "kearth-oreum-evidence-registry-v1"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "official_inventory": {
                "provider": "제주특별자치도",
                "catalog_url": OFFICIAL_OREUM_URL,
                "snapshot_date": "2024-03-31",
                "sha256": sha256(args.official_csv),
                "rows": len(official),
                "spatial_limit": "attributes and address only; no coordinate or boundary",
            },
            "user_attachment": {
                "provider": "user-provided pasted HTML; upstream provenance not established",
                "sha256": sha256(args.attachment_html),
                "rows": len(attachment),
                "scope": "Jeju-si records 1..210",
                "record_linkage_audit": attachment_audit,
            },
            "offline_osm": {
                "provider": "OpenStreetMap contributors via Geofabrik",
                "catalog_url": GEOFABRIK_URL,
                "sha256": sha256(args.osm_pbf),
                "named_peaks_considered": len(peaks),
                "named_places_considered": len(places),
            },
            "development_permits": {
                "provider": "국토교통부",
                "catalog_url": MOLIT_PERMIT_URL,
                "sha256": sha256(args.permit_csv),
                "use": "same-locality context only; not parcel evidence",
            },
            "model_scores": {
                "status": "loaded"
                if args.model_scores and args.model_scores.exists()
                else "pending",
                "sha256": sha256(args.model_scores)
                if args.model_scores and args.model_scores.exists()
                else None,
            },
            "rgb_review": {
                "status": "loaded"
                if args.rgb_review and args.rgb_review.exists()
                else "pending",
                "sha256": sha256(args.rgb_review)
                if args.rgb_review and args.rgb_review.exists()
                else None,
            },
        },
        "evidence_grades": {
            "A": "official record or time-aligned official polygon overlap",
            "B": "authoritative derived spatial evidence with verified time",
            "C": "current community-map point/name evidence",
            "D": "same-locality or distance-only context",
            "M": "model screening evidence; never causal by itself",
            "U": "unavailable, unresolved, or not yet run",
        },
        "summary": summary,
        "records": records,
    }
    if args.farmmap_manifest:
        farmmap_manifest = json.loads(args.farmmap_manifest.read_text(encoding="utf-8"))
        if farmmap_manifest.get("source_id") != "mafra_farmmap_jeju":
            raise ValueError("unexpected FarmMap manifest source")
        provenance["sources"]["farmmap"] = {
            "provider": farmmap_manifest["provider"],
            "catalog_url": farmmap_manifest["catalog_url"],
            "snapshot_date": farmmap_manifest["snapshot_date"],
            "sha256": farmmap_manifest["raw_sha256"],
            "rows": farmmap_manifest["row_count"],
            "use": (
                "dated state at current OSM peak point; C-grade because the point is "
                "not an official oreum boundary"
            ),
        }
    registry_json.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    coverage_json.write_text(
        json.dumps(
            {
                "schema": "kearth-oreum-evidence-coverage-v1",
                "generated_at": provenance["generated_at"],
                "summary": summary,
                "source_registry": registry_json.name,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    provenance["registry_sha256"] = sha256(registry_json)
    dashboard.write_text(
        render_dashboard(provenance, load_access_status(args.access_status)),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": summary,
                "osm_named_peaks": len(peaks),
                "osm_named_places": len(places),
                "outputs": [
                    str(attachment_out),
                    str(registry_csv),
                    str(registry_json),
                    str(coverage_json),
                    str(dashboard),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
