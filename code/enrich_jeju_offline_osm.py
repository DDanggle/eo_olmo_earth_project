#!/usr/bin/env python3
"""Join Jeju change candidates to an offline South Korea OSM snapshot.

The country extract is downloaded without disclosing candidate coordinates. All
candidate-level spatial operations happen locally. Official Jeju oreum names and
MOLIT permit rows are then joined conservatively: oreum identities require a name
match, while permits are reported only as same-locality context unless a parcel PNU
can later be resolved through an authorized cadastral layer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EARTH_RADIUS_M = 6_371_008.8
OSM_COPYRIGHT = "https://www.openstreetmap.org/copyright"
GEOFABRIK_SOURCE = "https://download.geofabrik.de/asia/south-korea.html"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def all_points(geometry: dict[str, Any] | None) -> list[tuple[float, float]]:
    if not geometry:
        return []
    coordinates = geometry.get("coordinates")
    points: list[tuple[float, float]] = []

    def visit(value: Any) -> None:
        if (
            isinstance(value, list)
            and len(value) >= 2
            and isinstance(value[0], (int, float))
            and isinstance(value[1], (int, float))
        ):
            points.append((float(value[0]), float(value[1])))
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(coordinates)
    if geometry.get("type") == "GeometryCollection":
        for child in geometry.get("geometries", []):
            points.extend(all_points(child))
    return points


def feature_center(feature: dict[str, Any]) -> tuple[float, float] | None:
    points = all_points(feature.get("geometry"))
    if not points:
        return None
    lon = sum(point[0] for point in points) / len(points)
    lat = sum(point[1] for point in points) / len(points)
    return lat, lon


def point_in_ring(lon: float, lat: float, ring: list[list[float]]) -> bool:
    inside = False
    if len(ring) < 3:
        return False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i][:2]
        xj, yj = ring[j][:2]
        crosses = (yi > lat) != (yj > lat)
        if crosses:
            x_at_lat = (xj - xi) * (lat - yi) / ((yj - yi) or 1e-15) + xi
            if lon < x_at_lat:
                inside = not inside
        j = i
    return inside


def point_in_polygon(lon: float, lat: float, polygon: list[list[list[float]]]) -> bool:
    if not polygon or not point_in_ring(lon, lat, polygon[0]):
        return False
    return not any(point_in_ring(lon, lat, hole) for hole in polygon[1:])


def geometry_contains(geometry: dict[str, Any] | None, lon: float, lat: float) -> bool:
    if not geometry:
        return False
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if kind == "Polygon":
        return point_in_polygon(lon, lat, coords)
    if kind == "MultiPolygon":
        return any(point_in_polygon(lon, lat, polygon) for polygon in coords)
    if kind == "GeometryCollection":
        return any(
            geometry_contains(child, lon, lat)
            for child in geometry.get("geometries", [])
        )
    return False


def normalize_name(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value or "")
    return re.sub(r"[^0-9A-Za-z가-힣]", "", value).lower()


def name_aliases(value: str) -> set[str]:
    normalized = normalize_name(value)
    aliases = {normalized}
    for suffix in ("오름", "악", "봉"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            aliases.add(normalized[: -len(suffix)])
    return {alias for alias in aliases if alias}


def tags(feature: dict[str, Any]) -> dict[str, str]:
    props = feature.get("properties") or {}
    return {str(key): str(value) for key, value in props.items() if value is not None}


def run_osmium(
    osmium: str, source: Path, workdir: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    filtered = workdir / "jeju_context.osm.pbf"
    sequence = workdir / "jeju_context.geojsonseq"
    expressions = [
        "nwr/natural",
        "nwr/landuse",
        "nwr/leisure",
        "nwr/man_made",
        "nwr/power",
        "nwr/construction",
        "nwr/place",
        "nwr/boundary=administrative",
        "w/highway",
        "nwr/building",
    ]
    subprocess.run(
        [osmium, "tags-filter", str(source), *expressions, "-o", str(filtered)],
        check=True,
    )
    subprocess.run(
        [osmium, "export", str(filtered), "-f", "geojsonseq", "-o", str(sequence)],
        check=True,
    )
    features = []
    with sequence.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.lstrip("\x1e").strip()
            if not line:
                continue
            feature = json.loads(line)
            if feature.get("geometry"):
                features.append(feature)
    fileinfo = json.loads(
        subprocess.run(
            [osmium, "fileinfo", "--json", str(source)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return features, fileinfo


def geometry_distance_m(feature: dict[str, Any], lat: float, lon: float) -> float:
    points = all_points(feature.get("geometry"))
    if not points:
        return float("inf")
    return min(
        haversine_m(lat, lon, point_lat, point_lon) for point_lon, point_lat in points
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as src:
        return list(csv.DictReader(src))


def compact_feature(feature: dict[str, Any], lat: float, lon: float) -> dict[str, Any]:
    feature_tags = tags(feature)
    center = feature_center(feature)
    return {
        "osm_id": feature.get("id") or feature_tags.get("@id"),
        "name": feature_tags.get("name") or feature_tags.get("name:ko"),
        "distance_m": round(geometry_distance_m(feature, lat, lon)),
        "center": {"lat": center[0], "lon": center[1]} if center else None,
        "tags": {
            key: feature_tags[key]
            for key in (
                "admin_level",
                "boundary",
                "natural",
                "place",
                "landuse",
                "leisure",
                "man_made",
                "power",
                "construction",
                "highway",
                "building",
                "surface",
                "generator:source",
                "generator:method",
                "plant:source",
                "operator",
                "sport",
            )
            if key in feature_tags
        },
    }


def unique_compact(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse relation/way duplicates while retaining distinct nearby objects."""
    output = []
    seen = set()
    for item in items:
        signature = (
            item.get("name"),
            tuple(sorted(item.get("tags", {}).items())),
            round(float(item.get("distance_m", 0)) / 10),
        )
        if signature in seen:
            continue
        seen.add(signature)
        output.append(item)
    return output


def candidate_context(
    candidate: dict[str, Any],
    features: list[dict[str, Any]],
    official_oreum: list[dict[str, str]],
    permits: list[dict[str, str]],
) -> dict[str, Any]:
    lat, lon = float(candidate["lat"]), float(candidate["lon"])
    nearby: list[tuple[float, dict[str, Any], dict[str, str]]] = []
    containing: list[tuple[dict[str, Any], dict[str, str]]] = []
    for feature in features:
        feature_tags = tags(feature)
        distance = geometry_distance_m(feature, lat, lon)
        if distance <= 5_000:
            nearby.append((distance, feature, feature_tags))
        if geometry_contains(feature.get("geometry"), lon, lat):
            containing.append((feature, feature_tags))

    peaks = [
        compact_feature(feature, lat, lon)
        for distance, feature, feature_tags in nearby
        if feature_tags.get("natural") == "peak"
        and (feature_tags.get("name") or feature_tags.get("name:ko"))
    ]
    peaks.sort(key=lambda item: item["distance_m"])
    places = [
        compact_feature(feature, lat, lon)
        for distance, feature, feature_tags in nearby
        if feature_tags.get("place")
        and (feature_tags.get("name") or feature_tags.get("name:ko"))
    ]
    places.sort(key=lambda item: item["distance_m"])

    official_index: dict[str, list[dict[str, str]]] = {}
    for row in official_oreum:
        for alias in sorted(name_aliases(row["오름명"])):
            official_index.setdefault(alias, []).append(row)
    official_best: dict[str, dict[str, Any]] = {}
    for peak in peaks:
        matched_rows: list[dict[str, str]] = []
        for alias in sorted(name_aliases(peak["name"] or "")):
            matched_rows.extend(official_index.get(alias, []))
        for row in matched_rows:
            key = row["연번"]
            item = {
                "official_record_no": key,
                "distance_m": peak["distance_m"],
                "osm_peak_name": peak["name"],
                "official_name": row["오름명"],
                "official_address": row["소재지"],
                "altitude_m": row["표고"],
                "area_m2": row["면적"],
                "shape": row["형태"],
                "identity_evidence": "OSM peak name matched to 제주특별자치도 official inventory",
            }
            if (
                key not in official_best
                or item["distance_m"] < official_best[key]["distance_m"]
            ):
                official_best[key] = item
    official_matches = list(official_best.values())
    official_matches.sort(
        key=lambda item: (item["distance_m"], item["official_record_no"])
    )

    admins = []
    landcover = []
    for feature, feature_tags in containing:
        if feature_tags.get("boundary") == "administrative" and (
            feature_tags.get("name") or feature_tags.get("name:ko")
        ):
            admins.append(compact_feature(feature, lat, lon))
        if any(key in feature_tags for key in ("landuse", "natural", "leisure")):
            landcover.append(compact_feature(feature, lat, lon))
    admins.sort(
        key=lambda item: int(item["tags"].get("admin_level", "0") or 0), reverse=True
    )

    building_count = sum(
        1
        for distance, _, feature_tags in nearby
        if distance <= 500 and "building" in feature_tags
    )
    road_features = unique_compact(
        [
            compact_feature(feature, lat, lon)
            for distance, feature, feature_tags in nearby
            if distance <= 500 and "highway" in feature_tags
        ]
    )
    noteworthy = unique_compact(
        [
            compact_feature(feature, lat, lon)
            for distance, feature, feature_tags in nearby
            if distance <= 1_000
            and any(
                feature_tags.get(key)
                in {
                    "construction",
                    "quarry",
                    "industrial",
                    "commercial",
                    "retail",
                    "landfill",
                    "solar",
                    "wind_turbine",
                    "works",
                    "plant",
                    "reservoir",
                    "golf_course",
                }
                for key in (
                    "landuse",
                    "natural",
                    "leisure",
                    "man_made",
                    "power",
                    "construction",
                )
            )
        ]
    )
    noteworthy.sort(key=lambda item: item["distance_m"])

    locality_names = []
    for item in admins:
        name = item.get("name")
        if not name or name in {"대한민국", "제주특별자치도", "제주시", "서귀포시"}:
            continue
        if name not in locality_names:
            locality_names.append(name)
    permit_token = locality_names[0] if locality_names else None
    permit_matches = [
        row
        for row in permits
        if permit_token and permit_token in (row.get("위치명") or "")
    ]
    permit_years = Counter(
        (row.get("허가일자") or "unknown")[:4] for row in permit_matches
    )
    permit_actions = Counter(
        (row.get("대표개발행위명") or "unknown").strip() for row in permit_matches
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "lat": lat,
        "lon": lon,
        "rgb_image": candidate["image"],
        "containing_administrative_areas": admins,
        "containing_current_landcover": landcover,
        "nearby_named_peaks_5km": peaks[:12],
        "official_oreum_name_matches_5km": official_matches[:12],
        "nearby_named_places_5km": places[:12],
        "current_building_count_500m": building_count,
        "current_road_features_500m": len(road_features),
        "nearest_roads": sorted(road_features, key=lambda item: item["distance_m"])[:8],
        "noteworthy_mapped_features_1km": noteworthy[:20],
        "permit_context": {
            "match_scope": "same named locality; not parcel-level",
            "locality_token": permit_token,
            "record_count": len(permit_matches),
            "by_year": dict(sorted(permit_years.items())),
            "top_actions": dict(permit_actions.most_common(10)),
            "sample_records": [
                {
                    key: row.get(key)
                    for key in (
                        "PNU",
                        "위치명",
                        "지목명",
                        "면적(㎡)",
                        "용도지역명",
                        "용도지구명",
                        "대표개발행위명",
                        "신청일자",
                        "허가일자",
                        "개발행위목적",
                    )
                }
                for row in permit_matches[:20]
            ],
            "negative_evidence_warning": (
                "The 2026-08-19 national snapshot has no 2023/2024 Jeju permit rows; "
                "no match cannot be interpreted as no permit."
            ),
        },
        "initial_after_context_judgment": {
            "effect": "context_only",
            "confidence": "low",
            "notes": "Current OSM context and locality-level permits cannot date or cause the RGB change.",
        },
    }


def project_points(
    value: Any, lat0: float, lon0: float, size: int, radius_m: float
) -> Any:
    if (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        x_m = (
            math.radians(float(value[0]) - lon0)
            * EARTH_RADIUS_M
            * math.cos(math.radians(lat0))
        )
        y_m = math.radians(float(value[1]) - lat0) * EARTH_RADIUS_M
        return [
            size / 2 + x_m / radius_m * size / 2,
            size / 2 - y_m / radius_m * size / 2,
        ]
    if isinstance(value, list):
        return [project_points(child, lat0, lon0, size, radius_m) for child in value]
    return value


def svg_path(coords: Any, kind: str) -> str:
    paths: list[str] = []

    def line(points: list[list[float]], close: bool) -> None:
        if not points:
            return
        command = "M" + " L".join(f"{point[0]:.1f},{point[1]:.1f}" for point in points)
        paths.append(command + (" Z" if close else ""))

    if kind == "LineString":
        line(coords, False)
    elif kind == "MultiLineString":
        for item in coords:
            line(item, False)
    elif kind == "Polygon":
        for ring in coords:
            line(ring, True)
    elif kind == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                line(ring, True)
    return " ".join(paths)


def render_map(site: dict[str, Any], features: list[dict[str, Any]]) -> str:
    size, radius = 420, 1_500.0
    lat, lon = site["lat"], site["lon"]
    layers = {
        "land": [],
        "infrastructure": [],
        "roads": [],
        "buildings": [],
        "labels": [],
    }
    for feature in features:
        feature_tags = tags(feature)
        distance = geometry_distance_m(feature, lat, lon)
        if distance > radius * 1.5:
            continue
        geometry = feature.get("geometry") or {}
        projected = project_points(geometry.get("coordinates"), lat, lon, size, radius)
        path = svg_path(projected, geometry.get("type", ""))
        if path and any(
            key in feature_tags for key in ("landuse", "natural", "leisure")
        ):
            value = (
                feature_tags.get("landuse")
                or feature_tags.get("natural")
                or feature_tags.get("leisure")
            )
            color = {
                "forest": "#2f6b43",
                "wood": "#2f6b43",
                "grass": "#6d8b4e",
                "meadow": "#7d9955",
                "farmland": "#9a8749",
                "orchard": "#7f8742",
                "residential": "#5d6470",
                "industrial": "#7d5548",
                "quarry": "#76584c",
                "construction": "#a8653e",
                "golf_course": "#507a50",
            }.get(value, "#40545a")
            layers["land"].append(
                f'<path d="{path}" fill="{color}" fill-opacity=".45" stroke="none"/>'
            )
        if path and (
            feature_tags.get("power") == "plant"
            or feature_tags.get("plant:source") == "solar"
        ):
            layers["infrastructure"].append(
                f'<path d="{path}" fill="#795bd8" fill-opacity=".72" stroke="#b7a7ff" stroke-width="1"/>'
            )
        if path and "highway" in feature_tags:
            width = (
                2.4
                if feature_tags["highway"] in {"primary", "secondary", "tertiary"}
                else 1.1
            )
            layers["roads"].append(
                f'<path d="{path}" fill="none" stroke="#cfd7da" stroke-width="{width}"/>'
            )
        if path and "building" in feature_tags and distance <= 700:
            layers["buildings"].append(
                f'<path d="{path}" fill="#d49262" stroke="#f0ba8d" stroke-width=".35"/>'
            )
        if feature_tags.get("natural") == "peak" and distance <= radius:
            center = feature_center(feature)
            name = feature_tags.get("name") or feature_tags.get("name:ko")
            if center and name:
                x, y = project_points([center[1], center[0]], lat, lon, size, radius)
                layers["labels"].append(
                    f'<path d="M{x:.1f},{y - 7:.1f} l-6,11 h12 z" fill="#f0cf67"/>'
                    f'<text x="{x + 8:.1f}" y="{y + 3:.1f}">{html.escape(name)}</text>'
                )
        if feature_tags.get("leisure") == "golf_course" and distance <= radius:
            center = feature_center(feature)
            name = feature_tags.get("name") or feature_tags.get("name:ko") or "골프장"
            if center:
                x, y = project_points([center[1], center[0]], lat, lon, size, radius)
                layers["labels"].append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#f0cf67"/>'
                    f'<text x="{x + 7:.1f}" y="{y + 3:.1f}">{html.escape(name)}</text>'
                )
        if feature_tags.get("power") == "plant" and distance <= radius:
            center = feature_center(feature)
            if center:
                x, y = project_points([center[1], center[0]], lat, lon, size, radius)
                layers["labels"].append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#d3c9ff"/>'
                )
    content = "".join(
        layers["land"]
        + layers["infrastructure"]
        + layers["roads"]
        + layers["buildings"]
        + layers["labels"]
    )
    return (
        f'<svg class="local-map" viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="{html.escape(site["candidate_id"])} local OSM context">'
        f'<rect width="{size}" height="{size}" fill="#182126"/>{content}'
        f'<circle cx="{size / 2}" cy="{size / 2}" r="8" fill="none" stroke="#3de1df" stroke-width="3"/>'
        f'<line x1="{size / 2 - 14}" y1="{size / 2}" x2="{size / 2 + 14}" y2="{size / 2}" stroke="#3de1df"/>'
        f'<line x1="{size / 2}" y1="{size / 2 - 14}" x2="{size / 2}" y2="{size / 2 + 14}" stroke="#3de1df"/>'
        f'<text x="12" y="{size - 14}" class="scale">3 km window · © OpenStreetMap contributors (ODbL)</text>'
        "</svg>"
    )


def render_dashboard(
    context: dict[str, Any],
    features: list[dict[str, Any]],
    assistant_review: dict[str, Any] | None = None,
) -> str:
    cards = []
    for site in context["sites"]:
        oreum = site["official_oreum_name_matches_5km"]
        nearest = oreum[0] if oreum else None
        oreum_text = (
            f"{html.escape(nearest['official_name'])} · {nearest['distance_m']} m "
            f"({html.escape(nearest['official_address'])})"
            if nearest
            else "5 km 안에서 공식 목록과 이름이 일치하는 OSM peak 없음"
        )
        land = [
            item["tags"].get("landuse")
            or item["tags"].get("natural")
            or item["tags"].get("leisure")
            for item in site["containing_current_landcover"]
        ]
        permit = site["permit_context"]
        farmmap_edges = site.get("farmmap_evidence", [])
        if farmmap_edges:
            farm = farmmap_edges[0]
            attributes = farm["attributes"]
            relation_label = {
                "official_pre_change_state_at_point": "변화 전 공식 상태",
                "official_state_within_change_window_at_point": "변화 구간 내 공식 상태",
                "official_post_change_state_at_point": "변화 후 공식 상태",
            }.get(farm["relation"], farm["relation"])
            gap_text = (
                f" · 경계와 {farm['day_gap']}일 간격"
                if farm.get("day_gap") is not None
                else ""
            )
            farmmap_text = (
                f'<span class="grade grade-b">B · 공식 상태</span> '
                f"{html.escape(attributes['farm_class'])} · PNU "
                f"{html.escape(farm.get('pnu') or '없음')} · "
                f"항공 {html.escape(attributes['flight_date'] or '미상')} · "
                f"갱신 {html.escape(attributes['update_date'] or '미상')}<br/>"
                f"{html.escape(relation_label)} · 변화 구간 "
                f"{html.escape(attributes['t_before'])} → "
                f"{html.escape(attributes['t_after'])}{gap_text}<br/>"
                f'<span class="warning">점이 농지 polygon 안에 있다는 뜻이며 변화 원인은 아님</span>'
            )
        else:
            farmmap_text = (
                '<span class="grade grade-u">U · point hit 없음</span> '
                "팜맵은 농경지 polygon만 포함하므로 무변화·비농지의 음성 근거가 아님"
            )
        cards.append(f"""
        <article class="card" data-id="{html.escape(site["candidate_id"])}">
          <h2>{html.escape(site["candidate_id"])}</h2>
          <div class="coords">{site["lat"]:.4f}, {site["lon"]:.4f}</div>
          <div class="visuals">
            <img src="candidate_images/{html.escape(Path(site["rgb_image"]).name)}" alt="4-year RGB review"/>
            {render_map(site, features)}
          </div>
          <div class="facts">
            <p><b>공식 오름명 결합</b> {oreum_text}</p>
            <p><b>현재 OSM 토지이용</b> {html.escape(", ".join(filter(None, land)) or "point containment 없음")}</p>
            <p><b>현재 지도 밀도</b> 반경 500 m 건물 {site["current_building_count_500m"]}개 · 도로 객체 {site["current_road_features_500m"]}개</p>
            <p><b>국토부 허가 문맥</b> {html.escape(str(permit["locality_token"]) if permit["locality_token"] else "지역명 연결 실패")} 동일 지역 {permit["record_count"]}건 · 필지 일치 아님</p>
            <p class="wide-fact"><b>2025 제주 공식 팜맵</b> {farmmap_text}</p>
          </div>
          <div class="review-row">
            <label>결합 후 변화 <select class="effect">
              <option value="context_only">문맥만 추가</option><option value="strengthens">개발 해석 강화</option>
              <option value="weakens">개발 해석 약화</option><option value="changes_category">범주 변경</option>
              <option value="unavailable">판정 불가</option>
            </select></label>
            <label>확신도 <select class="confidence"><option>low</option><option>medium</option><option>high</option></select></label>
            <label class="wide">메모 <textarea class="notes">{html.escape(site["initial_after_context_judgment"]["notes"])}</textarea></label>
          </div>
        </article>""")
    embedded = json.dumps(context, ensure_ascii=False).replace("</", "<\\/")
    embedded_review = json.dumps(
        assistant_review or {"reviews": {}}, ensure_ascii=False
    ).replace("</", "<\\/")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/><title>Jeju public-context evidence pack</title>
<style>
:root{{--bg:#0d1418;--panel:#162126;--line:#314047;--text:#e7eeee;--muted:#9bacb1;--cyan:#3de1df;--amber:#f0cf67}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:3;background:#0d1418ee;border-bottom:1px solid var(--line);padding:18px 24px}}
h1{{margin:0;font-size:22px}} header p{{margin:5px 0 0;color:var(--muted)}} .status{{color:var(--cyan)}}
main{{max-width:1500px;margin:auto;padding:22px;display:grid;gap:22px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}}
h2{{display:inline;margin:0 10px 0 0}} .coords{{display:inline;color:var(--muted)}} .visuals{{display:grid;grid-template-columns:minmax(0,2fr) minmax(320px,1fr);gap:14px;margin:15px 0}}
img,.local-map{{width:100%;border:1px solid var(--line);border-radius:10px;background:#111}} .local-map text{{font-size:10px;fill:#f4f0dc;paint-order:stroke;stroke:#182126;stroke-width:3px}} .local-map .scale{{fill:#aab9bd;stroke-width:2px}}
.facts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px 18px}} .facts p{{margin:0;padding:8px 0;border-top:1px solid #27343a}} b{{color:var(--amber)}}
.wide-fact{{grid-column:1/-1}} .grade{{display:inline-block;border-radius:999px;padding:2px 7px;margin-right:5px;font-weight:700}} .grade-b{{color:#071315;background:var(--cyan)}} .grade-u{{color:#dbe4e6;background:#46565d}} .warning{{color:var(--muted)}}
.review-row{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}} label{{color:var(--muted)}} select,textarea{{display:block;width:100%;margin-top:4px;background:#0d1519;color:var(--text);border:1px solid #405159;border-radius:7px;padding:8px}} .wide{{grid-column:1/-1}} textarea{{min-height:64px}}
.toolbar{{display:flex;gap:10px;align-items:center;margin-top:10px}} button{{background:var(--cyan);border:0;border-radius:7px;padding:8px 13px;font-weight:700;cursor:pointer}}
footer{{max-width:1500px;margin:0 auto;padding:0 22px 30px;color:var(--muted)}} a{{color:var(--cyan)}}
@media(max-width:900px){{.visuals,.facts{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>제주 변화 후보 × 한국 공공데이터</h1>
<p>RGB-only 판정을 유지한 채 OSM과 2025 제주 공식 팜맵 289,379개 polygon을 모두 로컬 결합했습니다. 팜맵 hit는 날짜가 있는 상태 근거이며 원인 판정은 아닙니다.</p>
<div class="toolbar"><span class="status" id="status">4개 사이트 · 결합 후 판정 4/4</span><button id="export">판정 JSON 내보내기</button></div></header>
<main>{"".join(cards)}</main>
<footer>지도 데이터: <a href="{OSM_COPYRIGHT}">© OpenStreetMap contributors, ODbL</a>, Geofabrik 대한민국 스냅샷.
오름명·속성: 제주특별자치도 오름현황. 개발행위: 국토교통부 토지이음 공개 CSV. 농지 polygon: 농림수산식품교육문화정보원 2025 제주 팜맵.
현재 지도 객체·팜맵 상태·지역 단위 허가는 단독으로 과거 변화의 원인·허가·위법성을 증명하지 않습니다.</footer>
<script>const context={embedded}; const assistantReview={embedded_review}; const key='jeju-public-context-review-v1'; let saved=JSON.parse(localStorage.getItem(key)||'{{}}');
document.querySelectorAll('.card').forEach(card=>{{const id=card.dataset.id; const base=assistantReview.reviews[id]||context.sites.find(x=>x.candidate_id===id).initial_after_context_judgment; const val=saved[id]||base;
card.querySelector('.effect').value=val.effect||'context_only'; card.querySelector('.confidence').value=val.confidence||'low'; card.querySelector('.notes').value=val.notes||'';
card.querySelectorAll('select,textarea').forEach(el=>el.addEventListener('input',save));}});
function save(){{document.querySelectorAll('.card').forEach(card=>{{saved[card.dataset.id]={{effect:card.querySelector('.effect').value,confidence:card.querySelector('.confidence').value,notes:card.querySelector('.notes').value}}}});localStorage.setItem(key,JSON.stringify(saved));update();}}
function update(){{const done=Object.values(saved).filter(x=>x.effect).length;document.getElementById('status').textContent=`4개 사이트 · 결합 후 판정 ${{done}}/4`;}} save();
document.getElementById('export').onclick=()=>{{const out={{schema:'jeju-public-context-human-review-v1',source_context:'candidate_public_context.json',reviews:saved}};const blob=new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='jeju_public_context_review.json';a.click();URL.revokeObjectURL(url);}};
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm-pbf", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oreum-csv", type=Path, required=True)
    parser.add_argument("--permit-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--osmium", default="osmium")
    parser.add_argument("--assistant-context-review", type=Path)
    parser.add_argument("--country-osm-md5-file", type=Path)
    parser.add_argument("--farmmap-evidence", type=Path)
    parser.add_argument("--farmmap-manifest", type=Path)
    args = parser.parse_args()
    if bool(args.farmmap_evidence) != bool(args.farmmap_manifest):
        parser.error(
            "--farmmap-evidence and --farmmap-manifest must be supplied together"
        )

    review = json.loads(args.review.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_id = {
        candidate["candidate_id"]: candidate for candidate in manifest["candidates"]
    }
    duplicate_pair = review["summary"]["duplicate_pair"]
    drop_duplicate = duplicate_pair[1]
    selected_ids = [
        candidate_id
        for candidate_id, judgment in review["reviews"].items()
        if judgment["is_persistent_change"] == "yes"
        and judgment["confidence"] == "high"
        and candidate_id != drop_duplicate
    ]
    if len(selected_ids) != 4:
        raise ValueError(f"expected 4 unique high-confidence sites, got {selected_ids}")

    official_oreum = read_csv(args.oreum_csv)
    permits = read_csv(args.permit_csv)
    with tempfile.TemporaryDirectory(prefix="jeju-osm-context-") as temp:
        features, fileinfo = run_osmium(args.osmium, args.osm_pbf, Path(temp))

    sites = [
        candidate_context(by_id[candidate_id], features, official_oreum, permits)
        for candidate_id in selected_ids
    ]
    farmmap_manifest = None
    if args.farmmap_evidence:
        evidence = json.loads(args.farmmap_evidence.read_text(encoding="utf-8"))
        if evidence.get("schema") != "kearth-evidence-edge-collection-v1":
            raise ValueError("unexpected FarmMap evidence schema")
        edges_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in evidence.get("edges", []):
            if edge.get("source_id") != "mafra_farmmap_jeju":
                raise ValueError(f"unexpected evidence source: {edge.get('source_id')}")
            edges_by_target[edge["target_id"]].append(edge)
        for site in sites:
            site["farmmap_evidence"] = sorted(
                edges_by_target.get(site["candidate_id"], []),
                key=lambda edge: edge["source_record_id"],
            )
        farmmap_manifest = json.loads(args.farmmap_manifest.read_text(encoding="utf-8"))
        if farmmap_manifest.get("source_id") != "mafra_farmmap_jeju":
            raise ValueError("unexpected FarmMap manifest source")
    timestamp = fileinfo["header"]["option"].get("timestamp")
    context = {
        "schema": (
            "jeju-public-context-evidence-v2"
            if farmmap_manifest
            else "jeju-public-context-evidence-v1"
        ),
        "selection": {
            "rule": "high-confidence persistent RGB records, duplicate pair reduced to first record",
            "candidate_ids": selected_ids,
            "frozen_rgb_review_sha256": sha256(args.review),
        },
        "privacy": {
            "candidate_coordinates_sent_to_external_services": False,
            "method": "download full South Korea OSM snapshot, then spatially join locally",
        },
        "sources": {
            "osm": {
                "provider": "OpenStreetMap contributors via Geofabrik",
                "source_url": GEOFABRIK_SOURCE,
                "license": "ODbL 1.0",
                "snapshot_timestamp": timestamp,
                "south_korea_source_md5": (
                    args.country_osm_md5_file.read_text(encoding="utf-8").split()[0]
                    if args.country_osm_md5_file
                    else None
                ),
                "local_jeju_extract_sha256": sha256(args.osm_pbf),
            },
            "official_oreum": {
                "provider": "제주특별자치도",
                "snapshot": "2024-03-31",
                "rows": len(official_oreum),
                "join_limit": "official file has address but no coordinates; OSM peak supplies location",
            },
            "development_permits": {
                "provider": "국토교통부",
                "snapshot": "2026-08-19",
                "jeju_rows": len(permits),
                "join_limit": "same named locality only until an authorized cadastral PNU layer is available",
            },
        },
        "sites": sites,
        "limitations": [
            "OSM is a current community map, not a historical ground-truth layer.",
            "An official oreum name match locates an OSM peak point, not an official oreum boundary.",
            "Permit context is locality-level and must not be described as a candidate parcel match.",
            "The permit snapshot has no 2023/2024 Jeju records, so absence is not negative evidence.",
        ],
    }
    if farmmap_manifest:
        context["sources"]["farmmap"] = {
            "provider": farmmap_manifest["provider"],
            "catalog_url": farmmap_manifest["catalog_url"],
            "snapshot": farmmap_manifest["snapshot_date"],
            "rows": farmmap_manifest["row_count"],
            "raw_sha256": farmmap_manifest["raw_sha256"],
            "crs": farmmap_manifest["crs"],
            "join_limit": (
                "official dated farm-state polygon covering the candidate point; "
                "not a change footprint or causal event"
            ),
        }
        context["limitations"].append(
            "FarmMap covers agricultural polygons only; a point miss is unknown, not negative evidence."
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "candidate_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for site in sites:
        source_image = args.manifest.parent / site["rgb_image"]
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        shutil.copy2(source_image, image_dir / source_image.name)
    context_path = args.out_dir / "candidate_public_context.json"
    dashboard_path = args.out_dir / "evidence_dashboard.html"
    context_path.write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    assistant_context_review = (
        json.loads(args.assistant_context_review.read_text(encoding="utf-8"))
        if args.assistant_context_review
        else None
    )
    dashboard_path.write_text(
        render_dashboard(context, features, assistant_context_review), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sites": selected_ids,
                "osm_features_considered": len(features),
                "official_oreum_matches": {
                    site["candidate_id"]: len(site["official_oreum_name_matches_5km"])
                    for site in sites
                },
                "permit_context_records": {
                    site["candidate_id"]: site["permit_context"]["record_count"]
                    for site in sites
                },
                "farmmap_point_hits": {
                    site["candidate_id"]: len(site.get("farmmap_evidence", []))
                    for site in sites
                },
                "outputs": [str(context_path), str(dashboard_path)],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
