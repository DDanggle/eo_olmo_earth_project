#!/usr/bin/env python3
"""Collect a bounded Jeju public-API snapshot and join it to OlmoEarth targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping

from kearth_public.api_snapshot import (
    RequestSpec,
    data_go_items,
    eia_features,
    execute_request,
    load_json_response,
    read_env_file,
    point_in_ring,
    vworld_features,
    vworld_semantic_status,
)


SOURCE_IDS = {
    "vworld",
    "building",
    "gk2a",
    "eia",
    "landcover",
}
JEJU_BBOX_4326 = (126.10, 33.10, 126.98, 33.62)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_targets(candidate_path: Path, registry_path: Path) -> list[dict[str, object]]:
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))["candidates"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if len(registry.get("records", [])) != 368:
        raise ValueError("oreum registry must preserve the official 368-record denominator")
    targets: list[dict[str, object]] = []
    for item in candidates:
        targets.append(
            {
                "target_id": item["candidate_id"],
                "target_kind": "olmoearth_candidate",
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
                "observation_dates": sorted(
                    {
                        value["acquisition_date"]
                        for value in item.get("season_aligned_rgb", {}).values()
                        if value.get("acquisition_date")
                    }
                ),
            }
        )
    for item in registry["records"]:
        location = item.get("location", {})
        if location.get("lat") is None or location.get("lon") is None:
            continue
        targets.append(
            {
                "target_id": item["oreum_id"],
                "target_kind": "oreum_offline_osm_point",
                "name": item["name"],
                "lat": float(location["lat"]),
                "lon": float(location["lon"]),
                "observation_dates": [],
            }
        )
    return sorted(targets, key=lambda item: (str(item["target_kind"]), str(item["target_id"])))


def legal_dong_codes(paths: Iterable[Path]) -> list[tuple[str, str]]:
    codes: set[tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as source:
                for row in csv.DictReader(source):
                    pnu = str(row.get("pnu", row.get("PNU", ""))).strip()
                    if len(pnu) == 19 and pnu.isdigit() and pnu.startswith(("50110", "50130")):
                        codes.add((pnu[:5], pnu[5:10]))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for edge in payload.get("edges", []):
                pnu = str(edge.get("pnu", ""))
                if len(pnu) == 19 and pnu.isdigit() and pnu.startswith(("50110", "50130")):
                    codes.add((pnu[:5], pnu[5:10]))
    return sorted(codes)


def build_specs(
    sources: set[str],
    targets: list[dict[str, object]],
    dong_codes: list[tuple[str, str]],
    *,
    vworld_domain: str,
    gk2a_latest_datetime: str,
) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    if "vworld" in sources:
        for target in targets:
            specs.append(
                RequestSpec(
                    source_id="vworld_cadastral",
                    endpoint="https://api.vworld.kr/req/data",
                    public_params={
                        "service": "data",
                        "version": "2.0",
                        "request": "GetFeature",
                        "format": "json",
                        "errorFormat": "json",
                        "size": 10,
                        "page": 1,
                        "data": "LP_PA_CBND_BUBUN",
                        "geometry": "true",
                        "attribute": "true",
                        "crs": "EPSG:4326",
                        "geomFilter": f"POINT({target['lon']:.7f} {target['lat']:.7f})",
                        "domain": vworld_domain,
                    },
                    secret_params={"key": "VWORLD_API_KEY"},
                    response_suffix=".json",
                    target_id=str(target["target_id"]),
                )
            )
    if "building" in sources:
        for endpoint_name, source_id in (
            ("getApBasisOulnInfo", "building_hub_basis"),
            ("getApDemolExtngMgmRgstInfo", "building_hub_demolition"),
        ):
            for sigungu, bjdong in dong_codes:
                specs.append(
                    RequestSpec(
                        source_id=source_id,
                        endpoint=f"https://apis.data.go.kr/1613000/ArchPmsHubService/{endpoint_name}",
                        public_params={
                            "sigunguCd": sigungu,
                            "bjdongCd": bjdong,
                            "startDate": "20230101",
                            "endDate": "20261231",
                            "_type": "json",
                            "numOfRows": 1000,
                            "pageNo": 1,
                        },
                        secret_params={"serviceKey": "DATA_GO_KR_SERVICE_KEY"},
                        response_suffix=".json",
                        target_id=f"{sigungu}{bjdong}",
                    )
                )
    observation_dates = sorted(
        {
            date
            for target in targets
            if target["target_kind"] == "olmoearth_candidate"
            for date in target.get("observation_dates", [])
        }
    )
    if "gk2a" in sources:
        for date in observation_dates:
            specs.append(
                RequestSpec(
                    source_id="gk2a_cloud",
                    endpoint="https://apis.data.go.kr/1360000/CloudSatlitInfoService/getGk2acldAll",
                    public_params={
                        "pageNo": 1,
                        "numOfRows": 10,
                        "dataType": "JSON",
                        "dateTime": date.replace("-", "") + "0300",
                        "resultType": "cld",
                    },
                    secret_params={"ServiceKey": "DATA_GO_KR_SERVICE_KEY"},
                    response_suffix=".json",
                    target_id=date,
                )
            )
        specs.append(
            RequestSpec(
                source_id="gk2a_cloud_current",
                endpoint="https://apis.data.go.kr/1360000/CloudSatlitInfoService/getGk2acldAll",
                public_params={
                    "pageNo": 1,
                    "numOfRows": 10,
                    "dataType": "JSON",
                    "dateTime": gk2a_latest_datetime,
                    "resultType": "cld",
                },
                secret_params={"ServiceKey": "DATA_GO_KR_SERVICE_KEY"},
                response_suffix=".json",
                target_id=gk2a_latest_datetime,
            )
        )
    if "eia" in sources:
        specs.append(
            RequestSpec(
                source_id="eia_business_area",
                endpoint="https://apis.data.go.kr/1480523/BsnsAreaService/getInfoWFS",
                public_params={
                    "srsName": "EPSG:4326",
                    "maxFeatures": 1000,
                    "resultType": "results",
                    "bbox": ",".join(str(value) for value in JEJU_BBOX_4326),
                },
                secret_params={"ServiceKey": "DATA_GO_KR_SERVICE_KEY"},
                response_suffix=".xml",
                target_id="jeju_bbox",
            )
        )
    if "landcover" in sources:
        candidate_targets = [
            target for target in targets if target["target_kind"] == "olmoearth_candidate"
        ]
        for target in candidate_targets:
            lon, lat = float(target["lon"]), float(target["lat"])
            bbox = f"{lon - 0.005:.7f},{lat - 0.005:.7f},{lon + 0.005:.7f},{lat + 0.005:.7f}"
            for year in (2023, 2024, 2025):
                specs.append(
                    RequestSpec(
                        source_id="mcee_landcover",
                        endpoint="https://api.mcee.go.kr/geoserver/wms",
                        public_params={
                            "service": "WMS",
                            "version": "1.1.1",
                            "request": "GetMap",
                            "layers": f"EGIS:lv3_{year}y",
                            "styles": "",
                            "format": "image/png",
                            "transparent": "true",
                            "srs": "EPSG:4326",
                            "bbox": bbox,
                            "width": 256,
                            "height": 256,
                        },
                        secret_params={},
                        response_suffix=".png",
                        target_id=f"{target['target_id']}:{year}",
                    )
                )
    return specs


def execute_specs(
    specs: Iterable[RequestSpec],
    *,
    output_dir: Path,
    env: Mapping[str, str],
    retrieved_at: str,
    offset: int = 0,
    total_hint: int | None = None,
) -> list[dict[str, object]]:
    specs = list(specs)
    total = total_hint or len(specs)
    records: list[dict[str, object]] = []
    for index, spec in enumerate(specs, offset + 1):
        print(f"[{index}/{total}] {spec.source_id} {spec.target_id or '-'}", flush=True)
        records.append(
            execute_request(
                spec,
                output_dir=output_dir,
                env=env,
                retrieved_at=retrieved_at,
            )
        )
    return records


def building_followup_specs(
    output_dir: Path,
    records: list[dict[str, object]],
    *,
    max_pages_per_code: int,
) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    for record in records:
        if not str(record["source_id"]).startswith("building_hub_"):
            continue
        _, meta = data_go_items(load_json_response(output_dir, record))
        try:
            total_count = int(meta.get("total_count") or 0)
            page_size = int(meta.get("num_of_rows") or 0)
        except (TypeError, ValueError):
            continue
        if page_size <= 0:
            continue
        page_count = (total_count + page_size - 1) // page_size
        if max_pages_per_code:
            page_count = min(page_count, max_pages_per_code)
        for page_no in range(2, page_count + 1):
            params = dict(record["public_params"])
            params["pageNo"] = page_no
            specs.append(
                RequestSpec(
                    source_id=str(record["source_id"]),
                    endpoint=str(record["endpoint"]),
                    public_params=params,
                    secret_params={"serviceKey": "DATA_GO_KR_SERVICE_KEY"},
                    response_suffix=".json",
                    target_id=str(record.get("target_id") or ""),
                )
            )
    return specs


def annotate_semantics(output_dir: Path, records: list[dict[str, object]]) -> None:
    for record in records:
        source_id = str(record["source_id"])
        if record["outcome"] != "http_success":
            record["semantic_status"] = "request_failed"
            continue
        if source_id == "vworld_cadastral":
            features, meta = vworld_features(load_json_response(output_dir, record))
            record["semantic_status"] = vworld_semantic_status(meta, len(features))
            record["semantic_item_count"] = len(features)
            record["api_error"] = meta.get("error")
        elif source_id.startswith("building_hub_") or source_id.startswith("gk2a_cloud"):
            items, meta = data_go_items(load_json_response(output_dir, record))
            header = meta.get("header", {})
            result_code = str(header.get("resultCode", "")) if isinstance(header, dict) else ""
            record["semantic_status"] = "api_success" if result_code in {"00", "0", "NORMAL_SERVICE"} else "api_error"
            record["semantic_item_count"] = len(items)
            record["api_result_code"] = result_code
            record["api_result_message"] = header.get("resultMsg") if isinstance(header, dict) else None
        elif source_id == "eia_business_area":
            raw_file = record.get("raw_file")
            features = eia_features((output_dir / str(raw_file)).read_bytes()) if raw_file else []
            record["semantic_status"] = "api_success" if features else "api_error_or_empty"
            record["semantic_item_count"] = len(features)
        elif source_id == "mcee_landcover":
            content_type = str(record.get("content_type", "")).lower()
            record["semantic_status"] = "api_success" if "image/png" in content_type else "api_error"
            record["semantic_item_count"] = 1 if record["semantic_status"] == "api_success" else 0
        else:
            record["semantic_status"] = "not_parsed"


def parcel_anchors(output_dir: Path, requests: list[dict[str, object]]) -> list[dict[str, object]]:
    anchors: list[dict[str, object]] = []
    for record in requests:
        if record["source_id"] != "vworld_cadastral":
            continue
        features, parse = vworld_features(load_json_response(output_dir, record))
        properties = features[0].get("properties", {}) if features else {}
        feature = features[0] if features else None
        anchors.append(
            {
                "target_id": record.get("target_id"),
                "request_hash": record["request_hash"],
                "request_outcome": record["outcome"],
                "api_status": parse.get("status"),
                "feature_count": len(features),
                "pnu": properties.get("pnu") if isinstance(properties, dict) else None,
                "address": properties.get("addr") if isinstance(properties, dict) else None,
                "feature": feature,
                "evidence_grade": "C" if feature else "U",
                "interpretation": "representative point parcel; not an official oreum boundary",
                "source_id": "vworld_cadastral",
            }
        )
    return anchors


def building_events(output_dir: Path, requests: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    events: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for record in requests:
        if not str(record["source_id"]).startswith("building_hub_"):
            continue
        items, meta = data_go_items(load_json_response(output_dir, record))
        coverage.append({**record, "response_meta": meta, "item_count": len(items)})
        for item in items:
            pnu = None
            plat_gb = str(item.get("platGbCd", "")).strip()
            sigungu = str(item.get("sigunguCd", "")).strip()
            bjdong = str(item.get("bjdongCd", "")).strip()
            bun = str(item.get("bun", "")).strip()
            ji = str(item.get("ji", "")).strip()
            if plat_gb in {"0", "1"} and sigungu.isdigit() and bjdong.isdigit() and bun.isdigit() and ji.isdigit():
                pnu = f"{sigungu.zfill(5)}{bjdong.zfill(5)}{'1' if plat_gb == '0' else '2'}{bun.zfill(4)}{ji.zfill(4)}"
            events.append(
                {
                    "source_id": record["source_id"],
                    "request_hash": record["request_hash"],
                    "legal_dong_code": record.get("target_id"),
                    "pnu": pnu,
                    "management_id": item.get("mgmPmsrgstPk") or item.get("mgmUpperBldrgstPk"),
                    "address": item.get("platPlc"),
                    "building_name": item.get("bldNm"),
                    "permit_date": item.get("archPmsDay"),
                    "construction_start_date": item.get("realStcnsDay"),
                    "use_approval_date": item.get("useAprDay"),
                    "created_date": item.get("crtnDay"),
                    "raw_item": item,
                }
            )
    return events, coverage


def eia_records(output_dir: Path, requests: list[dict[str, object]], targets: list[dict[str, object]]) -> list[dict[str, object]]:
    features: list[dict[str, object]] = []
    candidate_targets = [target for target in targets if target["target_kind"] == "olmoearth_candidate"]
    for record in requests:
        if record["source_id"] != "eia_business_area" or not record.get("raw_file"):
            continue
        for feature in eia_features((output_dir / str(record["raw_file"])).read_bytes()):
            matches = []
            for target in candidate_targets:
                if any(point_in_ring(float(target["lon"]), float(target["lat"]), ring) for ring in feature["rings_lon_lat"]):
                    matches.append(str(target["target_id"]))
            features.append({**feature, "candidate_point_matches": matches, "request_hash": record["request_hash"]})
    return features


def build_candidate_evidence(
    targets: list[dict[str, object]],
    anchors: list[dict[str, object]],
    events: list[dict[str, object]],
    eia: list[dict[str, object]],
    requests: list[dict[str, object]],
) -> list[dict[str, object]]:
    anchors_by_target: dict[str, list[dict[str, object]]] = {}
    for item in anchors:
        anchors_by_target.setdefault(str(item["target_id"]), []).append(item)
    events_by_dong: dict[str, list[dict[str, object]]] = {}
    for event in events:
        events_by_dong.setdefault(str(event.get("legal_dong_code", "")), []).append(event)
    request_by_target: dict[str, list[dict[str, object]]] = {}
    for record in requests:
        request_by_target.setdefault(str(record.get("target_id", "")), []).append(record)

    result: list[dict[str, object]] = []
    for target in targets:
        if target["target_kind"] != "olmoearth_candidate":
            continue
        target_id = str(target["target_id"])
        target_anchors = anchors_by_target.get(target_id, [])
        vworld_anchors = [
            item for item in target_anchors if item.get("source_id") == "vworld_cadastral"
        ]
        anchor = (vworld_anchors or target_anchors or [None])[0]
        pnus = sorted(
            {
                str(item.get("pnu"))
                for item in target_anchors
                if len(str(item.get("pnu") or "")) == 19
                and str(item.get("pnu")).isdigit()
            }
        )
        legal_dongs = sorted({pnu[:10] for pnu in pnus})
        matching_events = [
            event
            for legal_dong in legal_dongs
            for event in events_by_dong.get(legal_dong, [])
        ]
        exact_events = [
            event for event in matching_events if str(event.get("pnu") or "") in pnus
        ]
        observation_dates = sorted(str(value).replace("-", "") for value in target.get("observation_dates", []))
        aligned_exact_events = []
        if observation_dates:
            for event in exact_events:
                event_dates = [
                    str(event.get(field) or "").strip()
                    for field in ("permit_date", "construction_start_date", "use_approval_date", "created_date")
                ]
                if any(date.isdigit() and observation_dates[0] <= date <= observation_dates[-1] for date in event_dates):
                    aligned_exact_events.append(event)
        eia_matches = [feature for feature in eia if target_id in feature["candidate_point_matches"]]
        pnu_conflict = len(pnus) > 1
        grade = "B" if (aligned_exact_events and not pnu_conflict) or eia_matches else "U"
        if pnu_conflict:
            warning = (
                "parcel sources disagree on PNU; exact matches remain context until "
                "geometry and source-version conflict is resolved"
            )
        elif grade == "B":
            warning = (
                "B means official spatial/time corroboration for investigation, not proof "
                "that the administrative event caused the spectral change"
            )
        else:
            warning = (
                "same legal-dong events and state layers are context, not parcel-time causal matches"
            )
        result.append(
            {
                **target,
                "representative_parcel": anchor,
                "current_vworld_parcel": vworld_anchors[0] if vworld_anchors else None,
                "dated_farmmap_parcels": [
                    item
                    for item in target_anchors
                    if item.get("source_id") != "vworld_cadastral"
                ],
                "parcel_evidence": target_anchors,
                "parcel_pnu_values": pnus,
                "parcel_pnu_relation": (
                    "conflict" if pnu_conflict else "single" if pnus else "unresolved"
                ),
                "parcel_pnu_conflict": pnu_conflict,
                "same_legal_dong_building_event_count": len(matching_events),
                "same_legal_dong_building_events": matching_events,
                "exact_parcel_building_event_count": len(exact_events),
                "time_aligned_exact_parcel_events": aligned_exact_events,
                "eia_polygon_matches": eia_matches,
                "landcover_request_hashes": [
                    item["request_hash"]
                    for year in (2023, 2024, 2025)
                    for item in request_by_target.get(f"{target_id}:{year}", [])
                ],
                "causal_evidence_grade": grade,
                "decision": "investigate" if grade == "B" else "abstain",
                "warning": warning,
            }
        )
    return result


def secret_scan(output_dir: Path, env: Mapping[str, str]) -> list[str]:
    findings: list[str] = []
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    secrets = [
        value.encode()
        for name, value in env.items()
        if any(marker in name.upper() for marker in secret_markers) and value
    ]
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        body = path.read_bytes()
        if any(secret in body for secret in secrets):
            findings.append(str(path.relative_to(output_dir)))
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--oreum-registry", type=Path, required=True)
    parser.add_argument("--pnu-source", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", default=",".join(sorted(SOURCE_IDS)))
    parser.add_argument("--vworld-domain", default="")
    parser.add_argument("--retrieved-at", default="")
    parser.add_argument("--max-vworld-targets", type=int, default=0)
    parser.add_argument("--max-building-codes", type=int, default=0)
    parser.add_argument("--max-landcover-targets", type=int, default=0)
    parser.add_argument("--max-building-pages-per-code", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = {value.strip() for value in args.sources.split(",") if value.strip()}
    unknown = sources - SOURCE_IDS
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    env = {**read_env_file(args.env_file), **os.environ}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = args.retrieved_at or datetime.now(timezone.utc).isoformat()
    retrieved_datetime = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
    gk2a_latest_datetime = (retrieved_datetime - timedelta(days=2)).strftime("%Y%m%d0300")
    vworld_domain = args.vworld_domain or env.get("VWORLD_API_DOMAIN", "http://localhost")
    targets = load_targets(args.candidate_manifest, args.oreum_registry)
    dong_codes = legal_dong_codes(args.pnu_source)
    specs = build_specs(
        sources,
        targets,
        dong_codes[: args.max_building_codes] if args.max_building_codes else dong_codes,
        vworld_domain=vworld_domain,
        gk2a_latest_datetime=gk2a_latest_datetime,
    )
    if args.max_vworld_targets:
        vworld_specs = [spec for spec in specs if spec.source_id == "vworld_cadastral"]
        allowed_vworld_hashes = {spec.identity for spec in vworld_specs[: args.max_vworld_targets]}
        specs = [
            spec
            for spec in specs
            if spec.source_id != "vworld_cadastral" or spec.identity in allowed_vworld_hashes
        ]
    if args.max_landcover_targets:
        allowed_landcover_targets = {
            str(target["target_id"])
            for target in targets
            if target["target_kind"] == "olmoearth_candidate"
        }
        allowed_landcover_targets = set(sorted(allowed_landcover_targets)[: args.max_landcover_targets])
        specs = [
            spec
            for spec in specs
            if spec.source_id != "mcee_landcover"
            or str(spec.target_id).rsplit(":", 1)[0] in allowed_landcover_targets
        ]

    requests = execute_specs(specs, output_dir=output_dir, env=env, retrieved_at=retrieved_at)
    followups = building_followup_specs(
        output_dir,
        requests,
        max_pages_per_code=args.max_building_pages_per_code,
    )
    if followups:
        requests.extend(
            execute_specs(
                followups,
                output_dir=output_dir,
                env=env,
                retrieved_at=retrieved_at,
                offset=len(requests),
                total_hint=len(requests) + len(followups),
            )
        )
    annotate_semantics(output_dir, requests)
    anchors = parcel_anchors(output_dir, requests)
    events, building_coverage = building_events(output_dir, requests)
    eia = eia_records(output_dir, requests, targets)
    candidate_evidence = build_candidate_evidence(targets, anchors, events, eia, requests)
    write_json(output_dir / "requests.json", {"schema": "kearth-api-requests-v1", "requests": requests})
    write_json(output_dir / "parcel_anchors.json", {"schema": "kearth-parcel-anchors-v1", "anchors": anchors})
    write_json(output_dir / "building_events.json", {"schema": "kearth-building-events-v1", "events": events})
    write_json(output_dir / "building_coverage.json", {"schema": "kearth-building-coverage-v1", "requests": building_coverage})
    write_json(output_dir / "eia_features.json", {"schema": "kearth-eia-features-v1", "features": eia})
    write_json(output_dir / "candidate_evidence.json", {"schema": "kearth-candidate-api-evidence-v1", "records": candidate_evidence})

    outcomes = Counter(str(record["outcome"]) for record in requests)
    source_outcomes: dict[str, Counter[str]] = {}
    for record in requests:
        source_outcomes.setdefault(str(record["source_id"]), Counter())[str(record["outcome"])] += 1
    summary = {
        "schema": "kearth-api-snapshot-summary-v1",
        "retrieved_at": retrieved_at,
        "scope": {
            "official_oreum_denominator": 368,
            "vworld_target_points_requested": len(
                {record.get("target_id") for record in requests if record["source_id"] == "vworld_cadastral"}
            ),
            "resolved_oreum_points_available": sum(t["target_kind"] == "oreum_offline_osm_point" for t in targets),
            "olmoearth_candidates_available": sum(t["target_kind"] == "olmoearth_candidate" for t in targets),
            "building_legal_dong_codes_requested": len(
                {
                    record.get("target_id")
                    for record in requests
                    if str(record["source_id"]).startswith("building_hub_")
                }
            ),
            "jeju_bbox_4326": JEJU_BBOX_4326,
        },
        "request_count": len(requests),
        "outcomes": dict(sorted(outcomes.items())),
        "source_outcomes": {key: dict(sorted(value.items())) for key, value in sorted(source_outcomes.items())},
        "semantic_statuses": dict(sorted(Counter(str(record.get("semantic_status")) for record in requests).items())),
        "parcel_anchor_features": sum(item["feature_count"] for item in anchors),
        "building_event_rows": len(events),
        "eia_feature_rows": len(eia),
        "candidate_official_corroboration_b": sum(item["causal_evidence_grade"] == "B" for item in candidate_evidence),
        "candidate_records": len(candidate_evidence),
        "limits": [
            "VWorld parcels are representative point parcels, not official oreum boundaries.",
            "BuildingHub v1 is bounded to legal-dong codes already observed in existing PNU evidence; pages are exhausted unless an explicit page cap is set.",
            "GK2A and land-cover layers describe observation context/state, not a change cause.",
            "NGII aerial imagery is a manual application/download channel and is outside API collection.",
            "No-match remains unknown until source pagination, temporal coverage, and geometry coverage are complete.",
        ],
    }
    write_json(output_dir / "run_summary.json", summary)
    findings = secret_scan(output_dir, env)
    if findings:
        raise RuntimeError(f"credential material detected in artifacts: {findings}")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
