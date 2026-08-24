#!/usr/bin/env python3
"""Rebuild deterministic joins from an existing K-Earth API raw snapshot."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from collect_kearth_api_snapshot import (
    build_candidate_evidence,
    load_targets,
    write_json,
)
from kearth_public.api_snapshot import data_go_items, load_json_response


def source_pnu_records(paths: Iterable[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as source:
                for index, row in enumerate(csv.DictReader(source), 1):
                    pnu = str(row.get("pnu", row.get("PNU", ""))).strip()
                    if len(pnu) != 19 or not pnu.isdigit():
                        continue
                    records.append(
                        {
                            "pnu": pnu,
                            "source_id": "existing_farmmap_permit_link",
                            "source_record_id": str(row.get("farm_id") or index),
                            "target_id": None,
                            "attributes": {
                                "farm_class": row.get("farm_class"),
                                "farm_flight_date": row.get("farm_flight_date"),
                                "permit_date": row.get("permit_date"),
                            },
                        }
                    )
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for index, edge in enumerate(payload.get("edges", []), 1):
                pnu = str(edge.get("pnu", "")).strip()
                if len(pnu) != 19 or not pnu.isdigit():
                    continue
                records.append(
                    {
                        "pnu": pnu,
                        "source_id": str(edge.get("source_id") or "existing_evidence_edge"),
                        "source_record_id": str(edge.get("source_record_id") or index),
                        "target_id": edge.get("target_id"),
                        "attributes": edge.get("attributes", {}),
                    }
                )
    return records


def existing_candidate_anchors(records: list[dict[str, object]]) -> list[dict[str, object]]:
    anchors: list[dict[str, object]] = []
    for record in records:
        target_id = str(record.get("target_id") or "")
        if not target_id or target_id.startswith("JJ-OREUM-"):
            continue
        anchors.append(
            {
                "target_id": target_id,
                "request_hash": None,
                "request_outcome": "reused_existing_official_snapshot",
                "api_status": None,
                "feature_count": 1,
                "pnu": record["pnu"],
                "address": record.get("attributes", {}).get("farm_address"),
                "feature": None,
                "evidence_grade": "C",
                "interpretation": "candidate point inside a dated official FarmMap polygon; not an official oreum boundary",
                "source_id": record["source_id"],
                "source_record_id": record["source_record_id"],
            }
        )
    return anchors


def cross_source_links(
    pnu_records: list[dict[str, object]], building_events: list[dict[str, object]]
) -> list[dict[str, object]]:
    refs_by_pnu: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in pnu_records:
        refs_by_pnu[str(record["pnu"])].append(record)
    events_by_pnu: dict[str, list[dict[str, object]]] = defaultdict(list)
    for event in building_events:
        if event.get("pnu"):
            events_by_pnu[str(event["pnu"])].append(event)
    links: list[dict[str, object]] = []
    for pnu in sorted(refs_by_pnu):
        matches = events_by_pnu.get(pnu, [])
        links.append(
            {
                "pnu": pnu,
                "existing_record_count": len(refs_by_pnu[pnu]),
                "existing_records": refs_by_pnu[pnu],
                "building_event_count": len(matches),
                "building_events": matches,
                "relation": "exact_pnu_cross_source_context" if matches else "no_building_match_unknown",
                "negative_interpretation_allowed": False,
            }
        )
    return links


def observation_context(snapshot_dir: Path, requests: list[dict[str, object]]) -> dict[str, object]:
    gk2a: list[dict[str, object]] = []
    landcover: list[dict[str, object]] = []
    for record in requests:
        source_id = str(record["source_id"])
        if source_id.startswith("gk2a_cloud"):
            payload = load_json_response(snapshot_dir, record)
            items, meta = data_go_items(payload)
            item = items[0] if items else {}
            raw_values = str(item.get("value", ""))
            values = raw_values.split(",") if raw_values else []
            gk2a.append(
                {
                    "target_time": record.get("target_id"),
                    "semantic_status": record.get("semantic_status"),
                    "api_result_code": record.get("api_result_code"),
                    "api_result_message": record.get("api_result_message"),
                    "grid_metadata": {key: item.get(key) for key in ("dateTime", "gridKm", "xdim", "ydim", "x0", "y0", "unit")},
                    "grid_value_count": len(values),
                    "value_histogram": dict(sorted(Counter(values).items())) if values else {},
                    "raw_file": record.get("raw_file"),
                    "request_hash": record["request_hash"],
                }
            )
        elif source_id == "mcee_landcover":
            target_id, year = str(record.get("target_id", "")).rsplit(":", 1)
            landcover.append(
                {
                    "target_id": target_id,
                    "year": int(year),
                    "semantic_status": record.get("semantic_status"),
                    "raw_file": record.get("raw_file"),
                    "raw_sha256": record.get("raw_sha256"),
                    "raw_bytes": record.get("raw_bytes"),
                    "request_hash": record["request_hash"],
                }
            )
    return {
        "schema": "kearth-observation-context-v1",
        "gk2a": gk2a,
        "landcover_tiles": landcover,
        "limitations": [
            "The GK2A endpoint rejected historical OlmoEarth dates and returned only the latest allowed national grid.",
            "Land-cover tiles are annual classified state maps and do not by themselves identify a causal event.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--oreum-registry", type=Path, required=True)
    parser.add_argument("--pnu-source", type=Path, action="append", default=[])
    args = parser.parse_args()

    requests = json.loads((args.snapshot_dir / "requests.json").read_text(encoding="utf-8"))["requests"]
    targets = load_targets(args.candidate_manifest, args.oreum_registry)
    api_anchors = json.loads((args.snapshot_dir / "parcel_anchors.json").read_text(encoding="utf-8"))["anchors"]
    building_events = json.loads((args.snapshot_dir / "building_events.json").read_text(encoding="utf-8"))["events"]
    eia = json.loads((args.snapshot_dir / "eia_features.json").read_text(encoding="utf-8"))["features"]
    pnu_records = source_pnu_records(args.pnu_source)
    reused_anchors = existing_candidate_anchors(pnu_records)
    anchors_by_target = {str(anchor["target_id"]): anchor for anchor in reused_anchors}
    anchors_by_target.update(
        {
            str(anchor["target_id"]): anchor
            for anchor in api_anchors
            if anchor.get("feature_count", 0) > 0
        }
    )
    anchors = list(anchors_by_target.values())
    candidates = build_candidate_evidence(targets, anchors, building_events, eia, requests)
    links = cross_source_links(pnu_records, building_events)
    context = observation_context(args.snapshot_dir, requests)

    write_json(args.snapshot_dir / "candidate_evidence.json", {"schema": "kearth-candidate-api-evidence-v1", "records": candidates})
    write_json(args.snapshot_dir / "cross_source_pnu_links.json", {"schema": "kearth-cross-source-pnu-links-v1", "links": links})
    write_json(args.snapshot_dir / "observation_context.json", context)
    summary_path = args.snapshot_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    vm_probe_path = args.snapshot_dir / "vworld_vm_probe" / "requests.json"
    vm_probe_status = None
    if vm_probe_path.exists():
        vm_requests = json.loads(vm_probe_path.read_text(encoding="utf-8"))["requests"]
        if vm_requests:
            vm_probe_status = {
                "semantic_status": vm_requests[0].get("semantic_status"),
                "api_error_code": (vm_requests[0].get("api_error") or {}).get("code"),
                "target_id": vm_requests[0].get("target_id"),
            }
    summary.update(
        {
            "candidate_existing_parcel_anchors": len(reused_anchors),
            "cross_source_pnu_population": len(links),
            "cross_source_pnu_with_building_event": sum(link["building_event_count"] > 0 for link in links),
            "gk2a_current_grid_values": max((item["grid_value_count"] for item in context["gk2a"]), default=0),
            "landcover_tile_rows": len(context["landcover_tiles"]),
            "candidate_official_corroboration_b": sum(item["causal_evidence_grade"] == "B" for item in candidates),
            "vworld_vm_probe": vm_probe_status,
        }
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
