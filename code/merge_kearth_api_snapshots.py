#!/usr/bin/env python3
"""Combine a prior multi-source snapshot with a separately gated VWorld run.

The original failed snapshot and the VWorld-only collection remain immutable.
This script copies their redacted raw responses into a new directory, replaces
the stale VWorld request records, and rebuilds every deterministic join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from collect_kearth_api_snapshot import (
    annotate_semantics,
    build_candidate_evidence,
    load_targets,
    parcel_anchors,
    secret_scan,
    write_json,
)
from derive_kearth_api_snapshot import (
    cross_source_links,
    existing_candidate_anchors,
    observation_context,
    source_pnu_records,
)
from kearth_public.api_snapshot import read_env_file


def load_records(snapshot_dir: Path) -> list[dict[str, object]]:
    payload = json.loads((snapshot_dir / "requests.json").read_text(encoding="utf-8"))
    return [dict(record) for record in payload["requests"]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_snapshot_provenance(snapshot_dir: Path) -> dict[str, object]:
    summary_path = snapshot_dir / "run_summary.json"
    requests_path = snapshot_dir / "requests.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "path": str(snapshot_dir),
        "retrieved_at": summary.get("retrieved_at"),
        "requests_sha256": sha256(requests_path),
        "run_summary_sha256": sha256(summary_path),
    }


def verify_and_copy_raw(
    source_dir: Path,
    output_dir: Path,
    records: Iterable[Mapping[str, object]],
) -> None:
    for record in records:
        raw_file = record.get("raw_file")
        if not raw_file:
            continue
        relative = Path(str(raw_file))
        source = source_dir / relative
        target = output_dir / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        body = source.read_bytes()
        expected = str(record.get("raw_sha256") or "")
        actual = hashlib.sha256(body).hexdigest()
        if expected and actual != expected:
            raise ValueError(f"raw SHA mismatch: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != body:
            raise ValueError(f"raw filename collision: {relative}")
        if not target.exists():
            shutil.copy2(source, target)


def vworld_pnu_records(anchors: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for anchor in anchors:
        pnu = str(anchor.get("pnu") or "")
        if len(pnu) != 19 or not pnu.isdigit():
            continue
        records.append(
            {
                "pnu": pnu,
                "source_id": "vworld_cadastral",
                "source_record_id": str(anchor.get("request_hash") or anchor.get("target_id")),
                "target_id": anchor.get("target_id"),
                "attributes": {
                    "address": anchor.get("address"),
                    "evidence_grade": anchor.get("evidence_grade"),
                    "interpretation": anchor.get("interpretation"),
                },
            }
        )
    return records


def merge_request_records(
    base_records: Iterable[Mapping[str, object]],
    vworld_records: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    result = [dict(record) for record in base_records if record.get("source_id") != "vworld_cadastral"]
    result.extend(dict(record) for record in vworld_records if record.get("source_id") == "vworld_cadastral")
    identities = [str(record["request_hash"]) for record in result]
    if len(identities) != len(set(identities)):
        duplicates = sorted(key for key, count in Counter(identities).items() if count > 1)
        raise ValueError(f"duplicate request hashes after merge: {duplicates[:3]}")
    return result


def validate_vworld_snapshot(
    records: Iterable[Mapping[str, object]],
    anchors: Iterable[Mapping[str, object]],
    expected_target_ids: Iterable[str],
) -> None:
    records = list(records)
    anchors = list(anchors)
    expected = list(expected_target_ids)
    actual = [str(record.get("target_id") or "") for record in records]
    if len(actual) != len(set(actual)):
        duplicates = sorted(target for target, count in Counter(actual).items() if count > 1)
        raise ValueError(f"duplicate VWorld targets: {duplicates[:3]}")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"VWorld target mismatch: missing={missing[:3]} extra={extra[:3]}")
    anchors_by_target = {str(anchor.get("target_id") or ""): anchor for anchor in anchors}
    if set(anchors_by_target) != set(expected):
        raise ValueError("VWorld anchors do not exhaust the expected target set")
    for record in records:
        target_id = str(record.get("target_id") or "")
        semantic = str(record.get("semantic_status") or "")
        feature_count = int(anchors_by_target[target_id].get("feature_count") or 0)
        if semantic == "api_success" and feature_count != 1:
            raise ValueError(
                f"VWorld target {target_id} is ambiguous: api_success with {feature_count} features"
            )
        if semantic == "api_no_features" and feature_count != 0:
            raise ValueError(
                f"VWorld target {target_id} is inconsistent: no-features status with features"
            )
        if semantic not in {"api_success", "api_no_features"}:
            raise ValueError(f"VWorld target {target_id} failed semantically: {semantic}")


def request_summary(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    records = list(records)
    outcomes = Counter(str(record.get("outcome")) for record in records)
    semantic = Counter(str(record.get("semantic_status")) for record in records)
    source_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        source_outcomes[str(record.get("source_id"))][str(record.get("outcome"))] += 1
    return {
        "request_count": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "semantic_statuses": dict(sorted(semantic.items())),
        "source_outcomes": {
            source_id: dict(sorted(counter.items()))
            for source_id, counter in sorted(source_outcomes.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--vworld-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--oreum-registry", type=Path, required=True)
    parser.add_argument("--pnu-source", type=Path, action="append", default=[])
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to mix a new snapshot with existing artifacts: {args.output_dir}"
        )

    base_requests = load_records(args.base_snapshot)
    vworld_requests = load_records(args.vworld_snapshot)
    # Reclassify from retained raw JSON, so NOT_FOUND is empty coverage rather
    # than an authentication or transport error. No network call occurs here.
    annotate_semantics(args.vworld_snapshot, vworld_requests)
    requests = merge_request_records(base_requests, vworld_requests)
    targets = load_targets(args.candidate_manifest, args.oreum_registry)
    expected_vworld_targets = [str(target["target_id"]) for target in targets]
    vworld_anchors = parcel_anchors(args.vworld_snapshot, vworld_requests)
    validate_vworld_snapshot(vworld_requests, vworld_anchors, expected_vworld_targets)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_and_copy_raw(
        args.base_snapshot,
        args.output_dir,
        (record for record in base_requests if record.get("source_id") != "vworld_cadastral"),
    )
    verify_and_copy_raw(args.vworld_snapshot, args.output_dir, vworld_requests)

    building_events = json.loads(
        (args.base_snapshot / "building_events.json").read_text(encoding="utf-8")
    )["events"]
    building_coverage = json.loads(
        (args.base_snapshot / "building_coverage.json").read_text(encoding="utf-8")
    )["requests"]
    eia = json.loads((args.base_snapshot / "eia_features.json").read_text(encoding="utf-8"))[
        "features"
    ]
    existing_pnu_records = source_pnu_records(args.pnu_source)
    prior_candidate_anchors = existing_candidate_anchors(existing_pnu_records)
    all_anchors = vworld_anchors + prior_candidate_anchors
    candidates = build_candidate_evidence(
        targets, all_anchors, building_events, eia, requests
    )

    pnu_records = existing_pnu_records + vworld_pnu_records(vworld_anchors)
    links = cross_source_links(pnu_records, building_events)
    context = observation_context(args.output_dir, requests)

    write_json(args.output_dir / "requests.json", {"schema": "kearth-api-requests-v1", "requests": requests})
    write_json(
        args.output_dir / "parcel_anchors.json",
        {
            "schema": "kearth-parcel-anchors-v2",
            "anchors": all_anchors,
            "vworld_anchors": vworld_anchors,
            "prior_candidate_anchors": prior_candidate_anchors,
        },
    )
    write_json(args.output_dir / "building_events.json", {"schema": "kearth-building-events-v1", "events": building_events})
    write_json(
        args.output_dir / "building_coverage.json",
        {"schema": "kearth-building-coverage-v1", "requests": building_coverage},
    )
    write_json(args.output_dir / "eia_features.json", {"schema": "kearth-eia-features-v1", "features": eia})
    write_json(
        args.output_dir / "candidate_evidence.json",
        {"schema": "kearth-candidate-api-evidence-v1", "records": candidates},
    )
    write_json(
        args.output_dir / "cross_source_pnu_links.json",
        {"schema": "kearth-cross-source-pnu-links-v1", "links": links},
    )
    write_json(args.output_dir / "observation_context.json", context)

    vworld_semantic = Counter(
        str(record.get("semantic_status"))
        for record in requests
        if record.get("source_id") == "vworld_cadastral"
    )
    candidate_anchors = [anchor for anchor in vworld_anchors if not str(anchor.get("target_id")).startswith("JJ-OREUM-")]
    oreum_anchors = [anchor for anchor in vworld_anchors if str(anchor.get("target_id")).startswith("JJ-OREUM-")]
    pnus = [str(anchor.get("pnu")) for anchor in vworld_anchors if anchor.get("pnu")]
    pnu_counts = Counter(pnus)
    event_counts = Counter(str(event.get("pnu")) for event in building_events if event.get("pnu"))
    summary = {
        "schema": "kearth-api-snapshot-summary-v3",
        "retrieved_at": json.loads(
            (args.vworld_snapshot / "run_summary.json").read_text(encoding="utf-8")
        )["retrieved_at"],
        "provenance": {
            "base_snapshot": input_snapshot_provenance(args.base_snapshot),
            "vworld_snapshot": input_snapshot_provenance(args.vworld_snapshot),
            "merge_network_requests": 0,
            "merge_script_sha256": sha256(Path(__file__)),
            "source_selection_policy": (
                "retain every non-VWorld response from base; replace base VWorld responses "
                "with the gated fresh snapshot; preserve prior FarmMap and current VWorld "
                "parcel anchors as separate evidence"
            ),
            "superseded_vworld_request_identities": len(
                {
                    str(record.get("request_hash"))
                    for record in base_requests
                    if record.get("source_id") == "vworld_cadastral"
                }
                & {
                    str(record.get("request_hash"))
                    for record in vworld_requests
                    if record.get("source_id") == "vworld_cadastral"
                }
            ),
        },
        **request_summary(requests),
        "scope": {
            "official_oreum_denominator": 368,
            "resolved_oreum_points_available": len(oreum_anchors),
            "olmoearth_candidates_available": len(candidate_anchors),
            "vworld_target_points_requested": len(vworld_anchors),
        },
        "vworld": {
            "semantic_statuses": dict(sorted(vworld_semantic.items())),
            "parcel_anchor_features": sum(int(anchor.get("feature_count") or 0) for anchor in vworld_anchors),
            "candidate_parcel_anchors": sum(int(anchor.get("feature_count") or 0) for anchor in candidate_anchors),
            "oreum_parcel_anchors": sum(int(anchor.get("feature_count") or 0) for anchor in oreum_anchors),
            "unique_pnu": len(pnu_counts),
            "multi_point_parcels": sum(count > 1 for count in pnu_counts.values()),
            "not_found_targets": [
                record.get("target_id")
                for record in vworld_requests
                if record.get("semantic_status") == "api_no_features"
            ],
        },
        "parcel_anchor_features": sum(int(anchor.get("feature_count") or 0) for anchor in vworld_anchors),
        "building_event_rows": len(building_events),
        "eia_feature_rows": len(eia),
        "landcover_tile_rows": len(context["landcover_tiles"]),
        "gk2a_current_grid_values": max(
            (item["grid_value_count"] for item in context["gk2a"]), default=0
        ),
        "candidate_records": len(candidates),
        "candidate_exact_parcel_with_building_event": sum(
            record["exact_parcel_building_event_count"] > 0 for record in candidates
        ),
        "candidate_time_aligned_exact_parcel": sum(
            bool(record["time_aligned_exact_parcel_events"]) for record in candidates
        ),
        "candidate_official_corroboration_b": sum(
            record["causal_evidence_grade"] == "B" for record in candidates
        ),
        "candidate_multi_source_parcel_anchors": sum(
            len(record["parcel_evidence"]) > 1 for record in candidates
        ),
        "candidate_parcel_pnu_conflicts": sum(
            bool(record["parcel_pnu_conflict"]) for record in candidates
        ),
        "oreum_parcel_with_building_event": sum(
            bool(anchor.get("pnu")) and event_counts[str(anchor["pnu"])] > 0
            for anchor in oreum_anchors
        ),
        "cross_source_pnu_population": len(links),
        "cross_source_pnu_with_building_event": sum(
            link["building_event_count"] > 0 for link in links
        ),
        "limits": [
            "VWorld point parcels are representative parcels, not official oreum boundaries.",
            "Several oreum points can share one large forest parcel and are not independent parcel observations.",
            "An exact PNU match is not time-aligned causal evidence unless an event date falls inside the EO observation interval.",
            "VWorld NOT_FOUND is missing point coverage, not evidence that no development occurred.",
            "A request hash identifies normalized request intent, not a response snapshot; retrieved_at and raw SHA identify the response lineage.",
            "Conflicting PNU values from dated FarmMap and current VWorld remain separate evidence until geometry/version differences are resolved.",
            "Historical GK2A dates remain unavailable from the recent-two-day endpoint.",
        ],
    }
    write_json(args.output_dir / "run_summary.json", summary)

    scan_env = {**os.environ, **read_env_file(args.env_file)}
    findings = secret_scan(args.output_dir, scan_env)
    if findings:
        raise RuntimeError(f"credential material detected in merged artifacts: {findings}")
    write_json(
        args.output_dir / "COMPLETE.json",
        {
            "schema": "kearth-api-snapshot-completion-v1",
            "requests_sha256": sha256(args.output_dir / "requests.json"),
            "run_summary_sha256": sha256(args.output_dir / "run_summary.json"),
            "secret_scan_findings": 0,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
