"""Build the leakage-audited Jeju pilot for K-EvidenceShift.

The pilot is deliberately not a train/validation/test benchmark.  It preserves
the current 14 algorithm-selected candidates as an audit pool, separates an
assistant visual pre-annotation from independent ground truth, and records
which public-data fields would leak future information in a prospective run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "k-evidence-shift-jeju-pilot-v0"


@dataclass(frozen=True, slots=True)
class PilotInputs:
    config: Path
    candidate_manifest: Path
    assistant_review: Path
    candidate_evidence: Path
    observation_context: Path
    api_run_summary: Path
    api_requests: Path
    api_complete_marker: Path

    def items(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("config", self.config),
            ("candidate_manifest", self.candidate_manifest),
            ("assistant_review", self.assistant_review),
            ("candidate_evidence", self.candidate_evidence),
            ("observation_context", self.observation_context),
            ("api_run_summary", self.api_run_summary),
            ("api_requests", self.api_requests),
            ("api_complete_marker", self.api_complete_marker),
        )


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _parse_iso_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _haversine_m(left: dict[str, Any], right: dict[str, Any]) -> float:
    radius_m = 6_371_008.8
    lat1 = math.radians(float(left["lat"]))
    lat2 = math.radians(float(right["lat"]))
    delta_lat = lat2 - lat1
    delta_lon = math.radians(float(right["lon"]) - float(left["lon"]))
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    return 2 * radius_m * math.asin(math.sqrt(value))


def _spatial_groups(candidates: list[dict[str, Any]], buffer_m: float) -> tuple[dict[str, str], list[dict[str, Any]]]:
    parent = {candidate["candidate_id"]: candidate["candidate_id"] for candidate in candidates}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    close_pairs: list[dict[str, Any]] = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            distance_m = _haversine_m(left, right)
            if distance_m <= buffer_m:
                union(left["candidate_id"], right["candidate_id"])
                close_pairs.append(
                    {
                        "left": left["candidate_id"],
                        "right": right["candidate_id"],
                        "distance_m": round(distance_m, 3),
                    }
                )

    members: dict[str, list[str]] = defaultdict(list)
    for candidate_id in sorted(parent):
        members[find(candidate_id)].append(candidate_id)
    group_id_by_candidate: dict[str, str] = {}
    for member_ids in sorted(members.values()):
        token = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()[:12]
        group_id = f"jeju-spatial-{token}"
        for candidate_id in member_ids:
            group_id_by_candidate[candidate_id] = group_id
    return group_id_by_candidate, sorted(close_pairs, key=lambda value: (value["left"], value["right"]))


def _shared_value_groups(
    candidates: list[dict[str, Any]], values_for, prefix: str
) -> dict[str, str]:
    """Return deterministic connected components for shared source identities."""
    parent = {candidate["candidate_id"]: candidate["candidate_id"] for candidate in candidates}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    value_sets = {
        candidate["candidate_id"]: set(values_for(candidate)) for candidate in candidates
    }
    for index, left in enumerate(candidates):
        left_id = left["candidate_id"]
        for right in candidates[index + 1 :]:
            right_id = right["candidate_id"]
            if value_sets[left_id] & value_sets[right_id]:
                union(left_id, right_id)

    members: dict[str, list[str]] = defaultdict(list)
    for candidate_id in sorted(parent):
        members[find(candidate_id)].append(candidate_id)
    result: dict[str, str] = {}
    for member_ids in sorted(members.values()):
        token = hashlib.sha256("\n".join(member_ids).encode("utf-8")).hexdigest()[:12]
        for candidate_id in member_ids:
            result[candidate_id] = f"{prefix}-{token}"
    return result


def _window_tile_ids(candidate: dict[str, Any]) -> set[str]:
    result = set()
    for observation in candidate["season_aligned_rgb"].values():
        parts = str(observation["window"]).rsplit("_", maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"invalid window name: {observation['window']!r}")
        result.add(f"{parts[-2]}_{parts[-1]}")
    return result


def _source_item_ids(candidate: dict[str, Any]) -> set[str]:
    return {
        item
        for observation in candidate["season_aligned_rgb"].values()
        for item in observation["source_items"]
    }


def _transition_dates(candidate: dict[str, Any]) -> tuple[str, str]:
    raw = str(candidate["algorithm"]["when"])
    try:
        left_year, right_year = raw.split("->", maxsplit=1)
        observations = candidate["season_aligned_rgb"]
        return observations[left_year]["acquisition_date"], observations[right_year]["acquisition_date"]
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid transition for {candidate['candidate_id']}: {raw!r}") from error


def _cloud_stratum(value: float | None, config: dict[str, Any]) -> str:
    if value is None:
        return "unknown"
    if value <= float(config["clear_max"]):
        return "clear_proxy"
    if value <= float(config["mixed_max"]):
        return "mixed_proxy"
    return "high_proxy"


def _visual_label(review: dict[str, Any]) -> str:
    return {
        "yes": "change_preannotation",
        "no": "no_change_preannotation",
        "uncertain": "uncertain_preannotation",
    }[review["is_persistent_change"]]


def _compact_date(value: Any) -> date | None:
    raw = str(value or "").strip().replace("-", "").replace("/", "").replace(".", "")
    if len(raw) != 8 or not raw.isdigit():
        return None
    try:
        return datetime.strptime(raw, "%Y%m%d").date()
    except ValueError:
        return None


def transition_aligned_exact_events(
    evidence: dict[str, Any], t0: str, t1: str
) -> list[dict[str, Any]]:
    """Recompute transition alignment instead of trusting whole-series upstream flags."""
    interval_start, interval_end = date.fromisoformat(t0), date.fromisoformat(t1)
    parcel_pnus = set(evidence.get("parcel_pnu_values", []))
    fields = (
        "permit_date",
        "construction_start_date",
        "use_approval_date",
        "created_date",
    )
    aligned = []
    for event in evidence.get("same_legal_dong_building_events", []):
        if event.get("pnu") not in parcel_pnus:
            continue
        event_dates = [parsed for field in fields if (parsed := _compact_date(event.get(field)))]
        if any(interval_start <= event_date <= interval_end for event_date in event_dates):
            aligned.append(event)
    return aligned


def transition_aligned_eia_matches(
    evidence: dict[str, Any], t0: str, t1: str
) -> list[dict[str, Any]]:
    """Require an explicit event date before EIA overlap can support a transition."""
    interval_start, interval_end = date.fromisoformat(t0), date.fromisoformat(t1)
    fields = ("approval_date", "start_date", "event_date", "construction_start_date")
    aligned = []
    for match in evidence.get("eia_polygon_matches", []):
        event_dates = [parsed for field in fields if (parsed := _compact_date(match.get(field)))]
        if any(interval_start <= event_date <= interval_end for event_date in event_dates):
            aligned.append(match)
    return aligned


def _validate_sources(
    candidate_manifest: dict[str, Any],
    assistant_review: dict[str, Any],
    candidate_evidence: dict[str, Any],
    observation_context: dict[str, Any],
    api_run_summary: dict[str, Any],
    api_complete: dict[str, Any],
) -> None:
    candidates = candidate_manifest.get("candidates", [])
    candidate_ids = [value.get("candidate_id") for value in candidates]
    if len(candidate_ids) != 14 or len(set(candidate_ids)) != 14:
        raise ValueError("the v0 pilot requires exactly 14 unique fixed candidates")
    if set(candidate_ids) != set(assistant_review.get("reviews", {})):
        raise ValueError("assistant review IDs do not match the fixed candidate manifest")
    if assistant_review.get("manifest_sha256") != candidate_manifest.get("provenance", {}).get(
        "manifest_sha256"
    ):
        raise ValueError("assistant review does not reference the candidate manifest protocol hash")
    manifest_for_hash = json.loads(json.dumps(candidate_manifest))
    expected_protocol_hash = manifest_for_hash["provenance"]["manifest_sha256"]
    manifest_for_hash["provenance"]["manifest_sha256"] = None
    computed_protocol_hash = hashlib.sha256(
        json.dumps(manifest_for_hash, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if expected_protocol_hash != computed_protocol_hash:
        raise ValueError("candidate manifest protocol hash does not recompute")
    evidence_ids = [value.get("target_id") for value in candidate_evidence.get("records", [])]
    if set(candidate_ids) != set(evidence_ids) or len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("API evidence IDs do not match the fixed candidate manifest")
    evidence_by_id = {value["target_id"]: value for value in candidate_evidence["records"]}
    for candidate in candidates:
        evidence = evidence_by_id[candidate["candidate_id"]]
        if (float(candidate["lat"]), float(candidate["lon"])) != (
            float(evidence["lat"]),
            float(evidence["lon"]),
        ):
            raise ValueError(f"coordinate drift for {candidate['candidate_id']}")
        candidate_dates = sorted(
            observation["acquisition_date"]
            for observation in candidate["season_aligned_rgb"].values()
        )
        if candidate_dates != sorted(evidence["observation_dates"]):
            raise ValueError(f"observation-date drift for {candidate['candidate_id']}")
    landcover_counts = Counter(value.get("target_id") for value in observation_context.get("landcover_tiles", []))
    if any(landcover_counts[candidate_id] != 3 for candidate_id in candidate_ids):
        raise ValueError("each pilot candidate must have exactly three annual land-cover responses")
    landcover_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in observation_context["landcover_tiles"]:
        landcover_by_id[value["target_id"]].append(value)
    for candidate_id in candidate_ids:
        values = landcover_by_id[candidate_id]
        if {value["year"] for value in values} != {2023, 2024, 2025}:
            raise ValueError(f"land-cover years are incomplete for {candidate_id}")
        if any(value["semantic_status"] != "api_success" for value in values):
            raise ValueError(f"land-cover request failed for {candidate_id}")
    if api_run_summary.get("schema") != "kearth-api-snapshot-summary-v3":
        raise ValueError("the pilot requires the canonical v3 API snapshot")
    if api_run_summary.get("candidate_records") != len(candidate_ids):
        raise ValueError("API run summary candidate count does not match")
    semantic_total = sum(api_run_summary.get("semantic_statuses", {}).values())
    if semantic_total != api_run_summary.get("request_count"):
        raise ValueError("API semantic status counts do not cover every request")
    if api_run_summary.get("outcomes", {}).get("http_success") != api_run_summary.get(
        "request_count"
    ):
        raise ValueError("the canonical pilot snapshot must have complete HTTP coverage")
    if api_complete.get("schema") != "kearth-api-snapshot-completion-v1":
        raise ValueError("API completion marker is missing or has the wrong schema")
    if api_complete.get("secret_scan_findings") != 0:
        raise ValueError("API snapshot completion marker reports secret findings")


def _source_manifest(inputs: PilotInputs) -> dict[str, Any]:
    records = [
        {
            "source_id": name,
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for name, path in inputs.items()
    ]
    builder_path = Path(__file__).resolve()
    return {
        "schema": "k-evidence-shift-source-manifest-v1",
        "sources": records,
        "builder": {
            "logical_id": "code/kearth_benchmark/pilot.py",
            "path": builder_path.as_posix(),
            "sha256": sha256_file(builder_path),
            "bytes": builder_path.stat().st_size,
        },
    }


def build_pilot(inputs: PilotInputs) -> dict[str, Any]:
    config = read_json(inputs.config)
    candidate_manifest = read_json(inputs.candidate_manifest)
    assistant_review = read_json(inputs.assistant_review)
    candidate_evidence = read_json(inputs.candidate_evidence)
    observation_context = read_json(inputs.observation_context)
    api_run_summary = read_json(inputs.api_run_summary)
    api_complete = read_json(inputs.api_complete_marker)
    if sha256_file(inputs.api_run_summary) != api_complete.get("run_summary_sha256"):
        raise ValueError("API run summary SHA does not match COMPLETE.json")
    if sha256_file(inputs.api_requests) != api_complete.get("requests_sha256"):
        raise ValueError("API requests SHA does not match COMPLETE.json")
    _validate_sources(
        candidate_manifest,
        assistant_review,
        candidate_evidence,
        observation_context,
        api_run_summary,
        api_complete,
    )

    candidates = sorted(candidate_manifest["candidates"], key=lambda value: value["candidate_id"])
    reviews = assistant_review["reviews"]
    evidence_by_id = {value["target_id"]: value for value in candidate_evidence["records"]}
    landcover_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for value in observation_context["landcover_tiles"]:
        landcover_by_id[value["target_id"]].append(value)

    group_ids, close_pairs = _spatial_groups(candidates, float(config["spatial_group_buffer_m"]))
    window_group_ids = _shared_value_groups(candidates, _window_tile_ids, "jeju-window")
    scene_group_ids = _shared_value_groups(candidates, _source_item_ids, "jeju-scene")
    api_retrieved_at = _parse_iso_datetime(api_run_summary["retrieved_at"])
    gk2a_historical = [value for value in observation_context["gk2a"] if "-" in str(value["target_time"])]
    gk2a_current = [value for value in observation_context["gk2a"] if "-" not in str(value["target_time"])]

    records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        review = reviews[candidate_id]
        evidence = evidence_by_id[candidate_id]
        t0, t1 = _transition_dates(candidate)
        observations = []
        for year, observation in sorted(candidate["season_aligned_rgb"].items()):
            acquisition_date = observation["acquisition_date"]
            if acquisition_date < t0:
                temporal_role = "history_before_t0"
            elif acquisition_date <= t1:
                temporal_role = "transition_input"
            else:
                temporal_role = "future_after_t1_review_only"
            observations.append(
                {
                    "year": int(year),
                    "acquisition_date": acquisition_date,
                    "temporal_role": temporal_role,
                    "prospective_input_eligible": acquisition_date <= t1,
                    "source_items": observation["source_items"],
                    "window": observation["window"],
                    "context_cloud_proxy": observation["context_metrics"]["cloud_proxy"],
                    "detail_cloud_proxy": observation["detail_metrics"]["cloud_proxy"],
                }
            )
        event_observations = [value for value in observations if value["acquisition_date"] in {t0, t1}]
        event_cloud_proxy = max(
            value
            for observation in event_observations
            for value in (observation["context_cloud_proxy"], observation["detail_cloud_proxy"])
        )
        time_aligned_events = transition_aligned_exact_events(evidence, t0, t1)
        eia_matches = transition_aligned_eia_matches(evidence, t0, t1)
        supported = bool(time_aligned_events or eia_matches)
        retrieval_is_post_t1 = api_retrieved_at.date() > date.fromisoformat(t1)
        landcover = sorted(landcover_by_id[candidate_id], key=lambda value: value["year"])
        records.append(
            {
                "schema": "k-evidence-shift-site-event-v1",
                "site_event_id": f"{candidate_id}__{t0}__{t1}",
                "candidate_id": candidate_id,
                "site_group_id": group_ids[candidate_id],
                "spatial_window_group_id": window_group_ids[candidate_id],
                "source_scene_component_id": scene_group_ids[candidate_id],
                "split": "pilot_audit_pool",
                "split_eligible_for_metrics": False,
                "region": {"id": config["region_id"], "name": config["region_name"]},
                "location": {"lat": candidate["lat"], "lon": candidate["lon"]},
                "cohort": candidate["cohort"],
                "selection_contract": {
                    "regime": "deterministic_top_rank_and_coordinate_cohort",
                    "eligible_for_prevalence_estimation": False,
                    "inclusion_probability": None,
                    "selection_fields_allowed_as_model_features": False,
                },
                "detected_transition": {
                    "t0": t0,
                    "t1": t1,
                    "ranking_source": candidate["algorithm"]["source"],
                    "ranking_position": candidate["algorithm"]["rank"],
                    "anomaly_z": candidate["algorithm"]["z"],
                    "landcover_class_at_ranking": candidate["algorithm"]["landcover"],
                },
                "observations": observations,
                "label_axes": {
                    "visual_change": {
                        "value": _visual_label(review),
                        "source": "assistant_visual_preannotation",
                        "confidence": review["confidence"],
                        "review_category": review["label"],
                        "eligible_as_ground_truth": False,
                        "temporal_use": "retrospective_preannotation",
                        "uses_post_t1_observations": any(
                            value["temporal_role"] == "future_after_t1_review_only"
                            for value in observations
                        ),
                        "eligible_for_prospective_evaluation": False,
                    },
                    "official_event_supported": {
                        "value": "supported" if supported else "not_observed",
                        "time_aligned_exact_parcel_event_count": len(time_aligned_events),
                        "time_aligned_eia_event_count": len(eia_matches),
                        "upstream_whole_series_evidence_grade": evidence[
                            "causal_evidence_grade"
                        ],
                        "absence_is_negative_label": False,
                        "causal_claim_allowed": False,
                    },
                    "evidence_available": {
                        "current_cadastral_anchor": evidence.get(
                            "current_vworld_parcel", {}
                        ).get("api_status")
                        == "OK",
                        "annual_landcover_responses": len(landcover),
                        "historical_gk2a_successes": sum(
                            value["semantic_status"] == "api_success"
                            for value in gk2a_historical
                        ),
                        "historical_gk2a_requests": len(gk2a_historical),
                        "same_legal_dong_event_count_context_only": evidence[
                            "same_legal_dong_building_event_count"
                        ],
                        "direct_eia_overlap_count": len(eia_matches),
                        "source_states": {
                            "vworld_cadastral": (
                                "available_current_post_t1"
                                if retrieval_is_post_t1
                                else "available_publication_time_unfrozen"
                            ),
                            "building_hub": (
                                "time_aligned_event"
                                if time_aligned_events
                                else "event_outside_transition"
                                if evidence["exact_parcel_building_event_count"]
                                else "no_exact_match_not_interpretable_as_negative"
                            ),
                            "environmental_impact_assessment": (
                                "time_aligned_event"
                                if eia_matches
                                else "spatial_context_without_transition_date"
                                if evidence.get("eia_polygon_matches")
                                else "no_overlap_not_interpretable_as_negative"
                            ),
                            "annual_landcover": "available_state_layer_publication_time_unfrozen",
                            "gk2a_historical": "api_error_outside_endpoint_retention_window",
                        },
                    },
                    "cause": {
                        "value": None,
                        "status": "unknown_abstain",
                        "eligible_as_ground_truth": False,
                    },
                },
                "strata": {
                    "cloud_proxy_value": event_cloud_proxy,
                    "cloud_proxy_stratum": _cloud_stratum(event_cloud_proxy, config["cloud_proxy"]),
                    "cloud_measurement": "sentinel2_rgb_proxy_only",
                    "historical_gk2a": "unavailable_endpoint_window",
                    "parcel_pnu_relation": evidence["parcel_pnu_relation"],
                    "official_support": "supported" if supported else "not_observed",
                    "evidence_coverage": "incomplete_for_event_attribution",
                },
                "public_evidence": {
                    "representative_pnu": evidence.get("current_vworld_parcel", {}).get("pnu"),
                    "parcel_group_ids": [
                        f"pnu-{value}"
                        for value in sorted(set(evidence.get("parcel_pnu_values", [])))
                    ],
                    "parcel_anchor_count": len(evidence.get("parcel_evidence", [])),
                    "exact_parcel_event_count_any_time": evidence["exact_parcel_building_event_count"],
                    "time_aligned_exact_parcel_event_count": len(time_aligned_events),
                    "direct_eia_overlap_count_any_time": len(
                        evidence.get("eia_polygon_matches", [])
                    ),
                    "time_aligned_eia_event_count": len(eia_matches),
                    "upstream_whole_series_time_aligned_event_count": len(
                        evidence.get("time_aligned_exact_parcel_events", [])
                    ),
                    "landcover_years": [value["year"] for value in landcover],
                    "api_snapshot_retrieved_at": api_run_summary["retrieved_at"],
                    "prospective_input_eligible": False,
                    "prospective_exclusion_reason": (
                        "snapshot_retrieved_after_t1_and_source_publication_times_not_frozen"
                        if retrieval_is_post_t1
                        else "source_publication_times_not_frozen"
                    ),
                },
                "decision": {
                    "value": "abstain",
                    "reason": "no_independent_visual_ground_truth_or_time_aligned_official_event_corroboration",
                    "causal_claim_allowed": False,
                },
            }
        )

    source_manifest = _source_manifest(inputs)
    digest_payload = {
        "schema": SCHEMA,
        "sources": [
            {
                "source_id": value["source_id"],
                "sha256": value["sha256"],
                "bytes": value["bytes"],
            }
            for value in source_manifest["sources"]
        ],
        "builder": {
            "logical_id": source_manifest["builder"]["logical_id"],
            "sha256": source_manifest["builder"]["sha256"],
        },
    }
    input_digest = hashlib.sha256(canonical_bytes(digest_payload)).hexdigest()
    return {
        "schema": SCHEMA,
        "benchmark_id": config["benchmark_id"],
        "build_id": input_digest[:16],
        "status": "pilot_audit_only_not_publishable_benchmark",
        "research_route": "selective_change_detection_with_abstention",
        "records": records,
        "source_manifest": source_manifest,
        "close_pairs": close_pairs,
        "spatial_group_buffer_m": config["spatial_group_buffer_m"],
        "cloud_proxy_contract": config["cloud_proxy"],
        "gk2a_current_snapshot": {
            "available": bool(gk2a_current and gk2a_current[0]["semantic_status"] == "api_success"),
            "grid_value_count": gk2a_current[0]["grid_value_count"] if gk2a_current else 0,
            "eligible_for_historical_pairing": False,
        },
        "promotion_gate_config": config["promotion_gates"],
        "paired_input_contract": config["paired_input_contract"],
        "model_matrix_config": config["model_matrix"],
    }


def _counts_by(records: Iterable[dict[str, Any]], key) -> dict[str, int]:
    return dict(sorted(Counter(key(record) for record in records).items()))


def _unique_site_label_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        grouped[record["site_group_id"]].add(
            record["label_axes"]["visual_change"]["value"]
        )
    collapsed = [next(iter(values)) if len(values) == 1 else "conflicting_preannotation" for values in grouped.values()]
    return dict(sorted(Counter(collapsed).items()))


def _high_confidence_change_site_count(records: Iterable[dict[str, Any]]) -> int:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["site_group_id"]].append(record)
    return sum(
        all(
            item["label_axes"]["visual_change"]["value"] == "change_preannotation"
            and item["label_axes"]["visual_change"]["confidence"] == "high"
            for item in items
        )
        for items in grouped.values()
    )


def _summary(pilot: dict[str, Any]) -> dict[str, Any]:
    records = pilot["records"]
    return {
        "schema": "k-evidence-shift-pilot-summary-v1",
        "benchmark_id": pilot["benchmark_id"],
        "build_id": pilot["build_id"],
        "status": pilot["status"],
        "candidate_records": len(records),
        "unique_spatial_groups_500m": len({value["site_group_id"] for value in records}),
        "unique_source_window_groups": len(
            {value["spatial_window_group_id"] for value in records}
        ),
        "unique_source_scene_components": len(
            {value["source_scene_component_id"] for value in records}
        ),
        "independent_human_ground_truth_labels": 0,
        "assistant_visual_preannotations": len(records),
        "official_event_supported": sum(
            value["label_axes"]["official_event_supported"]["value"] == "supported" for value in records
        ),
        "cause_labels": sum(value["label_axes"]["cause"]["value"] is not None for value in records),
        "abstentions": sum(value["decision"]["value"] == "abstain" for value in records),
        "pnu_conflicts": sum(value["strata"]["parcel_pnu_relation"] == "conflict" for value in records),
        "post_t1_public_evidence_records": sum(not value["public_evidence"]["prospective_input_eligible"] for value in records),
        "cloud_proxy_strata": _counts_by(records, lambda value: value["strata"]["cloud_proxy_stratum"]),
        "visual_preannotation_values_by_record": _counts_by(
            records, lambda value: value["label_axes"]["visual_change"]["value"]
        ),
        "unique_site_visual_preannotation_values": _unique_site_label_counts(records),
        "high_confidence_change_preannotation_unique_sites": _high_confidence_change_site_count(
            records
        ),
        "warning": "No accuracy, transfer gain, calibration, or causal claim is estimable from this pilot.",
    }


def _leakage_audit(pilot: dict[str, Any]) -> dict[str, Any]:
    records = pilot["records"]

    def split_violations(group_field: str) -> dict[str, list[str]]:
        group_to_split: dict[str, set[str]] = defaultdict(set)
        for record in records:
            group_to_split[record[group_field]].add(record["split"])
        return {
            key: sorted(value) for key, value in group_to_split.items() if len(value) > 1
        }

    site_group_violations = split_violations("site_group_id")
    window_group_violations = split_violations("spatial_window_group_id")
    scene_group_violations = split_violations("source_scene_component_id")
    parcel_to_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for parcel_group_id in record["public_evidence"]["parcel_group_ids"]:
            parcel_to_splits[parcel_group_id].add(record["split"])
    parcel_group_violations = {
        key: sorted(value) for key, value in parcel_to_splits.items() if len(value) > 1
    }
    assistant_as_gt = [
        value["candidate_id"]
        for value in records
        if value["label_axes"]["visual_change"]["eligible_as_ground_truth"]
    ]
    prospective_evidence = [
        value["candidate_id"] for value in records if value["public_evidence"]["prospective_input_eligible"]
    ]
    post_t1_eo_enabled = [
        record["candidate_id"]
        for record in records
        if any(
            observation["acquisition_date"] > record["detected_transition"]["t1"]
            and observation["prospective_input_eligible"]
            for observation in record["observations"]
        )
    ]
    selection_fields_enabled = [
        record["candidate_id"]
        for record in records
        if record["selection_contract"]["selection_fields_allowed_as_model_features"]
    ]
    return {
        "schema": "k-evidence-shift-leakage-audit-v1",
        "benchmark_id": pilot["benchmark_id"],
        "spatial_buffer_m": pilot["spatial_group_buffer_m"],
        "close_pairs": pilot["close_pairs"],
        "site_group_split_violations": site_group_violations,
        "source_window_group_split_violations": window_group_violations,
        "source_scene_component_split_violations": scene_group_violations,
        "parcel_group_split_violations": parcel_group_violations,
        "scene_disjoint_quality_split_possible": len(
            {record["source_scene_component_id"] for record in records}
        )
        > 1,
        "selection_fields_enabled_as_model_features": selection_fields_enabled,
        "assistant_preannotations_misused_as_ground_truth": assistant_as_gt,
        "post_t1_or_unfrozen_public_evidence_enabled_as_input": prospective_evidence,
        "post_t1_eo_enabled_as_transition_input": post_t1_eo_enabled,
        "sealed_test_created": False,
        "result": "pass_for_audit_pool_not_a_train_test_split"
        if not site_group_violations
        and not window_group_violations
        and not scene_group_violations
        and not parcel_group_violations
        and not assistant_as_gt
        and not prospective_evidence
        and not post_t1_eo_enabled
        and not selection_fields_enabled
        else "fail",
    }


def _promotion_gates(pilot: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    required = pilot["promotion_gate_config"]
    observed_years = {
        observation["year"] for record in pilot["records"] for observation in record["observations"]
    }
    values = {
        "independent_regions": 1,
        "independent_human_labels": summary["independent_human_ground_truth_labels"],
        "double_reviewed_labels": 0,
        "independent_spatial_groups": summary["unique_spatial_groups_500m"],
        "completed_paired_baselines": 0,
        "pinned_model_checkpoints": 0,
        "observation_years": len(observed_years),
        "sealed_probability_test": False,
        "frozen_paired_input_contract": bool(
            pilot["paired_input_contract"]["frozen"]
            and pilot["paired_input_contract"]["status"] == "frozen"
        ),
    }
    gates = {
        "independent_regions": {
            "observed": values["independent_regions"],
            "required": required["minimum_independent_regions"],
        },
        "independent_human_labels": {
            "observed": values["independent_human_labels"],
            "required": required["minimum_independent_human_labels"],
        },
        "double_reviewed_labels": {
            "observed": values["double_reviewed_labels"],
            "required": required["minimum_double_reviewed_labels"],
        },
        "independent_spatial_groups": {
            "observed": values["independent_spatial_groups"],
            "required": required["minimum_independent_spatial_groups"],
        },
        "completed_paired_baselines": {
            "observed": values["completed_paired_baselines"],
            "required": required["minimum_completed_paired_baselines"],
        },
        "pinned_model_checkpoints": {
            "observed": values["pinned_model_checkpoints"],
            "required": required["minimum_pinned_model_checkpoints"],
        },
        "observation_years": {
            "observed": values["observation_years"],
            "required": required["minimum_observation_years"],
        },
        "sealed_probability_test": {
            "observed": values["sealed_probability_test"],
            "required": required["require_sealed_probability_test"],
        },
        "frozen_paired_input_contract": {
            "observed": values["frozen_paired_input_contract"],
            "required": required["require_frozen_paired_input_contract"],
        },
    }
    for gate in gates.values():
        if isinstance(gate["required"], bool):
            gate["pass"] = gate["observed"] is gate["required"]
        else:
            gate["pass"] = gate["observed"] >= gate["required"]
    return {
        "schema": "k-evidence-shift-promotion-gates-v1",
        "benchmark_id": pilot["benchmark_id"],
        "gates": gates,
        "cvpr_experiment_ready": all(value["pass"] for value in gates.values()),
        "federated_learning_status": "hold_until_three_or_more_real_non_exportable_institutional_silos",
        "cause_research_status": "not_supported_route_to_selective_change_detection",
    }


def _model_rows(pilot: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for model in pilot["model_matrix_config"]:
        rows.append(
            {
                **model,
                "decoder_contract": "same_task_head_within_paired_track",
                "compute_contract": "matched_trainable_parameters_and_search_budget_within_paired_track",
                "input_contract_status": pilot["paired_input_contract"]["status"],
                "status": "not_run",
                "blocked_by": "independent_ground_truth_sealed_split_and_frozen_input_contract_missing",
            }
        )
    return rows


def _data_card(summary: dict[str, Any], leakage: dict[str, Any], gates: dict[str, Any]) -> str:
    return f"""# K-EvidenceShift Jeju pilot v0 data card

## 판정

이 산출물은 **벤치마크가 아니라 누수 검사를 통과한 audit pilot**이다. 현재 14개 후보에는 독립 인간 정답이 0개이므로 정확도·전이효과·원인 규명 성능을 보고하면 안 된다.

## 현재 범위

- 후보 레코드: {summary['candidate_records']}
- 500 m 공간 그룹: {summary['unique_spatial_groups_500m']}
- 공유 materialized-window 그룹: {summary['unique_source_window_groups']}
- assistant 시각 pre-annotation: {summary['assistant_visual_preannotations']} (ground truth 사용 금지)
- 독립 인간 정답: {summary['independent_human_ground_truth_labels']}
- 시점 정렬 공식 사건 보강 근거: {summary['official_event_supported']} (인과 주장 불가)
- 보류: {summary['abstentions']}/{summary['candidate_records']}
- PNU 출처 충돌: {summary['pnu_conflicts']}

## 누수 계약

- 같은 500 m 공간 그룹은 향후 서로 다른 split에 둘 수 없다.
- 같은 rslearn materialized window를 공유하는 후보도 서로 다른 split에 둘 수 없다.
- 현재 scene graph가 연결되어 cloud/quality task의 scene-disjoint split은 만들 수 없다.
- API snapshot은 마지막 EO 관측 뒤에 수집됐고 공개시점이 동결되지 않았으므로 prospective model input에서 제외한다.
- 행정자료 no-match는 음성 원인 라벨이 아니다.
- 현재 GK2A 2 km grid는 수집됐지만 과거 6시점 조회는 API 제한으로 실패했다. Sentinel-2 B02 임계값은 cloud proxy일 뿐 한국형 구름 계측값이 아니다.
- sealed probability test와 이중 판독 세트는 아직 만들지 않았다.
- 다중 모델용 common Sentinel-2 입력 계약도 아직 동결되지 않았다. P0의 OlmoEarth 릴리스 감사는 별도의 exact-input 계약을 사용하므로 전이 성능표가 아니다.

누수 검사 결과: `{leakage['result']}`

## 승격 조건

CVPR 실험 준비 상태: `{gates['cvpr_experiment_ready']}`. 최소 3개 독립 지역, 300개 독립 블라인드 판독, 그중 120개 이중 판독, sealed probability test, 동결된 paired-input 계약, 4개 paired-input baseline 완료가 필요하다. 연합학습은 실제 원자료 반출 불가 기관 사일로가 3곳 이상일 때만 승격한다.
"""


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_bytes(value))


def write_pilot(pilot: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    allowed_outputs = {
        "pilot_manifest.json",
        "site_events.jsonl",
        "run_summary.json",
        "leakage_audit.json",
        "promotion_gates.json",
        "source_manifest.json",
        "model_matrix.csv",
        "DATA_CARD.md",
        "sha256_manifest.json",
        "COMPLETE.json",
    }
    if output_dir.exists():
        unexpected = sorted(
            path.name for path in output_dir.iterdir() if path.name not in allowed_outputs
        )
        if unexpected:
            raise ValueError(f"output directory contains unexpected stale files: {unexpected}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary(pilot)
    leakage = _leakage_audit(pilot)
    gates = _promotion_gates(pilot, summary)
    model_rows = _model_rows(pilot)

    _write_json(output_dir / "pilot_manifest.json", {key: value for key, value in pilot.items() if key != "model_matrix_config"})
    with (output_dir / "site_events.jsonl").open("w", encoding="utf-8") as output:
        for record in pilot["records"]:
            output.write(canonical_bytes(record).decode("utf-8"))
    _write_json(output_dir / "run_summary.json", summary)
    _write_json(output_dir / "leakage_audit.json", leakage)
    _write_json(output_dir / "promotion_gates.json", gates)
    _write_json(output_dir / "source_manifest.json", pilot["source_manifest"])
    with (output_dir / "model_matrix.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(model_rows[0]))
        writer.writeheader()
        writer.writerows(model_rows)
    (output_dir / "DATA_CARD.md").write_text(_data_card(summary, leakage, gates), encoding="utf-8")

    produced = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path.name not in {"sha256_manifest.json", "COMPLETE.json"}
    )
    sha_manifest = {
        "schema": "k-evidence-shift-output-sha256-v1",
        "files": [
            {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in produced
        ],
    }
    _write_json(output_dir / "sha256_manifest.json", sha_manifest)
    complete = {
        "schema": "k-evidence-shift-output-completion-v1",
        "build_id": pilot["build_id"],
        "sha256_manifest_sha256": sha256_file(output_dir / "sha256_manifest.json"),
        "file_count": len(sha_manifest["files"]),
    }
    _write_json(output_dir / "COMPLETE.json", complete)
    return {"summary": summary, "leakage": leakage, "gates": gates, "sha256": sha_manifest}
