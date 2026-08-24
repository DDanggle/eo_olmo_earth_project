#!/usr/bin/env python3
"""Build deterministic manifests and evidence edges for K-Earth public data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from kearth_public.canonical import file_sha256, write_json
from kearth_public.csv_sources import (
    audit_development_permits,
    ingest_jeju_forest_use,
    write_csv,
)
from kearth_public.farmmap import ingest_farmmap, load_targets


FOREST_CATALOG_URL = "https://www.data.go.kr/data/15056266/fileData.do"
FOREST_DOWNLOAD_URL = (
    "https://www.data.go.kr/cmm/cmm/fileDownload.do?"
    "atchFileId=FILE_000000003680509&fileDetailSn=1&insertDataPrcus=N"
)
FARMMAP_CATALOG_URL = "https://www.data.go.kr/data/15104491/fileData.do"
FARMMAP_DOWNLOAD_URL = (
    "https://www.data.go.kr/cmm/cmm/fileDownload.do?"
    "atchFileId=FILE_000000003642465&fileDetailSn=1&insertDataPrcus=N"
)


def write_edges(path: Path, edges: list[dict[str, Any]]) -> None:
    write_json(
        path,
        {
            "schema": "kearth-evidence-edge-collection-v1",
            "count": len(edges),
            "edges": edges,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forest-use-csv", type=Path, required=True)
    parser.add_argument("--farmmap-zip", type=Path, required=True)
    parser.add_argument("--development-permits-csv", type=Path, required=True)
    parser.add_argument("--candidate-context-json", type=Path, required=True)
    parser.add_argument("--candidate-manifest-json", type=Path, required=True)
    parser.add_argument("--oreum-registry-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--retrieved-at",
        required=True,
        help="Fixed ISO-8601 retrieval time; required to keep manifests reproducible",
    )
    args = parser.parse_args()
    inputs = [
        args.forest_use_csv,
        args.farmmap_zip,
        args.development_permits_csv,
        args.candidate_context_json,
        args.candidate_manifest_json,
        args.oreum_registry_json,
    ]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        parser.error(f"missing input files: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    forest = ingest_jeju_forest_use(
        args.forest_use_csv,
        catalog_url=FOREST_CATALOG_URL,
        download_url=FOREST_DOWNLOAD_URL,
        retrieved_at=args.retrieved_at,
    )
    forest_csv = args.out_dir / "jeju_forest_use_2008_2026.csv"
    forest_columns = [
        "year",
        "approval_count",
        "approved_area_ha",
        "snapshot_date",
    ]
    write_csv(forest_csv, forest.rows, forest_columns)
    forest_manifest = args.out_dir / "jeju_forest_use_manifest.json"
    write_json(forest_manifest, forest.manifest.to_dict())

    permits = audit_development_permits(args.development_permits_csv)
    permit_audit = args.out_dir / "development_permit_coverage_audit.json"
    write_json(
        permit_audit,
        {
            "schema": "kearth-coverage-audit-v1",
            "source_id": "molit_dev_permit",
            "raw_file_name": args.development_permits_csv.name,
            "raw_sha256": file_sha256(args.development_permits_csv),
            "quality": permits.quality,
        },
    )

    targets = load_targets(
        args.candidate_context_json,
        args.candidate_manifest_json,
        args.oreum_registry_json,
    )
    farmmap = ingest_farmmap(
        args.farmmap_zip,
        targets=targets,
        permit_rows_by_pnu=permits.pnu_to_rows,
        catalog_url=FARMMAP_CATALOG_URL,
        download_url=FARMMAP_DOWNLOAD_URL,
        retrieved_at=args.retrieved_at,
    )
    farmmap_manifest = args.out_dir / "jeju_farmmap_manifest.json"
    write_json(farmmap_manifest, farmmap.manifest.to_dict())
    edge_path = args.out_dir / "farmmap_evidence_edges.json"
    write_edges(edge_path, [edge.to_dict() for edge in farmmap.evidence_edges])
    target_summary_path = args.out_dir / "farmmap_target_summary.json"
    write_json(target_summary_path, farmmap.target_summary)
    permit_links_path = args.out_dir / "farmmap_permit_pnu_links.csv"
    permit_link_columns = [
        "farm_id",
        "pnu",
        "farm_class",
        "farm_address",
        "farm_flight_date",
        "farm_update_date",
        "permit_date",
        "permit_action",
        "permit_purpose",
        "relation",
        "causal_claim_allowed",
    ]
    if farmmap.permit_links:
        write_csv(permit_links_path, farmmap.permit_links, permit_link_columns)
    else:
        with permit_links_path.open("w", encoding="utf-8", newline="") as output:
            csv.DictWriter(output, fieldnames=permit_link_columns).writeheader()

    output_paths = [
        forest_csv,
        forest_manifest,
        permit_audit,
        farmmap_manifest,
        edge_path,
        target_summary_path,
        permit_links_path,
    ]
    summary = {
        "schema": "kearth-public-ingestion-run-v1",
        "retrieved_at": args.retrieved_at,
        "forest_use": forest.summary,
        "development_permits": permits.quality,
        "farmmap": {
            "rows": farmmap.manifest.row_count,
            "target_summary": farmmap.target_summary,
            "evidence_edge_count": len(farmmap.evidence_edges),
            "exact_permit_pnu_links": len(farmmap.permit_links),
            "exact_permit_unique_pnu": len(
                {row["pnu"] for row in farmmap.permit_links}
            ),
            "exact_permit_unique_farm_polygon": len(
                {row["farm_id"] for row in farmmap.permit_links}
            ),
        },
        "outputs": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output_paths, key=lambda item: item.name)
        },
        "claim_policy": {
            "farmmap_point_hit": "dated official state evidence; not cause",
            "exact_permit_farmmap_pnu": "cross-source parcel context; not cause",
            "farmmap_point_miss": "unknown; FarmMap covers agricultural polygons only",
            "development_permit_no_match": "unknown; source coverage audit failed",
        },
    }
    summary_path = args.out_dir / "run_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
