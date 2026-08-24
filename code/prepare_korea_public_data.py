#!/usr/bin/env python3
"""Normalize Korean public data needed by the Jeju evidence-pack audit.

This script intentionally performs no geocoding and sends no candidate coordinates
to an external service. It converts the official Jeju oreum inventory and filters
the nationwide MOLIT development-permit snapshot to Jeju records only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


OREUM_SOURCE_URL = "https://www.data.go.kr/data/15043497/fileData.do"
PERMIT_CATALOG_URL = "https://www.data.go.kr/data/15021109/fileData.do"
PERMIT_DOWNLOAD_PAGE = (
    "https://eum.go.kr/web/op/sv/svItemDet.jsp?"
    "dataCd=001&dataTypeCd=CSV&currentPageNo=1&selectType=subject"
)
VWORLD_WFS_URL = "https://www.data.go.kr/data/15058805/openapi.do"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_oreum(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="cp949", newline="") as src:
        rows = list(csv.DictReader(src))
    if len(rows) != 368:
        raise ValueError(f"expected 368 official oreum rows, got {len(rows)}")
    return rows


def read_jeju_permits(path: Path) -> tuple[list[dict[str, str]], str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise ValueError(f"expected one CSV in permit ZIP, got {names}")
        member = names[0]
        with archive.open(member) as raw:
            text = io.TextIOWrapper(raw, encoding="cp949", newline="")
            reader = csv.DictReader(text)
            rows = [
                row
                for row in reader
                if (row.get("지자체코드") or "").startswith("50")
                or "제주특별자치도" in (row.get("위치명") or "")
                or "제주특별자치도" in (row.get("지자체명") or "")
            ]
    return rows, member


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def permit_year(value: str) -> str:
    value = (value or "").strip()
    return value[:4] if len(value) >= 4 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oreum-csv", type=Path, required=True)
    parser.add_argument("--permit-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    oreum_rows = read_oreum(args.oreum_csv)
    permit_rows, permit_member = read_jeju_permits(args.permit_zip)
    if not permit_rows:
        raise ValueError("no Jeju development-permit records found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    oreum_out = args.out_dir / "jeju_oreum_official_20240331.csv"
    permits_out = args.out_dir / "jeju_development_permits_20260819.csv"
    write_csv(oreum_out, oreum_rows)
    write_csv(permits_out, permit_rows)

    years = Counter(permit_year(row.get("허가일자", "")) for row in permit_rows)
    municipalities = Counter(
        (row.get("지자체명") or "").strip() or "unknown" for row in permit_rows
    )
    actions = Counter(
        (row.get("대표개발행위명") or "").strip() or "unknown" for row in permit_rows
    )
    summary = {
        "schema": "jeju-korean-public-data-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "privacy": {
            "candidate_coordinates_sent_to_external_services": False,
            "spatial_join_status": "pending_explicit_coordinate-transmission_consent",
        },
        "sources": {
            "jeju_oreum_inventory": {
                "provider": "제주특별자치도",
                "catalog_url": OREUM_SOURCE_URL,
                "snapshot_date": "2024-03-31",
                "downloaded_file_sha256": sha256(args.oreum_csv),
                "license": "이용허락범위 제한 없음",
                "rows": len(oreum_rows),
                "spatial_limit": "주소만 제공하며 좌표·경계는 없음",
            },
            "development_permits": {
                "provider": "국토교통부",
                "catalog_url": PERMIT_CATALOG_URL,
                "download_page": PERMIT_DOWNLOAD_PAGE,
                "snapshot_date": "2026-08-19",
                "downloaded_file_sha256": sha256(args.permit_zip),
                "archive_member": permit_member,
                "license": "공공저작물 출처표시 제1유형",
                "jeju_rows": len(permit_rows),
                "spatial_limit": "PNU·위치명은 있으나 위경도는 없어 지적도 연결이 필요",
            },
            "vworld_wfs": {
                "provider": "국토교통부 국가공간정보센터",
                "catalog_url": VWORLD_WFS_URL,
                "status": "unavailable_without_api_key",
                "intended_role": "후보 좌표를 PNU·지적·건물·용도지역 객체에 공간 연결",
            },
        },
        "permit_summary": {
            "by_year": dict(sorted(years.items())),
            "by_municipality": dict(municipalities.most_common()),
            "top_representative_actions": dict(actions.most_common(20)),
        },
        "outputs": {
            "oreum_csv": oreum_out.name,
            "development_permits_csv": permits_out.name,
        },
    }
    summary_path = args.out_dir / "sources_and_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "oreum_rows": len(oreum_rows),
        "jeju_permit_rows": len(permit_rows),
        "permit_years": dict(sorted(years.items())),
        "outputs": [str(oreum_out), str(permits_out), str(summary_path)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
