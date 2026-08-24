"""Strict adapters for official CSV snapshots used by the K-Earth audit."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .canonical import file_sha256, schema_sha256
from .models import SourceManifest
from .pnu import PNU
from .temporal import CoverageAudit, parse_date


def read_csv_with_encoding(path: Path) -> tuple[list[dict[str, str]], str]:
    errors: dict[str, str] = {}
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with path.open(encoding=encoding, newline="") as source:
                reader = csv.DictReader(source)
                if not reader.fieldnames:
                    raise ValueError(f"CSV has no header: {path}")
                rows = list(reader)
                if None in reader.fieldnames:
                    raise ValueError(f"CSV contains an empty column name: {path}")
                return rows, encoding
        except UnicodeDecodeError as exc:
            errors[encoding] = str(exc)
    raise ValueError(f"CSV is neither UTF-8 nor CP949: {path}; errors={errors}")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty normalized CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True, slots=True)
class ForestUseResult:
    rows: list[dict[str, Any]]
    manifest: SourceManifest
    summary: dict[str, Any]


def ingest_jeju_forest_use(
    path: Path,
    *,
    catalog_url: str,
    download_url: str,
    retrieved_at: str,
) -> ForestUseResult:
    rows, encoding = read_csv_with_encoding(path)
    expected = ["해당연도", "건수", "면적(ha)", "데이터기준일자"]
    columns = list(rows[0]) if rows else []
    if columns != expected:
        raise ValueError(
            f"unexpected forest-use schema: {columns}; expected={expected}"
        )

    normalized: list[dict[str, Any]] = []
    seen_years: set[int] = set()
    snapshot_dates: set[str] = set()
    for index, row in enumerate(rows, start=2):
        try:
            year = int(row["해당연도"])
            count = int(row["건수"])
            area_ha = Decimal(row["면적(ha)"])
            snapshot = parse_date(row["데이터기준일자"]).isoformat()
        except (KeyError, ValueError, InvalidOperation) as exc:
            raise ValueError(f"invalid forest-use row {index}: {row}") from exc
        if year in seen_years or count < 0 or area_ha < 0:
            raise ValueError(
                f"invalid duplicate/negative forest-use row {index}: {row}"
            )
        seen_years.add(year)
        snapshot_dates.add(snapshot)
        normalized.append(
            {
                "year": year,
                "approval_count": count,
                "approved_area_ha": str(area_ha),
                "snapshot_date": snapshot,
            }
        )
    normalized.sort(key=lambda row: row["year"])
    expected_years = list(range(normalized[0]["year"], normalized[-1]["year"] + 1))
    actual_years = [row["year"] for row in normalized]
    if actual_years != expected_years:
        raise ValueError(f"forest-use years are not contiguous: {actual_years}")
    if len(snapshot_dates) != 1:
        raise ValueError(f"forest-use rows have mixed snapshot dates: {snapshot_dates}")

    normalized_columns = [
        "year",
        "approval_count",
        "approved_area_ha",
        "snapshot_date",
    ]
    manifest = SourceManifest(
        source_id="jeju_forest_use_aggregate",
        provider="제주특별자치도 제주시",
        catalog_url=catalog_url,
        download_url=download_url,
        snapshot_date=next(iter(snapshot_dates)),
        retrieved_at=retrieved_at,
        raw_file_name=path.name,
        raw_sha256=file_sha256(path),
        raw_bytes=path.stat().st_size,
        license="이용허락범위 제한 없음",
        access_method="public_portal_file_download",
        data_format="CSV",
        columns=tuple(normalized_columns),
        schema_sha256=schema_sha256(normalized_columns),
        row_count=len(normalized),
        spatial_coverage="제주시 aggregate; no parcel geometry",
        temporal_coverage=f"{actual_years[0]}-{actual_years[-1]} annual",
        quality={
            "source_encoding": encoding,
            "contiguous_years": True,
            "parcel_joinable": False,
            "no_match_audit": {
                "no_match_interpretable": False,
                "failed_checks": [
                    "geography_complete",
                    "event_population_complete",
                    "join_fields_complete",
                ],
                "reason": "city aggregate is a denominator, not parcel-level evidence",
            },
        },
    )
    by_year = {str(row["year"]): row for row in normalized}
    summary = {
        "years": [actual_years[0], actual_years[-1]],
        "row_count": len(normalized),
        "2023": by_year.get("2023"),
        "2024": by_year.get("2024"),
        "parcel_joinable": False,
    }
    return ForestUseResult(normalized, manifest, summary)


@dataclass(frozen=True, slots=True)
class PermitResult:
    rows: list[dict[str, str]]
    pnu_to_rows: dict[str, list[dict[str, str]]]
    quality: dict[str, Any]


def audit_development_permits(path: Path) -> PermitResult:
    rows, encoding = read_csv_with_encoding(path)
    required = {"PNU", "위치명", "허가일자", "개발행위명", "개발행위목적"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(
            f"development-permit CSV is missing columns: {sorted(required - set(rows[0]))}"
        )
    pnu_to_rows: dict[str, list[dict[str, str]]] = {}
    invalid_pnus: list[dict[str, Any]] = []
    years: Counter[str] = Counter()
    missing_dates = 0
    for index, row in enumerate(rows, start=2):
        raw_pnu = (row.get("PNU") or "").strip()
        if not PNU.is_valid(raw_pnu):
            invalid_pnus.append({"row": index, "value": raw_pnu})
            continue
        pnu_to_rows.setdefault(raw_pnu, []).append(row)
        raw_date = (row.get("허가일자") or "").strip()
        if not raw_date:
            missing_dates += 1
            years["unknown"] += 1
            continue
        try:
            years[str(parse_date(raw_date).year)] += 1
        except ValueError:
            missing_dates += 1
            years["invalid"] += 1
    audit = CoverageAudit(
        geography_complete=False,
        time_complete=False,
        event_population_complete=False,
        collection_complete=False,
        join_fields_complete=not invalid_pnus,
    )
    quality = {
        "source_encoding": encoding,
        "row_count": len(rows),
        "unique_pnu_count": len(pnu_to_rows),
        "valid_pnu_count": len(rows) - len(invalid_pnus),
        "invalid_pnus": invalid_pnus,
        "missing_or_invalid_permit_dates": missing_dates,
        "by_year": dict(sorted(years.items())),
        "no_match_interpretable": audit.no_match_interpretable,
        "failed_coverage_checks": audit.failed_checks(),
    }
    return PermitResult(rows, pnu_to_rows, quality)
