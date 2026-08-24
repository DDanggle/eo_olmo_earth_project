"""Validated output models for source manifests and evidence edges."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse


class EvidenceGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    M = "M"
    U = "U"


def _require_https(url: str, field_name: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTPS URL: {url!r}")


@dataclass(frozen=True, slots=True)
class SourceManifest:
    source_id: str
    provider: str
    catalog_url: str
    snapshot_date: str
    retrieved_at: str
    raw_file_name: str
    raw_sha256: str
    raw_bytes: int
    license: str
    access_method: str
    data_format: str
    columns: tuple[str, ...]
    schema_sha256: str
    row_count: int
    spatial_coverage: str
    temporal_coverage: str
    crs: str | None = None
    geometry_type: str | None = None
    download_url: str | None = None
    quality: dict[str, Any] = field(default_factory=dict)
    schema: str = "kearth-source-manifest-v1"

    def __post_init__(self) -> None:
        if not self.source_id or any(char.isspace() for char in self.source_id):
            raise ValueError("source_id must be a non-empty whitespace-free identifier")
        _require_https(self.catalog_url, "catalog_url")
        if self.download_url:
            _require_https(self.download_url, "download_url")
        if len(self.raw_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.raw_sha256
        ):
            raise ValueError("raw_sha256 must be a lowercase SHA-256 hex digest")
        if len(self.schema_sha256) != 64:
            raise ValueError("schema_sha256 must be a SHA-256 hex digest")
        if self.raw_bytes <= 0 or self.row_count < 0:
            raise ValueError("raw_bytes must be positive and row_count non-negative")
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("manifest columns must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceEdge:
    edge_id: str
    source_id: str
    source_record_id: str
    relation: str
    evidence_grade: EvidenceGrade
    target_id: str | None = None
    oreum_id: str | None = None
    change_id: str | None = None
    pnu: str | None = None
    spatial_method: str | None = None
    intersection_area_m2: float | None = None
    distance_m: float | None = None
    temporal_method: str | None = None
    day_gap: int | None = None
    no_match_interpretable: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)
    schema: str = "kearth-evidence-edge-v1"

    def __post_init__(self) -> None:
        if not self.edge_id or not self.source_id or not self.source_record_id:
            raise ValueError("edge_id, source_id and source_record_id are required")
        targets = [self.target_id, self.oreum_id, self.change_id]
        if not any(targets):
            raise ValueError("an evidence edge requires at least one target identifier")
        if self.evidence_grade == EvidenceGrade.B and (
            not self.spatial_method or not self.temporal_method
        ):
            raise ValueError(
                "grade B evidence requires explicit spatial_method and temporal_method"
            )
        if self.distance_m is not None and self.distance_m < 0:
            raise ValueError("distance_m cannot be negative")
        if self.intersection_area_m2 is not None and self.intersection_area_m2 < 0:
            raise ValueError("intersection_area_m2 cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["evidence_grade"] = self.evidence_grade.value
        return value
