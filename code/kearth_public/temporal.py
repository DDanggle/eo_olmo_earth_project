"""Explicit temporal-overlap rules for evidence joins."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta


def parse_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{4})[-/.]?(\d{2})[-/.]?(\d{2})", text)
    if not match:
        raise ValueError(f"unsupported date value: {value!r}")
    return date(*(int(part) for part in match.groups()))


@dataclass(frozen=True, slots=True)
class DateInterval:
    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"interval start {self.start} is after end {self.end}")

    @classmethod
    def from_values(
        cls, start: str | date | datetime, end: str | date | datetime
    ) -> "DateInterval":
        return cls(parse_date(start), parse_date(end))

    def expanded(self, days: int) -> "DateInterval":
        if days < 0:
            raise ValueError("interval expansion must be non-negative")
        delta = timedelta(days=days)
        return DateInterval(self.start - delta, self.end + delta)

    def overlaps(self, other: "DateInterval") -> bool:
        return self.start <= other.end and other.start <= self.end

    def day_gap(self, other: "DateInterval") -> int:
        if self.overlaps(other):
            return 0
        if self.end < other.start:
            return (other.start - self.end).days
        return (self.start - other.end).days


@dataclass(frozen=True, slots=True)
class CoverageAudit:
    geography_complete: bool
    time_complete: bool
    event_population_complete: bool
    collection_complete: bool
    join_fields_complete: bool

    @property
    def no_match_interpretable(self) -> bool:
        return all(
            (
                self.geography_complete,
                self.time_complete,
                self.event_population_complete,
                self.collection_complete,
                self.join_fields_complete,
            )
        )

    def failed_checks(self) -> list[str]:
        values = {
            "geography_complete": self.geography_complete,
            "time_complete": self.time_complete,
            "event_population_complete": self.event_population_complete,
            "collection_complete": self.collection_complete,
            "join_fields_complete": self.join_fields_complete,
        }
        return [name for name, passed in values.items() if not passed]
