"""Validation and parsing for Korea's 19-digit parcel identifier (PNU)."""

from __future__ import annotations

from dataclasses import dataclass


class InvalidPNU(ValueError):
    """Raised when a value cannot be a 19-digit PNU."""


@dataclass(frozen=True, slots=True)
class PNU:
    value: str
    legal_dong_code: str
    mountain: bool
    main_number: int
    sub_number: int

    @classmethod
    def parse(cls, raw: str) -> "PNU":
        value = str(raw or "").strip()
        if len(value) != 19 or not value.isdigit():
            raise InvalidPNU(f"PNU must contain exactly 19 digits: {raw!r}")
        land_code = value[10]
        if land_code not in {"1", "2"}:
            raise InvalidPNU(
                f"PNU land/mountain code must be 1 or 2 at index 10: {raw!r}"
            )
        legal_dong_code = value[:10]
        if legal_dong_code == "0" * 10:
            raise InvalidPNU(f"PNU legal-dong code cannot be all zero: {raw!r}")
        return cls(
            value=value,
            legal_dong_code=legal_dong_code,
            mountain=land_code == "2",
            main_number=int(value[11:15]),
            sub_number=int(value[15:19]),
        )

    @classmethod
    def is_valid(cls, raw: str) -> bool:
        try:
            cls.parse(raw)
        except InvalidPNU:
            return False
        return True
