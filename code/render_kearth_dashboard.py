#!/usr/bin/env python3
"""Render the K-Earth dashboard from an existing evidence registry.

This keeps presentation-only updates reproducible without rerunning OSM extraction
or rebuilding the 368-record registry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_kearth_oreum_registry import (
    DEFAULT_ACCESS_STATUS,
    load_access_status,
    render_dashboard,
    sha256,
)


def render_existing_registry(
    registry: Path, output: Path, access_status: Path = DEFAULT_ACCESS_STATUS
) -> None:
    payload = json.loads(registry.read_text(encoding="utf-8"))
    if not str(payload.get("schema", "")).startswith(
        "kearth-oreum-evidence-registry-v"
    ):
        raise ValueError(f"unexpected registry schema: {payload.get('schema')!r}")
    if payload.get("summary", {}).get("official_inventory") != 368:
        raise ValueError("dashboard requires the fixed 368-record oreum denominator")
    payload["registry_sha256"] = sha256(registry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_dashboard(payload, load_access_status(access_status)), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--access-status", type=Path, default=DEFAULT_ACCESS_STATUS)
    args = parser.parse_args()
    render_existing_registry(args.registry, args.output, args.access_status)


if __name__ == "__main__":
    main()
