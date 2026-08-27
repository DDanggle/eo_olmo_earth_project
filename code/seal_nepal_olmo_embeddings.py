#!/usr/bin/env python3
"""Seal OLMoEarth 768-band COG outputs for the five Nepal anchors."""
# ruff: noqa: D103
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--code-snapshot", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    expected = {"source_provisional", "rasuwagadhi", "timure", "syabrubesi", "dhunche"}
    rows = []
    for anchor in sorted(expected):
        layer = args.dataset / "windows" / "nepal" / anchor / "layers" / "embeddings"
        candidates = sorted(layer.rglob("*.tif")) if layer.exists() else []
        if len(candidates) != 1:
            rows.append({"anchor": anchor, "valid": False, "reason": f"expected_one_tif_found_{len(candidates)}"})
            continue
        path = candidates[0]
        with rasterio.open(path) as source:
            metadata = {
                "bands": source.count,
                "width": source.width,
                "height": source.height,
                "crs": str(source.crs),
                "transform": list(source.transform)[:6],
                "dtype": source.dtypes[0],
            }
        rows.append({
            "anchor": anchor,
            "valid": metadata["bands"] == 768,
            "path": str(path.relative_to(args.dataset)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            **metadata,
        })
    manifest = {
        "schema": "nepal-olmo-embeddings-v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "mode": args.mode,
        "model_id": "OLMOEARTH_V1_BASE",
        "embedding_dim": 768,
        "code_snapshot": str(args.code_snapshot),
        "anchors": rows,
        "valid": len(rows) == len(expected) and all(row["valid"] for row in rows),
    }
    out = args.dataset.parent / "embedding_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal = args.dataset.parent / "EMBEDDING_SHA256SUMS"
    seal.write_text(f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out), "valid": manifest["valid"], "seal_sha256": sha256_file(seal)}, sort_keys=True))
    if not manifest["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
