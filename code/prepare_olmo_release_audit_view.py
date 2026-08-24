#!/usr/bin/env python3
"""Create a non-destructive rslearn dataset view over immutable audit inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


OUTPUT_LAYERS = (
    "embeddings_audit_v1_legacy",
    "embeddings_audit_v1_2_legacy",
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def safe_symlink(source: Path, target: Path) -> None:
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f"existing symlink points elsewhere: {target}")
        return
    if target.exists():
        raise FileExistsError(target)
    target.symlink_to(source)


def build_view(
    source_root: Path,
    input_manifest: Path,
    output_root: Path,
    output_layers: tuple[str, ...] = OUTPUT_LAYERS,
) -> dict[str, Any]:
    if not output_layers or len(set(output_layers)) != len(output_layers):
        raise ValueError("output layers must be non-empty and unique")
    source_config = read_json(source_root / "config.json")
    output_config = json.loads(json.dumps(source_config))
    for layer_name in output_layers:
        existing = output_config["layers"].get(layer_name)
        expected = {"band_sets": [{"dtype": "float32", "num_bands": 768}], "type": "raster"}
        if existing is not None and existing != expected:
            raise ValueError(f"incompatible output layer already defined: {layer_name}")
        output_config["layers"][layer_name] = expected

    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "config.json"
    rendered_config = json.dumps(output_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if config_path.exists() and config_path.read_text(encoding="utf-8") != rendered_config:
        raise ValueError(f"refusing to replace a different config: {config_path}")
    config_path.write_text(rendered_config, encoding="utf-8")

    records = read_json(input_manifest)["records"]
    if not records:
        raise ValueError("input manifest contains no records")
    linked_layers = 0
    for record in records:
        source_window = Path(record["window_dir"])
        if source_window.parent.parent.parent.resolve() != source_root.resolve():
            raise ValueError(f"window escapes declared source dataset: {source_window}")
        target_window = output_root / "windows/default" / record["window_name"]
        target_window.mkdir(parents=True, exist_ok=True)
        for name in ("items.json", "metadata.json"):
            source = source_window / name
            target = target_window / name
            if target.exists() or target.is_symlink():
                if target.read_bytes() != source.read_bytes():
                    raise ValueError(f"existing metadata differs: {target}")
            else:
                shutil.copy2(source, target)
        target_layers = target_window / "layers"
        target_layers.mkdir(exist_ok=True)
        for period_index in range(12):
            layer_name = "sentinel2_l2a" if period_index == 0 else f"sentinel2_l2a.{period_index}"
            safe_symlink(source_window / "layers" / layer_name, target_layers / layer_name)
            linked_layers += 1

    manifest = {
        "schema": "olmoearth-release-audit-view-v1",
        "source_dataset": source_root.as_posix(),
        "input_manifest": input_manifest.as_posix(),
        "output_dataset": output_root.as_posix(),
        "windows": len(records),
        "linked_input_layers": linked_layers,
        "output_layers": list(output_layers),
        "source_inputs_mutated": False,
    }
    manifest_path = output_root / "audit_view_manifest.json"
    rendered_manifest = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != rendered_manifest:
        raise ValueError(f"refusing to replace a different manifest: {manifest_path}")
    manifest_path.write_text(rendered_manifest, encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument(
        "--output-layer",
        action="append",
        dest="output_layers",
        help="Output layer to define; repeat for multiple layers (defaults to the paired audit layers)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_view(
        args.source_dataset,
        args.smoke_manifest,
        args.output_dataset,
        tuple(args.output_layers) if args.output_layers else OUTPUT_LAYERS,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
