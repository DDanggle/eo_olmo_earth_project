#!/usr/bin/env python3
"""Resolve moving Hugging Face model IDs to immutable commits and file hashes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REPOS = (
    "allenai/OlmoEarth-v1-Base",
    "allenai/OlmoEarth-v1_2-Base",
)


def package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not-installed"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_revision(path: Path) -> str:
    # Hugging Face snapshot files are symlinks into blobs. Resolving the symlink
    # would erase the immutable `snapshots/<commit>` component we need to record.
    parts = path.parts
    try:
        return parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"cannot recover Hugging Face snapshot revision from {path}") from error


def resolve_repo(repo_id: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    initial_config = Path(hf_hub_download(repo_id=repo_id, filename="config.json"))
    revision = snapshot_revision(initial_config)
    config_path = Path(
        hf_hub_download(repo_id=repo_id, filename="config.json", revision=revision)
    )
    weights_path = Path(
        hf_hub_download(repo_id=repo_id, filename="weights.pth", revision=revision)
    )
    if config_path.parent != weights_path.parent:
        raise ValueError(f"config and weights resolved to different snapshots for {repo_id}")
    files = []
    for path in (config_path, weights_path):
        files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "repo_id": repo_id,
        "revision": revision,
        "snapshot_path": config_path.parent.as_posix(),
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", dest="repos")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/work/data/olmoearth/release_audit_p0/checkpoints.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repos = tuple(args.repos or DEFAULT_REPOS)
    if len(repos) != len(set(repos)):
        raise ValueError("repository IDs must be unique")
    result = {
        "schema": "olmoearth-checkpoint-resolution-v1",
        "resolved_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "huggingface_hub": package_version("huggingface-hub"),
            "olmoearth_pretrain": package_version(
                "olmoearth-pretrain-minimal", "olmoearth-pretrain"
            ),
            "rslearn": package_version("rslearn"),
        },
        "models": [resolve_repo(repo_id) for repo_id in repos],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
