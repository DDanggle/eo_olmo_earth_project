#!/usr/bin/env python3
"""Validate and optionally execute the exact-input OlmoEarth v1/v1.2 smoke audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS = (
    {
        "release_id": "olmoearth_v1_base",
        "repo_id": "allenai/OlmoEarth-v1-Base",
        "model_env": "OLMO_V1_MODEL_PATH",
        "config": "olmo_release_v1_legacy.yaml",
        "output_layer": "embeddings_audit_v1_legacy",
    },
    {
        "release_id": "olmoearth_v1_2_base",
        "repo_id": "allenai/OlmoEarth-v1_2-Base",
        "model_env": "OLMO_V1_2_MODEL_PATH",
        "config": "olmo_release_v1_2_legacy.yaml",
        "output_layer": "embeddings_audit_v1_2_legacy",
    },
)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoints(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    by_repo = {value["repo_id"]: value for value in payload["models"]}
    if set(by_repo) != {value["repo_id"] for value in RUNS}:
        raise ValueError("checkpoint manifest does not contain the exact two audit repositories")
    for model in by_repo.values():
        snapshot = Path(model["snapshot_path"])
        if snapshot.name != model["revision"]:
            raise ValueError(f"snapshot path/revision mismatch for {model['repo_id']}")
        for file_record in model["files"]:
            target = snapshot / file_record["name"]
            if target.stat().st_size != file_record["bytes"]:
                raise ValueError(f"checkpoint size drift: {target}")
            if file_sha256(target) != file_record["sha256"]:
                raise ValueError(f"checkpoint SHA drift: {target}")
    return by_repo


def validate_inputs(exact_inputs: Path, dataset_root: Path) -> list[dict[str, Any]]:
    payload = read_json(exact_inputs)
    if not payload.get("exact_tensor_file_pairing_ready"):
        raise ValueError("smoke inputs are not content-hashed")
    records = payload["records"]
    if len(records) != 8 or any(record["hash_policy"] != "sha256" for record in records):
        raise ValueError("the release smoke requires exactly eight SHA-256 input records")
    required_identity_fields = {
        "sample_id",
        "window_name",
        "input_bundle_identity",
        "spatial_cluster_id",
    }
    if any(not required_identity_fields.issubset(record) for record in records):
        raise ValueError("exact inputs are missing sample/window/bundle/spatial identities")
    if len({record["sample_id"] for record in records}) != 8:
        raise ValueError("exact input sample IDs are not unique")
    dataset_windows = {path.name for path in (dataset_root / "windows/default").iterdir() if path.is_dir()}
    if dataset_windows != {record["window_name"] for record in records}:
        raise ValueError("audit dataset windows differ from the exact input manifest")
    for record in records:
        view_window = dataset_root / "windows/default" / record["window_name"]
        for layer in record["input_layers"]:
            for field in ("geotiff", "metadata"):
                inventory = layer[field]
                source_file = Path(inventory["path"])
                if source_file.stat().st_size != inventory["bytes"]:
                    raise ValueError(f"input size drift: {source_file}")
                if file_sha256(source_file) != inventory["sha256"]:
                    raise ValueError(f"input SHA drift: {source_file}")
            view_layer = view_window / "layers" / layer["layer_name"]
            source_layer = Path(layer["geotiff"]["path"]).parents[1]
            if not view_layer.is_symlink() or view_layer.resolve() != source_layer.resolve():
                raise ValueError(f"audit view is not linked to the hashed input layer: {view_layer}")
        for field, view_name in (("items_json", "items.json"), ("window_metadata", "metadata.json")):
            inventory = record[field]
            source_file = Path(inventory["path"])
            view_file = view_window / view_name
            if file_sha256(source_file) != inventory["sha256"]:
                raise ValueError(f"input SHA drift: {source_file}")
            if file_sha256(view_file) != inventory["sha256"]:
                raise ValueError(f"audit-view metadata drift: {view_file}")
    return records


def filter_gpu_processes(selected_uuid: str, process_rows: str) -> list[str]:
    result = []
    for line in process_rows.splitlines():
        if not line.strip():
            continue
        parts = [value.strip() for value in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi process row: {line!r}")
        gpu_uuid, pid, process_name, used_memory = parts
        if gpu_uuid == selected_uuid:
            result.append(f"{pid}, {process_name}, {used_memory}")
    return result


def gpu_processes(gpu_index: str) -> list[str]:
    gpu_inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    uuid_by_index = {}
    for line in gpu_inventory.stdout.splitlines():
        if not line.strip():
            continue
        index, gpu_uuid = [value.strip() for value in line.split(",", maxsplit=1)]
        uuid_by_index[index] = gpu_uuid
    if gpu_index not in uuid_by_index:
        raise ValueError(f"GPU index {gpu_index!r} is not present in nvidia-smi inventory")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return filter_gpu_processes(uuid_by_index[gpu_index], completed.stdout)


def output_inventory(
    dataset_root: Path,
    layer_name: str,
    records: list[dict[str, Any]],
    minimum_mtime: float,
) -> list[dict[str, Any]]:
    result = []
    for record in records:
        window = record["window_name"]
        candidates = sorted(
            (dataset_root / "windows/default" / window / "layers" / layer_name).glob(
                "**/geotiff.tif"
            )
        )
        if len(candidates) != 1:
            raise ValueError(f"expected one output GeoTIFF for {window}/{layer_name}, found {len(candidates)}")
        target = candidates[0]
        if target.stat().st_mtime + 1e-6 < minimum_mtime:
            raise ValueError(f"stale output predates this release run: {target}")
        result.append(
            {
                "sample_id": record["sample_id"],
                "window": window,
                "input_bundle_identity": record["input_bundle_identity"],
                "spatial_cluster_id": record["spatial_cluster_id"],
                "path": target.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": file_sha256(target),
                "mtime_ns": target.stat().st_mtime_ns,
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--rslearn", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = validate_checkpoints(args.checkpoint_manifest)
    records = validate_inputs(args.exact_inputs, args.dataset_root)
    processes = gpu_processes(args.gpu)
    preflight = {
        "schema": "olmoearth-release-smoke-preflight-v1",
        "checkpoint_manifest_sha256": file_sha256(args.checkpoint_manifest),
        "exact_inputs_sha256": file_sha256(args.exact_inputs),
        "dataset_root": args.dataset_root.as_posix(),
        "records": len(records),
        "release_runs": len(RUNS),
        "selected_gpu": args.gpu,
        "gpu_processes": processes,
        "execute_requested": args.execute,
        "ready": not processes,
    }
    if args.preflight_output:
        args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_output.write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    if not args.execute:
        return
    if processes:
        raise SystemExit("refusing to perturb active GPU jobs; rerun when the GPU is idle")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_records = []
    for run in RUNS:
        model = checkpoints[run["repo_id"]]
        config_path = args.config_dir / run["config"]
        environment = os.environ.copy()
        environment.update(
            {
                "DATASET_PATH": args.dataset_root.as_posix(),
                run["model_env"]: model["snapshot_path"],
                "CUDA_VISIBLE_DEVICES": args.gpu,
            }
        )
        log_path = args.output_dir / f"{run['release_id']}.log"
        started_at = datetime.now(timezone.utc).isoformat()
        started_epoch = time.time()
        start = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [args.rslearn.as_posix(), "model", "predict", "--config", config_path.as_posix()],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{run['release_id']} failed; inspect {log_path}")
        outputs = output_inventory(
            args.dataset_root, run["output_layer"], records, started_epoch
        )
        run_records.append(
            {
                "release_id": run["release_id"],
                "repo_id": run["repo_id"],
                "revision": model["revision"],
                "checkpoint_files": model["files"],
                "config_path": config_path.as_posix(),
                "config_sha256": file_sha256(config_path),
                "timestamp_track": "legacy_timestamps",
                "started_at": started_at,
                "wall_seconds": round(time.monotonic() - start, 3),
                "outputs": outputs,
                "log_path": log_path.as_posix(),
                "log_sha256": file_sha256(log_path),
            }
        )
    result = {
        "schema": "olmoearth-release-smoke-result-v1",
        "status": "complete",
        "input_pairing": {
            "records": len(records),
            "exact_inputs_sha256": file_sha256(args.exact_inputs),
            "same_manifest_for_both_releases": True,
            "samples": [
                {
                    "sample_id": record["sample_id"],
                    "window_name": record["window_name"],
                    "input_bundle_identity": record["input_bundle_identity"],
                    "spatial_cluster_id": record["spatial_cluster_id"],
                }
                for record in records
            ],
        },
        "runs": run_records,
        "claims_allowed": ["paired_release_representation_audit"],
        "claims_forbidden": ["accuracy_improvement", "negative_transfer_reduction"],
    }
    result_path = args.output_dir / "run_summary.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    complete = {
        "schema": "olmoearth-release-smoke-completion-v1",
        "run_summary_sha256": file_sha256(result_path),
    }
    (args.output_dir / "COMPLETE.json").write_text(
        json.dumps(complete, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
