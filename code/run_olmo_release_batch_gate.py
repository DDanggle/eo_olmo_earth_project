#!/usr/bin/env python3
"""Benchmark safe rslearn batch settings on the exact eight-window v1.2 audit.

Each candidate receives a fresh dataset view and output layer.  The program
holds an OS-level GPU0 lock, refuses an occupied selected GPU, records one-second
telemetry, hashes every output, and writes a completion marker only after all
candidates finish.  Numerical equivalence is intentionally evaluated by the
separate ``analyze_olmo_release_batch_gate.py`` program.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import signal
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmo_release_semantic_contract import (
    fingerprint_rslearn_runtime,
    normalize_checkpoint_manifest,
    validate_launcher_runtime_binding,
)
from prepare_olmo_release_audit_view import build_view
from run_olmo_release_smoke import (
    file_sha256,
    gpu_processes,
    output_inventory,
    read_json,
    validate_checkpoints,
    validate_inputs,
)


RELEASES = {
    "v1": {
        "release_id": "olmoearth_v1_base",
        "repo_id": "allenai/OlmoEarth-v1-Base",
        "model_env": "OLMO_V1_MODEL_PATH",
        "output_layer": "embeddings_batch_gate_v1",
    },
    "v1_2": {
        "release_id": "olmoearth_v1_2_base",
        "repo_id": "allenai/OlmoEarth-v1_2-Base",
        "model_env": "OLMO_V1_2_MODEL_PATH",
        "output_layer": "embeddings_batch_gate_v1_2",
    },
}
PLACEHOLDERS = ("__BATCH_SIZE__", "__NUM_WORKERS__", "__OUTPUT_LAYER__")
CROPS_PER_WINDOW = 961
AUDIT_CODE_SCHEMA = "olmoearth-release-batch-audit-code-contract-v1"
AUDIT_HELPER_MODULES = {
    "olmo_release_semantic_contract": Path(
        fingerprint_rslearn_runtime.__code__.co_filename
    ).resolve(),
    "prepare_olmo_release_audit_view": Path(build_view.__code__.co_filename).resolve(),
    "run_olmo_release_smoke": Path(file_sha256.__code__.co_filename).resolve(),
}


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def stable_code_file_record(path: Path, *, module: str) -> dict[str, Any]:
    """Hash one Python source while rejecting an in-flight file replacement."""

    resolved = path.resolve(strict=True)
    before = resolved.stat()
    digest = file_sha256(resolved)
    after = resolved.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError(f"audit code changed while hashing: {resolved}")
    return {
        "module": module,
        "path": resolved.as_posix(),
        "bytes": after.st_size,
        "sha256": digest,
    }


def batch_audit_code_contract() -> dict[str, Any]:
    """Bind the batch runner and every local module it imports directly."""

    runner = stable_code_file_record(
        Path(__file__), module="run_olmo_release_batch_gate"
    )
    helpers = [
        stable_code_file_record(path, module=module)
        for module, path in sorted(AUDIT_HELPER_MODULES.items())
    ]
    inventory = {"runner": runner, "direct_local_helpers": helpers}
    return {
        "schema": AUDIT_CODE_SCHEMA,
        **inventory,
        "inventory_sha256": hashlib.sha256(canonical_bytes(inventory)).hexdigest(),
    }


def validate_batch_audit_code_contract(value: Any) -> dict[str, Any]:
    """Validate and canonically normalize a persisted batch audit code contract."""

    if not isinstance(value, dict) or value.get("schema") != AUDIT_CODE_SCHEMA:
        raise ValueError("unrecognized batch audit code contract")
    runner = value.get("runner")
    helpers = value.get("direct_local_helpers")
    if not isinstance(runner, dict) or not isinstance(helpers, list):
        raise ValueError("batch audit code contract lacks runner/helper inventory")
    if runner.get("module") != "run_olmo_release_batch_gate":
        raise ValueError("batch audit runner module identity drift")
    expected_helper_modules = sorted(AUDIT_HELPER_MODULES)
    if [record.get("module") for record in helpers] != expected_helper_modules:
        raise ValueError("batch audit direct-helper module inventory drift")
    records = [runner, *helpers]
    for record in records:
        path = record.get("path")
        size = record.get("bytes")
        digest = record.get("sha256")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("batch audit code path must be absolute")
        if not isinstance(size, int) or size <= 0:
            raise ValueError("batch audit code byte count is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("batch audit code SHA-256 is invalid")
    normalized_inventory = {
        "runner": dict(runner),
        "direct_local_helpers": [dict(record) for record in helpers],
    }
    expected_inventory_sha = hashlib.sha256(
        canonical_bytes(normalized_inventory)
    ).hexdigest()
    if value.get("inventory_sha256") != expected_inventory_sha:
        raise ValueError("batch audit code inventory digest mismatch")
    return {
        "schema": AUDIT_CODE_SCHEMA,
        **normalized_inventory,
        "inventory_sha256": expected_inventory_sha,
    }


def atomic_create(path: Path, content: bytes) -> None:
    """Create an immutable evidence file, accepting only an identical rerun."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace different evidence: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_replace(path: Path, content: bytes) -> None:
    """Atomically update a progress file that is not itself final evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def parse_candidate(value: str) -> tuple[int, int]:
    try:
        batch_text, workers_text = value.split(":", maxsplit=1)
        batch_size, workers = int(batch_text), int(workers_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("candidate must be BATCH:WORKERS") from exc
    if batch_size < 1 or workers < 0:
        raise argparse.ArgumentTypeError("batch must be positive and workers non-negative")
    if workers > 8:
        raise argparse.ArgumentTypeError("workers above eight are forbidden for this workload")
    return batch_size, workers


def candidate_id(batch_size: int, workers: int) -> str:
    return f"b{batch_size:03d}_w{workers:02d}"


def render_config(
    template_path: Path,
    output_path: Path,
    batch_size: int,
    workers: int,
    model_env: str,
    output_layer: str,
) -> None:
    rendered = template_path.read_text(encoding="utf-8")
    replacements = {
        "__BATCH_SIZE__": str(batch_size),
        "__NUM_WORKERS__": str(workers),
        "__OUTPUT_LAYER__": output_layer,
    }
    if "__MODEL_PATH__" in rendered:
        replacements["__MODEL_PATH__"] = "${" + model_env + "}"
    elif f"${{{model_env}}}" not in rendered:
        raise ValueError(f"template is not bound to model environment {model_env}")
    for placeholder, replacement in replacements.items():
        if rendered.count(placeholder) != 1:
            raise ValueError(f"template must contain {placeholder} exactly once")
        rendered = rendered.replace(placeholder, replacement)
    if any(value in rendered for value in PLACEHOLDERS):
        raise ValueError("unresolved batch-gate config placeholder")
    atomic_create(output_path, rendered.encode("utf-8"))


def gpu_uuid(gpu_index: str) -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = {}
    for line in completed.stdout.splitlines():
        index, value = [part.strip() for part in line.split(",", maxsplit=1)]
        values[index] = value
    if gpu_index not in values:
        raise ValueError(f"selected GPU {gpu_index} is absent")
    return values[gpu_index]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def available_host_memory_mib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def optional_nvidia_float(value: str) -> float | None:
    """Parse nvidia-smi numbers while preserving legitimate unavailable fields."""
    normalized = value.strip()
    if normalized in {"[Not Found]", "N/A", "[N/A]", "Not Supported"}:
        return None
    return float(normalized)


class TelemetrySampler:
    def __init__(self, gpu_index: str, interval_seconds: float = 1.0) -> None:
        self.gpu_index = gpu_index
        self.interval_seconds = interval_seconds
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5.0, self.interval_seconds * 3))
        if self._thread.is_alive():
            raise RuntimeError("telemetry thread did not stop")

    def _run(self) -> None:
        previous = None
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.current.sm",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2.5,
                )
                parts = [value.strip() for value in completed.stdout.strip().split(",")]
                if len(parts) != 8:
                    raise ValueError(f"unexpected telemetry row: {completed.stdout!r}")
                now = time.monotonic()
                self.rows.append(
                    {
                        "monotonic_seconds": now,
                        "gap_seconds": None if previous is None else now - previous,
                        "gpu_uuid": parts[0],
                        "gpu_utilization_percent": optional_nvidia_float(parts[1]),
                        "memory_utilization_percent": optional_nvidia_float(parts[2]),
                        "memory_used_mib": optional_nvidia_float(parts[3]),
                        "memory_total_mib": optional_nvidia_float(parts[4]),
                        "power_watts": optional_nvidia_float(parts[5]),
                        "temperature_celsius": optional_nvidia_float(parts[6]),
                        "sm_clock_mhz": optional_nvidia_float(parts[7]),
                        "host_mem_available_mib": available_host_memory_mib(),
                    }
                )
                previous = now
            except Exception as exc:  # telemetry failure must invalidate the candidate
                self.errors.append(f"{type(exc).__name__}: {exc}")
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, self.interval_seconds - elapsed))

    def summary(self) -> dict[str, Any]:
        if not self.rows:
            return {"samples": 0, "errors": self.errors}
        gpu_util = [
            row["gpu_utilization_percent"]
            for row in self.rows
            if row["gpu_utilization_percent"] is not None
        ]
        memory_used = [
            row["memory_used_mib"]
            for row in self.rows
            if row["memory_used_mib"] is not None
        ]
        memory_total = [
            row["memory_total_mib"]
            for row in self.rows
            if row["memory_total_mib"] is not None
        ]
        power = [
            row["power_watts"] for row in self.rows if row["power_watts"] is not None
        ]
        gaps = [row["gap_seconds"] for row in self.rows if row["gap_seconds"] is not None]
        return {
            "samples": len(self.rows),
            "errors": self.errors,
            "gpu_uuid_values": sorted({row["gpu_uuid"] for row in self.rows}),
            "gpu_utilization_available_samples": len(gpu_util),
            "gpu_utilization_p50_percent": percentile(gpu_util, 0.5),
            "gpu_utilization_p90_percent": percentile(gpu_util, 0.9),
            "peak_memory_used_mib": max(memory_used) if memory_used else None,
            "memory_total_mib": max(memory_total) if memory_total else None,
            "mean_power_watts": sum(power) / len(power) if power else None,
            "minimum_host_mem_available_mib": min(
                row["host_mem_available_mib"] for row in self.rows
            ),
            "maximum_sample_gap_seconds": max(gaps) if gaps else 0.0,
        }


def runtime_versions() -> dict[str, Any]:
    result: dict[str, Any] = {"python": os.sys.version}
    for package in ("rslearn", "torch", "lightning"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    try:
        import torch

        result.update(
            {
                "torch_cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    except Exception as exc:  # pragma: no cover - server-only dependency
        result["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def run_candidate(
    *,
    batch_size: int,
    workers: int,
    source_dataset: Path,
    exact_inputs: Path,
    candidate_root: Path,
    template_config: Path,
    rslearn: Path,
    model_path: str,
    gpu_index: str,
    expected_gpu_uuid: str,
    maximum_memory_mib: float,
    model_env: str,
    output_layer: str,
) -> dict[str, Any]:
    identifier = candidate_id(batch_size, workers)
    if candidate_root.exists():
        raise FileExistsError(f"candidate root already exists: {candidate_root}")
    dataset_root = candidate_root / "dataset"
    result_root = candidate_root / "result"
    config_path = candidate_root / "resolved_config.yaml"
    candidate_root.mkdir(parents=True)
    view = build_view(
        source_dataset,
        exact_inputs,
        dataset_root,
        output_layers=(output_layer,),
    )
    render_config(
        template_config,
        config_path,
        batch_size,
        workers,
        model_env,
        output_layer,
    )
    records = validate_inputs(exact_inputs, dataset_root)
    processes = gpu_processes(gpu_index)
    if processes:
        raise RuntimeError(f"selected GPU became occupied before {identifier}: {processes}")

    result_root.mkdir()
    log_path = result_root / "rslearn.log"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "DATASET_PATH": dataset_root.as_posix(),
            model_env: model_path,
            "CUDA_VISIBLE_DEVICES": expected_gpu_uuid,
        }
    )
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    telemetry = TelemetrySampler(gpu_index)
    telemetry.start()
    command = [
        rslearn.as_posix(),
        "model",
        "predict",
        "--config",
        config_path.as_posix(),
    ]
    process: subprocess.Popen[str] | None = None
    return_code = -999
    execution_error = None
    forwarded_signals: list[int] = []

    def forward_signal(signum: int, _frame: Any) -> None:
        forwarded_signals.append(signum)
        if process is not None and process.poll() is None:
            process.send_signal(signum)

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)
    }
    for signum in previous_handlers:
        signal.signal(signum, forward_signal)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = process.wait()
    except Exception as exc:
        execution_error = f"{type(exc).__name__}: {exc}"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        try:
            telemetry.stop()
        except Exception as exc:
            execution_error = execution_error or f"telemetry stop: {type(exc).__name__}: {exc}"
    wall_seconds = time.monotonic() - start
    telemetry_path = result_root / "telemetry.json"
    atomic_create(telemetry_path, canonical_bytes(telemetry.rows))
    telemetry_summary = telemetry.summary()
    failure_reasons = []
    if execution_error is not None:
        failure_reasons.append(f"execution_exception:{execution_error}")
    if return_code != 0:
        failure_reasons.append(f"rslearn_exit_{return_code}")
    if forwarded_signals:
        failure_reasons.append(f"runner_forwarded_signals:{forwarded_signals}")
    if telemetry_summary.get("errors"):
        failure_reasons.append("telemetry_error")
    if telemetry_summary.get("gpu_uuid_values") != [expected_gpu_uuid]:
        failure_reasons.append("gpu_uuid_mismatch")
    if telemetry_summary.get("maximum_sample_gap_seconds", 0.0) > 3.0:
        failure_reasons.append("telemetry_gap_above_3s")
    if telemetry_summary.get("gpu_utilization_available_samples", 0) == 0:
        failure_reasons.append("gpu_utilization_unavailable")
    peak_memory = telemetry_summary.get("peak_memory_used_mib")
    if peak_memory is None:
        failure_reasons.append("gpu_memory_telemetry_unavailable")
    elif peak_memory > maximum_memory_mib:
        failure_reasons.append("peak_gpu_memory_above_gate")

    outputs = []
    if return_code == 0:
        try:
            # Fresh candidate roots are proven absent before view creation.  Do not
            # use wall-clock mtime as freshness evidence because NFS/NTP corrections
            # can move file timestamps backward during a run.
            outputs = output_inventory(dataset_root, output_layer, records, 0.0)
        except Exception as exc:
            failure_reasons.append(
                f"output_inventory_exception:{type(exc).__name__}:{exc}"
            )
    if len(outputs) != len(records):
        failure_reasons.append("incomplete_output_inventory")
    try:
        validate_inputs(exact_inputs, dataset_root)
    except Exception as exc:
        failure_reasons.append(f"post_run_input_exception:{type(exc).__name__}:{exc}")
    crops = len(records) * CROPS_PER_WINDOW
    result = {
        "schema": "olmoearth-release-batch-candidate-v1",
        "status": "pass_execution" if not failure_reasons else "failed",
        "candidate_id": identifier,
        "batch_size": batch_size,
        "num_workers": workers,
        "started_at": started_at,
        "fresh_candidate_root_created_by_runner": True,
        "wall_seconds": round(wall_seconds, 6),
        "expected_crops": crops,
        "end_to_end_crops_per_second": crops / wall_seconds,
        "source_view": view,
        "source_view_manifest_sha256": file_sha256(
            dataset_root / "audit_view_manifest.json"
        ),
        "source_view_config_sha256": file_sha256(dataset_root / "config.json"),
        "config_path": config_path.as_posix(),
        "config_sha256": file_sha256(config_path),
        "log_path": log_path.as_posix(),
        "log_sha256": file_sha256(log_path),
        "telemetry_path": telemetry_path.as_posix(),
        "telemetry_sha256": file_sha256(telemetry_path),
        "telemetry_summary": telemetry_summary,
        "executed_command": command,
        "outputs": outputs,
        "failure_reasons": failure_reasons,
    }
    result_path = result_root / "candidate_summary.json"
    atomic_create(result_path, canonical_bytes(result))
    marker_name = "CANDIDATE_COMPLETE.json" if not failure_reasons else "CANDIDATE_FAILED.json"
    atomic_create(
        result_root / marker_name,
        canonical_bytes(
            {
                "schema": "olmoearth-release-batch-candidate-completion-v1",
                "status": result["status"],
                "candidate_summary_sha256": file_sha256(result_path),
            }
        ),
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--release", choices=sorted(RELEASES), default="v1_2")
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--template-config", type=Path, required=True)
    parser.add_argument("--rslearn", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--gpu-lock", type=Path, default=Path("/home/work/data/.jobs/gpu0.lock"))
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    parser.add_argument("--maximum-memory-mib", type=float, default=115000.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu != "0":
        raise SystemExit("the batch gate is hard-pinned to physical GPU0")
    args.source_dataset = args.source_dataset.resolve()
    args.exact_inputs = args.exact_inputs.resolve()
    args.checkpoint_manifest = args.checkpoint_manifest.resolve()
    args.template_config = args.template_config.resolve()
    args.rslearn = args.rslearn.resolve()
    args.output_root = args.output_root.resolve()
    fixed_lock = Path("/home/work/data/.jobs/gpu0.lock")
    if args.gpu_lock != fixed_lock:
        raise SystemExit(f"GPU0 lock path is fixed at {fixed_lock}")
    if len(set(args.candidate)) != len(args.candidate):
        raise SystemExit("candidate batch/worker pairs must be unique")
    if args.output_root.exists():
        raise SystemExit(f"refusing an existing batch-gate root: {args.output_root}")
    free_gib = shutil.disk_usage(args.output_root.parent).free / 1024**3
    if free_gib < args.minimum_free_gib:
        raise SystemExit(
            f"only {free_gib:.1f} GiB free; batch gate requires {args.minimum_free_gib:.1f} GiB"
        )
    release = RELEASES[args.release]
    initial_checkpoint_manifest_sha = file_sha256(args.checkpoint_manifest)
    initial_checkpoint_models = normalize_checkpoint_manifest(
        read_json(args.checkpoint_manifest)
    )
    checkpoints = validate_checkpoints(args.checkpoint_manifest)
    if (
        file_sha256(args.checkpoint_manifest) != initial_checkpoint_manifest_sha
        or normalize_checkpoint_manifest(read_json(args.checkpoint_manifest))
        != initial_checkpoint_models
    ):
        raise SystemExit("checkpoint manifest changed during batch preflight")
    model = checkpoints[release["repo_id"]]
    rslearn_runtime_fingerprint = fingerprint_rslearn_runtime(args.rslearn)
    current_runtime_versions = validate_launcher_runtime_binding(
        runtime_versions(), Path(os.sys.executable), rslearn_runtime_fingerprint
    )
    initial_audit_code_contract = batch_audit_code_contract()
    selected_uuid = gpu_uuid(args.gpu)
    if gpu_processes(args.gpu):
        raise SystemExit("selected GPU is occupied before the batch gate")

    args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with args.gpu_lock.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"GPU lock is held: {args.gpu_lock}") from exc
        args.output_root.mkdir(parents=True)
        preflight = {
            "schema": "olmoearth-release-batch-gate-preflight-v1",
            "selected_gpu": args.gpu,
            "selected_gpu_uuid": selected_uuid,
            "gpu_lock": args.gpu_lock.as_posix(),
            "checkpoint_manifest_sha256": initial_checkpoint_manifest_sha,
            "exact_inputs_sha256": file_sha256(args.exact_inputs),
            "template_config_sha256": file_sha256(args.template_config),
            "candidates": [
                {"batch_size": batch, "num_workers": workers}
                for batch, workers in args.candidate
            ],
            "free_gib_before": free_gib,
            "runtime_versions": current_runtime_versions,
            "rslearn_runtime_fingerprint": rslearn_runtime_fingerprint,
            "audit_code_contract": initial_audit_code_contract,
            "ready": True,
        }
        atomic_create(args.output_root / "preflight.json", canonical_bytes(preflight))
        results = []
        for batch_size, workers in args.candidate:
            result = run_candidate(
                batch_size=batch_size,
                workers=workers,
                source_dataset=args.source_dataset,
                exact_inputs=args.exact_inputs,
                candidate_root=args.output_root / candidate_id(batch_size, workers),
                template_config=args.template_config,
                rslearn=args.rslearn,
                model_path=model["snapshot_path"],
                gpu_index=args.gpu,
                expected_gpu_uuid=selected_uuid,
                maximum_memory_mib=args.maximum_memory_mib,
                model_env=release["model_env"],
                output_layer=release["output_layer"],
            )
            results.append(result)
            atomic_replace(
                args.output_root / "PROGRESS.json",
                canonical_bytes(
                    {
                        "schema": "olmoearth-release-batch-gate-progress-v1",
                        "completed_candidates": [value["candidate_id"] for value in results],
                        "last_status": result["status"],
                    }
                ),
            )
            if result["status"] != "pass_execution":
                raise RuntimeError(
                    f"candidate {result['candidate_id']} failed: {result['failure_reasons']}"
                )

        post_run_rslearn_runtime_verified = False
        post_run_rslearn_runtime_error = None
        try:
            live_rslearn_runtime = fingerprint_rslearn_runtime(args.rslearn)
            if live_rslearn_runtime != rslearn_runtime_fingerprint:
                raise ValueError(
                    "rslearn executable/interpreter/package source changed during batch execution"
                )
            post_run_rslearn_runtime_verified = True
        except Exception as exc:
            post_run_rslearn_runtime_error = f"{type(exc).__name__}: {exc}"

        post_run_checkpoints_verified = False
        post_run_checkpoint_error = None
        live_checkpoint_manifest_sha: str | None = None
        live_checkpoint_models: dict[str, dict[str, Any]] | None = None
        try:
            live_checkpoint_manifest_sha = file_sha256(args.checkpoint_manifest)
            if live_checkpoint_manifest_sha != initial_checkpoint_manifest_sha:
                raise ValueError("checkpoint manifest changed during batch execution")
            validate_checkpoints(args.checkpoint_manifest)
            live_checkpoint_models = normalize_checkpoint_manifest(
                read_json(args.checkpoint_manifest)
            )
            if live_checkpoint_models != initial_checkpoint_models:
                raise ValueError(
                    "checkpoint revisions or file hashes changed during batch execution"
                )
            post_run_checkpoints_verified = True
        except Exception as exc:
            post_run_checkpoint_error = f"{type(exc).__name__}: {exc}"
        checkpoint_marker_path = (
            args.output_root / "POST_RUN_CHECKPOINTS_VERIFICATION.json"
        )
        atomic_create(
            checkpoint_marker_path,
            canonical_bytes(
                {
                    "schema": "olmoearth-release-batch-post-run-checkpoint-verification-v1",
                    "status": "verified" if post_run_checkpoints_verified else "failed",
                    "checkpoint_manifest_path": args.checkpoint_manifest.as_posix(),
                    "initial_checkpoint_manifest_sha256": initial_checkpoint_manifest_sha,
                    "live_checkpoint_manifest_sha256": live_checkpoint_manifest_sha,
                    "initial_checkpoint_models": initial_checkpoint_models,
                    "live_checkpoint_models": live_checkpoint_models,
                    "error": post_run_checkpoint_error,
                }
            ),
        )

        post_run_audit_code_verified = False
        post_run_audit_code_error = None
        live_audit_code_contract: dict[str, Any] | None = None
        try:
            live_audit_code_contract = batch_audit_code_contract()
            if live_audit_code_contract != initial_audit_code_contract:
                raise ValueError(
                    "batch runner or directly imported local helper changed during execution"
                )
            post_run_audit_code_verified = True
        except Exception as exc:
            post_run_audit_code_error = f"{type(exc).__name__}: {exc}"
        audit_code_marker_path = args.output_root / "POST_RUN_AUDIT_CODE_VERIFICATION.json"
        atomic_create(
            audit_code_marker_path,
            canonical_bytes(
                {
                    "schema": "olmoearth-release-batch-post-run-audit-code-verification-v1",
                    "status": "verified" if post_run_audit_code_verified else "failed",
                    "initial_audit_code_contract": initial_audit_code_contract,
                    "live_audit_code_contract": live_audit_code_contract,
                    "error": post_run_audit_code_error,
                }
            ),
        )

        execution_verified = (
            post_run_rslearn_runtime_verified
            and post_run_checkpoints_verified
            and post_run_audit_code_verified
        )

        summary = {
            "schema": "olmoearth-release-batch-gate-run-v1",
            "status": "execution_complete_analysis_pending"
            if execution_verified
            else "failed_post_run_provenance_verification",
            "preflight_sha256": file_sha256(args.output_root / "preflight.json"),
            "release_id": release["release_id"],
            "repo_id": release["repo_id"],
            "revision": model["revision"],
            "selected_gpu": args.gpu,
            "selected_gpu_uuid": selected_uuid,
            "post_run_rslearn_runtime_verified": post_run_rslearn_runtime_verified,
            "post_run_rslearn_runtime_error": post_run_rslearn_runtime_error,
            "post_run_checkpoints_verified": post_run_checkpoints_verified,
            "post_run_checkpoint_error": post_run_checkpoint_error,
            "post_run_checkpoint_verification": {
                "path": checkpoint_marker_path.as_posix(),
                "sha256": file_sha256(checkpoint_marker_path),
            },
            "audit_code_contract": initial_audit_code_contract,
            "post_run_audit_code_verified": post_run_audit_code_verified,
            "post_run_audit_code_error": post_run_audit_code_error,
            "post_run_audit_code_verification": {
                "path": audit_code_marker_path.as_posix(),
                "sha256": file_sha256(audit_code_marker_path),
            },
            "candidates": results,
            "claims_allowed": ["batch_throughput_measurement_after_numerical_equivalence"],
            "claims_forbidden": [
                "accuracy_improvement",
                "release_compatibility",
                "generalization",
            ],
        }
        summary_path = args.output_root / "run_summary.json"
        atomic_create(summary_path, canonical_bytes(summary))
        marker_name = (
            "EXECUTION_COMPLETE.json"
            if execution_verified
            else "EXECUTION_FAILED.json"
        )
        atomic_create(
            args.output_root / marker_name,
            canonical_bytes(
                {
                    "schema": "olmoearth-release-batch-gate-execution-completion-v1",
                    "status": "complete"
                    if execution_verified
                    else "failed",
                    "run_summary_sha256": file_sha256(summary_path),
                    "post_run_rslearn_runtime_verified": post_run_rslearn_runtime_verified,
                    "post_run_checkpoints_verified": post_run_checkpoints_verified,
                    "checkpoint_manifest_sha256": initial_checkpoint_manifest_sha,
                    "post_run_checkpoint_verification_sha256": file_sha256(
                        checkpoint_marker_path
                    ),
                    "post_run_audit_code_verified": post_run_audit_code_verified,
                    "audit_code_contract_sha256": initial_audit_code_contract[
                        "inventory_sha256"
                    ],
                    "post_run_audit_code_verification_sha256": file_sha256(
                        audit_code_marker_path
                    ),
                }
            ),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        if not execution_verified:
            raise RuntimeError(
                "batch execution failed post-run provenance verification: "
                f"runtime={post_run_rslearn_runtime_error}; "
                f"checkpoints={post_run_checkpoint_error}; "
                f"audit_code={post_run_audit_code_error}"
            )


if __name__ == "__main__":
    main()
