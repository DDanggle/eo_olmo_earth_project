#!/usr/bin/env python3
"""Analyze the frozen 216-pair OlmoEarth release audit without task labels.

The program deliberately separates three roles:

* all 216 pairs may contribute to preregistered, split-stratified descriptive
  release-drift summaries;
* only the 30 calibration spatial clusters may fit a bridge or choose ridge
  regularization; and
* only the 16 sealed-test clusters may contribute to compatibility endpoints.

The embargo and previously disclosed audit clusters never fit a parameter and
never enter a headline compatibility metric.  The output therefore measures
representation/cache identity continuity, not accuracy or semantic utility.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import hashlib
import importlib
import io
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from finalize_olmo_release_full import (
    finalizer_code_contract,
    validate_finalizer_code_contract,
    validate_finalizer_code_evidence,
)
from run_olmo_release_full import (
    build_python_code_contract,
    full_runner_code_contract,
    validate_full_release_command,
    validate_full_runner_code_contract,
    validate_full_runner_code_evidence,
    validate_python_code_contract,
)


EXPECTED_EVIDENCE_SCHEMA = "olmoearth-release-full-paired-evidence-v1"
EXPECTED_PAIR_MARKER_SCHEMA = "olmoearth-release-full-paired-evidence-completion-v1"
EXPECTED_EXACT_SCHEMA = "olmoearth-release-exact-input-selection-v1"
EXPECTED_EXACT_MARKER_SCHEMA = "olmoearth-release-exact-input-completion-v1"
EXPECTED_SPLIT_SCHEMA = "olmoearth-release-spatial-split-v1"
EXPECTED_SPLIT_MARKER_SCHEMA = "olmoearth-release-spatial-split-completion-v1"
EXPECTED_SPLIT_CLUSTERS = {
    "calibration": 30,
    "embargo": 6,
    "sealed_test": 16,
    "disclosed_audit": 2,
}
CALIBRATION_X = (22528, 23552, 24576, 25600, 26624)
EMBARGO_X = (27648,)
EAST_X = (28672, 29696, 30720)
EXPECTED_Y = (-372736, -371712, -370688, -369664, -368640, -367616)
EXPECTED_DISCLOSED_EAST_KEYS = ("28672_-372736", "29696_-367616")
EXPECTED_YEARS = (2023, 2024, 2025, 2026)
LATTICE_SIDE = 16
QUERY_LATTICE_POSITIONS = tuple(range(1, LATTICE_SIDE, 2))
RETRIEVAL_K_VALUES = (1, 5, 10)
NEIGHBOR_K_VALUES = (1, 5, 10)
RIDGE_ALPHA_MULTIPLIERS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
BRIDGE_METHODS = (
    "identity_no_bridge",
    "mean_shift_translation_only",
    "translated_orthogonal_procrustes",
    "affine_ridge",
)
NEAR_TIE_COSINE_TOLERANCE = 1e-6
NATIVE_R1_FLOOR = 1.0
CROSS_R1_FLOOR = 0.95
CLUSTERED_R1_LOWER_FLOOR = 0.95
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MINIMUM_FORBIDDEN_CLAIMS = {
    "task_accuracy",
    "negative_transfer_reduction",
    "cloud_robustness",
    "korean_population_generalization",
}
EXPECTED_RELEASE_RUNS = {
    "v1": {
        "release_id": "olmoearth_v1_base",
        "repo_id": "allenai/OlmoEarth-v1-Base",
        "output_layer": "embeddings_full_v1_legacy",
    },
    "v1_2": {
        "release_id": "olmoearth_v1_2_base",
        "repo_id": "allenai/OlmoEarth-v1_2-Base",
        "output_layer": "embeddings_full_v1_2_legacy",
    },
}
ANALYZER_CODE_SCHEMA = "olmoearth-release-full-analyzer-code-contract-v1"
ANALYZER_CODE_OWNER_ROLE = "sealed_release_analyzer"
ANALYZER_POST_CODE_SCHEMA = (
    "olmoearth-release-full-analyzer-post-run-code-verification-v1"
)


def analyzer_helper_paths() -> dict[str, Path]:
    """Every local Python module imported directly by this analyzer."""

    return {
        "finalize_olmo_release_full": Path(
            finalizer_code_contract.__code__.co_filename
        ),
        "run_olmo_release_full": Path(full_runner_code_contract.__code__.co_filename),
    }


def analyzer_code_contract() -> dict[str, Any]:
    return build_python_code_contract(
        schema=ANALYZER_CODE_SCHEMA,
        owner_role=ANALYZER_CODE_OWNER_ROLE,
        owner_module="analyze_olmo_release_full",
        owner_path=Path(__file__),
        direct_local_helpers=analyzer_helper_paths(),
    )


def validate_analyzer_code_contract(
    value: Any, *, require_live_match: bool = True
) -> dict[str, Any]:
    return validate_python_code_contract(
        value,
        schema=ANALYZER_CODE_SCHEMA,
        owner_role=ANALYZER_CODE_OWNER_ROLE,
        owner_module="analyze_olmo_release_full",
        owner_path=Path(__file__),
        direct_local_helpers=analyzer_helper_paths(),
        require_live_match=require_live_match,
    )


def verify_analyzer_code_stability(initial: dict[str, Any]) -> dict[str, Any]:
    """Fail unless analyzer and direct local helpers match the sealed inventory."""

    validate_analyzer_code_contract(initial, require_live_match=False)
    live = analyzer_code_contract()
    if live != initial:
        raise ValueError(
            "analyzer or directly imported local helper changed during analysis"
        )
    validate_analyzer_code_contract(live, require_live_match=True)
    return live


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not all(isinstance(value, dict) for value in records):
        raise ValueError(f"JSONL contains a non-object record: {path}")
    return records


def atomic_create(path: Path, content: bytes) -> None:
    """Create immutable evidence, or accept an already byte-identical file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to replace different evidence: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _stable_live_inventory(path: Path, expected_bytes: int) -> dict[str, Any]:
    before = path.stat()
    if before.st_size != int(expected_bytes):
        raise ValueError(f"file-size drift: {path}")
    digest = file_sha256(path)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError(f"file changed while hashing: {path}")
    return {"path": path.as_posix(), "bytes": after.st_size, "sha256": digest}


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"missing or malformed SHA-256: {label}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"non-hexadecimal SHA-256: {label}") from exc
    return value


def _identity_fields(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        record["window_name"],
        record["spatial_cluster_id"],
        record["input_bundle_identity"],
    )


def _spatial_coordinates(spatial_key: str) -> tuple[int, int]:
    try:
        x_text, y_text = spatial_key.split("_", maxsplit=1)
        return int(x_text), int(y_text)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"invalid frozen spatial key: {spatial_key!r}") from exc


def _expected_split_for_key(spatial_key: str) -> str:
    x_value, y_value = _spatial_coordinates(spatial_key)
    if y_value not in EXPECTED_Y:
        raise ValueError(f"spatial key is outside the frozen y-grid: {spatial_key}")
    if x_value in CALIBRATION_X:
        return "calibration"
    if x_value in EMBARGO_X:
        return "embargo"
    if x_value in EAST_X:
        return (
            "disclosed_audit"
            if spatial_key in EXPECTED_DISCLOSED_EAST_KEYS
            else "sealed_test"
        )
    raise ValueError(f"spatial key is outside the frozen x-grid: {spatial_key}")


def cross_bind_paired_outputs_to_release_runs(
    pairs: dict[str, dict[str, Any]],
    v1_outputs: dict[str, dict[str, Any]],
    v1_2_outputs: dict[str, dict[str, Any]],
    v1_health: dict[str, dict[str, Any]] | None = None,
    v1_2_health: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Require pair inventories and identities to equal both canonical run maps."""
    if set(pairs) != set(v1_outputs) or set(pairs) != set(v1_2_outputs):
        raise ValueError("paired outputs and release-run output identities differ")
    for sample_id in sorted(pairs):
        pair = pairs[sample_id]
        for release_field, pair_health_field, run_outputs, run_health in (
            ("v1_output", "v1_value_health", v1_outputs, v1_health),
            ("v1_2_output", "v1_2_value_health", v1_2_outputs, v1_2_health),
        ):
            run_output = run_outputs[sample_id]
            run_identity = (
                run_output.get("window"),
                run_output.get("spatial_cluster_id"),
                run_output.get("input_bundle_identity"),
            )
            pair_identity = (
                pair.get("window"),
                pair.get("spatial_cluster_id"),
                pair.get("input_bundle_identity"),
            )
            if run_identity != pair_identity:
                raise ValueError(
                    f"paired/run output identity mismatch: {sample_id}/{release_field}"
                )
            canonical_inventory = {
                key: run_output.get(key) for key in ("path", "bytes", "sha256", "mtime_ns")
            }
            if pair.get(release_field) != canonical_inventory:
                raise ValueError(
                    f"paired/run output inventory mismatch: {sample_id}/{release_field}"
                )
            if run_health is not None:
                if set(run_health) != set(pairs):
                    raise ValueError("paired outputs and release-run health identities differ")
                health = run_health[sample_id]
                structural_fields = (
                    "height",
                    "width",
                    "count",
                    "dtypes",
                    "crs",
                    "transform",
                    "bounds",
                    "nodata",
                )
                value_fields = (
                    "usable_tokens",
                    "nonzero_usable_tokens",
                    "finite_values",
                    "total_values",
                    "all_values_finite",
                )
                if pair.get("raster_contract") != {
                    key: health.get(key) for key in structural_fields
                }:
                    raise ValueError(
                        f"paired/run raster-health structure mismatch: {sample_id}/{release_field}"
                    )
                if pair.get(pair_health_field) != {
                    key: health.get(key) for key in value_fields
                }:
                    raise ValueError(
                        f"paired/run raster value-health mismatch: {sample_id}/{release_field}"
                    )


@dataclass(frozen=True)
class FrozenEvidence:
    exact_records: dict[str, dict[str, Any]]
    pairs: dict[str, dict[str, Any]]
    split_by_sample: dict[str, str]
    split_by_cluster: dict[str, str]
    evidence_hashes: dict[str, str]
    execution_contract: dict[str, Any] | None
    code_contracts: dict[str, dict[str, Any]] | None = None


def _json_safe_runtime_value(value: Any) -> Any:
    """Convert build metadata to a deterministic JSON value without repr addresses."""
    if isinstance(value, np.generic):
        return _json_safe_runtime_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in numerical runtime configuration")
        return value
    if isinstance(value, dict):
        return {
            str(key): _json_safe_runtime_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_runtime_value(item) for item in value]
    return str(value)


def _module_binary_identity(module_names: Sequence[str]) -> dict[str, Any]:
    """Hash the first importable numerical extension from an explicit allowlist."""
    selected = None
    for module_name in module_names:
        try:
            selected = importlib.import_module(module_name)
            break
        except ImportError:
            continue
    if selected is None:
        raise RuntimeError(f"none of the required runtime modules import: {module_names}")
    module_path_text = getattr(selected, "__file__", None)
    if not isinstance(module_path_text, str):
        raise RuntimeError(f"runtime module has no on-disk identity: {selected.__name__}")
    module_path = Path(module_path_text).resolve()
    inventory = _stable_live_inventory(module_path, module_path.stat().st_size)
    return {"module": selected.__name__, **inventory}


def _numpy_configuration_identity() -> dict[str, Any]:
    raw_configuration = getattr(np.__config__, "CONFIG", None)
    if isinstance(raw_configuration, dict):
        configuration: Any = _json_safe_runtime_value(raw_configuration)
        source = "numpy.__config__.CONFIG"
    else:  # NumPy 1.x compatibility on execution hosts.
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            np.__config__.show()
        configuration = {"show_text": stream.getvalue().strip()}
        source = "numpy.__config__.show"

    blas_lapack: dict[str, Any] = {}

    def visit(value: Any, path: tuple[str, ...] = ()) -> None:
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            key_text = str(key)
            current = (*path, key_text)
            if "blas" in key_text.lower() or "lapack" in key_text.lower():
                blas_lapack["/".join(current)] = item
            else:
                visit(item, current)

    visit(configuration)
    if not blas_lapack:
        # Older NumPy emits free-form text.  Binding the full configuration is
        # safer than pretending that a partial parser recovered the backend.
        blas_lapack = {"full_configuration_fallback": configuration}
    configuration_sha = hashlib.sha256(canonical_bytes(configuration)).hexdigest()
    blas_lapack_sha = hashlib.sha256(canonical_bytes(blas_lapack)).hexdigest()
    return {
        "version": np.__version__,
        "configuration_source": source,
        "configuration": configuration,
        "configuration_sha256": configuration_sha,
        "blas_lapack_configuration": blas_lapack,
        "blas_lapack_configuration_sha256": blas_lapack_sha,
        "core_extension": _module_binary_identity(
            ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath")
        ),
        "linalg_extension": _module_binary_identity(("numpy.linalg._umath_linalg",)),
    }


def _thread_environment_identity() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in THREAD_ENVIRONMENT_VARIABLES:
        value = os.environ.get(name)
        if value is not None and re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value) is None:
            raise RuntimeError(
                f"unsafe or non-numeric numerical thread setting in {name}"
            )
        result[name] = value
    return result


def _rasterio_runtime_identity(rasterio: Any) -> dict[str, Any]:
    with rasterio.Env() as environment:
        raw_drivers = environment.drivers()
    drivers = [
        {"short_name": str(name), "description": str(description)}
        for name, description in sorted(raw_drivers.items())
    ]
    driver_sha = hashlib.sha256(canonical_bytes(drivers)).hexdigest()
    return {
        "version": str(rasterio.__version__),
        "gdal_version": str(getattr(rasterio, "__gdal_version__", "unknown")),
        "proj_version": str(getattr(rasterio, "__proj_version__", "unknown")),
        "geos_version": str(getattr(rasterio, "__geos_version__", "unknown")),
        "package": _module_binary_identity(("rasterio",)),
        "core_extension": _module_binary_identity(("rasterio._base",)),
        "gdal_driver_count": len(drivers),
        "gdal_drivers": drivers,
        "gdal_drivers_sha256": driver_sha,
    }


def analysis_runtime_identity() -> dict[str, Any]:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - server integration dependency
        raise RuntimeError("rasterio is required before sealing the analysis runtime") from exc
    identity = {
        "python_executable": Path(sys.executable).resolve().as_posix(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": _numpy_configuration_identity(),
        "rasterio_gdal": _rasterio_runtime_identity(rasterio),
        "numerical_thread_environment": _thread_environment_identity(),
    }
    identity["fingerprint_sha256"] = hashlib.sha256(canonical_bytes(identity)).hexdigest()
    return identity


def assert_analysis_runtime_unchanged(
    expected: dict[str, Any], observed: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Fail closed if the numerical/geospatial runtime changed after sealing."""
    current = analysis_runtime_identity() if observed is None else observed
    if canonical_bytes(current) != canonical_bytes(expected):
        raise ValueError("analysis numerical/geospatial runtime drifted after sealing")
    return current


def create_preanalysis_lock(
    output_dir: Path,
    evidence: FrozenEvidence,
    *,
    output_hash_workers: int,
    retrieval_chunk_size: int,
    argv: Sequence[str],
    runtime_identity: dict[str, Any] | None = None,
) -> tuple[Path, str, dict[str, Any]]:
    """Irreversibly seal one analysis invocation before any output-raster read."""
    if evidence.execution_contract is None:
        raise ValueError("preanalysis lock requires both canonical release-run contracts")
    if evidence.code_contracts is None:
        raise ValueError("preanalysis lock requires full-runner/finalizer code contracts")
    upstream_code_contracts = {
        "full_runner": validate_full_runner_code_contract(
            evidence.code_contracts.get("full_runner"), require_live_match=True
        ),
        "finalizer": validate_finalizer_code_contract(
            evidence.code_contracts.get("finalizer"), require_live_match=True
        ),
    }
    initial_analyzer_code_contract = analyzer_code_contract()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing one-time analysis directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    payload = {
        "schema": "olmoearth-release-full-preanalysis-lock-v1",
        "status": "sealed_before_any_output_raster_read",
        "one_time_analysis_directory": output_dir.resolve().as_posix(),
        "analyzer": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": initial_analyzer_code_contract["owner"]["sha256"],
        },
        "analyzer_code_contract": initial_analyzer_code_contract,
        "upstream_code_contracts": upstream_code_contracts,
        "metric_contract": {
            "lattice_side": LATTICE_SIDE,
            "query_lattice_positions": list(QUERY_LATTICE_POSITIONS),
            "retrieval_k_values": list(RETRIEVAL_K_VALUES),
            "neighbor_k_values": list(NEIGHBOR_K_VALUES),
            "ridge_alpha_multipliers": list(RIDGE_ALPHA_MULTIPLIERS),
            "bridge_methods": list(BRIDGE_METHODS),
            "near_tie_cosine_tolerance": NEAR_TIE_COSINE_TOLERANCE,
            "native_recall_at_1_floor": NATIVE_R1_FLOOR,
            "cross_recall_at_1_floor": CROSS_R1_FLOOR,
            "clustered_lower_bound_floor": CLUSTERED_R1_LOWER_FLOOR,
            "split_rule": {
                "calibration_x": list(CALIBRATION_X),
                "embargo_x": list(EMBARGO_X),
                "east_x": list(EAST_X),
                "disclosed_east_keys": list(EXPECTED_DISCLOSED_EAST_KEYS),
            },
            "bridge_fit": "calibration 8x8 anchor tokens only",
            "ridge_selection": "calibration spatial-x-block validation only",
            "ridge_selection_metric": "mean exact-paired cosine on the fixed 8x8 calibration query lattice",
            "lattice_axis_rule": "floor((index+0.5)*axis_length/16)",
            "retrieval_rule": "row-L2-normalized cosine; exact ties and fixed-tolerance near ties reported with optimistic/stable/pessimistic bounds",
            "clustered_retrieval_gate": "minimum leave-one-location-cluster-out R@1; normal interval remains descriptive only",
            "manifest_contrast_score": "one minus clipped cosine of pooled site-year embeddings",
            "headline_evaluation": "sealed_test only",
        },
        "invocation": {
            "argv": list(argv),
            "working_directory": Path.cwd().resolve().as_posix(),
            "output_hash_workers": output_hash_workers,
            "retrieval_chunk_size": retrieval_chunk_size,
        },
        "runtime": runtime_identity or analysis_runtime_identity(),
        "frozen_evidence": evidence.evidence_hashes,
        "release_execution": evidence.execution_contract,
        "sealed_test_policy": {
            "single_frozen_invocation": True,
            "no_hyperparameter_or_method_selection_on_sealed_test": True,
            "existing_or_partial_output_directory_is_not_resumable": True,
        },
    }
    path = output_dir / "PREANALYSIS_LOCK.json"
    atomic_create(path, canonical_bytes(payload))
    return path, file_sha256(path), payload


def validate_analyzer_code_evidence(
    *,
    lock_payload: dict[str, Any],
    analysis_summary: dict[str, Any],
    analysis_root: Path,
    completion: dict[str, Any] | None = None,
    require_live_match: bool = True,
) -> dict[str, Any]:
    """Validate lock→post-run proof→summary→optional completion code bindings."""

    if (
        lock_payload.get("schema")
        != "olmoearth-release-full-preanalysis-lock-v1"
        or lock_payload.get("status") != "sealed_before_any_output_raster_read"
    ):
        raise ValueError("preanalysis lock schema/status drift")
    contract = validate_analyzer_code_contract(
        lock_payload.get("analyzer_code_contract"),
        require_live_match=require_live_match,
    )
    analyzer_record = lock_payload.get("analyzer", {})
    if analyzer_record != {
        "path": contract["owner"]["path"],
        "sha256": contract["owner"]["sha256"],
    }:
        raise ValueError("preanalysis lock analyzer identity drift")
    upstream = lock_payload.get("upstream_code_contracts")
    if not isinstance(upstream, dict) or set(upstream) != {"full_runner", "finalizer"}:
        raise ValueError("preanalysis lock upstream code-contract inventory drift")
    validated_upstream = {
        "full_runner": validate_full_runner_code_contract(
            upstream["full_runner"], require_live_match=require_live_match
        ),
        "finalizer": validate_finalizer_code_contract(
            upstream["finalizer"], require_live_match=require_live_match
        ),
    }
    if (
        analysis_summary.get("analysis_code_contract") != contract
        or analysis_summary.get("analysis_code_sha256")
        != contract["owner"]["sha256"]
        or analysis_summary.get("post_analysis_code_verified") is not True
        or analysis_summary.get("upstream_code_contracts") != validated_upstream
    ):
        raise ValueError("analysis summary/code-contract binding drift")
    descriptor = analysis_summary.get("post_analysis_code_verification")
    if not isinstance(descriptor, dict) or set(descriptor) != {"path", "sha256"}:
        raise ValueError("analysis code-verification descriptor drift")
    expected_marker = analysis_root.resolve() / "POST_ANALYSIS_CODE_VERIFICATION.json"
    marker_path = descriptor.get("path")
    if (
        not isinstance(marker_path, str)
        or not Path(marker_path).is_absolute()
        or Path(marker_path).resolve() != expected_marker
        or not expected_marker.is_file()
        or file_sha256(expected_marker) != descriptor.get("sha256")
    ):
        raise ValueError("analysis code-verification marker drift")
    marker = read_json(expected_marker)
    if (
        not isinstance(marker, dict)
        or set(marker)
        != {
            "schema",
            "status",
            "initial_analyzer_code_contract",
            "live_analyzer_code_contract",
            "error",
        }
        or marker.get("schema") != ANALYZER_POST_CODE_SCHEMA
        or marker.get("status") != "verified"
        or marker.get("initial_analyzer_code_contract") != contract
        or marker.get("live_analyzer_code_contract") != contract
        or marker.get("error") is not None
    ):
        raise ValueError("analysis code-verification marker content drift")
    if completion is not None and (
        completion.get("post_analysis_code_verified") is not True
        or completion.get("analysis_code_contract_sha256")
        != contract["inventory_sha256"]
        or completion.get("analysis_code_sha256") != contract["owner"]["sha256"]
        or completion.get("post_analysis_code_verification_sha256")
        != descriptor["sha256"]
        or completion.get("full_runner_code_contract_sha256")
        != validated_upstream["full_runner"]["inventory_sha256"]
        or completion.get("finalizer_code_contract_sha256")
        != validated_upstream["finalizer"]["inventory_sha256"]
    ):
        raise ValueError("analysis completion/code-contract binding drift")
    return {
        "analyzer": contract,
        "full_runner": validated_upstream["full_runner"],
        "finalizer": validated_upstream["finalizer"],
    }


def _validate_release_execution(
    run_summary_path: Path,
    complete_path: Path,
    expected: dict[str, str],
    expected_run_sha: str,
    expected_complete_sha: str,
    bindings: dict[str, str],
) -> dict[str, Any]:
    """Verify that a finalized release came from the promoted GPU0 contract."""
    result_root = run_summary_path.parent.resolve()
    if (
        run_summary_path.resolve() != result_root / "run_summary.json"
        or complete_path.resolve() != result_root / "RELEASE_COMPLETE.json"
    ):
        raise ValueError(f"release execution path contract drift: {expected['release_id']}")
    run = read_json(run_summary_path)
    complete = read_json(complete_path)
    run_sha = file_sha256(run_summary_path)
    complete_sha = file_sha256(complete_path)
    if run_sha != expected_run_sha or complete_sha != expected_complete_sha:
        raise ValueError(f"final evidence release-summary hash drift: {expected['release_id']}")
    if (
        complete.get("schema") != "olmoearth-release-full-completion-v1"
        or complete.get("status") != "complete"
        or complete.get("run_summary_sha256") != run_sha
    ):
        raise ValueError(f"release completion marker mismatch: {expected['release_id']}")
    release = run.get("release", {})
    if (
        run.get("schema") != "olmoearth-release-full-result-v1"
        or run.get("status") != "complete"
        or run.get("failure_reasons") != []
        or run.get("records") != 216
        or len(run.get("outputs", [])) != 216
        or len(run.get("output_health", [])) != 216
        or release.get("release_id") != expected["release_id"]
        or run.get("repo_id") != expected["repo_id"]
        or release.get("output_layer") != expected["output_layer"]
    ):
        raise ValueError(f"release execution result contract mismatch: {expected['release_id']}")
    for assertion in (
        "post_run_inputs_verified",
        "post_run_checkpoints_verified",
        "post_run_rslearn_runtime_verified",
    ):
        if run.get(assertion) is not True:
            raise ValueError(f"release lacks post-run assertion {assertion}: {expected['release_id']}")
    post_run_markers: dict[str, dict[str, Any]] = {}
    for marker_field in (
        "post_run_checkpoint_verification",
        "post_run_rslearn_runtime_verification",
    ):
        marker = run.get(marker_field, {})
        marker_path = Path(marker.get("path", ""))
        if not marker_path.is_file() or file_sha256(marker_path) != marker.get("sha256"):
            raise ValueError(f"release post-run marker drift: {expected['release_id']}/{marker_field}")
        post_run_markers[marker_field] = read_json(marker_path)

    output_map: dict[str, dict[str, Any]] = {}
    for output in run["outputs"]:
        sample_id = output.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in output_map:
            raise ValueError(f"release output IDs are missing or duplicated: {expected['release_id']}")
        for identity_field in ("window", "spatial_cluster_id", "input_bundle_identity"):
            if not isinstance(output.get(identity_field), str) or not output[identity_field]:
                raise ValueError(
                    f"release output identity is incomplete: {expected['release_id']}/{sample_id}"
                )
        if (
            not isinstance(output.get("path"), str)
            or int(output.get("bytes", -1)) <= 0
            or int(output.get("mtime_ns", -1)) <= 0
        ):
            raise ValueError(
                f"release output inventory is incomplete: {expected['release_id']}/{sample_id}"
            )
        _require_sha(output.get("sha256"), f"{expected['release_id']}/{sample_id}")
        output_map[sample_id] = output
    health_by_sample = {value.get("sample_id"): value for value in run["output_health"]}
    output_ids = set(output_map)
    if len(health_by_sample) != 216 or set(health_by_sample) != output_ids:
        raise ValueError(f"release output-health identity drift: {expected['release_id']}")
    for sample_id, health in health_by_sample.items():
        if (
            health.get("count") != 768
            or health.get("dtypes") != ["float32"]
            or int(health.get("usable_tokens", 0)) < 2
            or int(health.get("nonzero_usable_tokens", 0)) < 2
            or int(health.get("finite_values", -1)) < 0
            or int(health.get("total_values", -1))
            != int(health.get("height", 0))
            * int(health.get("width", 0))
            * int(health.get("count", 0))
        ):
            raise ValueError(f"release output value-health drift: {expected['release_id']}/{sample_id}")

    preflight_path = run_summary_path.parent / "preflight.json"
    if not preflight_path.is_file() or file_sha256(preflight_path) != run.get("preflight_sha256"):
        raise ValueError(f"release preflight drift: {expected['release_id']}")
    preflight = read_json(preflight_path)
    if (
        preflight.get("schema") != "olmoearth-release-full-preflight-v1"
        or preflight.get("status") != "ready"
        or preflight.get("records") != 216
        or preflight.get("spatial_clusters") != 54
        or preflight.get("selected_gpu") != "0"
        or preflight.get("selected_gpu_uuid") != bindings["selected_gpu_uuid"]
    ):
        raise ValueError(f"release preflight contract mismatch: {expected['release_id']}")
    try:
        runner_code_contract = validate_full_runner_code_evidence(
            preflight=preflight,
            run_summary=run,
            completion=complete,
            result_root=result_root,
            require_live_match=True,
        )
    except ValueError as exc:
        raise ValueError(
            f"release full-runner code evidence drift: {expected['release_id']}: {exc}"
        ) from exc
    resolved_config_path = result_root.parent / "resolved_config.yaml"
    if (
        not resolved_config_path.is_file()
        or file_sha256(resolved_config_path) != run.get("resolved_config_sha256")
    ):
        raise ValueError(f"release resolved-config drift: {expected['release_id']}")
    command_contract = validate_full_release_command(
        run.get("executed_command"),
        rslearn_entrypoint=Path(
            preflight["rslearn_runtime_fingerprint"]["entrypoint"]["path"]
        ),
        resolved_config=resolved_config_path,
    )
    if run.get("executed_command_contract") != command_contract:
        raise ValueError(f"release executed-command contract drift: {expected['release_id']}")
    for field in (
        "exact_inputs_sha256",
        "exact_complete_sha256",
        "split_manifest_sha256",
        "split_complete_sha256",
    ):
        if preflight.get(field) != bindings[field]:
            raise ValueError(f"release preflight evidence binding drift: {expected['release_id']}/{field}")
    promoted = preflight.get("batch_contract", {})
    execution_contract = promoted.get("execution_contract", {})
    execution_check = preflight.get("promoted_execution_contract_check", {})
    if (
        promoted.get("schema") != "olmoearth-release-batch-contract-v1"
        or promoted.get("status") != "promoted"
        or promoted.get("full_run_allowed") is not True
        or promoted.get("promotion_pending") != []
        or execution_contract.get("schema") != "olmoearth-release-execution-contract-v1"
        or execution_check.get("schema")
        != "olmoearth-release-full-execution-contract-check-v1"
        or execution_check.get("status") != "matched"
        or execution_check.get("repo_id") != expected["repo_id"]
        or execution_check.get("physical_gpu")
        != {"index": "0", "uuid": bindings["selected_gpu_uuid"]}
    ):
        raise ValueError(f"release did not match the promoted execution contract: {expected['release_id']}")
    if hashlib.sha256(canonical_bytes(promoted)).hexdigest() != preflight.get(
        "batch_contract_sha256"
    ):
        raise ValueError(f"embedded promoted batch contract SHA drift: {expected['release_id']}")
    _require_sha(
        preflight.get("batch_contract_complete_sha256"),
        f"{expected['release_id']}/batch contract completion",
    )
    gate_checks = promoted.get("gate_checks", {})
    if not gate_checks or not all(value is True for value in gate_checks.values()):
        raise ValueError(f"promoted batch gates are incomplete: {expected['release_id']}")
    promotion_evidence = promoted.get("evidence_files", [])
    if len(promotion_evidence) < 3:
        raise ValueError(f"promoted batch evidence is incomplete: {expected['release_id']}")
    for item in promotion_evidence:
        evidence_path = Path(item.get("path", ""))
        if not evidence_path.is_file() or file_sha256(evidence_path) != item.get("sha256"):
            raise ValueError(
                f"promoted batch evidence drift: {expected['release_id']}/{evidence_path}"
            )
    selected_tuning = promoted.get("selected", {})
    if (
        selected_tuning.get("batch_size") != run.get("batch_size")
        or selected_tuning.get("num_workers") != run.get("num_workers")
        or execution_contract.get("selected_tuning") != selected_tuning
    ):
        raise ValueError(f"release tuning differs from promoted setting: {expected['release_id']}")
    if not execution_contract.get("validation") or not all(
        value is True for value in execution_contract["validation"].values()
    ):
        raise ValueError(f"promoted execution assertions are incomplete: {expected['release_id']}")
    if execution_check.get("semantic_config_core") != execution_contract.get(
        "semantic_config_core"
    ):
        raise ValueError(f"release semantic core differs from promotion: {expected['release_id']}")
    if execution_check.get("runtime_versions") != execution_contract.get("runtime_versions"):
        raise ValueError(f"release runtime differs from promotion: {expected['release_id']}")
    if execution_check.get("rslearn_runtime_fingerprint") != execution_contract.get(
        "rslearn_runtime_fingerprint"
    ):
        raise ValueError(f"release rslearn runtime differs from promotion: {expected['release_id']}")
    if execution_contract.get("physical_gpu") != execution_check.get("physical_gpu"):
        raise ValueError(f"release GPU differs from promotion: {expected['release_id']}")
    if execution_contract.get("checkpoint_manifest_sha256") != execution_check.get(
        "checkpoint_manifest_sha256"
    ):
        raise ValueError(f"release checkpoint manifest differs from promotion: {expected['release_id']}")
    promoted_release = execution_contract.get("releases", {}).get(expected["repo_id"], {})
    if (
        promoted_release.get("release_id") != expected["release_id"]
        or promoted_release.get("repo_id") != expected["repo_id"]
        or promoted_release.get("revision") != run.get("revision")
        or promoted_release.get("files") != run.get("checkpoint_files")
        or execution_check.get("revision") != run.get("revision")
    ):
        raise ValueError(f"release checkpoint/revision differs from promotion: {expected['release_id']}")
    checkpoint_marker = post_run_markers["post_run_checkpoint_verification"]
    if (
        checkpoint_marker.get("schema")
        != "olmoearth-release-post-run-checkpoint-verification-v1"
        or checkpoint_marker.get("status") != "verified"
        or checkpoint_marker.get("checkpoint_manifest_sha256")
        != preflight.get("checkpoint_manifest_sha256")
        or checkpoint_marker.get("repo_id") != expected["repo_id"]
        or checkpoint_marker.get("revision") != run.get("revision")
        or checkpoint_marker.get("checkpoint_files") != promoted_release.get("files")
    ):
        raise ValueError(f"post-run checkpoint marker content drift: {expected['release_id']}")
    runtime_marker = post_run_markers["post_run_rslearn_runtime_verification"]
    if (
        runtime_marker.get("schema")
        != "olmoearth-release-post-run-rslearn-runtime-verification-v1"
        or runtime_marker.get("status") != "verified"
        or runtime_marker.get("rslearn_runtime_fingerprint")
        != preflight.get("rslearn_runtime_fingerprint")
    ):
        raise ValueError(f"post-run rslearn marker content drift: {expected['release_id']}")
    execution_binding = execution_check.get("execution_binding", {})
    if (
        execution_binding.get("output_layer") != expected["output_layer"]
        or execution_binding.get("batch_size") != run.get("batch_size")
        or execution_binding.get("num_workers") != run.get("num_workers")
        or execution_binding.get("dataset_path_environment") != "DATASET_PATH"
    ):
        raise ValueError(f"release execution binding differs from promotion: {expected['release_id']}")
    return {
        "run_summary_sha256": run_sha,
        "complete_sha256": complete_sha,
        "preflight_sha256": file_sha256(preflight_path),
        "batch_contract_sha256": preflight.get("batch_contract_sha256"),
        "batch_contract_complete_sha256": preflight.get("batch_contract_complete_sha256"),
        "execution_contract_sha256": hashlib.sha256(
            canonical_bytes(execution_contract)
        ).hexdigest(),
        "semantic_config_core_sha256": hashlib.sha256(
            canonical_bytes(execution_check["semantic_config_core"])
        ).hexdigest(),
        "rslearn_runtime_fingerprint_sha256": hashlib.sha256(
            canonical_bytes(execution_check["rslearn_runtime_fingerprint"])
        ).hexdigest(),
        "full_runner_code_contract_sha256": runner_code_contract[
            "inventory_sha256"
        ],
        "full_runner_code_contract": runner_code_contract,
        "selected_gpu_uuid": bindings["selected_gpu_uuid"],
        "batch_size": run["batch_size"],
        "num_workers": run["num_workers"],
        "output_map_sha256": hashlib.sha256(
            canonical_bytes(
                {
                    sample_id: {
                        key: output_map[sample_id][key]
                        for key in (
                            "window",
                            "spatial_cluster_id",
                            "input_bundle_identity",
                            "path",
                            "bytes",
                            "sha256",
                            "mtime_ns",
                        )
                    }
                    for sample_id in sorted(output_map)
                }
            )
        ).hexdigest(),
        "_output_map": output_map,
        "_health_map": health_by_sample,
    }


def validate_frozen_evidence(
    *,
    evidence_summary_path: Path,
    evidence_complete_path: Path,
    paired_outputs_path: Path,
    exact_inputs_path: Path,
    exact_complete_path: Path,
    split_manifest_path: Path,
    split_complete_path: Path,
    v1_run_summary_path: Path | None = None,
    v1_complete_path: Path | None = None,
    v1_2_run_summary_path: Path | None = None,
    v1_2_complete_path: Path | None = None,
) -> FrozenEvidence:
    """Validate every marker/link in the frozen analysis evidence chain."""
    evidence_root = evidence_summary_path.parent.resolve()
    if (
        evidence_summary_path.resolve() != evidence_root / "evidence_summary.json"
        or evidence_complete_path.resolve()
        != evidence_root / "FULL_EVIDENCE_COMPLETE.json"
        or paired_outputs_path.resolve() != evidence_root / "paired_outputs.jsonl"
    ):
        raise ValueError("paired-evidence path contract drift")
    evidence_summary = read_json(evidence_summary_path)
    evidence_complete = read_json(evidence_complete_path)
    exact_payload = read_json(exact_inputs_path)
    exact_complete = read_json(exact_complete_path)
    split_manifest = read_json(split_manifest_path)
    split_complete = read_json(split_complete_path)

    evidence_summary_sha = file_sha256(evidence_summary_path)
    paired_outputs_sha = file_sha256(paired_outputs_path)
    exact_inputs_sha = file_sha256(exact_inputs_path)
    exact_complete_sha = file_sha256(exact_complete_path)
    split_manifest_sha = file_sha256(split_manifest_path)
    split_complete_sha = file_sha256(split_complete_path)

    if (
        evidence_complete.get("schema") != EXPECTED_PAIR_MARKER_SCHEMA
        or evidence_complete.get("status") != "complete"
    ):
        raise ValueError("unrecognized paired-evidence completion marker")
    if evidence_complete.get("evidence_summary_sha256") != evidence_summary_sha:
        raise ValueError("paired-evidence marker does not bind the evidence summary")
    if evidence_complete.get("paired_outputs_jsonl_sha256") != paired_outputs_sha:
        raise ValueError("paired-evidence marker does not bind paired_outputs.jsonl")
    if evidence_summary.get("schema") != EXPECTED_EVIDENCE_SCHEMA:
        raise ValueError("unrecognized paired-evidence schema")
    if evidence_summary.get("status") != "complete":
        raise ValueError("paired release evidence is not complete")
    try:
        code_contracts = validate_finalizer_code_evidence(
            evidence_summary=evidence_summary,
            evidence_completion=evidence_complete,
            evidence_root=evidence_root,
            require_live_match=True,
        )
    except ValueError as exc:
        raise ValueError(f"paired evidence code-contract drift: {exc}") from exc
    if evidence_summary.get("paired_outputs") != 216:
        raise ValueError("paired evidence does not contain exactly 216 pairs")
    if evidence_summary.get("paired_outputs_jsonl_sha256") != paired_outputs_sha:
        raise ValueError("evidence summary paired-output SHA drift")
    for field in (
        "raster_contracts_exact",
        "validity_masks_exact",
        "value_health_passed_all_432_outputs",
        "same_exact_inputs_both_releases",
    ):
        if evidence_summary.get(field) is not True:
            raise ValueError(f"paired evidence lacks required assertion: {field}")
    input_closure = evidence_summary.get("input_post_run_closure", {})
    output_closure = evidence_summary.get("output_post_run_closure", {})
    if (
        input_closure.get("files") != 5616
        or input_closure.get("all_sha256_match_frozen_manifest") is not True
    ):
        raise ValueError("post-run input closure is incomplete")
    if (
        output_closure.get("files") != 432
        or output_closure.get("all_sha256_match_release_manifests") is not True
    ):
        raise ValueError("post-run output closure is incomplete")
    if not isinstance(evidence_summary.get("selected_gpu_uuid"), str):
        raise ValueError("paired evidence does not bind a physical GPU UUID")
    for field in (
        "finalizer_code_sha256",
        "v1_run_summary_sha256",
        "v1_complete_sha256",
        "v1_2_run_summary_sha256",
        "v1_2_complete_sha256",
    ):
        _require_sha(evidence_summary.get(field), f"evidence_summary.{field}")
    if not MINIMUM_FORBIDDEN_CLAIMS.issubset(
        set(evidence_summary.get("claims_forbidden", []))
    ):
        raise ValueError("paired evidence omits mandatory forbidden claims")

    if exact_payload.get("schema") != EXPECTED_EXACT_SCHEMA:
        raise ValueError("unrecognized exact-input schema")
    if exact_payload.get("exact_tensor_file_pairing_ready") is not True:
        raise ValueError("exact tensor pairing is not ready")
    if exact_complete.get("schema") != EXPECTED_EXACT_MARKER_SCHEMA:
        raise ValueError("unrecognized exact-input completion marker")
    if exact_complete.get("exact_inputs_sha256") != exact_inputs_sha:
        raise ValueError("exact-input completion marker mismatch")
    if (
        exact_complete.get("records") != 216
        or exact_complete.get("spatial_clusters") != 54
        or exact_complete.get("years") != list(EXPECTED_YEARS)
        or exact_complete.get("unique_files") != 5616
    ):
        raise ValueError("exact-input completion counts drifted")
    if evidence_summary.get("exact_inputs_sha256") != exact_inputs_sha:
        raise ValueError("paired evidence is not bound to these exact inputs")
    if evidence_summary.get("exact_complete_sha256") != exact_complete_sha:
        raise ValueError("paired evidence is not bound to this exact-input marker")

    exact_records_list = exact_payload.get("records", [])
    if len(exact_records_list) != 216:
        raise ValueError("exact-input manifest is not 216 records")
    exact_records: dict[str, dict[str, Any]] = {}
    panel = Counter()
    spatial_key_by_cluster: dict[str, str] = {}
    cluster_by_spatial_key: dict[str, str] = {}
    for record in exact_records_list:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in exact_records:
            raise ValueError("exact-input sample IDs are missing or duplicated")
        if record.get("hash_policy") != "sha256":
            raise ValueError(f"non-SHA exact input: {sample_id}")
        _require_sha(record.get("input_bundle_identity"), f"input identity/{sample_id}")
        if len(record.get("input_layers", [])) != 12:
            raise ValueError(f"exact input does not have 12 periods: {sample_id}")
        if [value.get("period_index") for value in record["input_layers"]] != list(
            range(12)
        ):
            raise ValueError(f"exact input period order drift: {sample_id}")
        year = int(record["year"])
        if year not in EXPECTED_YEARS:
            raise ValueError(f"unexpected exact-input year: {sample_id}/{year}")
        cluster = record["spatial_cluster_id"]
        spatial_key = record["spatial_key"]
        previous_key = spatial_key_by_cluster.setdefault(cluster, spatial_key)
        if previous_key != spatial_key:
            raise ValueError(f"one spatial cluster has multiple keys: {cluster}")
        previous_cluster = cluster_by_spatial_key.setdefault(spatial_key, cluster)
        if previous_cluster != cluster:
            raise ValueError(f"one spatial key maps to multiple clusters: {spatial_key}")
        panel[(cluster, year)] += 1
        exact_records[sample_id] = record
    clusters = set(spatial_key_by_cluster)
    expected_panel = {(cluster, year) for cluster in clusters for year in EXPECTED_YEARS}
    if (
        len(clusters) != 54
        or len(cluster_by_spatial_key) != 54
        or set(panel) != expected_panel
        or any(value != 1 for value in panel.values())
    ):
        raise ValueError("exact inputs are not a complete 54-site by four-year panel")

    if split_complete.get("schema") != EXPECTED_SPLIT_MARKER_SCHEMA:
        raise ValueError("unrecognized spatial-split completion marker")
    if split_complete.get("split_manifest_sha256") != split_manifest_sha:
        raise ValueError("spatial-split completion marker mismatch")
    if split_complete.get("frozen_before_full_output_inspection") is not True:
        raise ValueError("spatial split was not frozen before output inspection")
    if split_manifest.get("schema") != EXPECTED_SPLIT_SCHEMA:
        raise ValueError("unrecognized spatial-split schema")
    if split_manifest.get("frozen_before_full_output_inspection") is not True:
        raise ValueError("spatial split lacks its freeze assertion")
    if split_manifest.get("exact_inputs", {}).get("sha256") != exact_inputs_sha:
        raise ValueError("spatial split is not bound to these exact inputs")
    if evidence_summary.get("split_manifest_sha256") != split_manifest_sha:
        raise ValueError("paired evidence is not bound to this spatial split")
    if evidence_summary.get("split_complete_sha256") != split_complete_sha:
        raise ValueError("paired evidence is not bound to this split marker")
    contract = split_manifest.get("analysis_contract", {})
    if (
        contract.get("bridge_fit") != "calibration only"
        or contract.get("hyperparameter_selection")
        != "grouped inner validation within calibration only"
        or contract.get("all_years_of_each_location_share_one_split") is not True
    ):
        raise ValueError("spatial split analysis contract drifted")
    expected_rule = {
        "calibration_x": list(CALIBRATION_X),
        "embargo_x": list(EMBARGO_X),
        "east_x": list(EAST_X),
        "disclosed_east_clusters_removed_from_test": list(
            EXPECTED_DISCLOSED_EAST_KEYS
        ),
    }
    if split_manifest.get("split_rule") != expected_rule:
        raise ValueError("spatial split rule differs from the frozen preregistration")
    prior_disclosure = split_manifest.get("prior_disclosure_inputs", {})
    prior_path_value = prior_disclosure.get("path")
    if not isinstance(prior_path_value, str):
        raise ValueError("spatial split lacks its prior-disclosure source binding")
    prior_path = Path(prior_path_value)
    if not prior_path.is_file() or file_sha256(prior_path) != prior_disclosure.get(
        "sha256"
    ):
        raise ValueError("prior-disclosure source evidence drifted")
    prior_payload = read_json(prior_path)
    prior_records = prior_payload.get("records", [])
    if not isinstance(prior_records, list) or not prior_records:
        raise ValueError("prior-disclosure source contains no records")
    prior_east_keys = sorted(
        {
            record.get("spatial_key")
            for record in prior_records
            if isinstance(record.get("spatial_key"), str)
            and _spatial_coordinates(record["spatial_key"])[0] in EAST_X
        }
    )
    if prior_east_keys != list(EXPECTED_DISCLOSED_EAST_KEYS):
        raise ValueError("prior-disclosure east-cluster membership drifted")
    expected_count_payload = {
        split_name: {
            "spatial_clusters": count,
            "site_years": count * 4,
            "adjacent_year_events": count * 3,
        }
        for split_name, count in EXPECTED_SPLIT_CLUSTERS.items()
    }
    if split_manifest.get("counts") != expected_count_payload:
        raise ValueError("spatial split count payload differs from the frozen contract")

    split_by_sample: dict[str, str] = {}
    split_by_cluster: dict[str, str] = {}
    assignments = split_manifest.get("assignments", [])
    if len(assignments) != 54:
        raise ValueError("spatial split does not contain 54 assignments")
    observed_split_counts = Counter()
    for assignment in assignments:
        split_name = assignment.get("split")
        if split_name not in EXPECTED_SPLIT_CLUSTERS:
            raise ValueError(f"unexpected split name: {split_name}")
        cluster = assignment.get("spatial_cluster_id")
        key = assignment.get("spatial_key")
        if (
            cluster in split_by_cluster
            or key not in cluster_by_spatial_key
            or spatial_key_by_cluster.get(cluster) != key
            or cluster_by_spatial_key.get(key) != cluster
        ):
            raise ValueError(f"duplicated or mismatched split cluster: {cluster}")
        if split_name != _expected_split_for_key(key):
            raise ValueError(f"split assignment violates the frozen coordinate rule: {key}")
        expected_ids = sorted(
            sample_id
            for sample_id, record in exact_records.items()
            if record["spatial_cluster_id"] == cluster
        )
        if assignment.get("sample_ids") != expected_ids:
            raise ValueError(f"split sample membership drift: {cluster}")
        if assignment.get("years") != list(EXPECTED_YEARS):
            raise ValueError(f"split year membership drift: {cluster}")
        split_by_cluster[cluster] = split_name
        observed_split_counts[split_name] += 1
        for sample_id in expected_ids:
            split_by_sample[sample_id] = split_name
    if dict(observed_split_counts) != EXPECTED_SPLIT_CLUSTERS:
        raise ValueError(f"spatial split counts drifted: {dict(observed_split_counts)}")
    if set(split_by_sample) != set(exact_records):
        raise ValueError("spatial split does not cover all exact sample identities")
    observed_disclosed = sorted(
        key
        for key, cluster in cluster_by_spatial_key.items()
        if split_by_cluster[cluster] == "disclosed_audit"
    )
    if observed_disclosed != list(EXPECTED_DISCLOSED_EAST_KEYS):
        raise ValueError("disclosed-audit membership differs from prior disclosure evidence")

    pairs_list = read_jsonl(paired_outputs_path)
    if len(pairs_list) != 216:
        raise ValueError("paired-output JSONL is not 216 records")
    pairs: dict[str, dict[str, Any]] = {}
    output_paths: set[str] = set()
    for pair in pairs_list:
        sample_id = pair.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in pairs:
            raise ValueError("paired output sample IDs are missing or duplicated")
        if sample_id not in exact_records:
            raise ValueError(f"paired output is not in exact inputs: {sample_id}")
        exact_record = exact_records[sample_id]
        paired_identity = (
            pair.get("window"),
            pair.get("spatial_cluster_id"),
            pair.get("input_bundle_identity"),
        )
        if paired_identity != _identity_fields(exact_record):
            raise ValueError(f"paired output identity drift: {sample_id}")
        if pair.get("validity_masks_exact") is not True:
            raise ValueError(f"paired output lacks exact-mask assertion: {sample_id}")
        contract_value = pair.get("raster_contract", {})
        required_contract_fields = {
            "height",
            "width",
            "count",
            "dtypes",
            "crs",
            "transform",
            "bounds",
            "nodata",
        }
        if set(contract_value) != required_contract_fields:
            raise ValueError(f"paired raster contract fields drifted: {sample_id}")
        if (
            int(contract_value["height"]) < LATTICE_SIDE
            or int(contract_value["width"]) < LATTICE_SIDE
            or int(contract_value["count"]) != 768
            or contract_value["dtypes"] != ["float32"]
        ):
            raise ValueError(f"paired raster shape/feature/dtype contract drift: {sample_id}")
        if pair.get("value_health_passed_both_releases") is not True:
            raise ValueError(f"paired output lacks value-health assertion: {sample_id}")
        expected_total = (
            int(contract_value["height"])
            * int(contract_value["width"])
            * int(contract_value["count"])
        )
        for health_field in ("v1_value_health", "v1_2_value_health"):
            health = pair.get(health_field, {})
            if (
                int(health.get("usable_tokens", 0)) < 2
                or int(health.get("nonzero_usable_tokens", 0)) < 2
                or int(health.get("finite_values", -1)) < 0
                or int(health.get("total_values", -1)) != expected_total
                or not isinstance(health.get("all_values_finite"), bool)
            ):
                raise ValueError(f"paired value-health drift: {sample_id}/{health_field}")
        for release_field in ("v1_output", "v1_2_output"):
            inventory = pair.get(release_field, {})
            path = inventory.get("path")
            if not isinstance(path, str) or path in output_paths:
                raise ValueError(f"missing or duplicate output path: {sample_id}/{release_field}")
            output_paths.add(path)
            if int(inventory.get("bytes", -1)) <= 0:
                raise ValueError(f"invalid output byte count: {sample_id}/{release_field}")
            _require_sha(inventory.get("sha256"), f"{sample_id}/{release_field}")
            if int(inventory.get("mtime_ns", -1)) <= 0:
                raise ValueError(f"invalid output mtime: {sample_id}/{release_field}")
        pairs[sample_id] = pair
    if set(pairs) != set(exact_records) or len(output_paths) != 432:
        raise ValueError("paired output identities/paths are not a complete 216 by two set")

    execution_paths = (
        v1_run_summary_path,
        v1_complete_path,
        v1_2_run_summary_path,
        v1_2_complete_path,
    )
    execution_contract: dict[str, Any] | None = None
    if any(path is not None for path in execution_paths):
        if any(path is None for path in execution_paths):
            raise ValueError("all four release execution paths are required together")
        bindings = {
            "exact_inputs_sha256": exact_inputs_sha,
            "exact_complete_sha256": exact_complete_sha,
            "split_manifest_sha256": split_manifest_sha,
            "split_complete_sha256": split_complete_sha,
            "selected_gpu_uuid": evidence_summary["selected_gpu_uuid"],
        }
        v1_execution = _validate_release_execution(
            v1_run_summary_path,  # type: ignore[arg-type]
            v1_complete_path,  # type: ignore[arg-type]
            EXPECTED_RELEASE_RUNS["v1"],
            evidence_summary["v1_run_summary_sha256"],
            evidence_summary["v1_complete_sha256"],
            bindings,
        )
        v1_2_execution = _validate_release_execution(
            v1_2_run_summary_path,  # type: ignore[arg-type]
            v1_2_complete_path,  # type: ignore[arg-type]
            EXPECTED_RELEASE_RUNS["v1_2"],
            evidence_summary["v1_2_run_summary_sha256"],
            evidence_summary["v1_2_complete_sha256"],
            bindings,
        )
        for field in (
            "batch_contract_sha256",
            "batch_contract_complete_sha256",
            "execution_contract_sha256",
            "semantic_config_core_sha256",
            "rslearn_runtime_fingerprint_sha256",
            "selected_gpu_uuid",
            "batch_size",
            "num_workers",
            "full_runner_code_contract_sha256",
        ):
            if v1_execution[field] != v1_2_execution[field]:
                raise ValueError(f"paired releases differ in promoted execution field: {field}")
        if (
            v1_execution.get("full_runner_code_contract")
            != v1_2_execution.get("full_runner_code_contract")
            or v1_execution.get("full_runner_code_contract")
            != code_contracts["full_runner"]
        ):
            raise ValueError(
                "release executions and paired evidence do not share one full-runner code contract"
            )
        cross_bind_paired_outputs_to_release_runs(
            pairs,
            v1_execution["_output_map"],
            v1_2_execution["_output_map"],
            v1_execution["_health_map"],
            v1_2_execution["_health_map"],
        )
        v1_execution = {
            key: value
            for key, value in v1_execution.items()
            if key not in {"_output_map", "_health_map"}
        }
        v1_2_execution = {
            key: value
            for key, value in v1_2_execution.items()
            if key not in {"_output_map", "_health_map"}
        }
        execution_contract = {
            "status": "both_releases_match_one_promoted_execution_contract",
            "v1": v1_execution,
            "v1_2": v1_2_execution,
        }

    return FrozenEvidence(
        exact_records=exact_records,
        pairs=pairs,
        split_by_sample=split_by_sample,
        split_by_cluster=split_by_cluster,
        evidence_hashes={
            "evidence_summary_sha256": evidence_summary_sha,
            "evidence_complete_sha256": file_sha256(evidence_complete_path),
            "paired_outputs_jsonl_sha256": paired_outputs_sha,
            "exact_inputs_sha256": exact_inputs_sha,
            "exact_complete_sha256": exact_complete_sha,
            "split_manifest_sha256": split_manifest_sha,
            "split_complete_sha256": split_complete_sha,
            "prior_disclosure_inputs_sha256": prior_disclosure["sha256"],
            "full_runner_code_contract_sha256": code_contracts["full_runner"][
                "inventory_sha256"
            ],
            "finalizer_code_contract_sha256": code_contracts["finalizer"][
                "inventory_sha256"
            ],
        },
        execution_contract=execution_contract,
        code_contracts=code_contracts,
    )


def validate_live_output_hashes(
    pairs: dict[str, dict[str, Any]], workers: int
) -> dict[str, Any]:
    if not 1 <= workers <= 2:
        raise ValueError("output hash workers must be one or two")
    inventories = []
    for sample_id in sorted(pairs):
        pair = pairs[sample_id]
        inventories.extend((pair["v1_output"], pair["v1_2_output"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        live = list(
            executor.map(
                lambda value: _stable_live_inventory(
                    Path(value["path"]), int(value["bytes"])
                ),
                inventories,
            )
        )
    for expected, observed in zip(inventories, live, strict=True):
        if observed["sha256"] != expected["sha256"]:
            raise ValueError(f"live output SHA drift: {expected['path']}")
    return {
        "files": len(live),
        "bytes": sum(int(value["bytes"]) for value in live),
        "all_sha256_match_paired_evidence": True,
    }


def lattice_axis(length: int, side: int = LATTICE_SIDE) -> np.ndarray:
    """Return unique center-of-bin pixel indices for a deterministic lattice."""
    if length < side or side < 1:
        raise ValueError(f"cannot place a {side}-point lattice on axis length {length}")
    indices = np.floor((np.arange(side, dtype=np.float64) + 0.5) * length / side).astype(
        np.int64
    )
    if len(np.unique(indices)) != side or indices[0] < 0 or indices[-1] >= length:
        raise ValueError("lattice construction did not yield unique in-bounds indices")
    return indices


def query_flat_indices() -> np.ndarray:
    return np.array(
        [
            row * LATTICE_SIDE + column
            for row in QUERY_LATTICE_POSITIONS
            for column in QUERY_LATTICE_POSITIONS
        ],
        dtype=np.int64,
    )


def _observed_raster_contract(dataset: Any) -> dict[str, Any]:
    return {
        "height": dataset.height,
        "width": dataset.width,
        "count": dataset.count,
        # The paired finalizer stores the canonical unique dtype set produced
        # by ``inspect_raster``, not one repeated entry per feature band.
        "dtypes": sorted(set(dataset.dtypes)),
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "transform": list(dataset.transform)[:6],
        "bounds": list(dataset.bounds),
        "nodata": dataset.nodata,
    }


def _verified_window_bounds(record: dict[str, Any]) -> tuple[str, list[float]]:
    inventory = record["window_metadata"]
    path = Path(inventory["path"])
    live = _stable_live_inventory(path, int(inventory["bytes"]))
    if live["sha256"] != inventory["sha256"]:
        raise ValueError(f"window metadata SHA drift: {path}")
    payload = read_json(path)
    projection, bounds = payload["projection"], payload["bounds"]
    xs = [bounds[0] * projection["x_resolution"], bounds[2] * projection["x_resolution"]]
    ys = [bounds[1] * projection["y_resolution"], bounds[3] * projection["y_resolution"]]
    return projection["crs"], [min(xs), min(ys), max(xs), max(ys)]


def read_verified_pair(
    pair: dict[str, Any], exact_record: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Read only the fixed lattice and revalidate its live grid and masks.

    Full-file SHA validation above proves that the rasters are the same files
    whose complete value/mask scans were sealed by the paired finalizer.  The
    analyzer therefore reads only 16 spatial rows per raster, avoiding a second
    432-file full-value scan on shared storage.
    """
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - server integration dependency
        raise RuntimeError("rasterio is required for full release analysis") from exc

    arrays, masks, contracts = [], [], []
    for release_field in ("v1_output", "v1_2_output"):
        with rasterio.open(Path(pair[release_field]["path"])) as dataset:
            observed = _observed_raster_contract(dataset)
            rows = lattice_axis(dataset.height)
            columns = lattice_axis(dataset.width)
            indexes = list(range(1, dataset.count + 1))
            if len(set(dataset.block_shapes)) != 1:
                raise ValueError(
                    f"output bands have different block shapes: {pair['sample_id']}/{release_field}"
                )
            block_height = int(dataset.block_shapes[0][0])
            block_starts = sorted({int(row // block_height * block_height) for row in rows})
            lattice_values = np.empty(
                (LATTICE_SIDE, LATTICE_SIDE, dataset.count), dtype=np.float32
            )
            lattice_masks = np.empty((LATTICE_SIDE, LATTICE_SIDE), dtype=bool)
            row_positions = {int(row): position for position, row in enumerate(rows)}
            for row_start in block_starts:
                row_stop = min(row_start + block_height, dataset.height)
                window = ((row_start, row_stop), (0, dataset.width))
                values = dataset.read(indexes=indexes, window=window, out_dtype="float32")
                value_masks = dataset.read_masks(indexes=indexes, window=window)
                for row in rows:
                    if not row_start <= row < row_stop:
                        continue
                    position = row_positions[int(row)]
                    local_row = int(row) - row_start
                    lattice_values[position] = values[:, local_row, columns].T
                    lattice_masks[position] = np.all(
                        value_masks[:, local_row, columns] > 0, axis=0
                    )
        if observed != pair["raster_contract"]:
            raise ValueError(
                f"live raster contract drift: {pair['sample_id']}/{release_field}"
            )
        contracts.append(observed)
        arrays.append(lattice_values)
        masks.append(lattice_masks)
    if contracts[0] != contracts[1]:
        raise ValueError(f"paired live raster grids differ: {pair['sample_id']}")
    if not np.array_equal(masks[0], masks[1]):
        raise ValueError(f"paired live validity masks differ: {pair['sample_id']}")
    valid = masks[0]
    for release_name, array in zip(("v1", "v1.2"), arrays, strict=True):
        if not np.isfinite(array[valid]).all():
            raise ValueError(
                f"valid output pixels contain non-finite values: {pair['sample_id']}/{release_name}"
            )
    expected_crs, expected_bounds = _verified_window_bounds(exact_record)
    if contracts[0]["crs"] != expected_crs or not np.allclose(
        contracts[0]["bounds"], expected_bounds, rtol=0.0, atol=1e-6
    ):
        raise ValueError(f"output does not match the frozen window grid: {pair['sample_id']}")
    return arrays[0], arrays[1], valid, contracts[0]


def extract_lattice(
    array: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if (
        array.ndim != 3
        or array.shape[:2] != (LATTICE_SIDE, LATTICE_SIDE)
        or valid_mask.shape != array.shape[:2]
    ):
        raise ValueError("embedding/mask dimensionality mismatch")
    tokens = array.reshape(-1, array.shape[-1]).astype(np.float32, copy=False)
    token_valid = valid_mask.reshape(-1)
    if tokens.shape[0] != LATTICE_SIDE**2 or not np.all(token_valid):
        raise ValueError("the frozen 16x16 lattice contains invalid/nodata tokens")
    if not np.isfinite(tokens).all():
        raise ValueError("the frozen 16x16 lattice contains non-finite tokens")
    return tokens, tokens[query_flat_indices()]


def _center(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2 or not np.isfinite(value).all():
        raise ValueError("metric input must be a finite 2D matrix with at least two rows")
    return value - value.mean(axis=0, keepdims=True)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left_centered, right_centered = _center(left), _center(right)
    if left_centered.shape[0] != right_centered.shape[0]:
        raise ValueError("CKA requires paired observations")
    left_gram = left_centered @ left_centered.T
    right_gram = right_centered @ right_centered.T
    denominator = float(
        np.linalg.norm(left_gram, ord="fro") * np.linalg.norm(right_gram, ord="fro")
    )
    if denominator <= 0:
        raise ValueError("CKA is undefined for a constant representation")
    return float(
        np.clip(np.sum(left_gram * right_gram) / denominator, 0.0, 1.0)
    )


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("row normalization requires a finite 2D matrix")
    norms = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("row normalization received a zero vector")
    return value / norms


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("rank input contains a non-finite value")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.asarray(left).shape != np.asarray(right).shape:
        raise ValueError("Spearman inputs must have the same shape")
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator <= 0:
        raise ValueError("Spearman correlation is undefined for constant ranks")
    return float(np.clip(np.dot(left_rank, right_rank) / denominator, -1.0, 1.0))


def kendall_tau_b(left: np.ndarray, right: np.ndarray) -> float:
    left_value = np.asarray(left, dtype=np.float64).reshape(-1)
    right_value = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_value.shape != right_value.shape or left_value.size < 2:
        raise ValueError("Kendall tau-b requires paired vectors of length at least two")
    concordant = discordant = left_only_ties = right_only_ties = 0
    for first in range(left_value.size - 1):
        left_delta = left_value[first] - left_value[first + 1 :]
        right_delta = right_value[first] - right_value[first + 1 :]
        left_sign, right_sign = np.sign(left_delta), np.sign(right_delta)
        concordant += int(np.sum(left_sign * right_sign > 0))
        discordant += int(np.sum(left_sign * right_sign < 0))
        left_only_ties += int(np.sum((left_sign == 0) & (right_sign != 0)))
        right_only_ties += int(np.sum((left_sign != 0) & (right_sign == 0)))
    denominator = math.sqrt(
        (concordant + discordant + left_only_ties)
        * (concordant + discordant + right_only_ties)
    )
    if denominator <= 0:
        raise ValueError("Kendall tau-b is undefined when every pair is tied")
    return float((concordant - discordant) / denominator)


def _euclidean_distances(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    squared = np.sum(value**2, axis=1, keepdims=True)
    distances_squared = np.maximum(squared + squared.T - 2.0 * (value @ value.T), 0.0)
    return np.sqrt(distances_squared)


def geometry_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_value, right_value = np.asarray(left), np.asarray(right)
    if left_value.shape[0] != right_value.shape[0] or left_value.shape[0] < 3:
        raise ValueError("geometry metrics require at least three paired identities")
    left_distances, right_distances = _euclidean_distances(left_value), _euclidean_distances(
        right_value
    )
    triangle = np.triu_indices(left_value.shape[0], k=1)
    return {
        "site_years": int(left_value.shape[0]),
        "embedding_dimensions": {
            "v1": int(left_value.shape[1]),
            "v1_2": int(right_value.shape[1]),
        },
        "linear_cka": linear_cka(left_value, right_value),
        "pairwise_euclidean_distance_spearman": spearman_correlation(
            left_distances[triangle], right_distances[triangle]
        ),
        "pairwise_distances_are_independent_replicates": False,
    }


def leave_one_cluster_out_geometry(
    left: np.ndarray, right: np.ndarray, clusters: Sequence[str]
) -> dict[str, Any]:
    if len(clusters) != np.asarray(left).shape[0]:
        raise ValueError("one cluster ID is required per geometry observation")
    rows = []
    for cluster in sorted(set(clusters)):
        keep = np.array([value != cluster for value in clusters], dtype=bool)
        if int(keep.sum()) < 3:
            continue
        metrics = geometry_metrics(np.asarray(left)[keep], np.asarray(right)[keep])
        rows.append(
            {
                "left_out_spatial_cluster_id": cluster,
                "linear_cka": metrics["linear_cka"],
                "pairwise_euclidean_distance_spearman": metrics[
                    "pairwise_euclidean_distance_spearman"
                ],
            }
        )
    if len(rows) < 2:
        return {"available": False, "reason": "fewer_than_two_cluster_deletions"}
    return {
        "available": True,
        "spatial_clusters": len(rows),
        "range": {
            metric: [
                float(min(row[metric] for row in rows)),
                float(max(row[metric] for row in rows)),
            ]
            for metric in ("linear_cka", "pairwise_euclidean_distance_spearman")
        },
    }


def cluster_jackknife_mean(
    values: Sequence[float], clusters: Sequence[str], *, bounded: bool = False
) -> dict[str, Any]:
    value_array = np.asarray(values, dtype=np.float64)
    cluster_array = np.asarray(clusters)
    if value_array.ndim != 1 or value_array.size != cluster_array.size:
        raise ValueError("jackknife values and clusters must be aligned vectors")
    unique_clusters = sorted(set(str(value) for value in cluster_array))
    if len(unique_clusters) < 2:
        return {
            "estimate": float(np.mean(value_array)),
            "available": False,
            "reason": "fewer_than_two_spatial_clusters",
        }
    leave_one_out = np.array(
        [
            np.mean(value_array[cluster_array != cluster])
            for cluster in unique_clusters
        ],
        dtype=np.float64,
    )
    loo_mean = float(np.mean(leave_one_out))
    standard_error = float(
        math.sqrt(
            (len(unique_clusters) - 1)
            / len(unique_clusters)
            * np.sum((leave_one_out - loo_mean) ** 2)
        )
    )
    lower = float(np.mean(value_array) - 1.96 * standard_error)
    upper = float(np.mean(value_array) + 1.96 * standard_error)
    if bounded:
        lower, upper = max(0.0, lower), min(1.0, upper)
    return {
        "estimate": float(np.mean(value_array)),
        "available": True,
        "spatial_clusters": len(unique_clusters),
        "standard_error": standard_error,
        "normal_95_interval": [lower, upper],
        "leave_one_cluster_out_range": [
            float(np.min(leave_one_out)), float(np.max(leave_one_out))
        ],
        "interval_note": "cluster jackknife normal interval; descriptive, not a population CI",
    }


@dataclass(frozen=True)
class AffineMap:
    source_mean: np.ndarray
    target_mean: np.ndarray
    matrix: np.ndarray
    method: str

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        value = np.asarray(matrix, dtype=np.float64)
        result = (value - self.source_mean) @ self.matrix + self.target_mean
        if not np.isfinite(result).all():
            raise ValueError(f"{self.method} bridge produced non-finite values")
        return result

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.method.encode("utf-8"))
        for value in (self.source_mean, self.target_mean, self.matrix):
            canonical = np.ascontiguousarray(value, dtype="<f8")
            digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
            digest.update(canonical.tobytes())
        return digest.hexdigest()


def identity_map(dimensions: int) -> AffineMap:
    return AffineMap(
        source_mean=np.zeros((1, dimensions), dtype=np.float64),
        target_mean=np.zeros((1, dimensions), dtype=np.float64),
        matrix=np.eye(dimensions, dtype=np.float64),
        method="identity_no_bridge",
    )


def fit_mean_shift_translation(source: np.ndarray, target: np.ndarray) -> AffineMap:
    """Fit only the calibration mean offset, with no rotation or rescaling."""
    source_value, target_value = np.asarray(source, dtype=np.float64), np.asarray(
        target, dtype=np.float64
    )
    if source_value.shape != target_value.shape or source_value.ndim != 2:
        raise ValueError("mean-shift translation requires equal paired 2D matrices")
    if source_value.shape[0] < 1 or source_value.shape[1] < 1:
        raise ValueError("mean-shift translation requires a non-empty matrix")
    if not np.isfinite(source_value).all() or not np.isfinite(target_value).all():
        raise ValueError("mean-shift translation inputs must be finite")
    return AffineMap(
        source_mean=source_value.mean(axis=0, keepdims=True),
        target_mean=target_value.mean(axis=0, keepdims=True),
        matrix=np.eye(source_value.shape[1], dtype=np.float64),
        method="mean_shift_translation_only",
    )


def fit_translated_orthogonal_procrustes(
    source: np.ndarray, target: np.ndarray
) -> AffineMap:
    source_value, target_value = np.asarray(source, dtype=np.float64), np.asarray(
        target, dtype=np.float64
    )
    if source_value.shape != target_value.shape or source_value.ndim != 2:
        raise ValueError("orthogonal Procrustes requires equal paired 2D matrices")
    source_mean = source_value.mean(axis=0, keepdims=True)
    target_mean = target_value.mean(axis=0, keepdims=True)
    cross = (source_value - source_mean).T @ (target_value - target_mean)
    left, _, right_transpose = np.linalg.svd(cross, full_matrices=False)
    rotation = left @ right_transpose
    if not np.isfinite(rotation).all():
        raise ValueError("orthogonal Procrustes produced non-finite parameters")
    return AffineMap(source_mean, target_mean, rotation, "translated_orthogonal_procrustes")


def _ridge_scale(centered_source: np.ndarray) -> float:
    scale = float(np.sum(centered_source**2) / centered_source.shape[1])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("ridge source covariance has non-positive scale")
    return scale


def fit_affine_ridge(
    source: np.ndarray, target: np.ndarray, alpha_multiplier: float
) -> tuple[AffineMap, float]:
    source_value, target_value = np.asarray(source, dtype=np.float64), np.asarray(
        target, dtype=np.float64
    )
    if source_value.shape[0] != target_value.shape[0] or source_value.ndim != 2:
        raise ValueError("affine ridge requires paired 2D matrices")
    if source_value.shape[1] != target_value.shape[1]:
        raise ValueError("this cache audit requires equal source/target dimensions")
    if alpha_multiplier <= 0:
        raise ValueError("ridge alpha multiplier must be positive")
    source_mean = source_value.mean(axis=0, keepdims=True)
    target_mean = target_value.mean(axis=0, keepdims=True)
    source_centered = source_value - source_mean
    target_centered = target_value - target_mean
    absolute_alpha = float(alpha_multiplier * _ridge_scale(source_centered))
    gram = source_centered.T @ source_centered
    cross = source_centered.T @ target_centered
    regularized = gram + absolute_alpha * np.eye(gram.shape[0], dtype=np.float64)
    weights = np.linalg.solve(regularized, cross)
    if not np.isfinite(weights).all():
        raise ValueError("affine ridge produced non-finite parameters")
    return (
        AffineMap(source_mean, target_mean, weights, "affine_ridge"),
        absolute_alpha,
    )


def paired_cosine_mean(left: np.ndarray, right: np.ndarray) -> float:
    left_normalized, right_normalized = row_normalize(left), row_normalize(right)
    return float(np.mean(np.sum(left_normalized * right_normalized, axis=1)))


def spatial_block_folds(clusters: Sequence[str], spatial_keys: Sequence[str]) -> list[str]:
    if len(clusters) != len(spatial_keys):
        raise ValueError("cluster IDs and spatial keys must align")
    cluster_to_x: dict[str, str] = {}
    for cluster, spatial_key in zip(clusters, spatial_keys, strict=True):
        try:
            x_value, _ = spatial_key.split("_", maxsplit=1)
            int(x_value)
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"invalid spatial key for blocked folds: {spatial_key}") from exc
        previous = cluster_to_x.setdefault(cluster, x_value)
        if previous != x_value:
            raise ValueError(f"one cluster crosses ridge-CV spatial blocks: {cluster}")
    if len(set(cluster_to_x.values())) < 3:
        raise ValueError("ridge selection requires at least three spatial x-blocks")
    return [cluster_to_x[cluster] for cluster in clusters]


def select_ridge_multiplier(
    source_gallery: np.ndarray,
    target_gallery: np.ndarray,
    clusters: Sequence[str],
    spatial_keys: Sequence[str],
    local_query_indices: np.ndarray,
    multipliers: Sequence[float] = RIDGE_ALPHA_MULTIPLIERS,
) -> dict[str, Any]:
    """Select ridge strength using calibration spatial blocks only."""
    source_value, target_value = np.asarray(source_gallery), np.asarray(target_gallery)
    if source_value.shape != target_value.shape:
        raise ValueError("ridge-selection source/target galleries must match")
    if source_value.shape[0] % len(clusters) != 0:
        raise ValueError("ridge-selection gallery rows do not form equal site-year blocks")
    rows_per_record = source_value.shape[0] // len(clusters)
    if np.any(local_query_indices < 0) or np.any(local_query_indices >= rows_per_record):
        raise ValueError("ridge-selection query index escapes a site-year block")
    folds = spatial_block_folds(clusters, spatial_keys)
    unique_folds = sorted(set(folds), key=lambda value: int(value))
    scores = {float(multiplier): [] for multiplier in multipliers}
    fold_rows = []
    record_rows = np.arange(source_value.shape[0]).reshape(len(clusters), rows_per_record)
    for fold in unique_folds:
        train_records = np.array([value != fold for value in folds], dtype=bool)
        validation_records = ~train_records
        train_rows = record_rows[train_records].reshape(-1)
        validation_query_rows = record_rows[validation_records][:, local_query_indices].reshape(-1)
        fold_source = np.asarray(source_value[train_rows], dtype=np.float64)
        fold_target = np.asarray(target_value[train_rows], dtype=np.float64)
        source_mean = fold_source.mean(axis=0, keepdims=True)
        target_mean = fold_target.mean(axis=0, keepdims=True)
        source_centered = fold_source - source_mean
        target_centered = fold_target - target_mean
        scale = _ridge_scale(source_centered)
        eigenvalues, eigenvectors = np.linalg.eigh(source_centered.T @ source_centered)
        projected_cross = eigenvectors.T @ (source_centered.T @ target_centered)
        for multiplier in multipliers:
            absolute_alpha = float(multiplier) * scale
            weights = eigenvectors @ (
                projected_cross / (eigenvalues[:, None] + absolute_alpha)
            )
            bridge = AffineMap(
                source_mean,
                target_mean,
                weights,
                "affine_ridge_spatial_cv",
            )
            score = paired_cosine_mean(
                bridge.transform(source_value[validation_query_rows]),
                target_value[validation_query_rows],
            )
            scores[float(multiplier)].append(score)
            fold_rows.append(
                {
                    "held_out_x_block": int(fold),
                    "alpha_multiplier": float(multiplier),
                    "absolute_alpha": absolute_alpha,
                    "paired_cosine_mean": score,
                    "validation_site_years": int(np.sum(validation_records)),
                }
            )
    means = {
        multiplier: float(np.mean(values)) for multiplier, values in scores.items()
    }
    selected = max(sorted(means), key=lambda value: (means[value], -value))
    return {
        "selected_alpha_multiplier": selected,
        "selection_metric": "mean exact-paired cosine on fixed 8x8 query lattice",
        "spatial_folds": [int(value) for value in unique_folds],
        "candidate_mean_scores": {
            str(multiplier): means[multiplier] for multiplier in sorted(means)
        },
        "fold_results": sorted(
            fold_rows,
            key=lambda value: (value["held_out_x_block"], value["alpha_multiplier"]),
        ),
        "sealed_test_used_for_selection": False,
    }


def exact_identity_retrieval(
    query: np.ndarray,
    gallery: np.ndarray,
    correct_gallery_indices: np.ndarray,
    query_clusters: Sequence[str],
    *,
    chunk_size: int,
    k_values: Sequence[int] = RETRIEVAL_K_VALUES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Rank the exact token and expose exact- and fixed-tolerance tie fragility."""
    query_value, gallery_value = row_normalize(query), row_normalize(gallery)
    correct = np.asarray(correct_gallery_indices, dtype=np.int64)
    if query_value.shape[0] != correct.size or len(query_clusters) != correct.size:
        raise ValueError("retrieval queries, identities, and clusters are misaligned")
    if query_value.shape[1] != gallery_value.shape[1]:
        raise ValueError("retrieval query/gallery dimensions differ")
    if np.any(correct < 0) or np.any(correct >= gallery_value.shape[0]):
        raise ValueError("correct gallery index is out of bounds")
    if gallery_value.shape[0] < 2:
        raise ValueError("retrieval diagnostics require at least two gallery tokens")
    if chunk_size < 1 or any(not 1 <= value <= gallery_value.shape[0] for value in k_values):
        raise ValueError("invalid retrieval chunk or k")
    ranks = np.empty(query_value.shape[0], dtype=np.int64)
    optimistic_ranks = np.empty(query_value.shape[0], dtype=np.int64)
    pessimistic_ranks = np.empty(query_value.shape[0], dtype=np.int64)
    tolerance_optimistic_ranks = np.empty(query_value.shape[0], dtype=np.int64)
    tolerance_pessimistic_ranks = np.empty(query_value.shape[0], dtype=np.int64)
    tie_counts = np.empty(query_value.shape[0], dtype=np.int64)
    near_tie_competitor_counts = np.empty(query_value.shape[0], dtype=np.int64)
    correct_to_best_competitor_margins = np.empty(
        query_value.shape[0], dtype=np.float64
    )
    gallery_indices = np.arange(gallery_value.shape[0], dtype=np.int64)
    for start in range(0, query_value.shape[0], chunk_size):
        stop = min(start + chunk_size, query_value.shape[0])
        similarities = query_value[start:stop] @ gallery_value.T
        local_correct = correct[start:stop]
        target = similarities[np.arange(stop - start), local_correct]
        greater = np.sum(similarities > target[:, None], axis=1)
        equal = similarities == target[:, None]
        tie_counts[start:stop] = np.sum(equal, axis=1)
        competitor_similarities = similarities.copy()
        competitor_similarities[np.arange(stop - start), local_correct] = -np.inf
        correct_to_best_competitor_margins[start:stop] = target - np.max(
            competitor_similarities, axis=1
        )
        within_tolerance = (
            np.abs(similarities - target[:, None]) <= NEAR_TIE_COSINE_TOLERANCE
        )
        within_tolerance[np.arange(stop - start), local_correct] = False
        near_tie_competitor_counts[start:stop] = np.sum(within_tolerance, axis=1)
        stable_before = np.sum(equal & (gallery_indices[None, :] < local_correct[:, None]), axis=1)
        optimistic_ranks[start:stop] = 1 + greater
        pessimistic_ranks[start:stop] = greater + tie_counts[start:stop]
        tolerance_optimistic_ranks[start:stop] = 1 + np.sum(
            similarities > target[:, None] + NEAR_TIE_COSINE_TOLERANCE,
            axis=1,
        )
        tolerance_pessimistic_ranks[start:stop] = np.sum(
            similarities >= target[:, None] - NEAR_TIE_COSINE_TOLERANCE,
            axis=1,
        )
        ranks[start:stop] = optimistic_ranks[start:stop] + stable_before
    reciprocal = 1.0 / ranks.astype(np.float64)
    result: dict[str, Any] = {
        "queries": int(ranks.size),
        "gallery_tokens": int(gallery_value.shape[0]),
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
        "median_rank": float(np.median(ranks)),
        "maximum_rank": int(np.max(ranks)),
        "queries_with_similarity_ties": int(np.sum(tie_counts > 1)),
        "tie_rule": "exact float tie then stable gallery identity order",
        "near_tie_cosine_tolerance": NEAR_TIE_COSINE_TOLERANCE,
        "queries_with_near_tie_competitors": int(
            np.sum(near_tie_competitor_counts > 0)
        ),
        "fraction_with_near_tie_competitors": float(
            np.mean(near_tie_competitor_counts > 0)
        ),
        "near_tie_competitors_total": int(np.sum(near_tie_competitor_counts)),
        "maximum_near_tie_competitors_per_query": int(
            np.max(near_tie_competitor_counts)
        ),
        "correct_to_best_competitor_similarity_margin": {
            "minimum": float(np.min(correct_to_best_competitor_margins)),
            "p01": float(np.quantile(correct_to_best_competitor_margins, 0.01)),
            "p05": float(np.quantile(correct_to_best_competitor_margins, 0.05)),
            "median": float(np.median(correct_to_best_competitor_margins)),
            "mean": float(np.mean(correct_to_best_competitor_margins)),
            "maximum": float(np.max(correct_to_best_competitor_margins)),
        },
        "queries_with_nonpositive_similarity_margin": int(
            np.sum(correct_to_best_competitor_margins <= 0.0)
        ),
        "queries_with_similarity_margin_at_most_tolerance": int(
            np.sum(
                correct_to_best_competitor_margins
                <= NEAR_TIE_COSINE_TOLERANCE
            )
        ),
        "recall_at_k": {},
        "exact_tie_recall_bounds_at_k": {},
        "tolerance_recall_bounds_at_k": {},
        "cluster_jackknife": {
            "mean_reciprocal_rank": cluster_jackknife_mean(
                reciprocal, query_clusters, bounded=True
            ),
            "correct_to_best_competitor_similarity_margin": cluster_jackknife_mean(
                correct_to_best_competitor_margins,
                query_clusters,
                bounded=False,
            ),
        },
    }
    for k_value in k_values:
        successes = (ranks <= k_value).astype(np.float64)
        optimistic_successes = (optimistic_ranks <= k_value).astype(np.float64)
        pessimistic_successes = (pessimistic_ranks <= k_value).astype(np.float64)
        tolerance_optimistic_successes = (
            tolerance_optimistic_ranks <= k_value
        ).astype(np.float64)
        tolerance_pessimistic_successes = (
            tolerance_pessimistic_ranks <= k_value
        ).astype(np.float64)
        result["recall_at_k"][str(k_value)] = float(np.mean(successes))
        result["exact_tie_recall_bounds_at_k"][str(k_value)] = {
            "optimistic": float(np.mean(optimistic_successes)),
            "pessimistic": float(np.mean(pessimistic_successes)),
        }
        result["tolerance_recall_bounds_at_k"][str(k_value)] = {
            "optimistic": float(np.mean(tolerance_optimistic_successes)),
            "pessimistic": float(np.mean(tolerance_pessimistic_successes)),
        }
        result["cluster_jackknife"][f"recall_at_{k_value}"] = cluster_jackknife_mean(
            successes, query_clusters, bounded=True
        )
        result["cluster_jackknife"][
            f"pessimistic_recall_at_{k_value}"
        ] = cluster_jackknife_mean(
            pessimistic_successes, query_clusters, bounded=True
        )
        result["cluster_jackknife"][
            f"tolerance_pessimistic_recall_at_{k_value}"
        ] = cluster_jackknife_mean(
            tolerance_pessimistic_successes, query_clusters, bounded=True
        )
    per_cluster = []
    cluster_array = np.asarray(query_clusters)
    for cluster in sorted(set(query_clusters)):
        selected = cluster_array == cluster
        row = {
            "spatial_cluster_id": cluster,
            "queries": int(np.sum(selected)),
            "mean_reciprocal_rank": float(np.mean(reciprocal[selected])),
            "median_rank": float(np.median(ranks[selected])),
            "queries_with_near_tie_competitors": int(
                np.sum(near_tie_competitor_counts[selected] > 0)
            ),
            "mean_correct_to_best_competitor_similarity_margin": float(
                np.mean(correct_to_best_competitor_margins[selected])
            ),
            "minimum_correct_to_best_competitor_similarity_margin": float(
                np.min(correct_to_best_competitor_margins[selected])
            ),
        }
        for k_value in k_values:
            row[f"recall_at_{k_value}"] = float(np.mean(ranks[selected] <= k_value))
            row[f"optimistic_recall_at_{k_value}"] = float(
                np.mean(optimistic_ranks[selected] <= k_value)
            )
            row[f"pessimistic_recall_at_{k_value}"] = float(
                np.mean(pessimistic_ranks[selected] <= k_value)
            )
            row[f"tolerance_optimistic_recall_at_{k_value}"] = float(
                np.mean(tolerance_optimistic_ranks[selected] <= k_value)
            )
            row[f"tolerance_pessimistic_recall_at_{k_value}"] = float(
                np.mean(tolerance_pessimistic_ranks[selected] <= k_value)
            )
        per_cluster.append(row)
    return result, per_cluster


def representation_retrieval_gate(
    cross_release: dict[str, Any], native_reference: dict[str, Any]
) -> dict[str, Any]:
    """Apply fixed absolute, native-sanity, and cluster-deletion stability gates."""
    native_r1 = float(native_reference["recall_at_k"]["1"])
    cross_r1 = float(cross_release["recall_at_k"]["1"])
    native_pessimistic = float(
        native_reference["exact_tie_recall_bounds_at_k"]["1"]["pessimistic"]
    )
    cross_pessimistic = float(
        cross_release["exact_tie_recall_bounds_at_k"]["1"]["pessimistic"]
    )
    native_tolerance_pessimistic = float(
        native_reference["tolerance_recall_bounds_at_k"]["1"]["pessimistic"]
    )
    cross_tolerance_pessimistic = float(
        cross_release["tolerance_recall_bounds_at_k"]["1"]["pessimistic"]
    )
    jackknife = cross_release.get("cluster_jackknife", {}).get("recall_at_1", {})
    deletion_range = jackknife.get("leave_one_cluster_out_range")
    clustered_deletion_minimum = (
        float(deletion_range[0])
        if isinstance(deletion_range, list) and len(deletion_range) == 2
        else None
    )
    interval = jackknife.get("normal_95_interval")
    descriptive_normal_lower = (
        float(interval[0])
        if isinstance(interval, list) and len(interval) == 2
        else None
    )
    pessimistic_jackknife = cross_release.get("cluster_jackknife", {}).get(
        "pessimistic_recall_at_1", {}
    )
    pessimistic_deletion_range = pessimistic_jackknife.get(
        "leave_one_cluster_out_range"
    )
    pessimistic_clustered_deletion_minimum = (
        float(pessimistic_deletion_range[0])
        if isinstance(pessimistic_deletion_range, list)
        and len(pessimistic_deletion_range) == 2
        else None
    )
    pessimistic_interval = pessimistic_jackknife.get("normal_95_interval")
    descriptive_pessimistic_normal_lower = (
        float(pessimistic_interval[0])
        if isinstance(pessimistic_interval, list) and len(pessimistic_interval) == 2
        else None
    )
    tolerance_pessimistic_jackknife = cross_release.get(
        "cluster_jackknife", {}
    ).get("tolerance_pessimistic_recall_at_1", {})
    tolerance_pessimistic_deletion_range = tolerance_pessimistic_jackknife.get(
        "leave_one_cluster_out_range"
    )
    tolerance_pessimistic_clustered_deletion_minimum = (
        float(tolerance_pessimistic_deletion_range[0])
        if isinstance(tolerance_pessimistic_deletion_range, list)
        and len(tolerance_pessimistic_deletion_range) == 2
        else None
    )
    ratio = cross_r1 / native_r1 if native_r1 > 0 else None
    checks = {
        "native_stable_recall_at_1_equals_1": native_r1 >= NATIVE_R1_FLOOR,
        "native_exact_tie_pessimistic_recall_at_1_equals_1": native_pessimistic
        >= NATIVE_R1_FLOOR,
        "native_tolerance_pessimistic_recall_at_1_equals_1": native_tolerance_pessimistic
        >= NATIVE_R1_FLOOR,
        "native_exact_similarity_ties_zero": native_reference[
            "queries_with_similarity_ties"
        ]
        == 0,
        "native_near_tie_competitors_zero": native_reference[
            "queries_with_near_tie_competitors"
        ]
        == 0,
        "cross_stable_recall_at_1_at_least_0_95": cross_r1 >= CROSS_R1_FLOOR,
        "cross_pessimistic_recall_at_1_at_least_0_95": cross_pessimistic
        >= CROSS_R1_FLOOR,
        "cross_tolerance_pessimistic_recall_at_1_at_least_0_95": cross_tolerance_pessimistic
        >= CROSS_R1_FLOOR,
        "cross_to_native_stable_recall_ratio_at_least_0_95": ratio is not None
        and ratio >= CROSS_R1_FLOOR,
        "minimum_leave_one_location_cluster_out_recall_at_1_at_least_0_95": clustered_deletion_minimum
        is not None
        and clustered_deletion_minimum >= CLUSTERED_R1_LOWER_FLOOR,
        "minimum_leave_one_location_cluster_out_pessimistic_recall_at_1_at_least_0_95": pessimistic_clustered_deletion_minimum
        is not None
        and pessimistic_clustered_deletion_minimum >= CLUSTERED_R1_LOWER_FLOOR,
        "minimum_leave_one_location_cluster_out_tolerance_pessimistic_recall_at_1_at_least_0_95": tolerance_pessimistic_clustered_deletion_minimum
        is not None
        and tolerance_pessimistic_clustered_deletion_minimum
        >= CLUSTERED_R1_LOWER_FLOOR,
    }
    return {
        "status": "representation_proxy_pass" if all(checks.values()) else "representation_proxy_fail",
        "pass": bool(all(checks.values())),
        "thresholds_preregistered_in_analysis_code": True,
        "native_reference_recall_at_1": native_r1,
        "native_reference_pessimistic_recall_at_1": native_pessimistic,
        "native_reference_tolerance_pessimistic_recall_at_1": native_tolerance_pessimistic,
        "cross_release_recall_at_1": cross_r1,
        "cross_release_pessimistic_recall_at_1": cross_pessimistic,
        "cross_release_tolerance_pessimistic_recall_at_1": cross_tolerance_pessimistic,
        "cross_to_native_recall_at_1_ratio": ratio,
        "minimum_leave_one_location_cluster_out_recall_at_1": clustered_deletion_minimum,
        "minimum_leave_one_location_cluster_out_pessimistic_recall_at_1": pessimistic_clustered_deletion_minimum,
        "minimum_leave_one_location_cluster_out_tolerance_pessimistic_recall_at_1": tolerance_pessimistic_clustered_deletion_minimum,
        "descriptive_location_clustered_normal_95_lower_bound": descriptive_normal_lower,
        "descriptive_pessimistic_location_clustered_normal_95_lower_bound": descriptive_pessimistic_normal_lower,
        "checks": checks,
    }


def _stable_top_k(scores: np.ndarray, identities: Sequence[str], k_value: int) -> list[int]:
    if len(scores) != len(identities) or not 1 <= k_value <= len(identities):
        raise ValueError("invalid stable top-k inputs")
    return sorted(
        range(len(identities)), key=lambda index: (-float(scores[index]), identities[index])
    )[:k_value]


def cross_site_neighbor_overlap(
    transformed_source: np.ndarray,
    target: np.ndarray,
    sample_ids: Sequence[str],
    clusters: Sequence[str],
    *,
    k_values: Sequence[int] = NEIGHBOR_K_VALUES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare source/target neighbor sets after excluding every same-site year."""
    source_value, target_value = row_normalize(transformed_source), row_normalize(target)
    if source_value.shape != target_value.shape or len(sample_ids) != source_value.shape[0]:
        raise ValueError("cross-site neighbor inputs are misaligned")
    if len(clusters) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("cross-site neighbor identities/clusters are invalid")
    source_similarity, target_similarity = source_value @ target_value.T, target_value @ target_value.T
    per_query = []
    for query_index, (sample_id, cluster) in enumerate(
        zip(sample_ids, clusters, strict=True)
    ):
        candidates = [
            index for index, candidate_cluster in enumerate(clusters) if candidate_cluster != cluster
        ]
        if len(candidates) < max(k_values):
            raise ValueError("too few cross-site candidates for requested neighbor k")
        candidate_ids = [sample_ids[index] for index in candidates]
        row = {"sample_id": sample_id, "spatial_cluster_id": cluster}
        for k_value in k_values:
            source_local = _stable_top_k(
                source_similarity[query_index, candidates], candidate_ids, k_value
            )
            target_local = _stable_top_k(
                target_similarity[query_index, candidates], candidate_ids, k_value
            )
            source_set = {candidates[index] for index in source_local}
            target_set = {candidates[index] for index in target_local}
            intersection = len(source_set & target_set)
            row[f"overlap_at_{k_value}"] = intersection / k_value
            row[f"jaccard_at_{k_value}"] = intersection / (2 * k_value - intersection)
        per_query.append(row)
    result: dict[str, Any] = {
        "site_year_queries": len(per_query),
        "same_spatial_site_all_years_excluded": True,
        "tie_rule": "descending cosine then sample_id",
        "metrics": {},
    }
    per_cluster = []
    for k_value in k_values:
        for metric in ("overlap", "jaccard"):
            key = f"{metric}_at_{k_value}"
            values = [float(row[key]) for row in per_query]
            result["metrics"][key] = cluster_jackknife_mean(
                values, clusters, bounded=True
            )
    cluster_array = np.asarray(clusters)
    for cluster in sorted(set(clusters)):
        selected = cluster_array == cluster
        row: dict[str, Any] = {
            "spatial_cluster_id": cluster,
            "queries": int(np.sum(selected)),
        }
        for k_value in k_values:
            for metric in ("overlap", "jaccard"):
                key = f"{metric}_at_{k_value}"
                row[key] = float(
                    np.mean(
                        [
                            per_query[index][key]
                            for index in np.flatnonzero(selected)
                        ]
                    )
                )
        per_cluster.append(row)
    return result, per_cluster


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_value = row_normalize(np.asarray(left).reshape(1, -1))[0]
    right_value = row_normalize(np.asarray(right).reshape(1, -1))[0]
    cosine = float(np.clip(np.dot(left_value, right_value), -1.0, 1.0))
    return 1.0 - cosine


def _ranking_core(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    event_ids: Sequence[str],
    top_ks: Sequence[int],
) -> dict[str, Any]:
    result = {
        "contrasts": int(len(event_ids)),
        "spearman": spearman_correlation(source_scores, target_scores),
        "kendall_tau_b": kendall_tau_b(source_scores, target_scores),
        "top_k_overlap": {},
    }
    for k_value in top_ks:
        source_top = set(_stable_top_k(source_scores, event_ids, k_value))
        target_top = set(_stable_top_k(target_scores, event_ids, k_value))
        intersection = len(source_top & target_top)
        result["top_k_overlap"][str(k_value)] = {
            "fraction": intersection / k_value,
            "jaccard": intersection / (2 * k_value - intersection),
        }
    return result


def consecutive_manifest_window_contrast_continuity(
    transformed_source: np.ndarray,
    target: np.ndarray,
    sample_ids: Sequence[str],
    clusters: Sequence[str],
    years: Sequence[int],
) -> dict[str, Any]:
    """Compare 48 consecutive annual-window contrasts on sealed sites only.

    These are contrasts between independently materialized rolling 12-period
    manifests.  They are not localized change events or temporal ground truth.
    """
    if not (
        transformed_source.shape == target.shape
        and len(sample_ids) == len(clusters) == len(years) == target.shape[0]
    ):
        raise ValueError("consecutive manifest-window inputs are misaligned")
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        by_cluster[cluster].append(index)
    contrast_ids, contrast_clusters, transition_labels, source_scores, target_scores = (
        [],
        [],
        [],
        [],
        [],
    )
    for cluster in sorted(by_cluster):
        indices = sorted(by_cluster[cluster], key=lambda index: int(years[index]))
        observed_years = [int(years[index]) for index in indices]
        if observed_years != list(EXPECTED_YEARS):
            raise ValueError(f"sealed cluster lacks a four-year panel: {cluster}")
        for left_index, right_index in zip(indices[:-1], indices[1:], strict=True):
            transition = f"{years[left_index]}_to_{years[right_index]}"
            contrast_ids.append(f"{cluster}::{transition}")
            contrast_clusters.append(cluster)
            transition_labels.append(transition)
            source_scores.append(
                _cosine_distance(
                    transformed_source[left_index], transformed_source[right_index]
                )
            )
            target_scores.append(_cosine_distance(target[left_index], target[right_index]))
    if len(contrast_ids) != 48 or len(set(contrast_clusters)) != 16:
        raise ValueError("sealed manifest-contrast endpoint is not 16 sites by three transitions")
    source_array, target_array = np.asarray(source_scores), np.asarray(target_scores)
    core = _ranking_core(source_array, target_array, contrast_ids, top_ks=(5, 10))
    transition_array = np.asarray(transition_labels)
    transition_results = {}
    for transition in ("2023_to_2024", "2024_to_2025", "2025_to_2026"):
        selected = transition_array == transition
        if int(selected.sum()) != 16:
            raise ValueError(f"sealed transition does not contain 16 locations: {transition}")
        selected_ids = [
            contrast_id
            for contrast_id, keep in zip(contrast_ids, selected, strict=True)
            if keep
        ]
        transition_results[transition] = {
            **_ranking_core(
                source_array[selected], target_array[selected], selected_ids, top_ks=(5, 10)
            ),
            "window_role": (
                "rolling_2026_window_contrast_not_a_prospective_change_event"
                if transition == "2025_to_2026"
                else "consecutive_frozen_annual_window_contrast"
            ),
        }
    leave_one_out = []
    cluster_array = np.asarray(contrast_clusters)
    for cluster in sorted(set(contrast_clusters)):
        keep = cluster_array != cluster
        reduced_ids = [
            contrast_id
            for contrast_id, selected in zip(contrast_ids, keep, strict=True)
            if selected
        ]
        reduced = _ranking_core(
            source_array[keep], target_array[keep], reduced_ids, top_ks=(5, 10)
        )
        leave_one_out.append(
            {
                "left_out_spatial_cluster_id": cluster,
                "spearman": reduced["spearman"],
                "kendall_tau_b": reduced["kendall_tau_b"],
                "top_5_overlap_fraction": reduced["top_k_overlap"]["5"]["fraction"],
                "top_10_overlap_fraction": reduced["top_k_overlap"]["10"]["fraction"],
            }
        )
    core["leave_one_spatial_cluster_out_range"] = {
        key: [
            float(min(row[key] for row in leave_one_out)),
            float(max(row[key] for row in leave_one_out)),
        ]
        for key in (
            "spearman",
            "kendall_tau_b",
            "top_5_overlap_fraction",
            "top_10_overlap_fraction",
        )
    }
    core["score"] = "one_minus_cosine_of_pooled_site_year_embeddings"
    core["same_site_year_panel"] = True
    core["endpoint_name"] = "consecutive_manifest_window_contrast_continuity"
    core["transition_results"] = transition_results
    core["temporal_change_detection_claim_allowed"] = False
    return core


def _summarize_per_window(
    rows: list[dict[str, Any]], split_names: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    strata = {"all_216": rows}
    strata.update(
        {
            split_name: [row for row in rows if row["split"] == split_name]
            for split_name in split_names
        }
    )
    for stratum, selected in strata.items():
        entry: dict[str, Any] = {
            "site_years": len(selected),
            "spatial_clusters": len({row["spatial_cluster_id"] for row in selected}),
            "metrics": {},
        }
        clusters = [row["spatial_cluster_id"] for row in selected]
        for metric in (
            "query_lattice_linear_cka",
            "query_lattice_row_normalized_linear_cka",
            "query_lattice_paired_cosine_mean",
        ):
            values = [float(row[metric]) for row in selected]
            entry["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "cluster_jackknife": cluster_jackknife_mean(
                    values, clusters, bounded=metric != "query_lattice_paired_cosine_mean"
                ),
            }
        result[stratum] = entry
    return result


def _geometry_by_stratum(
    old_pooled: np.ndarray,
    new_pooled: np.ndarray,
    split_names: Sequence[str],
    sample_splits: Sequence[str],
    clusters: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    strata = {"all_216": np.ones(len(sample_splits), dtype=bool)}
    strata.update(
        {
            split_name: np.asarray(sample_splits) == split_name
            for split_name in split_names
        }
    )
    cluster_array = np.asarray(clusters)
    for stratum, selected in strata.items():
        metrics = geometry_metrics(old_pooled[selected], new_pooled[selected])
        metrics["leave_one_spatial_cluster_out"] = leave_one_cluster_out_geometry(
            old_pooled[selected], new_pooled[selected], cluster_array[selected].tolist()
        )
        result[stratum] = metrics
    return result


def _bridge_bundle(
    source_calibration: np.ndarray,
    target_calibration: np.ndarray,
    calibration_clusters: Sequence[str],
    calibration_spatial_keys: Sequence[str],
    local_query_indices: np.ndarray,
) -> tuple[dict[str, AffineMap], dict[str, Any]]:
    dimensions = source_calibration.shape[1]
    identity = identity_map(dimensions)
    mean_shift = fit_mean_shift_translation(source_calibration, target_calibration)
    procrustes = fit_translated_orthogonal_procrustes(
        source_calibration, target_calibration
    )
    ridge_selection = select_ridge_multiplier(
        source_calibration,
        target_calibration,
        calibration_clusters,
        calibration_spatial_keys,
        local_query_indices,
    )
    ridge, absolute_alpha = fit_affine_ridge(
        source_calibration,
        target_calibration,
        ridge_selection["selected_alpha_multiplier"],
    )
    ridge_selection["final_absolute_alpha"] = absolute_alpha
    bridges = {
        "identity_no_bridge": identity,
        "mean_shift_translation_only": mean_shift,
        "translated_orthogonal_procrustes": procrustes,
        "affine_ridge": ridge,
    }
    if tuple(bridges) != BRIDGE_METHODS:
        raise AssertionError("bridge method order drifted from the preanalysis contract")
    metadata = {
        "ridge_selection": ridge_selection,
        "bridges": {
            method: {
                "parameter_sha256": bridge.digest(),
                "fit_site_years": len(calibration_clusters),
                "fit_tokens": int(source_calibration.shape[0]),
                "fit_spatial_clusters": len(set(calibration_clusters)),
            }
            for method, bridge in bridges.items()
        },
    }
    metadata["bridges"]["identity_no_bridge"].update(
        {"fit_site_years": 0, "fit_tokens": 0, "fit_spatial_clusters": 0}
    )
    return bridges, metadata


def analyze(
    evidence: FrozenEvidence,
    *,
    output_hash_workers: int,
    retrieval_chunk_size: int,
    preanalysis_lock_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _require_sha(preanalysis_lock_sha256, "preanalysis lock")
    live_output_closure = validate_live_output_hashes(evidence.pairs, output_hash_workers)
    sample_ids = sorted(evidence.exact_records)
    old_lattice, new_lattice = [], []
    old_pooled, new_pooled, rows = [], [], []
    clusters, spatial_keys, years, splits = [], [], [], []
    query_indices = query_flat_indices()
    dimensions: tuple[int, int] | None = None
    for sample_id in sample_ids:
        record, pair = evidence.exact_records[sample_id], evidence.pairs[sample_id]
        old_grid, new_grid, valid, raster_contract = read_verified_pair(pair, record)
        old_tokens, old_query = extract_lattice(old_grid, valid)
        new_tokens, new_query = extract_lattice(new_grid, valid)
        current_dimensions = (old_tokens.shape[1], new_tokens.shape[1])
        if dimensions is None:
            dimensions = current_dimensions
        elif current_dimensions != dimensions:
            raise ValueError(f"embedding dimensions drift across records: {sample_id}")
        if current_dimensions[0] != current_dimensions[1]:
            raise ValueError("identity/orthogonal cache baselines require equal release dimensions")
        old_lattice.append(old_tokens)
        new_lattice.append(new_tokens)
        old_pooled.append(old_tokens.mean(axis=0))
        new_pooled.append(new_tokens.mean(axis=0))
        cluster = record["spatial_cluster_id"]
        split_name = evidence.split_by_sample[sample_id]
        clusters.append(cluster)
        spatial_keys.append(record["spatial_key"])
        years.append(int(record["year"]))
        splits.append(split_name)
        rows.append(
            {
                "sample_id": sample_id,
                "window": record["window_name"],
                "spatial_cluster_id": cluster,
                "spatial_key": record["spatial_key"],
                "year": int(record["year"]),
                "split": split_name,
                "bad_proxy_mean": float(record["bad_proxy_mean"]),
                "height": int(raster_contract["height"]),
                "width": int(raster_contract["width"]),
                "embedding_bands": int(raster_contract["count"]),
                "valid_grid_tokens": int(pair["v1_value_health"]["usable_tokens"]),
                "gallery_lattice_tokens": int(old_tokens.shape[0]),
                "query_lattice_tokens": int(old_query.shape[0]),
                "query_lattice_linear_cka": linear_cka(old_query, new_query),
                "query_lattice_row_normalized_linear_cka": linear_cka(
                    row_normalize(old_query), row_normalize(new_query)
                ),
                "query_lattice_paired_cosine_mean": paired_cosine_mean(
                    old_query, new_query
                ),
            }
        )

    old_lattice_array = np.stack(old_lattice)
    new_lattice_array = np.stack(new_lattice)
    del old_lattice, new_lattice
    old_query_array = old_lattice_array[:, query_indices]
    new_query_array = new_lattice_array[:, query_indices]
    old_pooled_array = np.stack(old_pooled)
    new_pooled_array = np.stack(new_pooled)
    if old_lattice_array.shape[:2] != (216, 256) or old_query_array.shape[:2] != (
        216,
        64,
    ):
        raise ValueError("analysis lattice contract is not 216x256 with 216x64 queries")

    split_order = ("calibration", "embargo", "sealed_test", "disclosed_audit")
    descriptive = {
        "per_window_query_lattice": _summarize_per_window(rows, split_order),
        "pooled_site_year_geometry": _geometry_by_stratum(
            old_pooled_array,
            new_pooled_array,
            split_order,
            splits,
            clusters,
        ),
        "headline_compatibility_endpoint": False,
        "labels_used": 0,
    }

    calibration_mask = np.asarray(splits) == "calibration"
    test_mask = np.asarray(splits) == "sealed_test"
    if int(calibration_mask.sum()) != 120 or int(test_mask.sum()) != 64:
        raise ValueError("analysis split is not 120 calibration and 64 sealed site-years")
    calibration_clusters = np.asarray(clusters)[calibration_mask].tolist()
    calibration_keys = np.asarray(spatial_keys)[calibration_mask].tolist()
    test_clusters = np.asarray(clusters)[test_mask].tolist()
    test_sample_ids = np.asarray(sample_ids)[test_mask].tolist()
    test_years = np.asarray(years)[test_mask].astype(int).tolist()

    # The 8x8 anchor lattice is both preregistered and large enough to fit the
    # 768-D bridges (7,680 calibration pairs).  Keeping the remaining 16x16
    # points out of fitting also makes the gallery denser than the anchors.
    old_calibration = old_query_array[calibration_mask].reshape(-1, dimensions[0])
    new_calibration = new_query_array[calibration_mask].reshape(-1, dimensions[1])
    old_to_new, old_to_new_meta = _bridge_bundle(
        old_calibration,
        new_calibration,
        calibration_clusters,
        calibration_keys,
        np.arange(64, dtype=np.int64),
    )
    new_to_old, new_to_old_meta = _bridge_bundle(
        new_calibration,
        old_calibration,
        calibration_clusters,
        calibration_keys,
        np.arange(64, dtype=np.int64),
    )

    old_test_gallery = old_lattice_array[test_mask].reshape(-1, dimensions[0])
    new_test_gallery = new_lattice_array[test_mask].reshape(-1, dimensions[1])
    old_test_queries = old_query_array[test_mask].reshape(-1, dimensions[0])
    new_test_queries = new_query_array[test_mask].reshape(-1, dimensions[1])
    test_query_clusters = [cluster for cluster in test_clusters for _ in range(64)]
    correct_gallery_indices = np.concatenate(
        [record_index * 256 + query_indices for record_index in range(64)]
    )

    native_retrieval: dict[str, Any] = {}
    retrieval_cluster_rows: list[dict[str, Any]] = []
    for release_name, queries, gallery in (
        ("v1_native", old_test_queries, old_test_gallery),
        ("v1_2_native", new_test_queries, new_test_gallery),
    ):
        metric, cluster_rows = exact_identity_retrieval(
            queries,
            gallery,
            correct_gallery_indices,
            test_query_clusters,
            chunk_size=retrieval_chunk_size,
        )
        native_retrieval[release_name] = metric
        retrieval_cluster_rows.extend(
            {"endpoint": "exact_token_retrieval", "direction": release_name, "method": "native", **row}
            for row in cluster_rows
        )

    directional_specs = {
        "v1_2_query_to_v1_gallery": {
            "source_queries": new_test_queries,
            "source_pooled": new_pooled_array[test_mask],
            "target_gallery": old_test_gallery,
            "target_pooled": old_pooled_array[test_mask],
            "bridges": new_to_old,
            "bridge_metadata": new_to_old_meta,
            "native_reference": "v1_native",
        },
        "v1_query_to_v1_2_gallery": {
            "source_queries": old_test_queries,
            "source_pooled": old_pooled_array[test_mask],
            "target_gallery": new_test_gallery,
            "target_pooled": new_pooled_array[test_mask],
            "bridges": old_to_new,
            "bridge_metadata": old_to_new_meta,
            "native_reference": "v1_2_native",
        },
    }
    directional_results: dict[str, Any] = {}
    for direction, spec in directional_specs.items():
        method_results = {}
        for method, bridge in spec["bridges"].items():
            transformed_queries = bridge.transform(spec["source_queries"])
            retrieval, cluster_rows = exact_identity_retrieval(
                transformed_queries,
                spec["target_gallery"],
                correct_gallery_indices,
                test_query_clusters,
                chunk_size=retrieval_chunk_size,
            )
            target_native = native_retrieval[spec["native_reference"]]
            retrieval["fraction_of_target_native_recall_at_k"] = {
                str(k_value): (
                    retrieval["recall_at_k"][str(k_value)]
                    / target_native["recall_at_k"][str(k_value)]
                    if target_native["recall_at_k"][str(k_value)] > 0
                    else None
                )
                for k_value in RETRIEVAL_K_VALUES
            }
            retrieval_gate = representation_retrieval_gate(retrieval, target_native)
            transformed_pooled = bridge.transform(spec["source_pooled"])
            neighbors, neighbor_cluster_rows = cross_site_neighbor_overlap(
                transformed_pooled,
                spec["target_pooled"],
                test_sample_ids,
                test_clusters,
            )
            contrast = consecutive_manifest_window_contrast_continuity(
                transformed_pooled,
                spec["target_pooled"],
                test_sample_ids,
                test_clusters,
                test_years,
            )
            method_results[method] = {
                "bridge": spec["bridge_metadata"]["bridges"][method],
                "exact_token_retrieval": retrieval,
                "representation_retrieval_gate": retrieval_gate,
                "cross_site_neighbor_continuity": neighbors,
                "consecutive_manifest_window_contrast_continuity": contrast,
            }
            retrieval_cluster_rows.extend(
                {
                    "endpoint": "exact_token_retrieval",
                    "direction": direction,
                    "method": method,
                    **row,
                }
                for row in cluster_rows
            )
            retrieval_cluster_rows.extend(
                {
                    "endpoint": "cross_site_neighbor_continuity",
                    "direction": direction,
                    "method": method,
                    **row,
                }
                for row in neighbor_cluster_rows
            )
        directional_results[direction] = {
            "native_target_reference": spec["native_reference"],
            "ridge_calibration_only_selection": spec["bridge_metadata"][
                "ridge_selection"
            ],
            "methods": method_results,
        }

    proxy_passes = []
    for direction_name, direction in directional_results.items():
        for method, values in direction["methods"].items():
            proxy_passes.append(
                {
                    "direction": direction_name,
                    "method": method,
                    **values["representation_retrieval_gate"],
                }
            )

    summary = {
        "schema": "olmoearth-release-full-analysis-v1",
        "status": "complete",
        "analysis_code_sha256": file_sha256(Path(__file__)),
        "preanalysis_lock_sha256": preanalysis_lock_sha256,
        "frozen_evidence": evidence.evidence_hashes,
        "promoted_execution_contract": evidence.execution_contract,
        "live_output_closure": live_output_closure,
        "sample_contract": {
            "validated_site_year_pairs": 216,
            "validated_spatial_clusters": 54,
            "years": list(EXPECTED_YEARS),
            "labels": 0,
            "calibration": {"spatial_clusters": 30, "site_years": 120},
            "sealed_test": {"spatial_clusters": 16, "site_years": 64},
            "embargo": {"spatial_clusters": 6, "site_years": 24},
            "disclosed_audit": {"spatial_clusters": 2, "site_years": 8},
            "bridge_fit": "calibration only",
            "ridge_selection": "calibration-only spatial-x-block validation",
            "headline_evaluation": "sealed test only",
            "embargo_or_disclosed_used_for_fit_or_headline_evaluation": False,
            "units": {
                "per_window_spatial_drift": "site-year summarized with location-cluster jackknife",
                "pooled_geometry": "site-year; pairwise distances are not independent replicates",
                "exact_retrieval": "exact spatial token; uncertainty clustered by location",
                "cross_site_neighbors": "site-year query; same-location years excluded",
                "manifest_window_contrast": "consecutive annual-window contrast; uncertainty deletion unit is location",
            },
        },
        "token_contract": {
            "gallery_lattice": "16x16 deterministic center-of-bin pixel lattice",
            "gallery_tokens_per_site_year": 256,
            "query_lattice_positions_within_gallery": list(QUERY_LATTICE_POSITIONS),
            "query_tokens_per_site_year": 64,
            "bridge_fit_tokens": "calibration split 8x8 anchor lattice only",
            "invalid_lattice_policy": "fail closed",
            "retrieval_similarity": "row-L2-normalized cosine",
            "exact_identity": "sample_id plus exact 16x16 lattice coordinate",
            "exact_tie_policy": "report optimistic, stable-identity, and pessimistic ranks",
            "near_tie_policy": {
                "cosine_tolerance": NEAR_TIE_COSINE_TOLERANCE,
                "report_correct_to_best_competitor_margin": True,
                "report_tolerance_optimistic_and_pessimistic_bounds": True,
            },
            "bridge_methods": list(BRIDGE_METHODS),
        },
        "descriptive_release_drift": descriptive,
        "sealed_test_compatibility": {
            "native_retrieval_sanity_ceiling": native_retrieval,
            "directions": directional_results,
            "research_promotion_gate": {
                "representation_proxy_results": proxy_passes,
                "task_performance_within_one_percentage_point": None,
                "full_cache_compatibility_promoted": False,
                "reason": "label-free identity retrieval cannot establish downstream task utility",
            },
        },
        "claims_allowed": [
            "split_stratified_descriptive_release_drift_on_216_frozen_exact_inputs",
            "calibration_only_bridge_fit_and_sealed_test_representation_identity_retrieval",
            "sealed_test_cross_site_neighbor_and_consecutive_manifest_window_contrast_continuity",
        ],
        "claims_forbidden": [
            "task_accuracy_or_accuracy_improvement",
            "negative_transfer_or_negative_transfer_reduction",
            "cloud_robustness",
            "korea_jeju_or_other_population_generalization",
            "semantic_retrieval_correctness",
            "causal_change_attribution",
            "temporal_change_detection",
            "input_effect_vs_release_effect",
            "model_native_or_operational_backward_compatibility",
            "full_cache_compatibility_without_downstream_task_evaluation",
            "population_confidence_interval_or_significance",
            "post_sealed_bridge_method_selection_or_deployment_cherry_picking",
        ],
        "limitations": [
            "All 216 records are label-free legacy-recipe observations from one Jeju grid.",
            "The sealed endpoint tests representation identity and ranking, not task accuracy or decision utility.",
            "The 54 spatial clusters, not tokens or pairwise distances, are the independent uncertainty units.",
            "Cluster jackknife intervals are descriptive because this deterministic grid is not a probability sample.",
            "No BestClear input-recipe cell exists, so release shift cannot be separated from an input-recipe interaction.",
            "Consecutive annual-window contrasts are not localized change events; the 2025-to-2026 cell uses the rolling-2026 manifest.",
            "The 95% retrieval fraction is only one necessary representation proxy; the task-performance gate remains unavailable.",
            "All four preregistered bridge methods must be reported as one family; selecting a winner after this sealed run requires a new untouched test split.",
        ],
    }
    return summary, rows, retrieval_cluster_rows


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("cannot serialize an empty CSV")
    fieldnames = sorted({key for row in rows for key in row})
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        output.seek(0)
        return output.read().encode("utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-summary", type=Path, required=True)
    parser.add_argument("--evidence-complete", type=Path, required=True)
    parser.add_argument("--paired-outputs", type=Path, required=True)
    parser.add_argument("--exact-inputs", type=Path, required=True)
    parser.add_argument("--exact-complete", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-complete", type=Path, required=True)
    parser.add_argument("--v1-run-summary", type=Path, required=True)
    parser.add_argument("--v1-complete", type=Path, required=True)
    parser.add_argument("--v1-2-run-summary", type=Path, required=True)
    parser.add_argument("--v1-2-complete", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-hash-workers", type=int, default=2)
    parser.add_argument("--retrieval-chunk-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(
            f"refusing existing one-time analysis directory: {args.output_dir}"
        )
    if not 1 <= args.output_hash_workers <= 2:
        raise SystemExit("--output-hash-workers must be one or two")
    if not 1 <= args.retrieval_chunk_size <= 512:
        raise SystemExit("--retrieval-chunk-size must be between one and 512")
    evidence = validate_frozen_evidence(
        evidence_summary_path=args.evidence_summary,
        evidence_complete_path=args.evidence_complete,
        paired_outputs_path=args.paired_outputs,
        exact_inputs_path=args.exact_inputs,
        exact_complete_path=args.exact_complete,
        split_manifest_path=args.split_manifest,
        split_complete_path=args.split_complete,
        v1_run_summary_path=args.v1_run_summary,
        v1_complete_path=args.v1_complete,
        v1_2_run_summary_path=args.v1_2_run_summary,
        v1_2_complete_path=args.v1_2_complete,
    )
    lock_path, lock_sha256, lock_payload = create_preanalysis_lock(
        args.output_dir,
        evidence,
        output_hash_workers=args.output_hash_workers,
        retrieval_chunk_size=args.retrieval_chunk_size,
        argv=sys.argv,
    )
    summary, per_window_rows, per_cluster_rows = analyze(
        evidence,
        output_hash_workers=args.output_hash_workers,
        retrieval_chunk_size=args.retrieval_chunk_size,
        preanalysis_lock_sha256=lock_sha256,
    )
    if file_sha256(lock_path) != lock_sha256:
        raise ValueError("preanalysis lock changed during analysis")
    runtime_at_completion = assert_analysis_runtime_unchanged(
        lock_payload["runtime"]
    )
    initial_analyzer_code_contract = validate_analyzer_code_contract(
        lock_payload.get("analyzer_code_contract"), require_live_match=False
    )
    live_analyzer_code_contract = verify_analyzer_code_stability(
        initial_analyzer_code_contract
    )
    code_marker_path = args.output_dir / "POST_ANALYSIS_CODE_VERIFICATION.json"
    atomic_create(
        code_marker_path,
        canonical_bytes(
            {
                "schema": ANALYZER_POST_CODE_SCHEMA,
                "status": "verified",
                "initial_analyzer_code_contract": initial_analyzer_code_contract,
                "live_analyzer_code_contract": live_analyzer_code_contract,
                "error": None,
            }
        ),
    )
    summary["analysis_code_sha256"] = initial_analyzer_code_contract["owner"][
        "sha256"
    ]
    summary["analysis_code_contract"] = initial_analyzer_code_contract
    summary["post_analysis_code_verified"] = True
    summary["post_analysis_code_verification"] = {
        "path": code_marker_path.resolve().as_posix(),
        "sha256": file_sha256(code_marker_path),
    }
    summary["upstream_code_contracts"] = lock_payload["upstream_code_contracts"]
    summary["runtime_fingerprint_sha256"] = runtime_at_completion[
        "fingerprint_sha256"
    ]
    summary_path = args.output_dir / "analysis_summary.json"
    per_window_path = args.output_dir / "per_window_metrics.csv"
    per_cluster_path = args.output_dir / "sealed_test_per_cluster_metrics.csv"
    atomic_create(summary_path, canonical_bytes(summary))
    atomic_create(per_window_path, _csv_bytes(per_window_rows))
    atomic_create(per_cluster_path, _csv_bytes(per_cluster_rows))
    if file_sha256(lock_path) != lock_sha256:
        raise ValueError("preanalysis lock changed during analysis")
    completion_runtime = assert_analysis_runtime_unchanged(
        lock_payload["runtime"]
    )
    completion_analyzer_code_contract = verify_analyzer_code_stability(
        initial_analyzer_code_contract
    )
    if completion_analyzer_code_contract != live_analyzer_code_contract:
        raise ValueError("analyzer code contract drifted before completion sealing")
    marker = {
        "schema": "olmoearth-release-full-analysis-completion-v1",
        "status": "complete",
        "analysis_summary_sha256": file_sha256(summary_path),
        "per_window_metrics_sha256": file_sha256(per_window_path),
        "sealed_test_per_cluster_metrics_sha256": file_sha256(per_cluster_path),
        "analysis_code_sha256": initial_analyzer_code_contract["owner"]["sha256"],
        "analysis_code_contract_sha256": initial_analyzer_code_contract[
            "inventory_sha256"
        ],
        "post_analysis_code_verified": True,
        "post_analysis_code_verification_sha256": file_sha256(code_marker_path),
        "full_runner_code_contract_sha256": lock_payload[
            "upstream_code_contracts"
        ]["full_runner"]["inventory_sha256"],
        "finalizer_code_contract_sha256": lock_payload[
            "upstream_code_contracts"
        ]["finalizer"]["inventory_sha256"],
        "preanalysis_lock_sha256": lock_sha256,
        "runtime_fingerprint_sha256": completion_runtime[
            "fingerprint_sha256"
        ],
    }
    validate_analyzer_code_evidence(
        lock_payload=read_json(lock_path),
        analysis_summary=read_json(summary_path),
        analysis_root=args.output_dir,
        completion=marker,
        require_live_match=True,
    )
    atomic_create(args.output_dir / "ANALYSIS_COMPLETE.json", canonical_bytes(marker))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
