#!/usr/bin/env python3
"""Canonical scientific execution contract for the OlmoEarth release audit.

The raw YAML files intentionally differ between the batch gate and full run
(output layer, batch size, and model environment).  This module parses the
*resolved* YAML and separates those allowed execution bindings from the model
input/merging semantics that must remain byte-for-byte equivalent as canonical
JSON values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


SENTINEL2_LAYERS = [
    "sentinel2_l2a",
    *[f"sentinel2_l2a.{index}" for index in range(1, 12)],
]
SENTINEL2_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]

RELEASE_SPECS: dict[str, dict[str, str]] = {
    "allenai/OlmoEarth-v1-Base": {
        "release_id": "olmoearth_v1_base",
        "model_env": "OLMO_V1_MODEL_PATH",
        "batch_output_layer": "embeddings_batch_gate_v1",
        "full_output_layer": "embeddings_full_v1_legacy",
    },
    "allenai/OlmoEarth-v1_2-Base": {
        "release_id": "olmoearth_v1_2_base",
        "model_env": "OLMO_V1_2_MODEL_PATH",
        "batch_output_layer": "embeddings_batch_gate_v1_2",
        "full_output_layer": "embeddings_full_v1_2_legacy",
    },
}

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
_GPU_UUID = re.compile(r"GPU-[0-9A-Fa-f-]+\Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    before = path.stat()
    digest = _file_sha256(path)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise ValueError(f"file changed while fingerprinting: {path}")
    return {
        "path": path.relative_to(relative_to).as_posix()
        if relative_to is not None
        else path.as_posix(),
        "bytes": after.st_size,
        "sha256": digest,
    }


def fingerprint_python_sources(package_root: Path) -> dict[str, Any]:
    """Hash the deterministic ``*.py`` inventory below one imported package root."""

    root = package_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"rslearn package root is not a directory: {root}")
    records = []
    for path in sorted(root.rglob("*.py"), key=lambda value: value.as_posix()):
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"rslearn Python source escapes package root: {path}")
        records.append(_stable_file_record(resolved, relative_to=root))
    if not records:
        raise ValueError(f"rslearn package contains no Python sources: {root}")
    rendered = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "inventory_sha256": hashlib.sha256(rendered).hexdigest(),
    }


def _git_source_state(package_root: Path) -> dict[str, Any] | None:
    git = shutil.which("git")
    if git is None:
        return None
    probe = subprocess.run(
        [git, "-C", package_root.as_posix(), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        return None
    repository = Path(probe.stdout.strip()).resolve(strict=True)
    commit = subprocess.run(
        [git, "-C", repository.as_posix(), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", commit) is None:
        raise ValueError(f"invalid rslearn git commit: {commit!r}")
    status = subprocess.run(
        [
            git,
            "-C",
            repository.as_posix(),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "repository_root": repository.as_posix(),
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _resolve_shebang_interpreter(entrypoint: Path, shebang: str) -> Path:
    try:
        components = shlex.split(shebang[2:].strip())
    except ValueError as exc:
        raise ValueError(f"invalid rslearn entrypoint shebang: {shebang!r}") from exc
    if not components:
        raise ValueError("rslearn entrypoint has an empty shebang")
    executable = components[0]
    if Path(executable).name == "env":
        arguments = components[1:]
        if arguments[:1] == ["-S"]:
            arguments = shlex.split(" ".join(arguments[1:]))
        while arguments and arguments[0].startswith("-"):
            arguments.pop(0)
        if not arguments:
            raise ValueError(f"cannot resolve env shebang: {shebang!r}")
        resolved = shutil.which(arguments[0])
        if resolved is None:
            raise ValueError(f"shebang interpreter is absent from PATH: {arguments[0]}")
        candidate = Path(resolved).absolute()
    else:
        candidate = Path(executable).absolute()
    if not candidate.is_file():
        raise ValueError(f"shebang interpreter is not a file: {candidate}")
    # Preserve the venv invocation path for the import probe.  Executing the
    # resolved symlink target can bypass the adjacent pyvenv.cfg and import a
    # different site-packages tree than the console script actually uses.
    return candidate


def fingerprint_rslearn_runtime(rslearn: Path) -> dict[str, Any]:
    """Bind an rslearn console script to its interpreter and imported Python code.

    The package probe runs through the console script's own shebang interpreter,
    with ``PYTHONPATH`` removed, so the evidence describes the subprocess runtime
    rather than whichever Python happened to launch the audit runner.
    """

    entrypoint = rslearn.expanduser().resolve(strict=True)
    if not entrypoint.is_file():
        raise ValueError(f"rslearn entrypoint is not a regular file: {entrypoint}")
    with entrypoint.open("rb") as source:
        first_line = source.readline(4096)
    try:
        shebang = first_line.decode("utf-8").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise ValueError(f"rslearn entrypoint shebang is not UTF-8: {entrypoint}") from exc
    if not shebang.startswith("#!"):
        raise ValueError(f"rslearn entrypoint lacks a shebang: {entrypoint}")
    interpreter_invocation = _resolve_shebang_interpreter(entrypoint, shebang)
    interpreter = interpreter_invocation.resolve(strict=True)
    probe_code = (
        "import importlib.metadata,json,pathlib,rslearn,sys;"
        "roots=[pathlib.Path(p).resolve().as_posix() for p in rslearn.__path__];"
        "print(json.dumps({'sys_executable':pathlib.Path(sys.executable).resolve().as_posix(),"
        "'python_version':sys.version,'implementation':sys.implementation.name,"
        "'rslearn_version':importlib.metadata.version('rslearn'),'package_roots':roots},"
        "sort_keys=True))"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [interpreter_invocation.as_posix(), "-c", probe_code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError("rslearn interpreter probe did not return JSON") from exc
    if Path(probe.get("sys_executable", "")).resolve() != interpreter:
        raise ValueError("rslearn shebang interpreter differs from probed sys.executable")
    roots = probe.get("package_roots")
    if not isinstance(roots, list) or len(roots) != 1:
        raise ValueError(f"rslearn import must resolve to exactly one package root: {roots!r}")
    package_root = Path(roots[0]).resolve(strict=True)
    source_inventory = fingerprint_python_sources(package_root)
    result = {
        "schema": "olmoearth-rslearn-runtime-fingerprint-v1",
        "entrypoint": {
            **_stable_file_record(entrypoint),
            "shebang": shebang,
        },
        "interpreter": {
            **_stable_file_record(interpreter),
            "invocation_path": interpreter_invocation.as_posix(),
            "version": probe.get("python_version"),
            "implementation": probe.get("implementation"),
        },
        "rslearn_package": {
            "version": probe.get("rslearn_version"),
            "root": package_root.as_posix(),
            "python_sources": source_inventory,
            "git": _git_source_state(package_root),
        },
    }
    return validate_rslearn_runtime_fingerprint(result)


def validate_rslearn_runtime_fingerprint(value: Any) -> dict[str, Any]:
    """Validate and normalize a persisted runtime fingerprint."""

    fingerprint = _mapping(value, "rslearn runtime fingerprint")
    if fingerprint.get("schema") != "olmoearth-rslearn-runtime-fingerprint-v1":
        raise ValueError("unrecognized rslearn runtime fingerprint schema")
    entrypoint = _mapping(fingerprint.get("entrypoint"), "rslearn entrypoint")
    interpreter = _mapping(fingerprint.get("interpreter"), "rslearn interpreter")
    package = _mapping(fingerprint.get("rslearn_package"), "rslearn package")
    inventory = _mapping(package.get("python_sources"), "rslearn source inventory")
    for record, label in ((entrypoint, "entrypoint"), (interpreter, "interpreter")):
        path = record.get("path")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError(f"rslearn {label} path must be absolute")
        if not isinstance(record.get("bytes"), int) or record["bytes"] <= 0:
            raise ValueError(f"rslearn {label} byte count is invalid")
        validate_sha256(record.get("sha256"), f"rslearn {label}")
    if not isinstance(entrypoint.get("shebang"), str) or not entrypoint[
        "shebang"
    ].startswith("#!"):
        raise ValueError("rslearn entrypoint shebang is invalid")
    for key in ("version", "implementation"):
        if not isinstance(interpreter.get(key), str) or not interpreter[key]:
            raise ValueError(f"rslearn interpreter {key} is invalid")
    invocation_path = interpreter.get("invocation_path")
    if not isinstance(invocation_path, str) or not Path(invocation_path).is_absolute():
        raise ValueError("rslearn interpreter invocation path must be absolute")
    if not isinstance(package.get("version"), str) or not package["version"]:
        raise ValueError("rslearn package version is invalid")
    if not isinstance(package.get("root"), str) or not Path(package["root"]).is_absolute():
        raise ValueError("rslearn package root must be absolute")
    if not isinstance(inventory.get("files"), int) or inventory["files"] < 1:
        raise ValueError("rslearn source inventory file count is invalid")
    if not isinstance(inventory.get("bytes"), int) or inventory["bytes"] < 1:
        raise ValueError("rslearn source inventory byte count is invalid")
    validate_sha256(inventory.get("inventory_sha256"), "rslearn source inventory")
    git = package.get("git")
    if git is not None:
        git = _mapping(git, "rslearn git state")
        if not isinstance(git.get("repository_root"), str) or not Path(
            git["repository_root"]
        ).is_absolute():
            raise ValueError("rslearn git repository root must be absolute")
        if not isinstance(git.get("commit"), str) or _REVISION.fullmatch(git["commit"]) is None:
            raise ValueError("rslearn git commit is invalid")
        if not isinstance(git.get("dirty"), bool):
            raise ValueError("rslearn git dirty flag is invalid")
        validate_sha256(git.get("status_sha256"), "rslearn git status")
    return copy.deepcopy(fingerprint)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _at(value: Any, path: tuple[str | int, ...]) -> Any:
    current = value
    traversed: list[str] = []
    for component in path:
        traversed.append(str(component))
        try:
            if isinstance(component, int):
                current = _list(current, ".".join(traversed[:-1]))[component]
            else:
                current = _mapping(current, ".".join(traversed[:-1]) or "root")[component]
        except (KeyError, IndexError) as exc:
            raise ValueError(f"resolved config is missing {'.'.join(traversed)}") from exc
    return current


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} drifted: expected {expected!r}, found {actual!r}")


def validate_resolved_config(
    path: Path,
    *,
    model_env: str,
    output_layer: str,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    """Parse and normalize one resolved rslearn prediction configuration."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid resolved YAML: {path}") from exc
    root = _mapping(payload, "root")
    if set(root) != {"model", "data", "trainer"}:
        raise ValueError(f"unexpected resolved config root keys: {sorted(root)}")

    encoder = _list(
        _at(root, ("model", "init_args", "model", "init_args", "encoder")),
        "model encoder",
    )
    decoder = _list(
        _at(root, ("model", "init_args", "model", "init_args", "decoder")),
        "model decoder",
    )
    transforms = _list(
        _at(root, ("data", "init_args", "predict_config", "transforms")),
        "prediction transforms",
    )
    callbacks = _list(_at(root, ("trainer", "callbacks")), "trainer callbacks")
    if len(encoder) != 1 or len(decoder) != 1 or len(transforms) != 1 or len(callbacks) != 1:
        raise ValueError("audit config requires exactly one encoder/decoder/transform/writer")

    encoder_args = _mapping(_at(encoder[0], ("init_args",)), "encoder init_args")
    normalize_args = _mapping(_at(transforms[0], ("init_args",)), "normalize init_args")
    writer_args = _mapping(_at(callbacks[0], ("init_args",)), "writer init_args")
    merger = _mapping(_at(writer_args, ("merger",)), "writer merger")
    merger_args = _mapping(_at(merger, ("init_args",)), "merger init_args")
    data_args = _mapping(_at(root, ("data", "init_args")), "data init_args")
    prediction = _mapping(_at(data_args, ("predict_config",)), "predict_config")
    inputs = _mapping(_at(data_args, ("inputs",)), "inputs")
    if set(inputs) != {"sentinel2_l2a"}:
        raise ValueError(f"input modality drift: {sorted(inputs)}")
    sentinel = _mapping(inputs["sentinel2_l2a"], "sentinel2_l2a input")

    checks = (
        (
            _at(root, ("model", "class_path")),
            "rslearn.train.lightning_module.RslearnLightningModule",
            "lightning module",
        ),
        (
            _at(root, ("model", "init_args", "model", "class_path")),
            "rslearn.models.singletask.SingleTaskModel",
            "task model",
        ),
        (
            _at(encoder[0], ("class_path",)),
            "rslearn.models.olmoearth_pretrain.model.OlmoEarth",
            "encoder class",
        ),
        (encoder_args.get("model_path"), "${" + model_env + "}", "model path environment"),
        (encoder_args.get("patch_size"), 4, "encoder patch_size"),
        (encoder_args.get("use_legacy_timestamps"), True, "legacy timestamp mode"),
        (
            _at(decoder[0], ("class_path",)),
            "rslearn.train.tasks.embedding.EmbeddingHead",
            "decoder class",
        ),
        (
            _at(root, ("data", "class_path")),
            "rslearn.train.data_module.RslearnDataModule",
            "data module",
        ),
        (data_args.get("path"), "${DATASET_PATH}", "dataset path environment"),
        (sentinel.get("data_type"), "raster", "input data type"),
        (sentinel.get("layers"), SENTINEL2_LAYERS, "Sentinel-2 layer order"),
        (sentinel.get("bands"), SENTINEL2_BANDS, "Sentinel-2 band order"),
        (sentinel.get("passthrough"), True, "input passthrough"),
        (sentinel.get("dtype"), "FLOAT32", "input dtype"),
        (sentinel.get("load_all_layers"), True, "load_all_layers"),
        (
            _at(data_args, ("task", "class_path")),
            "rslearn.train.tasks.embedding.EmbeddingTask",
            "embedding task",
        ),
        (data_args.get("batch_size"), batch_size, "batch size"),
        (data_args.get("num_workers"), num_workers, "worker count"),
        (
            _at(transforms[0], ("class_path",)),
            "rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize",
            "normalization class",
        ),
        (
            normalize_args.get("band_names"),
            {"sentinel2_l2a": SENTINEL2_BANDS},
            "normalization band order",
        ),
        (prediction.get("load_all_crops"), True, "load_all_crops"),
        (prediction.get("crop_size"), 64, "crop size"),
        (prediction.get("overlap_pixels"), 32, "crop overlap"),
        (
            _at(callbacks[0], ("class_path",)),
            "rslearn.train.prediction_writer.RslearnWriter",
            "writer class",
        ),
        (writer_args.get("path"), "${DATASET_PATH}", "writer dataset path"),
        (writer_args.get("output_layer"), output_layer, "output layer"),
        (
            merger.get("class_path"),
            "rslearn.train.prediction_writer.RasterMerger",
            "merger class",
        ),
        (merger_args.get("overlap_pixels"), 8, "merger overlap"),
        (merger_args.get("downsample_factor"), 4, "merger downsample factor"),
    )
    for actual, expected, label in checks:
        _expect(actual, expected, label)

    semantic_core = {
        "schema": "olmoearth-release-semantic-core-v1",
        "encoder": {
            "class_path": "rslearn.models.olmoearth_pretrain.model.OlmoEarth",
            "patch_size": 4,
            "use_legacy_timestamps": True,
        },
        "input": {
            "modality": "sentinel2_l2a",
            "data_type": "raster",
            "layers": list(SENTINEL2_LAYERS),
            "bands": list(SENTINEL2_BANDS),
            "dtype": "FLOAT32",
            "load_all_layers": True,
        },
        "normalization": {
            "class_path": "rslearn.models.olmoearth_pretrain.norm.OlmoEarthNormalize",
            "band_names": {"sentinel2_l2a": list(SENTINEL2_BANDS)},
        },
        "prediction": {
            "load_all_crops": True,
            "crop_size": 64,
            "overlap_pixels": 32,
        },
        "decoder": "rslearn.train.tasks.embedding.EmbeddingHead",
        "merger": {
            "class_path": "rslearn.train.prediction_writer.RasterMerger",
            "overlap_pixels": 8,
            "downsample_factor": 4,
        },
    }
    return {
        "schema": "olmoearth-release-resolved-config-contract-v1",
        "semantic_core": semantic_core,
        "execution_binding": {
            "model_path_environment": model_env,
            "dataset_path_environment": "DATASET_PATH",
            "output_layer": output_layer,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
    }


def validate_runtime_versions(value: Any) -> dict[str, Any]:
    runtime = _mapping(value, "runtime_versions")
    required = ("python", "rslearn", "torch", "lightning", "torch_cuda", "cudnn")
    missing = [key for key in required if key not in runtime or runtime[key] is None]
    if missing:
        raise ValueError(f"runtime version evidence is incomplete: {missing}")
    normalized = {key: copy.deepcopy(runtime[key]) for key in required}
    if not all(isinstance(normalized[key], str) and normalized[key] for key in required[:5]):
        raise ValueError("runtime string versions must be non-empty")
    if not isinstance(normalized["cudnn"], (str, int)):
        raise ValueError("cuDNN version must be a string or integer")
    return normalized


def validate_launcher_runtime_binding(
    runtime_versions: Any,
    launcher_executable: Path,
    rslearn_runtime_fingerprint: Any,
) -> dict[str, Any]:
    """Require the audit runner and rslearn console script to share one venv."""

    runtime = validate_runtime_versions(runtime_versions)
    fingerprint = validate_rslearn_runtime_fingerprint(rslearn_runtime_fingerprint)
    launcher = launcher_executable.resolve(strict=True)
    interpreter = Path(fingerprint["interpreter"]["path"])
    if launcher != interpreter:
        raise ValueError(
            f"audit launcher interpreter differs from rslearn shebang interpreter: "
            f"{launcher} != {interpreter}"
        )
    if runtime["python"] != fingerprint["interpreter"]["version"]:
        raise ValueError("audit launcher Python version differs from rslearn interpreter")
    if runtime["rslearn"] != fingerprint["rslearn_package"]["version"]:
        raise ValueError("audit launcher rslearn version differs from executed rslearn package")
    return runtime


def validate_physical_gpu(index: Any, uuid: Any) -> dict[str, str]:
    if str(index) != "0":
        raise ValueError(f"release audit must use physical GPU0, found {index!r}")
    if not isinstance(uuid, str) or _GPU_UUID.fullmatch(uuid) is None:
        raise ValueError(f"invalid physical GPU UUID: {uuid!r}")
    return {"index": "0", "uuid": uuid}


def normalize_checkpoint_model(value: Any) -> dict[str, Any]:
    model = _mapping(value, "checkpoint model")
    repo_id = model.get("repo_id")
    if repo_id not in RELEASE_SPECS:
        raise ValueError(f"unexpected checkpoint repository: {repo_id!r}")
    revision = model.get("revision")
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError(f"invalid immutable checkpoint revision: {revision!r}")
    files = _list(model.get("files"), "checkpoint files")
    normalized_files = []
    names = set()
    for file_value in files:
        item = _mapping(file_value, "checkpoint file")
        name, size, sha256 = item.get("name"), item.get("bytes"), item.get("sha256")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"invalid or duplicate checkpoint file name: {name!r}")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"invalid checkpoint file size: {name!r}/{size!r}")
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise ValueError(f"invalid checkpoint file SHA-256: {name!r}")
        names.add(name)
        normalized_files.append({"name": name, "bytes": size, "sha256": sha256})
    if not {"config.json", "weights.pth"}.issubset(names):
        raise ValueError(f"checkpoint lacks config.json or weights.pth: {repo_id}")
    snapshot_path = model.get("snapshot_path")
    if not isinstance(snapshot_path, str) or f"/snapshots/{revision}" not in snapshot_path:
        raise ValueError(f"checkpoint snapshot path is not bound to revision: {repo_id}")
    spec = RELEASE_SPECS[repo_id]
    return {
        "release_id": spec["release_id"],
        "repo_id": repo_id,
        "revision": revision,
        "model_path_environment": spec["model_env"],
        "files": sorted(normalized_files, key=lambda item: item["name"]),
    }


def normalize_checkpoint_manifest(value: Any) -> dict[str, dict[str, Any]]:
    payload = _mapping(value, "checkpoint manifest")
    if payload.get("schema") != "olmoearth-checkpoint-resolution-v1":
        raise ValueError("unrecognized checkpoint manifest schema")
    models = [
        normalize_checkpoint_model(model)
        for model in _list(payload.get("models"), "models")
    ]
    by_repo = {model["repo_id"]: model for model in models}
    if len(by_repo) != len(models) or set(by_repo) != set(RELEASE_SPECS):
        raise ValueError("checkpoint manifest must contain exactly the two audit releases")
    return {repo_id: by_repo[repo_id] for repo_id in sorted(by_repo)}


def validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"invalid {label} SHA-256: {value!r}")
    return value


def validate_promoted_execution_contract(
    value: Any,
    *,
    repo_id: str,
    resolved_config_contract: dict[str, Any],
    batch_size: int,
    num_workers: int,
    runtime_versions: Any,
    gpu_index: Any,
    gpu_uuid: Any,
    checkpoint_manifest_sha256: str,
    checkpoint_models: dict[str, dict[str, Any]],
    rslearn_runtime_fingerprint: Any,
) -> dict[str, Any]:
    """Strictly compare a current full run with the promoted batch evidence."""

    contract = _mapping(value, "promoted execution contract")
    if contract.get("schema") != "olmoearth-release-execution-contract-v1":
        raise ValueError("unrecognized promoted execution contract schema")
    if repo_id not in RELEASE_SPECS:
        raise ValueError(f"unexpected full-run repository: {repo_id!r}")
    spec = RELEASE_SPECS[repo_id]
    tuning = _mapping(contract.get("selected_tuning"), "selected tuning")
    if tuning.get("batch_size") != batch_size or tuning.get("num_workers") != num_workers:
        raise ValueError("full-run tuning differs from promoted execution tuning")

    if resolved_config_contract.get("semantic_core") != contract.get(
        "semantic_config_core"
    ):
        raise ValueError("full resolved YAML differs from promoted semantic config core")
    binding = _mapping(
        resolved_config_contract.get("execution_binding"), "full config execution binding"
    )
    expected_binding = {
        "model_path_environment": spec["model_env"],
        "dataset_path_environment": "DATASET_PATH",
        "output_layer": spec["full_output_layer"],
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    if binding != expected_binding:
        raise ValueError("full resolved YAML has an unexpected release/output binding")

    current_runtime = validate_runtime_versions(runtime_versions)
    if current_runtime != contract.get("runtime_versions"):
        raise ValueError("full-run runtime versions differ from batch promotion")
    current_gpu = validate_physical_gpu(gpu_index, gpu_uuid)
    if current_gpu != contract.get("physical_gpu"):
        raise ValueError("full run is not using the promoted physical GPU")
    current_rslearn = validate_rslearn_runtime_fingerprint(rslearn_runtime_fingerprint)
    if current_rslearn != contract.get("rslearn_runtime_fingerprint"):
        raise ValueError("full-run rslearn executable/runtime/source differs from promotion")
    validate_sha256(contract.get("exact_smoke_inputs_sha256"), "exact smoke inputs")
    if validate_sha256(
        checkpoint_manifest_sha256, "current checkpoint manifest"
    ) != contract.get("checkpoint_manifest_sha256"):
        raise ValueError("full-run checkpoint manifest differs from batch promotion")
    promoted_releases = _mapping(contract.get("releases"), "promoted releases")
    if checkpoint_models != promoted_releases:
        raise ValueError("full-run release revisions/checkpoint file hashes differ from promotion")
    if checkpoint_models[repo_id]["model_path_environment"] != spec["model_env"]:
        raise ValueError("full-run model environment/repository binding drift")
    validation = _mapping(contract.get("validation"), "promotion validation")
    if not validation or not all(value is True for value in validation.values()):
        raise ValueError("promoted execution validation assertions are incomplete")
    return {
        "schema": "olmoearth-release-full-execution-contract-check-v1",
        "status": "matched",
        "repo_id": repo_id,
        "revision": checkpoint_models[repo_id]["revision"],
        "physical_gpu": current_gpu,
        "runtime_versions": current_runtime,
        "rslearn_runtime_fingerprint": current_rslearn,
        "semantic_config_core": resolved_config_contract["semantic_core"],
        "execution_binding": binding,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the fail-closed rslearn execution fingerprint as JSON."
    )
    parser.add_argument("--rslearn", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            fingerprint_rslearn_runtime(args.rslearn),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
