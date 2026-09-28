#!/usr/bin/env python3
"""One-shot bounded EO/Qwen interface launcher. Authoring/importing does not launch.

Run only after root has frozen local source and inputs. The source manifest is
ROOT/code_snapshot/vlm_v3/manifest.json with {"files": {relative_name: sha256}}
or per-file {"sha256": sha256} records. Both worker sources and this launcher
must appear. No downloads, retries, daemon, or external process termination.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import time
import traceback

ROOT = Path("/home/work/data/olmoearth/oe4_native_v12_v0")
PYTHON = ROOT.parent / ".venv-master/bin/python"
SNAPSHOT = ROOT / "code_snapshot/vlm_v3"
SOURCE_MANIFEST = SNAPSHOT / "manifest.json"
INPUTS = ROOT / "vlm_inputs_v0"
OUTPUT = ROOT / "runs/vlm_interface_gpu1_v3"
NATIVE_STATUS = ROOT / "bounded_sequence_gpu1_v3_status.json"
STATUS = ROOT / "vlm_sequence_gpu1_v3_status.json"
LOG = ROOT / "vlm_interface_gpu1_v3.log"
LOCK = ROOT / "vlm_sequence_gpu1_v3.lock"
MODEL = ROOT.parent / "models/Qwen3-VL-8B-Instruct"
MAX_NATIVE_WAIT = 45 * 60
MAX_GPU_WAIT = 10 * 60
MAX_WORKER_SECONDS = 1100
POLL_SECONDS = 10
PROTECTED = {
    "pilot_sen12_gp_heads.py": ("8f805359c2989cd474e89d9086545698c0acdbb9673b27a82cc1ad00743a1af1", 1788238578),
    "sen12_official_baselines.py": ("19232b2ae122dd6529bac56977e075ce7b94c0daf90f24c21557b4d5cc213f5b", 1787683824),
    "extract_sen12_fold_cache.py": ("197895a05f2f6dddff12f6385c01aacf4f72bdbb877d273b473c83dc5d3a5e0a", 1787659265),
    "audit_sen12_fold_cache.py": ("4287dbcdef64c9ea9ea2a707e961ce6ba606b85405ee794fa91f44ded2419ec9", 1787668636),
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Require a regular nonsymlink file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError(f"File changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "mtime_ns": after.st_mtime_ns, "bytes": after.st_size}


def protected_records():
    records = {}
    for name, (expected_sha, expected_mtime_seconds) in PROTECTED.items():
        record = file_record(ROOT.parent / "code" / name)
        records[name] = record
        if record["sha256"] != expected_sha or record["mtime_ns"] // 1_000_000_000 != expected_mtime_seconds:
            raise RuntimeError(f"Protected source identity mismatch: {name}")
    return records


def verified_source_records():
    manifest_record = file_record(SOURCE_MANIFEST)
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Source manifest must have a nonempty files mapping")
    required = {"vlm_interface_smoke.py", "real_h5_provider.py", "run_vlm_bounded.py"}
    if not required.issubset(files):
        raise RuntimeError(f"Missing mandatory source hashes: {sorted(required - set(files))}")
    records = {"manifest.json": manifest_record}
    for name, value in files.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or str(rel) != name or name == "manifest.json":
            raise RuntimeError(f"Invalid snapshot-relative manifest path: {name}")
        path = SNAPSHOT / name
        if not path.resolve().is_relative_to(SNAPSHOT.resolve()):
            raise RuntimeError(f"Source escapes snapshot: {name}")
        expected = value.get("sha256") if isinstance(value, dict) else value
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"Invalid SHA256 in source manifest: {name}")
        record = file_record(path)
        if record["sha256"] != expected:
            raise RuntimeError(f"Source SHA256 mismatch: {name}")
        records[name] = record
    unlisted = {str(p.relative_to(SNAPSHOT)) for p in SNAPSHOT.rglob("*.py")
                if "__pycache__" not in p.parts} - set(files)
    if unlisted:
        raise RuntimeError(f"Unhashed Python source in snapshot: {sorted(unlisted)}")
    if Path(__file__).resolve() != (SNAPSHOT / "run_vlm_bounded.py").resolve():
        raise RuntimeError("Run the frozen launcher under code_snapshot/vlm_v3")
    if file_record(SOURCE_MANIFEST) != manifest_record:
        raise RuntimeError("Source manifest changed during verification")
    return records


def input_records():
    required = {"rgb.png", "provider_config.json", "provenance.json"}
    files = {p.relative_to(INPUTS).as_posix(): p for p in INPUTS.rglob("*") if p.is_file()}
    if not required.issubset(files):
        raise RuntimeError(f"Frozen VLM inputs missing: {sorted(required - set(files))}")
    records = {str(path): file_record(path) for path in files.values()}
    config = json.loads((INPUTS / "provider_config.json").read_text())
    manifest = Path(config["manifest"])
    if not manifest.is_absolute() or not manifest.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("Provider data manifest must be an absolute local path under ROOT")
    records[str(manifest)] = file_record(manifest)
    return records


def _parse_idle_gpu_rows_strict(gpu_text, process_text):
    """Fail closed on malformed/unsupported nvidia-smi values; allow physical GPU 1 only."""
    busy = set()
    for row in csv.reader(process_text.splitlines()):
        if not row or not any(x.strip() for x in row):
            continue
        if len(row) != 2 or not row[0].strip().startswith("GPU-") or not row[1].strip().isdigit():
            raise RuntimeError(f"Unparseable compute-process row: {row}")
        busy.add(row[0].strip())
    observations = {}
    for row in csv.reader(gpu_text.splitlines()):
        if not row or not any(x.strip() for x in row):
            continue
        if len(row) != 4:
            raise RuntimeError(f"Unparseable GPU row: {row}")
        index, uuid, memory, utilization = (v.strip() for v in row)
        utilization_unavailable = utilization in {"[Not Found]", "N/A"}
        if (not all(x.isdigit() for x in [index, memory]) or not uuid.startswith("GPU-")
                or not (utilization.isdigit() or utilization_unavailable)):
            raise RuntimeError(f"Unsupported GPU query values: {row}")
        index = int(index)
        if index in observations:
            raise RuntimeError("Duplicate GPU index in nvidia-smi response")
        observations[index] = {"index": index, "uuid": uuid, "memory_mib": int(memory),
                               "utilization_percent": None if utilization_unavailable else int(utilization),
                               "utilization_unavailable": utilization_unavailable,
                               "has_compute_pid": uuid in busy}
    if not observations:
        raise RuntimeError("nvidia-smi returned no GPUs")
    idle = []
    for index in [1]:
        if index not in observations:
            continue
        observation = observations[index]
        if observation["has_compute_pid"]:
            continue
        if observation["utilization_unavailable"]:
            # Exact missing-value sentinels are eligible only at zero memory.
            if observation["memory_mib"] != 0:
                continue
            observation["idle_basis"] = "memory_zero_and_no_compute_PID"
        else:
            if observation["memory_mib"] >= 512 or observation["utilization_percent"] >= 5:
                continue
            observation["idle_basis"] = "memory_under_512_and_utilization_under_5_and_no_compute_PID"
        idle.append(observation)
    return idle, [observations[i] for i in sorted(observations)]


def parse_idle_gpu_rows(gpu_text, process_text):
    try:
        return _parse_idle_gpu_rows_strict(gpu_text, process_text)
    except (RuntimeError, ValueError, csv.Error) as e:
        return [], [{"query_error": repr(e), "treated_as": "BUSY; retry within existing wait budget"}]


def idle_gpu_snapshot():
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True, timeout=15, stderr=subprocess.PIPE)
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                             "--format=csv,noheader,nounits"], text=True, timeout=15, stderr=subprocess.PIPE)
        return parse_idle_gpu_rows(gpu, processes)
    except (subprocess.SubprocessError, OSError) as e:
        return [], [{"query_error": repr(e), "treated_as": "BUSY; retry within existing wait budget"}]


def same_records(current, expected, label):
    if current != expected:
        raise RuntimeError(f"{label} hashes/mtime/size changed since launcher start")


def sleep_bounded(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return
    time.sleep(min(POLL_SECONDS, remaining))


def main():
    if not ROOT.is_dir():
        raise RuntimeError("Existing ROOT is required; this launcher does not acquire assets")
    lock = LOCK.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Exclusive claims prevent accidental reuse or overwriting a previous run.
    with STATUS.open("x") as f:
        json.dump({"status": "initializing", "controller_pid": os.getpid(), "started_utc": utc()}, f)
        f.write("\n")
    state = {"status": "initializing", "started_utc": utc(), "controller_pid": os.getpid(),
             "scope": "One-shot engineering interface validation, not a semantic or novelty result",
             "maximum_native_wait_seconds": MAX_NATIVE_WAIT, "maximum_gpu_wait_seconds": MAX_GPU_WAIT,
             "maximum_worker_seconds": MAX_WORKER_SECONDS, "output_directory": str(OUTPUT),
             "protected_before": None, "source_before": None, "inputs_before": None}
    worker = None
    log = None

    def write_state():
        tmp = STATUS.with_name(f"{STATUS.name}.{os.getpid()}.tmp")
        with tmp.open("x") as f:
            json.dump(state, f, indent=2, allow_nan=False)
            f.write("\n")
        tmp.replace(STATUS)

    def transition(status, detail):
        state.update({"status": status, "detail": detail, "updated_utc": utc()})
        write_state()
        line = json.dumps({"utc": utc(), "status": status, "detail": detail})
        print(line, flush=True)
        if log is not None:
            log.write(line + "\n")
            log.flush()

    def terminate_own_worker():
        # Only the dedicated process group created by THIS launcher is eligible.
        if worker is not None and worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=15)

    def interrupted(signum, frame):
        raise InterruptedError(f"Launcher received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        log = LOG.open("x", buffering=1)
        if OUTPUT.exists():
            raise FileExistsError(OUTPUT)
        state["protected_before"] = protected_records()
        state["source_before"] = verified_source_records()
        state["inputs_before"] = input_records()
        if not PYTHON.is_file() or not MODEL.is_dir():
            raise RuntimeError("Local master Python and cached Qwen checkpoint are required")
        transition("waiting_for_native_sequence", "Waiting at most 45 minutes for the existing native sequence; no other jobs interrupted.")
        native_start = time.monotonic()
        deadline = native_start + MAX_NATIVE_WAIT
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Native sequence did not complete within 45 minutes")
            if NATIVE_STATUS.exists():
                native = json.loads(NATIVE_STATUS.read_text())
                if native.get("status") == "failed":
                    raise RuntimeError(f"Native sequence failed: {native.get('detail')}")
                if native.get("status") == "completed":
                    state["native_completion"] = {"observed_utc": utc(), "status_record": file_record(NATIVE_STATUS)}
                    break
            sleep_bounded(deadline)
        state["native_wait_seconds"] = time.monotonic() - native_start
        argv = [str(PYTHON), "-u", str(SNAPSHOT / "vlm_interface_smoke.py"),
                "--model-dir", str(MODEL), "--image", str(INPUTS / "rgb.png"),
                "--eo-provider", f"{SNAPSHOT / 'real_h5_provider.py'}:make_provider",
                "--eo-provider-config", str(INPUTS / "provider_config.json"),
                "--deps-root", str(ROOT / "deps"), "--out-dir", str(OUTPUT), "--device", "cuda:0"]
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # Equivalent to env -u PYTHONPATH.
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        transition("waiting_for_idle_gpu", "Physical GPU 1 only; three consecutive checks 10 seconds apart, followed by an immediate prelaunch recheck.")
        gpu_start = time.monotonic()
        deadline = gpu_start + MAX_GPU_WAIT
        stable_uuid, streak = None, 0
        stable_records = []
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("No stable idle GPU within 10 minutes; no other jobs interrupted")
            choices, observed = idle_gpu_snapshot()
            current = choices[0] if choices else None
            current_uuid = current["uuid"] if current else None
            if current_uuid is not None and current_uuid == stable_uuid:
                streak += 1
            else:
                streak = 1 if current is not None else 0
                stable_records = []
            stable_uuid = current_uuid
            if current is not None:
                stable_records.append({"observed_utc": utc(), **current})
            state["gpu_last_observations"] = observed
            state["gpu_idle_streak"] = streak
            state["gpu_poll_count"] = state.get("gpu_poll_count", 0) + 1
            write_state()
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU idle verification exceeded 10-minute budget")
            if streak >= 3:
                chosen = current
                state["gpu_wait_seconds"] = time.monotonic() - gpu_start
                state["gpu_stable_observations"] = stable_records
                same_records(protected_records(), state["protected_before"], "Protected sources")
                same_records(verified_source_records(), state["source_before"], "Frozen source")
                same_records(input_records(), state["inputs_before"], "Frozen VLM inputs")
                if OUTPUT.exists():
                    raise FileExistsError(OUTPUT)
                # UUID avoids CUDA/nvidia-smi index-order ambiguity.
                env["CUDA_VISIBLE_DEVICES"] = chosen["uuid"]
                state["worker"] = {"argv": argv, "gpu_physical_index": chosen["index"], "gpu_uuid": chosen["uuid"],
                    "environment_overrides": {"PYTHONPATH": None, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": chosen["uuid"]}}
                transition("prelaunch_recheck", "Source and inputs unchanged; checking GPU again immediately before dispatch.")
                # No asset work or sleeps between a successful recheck and process creation.
                choices, final_observed = idle_gpu_snapshot()
                if time.monotonic() >= deadline:
                    raise TimeoutError("GPU prelaunch verification exceeded 10-minute budget")
                if not any(gpu["uuid"] == chosen["uuid"] for gpu in choices):
                    state["gpu_last_observations"] = final_observed
                    state["prelaunch_busy_retries"] = state.get("prelaunch_busy_retries", 0) + 1
                    state.pop("worker", None)
                    stable_uuid, streak, stable_records = None, 0, []
                    transition("waiting_for_idle_gpu", "Immediate recheck was busy/unknown; restarting stability checks within the same 10-minute budget.")
                    sleep_bounded(deadline)
                    continue
                worker_start = time.monotonic()
                worker = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                break
            sleep_bounded(deadline)
        state["worker"].update({"pid": worker.pid, "started_utc": utc(), "final_idle_observations": final_observed})
        transition("running", "One bounded VLM interface worker; hard timeout 1100 seconds.")
        try:
            returncode = worker.wait(timeout=max(0.0, MAX_WORKER_SECONDS - (time.monotonic() - worker_start)))
        except subprocess.TimeoutExpired:
            terminate_own_worker()
            state["worker"]["timed_out"] = True
            raise TimeoutError("VLM worker exceeded 1100 seconds; only its own process group was stopped")
        state["worker"].update({"returncode": returncode, "finished_utc": utc(), "wall_seconds": time.monotonic() - worker_start})
        if returncode != 0:
            raise RuntimeError(f"VLM interface worker exited with status {returncode}; inspect its log and receipt")
        receipt_path = OUTPUT / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        state["worker"]["receipt"] = file_record(receipt_path)
        state["worker"]["receipt_status"] = receipt.get("status")
        if receipt.get("status") != "PASS_live_encoder_adapter_vlm_gradient_contract":
            raise RuntimeError("Worker did not establish the live-encoder interface contract")
        state["protected_after"] = protected_records()
        state["source_after"] = verified_source_records()
        state["inputs_after"] = input_records()
        same_records(state["protected_after"], state["protected_before"], "Protected sources")
        same_records(state["source_after"], state["source_before"], "Frozen source")
        same_records(state["inputs_after"], state["inputs_before"], "Frozen VLM inputs")
        state["completed_utc"] = utc()
        transition("completed", "Live encoder/adapter/VLM interface contract passed; this is not a semantic research result.")
    except BaseException as e:
        cleanup_error = None
        try:
            terminate_own_worker()
        except BaseException as cleanup:
            cleanup_error = repr(cleanup)
        state["failure"] = {"error": repr(e), "traceback": traceback.format_exc(), "cleanup_error": cleanup_error}
        # Preserve after-state evidence even when the worker or an earlier guard fails.
        for label, check in [("protected", protected_records), ("source", verified_source_records), ("inputs", input_records)]:
            try:
                current = check()
                state[label + "_after"] = current
                before = state.get(label + "_before")
                state[label + "_unchanged"] = current == before if before is not None else None
            except BaseException as after_error:
                state[label + "_after_check_error"] = repr(after_error)
        if worker is not None:
            state.setdefault("worker", {}).update({"returncode": worker.poll(), "finished_utc": utc()})
        state["finished_utc"] = utc()
        transition("failed", repr(e))
        raise
    finally:
        if log is not None:
            log.close()
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
