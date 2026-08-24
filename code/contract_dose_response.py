#!/usr/bin/env python3
"""계약 불일치 dose-response — 밴드 순서 축.

동기: `config/olmo_release_v1_legacy.yaml`은 밴드 순서를 **두 곳**에 선언한다.
  (1) data.inputs.sentinel2_l2a.bands
  (2) predict_config.transforms.OlmoEarthNormalize.band_names.sentinel2_l2a
한 곳만 어긋나면 정규화 통계가 다른 밴드에 붙는다. 파일은 정상이고 실행도 성공하며
차원도 같다. 즉 **조용히 틀린다.**

이 스크립트는 (1)만 k쌍 치환하고 (2)는 그대로 두어 알려진 용량의 불일치를 주입한 뒤,
dose 0 대비 표현이 얼마나 이동하는지 잰다. 원본 raster를 재사용하므로 재다운로드·
재materialize가 없다.

측정 목적은 두 가지다.
  W1  용량-반응 곡선: 불일치가 커지면 표현이 얼마나 이동하는가.
  W2  진단 눈멂: CKA/코사인 같은 값싼 진단이 그 이동을 탐지하는가.

GPU를 점유하므로 선택한 GPU에 다른 프로세스가 있으면 실행을 거부한다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# 12밴드 -> 인접 6쌍. dose k = 앞에서부터 k쌍을 뒤집는다.
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
DOSES = [0, 1, 2, 3, 6]
REVERSE_DOSE = "reverse"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def permute(bands: list[str], dose) -> list[str]:
    out = list(bands)
    if dose == REVERSE_DOSE:
        return out[::-1]
    for i, j in PAIRS[:dose]:
        out[i], out[j] = out[j], out[i]
    return out


def gpu_busy(index: str) -> list[str]:
    uuid = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    target = None
    for line in uuid.strip().splitlines():
        idx, val = [p.strip() for p in line.split(",", 1)]
        if idx == index:
            target = val
    if target is None:
        raise SystemExit(f"REFUSED: gpu index {index} not found")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout
    return [ln for ln in apps.strip().splitlines() if target in ln]


def build_configs(base_cfg: Path, out_dir: Path, layer_prefix: str):
    import yaml

    base = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    inputs = base["data"]["init_args"]["inputs"]["sentinel2_l2a"]
    original = list(inputs["bands"])
    norm = (base["data"]["init_args"]["predict_config"]["transforms"][0]
            ["init_args"]["band_names"]["sentinel2_l2a"])
    if list(norm) != original:
        raise SystemExit("REFUSED: base config already has mismatched band lists")

    out_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for dose in DOSES + [REVERSE_DOSE]:
        cfg = copy.deepcopy(base)
        permuted = permute(original, dose)
        # (1)만 바꾸고 (2) 정규화기 band_names는 그대로 둔다 -> 조용한 불일치
        cfg["data"]["init_args"]["inputs"]["sentinel2_l2a"]["bands"] = permuted
        layer = f"{layer_prefix}{dose}"
        cfg["trainer"]["callbacks"][0]["init_args"]["output_layer"] = layer
        text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        path = out_dir / f"dose_{dose}.yaml"
        path.write_text(text, encoding="utf-8")
        plan.append({
            "dose": dose,
            "swapped_pairs": (len(PAIRS) if dose == REVERSE_DOSE else dose),
            "bands": permuted,
            "normalizer_bands": list(norm),
            "displaced_positions": sum(1 for a, b in zip(permuted, original) if a != b),
            "output_layer": layer,
            "config_path": path.as_posix(),
            "config_sha256": sha256_text(text),
        })
    return original, plan


def register_layers(dataset_root: Path, plan, work_dir: Path) -> dict:
    """출력 레이어를 dataset config.json에 선언한다.

    rslearn은 선언되지 않은 output_layer에 쓰지 않는다. 기존 레이어와 frozen 입력은
    건드리지 않고 dose 레이어만 추가하며, 변경 전 원본을 백업하고 해시를 남긴다.
    """
    cfg_path = dataset_root / "config.json"
    before = cfg_path.read_text(encoding="utf-8")
    backup = work_dir / "dataset_config.before.json"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(before, encoding="utf-8")

    cfg = json.loads(before)
    layers = cfg.setdefault("layers", {})
    template = layers.get("embeddings_audit_v1_legacy")
    if template is None:
        raise SystemExit("REFUSED: reference layer embeddings_audit_v1_legacy missing")

    added = []
    for entry in plan:
        name = entry["output_layer"]
        if name in layers:
            continue
        layers[name] = copy.deepcopy(template)
        added.append(name)
    after = json.dumps(cfg, ensure_ascii=False, indent=2) + "\n"
    if added:
        cfg_path.write_text(after, encoding="utf-8")
    return {
        "config_path": cfg_path.as_posix(),
        "backup_path": backup.as_posix(),
        "sha256_before": sha256_text(before),
        "sha256_after": sha256_text(after),
        "layers_added": added,
        "template_layer": "embeddings_audit_v1_legacy",
        "note": "기존 레이어와 frozen 입력은 수정하지 않았다. dose 레이어만 추가했다.",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--rslearn", type=Path, required=True)
    ap.add_argument("--model-path", required=True, help="model snapshot path")
    ap.add_argument("--model-env", default="OLMO_V1_MODEL_PATH",
                    help="env var the config interpolates for the checkpoint")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--layer-prefix", default="embeddings_dose_")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    original, plan = build_configs(args.base_config, args.work_dir / "configs",
                                   args.layer_prefix)
    busy = gpu_busy(args.gpu)
    preflight = {
        "schema": "contract-dose-response-v1",
        "axis": "band_order_mismatch_between_input_and_normalizer",
        "original_bands": original,
        "doses": plan,
        "gpu": args.gpu,
        "model_env": args.model_env,
        "model_path": args.model_path,
        "base_config": args.base_config.as_posix(),
        "gpu_processes": busy,
        "dataset_root": args.dataset_root.as_posix(),
        "ready": not busy,
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in preflight.items() if k != "doses"},
                     ensure_ascii=False, indent=2), flush=True)
    for entry in plan:
        print(f"  dose={entry['dose']:>7}  displaced={entry['displaced_positions']:>2}"
              f"  layer={entry['output_layer']}", flush=True)

    if not args.execute:
        return 0
    if busy:
        raise SystemExit("REFUSED: selected GPU has active processes")

    registration = register_layers(args.dataset_root, plan, args.work_dir)
    (args.work_dir / "layer_registration.json").write_text(
        json.dumps(registration, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[layers] added={registration['layers_added']}", flush=True)

    runs = []
    for entry in plan:
        env = os.environ.copy()
        env.update({
            "DATASET_PATH": args.dataset_root.as_posix(),
            args.model_env: args.model_path,
            "CUDA_VISIBLE_DEVICES": args.gpu,
        })
        log = args.work_dir / f"dose_{entry['dose']}.log"
        start = time.monotonic()
        with log.open("w", encoding="utf-8") as fh:
            done = subprocess.run(
                [args.rslearn.as_posix(), "model", "predict",
                 "--config", entry["config_path"]],
                env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"dose {entry['dose']} failed; see {log}")
        runs.append({**entry, "wall_seconds": round(time.monotonic() - start, 3),
                     "finished_at": datetime.now(timezone.utc).isoformat()})
        print(f"[run] dose={entry['dose']} ok {runs[-1]['wall_seconds']}s", flush=True)

    (args.work_dir / "RUNS_COMPLETE.json").write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[done] runs complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
