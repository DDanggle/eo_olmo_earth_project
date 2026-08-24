#!/usr/bin/env python3
"""Frozen-head release swap — 배포된 LFMC 모델의 인코더를 v1→v1.2로 올리면 무슨 일이 일어나는가.

동기 (MEASURED_FINDINGS M1): 같은 입력에서 릴리스만 바꾸면 검색 identity가 양방향 0으로 깨진다.
그러나 `R@1=0`은 **좌표 호환성 실패**이지 task 실패가 아니다. 이 스크립트가 그 간극을 메운다.

설계 — 인코더만 교체하고 head는 고정한다:
  A0 control     v1   encoder + 우리 학습 head  → test MSE 558.8을 재현해야 한다 (타당성 게이트)
  A1 no-check    v1.2 encoder + 같은 head(고정) → 검사 없이 재사용했을 때의 실제 손해
  (후속) A2 upper bound: v1.2에서 head 재학습 / A3 bridge: v1.2→v1 선형맵 후 투입

구현: Lightning strict 로딩을 우회하지 않는다. v1.2 인코더로 모델을 만들고 그 state_dict의
`model.decoder.*`만 우리 ep33 체크포인트 값으로 덮어써 **키가 정확히 일치하는 병합 체크포인트**를
만든 뒤, 평소와 같은 `rslearn model test`로 평가한다. 그래야 두 arm의 지표 계산 경로가 동일하다.

타당성 게이트: 같은 절차를 v1로 수행한 병합본이 원본 ep33과 수치적으로 일치해야 한다
(일치하지 않으면 병합 자체가 결과를 오염시킨 것이므로 중단).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

OLMO = "/home/work/data/olmoearth"
LFMC_CFG = f"{OLMO}/olmoearth_projects/olmoearth_run_data/lfmc/model.yaml"
EP33 = f"{OLMO}/scratch/lfmc/trainer_checkpoints/epoch=33-step=22270.ckpt"
DATASET = f"{OLMO}/scratch/lfmc/dataset"
OUT = f"{OLMO}/frozen_head_swap"
VENV = f"{OLMO}/.venv/bin/python"
HF_HUB = "/home/work/data/.cache/huggingface/hub"
REPOS = {"v1": "models--allenai--OlmoEarth-v1-Base", "v1_2": "models--allenai--OlmoEarth-v1_2-Base"}
DECODER_PREFIX = "model.decoder."
ENCODER_PREFIX = "model.encoder."


def snapshot_dir(release: str) -> str:
    """HF 캐시의 불변 snapshot 디렉터리. revision을 고정해 재현성을 유지한다."""
    hits = sorted(glob.glob(f"{HF_HUB}/{REPOS[release]}/snapshots/*"))
    if len(hits) != 1:
        raise SystemExit(f"{release}: expected exactly one snapshot, found {len(hits)}: {hits}")
    for required in ("config.json", "weights.pth"):
        if not Path(hits[0], required).exists():
            raise SystemExit(f"{release}: missing {required} in {hits[0]}")
    return hits[0]


def write_variant_config(release: str, path: Path) -> None:
    """lfmc model.yaml에서 encoder만 해당 릴리스로 바꾼 변형을 쓴다. 나머지는 한 줄도 건드리지 않는다."""
    text = Path(LFMC_CFG).read_text(encoding="utf-8")
    original = "              model_id: OLMOEARTH_V1_BASE\n"
    if original not in text:
        raise SystemExit(f"encoder anchor not found in {LFMC_CFG}")
    text = text.replace(original, f"              model_path: {snapshot_dir(release)}\n", 1)
    path.write_text(text, encoding="utf-8")


def build_state_dict(release: str, cfg: Path) -> dict[str, torch.Tensor]:
    """변형 config로 모델을 만들고 그 state_dict를 얻는다 (v1.2 인코더 가중치가 이미 로드된 상태)."""
    script = f"""
import json, sys, torch, yaml
from jsonargparse import ArgumentParser
from rslearn.train.lightning_module import RslearnLightningModule
cfg = yaml.safe_load(open({str(cfg)!r}))
# RslearnLightningModule은 task를 요구한다. task는 yaml의 data 절에 선언돼 있으므로 주입한다.
model_cfg = cfg["model"]
model_cfg.setdefault("init_args", {{}})["task"] = cfg["data"]["init_args"]["task"]
parser = ArgumentParser()
parser.add_subclass_arguments(RslearnLightningModule, "model")
init = parser.instantiate_classes(parser.parse_object({{"model": model_cfg}}))
sd = init.model.state_dict()
torch.save(sd, {str(cfg.with_suffix(".sd.pt"))!r})
print(json.dumps({{"n_keys": len(sd)}}))
"""
    env = {**os.environ, "PYTHONPATH": "", "HF_HOME": "/home/work/data/.cache/huggingface"}
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([VENV, "-c", script], capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"state_dict build failed for {release}:\n{proc.stderr[-3000:]}")
    return torch.load(cfg.with_suffix(".sd.pt"), map_location="cpu")


def merge(release: str, cfg: Path, out_ckpt: Path) -> dict:
    """v1.2 인코더 + 우리 head → 키가 정확히 맞는 체크포인트."""
    fresh = build_state_dict(release, cfg)
    trained = torch.load(EP33, map_location="cpu")
    trained_sd = trained["state_dict"]

    merged, taken, kept, missing = {}, 0, 0, []
    for key, value in fresh.items():
        if key.startswith(DECODER_PREFIX):
            if key not in trained_sd:
                missing.append(key)
                merged[key] = value
                continue
            if trained_sd[key].shape != value.shape:
                raise SystemExit(f"decoder shape mismatch on {key}: {trained_sd[key].shape} vs {value.shape}")
            merged[key] = trained_sd[key]
            taken += 1
        else:
            merged[key] = value
            kept += 1

    if missing:
        raise SystemExit(f"{len(missing)} decoder keys absent from ep33 (first: {missing[:3]})")

    ckpt = {k: v for k, v in trained.items() if k != "state_dict"}
    ckpt["state_dict"] = merged
    # optimizer 상태는 릴리스가 바뀌면 의미가 없다. test에는 불필요하므로 제거한다.
    for drop in ("optimizer_states", "lr_schedulers", "loops"):
        ckpt.pop(drop, None)
    torch.save(ckpt, out_ckpt)
    return {"decoder_keys_from_ep33": taken, "encoder_keys_from_release": kept, "total": len(merged)}


def run_test(cfg: Path, ckpt: Path, split: str, tag: str) -> dict:
    """평소와 같은 rslearn model test. 두 arm의 지표 경로를 동일하게 유지한다."""
    log = Path(OUT, f"test_{tag}_{split}.log")
    env = {
        **os.environ,
        "DATASET_PATH": DATASET,
        "NUM_WORKERS": "8",
        "WANDB_MODE": "offline",
        "WANDB_PROJECT": "lfmc",
        "WANDB_NAME": f"swap-{tag}-{split}",
        "WANDB_ENTITY": "local",
        "TRAINER_DATA_PATH": f"{OUT}/trainer_{tag}",
        "PREDICTION_OUTPUT_LAYER": "output",
        "EXTRA_FILES_PATH": "/tmp",
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/home/work/data/.cache/huggingface",
    }
    env.pop("PYTHONPATH", None)
    cmd = [
        VENV, "-m", "rslearn.main", "model", "test",
        "--config", str(cfg), "--ckpt_path", str(ckpt),
        f"--data.init_args.test_config.tags.split={split}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log.write_text(proc.stdout + "\n===STDERR===\n" + proc.stderr, encoding="utf-8")
    metrics = {}
    for line in (proc.stdout + proc.stderr).replace("\r", "\n").splitlines():
        for name in ("test_mse", "test_loss", "test_regress"):
            if name in line:
                for token in line.replace("│", " ").split():
                    try:
                        metrics[name] = float(token)
                    except ValueError:
                        continue
    if "test_mse" not in metrics:
        raise SystemExit(f"{tag}/{split}: no test_mse in output, see {log}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="test,val")
    ap.add_argument("--arms", default="v1,v1_2")
    args = ap.parse_args()
    Path(OUT).mkdir(parents=True, exist_ok=True)

    result = {
        "schema": "frozen-head-release-swap-v1",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "head_checkpoint": EP33,
        "reference": {"ep33_test_mse_v1_native": 558.7871704101562},
        "snapshots": {r: snapshot_dir(r) for r in REPOS},
        "arms": {},
    }

    for arm in args.arms.split(","):
        cfg = Path(OUT, f"model_{arm}.yaml")
        ckpt = Path(OUT, f"head_ep33_encoder_{arm}.ckpt")
        write_variant_config(arm, cfg)
        merge_info = merge(arm, cfg, ckpt)
        print(f"[{arm}] merged: {merge_info}", flush=True)
        arm_result = {"merge": merge_info, "config": str(cfg), "checkpoint": str(ckpt), "metrics": {}}
        for split in args.splits.split(","):
            m = run_test(cfg, ckpt, split, arm)
            arm_result["metrics"][split] = m
            print(f"[{arm}/{split}] {m}", flush=True)
        result["arms"][arm] = arm_result

    # 타당성 게이트: v1 병합본이 원본 ep33 수치를 재현해야 한다
    v1_test = result["arms"].get("v1", {}).get("metrics", {}).get("test", {}).get("test_mse")
    if v1_test is not None:
        delta = abs(v1_test - result["reference"]["ep33_test_mse_v1_native"])
        result["validity"] = {"v1_merge_reproduces_ep33": delta < 0.5, "abs_delta": delta}
        print(f"[validity] v1 merged test_mse={v1_test} (ref 558.787, |Δ|={delta:.4f})", flush=True)

    if {"v1", "v1_2"} <= set(result["arms"]):
        for split in args.splits.split(","):
            a = result["arms"]["v1"]["metrics"].get(split, {}).get("test_mse")
            b = result["arms"]["v1_2"]["metrics"].get(split, {}).get("test_mse")
            if a and b:
                result.setdefault("headline", {})[split] = {
                    "v1_encoder_mse": a, "v1_2_encoder_mse": b,
                    "ratio": round(b / a, 4), "delta": round(b - a, 3),
                }
                print(f"[headline/{split}] v1 {a:.2f} → v1.2 {b:.2f} (×{b/a:.2f})", flush=True)

    Path(OUT, "frozen_head_swap.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
