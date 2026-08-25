#!/usr/bin/env python3
"""G-P pilot v2 — 한 개발 LOCO fold에서 P1/P2/P4를 감사 가능한 방식으로 비교한다. GPU1 전용.

목적: frozen OlmoEarth v1이 산사태 segmentation에서 task-specific 모델과 경쟁하는지 판정한다.
이 pilot에서 P4가 경쟁력 있거나 정확도 대비 비용 우위가 확인되면 10-fold·다중 seed로 확장한다.
밀리면 router 이전에 representation adaptation이 필요하다는 뜻이다.

개발 프로토콜 v2 (M23 test를 이미 보았으므로 confirmatory 사전등록이 아니다)
  입력    S12q. 세 arm이 **같은 12 timestep index**를 쓴다 (캐시가 이미 그것만 담고 있다)
  정규화  raw = uint16 / 10000 (반사도 [0,1] 가정). embedding = train split의 채널별 mean/std
  arm     P1 raw 시간평균 → shallow U-Net (2 down / 2 up)
          P2 raw 12시점 → P2-tiny factorized-pool 3D encoder (공식 3D U-Net 아님)
          P4 frozen emb 768x32x32 → 1x1 conv → 32→64→128 upsample decoder
  손실    BCEWithLogits, pos_weight = train의 (neg/pos), 50으로 상한
  최적화  AdamW lr 1e-3, wd 1e-4, cosine, batch 16, seed 1

프로토콜 계보 (2026-08-25, 사유 공개)
  1차 8 epoch 실행에서 **세 arm 모두 손실이 단조 하강 중**이었다 (P1 0.943→0.574,
  P2 0.961→0.389, P4 0.406→0.186). 즉 전부 미수렴이며, 시작 손실이 낮은 P4에 유리한
  비교였다. 따라서 예산을 **모든 arm에 동일하게** 늘리고 val IoU로 best epoch를 고른다.
  그러나 8-epoch test를 이미 본 뒤의 변경이므로 이 fold는 개발용이다. 최종 test는 나머지
  미열람 fold에서 사전등록 후 한 번만 연다.

  지표    IoU@0.5 · F1@0.5 · exact pixel AUPRC · pixel-micro ECE(15 bin) · Brier/NLL.
          positive-patch macro IoU와 표본별 TP/FP/FN도 함께 보존한다.
  기록    학습가능 파라미터 수 · head fit+val 시간 · peak GPU · 입력 캐시 바이트 · epoch별 이력
          best checkpoint SHA-256 · 표본별 평가 JSONL · cache audit seal.

주의: test는 fold의 held-out 지역 하나뿐이다. **region-macro는 10-fold 전체에서만 나온다.**
이 pilot의 수치를 region 일반화로 읽지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

EPOCHS, BATCH, SEED = 40, 16, 1
LR, WD = 1e-3, 1e-4
POS_WEIGHT_CAP = 50.0
BINS = 15


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_average_precision(scores, labels) -> float | None:
    """Binary AP with tied scores grouped, equivalent to step-wise PR integration."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError("scores/labels shape mismatch")
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    boundary = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tp = np.cumsum(sorted_labels, dtype=np.int64)[boundary]
    fp = (boundary + 1) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fixed_bin_ece(scores, labels, bins: int = BINS) -> float:
    """Equal-width pixel-micro ECE. Background dominance is reported as a caveat."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape or not len(scores):
        raise ValueError("non-empty equal-length scores/labels required")
    which = np.minimum((np.clip(scores, 0.0, 1.0) * bins).astype(np.int64), bins - 1)
    counts = np.bincount(which, minlength=bins).astype(np.float64)
    conf = np.bincount(which, weights=scores, minlength=bins)
    hits = np.bincount(which, weights=labels, minlength=bins)
    keep = counts > 0
    return float(np.sum((counts[keep] / len(scores))
                        * np.abs(conf[keep] / counts[keep] - hits[keep] / counts[keep])))


def seed_everything(torch, np, seed: int) -> None:
    """Arm 실행 순서와 무관한 초기화·shuffle을 만든다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani"))
    p.add_argument("--folds", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/loco_folds.json"))
    p.add_argument("--contract", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl"))
    p.add_argument("--fold", type=str, default="holdout_chimanimani")
    p.add_argument("--out", type=Path,
                   default=Path("/home/work/data/olmoearth/sen12_gp_pilot_v2"))
    p.add_argument("--cache-audit", type=Path, default=None,
                   help="기본값은 <cache>/cache_audit.json; all_gates_pass seal 필수")
    p.add_argument("--arms", type=str, default="P1,P2,P3,P4")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    return p.parse_args()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("refusing to run: CUDA_VISIBLE_DEVICES must be exactly 1")

    # Seed reset alone is not a reproducibility contract. cuBLAS/cuDNN and some
    # CUDA kernels may otherwise choose nondeterministic implementations. These
    # variables must be fixed before importing torch / initializing CUDA.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"

    import sys as _sys
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sen12_official_baselines import (OfficialUNet3D, OfficialUTAE, param_count)

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    audit_path = args.cache_audit or (args.cache / "cache_audit.json")
    if not audit_path.is_file():
        raise SystemExit(f"cache audit seal 없음: {audit_path}\n"
                         "audit_sen12_fold_cache.py를 먼저 통과시켜야 한다.")
    cache_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if cache_audit.get("all_gates_pass") is not True:
        raise SystemExit(f"cache audit gate 실패: {audit_path}")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    seed_everything(torch, np, SEED)
    device = torch.device("cuda")
    torch.cuda.set_device(0)   # CUDA_VISIBLE_DEVICES=1 이므로 physical GPU1

    folds = json.loads(args.folds.read_text(encoding="utf-8"))
    fold = next(f for f in folds["folds"] if f["fold"] == args.fold)
    records = {}
    for line in args.contract.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            records[r["sample_id"]] = r

    def members(split):
        regions = (fold["train_regions"] if split == "train"
                   else [fold["val_region"]] if split == "val" else [fold["test_region"]])
        return sorted(sid for sid, r in records.items()
                      if r["region"] in regions and not r.get("error")
                      and r.get("s15_eligible", True)
                      and (args.cache / "mask_u8" / f"{sid}.npy").exists())

    splits = {s: members(s) for s in ("train", "val", "test")}
    print("split:", {k: len(v) for k, v in splits.items()}, flush=True)
    if min(len(v) for v in splits.values()) == 0:
        raise SystemExit("캐시가 비었다. extract_sen12_fold_cache.py 를 먼저 완료해야 한다.")

    # ---- timestamp parity: S12q 12시점의 월 인덱스 ----
    # P4는 OlmoEarth wrapper 내부 position encoding으로 월(0~11)을 받는다. raw arm에도
    # 같은 정보를 준다. 이것이 M25가 지적한 정보 비대칭의 해소다.
    months_p = args.cache / "months.jsonl"
    months: dict[str, list[int]] = {}
    if months_p.exists():
        for line in months_p.read_text(encoding="utf-8").splitlines():
            if line:
                r = json.loads(line)
                months[r["sample_id"]] = r["months_0_11"]
    print("months 로드 %d개" % len(months), flush=True)

    class S12(Dataset):
        """arm이 요구하는 텐서만 읽는다. mmap으로 RAM을 아낀다."""

        def __init__(self, ids, kind, emb_stats=None):
            self.ids, self.kind, self.stats = ids, kind, emb_stats

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            sid = self.ids[i]
            y = np.load(args.cache / "mask_u8" / f"{sid}.npy", mmap_mode="r")
            y = torch.from_numpy(np.array(y, copy=True, order="C")).float().unsqueeze(0)
            if self.kind == "emb":
                x = np.load(args.cache / "emb_fp16" / f"{sid}.npy", mmap_mode="r")
                x = torch.from_numpy(np.array(x, copy=True, order="C")).float()
                if self.stats is not None:
                    x = (x - self.stats[0]) / self.stats[1]
            else:
                x = np.load(args.cache / "raw_u16" / f"{sid}.npy", mmap_mode="r")
                x = torch.from_numpy(np.array(x, copy=True, order="C")).float() / 10000.0
                x = x.clamp(0.0, 1.5)                       # C,T,H,W
                if self.kind == "raw_mean":
                    x = x.mean(dim=1)                        # C,H,W
            m = months.get(sid)
            if m is None:
                mt = torch.zeros(12, dtype=torch.float32)
            else:
                mt = torch.tensor(m[:12], dtype=torch.float32)
                if mt.numel() < 12:
                    mt = F.pad(mt, (0, 12 - mt.numel()))
            return x, y, mt, sid

    # ---- embedding 채널 통계는 **train split에서만** 낸다 (누수 방지) ----
    def emb_stats(train_ids, sample=400):
        """고정된 400표본 근사 통계. 전체 train 통계라고 과장하지 않는다."""
        idx = np.linspace(0, len(train_ids) - 1, min(sample, len(train_ids))).astype(int)
        acc = np.zeros((768,), dtype="float64")
        acc2 = np.zeros((768,), dtype="float64")
        n = 0
        for j in idx:
            a = np.load(args.cache / "emb_fp16" / f"{train_ids[j]}.npy").astype("float32")
            acc += a.mean(axis=(1, 2)); acc2 += (a ** 2).mean(axis=(1, 2)); n += 1
        mean = acc / n
        var = np.maximum(acc2 / n - mean ** 2, 1e-6)
        return (torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1),
                torch.tensor(np.sqrt(var), dtype=torch.float32).view(-1, 1, 1))

    def conv_bn(i, o, k=3, d=2):
        return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, k, padding=k // 2), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True))

    class ShallowUNet(nn.Module):
        """P1 — raw 시간평균 10x128x128 → mask."""

        def __init__(self, cin=10, base=32):
            super().__init__()
            self.e1, self.e2 = conv_bn(cin, base), conv_bn(base, base * 2)
            self.b = conv_bn(base * 2, base * 4)
            self.d2, self.d1 = conv_bn(base * 4 + base * 2, base * 2), conv_bn(base * 2 + base, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(F.max_pool2d(e1, 2))
            b = self.b(F.max_pool2d(e2, 2))
            d2 = self.d2(torch.cat([F.interpolate(b, scale_factor=2, mode="nearest"), e2], 1))
            d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="nearest"), e1], 1))
            return self.head(d1)

    class UNet3D(nn.Module):
        """P2-tiny — deterministic factorized-pool 3D encoder → 2D decoder.

        This remains a pilot stand-in, not the official Sen12 3D U-Net. CUDA
        max/avg_pool3d backward have no deterministic implementation in torch
        2.7, so strict replay factorizes pooling into adjacent-time averaging
        and deterministic 2D spatial max pooling.
        """

        def __init__(self, cin=10, base=16):
            super().__init__()
            def blk(i, o):
                return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o),
                                     nn.ReLU(inplace=True),
                                     nn.Conv3d(o, o, 3, padding=1), nn.BatchNorm3d(o),
                                     nn.ReLU(inplace=True))
            self.e1, self.e2 = blk(cin, base), blk(base, base * 2)
            self.b = blk(base * 2, base * 4)
            self.d2 = conv_bn(base * 4 + base * 2, base * 2)
            self.d1 = conv_bn(base * 2 + base, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):                                # B,C,T,H,W
            def pool_time_space(tensor):
                batch, channels, timesteps, height, width = tensor.shape
                if timesteps % 2:
                    raise RuntimeError("P2-tiny temporal pooling requires an even timestep count")
                # Pairwise temporal mean has a deterministic elementwise backward.
                tensor = (tensor[:, :, 0::2] + tensor[:, :, 1::2]) * 0.5
                tensor = tensor.permute(0, 2, 1, 3, 4).reshape(
                    batch * (timesteps // 2), channels, height, width)
                tensor = F.max_pool2d(tensor, 2)
                return tensor.reshape(batch, timesteps // 2, channels,
                                      height // 2, width // 2).permute(0, 2, 1, 3, 4)

            e1 = self.e1(x)
            e2 = self.e2(pool_time_space(e1))
            b = self.b(pool_time_space(e2))
            # 시간축을 평균으로 접어 2D decoder로 넘긴다
            e1s, e2s, bs = e1.mean(2), e2.mean(2), b.mean(2)
            d2 = self.d2(torch.cat([F.interpolate(bs, size=e2s.shape[-2:], mode="nearest"), e2s], 1))
            d1 = self.d1(torch.cat([F.interpolate(d2, size=e1s.shape[-2:], mode="nearest"), e1s], 1))
            return self.head(d1)

    class EmbDecoder(nn.Module):
        """P4 — frozen 768x32x32 → 32→64→128 upsample decoder → mask."""

        def __init__(self, cin=768, base=128):
            super().__init__()
            self.proj = nn.Sequential(nn.Conv2d(cin, base, 1), nn.BatchNorm2d(base),
                                      nn.ReLU(inplace=True))
            self.u1, self.u2 = conv_bn(base, base // 2), conv_bn(base // 2, base // 4)
            self.head = nn.Conv2d(base // 4, 1, 1)

        def forward(self, x):
            x = self.proj(x)
            x = self.u1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            x = self.u2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            return F.interpolate(self.head(x), size=(128, 128), mode="bilinear",
                                 align_corners=False)

    def forward(model, arm, x, mt):
        """arm별 입력 계약. **월 정보를 모든 raw arm에 동일하게 준다** (timestamp parity).

        P4는 OlmoEarth wrapper 내부 position encoding으로 월(0~11)을 이미 받는다.
        raw arm에 같은 정보를 주지 않으면 모델 차이와 정보량 차이가 섞인다(M25 지적).
        """
        if arm == "P4":
            return model(x)
        if arm == "P1":
            b, c, h, w = x.shape
            ch = (mt.mean(dim=1) / 11.0).view(b, 1, 1, 1).expand(b, 1, h, w)
            return model(torch.cat([x, ch], dim=1))
        b, c, t, h, w = x.shape
        ch = (mt[:, :t] / 11.0).view(b, 1, t, 1, 1).expand(b, 1, t, h, w)
        xin = torch.cat([x, ch], dim=1)
        if arm == "P3":
            return model(xin, mt[:, :t].long().clamp(0, 11))
        return model(xin)

    ARMS = {
        # P1은 대조 하한. 월 1채널을 받으므로 in_channels=11.
        "P1": ("raw_mean", lambda: ShallowUNet(cin=11),
               "raw 시간평균 shallow U-Net (+month parity)"),
        # P2/P3는 **공식 구조 이식본** (M27: 구조 변경 없음, 비결정적 커널만 교체)
        "P2": ("raw", lambda: OfficialUNet3D(in_channels=11),
               "공식 UNet3D 이식 — strided Conv3d, AdaptiveAvgPool3d→mean(dim=2) (+month)"),
        "P3": ("raw", lambda: OfficialUTAE(in_channels=11),
               "공식 U-TAE 이식 — enc[64,64,64,128] dec[32,32,64,128] k4/s2/p1, "
               "LTAE n_head16 d_model256 d_k4 att_group (+month)"),
        "P2_tiny": ("raw", UNet3D,
                    "M25의 P2-tiny stand-in. 참고용으로만 보존 (strong baseline 아님)"),
        "P4": ("emb", EmbDecoder, "frozen OlmoEarth v1 + spatial decoder"),
    }

    stats = emb_stats(splits["train"])
    results = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if arm not in ARMS:
            raise SystemExit(f"알 수 없는 arm: {arm}")
        # 이전 arm의 학습/평가가 다음 arm의 초기화·shuffle을 바꾸지 않게 독립 reset한다.
        seed_everything(torch, np, SEED)
        kind, cls, desc = ARMS[arm]
        st = stats if kind == "emb" else None
        loaders = {}
        for split_idx, s in enumerate(("train", "val", "test")):
            generator = torch.Generator().manual_seed(SEED + split_idx)
            loaders[s] = DataLoader(
                S12(splits[s], kind, st), batch_size=BATCH, shuffle=(s == "train"),
                num_workers=6, pin_memory=True, drop_last=(s == "train"),
                generator=generator, persistent_workers=False)
        model = cls().to(device)
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # pos_weight는 표본 300개 근사가 아니라 train mask **전체**에서 계산한다.
        pos = neg = 0
        for sid in splits["train"]:
            m = np.load(args.cache / "mask_u8" / f"{sid}.npy", mmap_mode="r")
            pos += int((m > 0).sum()); neg += int((m == 0).sum())
        pw = min(POS_WEIGHT_CAP, max(1.0, neg / max(pos, 1)))
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        history, best = [], {"val_iou": -1.0, "epoch": 0, "state": None}
        for ep in range(args.epochs):
            model.train()
            tot, nb = 0.0, 0
            for x, y, mt, _ in loaders["train"]:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                mt = mt.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = lossf(forward(model, arm, x, mt).float(), y)
                loss.backward(); opt.step()
                tot += float(loss.detach()); nb += 1
            sched.step()
            # val IoU로 best epoch을 고른다. test는 절대 보지 않는다.
            model.eval()
            vtp = vfp = vfn = 0.0
            with torch.no_grad():
                for x, y, mt, _ in loaders["val"]:
                    x, y, mt = x.to(device), y.to(device), mt.to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pr = (torch.sigmoid(forward(model, arm, x, mt).float()) > 0.5).float()
                    vtp += float((pr * y).sum()); vfp += float((pr * (1 - y)).sum())
                    vfn += float(((1 - pr) * y).sum())
            v_iou = vtp / max(vtp + vfp + vfn, 1e-9)
            history.append({"epoch": ep + 1, "train_loss": round(tot / max(nb, 1), 5),
                            "val_iou": round(v_iou, 5),
                            "seconds": round(time.perf_counter() - t0, 1)})
            if v_iou > best["val_iou"]:
                best = {"val_iou": v_iou, "epoch": ep + 1,
                        "state": {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}}
            if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
                print(f"  [{arm}] epoch {ep+1}/{args.epochs} loss {tot/max(nb,1):.4f} "
                      f"val_iou {v_iou:.4f} (best {best['val_iou']:.4f}@{best['epoch']}) "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
        train_s = time.perf_counter() - t0
        if best["state"] is not None:
            model.load_state_dict(best["state"])   # test는 best-val 가중치로만 평가한다

        checkpoint_dir = args.out / "checkpoints" / args.fold
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{arm}_best.pt"
        torch.save({
            "schema": "sen12-gp-checkpoint-v2",
            "arm": arm,
            "fold": args.fold,
            "seed": SEED,
            "best_val_epoch": best["epoch"],
            "best_val_iou": best["val_iou"],
            "model_state": best["state"],
        }, checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)

        @torch.no_grad()
        def evaluate(split):
            model.eval()
            tp = fp = fn = tn = 0.0
            score_chunks, label_chunks = [], []
            patch_rows = []
            # 공식 저장소의 S12LS-LD와 비교하려면 **마스크 > 50 픽셀** 부분집합이 필요하다
            # (README: "only annotated patches (>50 annotated pixels per patch)").
            # 같은 pass에서 전체(headline)와 LD 부분집합을 동시에 낸다.
            ld_tp = ld_fp = ld_fn = 0.0
            ld_n = 0
            positive_patch_ious = []
            for x, y, mt, sids in loaders[split]:
                x, y, mt = x.to(device), y.to(device), mt.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p = torch.sigmoid(forward(model, arm, x, mt).float())
                pred = (p > 0.5).float()
                batch_tp = (pred * y).flatten(1).sum(1)
                batch_fp = (pred * (1 - y)).flatten(1).sum(1)
                batch_fn = ((1 - pred) * y).flatten(1).sum(1)
                batch_tn = ((1 - pred) * (1 - y)).flatten(1).sum(1)
                tp += float(batch_tp.sum()); fp += float(batch_fp.sum())
                fn += float(batch_fn.sum()); tn += float(batch_tn.sum())
                mask_pixels = y.flatten(1).sum(1)
                # 공식 README 문구가 strictly >50이므로 50을 포함하지 않는다.
                keep = mask_pixels > 50
                if bool(keep.any()):
                    pk, yk = pred[keep], y[keep]
                    ld_tp += float((pk * yk).sum()); ld_fp += float((pk * (1 - yk)).sum())
                    ld_fn += float(((1 - pk) * yk).sum()); ld_n += int(keep.sum())
                pf = p.detach().cpu().numpy().astype("float32", copy=False)
                yf = y.detach().cpu().numpy().astype("uint8", copy=False)
                score_chunks.append(pf.reshape(-1))
                label_chunks.append(yf.reshape(-1))
                for i, sid in enumerate(sids):
                    p_tp, p_fp, p_fn = (float(batch_tp[i]), float(batch_fp[i]),
                                        float(batch_fn[i]))
                    den = p_tp + p_fp + p_fn
                    mask_n = int(mask_pixels[i])
                    patch_iou = p_tp / den if den else 1.0
                    if mask_n > 0:
                        positive_patch_ious.append(patch_iou)
                    patch_rows.append({
                        "sample_id": sid,
                        "region": records[sid]["region"],
                        "ann_id": records[sid].get("ann_id"),
                        "event_date": records[sid].get("event_date"),
                        "mask_positive_pixels": mask_n,
                        "prediction_positive_pixels": int(batch_tp[i] + batch_fp[i]),
                        "tp": int(p_tp), "fp": int(p_fp), "fn": int(p_fn),
                        "iou_at_0_5": round(patch_iou, 8),
                        "mean_probability": round(float(pf[i].mean()), 8),
                    })
            iou = tp / max(tp + fp + fn, 1e-9)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
            precision = tp / max(tp + fp, 1e-9)
            recall = tp / max(tp + fn, 1e-9)
            scores = np.concatenate(score_chunks)
            labels = np.concatenate(label_chunks)
            auprc = exact_average_precision(scores, labels)
            ece = fixed_bin_ece(scores, labels, BINS)
            brier = float(np.mean((scores - labels) ** 2))
            clipped = np.clip(scores.astype("float64"), 1e-7, 1 - 1e-7)
            nll = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))
            ld_iou = ld_tp / max(ld_tp + ld_fp + ld_fn, 1e-9)
            ld_f1 = 2 * ld_tp / max(2 * ld_tp + ld_fp + ld_fn, 1e-9)
            metrics = {"iou": round(iou, 6), "f1": round(f1, 6),
                    "precision": round(precision, 6), "recall": round(recall, 6),
                    "auprc_exact": (round(auprc, 6) if auprc is not None else None),
                    "ece_15bin_pixel_micro": round(ece, 6),
                    "brier_pixel_micro": round(brier, 6),
                    "nll_pixel_micro": round(nll, 6),
                    "positive_pixel_frac": round(float(labels.mean()), 8),
                    "positive_patch_macro_iou": (round(float(np.mean(positive_patch_ious)), 6)
                                                 if positive_patch_ious else None),
                    "positive_patch_n": len(positive_patch_ious),
                    "confusion_pixels": {"tp": int(tp), "fp": int(fp),
                                         "fn": int(fn), "tn": int(tn)},
                    # 공식 저장소 S12LS-LD 참고용 (마스크 >50 픽셀 표본만)
                    "ld_subset_n": ld_n, "ld_iou": round(ld_iou, 5),
                    "ld_f1": round(ld_f1, 5)}
            return metrics, patch_rows

        val_metrics, val_rows = evaluate("val")
        test_metrics, test_rows = evaluate("test")
        eval_dir = args.out / "per_sample" / args.fold
        eval_dir.mkdir(parents=True, exist_ok=True)
        per_sample_paths = {}
        for split, rows in (("val", val_rows), ("test", test_rows)):
            path = eval_dir / f"{arm}_{split}.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                            encoding="utf-8")
            per_sample_paths[split] = {"path": str(path), "sha256": sha256_file(path),
                                       "rows": len(rows)}

        results[arm] = {
            "desc": desc, "trainable_params": n_par, "pos_weight": round(pw, 3),
            "fit_plus_epoch_val_seconds": round(train_s, 1),
            "best_val_epoch": best["epoch"], "best_val_iou": round(best["val_iou"], 5),
            "history": history,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
            "per_sample": per_sample_paths,
            "val": val_metrics, "test": test_metrics,
        }
        print(f"  [{arm}] params {n_par:,} · train {train_s:.0f}s · "
              f"test {results[arm]['test']}", flush=True)
        del model, loaders
        torch.cuda.empty_cache()

    cache_bytes = {k: sum(p.stat().st_size for p in (args.cache / k).glob("*.npy"))
                   for k in ("emb_fp16", "raw_u16", "mask_u8")}
    cache_summary_path = args.cache / "cache_summary.json"
    cache_summary = (json.loads(cache_summary_path.read_text(encoding="utf-8"))
                     if cache_summary_path.is_file() else None)
    summary = {
        "schema": "sen12-gp-pilot-v2",
        "code_sha256": sha256_file(Path(__file__)),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        },
        "fold": args.fold, "test_region": fold["test_region"],
        "val_region": fold["val_region"],
        "split_counts": {k: len(v) for k, v in splits.items()},
        "evidence_status": "development_only_not_confirmatory",
        "test_exposure": (
            "chimanimani test는 M23 8-epoch 결과에서 이미 열람됐다. 이후 40-epoch와 v2는 개발 결과다. "
            "최종 주장은 미열람 지역 fold를 사전등록 후 한 번만 평가해야 한다."),
        "protocol_history": (
            "1차 8-epoch 실행에서 세 arm 모두 손실이 단조 하강 중(미수렴)이어서 예산을 모든 arm에 "
            "동일하게 늘리고 val IoU로 best epoch을 골랐다. 이 변경은 test 열람 뒤 이뤄졌으므로 "
            "confirmatory 사전등록이 아니다. 1차 결과는 M23에 보존한다."),
        "development_protocol_v2": {"epochs": args.epochs, "batch": BATCH, "seed": SEED,
                          "model_selection": "best val IoU; test never used for selection",
                          "decision_threshold": 0.5,
                          "lr": LR, "weight_decay": WD,
                          "pos_weight_cap": POS_WEIGHT_CAP,
                          "pos_weight_source": "all train masks, exact",
                          "raw_norm": "uint16/10000, clamp[0,1.5]",
                          "emb_norm": "fixed 400-sample train-only channel mean/std approximation",
                          "timestep_policy": "S12q (shared across arms)",
                          "auprc": "exact over all pixels; tied scores grouped",
                          "ece": "15 equal-width bins, pixel-micro",
                          "determinism": {
                              "strict_algorithms": True,
                              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                              "cudnn_benchmark": False,
                              "cudnn_deterministic": True,
                              "tf32": False,
                              "float32_matmul_precision": "highest",
                          },
                          "information_contract": {
                              "shared_timestep_indices": True,
                              "known_mismatch": (
                                  "P4 encoder received acquisition timestamps; P1/P2 only receive order"
                              ),
                              "claim_status": "not timestamp-matched",
                          }},
        "cache_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path),
                        "summary": cache_audit},
        "cache_build_summary": cache_summary,
        "cache_bytes": cache_bytes,
        "cost_scope": {
            "reported_fit_time": "head fit + validation every epoch; excludes cache construction",
            "reported_peak_cuda": "head training/evaluation only",
            "p4_warning": ("P4의 frozen OlmoEarth encoder 파라미터·cache extraction 비용을 0으로 "
                           "간주하지 않는다. end-to-end/cold 비용과 task 수에 따른 amortized 비용은 "
                           "별도 표가 나오기 전까지 비교 불가."),
        },
        "arms": results,
        "caveat": ("test는 held-out 지역 1개뿐이다. region-macro는 10-fold 전체에서만 나온다. "
                   "이미 열람한 개발 fold이므로 지역 일반화나 최종 test로 읽지 않는다. "
                   "patch들은 같은 inventory/event와 중첩될 수 있어 1,133개 독립 사건이 아니다. "
                   "pixel-micro ECE는 배경 픽셀이 지배하므로 router calibration 근거가 아니다."),
        "official_repository_benchmark_S12LS_LD": {
            "source": "PaulH97/Sen12Landslides README current benchmark, S2+DEM, seeds 42/123/777 평균",
            "U-TAE": {"AP": 67.75, "F1": 61.80, "IoU": 44.74},
            "U-ConvLSTM": {"AP": 65.13, "F1": 61.95, "IoU": 44.88},
            "Unet3d": {"AP": 62.08, "F1": 58.82, "IoU": 41.66},
            "ConvGRU": {"AP": 60.00, "F1": 59.06, "IoU": 41.91},
            "not_comparable_because": [
                "task: 그들은 mask>50 픽셀 표본만(양성 밀집). 우리 headline은 음성 포함",
                "split: 그들은 random 80/20. 우리는 leave-one-region-out",
                "input: 그들은 11채널(밴드+SCL+DEM) 15 timestep. 우리는 10밴드 12 timestep, DEM 없음",
                f"optimization: 그들은 75 epoch BCEDice(pos_weight 5, dice_w 0.5). 우리는 {args.epochs} epoch BCE(train exact pos_weight)",
            ],
            "corrections": [
                "위 표는 Scientific Data 논문의 원 표가 아니라 공식 저장소가 추가한 binary benchmark다",
                "Scientific Data 논문 자체는 50 epoch/cross-entropy 설정과 별도의 geographic cluster LOCO도 보고한다",
            ],
            "our_ld_subset_note": "ld_iou/ld_f1 은 >50 필터만 맞춘 참고값이며 split·input·loss는 여전히 다르다",
        },
    }
    (args.out / f"{args.fold}_pilot.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
