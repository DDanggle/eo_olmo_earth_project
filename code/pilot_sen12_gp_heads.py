#!/usr/bin/env python3
"""G-P pilot 2단계 — 한 LOCO fold에서 P1/P2/P4를 **동일 예산**으로 학습·평가한다. GPU1 전용.

목적: frozen OlmoEarth v1이 산사태 segmentation에서 task-specific 모델과 경쟁하는지 판정한다.
이 pilot에서 P4가 경쟁력 있거나 정확도 대비 비용 우위가 확인되면 10-fold·다중 seed로 확장한다.
밀리면 router 이전에 representation adaptation이 필요하다는 뜻이다.

사전 등록 (L4 — 실행 중 바꾸지 않는다)
  입력    S12q. 세 arm이 **같은 12 timestep index**를 쓴다 (캐시가 이미 그것만 담고 있다)
  정규화  raw = uint16 / 10000 (반사도 [0,1] 가정). embedding = train split의 채널별 mean/std
  arm     P1 raw 시간평균 → shallow U-Net (2 down / 2 up)
          P2 raw 12시점 → 3D U-Net (시간축도 downsample)
          P4 frozen emb 768x32x32 → 1x1 conv → 32→64→128 upsample decoder
  손실    BCEWithLogits, pos_weight = train의 (neg/pos), 50으로 상한
  최적화  AdamW lr 1e-3, wd 1e-4, cosine, batch 16, seed 1

프로토콜 수정 (2026-08-25, 사유 공개)
  1차 8 epoch 실행에서 **세 arm 모두 손실이 단조 하강 중**이었다 (P1 0.943→0.574,
  P2 0.961→0.389, P4 0.406→0.186). 즉 전부 미수렴이며, 시작 손실이 낮은 P4에 유리한
  비교였다. 따라서 예산을 **모든 arm에 동일하게** 늘리고 val IoU로 best epoch를 고른다.
  1차 결과를 폐기하지 않고 둘 다 보고한다. best epoch 선택은 **val IoU만** 쓰고 test는 보지 않는다.

  지표    IoU@0.5 · F1@0.5 · AUPRC · ECE(15 bin). test 지역과 val 지역을 따로 보고
  기록    학습가능 파라미터 수 · 학습시간 · peak GPU · 입력 캐시 바이트 · epoch별 이력

주의: test는 fold의 held-out 지역 하나뿐이다. **region-macro는 10-fold 전체에서만 나온다.**
이 pilot의 수치를 region 일반화로 읽지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

EPOCHS, BATCH, SEED = 40, 16, 1
LR, WD = 1e-3, 1e-4
POS_WEIGHT_CAP = 50.0
BINS = 15


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
                   default=Path("/home/work/data/olmoearth/sen12_gp_pilot"))
    p.add_argument("--arms", type=str, default="P1,P2,P4")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    return p.parse_args()


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("refusing to run: CUDA_VISIBLE_DEVICES must be exactly 1")

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
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

    class S12(Dataset):
        """arm이 요구하는 텐서만 읽는다. mmap으로 RAM을 아낀다."""

        def __init__(self, ids, kind, emb_stats=None):
            self.ids, self.kind, self.stats = ids, kind, emb_stats

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            sid = self.ids[i]
            y = np.load(args.cache / "mask_u8" / f"{sid}.npy", mmap_mode="r")
            y = torch.from_numpy(np.ascontiguousarray(y)).float().unsqueeze(0)
            if self.kind == "emb":
                x = np.load(args.cache / "emb_fp16" / f"{sid}.npy", mmap_mode="r")
                x = torch.from_numpy(np.ascontiguousarray(x)).float()
                if self.stats is not None:
                    x = (x - self.stats[0]) / self.stats[1]
            else:
                x = np.load(args.cache / "raw_u16" / f"{sid}.npy", mmap_mode="r")
                x = torch.from_numpy(np.ascontiguousarray(x)).float() / 10000.0
                x = x.clamp(0.0, 1.5)                       # C,T,H,W
                if self.kind == "raw_mean":
                    x = x.mean(dim=1)                        # C,H,W
            return x, y

    # ---- embedding 채널 통계는 **train split에서만** 낸다 (누수 방지) ----
    def emb_stats(train_ids, sample=400):
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
        """P2 — raw 10x12x128x128 → 시간축도 줄이는 3D U-Net → mask."""

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
            e1 = self.e1(x)
            e2 = self.e2(F.max_pool3d(e1, (2, 2, 2)))
            b = self.b(F.max_pool3d(e2, (2, 2, 2)))
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

    ARMS = {
        "P1": ("raw_mean", ShallowUNet, "raw 시간평균 shallow U-Net"),
        "P2": ("raw", UNet3D, "raw 12시점 3D U-Net"),
        "P4": ("emb", EmbDecoder, "frozen OlmoEarth v1 + spatial decoder"),
    }

    stats = emb_stats(splits["train"])
    results = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        kind, cls, desc = ARMS[arm]
        st = stats if kind == "emb" else None
        loaders = {s: DataLoader(S12(splits[s], kind, st), batch_size=BATCH,
                                 shuffle=(s == "train"), num_workers=6,
                                 pin_memory=True, drop_last=(s == "train"))
                   for s in ("train", "val", "test")}
        model = cls().to(device)
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # pos_weight는 train mask에서만 계산한다
        pos = neg = 0
        for j in np.linspace(0, len(splits["train"]) - 1, 300).astype(int):
            m = np.load(args.cache / "mask_u8" / f"{splits['train'][j]}.npy")
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
            for x, y in loaders["train"]:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = lossf(model(x).float(), y)
                loss.backward(); opt.step()
                tot += float(loss); nb += 1
            sched.step()
            # val IoU로 best epoch을 고른다. test는 절대 보지 않는다.
            model.eval()
            vtp = vfp = vfn = 0.0
            with torch.no_grad():
                for x, y in loaders["val"]:
                    x, y = x.to(device), y.to(device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        pr = (torch.sigmoid(model(x).float()) > 0.5).float()
                    vtp += float((pr * y).sum()); vfp += float((pr * (1 - y)).sum())
                    vfn += float(((1 - pr) * y).sum())
            v_iou = vtp / max(vtp + vfp + vfn, 1e-9)
            history.append({"epoch": ep + 1, "train_loss": round(tot / max(nb, 1), 5),
                            "val_iou": round(v_iou, 5),
                            "seconds": round(time.perf_counter() - t0, 1)})
            if v_iou > best["val_iou"]:
                best = {"val_iou": v_iou, "epoch": ep + 1,
                        "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
                print(f"  [{arm}] epoch {ep+1}/{args.epochs} loss {tot/max(nb,1):.4f} "
                      f"val_iou {v_iou:.4f} (best {best['val_iou']:.4f}@{best['epoch']}) "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
        train_s = time.perf_counter() - t0
        if best["state"] is not None:
            model.load_state_dict(best["state"])   # test는 best-val 가중치로만 평가한다

        @torch.no_grad()
        def evaluate(split):
            model.eval()
            tp = fp = fn = 0
            probs_sum = np.zeros(BINS); hits = np.zeros(BINS); cnts = np.zeros(BINS)
            ap_num = ap_den = 0.0
            pos_scores, neg_scores = [], []
            # 원 논문 S12LS-LD와 비교하려면 **마스크 >= 50 픽셀** 부분집합이 필요하다
            # (README: "only annotated patches (>50 annotated pixels per patch)").
            # 같은 pass에서 전체(headline)와 LD 부분집합을 동시에 낸다.
            ld_tp = ld_fp = ld_fn = 0.0
            ld_n = 0
            for x, y in loaders[split]:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    p = torch.sigmoid(model(x).float())
                pred = (p > 0.5).float()
                tp += float((pred * y).sum()); fp += float((pred * (1 - y)).sum())
                fn += float(((1 - pred) * y).sum())
                keep = (y.flatten(1).sum(1) >= 50)
                if bool(keep.any()):
                    pk, yk = pred[keep], y[keep]
                    ld_tp += float((pk * yk).sum()); ld_fp += float((pk * (1 - yk)).sum())
                    ld_fn += float(((1 - pk) * yk).sum()); ld_n += int(keep.sum())
                pf, yf = p.flatten().cpu().numpy(), y.flatten().cpu().numpy()
                b = np.clip((pf * BINS).astype(int), 0, BINS - 1)
                np.add.at(probs_sum, b, pf); np.add.at(hits, b, yf); np.add.at(cnts, b, 1)
                # AUPRC는 표본을 줄여 근사한다 (전 픽셀 정렬은 비싸다)
                k = np.random.RandomState(SEED).choice(len(pf), size=min(20000, len(pf)),
                                                       replace=False)
                pos_scores.extend(pf[k][yf[k] > 0]); neg_scores.extend(pf[k][yf[k] == 0])
            iou = tp / max(tp + fp + fn, 1e-9)
            f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
            nz = cnts > 0
            ece = float(np.sum(cnts[nz] / cnts.sum()
                               * np.abs(hits[nz] / cnts[nz] - probs_sum[nz] / cnts[nz])))
            # AUPRC 근사: threshold sweep
            ps, ns = np.array(pos_scores), np.array(neg_scores)
            auprc = None
            if len(ps) and len(ns):
                ths = np.quantile(np.concatenate([ps, ns]), np.linspace(0, 1, 101))
                prev_r, s = 1.0, 0.0
                for t in ths:
                    tp_ = (ps >= t).sum(); fp_ = (ns >= t).sum()
                    if tp_ == 0:
                        continue
                    prec = tp_ / (tp_ + fp_); rec = tp_ / len(ps)
                    s += prec * max(prev_r - rec, 0.0); prev_r = rec
                auprc = float(s)
            ld_iou = ld_tp / max(ld_tp + ld_fp + ld_fn, 1e-9)
            ld_f1 = 2 * ld_tp / max(2 * ld_tp + ld_fp + ld_fn, 1e-9)
            return {"iou": round(iou, 5), "f1": round(f1, 5),
                    "auprc": (round(auprc, 5) if auprc is not None else None),
                    "ece": round(ece, 5), "positive_pixel_frac":
                        round(float(len(ps) / max(len(ps) + len(ns), 1)), 6),
                    # 원 논문 S12LS-LD 비교용 (마스크 >=50 픽셀 표본만)
                    "ld_subset_n": ld_n, "ld_iou": round(ld_iou, 5),
                    "ld_f1": round(ld_f1, 5)}

        results[arm] = {
            "desc": desc, "trainable_params": n_par, "pos_weight": round(pw, 3),
            "train_seconds": round(train_s, 1),
            "best_val_epoch": best["epoch"], "best_val_iou": round(best["val_iou"], 5),
            "history": history,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "val": evaluate("val"), "test": evaluate("test"),
        }
        print(f"  [{arm}] params {n_par:,} · train {train_s:.0f}s · "
              f"test {results[arm]['test']}", flush=True)
        del model, loaders
        torch.cuda.empty_cache()

    cache_bytes = {k: sum(p.stat().st_size for p in (args.cache / k).glob("*.npy"))
                   for k in ("emb_fp16", "raw_u16", "mask_u8")}
    summary = {
        "schema": "sen12-gp-pilot-v1",
        "fold": args.fold, "test_region": fold["test_region"],
        "val_region": fold["val_region"],
        "split_counts": {k: len(v) for k, v in splits.items()},
        "protocol_amendment": (
            "1차 8-epoch 실행에서 세 arm 모두 손실이 단조 하강 중(미수렴)이어서 예산을 모든 arm에 "
            "동일하게 늘리고 val IoU로 best epoch을 골랐다. test는 선택에 쓰지 않았다. "
            "1차 결과는 폐기하지 않고 M-기록에 함께 남긴다."),
        "preregistered": {"epochs": args.epochs, "batch": BATCH, "seed": SEED,
                          "model_selection": "best val IoU; test never used for selection",
                          "lr": LR, "weight_decay": WD,
                          "pos_weight_cap": POS_WEIGHT_CAP,
                          "raw_norm": "uint16/10000, clamp[0,1.5]",
                          "emb_norm": "train-split channel mean/std (no leakage)",
                          "timestep_policy": "S12q (shared across arms)"},
        "cache_bytes": cache_bytes,
        "arms": results,
        "caveat": ("test는 held-out 지역 1개뿐이다. region-macro는 10-fold 전체에서만 나온다. "
                   "이 수치를 지역 일반화로 읽지 않는다."),
        "published_baseline_S12LS_LD": {
            "source": "PaulH97/Sen12Landslides README, S2+DEM, seeds 42/123/777 평균",
            "U-TAE": {"AP": 67.75, "F1": 61.80, "IoU": 44.74},
            "U-ConvLSTM": {"AP": 65.13, "F1": 61.95, "IoU": 44.88},
            "Unet3d": {"AP": 62.08, "F1": 58.82, "IoU": 41.66},
            "ConvGRU": {"AP": 60.00, "F1": 59.06, "IoU": 41.91},
            "not_comparable_because": [
                "task: 그들은 mask>=50 픽셀 표본만(양성 밀집). 우리 headline은 음성 포함",
                "split: 그들은 random 80/20. 우리는 leave-one-region-out (훨씬 어렵다)",
                "input: 그들은 11채널(밴드+SCL+DEM) 15 timestep. 우리는 10밴드 12 timestep, DEM 없음",
                "epochs: 그들은 75, loss는 BCEDice(pos_weight 5, dice_w 0.5). 우리는 BCE(pos_weight 35)",
            ],
            "our_ld_subset_note": "ld_iou/ld_f1 은 task만 맞춘 값이며 split·input·epoch은 여전히 다르다",
        },
    }
    (args.out / f"{args.fold}_pilot.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
