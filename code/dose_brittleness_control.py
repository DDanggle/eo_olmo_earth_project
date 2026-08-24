#!/usr/bin/env python3
"""dose-response 결과의 반증 대조군 — R@1이 원래 잘 깨지는 지표가 아닌가?

위협 1. **지표 취약성.** 한 window의 토큰은 공간적으로 인접해 서로 거의 같다.
   그렇다면 아주 작은 무작위 섭동만으로도 최근접이 옆 토큰으로 넘어가고,
   dose 실험의 R@1 붕괴는 `표현이 깨졌다`가 아니라 `지표가 원래 잘 깨진다`가 된다.
   -> dose 0 임베딩에 크기를 아는 Gaussian 잡음을 넣어 같은 곡선을 그린다.
      아주 작은 잡음이 dose 6과 같은 R@1을 만들면 **dose 주장은 크게 약해진다.**

위협 2. **틀린 이웃이 어디인가.** 최근접이 바로 옆 토큰이면 표현은 멀쩡하고
   지표만 예민한 것이다. -> 잘못 검색된 이웃의 공간 거리를 잰다.
      중앙값이 1~2 토큰이면 국소 혼동, 크면 실제 좌표계 붕괴다.

GPU를 쓰지 않는다. dose 0 raster만 읽는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NOISE_LEVELS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def read_embedding(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read(masked=True).astype(np.float64)
    return np.moveaxis(np.ma.filled(arr, np.nan), 0, -1)


def valid_mask(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1])
    return np.isfinite(flat).all(axis=1) & (np.linalg.norm(flat, axis=1) > 0)


def center_cka(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    num = float(np.linalg.norm(a.T @ b, ord="fro") ** 2)
    den = float(np.linalg.norm(a.T @ a, ord="fro") * np.linalg.norm(b.T @ b, ord="fro"))
    return float(np.clip(num / den, 0.0, 1.0)) if den > 0 else float("nan")


def retrieval(query: np.ndarray, gallery: np.ndarray, coords: np.ndarray,
              block: int = 512):
    """R@1과 '틀렸을 때 이웃이 공간적으로 얼마나 떨어졌는가'를 함께 낸다."""
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    g = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    hits = 0
    miss_dist = []
    n = q.shape[0]
    for s in range(0, n, block):
        e = min(s + block, n)
        nn = np.argmax(q[s:e] @ g.T, axis=1)
        truth = np.arange(s, e)
        ok = nn == truth
        hits += int(ok.sum())
        bad = ~ok
        if bad.any():
            d = np.linalg.norm(coords[nn[bad]] - coords[truth[bad]], axis=1)
            miss_dist.extend(d.tolist())
    return (hits / n,
            float(np.median(miss_dist)) if miss_dist else float("nan"),
            float(np.mean(np.asarray(miss_dist) <= 1.5)) if miss_dist else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--base-layer", default="embeddings_dose_0")
    ap.add_argument("--sample-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    wroot = args.dataset_root / "windows"
    for group in sorted(p for p in wroot.iterdir() if p.is_dir()):
        for win in sorted(p for p in group.iterdir() if p.is_dir()):
            base_dir = win / "layers" / args.base_layer
            if not base_dir.exists():
                continue
            tif = sorted(base_dir.rglob("geotiff.tif"))
            if not tif:
                continue
            arr = read_embedding(tif[0])
            h, w, _ = arr.shape
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            coords_all = np.stack([yy.ravel(), xx.ravel()], axis=1).astype(np.float64)
            m = valid_mask(arr)
            flat = arr.reshape(-1, arr.shape[-1])[m]
            coords = coords_all[m]
            if flat.shape[0] < 64:
                continue
            n = min(args.sample_tokens, flat.shape[0])
            idx = rng.choice(flat.shape[0], size=n, replace=False)
            base, cs = flat[idx], coords[idx]
            scale = float(np.median(np.linalg.norm(base, axis=1)))

            for lvl in NOISE_LEVELS:
                noisy = base + rng.normal(0.0, lvl * scale / np.sqrt(base.shape[1]),
                                          size=base.shape) if lvl > 0 else base.copy()
                r1, med_d, frac_adj = retrieval(noisy, base, cs)
                bn = base / np.linalg.norm(base, axis=1, keepdims=True)
                nn_ = noisy / np.linalg.norm(noisy, axis=1, keepdims=True)
                rows.append({
                    "window": win.name, "noise_level": lvl,
                    "same_token_mean_cosine": float(np.mean(np.sum(bn * nn_, axis=1))),
                    "linear_cka": center_cka(base, noisy),
                    "recall_at_1": r1,
                    "median_miss_spatial_distance_tokens": med_d,
                    "fraction_misses_adjacent": frac_adj,
                })
            print(f"[ok] {win.name}", flush=True)

    if not rows:
        raise SystemExit("REFUSED: no windows analysed")

    summary = {}
    for lvl in NOISE_LEVELS:
        sel = [r for r in rows if r["noise_level"] == lvl]
        if not sel:
            continue
        summary[str(lvl)] = {
            "windows": len(sel),
            "median_same_token_cosine": float(np.median([r["same_token_mean_cosine"] for r in sel])),
            "median_linear_cka": float(np.median([r["linear_cka"] for r in sel])),
            "median_recall_at_1": float(np.median([r["recall_at_1"] for r in sel])),
            "median_miss_spatial_distance_tokens": float(np.nanmedian(
                [r["median_miss_spatial_distance_tokens"] for r in sel])),
            "median_fraction_misses_adjacent": float(np.nanmedian(
                [r["fraction_misses_adjacent"] for r in sel])),
        }

    # dose 6의 R@1(0.2456)과 같은 수준을 만드는 최소 잡음을 찾는다.
    dose6_r1 = 0.2456
    matching = [float(l) for l in summary
                if summary[l]["median_recall_at_1"] <= dose6_r1]
    payload = {
        "schema": "dose-brittleness-control-v1",
        "base_layer": args.base_layer,
        "noise_levels": NOISE_LEVELS,
        "per_window": rows,
        "summary_by_noise": summary,
        "dose6_reference_recall_at_1": dose6_r1,
        "smallest_noise_matching_dose6": (min(matching) if matching else None),
        "verdict_rule": (
            "smallest_noise_matching_dose6가 0.01(1%) 이하이면 R@1은 구조적으로 취약한 지표이며 "
            "dose 실험의 R@1 붕괴를 '표현이 깨졌다'로 해석하면 안 된다. "
            "또한 median_fraction_misses_adjacent가 높으면(예: >0.5) 틀린 이웃이 대부분 바로 옆 "
            "토큰이라는 뜻이므로 좌표계 붕괴가 아니라 국소 혼동이다."
        ),
        "forbidden_claims": [
            "이 대조군은 task 정확도를 측정하지 않는다.",
            "제주 8 site-years 표본이며 모집단 추정이 아니다.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("\n=== noise control ===")
    print("noise | cos     | CKA    | R@1    | miss dist | adj frac")
    for lvl in NOISE_LEVELS:
        k = str(lvl)
        if k in summary:
            s = summary[k]
            print(f"{lvl:>5} | {s['median_same_token_cosine']:+.4f} | "
                  f"{s['median_linear_cka']:.4f} | {s['median_recall_at_1']:.4f} | "
                  f"{s['median_miss_spatial_distance_tokens']:>9.2f} | "
                  f"{s['median_fraction_misses_adjacent']:.3f}")
    print(f"\nsmallest noise matching dose6 R@1={dose6_r1}: "
          f"{payload['smallest_noise_matching_dose6']}")
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
