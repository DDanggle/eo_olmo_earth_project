#!/usr/bin/env python3
"""dose-response 분석 — 밴드 순서 계약 불일치가 표현을 얼마나 움직이는가.

각 dose를 dose 0과 비교해 두 가지를 동시에 낸다.

  W1 용량-반응 : 불일치가 커질수록 표현이 얼마나 이동하는가
  W2 진단 눈멂 : CKA/코사인 같은 값싼 진단이 그 이동을 탐지하는가

핵심 대조는 `R@1이 무너지는데 CKA는 높게 유지되는가`이다. 그렇다면 표현 유사도 지표로
캐시 재사용 자격을 판정하면 안 된다는 직접 증거가 된다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DOSE_ORDER = ["0", "1", "2", "3", "6", "reverse"]


def read_embedding(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        arr = ds.read(masked=True).astype(np.float64)
    arr = np.ma.filled(arr, np.nan)
    return np.moveaxis(arr, 0, -1)  # (H, W, C)


def center(m: np.ndarray) -> np.ndarray:
    v = np.asarray(m, dtype=np.float64)
    return v - v.mean(axis=0, keepdims=True)


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    a, b = center(a), center(b)
    cross = a.T @ b
    num = float(np.linalg.norm(cross, ord="fro") ** 2)
    den = float(np.linalg.norm(a.T @ a, ord="fro") * np.linalg.norm(b.T @ b, ord="fro"))
    if den <= 0:
        return float("nan")
    return float(np.clip(num / den, 0.0, 1.0))


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / den) if den > 0 else float("nan")


def recall_at_1(query: np.ndarray, gallery: np.ndarray, block: int = 512) -> float:
    """query[i]의 최근접 gallery가 자기 자신(i)인 비율. 코사인 기준."""
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    g = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    hits = 0
    for start in range(0, q.shape[0], block):
        sims = q[start:start + block] @ g.T
        hits += int(np.sum(np.argmax(sims, axis=1) == np.arange(start, min(start + block, q.shape[0]))))
    return hits / q.shape[0]


def tokens(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, arr.shape[-1])
    finite = np.isfinite(flat).all(axis=1)
    nonzero = np.linalg.norm(flat, axis=1) > 0
    return flat[finite & nonzero]


def layer_path(ds_root: Path, group: str, window: str, layer: str) -> Path | None:
    base = ds_root / "windows" / group / window / "layers" / layer
    if not base.exists():
        return None
    found = sorted(base.rglob("geotiff.tif"))
    return found[0] if found else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--layer-prefix", default="embeddings_dose_")
    ap.add_argument("--sample-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = json.loads((args.work_dir / "preflight.json").read_text(encoding="utf-8"))
    by_dose = {str(d["dose"]): d for d in plan["doses"]}

    windows = []
    for group_dir in sorted((args.dataset_root / "windows").iterdir()):
        if group_dir.is_dir():
            for w in sorted(group_dir.iterdir()):
                if w.is_dir():
                    windows.append((group_dir.name, w.name))
    if not windows:
        raise SystemExit("REFUSED: no windows found")

    rng = np.random.default_rng(args.seed)
    rows = []
    for group, window in windows:
        base_path = layer_path(args.dataset_root, group, window,
                               f"{args.layer_prefix}0")
        if base_path is None:
            print(f"[skip] {window}: dose 0 missing", flush=True)
            continue
        base = tokens(read_embedding(base_path))
        if base.shape[0] < 16:
            print(f"[skip] {window}: too few valid tokens", flush=True)
            continue
        n = min(args.sample_tokens, base.shape[0])
        idx = rng.choice(base.shape[0], size=n, replace=False)
        base_s = base[idx]

        for dose in DOSE_ORDER:
            if dose == "0":
                continue
            p = layer_path(args.dataset_root, group, window,
                           f"{args.layer_prefix}{dose}")
            if p is None:
                continue
            other = tokens(read_embedding(p))
            if other.shape[0] != base.shape[0]:
                print(f"[warn] {window} dose {dose}: token count differs "
                      f"({other.shape[0]} vs {base.shape[0]})", flush=True)
                continue
            other_s = other[idx]

            bn = base_s / np.linalg.norm(base_s, axis=1, keepdims=True)
            on = other_s / np.linalg.norm(other_s, axis=1, keepdims=True)
            cos = float(np.mean(np.sum(bn * on, axis=1)))

            m = min(512, n)
            sub = rng.choice(n, size=m, replace=False)
            db = base_s[sub] @ base_s[sub].T
            do = other_s[sub] @ other_s[sub].T
            iu = np.triu_indices(m, k=1)

            rows.append({
                "group": group, "window": window, "dose": dose,
                "displaced_positions": by_dose[dose]["displaced_positions"],
                "valid_tokens": int(base.shape[0]),
                "sampled_tokens": int(n),
                "same_token_mean_cosine": cos,
                "linear_cka": linear_cka(base_s, other_s),
                "pairwise_distance_spearman": spearman(db[iu], do[iu]),
                "recall_at_1_dose_to_base": recall_at_1(other_s, base_s),
            })
            print(f"[ok] {window} dose={dose} cos={cos:.4f} "
                  f"cka={rows[-1]['linear_cka']:.4f} "
                  f"r@1={rows[-1]['recall_at_1_dose_to_base']:.4f}", flush=True)

    if not rows:
        raise SystemExit("REFUSED: no comparable pairs produced")

    summary = {}
    for dose in DOSE_ORDER[1:]:
        sel = [r for r in rows if r["dose"] == dose]
        if not sel:
            continue
        summary[dose] = {
            "windows": len(sel),
            "displaced_positions": sel[0]["displaced_positions"],
            "median_same_token_cosine": float(np.median([r["same_token_mean_cosine"] for r in sel])),
            "median_linear_cka": float(np.median([r["linear_cka"] for r in sel])),
            "median_distance_spearman": float(np.median([r["pairwise_distance_spearman"] for r in sel])),
            "median_recall_at_1": float(np.median([r["recall_at_1_dose_to_base"] for r in sel])),
            "min_recall_at_1": float(np.min([r["recall_at_1_dose_to_base"] for r in sel])),
        }

    blind = {
        d: {
            "cka_stays_high_while_recall_collapses":
                bool(s["median_linear_cka"] >= 0.90 and s["median_recall_at_1"] <= 0.50),
            "median_linear_cka": s["median_linear_cka"],
            "median_recall_at_1": s["median_recall_at_1"],
        }
        for d, s in summary.items()
    }

    payload = {
        "schema": "contract-dose-response-analysis-v1",
        "axis": plan["axis"],
        "original_bands": plan["original_bands"],
        "sample_tokens": args.sample_tokens,
        "seed": args.seed,
        "per_window": rows,
        "summary_by_dose": summary,
        "diagnostic_blindness": blind,
        "reading": (
            "cka_stays_high_while_recall_collapses=true인 dose가 있으면, 표현 유사도 지표가 "
            "계약 불일치에 눈멀었다는 직접 증거다. 모든 dose에서 CKA와 R@1이 같이 무너지면 "
            "값싼 진단으로 충분하다는 뜻이므로 W2 주장을 철회한다."
        ),
        "forbidden_claims": [
            "이것은 task 정확도 결과가 아니다. 라벨이 없다.",
            "8 site-year 제주 smoke 표본이며 모집단 추정이 아니다.",
            "밴드 순서 축 하나이며 다른 계약 축으로 일반화하지 않는다.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print("\n=== summary ===")
    for d in DOSE_ORDER[1:]:
        if d in summary:
            s = summary[d]
            print(f"dose {d:>7} displaced={s['displaced_positions']:>2} "
                  f"cos={s['median_same_token_cosine']:+.4f} "
                  f"CKA={s['median_linear_cka']:.4f} "
                  f"spearman={s['median_distance_spearman']:.4f} "
                  f"R@1={s['median_recall_at_1']:.4f}")
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
