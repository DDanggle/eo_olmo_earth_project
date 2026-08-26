#!/usr/bin/env python3
"""E1 보조 v2 — 지역 균형 표본에서 crop-boundary artifact를 짝지어 진단한다.

v1은 정렬된 sample ID의 첫 300개를 써 지역 편향 가능성이 있고 평균만 보고했다.
v2는 각 headline region에서 같은 수를 결정론적으로 뽑고, tiled/full cache의
boundary excess에 대한 difference-in-differences와 지역-표본 계층 bootstrap CI를 낸다.

이 진단은 표현 artifact를 재며 downstream 성능 효과나 인과적 성능 개선을 증명하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np


TILED = pathlib.Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani/emb_fp16")
FULL = pathlib.Path(
    sys.argv[1] if len(sys.argv) > 1
    else "/home/work/data/olmoearth/sen12_pilot_full128/holdout_chimanimani/emb_fp16"
)
CONTRACT = pathlib.Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl")
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/cache_context_diagnostic_v2.json")
BND = 16
PER_REGION = 30
N_BOOT = 10_000
SEED = 20260826


def stable_order(sample_id: str) -> str:
    return hashlib.sha256(f"{SEED}:{sample_id}".encode()).hexdigest()


def neighbor_stats(a: np.ndarray) -> dict[str, float]:
    """Return cosine-distance means for pairs crossing the fixed crop axes and all others."""
    v = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-8)
    horizontal = 1 - (v[:, :, :-1] * v[:, :, 1:]).sum(0)
    vertical = 1 - (v[:, :-1, :] * v[:, 1:, :]).sum(0)
    cross = np.concatenate([horizontal[:, BND - 1], vertical[BND - 1, :]])
    inner = np.concatenate([
        horizontal[:, :BND - 1].ravel(), horizontal[:, BND:].ravel(),
        vertical[:BND - 1, :].ravel(), vertical[BND:, :].ravel(),
    ])
    cross_mean, inner_mean = float(cross.mean()), float(inner.mean())
    return {
        "cross_mean": cross_mean,
        "inner_mean": inner_mean,
        "boundary_excess": cross_mean - inner_mean,
        "ratio": cross_mean / max(inner_mean, 1e-12),
    }


def hierarchical_bootstrap(rows: list[dict], key: str) -> dict:
    """Equal-region estimand; resample regions, then samples within each selected region."""
    region_names = sorted({r["region"] for r in rows})
    grouped = {name: np.array([r[key] for r in rows if r["region"] == name], dtype=float)
               for name in region_names}
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        selected_regions = rng.integers(0, len(region_names), size=len(region_names))
        region_means = []
        for region_idx in selected_regions:
            values = grouped[region_names[region_idx]]
            selected_values = values[rng.integers(0, len(values), size=len(values))]
            region_means.append(float(selected_values.mean()))
        draws[i] = float(np.mean(region_means))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    observed = float(np.mean([values.mean() for values in grouped.values()]))
    return {
        "estimand": "equal-region macro mean",
        "observed": round(observed, 6),
        "ci95_percentile": [round(float(lo), 6), round(float(hi), 6)],
        "bootstrap_tail_fraction_le_0": round(float((draws <= 0).mean()), 6),
        "n_regions": len(region_names),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
    }


def main() -> None:
    common_ids = {p.stem for p in FULL.glob("*.npy")}
    common_ids &= {p.stem for p in TILED.glob("*.npy")}
    if not common_ids:
        raise SystemExit("비교할 샘플이 없음 — full128 캐시가 아직 안 만들어졌음")

    by_region: dict[str, list[str]] = {}
    for line in CONTRACT.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        sample_id = record["sample_id"]
        if sample_id in common_ids and record.get("s15_eligible") is True:
            by_region.setdefault(record["region"], []).append(sample_id)
    if len(by_region) < 2:
        raise SystemExit("region 층화에 필요한 두 지역 이상을 찾지 못함")

    selected: list[tuple[str, str]] = []
    for region, ids in sorted(by_region.items()):
        ordered = sorted(ids, key=stable_order)
        selected.extend((region, sample_id) for sample_id in ordered[:PER_REGION])

    rows = []
    for region, sample_id in selected:
        tiled = np.load(TILED / f"{sample_id}.npy").astype("float32")
        full = np.load(FULL / f"{sample_id}.npy").astype("float32")
        tiled_stats, full_stats = neighbor_stats(tiled), neighbor_stats(full)
        tiled_norm = tiled / (np.linalg.norm(tiled, axis=0, keepdims=True) + 1e-8)
        full_norm = full / (np.linalg.norm(full, axis=0, keepdims=True) + 1e-8)
        token_difference = 1 - (tiled_norm * full_norm).sum(0)
        boundary_band = np.zeros((32, 32), dtype=bool)
        boundary_band[BND - 1:BND + 1, :] = True
        boundary_band[:, BND - 1:BND + 1] = True
        rows.append({
            "sample_id": sample_id,
            "region": region,
            "tiled_cross": tiled_stats["cross_mean"],
            "tiled_inner": tiled_stats["inner_mean"],
            "tiled_excess": tiled_stats["boundary_excess"],
            "full_cross": full_stats["cross_mean"],
            "full_inner": full_stats["inner_mean"],
            "full_excess": full_stats["boundary_excess"],
            "difference_in_differences": (
                tiled_stats["boundary_excess"] - full_stats["boundary_excess"]
            ),
            "tiled_full_boundary_cosdist": float(token_difference[boundary_band].mean()),
            "tiled_full_interior_cosdist": float(token_difference[~boundary_band].mean()),
            "relative_frobenius_difference": float(
                np.linalg.norm(tiled - full) / (np.linalg.norm(tiled) + 1e-8)
            ),
        })

    selected_ids = "\n".join(sorted(r["sample_id"] for r in rows)) + "\n"
    per_region = {}
    for region in sorted(by_region):
        region_rows = [r for r in rows if r["region"] == region]
        if not region_rows:
            continue
        per_region[region] = {
            "n": len(region_rows),
            "tiled_excess_mean": round(float(np.mean([r["tiled_excess"] for r in region_rows])), 6),
            "full_excess_mean": round(float(np.mean([r["full_excess"] for r in region_rows])), 6),
            "difference_in_differences_mean": round(
                float(np.mean([r["difference_in_differences"] for r in region_rows])), 6
            ),
        }

    out = {
        "schema": "cache-context-diagnostic-v2",
        "selection": {
            "method": "SHA256(seed:sample_id), first 30 per region among common cache IDs",
            "n_samples": len(rows),
            "n_regions": len(per_region),
            "per_region_target": PER_REGION,
            "selected_sample_ids_sha256": hashlib.sha256(selected_ids.encode()).hexdigest(),
        },
        "primary_estimand": (
            "(tiled boundary−interior cosine distance) − "
            "(full boundary−interior cosine distance)"
        ),
        "difference_in_differences": hierarchical_bootstrap(rows, "difference_in_differences"),
        "tiled_boundary_excess": hierarchical_bootstrap(rows, "tiled_excess"),
        "full_boundary_excess": hierarchical_bootstrap(rows, "full_excess"),
        "per_region": per_region,
        "descriptive": {
            key: round(float(np.mean([r[key] for r in rows])), 6)
            for key in (
                "tiled_cross", "tiled_inner", "full_cross", "full_inner",
                "tiled_full_boundary_cosdist", "tiled_full_interior_cosdist",
                "relative_frobenius_difference",
            )
        },
        "limitations": [
            "표현 차이를 재며 downstream 성능 효과를 증명하지 않는다.",
            "고정 중심축과 실제 지형 경계의 우연한 정렬을 full-cache 대조로 완화하지만 제거하지 못한다.",
            "같은 샘플의 두 cache가 공유하는 encoder/입력 오차는 독립적으로 식별하지 못한다.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
