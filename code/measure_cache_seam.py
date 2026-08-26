#!/usr/bin/env python3
"""E1 보조 — 4x64 타일 캐시에 **seam이 실제로 있는지** 직접 잰다. GPU 불필요.

기존 캐시는 64 crop 4장을 독립 인코딩해 이어붙였다. 토큰 격자 32x32에서
경계는 x=16, y=16 이다. seam이 있다면 그 경계를 **가로지르는** 이웃 토큰 쌍의
차이가 내부 이웃 쌍보다 커야 한다.

두 캐시를 비교도 한다: 같은 샘플의 4x64 캐시와 1x128 캐시가 얼마나 다른가,
그리고 그 차이가 경계 근처에 몰려 있는가.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

TILED = pathlib.Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani/emb_fp16")
FULL = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else "/home/work/data/olmoearth/sen12_pilot_full128/holdout_chimanimani/emb_fp16")
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/cache_seam.json")
BND = 16   # 32 토큰 격자에서 64px crop 경계


def neighbor_stats(a: np.ndarray) -> dict:
    """수평·수직 이웃 토큰 간 코사인 거리. 경계를 가로지르는 쌍과 내부 쌍을 나눈다."""
    v = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-8)   # C,H,W 정규화
    hcos = (v[:, :, :-1] * v[:, :, 1:]).sum(0)     # H, W-1  (열 j 와 j+1)
    vcos = (v[:, :-1, :] * v[:, 1:, :]).sum(0)     # H-1, W
    hd, vd = 1 - hcos, 1 - vcos
    hcross = hd[:, BND - 1]                        # 열 15|16 경계
    hin = np.concatenate([hd[:, :BND - 1].ravel(), hd[:, BND:].ravel()])
    vcross = vd[BND - 1, :]
    vin = np.concatenate([vd[:BND - 1, :].ravel(), vd[BND:, :].ravel()])
    cross = np.concatenate([hcross, vcross]); inner = np.concatenate([hin, vin])
    return {"cross_mean": float(cross.mean()), "inner_mean": float(inner.mean()),
            "ratio": float(cross.mean() / max(inner.mean(), 1e-12))}


def main() -> None:
    ids = sorted(p.stem for p in FULL.glob("*.npy"))
    ids = [i for i in ids if (TILED / f"{i}.npy").exists()]
    if not ids:
        raise SystemExit("비교할 샘플이 없음 — full128 캐시가 아직 안 만들어졌음")
    ids = ids[:300]

    acc = {"tiled": [], "full": []}
    diffs = {"cross": [], "inner": []}
    rel = []
    for sid in ids:
        t = np.load(TILED / f"{sid}.npy").astype("float32")
        f = np.load(FULL / f"{sid}.npy").astype("float32")
        acc["tiled"].append(neighbor_stats(t))
        acc["full"].append(neighbor_stats(f))
        # 두 캐시의 토큰별 코사인 거리 지도
        tn = t / (np.linalg.norm(t, axis=0, keepdims=True) + 1e-8)
        fn = f / (np.linalg.norm(f, axis=0, keepdims=True) + 1e-8)
        d = 1 - (tn * fn).sum(0)                    # 32,32
        band = np.zeros((32, 32), dtype=bool)
        band[BND - 1:BND + 1, :] = True; band[:, BND - 1:BND + 1] = True
        diffs["cross"].append(float(d[band].mean()))
        diffs["inner"].append(float(d[~band].mean()))
        rel.append(float(np.linalg.norm(t - f) / (np.linalg.norm(t) + 1e-8)))

    def agg(rows, key):
        v = np.array([r[key] for r in rows]); return round(float(v.mean()), 6)

    out = {
        "schema": "cache-seam-v1", "n_samples": len(ids), "boundary_token_index": BND,
        "neighbor_cosine_distance": {
            "tiled_4x64": {k: agg(acc["tiled"], k) for k in ("cross_mean", "inner_mean", "ratio")},
            "full_1x128": {k: agg(acc["full"], k) for k in ("cross_mean", "inner_mean", "ratio")},
            "interpretation": "ratio가 1보다 크게 높으면 경계에서 표현이 끊긴다는 뜻",
        },
        "tiled_vs_full_difference": {
            "boundary_band_mean_cosdist": round(float(np.mean(diffs["cross"])), 6),
            "interior_mean_cosdist": round(float(np.mean(diffs["inner"])), 6),
            "relative_frobenius_diff_mean": round(float(np.mean(rel)), 6),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
