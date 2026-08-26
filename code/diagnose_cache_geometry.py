#!/usr/bin/env python3
"""통짜 캐시가 왜 나빴는지 — 캐시의 **기하·정보 구조**를 잰다. GPU 불필요.

M37: full_1x128 캐시가 tiled_4x64보다 downstream 성능이 나빴다(작은 −0.014, 큰 −0.096).
M34: 통짜가 seam은 없앴다. 그런데 성능은 나빠졌다. 기전이 설명되지 않았다.

세 가지 후보를 구분한다.

  A. 과도한 평활화   전역 attention이 국소 신호를 희석 → 공간 고주파 에너지가 낮아짐
  B. 표현 붕괴       유효 랭크(effective rank)가 낮아짐 → 쓸 수 있는 자유도가 줄어듦
  C. 척도 이동       채널 통계(평균·표준편차)가 크게 달라짐 → decoder 최적화 조건이 바뀜

셋 다 같은 성능 저하를 낳을 수 있지만 처방이 다르다.
A면 국소성 복원, B면 표현 자체 문제, C면 정규화만 고치면 된다.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

TILED = pathlib.Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani/emb_fp16")
FULL = pathlib.Path("/home/work/data/olmoearth/sen12_pilot_full128/holdout_chimanimani/emb_fp16")
OUT = pathlib.Path("/home/work/data/olmoearth/gp_official_bundle/cache_geometry.json")
N = 60          # SVD가 무거워 250에서 줄였다 (10분 초과)


def hf_energy(a: np.ndarray) -> float:
    """공간 고주파 비율. 인접 토큰 차분 에너지 / 전체 분산."""
    dx = np.diff(a, axis=2); dy = np.diff(a, axis=1)
    var = a.var(axis=(1, 2)).sum() + 1e-12
    return float((dx ** 2).mean(axis=(1, 2)).sum() + (dy ** 2).mean(axis=(1, 2)).sum()) / float(var)


def eff_rank(mat: np.ndarray) -> float:
    """유효 랭크 = exp(스펙트럼 엔트로피). mat: tokens x channels."""
    s = np.linalg.svd(mat - mat.mean(0, keepdims=True), compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main() -> None:
    ids = sorted(p.stem for p in FULL.glob("*.npy"))
    ids = [i for i in ids if (TILED / f"{i}.npy").exists()][:N]
    res = {"schema": "cache-geometry-v1", "n_samples": len(ids), "caches": {}}
    pooled = {}
    for name, root in (("tiled_4x64", TILED), ("full_1x128", FULL)):
        hfs, ers, means, stds, norms = [], [], [], [], []
        stack = []
        for sid in ids:
            a = np.load(root / f"{sid}.npy").astype("float32")   # C,H,W
            hfs.append(hf_energy(a))
            flat = a.reshape(a.shape[0], -1).T                    # tokens x C
            ers.append(eff_rank(flat))
            means.append(a.mean()); stds.append(a.std())
            norms.append(float(np.linalg.norm(a, axis=0).mean()))
            stack.append(flat[::32])                              # 전역 랭크용 부분표집
        allmat = np.concatenate(stack, axis=0)
        pooled[name] = allmat
        res["caches"][name] = {
            "hf_energy_mean": round(float(np.mean(hfs)), 6),
            "effective_rank_per_tile_mean": round(float(np.mean(ers)), 3),
            "effective_rank_global": round(eff_rank(allmat), 3),
            "channel_mean": round(float(np.mean(means)), 6),
            "channel_std": round(float(np.mean(stds)), 6),
            "token_norm_mean": round(float(np.mean(norms)), 4),
            "max_effective_rank_possible": int(min(allmat.shape))}

    a, b = res["caches"]["tiled_4x64"], res["caches"]["full_1x128"]
    res["verdict"] = {
        "A_oversmoothing": {
            "hf_ratio_full_over_tiled": round(b["hf_energy_mean"] / max(a["hf_energy_mean"], 1e-12), 4),
            "supported": bool(b["hf_energy_mean"] < a["hf_energy_mean"] * 0.9)},
        "B_representation_collapse": {
            "eff_rank_ratio_full_over_tiled": round(
                b["effective_rank_global"] / max(a["effective_rank_global"], 1e-12), 4),
            "supported": bool(b["effective_rank_global"] < a["effective_rank_global"] * 0.9)},
        "C_scale_shift": {
            "std_ratio_full_over_tiled": round(b["channel_std"] / max(a["channel_std"], 1e-12), 4),
            "norm_ratio_full_over_tiled": round(b["token_norm_mean"] / max(a["token_norm_mean"], 1e-12), 4),
            "supported": bool(not 0.8 < b["channel_std"] / max(a["channel_std"], 1e-12) < 1.25)},
    }
    res["caveat"] = ("어느 가설이 지지돼도 그것이 성능 저하의 **원인**이라는 증명은 아니다. "
                     "상관 관찰이며, C가 지지되면 정규화를 맞춘 재실행으로 분리해야 한다.")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
