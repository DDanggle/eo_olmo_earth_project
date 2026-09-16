#!/usr/bin/env python3
"""S3 — 오름 243곳을 동결된 계약대로 채점한다. 사건 쌍 하나, 시간 귀무 둘.

무엇을 재는가
-------------
토큰마다 Δz = 1 − cos(z_before, z_after). 사건 쌍은 2025→2026, 귀무는 2024→2025(주)와
2023→2024(부). 문턱은 **주 귀무의 유효 토큰 p99** 하나다. 그러면 구성상 주 귀무의 깃발율은
≈1% 가 되어야 하고, 벗어나면 파이프라인이 깨진 것이다.

부 귀무는 그 문턱을 **빌려 쓴다.** 주 귀무로 만든 문턱을 부 귀무에 대면 그것도 ≈1% 근처여야
한다 — 이게 구성상 보장되지 않는 첫 독립 검사다. 크게 벗어나면 "연간 평시 변화" 가 해마다
다르다는 뜻이고 그것 자체가 보고할 발견이다.

주장 경계
---------
* 결과는 **읽는 순서**다. "오름 훼손을 탐지했다" 가 아니다.
* 관측 불가 오름은 점수를 갖지 않는다. `abstain` 이고 분모에서 빠지지 않는다.
* 사건·귀무 양쪽에서 깃발이 선 토큰은 연간 변화가 아니라 지속 인공물(지형·수역·만성 구름)
  후보다. 따로 센다. 네팔에는 없던 값싼 필터다.

해상도
------
`--patch 4` 는 40 m 토큰(네팔에서 UNOSAT 으로 검증된 설정, **주 결과**). `--patch 2` 는 20 m
토큰으로 허가 필지(중앙값 1,216 m² < 1,600 m²)를 여러 토큰으로 덮기 위한 **부 결과**다.
20 m 는 미검증이므로 귀무 깃발율 검사를 통과해야만 쓴다.

입력 계약 (실측으로 확인한 것)
-------------------------------
* modality `sentinel2_l2a` 12밴드, 순서가 큐브 저장 순서와 일치함을 확인했다.
* 정규화는 **raw DN** 을 기대한다(DN/10000 을 넣으면 분포가 [0.05,0.36] 으로 눌린다).
  기존 model_s2.yaml 도 passthrough 로 DN 을 그대로 넘긴다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime

import numpy as np

from jeju_paths import ARTIFACT_ROOT, CACHE_ROOT, CONTRACT_ROOT, display_path, ensure
from oreum_observability import (MIN_VALID_TOKEN_FRAC, SCL_CLEAR, TOKEN_BAD_PIXEL_FRAC,
                                 contract_years_pairs)

MODALITY = "sentinel2_l2a"
CROP = 256              # 창 전체를 한 번에. H200 에서 patch 2 도 여유 있다.


def token_valid(scl: np.ndarray, patch: int, buffer: int = 0) -> np.ndarray:
    """유효 토큰. buffer>0 이면 무효 토큰 주위 buffer 토큰을 함께 무효로 한다.

    v8.1 수정(2026-09-16): 첫 지도에서 40 m 1위(문도지오름)의 전후 프레임이 둘 다 구름이고 깃발이
    **구름 가장자리**를 따라 섰다. SCL 은 구름 본체를 잡지만 가장자리·그림자·얇은 권운은 새고, 그
    토큰의 Δz 는 지표가 아니라 대기다. 구름 마스크 완충은 표준 처방이고 Δz 를 보지 않는 규칙이다.
    """
    h, w = scl.shape
    ht, wt = h // patch, w // patch
    good = np.isin(scl[: ht * patch, : wt * patch], SCL_CLEAR)
    v = good.reshape(ht, patch, wt, patch).mean(axis=(1, 3)) >= 1.0 - TOKEN_BAD_PIXEL_FRAC
    for _ in range(buffer):
        bad = ~v
        grown = bad.copy()
        grown[1:, :] |= bad[:-1, :]; grown[:-1, :] |= bad[1:, :]
        grown[:, 1:] |= bad[:, :-1]; grown[:, :-1] |= bad[:, 1:]
        v = ~grown
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    ap.add_argument("--patch", type=int, default=4, choices=(2, 4))
    ap.add_argument("--model", default="OLMOEARTH_V1_BASE")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--buffer", type=int, default=0, help="무효 토큰 주위를 이만큼 더 무효로 (구름 가장자리 완충). v8.1 = 1")
    ap.add_argument("--frame", default="A", choices=("A", "B"),
                    help="B = 제주 격자 공간 귀무. 문턱 세 개(사건·귀무A·귀무B 각각의 풀링 p99)를 내고 프레임 A 깃발율을 그 문턱으로 다시 잰다.")
    a = ap.parse_args()

    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    from rslearn.models.olmoearth_pretrain.model import ModelID, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage

    contract = json.loads((CONTRACT_ROOT / f"{a.contract}_contract.json").read_text())
    YEARS, PAIRS = contract_years_pairs(contract)
    scenes = contract["optical"]["scenes"]
    assign = contract["optical"]["frame_a_assignment"]
    suffix = "_B" if a.frame == "B" else ""
    cache = CACHE_ROOT / f"{a.contract}/prepare{suffix}"
    out_dir = ensure(CACHE_ROOT / f"{a.contract}/scan{suffix}_p{a.patch}")

    spec = Modality.get(MODALITY)
    normalizer = Normalizer(Strategy.COMPUTED, std_multiplier=2)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    model = OlmoEarth(patch_size=a.patch, model_id=getattr(ModelID, a.model), token_pooling=True,
                      use_legacy_timestamps=False, autocast_dtype=None).to(device).eval()
    print(f"device {device} · patch {a.patch} ({a.patch*10} m 토큰) · {a.model}", flush=True)

    def norm(year_cube: np.ndarray) -> np.ndarray:
        x = np.transpose(np.nan_to_num(year_cube, nan=0.0), (1, 2, 0))[None]     # (1,H,W,C)
        return np.transpose(normalizer.normalize(spec, x.astype("float32")), (3, 0, 1, 2)).astype("float32")

    def embed(xn: np.ndarray, t: datetime) -> torch.Tensor:
        img = torch.from_numpy(np.ascontiguousarray(xn)).to(device)
        inp = {MODALITY: RasterImage(image=img, timestamps=[(t, t)])}
        sample, _, _ = model._prepare_modality_inputs(ModelContext(inputs=[inp], metadatas=[]))
        with torch.no_grad():
            tm = model.model(sample, fast_pass=False, patch_size=a.patch)["tokens_and_masks"]
            z = getattr(tm, MODALITY); m = (getattr(tm, f"{MODALITY}_mask") != 2).unsqueeze(-1)
            f = ((z * m).sum(dim=(3, 4)) / m.sum(dim=(3, 4)).clamp(min=1))[0]     # (Ht,Wt,D)
        return f.permute(2, 0, 1).float().cpu()

    def delta(za, zb):
        num = (za * zb).sum(0)
        return (1 - num / (za.norm(dim=0).clamp(min=1e-8) * zb.norm(dim=0).clamp(min=1e-8))).numpy().astype("float32")

    if a.frame == "B":
        assign = {f.stem: None for f in sorted(cache.glob("JJ-GRID-*.npz"))}
    ids = sorted(assign)[: a.limit] if a.limit else sorted(assign)
    t0, done = time.time(), 0
    per_site: dict[str, dict] = {}
    for oid in ids:
        f = cache / f"{oid}.npz"
        if not f.exists():
            continue
        outp = out_dir / f"{oid}_delta.npz"
        d = np.load(f, allow_pickle=True)
        tile = str(d["tile"])
        if outp.exists():
            dd = np.load(outp)
            deltas = {k: dd[f"d_{k}"] for k in PAIRS}
            tv = {y: token_valid(d["scl"][i], a.patch, a.buffer) for i, y in enumerate(YEARS)}
            valids = {k: tv[b] & tv[t_] for k, (b, t_) in PAIRS.items()}
        else:
            z = {}
            for i, y in enumerate(YEARS):
                t = datetime.fromisoformat(scenes[tile][y]["datetime"]).replace(tzinfo=None)
                z[y] = embed(norm(d["cube"][:, i]), t)
            tv = {y: token_valid(d["scl"][i], a.patch, a.buffer) for i, y in enumerate(YEARS)}
            deltas = {k: delta(z[b], z[t_]) for k, (b, t_) in PAIRS.items()}
            valids = {k: tv[b] & tv[t_] for k, (b, t_) in PAIRS.items()}
            np.savez_compressed(outp, **{f"d_{k}": v for k, v in deltas.items()},
                                **{f"v_{k}": v for k, v in valids.items()})
        per_site[oid] = {"tile": tile, "deltas": deltas, "valids": valids}
        done += 1
        if done % 20 == 0:
            print(f"  {done}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)

    if a.frame == "B":
        pools = {k: np.concatenate([s["deltas"][k][s["valids"][k]] for s in per_site.values()]) for k in PAIRS}
        thr_b = {k: float(np.percentile(v, 99)) for k, v in pools.items()}
        # 프레임 A 를 프레임 B 사건 쌍 p99 로 다시 잰다 (공간 귀무): 사건 쌍 자체를 섬 전역에 풀링한 문턱.
        a_dir = CACHE_ROOT / f"{a.contract}/scan_p{a.patch}"
        a_ev = np.concatenate([np.load(f)["d_event"][np.load(f)["v_event"]] for f in sorted(a_dir.glob("*_delta.npz"))]) if a_dir.exists() else np.array([])
        res = {"schema": "jeju-v8-scan-frameB-v1", "frame": "B", "role": "공간 귀무 풀·보정 전용. 오름 결과로 제시 금지.",
               "contract": a.contract, "contract_sha256": contract.get("_self_sha256"), "patch_size": a.patch,
               "grid_points": len(per_site), "tokens_pooled": {k: int(v.size) for k, v in pools.items()},
               "p99_by_pair": thr_b,
               "assumption_check": {"claim": "연간 실질 변화가 섬의 1% 미만이면 사건 쌍의 전역 p99 는 귀무 쌍의 p99 와 비슷해야 한다",
                                    "event_p99_over_nullA_p99": thr_b["event"] / thr_b["null_temporal_primary"],
                                    "nullB_p99_over_nullA_p99": thr_b["null_temporal_secondary"] / thr_b["null_temporal_primary"]},
               "frame_a_event_flag_rate_under_frameB_event_p99": float((a_ev > thr_b["event"]).mean()) if a_ev.size else None,
               "frame_a_event_flag_rate_under_frameB_nullA_p99": float((a_ev > thr_b["null_temporal_primary"]).mean()) if a_ev.size else None,
               "claims": {"forbidden": ["프레임 B 점을 오름으로 표시", "프레임 A·B 조인"]}}
        path = ensure(ARTIFACT_ROOT / "results") / f"{a.contract}_scan_B_p{a.patch}.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"프레임 B 격자 {len(per_site)}점 · p99 사건 {thr_b['event']:.5f} · 귀무A {thr_b['null_temporal_primary']:.5f} · 귀무B {thr_b['null_temporal_secondary']:.5f}")
        print(f"사건/귀무A p99 비 {res['assumption_check']['event_p99_over_nullA_p99']:.3f} · 귀무B/귀무A {res['assumption_check']['nullB_p99_over_nullA_p99']:.3f}")
        if a_ev.size:
            print(f"프레임 A 사건 깃발율: B-사건 p99 기준 {res['frame_a_event_flag_rate_under_frameB_event_p99']*100:.2f}% · B-귀무A p99 기준 {res['frame_a_event_flag_rate_under_frameB_nullA_p99']*100:.2f}%")
        print(f"→ {display_path(path)}"); return

    # ---- 문턱: 주 귀무의 유효 토큰 p99 (전 오름 풀링) -------------------------------
    pool = np.concatenate([s["deltas"]["null_temporal_primary"][s["valids"]["null_temporal_primary"]]
                           for s in per_site.values()])
    thr = float(np.percentile(pool, 99))
    ev_pool = np.concatenate([s["deltas"]["event"][s["valids"]["event"]] for s in per_site.values()])
    nb_pool = np.concatenate([s["deltas"]["null_temporal_secondary"][s["valids"]["null_temporal_secondary"]]
                              for s in per_site.values()])
    rates = {"null_temporal_primary": float((pool > thr).mean()),
             "null_temporal_secondary": float((nb_pool > thr).mean()),
             "event": float((ev_pool > thr).mean())}
    for k in PAIRS:                       # v8x 추가 귀무(null_YYYY_YYYY): 문턱을 빌려 쓴 깃발율 → null-of-nulls 분포
        if k not in rates:
            pk = np.concatenate([s["deltas"][k][s["valids"][k]] for s in per_site.values()])
            rates[k] = float((pk > thr).mean())

    # ---- 오름별 --------------------------------------------------------------------
    sites_out = {}
    for oid, s in per_site.items():
        rec = {"tile": s["tile"]}
        for k in PAIRS:
            v = s["valids"][k]; dlt = s["deltas"][k]
            frac_valid = float(v.mean())
            obs = frac_valid >= MIN_VALID_TOKEN_FRAC
            rec[k] = {"valid_token_frac": round(frac_valid, 4), "observable": bool(obs),
                      "flag_frac": round(float((dlt[v] > thr).mean()), 4) if obs else None,
                      "n_flag": int((dlt[v] > thr).sum()) if obs else None,
                      "median_delta": round(float(np.median(dlt[v])), 5) if obs else None}
        both = s["valids"]["event"] & s["valids"]["null_temporal_primary"]
        persist = (s["deltas"]["event"] > thr) & (s["deltas"]["null_temporal_primary"] > thr) & both
        rec["persistent_tokens"] = int(persist.sum())          # 양쪽 다 깃발 → 인공물 후보
        rec["event_only_tokens"] = int(((s["deltas"]["event"] > thr) & ~(s["deltas"]["null_temporal_primary"] > thr) & both).sum())
        rec["verdict"] = "scored" if rec["event"]["observable"] else "abstain"
        sites_out[oid] = rec

    scored = {k: v for k, v in sites_out.items() if v["verdict"] == "scored"}
    ranking = sorted(scored, key=lambda k: -scored[k]["event"]["flag_frac"])
    for r, oid in enumerate(ranking, 1):
        sites_out[oid]["rank"] = r

    n_tok = (256 // a.patch) ** 2
    result = {
        "schema": "jeju-v8-scan-v1",
        "contract": a.contract, "contract_sha256": contract.get("_self_sha256"),
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "model": a.model, "patch_size": a.patch, "token_ground_m": a.patch * 10, "cloud_buffer_tokens": a.buffer,
        "tokens_per_window": n_tok, "device": str(device),
        "role": "primary (네팔에서 검증된 40 m)" if a.patch == 4 else "secondary (20 m, 허가 필지 조인용; 미검증)",
        "pairs": PAIRS,
        "threshold": {"source": "null_temporal_primary 유효 토큰 p99, 전 오름 풀링",
                      "value": thr, "n_null_tokens": int(pool.size)},
        "flag_rate_pooled": rates,
        "flag_rate_check": {
            "primary_should_be": 0.01, "primary_is": rates["null_temporal_primary"],
            "secondary_borrowed_threshold": rates["null_temporal_secondary"],
            "secondary_note": "부 귀무는 주 귀무의 문턱을 빌려 쓴다. 구성상 보장되지 않는 첫 독립 검사다.",
        },
        "summary": {
            "frame_a_total": len(assign), "cubes": len(per_site),
            "scored": len(scored), "abstain": len(per_site) - len(scored),
            "sites_with_any_event_flag": sum(1 for v in scored.values() if v["event"]["n_flag"]),
            "persistent_tokens_total": sum(v["persistent_tokens"] for v in sites_out.values()),
        },
        "claims": {
            "allowed": ["관측 가능한 오름 N곳 중 연간 변화가 평시 p99 를 넘은 토큰 비율로 읽는 순서",
                        "주 귀무 깃발율 ≈1% (구성상) · 부 귀무 깃발율 (독립 검사)"],
            "forbidden": ["오름 훼손을 탐지했다", "원인·시설 종류", "abstain 을 변화 없음으로",
                          "20 m 결과를 주 결과로 (귀무 검사 통과 전)"],
        },
        "ranking": ranking,
        "sites": sites_out,
        "elapsed_s": round(time.time() - t0, 1),
    }
    path = ensure(ARTIFACT_ROOT / "results") / (f"{a.contract}_scan_p{a.patch}.json" if a.buffer == 0 else f"{a.contract}_scan_p{a.patch}_buf{a.buffer}.json")
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1))

    print(f"\n문턱 p99 = {thr:.5f}  (귀무 토큰 {pool.size:,})")
    print(f"깃발율  주귀무 {rates['null_temporal_primary']*100:.2f}%  (≈1% 여야)  "
          f"부귀무 {rates['null_temporal_secondary']*100:.2f}%  사건 {rates['event']*100:.2f}%")
    print(f"채점 {len(scored)} · 보류 {len(per_site)-len(scored)} · 사건 깃발 있는 오름 {result['summary']['sites_with_any_event_flag']}")
    print("상위 10:")
    for oid in ranking[:10]:
        s = sites_out[oid]
        print(f"  {oid}  사건 {s['event']['flag_frac']*100:5.1f}%  주귀무 {s['null_temporal_primary']['flag_frac'] if s['null_temporal_primary']['flag_frac'] is None else round(s['null_temporal_primary']['flag_frac']*100,1)}%  지속 {s['persistent_tokens']}")
    print(f"\n→ {display_path(path)}  ({result['elapsed_s']}s)")


if __name__ == "__main__":
    main()
