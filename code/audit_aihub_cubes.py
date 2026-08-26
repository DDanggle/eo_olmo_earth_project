#!/usr/bin/env python3
"""AI-Hub 12밴드 큐브 전수 감사. GPU 불필요.

물질화기는 `boundless=True`로 읽는다. 타일이 STAC item의 MGRS 격자 밖으로 나가면
그 영역이 **0으로 채워진 채 성공으로 집계된다.** M31의 "2,539 성공"은 그래서
아직 신뢰할 수 없다. 전수로 다음을 잰다.

  nodata 비율      밴드별 0 화소 비율. 큰 값이면 격자 밖 채움을 의심한다
  전밴드 동시 0    실제 결측의 지표 (한 밴드만 0인 것은 물리적으로 가능)
  포화·이상치      65535 근처, 음수 불가(uint16)
  platform 일치    STAC platform vs AI-Hub 메타
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

D = pathlib.Path("/home/work/data/olmoearth/aihub/s2_12band")
OUT = D / "cube_audit.json"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def main() -> None:
    rows = [json.loads(l) for l in (D / "manifest.jsonl").read_text().splitlines() if l]
    if LIMIT:
        rows = rows[:LIMIT]
    per, plat_mismatch = [], []
    for i, r in enumerate(rows, 1):
        a = np.load(D / "arrays" / (r["key"] + ".npy"))
        zero_frac = (a == 0).mean(axis=(1, 2))
        all_zero = float((a == 0).all(axis=0).mean())
        per.append({"key": r["key"],
                    "zero_frac_max_band": float(zero_frac.max()),
                    "zero_frac_mean": float(zero_frac.mean()),
                    "all_band_zero_frac": all_zero,
                    "max_value": int(a.max()),
                    "cloud_cover": r.get("cloud_cover")})
        ps, pm = (r.get("platform_stac") or "").upper(), (r.get("platform_meta") or "").upper()
        if ps.replace("-", "") != pm.replace("-", ""):
            plat_mismatch.append({"key": r["key"], "stac": ps, "meta": pm})
        if i % 250 == 0:
            print(f"  [{i}/{len(rows)}]", flush=True)

    az = np.array([p["all_band_zero_frac"] for p in per])
    buckets = {
        "clean_lt_0.1pct": int((az < 0.001).sum()),
        "minor_0.1_1pct": int(((az >= 0.001) & (az < 0.01)).sum()),
        "notable_1_10pct": int(((az >= 0.01) & (az < 0.10)).sum()),
        "severe_ge_10pct": int((az >= 0.10).sum()),
    }
    worst = sorted(per, key=lambda p: -p["all_band_zero_frac"])[:10]
    out = {"schema": "aihub-cube-audit-v1", "n_cubes": len(per),
           "all_band_zero_fraction": {
               "mean": round(float(az.mean()), 6), "median": round(float(np.median(az)), 6),
               "p95": round(float(np.percentile(az, 95)), 6), "max": round(float(az.max()), 6),
               "buckets": buckets},
           "worst_10": worst,
           "platform_mismatch_n": len(plat_mismatch),
           "platform_mismatch_examples": plat_mismatch[:5],
           "max_value_over_all": int(max(p["max_value"] for p in per)),
           "why": "boundless=True 읽기는 격자 밖을 0으로 채우고도 성공으로 집계된다"}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "worst_10"},
                     ensure_ascii=False, indent=2))
    print("worst_10:", json.dumps(worst[:3], ensure_ascii=False))


if __name__ == "__main__":
    main()
