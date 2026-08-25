#!/usr/bin/env python3
"""timestamp parity 용 — S12q 12시점의 **월 인덱스**를 추출한다. GPU 불필요.

왜 필요한가: P4(frozen OLMo)는 wrapper 내부 position encoding으로 시점 정보를 받는다.
rslearn 주석에 따르면 **월(0~11)만** 쓴다. raw baseline(P2/P3)에 같은 정보를 주지 않으면
모델 차이와 정보량 차이가 섞인다(M25가 지적한 비대칭).

따라서 각 표본의 S12q 12시점에 대한 월 인덱스를 뽑아 캐시에 넣는다.
P2는 월을 채널로 broadcast하고, P3는 LTAE의 `positions` 인자로 받는다.
"""
from __future__ import annotations

import json
from pathlib import Path

CONTRACT = Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl")
DATA = Path("/home/work/data/sen12landslides/extracted")
CACHE = Path("/home/work/data/olmoearth/sen12_pilot/holdout_chimanimani")


def select_timestep_indices(rec, keep=12):
    q = rec.get("scl_clear_fraction")
    if not isinstance(q, list) or len(q) != 15:
        raise ValueError(rec.get("sample_id"))
    return sorted(sorted(range(len(q)), key=lambda i: (-float(q[i]), i))[:keep])


def main() -> None:
    import numpy as np
    import xarray as xr

    ids = sorted(p.stem for p in (CACHE / "mask_u8").glob("*.npy"))
    recs = {}
    for line in CONTRACT.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line)
            if r["sample_id"] in set(ids):
                recs[r["sample_id"]] = r
    out = CACHE / "months.jsonl"
    have = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line:
                have.add(json.loads(line)["sample_id"])
    n = 0
    with out.open("a", encoding="utf-8") as f:
        for sid in ids:
            if sid in have:
                continue
            rec = recs[sid]
            idx = select_timestep_indices(rec)
            with xr.open_dataset(DATA / rec["file"], decode_times=True, cache=False) as ds:
                t = np.asarray(ds["time"].values)[idx]
            months = [int(str(np.datetime_as_string(x, unit="M"))[5:7]) - 1 for x in t]
            years = [int(str(np.datetime_as_string(x, unit="Y"))) for x in t]
            f.write(json.dumps({"sample_id": sid, "timestep_indices": idx,
                                "months_0_11": months, "years": years},
                               ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  {n} / {len(ids)}", flush=True)
    total = sum(1 for line in out.read_text(encoding="utf-8").splitlines() if line)
    print(json.dumps({"schema": "sen12-s12q-months-v1", "written": n,
                      "total_rows": total, "samples": len(ids),
                      "file": str(out),
                      "note": "월 인덱스 0~11. OlmoEarth position encoding이 월만 쓰는 것과 맞춘다"},
                     ensure_ascii=False, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
