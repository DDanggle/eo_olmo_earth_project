#!/usr/bin/env python3
"""변화 슬라이더 — 월별 시계열의 각 달을 **정확히 12개월 전 같은 달**과 비교한 Δz 토큰 맵.

간격 규율을 지킨다: 인접한 달끼리 비교하면 계절 위상차가 Δz 에 실린다(v7.10 이 죽은 자리). 그래서
쌍은 항상 (t−12개월 → t) 이고, 둘 다 프레임이 있는 달만 만든다. 색 척도와 문턱은 **봉인된 40 m 계약의
p99**(scan_p4.json) 를 그대로 쓴다 — 슬라이더의 색이 지도 색계급과 같은 자를 쓰게 하려는 것이다.

이것은 **보는 용도**다. 궤도가 달마다 다를 수 있어(보는 용도로 궤도 무관 선택) 채점에 쓰지 않는다.
계약과 다른 궤도 쌍이면 JSON 에 `orbit_match: false` 로 적는다.

산출물: <WEB_DATA_ROOT>/series/<oreum_id>/<YYYY-MM>_delta.png, <oreum_id>.json 의 frames[key].delta 갱신
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

from jeju_paths import ARTIFACT_ROOT, CACHE_ROOT, WEB_DATA_ROOT, display_path
from oreum_export_web import delta_png
from oreum_scan import MODALITY, token_valid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="jeju_v8")
    ap.add_argument("--patch", type=int, default=4)
    a = ap.parse_args()
    import torch
    from olmoearth_pretrain.data.constants import Modality
    from olmoearth_pretrain.data.normalize import Normalizer, Strategy
    from rslearn.models.olmoearth_pretrain.model import ModelID, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage
    from PIL import Image

    scan = json.loads((ARTIFACT_ROOT / f"results/{a.contract}_scan_p4.json").read_text())
    thr = scan["threshold"]["value"]
    vmax = float(np.percentile([s["event"]["median_delta"] for s in scan["sites"].values()
                                if s["event"]["median_delta"] is not None], 99)) * 3
    series_root = WEB_DATA_ROOT / "series"
    cache_root = CACHE_ROOT / "jeju_v8/series"
    index = json.loads((series_root / "index.json").read_text())

    spec = Modality.get(MODALITY); nz = Normalizer(Strategy.COMPUTED, std_multiplier=2)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OlmoEarth(patch_size=a.patch, model_id=ModelID.OLMOEARTH_V1_BASE, token_pooling=True,
                      use_legacy_timestamps=False, autocast_dtype=None).to(dev).eval()

    def embed(cube, t):
        x = np.transpose(np.nan_to_num(cube, nan=0.0), (1, 2, 0))[None].astype("float32")
        xn = np.transpose(nz.normalize(spec, x), (3, 0, 1, 2)).astype("float32")
        inp = {MODALITY: RasterImage(image=torch.from_numpy(np.ascontiguousarray(xn)).to(dev), timestamps=[(t, t)])}
        sample, _, _ = model._prepare_modality_inputs(ModelContext(inputs=[inp], metadatas=[]))
        with torch.no_grad():
            tm = model.model(sample, fast_pass=False, patch_size=a.patch)["tokens_and_masks"]
            z = getattr(tm, MODALITY); m = (getattr(tm, f"{MODALITY}_mask") != 2).unsqueeze(-1)
            f = ((z * m).sum(dim=(3, 4)) / m.sum(dim=(3, 4)).clamp(min=1))[0]
        return f.permute(2, 0, 1).float()

    def shift12(key):
        y, m = map(int, key.split("-")); return f"{y-1}-{m:02d}"

    total = 0
    for oid in index["oreum_ids"]:
        meta_p = series_root / f"{oid}.json"; meta = json.loads(meta_p.read_text())
        z, tv = {}, {}
        for key, rec in meta["frames"].items():
            if not rec or not rec.get("frame"):
                continue
            d = np.load(cache_root / oid / f"{key}.npz", allow_pickle=True)
            z[key] = embed(d["cube"], datetime.fromisoformat(str(d["date"])))
            tv[key] = token_valid(d["scl"], a.patch)
        n = 0
        for key in list(z):
            prev = shift12(key)
            if prev not in z:
                continue
            za, zb = z[prev], z[key]
            dl = (1 - (za * zb).sum(0) / (za.norm(dim=0).clamp(min=1e-8) * zb.norm(dim=0).clamp(min=1e-8))).cpu().numpy()
            v = tv[prev] & tv[key]
            Image.fromarray(delta_png(dl, v, thr, vmax), "RGBA").resize((256, 256), Image.NEAREST).save(series_root / oid / f"{key}_delta.png")
            flag = float((dl[v] > thr).mean()) if v.any() else None
            meta["frames"][key]["delta"] = {"vs": prev, "frame": f"{key}_delta.png", "valid_frac": round(float(v.mean()), 3),
                                            "flag_frac": None if flag is None else round(flag, 4),
                                            "orbit_match": meta["frames"][prev].get("orbit") == meta["frames"][key].get("orbit")}
            n += 1
        meta["delta_rule"] = {"pair": "t-12개월 → t", "threshold_p99_from": f"{a.contract}_scan_p4.json", "threshold": thr,
                              "note": "보는 용도. 궤도가 다를 수 있어 채점에 쓰지 않는다."}
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        total += n
        print(f"  {oid} {meta['name']}: Δz 프레임 {n}개", flush=True)
    print(f"총 {total}개 → {display_path(series_root)}")


if __name__ == "__main__":
    main()
