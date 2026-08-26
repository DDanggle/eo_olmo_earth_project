#!/usr/bin/env python3
"""P4가 받는 시간 정보와 P2/P3가 받는 시간 정보가 **실제로 다른지** 확인한다. GPU1.

주장: P4는 OlmoEarth wrapper를 통해 exact timestamp를 받고, P2/P3는 month/11 한 채널만
받는다 — 그래서 비교가 오염된다.

확인 방법: 같은 큐브를 (a) 원래 timestamp, (b) 월만 같고 일(day)을 1일로 바꾼 timestamp로
인코딩해 임베딩이 달라지는지 본다. 달라지면 wrapper가 **월보다 세밀한 정보**를 쓰는 것이고
비대칭이 실재한다. 같으면 wrapper도 월 해상도이므로 비대칭은 없다.
"""
from __future__ import annotations
import json, os
from datetime import datetime
from pathlib import Path

OUT = Path("/home/work/data/olmoearth/gp_official_bundle/timestamp_asymmetry.json")


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("refusing to run: CUDA_VISIBLE_DEVICES must be exactly 1")
    import numpy as np, torch, xarray as xr
    from olmoearth_pretrain_minimal import ModelID
    from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "efc", Path(__file__).parent / "extract_sen12_fold_cache.py")
    efc = importlib.util.module_from_spec(spec); spec.loader.exec_module(efc)
    BANDS = efc.MODEL_BANDS

    contract = Path("/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl")
    recs = {}
    for line in contract.read_text(encoding="utf-8").splitlines():
        if line:
            r = json.loads(line); recs[r["sample_id"]] = r
    sids = [s for s in sorted(recs) if s.startswith("chimanimani_s2_")][:5]

    dev = torch.device("cuda")
    enc = OlmoEarth(patch_size=4, model_id=ModelID.OLMOEARTH_V1_BASE,
                    token_pooling=True, use_legacy_timestamps=False,
                    normalize=True, autocast_dtype="bfloat16").to(dev).eval()

    def embed(cube, times):
        image = torch.from_numpy(cube).to(dev)
        d = {"sentinel2_l2a": RasterImage(image=image,
                                          timestamps=[(t, t) for t in times])}
        enc.normalizer(d, {})
        sample, present, _ = enc._prepare_modality_inputs(ModelContext(inputs=[d], metadatas=[]))
        sample.sentinel2_l2a_mask[..., 2] = MaskValue.MISSING.value
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = enc.model(sample, fast_pass=False, patch_size=4)
            tm = o["tokens_and_masks"]
            m = (tm.sentinel2_l2a_mask != MaskValue.MISSING.value).unsqueeze(-1)
            return ((tm.sentinel2_l2a * m).sum(dim=(3, 4))
                    / m.sum(dim=(3, 4)).clamp(min=1))[0].float().cpu().numpy()

    rows = []
    root = Path("/home/work/data/sen12landslides/extracted")
    for sid in sids:
        rec = recs[sid]
        idx = efc.select_timestep_indices(rec)
        with xr.open_dataset(root / rec["file"], decode_times=True, cache=False) as ds:
            bands = [np.asarray(ds[b].values[idx], dtype="float32") if b in ds
                     else np.zeros((len(idx), 128, 128), dtype="float32") for b in BANDS]
            cube = np.stack(bands, axis=0)[:, :, :64, :64]
            times = [datetime.fromisoformat(str(np.datetime_as_string(t, unit="s")))
                     for t in np.asarray(ds["time"].values)[idx]]
        # (b) **월은 그대로 두고 날짜만 하루 이동**한다.
        # 처음엔 전부 (그 달 1일)로 뭉갰는데 같은 달에 속한 시점들이 timestamp 중복이 되어
        # wrapper가 거부했다 — 그 사실 자체가 "여러 시점이 한 달 안에 있다"는 증거다.
        # 하루 이동은 월 채널(month/11)을 거의 바꾸지 않으므로, 임베딩이 바뀌면
        # wrapper가 월보다 세밀한 정보를 쓴다는 뜻이다.
        from datetime import timedelta

        def shift_keep_month(ts):
            """월을 **바꾸지 않고** 날짜만 옮긴다.
            처음엔 무조건 +1일을 썼는데 일부 시점의 월이 넘어가 버려서
            차이가 월 때문인지 일 때문인지 구분할 수 없었다."""
            out, used = [], set()
            for x in ts:
                for delta in (1, -1, 2, -2, 3, -3):
                    y = x + timedelta(days=delta)
                    if y.month == x.month and y.year == x.year and y not in used:
                        out.append(y); used.add(y); break
                else:
                    out.append(x); used.add(x)
            return out

        coarse = shift_keep_month(times)
        same_month = len(times) - len({(t.year, t.month) for t in times})
        a, b = embed(cube, times), embed(cube, coarse)
        denom = float(np.linalg.norm(a)) + 1e-12
        rows.append({"sample_id": sid,
                     "n_timesteps": len(times),
                     "timesteps_sharing_a_month": same_month,
                     "month_channel_changed": bool(
                         [t.month for t in times] != [c.month for c in coarse]),
                     "days_distinct": sorted({t.day for t in times}),
                     "max_abs_diff": float(np.abs(a - b).max()),
                     "relative_frobenius_diff": float(np.linalg.norm(a - b) / denom),
                     "identical": bool(np.array_equal(a, b))})
        print(rows[-1], flush=True)

    ident = all(r["identical"] for r in rows)
    out = {"schema": "timestamp-asymmetry-v1", "n_samples": len(rows), "rows": rows,
           "all_identical": ident,
           "verdict": ("wrapper는 날짜 이동에 둔감 — 비대칭 근거 약함" if ident
                       else "wrapper가 날짜 수준 변화에 반응 — 월 채널만 주는 P2/P3와 비대칭 실재"),
           "method": "같은 큐브를 원래 timestamp와 +1일 이동 timestamp로 각각 인코딩해 비교"}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
