#!/usr/bin/env python3
"""C2-A: v1.2의 mask 소비 경로를 실측해 M1을 방어한다.

가설 (사용자 제안, 최신 GitHub 코드 독해 기반 — 미검증):
  v1.2는 12밴드 단일 group이므로 encoder가 mask[..., 0]만 읽고 slice 1·2는 무시한다.

이 스크립트는 그 가설을 **감사 환경(rslearn 0.1.13 / olmoearth_pretrain_minimal 0.0.6)** 에서
행동으로 확인한다. 최신 GitHub 코드가 아니라 실제 M1을 만든 코드가 근거여야 한다.

성공 조건 (사전 등록):
  G1. encoder 출력 S(band set) 축: v1=3, v1.2=1
  G2. rslearn이 만드는 입력 mask S축: 두 릴리스 모두 3
  G3. v1.2에서 mask slice 1·2를 MISSING으로 바꿔도 encoder 출력 토큰이 **byte-identical**
  G4. v1.2에서 mask slice 0을 MISSING으로 바꾸면 출력이 **달라진다**
  G5. v1에서는 slice 2를 바꾸면 출력이 달라진다 (v1은 실제로 3 set을 쓴다)

G1~G5 전부 통과하면: 두 릴리스의 출력은 pooling 후 동일한 공간 격자에 768-d 하나이고,
v1.2에서는 partial-group missingness를 표현할 수단이 아예 없다는 것이 실측으로 확정된다.

부가 관찰: rslearn의 fast_pass는 **입력** mask 전체(3 slice)를 보고 결정되지만,
pooling 시 masked-average는 **출력** mask(v1.2는 S=1)를 쓴다. 따라서 v1.2에서 band_set 2를
MISSING으로 표시하면 토큰화에는 아무 영향이 없으면서 pooling 코드 경로만 바뀐다.
이것도 측정한다 (G6).
"""
from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path

OUT = Path("/home/work/data/olmoearth/mask_path_c2a")
SNAPS = {
    "v1": "/home/work/data/.cache/huggingface/hub/models--allenai--OlmoEarth-v1-Base/snapshots",
    "v1_2": "/home/work/data/.cache/huggingface/hub/models--allenai--OlmoEarth-v1_2-Base/snapshots",
}
PATCH = 4
H = W = 32          # patch 4 → 8x8 patch grid
T = 2
SEED = 20260824


def build_sample(mask_missing_slices, n_band_sets, n_channels, device, MaskedOlmoEarthSample, MaskValue):
    """결정적 합성 입력. mask_missing_slices에 든 band_set index만 MISSING으로 채운다."""
    import torch

    g = torch.Generator(device="cpu").manual_seed(SEED)
    img = torch.rand((1, H, W, T, n_channels), generator=g, dtype=torch.float32) * 3000.0
    mask = torch.full(
        (1, H, W, T, n_band_sets), MaskValue.ONLINE_ENCODER.value, dtype=torch.int32
    )
    # slice 전체를 MISSING으로 두면 남는 토큰이 0이 되어 encoder가 assertion으로 죽는다.
    # 따라서 공간의 위쪽 절반만 MISSING으로 표시한다. 효과 유무 판정에는 충분하다.
    for idx in mask_missing_slices:
        mask[:, : H // 2, :, :, idx] = MaskValue.MISSING.value
    ts = torch.zeros((1, T, 3), dtype=torch.int32)
    for i, d in enumerate([datetime(2024, 3, 15), datetime(2024, 7, 15)][:T]):
        ts[0, i, 0], ts[0, i, 1], ts[0, i, 2] = d.day, d.month - 1, d.year
    return MaskedOlmoEarthSample(
        sentinel2_l2a=img.to(device),
        sentinel2_l2a_mask=mask.to(device),
        timestamps=ts.to(device),
    )


def main() -> None:
    import torch
    from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth

    mod = __import__("olmoearth_pretrain_minimal", fromlist=["*"])
    consts = __import__(
        "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants",
        fromlist=["Modality"],
    )
    Modality = consts.Modality
    fv = __import__(
        "olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit",
        fromlist=["MaskedOlmoEarthSample"],
    )
    MaskedOlmoEarthSample = fv.MaskedOlmoEarthSample
    ModelID = getattr(mod, "ModelID", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = Modality.get("sentinel2_l2a")
    rslearn_n_sets = len(m.band_sets)
    n_channels = sum(len(bs.bands) for bs in m.band_sets)

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "mask-path-c2a-v1",
        "environment": {
            "rslearn": __import__("importlib.metadata", fromlist=["version"]).version("rslearn"),
            "olmoearth_pretrain_minimal": __import__(
                "importlib.metadata", fromlist=["version"]
            ).version("olmoearth-pretrain-minimal"),
            "device": str(device),
        },
        "input_contract": {
            "rslearn_num_band_sets": rslearn_n_sets,
            "num_channels": n_channels,
            "note": "rslearn은 릴리스와 무관하게 이 수만큼 mask slice를 만든다",
        },
        "releases": {},
    }

    for tag, snap_root in SNAPS.items():
        snap = sorted(glob.glob(f"{snap_root}/*"))[0]
        entry = {"snapshot": snap}
        wrapper = None
        id_name = {"v1": "OLMOEARTH_V1_BASE", "v1_2": "OLMOEARTH_V1_2_BASE"}[tag]
        if ModelID is not None and id_name in [x.name for x in ModelID]:
            try:
                wrapper = OlmoEarth(patch_size=PATCH, model_id=id_name)
                entry["loaded_via"] = f"model_id={id_name}"
            except Exception as exc:  # noqa: BLE001
                entry["model_id_error"] = repr(exc)[:200]
        if wrapper is None:
            wrapper = OlmoEarth(patch_size=PATCH, model_path=snap)
            entry["loaded_via"] = "model_path"

        core = wrapper.model.to(device).eval()
        tc = getattr(core, "tokenization_config", None)
        entry["model_num_bandsets"] = (
            tc.get_num_bandsets("sentinel2_l2a") if tc is not None else None
        )

        def run(missing):
            sample = build_sample(
                missing, rslearn_n_sets, n_channels, device,
                MaskedOlmoEarthSample, MaskValue,
            )
            # rslearn의 fast_pass 결정 규칙을 그대로 재현한다 (입력 mask 전체를 본다).
            fast_pass = not bool(
                torch.any(sample.sentinel2_l2a_mask == MaskValue.MISSING.value)
            )
            with torch.no_grad():
                out = core(sample, fast_pass=fast_pass, patch_size=PATCH)
            tm = out["tokens_and_masks"]
            tok = getattr(tm, "sentinel2_l2a")
            omask = getattr(tm, "sentinel2_l2a_mask")
            return {
                "fast_pass": fast_pass,
                "token_shape": list(tok.shape),
                "output_mask_shape": list(omask.shape),
                "output_mask_has_missing": bool(
                    torch.any(omask == MaskValue.MISSING.value)
                ),
                "_tok": tok.detach().float().cpu(),
            }

        base = run([])
        entry["baseline"] = {k: v for k, v in base.items() if not k.startswith("_")}
        entry["encoder_output_band_set_axis"] = base["token_shape"][4]
        print(f"[{tag}] model_num_bandsets={entry['model_num_bandsets']}"
              f" token_shape={base['token_shape']}"
              f" output_mask_shape={base['output_mask_shape']}", flush=True)

        arms = {}
        for name, missing in [
            ("mask_slice_1_2_missing", [1, 2]),
            ("mask_slice_0_missing", [0]),
            ("mask_slice_2_missing", [2]),
        ]:
            try:
                r = run(missing)
            except Exception as exc:  # noqa: BLE001
                arms[name] = {"error": repr(exc)[:300]}
                print(f"[{tag}] {name}: ERROR {repr(exc)[:120]}", flush=True)
                continue
            d = (r["_tok"] - base["_tok"])
            arms[name] = {
                k: v for k, v in r.items() if not k.startswith("_")
            } | {
                "byte_identical_to_baseline": bool(torch.equal(r["_tok"], base["_tok"])),
                "max_abs_diff": float(d.abs().max()),
                "mean_abs_diff": float(d.abs().mean()),
            }
            print(f"[{tag}] {name}: identical={arms[name]['byte_identical_to_baseline']}"
                  f" max_abs_diff={arms[name]['max_abs_diff']:.6g}"
                  f" fast_pass={arms[name]['fast_pass']}", flush=True)
        entry["arms"] = arms
        result["releases"][tag] = entry
        del core, wrapper
        torch.cuda.empty_cache()

    def ident(entry, arm):
        """arm의 byte_identical 값. 실패한 arm은 None을 돌려 게이트가 조용히 통과하지 않게 한다."""
        return entry["arms"].get(arm, {}).get("byte_identical_to_baseline")

    v1, v12 = result["releases"]["v1"], result["releases"]["v1_2"]
    gates = {
        "G1_encoder_S_axis_v1_3_v12_1": (
            v1["encoder_output_band_set_axis"] == 3
            and v12["encoder_output_band_set_axis"] == 1
        ),
        "G2_input_mask_S_always_3": rslearn_n_sets == 3,
        "G3_v12_slice_1_2_no_effect": ident(v12, "mask_slice_1_2_missing") is True,
        "G4_v12_slice_0_has_effect": ident(v12, "mask_slice_0_missing") is False,
        "G5_v1_slice_2_has_effect": ident(v1, "mask_slice_2_missing") is False,
        "G6_v12_slice_2_flips_fast_pass_silently": (
            v12["arms"].get("mask_slice_2_missing", {}).get("fast_pass") is False
            and ident(v12, "mask_slice_2_missing") is True
        ),
    }
    result["gates"] = gates
    result["all_gates_pass"] = all(gates.values())
    (OUT / "mask_path_c2a.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("DONE")


if __name__ == "__main__":
    main()
