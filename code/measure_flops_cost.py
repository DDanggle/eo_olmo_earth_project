#!/usr/bin/env python3
"""벽시계 대신 **경합 불변** 비용을 잰다 — 샘플당 forward FLOPs. GPU1 전용(측정만).

왜: `fit_plus_epoch_val_seconds`는 같은 구성(P4/tiled/small)인데 M30에서 641 s,
M37에서 866.6 s로 나왔다(+35%). 두 실행 모두 GPU1에 다른 프로젝트 작업 2개가
동시에 돌던 중이었다. 손익분기와 Pareto 판정이 이 시계 위에 얹혀 있어 무효화 위험이 있다.

FLOPs는 하드웨어 점유와 무관하다. 이걸로 amortization을 하드웨어 독립 진술로 바꾼다.

  shared-cache 총비용 = E_enc + K x H_head
  raw-model  총비용  = K x F_task
  손익분기 K* = E_enc / (F_task - H_head)   (F_task > H_head 일 때만 존재)
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
OUT = Path("/home/work/data/olmoearth/gp_official_bundle/flops_cost.json")
T, H, W = 12, 128, 128
CIN_RAW = 11          # 10밴드 + month 1채널 (timestamp parity)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("refusing to run: CUDA_VISIBLE_DEVICES must be exactly 1")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.flop_counter import FlopCounterMode
    from sen12_official_baselines import OfficialUNet3D, OfficialUTAE

    dev = torch.device("cuda")

    def count(mod, *args):
        mod = mod.to(dev).eval()
        fc = FlopCounterMode(display=False)
        with fc, torch.no_grad():
            mod(*args)
        return int(fc.get_total_flops())

    res = {"schema": "flops-cost-v1",
           "why": "벽시계는 GPU 경합으로 같은 구성에서 641 s vs 866.6 s (+35%) 편차를 보였다",
           "shape": {"T": T, "H": H, "W": W}, "per_sample_forward_flops": {}}

    x_raw = torch.zeros(1, CIN_RAW, T, H, W, device=dev)
    res["per_sample_forward_flops"]["P2_official_unet3d"] = count(
        OfficialUNet3D(in_channels=CIN_RAW), x_raw)
    pos = torch.zeros(1, T, dtype=torch.long, device=dev)
    res["per_sample_forward_flops"]["P3_official_utae"] = count(
        OfficialUTAE(in_channels=CIN_RAW), x_raw, pos)

    # decoder들은 캐시(768x32x32)를 입력으로 받는다
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pilot", Path(__file__).parent / "pilot_sen12_gp_heads.py")

    # pilot 내부 클래스는 main() 안에 있어 임포트가 안 된다. 동일 정의를 여기 복제하지 않고
    # 구조만 재현한다 — 파라미터 수를 대조해 동일성을 확인한다.
    def conv_bn(i, o, k=3):
        # 원본 pilot의 conv_bn은 **conv 두 개**다 (처음에 하나로 복제해 파라미터가
        # 191,169로 나왔고 실제 237,537과 어긋났다).
        return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, k, padding=k // 2), nn.BatchNorm2d(o),
                             nn.ReLU(inplace=True))

    class Small(nn.Module):
        def __init__(self, cin=768, base=128):
            super().__init__()
            self.proj = nn.Sequential(nn.Conv2d(cin, base, 1), nn.BatchNorm2d(base),
                                      nn.ReLU(inplace=True))
            self.u1, self.u2 = conv_bn(base, base // 2), conv_bn(base // 2, base // 4)
            self.head = nn.Conv2d(base // 4, 1, 1)

        def forward(self, x):
            x = self.proj(x)
            x = self.u1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            x = self.u2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            return F.interpolate(self.head(x), size=(128, 128), mode="bilinear",
                                 align_corners=False)

    class Big(nn.Module):
        def __init__(self, cin=768, base=256):
            super().__init__()

            def blk(a, b):
                return nn.Sequential(
                    nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                    nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
            self.proj = blk(cin, base); self.d1 = blk(base, base // 2)
            self.d2 = blk(base // 2, base // 4); self.refine = blk(base // 4, base // 4)
            self.head = nn.Conv2d(base // 4, 1, 1)

        def forward(self, x):
            x = self.proj(x)
            x = self.d1(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            x = self.d2(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
            x = self.refine(x)
            return F.interpolate(self.head(x), size=(128, 128), mode="bilinear",
                                 align_corners=False)

    z = torch.zeros(1, 768, 32, 32, device=dev)
    small, big = Small(), Big()
    res["param_check"] = {
        "small_decoder": sum(p.numel() for p in small.parameters()),
        "big_decoder": sum(p.numel() for p in big.parameters()),
        "expected_small": 237537, "expected_big": 2989121}
    res["per_sample_forward_flops"]["P4_small_decoder"] = count(small, z)
    res["per_sample_forward_flops"]["P4c_large_decoder"] = count(big, z)

    # OlmoEarth 인코더 — 실제 캐시 생성 경로와 동일하게 128 단일 패스로 잰다
    from olmoearth_pretrain_minimal import ModelID
    from rslearn.models.olmoearth_pretrain.model import MaskValue, OlmoEarth
    from rslearn.train.model_context import ModelContext, RasterImage
    from datetime import datetime, timedelta
    enc = OlmoEarth(patch_size=4, model_id=ModelID.OLMOEARTH_V1_BASE,
                    token_pooling=True, use_legacy_timestamps=False,
                    normalize=True, autocast_dtype="bfloat16").to(dev).eval()
    res["encoder_params"] = sum(p.numel() for p in enc.parameters())
    img = torch.zeros(12, T, H, W, device=dev)
    ts = [datetime(2019, 1, 1) + timedelta(days=30 * i) for i in range(T)]
    d = {"sentinel2_l2a": RasterImage(image=img, timestamps=[(t, t) for t in ts])}
    enc.normalizer(d, {})
    ctx = ModelContext(inputs=[d], metadatas=[])
    sample, present, _ = enc._prepare_modality_inputs(ctx)
    sample.sentinel2_l2a_mask[..., 2] = MaskValue.MISSING.value
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        enc.model(sample, fast_pass=False, patch_size=4)
    res["per_sample_forward_flops"]["olmoearth_encoder_1x128"] = int(fc.get_total_flops())

    # ── 손익분기 ──
    # 처음에 "인코더 1 forward vs task 모델 1 forward"로 비교해 K≈98이 나왔는데 **틀렸다**.
    # 인코더는 샘플당 **1회**만 돌지만 task 모델은 40 epoch을 forward+backward로 돈다.
    # 그 배율을 넣지 않으면 인코더 비용이 100배 과대평가된다.
    f = res["per_sample_forward_flops"]
    N_CACHE, N_TRAIN, EPOCHS, BWD = 6834, 5542, 40, 3.0
    train_mult = N_TRAIN * EPOCHS * BWD          # forward-equivalent 횟수
    E_total = f["olmoearth_encoder_1x128"] * N_CACHE
    res["cost_model"] = {
        "n_cached_samples": N_CACHE, "n_train_samples": N_TRAIN, "epochs": EPOCHS,
        "backward_multiplier": BWD,
        "note": "backward를 forward의 2배로 잡아 fwd+bwd = 3x forward로 계산한다",
        "encoder_total_flops_one_time": E_total,
        "train_forward_equivalents": train_mult}
    out = {}
    for task_name in ("P2_official_unet3d", "P3_official_utae"):
        for head_name in ("P4_small_decoder", "P4c_large_decoder"):
            Fk = f[task_name] * train_mult
            Hk = f[head_name] * train_mult
            kstar = E_total / (Fk - Hk) if Fk > Hk else None
            out[f"{head_name}_vs_{task_name}"] = {
                "encoder_total_flops": E_total,
                "head_train_flops_per_task": int(Hk),
                "task_model_train_flops_per_task": int(Fk),
                "breakeven_K_real": round(kstar, 4) if kstar else None,
                "breakeven_K_integer": (int(kstar) + 1 if kstar else None),
                "note": ("공유 캐시가 더 쌈" if kstar else "head가 task model보다 비싸 손익분기 없음")}
    res["amortization"] = out
    # 벽시계와의 정합성 대조 — FLOPs 모델이 실측 시간과 같은 자릿수인지 확인한다
    res["sanity_vs_wallclock"] = {
        "encoder_total_flops": E_total,
        "p2_train_total_flops": int(f["P2_official_unet3d"] * train_mult),
        "measured_seconds": {"cache_extraction": 1130, "p2_training": 1491},
        "implied_tflops_per_s": {
            "cache": round(E_total / 1130 / 1e12, 1),
            "p2": round(f["P2_official_unet3d"] * train_mult / 1491 / 1e12, 1)},
        "interpretation": "두 값이 같은 자릿수면 FLOPs 모델이 실측과 모순되지 않는다"}
    res["caveat"] = ("forward FLOPs만이다. backward는 대략 2배이지만 arm마다 비율이 같지 않을 수 있다. "
                     "메모리 대역폭·커널 효율은 반영되지 않는다. 실제 시간은 아니다.")

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
