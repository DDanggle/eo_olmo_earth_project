#!/usr/bin/env python3
"""공식 Sen12 baseline 구조 이식 — `UNet3D` 와 `U-TAE`. strict deterministic 안전.

M27 근거로 **구조를 바꾸지 않는다**. 공식 config를 그대로 쓰고, 비결정적 커널 하나만
같은 수학 함수의 결정적 커널로 교체한다.

  공식 UNet3D  : strided Conv3d 로 downsample (max pooling 아님)
                 마지막 AdaptiveAvgPool3d((1,H,W)) → **mean(dim=2, keepdim=True)**
                 float64 검증: forward diff 3.331e-16, backward diff 5.551e-17
  공식 U-TAE   : str_conv_k 4 / s 2 / p 1 strided conv → strict에서 막히는 연산 없음
                 (실측: params 1,165,409 · replay max|diff| 0.0 · peak 2.86 GB)
                 encoder_widths [64,64,64,128], decoder_widths [32,32,64,128]
                 agg_mode att_group, encoder_norm group, n_head 16, d_model 256, d_k 4
                 padding_mode reflect

strict에서 추가로 발견한 것 (2026-08-26)
  `reflection_pad3d_backward_out_cuda` 도 결정적 구현이 없다. 그래서 3D 경로는 zeros padding을
  쓴다. 공식 UNet3D yaml에는 `padding_mode` 항목이 아예 없으므로 이는 **공식에 더 가까운** 선택이다.

정직하게 적어두는 한계
  - `UNet3D` 내부 채널 리스트·depth는 공식 yaml에 없고 클래스 기본값이다. 여기서는
    U-TAE와 같은 [64,64,64,128] 계열로 두고 **파라미터 수를 산출물에 기록**한다.
  - U-TAE `att_group` 집계는 Garnot & Landrieu(2021) 설계를 따르되 **원 구현과 bit 단위로
    같다는 보장은 없다.** 그래서 산출물에 `reimplementation: true` 를 박는다.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 공식 U-TAE yaml만 `padding_mode: reflect` 를 명시한다. UNet3D yaml에는 없다(클래스 기본값).
# 그리고 strict 모드에서 `reflection_pad3d_backward_out_cuda` 는 결정적 구현이 없다(실측).
# 따라서 2D(U-TAE)는 reflect, 3D(UNet3D)는 zeros 를 쓴다 — 이것이 공식에 **더 가깝다**.
PADDING_MODE_2D = "reflect"
PADDING_MODE_3D = "zeros"


def gn(c: int) -> nn.Module:
    """공식 encoder_norm='group'. 채널 수에 맞춰 그룹을 잡는다."""
    for g in (4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)
    return nn.GroupNorm(1, c)


# ────────────────────────────── 공식 UNet3D ──────────────────────────────
class ConvBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, dropout: float = 0.0):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, padding_mode=PADDING_MODE_3D),
            gn(cout), nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(cout, cout, 3, padding=1, padding_mode=PADDING_MODE_3D),
            gn(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


class OfficialUNet3D(nn.Module):
    """공식 구조: strided Conv3d downsample + transposed conv upsample + mean 시간축소.

    입력 `B,C,T,H,W` → 출력 `B,1,H,W`.
    시간축은 stride 1로 두어 T를 보존하고 공간만 내린다(공식이 2D mask를 내는 구조와 일치).
    """

    def __init__(self, in_channels: int, widths=(64, 64, 64, 128), dropout: float = 0.0):
        super().__init__()
        w = list(widths)
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        prev = in_channels
        for i, c in enumerate(w):
            self.enc.append(ConvBlock3D(prev, c, dropout))
            if i < len(w) - 1:
                # 공식과 동일하게 **strided conv** 로 내린다 (pooling 아님).
                self.down.append(nn.Conv3d(c, c, (1, 4, 4), stride=(1, 2, 2),
                                           padding=(0, 1, 1)))
            prev = c
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(len(w) - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(w[i], w[i - 1], (1, 4, 4),
                                              stride=(1, 2, 2), padding=(0, 1, 1)))
            self.dec.append(ConvBlock3D(w[i - 1] * 2, w[i - 1], dropout))
        self.head = nn.Conv2d(w[0], 1, 1)

    def forward(self, x):
        feats = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < len(self.down):
                feats.append(x)
                x = self.down[i](x)
        for up, dec, skip in zip(self.up, self.dec, reversed(feats)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-3:], mode="nearest")
            x = dec(torch.cat([x, skip], dim=1))
        # 공식 AdaptiveAvgPool3d((1,H,W)) 와 **수학적으로 동일**한 결정적 연산 (M27)
        x = x.mean(dim=2)
        return self.head(x)


# ────────────────────────────── 공식 U-TAE ──────────────────────────────
class ConvBlock2D(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, padding_mode=PADDING_MODE_2D),
            gn(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, padding_mode=PADDING_MODE_2D),
            gn(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.b(x)


class LTAE(nn.Module):
    """Lightweight Temporal Attention Encoder. 공식 n_head 16 / d_model 256 / d_k 4.

    `agg_mode='att_group'`: head를 채널 그룹에 대응시켜 그룹별 attention 가중으로 시간축을 접는다.
    """

    def __init__(self, in_ch: int, n_head: int = 16, d_model: int = 256, d_k: int = 4):
        super().__init__()
        self.n_head, self.d_k = n_head, d_k
        self.inconv = nn.Conv1d(in_ch, d_model, 1)
        self.norm_in = nn.GroupNorm(n_head, d_model)
        self.q = nn.Parameter(torch.zeros(n_head, d_k))
        nn.init.normal_(self.q, std=d_k ** -0.5)
        self.k_proj = nn.Linear(d_model, n_head * d_k)
        self.in_ch, self.d_model = in_ch, d_model

    def forward(self, x, positions):
        """x: B,C,T,H,W · positions: B,T (월 인덱스 0~11) → B,C,H,W"""
        b, c, t, h, w = x.shape
        # 시간별 전역 서술자에 sinusoidal position을 더한다
        desc = x.mean(dim=(3, 4))                                   # B,C,T
        desc = self.norm_in(self.inconv(desc)).permute(0, 2, 1)     # B,T,d_model
        pe = self._pos(positions, self.d_model, x.device, x.dtype)  # B,T,d_model
        desc = desc + pe
        k = self.k_proj(desc).view(b, t, self.n_head, self.d_k)     # B,T,H,d_k
        att = torch.einsum("bthd,hd->bht", k, self.q) / self.d_k ** 0.5
        att = torch.softmax(att, dim=-1)                            # B,H,T
        # head를 채널 그룹에 대응 (att_group)
        grp = c // self.n_head
        if grp == 0:                     # 채널이 head보다 적으면 head를 평균해 공유
            a = att.mean(1)                                         # B,T
            return torch.einsum("bcthw,bt->bchw", x, a)
        xg = x[:, :grp * self.n_head].view(b, self.n_head, grp, t, h, w)
        out = torch.einsum("bngthw,bnt->bnghw", xg, att).reshape(b, grp * self.n_head, h, w)
        if grp * self.n_head < c:        # 나머지 채널은 head 평균으로
            rest = torch.einsum("bcthw,bt->bchw", x[:, grp * self.n_head:], att.mean(1))
            out = torch.cat([out, rest], dim=1)
        return out

    @staticmethod
    def _pos(positions, d, device, dtype):
        p = positions.to(device=device, dtype=torch.float32).unsqueeze(-1)   # B,T,1
        i = torch.arange(d, device=device, dtype=torch.float32)
        ang = p / torch.pow(1000.0, 2 * torch.div(i, 2, rounding_mode="floor") / d)
        pe = torch.zeros(p.shape[0], p.shape[1], d, device=device, dtype=torch.float32)
        pe[..., 0::2] = torch.sin(ang[..., 0::2])
        pe[..., 1::2] = torch.cos(ang[..., 1::2])
        return pe.to(dtype)


class OfficialUTAE(nn.Module):
    """공식 U-TAE: 시점을 공유 encoder로 통과 → 최하단에서 LTAE로 시간 접기 → 2D decoder."""

    def __init__(self, in_channels: int, encoder_widths=(64, 64, 64, 128),
                 decoder_widths=(32, 32, 64, 128), n_head: int = 16,
                 d_model: int = 256, d_k: int = 4,
                 str_conv_k: int = 4, str_conv_s: int = 2, str_conv_p: int = 1):
        super().__init__()
        ew, dw = list(encoder_widths), list(decoder_widths)
        self.stem = ConvBlock2D(in_channels, ew[0])
        self.enc, self.down = nn.ModuleList(), nn.ModuleList()
        for i in range(1, len(ew)):
            # 공식 str_conv_k/s/p 를 그대로 쓴다
            self.down.append(nn.Conv2d(ew[i - 1], ew[i], str_conv_k, stride=str_conv_s,
                                       padding=str_conv_p, padding_mode=PADDING_MODE_2D))
            self.enc.append(ConvBlock2D(ew[i], ew[i]))
        self.ltae = LTAE(ew[-1], n_head=n_head, d_model=d_model, d_k=d_k)
        self.up, self.dec = nn.ModuleList(), nn.ModuleList()
        prev = ew[-1]
        for i in range(len(ew) - 2, -1, -1):
            self.up.append(nn.ConvTranspose2d(prev, dw[i], str_conv_k, stride=str_conv_s,
                                              padding=str_conv_p))
            self.dec.append(ConvBlock2D(dw[i] + ew[i], dw[i]))
            prev = dw[i]
        self.head = nn.Conv2d(dw[0], 1, 1)

    def forward(self, x, positions):
        b, c, t, h, w = x.shape
        z = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        z = self.stem(z)
        skips = [z]
        for dn, blk in zip(self.down, self.enc):
            z = blk(dn(z))
            skips.append(z)
        # 최하단만 시간축을 접는다. 나머지 skip은 시간 평균으로 2D화한다.
        deep = skips[-1]
        dc, dh, dw_ = deep.shape[1], deep.shape[2], deep.shape[3]
        deep = deep.view(b, t, dc, dh, dw_).permute(0, 2, 1, 3, 4)
        y = self.ltae(deep, positions)
        for i, (up, dec) in enumerate(zip(self.up, self.dec)):
            skip = skips[len(skips) - 2 - i]
            sc, sh, sw = skip.shape[1], skip.shape[2], skip.shape[3]
            skip2d = skip.view(b, t, sc, sh, sw).mean(dim=1)
            y = up(y)
            if y.shape[-2:] != skip2d.shape[-2:]:
                y = F.interpolate(y, size=skip2d.shape[-2:], mode="nearest")
            y = dec(torch.cat([y, skip2d], dim=1))
        return self.head(y)


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
