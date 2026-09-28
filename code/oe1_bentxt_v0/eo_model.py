"""OE1 single-observation native S2 adapter; no implicit downloads or fake bands.

Use official local config.json/weights.pth with from_pretrained(). Inputs are raw
L2A DN (reflectance * 10000), already aligned by the data builder to a 10 m grid.
This module neither resamples original bands nor invents missing acquisition dates.
The encoder is called directly, preserving autograd; decoder/EMA target are unused.
Default eval-mode encoder disables stochastic layers in BOTH comparison conditions.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from contextlib import nullcontext
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

S2_BANDS = ("B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquisition_day(value: str) -> date:
    """Accept real day precision or timezone-qualified acquisition timestamps."""
    if not isinstance(value, str):
        raise ValueError("Acquisition date must be an explicit ISO string")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return date.fromisoformat(value)
    if "T" not in value:
        raise ValueError("Require YYYY-MM-DD or timezone-qualified ISO datetime")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("Acquisition datetime requires timezone; no fabricated UTC")
    return stamp.astimezone(timezone.utc).date()


def _official_api():
    from olmoearth_pretrain_minimal import load_model_from_path
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.data.normalize import load_computed_config
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample, MaskValue
    return load_model_from_path, load_computed_config, Modality, MaskedOlmoEarthSample, MaskValue


def _tensor_sha(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous()
    header = json.dumps([str(data.dtype), list(data.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


class EOImageEncoder(nn.Module):
    """Returns [B,16,D] (4x4), [B,1,D] (mean), or [B,H/p*W/p,D]."""

    def __init__(self, encoder: nn.Module, *, trainable: bool = False,
                 patch_size: int = 4, pooling: str = "4x4",
                 stochastic_train_mode: bool = False, provenance: dict | None = None):
        super().__init__()
        if type(trainable) is not bool or type(stochastic_train_mode) is not bool:
            raise ValueError("Training flags must be booleans")
        if type(patch_size) is not int or not 1 <= patch_size <= 8:
            raise ValueError("patch_size must be an integer in [1,8]")
        if pooling not in ("4x4", "mean", "spatial"):
            raise ValueError("pooling must be 4x4, mean, or spatial")
        _, norm_loader, modalities, self._sample_type, self._mask_values = _official_api()
        if tuple(modalities.SENTINEL2_L2A.band_order) != S2_BANDS:
            raise RuntimeError("Official S2 band contract changed")
        norm = norm_loader()["sentinel2_l2a"]
        means = [float(norm[b]["mean"]) for b in S2_BANDS]
        stds = [float(norm[b]["std"]) for b in S2_BANDS]
        if not all(math.isfinite(x) for x in means + stds) or min(stds) <= 0:
            raise ValueError("Invalid official normalization statistics")
        self.register_buffer("band_mean", torch.tensor(means).view(1, 12, 1, 1))
        self.register_buffer("band_std", torch.tensor(stds).view(1, 12, 1, 1))
        self.encoder = encoder
        self.patch_size, self.pooling = patch_size, pooling
        self.stochastic_train_mode = stochastic_train_mode
        self.provenance = dict(provenance or {"checkpoint_loaded": False})
        self.provenance.update({
            "band_order": list(S2_BANDS), "input_units": "S2_L2A_DN_reflectance_times_10000",
            "normalization": "(DN - mean) / (4 * std) + 0.5; no clipping",
            "normalization_stats_sha256": hashlib.sha256(json.dumps(norm, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            "timestamp_encoding": "UTC calendar day, month-1, year; date-only preserves stated day precision",
            "forward_path": "loaded_model.encoder(sample, patch_size, fast_pass=True)",
            "stochastic_train_mode": stochastic_train_mode,
        })
        self.audit_counters = {"optimizer_steps_observed": 0, "encoder_update_steps": 0}
        self.set_trainable(trainable)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str | Path, **kwargs) -> "EOImageEncoder":
        """Load a local official checkpoint; never fetch network artifacts."""
        path = Path(checkpoint_dir).resolve(strict=True)
        pins = {name: sha256_file(path / name) for name in ("config.json", "weights.pth")}
        loader = _official_api()[0]
        model = loader(str(path), load_weights=True)
        if not hasattr(model, "encoder"):
            raise TypeError("Official checkpoint has no online encoder")
        # Verify bytes have not changed during deserialization.
        if pins != {name: sha256_file(path / name) for name in pins}:
            raise RuntimeError("Checkpoint changed while loading")
        source_files = {}
        for obj in (loader, type(model.encoder), _official_api()[1]):
            source = Path(inspect.getfile(obj)).resolve()
            source_files[str(source)] = sha256_file(source)
        return cls(model.encoder, provenance={"checkpoint_loaded": True,
                   "checkpoint_dir": str(path), "checkpoint_sha256": pins,
                   "official_source_sha256": source_files}, **kwargs)

    @property
    def output_dim(self) -> int:
        return int(self.encoder.embedding_size)

    def set_trainable(self, trainable: bool) -> None:
        if type(trainable) is not bool:
            raise ValueError("trainable must be boolean")
        self.encoder_trainable = trainable
        self.encoder.requires_grad_(trainable)
        for param in self.encoder.parameters():
            param.grad = None
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "encoder_trainable"):
            self.encoder.train(mode and self.encoder_trainable and self.stochastic_train_mode)
        return self

    def forward(self, images: torch.Tensor, acquisition_dates: Sequence[str], *,
                band_names: Sequence[str], input_units: str = "S2_L2A_DN_reflectance_times_10000") -> torch.Tensor:
        if input_units != "S2_L2A_DN_reflectance_times_10000":
            raise ValueError("No automatic scale conversion; pass raw L2A DN")
        if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 12:
            raise ValueError("Expected [B,12,H,W] raw actual S2 bands")
        if len(band_names) != 12 or set(band_names) != set(S2_BANDS):
            raise ValueError("All 12 actual S2 L2A bands required exactly once; no band synthesis")
        b, _, h, w = images.shape
        if b < 1 or min(h, w) < self.patch_size or h % self.patch_size or w % self.patch_size:
            raise ValueError("Nonempty spatial dimensions must be divisible by patch_size")
        if self.pooling == "4x4" and min(h, w) // self.patch_size < 4:
            raise ValueError("4x4 pooling requires at least a 4x4 source token grid")
        if isinstance(acquisition_dates, str) or len(acquisition_dates) != b:
            raise ValueError("One actual acquisition date per image required")
        days = [acquisition_day(value) for value in acquisition_dates]
        if images.device != self.band_mean.device:
            raise ValueError("Move input and encoder to the same device explicitly")
        raw = images.float()
        if not torch.isfinite(raw).all() or (raw < 0).any() or (raw > 65535).any():
            raise ValueError("Nonfinite or out-of-range raw S2 DN")
        order = torch.tensor([band_names.index(name) for name in S2_BANDS], device=raw.device)
        raw = raw.index_select(1, order)
        norm = (raw - self.band_mean) / (4 * self.band_std) + 0.5
        timestamps = torch.tensor([[[d.day, d.month - 1, d.year]] for d in days], dtype=torch.int32, device=raw.device)
        token_config = getattr(self.encoder, "tokenization_config", None)
        nsets = token_config.get_num_bandsets("sentinel2_l2a") if token_config is not None else 3
        mask = torch.full((b, h, w, 1, nsets), self._mask_values.ONLINE_ENCODER.value, dtype=torch.int32, device=raw.device)
        sample = self._sample_type(timestamps=timestamps, sentinel2_l2a=norm.permute(0, 2, 3, 1).unsqueeze(3), sentinel2_l2a_mask=mask)
        context = nullcontext() if self.encoder_trainable else torch.no_grad()
        with context:
            output = self.encoder(sample, patch_size=self.patch_size, fast_pass=True)
            tokens = output["tokens_and_masks"].sentinel2_l2a
            if tokens.ndim != 6 or tokens.shape[0] != b or tokens.shape[3] != 1:
                raise RuntimeError("Unexpected official encoder token contract")
            feature_map = tokens.mean(dim=(3, 4)).permute(0, 3, 1, 2)
            if self.pooling == "4x4":
                feature_map = F.adaptive_avg_pool2d(feature_map, (4, 4))
            elif self.pooling == "mean":
                feature_map = feature_map.mean(dim=(2, 3), keepdim=True)
            result = feature_map.flatten(2).transpose(1, 2).contiguous()
        if not torch.isfinite(result).all():
            raise RuntimeError("Nonfinite EO features")
        if self.encoder_trainable and torch.is_grad_enabled() and not result.requires_grad:
            raise RuntimeError("Encoder gradient path was detached")
        return result

    get_features = forward

    def capture_parameter_state(self) -> dict:
        """Exact whole-encoder parameter hashes; call before optimizer.step()."""
        return {name: _tensor_sha(param) for name, param in self.encoder.named_parameters()}

    def gradient_report(self) -> dict:
        report = {"parameter_tensors": 0, "trainable_tensors": 0, "with_grad": 0,
                  "nonzero_grad_tensors": 0, "finite": True, "l2_norm": 0.0}
        squared = 0.0
        for param in self.encoder.parameters():
            report["parameter_tensors"] += 1
            report["trainable_tensors"] += int(param.requires_grad)
            if param.grad is None:
                continue
            grad = param.grad.detach().double()
            report["with_grad"] += 1
            report["finite"] = report["finite"] and bool(torch.isfinite(grad).all())
            report["nonzero_grad_tensors"] += int(bool(torch.count_nonzero(grad)))
            squared += float(grad.square().sum().cpu())
        report["l2_norm"] = math.sqrt(squared) if math.isfinite(squared) else None
        return report

    def audit_after_step(self, before: dict) -> dict:
        """Call after optimizer.step(), BEFORE zero_grad; reports observed changes.

        Counters do not prove optimizer calls independently: caller must use this
        protocol once per step and preserve its batch/loss/step log separately.
        """
        after = self.capture_parameter_state()
        if set(before) != set(after):
            raise ValueError("Encoder parameter inventory changed")
        changed = [name for name in before if before[name] != after[name]]
        grad = self.gradient_report()
        if not grad["finite"]:
            raise RuntimeError("Nonfinite encoder gradients")
        if not self.encoder_trainable and (changed or grad["with_grad"]):
            raise RuntimeError("Frozen encoder parameters or gradients changed")
        self.audit_counters["optimizer_steps_observed"] += 1
        self.audit_counters["encoder_update_steps"] += int(bool(changed))
        encode = lambda mapping: hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        return {"trainable": self.encoder_trainable, "before_sha256": encode(before),
                "after_sha256": encode(after), "changed_parameter_tensors": len(changed),
                "changed_parameter_names": changed, "gradients": grad,
                "counters": dict(self.audit_counters)}
