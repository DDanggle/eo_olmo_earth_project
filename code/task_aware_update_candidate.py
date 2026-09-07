"""Untrained engineering candidate, NOT a replacement for registered T1 v0.

State and observation must already be aligned to a common coordinate system
using source-only statistics/projections. This module does not make a full-
window encoder causal, infer quality, calibrate uncertainty, or promise gains.
Use actual past-only deltas and validity metadata, not future-selected scenes.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TaskAwareResidualUpdate(nn.Module):
    """Small spatial residual. Starts at exact no-update, retains chronological input.

    Call once per acquisition (not once per unordered pair). Quality is an
    observed reliability weight, not a learned hazard probability. Invalid
    observations can be NaN; they are masked before arithmetic. State must be finite.
    A zero validity mask enforces hold-state, which is an engineering guarantee,
    not a claim that holding stale state is accurate in a changing world.
    """
    def __init__(self, channels: int = 768, hidden: int = 64):
        super().__init__()
        if channels < 1 or hidden < 1:
            raise ValueError("positive channel counts required")
        self.channels = channels
        self.project = nn.Conv2d(3 * channels + 2, hidden, 1)
        self.spatial = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.delta = nn.Conv2d(hidden, channels, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, state, observation, elapsed_days, quality, valid):
        if state.ndim != 4 or observation.shape != state.shape or state.shape[1] != self.channels:
            raise ValueError("state/observation must have the same N,C,H,W shape")
        target_shape = (state.shape[0], 1, state.shape[2], state.shape[3])
        if quality.shape != target_shape or valid.shape != target_shape or valid.dtype != torch.bool:
            raise ValueError("quality and boolean validity must have shape N,1,H,W")
        if elapsed_days.shape != (state.shape[0],):
            raise ValueError("elapsed_days must have shape N")
        if not torch.isfinite(state).all() or not torch.isfinite(quality).all():
            raise ValueError("state and quality must be finite")
        if ((quality < 0) | (quality > 1)).any():
            raise ValueError("quality must lie in [0,1]")
        if not torch.isfinite(elapsed_days).all() or (elapsed_days < 0).any():
            raise ValueError("elapsed_days must be finite and nonnegative")
        mask = valid & (quality > 0)
        if not torch.isfinite(observation.masked_select(mask.expand_as(observation))).all():
            raise ValueError("valid observation values must be finite")
        observed = torch.where(mask, observation, state)
        dt = torch.log1p(elapsed_days).view(-1, 1, 1, 1).expand(target_shape)
        x = torch.cat([state, observed, observed - state, dt, quality], dim=1)
        residual = self.delta(F.gelu(self.spatial(F.gelu(self.project(x)))))
        return state + mask.to(state.dtype) * quality * residual


def preservation_loss(student, teacher, frozen_readouts, *, feature_weight=1.0, task_weight=1.0):
    """Normalized feature MSE + average Bernoulli KL in frozen task-output space.

    Readouts include the SOURCE normalization and return binary logits. They
    must be frozen and in eval mode; gradients pass THROUGH them to the student.
    Teacher is detached. No query labels are accepted. This is a baseline loss,
    not a novel distillation objective or proof of generalization to unseen tasks.
    Prefix-specific semantics need dated labels before adding supervised losses.
    """
    if student.shape != teacher.shape or not frozen_readouts:
        raise ValueError("matched features and at least one frozen readout required")
    if feature_weight < 0 or task_weight < 0 or feature_weight + task_weight <= 0:
        raise ValueError("nonnegative loss weights, at least one positive, required")
    reference = teacher.detach()
    feature = F.mse_loss(student, reference)
    losses = []
    for head in frozen_readouts:
        if any(m.training for m in head.modules()) or any(p.requires_grad for p in head.parameters()):
            raise ValueError("task readouts must be recursively eval and frozen")
        student_logits = head(student)
        with torch.no_grad():
            teacher_logits = head(reference)
            probabilities = torch.sigmoid(teacher_logits)
            log_p = F.logsigmoid(teacher_logits)
            log_not_p = F.logsigmoid(-teacher_logits)
        kl = probabilities * (log_p - F.logsigmoid(student_logits))
        kl += (1 - probabilities) * (log_not_p - F.logsigmoid(-student_logits))
        losses.append(kl.mean())
    task = torch.stack(losses).mean()
    return {"loss": feature_weight * feature + task_weight * task,
            "feature_mse": feature, "task_kl": task}
