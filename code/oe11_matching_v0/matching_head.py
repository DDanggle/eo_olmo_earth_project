"""Small query-conditioned object matcher. CPU wiring, not a validated EO method.

Inputs are features, not class IDs or gold masks. Date averaging and mask-pooled
object extraction remain the existing OE10 features() implementation's job.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class RoleObjectMatchingHead(nn.Module):
    """Each query location selects support objects separately for the two roles.

    Query: [B,H,W,D]. Positive/counter: [B,K,D]. Text: [B,2,L], with
    role order positive then counterexample. No object index embeddings are used.
    A shared role scorer plus a query-only background score supports both roles
    being absent. Swapping objects/text exchanges role probabilities rather than
    complementing them; their sum is at most one.
    """
    def __init__(self, dimension, text_dimension, *, attention_dimension=64,
                 hidden_dimension=128, output_size=(128, 128)):
        super().__init__()
        for value in (dimension, text_dimension, attention_dimension, hidden_dimension):
            if type(value) is not int or value < 1:
                raise ValueError('Positive integer feature dimensions required')
        if len(output_size) != 2 or any(type(x) is not int or x < 1 for x in output_size):
            raise ValueError('Positive output height/width required')
        self.dimension = dimension
        self.text_dimension = text_dimension
        self.attention_dimension = attention_dimension
        self.hidden_dimension = hidden_dimension
        self.output_size = tuple(output_size)
        self.visual_norm = nn.LayerNorm(dimension)
        self.text_norm = nn.LayerNorm(text_dimension)
        self.text_project = nn.Linear(text_dimension, dimension, bias=False)
        self.query_key = nn.Linear(dimension, attention_dimension, bias=False)
        self.object_key = nn.Linear(dimension, attention_dimension, bias=False)
        self.text_key = nn.Linear(dimension, attention_dimension, bias=False)
        self.scorer = nn.Sequential(nn.Linear(4 * dimension, hidden_dimension),
                                    nn.GELU(), nn.Linear(hidden_dimension, 1, bias=False))
        self.background_scorer = nn.Sequential(nn.Linear(dimension, hidden_dimension),
                                               nn.GELU(), nn.Linear(hidden_dimension, 1))

    def configuration(self):
        return dict(dimension=self.dimension, text_dimension=self.text_dimension,
                    attention_dimension=self.attention_dimension,
                    hidden_dimension=self.hidden_dimension,
                    output_size=list(self.output_size))

    def forward(self, query, positive, counterexample, role_text, *,
                support_valid=None, return_details=False):
        if query.ndim != 4 or query.shape[-1] != self.dimension:
            raise ValueError('Query must be [B,H,W,D]')
        b, h, w, d = query.shape
        if min(b, h, w) < 1:
            raise ValueError('Empty query dimension')
        if (positive.ndim != 3 or positive.shape != counterexample.shape or
                positive.shape[0] != b or positive.shape[2] != d or
                positive.shape[1] not in (1, 2, 4, 8)):
            raise ValueError('Both support roles must be [B,K,D], K=1/2/4/8')
        if role_text.shape != (b, 2, self.text_dimension):
            raise ValueError('Role text must be [B,2,L] in positive/counter order')
        for tensor in (query, positive, counterexample, role_text):
            if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Finite floating point feature tensors required')
            if tensor.device != query.device or tensor.dtype != query.dtype:
                raise ValueError('Feature device and dtype must match')
        k = positive.shape[1]
        if support_valid is None:
            support_valid = torch.ones((b, 2, k), dtype=torch.bool, device=query.device)
        if (support_valid.shape != (b, 2, k) or support_valid.dtype != torch.bool or
                support_valid.device != query.device):
            raise ValueError('Support validity must be bool [B,2,K] on feature device')
        if not bool(support_valid.any(-1).all()):
            raise ValueError('Each support role needs a valid object')
        q = self.visual_norm(query).reshape(b, h * w, d)
        objects = self.visual_norm(torch.stack((positive, counterexample), dim=1))
        text = self.text_project(self.text_norm(role_text))
        # Text changes the query BEFORE object-wise dot products; adding one
        # constant per role after these products would cancel inside softmax.
        keys = self.query_key(q)[:, :, None, :] + self.text_key(text)[:, None, :, :]
        similarity = torch.einsum('bprm,brkm->bprk', keys, self.object_key(objects))
        similarity = similarity / math.sqrt(self.attention_dimension)
        weights = similarity.masked_fill(~support_valid[:, None], -torch.inf).softmax(-1)
        matched = torch.einsum('bprk,brkd->bprd', weights, objects)
        q_roles = q[:, :, None, :].expand(-1, -1, 2, -1)
        text_roles = text[:, None, :, :].expand(-1, h * w, -1, -1)
        # Direct image-text interaction preserves a text route even at K=1,
        # where attention weights necessarily equal one.
        features = torch.cat((q_roles, matched, q_roles * matched,
                              q_roles * text_roles), dim=-1)
        role_scores = self.scorer(features).squeeze(-1)
        background = self.background_scorer(q).squeeze(-1)
        alternatives = torch.stack((role_scores[:, :, 1], background), dim=-1)
        # Binary target probability from a three-outcome model: positive,
        # counterexample, and neither. Two absent roles need not sum to one.
        low_logits = (role_scores[:, :, 0] - torch.logsumexp(alternatives, dim=-1)).reshape(b, 1, h, w)
        logits = F.interpolate(low_logits, size=self.output_size, mode='bilinear',
                               align_corners=False)[:, 0]
        if not return_details:
            return logits
        return dict(logits=logits, attention=weights.reshape(b, h, w, 2, k),
                    role_scores=role_scores.reshape(b, h, w, 2),
                    background_scores=background.reshape(b, h, w))
