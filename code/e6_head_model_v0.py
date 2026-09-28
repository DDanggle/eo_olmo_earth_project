"""Draft E6 pure head, metadata encoder, and full EO transform.

This is model-component preparation, not an experiment runner or preregistration.
No files, datasets, IDs, labels, checkpoints, encoders, or LLMs are read. Public
inference accepts only EO pairs, phenomenon strings, and ordered ISO dates.
Equal example exposure never means equal FLOPs, parameters, or optimization.

Initialization: CPU float32; independent modules in declaration order, PyTorch
default Linear initialization, LayerNorm ones/zeros, type N(0,.02), CLS zero.
Fixed positions use Python float64 sin/cos, rounded once to float32. Attention
uses explicit QKV/matmul/softmax (no fused attention or inference fastpath).
"""
from datetime import date
import hashlib
import json
import math
import re

import torch
from torch import nn

WIDTH = 128
N_PARAMETERS = 367361
PAIR_SHAPE = (2, 64, 768)
ISO_DAY = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_tensor(value, name):
    require(isinstance(value, torch.Tensor), name + ' must be a Tensor')
    require(bool(torch.isfinite(value).all()), name + ' must be finite')


def full_transform(pair):
    """(2,64,768) or (N,2,64,768) float32 -> full tokens plus 192 int64 types.

    Every returned tensor is newly allocated; source frames remain unchanged.
    The arithmetic remains differentiable, including D=B-A.
    """
    require(isinstance(pair, torch.Tensor), 'pair must be a Tensor')
    require(pair.dtype == torch.float32, 'pair must be float32')
    require((pair.ndim == 3 and tuple(pair.shape) == PAIR_SHAPE) or
            (pair.ndim == 4 and len(pair) > 0 and tuple(pair.shape[1:]) == PAIR_SHAPE),
            'pair must have shape (2,64,768) or nonempty (N,2,64,768)')
    finite_tensor(pair, 'pair')
    earlier, later = pair.unbind(dim=-3)
    delta = later-earlier
    finite_tensor(delta, 'B-A')
    tokens = torch.cat((earlier, later, delta), dim=-2)
    types = torch.tensor([0]*64+[1]*64+[3]*64, dtype=torch.int64, device=pair.device)
    return tokens, types


def encode_metadata(phenomena, dates):
    """Return CPU float32 (N,8), using only phenomenon and two exact ISO dates.

    Date order is strictly increasing. Dict/item records are deliberately not
    accepted: ID, source answer, kind, event and quality fields cannot be read.
    """
    require(isinstance(phenomena, (list, tuple)) and isinstance(dates, (list, tuple)),
            'phenomena and dates must be explicit sequences')
    require(len(phenomena) == len(dates) and len(phenomena) > 0, 'Metadata batch lengths differ/empty')
    output = []
    for phenomenon, pair in zip(phenomena, dates):
        require(type(phenomenon) is str and phenomenon in ('flood', 'landslide'), 'Unknown phenomenon')
        require(isinstance(pair, (list, tuple)) and len(pair) == 2, 'Two ordered dates required')
        require(all(type(value) is str and ISO_DAY.fullmatch(value) is not None for value in pair),
                'Dates must be exact YYYY-MM-DD strings')
        earlier, later = (date.fromisoformat(value) for value in pair)
        require(earlier < later, 'Dates must be distinct and chronological')
        row = [float(phenomenon == 'flood'), float(phenomenon == 'landslide')]
        for day in (earlier, later):
            row.extend(((day.year-2000)/100, (day.month-1)/11, (day.day-1)/30))
        output.append(row)
    result = torch.tensor(output, dtype=torch.float32, device='cpu')
    finite_tensor(result, 'encoded metadata')
    return result


def spatial_positions():
    """192x128 fixed row/column sincos; equal position vectors for A/B/D.

    Axis components interleave sin then cos for k=0..31. Tokens are row-major.
    """
    def axis(coordinate):
        result = []
        for k in range(32):
            phase = coordinate/(10000.0**(k/32.0))
            result.extend((math.sin(phase), math.cos(phase)))
        return result
    grid = torch.tensor([axis(row)+axis(col) for row in range(8) for col in range(8)],
                        dtype=torch.float32, device='cpu')
    return grid.repeat(3, 1)


class AttentionBlock(nn.Module):
    """Pre-LN width128, four heads, FF256, exact GELU, no dropout/causal mask."""
    def __init__(self):
        super().__init__()
        factory = {'device': 'cpu', 'dtype': torch.float32}
        self.norm_attention = nn.LayerNorm(WIDTH, eps=1e-5, **factory)
        self.qkv = nn.Linear(WIDTH, 3*WIDTH, bias=True, **factory)
        self.attention_output = nn.Linear(WIDTH, WIDTH, bias=True, **factory)
        self.norm_ff = nn.LayerNorm(WIDTH, eps=1e-5, **factory)
        self.ff_in = nn.Linear(WIDTH, 256, bias=True, **factory)
        self.gelu = nn.GELU(approximate='none')
        self.ff_out = nn.Linear(256, WIDTH, bias=True, **factory)

    def forward(self, values):
        batch, length, _ = values.shape
        normalized = self.norm_attention(values)
        q, k, v = self.qkv(normalized).reshape(batch, length, 3, 4, 32).permute(2, 0, 3, 1, 4).unbind(0)
        weights = torch.softmax((q @ k.transpose(-2, -1))/math.sqrt(32.0), dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, length, WIDTH)
        values = values+self.attention_output(attended)
        return values+self.ff_out(self.gelu(self.ff_in(self.norm_ff(values))))


class SpatialPairHead(nn.Module):
    def __init__(self):
        super().__init__()
        factory = {'device': 'cpu', 'dtype': torch.float32}
        self.input_norm = nn.LayerNorm(768, eps=1e-5, **factory)
        self.input_linear = nn.Linear(768, WIDTH, bias=True, **factory)
        self.input_gelu = nn.GELU(approximate='none')
        self.input_output_norm = nn.LayerNorm(WIDTH, eps=1e-5, **factory)
        self.type_embedding = nn.Embedding(4, WIDTH, **factory)
        nn.init.normal_(self.type_embedding.weight, mean=0.0, std=0.02)
        self.cls = nn.Parameter(torch.zeros(WIDTH, **factory))
        self.metadata_linear = nn.Linear(8, WIDTH, bias=True, **factory)
        # Separate constructor calls, not copied/cloned prototype layers.
        self.blocks = nn.ModuleList([AttentionBlock(), AttentionBlock()])
        self.output_norm = nn.LayerNorm(WIDTH, eps=1e-5, **factory)
        self.classifier = nn.Linear(WIDTH, 1, bias=True, **factory)
        self.register_buffer('positions', spatial_positions(), persistent=True)
        self.register_buffer('token_types', torch.tensor([0]*64+[1]*64+[3]*64,
                                                       dtype=torch.int64, device='cpu'), persistent=True)
        require(sum(p.numel() for p in self.parameters()) == N_PARAMETERS, 'Architecture parameter count changed')

    def forward(self, pairs, metadata):
        """Trainable logits for explicit (N,2,64,768) pairs and encoded (N,8).

        Labels belong only in the external BCE target, never this interface.
        No mapping/item lookup exists in this component.
        """
        require(isinstance(pairs, torch.Tensor) and pairs.ndim == 4, 'Head requires a batch of pairs')
        require(isinstance(metadata, torch.Tensor) and metadata.dtype == torch.float32
                and tuple(metadata.shape) == (len(pairs), 8), 'Metadata must be float32 (N,8)')
        require(pairs.device == metadata.device == self.cls.device, 'Input/model devices differ')
        finite_tensor(metadata, 'metadata')
        require(bool(((metadata[:, :2] == 0) | (metadata[:, :2] == 1)).all())
                and bool((metadata[:, :2].sum(dim=1) == 1).all()), 'Phenomenon encoding is not one-hot')
        tokens, types = full_transform(pairs)
        require(torch.equal(types, self.token_types), 'Token type buffer differs')
        values = self.input_output_norm(self.input_gelu(self.input_linear(self.input_norm(tokens))))
        values = values+self.type_embedding(types)[None]+self.positions[None]
        cls = self.cls[None]+self.metadata_linear(metadata)
        values = torch.cat((cls[:, None], values), dim=1)
        for block in self.blocks:
            values = block(values)
        logits = self.classifier(self.output_norm(values[:, 0])).squeeze(-1)
        finite_tensor(logits, 'head logits')
        return logits

    @torch.no_grad()
    def predict(self, pairs, phenomena, dates):
        """Label/ID-free inference; no model state or training-mode mutation."""
        metadata = encode_metadata(phenomena, dates).to(device=pairs.device)
        logits = self(pairs, metadata)
        return {'logits': logits, 'probabilities': torch.sigmoid(logits), 'predictions_yes': logits >= 0}


def make_head(seed):
    """Fresh CPU head with reproducible scoped initialization; preserve CPU RNG.

    No CUDA discovery or GPU initialization is performed. Matching seeds across
    different architectures do not imply matching initial representations.
    """
    require(type(seed) is int and 0 <= seed < 2**63, 'Seed must be nonnegative int below 2**63')
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return SpatialPairHead()


def state_hash(model):
    """Canonical named CPU tensor bytes/dtype/shape; reject nonfinite state."""
    state = model.state_dict()
    h = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        finite_tensor(tensor, 'state '+name)
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode('utf8')
        h.update(len(header).to_bytes(8, 'little')); h.update(header)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        h.update(len(raw).to_bytes(8, 'little')); h.update(raw)
    return h.hexdigest()
