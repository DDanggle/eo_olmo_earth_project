"""Pure, label-free token interventions for the frozen E2 EO reader.

Input is exactly two pooled observations, (2, 64, 768), in earlier/later
order. The model-facing layout stays 192 tokens with type IDs 0, 1, 3.
Removing an observation also removes its derived difference; otherwise
the difference would reveal the supposedly removed observation.

These are interventions on input embeddings, not annotations of physical
events. In particular, repeated-frame and reverse arms do not supply new
yes/no gold labels. Any interpretation belongs in the experiment contract.

The input is never modified. Returned arrays own their memory, have no
shared mutable global state, and can safely be passed to torch.from_numpy.
"""

from __future__ import annotations

import numpy as np

ARMS = (
    "real",
    "earlier_only",
    "later_only",
    "repeat_earlier",
    "repeat_later",
    "no_delta",
    "reverse",
)
PAIR_SHAPE = (2, 64, 768)
TOKEN_SHAPE = (192, 768)


def transform_pair(pair: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return independent (tokens, type_ids) arrays for one intervention.

    ``pair`` must be a finite NumPy float16/float32/float64 array of shape
    (2, 64, 768). No implicit casting, resizing, or nonfinite imputation is
    performed. Token dtype is preserved; type IDs are int64. The returned
    token array is C-contiguous. Finite input that overflows a required
    difference is rejected rather than silently changing arithmetic dtype.

    Layouts, with A=earlier and B=later:
      real           [A, B, B-A]
      earlier_only   [A, 0, 0]
      later_only     [0, B, 0]
      repeat_earlier  [A, A, 0]
      repeat_later    [B, B, 0]
      no_delta       [A, B, 0]
      reverse        [B, A, A-B]

    Type IDs remain [0]*64 + [1]*64 + [3]*64 in every arm. They identify
    model-facing slots, so reverse swaps content but does not swap IDs.
    """
    if not isinstance(arm, str) or arm not in ARMS:
        raise ValueError(f"Unknown intervention arm: {arm!r}; expected one of {ARMS}")
    if not isinstance(pair, np.ndarray):
        raise TypeError("pair must be a NumPy ndarray; implicit conversion is not permitted")
    if pair.shape != PAIR_SHAPE:
        raise ValueError(f"pair must have shape {PAIR_SHAPE}; got {pair.shape}")
    if pair.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
        raise TypeError(f"pair must use float16, float32, or float64; got {pair.dtype}")
    if not np.isfinite(pair).all():
        raise ValueError("pair contains NaN or infinity")

    earlier, later = pair
    tokens = np.zeros(TOKEN_SHAPE, dtype=pair.dtype)
    if arm in ("real", "earlier_only", "no_delta"):
        tokens[:64] = earlier
    if arm in ("real", "later_only", "no_delta"):
        tokens[64:128] = later
    if arm == "repeat_earlier":
        tokens[:64] = earlier
        tokens[64:128] = earlier
    elif arm == "repeat_later":
        tokens[:64] = later
        tokens[64:128] = later
    elif arm == "reverse":
        tokens[:64] = later
        tokens[64:128] = earlier

    if arm in ("real", "reverse"):
        with np.errstate(over="ignore", invalid="ignore"):
            np.subtract(tokens[64:128], tokens[:64], out=tokens[128:])
        if not np.isfinite(tokens[128:]).all():
            raise ValueError(f"{arm} difference is nonfinite in dtype {pair.dtype}")

    types = np.repeat(np.array([0, 1, 3], dtype=np.int64), 64)
    return tokens, types
