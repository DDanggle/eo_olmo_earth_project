"""Dependency-only cache checks. No tensor/model parity or autograd is executed.

Identity values must describe actual computation, not a guessed filename. An EO
window is ordered and complete: adding a visit can change every contextual token.
KV identity covers the unchanged causal prefix only. A suffix can change without
invalidating that prefix; a query used by a selector belongs in projection keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from typing import Any, Mapping


class Stage(IntEnum):
    RAW = 0
    EO = 1
    PROJECTION = 2
    KV = 3


REQUIRED = (
    frozenset(("content_hashes", "crop_crs", "band_order", "calibration",
               "normalization", "validity", "preprocessor", "augmentation",
               "dtype")),
    frozenset(("encoder_weights", "encoder_config", "input_window",
               "attention_mask", "mode", "dtype", "runtime")),
    frozenset(("adapter_weights", "resampler_weights", "codebook_version",
               "codebook_weights", "eo_embedding_weights",
               "projector_config", "selector_revision", "selector_mode",
               "query_identity", "dtype", "runtime")),
    frozenset(("reader_weights", "reader_config", "causal_prefix_identity",
               "position_ids", "rope_config", "attention_mask",
               "multimodal_layout", "dtype", "cache_format", "runtime")),
)
SCHEMA = "oe-cache-dependencies-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class Request:
    # Canonical strings snapshot mutable caller inputs and retain ordered lists.
    stages: tuple[str, ...]

    @classmethod
    def build(cls, *, raw: Mapping, eo: Mapping, projection: Mapping,
              kv: Mapping, query_suffix: str = "") -> "Request":
        del query_suffix  # Never part of a cached *prefix*. See module contract.
        parts = (raw, eo, projection, kv)
        for stage, values, keys in zip(Stage, parts, REQUIRED):
            missing = keys - values.keys()
            if missing:
                raise ValueError(f"{stage.name}: missing {sorted(missing)}")
            for key in keys - {"query_identity"}:
                if values[key] is None or values[key] in ("", [], {}):
                    raise ValueError(f"{stage.name}: empty identity {key}")
        window = eo["input_window"]
        if eo["mode"] != "eval":
            raise ValueError("EO: exact persistent reuse requires deterministic eval")
        if not isinstance(window, (list, tuple)) or not window:
            raise ValueError("EO: input_window must be a nonempty ordered list")
        for visit in window:
            if not isinstance(visit, dict) or not {
                    "source_id", "acquired_at", "sensor"} <= visit.keys():
                raise ValueError("EO: each visit needs source_id/acquired_at/sensor")
            if any(not visit[k] for k in ("source_id", "acquired_at", "sensor")):
                raise ValueError("EO: visit identities must be nonempty")
        mode = projection["selector_mode"]
        query = projection["query_identity"]
        if mode not in {"static", "query_conditional"}:
            raise ValueError("PROJECTION: unknown selector_mode")
        if (mode == "static" and query is not None) or (
                mode == "query_conditional" and not query):
            raise ValueError("PROJECTION: selector/query identity mismatch")
        return cls(tuple(_canonical(p) for p in parts))

    def dependencies(self, stage: Stage) -> tuple[str, ...]:
        return tuple(hashlib.sha256((SCHEMA + ":" + part).encode()).hexdigest()
                     for part in self.stages[:int(stage) + 1])


@dataclass(frozen=True)
class Training:
    raw_requires_grad: bool = False
    encoder_trainable: bool = False
    projector_trainable: bool = False
    reader_trainable: bool = False

    def needs_graph(self, stage: Stage) -> bool:
        return (self.raw_requires_grad or
                (stage >= Stage.EO and self.encoder_trainable) or
                (stage >= Stage.PROJECTION and self.projector_trainable) or
                (stage >= Stage.KV and self.reader_trainable))

    def assert_reader_forward(self, *, no_grad: bool) -> None:
        if no_grad and self.needs_graph(Stage.KV):
            raise ValueError("reader no_grad blocks required training gradients")


@dataclass(frozen=True)
class Entry:
    stage: Stage
    dependencies: tuple[str, ...]
    fidelity: str = "exact"
    schema: str = SCHEMA

    @classmethod
    def capture(cls, request: Request, stage: Stage,
                *, fidelity: str = "exact") -> "Entry":
        return cls(stage, request.dependencies(stage), fidelity)


@dataclass(frozen=True)
class Decision:
    reusable: bool
    reason: str
    first_changed_stage: str | None = None


def check_reuse(entry: Entry, request: Request,
                training: Training = Training()) -> Decision:
    """Check detached stored artifacts; graph-preserving intra-step reuse differs."""
    if entry.schema != SCHEMA:
        return Decision(False, "schema_mismatch")
    if entry.fidelity != "exact":
        return Decision(False, "approximate_cache_unsupported")
    if not isinstance(entry.stage, Stage):
        return Decision(False, "invalid_stage")
    if training.needs_graph(entry.stage):
        return Decision(False, "detached_cache_blocks_gradients")
    expected = request.dependencies(entry.stage)
    if len(entry.dependencies) != len(expected):
        return Decision(False, "invalid_dependency_count")
    for index, (old, new) in enumerate(zip(entry.dependencies, expected)):
        if old != new:
            return Decision(False, "dependency_changed", Stage(index).name)
    return Decision(True, "exact_dependencies_match")
