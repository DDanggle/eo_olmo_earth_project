# OE11 bounded train / fresh-process resume worker

Implemented engineering worker, **actual EO/Qwen GPU execution not yet tested**.
It trains only the fixed registered TRAIN cases with binary mask loss. No native
OlmoEarth replay, language CE, generation, development/final evaluation, scheduler,
subprocess, or background worker is created by production code.

The CPU test uses the actual matching head with a tiny synthetic encoder/reader
and dropout (to make RNG state consequential). B2+facts and B0+names each passed
three distinct processes: uninterrupted4, split2, resume2. Model, AdamW moments/
step, RNG, next-step and final pre/post-update logits, and losses matched exactly.
Two deliberately wrong-RNG checkpoints restored but were correctly rejected by
trajectory comparison. This does not verify actual EO/Qwen or CUDA behavior.

## Interface

```sh
python worker.py --protocol /absolute/protocol.json --protocol-sha256 HASH \
  --stage uninterrupted --arm B2 --condition matched_knowledge --out NEW_DIRECTORY
```

Stages: `uninterrupted`, `split`, `resume`. Stage output must not exist beforehand.
Resume additionally requires:

```sh
--resume-from SPLIT/checkpoint.pt --resume-sha256 HASH \
--reference-result UNINTERRUPTED/final_comparison.pt --reference-sha256 HASH
```

`final_comparison.pt` binds its sibling `next_step_comparison.pt` by SHA. All own
checkpoint/reference reads require their expected SHA and use
`torch.load(weights_only=True, map_location='cpu', mmap=True)`; Python/NumPy RNG
state is serialized as primitive containers plus Torch RNG tensors.

Final receipt statuses: `uninterrupted_completed`, `split_completed`,
`resume_passed`; exceptions become `failed`. The controller must inspect exit
status and the final receipt, not intermediate progress messages.

`--verify-only` checks protocol/source/input pins without importing Torch or
creating output/model/GPU state. Required ordinary CLI fields remain required;
actual-mode controller environment markers are checked even for verification.

## Protocol contract

- `schema: oe11_training_pilot_v0`; `backend: actual` for the real controller.
  `tiny_cpu` is exclusively the synthetic test path.
- `seed`, `total_steps`, `split_step`, `max_stage_seconds <= 1800`;
  `resume_tolerance: 1e-6`.
- `allowed_arms`, `allowed_conditions` must include the selected CLI values.
- `optimizer: {encoder_lr,head_lr,weight_decay,eps,clip_grad_norm}`. The intended
  actual pilot values are `1e-5,1e-3,.01,1e-6,1.0`; AdamW has `foreach=False`,
  `fused=False` and fixed LR with no copied P2 warmup/scheduler.
- `max_text_tokens:1024`, `max_text_cache_entries` in0..64.
- `paths`: absolute `worker`, `cases`, `contexts`, `model_identity`, `eo_source`,
  `deps`, `eo_weights`, `qwen_weights`, `prepared`, `episodes`, `matching_code`,
  `loader_code`, `text_mask_code`.
- `file_sha256`: **absolute file path -> SHA**. Required pins include the three
  production files (`worker.py`, `training_core.py`, `actual_backend.py`), cases,
  contexts, model identity, matching2 source files, loader2 files including the
  frozen base, text-mask5 frozen files, prepared manifest, pooled episode
  contract, train catalog and train scoring file. Every supplied pin is checked
  before and after the stage. The actual backend separately checks all pinned
  EO/Qwen/tokenizer/normalization source and weight files from model_identity.

Case JSON must have `split:train`, matching seed/step/split values, unique
`episode_ids`, exact `order` of total_steps, optional `fit_episode_ids` subset,
`query_positions:[2,5]`, `support_positions:[0..7]`. Optional `input_hashes` must
match corresponding protocol pins. The intended actual v1 case bundle is24 IDs
(12 fixed query/role cases×K1/K8),48 updates, split24,12 K1 fit probes.

Production CUDA binding is explicit: controller exports
`OE11_MANUAL_BOUNDED_JOB=1`, `OE11_PHYSICAL_GPU_INDEX=0|1`,
`OE11_EXPECTED_GPU_UUID=GPU-...`, and `CUDA_VISIBLE_DEVICES` equal to that UUID.
Torch logical device0's UUID must match before model loading. The worker does
not select GPUs or manage budget reservations; the outer controller must enforce
idle ownership, cumulative budget, parent-death cleanup and the whole-job cap.

## Gates and interpretation

At the split boundary both trajectories clear caches before checkpoint/next
step. A fresh process constructs models/optimizer first, loads their state and
cost/order cursor, clears caches, then restores RNG last. Immediate restored
model/optimizer/RNG equality is exact; next and final trajectory comparisons use
absolute tolerance1e-6 for floating tensors/loss and exact RNG/order metadata.

Finite gradients are checked every update; encoder/head parameters must actually
change under B2, while B0 encoder parameters stay frozen. Qwen/unused connector
must retain `requires_grad=False`, `grad=None`, and unchanged parameter versions.
This is an optimizer/version guard, **not a full Qwen weight-byte reread**. Whole
trainable parameters/AdamW state are checked for finiteness at saved checkpoints
and comparison steps. K1 attention gradients may be zero by construction;
query_key/object_key/text_key must receive nonzero gradients on some K8 training
step. A missing K8 learning route fails at completion.

Fixed train-fit diagnostics report loss, positive-only IoU/recall and absent-case
positive fraction separately. Empty targets do not inflate positive IoU. Loss
improvement is recorded, not imposed as an arbitrary48-step accuracy gate.
Names/facts/removal/name-fixed swapped-fact sensitivity is a one-case diagnostic,
not evidence of semantic understanding. Both roles/text use the same matching
head; no class-specific checkpoint/threshold selection occurs.

Per-step `fetch_seconds`, `forward_backward_optimizer_seconds`, and
`full_step_seconds` separate I/O from compute. Controller elapsed time remains
the authoritative GPU-occupancy budget; worker timing does not replace it.

CUDA deterministic algorithms are required. Unsupported actual operations fail
explicitly; no automatic nondeterministic fallback or extra training is started.
Actual CUDA RNG, official-model update behavior and fresh-process reproducibility
remain pending until the authorized idle-GPU pilot runs.

## CPU test

`test_fresh_process.py` alone spawns synthetic CPU child processes; production
worker code never does. It uses preexisting local cached PyTorch2.9.1 packages,
sets CUDA_VISIBLE_DEVICES empty, and needs no installation/network/server.

```sh
/Users/dongdong/.local/share/uv/python/cpython-3.12.10-macos-aarch64-none/bin/python3 test_fresh_process.py
```

On Linux use its installed CPU-capable Torch/NumPy environment. Cached-package
paths are added only when they exist on macOS. Set `OE11_MATCHING_CODE` to the
deployed matching source directory and `OE11_TEST_OUTPUT_DIRECTORY` to a new
results directory outside the immutable snapshot. If output is unspecified a
fresh temporary results directory is created; tests never write into the source.

The resulting `cpu_fresh_process_receipt.json` records six successful stages,
two deliberate wrong-RNG failures, actual distinct process IDs and all numeric
differences. It must remain labelled synthetic CPU evidence.
