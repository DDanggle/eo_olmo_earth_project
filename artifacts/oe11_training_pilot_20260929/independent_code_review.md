# OE11 bounded training pilot: independent source gate review

2026-09-29. Read-only local review; no server access, GPU use, model launch, or edits to worker/launcher. Cases review is separately recorded in `oe11_pilot_case_review_v1.md`.

**Current recommendation: conditional go for the bounded engineering pilot when the existing launcher preflight passes. No remaining critical source defect found in the inspected paths. This is not an actual-weight/GPU success receipt or a performance endorsement.**

Inspected `training_core.py`, `worker.py`, `actual_backend.py`, `launcher.py`, its pinned reused ledger/entry helpers, and the meaningful negative-test code. The worker's synthetic fresh-process receipt reports two arms × three distinct processes, exact immediate model/AdamW/RNG restore, next-step and final comparison differences0, and two intentional wrong-RNG checkpoints rejected. These checks use a tiny synthetic encoder/reader and the real matching head; actual EO/Qwen/GPU wiring remains unverified.

## Scope and gradients

- Actual backend restricts fetches to the pinned24 train episodes and uses the train-only loader/target accessor. It requires original EO and reader identities, fixed named/knowledge contexts, query2/support8. No development/final scorer or generator is called.
- This is mask-only adaptation, with frozen Qwen text features. It does not run the old official native replay or language CE. Worker receipts explicitly distinguish those scopes.
- Every present gradient must be finite. Head and B2 EO gradients must be nonzero; actual anchor changes are checked after AdamW. K8 attention-route evidence is accumulated separately because K1 singleton attention gradients can be exactly zero. Reader, unused connector, and B0 encoder must remain frozen with grad=None and unchanged parameter-version records.
- Full trainable-parameter and AdamW-state finiteness is checked at checkpoint/comparison boundaries. Frozen parameter checks use PyTorch version/grad invariants, not a full second bytewise reading of all frozen weights; the receipt explicitly says the latter was not performed.

## Resume sequencing

- Both uninterrupted and split paths clear caches after update24. Split saves model/buffers, optimizer, RNG, step/order, identity, and diagnostics. Resume uses a distinct PID, verifies externally supplied artifact hashes, loads with `weights_only=True`, restores caches/modes before restoring RNG, and compares immediate model/optimizer/RNG exactly.
- Update25 and update48 compare model/AdamW floating tensors, logits before and after the update, and loss at the frozen1e-6 tolerance; integer state/order/RNG remain exact. The extra comparison forward occurs at the same points on both trajectories, so it does not create an asymmetric RNG/cache path.
- SHA-bound final-reference metadata binds the separate next-step comparison artifact. Startup/protocol/source/input identities must agree. A failed source recheck or mismatched trajectory exits unsuccessfully; successful intermediate receipt text does not override a failing process exit.

## Launcher corrections independently rechecked

Two source review findings were fixed before launch:

1. Idle detection is tied to an exact GPU UUID. The launcher now sets `CUDA_VISIBLE_DEVICES` to that UUID and sends the physical index separately; the worker checks the actual Torch device UUID. This closes possible disagreement between nvidia-smi indexes and CUDA enumeration.
2. The launcher rechecks the global deadline immediately before Popen, after GPU telemetry, dynamic artifact hashing, log creation, and environment preparation. A delayed preparation cannot start a new GPU process after its reservation is exhausted. The new negative test models this delayed-telemetry path and expects no spawned worker.

All three stages use the same selected physical GPU. Each stage launches a new owned process group. The original48-second charge and existing ledger lineage are mandatory; no new budget initialization or paused automation restart occurs. The reservation spans the whole three-stage job, including loading, diagnostics, comparisons, retries/failures, and cleanup. Actual elapsed time is rounded up; unresolved cleanup retains at least the full reservation. Failure stops subsequent stages. Busy-GPU preflight is read-only and does not reserve or create a job directory. Launcher author reports25 CPU tests passed; review inspected the relevant negative tests, without redundantly rerunning that suite.

## Reporting boundaries

The actual frozen cases have12 unique train images, four per source parent, 24 K1/K8 tasks, and two complete24-task cycles. Pre/post fit probes cover only the12 K1 tasks. Loss decrease is explicitly diagnostic rather than a computational pass gate. Keep target-present fit and absent-case false positives visible; the mostly-negative cohort can otherwise make background collapse look like improvement. No CVPR gain, semantic understanding, convergence, EO capability preservation, or unseen-region generalization follows from this pilot.

Timing now separates data fetch, forward/backward/optimizer, and full step. The older9.065-second B2 workload included a different native/CE path and is only a planning reference; use actual job occupancy for budget accounting.

Reviewed source SHA256 values (later source changes require a scoped follow-up):

- worker.py after scoped follow-up: `99f89da1d2440eabd3fadb4ecfe24ca1ffa1caee81625e1af353bc415c37dec0`
- training_core.py: `855bad78787ee6e17da1aa8b829b93057526d3b6467fb260851557791ced4af5`
- actual_backend.py: `9657457dd5cdca234a9d5747bef8d1c03486d227a581d38ddb361d52cedba180`
- launcher.py: `bbfd28163d6921097840059e664337c442684c9918644892db77d714395ac8b9`
- CPU fresh-process receipt: `ebd473bbaff6acbc9f18bad1c9555a47acc7a7bd36d3e5c18e5b008bcb6b591e`

## Scoped follow-up: verification-only CLI and manual guard

The previously reviewed worker hash was `82645871664308f8c57e922a09c876edbd38d3602b4509b52316fe568f788e9c`. The new `99f89da1...` worker adds `--verify-only` and requires the actual-backend manual-controller marker. The verification-only branch returns before output-directory creation, Torch import, CUDA initialization, or backend/model construction. Its status explicitly says protocol/source/input verification only; it does not claim model/optimizer/resume-artifact verification. The marker matches the existing launcher's environment. Training, checkpoint, and comparison sequencing remains as reviewed; core/actual-backend/launcher hashes above are unchanged. No new blocking defect found.

Also inspected `/private/tmp/oe11_build_pilot_protocol.py`: fixed48 updates/split24, 96 total optimizer updates across the three processes, one1800-second shared job cap, original ledger48-second charge, K1-only12-case fit scope, and mask-only/non-efficacy claims are consistent. The wording “No development/final-region data read” should be narrowed to image/label payloads and model predictions: the full shared prepared manifest includes calibration metadata and is read by the loader. This is a reporting precision correction, not evidence of calibration-label training.

The actual CPU token audit reports48 serialization checks but only8 unique serialized role inputs, maximum208 tokens, and no truncation. These are tokenizer-budget checks, not48 independent language examples or semantic validation. This audit does not load model weights or use a GPU.
