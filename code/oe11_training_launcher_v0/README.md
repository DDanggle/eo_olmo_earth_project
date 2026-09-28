# OE11 manual bounded training pilot launcher

This controller runs one manual pilot as three sequential fresh processes:
`uninterrupted`, `split`, `resume`. The worker protocol owns the update counts,
training conditions and case identities. This controller does not change them.

Deployment location is fixed to
`/home/work/data/olmoearth/oe11_training_pilot_v0/code_snapshot/oe11_training_launcher_v0`.
The model worker may be in the protocol's separately pinned snapshot.

## Preserved budget and process controls

- Uses the existing `oe10_text_mask_identity_v2/pilot_budget_ledger_v0.json`
  and initialization witness. It refuses a missing ledger and retains the original
  `connection_20260928_01` completed charge of 48 seconds.
- Maximum cumulative reservation/charge is 7200 seconds; each job reserves at most
  1800 seconds. The three workers share that one wall-clock deadline, including
  shutdown grace, telemetry, checkpoint hashing and stage preparation after the
  reservation. Elapsed failures/timeouts are rounded up and retained. Any unresolved
  process cleanup keeps at least the complete reservation charged.
- Exact copies `_ledger_v1.py` and `_worker_entry_v1.py` reuse the prior ledger lock,
  atomic writes, arithmetic, timeout handling and Linux parent-death guard. Their
  source SHA-256 values are checked on import. The new controller refuses the
  initialization branch of the reused ledger class.
- An exclusive shared ledger lock is held from final preflight through settlement.
  Each subprocess has its own new process group. Only a controller-owned group is
  terminated; other users' processes are never targeted.
- GPU selection is physical GPU1 if idle, otherwise physical GPU0 if idle. A fixed
  `execution.gpu_index` is also accepted. Both physical index/UUID pairs are allowlisted.
  Every stage rechecks idle telemetry and uses the same GPU UUID; the selected UUID
  is passed as `CUDA_VISIBLE_DEVICES`, with `OE11_PHYSICAL_GPU_INDEX` and
  `OE11_EXPECTED_GPU_UUID` for the worker's own UUID validation.
- Busy GPUs cause an immediate blocked result, without a queue, job directory or
  budget reservation. The operating system does not provide an exclusive GPU lease;
  another user's process can start after a telemetry check. Such a process is never
  killed by this launcher.

## CLI

All three modes require these arguments:

```
--protocol ABSOLUTE_JSON --protocol-sha256 SHA256
--initial-budget EXISTING_BOOTSTRAP_JSON --initial-budget-sha SHA256
--attestation EXISTING_P2_TERMINAL_ATTESTATION_JSON --attestation-sha SHA256
--launcher-sha SOURCE_MANIFEST_SHA256
--job-id UNUSED_JOB_ID --seconds 1800
```

Choose exactly one mode:

- `--preflight-only`: read-only GPU, terminal/P2, ledger and pinned input/model checks.
  Busy GPUs skip large input/model hashing and explicitly report that it was skipped.
  Exit 2 means blocked; exit 0 means ready at the time checked.
- `--verify-inputs-only`: performs CPU identity checks even while GPUs are occupied.
  A passing result is explicitly `execution_ready: false`; it cannot authorize launch.
- `--execute`: repeats preflight while holding the shared ledger lock, reserves once,
  and runs all three worker stages. No retries or automation are created.

The protocol requires `backend: actual`; synthetic CPU backends are refused.
Its `paths` include `worker`, `cases`, `contexts`, `model_identity`, `prepared`,
`episodes`, `eo_source`, `deps`, `eo_weights`, and `qwen_weights`. The absolute-path
`file_sha256` mapping binds the worker and production dependencies, cases, contexts,
model identity, prepared manifest and training catalog/scoring files. If
`paths.case_preflight` is included, the launcher checks its pinned PASS result and
its five input identity fields against the current protocol's hashes. Pretrained
reader, EO and EO source files are independently rehashed from model identity.

`execution.stages` is exactly `["uninterrupted", "split", "resume"]`. Arm and
condition must be allowed by the protocol. All worker CLI arguments are structured
argv; no shell is involved. Resume arguments are hashes of this job's own
`split/checkpoint.pt` and `uninterrupted/final_comparison.pt`, computed only after
those stages succeed. Symlink artifacts are rejected. The worker, not the launcher,
creates each stage output directory.

## Artifacts and checks

Each job retains reservation/preflight, command and environment receipts, three
stage logs, status and a final launcher receipt beneath `runs/JOB_ID`. Expected
worker receipt statuses are `uninterrupted_completed`, `split_completed`, and
`resume_passed`; zero process exit without the matching receipt is failure.

Local CPU verification:

```
python3 -B -m unittest -v test_launcher.py
```

The 26 tests use temporary ledgers and fake telemetry/workers. They cover busy
refusal without mutation, existing 48-second charge, cumulative and per-job caps,
duplicate and unresolved jobs, real `flock` contention, failure rounding, timeout
grace, late stage preparation, same-UUID fresh stages, owned checkpoint hashes,
symlink rejection, stale preflight identities and identity-only launch refusal.
These tests do not assert actual CUDA training, Linux parent-death behavior or
model resume equivalence; those are separate validation steps.
