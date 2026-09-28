# Frozen connection launcher v1 — engineering only

This stdlib launcher is prepared for the single frozen real-weight connection
protocol. Writing, importing and testing it neither starts a GPU job nor creates
a production budget reservation. It does not change the model, data, training,
score thresholds, original P2 cap, or research protocol.

The only production code location is:

    /home/work/data/olmoearth/oe10_text_mask_identity_v2/code_snapshot/oe10_connection_launcher_v1

Production execution/preflight requires Linux at that exact path. Protocol and
bootstrap files belong below the same oe10_text_mask_identity_v2 root, normally
under config/. The protocol SHA is embedded in launcher.py; the caller must
also supply the independently frozen launcher source-manifest SHA. Model source,
identity and complete episode export hashes come from the frozen protocol.
Full reader/EO weight and selected EO source hashes are rechecked by the frozen
model CLI inside the reservation before it loads the models. No shell is used.

## Read-only preflight

Use --preflight-only, not --execute. Supply:

    --protocol ROOT/config/oe10_text_mask_connection_protocol_20260928.json
    --initial-budget ROOT/config/oe10_pilot_budget_bootstrap_20260928.json
    --initial-budget-sha 0c7a2c815fc12270a68fa5cb9e8739a36614efe52837d94dcd13bd47b094cd20
    --launcher-sha EXTERNALLY_FROZEN_LAUNCHER_MANIFEST_SHA
    --job-id prospective_connection_01
    --seconds 1800

ROOT above abbreviates the full oe10_text_mask_identity_v2 server root. Preflight
prints reasons and exits 2 when blocked; it writes no ledger/lock/reservation or
result. It reads actual P2 status first, so running P2 is reported even when no
terminal attestation exists. A later successful preflight additionally requires
--attestation PATH and --attestation-sha EXTERNAL_SHA. Creating/reviewing that
attestation is an independent action; the launcher never creates one.

Required attestation schema (a specification, not an existing attestation):

    schema_version: oe10_p2_terminal_audit_attestation_v1
    audit_complete: true
    p2_status_sha256: digest of the actual terminal training_v0/status.json
    p2_terminal_status: completed or failed, matching the status
    completed_runs: actual completed training receipt count
    expected_runs: 6
    p2_result: complete only for all six completed, otherwise incomplete
    audited_utc: timezone-aware time after status.finished_utc
    audit_files: list of {path: absolute file under oe10_p2_v0, sha256: digest}

P2's protected_and_snapshot_unchanged must be true. All recorded controller and
worker PIDs must be inactive, and ps must show no P2 controller/worker command.
An honestly audited capped incomplete result may pass this engineering launch
prerequisite; it is still recorded incomplete, never promoted to a full P2 gate.

GPU1 must match the fixed UUID and have no compute processes. Numeric utilization
requires finite zero utilization and finite memory from 0 through 256 MiB. If
utilization is exactly [Not Found] or N/A, memory must instead be exactly zero.
The unavailable reading is recorded as null with its original marker, never as
a measured zero. Unknown/nonfinite/malformed readings are refused. Other GPU
work is neither rejected nor terminated. GPU1 is checked again directly before spawning the owned worker.

## One reserved attempt

Only explicit --execute can initialize/reserve. The dedicated server ledger is
ROOT/pilot_budget_ledger_v0.json. An advisory flock is held for the entire owned
attempt. Bootstrap accepts the externally pinned zero pilot_budget snapshot;
initialization never overwrites a ledger. A durable initialization witness also
blocks automatic recreation after a ledger disappears. Unresolved reserved,
running or cleanup-unconfirmed attempts retain their full reservation and block
later attempts, even if their controller crashed. This code has no reset/retry
or recovery command. Unique job IDs cannot be reused. All finished occupancy is
rounded up and charged, with a maximum reservation of 1800 seconds and cumulative
budget of 7200 seconds. The remaining budget is never reset per heartbeat.

The new process starts its own session. TERM and KILL target only that recorded
process group. The TERM grace and final reap window are inside the reservation,
not appended to it. Surviving descendants of an exited leader trigger cleanup;
if the group cannot be confirmed gone at the deadline, its reservation remains
fully charged. SIGINT/SIGTERM/SIGHUP on the launcher cause immediate cleanup.
Kernel scheduling/uninterruptible tasks are not assumed to obey an exact
realtime deadline; actual elapsed time and unconfirmed cleanup remain recorded.

Before importing torch/transformers, worker_entry enables Linux
PR_SET_PDEATHSIG(SIGKILL) and checks the expected parent PID both before and after
the call. It then records actual Python/import paths/package versions/CUDA and
environment before exec of the frozen model CLI. Ordinary exec preserves this
setting; setuid/setgid/file-capability interpreters are rejected. This covers the
fixed single-process Python worker (with threads), **not arbitrary fork
subprocess trees**. The broader fork limitation is explicit in the
[Linux API specification](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html).
The launcher separately checks its owned group; a crash reservation is never
released automatically on the assumption that all descendants stopped.

PYTHONPATH is removed; physical CUDA_VISIBLE_DEVICES=1, offline Hugging Face and
bounded CPU thread variables are passed as an environment mapping. Import
receipt, exact argv, preflight, frozen-model reservation, logs and elapsed/exit
receipt are saved under ROOT/connection_runs_v0/JOB_ID. Successful launcher exit
requires the frozen model's success receipt as well as exit code zero. This is
an engineering connection check, not training, human correction, semantic
knowledge improvement or performance evidence.

## CPU tests

    python -B -m unittest discover -s code/oe10_connection_launcher_v1 -p test_launcher.py -v

Tests use fake GPU/clock/worker responses and temporary ledgers. They exercise
concurrent reservation locks, unknown/crashed reservation charge, no reset,
retry IDs, cumulative budget, live P2 rejection, terminal-audit integrity, GPU
busy/UUID failure, only-owned-group timeout, surviving descendant cleanup and
read-only preflight. Linux additionally runs isolated CPU subprocess tests for
parent-death across ordinary exec and the expected-parent race check. No test
loads the real EO/Qwen weights, creates a production reservation, or uses CUDA.

## v1 operational correction

v0 is preserved. Only the fixed launcher source path and GPU-idle telemetry
handling changed in production code. Actual GPU1 reported zero memory and
[Not Found] utilization after P2 ended, while compute processes existed only on
GPU0. This correction does not change model, protocol, budget/ledger paths,
PID ownership, terminal-audit requirements or resource caps. CPU tests include
unavailable telemetry, occupancy, strict zero memory, malformed/nonfinite values
and the original numeric path. Server preflight and GPU execution remain the
root agent's separate actions.
