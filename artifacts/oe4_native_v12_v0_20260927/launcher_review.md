# Bounded VLM launcher validation record

Validated locally on 2026-09-27 KST. Artifact: `run_vlm_bounded.py`.

**No server command, real subprocess, nvidia-smi invocation, GPU allocation, model load, or training run was performed by these checks.** The launcher was imported as a module; its `main()` entry point was exercised only with temporary filesystem paths and mocked GPU queries/process creation. Temporary fixtures lived under `/private/tmp` and were removed on completion.

## Source revision and native-controller dependency

The original checked launcher waited for `bounded_sequence_v1_status.json`. Parent subsequently reported that the native CPU preflight found a missing public-config loss registry type and that only the parent's own idle native controller was stopped. The sole launcher code change following that report is:

```python
NATIVE_STATUS = ROOT / "bounded_sequence_v2_status.json"
```

All waiting, GPU selection, hashing, timeout, process, and output behavior remains identical. Python syntax compilation was rerun after this constant change. The v1 mock run below remains a test of the same control logic; it does not establish that native v2 has completed or that either GPU is free.

## Completed local checks

- `python3 -m py_compile /private/tmp/oe4_execute_20260927/run_vlm_bounded.py`: exit 0 before and after the native-status constant change.
- Pure GPU parser with two idle GPUs selected physical GPU 1 before GPU 0.
- A compute PID on GPU 1 excluded it and left GPU 0 eligible.
- Memory exactly 512 MiB and utilization exactly 5% were rejected, preserving the strict `<512` and `<5` criteria.
- `[Not Found]` utilization, `N/A` memory, malformed GPU output, malformed process output, and empty GPU output returned no eligible GPU and a BUSY diagnostic.
- A mocked `subprocess.check_output` `OSError` also returned BUSY rather than treating the query as zero utilization or terminating the wait.
- With an existing native status of `failed`, the mocked launcher wrote its own `failed` status and issued zero GPU queries and zero worker dispatches.
- With native status `completed`, the GPU-query sequence was: BUSY; idle; idle; idle; BUSY on immediate prelaunch recheck; idle; idle; idle; idle on immediate recheck. The launcher reset stability after the first final recheck, performed nine queries in total, and dispatched exactly one mocked worker after the final successful recheck.
- The mock worker returned the expected live-encoder interface receipt. Completion required that exact receipt status and unchanged protected/source/input records.
- The mock dispatch had `CUDA_VISIBLE_DEVICES=GPU-one`, no `PYTHONPATH`, `OMP_NUM_THREADS=2`, `start_new_session=True`, and `--device cuda:0`. Its requested wait timeout was positive and at most 1,100 seconds.

## Preserved mock console transcript

This is the local mock-test output from 2026-09-26 17:31:35 UTC (2026-09-27 KST), **not an actual server run**. Sub-millisecond timing reflects mocked sleeps and must not be interpreted as a real ten-second GPU stability observation.

```text
waiting_for_native_sequence
failed: RuntimeError('Native sequence failed: None')
waiting_for_native_sequence
waiting_for_idle_gpu
prelaunch_recheck
waiting_for_idle_gpu: Immediate recheck was busy/unknown; restarting stability checks within the same 30-minute budget.
prelaunch_recheck
running
completed
checks: idle preference, PID/memory/utilization boundaries, transient unknown/query failures, native failure stop, prelaunch retry then exactly one mocked dispatch, environment and protected records
actual_processes_launched: 0
pre-v2-constant source SHA256: fef823611fff856b80b5e288cc60d4afbff9959cf9d319e40b9bdfee9dfe0f61
```

## Remaining runtime validation

The mocked `main()` checks used fixed dummy return values for the protected/source/input record functions, so they test comparison and control flow, not the real server files or manifest contents. Live SHA identities, Unix flock behavior across concurrent launchers, the real three-hour/thirty-minute clocks, CUDA UUID binding, and an actual 1,100-second timeout have not been exercised here. Root must audit and freeze the source manifest before dispatch. Real worker completion still requires a valid `PASS_live_encoder_adapter_vlm_gradient_contract` receipt; none is asserted by this document.
