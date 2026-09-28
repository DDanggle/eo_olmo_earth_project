# E3 GPU0 operational amendment (draft for main review)

This bundle does not reselect items, repool tokens, change labels, train, score, or modify the v0 directory. The runner differs from the reviewed v0 source only in four GPU guard literals. The preregistration retains its scientific ID/schema and changes only `compute.gpu` and `compute.outputs`; a separate `amendment.json` records the relocation.

Main must first perform its audited quiescent handoff: verify the own waiting process identity/start time, SIGSTOP it, prove no children/no started runner/no inference artifacts and prepared state, then stop only that waiter and verify termination. Do not touch the unrelated GPU1 job. Preserve queue-before and the handoff record. The helper independently checks old waiter/runner absence under locks, including a possible old runner still hashing model files before updating status.

Upload this entire bundle into a separate directory such as `/home/work/data/olmoearth/code/e3_gpu0_amendment_v1`. Do not overwrite `code/e3_pair_dependence_v0.py` or the existing v0 snapshot. The CLI below assumes the handoff JSON is at `e3_gpu0_handoff.json`; substitute the actual reviewed path.

```sh
env -u PYTHONPATH .venv-master/bin/python -B code/e3_gpu0_amendment_v1/prepare_e3_gpu0_v1.py --runner code/e3_gpu0_amendment_v1/e3_pair_dependence_v0.py --config code/e3_gpu0_amendment_v1/e3_pair_dependence_prereg_v1.json --launcher code/e3_gpu0_amendment_v1/run_e3_pair_when_idle_v1.py --handoff e3_gpu0_handoff.json
```

Preparation creates only a new `e3_pair_dependence_v1` directory. It verifies the reviewed parent manifest SHA, all original frozen source/prepared data hashes, the exact allowed code/config changes, and the copied bytes. It records the parent snapshot and handoff. Model/tokenizer hashes are checked by the unchanged run path before GPU allocation. An interrupted preparation remains incomplete and cannot launch; do not overwrite it.

Launch only the copied launcher from the new output directory, using the project's normal detached `setsid nohup` pattern:

```sh
env -u PYTHONPATH .venv-master/bin/python -B e3_pair_dependence_v1/launcher_snapshot/run_e3_pair_when_idle_v1.py
```

The launcher holds the E3-global, GPU0, and legacy GPU1 cooperative locks. It forwards their file descriptors to the runner with `pass_fds`, so a launcher exit does not release exclusion while its child survives. It checks GPU0 is idle and the runner repeats that check immediately before allocation. It does not allocate on GPU1. These locks cannot prohibit an unrelated noncooperative process from beginning on GPU0 after the check; no other jobs are killed or interrupted.

Interpretation stays gated by the original real-arm reproduction threshold, parse/finite checks, exact coverage and original statistical rules. A GPU-related reproduction failure is invalid, with no threshold relaxation. This is an operational v1 of the same prepared E3 experiment, not a new scientific sample or altered decision rule.
