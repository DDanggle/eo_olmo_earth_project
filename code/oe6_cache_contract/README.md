# EO cache dependency contract prototype

Pure Python standard library. `cache_contract.py` defines four cache layers:
preprocessed raw inputs → EO features → projected/code embeddings → LLM K/V.
`test_cache_contract.py` checks dependency invalidation and detached-cache training guards.

Run from this directory:

```sh
python3 -m unittest -v test_cache_contract.py
```

This module **does not execute OlmoEarth/Qwen, inspect actual tensors, prove numerical
parity, compute source hashes, or verify that supplied identities describe reality**.
The runtime caller must supply hashes of the actual artifacts/configuration and call
the guard before reuse. Model parity and current-gradient checks remain GPU work.

`Request.build` snapshots provenance into canonical JSON so mutation of caller
dictionaries cannot retroactively change a request. Ordered window membership,
acquisition times and sensors affect downstream fingerprints. A query suffix is
deliberately excluded from the immutable prefix identity. If that query participates
in EO selection, it must also be given as the query-conditional selector identity;
if it appears in the prefix, it must change the causal prefix identity.

All identity fields are mandatory; use an explicit `not-applicable` identity for a
known absent component, never an empty field. Exact persistent EO reuse accepts
only the declared deterministic evaluation path. This declaration is not a test of
deterministic kernels. All other relevant runtime choices must be represented by
configuration/runtime identities. The training guard concerns detached persistent
artifacts; live intra-step autograd graph reuse is outside this module.

Any approximate/merged/compressed entry marked with fidelity other than `exact`
fails closed as unsupported. Such methods need their own error/grounding evaluation.
Schema mismatches and malformed entry dependency counts also fail closed.

The implementation has not been wired into the GPU worker or published to the
research repository. Unit results are **dependency-contract results only**.
