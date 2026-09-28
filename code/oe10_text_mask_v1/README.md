# Text-conditioned mask baseline v0

Prepared engineering code, with no actual Qwen/OlmoEarth forward or new GPU use
in this implementation step. Small CPU substitute tests only exercise wiring.
This is a standard conditional dense baseline, not a novelty claim or evidence
that descriptions improve localization.

The pinned P2 feature extractor retains the original 2 acquired query dates,
all 8 dates per support, K object prototypes and B0/B2 encoder freeze policy.
A frozen Qwen text-only user prefill supplies its last valid token hidden vector
to a new dense head alongside EO query/positive/counterexample features.
The old connector and language CE are unused. No RGB, EO slots, assistant answer,
query label, identifier or scoring metadata enters that text prefill.
This differs from P2; all three language conditions must be compared in this
same baseline. No comparison against P2 scores establishes a language benefit.

Model context has exactly instruction/positive/counterexample keys. The
instruction equals contracts.PROMPT. IDs, sources, class IDs and conditions
remain outside model input. Generic, names-only and names+facts use identical
architecture. Removing or swapping facts is a diagnostic. A context builder
must enforce provenance; the string boundary cannot prove that arbitrary prose
contains no leaked annotation. Public crop summaries are not human corrections.

Text is one unpadded request; the actual chat template uses a user turn only and
no generation prefix. The 1024-token limit fails rather than silently truncates.
Lengths/costs are recorded; conditions are not asserted compute-matched. Last
valid-token pooling is tested for both padding directions, while the production
prefill rejects padding. No generated explanations or actual semantic fidelity
are tested by this adapter.

EO feature cache rules are inherited from the immutable P2 copy. Default text
cache is disabled. If enabled, keys include rendered text, token IDs/mask and
reader identity; ordinary reader tensor version mutations invalidate it. Do
not mutate parameters using .data or change tokenizer files in a live process.
Checkpoints preserve arm, source, reader, EO/normalization and architecture
identity, plus head and B2 encoder state. Reader weights are excluded. They
must be initialized from the verified frozen files before restore. No optimizer
or full worker resume is implemented here; the future pilot must retain its own
optimizer/RNG/cost ledger and perform a fresh-process resume test if training.

Run standard-library contracts locally:

    python -B -m unittest discover -s code/oe10_text_mask_v0 -p test_contracts.py -v

Run all lightweight CPU tests in the existing torch environment:

    env -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B -m unittest discover -s code/oe10_text_mask_v0 -p 'test_*.py' -v
    env -u PYTHONPATH python -B code/oe10_text_mask_v0/check_connection.py --import-only

`check_connection.py --execute` is prepared for a later, separately reserved
GPU1 connection check after P2 terminates and is audited. It verifies expected
reader/EO/source/prepared/context hashes before loading actual models, opens one
train K1 case, compares five text conditions, checks mask-only gradient boundaries,
and performs same-process save/load. Its identity manifest and durable budget
reservation must be frozen by the caller before any execution. CLI alone is not
the research scheduler: the caller must check GPU availability, enforce an outer
wall-clock timeout, exclude concurrent owned jobs, preserve source snapshot hashes
and debit actual/retry GPU time from the cumulative 2-hour ledger.

Required identity-manifest fields: reader_files_sha256 (all JSON/Jinja/TXT/model
safetensors), eo_files_sha256 (public config.json/weights.pth),
eo_source_files_sha256 (including model_loader, normalize, constants),
prepared_manifest_sha256, contexts_sha256. Mandatory reservation fields:
status=reserved, p2_finished_and_audited=true, purpose=text_mask_connection_check,
reserved_seconds<=1800, prior_gpu_used_seconds>=0, whose sum is <=7200.
These are necessary inputs, not evidence that authorization/audit has happened.

Remaining limitations: actual Qwen API/forward, text sensitivity, EO gradients,
GPU memory/time, fresh-process resume, training effectiveness, new-region transfer,
expert agreement and free-form explanation quality all require their own evidence.

## v1: externally pinned complete episode identity

This copy preserves v0 model/head/features, immutable P2 modules, existing tests,
thresholds and runtime v2 payloads byte-for-byte. Only the execution identity
boundary is extended. The v0 source manifest is retained separately; the v1
source manifest records its SHA and contains path/bytes/SHA for every v1 file.

Every mode now requires `--source-manifest-sha256 SHA` from an independently
frozen caller record. It hashes this full source tree before contracts, torch or
model imports. Generated `__pycache__/*.pyc` files are excluded; symlinks and
unlisted source files are rejected. Python must already execute the entrypoint
and import the standard-library verifier to perform this check. This is a
replacement/error guard, not a defense against malicious runtime code.

`--execute` additionally requires `--identity-manifest-sha256 SHA` and the existing
identity manifest must contain `episodes_export_manifest_sha256`. The digest
pins `--episodes/export_manifest.json`, which in turn verifies the size/hash of
every payload (not only the selected case) and exact file coverage. Required
payloads include episode_contract.json, episodes_train.jsonl and
scoring/scoring_train.jsonl. All root/ancestor/payload symlinks, absolute paths,
path traversal, noncanonical or duplicate paths, omissions and extra files fail.
The guard reads scoring bytes for hashing but never parses annotation contents.
It runs before EpisodeLoader initialization or model imports.

`--identity-only` runs these same source/episode guards using only the Python
standard library and emits a JSON receipt; no reservation, torch, model loading,
label interpretation or GPU is involved. It is mutually exclusive with the
other modes. Example (fill digests from the saved external pin record):

    python -B code/oe10_text_mask_v1/check_connection.py --identity-only \
      --source-manifest-sha256 SOURCE_SHA \
      --identity-manifest PATH_TO_PINNED_IDENTITY_JSON \
      --identity-manifest-sha256 IDENTITY_SHA \
      --episodes PATH_TO_CANONICAL_RUNTIME_V2

Run new boundary tests plus unchanged stdlib contracts:

    python -B -m unittest discover -s code/oe10_text_mask_v1 -p test_input_identity.py -v
    python -B -m unittest discover -s code/oe10_text_mask_v1 -p test_contracts.py -v

The full reader/EO/prepared/context identity, fixed sample, P2 termination/audit,
external budget reservation, idle GPU1, outer timeout and duplicate-run exclusion
remain the caller's prerequisites. This change is not a GPU launch approval.
No optimizer, training scheduler, protocol or new performance metric is added.
A pinned source tree and its inputs must remain immutable during the run; the
guard does not lock the filesystem against concurrent modification.
