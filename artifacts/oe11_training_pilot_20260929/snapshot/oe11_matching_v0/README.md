# OE11 minimal role-conditioned object matching head

Status: implemented and **13/13 synthetic CPU tests passed** using existing local
PyTorch 2.9.1/Python 3.12 packages. No installation, server access, GPU, actual
EO/Qwen weights, real data, training experiment, or performance measurement.
Frozen OE10/P2 files were read and hashed; none were modified.

## What changes

The unchanged OE10 `features()` returns query `[1,32,32,D]` and positive/confuser
objects `[1,K,D]`. It still averages dates in EO token space, mask-pools each
support object, uses **two query dates/eight dates per support**, and preserves
B0 frozen/B2 trainable EO behavior and cache rules. This module removes the
subsequent mean over K: each query location selects individual support objects.

For each role, a projected text vector is added to the projected query **before**
the dot product with object keys. Thus text can change relative object weights;
adding one scalar after the dot product would cancel inside softmax. Weighted
objects and image–object/image–text interactions feed one shared role scorer.
At K=1 weights necessarily equal one, but the image–text score route remains.

A separate query-based background scorer allows both concepts to be absent:

`target_logit = positive_score - logsumexp(counter_score, background_score)`.

This is the binary target logit induced by positive/counter/neither scores.
Swapping both role texts and objects swaps role probabilities; they are **not**
forced to be complements. The sum of the two directed target probabilities is
at most one. This constraint assumes distinct mutually exclusive target/confuser
concepts, as in the current annual crop classes; do not reuse it unchanged for
overlapping attributes or multilabel concepts.

The first draft used a pure positive-minus-counter score. Root review found it
could not reject both absent roles (a material case in the prepared data). That
form was replaced before any data/model execution. No improvement claim follows
from fixing this representational constraint.

## Integration

Files are deliberately outside the research repo pending root integration:

- `matching_head.py`: standalone `torch.nn.Module`; batched features supported.
- `episode_adapter.py`: explicit factory over hash-pinned, unchanged OE10 v2.
- `test_matching.py`: synthetic head and toy encoder/reader wrapper tests.
- `run_cpu_tests_local.py`: optional existing-cache local environment helper.

Example in a new training source snapshot (do not patch frozen OE10 sources):

```python
from episode_adapter import make_episode_model_class
Episode = make_episode_model_class(PATH_TO_FROZEN_OE10_TEXT_MASK_V2)
model = Episode(
    encoder, arm, device, qwen_path,
    reader_identity=verified_reader_and_tokenizer_identity,
    eo_identity=verified_encoder_and_normalization_identity,
    max_text_tokens=1024,
)
result = model(model_input, context, target, return_details=True)
result['loss'].backward()
```

Constructor arguments follow `TextConditionedEpisodeModel`; use existing
explicit `qwen_model`, `processor`, and sample arguments when already loaded.
The launcher must verify actual weights/tokenizer/normalization and input
catalog/scoring hashes. Merely supplying identity dictionaries does not verify
files. The factory checks five frozen Python dependencies. Freeze/hash these
new implementation files in the eventual launcher source manifest as well.

Only `instruction/positive/counterexample` context fields are accepted, and the
instruction remains pinned. Both role concepts use the **same isolated context
template**, each occupying its `positive` text slot with the same neutral
counter text. The actual positive/confuser meaning is routed by the head's role
axis. This reuses v2's safe frozen-Qwen text prefill, token limit, last-valid-token
pooling, cache identity and mutation invalidation without modifying its source.
It is a minimal serialization, not verified agronomic reasoning. No query gold,
class ID, source ID, assistant answer or coverage supervision enters prefill.

This wrapper needs up to **two separate text prefills**, each bounded at 1024
tokens with no truncation. Actual tokenizer counts and total prefill cost must
be checked before a real run. Name-only and name-plus-facts comparisons must
use this same role serialization and head. Do not equate compute with the older
one-prefill global-text baseline; log token counts, cache use and actual cost.

`trainable_parameters()` and identity-checked trainable state save/load are
inherited; the checkpoint identity adds the new architecture, head dimensions,
role serialization and pinned dependencies. Save state dictionaries, not the
dynamically generated class object. Same-process new-instance restoration is
tested; fresh-process optimizer/RNG resume with actual weights is not.

No VLM language generation or language loss runs here, and EO/RGB does not enter
the Qwen text prefill. This is a language-conditioned EO mask model component,
not evidence of a complete VLM's explanatory capability.

## Tests and evidence

In a normal CPU PyTorch environment:

```sh
OE11_BASE_ROOT=/path/to/frozen/oe10_text_mask_v2 python -m unittest discover -s . -p test_matching.py -v
```

Current local environment command (no package installation):

```sh
/Users/dongdong/.local/share/uv/python/cpython-3.12.10-macos-aarch64-none/bin/python3 /private/tmp/oe11_matching_head/run_cpu_tests_local.py
```

Validated K=1/2/4/8, batch independence, object permutation equivariance, role
exchange, masked/padded objects, query/text effects on matching, both-absent
rejection representability and background gradient, binary mask loss gradients
to toy EO and head, B0 freeze/Qwen freeze, gold-to-logit independence, dependency
hash rejection, checkpoint identity/load, text-cache mutation and token limits.

The first run's one failure was a test accounting error: B2 gradient
checkpointing repeats five forward encodes during backward. The test now checks
34 forward patch-date encodes and 68 after backward with five recomputations.
Both `cpu_test_receipt_v0_failed.json` and corrected `cpu_test_receipt.json` are
preserved. These are execution-cost assertions, not speed measurements.

Next real integration gate: actual frozen reader/tokenizer role-prefill check,
actual EO forward/backward and background/role diagnostics, then an explicitly
budgeted learning/resume pilot on the already prepared two-class training pack.
Performance comparisons and new-region/semantic claims remain untested.
