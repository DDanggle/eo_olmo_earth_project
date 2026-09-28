# OE11 fixed request context bundle

This bundle contains 24 model contexts: 12 directed pairs of classes1/3/8/14,
each with `names_only` and `matched_knowledge`. It opens no imagery, query labels,
runtime outputs or model assets. The generator uses the standard library and
imports the existing pure-Python `code/oe10_text_mask_v2/contracts.py` only.

Runtime routing:

```python
context = contexts[episode["pair_id"] + ":" + condition]
# Pass only this three-field context value to the model.
```

`contexts.json` values contain only `instruction`, `positive`, and
`counterexample`. Pair/class/source IDs, conditions, SHA values, and source
records are in `provenance.json`, separate from model input. Pair IDs reproduce
the existing OE8 namespace. Reverse pairs swap the exact positive/counterexample
role strings. `matched_knowledge` copies four existing card fields verbatim,
joined by newlines: concept name, claim text, inspection hypothesis, transfer
limit. No factual text was rewritten or added.

The provenance `source_sha256` hashes the canonical saved JSON **source record**,
not the remote website contents. Original card-file hashes and card-record
hashes are recorded separately. No web source was re-fetched in this step.

Verification: 24 actual v2 contract passes, 24 unique IDs/payloads, 24 reverse-role
checks, all source IDs resolved, and rejection of provenance inserted into model
context. This is construction verification, not tokenization or model efficacy.

Fixed role texts can identify classes without semantic reasoning. These are
public agronomy summaries, not parcel-specific regional knowledge or new human
corrections. Human responses and patch-level expert annotations remain zero.
