# OE8 input preparation evidence

See ../../docs/OE8_PASTIS_INPUT_PREPARATION_20260927.md for results and limits.

- review_bundle_v0/export_manifest.json inventories 1,777 payload files. The manifest itself is additional.
- Full 80 image/label packets remain on the server. Only three representative packets are local.
- source_verification_v0.json inside the bundle covers all 80 source rereads; episode_independent_audit.json covers exported catalogs/masks plus three local original packets.
- extractor_review.md predates two fixes already applied before execution. episode_builder_review.md reviews the executed builder.
- Do not feed complete public episode JSON to a language model: stable IDs and paths are bookkeeping, not prompt content. The runtime whitelist and observation-acquisition guard remain to be implemented.
- The full 192 development query-pair cohort supports K1/2/4; the fixed 96 cohort supports K1/2/4/8. Do not mix these curves.
- These are public-label-derived examples, not collected expert corrections or model performance results.
