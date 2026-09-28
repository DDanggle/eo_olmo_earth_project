# E4 difference-block experiment — prepared before E4 outputs

The E3 collapse generated this hypothesis: zeroing D produced `no` throughout five
arms, including `[A,B,0]`. This is sequential exploration on exposed data.

**D=B−A still contains both observations.** Successful `[0,0,D]` would support a
route through the difference block, not the claim that earlier observations or
historical information are unnecessary.

Files ready for integration:

- `e4_delta_probe_prereg_v0.json`: four fixed arms, pinned E3 v1 inputs/models,
  209 unchanged items × 3 seeds × 4 arms = **2,508 generations**, no training.
- `e4_delta_probe_v0.py`: pure `transform_pair(pair, arm)` and
  `score_run(rows, items, references)`; no files, GPU or models are opened.
- `test_e4_delta_probe_v0.py`: eight synthetic tests for controls and decisions.

The four arms are `real`, `delta_only`, `delta_sign_flip`, and
`delta_feature_permute`. The latter uses the single fixed feature permutation
`j -> (j+257)%768`; it preserves each token's feature-value multiset and preserves
mean/variance/L2 within numerical tolerance. The original A/B blocks remain
unchanged in the sign/permutation arms. Neither sign nor permutation receives a
new physical-scene gold label.

The delta-only primary rule retains the prior BA/CI margins: real BA ≥ .60,
delta-only BA ≥ .60, and event-paired delta CI lower ≥ −.05 in at least two seeds.
All ten E3 flood events remain; do not import C1's support exclusions. Secondary
real/delta-only decision agreement ≥ .99 is descriptive and counts unparsed
answers as nonagreement. Sign/permutation results have no additional success
threshold or adaptive arm search.

Runner integration requirements:

1. Create a new exclusive `/home/work/data/olmoearth/e4_delta_probe_v0/` directory;
   preserve every E3 artifact. Copy the frozen E3 items and pair archive unchanged.
2. Use the E4 arm list for **all 209 items**. The old `items.allowed_arms` field is
   E3 provenance only; E4 hard negatives also receive all four arms.
3. Reuse the audited E3 v1 model/projector loading, dtype, token types, prompts,
   decoding and finite checks. Freeze the new runner, pure module and prereg hashes.
4. Build `references = {'e2_real': saved_e2_real, 'e3_real': e3_real_answers}` with
   string seed keys and exact `id -> parsed` dictionaries. Check byte hashes from
   the plan before use. Run all real seeds first; pass .99 reproduction separately
   against each reference in every phenomenon plus parse ≤ .01 **before generating
   any intervention output**. The pure scorer verifies the final reproduction again
   but cannot enforce execution order itself.
5. Output each row with seed, arm, id, tile, cluster, phen, kind, unchanged
   source_gold, explicit `transformed_gold:null`, answer_raw and parsed yes/no/null.
6. Keep the same inherited cooperative GPU locks and record GPU UUID. Never change
   an E3 code snapshot or reuse its output directory.
7. Treat invalid runs as invalid; preserve partial artifacts. No exclusion,
   threshold relaxation, per-item permutation or fitting follows automatically.

No E4 inference, training, upload, or model-result inspection was performed while
preparing these files. Only existing E3 provenance files were hashed to pin inputs.
