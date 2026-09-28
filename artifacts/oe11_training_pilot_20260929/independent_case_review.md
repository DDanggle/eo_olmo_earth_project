# OE11 train pilot cases v1: independent review

2026-09-29. Read-only local review of `code/oe11_pilot_prepare_v1/prepare_cases.py`, the actual `train_pilot_cases_v1.json`, and the previously exported prepared manifest. No server, GPU, model, or label-array access.

**No blocking defect found in this frozen case packet.**

Verified by assertions:

- Preparation script SHA equals the case receipt; exported manifest SHA equals the pinned input SHA.
- Exactly 12 distinct query patches; all are `train_pool`, `supervised_training_allowed=True`, strict and clean eligible.
- Exactly four target classes `[1,3,8,14]` × three strata (target-present, confuser-only, neither). Counter classes remain distinct and within the fixed class scope. Case target/counter/valid pixel counts agree with the pinned manifest's class counts.
- Actual parent distribution is 4/4/4 over t31tfj/t31tfm/t32ulu. This is the actual frozen selection outcome. The greedy least-represented-parent preference does not impose a universal hard maximum of four for arbitrary future inputs.
- Twelve base episodes each have K1/K8, giving 24 unique episode IDs. Both 24-step halves of the 48-step order contain all 24 IDs exactly once; shuffle regeneration with seed290929 matches exactly. Each K receives24 optimizer exposures. Checkpoint split24 is a complete first cycle.
- `fit_episode_ids` contains exactly the selected 12 K1 cases. Pre/post fit findings must be described as K1 training diagnostics, not K8 fit validation or generalization.
- Pair IDs recompute correctly from the fixed OE8 directed-pair namespace.

Preparation reads the pinned training scoring map and hashes selected training label files; it does not open development/final scoring files or label arrays. The common prepared manifest contains metadata for all partitions, so avoid claiming the manifest itself was never exposed. Supervised train-case selection from labels is explicit and appropriate for this engineering diagnostic.

Minor implementation limitations, nonblocking for this exact packet: role/parent balancing and unique-query selection are preferences with fallback, and the label-valid threshold uses `int(.95*128*128)` (floor). All actual selected cases exceed the true95% threshold, are unique, and achieve4/4/4. A future generic protocol claiming hard balance or a strict95% threshold should enforce those properties explicitly rather than rely on this outcome.

The packet deliberately treats loss change as diagnostic, not a pass gate. Computational/resume success must remain separate from trainability, language semantics, performance, or CVPR claims. Cases include24 tasks but only12 underlying images; they are not24 independent observations or human corrections.
