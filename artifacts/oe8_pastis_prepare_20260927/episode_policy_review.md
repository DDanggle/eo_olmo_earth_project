# OE8 episode policy — adopted implementation, 2026-09-27

This replaces the earlier review suggestions. The implemented contract is in `episode_builder.py`; the initial proposals for 100% purity, border exclusion, opposite-parent training supports, and three unordered pairs were not adopted. This preparation is an engineering catalog, not a completed learning experiment.

## Frozen source roles

Keep all 80 frozen candidate IDs. Within each of t32ulu and t31tfj, hash-sort its 32 patch IDs by `sha256('oe8-bank-v0:' + patch_id)` and reserve the first eight as `source_bank`. The other 24 per parent are `train_pool`; the sixteen t31tfm patches are `dev_query`. Counts are therefore 48/16/16. The split cannot be repaired after reading class coverage.

Source-bank images, masks, objects and corrections are excluded from encoder, connector, selector, utility teacher, replay, and supervised training. They are supplied only as development inference support. Historical foundation-model exposure is unknown. Patch ID separation is enforced; exact footprint separation and globally unique cross-patch parcel identity are not established. No claim of an unseen location, unseen object, or unseen class follows from this catalog.

## Objects, classes, pairs

Support objects have a source-local instance ID greater than zero, a dominant crop class in 1..18 with purity at least 0.95 over the complete instance, and at least 64 pixels in that crop class. The selected support mask contains only the dominant-class pixels inside that instance. Every pixel of the complete instance, including excluded minority pixels, must have valid measured-band inputs on all eight candidate dates. This is nodata eligibility; it does not certify cloud-free imagery. Image-border contact is recorded without exclusion. The object key is `patch_id:instance_id`.

Only eligible `train_pool` objects choose the class inventory. Rank classes with at least four eligible training patches by patch coverage descending and then numeric class ID ascending; take at most four classes and all their ordered pairs, at most twelve. The source bank never changes which classes are selected. Development class histograms in the preparation manifest are ignored. A label-defined counterclass is not yet a human-confirmed visual confusion.

For each class, hash-sort eligible objects deterministically, cap at two objects per patch, and take at most eight. Training support comes from `train_pool` excluding the query patch. Development support comes only from `source_bank` and uses one shared class sequence for all development queries. Class-only sampling means reverse-direction tasks swap exactly the same support objects. K=1/2/4/8 are nested prefixes. No padding, repetition, or threshold relaxation supplies a missing K.

## Query cohorts and input completeness

Keep all 48 training query candidates and all 16 development query candidates before support availability is applied. Cross these with every source-frozen directed pair, including queries where the target is absent. Public catalog construction never consults query target presence.

`strict_no_missing_input_eligible` means all eight image packets are complete. `supervised_training_allowed` means partition `train_pool`. `clean_training_eligible` is their conjunction; therefore a clean development image still has `clean_training_eligible=false`. All three are recorded for the loader and audit, never as selector features. Until a missing-aware adapter exists, the real trainer must hold incompatible inputs; it cannot silently present the remaining subset as the original full correction curve. All 80 actual inputs were reported complete by the parent extractor; this policy also explicitly handles future missing inputs.

The initial observations are positions [2,5], with fixed additional observations [0,7]. These are checked against the prepared `quality_policy.json`. Dates come from manifest `selected_dates` and are cross-checked against NPZ `timestamps` in day/month0/year order. The input NPZ stores all eight candidates, so the real loader must enforce acquisition access; merely storing an acquisition ledger is insufficient if the model already saw unacquired pixels/features/quality masks. All eight support observations count toward support acquisition/encoding cost, with cache reuse stated separately.

## Gold separation and model-visible fields

Input NPZs must contain no semantic, instance, or label-valid arrays. Full source labels are read to construct selected support masks; the model receives only selected masks, never unrelated full semantic maps. Source and query label hashes are checked against the prepared manifest.

Public `episodes_train.jsonl` and `episodes_development.jsonl` are frozen and hashed before development label NPZs are opened. Only then does a separate scorer write target classes, private query-label references, target/counter presence, and valid pixel counts. Query scoring uses source `label_valid`, without an all-eight-image-valid filter. Target absence remains an evaluation case; it never replaces the task or support pair.

An episode file is loader metadata, not an LLM prompt. Only the generic task text, acquired query observations and dates/validity, and selected support observations/masks and dates/validity may reach the model. Episode/base/pair IDs, patch/parent/object IDs, paths, hashes, class mappings, gold paths, and aggregate input eligibility are retrieval/scoring metadata. Stable pair IDs could otherwise encode a target shortcut. `pair_catalog.json`, source-object inventories, preparation class histograms, and `scoring/` are analysis or target artifacts only.

Input references resolve against the prepared root; support-mask references resolve against the episode output root. Both roots and SHA hashes are explicit in the contract. This is an interface/dataflow separation, not an operating-system access-control claim. A later trainer/loader audit must enforce it.

## Coverage and interpretation

Every query×pair base task records its supported K values, including zero-support cases in `coverage.json`. K=8 AUC uses the same K=8-complete base cohort at every K. Lower-K-only episodes remain separately reportable and cannot be mixed into that curve. Development target-presence strata are scoring-only. If foreground is absent everywhere, foreground IoU is undefined while absent-query false positives remain measurable.

Synthetic tests cover frozen partitioning, development-gold metadata invariance, retention of incomplete training/development queries, support/query separation, prefix nesting, exact reverse replay, per-patch caps, missing-K reporting, source-only class ranking, whole-instance nodata checks, and the month-zero timestamp contract. They test preparation contracts, not scientific validity or learning quality.

Readiness stays `scientific_training_ready=false`: parcel/footprint checks, loader acquisition enforcement, actual training/checkpoint reload, and fair baseline comparisons remain outstanding. The supports are synthetic corrections derived from public masks, not paid expert annotations. The existing human budget is unchanged.
