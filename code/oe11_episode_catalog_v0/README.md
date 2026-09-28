# OE11 fixed-class episode catalog adapter

CPU preparation only. This wrapper imports the SHA-pinned OE8 helper functions and never calls their 80-patch role assignment, class ranking, or main. It does not train, score model predictions, download data, or use a GPU.

Inputs: `manifest.jsonl`, `selection_policy.json`, input-only NPZ files, and separate annual-label NPZ files under `--prepared-root`. All three parents (`t31tfj`, `t31tfm`, `t32ulu`) must each have `train_pool`, `source_bank`, and `calibration` records. Counts are arbitrary positive counts; optional `caps_per_parent` in the input policy is enforced as a maximum, not misinterpreted as a required count. Actual parent/partition counts and whether caps were filled are recorded. **t31tfm is now a source region. Calibration is not unseen-region evaluation.** t30uxv is rejected.

The prepared partition `calibration` maps to the helper's `dev_query` only in the in-memory role map. Prepared files are unchanged. The separate OE11 loader adapter is needed for the old loader's partition alias.

```sh
python3 episode_catalog.py \
  --prepared-root /absolute/path/to/prepared \
  --out /absolute/path/to/new_catalog_pooled \
  --base-builder /absolute/path/to/code/oe8_episode_prepare_v0/episode_builder.py

python3 episode_catalog.py \
  --prepared-root /absolute/path/to/prepared \
  --out /absolute/path/to/new_catalog_cross_parent \
  --base-builder /absolute/path/to/code/oe8_episode_prepare_v0/episode_builder.py \
  --train-support-parent-policy cross_parent
```

Each output directory must be new and outside the prepared packet. Base source SHA256 must be `64c6df1fe10b40da8cc076286a6184be05ec0db261a4462af0548aaba8eff6d6`.

Classes are fixed to `[1, 3, 8, 14]` in both target and counterexample roles, giving 12 directed pairs. The adapter never adds winter wheat or selects classes using performance. Source eligibility, deterministic supports, reverse-pair symmetry, nested K=1/2/4/8, and query dates at indices 2/5 are inherited from the pinned helper. Additional fixed acquisition positions remain 0/7. Support objects require >=64 class pixels, >=.95 class purity, whole-instance input validity at all eight observations, and <=2 objects from one patch per class.

`pooled` training uses train_pool supports excluding the query patch. `cross_parent` uses only train_pool supports from the other two parents, while keeping the same query candidates and requested task pairs. Development uses the global source_bank in both modes, so development catalog hashes should be equal. The modes deliberately reuse legacy task IDs: **identify a task by catalog hash plus episode ID**, not episode ID alone. Different support availability can leave different K subsets; paired comparisons require their common pre-score scope.

`pair_catalog.json` always describes the full requested 12 pairs. Public episodes are emitted only at K values with enough real support. Shortages produce a valid catalog with explicit pair/K coverage and an incomplete intended-subset flag, not padded/replaced supports or a failing process. Choose a common K scope using availability before model scoring. `training_ready` remains false because catalog construction alone does not validate a training launch, loader, or spatial independence.

Source masks are separately stored under `support_masks/train_pool` and `support_masks/source_bank`. Calibration masks are not exported as supports. Public catalogs and source artifacts are hashed and frozen before this invocation opens calibration gold; scoring maps are then generated separately in `scoring/`. Calibration-label mutation tests verify public-catalog invariance. This is **not** a claim that labels were never seen: earlier CPU availability probing occurred, and train query labels may supply supports for other train queries.

Metadata and class/scoring maps are not model features. Input packets contain all eight observations; the loader must enforce acquisition access. Supports use eight dates and that information cost must be counted. Missing-input queries remain cataloged with explicit eligibility flags; the strict trainer must hold them until a missing-aware adapter exists. Annual labels do not establish per-date change or cloud visibility. This builder checks unique patch references, not exact footprints, upstream block assignments, or cross-patch parcel independence.

Outputs include `episodes_{train,development}.jsonl`, `pair_catalog.json`, `source_objects.jsonl`, `roles.jsonl`, `coverage.json`, two pre-scoring hash records, partitioned support masks, `scoring/scoring_{train,development}.jsonl`, and `episode_contract.json`. The contract includes `scoring_file_sha256` for lazy explicit scoring access.

Synthetic CPU tests (no real dataset used here):

```sh
OE8_BASE_BUILDER=/absolute/path/to/pinned/episode_builder.py \
  python3 -m unittest discover -s /absolute/path/to/oe11_episode_catalog -p 'test_*.py' -v
```

Eight tests passed: three-parent arbitrary-count scope, forbidden scope/partition rejection, pre-import source pin, full NPZ catalog and nested/reverse supports, calibration gold read order and mutation invariance, lower-K shortage preservation, cross-parent support restriction plus unchanged development tasks, policy conflicts and non-overwriting output.
