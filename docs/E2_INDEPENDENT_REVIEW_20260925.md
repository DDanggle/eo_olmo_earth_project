# E2 / MS-157 independent audit — 2026-09-25

The registered `reads_both` verdict is arithmetically reproducible. It is evidence
that the fixed projector-plus-frozen-LLM system uses embedding information for
the constructed binary tasks. It does not establish stable landslide reading
across regions, event-semantic understanding, temporal comparison, grounding,
memory necessity, or previously unexposed flood test performance.

This audit performed no model execution, retraining, label edits or original
file changes. `audit.py` and `audit.json` reproduce and describe existing outputs.

## Verified

- 23 locally available files match `SHA256SUMS`; the six `projector.pt` files
  listed in that manifest are not local and were not verified from their bytes.
- Current E2 and imported E0 source hashes exactly match the execution manifest.
- 22,212 answer rows have complete expected arm counts, no duplicate question
  IDs within an arm, and identical question/control identities across seeds.
- Real rows use their own gold; within/cross-tile rows use the donor's gold;
  within-tile donors share tile/fold/phenomenon and have opposite gold;
  cross-tile donors are the deterministic same-fold opposite-gold selections.
- All raw strings are `yes`, `no`, or `No`; parser outputs match and failures=0.
- Stored balanced accuracies, d values and all registered tile-bootstrap CIs
  reproduce independently within 1e-12.
- The gate remains landslide 2/3 seeds, flood 3/3 seeds, hence `reads_both`.
- The prereg commit time 01:32:29 KST precedes manifest start 01:32:55 by 26s.
  This is Git chronology evidence, not an external immutable registration.

| Phenomenon | Seed | Reader BA | Reader minus blind BA | d_swap | Registered pass |
|---|---:|---:|---:|---:|---|
| Landslide | 1 | .7891 | +.2839 | .5938 | yes |
| Landslide | 2 | .6823 | +.2005 | .3490 | yes |
| Landslide | 3 | .5260 | -.0573 | .0260 | no |
| Flood | 1 | .8025 | +.3025 | .6039 | yes |
| Flood | 2 | .7976 | +.2976 | .5864 | yes |
| Flood | 3 | .7872 | +.2702 | .5536 | yes |

## New material limitations

### Landslide is effectively one-region evidence

The 192 paired tiles are **186 Hiroshima and only 6 Indonesia**, not comparable
coverage of two holdout regions. Hiroshima accounts for 96.875% of paired tiles.

| Region | Pairs | Seed 1 BA / d_swap | Seed 2 BA / d_swap | Seed 3 BA / d_swap |
|---|---:|---|---|---|
| Hiroshima | 186 | .8011 / .6183 | .6855 / .3602 | .5215 / .0161 |
| Indonesia | 6 | .4167 / -.1667 | .5833 / .0000 | .6667 / .3333 |

Indonesia is descriptive only at n=6. The two successful seeds have no positive
Indonesia swap contrast. Do not describe the pooled pass as stable cross-region
landslide reading. Both regions were already exposed in earlier QA work.

### Flood events were already used in prior research

All 10 E2 flood test events appear in the 2026-09-09 KuroSiwo updater evaluation
(`artifacts/streaming_review_20260909/summary.json`,
`kuro.event_delta_gru_minus_teacher_seed_mean`). Those prior evaluations used
1,999 test tiles per decoder and reported event-level results.

The statement “never used before / 처음 쓰는 10사건” is incorrect at project scope.
Use “events disjoint from this E2 training, already exposed during prior research.”
The metadata split is event-disjoint; this audit did not independently reconstruct
the entire E2 training input list from remote files.

### Event uncertainty is larger than the registered tile uncertainty

Flood has 457 positive/within-tile-negative pairs plus 457 hard-negative tiles.
Event 1111013 supplies 261/457 pairs; 1111007 supplies 101/457. Together they
account for 79.2% of the paired-tile contrast. Other event groups have as few as
1, 2, 3, 5, or 6 positive pairs.

Post-hoc event-cluster bootstrap, 10,000 draws, resampling all paired rows of a
selected event together, keeps the tile-weighted estimand but broadens uncertainty:

| Seed | Registered tile CI for d_swap | Post-hoc event CI | Event CI for reader−blind BA |
|---|---|---|---|
| 1 | [.5558, .6499] | [.4816, .9138] | [.2492, .4532] |
| 2 | [.5361, .6346] | [.4551, .9023] | [.2346, .4464] |
| 3 | [.5011, .6061] | [.4305, .9231] | [.2118, .4049] |

The flood result remains positive in this diagnostic; the registered verdict is
not changed. These intervals are post-hoc and do not turn exposed events into a
new independent benchmark. Per-event tables are in `audit.json`.

### Seed 3 is failed label discrimination, not a proven training crash

Landslide reader seed 3 predicts yes on 64/384 real questions. Its paired real
outcomes (prediction on positive embedding | negative embedding) are:
`yes|no 33`, `no|yes 23`, `no|no 132`, `yes|yes 4`.
Under swap they become `yes|no 30`, `no|yes 25`, `no|no 135`, `yes|yes 2`.
Its contrast is near zero, and BA trails the matched blind seed by .0573.

Swap predictions still match the same donor's real prediction in 98.70% of rows
(seed 1/2: 99.22%). This is consistent with embedding-dependent but poorly
label-aligned behavior, not proof that the model ignores embeddings entirely.
All zero-embedding answers are no. Flood remains strong under the same seed.
Mixed-task imbalance is a possible cause, not identified by these outputs.

## Remaining scientific interpretation

- A within-tile swap is intentionally inconsistent with the source prompt's
  dates. Donor-gold scoring is correct for this intervention, but this diagnoses
  dependence on the supplied embedding pair, not successful adherence to the
  dates named in the source question.
- Current yes/no labels are linked to pre/post slots and post-event masks.
  Flood dates in prompts are -24/-12/day-event approximations, not verified
  acquisition timestamps. Flood-positive selection uses >=2% of valid mask
  pixels; unlike hard negatives, positives do not require >=90% valid coverage.
  Actual positive validity distribution is not saved in E2 outputs.
- Low hard-negative FPR argues against an unconditional “post slot implies yes”
  rule. It does not exclude a post-image-only flood-state detector. Hard-negative
  and within-tile-negative event distributions differ; their pooled FPR gap is
  not itself a matched causal contrast.
- Sensor and phenomenon are perfectly coupled (S1=flood, S2=landslide), and the
  wording is one binary template. The study does not isolate phenomenon-specific
  interpretation from generic event-related appearance or text-task routing.

## Reproducibility gaps to close in new runs, not by rewriting this run

The manifest contains source-code hashes and class/arm counts, but not full
input-item/embedding hashes, ordered train/test IDs, projector-to-answer hash
bindings, LLM/tokenizer revision hashes, or a training-status/loss history.
Answers omit source/donor dates, slots and event IDs; event analysis here joins
an older locally retained metadata export whose hash is recorded in audit.json.

`run()` rewrites `manifest.json` on every start, and skips any directory that has
`scores.json` without checking the old input/weight/config hashes. The prereg's
“training crash => invalid” is not durably enforced across resumes. There is no
evidence of an actual inconsistent resume in the supplied artifact; this is a
limit on what this snapshot can certify.

The earlier known non-finite test tile ks_05418 is absent from these E2 answers.
That does not certify all E2 input arrays finite, because run-time finite checks
and input-array hashes are not recorded.

The ledger's “training run ~14 minutes” applies to reader seed 1 (839.6s).
Other five training runs are 331.5–336.6s (about 5.5 minutes). The six training +
evaluation durations sum to 4,064.35s, consistent with the 4,087s wall-clock span
plus loading/probe overhead. Different elapsed time is not evidence of unequal
optimization steps; all declared runs use the same epoch/batch budget.

## Suitable next control

Before another retraining, retain the current projector seeds and fixed labels,
and compare full pair with pre-only / post-only / delta removal or replacement
under a newly frozen exposed-development protocol. Report Hiroshima/Indonesia
separately and flood by event; preserve seed 3. Hash actual checkpoint and input
snapshots. Those tests address what information carries current performance;
they do not rehabilitate MS-154, prove a memory bottleneck, or establish damage.
