# E5 candidate: equal-budget input comparison trained from initialization

2026-09-25. **Proposal only. This is not a preregistration. No new statistical
thresholds are registered and no E5 preparation, training or inference has run.**
The E5 name is a planning label. The parent reports that E4 completed 2,508
responses with its registered verdict `mixed_or_inconclusive`; this proposal does
not depend on selectively choosing an E4 seed or metric. Any eventual experiment
requires a complete runner/input/scoring contract frozen before its outputs.

## Why this is the next useful comparison

E3 changed inputs of a reader trained on `[A,B,D]`, where `D=B-A`. Its five zero-D
conditions all produced `no`, including `[A,B,0]`. A frozen-input failure does not
show that a reader trained on that input distribution cannot solve the task.
E4 adds difference-block interventions, but also uses the frozen reader.

The next bounded question is whether the explicit difference block helps under
the same training budget, or whether a reader trained from initialization on raw
pairs recovers useful source-label discrimination. The proposed primary contrast
is **trained pair minus trained full**. Later-only and delta-only are secondary
comparisons. This is a development diagnosis on already exposed data, not a fresh
unseen-test claim, and it does not establish a memory bottleneck.

## Four arms and what stays equal

| Arm | Input during every training and evaluation example | Narrow interpretation |
|---|---|---|
| full | `[A,B,B-A]` | Newly trained common reference |
| pair | `[A,B,0]` | Utility of explicitly supplying the difference representation |
| later | `[0,B,0]` | Utility accessible from the second slot plus the unchanged prompt |
| delta | `[0,0,B-A]` | Utility accessible through the difference representation alone |

Each block has 64 tokens of width 768. All arms retain 192 EO tokens, types
`[0]*64+[1]*64+[3]*64`, token positions, prompt, source dates, answer text and
attention-mask construction. Compute D from the unchanged float32 A/B pair before
masking; later must not retain D or another route from A. Do not delete tokens or
change the prompt in only one arm.

Zero raw blocks pass through the same trainable projector and type embeddings.
They therefore become learnable constant prefixes, not absent LLM tokens. Model
parameter count and sequence length remain equal, while available input
information deliberately differs. D already contains information from both
observations; delta is not a single-image or no-history model.

Use **4 arms × seeds 1, 2, 3 = 12 newly initialized projectors**. Old E2 full
checkpoints remain historical context, not the sole reference for newly trained
arms. Within a seed, clone and hash one common initial state_dict for all four
arms, create a fresh optimizer for each, and use the identical ordered batch plan.
Keep the frozen LLM in eval mode with requires_grad=False; gradients must still
flow through its operations to the projector during training.

## Exact inherited training recipe

The source is
`/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/e2_multi_reader_v0.py`.

- Lines 143–174: source loaders and original cache-availability eligibility.
- Lines 184–204: frozen Olmo-3-7B-Instruct in bfloat16; projector with input
  LayerNorm(768), Linear(768,2048), GELU, Linear(2048,H), output LayerNorm(H),
  learnable gain, and four type embeddings. H comes from the actual frozen LLM.
- Lines 208–225: S2 pooling by 4 and S1 pooling by 6, yielding 64 spatial tokens
  per observation, then explicit B-A; original caches are converted to float32.
- Lines 228–247: original chat template, date strings, answer plus EOS, and
  answer-only language-model labels.
- Lines 249–277: torch.manual_seed(seed), NumPy default_rng(seed), new projector,
  AdamW learning rate 1e-4 and weight decay .01, batch 8, and three epochs.
- Lines 279–292: greedy generation, max_new_tokens 16, and binary parsing.

The original E2 preregistration is
`config/e2_multi_reader_prereg_v0.json`. Inherit the explicit training settings;
also capture resolved optimizer defaults and library versions instead of silently
relying on unspecified defaults. There is no new LR, optimizer, scheduler,
class weighting, gradient accumulation, early stopping, epoch selection, prompt
search, or dataset rebalance in this minimal comparison.

There are **4,234 training examples**. Three epochs give **12,702 example
exposures** and `3 * ceil(4234/8) = 1,590 optimizer updates` per model. Each epoch
has 529 batches of eight and a final batch of two. Preserve that final batch.
Training loss/gradient/parameter finiteness and epoch losses must be recorded;
poor learning under this fixed budget is not proof that the input lacks useful
information.

## Current exact population and the historical provenance limit

The best available frozen reconstruction is:

- `artifacts/c0_linear_view_probe_v1_20260925/items.jsonl`
- `artifacts/c0_linear_view_probe_v1_20260925/eligibility_audit.json`
- `artifacts/c0_linear_view_probe_v1_20260925/manifest.json`

All relative research paths in this document resolve under
`/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project`.

| Partition | Landslide | Flood | Total |
|---|---:|---:|---:|
| Train | 518 pos + 518 neg | 1,066 pos + 1,066 neg + 1,066 hard_neg | 4,234 |
| Test | 192 pos + 192 neg | 457 pos + 457 neg + 457 hard_neg | 1,755 |

The reconstructed train contains 27 flood events and seven landslide regions;
test contains ten flood events and two landslide regions. Tile and event/region
overlap checks were enforced in C0. Test landslide support is Hiroshima 186
paired tiles and Indonesia six. The former E2 loader excluded 148 training and
22 test landslide questions for missing caches; these exclusions are explicitly
recorded in the C0 eligibility audit, rather than selected using predictions.

C0 items SHA256:
`e8058fd5193374a1c32968b063e6b67032d8a4f2bc0bf659e490e09816a67a3a`.
C0 eligibility SHA256:
`6c425290c4cfaafbf1aa46be11cd2cfe39e42c5d159d3e2080a4a7e03d6dacf3`.
C0 sorted train-ID digest:
`c19ae9b0d83b4dbfa9707e94d2428ca7e3f1b38434552575087a804d0caea9bc`.
C0 sorted test-ID digest:
`1af5b59921f537f2d9e2242a8a3d5cdcb2f339db52b54fd7e878cb1ed3dda93a`.

**Ordered IDs are a separate contract from these membership digests.** Original
E2 concatenated landslide then flood in each source file's order. C0 sorts by
`(partition,phen,id)` (`code/c0_linear_view_probe_v1.py:159`), so its training rows
start with flood. Reusing C0 order with the same RNG seed does not reproduce E2's
historical batch sequence. For the new comparison, choose and freeze one common
ordered train list, plus the actual seed-by-epoch permutations and batch IDs,
before training. This makes the four new arms mutually fair without asserting
historical exact replay.

Original E2 saved counts and code hashes but not its complete ordered train IDs,
original QA dates/slots, cache bytes, or complete training history. The C0 freeze
certifies the current reconstruction; it cannot retrospectively prove all E2
training inputs. Full E2 test IDs and labels do match the reconstruction in all
three saved seeds.

## Cached data and preparation cost

Server source paths used by E2:

- `/home/work/data/olmoearth/sentinel_qa_train_v0/items.jsonl`
- `/home/work/data/olmoearth/sentinel_qa_v0_1/items.jsonl`
- `/home/work/data/olmoearth/flood_qa_v0/items.jsonl`
- `/home/work/data/olmoearth/sen12_gp_contract/sample_contract.jsonl`
- `/home/work/data/olmoearth/olmo_streaming_dev/single_fp16/<tile>.npy`
- `/home/work/data/olmoearth/kurosiwo_s1_cache/single_fp16/<tile>.npy`
- `/home/work/data/olmoearth/olmo_llm/Olmo-3-7B-Instruct/`

C0 records 3,756 used raw caches and their hashes/stat metadata, totalling
45,740,152,320 bytes (42.60 GiB). Availability was checked at the C0 preparation;
this review did not reconnect to the server or assert their present availability.
Before a new run, check those exact sources against the frozen manifest. If
availability or bytes have changed, stop and record the discrepancy; do not
silently rebuild a different eligible population.

**Do not use C0's `global_features.npz` as the VLM input.** It contains spatial
means of shape `(N,2,768)`. Rebuild the 64-token A/B pairs from the hash-verified
raw caches using the E2 pooling recipe, then archive them and their item mapping.
All 5,989 train+test pairs in float32 `(2,64,768)` require approximately 2.19 GiB
uncompressed. Existing E3 pairs cover only 209 test items, not the training set.
Input preparation and verification are additional to measured training runtime.

## Evaluation populations must remain distinct

| Population | Scope | Proposed role |
|---|---|---|
| Full E2 test | 1,755 questions: flood 1,371 + landslide 384 | Shared evaluation for all 12 newly trained models |
| Full E2 paired flood component | 914 questions from 457 paired tiles, ten events | Proposed primary pair-minus-full source-label BA contrast, equal event weighting |
| Full E2 flood hard negatives | 457 questions, with their actual event support | Separate event/pooled FPR and parse counts; never merge silently into paired BA |
| E3/E4 subset | 209 questions: flood paired 114, flood hard 51, landslide 44 | Descriptive link to the prior frozen-input experiments; not an additional independent dataset |
| C1 same-prompt, quality-controlled primary subset | 902 questions: 445 flood positives + 457 hard negatives, eight events | Fixed complementary check controlling reconstructed event/prompt/date/slot and common label-coverage criterion |

The C1 902 questions are neither the full 1,755 nor E3's 114 paired questions.
They compare different tiles under the same reconstructed prompt. Seven original
positives fail the existing C1 quality criterion, and another five positives are
in two events without hard-negative support. Freeze the existing C1 IDs,
quality/source hashes and strata before new predictions. Do not use this subset
to change the training population. Its preprocessing and support restrictions
must be disclosed rather than described as whole-test performance.

For full paired flood, compute recall/specificity and BA inside each event, then
an equal-event mean. For C1, retain its defined equal-class BA within exact-prompt
stratum, equal-stratum mean within event, then equal-event mean. Report every
seed and event, and a seed-mean paired event contrast if a summary CI is desired;
do not treat three seeds as three times as many independent events. Landslide
Hiroshima and Indonesia remain separate descriptive region results.

The primary candidate is pair-minus-full; later and delta are secondary. Before
launch, decide and document the uncertainty method, primary support, any
noninferiority/success margins, multiple-contrast treatment, and invalid-run
rules in the full reviewed contract. **This proposal registers none of those
new thresholds.** A nonsignificant difference is not evidence of equivalence.
Generate all four models' full-test answers once; the fixed subset reports can
be computed from those saved answers without extra inference.

## Consequential remaining confounds

- Flood `pre_1/pre_2/post` dates are event-date minus 24/minus 12/zero days, not
  verified acquisition times. These approximations are supplied to the encoder
  as well as the prompt (`code/extract_kurosiwo_s1_cache.py:44`,
  `code/e1_flood_qa_v0.py:49`). New training does not repair that provenance.
- Flood positives derive from a post-reference mask; pre/pre `no` assumes event
  absence between the pre slots. Successful discrimination is not independently
  verified emergence of flooding. Positives and hard negatives originally have
  asymmetric label-coverage eligibility; the fixed C1 report addresses part of
  this issue without changing the historical source task.
- Landslide Q1 labels use event-relative pre/post and pre/pre or post/post pairs
  from the source contract (`code/sentinel_qa_gen.py:29–44`). They also do not
  supply an independently reviewed change annotation for every image pair.
- S1 is coupled to flood and S2 to landslide. Two binary prompt templates do not
  establish general natural-language grounding or sensor-independent semantics.
- All ten flood test events and both landslide test regions are already exposed
  in the project. Model/input choices motivated by E2–E4 remain exploratory.
- Under a fixed three-epoch budget, an advantage can reflect representation and
  optimization convenience. A weak pair model does not prove A/B lack the
  information; a strong delta model does not show history is unnecessary.

## What a result could support

If newly trained pair recovers strong performance, the old `[A,B,0]` collapse is
consistent with a limitation of the frozen training/input route rather than an
inability to use paired source information. That is not an identification of the
sole mechanism, because the new model has learned different weights.

If later performs similarly under the prespecified comparisons, this source task
admits a useful second-observation-plus-prompt baseline. If full/pair is better,
historical input or explicit difference representation is useful under this
budget. Neither outcome by itself establishes temporal reasoning, actual event
timing, physical causality, or a need for compressed memory. A scientifically
grounded change/memory claim still needs verified temporal labels and evidence
that its target requires observations unavailable to the single-view baseline.

## Feasible runtime from saved measurements

Saved E2 reader training times (`artifacts/e2_multi_reader_v0/reader_seed*/scores.json`)
are 839.621, 333.561 and 336.631 seconds. Extrapolating their observed minimum and
maximum to 12 training runs gives 66.7–167.9 minutes; repeating the observed
three-seed time composition four times gives 100.7 minutes. The reason for the
slow seed-1 run is not established, so do not label it a known cold-start cost.

Each original E2 reader evaluated 5,649 answers in 528.266, 356.882 or 348.910
seconds. The proposed real-only evaluations contain `12*1755 = 21,060` answers;
a simple throughput extrapolation is an additional 21.7–32.8 minutes. These are
planning estimates for comparable hardware/runtime, not promises or new measured
training results. **About 2–3.5 hours plus input preparation** is a reasonable
single-GPU planning range; 67 minutes is an optimistic training-only lower bound.
The E2 six-model entire wall time (68.1 minutes) must not be quoted as the budget
for twelve new models. No second GPU or concurrency speedup is assumed here.

## Required freeze before any eventual run

1. Exclusive new output directory, actual executed code snapshot, proposed
   config, source QA/contracts, current C0 membership/exclusion audit, and every
   used cache hash. Preserve all E2–E4 artifacts.
2. Ordered train/test IDs; pooling and date/slot mappings; finite archived pair
   arrays; exact prompt/tokenizer outputs; answer token IDs; all 12 model roles.
3. Frozen LLM/tokenizer weights and config; runtime/library versions; projector
   architecture/dtype; one initial state per seed; identical initial state hashes
   across arms; resolved optimizer groups and fresh state; seed/epoch batch IDs.
4. Exact 1,590-step and 12,702-exposure audit per model; finite loss, gradients,
   parameters and projected tokens; fixed final checkpoint only; no score-based
   resume, arm dropping, best-seed selection or silent retry with changed settings.
5. Full 1,755-ID output contract plus explicitly mapped E3 and C1 reports; source
   labels remain source labels; prereviewed parser/invalid handling and independent
   scorer; event support and all seed outcomes retained.
6. Final reviewed analysis/decision contract before model outputs, plus resource
   identity, cooperative GPU lock, timing and checkpoint-to-answer bindings.

This document is a concrete proposal for the next research step. It grants no
statistical registration status and records no E5 execution or result.
