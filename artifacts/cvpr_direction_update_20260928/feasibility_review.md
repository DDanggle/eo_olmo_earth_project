# Current-data feasibility review: region-aware expert definitions

2026-09-28. Read-only local audit by text_mask_path. No server access, GPU use,
new image/label opening, source edit or frozen-protocol change. Counts below were
recomputed from already recovered manifest/catalog/context metadata. This is a
proposal to the root agent, not a new preregistration or permission to resume the
paused automation.

## Verdict

The current assets can test whether a language-conditioned mask head works and
whether crop-name/fact vectors help this small closed-set development task.
They cannot yet identify regional-knowledge reasoning, expert correction transfer,
or understanding of temporal evidence. The next useful addition is **image-linked
correction statements whose validity varies within the same crop and question**,
not more copies or paraphrases of a crop-level knowledge card.

The existing 08:58 record reports a successful real-weight connection check:
text affects masks, B2 EO gradients reach 212 tensors, same-process restoration
matches, and optimizer updates remain zero. Those are useful engineering results.
P2 finished only 5/6 runs, all five failed the existing stability criterion. This
review does not promote either outcome into a method/performance claim.

## Verified effective sample size and confounds

| Existing asset | Actual quantity | What it supports |
|---|---:|---|
| Prepared PASTIS | 80 patches: 48 train, 16 source-bank, 16 previously exposed dev | A small development substrate |
| Train geography | t31tfj 24 patches; t32ulu 24 | Two source parent tiles, not 48 independent regions |
| Bank geography | 8 patches per source parent | Separate support bank, with strong class imbalance |
| Development geography | t31tfm 16 patches | One exposed development parent |
| Expansion catalog | 384 = 48 query x 2 target directions x 4 K values | Repeated tasks, not 384 independent images |
| Expansion supports | 221 copied masks; 243 eligible train objects in source inventory | Synthetic support, not expert interactions |
| Query observations | 2 of 8 prepared dates, positions 2 and 5 | No observation of the six unacquired query frames by current model |
| Support observations | 8 dates per positive and negative object | 16K support date instances plus 2 query instances |
| Expansion contexts | 5 conditions but only 7 distinct input strings | Closed set of language codes |
| Human responses | 0 | No measured expert consistency or annotation efficiency |

Each source parent has exactly one selected eight-date schedule, shared by all
24 training patches. Acquired query dates are 2019-01-25/2019-07-04 for t31tfj and
2019-02-14/2019-07-24 for t32ulu. Thus date-pair identity perfectly identifies
source parent in this sample. The model receives dates, even though parent IDs
and centroids are excluded. A date-only/location-prior diagnostic is necessary
before calling an improvement regional or phenological understanding.

Counts of train patches containing each annual class, from existing manifest:

| Parent / 24 patches | Meadow | Winter wheat | Corn | Winter barley | Grapevine | Leguminous fodder |
|---|---:|---:|---:|---:|---:|---:|
| t31tfj | 23 | 3 | 3 | 7 | 14 | 13 |
| t32ulu | 24 | 19 | 20 | 14 | 2 | 5 |

Grapevine and fodder co-occur in only 9/48 train patches; 23/48 contain neither,
7 contain grapevine only and 9 fodder only. Their directed 96 query-target tasks
therefore comprise 34 present and 62 absent; repetition over four K values gives
136 present / 248 absent rows. This is not 384 hard positive/confuser examples.
The absence result must stay separate from positive IoU, as in the frozen score.

For the K1 expansion catalog, 83/96 positive references originate in t31tfj;
the same 83/96 count holds for counterexamples because both task directions are
included. Across all train-source inventory, grapevine has 161 objects in 14
western patches versus 22 in 2 eastern patches; fodder has 57 in 12 versus 3 in 3.
Bank asymmetry is more severe: t31tfj has **zero corn** objects; t32ulu has **zero
grapevine** objects. Cross-region support swapping cannot be assumed to support
the same classes/K without adding eligible data prospectively.

Existing train co-occurrence is reasonably broad for the original classes:
meadow/wheat 21, meadow/corn 22, meadow/barley 21, wheat/corn 18, wheat/barley 16,
corn/barley 13 patches. These are availability facts, not recommendations to
exclude grapevine/fodder or select a higher-scoring class. Keep the existing
winter-wheat failure table and old primary population intact.

## Why seven strings are insufficient

The unique counts are generic 1, names-only 2, matched facts 2 and swapped facts 2.
Removed-facts inputs are exactly names-only. No card changes by actual source
region, observation year, acquisition date, parcel, cultivar, local weather or
management history. Dates occur only in audit metadata and EO input, not in the
seven Qwen request strings. The generic French/global scope caveats are useful
provenance but not local regional evidence.

Qwen is frozen, runs text-only and returns a deterministic last-token vector.
Under these seven fixed prompts, a table of seven cached hidden vectors exactly
replaces its online computation before the mask head. This does not show that
its pretrained semantic geometry is useless. It does show that current results
cannot distinguish a useful class descriptor from regional VLM reasoning.
Increasing row count or paraphrasing the same two classes does not by itself fix
this identification problem. A learned pair/class-code embedding is a necessary
control; new definitions and language forms must also be held out from training.

The matched-versus-swapped intervention controls the query image and support,
which is good. But the swapped crop facts can contradict the retained crop name
without looking at any image. A language-only model can detect that contradiction.
Wrong-context rejection is only evidence-grounded if its truth cannot be decided
from the crop name or writing style alone. The current conditions also omit an
irrelevant-but-true context and a parameter-matched no-language condition. Generic
text is still a frozen Qwen vector, not strictly a no-language implementation.

Annual crop masks do not label mowing dates, phenological stage, drought causes,
vine removal, species within the fodder class, or whether a stage is visible.
The cards correctly acknowledge this. They cannot serve as ground-truth labels
for these additional claims. Two dates cannot certify an unobserved intermediate
cut/recovery event; a specialist must also be allowed to answer unobservable.

## Architectural limitation that can be tested rather than presumed

`text_mask_model.py:40` broadcasts one text vector to all spatial locations.
At line 98 Qwen sees only the request, not EO features. At line 114 the mask head
receives query features and the means of positive/negative prototypes. The
unused original connector stays frozen and generation is explicitly unevaluated.
This is a text-conditioned segmenter using an LLM representation; successful
mask changes are not EO-grounded Qwen explanations.

The inherited encoder at `base_snapshot/episode_model.py:200` averages returned
features over dates and bandset after the native encoder; line 228 spatially
pools support within a 4-pixel grid and masks; line 114 in the new wrapper then
averages the K object prototypes. Date-/object-specific evidence is no longer
explicitly addressable by the readout. **Do not claim time information is
mathematically destroyed or that chronology is permutation-invariant:** native
EO features already incorporate timestamps and contextual processing. The
empirical question is whether retaining addressable date/object features solves
a concrete error at the same observation/compute budget.

## One proposed dataset: correction statements grounded in support observations

Keep PASTIS annual masks as the localization target and public support regions.
Add a small, separately versioned layer of **support-grounded observational
corrections**. A correction identifies a visible difference between positive and
confuser examples, the regions/dates that support it, and what cannot be inferred.
It must not invent a management event or causal label from a calendar/card.
Examples are observed greenness trajectories or differences visible in supplied
examples, not unverified statements that a particular field was harvested.

Start with **20 independently double-read source support pairs**, reusing already
prepared source observations and recording actual time. This is a new annotation
task, not the existing 20-image quality questionnaire. Ask annotators separately
whether a proposed distinguishing statement is supported, contradicted or
unobservable, and which supplied dates/regions justify that judgment. Crop-name
knowledge and image evidence must be separate annotation fields. Report agreement
and the fraction that is observable before committing the remaining human budget.
Do not call ordinary labelers agronomists or public-source summaries human expertise.

For each same query/support/target definition, form four controlled contexts:

1. A correct, support-grounded statement.
2. A plausible statement true for another support set of the **same crop**, but
   contradicted by this support set; crop name and prose style remain matched.
3. A true but non-discriminating/irrelevant statement, length/source-style matched.
4. Names-only/no-facts. Also run a strict no-language and class-code baseline.

This requires finding within-class variation in actual observed support, not
synthesizing arbitrary wrong crop calendars. If the 20 pairs cannot provide
reliable within-class contrast, this dataset cannot test the proposed regional
knowledge claim and must not be enlarged by paraphrases.

Use the six existing availability-selected classes as candidates, including
winter wheat. Create an availability matrix before performance evaluation for
class, target/confuser pair, parent, valid support count and visible temporal
contrast. Do not drop low-scoring classes. Existing source-parent cross-validation
can be development only: both parents and their data are already exposed.

For a genuine transfer result, add disjoint source/support/query blocks with
verified parcel/footprint separation, then evaluation across more than one
previously untouched target region or year. t30uxv is an unopened reserve, not an
already certified test. Its capacity/identity must be audited before opening
labels, and a single extra tile does not establish broad transfer. Existing
911-row partial-catalog availability is not evidence that the required new
region/year/phenology labels are already present. Acquire missing data under a
new manifest before training and keep exact availability failures visible.

Hold out regional groups and statement-source/templates jointly. For new-concept
claims, hold out an entire class from the episodic training objective; it is not
a new concept merely because the support patch is new. Original pretraining
exposure remains unknown. No class/region selection may depend on development
method scores.

## Minimal method comparison after that data gate

First expose the failure with the current common conditional baseline and a
strong time-preserving shared baseline. If retaining per-date/object tokens with
ordinary cross-attention solves it, treat pooling as an ablation, not a new method.
No need to build a tokenizer, KV cache, simulator or RL loop for this test.

A single testable method candidate is **evidence-conditioned use of correction
statements**: read statements against dated support/query features; allow a
language-conditioned residual to change the mask only when observed evidence
supports its use, and learn to ignore unsupported/irrelevant statements. Keep a
usable image/support branch. This is a candidate hypothesis, not established
novelty: generic gating, contrastive alignment and auxiliary reliability labels
are strong obvious baselines.

For the smallest fair comparison use the same date-preserving architecture,
OlmoEarth initialization, frozen reader, raw images, support, statements,
reliability labels, mask labels, updates and tuning budget in both main arms:

- **Generic joint baseline:** mask loss plus ordinary auxiliary statement
  reliability/evidence loss, receiving all true/wrong/irrelevant examples.
- **Candidate:** same baseline plus a pre-specified counterfactual consistency
  rule tying statement reliability to mask changes. Unsupported/irrelevant facts
  should not displace image evidence; true distinguishing facts may help.

Give the baseline all labels and corrupted contexts too. Comparing clean-only
baseline training to a corruption-trained proposal would confound supervision
with method. Add the class-code and strict no-language controls as scientific
nulls. Frozen-EO versus trainable-EO is a separate attribution test once this
phenomenon exists, not a reason to reopen or redefine the frozen P2 result.

Primary outputs: paired target IoU and confuser false-positive rate under correct
context; degradation under wrong context relative to names-only; invariance to
irrelevant context; target-absent false-positive area; and K/actual correction-time
curves. Report by independent region and target, not by the 384 repeated rows.
For evidence claims, measure human-validated region/date support and observed
risk versus abstention, not merely whether a cited ID exists. A language-only
context-validity control must fail where image evidence is necessary.

The data gate fails if statement validity is predictable from class/name/style,
if same-class true/wrong contexts are not reliably observable, or if source/query
independence cannot be established. The method gate fails if sufficiently trained
generic joint/class-code/no-language baselines match it, if gains disappear on
held-out regions/definitions, or if correct-context gains require unacceptable
wrong-context harm. Exact margins/sample sizes must be frozen prospectively;
current unstable P2 runs do not justify claiming a numerical threshold now.

Finally, an OlmoEarth-specific contribution requires exporting the learned EO
encoder and demonstrating an independent EO/readout or new-reader benefit under
matched adaptation budgets. Improved masks plus a templated explanation from the
same mask are not independent capabilities. This next comparison can establish a
useful failure mechanism and a possible remedy; it cannot yet establish CVPR
novelty or acceptance probability.

## Local evidence inspected

- `artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/manifest.jsonl`
- `artifacts/oe8_pastis_prepare_20260927/review_bundle_v0/episodes/source_objects.jsonl`
- `artifacts/oe10_expansion_catalog_20260928/oe10_expansion_catalog_v0/runtime_v2/episodes_train.jsonl`
- runtime_v2 preparation/runtime receipts; no underlying raw or label NPZ opened.
- `artifacts/oe10_text_mask_prepare_20260928/contexts_v0/contexts.jsonl`
- contexts_v0 knowledge_cards/preparation receipt and CPU tokenizer receipt.
- `code/oe10_text_mask_v2/text_mask_model.py` and pinned base feature extractor.
- `docs/OE8_PASTIS_INPUT_PREPARATION_20260927.md`, existing v6 design,
  `RESTART_HERE.md`, latest P2 execution/worklog status.
