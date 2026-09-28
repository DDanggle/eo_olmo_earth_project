# Minimal implementation route for useful language-conditioned EO matching

2026-09-28; read-only design plus synthetic NumPy contracts. No frozen source or
protocol changed; no server, GPU, new raw imagery, label array or held-out payload
opened. This is a concrete next implementation candidate, not a measured gain.

## Recommendation

First make the model use **different positive/confuser examples differently for
each query location**, with language conditioning that matching decision. Keep
OlmoEarth, the 128-pixel inputs, two query dates, eight dates per support and mask
supervision. Train a small readout before considering full VLM tuning. Add explicit
date alignment only if object matching and ordinary training still leave a
repeatable temporal error.

One cheap shared-input optimization should precede both: reuse the **same
attached encoder graph within one forward** when multiple support masks use the
same image/date stack. Root's metadata audit was independently reproduced exactly:
K8 currently calls the support encoder 16 times, but unique support inputs average
11.9896 in the original catalog and 11.6458 in the expansion. With two query dates,
patch-date encoding counts could fall from 130 to 97.9167 / 95.1667. This is a count,
not a measured speedup; native replay, checkpoint recomputation and language cost
are not included. Dedup is engineering shared by all competing models, not a
novel method. Comparing eight query dates against two is a separate observation
ablation even if dedup offsets some cost.

There is an earlier optimization gate. The absolute-score reviewer reports that
winter-wheat positive training loss remains much larger than corn and its positive
gradient norms are clipped on every final-window step. The observed AP/IoU gap
also leaves calibration in play. Other classes also clip on approximately 95.9-100%
of steps, so clipping is not winter-wheat-specific evidence. These do not prove
a single cause, but language
or architecture should not be claimed to solve a representation limitation until
the common head can fit a small, prospectively hash-selected positive/confuser
training subset. Use the same subset and diagnosis for all methods; do not rescue
one class by discarding its evaluation or relabeling synthetic supports as experts.

## Candidate 1: conditioned query-to-object matching

**Concrete change:** replace unconditional `positive.mean(1)` and
`counter.mean(1)` in the current v2 head with shared query-conditioned attention
over individual support objects. The first comparison keeps the existing date
average. That isolates object mixing from temporal representation.

At a query position x, for role r and object k:

    a[r,k](x) = softmax_k( dot(norm(Wq q(x) + U c[r]), norm(Wp p[r,k])) / temperature )
    m[r](x)   = sum_k a[r,k](x) * V p[r,k]
    logits(x) = H(q(x), m[positive](x), m[confuser](x), matching score difference)

Here c[r] is the name/definition representation for that role. With no language,
use a common null vector; names-only and names+facts go through exactly the same
head. Positive and confuser projections share weights so the model learns matching
rather than two unrelated classifiers. Keep role labels explicit. A generic
cross-attention implementation of this equation is a strong baseline, not itself
a novelty claim.

Why this is worth testing: uniform averaging mixes different examples before
the query can decide which ones apply. A larger K can therefore dilute a useful
example with a different appearance, quality or season. This is a hypothesis
about the observed lack of K benefit, not a demonstrated causal explanation.
Conditioning the match is more direct than broadcasting one language vector to
every pixel after support aggregation.

**Important implementation trap:** adding a text scalar that is constant over k
to attention logits has no effect after softmax. The text must change the query,
keys, their interaction or a genuinely object-dependent score. The equation
above does that. Test attention weights as well as final logits, and compare
against a class-code vector and null vector with the same parameterization.

Minimal touched methods in a new version:

- Keep the old EO `_encode_tensor`, timestamp handling and spatial pooling for
  this first object-only comparison.
- Return support tensors `[B, role=2, K, D]` before mean-over-K.
- Add shared object matching and feed its two spatially varying results to the
  dense head. Preserve output grid/upscaling and loss initially.
- Keep Qwen frozen; train matching projections/head, then OlmoEarth with the same
  schedule in all arms. Monitor head and encoder gradient clipping separately.
- Implement a local per-forward dictionary for repeated **image** encodings;
  mask pooling remains distinct per object. Never detach shared B2 graphs and
  never carry a trainable feature cache across optimizer updates.

## Candidate 2, conditional on a remaining date-specific failure

Do not add a new observation selector yet. Retain the date axis that the native
encoder already emits, and let the query use date-specific support evidence.

Existing native output is `[B,32,32,T,1,D]`. Current code averages T and bandset.
A new path drops only the singleton bandset, retaining:

| Tensor | Shape |
|---|---|
| Query feature | `[B, P=1024, Tq=2, D]` |
| Support feature after object-mask pooling | `[B, role=2, K, Ts=8, D]` |
| Role/statement text | `[B, role=2, J, d_text]` |
| Query dates | `[B,Tq, date features]` |
| Support dates | `[B,role,K,Ts,date features]` |
| Match weights | `[B,P,Tq,role,K,Ts]` |

The attention above expands from k to (k,s). Its bias may depend on cyclic
calendar difference and supported region/season scope. Matched positive and
confuser evidence is returned separately for each acquired query date. A small
shared readout can consume the two results and their signed difference; it must
not immediately average the dates before any temporal interaction.

The old mean path is exactly recoverable by averaging the retained tensors; this
makes a useful implementation check. It does **not** prove OlmoEarth's current
features lack temporal information: the native model already uses timestamps and
context before averaging. Require a same-observation mean-vs-object-only-vs-dated
comparison. If ordinary dated attention is enough, keep it as the common baseline
rather than attributing the gain to a special knowledge mechanism.

At K8 the matching score tensor has 1024*2*2*8*8 = 262,144 values, about 1 MiB in
FP32 before gradients/temporary buffers. This is small relative to the EO model,
but retaining native support grids and autograd activations can still be expensive.
Pool each object's spatial mask promptly into per-date prototypes and checkpoint
the encoder. Do not cache 128 full eight-date grids on the assumption that the
old one-grid cache's memory bound still applies. An object-prototype cache, when
encoder-frozen, must include mask content, dates, normalization and encoder state
in its key. A role or object ID alone is not a sufficient cache identity.

## Language representation: change the interface before changing the LLM

Current last-token pooling is a valid contextual representation; it is not a
proven bug. It mixes positive definition, confuser definition and caveats in one
vector. The targeted alternative is **separate role/content representations**:

1. Encode each role's concept name and each short observation-checkable claim
   separately with the frozen reader and a fixed prefix.
2. Pool only that content's non-padding hidden tokens, excluding fixed prefix,
   assistant/chat boundary and end tokens. Token boundaries must come from exact
   token assembly or verified offsets, never substring token-count guesses.
3. Keep the name vector and claim vectors separate. Cross-attention can choose a
   claim; do not average a long disclaimer into the same untyped class vector.
4. Keep negation, possible-versus-universal scope and missingness explicit. A
   schema such as `{feature_family, relation, temporal_scope, applicability,
   uncertainty}` can accompany the language embedding. Unknown values remain
   unknown; structured fields must not be fabricated crop-stage labels.

For example the existing fodder card says alfalfa is one subtype that can regrow
after cutting. It does not license a universal `fodder has repeated cutting`
label. The vineyard floor-cover card does not guarantee rows are visible at
Sentinel resolution. Convert these to optional hypotheses with scope, not hard
rules. Both the generic baseline and candidate must receive the same verified
structured fields, if used.

Test last-token versus content-mean using the same words, reader and main
architecture. Mean pooling is not inherently superior. Freeze the choice on
training/development evidence before final evaluation; do not combine changed
wording, new fields and new pooling into a single unexplained performance gain.
A small decoder trained on role vectors is sufficient for this step. Full Qwen
fine-tuning is not required for the gradient to OlmoEarth or for text-conditioned
matching to learn.

## Regional and seasonal input that is legitimate now

Use real acquisition dates already present in the input. Interpret native
`[day, month0, year]` correctly, including leap years. Cyclic day-of-year features
avoid a discontinuity at New Year. They are calendar coordinates, **not measured
crop phenological stage**.

For a region/season cue, prefer similarity to the actually supplied source
trajectories over a learned tile-ID embedding. A query at a given date can attend
to a different support date if spectral evidence supports it; calendar difference
is a soft bias, not a hard nearest-month match. Both target and confuser tracks
must be available to the matcher. Optional normalized reflectance/index summaries
can be derived from the already acquired bands, with band-order/missingness checks;
these do not establish cloud-free or true crop-stage labels.

Do not add Nepal-monsoon/Korea-rainy-season tokens to these French patches or infer
weather from date alone. Region-specific climate/calendar records are not in the
current model inputs. If later verified sources supply regional scope or seasonal
windows, expose the same facts to both baseline and candidate; preserve unknown
scope and evaluate matched/wrong/irrelevant context. A learned region offset could
just memorize two source environments, so its benefit must survive held-out-region
and within-region controls. Exact crop stage, drought, harvest or cause needs
appropriate ground truth and is outside this first segmentation experiment.

## Minimal fair comparison and decision order

A small nested comparison is more informative than assembling many new modules:

- **Reference:** current pooled common head, but trained sufficiently and given
  the same safe within-forward dedup optimization.
- **First candidate:** object-only matching. Within that same structure compare
  null/class-code, names-only and names+the same sourced facts. The facts effect
  is names+facts minus names-only, not versus unnamed examples alone.
- **Only if needed:** dated object matching, with the same images and language.
  This isolates temporal addressability; adding six more query dates is a separate
  paired observation experiment, not part of the architecture contrast.

For method novelty after this strong baseline, the promising question is whether
an evidence-conditioned correction rule can use a valid local clue and ignore an
unsupported one. Both methods must see identical true/wrong/irrelevant examples
and reliability supervision. Compare ordinary joint auxiliary reliability training
against the proposed consistency rule; do not obtain an apparent method gain by
giving the proposal better labels, more dates or a stronger readout.

Primary practical outputs stay localization on target-present cases, confuser
false-positive rate, target-absent false-positive area and benefit as K grows.
Report annotation/pretraining budgets and measured wall time separately. Temporal
citation/explanation quality needs another actual evaluation; the current mask
head has no learned generative explanation behavior. Only then consider training
an EO-token-to-frozen-LLM projector or small language adapter for constrained
region/date-grounded explanations. Full VLM training becomes a justified option
only if this stronger frozen-reader route demonstrably cannot express the task;
it is not a prerequisite for the first score improvement experiment.

## CPU evidence produced in this review

`/private/tmp/oe10_improvement_tensor_contract_20260928.py` ran with the bundled
NumPy runtime: **10/10 tests passed**. They check native-to-dated tensor shapes,
old-average reconstruction, fractional object-mask pooling, role/object/date
indices, all-K matching, context-sensitive weights, joint feature/date
permutations, role swapping, invalid-token masking and zero-based month/leap-year
conversion. The constructed opposite-temporal-signature example demonstrates an
operator possibility, not an observed EO error. No gradients, pretrained features,
learning, semantic representation quality or performance were tested.

Receipt: `/private/tmp/oe10_improvement_tensor_contract_result_20260928.json`.
Root support-compute audit was independently rerun to
`/private/tmp/oe10_support_compute_independent_20260928.json`, with exact JSON
agreement and separate counts/hash checks for both catalogs and every K value.
Before implementing dedup, compare full forward values plus encoder/head gradients
with and without sharing under actual stochastic/checkpoint settings. Equal
inputs and `.eval()` alone are not sufficient evidence of measured equivalence.
