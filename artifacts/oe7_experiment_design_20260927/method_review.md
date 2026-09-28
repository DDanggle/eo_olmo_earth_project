# OE7 independent method review: one learnable method and a decisive minimum comparison

2026-09-27. Read-only review of `docs/OLMOEARTH_CROSSDOMAIN_LOOP_PLAN_20260927.md`, `docs/OE6_PASTIS_INPUT_PROGRESS_20260927.md`, and `config/oe6_execution_and_crossdomain_status_20260927.json`. This is a proposed development experiment, not a preregistration, new performance result, or established novelty claim. No repository, server, or GPU changes. Existing KRW 1,000,000 annotation plan is unchanged.

## 1. What is actually available

- Verified PASTIS shard 0000 has 911 official-train records, across four parent tiles. Shard 0001 failed SHA; 0002 is missing. The corrupt/missing files cannot participate.
- Only two native engineering inputs have been prepared and independently checked. Their final dates were chosen by assistant visual QA; these hand-selected dates are not an unbiased observation-selection policy.
- The 64 train / 16 development manifest selects 32 records each from t32ulu and t31tfj, and 16 from t31tfm. Those 80 inputs are not yet prepared. t30uxv is a reserved region in this stage, not a proven globally unseen test set.
- Native inputs have ten measured S2 bands and two declared imputed bands. Native normalization is once, actual frame dates share image indices, annual semantic labels exclude void 19. Exact footprints and cloud masks are absent. Annual labels do not identify crop health, a change date, flood progression, or whether a particular date visually resolves the crop.
- Actual Qwen3-VL 8B FP32 one-step integration has passed. A scientific trainer, its save/reload, full reader identity, and matched comparative training remain unverified.

**First permissible claim:** a development comparison of crop distinction in one held-out parent tile. No claim of new classes, climate understanding, multi-country generalization, temporal-change understanding, or expert-label savings from these 80 samples alone.

## 2. One candidate method: train OlmoEarth to retain the distinction expressed by a correction

Use a simple **support contrast objective** as the first test of a learning hypothesis, not as a newly invented method. Keep architecture equal to a strong episodic dense baseline initially; do not introduce region gates, new tokenizer, physical simulator, and RL simultaneously.

An episode defines target class c+ and a visually confusable class c−, one or more confirmed support regions for each, and a separate query patch. Both class names and both types of support are supplied to every learned baseline. Crop pairs are selected using training labels only; the complete pair inventory and class coverage are frozen before development-label results. No claim that c− is human-verified visually confusable unless actual people verified it. Until then call it a training-defined comparison class.

The shared architecture is:

1. Selected native frames, dates, and declared input validity enter a trainable OlmoEarth encoder Eθ. It emits a spatial/temporal grid Z; do not flatten away the date identity.
2. The same encoder processes positive and comparison support patches. Their public source masks select support features. A shared projection g maps all grid features to normalized vectors.
3. Pool support-region features into p+ and p−. Query features and these prototypes enter an ordinary learned cross-attention/prototype fusion block, a dense head, and an ordinary learned resampler for Qwen. The plain baseline already receives p+, p−, names, dates, and the same raw support.
4. Dense output is an actual mask probability map at the declared source pixel grid, through the same upsampling decoder in every arm. Deterministic mask operators produce pixel counts; no geographic area claim without a footprint/grid contract.
5. Qwen receives the same question format, M=64 continuous EO positions as a provisional throughput point, source IDs, and the same format of predicted dense outputs. The baseline must not be denied the dense branch.

### Common objective

`Lcommon = Lmask + λtxt Ltext + λnative Lnative`.

- Lmask is binary target-vs-rest pixel BCE plus Dice on non-void labeled pixels; all arms receive identical mask supervision and class balancing. Class 0 is source background, not cloud/absence of observation. Only foreground target labels produce positive support regions.
- Ltext is short structured supervised output from existing annotation-derived tasks. Use the same templates/questions/targets across arms. Ground-truth masks/counts must never enter the input prompt at training or evaluation. If predicted counts are supplied, correct verbalization tests faithfulness to those counts, not a second independent measurement ability.
- Lnative is the pinned official v1.2 continued-pretraining objective using the same authorized native corpus/replay schedule in every supervised arm. Preserve original masks/teacher mechanism and log replay observation count. Official additional training is a substantive separate baseline, not an optional regularizer only for the proposal.
- Start with Qwen frozen but preserve gradients from its input back into the connector/encoder. Same head warmup and encoder trainable scope apply to both representation arms. Reader LoRA is a later matched expansion.

### Proposed additional loss

For query feature u and prototypes p+/p−, let `d(u)=cos(g(u),p+)−cos(g(u),p−)`. For target/comparison query cells, `y=+1/−1`. Add

`Lcontrast = mean softplus((m − y*d(u))/τ)`.

First development setting: m=0.2, τ=0.1, λcontrast=0.1 after a common short connector/head warmup. These numbers are starting values to record, not validated optimal hyperparameters. If tuning is allowed, the same trial budget goes to ordinary supervised contrastive/prototype objectives in the controls. Do not retune only the proposal.

The contrast loss uses only target/comparison regions, whereas Lmask uses all valid label pixels. To avoid making mixed boundary cells falsely pure, pool soft source-mask fractions onto the feature grid and weight by those fractions, or restrict to a declared >=80% pure cell rule; choose one rule before comparing. Do not interpolate categorical labels bilinearly. Do not pretend annual labels create per-date phenology labels: this loss is on a feature representation of the currently observed window, pooled over its time axis after encoding.

Train **R0** with Lcommon and **R1** with Lcommon + λcontrast Lcontrast. Architecture, parameter count, source data, masks, support, questions, replay, optimizer family, and total training budget are equal. The additional loss is an established metric-learning type mechanism, not independently novel. Its point is to test whether a correction-specific distinction improves the encoder beyond ordinary dense/VLM co-training.

Minimal support handling: support and query must have different patch IDs; no query mask enters support. Training support libraries use only the two training parent tiles. Primary development queries receive training-region support only. An optional target-region K-shot setting is a separate experiment with a disjoint development support/query allocation, whose labels are explicitly charged. With only 16 development patches, do not promise K=20 independent patches, and do not count multiple parcels in one patch as independent regions. Without exact footprints, same-parent overlap cannot be fully excluded; parent separation protects the first train→development comparison but not all within-training sampling.

## 3. Observation selection is a separate, learned component

The smallest useful action is selecting **one more actual date**, not zooming, masking, retrieving a correction, acquiring a new sensor, and rewriting text at once. These other actions can follow if date selection shows value.

Prepare up to four candidate observations from each patch by a label-independent raw-date rule, e.g. ordered sequence quantile indices at 1/8, 3/8, 5/8, 7/8 with duplicate removal. Store original acquisition dates. This choice is a development input policy, not a phenology-aware oracle. All methods access exactly the same candidate pool. Start from one declared common candidate and permit at most two additional dates. Actual data quality may require revising this rule before model comparison; any revision is recorded and applied to all arms.

The candidate selector sees:

- the question/class-pair definition and the same support summaries as the reader;
- date/sensor IDs that existed before pixel reading;
- features, dense uncertainty, and history for **already read** observations only.

It **does not see the OlmoEarth embedding, thumbnail, spectrum, model confidence, or derived cloud score of an unread candidate**. If a preview or cloud detector is used, its raw-pixel I/O and computation are charged and the same preview is given to all comparable selectors. No PASTIS cloud quality metadata has currently been established; do not silently manufacture this field as free metadata.

A small shared MLP scorer can combine a pooled already-read state with the candidate date embedding and output candidate utility. This is deliberately less complex than asking Qwen to write a long plan. A generic Qwen tool selector is a separate strong baseline using the exact same information and action budget.

### Selector training, without hidden query gold

Offline training targets may use **training-region labels** to measure the incremental mask-loss reduction from adding a candidate, minus a declared read/computation penalty. The target computation must explicitly encode the additional observation and is part of the training cost. At inference there is no target query label, loss difference, oracle rank, or uncharged candidate pass.

Use cross-fitted utility on parent groups: a cheap frozen-encoder dense teacher trained on parent A supplies utility records on parent B, and vice versa. Keep this utility teacher, label source, candidate rules, and targets common to R0/R1 when fitting the first selector. These two parents do not support a reliable claim of general utility transfer; this is only a bounded development construction. If a class cannot be supported in the opposite fold, mark it unsupported rather than borrowing development labels.

Fit the ranker by pairwise logistic ranking or utility regression on those records. A metadata-only ranker with identical training utility is mandatory: it checks whether the “adaptive” model just learned a crop/season lookup. A generic attention/router given the same utility is mandatory before claiming a specific selector mechanism. This stage is supervised observation selection. It is not RECAP, policy-gradient RL, or autonomous reinforcement learning.

Avoid selecting a method by the same development score that later serves as confirmation. All development choices and utility teachers are frozen before a separate unopened evaluation is constructed. Existing 16 development queries cannot simultaneously be hyperparameter tuning data and an untouched success test.

## 4. Minimum 2×2 and the additional baselines needed to interpret it

| Cell | Representation learning | Observation policy at inference | Isolates |
|---|---|---|---|
| R0S0 | ordinary common joint training | fixed metadata-defined sequence | strong plain baseline |
| R1S0 | same training plus support contrast | identical fixed sequence | correction representation effect |
| R0S1 | ordinary common joint training | trained selector | selection effect without proposed representation |
| R1S1 | support contrast | the same selector architecture/training target recipe | combination and interaction |

For the first diagnostic, fit one shared selector on a declared mixture of R0/R1 already-read training states using the same common utility targets, with equal state counts; apply it unchanged to both rows. This avoids attributing a different selector optimization budget to representation learning. Also apply S0/S1 to the same cached final trained encoder checkpoint. If representation training was itself exposed to different observation policies, that would add a third factor; avoid this by training both R0/R1 on the same randomized date-subset schedule.

The interaction to report is `(R1S1 − R1S0) − (R0S1 − R0S0)`, with the same support draws and query packets. It is descriptive at this scale, not proof of a novel synergistic mechanism. R1S0−R0S0 is the main representation comparison; do not let a selector improvement substitute for it.

Before this 2×2 is scientifically interpretable, establish:

| Baseline | Purpose | Fairness condition |
|---|---|---|
| Original frozen OlmoEarth + trained dense/readout/resampler | what pretrained features already contain | same support, labels, questions and head tuning |
| Official continued-pretrained OlmoEarth + same frozen-encoder adaptation | what additional native training alone buys | report original-vs-CPT backbone under the exact same downstream readout training |
| R0 ordinary joint | effect of ordinary joint adaptation | no extra information unique to R1 |
| Original frozen full-grid reader vs trained learned resampler | whether compression is the main bottleneck | same dense decoder and head optimization; full-grid compute is reported |
| Ordinary supervised contrastive/prototype objective with same support-label budget | whether the new loss is just generic metric learning | same weights/search budget and class balance |
| Fixed, random, metadata-only, uncertainty/generic utility selectors | whether the selector is merely season or uncertainty | same candidate availability and read/compute costs |

A native-CPT model trained on 64 PASTIS records with crop labels is not “official continued pretraining.” The native objective and its input corpus/replay need their own receipt. Conversely, using a much larger native corpus only for one arm is an extra-data confound. Two ledgers are useful: a strict matched total GPU-time comparison, and an explicitly labeled additional-compute scientific ablation. Neither equal steps nor equal M tokens alone equals equal cost.

The ordinary supervised contrastive control must use **exactly the same positive/comparison class pairs, source regions, eligible query cells, and number of pair evaluations**, not just the same overall dataset. Give it the same representation projection and hyperparameter search budget. If it matches R1, conclude that generic contrastive supervision explains the effect. The R0/R1 pair is a mechanism screening experiment; at present the paper's novel training rule has not been established.

Keep the first pipeline manageable: prepare/gate inputs → frozen dense + learned resampler and full-grid references → native-CPT reference + R0/R1 → the policy 2×2 on frozen final checkpoints → a second reader. A failed data/trainer gate should not trigger adding more architecture modules.

## 5. The accounting issue that can invalidate the entire loop story

OlmoEarth jointly encodes selected dates. Adding one date can change previously computed tokens. Exact adaptive inference must recompute the selected temporal window unless a different causal/separable encoder has been implemented and tested. Record raw I/O, total encoded pixel-dates, attention cost, wall time, peak memory, Qwen input/generated positions, and cached bytes separately.

For example, reading 1→2→3 dates requires encoder work for windows of sizes 1,2,3, not three free append operations. A one-shot model with all four candidates could be faster or better. **It must be included on the measured cost–quality curve if it fits the same budget.** Artificially forcing the fixed baseline to waste repeated passes does not establish an efficiency gain.

If whole HDF5/NPZ sequences are loaded to CPU before selecting frames, there may be no raw-I/O saving. Restrict the claim to measured encoder/VLM computation until an actual selective read path exists. The model code should accept only the selected frame tensor; the selector's function signature should not receive the candidate pixel/embedding bank. This is auditable by logging every encoder invocation and its input observation IDs.

Each completed adaptive episode gets a **replay baseline**: the same final chosen observation set and support are encoded once and passed to the same reader. The replay is an oracle-access ablation for attribution because it is handed the loop's final choices; it is not a deployable selection competitor. Equal accuracy in replay means the benefit came from chosen information, not repeated language reasoning. Runtime savings in replay are not the deployed selector's savings.

An all-candidate full-grid reference sees more information/compute and is labeled accordingly, unless actual measured costs show it fits the same cap. Feature caches are allowed for frozen teachers and evaluation models with unchanged complete observation windows; training OlmoEarth on stale feature caches cannot demonstrate encoder improvement.

## 6. Same dense outputs to Qwen: make the VLM attribution test unambiguous

There are two different comparisons, and both should be named:

1. **End-to-end task quality:** R0/R1 generate their own masks/features and the same Qwen answers. This measures system quality; a better mask may fully explain the answer improvement.
2. **Reader attribution:** save one canonical set of predicted masks, counts, and provenance IDs and give exactly those artifacts, with the same query and text, to both reader variants. A deterministic renderer/template plus the same Qwen is the strong tool pipeline control. If answers are equally correct, the contribution lies in EO perception/selection, not new language reasoning.

A source date cited because it was read is provenance, not proven explanatory necessity. To test necessity, remove/replace that date under a valid data contract and inspect the answer/mask change. PASTIS annual crop labels do not provide date-specific causal explanations. Do not supervise “the crop changed on date d” from these masks.

For a second-reader transfer claim, reinitialize and equally train the new connector on a fixed adaptation set for original, native-CPT, R0, and R1 encoders. Giving one encoder its old trained connector is not a clean representation comparison. This can be deferred until the first simple comparison shows a stable useful effect.

## 7. Minimal outcomes and stop/expand logic

Primary development metrics: target foreground IoU; false-positive rate specifically on the declared comparison class; presence/absence error when the target is not labeled; time/compute per query. Report per-class support coverage and unsupported pairs. Background accuracy must not dominate the result.

Report paired effects across the same query packets and support draws. Multiple seeds or pixels do not create independent parent regions. With one development parent tile, intervals over its patches are conditional development uncertainty, not geographic-generalization confidence. A stronger geographic result needs further unopened parent/event groups and separate materials.

- If frozen full-grid works and generic learned resampling recovers it, the discovered issue is connector compression; complex selection/contrast may be unnecessary.
- If R1S0 does not beat ordinary joint training/generic contrastive control at comparable cost, there is no support yet for the correction representation mechanism.
- If R0S1 accounts for the entire combined gain, report a selection improvement, not encoder learning.
- If all-candidate one-shot dominates the loop at equal or lower measured cost, the loop is not an efficiency contribution at this scale.
- If native EO capability drops, report the tradeoff and compare official replay/head warmup/gradient routing before claiming an improved general EO model.
- If the 80-record development contract or meaningful learning curves fail, fix those first. Two curated engineering examples cannot decide these hypotheses.

Human annotation remains the existing 20-case timed pilot inside the original KRW 1,000,000 budget. Public-mask-generated support is simulated correction, not already purchased expert insight. No extra annotation volume, API spending, data repair, new model download, or multi-day GPU allocation is presumed by this note.

## 8. Later candidate: preserve the correction-defined distinction through compression

A more specifically targeted hypothesis is that a correction's positive-vs-comparison distinction exists in the EO grid but is damaged by compression. Only pursue it after full-grid, plain learned-resampler, and actual label results establish this failure; the previous linear pooling toy does not establish it.

One candidate target is a training-only teacher discrimination score `d_grid` from the same positive/comparison regions, with a student score `d_comp` decoded from the compressed representation using a common query-conditioned readout. Train an extra consistency term such as squared score error or KL between calibrated target/comparison logits. All arms must get the same region/pair supervision and readout. Compare against ordinary feature distillation, mask-logit distillation, and supervised contrastive learning using the exact same pairs, total target dimensions, and teacher computation. A larger/residual resampler is another strong cost-matched comparator.

This has several pitfalls: (i) a compressed token has no automatic one-to-one pixel alignment, so arbitrary tokenwise MSE cannot be called preservation of a located distinction; (ii) a grid teacher may be wrong or encode label priors, and preserving its scores may preserve errors; (iii) a dense branch that bypasses compression may make this loss irrelevant to the actual downstream output; (iv) query-conditioned feature distillation is already a broad known method family. Teacher generation can use training labels and must be costed, but evaluation gold cannot choose the teacher target or compression layout.

The decisive observation would be: relative to equally supervised generic distillation, a compressed reader retains **previously demonstrated correct pair distinctions** and improves unseen-region mask/query outcomes at the same budget, while the trained encoder also transfers to a fresh reader or independent EO evaluation. Even that result would motivate a further close prior-art audit before claiming a novel principle. This follow-up is neither implemented nor a justification for postponing the ordinary baselines.
