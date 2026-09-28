# EO vocabulary, continuous/discrete paths, and streaming cache: independent research note

2026-09-27. Read the current `docs/OLMOEARTH_VLM_ARCHITECTURE_DATA_PLAN_20260927.md`. This is literature/design work only: no GPU run, new data acquisition, model evaluation, or claim of novelty. Seven research lines were checked against primary sources. Recommendations below are our hypotheses and engineering deductions, not the papers' EO results.

## Recommendation

Prioritize **a measured continuous-versus-discrete interface experiment**, after real labeled PASTIS inputs and a trained continuous baseline exist. Keep **cache correctness** as a separate engineering experiment; only make memory a research contribution if limited-memory errors are independently demonstrated. An EO dictionary is attractive as a reusable intermediate representation, but ordinary quantization and attaching code IDs to language embeddings are already established. The stronger question is whether a compact representation preserves the distinctions that expert corrections require, including a small target versus a visually similar exclusion, across environment changes.

Do not implement tokenizer, learned memory, expert routing, physical simulator, and an RL loop simultaneously. That would prevent identification of which change helps and multiply data/compute requirements.

## Seven primary research lines

| Work and verified primary sources | What is established | Transfer idea and boundary |
|---|---|---|
| **FAST**, first submitted 2025-01-16. [Paper](https://arxiv.org/html/2501.09747v1), [official tokenizer](https://huggingface.co/physical-intelligence/fast) | Robot action chunks are transformed with DCT, quantized, and compressed by BPE. The finite vocabulary is compatible with autoregressive VLA prediction. Public implementation and FAST+ tokenizer are available. | Borrow task-relevant rate–distortion evaluation and analytical compression baselines. Smooth, regularly sampled robot trajectories differ from irregular cloudy satellite visits. Do not import its tokenizer weights or reported training speedups as EO evidence. DCT of acquisition index is not DCT of elapsed physical time. |
| **UniTok**, 2025-02-27; v3 2025-10-24, NeurIPS 2025 Spotlight. [Paper](https://arxiv.org/html/2502.20321v3), [official code](https://github.com/FoundationVision/UniTok) | Multi-codebook quantization increases discrete representational capacity for visual understanding and reconstruction; integration with MLLMs is studied. | Compare ordinary VQ/FSQ against a modest residual/product codebook while keeping the reader and train data fixed. Reconstruction alone cannot establish correction transfer. Multiple codebooks can increase either sequence length or embedding/decoder work; record which implementation is used. |
| **TerraMind**, 2025-04-15, ICCV 2025. [Paper](https://arxiv.org/html/2504.11171v3), [official code](https://github.com/IBM/terramind) | EO image modalities have discrete patch codes; coordinate/caption sequences use modified BERT WordPiece; raw pixel patches and token representations are jointly used. It also generates intermediate modalities through Thinking-in-Modalities. The paper discusses details lost by aggressive tokenization. | This directly precludes a novelty claim for “Earth dictionary + text tokens + pixels” or a generated-modality loop alone. Our comparison should test whether expert-relevant distinctions and measured quantities survive a language interface. TerraMind is not simply a pretrained chat LLM with a new vocabulary; do not conflate these architectures. |
| **Phaedra**, first submitted 2026-02-03. [Paper](https://arxiv.org/html/2602.03915v1) | A physical-science tokenizer separates amplitude and morphology to improve numerical reconstruction; evaluations include PDEs and out-of-distribution EO/weather. | Borrow the diagnostic: preserving shape need not preserve magnitude. A candidate EO code can combine a pattern code and explicit scale/residual. This is not novel by itself, nor does a good PDE or EO reconstruction prove flood dynamics or causal physical validity. Code/checkpoint availability was not established in this audit. |
| **EO-VAE**, 2026-02-12; v2 2026-03-02. [Paper](https://arxiv.org/abs/2602.12177v2), [author model card](https://huggingface.co/nilsleh/eo-vae/blob/main/README.md) | A multi-sensor VAE uses dynamic hypernetworks for flexible channel combinations and evaluates reconstruction on TerraMesh. The author card exposes continuous latent extraction and a checkpoint. | Useful reference for a continuous EO compression path and sensor contract. “Tokenizer” does not imply discrete code IDs or a semantic vocabulary. It is not yet evidence that such latents improve a language reader or expert transfer. |
| **StreamMem**, first submitted 2025-08-21. [Paper](https://arxiv.org/html/2508.15717v1) | Query-agnostic streaming video KV compression uses proxy attention, pruning/merging, and a fixed memory budget. Position handling matters. | Compare EO memory against generic selection/merging if a streaming research question survives. Natural-video results do not establish that rare old flood or seasonal evidence survives. Preserve source dates separately; a merged KV prototype is not a new observation. Public implementation availability was not established here. |
| **StreamKV**, first submitted 2025-11-10. [Paper](https://arxiv.org/html/2511.07278v1), [linked official code](https://github.com/sou1p0wer/StreamKV) | Training-free video QA combines semantic segment partitioning, layer-adaptive KV selection, retrieval and compression. | A strong related family for event/observation-group retrieval. Segment-specific stored KVs and truly length-independent total memory are distinct settings; count CPU/disk storage as well as GPU. Do not present retrieval over the complete archive as bounded total memory. |

The search is selective; it cannot establish absence of closer work. Publication dates above use arXiv metadata, not crawler-relative dates. No external result numbers are adopted as project targets.

## What an Earth vocabulary could mean

Three things should have separate names:

1. **Quantization dictionary:** `q(z) -> code_id`, a learned or fixed partition of latent space. Code 417 does not intrinsically mean “rice,” “flood,” or “unknown.” Inspect code usage, collisions and geographic stability. A prototype gallery and expert descriptions can help interpret it, but are not universal definitions.
2. **Structured EO evidence:** a code or continuous vector accompanied by source footprint, acquisition time, sensor, resolution and validity. This is where grounding comes from. Keep those fields even if latent codes compress well.
3. **Human semantic ontology:** validated definitions such as permanent water versus inundation. These can be many-to-many with code IDs and depend on observations/context. A latent ID should not become a label without validation.

A literal bag-of-words histogram discards ordering, location and source linkage. It is a useful negative/control baseline for coarse retrieval; it is unsuitable as the sole representation for “where/when.” Use an ordered, spatially indexed code sequence or evidence graph instead. This is a design choice, not a novelty claim.

For the first reader, `code_id -> EO embedding table -> current input-embedding path` avoids changing the entire text tokenizer. A text token such as `<eo_417>` is only an identifier, not a pretrained English concept. If we later require the LLM to predict EO codes, add trainable code output rows and constrained output segments, pin the vocabulary/version, check save/reload and retain text-only capability. New rows are new parameters even when the old LLM is frozen. Compare these added parameters and training costs fairly. Do not turn arbitrary integer IDs into prose and assume the reader understands them.

## Cache contract: four different objects

| Cache | Validity requirements | Training implications |
|---|---|---|
| Raw chip/decoded tensor | Content hash, crop/CRS/band order/calibration/normalization/version and masks | Usually reusable preprocessing, but random augmentation is part of the input contract |
| OlmoEarth feature | All above plus encoder revision, exact date/sensor set, attention/masking/window contract, precision and deterministic mode | Detached cached features cannot carry current gradients into an updating encoder |
| Projected EO feature | All above plus adapter/resampler/vocabulary revision and prompt-dependent selection if any | Invalid when projector/codebook changes; quantization training requires an explicit gradient estimator and audit |
| LLM K/V | All above plus reader revision, entire preceding causal prefix, position IDs/RoPE, attention mask, multimodal layout, precision and cache format | Frozen LLM weights do not imply fixed K/V when EO inputs change. Do not use `no_grad` around a frozen reader if its input gradient must train OlmoEarth/adapter |

Appending new EO data can invalidate old features **even with frozen weights** if old and new dates are jointly contextualized by the encoder. Compare old features under `encode(old)` and the corresponding slice of `encode(old + new)` before assuming append-only reuse. Existing single-observation cache paths are a different contract; their past correctness does not prove a joint-window cache is appendable. A genuinely causal or independent-chunk encoding path would be an architectural change requiring matched comparison.

Exact causal LLM prefix reuse only applies to an unchanged prefix with unchanged position/mask semantics. If the query or expert correction comes before image tokens, changing it changes downstream KVs. If it comes after an immutable image prefix, only that image prefix may be reusable. Qwen multimodal position handling must be tested in the actual runtime. Arbitrary KV averaging/eviction is approximate and must be evaluated as a separate memory method.

Cache key minimum proposal: hashes of input data and order, full temporal window, preprocessing and validity, encoder, projector/resampler/codebook, reader weights+config, precision/runtime, prompt prefix, multimodal positions and masks. A mismatch should recompute from the first affected stage. Cache-byte reporting includes positions, indices, retrieval summaries and metadata. Recomputing the encoder after every optimization step is the reference, not a failure of caching.

## Experiment T1: does discretization preserve useful expert distinctions?

**Falsifiable hypothesis:** at a fixed LLM context length, an EO code interface can retain expert-relevant class/region distinctions more efficiently in stored bytes, without worse grounding; a scale-aware code variant helps only if ordinary quantization demonstrably loses a relevant magnitude distinction. This is exploratory, not a preregistered improvement claim.

**Cheap first stage:** reuse the planned 200–1,000 real development observations once their contracts pass. Freeze one OlmoEarth checkpoint, extract each exact input once, and fit codecs/readouts on train regions only. Test (C) sufficiently trained continuous resampler; (D) plain VQ/FSQ; (R) residual/product or shape-plus-scale variant with the same final decoder/readout capacity. Full grid is an uncompressed reference, not a compute-matched competitor. A histogram of the same code IDs is a location/time-loss control. Do not retain a hidden uncompressed feature bypass in only one compressed arm.

**Two fair frontiers, not one misleading equality:**

- Context frontier: same 16/64/128 reader positions, metadata, reader and training/tuning budget. Multi-codebook entries can be combined into one position or serialized; count actual final positions, embedding parameters, FLOPs and GPU time. Discrete IDs expand to dense vectors inside the LLM, so small disk payload does not imply smaller LLM K/V.
- Storage frontier: same measured payload bytes (codes + residual/scale + position/time/validity + amortized dictionary cost) versus scalar-quantized/PCA/product-quantized continuous features. Equal token count alone is not equal bits. Show quality–bytes–latency curves; do not hide dictionary/training costs or compression/decompression time.

**Measures:** actual crop/parcel IoU or validated flood/standing-water confusion; paired expert-support retrieval consistency; latent reconstruction only as a diagnostic. Reflectance/index error is meaningful only if a decoder reconstructs the corresponding physically calibrated source, not merely latent dimensions. PASTIS annual labels do not evaluate event onset.

**Error tests:** code ID permutation with matching embedding-table permutation leaves predictions unchanged; permutation without matching table should fail; same code histogram with changed source arrangement must not imply same localized answer; unseen-region code occupancy/collapse and rare-class collisions; missing data differs from measured zero; serialization round trip reproduces codes/metadata. Report source-region uncertainty, not pixel counts as replication.

**Escalation gate:** if plain continuous/resampler dominates the rate–quality curve, keep continuous tokens and stop vocabulary expansion. If a discrete variant survives multiple support draws/regions, train encoder+interface jointly with matched native replay and the same labels; only this later stage can support improvement to OlmoEarth itself. Frozen-feature codec results alone cannot support that claim.

## Experiment M1: exact reuse first, then a bounded observation loop

**Falsifiable hypothesis:** repeatedly inspecting a region can reuse immutable computation with unchanged answers; an approximate memory policy offers an additional measured quality–latency advantage at a fixed total storage budget. Treat the two hypotheses separately.

**Cheap exact stage:** select a small set of real development episodes, deterministic evaluation and one pinned FP32 model initially. Compare (A) recompute all, (B) raw/EO feature cache with fresh language prefill, and (C) unchanged-prefix K/V reuse. Use repeated questions plus expert correction additions and 2/4/8-visit prefixes where real dates exist. All receive the same observations; there is no future observation preselection. Annual crop labels support crop questions, not timestamped change claims.

Check teacher-forced logits, top-1 and final grounded predictions against A before benchmarking latency. Declare tolerances before reading new results; retain max absolute/relative error and tie cases. Measure cold and warm encoder time, prefill, generation, cache load, peak allocated memory and total stored bytes. Timing includes data transfer. Repeated identical prompts are a serving benchmark, not a scientific accuracy gain.

**Required invalidation tests:** change one date's pixels, crop, validity mask, timestamp, temporal-window membership, prompt prefix, expert correction position, encoder weight, adapter/codebook weight, reader weight or precision. Cache must either be known unaffected or invalidate; a one-step encoder update must produce nonzero gradients and use current features. Compare old-window tokens with slices of appended-window tokens explicitly. Save/reload must preserve vocabulary, cache schema and positions; do not reuse caches from an unpinned runtime.

**Optional approximate stage only after exact stage passes:** fixed-byte comparison of recency, uniform/reservoir (causal), generic salience/StreamMem-style merging, and a candidate source-aware memory retaining selected observations/support. Use matched update compute and count summaries, source IDs, CPU/disk archive and re-encoding. Full archive is an upper reference. Unknown future questions must remain hidden from query-agnostic writers. A query-aware reader can retrieve only from that writer's retained data unless an explicit archive-retrieval setting is compared for all methods.

At most two extra inspection actions per answer in the first loop: select an existing date/region or request an allowed expert correction, then measure/answer. Compare fixed one-pass and fixed two-pass readers with the same token/observation budget. Do not equate generating more prose or self-agreement with new evidence. Cache acceleration alone is engineering; a research memory claim requires demonstrated bounded-memory grounding/retention improvements and stronger generic baselines.

## Integration into current plan

The existing real-data/continuous-reader baseline remains the first execution priority. Add T1 as a small interface diagnostic, M1 as correctness/serving validation, and a bounded inspection loop only if the task benefits from additional evidence. The same split, expert-label budget and independent EO evaluation continue to apply. Neither proposal grants new evidence that the present model is CVPR-ready.
