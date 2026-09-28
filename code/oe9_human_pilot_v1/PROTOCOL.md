# EO candidate identification and timing pilot v1

This is a 20-case, two-reader feasibility pilot. It measures whether people can distinguish an existing image candidate using positive and counterexample sequences and how long this takes. It does not measure model learning, expert-correction label efficiency, newly drawn dense masks, temporal change accuracy, or generalization to a held-out region. No completed human annotations exist until actual independently collected responses are received.

## Cohort and stimulus

The 20 query patch IDs are the first 20 under the fixed `oe9-human-candidate-v1:query` hash among the frozen 48 `train_pool` IDs. Neither development patches nor the source support bank are eligible. Selection is frozen before any query label is opened. There is no replacement based on visibility, crop class, task difficulty, or a favorable result.

Each selected query receives one deterministic K=1 episode from the previously source-defined task pairs. Supports are query-disjoint `train_pool` objects. A single existing query instance geometry of at least 64 pixels is chosen by an independent hash, without reading its semantic class for candidate selection. This is a candidate-identification task, not free-form discovery or new full-mask annotation. Object geometry is a provided annotation, so the entire task is not annotation-free. There is no claim that a selected candidate represents the whole scene or that class absence was assessed throughout the patch.

The query and each of its two support objects have eight source observations. Show source dates, neutral candidate outlines, natural color (B04/B03/B02), and optional NIR false color (B08/B04/B03). Use a fixed raw clip of [0,3000] and gamma 2.2 for all images. Do not interpret the display as calibrated reflectance measurements. Nodata is purple; unmarked pixels may still contain clouds, shadows, haze, or other observation limitations. No cloud masks were supplied.

The supplied positive/counterexample masks come from public dataset labels. They are synthetic task examples, not fresh expert corrections. The private query reference is an annual public class label and is not human-verified expert gold. Query-source IDs, crop-class IDs/names, automatic reference answers, AI opinions, and model predictions are omitted from the distributed package.

## Two independent reviews

Assign A and B to two different people. A/B identifies reader assignment, not alternate models or different evidence. They review the same anonymous cases in different predetermined orders. Give each person only the `reviewer_package` directory. They must not discuss cases or inspect each other's answers until both independent exports are frozen. Separate devices/browser profiles are recommended. The software's distinct IDs do not prove reviewer independence; the coordinator must verify the process.

Open `index.html` directly or serve only the reviewer directory with a local static server. Never serve its parent containing `private/`. Enter an anonymous reviewer ID and assigned A/B slot. The first start reveals the images. Select evidence dates from the query and optionally the examples. Record one of:

- `target`: the candidate matches the positive example;
- `counterexample`: it matches the counterexample;
- `neither`: it appears to be a different kind from both;
- `uncertain`: observations are available but kind cannot be reliably decided;
- `unobservable`: clouds, resolution, missing relevant dates, or another viewing limit prevent the required observation.

Record a concrete reason, relevant limitation flags, and confidence. An unobservable response needs its limitation; any completed response needs at least one query evidence date. A neither response refers only to the outlined candidate. Untouched or unfinished cases remain explicitly untouched/unfinished. Do not fill missing reviewers or cases with guessed answers, duplicated responses, or AI responses.

Pause for interruptions. Timing records start/stop intervals, wall elapsed time, and estimated active time in a foreground page with interaction during the preceding 60 seconds. Use “계속 관찰 중” while studying an image for an extended period. Tab hiding pauses the timer; interrupted browser sessions recover only through their last persisted timestamp. These are self-recorded effort estimates, not verified human-attention telemetry. Export JSON after the session. Answers are saved only in local browser storage until export.

## Initial operational decision rules

These are explicit planning assumptions for this task, not borrowed D1 criteria, validated scientific cutoffs, or a paper acceptance criterion. Freeze them before receiving human responses.

1. Obtain all 20 completed cases from each of two distinct independent reviewers. Until then, report actual completed counts and wait; do not issue a pass/fail verdict.
2. Use the public-label private reference only to describe availability strata. Require at least eight target-or-counterexample candidate cases in total, with at least three of each. If the fixed cohort contains too many neither/mixed cases, report insufficient informative coverage. Keep the original sample unchanged. A revised stratified sample would be a separate future protocol.
3. Within this informative reference stratum, both reviewers must issue target/counterexample decisions on at least 75% of cases, and exact agreement across the full informative stratum must reach 75%. In addition, each reviewer must use both target and counterexample at least twice, preventing a constant answer from passing. Report numerator and denominator, the complete five-way confusion matrix, and neither/uncertain/unobservable counts. Overall agreement alone cannot pass.
4. Per reader, median active time should be at most 10 minutes and nearest-rank p90 at most 15 minutes per case for the first task to be practical. These limits are planning assumptions; retain interval records to diagnose undercounting, long idle observation, or workflow issues. No person-hour or monetary budget is changed by this protocol.

Report exact agreement and descriptive Cohen's kappa separately overall and in each public-reference availability stratum. At 20 cases kappa is unstable and may be undefined; do not replace undefined values with zero. The scorer does not call concordance with public labels expert accuracy. Agreement can still reflect shared mistakes, so actual disagreements and representative agreements should receive a later independent adjudication before treating annotations as research gold.

A provisional pass permits planning a larger candidate-identification annotation exercise. It does not validate free-form expert corrections, counterexample learning, training gains, temporal explanations, or a CVPR contribution. A low coverage result asks for a revised task sample; unobservable cases ask for improved rendering/data; disagreement asks for improved task instructions or domain expertise. Keep all three diagnoses separate.

## Generator and scorer

CPU generation from the verified OE8 data:

```sh
python build_human_pilot.py --prepared-root PREPARED80 --episodes-root EPISODES --out OUTPUT
```

Distribute only `OUTPUT/reviewer_package/`. Keep `OUTPUT/private/` and `build_contract.json` with the coordinator.

After collecting actual A/B exports:

```sh
python score_human_pilot.py --package OUTPUT/reviewer_package --private-reference OUTPUT/private/reference_mapping.json --a A.json --b B.json --out pilot_report.json
```

Omit `--a` and/or `--b` to report the actual pending state without fabricating annotations. `--allow-synthetic` is only for software tests and always produces a synthetic-only report that cannot pass the human gate. Test fixture files must be visibly marked and stored separately from actual responses.
