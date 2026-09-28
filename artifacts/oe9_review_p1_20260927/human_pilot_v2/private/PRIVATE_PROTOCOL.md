# Coordinator-only v2 sampling and feasibility rules

Do not distribute this file, the private mapping, the build contract, or a parent directory containing them to reviewers. Distribute only `reviewer_package/`.

## Why a separate revision exists

The untouched v1 package's fixed label-blind candidate sample contained target 2, counterexample 1, neither 11, and reference_unavailable 6. No human annotations had been collected. Its outcome was `pilot_design_insufficient_informative_cases`, not a failed human-readability test. Preserve that package and its report unchanged.

V2 is a new preparation protocol fixed before human answers and without using model scores or image visibility. It uses public annual semantic labels to construct informative reference strata: exactly 8 target, 8 counterexample, 4 neither, each from a different training query patch. These are provided-label sampling strata, not expert truths or known expected human responses. Class proportions are artificial and cannot estimate natural prevalence or representative overall accuracy.

## Deterministic selection

1. Reconstruct the exact v1 twenty query IDs using the original v1 query hash on the frozen train_pool48. Only train_pool inputs, labels, and support objects are eligible. Development and source-bank patches are excluded.
2. In each training query, enumerate existing instance geometries with at least 64 pixels. Require a dominant annual crop class 1..18 with at least .95 purity over the complete instance. Exclude background/void and mixed candidates that fail the threshold. Do not rank cloudiness, contrast, object area, model success, or human confidence.
3. Pair each eligible candidate with each existing K=1 task for that query. All supports must remain query-disjoint train_pool objects. The public source class of the candidate and the task's two class definitions determine its target/counterexample/neither reference stratum.
4. Keep the canonical hash-minimum option for each query×stratum. Solve a deterministic bipartite matching of the twenty stratum slots to distinct query IDs. Try the original twenty first, with equal counts per source parent within each stratum (4/4,4/4,2/2), then without that regional constraint. Retaining all original query IDs has priority over perfect regional balance.
5. Only if the original twenty cannot satisfy the exact quotas, try all train_pool48, first with the regional constraint and then without it. Record every attempt, the chosen cohort, the precise IDs, whether all v1 IDs were retained, parent counts, and parent×stratum counts. If no unique-query matching exists, stop with an explicit error. Never duplicate or pad a query.
6. After matching, independently hash-order the selected query IDs before assigning anonymous P001..P020. A/B display orders use their separate hash namespaces. Neither anonymous IDs nor position may encode a contiguous stratum block.

The candidate outline remains in original 128×128 source coordinates. The RGB/NIR images, outlines, selected source instance, and private reference must agree. Recheck the reference class and stratum after loading the selected mask for rendering. Keep the positive/counterexample supports of the chosen K=1 episode unchanged.

The reviewer payload, JavaScript, HTML, image filenames, and public protocol contain no query/source IDs, automatic category mapping, exact quota, class IDs/names, or candidate reference classes. Numeric case count and acquisition dates remain public. Private hashes and selection metadata support reproducibility without becoming prompt content.

## Decision rules unchanged from v1

Two distinct independently assigned people each complete all 20 cases. Missing or unfinished work stays missing/unfinished. Independence still needs coordinator verification.

The original explicitly provisional operating assumptions remain unchanged: reference availability needs at least three target cases, three counterexample cases, and eight combined. In the informative target/counterexample reference stratum, at least 75% must receive target/counterexample responses from both reviewers and at least 75% must have exact agreement; each reviewer must use target and counterexample at least twice. Per reviewer, median active time must be at most 600 seconds and nearest-rank p90 at most 900 seconds. Report numerator/denominator, all five response counts, strata, descriptive kappa, and timing distributions. These are workflow assumptions, not D1 rules, validated scientific cutoffs, or a CVPR threshold.

The scorer is deliberately unchanged. It never counts software fixtures as humans, and an overall agreement score cannot by itself pass. Public-reference agreement is not expert accuracy. V2 gives a better-posed candidate-identification/time exercise; it does not validate free-form corrections, human labeling of full masks, or correction-conditioned learning efficiency. Human annotations collected remain zero until actual independent exports arrive. No budget change follows.

## Execution

Use the same CLI as v1, but a new output directory:

```sh
python build_human_pilot.py --prepared-root PREPARED80 --episodes-root EPISODES --out OUTPUT_V2
python score_human_pilot.py --package OUTPUT_V2/reviewer_package --private-reference OUTPUT_V2/private/reference_mapping.json --out pending_v2.json
```

After real reviews, supply `--a A.json --b B.json`. The response schema and UI are unchanged from the browser-tested common implementation. V2 has its own package identity and sampling protocol marker, so responses cannot be scored against v1 accidentally.
