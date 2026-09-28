# EO candidate identification and timing pilot — reviewer instructions

This is a 20-case, two-reader feasibility exercise. Identify an already outlined image candidate using positive and counterexample observation sequences and record the time required. It is not a model comparison, free-form object discovery, or a task to draw new dense masks.

Cases and examples come only from the study's training partition. Public annual class references were used to make the candidate types informative for the exercise, before any human responses or model scores were considered. The images were not selected because they looked easy or a model performed well. These source labels are not fresh expert gold. The private reference answers and the planned distribution of candidate types are withheld from the reviewer package; do not try to infer an answer from the anonymous case number or sequence position.

Each candidate and each example has eight source observation dates. Compare natural color (B04/B03/B02) and optional near-infrared false color (B08/B04/B03). Every image uses the same fixed display transformation: raw values clipped to [0,3000], gamma 2.2. This display is not a quantitative reflectance measurement. White outlines identify the existing candidate geometry. Purple marks input nodata. Other pixels may still contain cloud, haze, shadows, or inadequate detail; no certified cloud masks were available.

The positive and counterexample supports are synthetic task examples made from public dataset annotations. The selected query is another existing candidate. A candidate's geometry is supplied, but its category is not. Identifying that candidate does not establish whether the target is absent throughout the entire scene.

## Independent review

A and B must be two different people. They see the same anonymous cases in different fixed orders; A/B does not indicate a model or alternate visual condition. Use your assigned anonymous ID and A/B slot. Do not exchange judgments or view another reviewer's export until both reviews are frozen. Separate browser profiles or devices are recommended. Distinct IDs alone do not certify independence; the coordinator must verify the process.

Open `index.html` directly or serve only this reviewer directory. Select “시작 / 재개” to begin a case. Compare the outlined query candidate with the two examples and choose:

- **대상 / target:** matches the positive example;
- **혼동 / counterexample:** matches the counterexample;
- **둘 다 아님 / neither:** appears to be a different kind from both;
- **불확실 / uncertain:** observations can be viewed, but kind cannot be reliably decided;
- **관측 불가 / unobservable:** clouds, resolution, relevant dates, or another observation limit prevent seeing what is needed.

Select the query dates supporting the decision, and optionally relevant example dates. Give a concrete reason, limitation flags, and confidence. An unobservable response must say what prevented observation. Neither concerns only the outlined candidate. Do not fill an untouched case or missing person's response with a guessed or AI-generated answer.

Pause when interrupted. Time is recorded as start/stop intervals, wall elapsed time, and an active-time estimate that counts foreground viewing with interaction during the preceding 60 seconds. Press “계속 관찰 중” during extended inspection. Hiding the page pauses timing. This is an effort estimate, not verified human attention. Untouched and unfinished cases remain distinct from completed reviews.

Export your JSON when finished or before changing devices. Answers otherwise remain only in this browser's local storage. This interface neither imports nor displays another reviewer's answers. No automatic reference answers, AI opinions, or model predictions are included in the distributed package.

## Interpretation

The coordinator will report actual completed counts, five-way agreement, response distributions, observation limitations, and time. Overall agreement alone is insufficient if it comes from a single repetitive response. Public annual labels define the task's availability strata and do not replace expert adjudication. Disagreements and representative agreements need subsequent review before answers become research gold.

This pilot can guide the next annotation workflow. It cannot establish correction learning efficiency, dense-mask annotation cost, change detection quality, geographic generalization, or a paper's scientific contribution. The existing human-work budget remains unchanged.
