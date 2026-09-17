# OlmoEarth — what I built, what I fixed, what I'd ask for (DG, 2026-09-17)

Thanks again for the call last week. Below is the one-page summary I promised, plus a request for a Studio account so I can try the same workflows on the platform side.

**[nepal-twin]** — https://github.com/DDanggle/eo-rasuwa
- Before/after change-review map for the 2026 Nepal (Rasuwa) disaster, built on frozen OlmoEarth v1 Base embeddings; no fine-tuning, no event labels.
- Pipeline: images → embeddings → change score (1 − cosine, pre vs post) → placebo baseline from normal pre-event change (p99) → ranked windows for review.
- Added Sentinel-1 when optical was cloud-limited: scorable windows went from 47 to 97 of the same 100; two radar tracks shared 9 of their top 10 locations.
- Against UNOSAT's map, my 6 flagged windows overlapped their mapped source area ~7.06% on average vs 0.093% for the other 50; the top radar window was 1.2 km from the mapped source lake (first news report: 2.9 km).
- Baselines: on Sen12-Landslides, OlmoEarth embeddings beat NDWI; on the Nepal flood check, NDWI scored higher on AUPRC. I report both.
- Nano matched Large on this overlap check (Base was used for the published run).

**[PR / issues]**
- (links to be added)

**[end-to-end pipeline I built]**
- Went through the full loop once: raw Sentinel-2/1 → rslearn dataset → OlmoEarth embedding cache → decoder training → evaluation, on Sen12-Landslides (15 regions), KuroSiwo (S1 flood, GEO-Bench-2) and Korean AI-Hub land-cover/logging/landslide labels (public data, spatial holdout).
- One cache serves several task heads; reusing frozen embeddings with small decoders beat training a raw model from scratch on the same labels in every case I measured.
- Streaming update: a small GRU updater refreshes a stored embedding with new single-date embeddings instead of re-encoding the whole window. Recovers 77–95% of the re-encoded decoder AP on Sen12 and 107–115% on KuroSiwo, at 2.3× less wall time and 4.5× less raw read.
- Things I had to fix along the way to make OlmoEarth run for me: sample quick-start schema (PR #68), SCL cloud scoring read with bilinear resampling on categorical values (local nearest patch; PR coming), a radar dB scaling bug in my own inputs, `load_all_crops` OOM at 1024 px (workers 6 / batch 4), a deterministic-mode pooling failure (replaced with an equivalent mean), and the runner/rslearn lockfile skew (runner 0.1.14 + rslearn 0.0.27 works).
- Also found that the released LFMC checkpoint scores test MSE 951.9 vs 580.6 in the docs; retraining on the released data gives 558.8 (issue #69).

**[comparison with other models]**
- (to be added from separate notes)

**[6 improvement requests]**
1. Cloud: a per-token "this is cloud, don't trust it" signal would remove the SCL mask + hand rule (≥20% clear) I run outside the model; cloud handling took the most time.
2. Scale: 1 − cosine has a different typical size per model (Nano ≈0.10, Base ≈0.25, Large ≈0.02); a "normal-times Δ" number in the model card, or a calibrated score, would remove the placebo machinery.
3. Season: monsoon-month normal change pushed my threshold up; a seasonal notion of "normal" would separate event change from seasonal change.
4. Sensors: for the same place, S1 and S2 change values have different distributions, so I normalized per sensor; a truly sensor-invariant space should give similar change from either — measurable on your side.
5. Resolution: houses, roads and bridges vanish at 40 m; a 10 m token mode for built-up areas would unlock recovery monitoring. (Caveat from my side: just extracting at 20 m did not help small objects in my landslide tests.)
6. Size guidance: Nano was enough for my task but everyone downloads Base (52,768 vs 8,517); one line in the model card would help.

**[ask]**
- A Studio / platform account, so I can run the Nepal and Korea workflows on your side and send feedback on where partners get stuck.
