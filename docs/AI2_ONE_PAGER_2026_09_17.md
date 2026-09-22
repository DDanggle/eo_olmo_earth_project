# OlmoEarth — what I built, what I fixed, what I'd ask for (DG, 2026-09-17)

Thanks again for the call last week. Here is the one-page summary I promised, plus a request for a Studio account.

**Nepal change-review map** — https://github.com/DDanggle/eo-rasuwa
- Before/after review map for the 2026 Rasuwa disaster on frozen OlmoEarth v1 Base embeddings. No fine-tuning, no event labels.
- Pipeline: Sentinel-2/1 → embeddings → change score (1 − cosine, pre vs post) → placebo threshold from normal pre-event change (p99) → ranked windows for a human to read.
- Adding Sentinel-1 through the same embedding space raised scorable windows from 47 to 97 of 100 when optical was cloud-limited.
- Against UNOSAT's map, my 6 flagged windows overlap the mapped source ~7 % on average vs 0.09 % for the other 50; the top radar window is 1.2 km from the source lake (first news report: 2.9 km).
- Nano matched Base on that check. On Sen12-Landslides the embeddings beat NDWI; on the Nepal flood check NDWI scored higher. I report both.

**Two model families on the same contract** — same 56 Nepal windows, same dates, same placebo threshold; only the encoder changes. Scored against the UNOSAT source polygons.
- OlmoEarth Nano through Large (1.4 M → 308 M params): all four put the same source windows in their top 6, with identical overlap (p ~ 0.002). No gain from size on this task, so Nano is enough for rapid triage.
- Galileo (nasaharvest, nano / tiny / base) on the identical contract: nano and base also point at the source, and Galileo nano shares its top window with OlmoEarth Base. Galileo tiny fails outright. Size-invariance here is a property of the OlmoEarth family, not of the task.
- Limits I'd state out loud: only 4 + 4 windows touch the polygons, so this yardstick is effectively pass/fail and cannot rank the models that pass. Whole-ranking correlation across families runs -0.07 to 0.71 — the families agree on the strong signal, not on the ordering. One event, one region.

**End-to-end pipeline I ran on your stack**
- Full loop once: raw S2/S1 → rslearn dataset → embedding cache → small decoder → evaluation, on Sen12-Landslides (15 regions), KuroSiwo (S1 flood) and Korean AI-Hub land-cover / logging / landslide labels with spatial holdout.
- One cache serves several heads; frozen embeddings + small decoders beat a raw model trained from scratch on the same labels in every case I measured.
- Streaming update: a small GRU refreshes a stored embedding with new single-date embeddings instead of re-encoding the window. 77–95 % of re-encoded AP on Sen12, 107–115 % on KuroSiwo, at 2.3× less wall time and 4.5× less raw read.

**What I had to fix to make it run** (PR / issue links)
- PR #68 — quick-start sample `annotation_features.geojson` migrated to the `oe_*` schema: https://github.com/allenai/olmoearth_projects/pull/68
- Issue #69 — released LFMC checkpoint scores test MSE 951.9 vs 580.6 in the docs; retrained on the released data gives 558.8: https://github.com/allenai/olmoearth_projects/issues/69
- Local fixes, PRs to follow: SCL cloud mask resampled bilinearly on categorical values (nearest fixes it); `load_all_crops` OOM at 1024 px; a deterministic-mode pooling failure; runner/rslearn lockfile skew (runner 0.1.14 + rslearn 0.0.27 works).
- On my side: an S1 dB-vs-linear mistake that failed silently — the pipeline ran, the numbers looked fine, the source vanished. Neither model card states the expected unit; one line would prevent this.

**Jeju oreum tracker — same recipe, Korea-only data** (in progress, not public yet)
- 243 volcanic cones, annual change 2025→2026, same frozen Base + placebo p99, with a hashed pre-registered contract so scene selection can't be tuned after seeing the map. 153 scored, 90 abstain ("could not see", never "no change").
- Korea-only inputs: VWorld aerial imagery (~0.3 m) as the human-reading background, cadastral parcels joined to 2026 development permits, conservation / heritage / national-park layers as denominators.
- Lesson from the other side: scene selection dominates ranking. Re-picking scenes under a 6-year contract keeps 1 of the top 10; cones with 20–40 % valid tokens collapse, ≥ 80 % hold. The map dims low-validity cones and marks the 6 that survive both contracts.
- Now: a person labels the top 20 against the aerial imagery (real change / persistent structure / cloud / unreadable) as ground truth, then a bare-soil index from the aerial imagery as a linear probe on the 40 m embeddings. Climate data joins are next; no number yet.

**Six requests**
1. Cloud: a per-token "don't trust this" signal would replace the SCL mask + hand rule I run outside the model. Cloud handling took the most time.
2. Scale: 1 − cosine has a different typical size per model (Nano ≈ 0.10, Base ≈ 0.25, Large ≈ 0.02). A "normal-times Δ" number in the card, or a calibrated score, would remove the placebo machinery.
3. Season: monsoon-month normal change pushed my threshold up. A seasonal notion of "normal" would separate event change from seasonal change.
4. Sensors: S1 and S2 change values have different distributions for the same place, so I normalize per sensor. A sensor-invariant space should not need that.
5. Resolution: houses, roads and bridges vanish at 40 m. A 10 m token mode for built-up areas would unlock recovery monitoring. (Extracting at 20 m alone did not help small objects in my tests.)
6. Size guidance: Nano was enough for my task, but everyone downloads Base. One line in the card would help.

**Ask**: a Studio / platform account, so I can run the Nepal and Korea workflows on your side and report where partners get stuck.
