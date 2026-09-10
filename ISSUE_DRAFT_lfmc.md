# [DRAFT] Issue for allenai/olmoearth_projects

> 최종본(2026-09-10). 제출은 `gh issue create --repo allenai/olmoearth_projects` 또는 웹. 이슈 작성은 외부 계정에 열려 있음(#40~#65 확인).
> 대상: https://github.com/allenai/olmoearth_projects/issues

**Title:** LFMC: released checkpoint evaluates to test MSE 951.9 vs 580.6 in docs/lfmc.md; retraining on the released dataset reaches 558.8 — can you confirm the checkpoint revision?

**Body:**

## Summary

The LFMC model card (`docs/lfmc.md`) states *"It achieves a mean squared error of 580.6 on our test set."* Evaluating the released checkpoint (`allenai/OlmoEarth-v1-FT-LFMC-Base`) on the released rslearn dataset's test split yields **MSE 951.9** — but retraining from scratch with the released dataset + the repo's `model.yaml` recipe reaches **MSE 558.8 by epoch 33**, slightly better than the documented number. So the released dataset, config and recipe reproduce the documented number; the gap is specific to the released weights. One hypothesis is that the HF file is from a different run than the documented one, but we may also be missing an evaluation detail (normalization, split definition, or a newer revision) — asking for confirmation rather than asserting a bad upload.

## Reproduction

Environment: `olmoearth-runner==0.1.14`, `rslearn==0.0.27`, `lightning==2.5.1.post0`, Python 3.11, single H200.

1. Download released artifacts:
   - dataset: `.../projects/lfmc/20251029/dataset.tar` (44,022 windows)
   - checkpoint: `OlmoEarth-v1-FT-LFMC-Base/model.ckpt`
2. Evaluate released checkpoint on the test split (split property shipped inside the windows):
   ```
   rslearn model test --config model.yaml --ckpt_path model.ckpt \
       --data.init_args.test_config.tags.split=test
   # → test_mse 951.9  (4,585 test windows)
   # on the val split: 995.3
   ```
3. Fine-tune from scratch with the released dataset + repo recipe (two small config
   edits for PyPI-rslearn compatibility: drop `enable_confusion_matrix`, swap
   `BestLastCheckpoint` → lightning `ModelCheckpoint`):
   ```
   # best (epoch 33 of 100): val_mse 652 → test split:
   # → test_mse 558.8
   ```

## Controls we ran (to rule out our own error)

- **Library-version effect**: evaluating the released ckpt under rslearn 0.0.27 vs
  current master gives 995.3 vs 995.4 (val split) — identical, so not a version issue.
- **Split confusion**: measured on both val (995.3) and test (951.9); docs number is
  for test. Our retrain evaluated with the exact same command/protocol.
- **Checkpoint metadata**: released ckpt reports epoch 91, 60,260 steps → 662
  steps/epoch, while the released dataset yields 655 steps/epoch — a ~1% difference
  suggesting it was trained on a slightly different data snapshot.

## Suggested fix

Re-upload the checkpoint corresponding to the documented run (or update docs).
Happy to share full logs/configs, or the 558.8 checkpoint if useful.

## Artifacts checked (2026-09-10)

- HF `allenai/OlmoEarth-v1-FT-LFMC-Base` `model.ckpt`: 1,139,505,083 bytes, LFS oid prefix `20064f6a0a7a`, last modified 2025-11-03 — unchanged since our measurement.
- `docs/lfmc.md` on `main` (23a3d7b) still states 580.6.
- Related open issues #45 (dataset path) and #46 (label format) do not cover this mismatch.

(The `model.yaml` API remarks from an earlier draft are dropped: current rslearn releases carry those APIs; the remaining problem is the pinned lock, tracked separately.)
