# [DRAFT] Issue for allenai/olmoearth_projects

> 제출 전 사용자 검토용 초안. 제출은 gh CLI 설치 후 또는 웹에서 수동.
> 대상: https://github.com/allenai/olmoearth_projects/issues

**Title:** LFMC released checkpoint scores MSE ~952 on the released test split (docs claim 580.6); retraining from the released dataset reproduces 559

**Body:**

## Summary

The LFMC model card (`docs/lfmc.md`) states *"It achieves a mean squared error of 580.6 on our test set."* Evaluating the released checkpoint (`allenai/OlmoEarth-v1-FT-LFMC-Base`) on the released rslearn dataset's test split yields **MSE 951.9** — but retraining from scratch with the released dataset + the repo's `model.yaml` recipe reaches **MSE 558.8 by epoch 33**, slightly better than the documented number. This suggests the dataset, config, and recipe are all fine, and the checkpoint file on HuggingFace may be from a different run (or a bad upload).

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

---
(Additional minor issues we hit while reproducing — happy to file separately:
`model.yaml` on main uses APIs not in any PyPI rslearn release; sample project's
`annotation_features.geojson` still uses the legacy `es_*` schema — PR ready.)
