# [DRAFT] Issue for allenai/olmoearth_projects

> 검토본(2026-09-13). 미제출. 본문은 아래 Body부터 사용한다. 제출 전 과거 평가 파일의 full hash,
> 실행 config/로그, historical master commit을 묶어야 한다. 외부 작성 이력은 있지만 현재 계정 권한은 미검증.
> 대상: https://github.com/allenai/olmoearth_projects/issues

**Title:** LFMC: test MSE 951.9 vs documented 580.6 — could you confirm the checkpoint and evaluation recipe?

**Body:**

## Summary

The [LFMC model card](https://github.com/allenai/olmoearth_projects/blob/23a3d7b799ba1fbb0c9138cb1444166ae1d3dd0a/docs/lfmc.md) reports test MSE **580.6**. In our recorded evaluation, the released checkpoint (`allenai/OlmoEarth-v1-FT-LFMC-Base`) yields test MSE **951.9** on the released dataset. A separate task fine-tuning run initialized from the pretrained OlmoEarth backbone reaches **558.8** at epoch 33 using the configuration with the compatibility edits below.

This shows that a nearby score is achievable in our setup, but does not reproduce the original run or establish an upload error. Could you confirm the checkpoint revision, dataset/split revision, and evaluation configuration corresponding to 580.6? We may be missing a normalization, aggregation, or other evaluation detail. The evaluation numbers below are from our earlier runs; the September 11 check rechecked public metadata, not a new evaluation.

## Reproduction

Environment: `olmoearth-runner==0.1.14`, `rslearn==0.0.27`, `lightning==2.5.1.post0`, Python 3.11, single H200.

1. Download released artifacts:
   - dataset: [20251029 dataset.tar](https://storage.googleapis.com/ai2-olmoearth-projects-public-data/projects/lfmc/20251029/dataset.tar) (44,022 windows)
   - checkpoint: `OlmoEarth-v1-FT-LFMC-Base/model.ckpt`
2. Evaluate released checkpoint on the test split (split property shipped inside the windows):
   ```
   rslearn model test --config model.yaml --ckpt_path model.ckpt \
       --data.init_args.test_config.tags.split=test
   # → test_mse 951.9  (4,585 test windows)
   # on the val split: 995.3
   ```
3. Start a new task fine-tuning run from the pretrained OlmoEarth backbone with the released dataset + repo recipe (two config
   edits for PyPI-rslearn compatibility: drop `enable_confusion_matrix`, swap
   `BestLastCheckpoint` → lightning `ModelCheckpoint`):
   ```
   # best (epoch 33 of 100): val_mse 652 → test split:
   # → test_mse 558.8
   ```

## Checks performed and remaining uncertainty

- **Two tested library environments**: evaluating the released ckpt under rslearn 0.0.27 vs
  the master checkout used at the time gives 995.3 vs 995.4 on val. That tested environment
  change does not explain the gap; it does not rule out all version/configuration effects.
- **Split confusion**: measured on both val (995.3) and test (951.9); docs number is
  for test. Our retrain evaluated with the exact same command/protocol.
- **Checkpoint metadata**: the released ckpt reports epoch 91 and 60,260 steps; our
  loader reports 655 steps/epoch. We have not reconciled epoch indexing, batching,
  accumulation, distributed sampling, or resumption, so this is not evidence by itself
  that different data were used.

## Requested clarification

Could you identify the weights, split, and evaluation recipe used for the documented
score? If they differ from the released artifacts, a revision-pinned link or updated
documentation would help. We can provide the full logs, executed configurations,
split counts, and our separate fine-tuning result for comparison.

## Public artifact metadata (file/card rechecked 2026-09-13; repo metadata checked 2026-09-11)

- HF repository revision: `92f32915e29014d50b55f43522a91e3dd2fe610a`, repository lastModified 2025-11-03.
- `model.ckpt`: 1,139,505,083 bytes; LFS SHA-256 `20064f6a0a7a70acc1d5e304bc8050bbe07e86891a60806ee585fffbeb86f92f`.
- File lastCommit: `0c27933f49c5ec99444783845481bf7b972af812`, 2025-10-30. The repository modification date is not the file upload date.
- `docs/lfmc.md` on `main` (23a3d7b) still states 580.6.
- Related open issues #45 (dataset path) and #46 (label format) do not cover this mismatch.

(The `model.yaml` API remarks from an earlier draft are dropped: current rslearn releases carry those APIs; the remaining problem is the pinned lock, tracked separately.)
