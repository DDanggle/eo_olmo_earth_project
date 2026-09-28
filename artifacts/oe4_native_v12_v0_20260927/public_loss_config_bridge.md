# Public checkpoint loss registry bridge

2026-09-27, staging-only correction in **`runtime_smoke_v1.py`**. Original `runtime_smoke.py`, downloaded checkpoint files, existing receipts and queued controller are unchanged by this agent.

## Observed failure

The actual downloaded public v1.2 Base configuration has neither `train_module.loss_config.loss_config.type` nor `train_module.contrastive_config.loss_config.type`. The former retains tau0.1, same-target threshold0.999 and the six decode-only modalities; the latter retains weight0.05. Outer `_CLASS_` values identify the **LossConfig dataclass**, not the registered loss implementation, so they cannot substitute for `type`.

Evidence: `/private/tmp/oe4_execute_20260927/receipts_snapshot_01/runs/data_prepare_v1/receipt.json`, with public config SHA256 `0d531a67ad3e477e7011efabcceb01ed80f430aa0a0a3d344fe18cec0f229b8a` and public weights SHA256 `57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec`.

## Source mechanism

Official `olmoearth_pretrain/train/loss.py:1303` does `self.loss_config.pop("type")`, mutating the supplied config. Both loss objects are built during train-module initialization. Official `olmoearth_pretrain/internal/experiment.py:317` builds the train module before `config.as_config_dict()` at line328. This provides a direct source path by which both registry keys disappear before checkpoint config serialization. In contrast, MaskingConfig first copies its strategy dictionary before popping `type`, and the actual public masking configuration retains its key.

The reviewed official v1.2 recipe, `scripts/official/v1_2/base.py:189`, specifies:

- Base: `modality_patch_discrimination_masked_negatives_vec`, tau0.1, same-target threshold0.999, decode-only modality list matching the public file.
- Contrastive: `InfoNCE`, weight0.05.

Recipe SHA256: `d88b23daca3d8a5657ede3ccbe98ce7e2ea6cdbf0e7b026b24fd15e61205f9ee`.

## Minimal correction

The new script parses those exact dictionaries from the pinned recipe's AST without executing the recipe or inventing parameters. It permits restoring missing keys only when the downloaded config SHA and recipe SHA match the reviewed values **and every remaining loss parameter exactly equals the recipe**. It deep-copies the training config and restores only the two missing `type` entries in memory. Explicit loss configs must also match this narrow v1.2 probe. Unknown/mismatched configurations fail closed.

The receipt records the original hashes, exact restored paths/values and source recipe. Both official loss objects and the masking-strategy constructor are now built **before any model/GPU allocation**, including in `--prepare-only`. Actual mask tokenization is still checked once the real model has loaded. Public files are not edited; saved inference snapshots continue to preserve the downloaded config, so a later request to train from a reserialized snapshot with a different config hash requires its own reviewed bridge or explicit restored config.

Local checks passed: Python compilation; AST extraction against the actual downloaded-config receipt; exact recipe/parameter match; no mutation of the input config. Actual official constructors require the server's isolated runtime and must be retried there before GPU requeueing. A previous failure at construction is not an optimizer update or training result.
