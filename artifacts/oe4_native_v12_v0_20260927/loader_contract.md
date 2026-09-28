# OE4: official v1.2 Base loader and bounded training contract

Reviewed 2026-09-27. This is a staging artifact. No remote execution, package installation, download or GPU run was performed by this reviewer. `runtime_smoke.py` passed local Python syntax compilation and its argparse help runs. Real import/model/H5/backward validation remains the server run's job.

## What can run first

Use full official source commit `0497dfbb6711ded4e6bf10cf089fc1e4d58c186b`, public `allenai/OlmoEarth-v1_2-Base` revision `2e99a734a30e9aeb993aaa39946c6dcf554739e0`, and the official 1k H5 archive. The supplied standalone loop uses the **unchanged official loader, normalization, masking and objective implementations** with PyTorch AdamW, and records actual encoder gradients/weight deltas. It is a bounded engineering run, **not a reproduction of the full official continued pretraining recipe**.

Initial two-scene train/two-scene diagnostic smoke:

```sh
env -u PYTHONPATH /home/work/data/olmoearth/.venv-master/bin/python \
  /home/work/data/olmoearth/oe4_native_v12_v0/code/runtime_smoke.py \
  --source-root /home/work/data/olmoearth/oe4_native_v12_v0/source \
  --deps-root /home/work/data/olmoearth/oe4_native_v12_v0/deps \
  --checkpoint-dir /home/work/data/olmoearth/oe4_native_v12_v0/models/OlmoEarth-v1_2-Base \
  --h5-root /home/work/data/olmoearth/oe4_native_v12_v0/data/official_subset_1k \
  --out-dir /home/work/data/olmoearth/oe4_native_v12_v0/runs/real_h5_smoke_001 \
  --device cuda:0 --train-files 2 --dev-files 2 --steps 2 \
  --batch-size 2 --crop-size 32 --timesteps 2 --patch-size 4 \
  --lr 0.000001 --max-wall-seconds 900 --save-weights
```

Root must select an actually idle GPU immediately before launch; the example `cuda:0` is not a reservation. The `code/` path is the proposed published location, not proof of transfer. Output directories must be new; the script refuses an existing directory.

After that run passes, a useful bounded engineering development run uses `--train-files 64 --dev-files 8 --steps 32 --batch-size 2`, a new output directory, and the same controls. This exposes 64 distinct train files once. Increasing requested train files without increasing steps does not expose all of them; the receipt lists exact exposed indices. `--prepare-only` performs data/receipt preparation without constructing the model or using CUDA, but still verifies required imports. No outcome-based sample selection occurs.

## Runtime dependencies and source identity

- `model_loader.load_model_from_path` can load inference models without olmo-core via the standalone Config implementation. **Correction to preliminary advice:** importing official `train.loss` or `train.masking` executes `train/__init__.py`, which requires a working `olmo_core.config.Config` import. Therefore this official-objective loop does require olmo-core. Do not fake or bypass the package guard.
- Root's isolated `--target ROOT/deps` installation strategy leaves the existing master environment unchanged. Required new packages include `ai2-olmo-core` (with its actual import dependencies), `class-registry>=2.1.2`, and `hdf5plugin>=6.0.0`. Also required: torch, numpy, einops, h5py, universal-pathlib and loader dependencies. `--no-deps` installation is only sufficient if transitive imports are already present; test the actual imports before claiming readiness.
- The full source declares torch `>=2.9,<2.10`, Python `>=3.11,<3.14`. The observed server torch 2.13/CUDA13/Python3.11 runtime is a **compatibility development environment**, not the official pinned runtime. Preserve and report it; don't mutate the existing environment to satisfy a declared range.
- The script puts `source` first and isolated `deps` second in `sys.path`, asserts the imported source package is under that source root, hashes the loader/loss/masking/recipe, and hashes both public checkpoint files.
- Do not use the minimal inference package for this objective loop. Its current loader restricts input modalities to S1/S2/Landsat, including the target encoder. Official training targets require decode-only map modalities as well.

## H5 contract

Official entry points: `olmoearth_pretrain/dataset/convert_to_h5py.py`, `olmoearth_pretrain/data/dataset.py:read_h5_file`, `_fill_in_missing_timesteps`, `normalize_image`, and `olmoearth_pretrain/dataset/sample.py`.

| Field | Storage | Rule |
|---|---|---|
| `sentinel2_l2a` | `[H,W,T_available,12]` | Raw DN reflectance; normalize once with official computed statistics. |
| `sentinel1` | `[H,W,T_available,2]` | Already converted to dB by H5 conversion. Do not log-transform again. Not consumed by this first script. |
| `worldcover` | ordinarily `[H,W,1]` | Static raw categorical code. Add singleton time axis to `[H,W,1,1]`; official native target uses its normalized value, not an invented one-hot mask. |
| `timestamps` | `[T_longest,3]` | `[day, month-1, year]`; real day is 1-based, month 0-based. |
| `missing_timesteps_masks/<modality>` | `[T_longest]` | **True means present**, despite the group's name. The sensor T-axis stores only present timesteps. |

To read the first two real S2 observations, read compact data `[..., :2, :]` and timestamps at `flatnonzero(present)[:2]`. Never align compact observations to timestamps `[:2]` without checking the presence mask. Exact compact-count/mask consistency is asserted. Without a mask, direct alignment is accepted only if sensor T equals timestamp T.

The first script selects aligned center crops from S2 and WorldCover, requiring equal spatial sizes and no sentinel missing pixels. It rejects malformed/short/nonfinite/duplicate crops and records every rejection. This deliberate fully observed crop subset is a loader/backward probe; it cannot establish cloud/missingness robustness. The script does not interpret zeros as cloud labels. `hdf5plugin` must be imported before compressed H5 reads.

Normalizer: official `Normalizer(Strategy.COMPUTED)` uses mean ± 2 standard deviations as the scaling range, equivalent to `(x-mean)/(4*std)+0.5`, with no clipping; official predefined fallback is used when needed. Missing sentinel is `-99999` and would be restored after normalization; the first run rejects such crops. WorldCover's computed normalization was checked in the local source and works.

## Model and objective

v1.2 uses one S2 bandset with order `[B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09]`, unlike v1's three bandsets. Its model config carries `rope_3d_mixed` positional encoding and a patch embedding MLP hidden size `[64]`. Load the public config rather than constructing a v1 lookalike. The script propagates `model.encoder.tokenization_config` into official masking and asserts that S2 has one bandset.

`MaskedOlmoEarthSample` mask values: online=0, target-only=1, decoder=2, missing=3. `unmask()` maps every nonmissing entry to online while retaining missing entries. Encoder token shape for S2 is `[B,H/patch,W/patch,T,1,D]`.

Official v1.2 native objective from `scripts/official/v1_2/base.py`:

- Random/time masking with encode/decode ratio .5, random ratio .5, designated decode-only maps.
- `modality_patch_discrimination_masked_negatives_vec`, tau .1, same-target threshold .999; same-target masking for map modalities.
- Two independently masked views; average base loss plus InfoNCE with weight **.05**, not v1's .1.
- Frozen target, all target exits at depth0, EMA `(1,1)`. The script asserts those public checkpoint settings; it fails if the checkpoint needs unimplemented regularization/MAE/EMA behavior.
- Online band dropout is explicitly enabled using `model.encoder.enable_band_dropout()`, matching the official trainer initializer. `model.train()` alone is insufficient. The target's effective rate is asserted zero; evaluation disables online dropout through eval mode.
- Loss is computed outside autocast. For the first probe the entire loop uses FP32. Optional external flash-attn is disabled in favor of official SDPA and any changed module flags are recorded.

**Scope restriction:** this first input subset has S2 plus decode-only WorldCover. With the official one-bandset role selection, S2 supplies encoded context and WorldCover supplies the decoded patch target; InfoNCE links two masked S2 views. It is therefore a native-loss partial-modality probe, not full S1/S2/Landsat plus all-map training. This matters when interpreting any improvement. Full official-style baseline preparation still needs the complete modality/timestamp loader, full sampling/augmentation schedule, and balanced exposure accounting.

Differences from the full recipe are explicit: small fixed crops and times, partial modalities, tiny global batch, no spatial augmentation, FP32/no FSDP, constant development LR (default 1e-6), no warmup/scheduler, only bounded steps. AdamW weight decay .02 and gradient clipping1 are retained. These choices prove the path and measure runtime; they do not estimate a final training budget.

## Receipts and stop conditions

`data_manifest.json` is written before model metrics and hashes selected raw crop values/timestamps, records filenames/crops/latlon when available, and distinguishes train vs dev. The split is **file-disjoint only**. Multiple H5 subtiles may come from the same larger region; neither this split nor these two original-pretraining subsets prove independent geographic or downstream generalization.

`receipt.json` includes imports/versions, source/checkpoint hashes, exact loss configuration, dev objective before/after under the same masks in eval mode, nonzero encoder gradient tensor counts, sampled encoder parameter deltas including attention parameters when available, elapsed time and peak memory. Base and InfoNCE positive step/batch counts are separate: homogeneous WorldCover can leave base loss zero, so an encoder update may be driven only by InfoNCE. That is not reported as successful coverage of both objectives. `train_log.jsonl` survives partial failures. `--save-weights` writes a loader-compatible full model snapshot plus unchanged config, but does not save optimizer state or claim full trainer resumability. It then reloads via the official loader, compares an actual fixed-input S2 encoder token tensor and every paired development loss component (`atol=1e-6`, `rtol=1e-5`), and fails on mismatch.

Pass requires finite loss/gradients, nonzero encoder gradients, no target gradients, and observed encoder weight updates. A negative dev-loss delta is **not** a pass criterion. Fail on invalid data/config, missing dependencies, nonfinite values, no encoder update, or wall budget. GPU idleness/permission is root's pre-launch gate, not implemented as a scheduler inside this script.

After the first successful run, retain the measured step times/memory and inspect the rejected/accepted data distribution before deciding a larger real development subset. These numbers must not be extrapolated to the full long-sequence/multimodal configuration without profiling it.
