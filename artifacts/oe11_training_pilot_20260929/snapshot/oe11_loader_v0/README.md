# OE11 read-only calibration loader adapter

This new module leaves the frozen OE10/P2 loader and all dataset files unchanged.
The base loader SHA must equal
`536de03b77bcd5bfedf439b5456d9e0fe60a3789c5f9c1a53c5511ded05d07f6`.

The base loader's `load`, NPZ validation, acquisition copies, support-mask hash
checks, target joins and label checks run unchanged. No global guard is patched.
The new catalog has two spelling differences handled only in the temporary
validation copy:

1. Development query `training_partition=calibration` maps to `dev_query` for
   the inherited validator. The original role must remain `development`, with
   `supervised_training_allowed=False`. Train/source-bank flags remain strict.
2. `support_masks/{train_pool,source_bank}/{object_key}.npz` maps to the old flat
   spelling for validation only. Before that alias, the adapter independently
   enforces the exact partition, object basename, and path containment. Actual
   mask reads use the original partitioned catalog path and original hash.

Both reference roots in the new contract must exactly match the supplied roots.
Currently `support_mask_reference_root` equals the catalog output root; differing
external mask roots are rejected rather than adding a path-routing bypass.

Allowed parents are exactly t31tfj/t31tfm/t32ulu. All three may have train_pool,
source_bank and calibration rows. t30uxv is rejected. No exact counts are baked
into the loader (the synthetic fixture is smaller than the actual 192/48/48).

`train_support_parent_policy=cross_parent` forbids same-parent **train** supports.
Development uses the same global source bank under both pooled/cross_parent
catalog variants. This is calibration on source parents, not a held-out-region
evaluation. Bank/calibration target access cannot authorize encoder training.

## Use

```python
from loader_adapter import make_loader_class
Loader = make_loader_class('/path/to/pinned/episode_loader.py')
loader = Loader(prepared_root, catalog_root, 'train', support_mode='pooled')
sample = loader.load(episode_id)  # query2 / support8; no query gold
target = loader.load_target(episode_id, purpose='training', training=True)
```

For development use `split='development'` and an explicit
`load_target(episode_id, purpose='evaluation')`. The existing training_target and
evaluation_target accessors remain available. Scoring-file SHA is checked lazily
on every explicit target call, followed by the original per-record join/label
checks. Construction and `load` do not open scoring or query-label files.

Only `sample['model_input']` enters the model; audit/metadata/target stay separate.
`load(..., acquired_positions=[0,2,5,7])` and `range(8)` work at the loader level.
The current model wrapper still hardcodes **two query dates**; this loader does
not implement or claim a four/eight-date model comparison. Initial indices2/5
remain mandatory and returned arrays are independent copies of acquired data.

## Bounded real-packet smoke CLI

```sh
env -u PYTHONPATH python smoke_packets.py \
  --prepared-root /home/work/data/olmoearth/oe11_source_expand_v0/prepared_v0 \
  --base-loader /path/to/pinned/episode_loader.py \
  --catalog pooled=/path/to/pooled/catalog \
  --catalog cross_parent=/path/to/cross_parent/catalog \
  --output /path/to/new/loader_smoke.json
```

Default: one case per mode/split/K cell, K1/K8, three acquisition sizes, **24 total
loads when both modes have every cell**. `--cases-per-cell 3` prefers three
distinct query parents and caps the smoke at72 loads. Missing K cells are reported
as shortages, never padded/replaced. `--check-targets` separately enables explicit
target checks; default is no query gold/scoring access. This CLI does not certify
cloud quality, geographic parcel independence, or model capability. It has not
yet been run on the actual server catalog by this implementation task.

## Local CPU verification

`test_loader_adapter.py` uses the separate OE11 catalog builder's synthetic
fixture, adding only the missing full packet fields before catalog construction.
Set `OE11_CATALOG_CODE`, `OE8_BASE_BUILDER`, `OE11_BASE_LOADER` when relocated.

```sh
python -m unittest discover -s /private/tmp/oe11_loader_adapter -p test_loader_adapter.py -v
```

Eight tests passed on local NumPy CPU: original bytes/memory partitions intact;
explicit gold access; inference denied any label/scoring file opening; K1/K8 with
2/4/8 query dates; inherited gold/role/path guards; cross-parent train and shared
development bank; dependency/manifest/scoring SHA rejection; reference-root and
held-out-parent rejection; all24 bounded smoke load combinations.

No server/GPU/model/actual-weight execution occurred. Actual prepared metadata
was read only to confirm field names/roles, and no final held-out data were opened.
