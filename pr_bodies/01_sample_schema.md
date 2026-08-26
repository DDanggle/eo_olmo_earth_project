## Problem

The documented quick-start for the sample project fails out of the box:

```
python -m olmoearth_projects.main olmoearth_run prepare_labeled_windows \
    --project_path olmoearth_run_data/sample --scratch_path /tmp/scratch
```

```
ValueError: Error reading annotation features file '.../annotation_features.geojson'
  Field required [type=missing] features.0.properties.oe_annotations_task_id
```

## Cause

The sample project ships two paired GeoJSON files, but only one was migrated to the
`oe_*` property schema:

- `annotation_task_features.geojson` — already uses `oe_annotations_task_id`, `oe_start_time`, `oe_end_time` ✅
- `annotation_features.geojson` — still uses legacy `es_*` keys and a scalar `es_label` ❌

`olmoearth-runner` (>= 0.1.12) validates annotation features against
`AnnotationFeatureProperties`, which requires `oe_annotations_task_id` and a dict-valued
`oe_labels`.

## Fix

Migrate the remaining file: `es_* → oe_*`, and `es_label: <int>` → `oe_labels: {category: <int>}`
(the dict form the model expects, keyed to match `label_property: "category"` in the
sample's `olmoearth_run.yaml`).

## Verification

After the change, `prepare_labeled_windows` completes and writes 6 labeled windows:

```
/tmp/scratch/dataset/windows/post_random_split/
  task_164679b9-..._annotation_0_point_-118.15229910783472_33.7362279043863/
  ... (6 total, EPSG:32611, 10 m, split=train/val)
```

Verified end-to-end on Linux (Python 3.11, olmoearth-runner 0.1.14). On macOS,
the updated file passes schema/JSON validation, but the documented CLI still hangs
later in multiprocessing; that is a separate pre-existing issue and not evidence for
or against this data-only fix.

The updated FeatureCollection also parses as JSON, contains 6 features, has no
legacy `es_*` property keys, and each feature has a dict-valued `oe_labels.category`.

## Scope

This is intentionally a data-only schema repair. `annotation_task_features.geojson`
already uses the required `oe_*` schema, so it is unchanged.
