The sample quick-start fails right away:

```
python -m olmoearth_projects.main olmoearth_run prepare_labeled_windows \
    --project_path olmoearth_run_data/sample --scratch_path /tmp/scratch
# ValueError: ... Field required [type=missing] oe_annotations_task_id
```

`annotation_task_features.geojson` was already moved to the `oe_*` keys, but
`annotation_features.geojson` still has the old `es_*` keys and a scalar
`es_label`. This PR renames the keys and changes `es_label: N` to
`oe_labels: {"category": N}`, which is what `AnnotationFeatureProperties`
in olmoearth-runner expects (checked in 0.1.12 and 0.1.14).

Data-only change, 6 features. After the fix `prepare_labeled_windows`
runs and writes 6 labeled windows (Linux, py3.11, runner 0.1.14).
