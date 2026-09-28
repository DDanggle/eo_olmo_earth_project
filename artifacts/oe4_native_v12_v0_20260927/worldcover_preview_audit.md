# WorldCover preview audit — 2026-09-27

**Finding: the large pale-purple area is valid Cropland, not an unrecognized code.**

The first preview, `sample_1229.h5`, has 1,010 pixels with raw WorldCover code40, 13 with code30, and one with code10. Thus 98.6328125% is code40, whose official palette color is `#f096ff` (Cropland). The fallback for an unrecognized code is a different color, `#ff00ff`. All four displayed crops have zero unrecognized pixels according to the separately exported raw-value counts; no zero-index encoding is present in these examples.

Evidence: `/private/tmp/oe4_execute_20260927/preview_v0/preview_v0/input_provenance.json`, corresponding gallery inspected visually; source `olmoearth_pretrain/data/visualize.py:22` defines the raw-code palette. The original v0 PNG/JSON were not changed.

Official computed WorldCover normalization uses mean34.04615796218091, std24.232421963981864 and range mean±2std. Recalculation in float32 gives code10→0.2519216537, code40→0.5614243150 and code50→0.6645919085, exactly matching the selected provenance's minima/maxima. `data/dataset.py:642` tries computed normalization first, and `dataset/convert_to_h5py.py:393` writes the loaded WorldCover values without a class-index remap. Runtime normalization is consistent with the official path for these four inspected crops. This is not a complete corpus validation.

The needed preview correction is explanatory: add visible color swatches, an unknown-pixel count on each WorldCover panel, and the dominant raw class plus percentage. The revised `export_preview.py` implements that without changing data, normalization, RGB display or official class colors. Neither the runtime script nor queued controller was edited.

Two material research limitations remain:

1. `sample_1267` and `sample_1266` share exactly the same saved parent-center coordinate `[-39.63349151611328,-69.50154876708984]` and observation dates. Official conversion copies the parent's center into each subtile. The two dev files therefore cannot count as two independent geographic regions; exact-coordinate grouping is a concrete next-stage repair, though different coordinates alone do not guarantee spatial independence.
2. WorldCover is a static target without an acquisition timestamp in these H5 records. In particular, the first S2 example is from2016. Native static-map supervision may follow the original recipe, but this preview cannot establish contemporaneous ground truth or a land-cover change between those observations. Cloud labels or change semantics were not inferred from RGB.

The near-homogeneous first crop also makes it important to inspect the native patch-loss activity receipt: identical targets can eliminate contrastive negatives. Pixel class validity does not guarantee useful patch discrimination for every crop.
