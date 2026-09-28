# Two fixed E3/E4 image cases: provenance and bounded export proposal

> Update 2026-09-25: original export, local rendering, and v7 source-image connection are complete. See [completed results](TWO_CASE_SOURCE_GROUNDING_RESULTS_20260925.md). The proposal below preserves the pre-execution record.

2026-09-25. Read-only investigation of existing research files. No server action,
download, raw-image export, model execution, new annotation or E5 change occurred.
The two IDs were supplied before future E5 outcomes. They remain illustrative
already-exposed development cases, not a random sample or a new evaluation.

## What is already established locally

Both IDs occur in the frozen E3 items and the C0 reconstruction as **test** data,
event **562**, AOI **13**, event date **2022-01-29**.

| Property | ks_06770 | ks_05265 |
|---|---|---|
| Existing E3 source items | q1_pos and q1_neg | q1_hard |
| Source centroid (longitude, latitude) | 139.612164, -17.738688 | 139.491430, -17.719522 |
| Whole 224×224 tile reference pflood | 23.341836734693878% | 0% |
| 192×192 model crop: labelled pixels | 20,871 / 36,864 (56.6162109375%) | 36,864 / 36,864 (100%) |
| Model crop: unknown label pixels | 15,993 (43.3837890625%) | 0 |
| Model crop: source flood pixels | 8,452 | 0 |
| Flood share of labelled crop | 40.496382540367015% | 0% |
| Flood share of whole crop | 22.92751736111111% | 0% |
| C1/E5 symmetric-quality primary eligibility | **Excluded** | Included |

**These are not a pair of representative C1/E5 primary cases.** In ks_06770 the
separate `valid_u8` array equals one everywhere, while 15,993 pixels have remapped
mask0 (raw mask3, unknown). Displaying only validity, or calling its .405 label
fraction a fraction of the entire crop, hides a substantial limitation. Keep this
case to inspect that limitation rather than silently replacing it.

For source-positive/hard comparisons, both date strings are 2022-01-17 and
2022-01-29 with slots pre_2/post. The same-tile q1_neg uses 2022-01-05 and
2022-01-17 with pre_1/pre_2. **All these dates are known event-derived
approximations, not recovered acquisition times.** The reference mask describes
the post slot; it does not independently certify no flooding in either pre slot.

## Exact local evidence

Research root:
`/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project`.

- `artifacts/e3_pair_dependence_v1_20260925/items.jsonl`: selected items, source
  dates/slots, pair keys 155, 161 and 162; SHA256
  `f3ae0abc245b3e800f1a39b4ca5a250b36f4cf503fe364f3003a556c7f907502`.
- `artifacts/streaming_review_20260909/kurosiwo_s1_cache/meta.jsonl`: event, AOI,
  split, source centroid and whole-tile mask fractions; SHA256
  `23c0df0e058df6664bcfaf007fb7da48a370b247153896fb15261bafe100612b`.
- `artifacts/c1_flood_label_quality_v0_20260925/quality.jsonl`: crop counts,
  mask/valid hashes and known-label coverage; SHA256
  `8f654dfbaabc41e3b27de9a04dfe0768c1f0a748439c70acb61ddae08a17a78f`.
- `artifacts/c0_linear_view_probe_v1_20260925/manifest.json`: cache hashes and
  split provenance; SHA256
  `5055bd2d73f1e9f162a43899a2d0f5c46cccef53137cb3a5ae31e7a2675bcb92`.
- `code/kurosiwo_export_npy.py`: asset indices 0/1/2 become VV/VH × pre_1/pre_2/post;
  index4 is the raw mask; index5==1 is validity. The optional DEM index3 is not
  needed for this bounded image inspection.
- `code/extract_kurosiwo_s1_cache.py`: model crop rows/columns `[16:208]`, linear
  SAR converted to dB with zeros filled at -30 dB for the encoder, and mask remap
  `{3:0,0:1,1:2,2:3}`. The new preview preserves raw arrays instead of hiding zeros.

## Minimal existing-data export path

The existing server container is
`/home/work/data/olmoearth/geobench2/kurosiwo/kurosiwo/geobench_kuro_siwo.tortilla`.
`tacoreader.load` locates the row by numeric tortilla:id, then `table.read(row)`
locates the nested assets. Current known asset names are pre_event_1,
pre_event_2, post_event, dem, mask and invalid_data.

The draft `/private/tmp/export_two_eo_cases_20260925.py` exports **only the two
fixed IDs**, using the established `.venv-geobench` CPU dependency environment.
It does not read model predictions or any E5 artifacts, and has no network
fallback. An existing destination is rejected.

For each case it:

1. Copies the **original embedded TIFF bytes** for indices 0, 1, 2, 4 and 5, using
   the local bounded `/vsisubfile/offset_length,container` locator. It hashes the
   exact selected bytes, avoiding a full 7.5 GB container read. It verifies the
   container stat before and after export.
2. Retains raw asset rows, TIFF tags/band tags, CRS, affine and source centroid.
   It computes both whole-tile and model-crop WGS84 footprints from that affine.
3. Checks original TIFF pixel arrays against existing `kurosiwo_npy` raw/mask/
   valid arrays, then checks cropped/remapped masks against the C1 hashes/counts.
   It also verifies each embedding cache against its C0 hash. A discrepancy stops
   the export and preserves partial evidence; no case replacement is allowed.
4. Saves crop arrays with full four-class raw mask and separate source-valid and
   labelled-valid arrays. Optional `--render` produces six SAR panels (VV and VH
   for all three slots), the post reference mask and labelled support. The fixed
   [-25,0] dB display has no per-image contrast fitting; nonpositive/nonfinite raw
   values are visibly marked. Raw values remain available in the NPZ/TIFFs.
5. Keeps acquisition dates null and includes all source timestamp fields and TIFF
   tags for explicit audit. It never substitutes event dates or imputed dates.

The CLI requires the frozen E3 items, C1 quality JSONL and C0 manifest paths, and
checks their hashes above. It also requires an explicit new `--out` path. Main
must install/review the draft before any invocation; **no export command has
been executed as part of this task.**

## Actual acquisition dates: still unresolved for these two cases

The old generic NPY metadata export only retained event dates. For the previously
recovered train pilot ks_00276, all three nested SAR asset rows repeated the event
timestamp, and original TIFF tags contained only AREA_OR_POINT. That proves the
pilot did not provide distinct acquisition dates; it does **not** prove the two
current cases have identical metadata. Their own original TIFF/sample rows still
need inspection. The exporter records candidate fields without automatically
promoting any to true acquisition dates.

## Leakage and interpretation boundaries

The current C0 items contain event562 only in test (six pos, six pre/pre neg,
two hard negatives); neither selected ID occurs in training. Their complete
embedding SHA256 values occur only once among the 3,756 frozen caches, with no
matching training-cache hash:

- ks_06770: `e64779c9c22cc35557deb6651a64e73288f59406fced38dc7b41bf55b27baa6b`
- ks_05265: `8864c48feddc69b6f2e82c60e6bb5a9b9e7672cfd916097d299c4b42154a56d8`

These checks rule out those particular ID/event/exact-cache duplications in the
current reconstruction, not all spatial overlap, shared-scene information,
pretraining exposure or historical provenance problems. Event562 was already
exposed in prior project evaluations. Inspecting it is development diagnosis,
not newly unseen validation.

The centroid is not a flood boundary. Different tiles from the same event can
differ in geography, permanent water, observation quality and unknown support.
SAR brightness is not itself a flood classification; the colored mask is a
dataset reference, not an independent VLM localization or fresh human annotation.
There is no land-cover overlay for these two cases here, no measured crop loss,
and no identified causal impact. Describe a source flood-label observation and
its uncertainty, not damage, event onset, or causal effects.

## Draft validation performed locally

The draft compiles. Synthetic CPU checks confirmed that raw class0 remains valid
no-water, raw class3 is unknown, valid0 is excluded, permanent water is not flood,
and local embedded-byte identity/bounds/network-locator guards behave as intended.
No real image was exported, rendered or interpreted. The geospatial export and
preview still require actual-case execution and visual QA after main review.
