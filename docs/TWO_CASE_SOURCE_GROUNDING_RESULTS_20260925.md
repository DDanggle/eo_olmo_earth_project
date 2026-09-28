# Two fixed EO source cases: completed grounding results

> 후속: [원저자 날짜 metadata 복구](KURO_DATE_RECOVERY_FEASIBILITY_20260925.md)에서 두 타일의 정확한 grid 조인과 공개 슬롯 매핑을 확인했다. 원 센서 픽셀 대응·현재 archive 생성 revision은 미검증이며 기존 export의 null 날짜는 보존한다. 아래는 그 이전 확인 단계의 기록이다.

2026-09-25. The original export and local rendered copy are complete. This note
updates the execution status of the earlier
[provenance proposal](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/docs/E3_E4_TWO_CASE_PROVENANCE_20260925.md).
It reports an inspection of two already-exposed development cases, fixed before
E5 outcomes. It adds no model inference, annotation, performance estimate, or
change to E3/E4/E5 evaluation membership.

The completed sources are the
[raw export index](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/fixed_case_grounding_v0_20260925/case_index.json)
and [rendered copy index](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/fixed_case_grounding_rendered_v0_20260925/case_index.json).
The raw export contains 16 files totaling **3,684,832 bytes**: ten original
embedded GeoTIFFs, two crop-array NPZs, two case manifests, selection, and index.
The rendered copy contains 18 files, adding two 1840×1440 PNGs. Export completed
at 06:26:33 UTC; rendering provenance is dated 06:28:43 UTC.

The exporter records successful original-TIFF versus historical-NPY equality,
cropped/remapped C1 mask/valid hash and count checks, and C0 embedding-cache hash
checks. Independent local review recalculated NPZ counts, verified every raw
file against the rendered copy's parent pins, verified all TIFF/NPZ copies and
manifest/PNG hashes, and checked that every displayed reference-mask pixel
equals the source palette at 2× nearest-neighbor scale. No discrepancy was found.
The server's original archive and historical NPYs were not re-read in this local
review. The full 7,529,588,233-byte archive is not hashed; the ten selected TIFF
byte slices are individually hashed.

Both cases belong to test event **562**, AOI **13**, event date **2022-01-29**.
They are different geographic tiles. Counts below were independently calculated
from each exported 192×192 crop, rows/columns `[16:208]` of its 224×224 source.

| Crop property | ks_06770 | ks_05265 |
|---|---:|---:|
| Source centroid, longitude / latitude | 139.612164 / −17.738688 | 139.491430 / −17.719522 |
| Separate source-valid pixels | 36,864 / 36,864 (100%) | 36,864 / 36,864 (100%) |
| Known-label pixels | 20,871 (56.6162%) | 36,864 (100%) |
| Unknown-label pixels | 15,993 (43.3838%) | 0 |
| Raw class0: no-water | 9,751 | 36,864 |
| Raw class1: permanent water | 2,668 | 0 |
| Raw class2: flood | 8,452 | 0 |
| Raw class3: label unknown | 15,993 | 0 |
| Flood / entire crop | 22.9275% | 0% |
| Flood / known-label crop | 40.4964% | 0% |
| C1/E5 symmetric-quality primary eligibility | Excluded | Included |

Here **known-label = source-valid==1 AND raw mask!=3**. Raw class0 remains
semantic no-water even though the GeoTIFF nodata field is also zero; treating
that metadata value as categorical missingness would erase the valid no-water
reference. Permanent water is excluded from flood. Both cases have finite,
positive SAR crop values in all six band/slot panels, but that does not turn
unknown reference labels into known ones. The source-row whole-tile flood
percentages are 23.3418% and 0%; the first must not be substituted for either
crop denominator above.

**Actual acquisition dates remain unverified after source recovery.** In both
cases, all three SAR asset rows repeat `stac:time_start = stac:time_end =
1643414400.0` (2022-01-29 00:00:00 UTC) and `flood_date = 2022-01-29 07:00:00`.
The latter string supplies no explicit timezone. TIFF dataset tags contain only
`AREA_OR_POINT: Area`, and band tags are empty. These metadata do not establish
distinct acquisition dates or intervals; they also do not establish that the
three images were acquired simultaneously. All three `acquisition_dates`
remain null. The QA dates Jan 5 / Jan 17 / Jan 29 are event-derived
approximations and must not be presented as recovered sensor dates.

The previews show all three VV/VH slots, a **dataset post reference mask**, and
known-label support. Display uses a fixed clipped −25 to 0 dB range with legends
outside image pixels. Neither preview is a model segmentation. Their value is
traceable scene evidence: the ks_06770 crop contains 8,452 reference flood pixels
within incomplete label coverage, while ks_05265 has a fully labelled no-water
reference. Keeping the low-coverage case makes this limitation inspectable.
The two examples are not a representative C1/E5 primary matched pair and provide
no estimate of generalization, temporal reasoning, or memory improvement.

The post mask does not independently establish pre-event absence, change onset,
or persistence. These artifacts contain no land-cover overlay, crop-loss
measurement, or identified causal impact. Their EPSG:3857 grid has 10-unit pixel
spacing; pixel counts are reported here without promoting projected pixel area
to ground hectares. Current byte equality supports data traceability but does
not prove historical encoder lineage or exclude all spatial/pretraining leakage.
Existing scientific conclusions and primary eligibility remain unchanged.

Reproducibility note: the raw index SHA256 is
`903dffdc1005549d433ccc3a0b823ca1de2f67191b223216f50673b85d526e04`;
the rendered index SHA256 is
`6ceef9b6a26e755498781c6e32b4316695d6f6459a308bec767414c9d3e8173f`.
The executed exporter SHA256 matches current source
`f668e506ca949bcb0e9e62a659baae789d2eedb9345682f8b0935513a1e417a6`.
The recorded rendering helper SHA256 is
`fd3d484abe8a49e1d75551b9c33e0a5cfa1d2cd21541a5c570ae35eda60d5c60`;
current helper SHA256 is
`88d276fe09e58b1737f0644582ae752870055ba4ab45df609803908bd4657527`.
Their only difference is the current helper's additional guard requiring index
and manifest quality-eligibility fields to agree. Those actual fields were
independently checked and agree for both cases; rendering code is unchanged.
A [post-execution reconstruction](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/two_case_render_execution_reconstruction_20260925/render_two_eo_cases_20260925.py)
was recovered by removing precisely that guard, and its SHA256 exactly matches
the execution record. This is **a reconstruction made after rendering, not a
snapshot captured before execution**. The
[unified diff](/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/artifacts/two_case_render_execution_reconstruction_20260925/executed_reconstruction_to_current_guarded.diff)
records the single-line change. The latest guarded helper remains unchanged.

The exporter and current helper have 21 passing synthetic tests covering byte
bounds/path containment, raw0 and unknown semantics, fixed-case selection,
immutable copying, hash/metadata rejection, and unobscured mask rendering.
This local review made no server request or research-code/artifact modification.
