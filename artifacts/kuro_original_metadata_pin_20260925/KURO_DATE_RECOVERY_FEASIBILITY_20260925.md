# KuroSiwo 두 타일 원본 날짜 메타데이터 복구

2026-09-25. 결과는 **원저자 metadata의 날짜·grid ID 복구와 공개 GEO-Bench-2 슬롯 매핑 확인 성공, 현재 tortilla를 생성한 정확한 revision·원 raster 대응은 미확인**이다. `ks_06770`, `ks_05265` 모두 공개 원저자 grid metadata와 전체 타일 geometry 및 사건·비율 필드가 정확히 일치하는 유일한 항목을 찾았다. 서버, 모델, E5 결과에 접근하지 않았고 기존 실험·날짜·캐시·앱을 바꾸지 않았다.

## 확인된 날짜와 출처

두 grid가 가진 원본 `sources`는 같다. 아래는 실제 원저자 metadata 문자열이며 사건일에서 추정한 날짜가 아니다. 다만 이를 현재 GEO-Bench 슬롯의 취득일로 자동 대입하지 않는다.

| 원본 source key | source_date | s1_ids의 원본 UUID | master / crank |
|---|---|---|---|
| SL1 | 2021-08-19 | `502f8040-709d-5c6e-b9fb-947ba4b47925` | false / 1 |
| SL2 | 2021-08-07 | `4bd406ac-3ce2-5a04-94e3-fca64f5878d1` | false / 2 |
| MS1 | 2022-02-03 | `2fa3885a-52ca-5fae-948f-60cbde7102c5` | true / 1 |

`source_date`는 일 단위이며 정확한 UTC 취득시각이 아니다. `s1_ids`는 UUID여서 `S1A_IW_GRDH_...` 등의 원 제품명이나 궤도 식별자를 이번 조사에서 복구한 것은 아니다. 각 source의 coverage는 100.0이다. 원 `datasets`에도 IVV/IVH 파일명과 같은 날짜·UUID가 반복된다. 예: `SL1_IVV_562_13_20210819`, `SL2_IVV_562_13_20210807`, `MS1_IVV_562_13_20220203`.

두 타일은 사건 `actid=562`, AOI `13`, `flood_date=2022-01-29 07:00:00`이다. 원저자 설정은 이 사건/AOI를 Australia, Burketown으로 명시한다. [고정 revision 설정](https://github.com/Orion-AI-Lab/KuroSiwo/blob/4347ed173c4e48f5a9d578bb5fe8453706b08e5e/catalogue/catalogue.yaml#L345)

## 어떤 조인으로 확인했는가

로컬 raw export의 `source_manifest.json`에서 실제 원 TIFF의 affine/shape/CRS와 source row를 사용했다. 중심점 반올림, 최근접 타일, 이미지 육안 판독으로 맞추지 않았다.

| 현재 ID | 원 grid UUID | EPSG:3857 전체 bounds (west, south, east, north) | pwater / pflood |
|---|---|---|---|
| ks_06770 | `e2f98639-b3ca-5c43-8d3b-79c11b55bc5c` | 15540435, -2008105, 15542675, -2005865 | 6.311782525510204 / 23.341836734693878 |
| ks_05265 | `d762ee26-ef58-5723-b9f7-d9d18f419bd8` | 15526995, -2005865, 15529235, -2003625 | 0.0 / 0.0 |

검증 조건은 다음과 같다.

- 내보낸 5개 asset 모두 EPSG:3857, 224×224, 10 m, 회전 없는 같은 affine인지 확인했다. 여기서는 192×192 모델 crop 대신 전체 원 타일을 썼다.
- 원본 WKT가 닫힌 5점 사각형이고 축에 평행하며 교차하지 않는지 검사했다. 네 모서리 집합이 TIFF affine에서 계산한 네 모서리와 **허용오차 없이 정확히 같아야** 한다. 원 설정 CRS도 3857이다. [설정 14행](https://github.com/Orion-AI-Lab/KuroSiwo/blob/4347ed173c4e48f5a9d578bb5fe8453706b08e5e/catalogue/catalogue.yaml#L14)
- `actid`, `aoiid`, `flood_date`, `pcovered`, `pwater`, `pflood`가 source row와 정확히 같은지 별도로 확인했다. 두 타일의 pcovered는 100.0이다.
- 원 dictionary key가 grid UUID의 hyphen 제거형이고 path가 `562/13/<gridhex>`인지 확인했다.
- 31,707개 metadata 중 act562/AOI13 후보는 67개다. ks_06770은 정확한 후보 1개, ks_05265는 0개였다.
- 67,490개 test metadata 중 같은 사건/AOI 후보는 77개다. 두 타일 모두 정확한 후보가 각 1개다. ks_06770의 두 파일 내 metadata record도 동일하다.

검토 가능한 전체 원 record, 필드별 비교, source manifest/asset SHA는 [two_case_metadata_join_audit.json](/private/tmp/kuro_date_metadata_20260925/two_case_metadata_join_audit.json)에 남겼다. 최종 `two_case_metadata_join_audit.json`은 표준 JSON이다. 탐색 단계의 `*.exact_matches.json`은 원 NaN 표현을 포함한 진단 기록으로 보존했으므로 표준 JSON 교환 파일로 사용하지 않는다. 원 DEM `nodata=NaN`은 JSON 안에서 `{"upstream_nonfinite_float":"nan"}`로 명시해 원 값의 존재를 보존했고, 날짜 조인에는 쓰지 않았다.

## 가져온 파일과 고정 계보

원저자 GitHub revision은 `4347ed173c4e48f5a9d578bb5fe8453706b08e5e`이다. 다음 두 파일은 raster가 없는 압축 grid metadata이며 총 27,647,906 B를 받았다. 영상 TAR, 모델, Sentinel 영상은 다운로드하지 않았다.

| metadata | 압축 크기 | SHA-256 |
|---|---:|---|
| [KuroV2_grid_dict.gz](https://github.com/Orion-AI-Lab/KuroSiwo/blob/4347ed173c4e48f5a9d578bb5fe8453706b08e5e/pickle/KuroV2_grid_dict.gz) | 8,980,125 B | `6235bf33fdf188ae23386134d48d70f57de90c2994e8fd9c33d000f9bd5420cc` |
| [KuroV2_grid_dict_test_0_100.gz](https://github.com/Orion-AI-Lab/KuroSiwo/blob/4347ed173c4e48f5a9d578bb5fe8453706b08e5e/pickle/KuroV2_grid_dict_test_0_100.gz) | 18,667,781 B | `dfc6e9a752d05aaa0a38c3fd6c18ce12e090f95635e5e4709025cb8512f020ad` |

다운로드 URL·크기·SHA는 [third_fetch_report.json](/private/tmp/kuro_date_metadata_20260925/third_fetch_report.json), [fourth_fetch_report.json](/private/tmp/kuro_date_metadata_20260925/fourth_fetch_report.json)에 있다. 파일별 Git blob SHA-1을 다시 계산해 고정 revision의 tree와 일치함도 확인했다: [github_blob_integrity.json](/private/tmp/kuro_date_metadata_20260925/github_blob_integrity.json). 원본 압축 bytes는 같은 폴더에 보존했다.

파서는 [inspect_grid_metadata.py](/private/tmp/kuro_date_metadata_20260925/inspect_grid_metadata.py)다. `pickle.load`/`loads`, 객체 import/생성/함수 호출을 쓰지 않고 `pickletools.genops`로 기본 문자열·숫자·bool·list·dict 및 memo 연산만 해석한다. `GLOBAL`, `STACK_GLOBAL`, `REDUCE`, `BUILD` 등 whitelist 외 opcode는 거부한다. gzip 확장 입력은 200,000,000 B 상한이다. **범용 악성 pickle sandbox가 아니라 이 SHA로 고정된 두 metadata의 제한 파서**이며, 다른 파일이나 구조를 자동 수용하는 용도가 아니다. 독립 synthetic 5개 테스트가 통과했다([test_kuro_metadata_parser_20260925.py](/private/tmp/test_kuro_metadata_parser_20260925.py), SHA-256 `a9c05efc49d4635dd8748a0b2f89a44874afebe4f8b1b415f788c9828a677e14`). opcode whitelist 밖 실행은 차단하지만 FRAME/PROTO의 전체 validation, STOP 뒤 bytes, 객체/memo 확장량까지 제한하지는 않으며 순환 참조·NaN/Inf도 허용한다. 현재 고정 SHA와 별도 구조 검사가 전제다. geometry·필드 일치 재현은 [audit_two_grid_joins.py](/private/tmp/kuro_date_metadata_20260925/audit_two_grid_joins.py)로 실행했다.

## 남은 불확실성과 해석 범위

현재 raw export에서 pre_event_1/pre_event_2/post_event의 STAC 시간은 모두 사건일이고, TIFF tag에 별도 source date는 없다. 기존 acquisition_dates가 null인 판단은 당시 근거에 맞으며 이번에도 기존 파일을 수정하지 않았다.

원저자 [Dataset.py 948–966행](https://github.com/Orion-AI-Lab/KuroSiwo/blob/4347ed173c4e48f5a9d578bb5fe8453706b08e5e/dataset/Dataset.py#L948)은 SL1→pre_event_1, SL2→pre_event_2라고 이름 붙인다. 그러나 root가 병행 확보한 **공식 GEO-Bench-2 generator는 이 이름을 뒤집어 시간순으로 매핑**한다. [generator 37–40행 및 517–520행](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/d7ae6aafc18e1da1bd6efb130f0a306125ee0fef/geobench_v2/generate_benchmark/kuro_siwo.py#L37)은 SL2→pre_event_1, SL1→pre_event_2, MS1→post_event를 규정한다. [548–575행](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/d7ae6aafc18e1da1bd6efb130f0a306125ee0fef/geobench_v2/generate_benchmark/kuro_siwo.py#L548)은 tortilla 순서가 pre1/pre2/post이고 모든 modality STAC time_start/end를 flood_date로 기록하는 이유도 설명한다. [loader 124–126행과 160–165행](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/d7ae6aafc18e1da1bd6efb130f0a306125ee0fef/geobench_v2/datasets/kuro_siwo.py#L124)은 slot 0/1/2에서 해당 이름을 그대로 읽는다.

따라서 공개 코드 계약을 적용한 대응은 다음과 같이 시간순이다. 원저자 loader의 이름만 보고 pre 방향이 뒤집혔다고 단정하는 것은 잘못이다.

| 현재 이름 | 공개 GEO-Bench 매핑 | 원 metadata 날짜 |
|---|---|---|
| pre_event_1 / pre_1 | SL2 | 2021-08-07 |
| pre_event_2 / pre_2 | SL1 | 2021-08-19 |
| post_event / post | MS1 | 2022-02-03 |

확인한 GEO-Bench-2 revision은 `d7ae6aafc18e1da1bd6efb130f0a306125ee0fef`이며 두 파일의 Git blob을 해당 revision tree와 다시 대조했다. generator SHA-256은 `57a89a697219539d1949de6a5999101c90b8ed65cbdcd094f59b0e55720cbcb1`, loader SHA-256은 `18583f750490cb37d8228d00170c6d3b0bf103b8ab815e7440b3a556488f3521`이다. URL·파일 hash·line·조건부 날짜 계약은 [published_geobench_slot_contract.json](/private/tmp/kuro_date_metadata_20260925/published_geobench_slot_contract.json)에 별도로 남겼다.

이 확인은 **공개 코드 계약의 mapping**을 해결한다. 이 exact revision이 이미 서버에 있는 tortilla를 만든 버전이라는 실행 계보, 원저자 영상과 현재 영상의 값 대응까지 확인한 것은 아니다. 그래서 기존 export의 acquisition_dates를 수정하지 않았고 최초 two_case_metadata_join_audit의 `geobench_slot_mapping_status=unverified`는 최초 geometry 감사 시점의 기록으로 보존했다. 후속 공개 코드 계약은 위 별도 파일을 함께 읽어야 한다.

원 metadata상 SL1→MS1 간격은 168일, SL2→MS1은 180일이다. 이 값은 사건 주변 −12/−24일의 합성 시간과 크게 다르다. 연구상 판단으로는 홍수 전후 효과와 장기간·계절 변화가 함께 있을 가능성을 별도로 검토해야 한다. 이것만으로 모델이 계절 단서를 썼다고 증명한 것은 아니다. post mask는 pre 영상의 무침수 정답이나 변화 발생시각을 검증하지 않는다.

S1 제품 UUID→제품명·정확 UTC 시각, 원본 raster와 현재 byte/value 대응, 현재 tortilla 생성의 정확한 revision은 아직 미검증이다. 공개 GEO-Bench 슬롯 순서는 코드에서 확인했다. E5는 합성 timestamp를 포함한 동결 입력에 대한 실험으로 유지하고, 이번 metadata 발견을 사후 입력 교체의 근거로 사용하지 않는다.

## 시도했지만 직접 날짜 복구에는 쓰지 못한 경로

- [catalogue.py](https://github.com/Orion-AI-Lab/KuroSiwo/blob/4347ed173c4e48f5a9d578bb5fe8453706b08e5e/catalogue/catalogue.py#L221)는 `source_date`, `s1_ids`, `master`, `crank` schema를 확인시켰다. 공개 GitHub tree에는 catalogue.gpkg 자체가 없다.
- [원저자 다운로드 스크립트](https://github.com/Orion-AI-Lab/KuroSiwo/blob/4347ed173c4e48f5a9d578bb5fe8453706b08e5e/download_kuro_siwo.sh#L13)에 catalogue.gpkg Dropbox URL이 있다. 공식 GitHub/HF만 조사한다는 범위 때문에 외부 Dropbox를 열지 않았다. 따라서 그 파일 크기·접근성은 확인하지 않았다. 이번에는 위 두 GitHub metadata로 대체할 수 있었다.
- [GeoTIFF 배포](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-GeoTIFFs/tree/2b149732bb088e1dda01c5bbf1c1cc354ff05689)의 재귀 tree에는 개별 info.json/GPKG/GeoJSON이 없고 GRD TAR 35개가 총 699,148,625,920 B다. TAR를 받거나 순회하지 않았다. [Webdataset](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset/tree/main)도 개별 metadata가 분리돼 있지 않았다.
- 공식 GitHub의 `json/slc_grid_pwater_0.0001.json` 1,587,921 B에는 actid/path/aoiid/clz만 있었다. geometry·날짜 조인에 사용하지 않았다.
- [별도 공식 annotation repo](https://github.com/Orion-AI-Lab/KuroSiwo-annotations/tree/b930dd278dfec69986f4e33c742958c1199ec86c/polygons/EMSR562_Australia/aoi/13/r1_v2)에서 작은 AOI geometry(236 B), AOI 속성(595 B), event 속성(121,729 B)을 읽었다. Burketown과 polygon label 속성은 확인됐으나 source acquisition date는 없다. `dmg_src_id`는 0/2의 속성값으로 Sentinel 제품 ID가 아니며 임의로 연결하지 않았다.

## 전체 7,000타일로 확장할 때

공개 [GEO-Bench generator 473–480행](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/d7ae6aafc18e1da1bd6efb130f0a306125ee0fef/geobench_v2/generate_benchmark/kuro_siwo.py#L473)은 `KuroV2_grid_dict_test_0_100.gz`에서 전체 metadata를 읽고 event별 train/validation/test를 부여한다. 직접 전체 행을 읽은 결과도 67,490 unique grid / 43 event / 72 event-AOI로 확인됐다. 공개 split 목록 기준 train 48,768, validation 2,826, test 15,896이다. 목록에 있으나 metadata에서 빠진 사건은 validation 514 하나다. train/test 목록의 사건은 모두 존재한다. 파일 이름의 test는 범위를 test-only로 뜻하지 않는다.

일반 `KuroV2_grid_dict.gz`는 31,707 unique grid / 43 event / 72 event-AOI이며 pwater=0 AND pflood=0인 행이 없다. 무침수 hard-negative를 포함하는 T0 조인에는 **이미 받은 test_0_100 파일 하나를 기본**으로 쓰는 것이 적합하다. 공식 GitHub tree에 별도 train/val pickle은 없다. [metadata_coverage_inventory.json](/private/tmp/kuro_date_metadata_20260925/metadata_coverage_inventory.json)에 사건별 개수·zero-water 개수·공개 split 목록을 저장했다. 이는 파일 범위 감사이며 현재 7,000개 전체의 포함·유일 대응을 확인한 결과는 아니다.

필요한 다음 입력은 **각 현재 타일의 전체 원 affine·shape·CRS와 actid/aoiid/flood_date/pcovered/pwater/pflood가 든 작은 metadata table**이다. 현재 검색 catalog의 반올림된 centroid만으로 exact geometry를 주장하면 안 된다. 서버 원 tortilla metadata를 읽기 전용으로 소형 JSONL로 내보내는 별도 단계가 적합하다. 이후 (actid,aoiid,전체 사각형)으로 dictionary index를 만들고, metadata 두 파일 간 중복 UUID의 record 일치·0개/1개/복수 후보·필드 불일치를 모두 보존한다. 누락은 가까운 grid로 대체하지 않는다.

우선순위는 **① 공개 GEO-Bench mapping과 현재 두 타일의 실행 계보·원 raster 대응 연결 → ② 전체7,000타일 metadata 조인 및 원 날짜 간격 분포 감사 → ③ 검증된 별도 데이터 계약으로 이후 실험 설계**다. 기존 E5의 입력·판정·결과를 소급 수정하는 작업은 아니다.
