# OE11 — 학습 지역 확장을 위한 실제 자료 경로

확인일: 2026-09-29. 공식 논문·저장소·Zenodo 레코드만 읽었다. 데이터 다운로드, 서버 접속, GPU 실행, 유료 API 호출 없음. 현재 80 patch의 split 정보는 root가 전달한 조건이며 이 검토에서 직접 검증하지 않았다.

**바로 할 일은 보유 PASTIS archive에서 두 train tile의 미사용 patch를 늘리는 것이다. 이는 표본 확대이며 새 지역 추가는 아니다. t31tfm dev와 t30uxv final을 유지하면 PASTIS 안에서 추가할 다섯 번째 tile은 없다. PASTIS-R도 지역 확대가 아니다.**

## 1. PASTIS: 재다운로드 전에 기존 archive 식별

| 공식 배포 | 정확한 파일명 | 레코드의 크기 / MD5 | 활용 |
|---|---|---|---|
| [PASTIS, record 5012942](https://zenodo.org/records/5012942) | `PASTIS.zip` | 28.8 GB / `cfc441bf18137ff0bbf4fad58828fb98` | 기존 S2 dense crop mask 학습에 직접 호환 |
| [PASTIS-R, record 5735646, v1](https://zenodo.org/records/5735646) | `PASTIS-R.zip` | 53.7 GB / `4887513d6c2d2b07fa935d325bd53e09` | 동일 patch의 S1A/S1D 추가, 새 지역 없음 |
| [PASTIS-R PixelSet, record 5745151](https://zenodo.org/records/5745151) | parcel pixel-set 배포 | 공식 README 기준 27 GB zipped | dense segmentation용 원래 patch와 혼동하지 말 것 |

파일은 각 레코드 Files의 해당 Download 링크로 접근한다. ZIP 다운로드나 HEAD는 수행하지 않았으므로 서버의 약 40 GB 파일이 어느 배포인지 크기만으로 단정하지 않는다. 원본 ZIP이면 위 checksum, 재패키징본이면 내부 manifest·출처 기록으로 확인한다. PASTIS 레코드 직접 open은 일시 오류였지만 공식 저장소 링크와 Zenodo 검색 본문의 파일명·checksum은 확인했다.

[공식 README](https://github.com/VSainteuf/pastis-benchmark)는 2,433 patch, 128×128, 10 m, S2 10 bands와 PASTIS-R의 동일 위치 확장을 명시한다. [공식 supplementary](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Garnot_Panoptic_Segmentation_of_ICCV_2021_supplemental.pdf)는 네 tile을 T30UXV/T32ULU/T31TFM/T31TFJ로 제시한다. 공식 5 folds는 지역 holdout과 같지 않다.

[공식 loader](https://github.com/VSainteuf/pastis-benchmark/blob/main/code/dataloader.py)에서 확인한 추출 최소 항목:

- `metadata.geojson`: `ID_PATCH`, `Fold`, `dates-S2`; geometry와 실제 archive의 tile 필드를 이용해 지역을 확인한다. 이 검토에서는 region 필드명을 추측하지 않았다.
- `DATA_S2/S2_<ID_PATCH>.npy`: `T×C×H×W`.
- `ANNOTATIONS/TARGET_<ID_PATCH>.npy`: semantic label은 첫 번째 plane인 `target[0]`.
- 원래 `NORM_S2_patch.json`을 그대로 쓰면 모든 fold의 통계가 섞일 수 있으므로 기존 train-only normalization 계약을 유지한다.

현재 대상 ID는 meadow=1, corn=3, grapevine=8, leguminous fodder=14이며 [IGN 공식 PASTIS-HD label table](https://huggingface.co/datasets/IGNF/PASTIS-HD/blob/main/documentation/label_names.json)과 대조할 수 있다. 추출할 원본 annotation의 동일 nomenclature도 확인한다.

실행 제안: 기존 train 48에서 128→256 patch로 늘리되, 먼저 두 train tile의 가용 ID와 클래스·날짜 coverage만 inventory한다. 16 bank/16 dev 및 공간 인접·동일 parcel 충돌을 제외한 고정 manifest를 만든다. 특정 희귀 클래스가 부족하면 그 사실을 기록하고 중복 patch로 목표 수를 채우지 않는다. t30uxv 이미지·label은 열지 않는다.

## 2. 실제 새 지역 후보 하나: TimeMatch — raw crop code를 써야 부분 호환

[공식 저장소](https://github.com/jnyborg/timematch), [논문 §4.2](https://arxiv.org/html/2111.02682v2), [최신 공식 데이터 v2, record 6542639](https://zenodo.org/records/6542639).

- `timematch_data.zip`: 73.9 GB, MD5 `b7909ba1d03e5c1abdac1c8c6a23f9ad`; v1과 같은 데이터 checksum. 저장소는 약 78 GB extracted라고 설명한다. v2에는 별도 `thermal_time.zip` 193.6 MB도 있다.
- 2017 S2 L1C TOA, 10 bands. 새 위치는 DK `32VNH`, FR `30TXT`·`31TCJ`, AT `33UVP`.
- 배포 loader 입력은 dense image가 아니라 각 parcel의 `T×C×N_pixels` zarr다. 20 m erosion 및 1 ha 미만 parcel 제외가 적용됐다. PASTIS와 radiometry·연도·경계 조건이 다르므로 바로 concat하지 않는다.

**현재 네 target에 대한 실제 라벨 호환성:**

| 현재 target | TimeMatch 공식 mapping | 안전한 취급 |
|---|---|---|
| corn | FR `MIS/MIE/MID`, DK `216/5/423`, AT `105/109/106/173/107` → corn | 가장 직접적인 공통 클래스. 세부 정의·관측 수 확인 후 확장 가능 |
| meadow | 공식 meadow에 일반 초지뿐 아니라 alfalfa/clover/fodder-legume mix 포함 | 그대로 매핑하면 현재 meadow/leguminous-fodder 경계를 훼손 |
| leguminous fodder | FR `LU*`, `TR*`, `MH*`, `MLG` 등 일부가 meadow에 병합; 다른 fodder는 unknown | FR 원 crop code를 분리해 PASTIS의 정확한 정의와 재대조해야 함 |
| grapevine | FR `VRC/VRT/RVI`가 unknown 아래 있음 | `VRC/VRT`를 원 코드에서 복구 가능; restructuring `RVI`는 별도 판단/제외 |

근거는 실제 [France mapping YAML](https://github.com/jnyborg/timematch/blob/main/class_mapping/france_class_mapping.yml), [Denmark YAML](https://github.com/jnyborg/timematch/blob/main/class_mapping/denmark_class_mapping.yml), [Austria YAML](https://github.com/jnyborg/timematch/blob/main/class_mapping/austria_class_mapping.yml)이다. 프랑스 YAML은 일부 fodder를 meadow에 합친 이유가 Denmark 자료에서 분리가 안 되기 때문이라고 명시한다. YAML 괄호의 수치를 우리 조건에 맞는 최종 가용 sample 수로 쓰지는 않는다.

[공식 dataset.py](https://github.com/jnyborg/timematch/blob/main/dataset.py)는 `meta/metadata.pkl`의 parcel별 `label` 원 crop code를 읽고 runtime에 YAML mapping을 적용한다. 따라서 **FR 두 지역을 먼저 골라 원 코드를 다시 매핑**할 실질적인 경로가 있다. unknown을 background로 바꾸면 안 된다.

활용 순서는 두 갈래다. parcel classification 보조 실험이면 TimeMatch pixel-set loader를 별도로 붙일 수 있다. 현재 OlmoEarth 공간 patch+분할 학습에 넣으려면 원 parcel geometry와 S2 장면에서 연속 raster patch를 다시 구성해야 한다. 배포에 좌표 복원 정보가 충분한지는 다운로드 전 문서만으로 확인되지 않았다. pixel-set을 임의 격자로 reshape해 위성 영상처럼 넣지 않는다.

**권고:** 당장 실행은 보유 PASTIS train 두 tile의 표본·계절 coverage 확장. 실제 새 지역은 TimeMatch FR `30TXT/31TCJ`의 원 code inventory와 공간 복원 가능성 확인을 다음 작은 준비 단계로 잡는다. 4개 target과 dense 입력이 이미 완벽하게 호환되는 즉시용 외부 archive를 찾았다고 보고할 근거는 없다. 이 차이를 명확히 해야 다운로드 후 라벨 재설계로 되돌아가는 일을 줄인다.
