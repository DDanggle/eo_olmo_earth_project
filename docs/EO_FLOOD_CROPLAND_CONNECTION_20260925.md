# 홍수와 경작지의 실제 공간 연결 — 2026-09-25

**현재 한계:** 선택한 train 사례 1개를 계산한 개발 결과다. 자동 탐지·작물 피해·회복·VLM 성능·기억 효과를 검증한 실험은 아니다. 전체 7,000칩 중 6,999칩의 교집합은 미계산이다. SAR의 실제 취득 시각은 원본에서도 확보하지 못했다.

그 범위 안에서 **지역 → 원 좌표의 영상과 침수 라벨 → 과거 경작지 지도 → 공간 교집합 → 근거 그림·출처·미확인 항목**이 실제로 연결됐다. 문서상의 설계에서, 재현 가능한 연구용 기준선으로 한 단계 진행했다.

## 직접 볼 곳

- 로컬 화면: http://127.0.0.1:8774/ → **2020 경작지와 겹친 침수**.
- 새 검색 자료: `artifacts/eo_evidence_search_v1_20260925/`. 이전 v0 자료는 보존.
- 계산 수치·입력 해시: `artifacts/eo_flood_overlap_v1_20260925/ks_00276_overlap/overlap.json`.
- 전후 SAR·라벨·토지피복·교집합 그림: 같은 폴더 `preview.png`.
- 원 래스터/메타데이터: `artifacts/eo_flood_overlap_v1_20260925/ks_00276/`.
- 실제 관측 footprint: `ks_00276_overlap/footprint.geojson`. 침수 경계가 아니라 원 래스터의 범위이며 화면에서 GeoJSON 저장 가능.

## 이번에 확인한 수치

공식 KuroSiwo catalogue의 activation1111009 / AOI01은 **Pakistan, Larkana**, 참조 사건일은 2022-09-10이다. 원 카탈로그와 동일한 `ks_00276`, train 사례를 선택했다. 이미 침수율이 높은 것으로 알려진 사례를 연결 검증용으로 골랐으며 대표 표본이 아니다.

| 항목 | 계산값 | 의미 |
|---|---:|---|
| 칩 범위 | 약 390.0 ha | 원 224×224 격자의 WGS84 지표면 면적 |
| 침수 참조 라벨 | 약 377.3 ha | raw class2 영역 |
| 2020 경작지 | 약 375.6 ha | 함께 유효한 영역의 WorldCover class40 |
| 경작지 × 침수 | **약 366.5 ha** | 3,665,141.601㎡, 47,151픽셀 |
| 위 경작지 중 침수와 겹친 비율 | 97.6% | 과거 경작지 375.6ha가 분모 |
| 마스크상 확인 불가 | 0.0 ha | 이번 칩의 유효 마스크·토지피복 클래스상 결측 없음 |
| 2020 수목 피복 × 침수 | 0.0 ha | 이 칩에는 class10 자체가 없음; 지역 전체의 산림 피해 부재가 아님 |

중심점은 위도27.954155 / 경도67.694659. 범위는 경도67.684598–67.704720, 위도27.945267–27.963042. 좌표는 원 TIFF에서 복구했으며 중심점으로 임의의 사각형을 생성하지 않았다. 검색 bbox는 기존 계약대로 중심점 포함 여부를 사용한다.

## 계산에서 고친/분리한 것

1. **평면 격자 면적과 지표면 면적.** 원 TIFF는 EPSG:3857, 격자10×10이다. 50,176픽셀×100㎡는501.76ha지만 지표면 면적은390.03ha로 약28.6% 과대평가한다. 각 픽셀 네 모서리를 WGS84로 변환해 타원체 위 면적을 합산했다. 임의로 100㎡를 곱하지 않는다.
2. **클래스와 nodata의 충돌.** raw TIFF는0비수역/1상시수역/2홍수/3미확인이다. TIFF의 nodata=0 속성을 그대로 적용하면 정상 비수역이 지워진다. 공식 클래스 규칙과 `invalid_data==1` 유효 마스크를 결합했다. loader의 재매핑 후 class3=flood와 원 TIFF의 class2=flood를 구별한다.
3. **실제 취득일과 사건일.** pre_event_1/pre_event_2/post_event 하위 메타데이터가 모두 같은 사건 timestamp를 반복한다. TIFF tags에도 취득일이 없다. 세 취득일을 null로 유지하며 기존 캐시의 -24/-12일 근사를 사용하지 않는다. 두 pre-event 영상 사이의 실제 간격·순서도 확인하지 않았다.
4. **지도 기준연도와 공개일.** WorldCover2020 v100은2021-10-20 공개라 참조 사건일보다 앞선다. 2021지도는2022-10-28 공개로 사건 후 자료다. 이번2020 S3 객체의 Last-Modified는2022-10-25여서 현재 바이트가 사건 당시 공개 파일과 동일한지는 미검증이다. 따라서 완전한 당시 가용자료 재현으로 주장하지 않는다.
5. **범주형 재격자.** 공식 COG의 필요한 영역만 읽고 nearest neighbor로 원 홍수 격자에 맞췄다. 유효 홍수 라벨과 알려진 토지피복 클래스가 동시에 있는 픽셀에서만 계산한다. 원본과 정렬된 TIFF, 배열, 분석 코드의 SHA-256을 기록했다.
6. **자료 없음과 관측 겹침0.** 검색 조건을 먼저 적용한 칩 수를 분모로 한다. 전체 자료에서는1계산/6,999미계산. 기본 경작지 예시의 사건1111009+침수비율≥10%에서는423대상/1계산/422미계산. 수목 피복 겹침0도 `partial_coverage`로 반환한다. 아직 계산하지 않은 다른 사건은 `needs_data`다.

WorldCover class40은 **연간 초본성 경작지**다. 다년생 목본 작물·온실을 모두 포함한 농업용지가 아니다. 2020년 이후 실제 토지 변화, 분류 오차, 정합·경계 재격자 오차는 남아 있다. 결측0은 정확도100% 또는 불확실성0을 뜻하지 않는다. 픽셀 수와 소수점은 계산 재현용이며 현장 정밀도 주장이 아니다.

## 논문·VLM과의 연결

이번 경로는 **도구로 계산 가능한 답과 시각 판독이 필요한 답을 구분하는 기준선**으로 유의미하다. 다만 라벨과 지도만으로 구할 수 있는 면적을 VLM 성과로 주장하면 안 된다.

| 질문 | 현재 답변 범위 | VLM 실험에서 필요한 구분 |
|---|---|---|
| 어느 관측 범위에서 과거 경작지와 침수가 겹쳤나? | 공식 참조 라벨의 공간 계산으로 답 가능 | 참조 라벨은 평가용 정답/상한. 자동 후보 생성에 몰래 입력하지 않음 |
| 실제 영상에서 침수 증거를 읽는가? | 이번에는 모델 미실행 | 원 임베딩·교체·제거, 같은 예산의 텍스트/메타데이터 기준선 |
| 정확히 언제 침수됐나? | 사건 참조일만 있음 | 실제 취득 시각 복구 전에는 정밀 시점 정답을 만들지 않음 |
| 경작지에 얼마나 피해가 났나? 언제 회복했나? | 확인 불가 | 사건 후 연속 관측·작물/피해 또는 회복 정답이 별도로 필요 |
| 제한된 기억으로 같은 답을 보존할 수 있나? | 미검증 | reader가 내용을 읽는지 확인 후, 동일 reader/자료/예산에서 저장 전략 비교 |

연구의 현재 방향은 `docs/DIRECTION_EO_EMBEDDING_READER_2026_09_25.md`와 E0/E1이다. 이 응용은 그 방향의 입력 계약·근거 표시·강한 공간 도구 기준선으로 연결한다. D1 H는 A 단독 완료/합의 미정 상태로 보류이며, 이번 자료 연결로 H를 통과시키거나 이전 실패를 덮지 않는다.

**다음 독립 실험:** 한 칩을 긍정 결과로 발표하는 대신, train의 원본 좌표·실제 시각 계약을 먼저 정리하고 지역/사건 단위 표본을 고정한다. 참조 라벨을 분리한 자동 검색과 단순 공간 계산 기준선을 비교한다. 모델 평가는 보류 사건에서 grounding 정확도, 알려진 것/모르는 것의 판별, 근거를 교체했을 때 답의 변화로 평가한다. 실제 피해·회복은 정답이 없는 상태에서 문구만 생성하지 않는다.

## 재현

서버 원본 복구와 교집합은 각각 `code/eo_flood_source_probe_v0.py`, `code/eo_flood_overlap_v0.py`. 모두 CPU이며 원본/기존 학습 파일을 변경하지 않는다. 해당 서버 환경은 `.venv-geobench`; 조작은 `./bin/nx`로 한다. 같은 출력 폴더가 이미 존재하면 덮어쓰기를 거부한다.

```bash
# 로컬 미리보기 재실행
python3 -B code/eo_evidence_search_v0.py serve \
  --out artifacts/eo_evidence_search_v1_20260925 --port 8774

# HTTP 통합 검증
python3 -B code/verify_eo_overlap_search_v0.py \
  --out artifacts/eo_evidence_search_v1_20260925/api_checks.json

# 검색 계약 검사
PYTHONPATH="$PWD/code" python3 -B -m unittest discover -s tests -p 'test_eo_query_core_v0.py'
PYTHONPATH="$PWD/code" python3 -B -m unittest discover -s tests -p 'test_eo_landcover_partial_coverage.py'
```

새 자료 조립은 `code/eo_attach_flood_overlap_v0.py --help`. 입력은 기존v0검색폴더, overlap.json, source_manifest.json, preview.png이며 입력 해시·사건ID/날짜/분할을 검사한다. 새 결과 폴더만 생성한다. 그림 생성은 `code/render_eo_overlap_v0.py`(NumPy/Pillow). 각 단계의 산출물과 코드는 함께 보존한다.

검증 결과: **검색 계약40개 + 실제좌표/면적/마스크10개 + HTTP24개 통과**. 브라우저에서 경작지 양성, 수목 피복 관측0+미계산, 다른 사건 자료부족, SN7기존검색·이미지·좁은 화면을 확인했다. 과학적 모델 정확도 검사와는 별개다. 서버 보호 실행파일4개의 SHA-256·mtime이 전후 동일하며 GPU학습을 실행하지 않았다.

## 1차 출처

- [KuroSiwo 공식 지역·사건 catalogue](https://github.com/Orion-AI-Lab/KuroSiwo/blob/main/catalogue/catalogue.yaml)
- [KuroSiwo valid mask 처리](https://github.com/Orion-AI-Lab/KuroSiwo/blob/main/dataset/Dataset.py)
- [GEO-Bench-2 raw/재매핑 클래스](https://github.com/The-AI-Alliance/GEO-Bench-2/blob/main/geobench_v2/datasets/kuro_siwo.py)
- [ESA WorldCover 데이터·분류·사용 조건](https://esa-worldcover.org/en/data-access)
- [ESA 제품 공개일 설명](https://eo4society.esa.int/2022/11/11/2021-worldcover-product/)
- [WorldCover2020 DOI](https://doi.org/10.5281/zenodo.5571936)

© ESA WorldCover project 2020 / Contains modified Copernicus Sentinel data (2020) processed by ESA WorldCover consortium. 일부 영역 추출·재투영됨.
