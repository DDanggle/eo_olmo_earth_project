# 외부 과업 후보 — 시간 증거 보존 캐시(후보 1) 검증용 (2026-09-07 조사)

조사 방법: 웹 확인 2건(GRAM·GEO-Bench-2 / 슬럼·해양·인도적·변화 데이터셋). 미확인 항목은 "미확인"으로 둠. 다운로드·해시·계약 감사 전이므로 **후보**임.

## 0. 가장 중요한 발견
**GEO-Bench-2의 시계열 프로토콜은 "시점별로 따로 인코딩해 임베딩을 평균 → 디코더"임**(arXiv 2511.15658 v2). 우리 봉인 캐시가 (시간, 밴드그룹) 평균인 것과 같은 선택임. 따라서 후보 1의 baseline(mean)이 곧 벤치마크 공식 프로토콜이고, 시간 스케치가 이기면 **벤치마크 프로토콜 자체를 개선**하는 결과가 됨. 실험 코드·사전등록: `code/extract_olmo_temporal.py`, `code/temporal_readout_train.py`, `config/temporal_readout_screen_prereg_v0.json`.

## 1. GRAM (DS4H-GIS) — 사용자 질문
- AAAI 2026, 슬럼 분할용 Region-aware MoE + test-time adaptation. 영상은 ESRI Wayback **1.2 m 단일 시점 RGB**, 영상 자체는 배포 안 함(라벨만). Sentinel도 시계열도 아님.
- 판단: 우리 캐시(10–20 m S2/S1) 검증 대상 **아님**. 다만 "지역별 expert + 새 지역 TTA"라는 문제 설정은 우리 field-adaptation과 같은 질문이라 관련연구로 인용. 슬럼을 Sentinel 스케일로 하려면 IDEABench(아래)가 유일함.

## 2. GEO-Bench-2 안에서 시점별 텐서를 주는 과업(후보 1을 그대로 돌릴 수 있는 곳)
| 과업 | 센서/GSD | 시간 | 라벨 | 크기 | 비고 |
|---|---|---|---|---|---|
| PASTIS-R | S1+S2 10 m | 시계열(최근 N, `num_time_steps`) | 작물 19 seg | 1455/482/496 | 우리 sha256 불일치 재해결 필요 |
| BioMassters | S1+S2 10 m | 월별 최대 12 | 바이오매스 회귀 | 4011/1739/2776, 138 GB | sha256 불일치 재해결 |
| KuroSiwo | S1+DEM 10 m | pre1·pre2·post 3시점 | 홍수 4 seg | 4000/1000/2000, 7.5 GB | **1순위**: 작고 인도적 |
| Fields of the World | S2 10 m | 2 window | 경계 2 seg | 4000/1000/2000 | 확보됨 |
| DynamicEarthNet | Planet 3 m(+S2 문서) | 일/주 | LULC 7 | 700/100/200 | OlmoEarth 캐시 16,000칩 완료(시간 평균) |
| TreeSatAI | S2 TS(문서) | 미확인 | 13 cls | 20000/4000/4000 | 로더 미확인 |
GEO-Bench-2에 진짜 변화탐지(이시점 라벨) 과업은 없음. 시점 사용 수(공식): BioMassters 7, PASTIS 7, DEN 6, KuroSiwo 2.

## 3. 벤치마크 밖 후보(슬럼·해양·인도적·변화) — Sentinel 스케일 우선
| 후보 | 왜 | 시간 구조 | 라이선스/크기 |
|---|---|---|---|
| **KuroSiwo** | S1 3시점, 43 홍수, 수작업 마스크 | 3시점 | CC-BY, 7.5 GB |
| **SenForFlood** | S1+S2 전/중 쌍, 353 사건, CEMS 마스크 | 쌍 | CC-BY-SA, HF 15 GB 부분집합 |
| **CaBuAr** | S2 전/후 12밴드, 산불 534건 | 쌍 | CDLA-P-2.0, 104 GB |
| SEN12-FLOOD | S1+S2 진짜 시계열(약 9시점) | 시계열 | IEEE DataPort, 장면 단위 라벨만 |
| IDEABench | 유일한 Sentinel 스케일 슬럼(8도시, S1+S2, 47,476 패치) | 단일 시점 | CC-BY, 3.7 GB |
| MARIDA / MADOS | 해양 쓰레기·유류 S2 | 부분/단일 | 공개 |
| Global Mangrove Watch | 연도별 맹그로브 변화 라벨(영상 없음) | 연도 | CC-BY, S2 직접 수집 필요 |
| UNOSAT 우크라이나 S1 | 월별 S1 2022–24 + 피해 라벨 | 월별 | CC-BY, 22.8 GB |
| 없음 | 양식장·난민캠프 성장 Sentinel 공개 벤치마크 | — | 논문만 존재 |

## 4. 결정
1. 후보 1(시간 증거) 스크린은 Sen12 개발 2폴드에서 먼저(등록 완료, GPU 순서는 4-arm 뒤).
2. 통과 시 외부 확증 순서: **KuroSiwo → SenForFlood → PASTIS-R/BioMassters(해시 재해결)**. 전부 시점별 텐서가 있어 시간 보존 캐시를 그대로 만들 수 있고, 비교 대상은 GEO-Bench-2 공식 평균 프로토콜.
3. 한국 3-task는 외부 시스템 검증 1건으로 보존(방법 선택에 소모 안 함).
4. 슬럼(IDEABench)·해양(MARIDA)은 단일 시점이라 후보 1에는 부적합. 후보 2(공간 잔차)나 few-shot 전이 검증에는 가능. GRAM은 인용만.
