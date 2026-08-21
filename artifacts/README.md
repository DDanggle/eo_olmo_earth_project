# 아티팩트 — 눈으로 확인하며 만든 것들

각 그림은 실험 하나의 판정 근거입니다. **실패한 그림도 남깁니다** — 실패가 다음 설계의
근거였고, 그 사슬이 이 프로젝트의 방법론이기 때문입니다.

## 발행된 웹 페이지 (claude.ai 아티팩트, 비공개)

| 페이지 | 내용 | URL |
|---|---|---|
| Madre de Dios Forest Loss | 페루 아마존 산림손실 100건 분류 결과 지도 + 5차례 디버깅 여정 | https://claude.ai/code/artifact/b5a54835-93cd-4ebc-9905-c2a7813b977e |
| Korea Earth Search | 임베딩 검색엔진 구축 리포트 (구조·정량평가·데모·삽질 연대기) | https://claude.ai/code/artifact/fe29beda-4856-460a-b0ac-9a7a6346f382 |

## 그림 (`figures/`)

### Korea Earth Search — 검색

| 파일 | 실험 | 무엇을 보여주나 | 판정 |
|---|---|---|---|
| `first_search.png` | 완도 유사도 검색 v1 (원시 임베딩) | 바다 쿼리와 산림 쿼리의 히트맵이 사실상 동일 | ❌ **실패** — 이방성 때문에 모든 cosine이 ~0.7에 몰림. 이 실패가 mean-centering을 찾게 함 |
| `search_v2.png` | 완도 검색 v2 (mean-centering) | 산림/수역/농경지 쿼리가 각각 다른 영역을 점등 | ✅ 판별력 확보. WorldCover 기준 built ×26.0 |
| `jeju_demo.png` | 제주 쿼리 3종 | ① 한라산 아고산 → 정상부 정확히 구획 ② 다랑쉬오름 → 동부 오름·초지(특이도 낮음) ③ 완도 바다 → 제주 바다 전역(의미 특이성 없음) | ⚠️ 부분 성공. ③의 한계가 프로토타입 검색으로 이어짐 |
| `farm_query.png` | 양식장 프로토타입 (few-shot) | 좌: 완도 연안 양식 격자 점등 + held-out 표시 / 우: 제주 내륙 소등, 해안 링 점등 | ✅ held-out 백분위 중앙값 100.0, 제주 교차지역 9/9가 96+ |

### 제주 다개년 변화탐지 — 실패 계보

| 파일 | 버전 | 무엇을 보여주나 | 판정 |
|---|---|---|---|
| `jeju_change_4yr.png` | v1 (연도 간 cosine) | 육지는 어둡고 바다가 온통 밝음, Top-20 전부 바다 | ❌ 파도·반사가 변화로 잡힘 |
| `jeju_change_v2.png` | v2 (계단형 + 층화 z) | 육지 후보 확보, 그러나 서귀포 서부 한 띠에 26/30 집중 | ⚠️ 요동은 잡았으나 오염은 못 잡음 |
| `verify_candidates.png` | v2 후보 육안 검증 | **5곳 중 4곳의 2023a 칸이 구름으로 하얗게 덮임** | ❌ 결정적 실패 판정 — "변화"가 아니라 구름이었음 |
| `jeju_change_v3.png` | v3 (구름 평균 마스킹) | 구름 오염도 지도 + 깨끗한 육지 후보 Top-30 | ⚠️ 시점·공간 분산됨(개선), 그러나 3/5 구름 잔존 |
| `verify_v3.png` | v3 후보 육안 검증 | 2번째 후보(33.5087N 126.5747E)에서 숲 → 나지가 연차적으로 확장 | ✅ **진짜 변화 1건 확인** / ⚠️ 나머지는 여전히 구름 |
| `jeju_change_v4.png` | v4 (최악 모자이크 기준) | 최악-모자이크 구름 지도 + 거의 전멸한 후보 | ❌ 생존 픽셀 1.2%, 전부 바다 → **사후 마스킹 원리적 불가** 증명 |

## 결과 데이터 (`results/`)

| 파일 | 내용 |
|---|---|
| `forest_loss_peru_100events.geojson` | 페루 산림손실 100건의 분류 결과 (클래스, 확률벡터, 좌표) |
| `jeju_change_top.json` | v1 변화 후보 (전부 바다 — 실패 기록) |
| `jeju_change_v2_top.json` | v2 후보 30곳 (구름 오염 — 실패 기록) |
| `jeju_change_v3_top.json` | v3 후보 30곳 + 클래스별 통계 + 깨끗한 픽셀 비율 63.1% |
| `jeju_change_v4_top.json` | v4 결과 (후보 0건, 생존 1.2%) |
| `osm_aqua_wando.json` / `osm_aqua_jeju.json` | OSM 양식장 좌표 (프로토타입 쿼리 라벨) |

## 서버에만 있는 것 (용량 때문에 미포함)

`/home/work/data/olmoearth/` 아래:
- `embed_search/dataset/` — 244윈도우 임베딩 스토어 (768차원 × 40m, 약 540만 벡터)
- `embed_jeju_v2/` — 구름 강건 합성으로 재수집한 제주 4개년 (v5용)
- `cloud_stats.npz` — 제주 4개년 픽셀별 구름 통계(평균·최댓값). **품질 마스크 자산**
- `scratch/lfmc/trainer_checkpoints/epoch=33-*.ckpt` — 우리가 재학습한 LFMC 모델 (test MSE 558.8)
