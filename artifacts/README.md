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
| `jeju_v5_quality_summary.png` | v1↔v5 전수 품질 감사 | cloud/zero/bad, strict clean coverage, window 분포가 완전히 겹침 | ❌ `PER_PERIOD_MOSAIC` 변경이 실제 입력을 바꾸지 않음 |
| `jeju_v5_rgb_blind_pairs.png` | v1에서만 선택한 blind 5쌍 | 좌(v1)·우(v5)의 구름 경계와 질감이 5/5 동일 | ❌ 육안 개선 0/5; 수치 감사와 일치 |
| `jeju_change_v6_4vs12.png` | v6 (4기간 vs 12기간) | Top-30 교집합 5곳, Jaccard 0.091로 입력 기간 수에 따라 후보 지도가 크게 이동 | ⚠️ **민감도 확인, 정답 확인 아님.** 그림의 “Jan-Apr” 표기는 잘못됐고 실제 4기간은 역시간순이며 2026과 계절이 불일치한다. 실패 계보 증거로 원본 보존 |
| `v7_rgb_pairs.png` | v7 SCL BestClear golden window | 사전 고정 target의 period 3이 큰 밝은 구름에서 clear scene으로 바뀌고 period 0 구름 패치도 제거 | ✅ 1윈도우 수치·육안 게이트 통과 / ⚠️ 제주 전체 일반화는 미검증 |

## 결과 데이터 (`results/`)

### Transfer 본선 compact audits

| 파일 | 내용 |
|---|---|
| `confirmatory_mechanism_audit_v1.json` | M65의 72 test JSONL을 다시 읽은 M86 CPU 감사. 8지역 empty-tile FP, paired P2/P4 tile win/loss, target-label oracle headroom, 모든 source SHA-256과 금지 주장 포함 |
| `nepal_np89_robustness_audit_v1.json` | sibling Nepal의 NP-88을 사후 재감사. AUPRC·기관별 결과·강한 수분지수/spectral baseline·5.12 km 공간 block·같은 창/강거리 조건부 AUROC와 source SHA 포함. 122,558 토큰은 독립 n이 아니며 event n=1 |

| 파일 | 내용 |
|---|---|
| `../KOREA_PUBLIC_DATA_CATALOG.md` | 공식 공공데이터 23개 연결 후보의 접근성·PNU/공간/시간 join·evidence 역할·no-match 계약과 P0 구현 순서 |
| `forest_loss_peru_100events.geojson` | 페루 산림손실 100건의 분류 결과 (클래스, 확률벡터, 좌표) |
| `jeju_change_top.json` | v1 변화 후보 (전부 바다 — 실패 기록) |
| `jeju_change_v2_top.json` | v2 후보 30곳 (구름 오염 — 실패 기록) |
| `jeju_change_v3_top.json` | v3 후보 30곳 + 클래스별 통계 + 깨끗한 픽셀 비율 63.1% |
| `jeju_change_v4_top.json` | v4 결과 (후보 0건, 생존 1.2%) |
| `jeju_change_v6_top.json` | 4기간↔12기간 Top-30과 Jaccard 0.091. 시간축·RGB 검증 전의 민감도 결과 |
| `jeju_v5_quality_summary.json` / `jeju_v5_quality_per_window.csv` | 216×12기간 v1↔v5 cloud/zero 전수 감사 |
| `jeju_v5_equivalence.json` | 동일 handler, source-group 2,592/2,592, 원본·임베딩 표본 동일 증거 |
| `jeju_v5_rgb_blind_pairs.json` / `jeju_v5_rgb_manual_review.json` | blind RGB 선택과 0/5 육안 개선 판정 |
| `jeju_time_axis_summary.json` / `jeju_time_axis_manifest.csv` | 실제 월 순서·source item 수·ordered hash manifest |
| `jeju_candidate_time_contract_audit.json` | 2025/rolling-2026 184일 중첩·4기간 계절 불일치와 기존 14후보 lineage 재감사. 중첩 5, 4기간 5, 합집합 9/14는 annual-change claim 부적격이며 시각 false positive 수와는 다름 |
| `v7_summary.json` / `v7_rgb_manual_review.json` | 1-window v1↔v7 수치 게이트와 고정 RGB 육안 판정. bad −95.64%, target 1.00→0.00 |
| `jeju-v7-smoke-attempt1.log` ~ `attempt3.log` / `jeju-v7-smoke.log` | enum, import path, SCL 보조-band 등록 실패 3건과 최종 성공 로그 |
| `human_review_v1/dashboard.html` | 14개 후보의 2023~2026 5월 정렬 RGB와 수정 가능한 Codex 1차 판정. 브라우저 저장·JSON export 지원 |
| `human_review_v1/manifest.json` / `assistant_review.json` | RGB 전 후보 선택 provenance와 분리된 사람 판정. 고확신 5 records = 중복 제거 4 sites |
| `human_review_v1/candidates/*.png` | 후보별 1.28 km 맥락 + 400 m 상세, 동일 0–3000 DN stretch의 4개년 비교 14장 |
| `external_data/korea_public_v1/evidence_dashboard.html` | 4개 고유 고확신 후보의 RGB-only 판정과 한국 공공/지도 문맥을 나란히 비교·수정·JSON export하는 self-contained UI |
| `external_data/korea_public_v1/candidate_public_context.json` / `assistant_context_review.json` | offline OSM 공간 결합 결과와 분리된 Codex 결합 후 판정. 2건 강화, 2건 문맥 추가, 0건 변화 판정 번복 |
| `external_data/korea_public_v1/jeju_oreum_official_20240331.csv` | 제주특별자치도 공식 오름현황 368건 UTF-8 정규화본. 주소·속성은 있으나 좌표/경계 없음 |
| `external_data/korea_public_v1/jeju_development_permits_20260819.csv` | 국토부 전국 개발행위허가 최신 ZIP에서 추출한 제주 240건. 2023/24 행 부재로 불일치를 음성 증거로 사용 금지 |
| `external_data/korea_public_v1/sources_and_summary.json` | 제공기관·기준일·라이선스·원본 hash·허가 연도 분포·키 필요 레이어 상태 |
| `external_data/korea_public_v1/candidate_images/*.png` | evidence dashboard를 좁은 로컬 서버로 안전하게 제공하기 위한 선택 4사이트 RGB 사본 |
| `external_data/kearth_public_ingest_v1/run_summary.json` | 공식 FarmMap 289,379건·개발행위 240건·산지이용 19년의 한 번 실행 요약, claim policy와 모든 파생파일 SHA-256 |
| `external_data/kearth_public_ingest_v1/jeju_farmmap_manifest.json` | 2025 제주 FarmMap 원본 hash·CRS·스키마·시군별 행수·등급/촬영/갱신일 coverage manifest |
| `external_data/kearth_public_ingest_v1/farmmap_evidence_edges.json` | 4 변화좌표와 243 OSM 오름점의 offline point-in-polygon edge. `r08` B급 변화 전 농지상태 1건, 오름점 C급 7건; 원인 주장 금지 |
| `external_data/kearth_public_ingest_v1/farmmap_permit_pnu_links.csv` | FarmMap↔개발행위허가 exact PNU 문맥 206행·50 PNU·144 농지 polygon. 사건시점·변화 footprint가 없어 원인 근거가 아님 |
| `external_data/kearth_public_ingest_v1/development_permit_coverage_audit.json` | 보유 제주 허가 240행의 PNU/날짜/연도 coverage와 no-match 해석 불가 판정 |
| `external_data/kearth_public_ingest_v1/jeju_forest_use_2008_2026.csv` | 제주시 산지이용지정현황 공식 연간 19행 정규화본. parcel join이 아닌 행정활동 coverage 경보 |
| `external_data/kearth_public_ingest_v1/raw/` | 공식 원본 보존: FarmMap ZIP 약 98 MB와 산지이용 CSV. Git 반영 전 LFS/객체저장소 정책 필요 |
| `external_data/kearth_api_snapshot_v1/dashboard.html` | OlmoEarth 14후보와 BuildingHUB 8,794행·EIA 13 polygon·GK2A·환경부 토지피복 42장을 같은 evidence gate로 보여주는 API 결합 보드. 결론 14/14 보류 |
| `external_data/kearth_api_snapshot_v1/requests.json` / `raw/` | 207개 secret-safe request manifest와 redaction된 원본 응답 207개. HTTP success와 API semantic success(200)를 분리하고 SHA-256 보존 |
| `external_data/kearth_api_snapshot_v1/building_events.json` / `eia_features.json` | 기존 PNU에서 확인된 제주 45법정동의 BuildingHUB 2023–2026 8,794행과 제주 bbox EIA 13 polygon |
| `external_data/kearth_api_snapshot_v1/observation_context.json` | 과거 6시점 조회 제한·최신 GK2A 127,040값과 14후보×3개년 토지피복 tile 42장의 시점·hash·파일 연결 |
| `external_data/kearth_api_snapshot_v1/candidate_evidence.json` / `cross_source_pnu_links.json` | 14후보 필지·법정동·EIA 결합과 기존 PNU 58개↔BuildingHUB exact 9개 감사. `r08` 같은 법정동 77건·exact PNU 0으로 보류 |
| `external_data/kearth_api_snapshot_v1/vworld_vm_probe/` | H100 VM 대표점에서도 로컬과 같은 `INCORRECT_KEY`를 반환한 secret-safe 실패 증거. key/domain 수정 전 반복 수집 금지 |
| `external_data/kearth_vworld_probe_20260822/` | 재승인 뒤 대표 후보 1점이 HTTP 200·`status=OK`·feature 1을 반환한 전수 확장 gate |
| `external_data/kearth_vworld_snapshot_v1/` | 후보 14+위치화 오름 243의 VWorld-only 응답. 256 feature·1 `NOT_FOUND`, 고유 PNU 235, raw/request SHA 보존 |
| `external_data/kearth_api_snapshot_v2/` | v7.6+VWorld 첫 결합본. 수치는 맞지만 candidate record에서 기존 FarmMap anchor를 보존하지 않은 결함 때문에 superseded; 실패 계보로 유지 |
| `external_data/kearth_api_snapshot_v3/dashboard.html` | 현재 canonical API 결합 보드. HTTP 463/463, semantic 성공 456·유효 무항목 1·과거 GK2A 오류 6, VWorld 후보 14/14·오름 242/243, 원인근거 0/14 |
| `external_data/kearth_api_snapshot_v3/requests.json` / `raw/` / `COMPLETE.json` | v7.6 비-VWorld 206응답+VWorld 257응답의 redacted raw, 입력 snapshot SHA·응답 lineage·완주/secret-scan marker |
| `external_data/kearth_api_snapshot_v3/parcel_anchors.json` / `candidate_evidence.json` | current VWorld와 dated FarmMap 필지를 분리 보존. `r08` PNU 충돌 1건, `r04` exact 건축사건 1건이나 시간정렬 0으로 14/14 보류 |
| `external_data/kearth_oreum_v1/dashboard.html` | 공공데이터 신청·현재 자산·사업·한국 연구·EarthRoute의 5축과 데이터 custody 경계를 먼저 보여주고, 기존 공식 368개 오름의 목록·위치·모델 screen·행정근거·사람 검수·10% 보류 gate를 그대로 보존한 프로그램 보드 |
| `external_data/kearth_oreum_v1/oreum_evidence_registry.json` / `oreum_registry_368.csv` | 제주 368건 고정 분모의 K-Earth Evidence Graph. 공식 속성 A, OSM point C, 지역 허가 D, 모델 M, 미확인 U를 별도 저장 |
| `external_data/kearth_oreum_v1/attachment_jeju_city_210.csv` | 사용자 제공 HTML 표의 재현 가능한 정규화본. 자체 1–210 순번이며 공식 연번과 직접 결합 금지 |
| `external_data/kearth_oreum_v1/evidence_coverage.json` | 전수 상태화와 전수 판정을 구분한 coverage, A/B급 원인 근거율, 선택적 변화탐지 전환 상태 |
| `external_data/kearth_oreum_v1/oreum_model_scores.json` | OSM 위치가 해결된 243건의 기존 4/12기간 점별 변화 screen. high-stable 8, high-unstable 34, moderate-stable 4, low/unstable 197 |
| `external_data/kearth_oreum_v1/rgb_review/dashboard.html` / `manifest.json` | 모델 안정 8건 + 기존 후보 500m 이내 1건의 동일계절·고정 stretch RGB 검수 UI와 입력 provenance |
| `external_data/kearth_oreum_v1/rgb_review/assistant_review.json` | 9건 육안 판정. 고확신 지속 변화 0, 구름·해무/공유입력 오염 8, 추가 입력 필요 1(성산일출봉) |
| `external_data/kearth_oreum_v1/rgb_review/candidates/*.png` | 2023–2026 5월 최근접 1.28 km context + 400 m detail 검수 패널 9장 |
| `benchmarks/k_evidence_shift_jeju_pilot_v0/` | 14개 후보를 성능표가 아닌 audit pool로 고정한 site-event manifest, 시간·공간·scene·PNU 누수 검사, promotion gate, 결정적 SHA와 COMPLETE marker |
| `release_audit_p0/checkpoints.json` | OlmoEarth v1/v1.2의 Hugging Face commit과 config/weight SHA-256, 서버 Python·rslearn·olmoearth-pretrain 환경 고정본 |
| `release_audit_p0/smoke_inputs_exact.json` / `preflight.json` | label-free smoke 8개에 쓰이는 208개 tensor/metadata exact hash와 GPU0만 비어 있음을 확인한 selected-device preflight |
| `release_audit_p0/results/run_summary.json` / `COMPLETE.json` | GPU0에서 순차 실행한 v1/v1.2 각 8개 output의 SHA·mtime·identity inventory와 완료 증명 |
| `release_audit_p0/results/analysis/analysis_summary.json` / `per_window_metrics.csv` | 8 label-free·7 spatial-cluster의 CKA·shift-null·이웃 보존·거리 순위상관과 금지 주장 |
| `release_audit_p0/results/verification_v1/verification.json` / `VERIFICATION_COMPLETE.json` | 서버 raw 228파일·7,851,565,383 bytes, preflight·GPU0 로그·config·identity·marker를 758/758 재검증한 `FULL_EVIDENCE_VERIFIED` 증명 |
| `release_audit_p0/results/verification_v1/reanalysis/` | 동일 raw에서 새로 계산해 기존 analysis JSON/CSV와 byte-identical임을 확인한 결정성 재실행 |
| `release_audit_full216_v1/README.md` | 제주 54위치×4년 전체 v1/v1.2 감사의 약점·실행·봉인 평가·허용/금지 주장. 최선 ridge R@1 0.697/0.609로 0.95 cache gate 실패 |
| `release_audit_full216_v1/paired_evidence_strict1/` | 입력 5,616파일·56.68GB와 출력 432파일·105.59GB의 SHA/grid/mask/value-health 폐쇄성, 216 paired manifest와 완료 marker |
| `release_audit_full216_v1/analysis_strict1/` | calibration-only bridge와 sealed 16위치/64건의 identity retrieval·neighbor·manifest-window continuity. analysis summary/CSV/preanalysis lock·종료 코드 검증 |
| `release_audit_full216_v1/v1/result/` / `v1_2/result/` | GPU0 batch8/workers4 실행 summary·telemetry·로그·사후 입력/checkpoint/rslearn/code 검증 marker. raw GeoTIFF는 서버에만 유지 |
| `osm_aqua_wando.json` / `osm_aqua_jeju.json` | OSM 양식장 좌표 (프로토타입 쿼리 라벨) |

## 서버에만 있는 것 (용량 때문에 미포함)

`/home/work/data/olmoearth/` 아래:
- `embed_search/dataset/` — 244윈도우 임베딩 스토어 (768차원 × 40m, 약 540만 벡터)
- `embed_jeju_v2/` — v1/v5(의미적 중복) 제주 4개년 원본 + 4기간/12기간 임베딩
- `embed_jeju_v7_smoke/` — SCL BestClear 1-window 입력 개입과 audit 산출물
- `release_audit_p0/smoke_dataset/` — 원본을 수정하지 않고 8개 smoke window의 96 S2 layer를 symlink한 v1/v1.2 공용 audit view
- `release_audit_p0/results/` — GPU0에서 완주한 16개 paired release output GeoTIFF. 로컬에는 로그·SHA inventory·완료 marker·소형 분석표만 복사
- `release_audit_full216_v1/v1_b008w04_strict1/` / `v1_2_b008w04_strict1/` — 216×2 full
  output GeoTIFF 약 98.34 GiB. 로컬에는 실행 summary·telemetry·paired manifest·봉인 분석만 복사
- `cloud_stats.npz` — 제주 4개년 픽셀별 구름 통계(평균·최댓값). **품질 마스크 자산**
- `scratch/lfmc/trainer_checkpoints/epoch=33-*.ckpt` — 우리가 재학습한 LFMC 모델 (test MSE 558.8)
