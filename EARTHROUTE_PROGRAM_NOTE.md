# EarthRoute — Decision Continuity Program Note

원문 작성: 2026-08-04  
현재 프로젝트와 통합·재검토: 2026-08-22  
관련 문서: `RESEARCH_STRATEGY.md`, `K_EVIDENCE_SHIFT_BENCHMARK.md`,
`PAPER_READING_LIST.md`, `KOREA_PUBLIC_DATA_CATALOG.md`

> **2026-08-23 위치 보정.** 현재 프로그램 중심축은 K-ALIGN이다. 이 노트는 대체되지 않고
> **뒤로 간다**. 순서는 `K-ALIGN(무엇을 다시 계산하지 않고도 같은 좌표계에서 말할 수 있는가)`
> → `FoldRefresh(그 부분 갱신 위의 통계가 유효한가)` → `EarthRoute(다음에 관측·모델·행정근거·
> 사람검수 중 무엇을 살 것인가)`다. 현재 결정은 `K_ALIGN_PROGRAM_NOTE.md`,
> authoritative 실험 계약은 `KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`가 맡는다.

## 결론부터

EarthRoute는 **연구 프로그램으로 유의미하고 사업으로도 시험할 가치가 있다.** 그러나 팔 것은
`transfer learning`, `OlmoEarth fine-tuning`, `제주 변화지도`, `범용 EO scheduler`가 아니다.

> 위성 관측·입력 파이프라인·기반모델 릴리스·행정근거가 바뀌었을 때, 기존 결정 중 무엇을
> 계속 믿고 무엇을 재계산·재검수·보류해야 하는지 판정하며 그 근거를 감사 가능하게 남긴다.

연구 이름은 EarthRoute를 유지할 수 있지만, 첫 고객 상품은 **Decision Continuity Audit** 또는
**K-Earth Evidence Pack**이어야 한다. `transfer learning`과 `router`는 그 결과를 더 적은 라벨과
비용으로 만드는 내부 수단이다.

현재 준비도는 비대칭이다.

| 축 | 판정 | 근거 |
|---|---|---|
| 연구 문제 | 중상 | FoldRefresh 자산, 제주 v1→v7 실패 계보, 공식 오름 368·FarmMap ingest, 명시적 kill rule |
| 기술 | 중 | 재현·검색·변화탐지·공식 evidence edge까지 작동; full EarthRoute oracle은 없음 |
| 제품 | 하 | 고객 입력 형식, SLA, 반복 산출물, 기존 업무시간 절감이 아직 측정되지 않음 |
| 시장 | 하 | 실제 구매자 인터뷰·지불의사·유료 파일럿 0; MARC는 검증 파트너 후보이지 구매자로 확인되지 않음 |

따라서 지금의 올바른 문장은 “사업이 된다”가 아니라 **“고단가 감사 서비스로 검증 가능한
사업 가설이 생겼다”**다.

## 이 문서의 provenance와 보정 사항

사용자가 제공한 2026-08-04 Slack export에는 같은 handoff note가 두 번 들어 있고 첫 복사본은
중간에 잘렸으며 UI 문구가 섞여 있었다. 이 문서는 중복을 제거하고, 2026-08-22 현재의
Jeju K-Earth 실측 및 최신 1차 문헌과 합친 canonical 버전이다.

원문에서 그대로 사실로 옮기지 않은 항목은 다음과 같다.

- `아무도 공개하지 않았다` 같은 절대 novelty 문장: 현재 조사 범위의 가설로 낮췄다.
- award·venue·모델 크기: primary source와 릴리스가 확인된 것만 사용한다.
- 원문의 9차원 action space: 첫 실험에서 식별 가능한 세 단계 action으로 축소한다.
- `risk-bounded`: 예측 위험, 모집단 추론, 행정근거 누락을 한 score로 뭉치지 않는다.
- trace asset은 무라벨로 만들 수 있지만, `품질 1%p 이내` 주장은 독립 라벨 없이 검증할 수 없다.
- 비용·일정·가격 숫자는 실측, 논문 보고치, 인터뷰용 가설을 분리한다.

### 원문 아이디어별 가능성 판정

| 원문 아이디어 | 구현 가능성 | 연구 가치 | 사업 가치 | 현재 결정 |
|---|---|---|---|---|
| execution/evidence trace asset | 높음 | 중상 — action·cost·outcome 계약이 있고 독립 검수 subset이 있을 때 | 낮음 — 데이터셋만으로 구매 이유는 약함 | Paper 1·파일럿 실행의 부산물로 먼저 축적 |
| FoldRefresh 기반 `reuse_cached_release` | 높음 | 높음 — 릴리스 변경 아래 모집단 결정 유지 | 중상 — 재계산·재검수 비용이 실제로 큰 고객에서만 | rslearn/K-Earth port를 다음 시스템 baseline으로 사용 |
| 지역·태스크 transfer learning | 중간 | 중상 — 강한 supervised baseline과 OOD split을 이길 때 | 보조 수단 — 제품명으로 팔 수 없음 | 라벨 50% 절감/동일 라벨 5%p/두 번째 태스크 gate로 검증 |
| data+compute+evidence router | 중간 | 높을 수 있으나 novelty 미확정 | 반복 refresh가 생기면 높음 | 세 action oracle headroom을 먼저 측정 |
| LLM/agent teacher | 중하 | 낮음–중간 | 낮음 | deterministic·greedy baseline을 못 이기면 제거 |
| 한국 전역 범용 플랫폼 | 기술적으로 가능 | 현재는 낮음 — 제주 오류를 확대할 위험 | 시장 증거 없음 | 2개 독립 고객의 반복 갱신 전 금지 |
| Decision Continuity Audit/Evidence Pack | 중상 | 중상 — 결론·보류·누락을 함께 측정 | 현재 가장 나은 wedge이나 미검증 | 서비스형 유료 파일럿으로 검증 |

즉 가장 좋은 순서는 `trace를 대량 생성 → router 학습 → 제품 출시`가 아니라
`반복 결정을 찾음 → 작은 Evidence Pack으로 시간·오류 측정 → FoldRefresh/세-action oracle →
반복 수요가 있을 때 router`다.

### FoldRefresh 상태를 두 저장소 사이에서 화해시키기

별도 저장소 `../decision-ready-earth-ai/`에서 다음 로컬 자산을 확인했다.

- `../decision-ready-earth-ai/REPRODUCTION.md`: 84개 실험 스크립트, 입력·결과 JSON,
  preregistration과 claim 경로.
- `paper/verify_foldrefresh_claims.py`: 논문 수치를 결과 JSON에서 재계산하는 guard.
- v1→v1.2 3,996 paired tiles, 1,539-unit inference population과 여러 외부 replay의 추적 자산.
- 등록 초록·AAAI Author Kit build chain·submission 관련 commit.

반면 로컬 submission checklist에는 OpenReview metadata와 clean-environment archive 확인 같은 수동
항목이 남아 있고, 이 작업에서는 공개 OpenReview receipt/ID를 확인하지 않았다. 따라서 상태는
다음처럼 기록한다.

> **FoldRefresh method/evidence chain: locally verified. AAAI-27 submission: attachment/local record에
> 의해 claimed, external receipt 미확인. eo_olmo_earth_project의 rslearn port: 미완료.**

즉 `GOAL.md`의 루프 3은 방법을 새로 발명한다는 뜻이 아니라, 기존 FoldRefresh를 이 저장소의
rslearn 출력·4축 manifest·K-Earth 결정지표에 이식하는 작업이다.

## 하나의 프로그램, 세 층

| 층 | 질문 | 현재 자산 | 고객/연구 출력 |
|---|---|---|---|
| **K-Earth evidence plane** | 이 변화에 대해 무엇을 안전하게 말할 수 있는가? | 오름 368 고정분모, PNU 계약, FarmMap 289,379 polygon, 등급 A–U | claim/evidence manifest, 보류 사유, 조사 후보 |
| **FoldRefresh continuity plane** | 모델 릴리스가 바뀌어도 어떤 통계·결정을 재사용할 수 있는가? | 별도 저장소의 design-based partial refresh | old/new release audit, selective recompute |
| **EarthRoute execution plane** | 어떤 관측·모델·근거·사람검증을 다음에 살 것인가? | 아이디어와 인접문헌; oracle 미구축 | 비용–위험 정책, 반복 refresh |

이 관계는 직렬이다.

`변화 screen → 근거 coverage 감사 → 보류/판정 → 릴리스 변경 감사 → 필요한 부분만 재계산·검수`

K-Earth 없이 EarthRoute를 돌리면 싼 계산을 고를 수는 있어도 결론이 안전한지 모른다.
FoldRefresh 없이 K-Earth만 두면 모델이 바뀔 때 전수 재실행한다. EarthRoute는 둘이 실제로 반복될
때에만 가치가 생긴다.

## 연구 질문

### RQ-E1. 비용을 줄여도 결정이 유지되는가?

전체 pipeline 대신 저비용 action을 썼을 때 pixel/task accuracy만이 아니라 Top-k 조사 목록,
행정구역 집계, risk–coverage, confidence interval이 유지되는지 본다.

- 가설: full run의 30–40% 이상 실제 end-to-end 비용을 줄이며 사전 고정 decision metric 저하를
  1%p 이내로 유지할 수 있다.
- 반증: FLOPs만 줄고 bytes, CPU/GPU-second, p95 latency가 1.5배 이상 좋아지지 않거나,
  decision metric이 허용치를 넘으면 systems claim을 폐기한다.

### RQ-E2. 전이는 언제 실제로 유용한가?

`pretrained model을 썼다`가 아니라 네 종류의 전이를 나눈다.

| 전이 | 질문 | 유료 가치가 생기는 조건 |
|---|---|---|
| 지역 | 제주에서 만든 head/policy가 두 번째 지역에서 유지되는가? | 같은 품질에서 현지 라벨·검수 절감 |
| 태스크 | 변화/분류/회귀 중 다른 task family에서도 선택 정책이 유지되는가? | 고객별 재개발을 줄임 |
| 릴리스 | v1 결과를 v1.2로 옮겨도 이웃·집계·후보가 유지되는가? | 전수 재계산·재검수 회피 |
| 결정 | 한 파트너의 보류 기준이 다른 workflow에서도 risk를 제어하는가? | 반복 가능한 audit template |

전이의 평균 이득만 본 뒤 라벨을 무작위로 늘리지 않는다. scratch·일반 vision/EO·여러 GeoFM의
지역·연도·센서·구름별 전이효과를 먼저 측정하고, 품질 게이트를 통과한 model/release disagreement와
공간 다양성·라벨 비용으로 다음 target label을 고르는 active acquisition을 별도 RQ로 둔다.
상세 목적함수와 baseline은 `K_EVIDENCE_SHIFT_BENCHMARK.md`가 canonical이다.

### RQ-E3. 무엇을 더 관측해야 말할 수 있는가?

원문의 data/compute router를 **evidence acquisition policy**로 확장한다. 높은 모델 confidence가
있어도 공식 경계·시점 근거가 없으면 `원인`을 말하지 않는다. 다음 action은 더 큰 모델이 아니라
지적경계, 행정사건, 과거 항공영상, 사람검수일 수 있다.

### RQ-E4. 근거 누락이 누구를 침묵시키는가?

행정자료의 no-match를 사건 부재로 해석하지 않고, 지역·오름 유형·토지피복·행정구역별
time-aligned coverage와 abstention을 측정한다. 이것이 현재 K-Earth의 가장 독자적인 질문이다.

## 최소 action space

원문의

`provider × sensor × time × resolution × model × patch × depth × halo × stop`

를 한 번에 열면 어느 축이 이득을 만들었는지 식별할 수 없다. 첫 oracle은 다음 세 행동만 둔다.

| action | 실행 | 용도 |
|---|---|---|
| `reuse` | 이전 release output + FoldRefresh correction | 안정적인 window의 기본값 |
| `cheap_refresh` | S2 중심 짧은 시간축 + 작은 모델 | 불확실하지만 full run 전 단계 |
| `escalate` | S1+S2 긴 시간축 + Base + RGB/전문가 검수 | 고위험·불일치 window |

그 다음 ablation에서만 `request_official_record`와 `request_field_review`를 추가한다. provider,
resolution, patch, depth, halo는 oracle headroom이 확인된 뒤 하나씩 연다.

초기 정책은 RL/LLM이 아니라 deterministic cost-aware baseline이어야 한다.

1. always-full
2. always-cheap
3. random budget-matched
4. quality-only(SCL/nodata)
5. uncertainty-only
6. release-drift-only
7. quality × drift × decision-sensitivity
8. learned router
9. agent-teacher는 위 1–8보다 좋아질 때만 유지

## `risk-bounded`의 정확한 뜻

한 개의 confidence threshold로 부르면 안 된다.

| 보증 | estimand | 필요한 검증 |
|---|---|---|
| 선택적 예측 risk | 시스템이 말한 후보 중 사람 기준 오답률 | 사전 고정 사람 라벨, risk–coverage, 가능하면 conformal/selective guarantee |
| 모집단 통계 validity | 전체 AOI의 면적·비율·회귀계수 | 확률표본 ground truth + PPI/설계기반 CI |
| evidence coverage | 공식 원인자료가 시공간적으로 조회 가능한 비율 | source별 denominator, snapshot/date, missingness audit |
| spatial consistency | seam·인접타일·구역 집계 안정성 | seam error, spatial block bootstrap, worst-region metric |

EarthRoute의 제약식은 개념적으로 다음과 같다.

`min(bytes, CPU-s, GPU-s, latency, human-minutes)`  
`subject to selective risk ≤ ε, CI coverage ≥ 1-α, evidence coverage reported, seam error ≤ δ`

`evidence coverage reported`는 coverage가 높다는 뜻이 아니라, 낮을 때 그 사실과 침묵 편향을
숨기지 않는다는 뜻이다.

## 전이 실험 설계

### 비교할 적응법

1. handcrafted/non-GFM supervised baseline
2. compute/data-matched U-Net·ViT scratch와 generic vision pretrained baseline
3. frozen OlmoEarth encoder + linear probe
4. OlmoEarth 외 Prithvi와 CROMA/TerraMind 계열
5. parameter-efficient adaptation(adapter/LoRA 계열)
6. full fine-tuning
7. no-target-tuning zero-shot/few-shot prototype where applicable

### 최소 평가 격자

| 축 | 최소 수준 |
|---|---|
| 태스크군 | 분류/검색 1 + segmentation 또는 regression 1 |
| 지역 | 제주 + 사전 고정 두 번째 지역 |
| 릴리스 | OlmoEarth v1 + v1.2 |
| 라벨 예산 | 10, 25, 50, 100 + full |
| 시간 | train 이전/이후의 temporal holdout |

평균 점수 외에 rare-class recall, worst-region, calibration, risk–coverage, label-hours,
download/materialize/GPU/human-review cost를 보고한다. target region을 보고 hyperparameter를 고르면
그 셀은 zero-shot transfer가 아니다.

### promotion gate

- 같은 성능에서 현지 라벨 50% 절감, 또는 같은 라벨에서 강한 baseline보다 5%p 개선.
- 두 태스크군 × 두 지역에서 방향이 반복됨.
- 새 release에서 calibration·decision metric이 사전 허용범위 안.
- 실패하면 `foundation-model transfer`를 지우고 task-specific adaptation 결과로 낮춘다.

현재 완도→제주 양식장 9/9 상위 4%는 좋은 smoke test지만, 양성 표본만 있고 OSM 표본편향과
유형 차이가 있어 사업 성능주장에는 사용할 수 없다.

## 경쟁 경계: 만들지 않을 것

| 이미 있는 층 | 공식 근거 | 정면 경쟁을 피하는 이유 |
|---|---|---|
| fine-tuning·annotation·inference platform | [OlmoEarth Studio 문서](https://docs.olmoearth.allenai.org/) | Ai2가 dataset→fine-tune→deploy를 이미 제공 |
| 대륙 규모 실행·provider index·향후 alerts/agents/embedding | [OlmoEarth infrastructure](https://allenai.org/blog/olmoearth-infrastructure) | 범용 scheduler/agent는 Ai2 roadmap과 직접 중복 |
| 전지구 분석용 embedding product | [AlphaEarth Earth Engine dataset](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) | plain embedding 판매의 차별성 부족 |
| 일반 변화탐지 feed | [Planet Analytics](https://docs.planet.com/develop/apis/analytics/) | 변화 위치를 찾는 것 자체는 상품화됨 |
| 영상 주문·처리 marketplace | [UP42 platform](https://up42.com/platform/platform-overview) | provider procurement/processing orchestration은 기존 시장 |
| no-code custom change model | [Picterra Forge](https://picterra.ai/technology/picterra-forge/) | change-model UI는 moat가 아님 |

따라서 다음은 하지 않는다.

- 또 하나의 EO workflow scheduler, tile viewer, embedding API.
- `OlmoEarth에 agent를 붙였다`를 기여로 삼기.
- 불법개발을 자동 확정하거나 공무원의 법적 판단을 대체하기.
- 제주 네 사례를 전국 성능으로 표현하기.
- model confidence를 공식 원인근거로 승격하기.

## 사업 가설

### 가장 현실적인 첫 wedge

1순위 가설은 **사후환경영향조사·환경평가 evidence assistant**다. 관련 법령은 착공 후 조사와
진행현황·공정률·승인/변경일 등을 포함한 반복 보고 workflow를 둔다. 이는 시장 수요가 확인됐다는
뜻은 아니지만, 새 예산항목보다 기존 용역의 증거수집·검수시간을 줄이는 방식으로 들어갈 수 있다는
근거다. 법적 판단과 보고서 책임은 등록된 평가업자에게 남긴다.

- [환경영향평가법 제35–37조](https://www.law.go.kr/lsLinkCommonInfo.do?chrClsCd=010202&lsJoLnkSeq=1029883855)
- [사후환경영향조사 결과 작성 방법](https://www.law.go.kr/LSW/flDownload.do?bylClsCd=200201&flNm=%5B%EB%B3%84%ED%91%9C+9%EC%9D%982%5D+%EC%82%AC%ED%9B%84%ED%99%98%EA%B2%BD%EC%98%81%ED%96%A5%EC%A1%B0%EC%82%AC%EA%B2%B0%EA%B3%BC%EC%9D%98+%EC%9E%91%EC%84%B1+%EB%B0%A9%EB%B2%95%28%EC%A0%9C40%EC%A1%B0%EC%9D%982+%EA%B4%80%EB%A0%A8%29&flSeq=150235727)

### 고객–업무–파일럿 표

| 우선 | 구매자 가설 | 반복 업무 | 첫 전달물 | 지불 신호 |
|---:|---|---|---|---|
| 1 | 환경영향평가 대행사·사업자 환경팀 | 사업구역 진행·토지변화·협의근거를 분기/연간 취합 | `Post-EIA Evidence Pack` + source/date/geometry manifest | 같은 구역 두 번째 refresh 유료 전환 |
| 2 | EO/GIS 컨설팅사·공공 지도 운영팀 | 모델·센서·합성법 변경 때 결과 회귀와 재계산 범위 판단 | `GeoFM Release Readiness Audit` | audit fee 또는 LOI |
| 3 | 농업·산림·연안 연구/보전 조직 | 적은 라벨로 지역 map을 만들고 반복 갱신 | `Local Adaptation Sprint` | 실제 라벨 제공 + 검수시간/결정 변화 기록 |
| 4 | 대규모 EO 플랫폼 | 반복 추론의 imagery/GPU/검수 비용 | EarthRoute cost–risk curve/OEM | 국가급 workload에서 절감 재현 |

MARC형 조직은 3번의 **현장 검증 파트너 후보**다. 예산권자와 반복 업무가 확인되기 전에는
고객이라고 부르지 않는다.

### 상품 사다리

1. **10일 Release Audit** — 한 AOI·한 task·한 model transition.
2. **6주 Evidence Pilot** — 한 사업구역, 공식근거 연결, domain expert review.
3. **분기 Continuity Refresh** — 같은 schema로 반복 업데이트.
4. **Control plane/OEM** — 두 독립 고객이 두 번 이상 갱신한 뒤에만.

가격은 아직 사실이 아니다. 인터뷰용 anchor로만 `10일 500만–1,500만원`, `6주 2,000만–
5,000만원`, `분기 refresh 300만–1,000만원 + 상용영상/컴퓨트 실비`를 시험할 수 있다. 가격단위는
pixel/km²보다 **결정 주기와 줄인 검수시간**이 낫다. 고객 절감액이 계약금의 3배 미만이면 중단한다.

### 첫 Evidence Pack의 파일 계약

- AOI/필지/사업구역 geometry와 CRS.
- before/after observation ID, 실제 촬영일, 합성 recipe, cloud/nodata 품질.
- model release, weight hash, code commit, head, threshold.
- 연결한 공식 source, snapshot/retrieval date, join key, coverage 계약.
- 변화 footprint와 evidence edge, 등급 A–U, 원인 주장 허용 범위.
- 사람 검수 chip, 판정자/시점, disagreement.
- `confirm / investigate / abstain` 결정과 이유.
- 이전 refresh 대비 바뀐 후보·집계·비용.

이 산출물은 `지도 한 장`이 아니라 감사 가능한 dossier다.

## 논문 프로그램 보정

원문의 “trace dataset 먼저, router 다음”은 논리적으로 맞지만 현재 프로젝트의 가장 강한 실측을
반영해 순서를 조정한다.

### Paper 0 — FoldRefresh

별도 저장소의 release-statistic partial refresh. 이 프로젝트에서는 새 논문처럼 다시 쓰지 않고
K-Earth/rslearn integration baseline으로 쓴다. 동료심사·외부 제출 상태는 확인된 범위대로 표기한다.

### Paper 1 — K-Earth Selective Evidence

가제: *Selective Change Detection under Incomplete Administrative Evidence*.

- 고정분모 오름 368, official source coverage, missingness/abstention bias.
- evidence-source ablation:
  `model+OSM → +PNU → +time-aligned state → +administrative event → +air/field review`.
- 지표: time-aligned coverage, risk–coverage, 지역별 침묵률, 사람검수 오류, source 추가 전후 결정.
- 원인근거가 계속 10% 미만이면 원인 분류를 하지 않는 현재 gate를 유지.

### Paper 2 — EarthRoute

Paper 1의 evidence risk와 FoldRefresh의 release action을 실행정책으로 확장한다.

- 먼저 execution/evidence trace schema와 oracle headroom을 공개.
- oracle이 실제 end-to-end cost를 30–40% 줄이지 못하면 router 논문을 중단.
- learned router는 random/quality/uncertainty/drift deterministic baselines를 이긴 뒤에만.
- agent는 greedy/heuristic보다 이득이 있을 때 teacher로만 유지하고 production에는 compile된 정책 사용.

무라벨 trace만으로는 quality claim을 할 수 없으므로, Paper 1의 확률표본 또는 독립 검수 subset을
Paper 2의 ground truth로 재사용한다.

## 90일 증명 순서

### 0–30일: 문제와 baseline

- 환경영향평가업체 3곳, EO/GIS 업체 2곳, 공공 연구기관 2곳, 보전조직 2곳 문제 인터뷰.
- 실제 산출물 형식, 수작업 시간, FP/FN 비용, 갱신주기, 예산권자를 기록.
- 제주 현재 4사이트 데모가 아니라 사전 고정 최소 100후보에서 evidence workflow를 실행.
- EarthShift/PANGAEA 방식의 spatial holdout과 비-GFM baseline을 고정.

### 31–60일: 유료 가능성·전이

- 한 실제 사업구역에서 기존 GIS workflow와 Evidence Pack을 blind time study로 비교.
- 검수시간 30% 이상 절감, 잘못된 원인 단정 0, source lineage 100%를 gate로 둔다.
- 두 번째 지역/태스크에서 frozen/PEFT/full FT의 label-efficiency를 비교.

### 61–90일: 반복성·EarthRoute oracle

- 같은 고객/AOI의 두 번째 refresh를 실행하고 유료 전환 여부 확인.
- `reuse / cheap_refresh / escalate` oracle cost–risk curve 작성.
- oracle headroom과 transfer gate를 통과할 때만 learned router와 논문 2를 연다.

## 중단·축소 조건

- 인터뷰 10회 뒤에도 반복 결정, 예산권자, 실패 비용이 특정되지 않는다.
- official spatiotemporal evidence coverage가 10% 미만인데 고객은 자동 원인확정만 원한다.
- 기존 GIS/Planet/GEE/ArcGIS workflow와 비교해 검수시간을 30% 이상 줄이지 못한다.
- 두 태스크군·두 지역에서 transfer 방향이 반복되지 않는다.
- oracle이 full run 대비 실제 end-to-end 비용 30–40%를 줄이지 못한다.
- learned router가 oracle saving의 70%를 회수하지 못한다.
- agent가 greedy/uncertainty heuristic을 이기지 못한다.
- 연간 고객 절감액이 계약금의 3배 미만이다.
- 데이터·모델 license가 유료 사용 또는 파생 산출물 전달을 허용하지 않는다.

중단은 실패가 아니라 범위 판정이다. router가 죽어도 K-Earth Evidence Pack과 selective evidence
paper는 남을 수 있고, 사업이 죽어도 release/evidence audit 연구는 남는다.

## 라이선스·책임 경계

- [OlmoEarth Artifact License](https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE)는
  artifacts/derivatives의 사용·수정·배포를 허용하지만 군사·방위·인간감시/치안, 일부 추출 활동을
  금지하고 downstream 배포에도 제한을 승계한다.
- [OlmoEarth Platform EULA](https://olmoearth.allenai.org/eula)는 Platform에서 만든 derivative
  model·prediction의 상업화에 Ai2 사전 서면승인을 요구한다.
- 따라서 local open artifact 경로와 hosted Platform 경로의 조건을 같은 것으로 취급하지 않는다.
- 공공데이터도 source별 이용조건·개인정보·재배포 범위를 manifest에 둔다.
- Evidence Pack은 법적 허가판정, 환경영향평가서, 현장조사를 대체하지 않는다.

첫 유료 파일럿 전에는 사용 경로별 license와 계약을 법률 전문가에게 확인한다.

## 주장 검증표

| 주장 | 상태 | 근거/다음 확인 |
|---|---|---|
| 제주 official oreum 368 고정분모 | 로컬 실측 | `artifacts/external_data/kearth_oreum_v1/` |
| FarmMap 289,379 polygons ingest | 로컬 실측 | `artifacts/external_data/kearth_public_ingest_v1/` |
| A/B급 원인근거 0/368 | 로컬 실측, 현재 snapshot에 한정 | coverage 확대 후 재계산 |
| FoldRefresh 수치·재현 chain | 별도 로컬 자산 확인 | `../decision-ready-earth-ai/REPRODUCTION.md` |
| FoldRefresh AAAI-27 최종 제출 | 사용자/로컬 기록, 외부 receipt 미확인 | OpenReview submission ID/receipt 필요 |
| EarthRoute가 최초의 joint router | 미확정 검색 가설 | 체계적 검색·제외 로그 필요 |
| transfer가 사업비를 줄임 | 미검증 | 두 태스크·두 지역 label/time cost 실험 |
| EIA 업체가 지불함 | 미검증 | 인터뷰·LOI·paid pilot 중 하나 필요 |
| generic scheduler/embedding/change feed가 혼잡 | 공식 제품으로 확인 | Ai2, AlphaEarth, Planet, UP42, Picterra 링크 |

## 한 문장 포지셔닝

연구:

> **EarthRoute chooses which observation, model, administrative evidence, or human validation to
> acquire next so that a changing map can support a valid decision at bounded cost.**

사업:

> **모델과 데이터가 바뀔 때 무엇을 다시 믿어도 되는지 감사하고, 부족한 근거는 자동으로
> 보류하며, 필요한 부분만 다시 계산·검수하는 한국형 Earth Intelligence 운영층.**
