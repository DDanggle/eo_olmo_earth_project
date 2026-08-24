# K-ALIGN CVPR main 준비도 감사와 실행 가능한 실험 설계

작성: 2026-08-23  
상태: **문헌·자산 감사 결과. 새 방법의 성능 결과는 아직 없음**  
기준 문서: `KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`, `K_ALIGN_PROGRAM_NOTE.md`,
`PAPER_READING_LIST.md`

## 0. 약점부터 — 현재 그대로는 CVPR main 논문이 아니다

현재 저장소에는 강한 **문제 증거**와 강한 **감사 엔지니어링**이 있다. 그러나 새 학습 방법,
한국 공공데이터가 EO 표현을 개선했다는 결과, 독립 task label은 없다.

- 제주 216 site-year의 OlmoEarth v1↔v1.2 실험은 raw cross-release R@1이 양방향 0이고,
  calibration-only affine ridge도 0.6973/0.6089라는 실패를 봉인했다. 이 결과는 좋은 Figure 1이지만
  label 0·제주 단일 grid이므로 task accuracy나 한국 일반화를 말하지 못한다.
- 공공 API 수집은 463/463 HTTP 성공, 의미상 성공 456건까지 닫혔다. 그러나 14후보에서 exact PNU
  건축사건 1건, **변화구간 시간정렬 0건**, EIA 직접 근거 0건이다. 오름 368의 공식 원인근거도
  0건이다.
- 현재 BuildingHUB 레코드에는 허가·착공·사용승인·레코드 생성일은 있으나, prospective input에
  필요한 **실제 공개시각 `published_time`은 없다**. `retrieved_time`이나 `created_date`를 공개시각으로
  바꾸어 쓰면 안 된다.
- `code/`에는 Olmo release audit·공공데이터 canonicalization·dashboard·leakage 검증은 있지만,
  공통 multi-model encoder interface, student/bus 학습, public dense-task trainer, compression evaluator는
  아직 없다.
- 첨부 노트의 26편은 이번 감사 전 대부분 검색 snippet 수준이었다. 아래 핵심 문헌은 공식
  proceedings/arXiv/저자 repo의 초록과 관련 본문을 확인했지만, **우리 환경 재현은 별도**다.

따라서 현재의 정직한 판정은 다음과 같다.

| 후보 | 지금의 CVPR main 준비도 | 잠재력 | 결정 |
|---|---|---|---|
| frozen EO release/family 호환성 | 낮음–중간 | 중간–높음 | **이번 주기 1순위**. 단 BCT 변형을 넘어야 함 |
| 한국 public-context privileged distillation | 매우 낮음 | 높음 | 데이터 gate 전에는 방법 학습 금지. 다음 주기 유력 |
| 압축·부분 refresh | 낮음 | 보조축 | NeuCo-Bench/NEC/ESD를 이기는 독립 기여가 아니면 표 한 칸 |
| 온보드·로봇·simulation | 없음 | 장기 | 실제 hardware/trajectory/task가 생길 때 후속 |

### 거시적 재정렬 — 무엇을 만드는 논문인가

이 프로젝트를 “EO adapter 하나를 더 만드는 논문”으로 두지 않는다. 연구 프로그램은 아래 세
질문을 순서대로 닫는다.

1. **Necessity — 정말 갱신 문제가 존재하는가?** archive 크기 `N`, query volume `Q`, raw raster
   접근성에 따라 full re-embedding, dual-index, compatibility adapter의 비용 경계를 먼저 그린다.
2. **Predictability — 언제 사후 정렬할 수 있는가?** architecture·band·GSD·time recipe 변화와
   layer CKA/probe/geometry가 실제 held-out alignment 성공을 예측하는지 특성화한다.
3. **Intervention — 어느 구간에 새 방법이 필요한가?** old gallery의 quantizer/codebook을 고정한
   채 task utility와 cross-version retrieval을 함께 지켜야 하는 구간에서만 quantizer-aware
   compatibility를 제안한다.

따라서 논문의 뼈대는 **characterization**, 방법의 날은 **fixed-quantizer constraint**, 한국은
**실제 release failure와 공공 provenance stress test**다. 공공데이터 표현강화는 이 뼈대의 필수
구성요소가 아니라 데이터 gate를 통과했을 때 여는 독립 후속축이다.

## 1. 이번 문헌 감사가 기존 계획에서 고친 것

### 1.1 `published_time` 하나만으로는 novelty가 아니다

GeoLink는 OSM을 EO pretraining과 downstream에 넣고, MMEarth·OmniSat·SatMIP·Galileo는
멀티모달/metadata supervision을 이용해 표현을 강화한다. WildSAT은 수백만 geotagged wildlife
관찰을 위성 표현 학습에 쓴다. Auxiliary Modality Learning은 train 때만 보조모달을 쓰고 test에는
더 적은 모달을 쓰는 문제를 이미 일반화했다.

남는 빈칸은 단순한 `한국 API + EO`가 아니라 다음 세 조건의 **결합**이다.

1. event/observed/published/retrieved time이 분리되고 미래정보가 기계적으로 차단된다.
2. 자연 누락·지연·충돌 아래서 public context가 없는 EO-only student에도 task utility가 전달된다.
3. teacher/model release가 바뀌어도 old gallery와 old task head가 유지된다.

이 셋 중 하나만 보여주면 기존 multimodal learning, auxiliary-modality distillation, embedding
compatibility의 응용에 가깝다.

### 1.2 기존 `derivability` hard exclusion은 방향이 잘못됐다

기존 계약은 EO-only probe가 public token을 잘 예측하면 그 source를 teacher에서 제외했다. 하지만
EO-only student가 test 때 context 없이 그 지식을 쓰려면, public signal 중 일부는 EO에서
**회복 가능해야 한다**. 완전히 회복 불가능한 행정정보는 inference-time residual에는 유용할 수 있어도
EO-only embedding으로 증류될 수 있는 정보에 상한이 있다. 반대로 토지피복처럼 매우 잘 회복되는
source는 새 정보라기보다 weak/pseudo-label supervision일 수 있지만 label efficiency는 높일 수 있다.

source 자격은 하나의 `D(source)`가 아니라 다음 세 수치로 판정해야 한다.

```text
R_source = EO-only 모델이 public token을 회복하는 정도
V_source = 독립 task Y에서 EO+context가 EO-only보다 주는 조건부 가치
T_source = context를 train 때만 본 EO-only student가 no-context student보다 얻는 전이 이득
```

| 관찰 | 해석 | 논문에서 허용할 역할 |
|---|---|---|
| `R` 높음, `V` 낮음 | 중복/shortcut | sanity·pseudo-label baseline |
| `R` 낮음, `V` 높음 | EO-only로 옮기기 어려운 독점 정보 | inference residual/abstention |
| `R` 중간, `V` 높음, `T` 양수 | 실제 privileged transfer 후보 | `E_repr` teacher |
| `T`가 shuffle/time-shift에서도 유지 | 위치·연도 shortcut 또는 leak | cell 무효 |

즉 `R`은 제외 기준이 아니라 **기전 진단**이다. `E_repr`은 독립 label과 `T_source`가 없으면 열지 않는다.

### 1.3 E-07의 1 KB 해석을 바로잡는다

Embedding-Only Uplink 논문은 embedding+metadata를 지상에서 궤도로 보내는 구성을 평가한다. 그러나
본문의 약 598–690 B/query는 주로 내려오는 JSON telemetry 크기이고, hint gallery 업링크 비용은
`N_hints × D × bytes_per_value`로 별도 계산한다. 따라서 다음은 구분한다.

- 허용: embedding-only link와 onboard retrieval은 cache/representation 효율의 운영 동기다.
- 금지: 논문이 “전체 gallery backfill이 1 KB 예산 때문에 물리적으로 불가능함”을 증명했다고 쓰는 것.
- 우리 실험: bandwidth를 가정한 simulation으로만 보고하고, satellite hardware 실측이라 부르지 않는다.

### 1.4 backward compatibility 선행은 예상보다 훨씬 붐빈다

BCT·FCT·LCE뿐 아니라 UniBCT(IJCAI 2022), BiCT(raw gallery image 없이 old embedding을 갱신),
Darwinian Model Upgrades(AAAI 2023, selective compatibility/gallery evolution), AdvBCT(CVPR 2023),
BT²(ICCV 2023), multi-version BC-Aligner(KDD 2022), hyperbolic backward compatibility(ICML 2025),
cross-modal XBT(ICCV 2025), online backfilling(WACV 2025)까지 있다. 따라서 raw raster가 없다는
제약이나 gallery를 일부 갱신한다는 사실만으로도 novelty가 되지 않는다.

살릴 수 있는 경계는 더 좁다. 1차 검색에서는 compatibility와 learned/PQ quantization을 각각
다룬 문헌은 많았지만, **frozen third-party EO release의 new query를 old PQ codebook/code에 직접
호환시키는 결합 문제**는 명시적으로 확인하지 못했다. 이것은 부재 증명이 아니므로 systematic
search와 BiCT/DMU/online-backfill의 정확한 기능 비교가 A0에 선행한다.

> **학습 파이프라인을 통제하지 못하는 frozen third-party EO releases에 대해, EO 특유의 band·GSD·
> time-grid 변화와 압축 gallery를 함께 다루면서, 새 모델 task utility와 old-gallery retrieval을
> 동시에 보존하는 post-hoc compatible distillation.**

이 문장도 아래 full matrix에서 여러 family/release와 task를 통과했을 때만 허용한다.

## 2. 1순위 논문 — Predictable EO Upgrade under a Fixed Quantizer

### 2.1 연구 질문

> **구·신 frozen EO encoder의 사후 정렬 가능성을 모델·입력 변화의 측정치로 예측할 수 있는가?
> 그리고 old quantizer/codebook이 고정된 경우, 새 모델의 task utility와 기존 compressed gallery
> 검색을 동시에 보존할 수 있는가?**

가제:

> **EarthUpgrade: Predicting and Preserving Compatibility under Frozen EO Quantizers**

한국은 이 논문의 유일한 평가셋이 아니라 다음 역할을 맡는다.

- 실제 Olmo v1→v1.2 release drift를 발견한 motivation/failure atlas.
- 행정경계·구름·누락 context가 있는 hard stress test.
- 공개 benchmark에서 검증한 방법을 한국 asynchronous setting에 외삽하는 사례.

### 2.2 Day 0 — 방법보다 먼저 필요성을 판정한다

첫 그림은 accuracy가 아니라 `N×Q` 운영 경계다.

```text
full re-embed = raw access + N × (read + new encoder + compact write)
dual index    = second-gallery storage + Q × dual retrieval/merge
adapter       = anchor extraction + adapter training + Q × adapter overhead
```

현재 216건 실행의 v1.2 37.5분은 49.1 GiB full token GeoTIFF 쓰기를 포함한다. 이를 production의
compact re-embedding 비용으로 선형 외삽하면 안 된다. pooled/16×16 FP16만 쓰는 compact path를
별도로 측정해 `N={10³,10⁴,10⁵,10⁶,10⁷}`와 실제 query-volume 구간에서 비용곡선을 만든다.

- 현실적인 `N`에서 compact full re-embedding이 가장 싸고 raw raster도 계속 접근 가능하면
  compatibility 논문을 중단한다.
- raw raster가 삭제·라이선스 만료·기관 반출제한으로 접근 불가하다면 비용 문제가 아니라
  **irreversible archive constraint**로 정의하고 그 증거를 데이터 계약에 남긴다.
- adapter가 유리한 영역이 실제 archive 규모와 query volume 안에 있을 때만 다음 절로 간다.

### 2.3 비교할 release 축

| family | old/new | 공정한 paired view | 역할 |
|---|---|---|---|
| OlmoEarth | v1 / v1.2 | 동일 S2 L2A 12-band acquisition·AOI·time recipe | 핵심 release pair |
| Prithvi-EO | 1.0 / 2.0 | 동일 HLS 6-band·30 m·공통 timestep view | 독립 release-family 반복 |
| TerraMind | v1 Base | 동일 S2 L2A 12-band의 cross-family ceiling | release pair가 아니므로 보조 |

Prithvi 1.0과 2.0은 모두 HLS 6 band·30 m를 쓸 수 있으므로 2×2 release-family 반복 가능성은 있다.
Olmo와 Prithvi의 raw 점수를 같은 입력 표에서 직접 우열로 읽지 않고 **family 내부 release retention**을
먼저 계산한다.

### 2.4 데이터와 split

1. **방법 선택용 anchor**: 새 지역 3곳, 2,048–10,000 site-years. 기존 제주 54 spatial key와
   공개된 smoke 위치는 전부 제외한다.
2. **공개 downstream task**: PANGAEA/NeuCo-Bench에서 최소 3개. 권장 조합은 HLS Burn Scars
   (HLS), PASTIS-R 또는 DynamicEarthNet(다중시계열), Sen1Floods11 또는 SpaceNet7(조밀예측/변화).
3. **split**: parcel/AOI/event group + spatial buffer, 과거 train→다음 연도 val→최신 test.
   같은 event의 before/after는 한 split에 둔다.
4. **method lock**: 한국/공개 calibration에서 architecture·loss·dimension을 고정하고, untouched
   task test는 한 번만 연다. 이미 본 제주 sealed 64는 Figure 1 외 사용 금지다.

### 2.5 characterization과 방법·baseline matrix

먼저 각 release pair에서 architecture·band order·reflectance scaling·normalization·timestamp·pooling을
재감사한다. 채널 순열이나 preprocessing mismatch가 R@1=0을 설명하면 model-drift 주장을 중단하고
재현성 결함으로 보고한다.

그 다음 다음 특성값이 held-out pair의 alignment 가능성을 예측하는지 검증한다.

- layer별 linear/row-normalized CKA, distance-rank retention, effective rank.
- old/new linear-probe transfer와 affine residual spectrum.
- 입력·architecture 변화의 크기와 quantization distortion.
- predictor는 일부 release pair에서 fit하고 보지 않은 family/release에서 평가한다.

“언제 정렬 가능한가”가 새 family에서 예측되지 않으면 characterization 기여도 성립하지 않는다.

| 종류 | 필수 비교군 |
|---|---|
| 재계산 ceiling | old/old, new/new, full gallery backfill, dual-index |
| 단순 post-hoc | identity, mean shift, orthogonal Procrustes, affine ridge, 2-layer MLP |
| compatibility | BCT, UniBCT, LCE, AdvBCT, BT², FCT oracle, BiCT, DMU, online backfill, BC-Aligner, hyperbolic BCRL |
| distillation | single teacher, AM-RADIO-style multi-teacher, proposed relational/token bus |
| 압축 | FP32, PCA64+int8, OPQ/PQ, Matryoshka, NEC/NeuCo-Bench-compatible learned compression |
| quantizer-aware | frozen old codebook, decode-then-align, align-then-quantize, joint quantization-aware adapter |
| task baseline | frozen linear/UPerNet, PEFT/SLR adapter, full fine-tuning, scratch U-Net/ViT |

`BCT-surrogate` 하나로 기존 compatibility 문헌을 대표하지 않는다. 원 방법을 적용할 수 없는 경우
“not applicable”의 정확한 가정을 표로 쓰고, 가능한 post-hoc 방법은 실제로 실행한다.

### 2.6 primary endpoints와 사전 gate

| 질문 | primary endpoint | GO | KILL |
|---|---|---|---|
| 필요한가 | `N×Q`별 GPU-hour·I/O·storage·latency | 현실 archive 구간에서 adapter Pareto 영역 존재 | compact re-embed가 항상 우세 |
| 호환되는가 | gallery-size별 new-query→old-gallery mAP/R@1 | native new/new의 ≥95%, cluster lower bound도 통과 | identity/affine/강한 post-hoc와 같음 |
| 예측 가능한가 | held-out release의 alignment success/error 예측 | 새 family/release에서 사전 오차한계 통과 | 같은 pair 설명에만 성공 |
| 새 지식이 남는가 | old head/동일 task decoder retention | new-native 대비 ≤1%p 손실 | task −2%p 초과 |
| 여러 release에 일반화하는가 | pair×task macro와 worst pair | 2 family release pair·3 task 중 ≥2/3 같은 방향 | Olmo 한 pair에서만 성립 |
| 압축에서도 남는가 | utility–bytes–latency Pareto | PCA64+int8/PQ보다 Pareto 우위 | float32 대비 절감만 존재 |
| 비용이 줄었는가 | backfilled bytes, GPU-hour, wall, query latency | utility gate와 비용 gate 동시 통과 | 비용만 개선 |

5 seeds는 학습 head/student에 사용한다. 위치·event가 통계 단위이며 pixel/token을 독립 표본으로
세지 않는다. CI는 site/event spatial bootstrap 또는 location-cluster jackknife로 낸다.
retrieval은 216개 gallery 한 점으로 끝내지 않고 고정 distractor pool로 `10³–10⁶+` 규모 곡선을
보고한다. 실제 embedding을 만들 수 없는 규모의 extrapolation은 별도로 표시하고 headline에 쓰지 않는다.

## 3. 2순위 논문 — Korea Public-Context Privileged Transfer

### 3.1 연구 질문

> **공개시각·coverage·geometry가 명시된 한국 행정/환경 기록을 train-only privileged signal로 쓸 때,
> context가 없는 EO-only student의 labels-to-target과 OOD worst-region 성능이 실제로 좋아지는가?**

이 논문은 잠재 novelty가 더 높지만 현재 데이터로는 실행 불가다. 후보-first로 14개 변화를 찾은 뒤
행정기록을 붙이는 현재 경로는 positive coverage가 너무 낮다. **event-first sampling**으로 바꾼다.

이 축은 현재 형태로는 CVPR vision-method 주기여보다 **multimodal data/evaluation methodology**에
가깝다. `R/V/T`가 특정 vision representation의 새로운 학습 원리와 task gain을 뒷받침하지 못하면
CVPR에 억지로 합치지 않고 NeurIPS Datasets & Benchmarks, ISPRS JPRS/TGRS 계열로 venue를 바꾼다.
`published_time`을 증명할 수 있는 source가 없으면 prospective claim은 실험으로 보완할 수 없으므로
즉시 retrospective auxiliary/inference residual로 격하한다.

```text
Building/EIA event universe
  → PNU/official geometry가 있고 event time이 유효한 record 선별
  → event 전후의 canonical EO acquisition 수집
  → 같은 지역·연도·필지유형의 matched control
  → source와 독립인 항공/사람 label로 task 판정
```

### 3.2 데이터를 열기 위한 gate

- event date와 `published_time`을 분리할 수 있는 source. 공개시각이 없으면 retrospective auxiliary로만.
- 3지역·2 event type에서 최소 수천 건의 spatially distinct record. 5% coverage는 feasibility 신호일
  뿐 main claim을 지탱하는 충분조건이 아니다.
- source와 독립인 task label. BuildingHUB event를 feature와 building-change label에 동시에 사용 금지.
- context-only, location+year-only, shape-only, missingness-only, region×year shuffle, ±1년 time-shift.
- `R_source`, `V_source`, `T_source`를 모두 보고하고, `T_source`가 독립 label에서 양수일 때만
  “EO-only representation transfer”를 주장한다.

### 3.3 지금 당장 가능한 데이터 P0

1. BuildingHUB 8,794행을 candidate가 아니라 **event universe**로 재구성한다.
2. exact PNU·event date·geometry를 가진 unique events와 중복/정정 레코드를 분리한다.
3. prospective 사용에 필요한 publication timestamp 필드가 실제 endpoint에 존재하는지 확인한다.
4. 200 events를 층화 추출해 S2 before/after availability, cloud, visible footprint, NGII lead time을 잰다.
5. 이 P0에서 유효 event가 100 미만이거나 독립 판독 agreement가 0.60 미만이면 학습하지 않는다.

이 단계는 GPU보다 데이터 정의가 병목이다.

## 4. 엔지니어링 준비도

### 4.1 이미 충분히 강한 부분

- credential redaction, request hash, raw SHA-256, HTTP/API 의미 성공 분리, pagination.
- exact-input/release/checkpoint/runtime/GPU/code provenance와 fail-closed marker.
- spatial split 봉인, pre-analysis lock, calibration/test 누출 방지.
- 공공 source canonicalization, PNU join, coverage/no-match/conflict 보존.
- 122 pass + 2 optional skip으로 봉인됐던 release-audit test chain.

이것은 CVPR 실험의 재현성 기반으로 충분하다. 다만 그 자체가 새 vision method는 아니다.

### 4.2 아직 없는 핵심 ML stack

```text
code/k_align/
  schemas.py              # site/event/input/public-cutoff contract
  data/canonical.py       # 한 번 수집한 canonical tensors
  data/model_views.py     # Olmo/Prithvi/TerraMind band·GSD·time view
  models/protocol.py      # encode() 공통 interface
  models/{olmo,prithvi,terramind}.py
  models/student_bus.py
  losses/{teacher,relation,compat,context}.py
  compression/{pca,pq,matryoshka,neural}.py
  tasks/pangaea.py
  train.py
  evaluate.py
  statistics.py
  run_matrix.py
configs/k_align/
```

추가로 필요한 것은 environment lock을 모델별로 분리하고, acquisition/input hash는 공유하되 model
view hash를 별도로 저장하는 것이다. 같은 원본을 모델마다 다시 다운로드하지 않는다.

### 4.3 저장·실행 설계

현재 full audit의 216 site-years는 입력 56.68 GB, 두 release 출력 105.59 GB였다. 이를 10,000으로
그대로 선형 확장하면 입력 약 2.62 TB, 두 release full token raster 약 4.89 TB, v1/v1.2 추론만 약
77 GPU-wall-hour가 된다. **full 256×256×768 raster를 teacher마다 저장하면 안 된다.**

권장 feature store:

- pooled 768-d fp16 + 고정 16×16 token lattice fp16.
- 16×16×768 fp16은 release당 약 0.375 MiB/site-year, 10k×2가 약 7.3 GiB다.
- dense task는 필요한 crop에서만 on-the-fly teacher inference 또는 중간 feature pyramid를 캐시한다.
- raw imagery는 STAC asset ID·etag/checksum과 compact chip을 분리하고, 중복 scene은 content-addressed로
  한 번만 저장한다.

GPU0 한 장은 adapter/small-student 실험에 충분할 가능성이 높다. 병목은 VRAM보다 NFS I/O,
teacher feature extraction, public join, task decoder matrix다.

## 5. 14일 P0와 full 제출 계획

### 5.1 14일 P0 — 먼저 방법 가능성만 판정

| 날짜 | 산출물 | fail-closed gate |
|---|---|---|
| D1 | compact re-embed/dual-index/adapter `N×Q` 비용곡선 | 현실적 N에서 re-embed가 항상 우세하면 중단 |
| D2–4 | preprocessing 재감사 + untouched split·gallery-size protocol freeze | 사소한 mismatch가 R@1=0을 설명하면 drift 주장 중단 |
| D5–8 | 2,048 anchor + layer/geometry predictor + 단순 bridge | predictor가 held-out pair에 일반화하지 않으면 characterization 격하 |
| D9–11 | frozen old quantizer에서 baseline/quantizer-aware P0 | 기존 decode/rebuild/PQ baseline과 같으면 방법 중단 |
| D12–14 | 공개 task 1개 **frozen probe만** + P0 memo | task utility 붕괴 또는 dual-index Pareto 우위면 중단 |

P0는 논문 결과가 아니라 **full experiment에 돈을 쓸지 결정하는 gate**다.

### 5.2 P0 통과 뒤 6–8주

1. 2 release families와 3 public tasks를 동결한다.
2. 10k+ anchors의 compact teacher store를 만든다.
3. 모든 compatibility/compression/task baseline을 compute-matched로 실행한다.
4. 5 seeds, spatial cluster CI, worst pair/region/year/cloud를 낸다.
5. 한국 event-first P0가 닫혔을 때만 public-context cell을 추가한다.
6. method와 test를 고른 뒤 새로운 한국 untouched region을 한 번만 연다.

## 6. CVPR main 가능성 판정

### 지금 제출하면

**main 가능성은 낮다.** 한 release pair의 label-free failure와 좋은 시스템 감사만으로는 method
novelty와 downstream relevance가 부족하다. 기술 리포트, dataset/benchmark track, Findings 성격에
가깝다.

### main으로 올라가는 최소 조건

다음을 모두 만족하면 **도전할 만한 main-track paper**가 된다.

1. compact re-embedding과 dual-index 대비 현실적인 `N×Q` 구간에서 compatibility가 필요한 이유.
2. held-out family/release의 alignment 가능성을 예측하는 characterization 결과.
3. 고정 old quantizer에서 최신 compatibility baseline을 이기는 방법.
4. Olmo와 Prithvi 두 release family, 최소 3개 공개 task에서 task utility·retrieval 동시 유지.
5. gallery-size 곡선과 compressed-gallery Pareto 우위.
6. 제주는 motivation, 새 지역은 untouched external stress test이며 코드·split·asset을 공개.

반대로 아래 중 하나면 main 주장을 즉시 낮춘다.

- Olmo v1/v1.2 한 pair에서만 개선.
- compact re-embedding이 현실적인 archive 크기에서 더 싸고 raw imagery도 계속 접근 가능.
- alignment predictor가 같은 pair를 설명하지만 새 family/release를 예측하지 못함.
- retrieval proxy만 좋고 mIoU/F1/RMSE 또는 old head retention이 없음.
- BCT/LCE/AdvBCT/BT²/강한 post-hoc adapter와 같음.
- public context 효과가 location/year 또는 missingness-only로 설명됨.
- 한국 public source에 공개시각·독립 label·충분한 aligned coverage가 없음.

CVPR 2027은 2027-06-20~24 Seattle 개최만 공식 확인됐고, 논문 마감은 이 문서 작성 시점에 공식
공개되지 않았다. `2026-10-31`은 **내부 완료일**로 유지하되 공식 deadline이라고 쓰지 않는다.

## 7. 근거가 된 1차 문헌

- [OlmoEarth (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Herzog_OlmoEarth_Stable_Latent_Image_Modeling_for_Multimodal_Earth_Observation_CVPR_2026_paper.html)
- [OlmoEarth v1/v1.2 공식 저장소](https://github.com/allenai/olmoearth_pretrain)
- [PANGAEA](https://arxiv.org/abs/2412.04204)
- [GeoLink (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f681209306654a0c1f690f65810e8e45-Abstract-Conference.html)
- [MMEarth (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8085_ECCV_2024_paper.php)
- [OmniSat (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4127_ECCV_2024_paper.php)
- [SatMIP (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3849_ECCV_2024_paper.php)
- [Galileo (ICML 2025)](https://proceedings.mlr.press/v267/tseng25a.html)
- [WildSAT (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Daroya_WildSAT_Learning_Satellite_Image_Representations_from_Wildlife_Observations_ICCV_2025_paper.html)
- [Auxiliary Modality Learning (ICML 2023)](https://proceedings.mlr.press/v202/shen23f.html)
- [AM-RADIO (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Ranzinger_AM-RADIO_Agglomerative_Vision_Foundation_Model_Reduce_All_Domains_Into_One_CVPR_2024_paper.html)
- [BCT (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html)
- [FCT (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Ramanujan_Forward_Compatible_Training_for_Large-Scale_Embedding_Retrieval_Systems_CVPR_2022_paper.html)
- [LCE (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Meng_Learning_Compatible_Embeddings_ICCV_2021_paper.html)
- [AdvBCT (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/html/Pan_Boundary-Aware_Backward-Compatible_Representation_via_Adversarial_Learning_in_Image_Retrieval_CVPR_2023_paper.html)
- [BT² (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Zhou_BT2_Backward-compatible_Training_with_Basis_Transformation_ICCV_2023_paper.html)
- [Hyperbolic BCRL (ICML 2025)](https://openreview.net/forum?id=KUphSx7PAC)
- [Cross-modal XBT (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Towards_Cross-modal_Backward-compatible_Representation_Learning_for_Vision-Language_Models_ICCV_2025_paper.html)
- [NeuCo-Bench (CVPRW 2026)](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Vinge_NeuCo-Bench_A_Novel_Benchmark_Framework_for_Neural_Embeddings_in_Earth_CVPRW_2026_paper.html)
- [Neural Embedding Compression for EO](https://arxiv.org/abs/2403.17886)
- [Embedding-Only Uplink (ICLR 2026 OpenReview)](https://openreview.net/pdf?id=IbzEpGdblY)
- [FLAIR-HUB](https://arxiv.org/abs/2506.07080)
- [Prithvi-EO-2.0](https://arxiv.org/abs/2412.02732)
- [CVPR 2027 개최 정보](https://www.thecvf.com/?p=137)
