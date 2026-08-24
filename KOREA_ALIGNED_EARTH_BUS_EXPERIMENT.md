# K-ALIGN Bus — 한국 공공데이터 정렬형 compatible Earth representation 실험 계약

최종 갱신: 2026-08-23  
상태: **사전 실험 계약. 아직 transfer/task 개선 결과 없음**

**2026-08-23 5차 전략 보정:** 이 계약은 한국 public-context track의 조건부 실험 계약이다.
현재 CVPR 1순위는 `K_ALIGN_BIG_PICTURE.md`의 cost-first characterization × fixed-quantizer
compatibility이며, event-first coverage·독립 label·공개시각 gate가 열리기 전에는 이 계약을
main paper와 합치지 않는다.

관련 문서: `K_ALIGN_PROGRAM_NOTE.md`(프로그램 수준 결정·승률 보정),
`K_GAIN_AXES.md`(정확도·임베딩·속도·위성유도 네 축의 기전·선행연구·빈칸),
`K_CONTEXT_FUSION_EXPERIMENT.md`, `EMBEDDING_TRANSFER_CVPR_TRACKS.md`,
`KOREA_PUBLIC_DATA_CATALOG.md`, `PAPER_READING_LIST.md`

2026-08-23 보정: 아래 계약은 `K_ALIGN_PROGRAM_NOTE.md`의 여덟 보정(R1–R8)을 반영한다.
프로그램 노트와 이 계약이 어긋나면 **이 계약이 이긴다**.

## 0. 먼저 약점과 논문 경계

- 현재 검증된 것은 제주 legacy input 216 site-years에서 OlmoEarth v1↔v1.2의 raw cross-release
  R@1이 0이고 affine ridge도 0.6973/0.6089에 그쳤다는 representation-proxy failure다. 다른
  backbone teacher나 한국 공공데이터가 task를 개선했다는 결과는 없다.
- 기존 sealed 64 site-years는 이미 결과를 봤다. 새 bus·distillation·residual 방법의 test로 다시
  쓰지 않고 Figure 1의 문제 동기로만 사용한다.
- 한국 공공데이터는 많지만 독립 ground truth가 아니다. 특히 BuildingHUB/EIA/FarmMap no-match는
  `no change`나 `no cause`가 아니며, target 사건 record를 feature와 label에 동시에 쓰지 않는다.
- cache 재사용은 높은 CKA가 아니라 old/new/family 교차 query–gallery와 실제 downstream head에서
  판정한다. operational cache 호환성은 backfill 비용과 task gate까지 통과했을 때만 주장한다.
- 로봇·simulation은 이 논문의 주기여가 아니다. stable bus가 검증된 뒤 field-side query encoder나
  trajectory world model의 후속 consumer로만 둔다.
- **순환논증 위험**: 연도별 토지피복·FarmMap·DEM은 그 자체가 EO 파생 산출물이다. 이것을
  privileged teacher로 써서 EO-only student가 좋아지면 독립적인 새 정보가 아니라 weak supervision
  효과일 수 있다. 따라서 §4의 `R/V/T` source transferability triage로 역할을 분리한다.
- **재현성 위험**: 심사자에게 VWorld/BuildingHUB/GK2A 접근권이 없다. 한국 API에만 근거한
  dual-speed 주장은 재현 불가능한 결과로 읽힌다. 공개 데이터만으로 publication lag·coverage
  hole·conflict를 합성하는 harness(§11 A6)가 없으면 일반 주장을 열지 않는다.
- **호환성 novelty 위험**: `S1`에 compat loss를 걸고 `S1→S0` R@1이 높은 것은 BCT가 이미 하는
  일이다. §7의 black-box 불가능성 baseline이 없으면 새 방법 주장을 쓰지 않는다.

## 1. 한 문장 질문과 가제

> **여러 글로벌 EO model family/release의 지식을 한국의 시점화된 공공데이터로 정렬해 안정적인
> EO-only representation bus로 증류하고, 모델 릴리스와 공공 context가 서로 다른 속도로 바뀌어도
> 기존 gallery와 task head를 안전하게 재사용할 수 있는가?**

가제:

> **K-ALIGN: Provenance-Aware Compatible Distillation for Earth Models under Asynchronous
> Public Context**

핵심 novelty 후보는 `한국 데이터가 많다`가 아니라 다음의 결합이다.

1. frozen heterogeneous EO teachers를 공통 stable bus로 옮기는 compatible distillation.
2. EO-only stable cache와 timestamped Korea-context residual을 분리하는 dual-speed representation.
3. model release, EO observation, public-record publication이 서로 다른 시계로 갱신되는 benchmark.
4. 자연 누락·공개 지연·충돌에서 task와 cache migration을 함께 평가하는 contract.

## 2. stable cache와 dynamic residual을 분리한다

```text
canonical EO x ── student encoder S ── z_stable ───────────────┐
       │                     │                                  │
       ├─ Olmo v1/v1.2 ── teacher projectors                   ├─ task/retrieval head
       └─ TerraMind ───── teacher projectors                   │
                                                                │
cutoff-valid Korea context c,p ─ context encoder ─ r_context ──┘
                                      ↑
                         time·coverage·source provenance
```

최종 결정 표현은 개념적으로 다음처럼 둔다.

```text
z_decision = z_stable + gate(provenance, coverage) * r_context
```

- `z_stable`: EO observation으로만 계산하며 model-family와 release를 가로지르는 cache 좌표계다.
- `r_context`: prediction cutoff 이전에 공개된 한국 공공 context의 정보만 담는다.
- context가 없거나 stale/conflicting이면 residual gate가 0 또는 abstain 방향으로 가야 한다.
- `r_context`를 제거한 `z_stable`만으로도 독립 EO task와 cross-model retrieval을 평가한다.
- 공공데이터가 업데이트돼도 stable gallery 전체를 다시 계산하지 않고 residual만 refresh한다.

### 네 개의 효과를 합치지 않는다

| estimand | 비교 | 말할 수 있는 것 |
|---|---|---|
| `E_repr` | 독립 label에서 `V/T`를 통과한 public-context teacher로 학습한 EO-only `z_stable` vs context 없는 student | 한국 context가 EO-only 표현 학습에 준 정보 |
| `E_compat` | upgraded query→old gallery, old query→upgraded gallery vs native | model/release cache 좌표 호환성 |
| `E_fusion` | `z_stable+r_context` vs `z_stable` | cutoff-valid context의 추론 시 추가 정보 |
| `E_refresh` | full re-encode/backfill vs residual/query-only update, **FoldRefresh cross-fitted 인증 포함** | 갱신 비용–utility trade-off와 부분 갱신 위 통계의 유효성 |

`E_fusion`만 좋아졌다면 EO embedding 강화가 아니다. `E_repr`만 좋아져도 old gallery와 호환되지
않으면 cache 재사용 기여가 아니다. `E_refresh`의 비용 절감만 있고 부분 갱신 위 통계 보증이
깨지면 논문 기여가 아니라 engineering report다.

라벨 의존성으로도 나눈다. `E_compat`·`E_refresh`와 `R(source)`·coverage 진단은 **사람 라벨이
0개**여도 평가되지만, `V/T`와 `E_repr`·`E_fusion`은 독립 label을 요구한다. 따라서 label-free
호환성 core와 public-context transfer를 별도 임계경로로 관리한다(§11).

## 3. 모델 갱신 protocol

### Stage 0 — base bus

- frozen teacher: OlmoEarth v1 Base.
- student `S0`: compact ViT/temporal encoder 한 종류.
- training: canonical S2 + cutoff-valid public-context privileged teacher.
- output: frozen `bus_v0` gallery와 task head.

### Stage 1 — upgraded bus

- 새 teacher: OlmoEarth v1.2 Base + TerraMind Base.
- student `S1`: 동일 architecture 또는 사전 고정한 더 작은 architecture.
- `S1`은 새 teacher의 지식을 배우되 `S0` gallery와 직접 검색 가능하도록 compatible loss를 받는다.
- test 때 `S0` gallery/task head는 다시 학습하거나 backfill하지 않는다.

### 필수 query/gallery 네 칸

| query | gallery | 역할 |
|---|---|---|
| `S0` | `S0` | old native lower reference |
| `S1` | `S1` | new native upper reference |
| `S1` | `S0` | 실제 새 query→기존 cache |
| `S0` | `S1` | reverse compatibility 진단 |

teacher raw space와 student bus space를 한 표에서 같은 embedding이라고 부르지 않는다. teacher별
projector는 증류 loss용이고, 배포 retrieval은 projector 없는 `S0/S1` stable bus로 측정한다.

## 4. 한국 공공데이터 alignment 계약

### source role

| source family | 예 | stable/context 역할 | 금지 |
|---|---|---|---|
| 공간 anchor | VWorld PNU·필지 geometry, 행정경계 | AOI group·geometry relation, cache key 보조 | 대표점을 오름/필지 전체 경계로 간주 |
| 관측품질 | SCL, GK2A, KMA 관측 | cloud/weather `observation` residual, 품질 strata | 현재 GK2A를 과거 구름 정답으로 사용 |
| 상태 map | 연도별 토지피복, FarmMap, DEM | cutoff-valid auxiliary teacher·state token | 제품 기준일을 실제 변화일로 사용 |
| 행정 event | BuildingHUB, EIA, 개발·환경 기록 | target과 독립인 auxiliary 또는 post-evidence | visual-change label과 같은 event를 feature로 재사용 |
| 독립 task label | NGII 전후 항공사진, 블라인드 EO 판독 | task ground truth | API record를 그대로 정답으로 사용 |
| coverage | endpoint 모집단, 기준일, error, lag | residual gate·평가 stratum | missing/no-match를 0으로 숨김 |

각 public token은 다음을 잃지 않는다.

```text
(source_id, record_id_hash, geometry_relation,
 event_time, observed_time, published_time, retrieved_time,
 prediction_cutoff, staleness_days, coverage_state,
 value_or_embedding, raw_sha256)
```

prospective input 조건:

```text
published_time <= prediction_cutoff
retrieved snapshot contains the record
geometry relation is explicit
source role does not duplicate the task label
```

`published_time`이 불명확하면 prospective residual에서 제외하고 auxiliary sensitivity 또는
post-prediction evidence로만 쓴다.

### source transferability triage — `E_repr` teacher의 사전 자격

연도별 토지피복·FarmMap·DEM은 EO 관측에서 파생된 산출물이라 독립적인 새 정보로 해석하기 어렵다.
그러나 EO-only probe가 source를 잘 예측한다는 이유만으로 teacher에서 제외하는 것도 잘못이다.
test 때 context가 없는 student로 지식을 옮기려면 public signal 중 일부가 EO에서 **회복 가능해야**
하기 때문이다. 반대로 EO에서 전혀 회복되지 않는 행정정보는 inference residual에는 유용할 수 있어도
EO-only embedding으로 전달되는 정보에는 상한이 있다. 따라서 단일 `D(source)` hard exclusion을
폐기하고 recoverability·조건부 task value·실제 transfer를 분리한다.

```text
R(source) = 같은 EO 입력·같은 split으로 학습한 EO-only probe가
            그 public token을 얼마나 회복하는가
            연속값: R^2 | 범주: balanced AUPRC | 사건 시각: ±k일 tolerance AUPRC

V(source) = 독립 task label Y에서 EO+context가 EO-only보다 주는 조건부 가치
T(source) = context를 train 때만 본 EO-only student가 no-context student보다 얻는 전이 이득
```

- `R`이 높고 `V`가 낮으면 중복/shortcut이며 pseudo-label baseline으로만 쓴다.
- `R`이 낮고 `V`가 높으면 inference-time residual/abstention 후보이며 EO-only transfer를 주장하지 않는다.
- `R`이 중간이고 `V`와 `T`가 양수일 때만 privileged `E_repr` teacher로 승격한다.
- `R` 진단 임계값과 `V/T` promotion gate는 A0에서 고정한다. `R` 진단 제안값은 R² 0.60,
  AUPRC-over-prior lift 2.0, 사건 시각 `k = 30`일이다. 이 값만으로 source를 탈락시키지는 않는다.
- probe는 teacher 학습과 **같은 EO 입력·같은 split**을 쓴다. 아니면 screen 자체가 leak이다.
- `R/V/T`는 실패해도 전부 보고한다. 어떤 공공기록이 회복 가능하고 task에 조건부 가치가 있으며
  EO-only student로 옮겨지는지의 구분 자체가 산출물이다.

사전 예상(반드시 실험으로 뒤집힐 수 있도록 먼저 기록):

| source | 예상 `R` | 근거 |
|---|---|---|
| 건축 인허가·착공·사용승인 일자 | 낮음 | 행정 시각은 물리 관측에 없음 |
| EIA 사업구역 경계 | 낮음 | 법적 경계는 지표 반사와 무관 |
| VWorld PNU 필지 경계 | 낮음–중간 | 지적선과 물리 경계가 자주 불일치 |
| GK2A 독립 구름 산출 | 중간 | SCL과 상관 있으나 센서가 독립 |
| FarmMap 필지 | 중간–높음 | 항공영상 기반 |
| 연도별 토지피복 | 높음 | EO 파생 산출물 |
| DEM | 높음 | 지형은 EO에서 상당 부분 회복 |

**coverage 하한.** 유효 event/published time·geometry를 가진 source coverage가 대상 site-event의
5% 미만이면 `E_repr` feasibility를 열지 않는다. 5%는 학습을 검토할 최소 하한일 뿐 CVPR main의
충분조건이 아니다. 현재 반증 신호가 이미 있다 — 14후보에서 exact PNU 일치 1건, 시간정렬 0건,
EIA 직접중첩 0건이다. candidate-first join이 계속 실패하면 BuildingHUB/EIA event universe에서
before/after EO를 역으로 수집하는 event-first pilot으로 전환한다.

### alignment 단위

기본 단위는 필지/공간사건과 시점이 결속된 다음 record다.

```text
(site_event_id, spatial_group_id, geometry, t0, t1,
 acquisition_ids, input_tensor_hash, public_snapshot_cutoff,
 teacher_release_ids, bus_release_id, task_label_source)
```

같은 PNU, 500 m spatial buffer, Sentinel scene, 실제 사건, 오름/AOI는 split을 가로지르지 않는다.

## 5. 학습 목적

```text
L = L_task
  + lambda_teacher * L_multi_teacher
  + lambda_relation * L_spatiotemporal_relation
  + lambda_compat * L_bus_compat
  + lambda_context * L_public_context
  + lambda_preserve * L_native_preserve
```

| loss | 목적 | fail-closed 조건 |
|---|---|---|
| `L_task` | 독립 visual/site-state task | 사람/독립 영상 label이 있을 때만 |
| `L_multi_teacher` | Olmo/TerraMind pooled·spatial token 지식 전달 | model별 공식 input view와 projector를 기록 |
| `L_spatiotemporal_relation` | site·token·시점의 이웃/차이 구조 보존 | pixel을 독립 표본으로 취급 금지 |
| `L_bus_compat` | `S1` query가 frozen `S0` gallery에서 작동 | old gallery·head를 update에 사용 금지 |
| `L_public_context` | cutoff-valid context alignment/masked prediction | no-match를 negative target으로 사용 금지 |
| `L_native_preserve` | 새 bus task utility 붕괴 방지 | train/validation만 사용, sealed 통계 금지 |

public-context teacher는 task label과 다른 source/role만 쓴다. `BuildingHUB 사건 = visual change`처럼
동일 record를 teacher input과 target에 중복하면 그 cell 전체를 leak으로 판정한다.

## 6. 데이터와 split

### 한국 frame

- 지역: 제주, 강원 산림권, 수도권 외곽 개발권.
- 시점: 2021–2024 train, 2025 validation, 가장 최신 완결연도·새 acquisition은 untouched test.
- unlabeled privileged-distillation frame: 최소 10,000 parcel/site-years.
- supervised 권장: train 600, validation 200, sealed geographic-future test 400.
- independent double review: sealed 400 전수, train 최소 120; agreement <0.60이면 모델표를 열지 않는다.

### 기존 제주 full-216의 역할

- 허용: model release가 같은 입력에서 cache identity를 깨뜨릴 수 있다는 motivation.
- 금지: 새 loss·architecture·hyperparameter 선택, 새 method headline test.
- 새 test는 기존 54 spatial key와 smoke에서 공개된 위치를 모두 제외한다.

### 외부 task

CVPR main claim에는 한국 3지역만으로 끝내지 않고 PANGAEA/GEO-Bench 계열의 공개 dense task 최소
하나를 재사용한다. 외부 task에는 한국 public residual을 억지로 만들지 않고 `compatible
multi-teacher bus`의 일반성만 검사한다. 한국에서만 가능한 provenance contribution과 일반
representation contribution을 같은 수치로 합치지 않는다.

## 7. baseline matrix

### compatibility·distillation

1. Olmo v1/v1.2 raw identity.
2. train-only mean/translation, Procrustes, affine ridge.
3. per-teacher linear/MLP projector.
4. single-teacher feature/logit distillation.
5. BCT, FCT, LCE, AdvBCT.
6. **black-box 불가능성 baseline (필수)** — 아래 세 개는 novelty 문단의 근거이므로 생략 불가.
   - `BCT-surrogate`: 공개 release에는 구 모델의 분류기·학습데이터가 없다. `S0` gallery에서
     surrogate head를 학습해 BCT influence loss를 건다. 원 가정이 깨졌을 때의 손실을 측정한다.
   - `FCT-posthoc`: 구 모델 쪽 side-information을 준비할 수 없으므로 사후 변환만으로 흉내낸다.
     성립 여부 자체가 결과다.
   - `contract-mismatch`: teacher의 band·GSD·temporal stride가 다를 때 위 두 방법의 추가 손실.
7. AM-RADIO-style multi-teacher distillation.
8. proposed stable bus without public context.
9. proposed stable bus with public-context privileged teacher.

주장 형태는 "BCT/FCT가 나쁘다"가 아니라 **"BCT/FCT의 가정이 공개 EO release에서 성립하지 않으며,
그 조건에서 우리 bus만 gate를 통과한다"**이다. `BCT-surrogate`가 우리 bus와 통계적으로 같으면
새 방법 주장을 중단한다(§13).

### public-context

1. EO-only.
2. location+year-only.
3. context-only.
4. late concat MLP, STACK, TOKEN-FUSE.
5. GeoLink-style object/parcel fusion, CLIP4Geo식 POI/텍스트 정렬, WildSAT식 비-EO 관찰기록 정렬,
   raster×vector 통합(Beyond-Pixels 계열).
6. privileged-modality distillation: JDCNet식 confidence-gated, InfraNet식 학습전용 보조모달.
7. `z_stable + r_context`.
8. full re-encode/full-context upper bound.
9. 집계 통계 보정(W-23 계열)과 약지도 필지 라벨. **이 둘은 baseline이지 기여가 아니다** —
   전지구 10 m 필지 경계 지도(241개국 31.7억 polygon)와 FLAIR-HUB가 이미 존재한다.

### efficiency

- full backfill, query-only adapter, residual-only refresh.
- 각 refresh 경로에 **FoldRefresh(cross-fitted partial refresh) 인증**을 붙인다. 부분 갱신된
  cache 위에서 계산한 지도 수준 통계가 유효한 유한모집단 보증을 유지하는지 함께 보고한다.
  BCT·FCT·AM-RADIO·Matryoshka 어디에도 이 보증은 없다.
- PCA, product quantization, **binary quantization(1 bit/dim)**, independent low-dim model,
  Matryoshka 64/256/768d.
- **bytes 비교의 기준선은 PCA(64)+int8이다.** float32 대비 절감은 이미 알려진 압축이므로
  성과로 세지 않는다. NeuCo-Bench 계열 EO 손실 신경압축 baseline을 함께 보고한다.
- `E_refresh`의 운영 동기 중 하나는 온보드 임베딩 링크다. E-07의 약 598–690 B/query는 주로
  downlink JSON telemetry이고, hint gallery 업링크는 `N_hints × D × bytes`로 별도다. 따라서
  **시뮬레이션된 대역폭 예산으로만 다루고, 1 KB gallery budget·물리적 backfill 불가능·온보드
  실측을 주장하지 않는다.**
- 모두 같은 acquisition, AOI support, labels, split, head/search GPU-hours를 사용한다.

## 8. 반증 control

- region×year 안에서 public record shuffle.
- context event/published time을 ±1년 이동.
- PNU/geometry shape only, missingness mask only, location/year only.
- target label과 같은 source event를 의도적으로 넣는 leak-positive sentinel: pipeline이 100% 차단해야 함.
- natural missingness와 random modality dropout을 별도 곡선으로 보고.
- context residual을 제거한 EO-only student와 teacher/context-only model을 항상 함께 보고.
- teacher마다 acquisition·GSD·band가 다른 native ceiling은 paired-input 표에서 분리.

## 9. 지표와 promotion gate

| 축 | primary metric | 사전 GO | KILL |
|---|---|---|---|
| `E_repr` | **primary: labels-to-target(라벨 절감)** · secondary: EO-only frozen-probe AUPRC | no-context student 대비 **label 20% 절감**(primary) 또는 +2%p(secondary), CI>0, ≥2/3 지역 | location/year-only 또는 context-only와 같음. **full-label 구간에서 강한 supervised baseline(UNet 등)에 지면 정확도 주장을 쓰지 않는다** |
| `E_compat` | cross-bus R@1/mAP, task-head retention | `S1/S0`이 `S1/S1`의 95% 이상, old head 손실 ≤1%p | affine bridge와 같거나 native task −2%p 초과 |
| `E_fusion` | event AUPRC, high-cloud false alarm, AURC | best simple fusion +2%p 또는 같은 성능에서 label 20% 절감 | shuffle/time-shift에서도 이득 유지 |
| `E_refresh` | backfill bytes, wall/GPU-hour, query latency **+ FoldRefresh 인증 통계** | backfill bytes 10×, query 5× 또는 embedding bytes 절감. **bytes 기준선은 float32가 아니라 PCA(64)+int8** — 그 위에서 추가 절감이 있어야 한다. 동시에 부분 갱신 위 통계 보증 유지 | full re-encode와 비용 차이 없음. PCA64+int8 대비 추가 절감 없음. 절감만 되고 보증이 깨지면 engineering report로 축소 |
| source transferability | `R(source)`, `V(source)`, `T(source)`와 유효 source coverage | coverage ≥5%인 feasibility frame에서 독립 label 기준 `V/T` 양수 | 유효 공개시각·coverage·독립 label이 없거나 `T`≤0 → `E_repr`을 열지 않음 |
| black-box 호환성 | `BCT-surrogate`·`FCT-posthoc` 대비 우리 bus의 cross-bus R@1 | `BCT-surrogate` 대비 유의한 우위 (CI>0) | 통계적으로 같으면 새 방법 주장 중단 |
| 일반성 | 합성 비동기 harness에서의 효과 방향 | 공개 데이터 harness에서 방향 재현 | 방향 소멸 시 한국 전용 사례연구로 축소 |
| worst group | region/year/cloud/coverage macro·minimum | 어떤 group도 −2%p 초과 악화 없음 | 두 번째 지역에서 방향 소멸 |
| leak | sentinel rejection | future/duplicate-role 100% 차단 | 한 건이라도 통과 |

위 네 효과 중 `E_repr + E_compat`가 통과해야 compatible representation paper다. `E_fusion`만
통과하면 K-Context fusion paper로 축소하고, `E_refresh`만 통과하면 engineering report다.

## 10. 최소 Figure/Table

1. stable EO bus와 timestamped residual의 dual-speed architecture.
2. `S0/S0`, `S1/S1`, `S1/S0`, `S0/S1` query/gallery matrix.
3. teacher family×release×task의 native/compatible performance.
4. `E_repr`, `E_fusion`, natural missingness·time-shift/shuffle ablation.
5. region×year×cloud×coverage worst-group heatmap.
6. task/retrieval 대 backfill bytes·latency·embedding dimension Pareto curve.
7. source별 recoverability `R`, 조건부 task value `V`, EO-only transfer `T`와 유효 coverage 표.
8. `BCT-surrogate`/`FCT-posthoc`/`contract-mismatch` 대비 우리 bus의 cross-bus 표.
9. 합성 비동기 harness의 lag·coverage·conflict 스윕과 한국 실측 분포 중첩.
10. 기존 제주 full-216 failure atlas는 motivation figure로만 사용.

## 11. 실행 순서

라벨 의존성으로 두 갈래로 나눈다. 왼쪽 A 계열이 임계경로, B 계열은 병렬이다.
**라벨(B1)이 A5·A6·A7을 막지 않는다.** 파트타임 실행에서 라벨은 단일 실패점이었다.

| 단계 | 기간 | 산출물 | 다음 단계 gate |
|---|---:|---|---|
| A0 source/role/transferability 계약 freeze | 3일 | source×cutoff×role×license manifest, leak test, `R/V/T` gate 고정, 합성 harness 기반 공개 데이터셋 확정 | 시간·role 불명확 source 제외 |
| A1 canonical 3-region anchors + untouched split hash-freeze | 1주 | ≥10k site-years, exact input/public snapshot hash | 두 번째 지역 join 실패 시 중단 |
| A2 teacher contract smoke | 3일 | 32 windows × Olmo v1/v1.2 · TerraMind · (선택) Prithvi 1.0/2.0의 token·stride·mask·runtime | paired support 불일치면 family×release 격자를 축소 |
| A3 source transferability triage | 4일+label pilot | source별 `R/V/T`, 유효 공개시각·coverage; label 전에는 `R/coverage`만 | 독립 label 없이 `E_repr` 개방 금지 |
| A4 bridge + black-box 불가능성 | 1주 | identity/linear/MLP/relational + `BCT-surrogate`/`FCT-posthoc`/`contract-mismatch` 표 | affine=MLP면 nonlinear 확장 중단 |
| A5 bus P0 — `E_compat` + `E_refresh` | 2주 | `S0`/`S1`, 4칸 query/gallery, FoldRefresh 인증 refresh 곡선, 256/768d | 두 gate 미달이면 bus 주장 중단 |
| A6 합성 비동기 harness | 1주 | 공개 task 위 publication lag·coverage hole·conflict 스윕, 한국 실측 분포 중첩 그림 | 효과 방향 소멸 시 dual-speed 일반 주장 중단 |
| A7 asset 패키징 | 1주 | 공개 가능한 비동기 benchmark + failure atlas + source transferability 표 | 라이선스 미해결 항목은 제외하고 진행 |
| B1 (병렬) 라벨 수집 | 즉시 착수, 4–8주 | NGII 전후 항공 신청, 블라인드 이중판독, train 600 / val 200 / sealed 400 | agreement <0.60이면 점수표를 열지 않음 |
| B2 `E_repr` + `E_fusion` | B1 이후 2주 | 4 estimand 전체 표, 5 seeds, CI·Pareto | 모든 주장을 사전 gate로만 판정 |

GPU0가 비어 있다는 이유로 A2부터 시작하지 않는다. A0/A1의 source-role, cutoff, exact input,
untouched split이 먼저 닫혀야 한다.

### family × release 격자

"family/release 호환성"을 주장하려면 최소 2 family × 2 release가 필요하다. 현재 구성
(Olmo v1/v1.2 + TerraMind Base)은 release 축이 한 family에만 있다.

| | release A | release B |
|---|---|---|
| Olmo | v1 | v1.2 |
| 두 번째 family (후보) | Prithvi-EO 1.0 | Prithvi-EO 2.0 |
| 세 번째 (선택) | TerraMind Base | — |

Prithvi는 6-band·30 m HLS 계약이라 S2 10 m와 paired input이 성립하지 않을 수 있다. **A2에서
판정한다.** 성립하지 않으면 격자를 포기하고 "1 family 2 release + 2 family cross-sectional"로
주장을 낮춘다. 억지로 밀면 native ceiling 혼동이 생긴다.

### 제출 사다리

| 순위 | 산출물 | 필요 조건 |
|---:|---|---|
| 0 | arXiv 기술 리포트 + 비동기 benchmark asset | A0–A3 + A7 |
| 1 | K-ALIGN 본편 | `E_repr` + `E_compat` 동시 통과 **그리고** A6 통과 |
| 2 | Compatible Earth Representation Bus | `E_compat`만 통과 |
| 3 | K-Context Fusion | `E_fusion`만 통과 |
| 4 | Partial-Refresh Engineering Report | `E_refresh`만 통과 |

두 축이 통과해도 **A6(합성 harness)이 없으면 main으로 올리지 않는다.** 재현 불가능한 지역
데이터에만 근거한 일반 주장은 리뷰에서 살아남지 않는다.

## 12. 현재 허용·금지 주장

현재 허용:

- full-216은 cross-release raw cache identity가 깨질 수 있다는 문제를 보여준다.
- K-ALIGN은 stable cache와 dynamic public-context update를 분리해 검증하도록 설계됐다.
- multi-teacher distillation, compatibility, public alignment, refresh cost는 별도 estimand다.
- BCT/FCT는 구 모델을 통제할 수 있다고 가정하며, 공개 EO release에는 그 가정이 성립하지 않는다
  (이것은 문헌 독해에 근거한 설계 판단이며, 정량 손실은 A4 실행 후에만 수치로 말한다).

현재 금지:

- 한국 공공데이터가 OlmoEarth/TerraMind embedding을 개선했다.
- 새 bus가 cache backfill을 없앴거나 operational compatibility를 달성했다.
- 한국 전체·실시간·로봇 navigation으로 일반화됐다.
- task label 0인 release audit을 supervised transfer 근거로 사용했다.
- 어떤 공공 source가 EO에서 회복 불가능하거나 EO-only student에 유용하다 (`R/V/T` 측정 전까지는 예상일 뿐이다).
- 부분 갱신된 cache 위의 통계가 유효하다 (이 저장소에서 FoldRefresh 인증을 재현하기 전까지).
- 우리 bus가 BCT/FCT보다 낫다 (A4 표가 나오기 전까지).

## 13. 프로그램 수준 중단 조건

§9의 estimand별 KILL과 별개로, 아래는 프로그램 자체를 축소·중단시킨다.

1. **데이터 중단** — A3에서 screen 통과 source의 시간정렬 coverage가 5% 미만이면 `E_repr`을
   열지 않고 순위 0(리포트+asset)으로 전환한다.
2. **호환성 중단** — A4에서 `BCT-surrogate`가 우리 bus와 통계적으로 같으면 새 방법 주장을
   중단하고 "공개 release에 BCT를 적용하는 법"이라는 더 작은 논문으로 낮춘다.
3. **일반성 중단** — A6 합성 harness에서 효과 방향이 사라지면 한국 전용 사례연구로 낮춘다.
4. **보증 중단** — `E_refresh`의 절감은 있으나 FoldRefresh 보증이 깨지면 통계 주장을 삭제하고
   비용만 보고한다.
5. **인력 중단** — B1의 이중판독 agreement가 0.60 미만이면 라벨 축 전체를 닫고 label-free
   산출물만 낸다. 낮은 agreement로 만든 점수표는 자기기만이다.
6. **시간 중단** — 마감 4주 전까지 A5가 닫히지 않으면 순위 0으로 전환하고 본편을 다음 마감으로
   넘긴다. **마감을 맞추려고 gate 수치를 사후에 낮추지 않는다** (L4).

### 최악의 경우에도 남는 것

모든 gate가 실패해도 아래 두 개는 이미 봉인된 수치로 성립한다.

- **CKA는 호환성이 아니다** — pooled CKA 0.97857, 거리 Spearman 0.95251, 그런데 같은 token의
  raw cosine −0.00860이고 cross-release R@1은 양방향 0.0000이며 사전등록 8 gate가 전부 실패했다.
  표현 유사도 지표로 모델 호환성을 주장하는 관행에 대한 반례다. 제주 한 지역·label 0의
  결과이므로 일반화 문장을 붙이지 않는다.
- **비동기 benchmark asset** — 216 site-years × 2 릴리스 paired 출력, 5,616파일 exact-input
  freeze, 5,184행 시간축 manifest, 463 request provenance snapshot, failure atlas.
