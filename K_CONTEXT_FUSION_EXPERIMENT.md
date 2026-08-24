# K-Context — 시점·coverage가 있는 공공 context로 EO 표현을 강화하는 실험 계약

최종 갱신: 2026-08-23  
상태: **문헌 대조를 마친 사전 실험 계약. 아직 supervised 결과 없음**

관련 문서: `K_EVIDENCE_SHIFT_BENCHMARK.md`, `RESEARCH_EXECUTION_PLAN.md`,
`KOREA_PUBLIC_DATA_CATALOG.md`, `PAPER_READING_LIST.md`,
`KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`

## 먼저 약점과 경쟁 경계

- 현재 제주 자산은 독립 사람 정답이 0개이고, 공식 원인 근거도 0/14다. full-216 감사는
  OlmoEarth v1↔v1.2 표현 proxy 호환성을 기각했을 뿐, 공공데이터가 task 정확도나 임베딩
  utility를 높인다는 증거가 아니다.
- “EO 영상에 지도·날씨·위치정보를 붙이면 좋아진다” 자체는 새 주장이 아니다. GeoLink는
  위성영상과 OpenStreetMap 객체 graph 127만 쌍을 정렬했고, MMEarth·Galileo·TerraMind는
  광학·SAR·고도·날씨·지도 label·시공간 metadata의 다중모달 학습을 이미 보였다.
- 따라서 정적 지도를 하나 더 붙이는 late fusion, 제주 단일 평균 향상, 미래 행정기록 누출,
  API no-match를 음성으로 바꾸는 실험은 workshop 수준의 데모에도 못 미친다.
- 한국 공공데이터의 가능한 새 기여는 **동적·시점화된 공식 context**, **자연적으로 비무작위인
  누락과 공개 지연**, **출처별 역할 분리**, **라벨이 적을 때 EO-only 표현으로 증류되는 정보 이득**을
  함께 측정하는 데 있다.

## 메인 논문 질문

> 시점·출처·coverage가 명시된 한국 공공 context를 글로벌 Earth foundation model의
> privileged supervision과 inference-time context로 결합하면, 영상-only 모델보다 라벨 효율,
> 지역·미래연도 OOD, high-cloud 성능을 높이면서 자연 누락·지연 아래에서도 그 이득을 유지하는가?

가제:

> **Context Under Coverage: Provenance-Aware Public Data Adaptation for Earth Foundation Models**

논문에서 분리해 추정할 효과는 세 개다.

| 효과 | train 입력 | test 입력 | 허용되는 해석 |
|---|---|---|---|
| `E_repr` | EO + public context auxiliary loss | **EO only** | 공공 context가 EO embedding에 증류됐는가 |
| `E_fusion` | EO + public context | EO + cutoff-valid context | 추론 시 추가 정보가 prediction을 개선했는가 |
| `E_decision` | 영상 예측 후 별도 evidence | 영상 예측 + evidence/coverage | 보류·검증 결정이 안전해졌는가 |

`E_fusion`이나 `E_decision`만 좋아졌다면 “임베딩이 강화됐다”고 쓰지 않는다.

### K-ALIGN과의 관계

이 문서는 public-context source role·provenance·natural missingness의 기본 계약이다. 사용자가
요청한 cache 재사용·다른 backbone teacher 증류까지 포함한 제출 후보는
`KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`가 authoritative하다. 그 문서는 stable EO cache
`z_stable`과 timestamped residual `r_context`를 분리하고 여기에 `E_compat`과 `E_refresh`를
추가한다. `E_repr`와 cross-model compatibility가 모두 사전 gate를 통과할 때만 두 축을 하나의
K-ALIGN main paper로 합치며, 하나만 통과하면 이 K-Context 또는 compatible-bus 논문으로 다시
분리한다.

## 선행연구를 넘는 정확한 빈칸

| 선행 축 | 이미 된 것 | 이 프로젝트가 추가해야 할 것 |
|---|---|---|
| GeoLink | EO–OSM region/object alignment, object-patch fusion, random object deletion | 정적 OSM이 아닌 event/publish/retrieval time이 있는 공식 기록, 자연 coverage·누락·충돌 |
| MMEarth | 12 modality를 multimodal pretext로 사용해 optical representation 강화 | 한국 prospective cutoff와 source-role leakage 계약, 지역·미래연도 실험 |
| SatMIP | 위치·시간 metadata를 text supervision으로 정렬 | 좌표 shortcut을 넘는 parcel/event context와 shuffled/time-shift control |
| Galileo/TerraMind | 대규모 native multimodal foundation model과 missing-modality 처리 | 동결 OlmoEarth에 붙는 경량 adapter, 공개자료 지연·no-match 아래 utility/abstention |
| Rao–Rolf | 단순 STACK/TOKEN-FUSE가 저라벨·OOD에서 강한 workshop baseline | hard-coded fusion을 이기는 main-track method와 독립 라벨·sealed test |

이 빈칸이 유지되려면 논문 contribution은 다음 셋을 모두 가져야 한다.

1. `event_time`, `observed_time`, `published_time`, `retrieved_time`, coverage 상태가 있는 benchmark.
2. EO-only student와 EO+context teacher를 분리하는 provenance-aware lightweight adapter.
3. 지역·미래연도·구름·자연 누락에서의 label efficiency, calibration, selective risk 평가.

## 데이터 역할 — 한 source는 한 셀에서 한 역할만

| 역할 | 후보 source | 사용할 수 있는 정보 | 금지 |
|---|---|---|---|
| 물리 context 입력 | S1/S2/SCL, GK2A/KMA, DEM, t0 토지피복, 필지형상 | 예측 cutoff 이전에 관측·공개된 값 | 미래 관측·사후 보정값 |
| privileged train signal | cutoff-valid land-cover/context 속성, 독립 target과 다른 행정 속성 | masked context prediction·teacher 입력 | target 사건 record를 target label과 동시 사용 |
| task label | NGII/독립 전후영상의 블라인드 이중판독 | `visual_change`, 선택적 mask | API 사건을 그대로 시각변화 정답으로 사용 |
| post-prediction evidence | BuildingHUB, EIA, FarmMap, 허가·규제 record | 공간·시간이 맞는 support와 coverage | no-match를 `no_change`/`no_cause`로 바꿈 |
| coverage metadata | endpoint 모집단, 기준일, 오류, publication lag | available/missing/error/out-of-window/conflict | missing row를 0으로 impute하고 숨김 |

모든 context token은 최소 다음 필드를 가진다.

```text
(source_id, geometry_relation, event_time, observed_time,
 published_time, retrieved_time, staleness_days,
 coverage_state, value_or_embedding, raw_sha256)
```

prospective 셀은 `published_time <= prediction_cutoff`인 record만 입력한다. 공개시점을 모르면
train auxiliary 또는 post-hoc evidence로만 쓰고 prospective input에서는 제외한다.

## 방법 가설 — Provenance-Aware Context Adapter

### 표현

1. 동결 OlmoEarth v1.2에서 EO spatial token `Z_eo`를 추출한다.
2. 수치·범주·geometry·time·source별 encoder로 context token `Z_ctx`를 만든다.
3. 작은 gated cross-attention/residual adapter가 `Z_eo`에 context를 결합한다.
4. `state` token과 cloud/weather 같은 `observation` token을 분리해 지표면 변화와 관측조건을
   동일 latent에 무분별하게 섞지 않는다.
5. 자연 coverage 상태를 보존한 source dropout을 사용한다. 무작위 modality dropout은 별도
   sensitivity이며 자연 누락의 대체물이 아니다.

### 학습 목적

| loss | 목적 | 적용 조건 |
|---|---|---|
| `L_task` | 독립 visual-change/site-state task | 사람/독립 영상 label이 있을 때만 |
| `L_ctx` | masked context reconstruction 또는 contrastive alignment | coverage=`available`인 source만; no-match를 negative로 사용 금지 |
| `L_consistency` | cloud/observation 변화 아래 land-state 일관성 | 같은 site의 유효 다중관측이 있을 때 |
| `L_distill` | EO+context teacher의 정보를 EO-only student에 전달 | privileged-train track |
| `L_preserve` | 원 Olmo embedding/task utility 과도한 망각 방지 | source/clear locked validation에서 측정 |

처음부터 새 foundation model을 pretrain하지 않는다. `frozen head → adapter/LoRA → full FT` 순서로
headroom을 확인하고, adapter trainable parameter는 backbone의 2% 이하를 1차 목표로 둔다.

## P0 — 라벨을 대량 만들기 전에 죽일 가설

목적은 “정확도가 올랐다”가 아니라 context가 좌표·연도 shortcut 이상으로 EO 표현에 정보와
headroom을 주는지 판정하는 것이다.

### 표본

- 제주, 강원 산림권, 수도권 외곽 개발권 3지역.
- 최소 10,000 parcel/site-year의 unlabeled frame; 과거 2021–2024 train, 2025 future validation.
- 각 source의 cutoff-valid snapshot, coverage state, geometry/time join을 frozen manifest로 만든다.
- P0 task probe에는 독립적으로 만들 수 있는 소규모 상태 label만 사용하고 변화 성능으로 일반화하지
  않는다.

### P0 비교

1. OlmoEarth v1.2 frozen embedding.
2. `location + year` only.
3. context-only.
4. late concatenation MLP.
5. hard-coded STACK/TOKEN-FUSE.
6. provenance-aware adapter의 EO+context teacher.
7. teacher에서 증류한 EO-only student.

### P0 필수 반증 control

- 지역×연도 안에서 context row shuffle.
- context를 ±1년 이동한 time-shift.
- missingness mask만 투입.
- geometry/PNU shape만 투입.
- 미래 record를 의도적으로 넣은 leak-positive sentinel. 정상 pipeline에서는 반드시 차단돼야 한다.
- context-only가 fused model과 같아지는지 검사.

P0 GO:

- EO-only student frozen probe가 동일 Olmo baseline보다 macro-AUPRC `+2%p` 이상 또는 같은 성능의
  label을 `20%` 이상 절감하고 spatial clustered 95% CI가 0을 넘는다.
- 3지역 중 2곳 이상 같은 방향이며 shuffled/time-shift context는 관측 이득의 80% 이상을 잃는다.
- context-only가 fused model을 따라잡지 않고, clear/source locked 성능 손실이 1%p 이하다.

P0 KILL:

- 개선이 location/year-only 또는 missingness-mask-only와 구분되지 않는다.
- 개선이 미래 record나 target과 동일한 행정 사건을 넣을 때만 생긴다.
- best simple STACK/TOKEN-FUSE보다 proposal이 2%p도 못 낫고 parameter/compute 이점도 없다.
- 두 번째 지역에서 방향이 사라진다.

KILL이면 adapter 논문은 멈추고 `동적 공공 context의 shortcut·누락 benchmark` 또는 선택적 decision
논문으로 범위를 낮춘다.

## P1 — main-track supervised 실험

### sampling frame과 라벨

- 3지역×최소 2전이구간×clear/high-cloud를 사전 층화한다.
- 권장 총 독립 site-event label 1,200건:
  - train 600, validation 200, sealed probability/geographic-future test 400.
  - sealed 400은 전수 이중판독; train에서는 최소 120을 이중판독한다.
- 동일 필지·사건·500 m buffer·Sentinel scene·PNU connected component는 한 split에만 둔다.
- 모델 score, 공공근거, assistant pre-annotation을 보지 않고 sealed test를 추출·hash·봉인한다.
- prevalence/모집단 주장은 포함확률이 있는 probability sample만 사용한다.

### primary task와 optional task

| 우선순위 | task | label | 이유 |
|---|---|---|---|
| primary | parcel/site-event visual change | yes/no/uncertain, 선택적 mask | 공공 사건 record와 독립인 평가 가능 |
| secondary | land-state/site type | 독립 지도·판독 label | embedding label-efficiency 측정 |
| safety | evidence-conditioned selective report | supported/unknown/conflict | 원인 단정 대신 coverage-aware 보류 |

`uncertain`은 음성으로 합치지 않는다. 평가 coverage와 abstention으로 보존한다.

### 비교 모델

#### Paired-input 표

- task-specific U-Net/temporal transformer scratch.
- architecture-matched random-init transformer.
- OlmoEarth v1.2 frozen linear/MLP, SLR adapter, LoRA/full FT.
- simple late concat, STACK, TOKEN-FUSE.
- GeoLink-style object-patch cross-attention.
- proposed provenance-aware adapter: EO+context teacher와 EO-only student 둘 다.
- context-only, location/year-only.

모두 같은 AOI, acquisition IDs, band intersection, labels, split, head/search GPU-hours를 사용한다.

#### Native-ceiling 표

- Galileo 또는 TerraMind의 지원 S1/S2/DEM/weather multimodal 입력.
- GeoLink official implementation이 한국 object graph를 수용할 때의 native fusion.
- CROMA는 동시 S1/S2 coverage가 충분할 때만.

native-ceiling의 추가 sensor/modalities 이득을 pretraining 우월성으로 해석하지 않는다.

### 평가 계약

- label fractions: `1/5/10/25/50/100%`, seed 최소 5개.
- 시간: 과거 train, 중간연도 validation, 최신연도 test.
- 지역: 한 지역 완전 holdout과 region-macro/worst-region을 함께 보고한다.
- 통계 단위: pixel이 아니라 site/event 또는 spatial block clustered bootstrap.
- 모델 선택·normalization·calibration은 train/validation만 사용한다.

| 주장 | primary metric | 함께 보고할 것 |
|---|---|---|
| 표현 강화 | EO-only student frozen-probe AUPRC, labels-to-target | context-only/location-only/shuffle control |
| fusion 이득 | event AUPRC, macro-F1 | fixed-recall false alarm, trainable params/GPU-hours |
| cloud 강건성 | high-cloud AUPRC·false alarm | clear 성능 손실, S1/weather/source ablation |
| OOD | future-year·held-out-region AUPRC | ID→OOD drop, worst-region |
| 누락 강건성 | natural coverage strata의 AURC | source dropout curve, conflict/error strata |
| 선택적 결정 | risk–coverage, AURC, Brier/ECE | unsupported-cause rate, abstention rate |

### main-claim promotion gate

| 주장 | 사전 gate |
|---|---|
| EO 표현 강화 | EO-only student `+2%p AUPRC` 또는 label 20% 절감, CI>0, ≥2/3 지역 같은 방향 |
| 추론 fusion | best simple fusion 대비 `+2%p AUPRC` 또는 같은 AUPRC에서 label 20% 절감 |
| high-cloud | fixed-recall false alarm 15% 감소 또는 AUPRC +3%p, clear 손실 ≤1%p |
| 지역/시간 OOD | worst-region +3%p 또는 ID→OOD gap 20% 상대 감소 |
| 자연 누락 | baseline 대비 AURC 10% 감소, 어떤 coverage group도 2%p 초과 악화 없음 |
| 경량 적응 | trainable params ≤2%, full FT 대비 -1%p 이내 또는 frozen보다 +2%p |
| 누출 반증 | shuffle/time-shift에서 주장 이득의 ≥80% 소멸; future sentinel 100% 차단 |

평균 하나가 gate를 넘더라도 clear/region/missingness worst group이 악화되면 main claim으로 승격하지
않는다.

## 논문을 메인 트랙으로 만들 최소 범위

### CVPR/ICCV main-track 형태

1. 동적 public-context benchmark와 prospective provenance contract.
2. provenance-aware adapter + EO-only privileged distillation.
3. 3지역·미래연도·구름·자연 누락에서 strong simple/main-track baselines 비교.
4. 최소 두 task 또는 한 한국 primary task + 외부 benchmark 재현.

K-ALIGN 통합본으로 승격할 때는 여기에 Olmo v1/v1.2+TerraMind의 `S0/S0`, `S1/S1`, `S1/S0`,
`S0/S1` query/gallery matrix와 full-backfill 대비 refresh bytes/latency Pareto가 추가된다. public
fusion 수치만으로 compatible representation 논문이라 부르지 않는다.

외부 재현 후보는 PANGAEA dense task 또는 MMEarth-Bench의 공개 protocol이다. MMEarth-Bench는
arXiv에 ECCV 2026 게재가 표기돼 있지만 공식 proceedings와 우리 재현을 확인하고, 논문의 유일한
외부 근거로 의존하지 않는다.

### NeurIPS Datasets & Benchmarks 형태

방법 이득이 약하더라도 3지역 이상, 충분한 event/time coverage, 자연 missingness·publication-lag,
독립 라벨, 재현 가능한 source snapshot을 갖추면 benchmark 논문으로 전환할 수 있다. 단, 한국 API
목록을 많이 붙인 것만으로는 부족하며 “누락·지연이 모델 평가를 어떻게 바꾸는가”를 정량화해야 한다.

### 이번 범위에서 빼는 것

- 실제 비반출 기관 3곳이 생기기 전 federated learning.
- P0 transfer headroom을 보기 전 active-label method 신기여.
- 새 untouched test 없이 full-216 sealed 결과를 본 뒤 고른 nonlinear cache bridge.
- 원인규명, 전국 실시간 서비스, 한국 전체 prevalence.

## 실행 순서

| 단계 | 기간 | 산출물 | 중단 조건 |
|---|---:|---|---|
| C0 source-role freeze | 3일 | source×role×cutoff×license manifest, leak tests | publication/observed time을 구분 못함 |
| C1 3-region frame | 1주 | ≥10k parcel/site-years, natural coverage table | 두 번째 지역 join 실패 |
| C2 P0 context headroom | 1주 | 7 baseline×controls×future validation | P0 GO 미달 |
| C3 blind label pilot | 1주 | 120 이중판독, 합의도·비용 | agreement <0.60 |
| C4 supervised scale-up | 2–4주 | 1,200 labels, sealed 400 | split/leak/coverage contract 실패 |
| C5 main matrix | 2주 | paired/native tables, 5 seeds, CIs | simple fusion을 이기지 못함 |

첫 GPU 작업은 C0/C1 manifest와 leak test가 통과한 뒤 `C2`만 실행한다. 현재 서버가 비어 있거나
GPU0를 쓸 수 있다는 사실은 잘못 정의된 target이나 미래 누출을 보완하지 못한다.

## 허용·금지 주장

현재 허용:

- 이 설계가 정적 map fusion과 동적 public-evidence adaptation을 분리한다.
- full-216 결과 때문에 cross-release cache compatibility와 downstream context utility는 별도
  estimand로 측정해야 한다.

현재 금지:

- 한국 공공데이터로 OlmoEarth 임베딩/정확도가 향상됐다.
- cloud robustness, negative transfer 감소, Korea generalization이 입증됐다.
- 행정 no-match가 실제 사건 부재를 뜻한다.
- FL이 privacy를 보장하거나 이 데이터에 필요하다.
