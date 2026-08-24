# K-EvidenceShift — 한국 시공간 전이·근거·라벨 획득 벤치마크 설계

최종 갱신: 2026-08-23

상태: **설계 + 제주 audit pilot v0 실행**. 아래 방법·기준은 실험 전 계약이며, 현재 pilot은
성능 benchmark나 논문 결과가 아니다.

관련 문서: `RESEARCH_STRATEGY.md`, `KOREA_PUBLIC_DATA_CATALOG.md`,
`PAPER_READING_LIST.md`, `K_CONTEXT_FUSION_EXPERIMENT.md`, `EARTHROUTE_PROGRAM_NOTE.md`

## 먼저 약점부터

현재 실측은 이 논문의 가능성을 보여주지만 아직 성능 주장을 지지하지 않는다.

- SCL BestClear는 제주 한 window에서 bad-pixel proxy를 95.64% 줄였지만, 여러 지역·연도의
  downstream 성능 향상은 아직 측정하지 않았다.
- 4기간·12기간 OlmoEarth 점수가 함께 높았던 오름 8건은 모두 같은 구름·해무를 공유한
  거짓 양성이었다. 모델 합의나 높은 confidence만으로 라벨 우선순위를 만들면 오염을 더 모을 수 있다.
- canonical v3 공공 API snapshot은 HTTP 463/463, 의미상 성공 456, valid empty 1,
  과거 GK2A 오류 6이었다. 기존 14후보는 모두 보류됐다. 공공데이터는
  현재 **정확도를 높였다는 증거**가 아니라 근거 coverage와 보류 이유를 더 잘 측정한 증거다.
- 제주 1지역, 원인 A/B급 근거 0/368, 독립 사람 라벨과 확률표본 부족 상태다.
- 과거 GK2A cloud 자료는 현재 endpoint의 최근 2일 제한 때문에 확보하지 못했다. 현 상태를
  `실시간 한국 시계열`이라고 일반화하지 않고 source별 `event-time / snapshot / near-real-time /
  manual archive`로 구분한다.
- 중앙에서 내려받을 수 있는 공개데이터만으로는 연합학습(Federated Learning, FL)의 필요성이 없다.

## 0주차 실행 결과 — 무엇이 실제로 생겼는가

`artifacts/benchmarks/k_evidence_shift_jeju_pilot_v0/`에 첫 실행 가능한 audit pilot을 만들었다.

- algorithm-selected 후보 14건, 500 m 기준 독립 공간그룹 13개, materialized-window 그룹 8개다.
- assistant visual pre-annotation은 record 기준 `change 6 / no-change 5 / uncertain 3`, 독립 site 기준
  `5 / 5 / 3`이다. 이는 ground truth와 유병률 추정에 쓸 수 없다.
- 공식 사건 보강 근거 0/14, cause label 0/14, PNU 충돌 1건, 보류 14/14다.
- scene graph는 하나로 연결돼 cloud/quality task의 scene-disjoint split을 만들 수 없다.
- 2023→2024처럼 이른 transition의 2025·2026 RGB는 `future_after_t1_review_only`로 내리고,
  API snapshot과 공개시점 미동결 자료도 prospective input에서 차단했다.
- 동일 site, 500 m buffer, 공유 materialized window, 동일 Sentinel scene, 동일 PNU가 split을
  넘는지 검사하며 현재 결과는 `pass_for_audit_pool_not_a_train_test_split`이다.

### 2026-08-23 시간계약 재감사 — 후보보다 먼저 입력을 막는다

- `jeju25`와 `jeju26r`는 2025-07-01~2026-01-01의 184일을 공유하므로 `2025→2026` annual
  transition으로 사용할 수 없다.
- 4기간 경로는 2023–2025의 9–12월과 rolling-2026의 3–6월을 비교했고,
  `model_first4_season_aligned_across_years=false`다.
- 12기간도 `all12_cover_same_calendar_month_set=false`이므로 현재 파일을 canonical annual-change
  입력이라고 부르지 않는다.
- 4/12기간 Top-30 Jaccard는 0.091이며, 14후보 중 중첩 전이 5·4기간 source 5·합집합 9건이
  contract-exposed다. 이는 9건 모두 시각 false positive라는 뜻이 아니라 annual-change claim의
  lineage가 무효라는 뜻이다.
- 앞으로 `time-window overlap=0`, 동일 계절 support, 실제 acquisition list/hash가 후보 산출 전에
  통과하지 않으면 점수·후보·공공근거 결합을 만들지 않는다. 768 feature에서 월별 축을 사후
  복원하지 않고 valid input으로 encoder를 다시 실행한다.
- 입력 SHA, builder SHA, output SHA manifest, completion marker를 남기고 로컬 64개 전체 테스트 중
  62개 통과, 선택적 geospatial/raster dependency 2개가 skip됐다. 서버에서는 rasterio를 포함한
  release 계약·synthetic 16-GeoTIFF end-to-end 테스트 15/15가 통과했다.

OlmoEarth release P0와 full label-free audit도 완료했다.

- 제주 54 window × 4년 = 216 site-years, 인접연도 162 site-events를 재검증했다.
- 사람·행정 라벨을 보지 않고 2023/2026의 clear/contaminated proxy 양끝에서 smoke 8개를 고정했다.
- smoke 8개의 tensor/metadata 208파일을 SHA-256으로 고정하고, 원본을 수정하지 않는 96개
  symlink dataset view를 만들었다.
- v1 Base는 HF commit `93589e2d…`, weight SHA `551c1cc5…`; v1.2 Base는 commit
  `581aa9ba…`, weight SHA `57f7b66f…`로 고정했다.
- smoke 16/16 뒤 216 site-years×2 release=432 output을 GPU0에서 순차 생성하고 입력 5,616파일과
  출력 432파일의 SHA·grid·mask·finite/nonzero를 봉인했다.
- sealed 16위치에서 same-release R@1은 1.0, raw cross-release R@1은 양방향 0.0이었다. 최선
  affine ridge도 0.6973/0.6089로 사전 0.95 gate를 실패했다. pooled site-year CKA 0.9786은
  패널의 관계적 구조가 남는다는 기술통계이며 downstream task나 cache 호환성 증거가 아니다.

따라서 현재 새로 말할 수 있는 것은 **실험 계약·입력 pairing·release representation-proxy 실패가
재현 가능해졌다**는 것뿐이다. 전이 성능, 공공 context의 정확도 개선, negative transfer 감소는
여전히 미검증이다.

따라서 지금 말할 수 있는 것은 “한국 데이터를 붙이면 좋아진다”가 아니라 다음이다.

> 한국의 지역·연도·센서·구름·공식근거 이동에서 글로벌 Earth foundation model의 전이 효과와
> 실패를 측정하고, 제한된 한국 라벨을 어디에 추가해야 그 실패를 가장 효율적으로 줄이는지
> 검증할 수 있는 실험 설계가 생겼다.

## 2026-08-23 논문 pivot — 가장 깨끗한 중심 질문

최신 경쟁 문헌과 full audit 뒤 Paper A의 질문을 다음으로 좁혔다.

> **시점·출처·coverage가 명시된 한국 공공 context를 글로벌 Earth foundation model의
> privileged supervision과 inference-time context로 결합하면, 영상-only보다 라벨 효율·지역/
> 미래연도 OOD·high-cloud 성능을 높이면서 자연 누락·지연 아래에서도 이득을 유지하는가?**

전이 실패 지도와 active-label acquisition은 이 질문의 headroom이 확인된 뒤의 분석/후속 실험이다.
Paper A의 authoritative method·split·baseline·promotion gate는
`K_CONTEXT_FUSION_EXPERIMENT.md`에 고정한다. 이 문서는 site-event benchmark와 evidence/coverage
schema의 authoritative contract로 유지한다.

이 질문에서는 한국 공공데이터가 세 역할을 맡는다.

1. **target domain 정의**: 필지·행정구역·토지피복·기상·관측시점으로 한국의 shift를 구조화한다.
2. **약한 근거와 query 후보**: 건축·환경·농지 사건을 사람이 볼 후보와 라벨 종류로 바꾼다.
3. **결정 검증**: 공식 근거가 없거나 충돌할 때 모델이 보류하는지 측정한다.

공공데이터 자체가 항상 정답은 아니다. 같은 source를 입력과 정답으로 동시에 써서 성능을
부풀리지 않으며, 물리적 변화 정답은 독립 시점 영상과 사람 판독으로 확인한다.

## 두 논문과 하나의 박사 프로그램

한 CVPR 논문에 transfer, active learning, evidence missingness, release compatibility, FL을 모두
주기여로 넣지 않는다.

| 범위 | 중심 기여 | 나머지 요소의 역할 |
|---|---|---|
| **Paper A — CVPR/ICCV main stretch** | provenance-aware public-context adapter + EO-only privileged distillation + 자연 누락 benchmark | negative transfer는 subgroup 분석, v1↔v1.2는 stress test |
| **Paper B — E&D/TMLR/후속** | 시점이 있는 행정근거의 누락·지연·충돌 아래 evidence-aware selective detection | Paper A 모델과 라벨을 재사용, PPI·release continuity를 확장 |
| **박사 프로그램** | 어떤 관측·라벨·근거·재계산을 다음에 살지 결정하는 Decision-Continuous Earth Intelligence | 실제 기관 silo가 생기면 federated adapters를 추가 |

Paper A의 가칭은 다음이 가장 설명적이다.

> *Context Under Coverage: Provenance-Aware Public Data Adaptation for Earth Foundation Models*

`K-EvidenceShift`는 데이터/평가 자산명으로 유지한다. 투고 전 유사 이름을 다시 검색한다.

## 주장 사다리

서로 다른 개선을 한 문장으로 합치지 않는다.

| 단계 | 묻는 질문 | 주 지표 | 현재 근거 |
|---|---|---|---|
| L0 연결 | source를 시공간적으로 재생성·결합했는가? | semantic success, row/geometry/time coverage | v3 API 456/463 의미상 성공, snapshot별 제약 보존 |
| L1 입력 | 구름·결측이 실제 모델 입력에서 줄었는가? | bad-pixel/cloud strata, item/pixel hash | 한 golden window만 확인 |
| L2 표현 | 한국 task에 유용한 embedding인가? | frozen probe, CKA, neighbor stability | 양식장 smoke 외 정식 라벨평가 없음 |
| L3 예측 | scratch·일반 모델보다 한국 holdout에서 좋은가? | event AUPRC, mIoU, worst-group, calibration | 미검증 |
| L4 선택 | 같은 label budget에서 실패를 더 빨리 줄이는가? | learning curve/AULC, labels-to-target, worst-group regret | 미검증 |
| L5 결정 | 말한 사례의 위험과 모집단 결론이 유효한가? | AURC, risk@coverage, PPI/CI, evidence coverage | 라벨·확률표본 부족 |

`embedding이 좋아졌다`는 L2/L3의 paired holdout 결과가 있을 때만 쓴다. 공공근거를 late fusion이나
검증에만 사용했다면 개선 대상은 embedding이 아니라 prediction/decision이다.

## 벤치마크 단위와 provenance

기본 표본 단위는 픽셀이 아니라 다음의 **site-event**다.

`(entity geometry, t0, t1, EO input revision, public-evidence snapshot time)`

최소 필드:

| 묶음 | 필드 |
|---|---|
| identity | `entity_id`, geometry/PNU, admin region, ecological/land-cover stratum |
| time | `t0`, `t1`, sensor acquisition times, public-record event/observed/updated/retrieved times |
| EO | item IDs, bands, masks, compositor, tensor hash, CRS/GSD, cloud/nodata statistics |
| model | model ID, revision/weight hash, code commit, normalization, output schema |
| evidence | source ID, snapshot hash, join type, geometry/time gap, availability and no-match reason |
| labels | `visual_change`, `official_event_supported`, `evidence_available`, `cause`, annotator/conflict |
| cost | annotation minutes, travel/manual retrieval cost, bytes, CPU/GPU seconds |

라벨 네 개를 분리한다.

- `visual_change`: 독립 전후 영상에서 관찰 가능한 변화.
- `official_event_supported`: 사전 고정 공간·시간창에서 공식 사건과 중첩.
- `evidence_available`: 그 source가 그 지역·기간을 실제로 포괄하는지.
- `cause`: 공식 geometry/time과 독립 판독이 모두 맞을 때만 제한적으로 부여.

행정자료의 no-match는 `visual_change=0`이나 `cause=none`이 아니라 `unknown`일 수 있다.

## P0와 P1을 섞지 않는다

### P0 — 1주 release/input audit

현재 즉시 가능한 주 실험은 정확도표가 아니라 다음 2×2 paired audit이다.

| 축 | 수준 | 현재 상태 |
|---|---|---|
| 입력 recipe | legacy 12-period / SCL BestClear 12-period | legacy 216 완료, BestClear 전수 미완료 |
| release | OlmoEarth v1 Base / v1.2 Base | immutable checkpoint 고정 완료 |
| timestamp | legacy dummy timestamps | release-only primary track으로 명시 |
| 단위 | 216 site-years / 162 adjacent-year events | metadata manifest 완료 |

primary outcome은 site-event accuracy가 아니라 Top-1% 후보의 input effect와 release effect다.

`(1 - Jaccard_recipe) - (1 - Jaccard_release)`

54 spatial window를 cluster로 bootstrap한다. secondary로 raw/row-normalized CKA와 spatial-shift
null, neighbor overlap@50, Kendall tau와 GPU-second/window를 보고한다. Procrustes는 8건 smoke에는
금지하고, 216건에서 spatial calibration/test split이 생긴 뒤 train-only fit으로만 평가한다. P0는 **accuracy improvement나 negative transfer를
증명하지 않고**, P1을 열 headroom과 release/cache drift가 있는지만 판정한다.

### P1 — context adaptation/transfer benchmark

최소 300개 독립 라벨은 engineering gate일 뿐 최종 worst-group benchmark의 충분 표본이 아니다.
Paper A로 승격하려면 다음처럼 label pool과 모집단 test를 분리한다.

- sealed probability test 최소 300건
- train pool 별도 최소 300건; active acquisition은 main context signal 확인 뒤 후속으로 사용
- 이중판독 최소 120건과 adjudication
- 3개 지역 × 2시기 × 2 cloud strata를 보고할 때 권장 목표 총 1,200건

300건 미만에서는 scratch·GeoFM accuracy 표를 main result로 만들지 않고 release/representation
stress test와 annotation asset만 보고한다.

## 데이터 역할과 누수 금지

각 source는 한 실험 셀에서 아래 역할 중 하나만 맡는다.

| 역할 | 예 | 금지 |
|---|---|---|
| 사전 입력 | t0 이전 토지피복·지형·과거 기상·필지형상 | t1 이후 갱신정보를 prospective prediction에 투입 |
| 약한 라벨/공식 근거 | t0–t1 건축·환경·농지 사건 | 같은 event record를 입력과 정답에 동시 사용 |
| 독립 감사 | NGII 전후 항공사진, 블라인드 사람 판독, 현장 검증 | 모델/행정근거를 본 뒤 label 정의 변경 |
| coverage metadata | endpoint 모집단·기준일·누락 사유 | API 0행을 사건 부재로 처리 |

현재 source 후보는 Sentinel-1/2, SCL, KMA/GK2A, VWorld 연속지적, BuildingHUB, 환경영향평가,
토지피복, FarmMap, NGII 항공사진이다. 사용 전 `KOREA_PUBLIC_DATA_CATALOG.md`의 라이선스,
공간단위, 시간 의미, 재배포 가능성을 다시 고정한다.

## Task A — 공공 context 적응과 전이 효과

### 비교 질문

1. train-time public context supervision 뒤 test에서 EO만 쓰는 student가 frozen Olmo보다 적은
   한국 label로 같은 성능에 도달하는가?
2. test에서도 context를 쓰는 fusion 이득이 location/year shortcut이나 label-source leakage와
   구분되는가?
3. 평균에서는 좋아 보여도 지역·미래연도·구름·자연 coverage 중 어디서 손해를 만드는가?

세 estimand를 분리한다.

- `E_repr`: privileged context train → EO-only test.
- `E_fusion`: EO+context train/test → 영상-only baseline.
- `E_decision`: 영상 예측 뒤 evidence/coverage를 써서 선택적 보고.

source별 time/coverage token, proposal architecture, simple-fusion control과 정확한 gate는
`K_CONTEXT_FUSION_EXPERIMENT.md`를 따른다.

그룹 `g`와 label budget `b`에서 전이 효과를 다음처럼 고정한다.

`Delta(m, g, b) = score(pretrained m + adaptation; g, b) - score(matched scratch; g, b)`

- compute, decoder, input bands/timestamps, augmentation, hyperparameter-search budget을 맞춘다.
- `Delta < 0`인 점만으로 확정하지 않는다. site/event 단위 paired spatial bootstrap 95% CI가
  0 아래일 때 `confirmed negative transfer`로 부른다.
- 평균, worst-region, worst-year, high-cloud, rare-class Delta를 모두 보고한다.

### 모델 매트릭스

| 층 | 최소 모델 | 목적 |
|---|---|---|
| task-specific | U-Net/temporal CNN 또는 ChangeFormer, ViT scratch | GFM 우월성을 가정하지 않는 강한 기준 |
| generic vision | ImageNet/self-supervised vision backbone 1개 | EO pretraining 자체의 가치 분리 |
| Olmo release | OlmoEarth v1 Base, v1.2 Base | 같은 tensor의 release drift 통제 |
| simple context | location/year-only, context-only, late concat, STACK/TOKEN-FUSE | shortcut와 hard-coded fusion 기준 |
| context method | provenance-aware adapter, EO-only distilled student | `E_fusion`과 `E_repr` 분리 |
| object-map fusion | GeoLink-style object-patch cross-attention | 정적 지도 결합의 강한 main-track 기준 |
| temporal EO | Prithvi-EO-2.0 | 시계열 transfer 비교 |
| cloud/multimodal | CROMA 또는 TerraMind | S1–S2/다중모달이 제주 구름을 보완하는지 |
| 확장 | AnySat | 센서·해상도 유연성이 실제 headroom을 없애는지 |

모델마다 native input이 다르므로 결과표를 두 track으로 나눈다.

- **paired-input track**: 공통으로 소비 가능한 S2 tensor와 동일 decoder/protocol.
- **native-ceiling track**: 각 모델의 공식 최적 modality/recipe. 더 좋은 결과여도 architecture와
  추가정보 효과를 분리해 주장하지 않는다.

적응법은 frozen probe, target self-supervised adapter, LoRA/adapter, full fine-tuning을 비교한다.

첫 2주의 구현 순서는 더 좁다.

1. exact tensor의 OlmoEarth v1/v1.2 frozen release audit
2. TerraMind Small adapter smoke 후, label gate를 통과하면 Base frozen comparator
3. Prithvi-EO-2.0 300M은 6-band·30 m compatibility smoke만; 공통 view를 동결하기 전 점수표 금지
4. CROMA는 유효한 제주 S1–S2 pairing을 확보한 뒤 native-ceiling track에 추가
5. scratch·generic model 성능표는 독립 라벨과 sealed split 뒤에만 실행
6. FL 코드는 실제 비공유 기관 silo 3곳이 생기기 전까지 구현하지 않음

## Task B — 후속 제한 예산 active target-label acquisition

Task A의 context/transfer headroom이 통과한 뒤에만 연다. 목표는 라벨 수를 많이 늘리는 것이 아니라,
한국 target risk를 가장 크게 줄이는 라벨을 고르는 것이다. Paper A의 필수 방법 기여는 아니다.

### Earth용 목적함수

PDE 연구의 `beta`, advection–diffusion, 모호성선, D-opt 해석을 그대로 옮기지 않는다. 여기서
실험 단위와 estimand는 다음처럼 새로 정의한다.

| PDE에서 익숙한 표현 | Earth에서 새로 정의할 것 |
|---|---|
| 물리 parameter `beta` | 지역·연도·센서·구름 그룹별 **전이 효과 Delta** |
| advection–diffusion dynamics | 실제 물리과정 모델이 없는 한 사용하지 않음; EO observation process와 event time을 기록 |
| 모호성선 | 물리적 경계가 아니라 모델·릴리스·센서 간 **경험적 disagreement region** |
| D-optimality | 식별성 보장이 아니라 embedding diversity를 고르는 한 baseline |

query는 `(site-event i, label/evidence action a)`이고 비용 `c(i,a)`를 갖는다. 목표는 고정 test
distribution에서 label budget 이후 worst-group risk와 AURC를 줄이는 것이다.

`min_Q  worst_group_risk(f[L union Q]) + lambda * AURC(f[L union Q])`

`subject to sum c(i,a) <= budget`

### acquisition baseline

1. random
2. 지역·연도·cloud 층화 random
3. uncertainty only
4. Olmo v1/v1.2 disagreement
5. cross-family disagreement(Olmo/Prithvi/CROMA 또는 TerraMind)
6. embedding k-center/core-set
7. D-opt/log-det diversity baseline
8. CLUE형 uncertainty-weighted clustering
9. cost-aware spatial acquisition(`Mapping on a Budget` 계열)
10. 제안 후보: quality-gated shift × disagreement × evidence-gap × diversity / cost

현재 8/8 cloud false positive 때문에 disagreement를 바로 query score로 쓰지 않는다.

- 먼저 clear/high-cloud/persistent-cloud를 분리하고, 오염 자체를 감사할 query와 task label query를
  구분한다.
- 같은 S2 입력을 공유한 모델의 합의는 독립 증거가 아니다. 가능한 경우 S1, 다른 모델 family,
  다른 시점, 사람 판독을 섞어 common-mode error를 줄인다.
- 한 지역의 경계 tile을 반복 선택하지 않도록 spatial diversity와 site-level deduplication을 둔다.

### 평가 계약

- seed set과 batch budget을 사전 고정하고 예: 1/5/10/25/50/100% learning curve를 그린다.
- 모든 acquisition method는 같은 unlabeled pool, 같은 oracle label, 같은 adaptation/search budget을 쓴다.
- 5개 이상 seed와 site-level bootstrap으로 AULC(area under learning curve), labels-to-target,
  event AUPRC, worst-group gap, AURC를 비교한다.
- adaptive labels는 편향된 표본이다. 별도의 봉인된 spatial-temporal test와 층화 확률표본을 유지하며,
  active set만으로 한국 전체 변화율이나 PPI 신뢰구간을 추정하지 않는다.

## Task C — evidence-aware selective detection

이미지 모델의 confidence와 공식근거 availability를 따로 입력한다.

`image only -> + S1/cloud/weather -> + t0-valid state -> + official event -> + independent audit`

각 단계에서 event AUPRC, fixed-recall false alarms, Brier/ECE, risk–coverage/AURC,
evidence-supported precision/recall, abstention rate, high/low/missing-evidence worst-group gap을 측정한다.

핵심 negative controls:

- evidence row를 다른 지역에 shuffle
- event timestamp를 미래/과거로 shift
- geometry를 동일 거리의 비사건 필지로 swap
- source availability mask만 주고 실제 내용은 제거

이 controls를 통과해야 `공공데이터의 정보가 도움`이라고 말할 수 있다. 행정구역 이름이나
coverage pattern만 학습해도 좋아지는 경우는 데이터 편향 발견이지 원인 근거 결합 성공이 아니다.

## Task D — OlmoEarth release와 cache 연속성

공개 계열의 정확한 비교명은 현재 `v1 -> v1.1 -> v1.2`다. 사용자가 말한 `v1 -> v2`는
프로젝트 내부 파이프라인 버전과 모델 릴리스를 분리해 기록한다.

동일 tensor에서 다음을 비교한다.

- old-query/old-gallery, new-query/new-gallery, new-query/old-gallery
- CKA, neighbor overlap, top-k Jaccard/Spearman, prediction disagreement. Procrustes·linear bridge는
  spatial calibration split에만 fit하고 별도 held-out gallery에서 평가한다.
- calibration, active query overlap, public-evidence 결합 후 최종 결정 변화

평균 downstream score가 같아도 query pool이나 조사 우선순위가 크게 바뀌면 drop-in decision
replacement가 아니다.

## split과 leakage 방지

| 축 | 계약 |
|---|---|
| 공간 | 같은 parcel/oreum/event와 buffer 안 인접 tile은 한 split에만 |
| 시간 | 과거 train, 다음 연도 validation, 최신 연도 prospective test |
| 사건 | 한 사건의 before/after와 중복 record는 한 split에만 |
| 지역 | 제주 내부 spatial holdout + 사전 고정 내륙/도시 또는 산림 지역 2곳 |
| cloud | clear/thin/heavy/persistent를 label 전에 층화 |
| evidence | high/low/missing/conflicting coverage를 분리; no-match를 negative로 바꾸지 않음 |
| release | 같은 tensor를 v1/v1.2에 넣고 manifest hash로 paired 여부 확인 |
| active loop | test label은 query 가능 pool에서 영구 제외 |

random pixel split은 사용하지 않는다. 표본 수와 CI의 단위도 pixel이 아니라 site/event다.

## 전체 지표

- **입력**: cloud/nodata fraction, valid observations, observation latency, tensor hash change.
- **task**: event AUPRC/F1, mIoU, fixed-recall false-positive rate, change-time error.
- **전이**: Delta, EarthShift-style effective robustness, ID->OOD drop, rare-class/worst-group.
- **active label**: AULC, labels-to-target, redundancy, geographic coverage, cost-to-target.
- **신뢰성**: Brier, ECE, risk–coverage, AURC, evidence precision, abstention.
- **release**: neighbor/top-k/rank/prediction/query-set stability.
- **운영**: label/person-hours, bytes, CPU/GPU seconds, trainable params, p95 latency.
- **모집단**: 별도 확률표본이 있을 때만 PPI/설계기반 bias, CI width, nominal coverage.

## Federated learning 승격 조건

공개데이터를 17개 시도로 나누는 것은 `synthetic geographic clients`이지 실제 cross-silo FL이 아니다.
아래 조건을 모두 만족할 때만 Paper A 본문에 넣는다.

1. 원시영상·현장라벨·고해상도 자료를 공유할 수 없는 독립 기관 3곳 이상.
2. 기관별 지역·센서·라벨 분포가 다르고 실제 분산 실행을 최소 한 번 수행.
3. 중앙 pooled 학습이 불가능한 법·계약·비용 근거와 threat model.
4. local-only, pooled upper bound, FedAvg, FedProx/SCAFFOLD adapter, personalized head와 비교.
5. macro/worst-client, unseen-region, upload/download bytes를 보고. privacy는 DP나 secure aggregation의
   실제 구현 없이 주장하지 않음.

승격 기준 후보는 `local-only 대비 macro +2%p`, `worst-client 저하 <=1%p`, `pooled와 <=2%p`,
`full-model FedAvg 대비 통신 >=10x 감소`이며 site-level 95% CI를 요구한다. 실패하거나 실제 silo가
없으면 FL은 부록의 simulation 또는 후속 연구로 내린다.

## 기존 benchmark를 재사용하는 법

| 자산 | 재사용 | K-EvidenceShift가 추가할 것 |
|---|---|---|
| GEO-Bench-2 | capability grouping, common task harness | 한국 시공간·근거·release shift |
| PANGAEA | dense task, frozen encoder+decoder, label fractions | active acquisition과 selective risk |
| EarthShift | geography/time/sensor/scale/source OOD protocol | administrative evidence missingness와 model-release shift |
| AllClear/CloudSEN12 | cloud strata·S1/S2 품질 평가 | 복원 PSNR이 아니라 downstream change/AURC |
| Copernicus-Bench | cloud/Sentinel task와 fixed splits | 한국 official evidence와 decision abstention |
| FedRS-Bench | 실제 출처·지역 client FL 비교 | 시간변화·dense task·근거 보류; 실제 silo가 있을 때만 |
| GeoLink | EO–OSM object/region alignment와 object-patch fusion | 동적 행정기록의 publication time·자연 누락·conflict, EO-only distillation |
| MMEarth | multimodal pretext로 optical representation 강화 | prospective cutoff와 source-role leakage, 한국 지역·미래연도 OOD |
| Galileo/TerraMind | native weather/SAR/DEM multimodal ceiling | 동결 Olmo에 붙는 경량 adapter와 public-record provenance |
| SatMIP | location/time metadata alignment | location/year shortcut·shuffle/time-shift 반증 control |

새 benchmark 전체를 처음부터 다시 만들지 않고 GEO-Bench-2/PANGAEA adapter를 우선 구현한다.

## 사전 promotion·kill gate

| 주장 | promotion gate | 실패 시 |
|---|---|---|
| GeoFM 전이 이득 | scratch 대비 +2%p 또는 동일 성능 label 50% 절감, CI>0, 3지역·2연도 방향 반복 | task-specific 결과로 축소 |
| negative transfer atlas | 사전 그룹에서 CI<0인 실패가 반복되고 데이터/compute mismatch로 설명되지 않음 | 단순 평균 benchmark로 축소 |
| active acquisition | stratified random/CLUE 대비 AULC 개선, worst-group 악화 없음, 5 seeds | 방법 기여 삭제, 데이터 설계 분석만 보고 |
| 공공데이터 결합 | image-only 대비 AUPRC +2%p 또는 same recall FP -10%, clear 성능 -1%p 이내 | accuracy 주장을 지우고 coverage/audit로 전환 |
| selective detection | matched coverage에서 AURC 감소, 봉인 test와 독립 calibration 유지 | confidence dashboard로만 제한 |
| release compatibility | cross-version retrieval이 new/new의 95% 이상, 새 모델 성능 -1%p 이내 | full refresh 필요 사례로 보고 |
| CVPR main | 6주차까지 multi-model baseline+300-label audit+재생성 계약 완료 | IGARSS/TMLR/EarthVision 경로로 전환 |

수치는 시장·문헌 사실이 아니라 비교를 사후 조정하지 않기 위한 내부 기준이다.

## 12주 실행 순서

1. **1주** — site-event schema, label taxonomy, target distribution, forbidden leakage, model/input
   revision을 동결한다.
2. **2주** — 제주 포함 3지역의 API·영상·라이선스·재생성 계약과 300건 annotation pilot pool을 만든다.
3. **3–4주** — 블라인드 이중판독, agreement, evidence high/low/missing 비율을 측정한다.
4. **5–6주** — scratch/generic vision/Olmo v1·v1.2/Prithvi/CROMA 또는 TerraMind baseline을 같은
   split으로 실행한다. 이때 CVPR go/no-go를 결정한다.
5. **7–8주** — frozen/PEFT/full FT와 active acquisition offline replay를 5 seeds로 비교한다.
6. **9주** — evidence-conditioned fusion·abstention. 실제 silo가 있을 때만 federated adapter.
7. **10주** — cloud/season/source missingness/release stress와 shuffled-time/geometry controls.
8. **11주** — datasheet, source license, manifest, 익명 code, 재배포 가능 sample.
9. **12주** — 독립 재실행, 표·그림·claim 동결.

2026-08-22 현재 CVPR 2027의 공식 paper deadline은 발표되지 않았다. 2026 일정을 2027 사실로
사용하지 않고, 내부적으로 2026-10-31 초록 동결·2026-11-06 제출 가능본을 목표로 한다.

## 첫 번째로 만들 표

코드를 확장하기 전에 아래 빈 표를 먼저 채운다.

| model/adaptation | Jeju clear | Jeju cloud | Jeju future | Region-2 | Region-3 | worst group | labels | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scratch | | | | | | | | |
| generic vision | | | | | | | | |
| Olmo v1 frozen | | | | | | | | |
| Olmo v1.2 frozen | | | | | | | | |
| Olmo v1.2 PEFT | | | | | | | | |
| Prithvi PEFT | | | | | | | | |
| CROMA/TerraMind PEFT | | | | | | | | |

이 표에서 실제 negative transfer와 label-efficiency headroom이 확인된 뒤에만 새 acquisition method를
구현한다. headroom이 없으면 active learning을 붙일 이유도 없다.
