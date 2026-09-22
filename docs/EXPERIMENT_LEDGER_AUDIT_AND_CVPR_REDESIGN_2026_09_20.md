# MS-122~149 독립 감사와 CVPR 재설계 — 시간 갱신·근거 기억·VLM

문서 식별일: 2026-09-20. 감사 실행 환경일: 2026-09-19 KST. 상태: **로컬 코드·설정·산출물 및 1차 문헌에 대한 독립 감사; 새 실험 미실행**.

이 문서는 `EXPERIMENT_LEDGER_2026_09_20.md`를 삭제하거나 사후 수정하지 않는다. 기존 원장은
MS 색인으로 보존하되, 논문 주장과 다음 실험의 판단에는 이 감사의 정정을 우선한다. 서버의 현재
상태, 진행 중인 E 소거 arm, 원격 snapshot은 다시 조회하지 않았다. 로컬 Git만으로 모든 prereg가
실행 전에 봉인됐는지도 독립 검증하지 않았다.

## 0. 결론부터

현재 결과는 **CVPR 본회의 방법 논문으로 아직 부족하다**. 이유는 단순히 학습 arm이 많이 실패해서가
아니다. 현재 원장의 중심 주장 몇 개가 실제 코드의 정보 집합과 다르고, 온라인 갱신 평가에 미래
정보가 섞였으며, VLM의 gold와 scorer가 실제 시각 근거를 검증하지 못하고, 새 사건에 대한 일반화를
말할 독립 단위가 네 사건뿐이기 때문이다.

그러나 연구 질문은 살릴 수 있다. 시간 보간을 본체로 두기보다 다음 질문으로 좁히는 것이 가장
방어 가능하다.

> 불규칙하고 품질이 다른 EO 관측이 순차적으로 도착할 때, 관측 provenance를 보존하는 제한 용량
> 증거 기억이 full-history VLM과 규칙 로그보다 낮은 비용으로, 어떤 관측의 어느 위치가 판단을
> 바꿨는지 더 정확하게 지목하고 적절히 보류·확정·정정할 수 있는가?

가칭 방법은 **Causal Evidence Ledger for Earth Observation (CEL-EO)**로 둔다. 핵심은 또 다른 GRU나
보간기가 아니라 다음 세 가지다.

1. `observed`와 `predicted` 상태를 분리한다. 보간값은 정상 기대 상태일 수는 있어도 사건을 확인하는
   증거가 될 수 없다.
2. 서로 다른 날짜의 토큰을 한 슬롯에 평균하지 않고, 관측 ID·시각·지역·센서·품질·원본 chip을
   가리키는 불변의 전후 증거를 보존한다.
3. 답만 맞히는 것이 아니라 `답 + 근거 관측 + 근거 영역 + 보류/정정 행동`을 함께 평가한다.

이 전환 뒤에도 새 독립 사건과 사람 검증 prefix gold를 확보하지 못하면, 가장 정직한 산출물은
CVPR 본편이 아니라 강한 workshop/negative characterization 논문이다.

## 1. 감사 범위와 증거 우선순위

이번 감사에서 대조한 로컬 근거는 다음과 같다.

- 원장과 상세 기록: `docs/EXPERIMENT_LEDGER_2026_09_20.md`, `MEASURED_FINDINGS.md`.
- 입력·채점 코드: `extract_olmo_perturb.py`, `extract_olmo_streaming.py`,
  `t5_prefix_latency_v0.py`, `sequential_qa_gen_v1.py`, `earthtalk_seq_v0.py`,
  `earthtalk_seq_v2.py`, `seasonal_baseline_v0_1.py` 등.
- 등록 설정: frozen sensitivity, interpolation, P1, LOEO, sequential QA, VLM stage 1/2,
  trust, seasonal baseline 관련 JSON.
- 로컬 산출물: post-k 두 지역, T5 latency, latent interpolation v0/T4 6 runs,
  VLM stage 1/2, token calibration, trust, seasonal baseline 요약.

논문 수치의 우선순위는 `원시 행/산출물 → 실행 코드 → 해당 실행의 등록 설정 → 상세 MS 기록 → 원장
요약`으로 둔다. 코드와 이름이 충돌하면 이름이 아니라 실제 정보 집합과 계산식을 따른다.

## 2. 원장 집계 감사

### 2.1 `3/11/3/9`는 계산되지 않는다

원장의 현재 집계는 같은 단위를 세지 않는다.

| 원장 구분 | 원장 숫자 | 감사 결과 |
|---|---:|---|
| 통과 | 3 | 129는 pipeline 성립, 142는 다음 stage 승인, 148-v0.1은 empirical gate 통과다. 같은 종류의 통과가 아니다. |
| 실패 | 11 | 행에 적힌 MS ID 자체가 13개이고, 표에서 실패한 131이 빠졌다. |
| 무효 | 3 | 집계는 128·136·139지만 §4의 “설계 실수 3건”은 128·136·148로 바뀐다. 144 event 비교도 `n=3` 설계 결함이다. |
| 측정 | 9 | 열거된 ID는 125·126·127·132·133·134·138·144·145·146의 10개다. |

MS ID를 억지로 하나의 최종 상태로만 분류하면 한 가지 가능한 집계는 `통과 3 / 실패 13 /
무효 2 / 측정 10`이다. 그러나 MS-136·148처럼 한 MS 아래 유효한 실패, 무효 arm, 재등록 통과가
함께 있으므로 이 숫자도 논문용으로는 부적절하다.

다음 버전부터는 두 표를 분리해야 한다.

- **MS 최종 상태 표:** 연구 질문의 현재 상태만 한 행으로 기록.
- **atomic run/gate 표:** `run_id`, prereg version, validity, gate outcome, empirical direction,
  method family, reused test events를 한 행씩 기록.

`pass`도 `pipeline-ready`, `stage-advance`, `method-efficacy`로 분리한다. “학습 후보 8개 전부 실패”는
후보 taxonomy가 없고 MS-131이 빠지며, trust head는 구름 예측 AUC 자체는 성공하고 downstream
payoff가 실패했다. 안전한 문장은 **“등록된 end-to-end utility gate는 통과하지 못했다”**이다.

원장의 “사전등록 25건”도 scope manifest 없이는 재현되지 않는다. 현재 `config/*prereg*.json`은
draft·amendment·다른 기간 파일을 합쳐 52개다. 25개의 정확한 포함 목록과 각 실행 hash를 따로
고정해야 한다.

## 3. 논문 후보 문장별 재판정

| 원장의 문장 | 코드·산출물로 확인된 사실 | 논문에서 쓸 안전한 문장 | 상태 |
|---|---|---|---|
| 사건 후 1/2/3장 = 21/75/96%, 판단에 2장 필요 | Hiroshima의 control 대비 mean IoU 비율은 .208/.750/.958이고 Indonesia는 .215/.638/.870이다. 탐지 확률이 아니다. | 현재 pooled frozen state/readout에서는 첫 post 관측만 남길 때 control IoU 회복이 낮고, 두 번째·세 번째 관측에서 증가했다. | 정정 |
| 최신 1장만 5% | `postlast1`은 모든 pre-event 관측 + 마지막 post 한 장이다. Indonesia는 15.9%다. | 마지막 post 한 장만 추가한 pre+postlast arm은 두 지역에서 control IoU의 5~16%였다. latest-only는 별도 실험이 필요하다. | 철회·재명명 |
| F1은 11개 옛 관측 평균에 1개 신호가 1/12로 희석 | attention, pooling, head mismatch가 분리되지 않았다. | 초기 post evidence가 현재 window/readout에서 약했다. 원인은 미확정이다. | 가설로 강등 |
| 첫 탐지 중앙값 25일, 대부분 revisit interval | 코드는 사건일이 아니라 첫 선택된 post 관측부터 센다. 388개 중 detected 374개에 조건부이며 14개는 censored다. | 첫 선택 post 관측 이후 추가 확인 대기의 조건부 중앙값이 25일이었다. | 정정 |
| 반복성 상한 cos .848 | ≤10일 연속 관측의 평균 cosine이다. 실제 변화와 대기·센서 잡음이 섞였고 linear 전체 평균 .868이 더 높다. | ≤10일 인접 관측 유사도 통계 .848. 복원 상한이나 noise floor가 아니다. | 상한 주장 철회 |
| 학습 보간기는 상한까지만 감 | T4는 선택한 3개 challenge region에서 linear보다 +.0285~+.0317였다. exact split의 mean-all/seasonal baseline은 계산되지 않았다. | 등록 +.03 gate에는 대부분 근소 미달했지만 linear 대비 일관된 개선. 강한 동일-split 기준선 전에는 방법 우위 미확정. | 보류 |
| 월 내 순서 미표현 | swap은 영상만 바꾸고 timestamps는 고정하며, 전체 fold aggregate에는 no-pair fallback이 섞인다. 모델은 slot position도 갖는다. | exact intra-month 날짜·간격은 주어지지 않았고, pair cohort의 pooled frozen output/readout이 swap에 거의 불변인지 재평가가 필요하다. | cohort 재계산 필요 |
| 연도 무시·격자/월 라벨에 취약하지 않음 | 특정 perturbation과 frozen pooled state/readout에서 등록 문턱 미달이다. | 해당 계약에서 큰 decision sensitivity를 관측하지 못했다. 모델 전체 능력의 불가능성은 아니다. | 범위 축소 |
| 구름 쌍 변화가 사건보다 큼 | cloud .351 대 normal .121은 유효 기술통계지만 event .224는 3쌍뿐인 설계 결함 arm이다. | 해당 normal/cloud 표본에서 cloudy-pair score가 normal보다 약 3배 컸다. 사건과의 대소 비교는 폐기한다. | 일부 보존 |
| 토큰 z AUC .80, 순위는 이식 | 이것은 사건 전후 pair가 주어진 뒤 final mask 안/밖 localization이다. LOEO AUC는 사건별 .65~.69 수준도 있다. | known event-pair 조건의 지역 내 localization rank가 일부 전이되지만 편차가 크고 threshold는 이식되지 않았다. | 범위 축소 |
| hokkaido 과분산 원인은 계절 | target-site gap×season cross-fit에서 normal z≥2가 .145→.048, AUC .669→.717였다. 사건 후 normal pair도 fit에 들어가며 online past-only가 아니다. | target-site 계절/표면 분포 이동과 일치하고 retrospective local calibration으로 크게 감소했다. 인과 원인과 online 효용은 미확정이다. | 정정 |
| 로그 규칙이 VLM보다 구름에 안정적 | 현재 synthetic state-machine gold와 stability 정의에서 관측됐다. 안정적으로 계속 틀리는 답도 보상될 수 있다. | 현재 자동 gold/scorer 아래에서 LOG가 VLM reader보다 높은 state constancy를 보였다. 정확한 근거 안정성 비교는 재채점 필요. | 보류 |

## 4. Publication-blocking 프로토콜 문제

### 4.1 사건 네 개는 타일 수로 늘어나지 않는다

타일 bootstrap은 한 사건 내부의 공간 반복에 대한 조건부 정밀도만 보여준다. 새 사건 일반화의 표본
크기는 실질적으로 네 개다. 방향이 4/4 같아도 단순 one-sided sign test는 `p=1/16=.0625`다.
Hiroshima test hash는 P1 v0~v0.3에서 반복 사용됐고, 세 외부 사건도 MS-140 이후 여러 방법 선택에
재사용됐다. 이제 네 사건 중 어느 것도 연구 전체 관점의 untouched confirmatory event가 아니다.

새 데이터는 타일이 아니라 `trigger episode = geography × event time interval`로 색인해야 한다.
서로 인접한 타일, 같은 궤도, 같은 재난은 같은 cluster다. 사건/AOI를 바깥 cluster로 bootstrap하고,
타일 bootstrap은 “이 사건 안에서의 공간 정밀도”로만 표기한다.

Sen12Landslides 원 데이터는 15개 전 지구 inventory, 13,628개 Sentinel-2 patch, 74,956개 refined
landslide를 보고한다. 현재 네 사건 제한은 원천 데이터의 절대 한계가 아니다. 다만 inventory와 독립
trigger episode가 일치하지 않을 수 있고, NDVI drop으로 추정한 사건일은 temporal detector 평가와
순환될 수 있으므로 날짜 출처와 confidence를 분리해야 한다.

### 4.2 현재 latency는 causal online replay가 아니다

MS-125/133은 전체 15장의 SCL을 먼저 본 뒤 가장 맑은 12장을 고르고 prefix를 만든다. 이는 미래
품질을 아는 archive retrospective 실험이다. 논문용 latency는 `extract_arrival_v0.py`처럼 그 시점까지
도착한 관측 중 last-12만 사용하는 causal replay로 다시 계산해야 한다.

측정 시계도 분리한다.

- event interval 시작 → 첫 usable acquisition: observation lag.
- acquisition → 실제 이용 가능: data/arrival lag.
- 첫 관측 → first/provisional/confirmed: algorithm/confirmation lag.
- 미탐지는 right-censored로 포함한다.

`any positive pixel` 탐지율과 행 단위 pre-event positive rate .312는 운영 FAR가 아니다. 독립 event
cluster에서 `false alert episodes / km²-year`, fixed-FAR recall, survival curve, time-to-confirm을 낸다.

### 4.3 계절 calibration도 현재는 retrospective local calibration이다

MS-148 v0.1은 hash spatial half로 fit/eval을 나누지만 동일 획득 날짜·날씨·계절 schedule을 공유하며,
normal pair 정의에 사건 후 같은-side pair도 포함한다. 이는 target-site cross-fit이지 past→future online
calibration이나 zero-shot transfer가 아니다.

다음 평가는 반드시 다음 셋을 분리한다.

1. no target calibration.
2. 사건 전 target history만 쓰는 rolling calibration.
3. retrospective target-site cross-fit.

calibration에 사용한 면적·개월·관측 수를 모델 budget으로 함께 보고한다.

### 4.4 VLM gold와 scorer는 현재 grounding을 증명하지 못한다

`sequential_qa_gen_v1.py`는 final static mask가 있으면 positive로 두고, 알려진 post index 이후 첫
`clear≥.5` 관측을 confirmed로 만든다. 그 시점에 실제 산사태 흔적이 보이는지는 검수하지 않으며,
위치 gold는 final mask quadrant를 복사한다. `revise` 예제는 한 건도 없다. 따라서 이는 실제 근거
발견 벤치마크가 아니라 **event-date + cloud-availability로 만든 synthetic evidence-availability proxy**다.

또한 등록은 Q_where quadrant Jaccard인데 `earthtalk_seq_v0.py`는 모든 질문을 exact match로 채점한다.
예측 quadrant는 알파벳순으로 정렬하지만 gold 생성 순서는 NW, NE, SW, SE라 동일 집합도 오답이 될 수
있다. MS-142/143의 Q_where를 재채점하기 전에는 공간 grounding 결과로 쓰지 않는다.

`Q_when=none`이 약 74~80%라 전체 accuracy도 상수 답 shortcut에 취약하다. 최소한 다음을 다시 낸다.

- confirmed/keep 조건부 evidence-date와 quadrant score.
- no-evidence prefix의 false-date/false-region rate.
- state와 update의 macro/balanced score.
- tile별 first-evidence observation error.
- correct retention과 stale-error retention을 분리한 stability.

### 4.5 D의 실패는 evidence memory 전체를 기각하지 않는다

현재 D는 매 step 가장 오래된 K=4 slot을 무조건 선택하고, `clear + global mean change + Δt`로 만든
타일 전체 scalar gate로 현재 64 spatial token과 4-step 전 token을 혼합한다. 혼합된 내용에 최신
관측 날짜를 덮어써 provenance가 깨진다. skip-write, evidence ID, 전후 관측 분리, query-conditioned
retrieval, diversity/preservation loss가 없다.

따라서 MS-143은 **날짜 provenance가 깨지는 round-robin latent mixer**를 기각했다. 메타정보 조건부
공간 기억이나 제한 용량 증거 기억 일반을 기각하지 않았다. 더구나 C는 명시적 clear/area/quadrant를
받지만 D reader의 메타정보는 동등하지 않아 순수 memory 효과도 아니다.

## 5. 시간 보간 트랙의 정확한 위치

### 5.1 현재 T4는 hyperlocal short-gap 실험이 아니다

MS-126 전체 15,000 query 중 ≤20일은 405개, 2.7%였다. T4 exact test는 21,140 query 중 ≤20일
428개, 2.0%이고, >120일은 15,673개, 74.1%다. resMLP seed1의 linear 대비 개선도 ≤20일
`+.0170`, >120일 `+.0325`다. 현재 결과는 사용자가 말한 좁은 지역 short-gap 보간의 근거가 아니라
대부분 장기 gap representation imputation 결과다.

### 5.2 현재 cosine은 정적인 장소 정체성을 보상할 수 있다

MS-126 전체에서는 `mean_all=.8747 > date-linear=.8681`이었다. T4가 선택한 세 region의 v0 641
tile을 가중하면 mean-all은 약 .865지만, T4 exact 2,114-tile split에는 아직 계산되지 않았다. T4
모델은 약 .849~.852, linear는 .8205다. 모집단이 달라 직접 비교할 수 없으므로 **동일 exact split의
mean-all 결과가 나오기 전에는 learned interpolation 우위를 주장할 수 없다**.

동일 split에서 다음을 추가한다.

- previous, nearest, date-linear, spherical interpolation.
- mean-all, past-only mean, day-of-year harmonic/HANTS, periodic GP.
- Kalman smoother/switching state-space, Δt-GRU, EO ODE-RNN.
- tile temporal mean을 제거한 residual cosine/MSE.
- 변화 mask IoU, 물리 변수, QA/readout utility.

양쪽 이웃을 쓰는 interpolation/smoothing과 과거만 쓰는 filtering/forecasting을 표와 코드에서 분리한다.
미래 관측을 쓴 결과를 early detection으로 합치지 않는다.

### 5.3 PDE/ODE보다 change-point가 먼저다

식생·기상처럼 매끄러운 현상에는 neural field/ODE와 weather forcing이 자연스럽다. 산사태는 갑작스러운
상태 전이이므로 generic smooth latent ODE/PDE는 onset을 흐릴 수 있다. 명시적인 상태 변수,
보존식/동역학, observation operator가 없다면 “PDE-informed”라고 부르지 않는다. 이 과업에서는
`seasonal normal state + switching/change-point + repeated observed evidence`가 더 직접적인 baseline이다.

시간 모델을 논문 본체에 넣는 최소한의 안전한 방식은 다음이다.

```text
past observed history
  -> gap x day-of-year expected normal state + uncertainty
query time
  -> predicted state [type=predicted; cannot confirm]
new usable observation
  -> innovation from expected normal
  -> first / provisional / confirmed / finished or contradicted
  -> immutable observed evidence IDs and region
```

### 5.4 정말 short-gap을 연구하려면 데이터 트랙을 분리한다

사용자가 원하는 “특정 좁은 지역의 시간 해상도 극세화”는 별도 데이터 계약이 필요하다.

- daily PlanetScope 기반 DynamicEarthNet 같은 dense stream, 또는 충분히 긴 특정 AOI archive.
- S1/S2 asynchronous stream과 실제 timestamp; optical reconstruction이면 AnytimeFormer/AGFlow류가
  직접 기준선이다.
- query를 ≤20일로 의도적으로 구성하고 AOI 내부 past→future chronological split을 쓴다.
- 단일 noisy latent cosine뿐 아니라 reflectance/semantic change/downstream decision을 함께 평가한다.

이 결과가 CEL-EO에 들어오더라도 predicted latent는 예상 정상상태나 retrieval priority로만 쓰고,
실제 관측 전에는 사건 근거로 승격하지 않는다.

## 6. VLM 방향의 최소 방어 아키텍처: CEL-EO

### 6.1 관측 레코드

각 관측은 최소 다음 typed record로 저장한다.

```text
obs_id, aoi_id, bbox/geometry, CRS/affine,
acquisition_time, arrival_time, sensor, bands, GSD,
quality/SCL map, encoder/version/hash,
spatial embedding, raw-chip pointer, observed_or_predicted
```

LLM prompt의 자연어 metadata와 updater/retriever가 쓰는 typed metadata를 구분한다. 위치·계절을
shortcut으로 외우지 않았는지 metadata-only, dates-only, metadata shuffle/dropout으로 검사한다.

### 6.2 변화 proposal과 상태기계

동일 지역 past-only seasonal normal로 innovation의 percentile/rank를 만들고, 전 지역 공통 z threshold를
강요하지 않는다. proposal은 `candidate → provisional → supported / contradicted / finished` 상태를
가진다. threshold와 persistence는 deterministic baseline부터 시작한다.

예측 상태는 proposal 생성에 도움을 줄 수 있지만, `supported`로 바꾸는 것은 실제 관측만 가능하게
한다. 새 관측이 없는데 LLM 문장이 스스로 claim을 확정하는 loop를 막는다.

### 6.3 provenance-preserving evidence memory

현재 상태 map과 사건 ledger를 분리한다. 각 event record는 pre/post observation을 따로 가리키고,
각각의 ID·날짜·품질·영역을 보존한다. 날짜가 다른 feature를 한 slot에서 평균하지 않는다.

첫 baseline은 학습 writer가 아니라 deterministic top-K다.

`score = change_rank × usable_quality × temporal_diversity × spatial_diversity`

동일 memory byte/token budget에서 recent-K, uniform-K, current D, top-K provenance, oracle evidence,
raw-chip retrieval을 비교한다. 사건 수가 충분해진 뒤에만 learned selector를 붙인다.

### 6.4 grounded reader와 제한 출력

query planner가 관심 AOI·기간·변화 유형을 typed memory에서 검색한다. 선택된 실제 pre/post chip에는
차분/local-causal attention 계열과 mask head를 적용한다. LLM 출력은 자유 문장만이 아니라 다음
구조를 반드시 동반한다.

```json
{
  "claim": "...",
  "status": "insufficient|provisional|supported|contradicted",
  "evidence_obs_ids": ["..."],
  "time_interval": ["...", "..."],
  "geometry": "mask-or-bbox",
  "uncertainty": "calibrated field"
}
```

verifier는 존재하지 않는 obs ID, predicted-only support, 시간 범위 밖 observation, 근거 없는 geometry를
거부한다. 언어는 ledger를 설명하고 질의하는 인터페이스이며, 검증된 detector state를 임의로
덮어쓰지 않는다.

이 설계는 로봇 기억과 닮았지만 논문에서는 “robot VLM”보다 **online partially observed EO decision
system**으로 정의하는 편이 정확하다. 4D retrieval, dual memory, metadata token 자체는 이미 혼잡한
아이디어다. EO 신규성은 irregular arrival, cloud/season shift, observed-vs-predicted provenance,
geographic evidence와 실제 correction에 있어야 한다.

## 7. 새 벤치마크 계약

### 7.1 사람 검수 gold

전문가가 각 prefix만 보고 다음을 표기한다.

1. 현재 답할 수 있는가.
2. 최초로 실제 흔적이 보이는 관측 ID는 무엇인가.
3. 그 관측에서 근거 영역은 어디인가.
4. 새 관측이 기존 고정 명제를 지지·반박·불충분 중 무엇으로 바꾸는가.

최종 사건 mask를 모든 prefix에 복사하지 않는다. `unknown→supported` evidence accumulation,
실제 현상 변화, 잘못된 판단의 `supported→contradicted/revised`를 별도 유형으로 둔다. 실제 revise
사례가 없으면 belief revision이라는 말을 쓰지 않는다.

### 7.2 필수 causal intervention

- critical evidence observation 삭제.
- irrelevant clear/cloud observation 추가.
- duplicate와 순서 교란.
- evidence를 비사건 관측으로 교체.
- 좌표/registration 20m shift를 token localization과 mask에 직접 적용.
- metadata 날짜·지역 shuffle과 raw visual 제거.

critical evidence를 없애도 같은 확정 답을 내면 grounded system이 아니다. 대체 가능한 복수 근거는
gold evidence set으로 관리한다.

### 7.3 필수 지표

- `answer correct AND evidence observation correct AND spatial mask correct` joint score.
- evidence-observation F1, mask IoU, evidence-date interval error.
- unsupported assertion rate와 predicted-as-evidence violation rate.
- abstention precision/recall, risk-coverage/AURC.
- unjustified flip rate와 genuine correction rate.
- false-alert episodes/km²-year, fixed-FAR recall, right-censored time-to-confirm.
- 성능 대 memory bytes/tokens, latency, raw archive reads의 Pareto curve.
- event/AOI outer cluster bootstrap; tile CI는 사건 내부 조건부 표로 분리.

### 7.4 필수 기준선

| 축 | 기준선 |
|---|---|
| 관측 사용 | latest-only, latest-clear, all-history frozen readout/VLM, all-mean |
| online state | previous, quality-aware EMA, Kalman/robust recursive update, Δt-GRU, rolling seasonal+persistence |
| memory | recent-K, uniform-K, 현재 D, deterministic top-K provenance, oracle evidence, raw-chip retrieval |
| language | metadata-only learned model, dates-only, deterministic template, LOG, LOG+VLM, full-history VLM |
| grounding | pair grounder, evidence pointer+mask, evidence-removal controls |
| representation | OLMoEarth v1 base뿐 아니라 현재 v1.2와 최소 한 개 경쟁 EO encoder |

현재 15 frame × 64 visual token은 960 token이라 full-history 입력이 불가능한 길이가 아니다.
LongEarth-R1은 평균 15.14, 최대 30 frame을 직접 다룬다. full-history VLM을 빼면 bounded-memory 필요성이
성립하지 않는다.

## 8. 실행 순서와 kill gate

### P0 — 원장·scorer 수리, 학습 없음

- 집계를 MS 상태와 atomic run/gate로 분리한다.
- MS-125, 132, 133, 144, 148의 과장 문장을 위 표대로 수정하되 원문 계보는 보존한다.
- Q_where를 등록한 Jaccard로 재채점하고 class-conditional/macro score를 낸다.
- T4 exact 2,114-tile split에 mean-all/past-mean/seasonal baseline을 계산한다.
- same-month pair/no-fallback cohort만 재계산한다.

**Gate P0:** 재채점 뒤 VLM의 주요 우위가 사라지거나 T4가 mean-all에 지면 해당 방법 주장은 종료한다.

### P1 — causal arrival와 통계 단위 수리

- future quality selection 없는 arrival replay.
- event-date/first-acquisition/arrival/confirmation clock 분리와 censoring.
- pre-event-only rolling calibration.
- registration shift를 token z/mask 평가에 직접 적용.
- 각 표에 `n_events`, `n_AOI`, `n_tiles`, calibration budget을 함께 표기한다.

**Gate P1:** deterministic seasonal+persistence가 learned updater와 동률이면 learned temporal updater를
논문 기여에서 뺀다.

### P2 — 새 사건과 실제 prefix gold

- Sen12의 남은 inventory부터 episode index를 구축한다.
- 사건 날짜 source/confidence와 NDVI-inferred 날짜를 분리한다.
- development 사건과 완전히 봉인된 final 사건 묶음을 새로 만든다.
- 독립 사건 수 목표는 임의의 숫자가 아니라 event-cluster power analysis로 정하되, 현재 네 건에서
  “수십 개 독립 trigger episode” 규모로 늘리는 것을 계획값으로 둔다.
- first-visible observation, mask, answerability, support/contradiction을 검수한다.

**Gate P2:** 새 독립 사건과 실제 per-prefix visible evidence가 없으면 CVPR main 주장을 중단한다.

### P3 — deterministic ledger부터 평가

- 규칙 seasonal proposal + immutable evidence ledger + template.
- 같은 ledger를 읽는 VLM.
- full-history VLM과 raw-chip retrieval.
- 그 다음에만 learned retrieval/writer를 추가한다.

**Gate P3:** VLM이 deterministic ledger+template보다 사실성·근거·보류에서 못 이기면 VLM은 UI로만
두고 방법 기여에서 뺀다. learned writer가 deterministic top-K를 event-level CI 밖에서 못 이기면
learned memory 주장을 뺀다.

### P4 — grounded update와 비용 Pareto

- 답+obs pointer+mask joint supervision.
- evidence deletion/replacement와 real contradiction.
- K=1/2/4/8, full-history, archive retrieval의 정확도–메모리–latency 곡선.

**Gate P4:** full-history가 동일 비용 범위에서 같거나 좋으면 현재 데이터에서 bounded-memory 우위를
철회한다. critical evidence 삭제에 답이 반응하지 않으면 grounded claim을 철회한다.

### P5 — 선택적 short-gap 보간

P0 exact-split baseline을 통과한 뒤에만 별도 dense dataset에서 진행한다. held-out observation 복원과
downstream decision이 모두 개선될 때만 CEL-EO의 expected-normal 모듈로 연결한다.

**Gate P5:** short-gap residual metric과 downstream에서 strong baseline을 못 이기면 interpolation은
characterization 부록으로 남기고 VLM 본체에 넣지 않는다.

## 9. CVPR 논문 형태

### 9.1 가능한 제목과 중심 주장

가제:

> **Observe, Predict, or Abstain: Provenance-Preserving Evidence Memory for Online Earth Observation VLMs**

중심 주장은 다음처럼 검증 가능해야 한다.

> irregular EO stream에서 CEL-EO는 동일 memory/compute budget의 latent mixing memory와 규칙 로그,
> full-history VLM보다 event-held-out 조건에서 근거 관측·영역을 더 정확히 선택하고, 낮은 false-alert와
> unsupported assertion으로 적절히 보류·정정한다.

기여는 세 개면 충분하다.

1. 실제 per-prefix 관측/영역/answerability/revision이 검증된 multi-event online EO benchmark.
2. observed/predicted provenance를 분리하는 bounded evidence ledger와 grounded reader.
3. event-level generalization, fixed-FAR latency, memory/archive cost를 함께 보는 평가.

“embedding을 language로 바꾼다”, “metadata를 많이 넣는다”, “4D memory를 쓴다”, “긴 EO 시계열을
VLM이 본다”는 각각 이미 선행연구가 가까워 단독 기여가 아니다.

### 9.2 현재 가능한 주장과 불가능한 주장

현재 가능한 가장 강한 관찰:

- frozen EO 상태 위 단순 learned temporal/memory head는 사건 이동에서 calibration과 spatial utility가
  불안정했다.
- target-site retrospective seasonal normalization과 명시적 규칙 log가 강한 baseline이다.
- current synthetic QA에서는 learned slot memory가 rule log reader를 이기지 못했다.

현재 불가능한 주장:

- “사건 판단에는 본질적으로 관측 두 장이 필요하다.”
- “.848이 보간의 물리적 상한이다.”
- “월 내 순서를 OlmoEarth가 표현할 수 없다.”
- “계절이 Hokkaido 과분산의 인과 원인이다.”
- “학습 memory 일반이 실패했다.”
- “VLM이 실제 EO 근거를 업데이트·수정했다.”

### 9.3 제출 판단

| 조건 | 판단 |
|---|---|
| 현재 원장과 네 사건 automatic QA 그대로 | CVPR main 불충분; workshop/characterization 후보 |
| P0~P1만 완료 | 신뢰 가능한 negative/benchmark paper 가능, 방법 기여는 약함 |
| P0~P4 + 새 sealed events + 사람 prefix gold + full-history 우위 | CVPR main을 진지하게 노릴 수 있음 |
| short-gap dense data까지 별도 성공 | 부가적인 temporal module/두 번째 논문 가능; 필수는 아님 |

## 10. 최신 1차 문헌과 직접 충돌점

아래 표의 “충돌”은 아이디어를 포기하라는 뜻이 아니라, 무엇이 더 이상 신규성의 중심이 될 수 없는지
보여준다.

| 문헌 | 확인된 범위 | 직접 충돌 또는 사용법 |
|---|---|---|
| [OLMoEarth official repository](https://github.com/allenai/olmoearth_pretrain), [pretraining dataset](https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Pretraining-Dataset.md) | 공식 코드·문서 | 현재 v1.2와 temporal RoPE 지원이 있어 v1 base만으로 결론 내리면 안 된다. 원 학습 자료는 360일에 30일 mosaic 12개이므로 raw per-acquisition duplicate-month stream은 granularity shift다. |
| [LongEarth-R1](https://arxiv.org/abs/2608.13344) | 2026 arXiv 본문/초록 | 평균 15.14, 최대 30 frame, 약 120k QA, temporal/spatial reward. “긴 EO 시계열 VLM”과 frame grounding은 이미 직접 경쟁이다. |
| [EarthDial, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Soni_EarthDial_Turning_Multi-sensory_Earth_Observations_to_Interactive_Dialogues_CVPR_2025_paper.html) | CVF 공식 | multi-sensory EO dialogue 자체는 신규성이 아니다. |
| [TEOChat](https://arxiv.org/abs/2410.06234) | 논문 | temporal EO chat/change QA의 직접 기준선이다. |
| [TerraScope](https://arxiv.org/abs/2603.19039) | 2026 arXiv 본문 | bi-temporal grounding과 mask-linked language. 단순 mask token 추가로는 부족하다. |
| [R4, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Sohn_R4_Retrieval-Augmented_Reasoning_for_Vision-Language_Models_in_4D_Spatio-Temporal_Space_CVPR_2026_paper.html) | CVF 공식 | semantic·공간·시간 4D persistent retrieval memory를 이미 제안한다. EO의 관측 품질·계절·provenance·보류가 차별점이어야 한다. |
| [AbstainEQA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_When_Robots_Should_Say_I_Dont_Know_Benchmarking_Abstention_in_CVPR_2026_paper.pdf) | CVF 공식 | 보류와 text-only shortcut을 이미 정면 평가한다. 같은 질문에 visual prefix만 바꾼 대조쌍이 필요하다. |
| [DIST-ALERT, Nature Communications 2025](https://www.nature.com/articles/s41467-025-64014-9) | 본문 | 계절 rolling baseline과 first→provisional→confirmed 상태 누적은 이미 운영적 선행이다. 필수 baseline이다. |
| [NRT-MONITOR](https://research.fs.usda.gov/treesearch/65240) | USFS/RSE 자료 | forgetting-factor recursive update와 반복 clear confirmation. learned filter 이전의 직접 baseline이다. |
| [EO ODE-RNN](https://arxiv.org/abs/2012.02542) | 논문 | continuous prediction + observation update를 irregular clouded SITS에 적용했다. ODE/predict-correct 자체는 신규성이 아니다. |
| [ALISE](https://arxiv.org/abs/2407.08448) | 논문 | irregular/unaligned SITS의 query-aligned representation. 시간 query projection 기준선이다. |
| [AnytimeFormer](https://doi.org/10.1016/j.rse.2025.115120), [AGFlow, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/MORSE/html/Fallah_Asynchronous_Remote_Sensing_Time-Series_Fusion_for_Cloud_Removal_and_Anytime_CVPRW_2026_paper.html) | publisher/CVF | 불규칙 비동기 S1/S2와 arbitrary-time reconstruction을 이미 다룬다. short-gap pixel reconstruction의 직접 경쟁이다. |
| [DynamicEarthNet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.html) | CVF 공식 | 75 AOI의 daily Planet imagery. 진짜 dense short-gap 계약에 더 적합하다. |
| [Sen12Landslides dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12606323/) | 원 논문 | 15 inventory·13,628 S2 patch·74,956 refined landslide. 새 독립 episode 구축의 우선 원천이다. |

## 11. 즉시 할 일 — 우선순위

1. **새 학습을 잠시 멈추고 P0를 끝낸다.** Q_where 재채점, exact-split mean-all, same-month cohort는
   현재 결과의 의미를 바꿀 수 있다.
2. **arrival-valid latency를 다시 정의한다.** 미래 quality 선택을 제거하고 event/observation/data/
   confirmation lag와 censoring을 분리한다.
3. **EventIndex를 만든다.** 15 inventory를 trigger episode로 분해하고 지금까지 보지 않은 final event
   묶음을 봉인한다.
4. **100~200개 pilot prefix를 이중 검수한다.** first-visible evidence와 mask 일치도가 낮으면
   automatic QA를 폐기하거나 proxy로 명시한다.
5. **학습 없는 immutable ledger를 먼저 만든다.** 이것이 LOG+template, full-history VLM, current D를
   이기는지 본 뒤 learned writer를 허용한다.
6. **OLMoEarth v1.2와 full-history VLM을 기준선에 추가한다.** 이 둘 없이 2027 심사에서 current-model
   comparison과 bounded-memory necessity를 방어하기 어렵다.

최종적으로 가장 유망한 논문은 “OlmoEarth 임베딩을 텍스트로 번역”하는 논문도, “1과 3으로 2를
만드는 latent interpolation” 논문도 아니다. **예측된 지구 상태와 실제 관측 근거를 혼동하지 않고,
제한된 기억에서 검증 가능한 관측·공간 근거를 유지하며 판단을 보류·확정·정정하는 EO VLM**이 현재
결과와 최신 문헌 사이에서 남아 있는 가장 선명한 공간이다.
