# OLMoEarth 추가 학습: JEPA 아이디어를 실제 갱신 능력으로 바꾸기

작성: 2026-09-09. **설계 초안, 미등록·미실행.** 새 성능 결과가 아니다.
사용자 첨부 `6c5f0d0b-7b7f-42c4-bb28-4df3b079e582/pasted-text.txt`를 검토했다.
실측 기준은 [9/9 연구판](STREAMING_RESEARCH_UPDATE_2026_09_09.md)과 그 보존 보고서다.
이 문서는 다음 학습의 준비안이며 기존 prereg/서버 queue를 대체하거나 자동 실행하지 않는다.

## 1. 결론과 큰 그림

**추가 학습으로 개선하는 방향을 채택한다. 다만 JEPA·GRU·센서 정렬을 한꺼번에 새 방법으로
선언하지 않는다.** 먼저 다음 한 문장을 검증한다.

> 새 위성 영상만 처리하는 저비용 경로에서도, 기존 OLMoEarth 임베딩의 판독 가능성을
> 유지하도록 추가 학습할 수 있는가? 실제로 과거 상태가 도움이 되는가?

현재까지 cache/few-shot은 **저장 표현이 쓸모 있다**는 기반, MS-116/117은 **새 관측으로
표현을 갱신할 수 있다**는 근거다. 다음은 **그 갱신 경로에 맞춰 학습 목적과 일부 가중치를
바꾸면 더 좋아지는가**이다. 센서 결손과 surprise는 이 본체를 평가·확장하는 별도 가지다.
FoldRefresh/release bridge는 모델 버전 변경 자산, Nepal은 응용 이력으로 남긴다.

KuroSiwo 수치는 조건부 양성이지만 clean-validation 재선택은 미완결이다. 현재 4 Sen12 지역과
이미 열람한 Kuro test 사건은 새 방법의 개발 근거로만 쓴다. 다시 실행한다고 untouched가 되지 않는다.
현재 GRU보다 추가 학습이 좋아질 가능성은 열려 있지만, 향상 크기·속도·CVPR 채택은 예측하지 않는다.

## 2. 첨부 설명에서 고칠 것

| 첨부의 해석 | 검토 결과 / 사용할 표현 |
|---|---|
| 관측 없는 GRU ≈0% → 산사태는 예측 불가능 | 해당 모델·입력·목표에서 실패했을 뿐이다. 재해 예측 불가능성의 증명이 아니다. |
| 산사태·홍수는 정의상 예측 못 한 것, surprise=탐지 | 틀리다. surprise는 우선 모델의 예상 불일치다. 계절·구름·레이더 기하·정합 오차도 원인이다. |
| 단일 임베딩 − 예측한 창 상태 | 같은 768차원이어도 목표 의미가 다르다. **단일 관측의 예상 − 같은 계약의 단일 관측**으로 바꾼다. |
| frozen target이면 모든 붕괴가 방지됨 | target 자체의 동반 붕괴는 막지만 학생의 평균값 예측·정보 무시는 남는다. 분산·유효 rank·상수 예측 대조가 필요하다. |
| 현재 updater는 이미 world model, Δt는 action | 정확한 현재 이름은 관측 조건부 상태 갱신기다. Δt는 시간 조건이며 제어 행동이 아니다. 관측 전 예측을 따로 평가해야 한다. |
| S1 cosine .39가 실패 원인을 확정 | 상관/현상이다. 시간·표본·센서·single/window 차이가 동시에 섞여 원인을 분리하지 못했다. |
| Hiroshima test 간격 8일이라 정렬 문제 아님 | train 평균은 52.39일이다. 8일도 사건 변화에는 작다고 보장 못 한다. 날짜 정렬 원인 배제 불가. |
| S1-only를 사전학습에서 거의 못 봄 | v1 논문은 전체 bandset masking을 명시한다. 실제 checkpoint의 sampling 빈도 확인 없이 단정 불가. |
| 다중모달 FM만 센서 번역 가능 | 별도 센서 encoder도 paired distillation이 가능하다. OLMo의 이득은 비교 실험으로 보일 일이다. |
| joint S1+S2 teacher면 번역 문제가 사라짐 | 목표를 바꾸는 다른 실험이다. 현재 S2가 미가용이면 joint teacher도 오래된 S2+현재 S1만 본다. 결손 정보는 자동 복원되지 않는다. |
| 외부 3task·회복50%·6주 gate가 CVPR 조건 | 저자의 자원 배분 규칙일 수는 있으나 학회 기준은 아니다. 특정 arm 실패가 모든 method 연구 종료를 뜻하지 않는다. |

근거: `code/streaming_update_train.py`, `code/extract_olmo_streaming.py`,
`code/extract_sen12_s1_singles.py`, `artifacts/streaming_review_20260909/summary.json`.
레이더에서 물이 항상 어둡다는 단순화도 피한다. 도시 double-bounce, 바람, shadow/layover를
포함한 실패 조건을 별도로 본다.

## 3. 최근 선행이 주는 방향 — 이름보다 학습 목표

아래는 공식 논문 확인에 따른 요약이다. 결과 수치를 우리 설정에 옮겨 쓰지 않는다.

| 원문 | 이미 있는 것 | 우리 설계에 반영 |
|---|---|---|
| [OLMoEarth v1 §2.3–2.4](https://arxiv.org/html/2511.13655v1) | SLIM/Latent MIM Lite, 고정 random-projection target, bandset masking, contrastive 학습 | JEPA와 연관은 있지만 I-JEPA와 같은 학습법은 아니다. 우리의 frozen trained encoder distillation도 원 논문의 random target과 다르다. |
| [I-JEPA](https://arxiv.org/abs/2301.08243), [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 잠재 예측; V-JEPA 2의 action-conditioned post-training | 가린 토큰 복원, 미래 예측, 관측 후 보정을 분리한다. 제어·planning 능력은 주장하지 않는다. |
| [V-JEPA 2.1 §2](https://arxiv.org/html/2603.14482v1), 2026-03 | dense predictive loss, intermediate-layer supervision | 세밀한 지도 품질 보존에 참고. 현재 MSE도 이미 dense token loss라 단순 추가를 신규성으로 쓰지 않는다. 중간층 전체 저장은 비용을 잰 뒤 검토한다. |
| [TerraFlow 및 Appendix A](https://arxiv.org/html/2603.12762v1), 2026-03 | EO temporal continual pretraining; temporal-disjoint sampling | 실제 EO 모델 추가 학습의 직접 선행이다. forward-only sampling은 초기 실험에서 기대 개선이 없었다고 보고하므로 인과 마스킹 자체를 만능으로 보지 않는다. |
| [Cloud-aware EO latent world model §8.3](https://arxiv.org/html/2607.13651v1), 2026-07 preprint | JEPA 계열 EO 관측성 예측, 합성 temporal inconsistency anomaly | “EO+JEPA+surprise 최초” 주장 금지. 관측 품질 이상과 실제 지표 변화의 구분이 필요하다. |
| [KalmanNet](https://arxiv.org/abs/2107.10043), [Recursive Bayesian Classification](https://arxiv.org/abs/2301.01796) | predict/correct 학습 필터, EO 재귀 지도 갱신 | GRU/innovation/predict-correct 자체는 신규성이 아니다. 우리 대상은 기존 판독기를 유지하는 materialized feature state다. |

추가 감시 목록: [UniJEPA](https://arxiv.org/abs/2608.07409),
[Orthogonal JEPA](https://arxiv.org/abs/2608.20065)는 2026-08 **초록 확인만** 했다.
사진·시간 예측 결합이나 factorized latent state를 신규성으로 내세우기 전 전문 비교가 필요하다.

**신규성 후보:** frozen global model을 단순히 재사용하는 것이 아니라, 부분 관측으로 업데이트되는
경로에 맞춰 post-training하면서 **오래된 cache와 고정된 여러 readout의 호환성**을 유지하고,
도착 시각 기준 품질–비용을 개선하는 것. 검색상 공백은 최초성의 증명이 아니다.

## 4. 과거 상태·새 관측·예측을 정확히 구분한다

- `E0`: 버전·정규화·공간 격자까지 고정한 reference OLMoEarth.
- `Z*_t = E0(X_available_at_t)`: **그때까지 사용 가능한** 전체 창의 teacher 상태.
- `q_t = E0(x_t)`: 새 취득 한 장의 reference 표현. `Z*_t`와 같은 형식이어도 다른 계약이다.
- `M_(t-1)`: 저장된 학생 상태. 인코더 내부 KV cache와 다르다.
- `qhat_t = P(M_(t-1), Δt, sensor, calendar)`: 현재 관측을 보기 전, **q_t와 동일한 계약**을 예측.
- `innovation_t = q_t - qhat_t`: 예상 불일치. hazard probability가 아니다.
- `M_t = G(M_(t-1), q_t, metadata)`: 관측을 반영한 새 상태. 첫 단계에는 innovation을 추가 입력하지
  않아 예측 보조 손실 효과와 새 구조 효과를 분리한다.
- `D0(M_t)`: 기존 고정 판독기의 지도. 학습된 posterior와 observation-free prior를 따로 평가한다.

예측의 label은 frozen encoder가 만든 표현이므로 수동 정답지 없이 학습할 수 있다.
그러나 segmentation head 학습, model selection, 임계값 조정에 쓴 라벨은 따로 계산한다.
“target label 0”과 “전체 과정 무라벨”은 다른 주장이다.

**시간 계약:** 취득시각과 게시/사용가능시각을 별도 보존한다. 게시시각이 없는 과거 데이터는
acquisition-order replay로만 부르고 실시간 카탈로그 지연을 재현했다고 하지 않는다.
현재 Sen12는 전체 기간에서 고른 clearest-12의 prefix다. 이는 offline 실험으로 보존하되
미래 구름 정보를 사용하지 않는 arrival-prefix 실험과 섞지 않는다. Kuro의 가정된 날짜도 복구 전
정확한 Δt/예측 지연의 근거로 쓰지 않는다. 같은 acquisition을 여러 번 새 증거로 투입하지 않는다.

학습에서 `t+1` 표현을 정답 target으로 보는 것은 허용한다. predictor input에는 들어가지 않는다.
test 지역의 미래 큐브로 predictor/normalization을 학습하면 transductive 조건이므로 기본 실험과
분리한다. “라벨이 없다”는 이유로 전체 Sen12/Kuro/Korea 큐브를 학습에 넣지 않는다.

## 5. 실제 파인튜닝을 포함하는 최소 연구 설계

### P0. 이미 있는 자산의 출발점 정리 — 새 방법 실험과 중복하지 않기

1. Kuro 입력 결함을 처리한 새 data revision 및 clean-validation checkpoint 선택을 준비한다.
   복구 불가 입력은 사전 기준의 abstention/coverage로 보고하고 어려운 타일을 성능 분모에서 숨기지 않는다.
2. **POST_ONLY**: 새 단일 표현에 source-trained readout. **NO_MEMORY**: 과거 상태 없이
   새 관측만 받도록 훈련한 capacity-matched adapter. 기존 no-observation GRU는 반대 대조다.
3. 본문 teacher-fidelity와 task-utility를 분리한다. teacher를 넘는 AP는 가능하나 “표현 복원 100% 초과”와
   같지 않다. 작은 teacher−stale 분모의 회복률 대신 절대 AP와 사건별 결과를 함께 쓴다.

기존 aux logit-MSE는 **이미 실행**됐다. `task_aware_update_candidate.py`의 작은 residual과
readout-KL은 **구현·테스트만**, 미학습이다. Δt/attention/공간 GRU·시간차분도 이미 시험했다.
같은 것을 새 JEPA 실험으로 다시 세지 않는다.

### P1. 본체 — cache 호환성을 유지하는 encoder post-training

가장 중요한 변경: **encoder 학습을 무조건 future work로 보내지 않는다.**
현재 H200 환경에서 작은 LoRA pilot의 가능성은 tensor 계약·10-step 메모리 계측으로 판단한다.
90M이라는 parameter 수만으로 불가능하거나 Ai2 내부에서만 할 수 있다고 말할 근거는 없다.

첫 2×2는 기존 source data만 사용하며 **D0를 고정**한다. 모든 arm은 같은 GRU, 같은 teacher,
같은 prefix, 같은 학습/선택 예산을 쓴다. 경량 forecast branch는 training-only다.

| arm | 새 영상 encoder | forecast 보조 손실 | 식별할 것 |
|---|---|---|---|
| F0 | frozen E0 | 없음 | feature + readout 보존의 기준 |
| F1 | frozen E0 | 있음 | 시간 예측 학습 자체의 효과 |
| L0 | E0 + LoRA | 없음 | 추가 trainable capacity/encoder 적응 효과 |
| L1 | E0 + LoRA | 있음 | encoder 적응 위에서 예측 학습이 더 주는 효과 |

**B0 plain GRU+feature-MSE**도 같은 수정 데이터/recipe로 비교한다. F0가 기존 logit-MSE와
동일해지는 설정이면 기존 개념의 재현임을 표시한다. F1−F0, L0−F0, L1−L0를 구분한다.
“LoRA+JEPA 조합이 좋다”는 말만으로 효과의 원인을 확정하지 않는다.

시작값 제안: attention q/v, 마지막 2개 block, rank8. **설치 모델에서 module name을 확인하기 전에는
적용하지 않는다.** 각 arm의 trainable tensor 목록과 실제 parameter 수를 저장한다.
head와 updater까지 한 번에 바꾸지 않는다. LoRA 예산 초과 시 범위를 줄이는 것은 새 recipe로 기록한다.

손실 초안:

`L = λ_state L(M_t, stopgrad(Z*_t)) + λ_readout KL(D0(Z*_t), D0(M_t))`

`    + λ_forecast L(P(M_(t-1), metadata_t), stopgrad(q_t))`

`    + λ_anchor L(Q(Eθ(x_t)), stopgrad(q_t))`

`Q`는 관측 표현을 reference 계약에 맞추는 사영이며 frozen arm은 identity다. feature 손실은
source-only scale로 정규화한다. λ와 checkpoint selection은 source validation에서 정하고,
test에서 바꾸지 않는다. 현재 관측을 이미 본 상태로 현재 q를 예측하게 만들지 않는다.
서로 다른 시간의 latent가 같아지도록 평탄화하는 temporal smoothness를 주목적으로 쓰지 않는다.

LoRA arm의 **초기 과거 cache는 기존 E0 값 그대로** 둔다. 새 영상만 Eθ로 처리하고 Q/G를 거쳐
canonical state로 기록한다. 이는 학습 목표이지 호환성 보장이 아니다. 기존 head의 retention을
검사하고, 새 model/adapter/state revision을 명시한다. 재임베딩 필요분은 비용에 포함한다.
온라인에도 forecast branch가 필요하면 그 비용을 추가한다. LoRA라서 raw-free라고 쓰지 않는다.

첫 smoke는 개발 fold1×seed1에서 B0/F0/F1/L0/L1, test 판독 없이 gradient/identity/NaN/memory와
source-validation만 확인한다. 이후 동결한 2개 개발 설정×3seed 전체면 **30회**다. 30회를
먼저 무작정 실행하지 않는다. 기존 보고서 값과 새 recipe 값을 한 표의 matched 비교로 섞지 않는다.

### P1b. decoder joint tuning — 빠른 utility 가지, 본체와 분리

첨부의 joint decoder+updater는 가능하다. 다만 D를 바꾼 결과는 **기존 head 호환성**의 증거가 아니다.
별도 D1 복사본을 두고 (i) D1-only on frozen GRU states, (ii) G+D1 joint, (iii) teacher와 student
두 경로에 같은 supervised budget을 주는 dual-path를 비교한다. D1(Z*) 성능도 함께 보고한다.
terminal label은 해당 terminal state에서만 사용한다. Sen12 끝 시점 마스크를 모든 과거 prefix의
현재 정답으로 쓰지 않는다. 이 가지가 성공해도 “EO encoder 자체가 향상”이라고 말하지 않는다.

### P2. 센서 정렬 — 먼저 관측 계약, 다음 학습

현재 nearest±/gap무제한 매칭을 그대로 projection에 넣지 않는다. actual-time past-only pairing,
duplicate 금지, 허용 gap별 coverage, 같은 train/val/test cohort를 우선 고정한다.

1. source에서 S1-single→S2-single의 identity/선형/작은 nonlinear 정렬을 비교한다. 같은 격자,
   근접시각, 유효광학·SAR기하 층화가 필요하다. paired target 사용은 학습 시에만 허용한다.
2. 실제 배포 시험은 old S2 state + new S1, 현재 S2는 미가용이다. frozen S2 teacher는 사후 평가용
   privileged reference로 구분한다. 미래 S2를 inference feature로 넣지 않는다.
3. 별도 joint-state arm은 teacher를 old S2+available S1로 바꾼다. 이는 2번과 다른 목표이므로
   각각 native teacher·native head·raw/no-memory 대조를 만든다.

cosine 개선만으로 성공을 정하지 않는다. mapping AP/FP와 retention을 본다. 실패하면 sensor mismatch,
temporal gap, training coverage, 물리적 비식별성 중 무엇이 남는지 한정한다. 재해 광학색 복원은
물리적으로 유일하지 않을 수 있다. **정렬을 학습했다 = 실제 광학을 관측했다**가 아니다.

### P3. surprise — 갱신 성능과 별개 평가

F1의 예측을 우선 재사용해 단일-sensor에서 계산한다. sensor를 바꾸는 실험과 동시에 시작하지 않는다.
raw residual과 Δt/계절/관측 품질로 source-calibration한 residual을 모두 보고한다.

- 대조: 연속 single Δz, persistence forecast, source 계절 평균, 단순 affine predictor;
  S2는 ΔNDVI/밴드차, S1은 같은 기하의 log-ratio 등 센서 적합 대조.
- 지표: 사건 동일가중 AP/AUPRC, 고정 검수 면적/FP 예산의 recall, coverage, 공간/사건 CI.
- 정상 시기·영구수역·구름·shadow·날짜 gap을 negative/층화로 포함한다. anomaly가 관측 불량만
  찾는지 확인한다. 관측 품질 경보와 지표 변화 후보를 서로 다른 출력으로 둔다.
- 정확한 onset label이 없으면 “조기 탐지/미래 재해 예측”을 쓰지 않는다. 모델 선택에 사건 라벨을
  썼다면 unsupervised training과 label-assisted selection을 구분한다.

surprise가 실패해도 P1의 업데이트 개선은 별도로 살아남을 수 있다. 반대로 surprise AP만 좋아져도
공유 cache 갱신 성능이 좋아졌다는 뜻은 아니다.

## 6. 외부 검증과 Korea shared-cache 3-task

- **Sen12:** 현재 exposed 개발·기존 비교 자산. 종점 label로 onset을 주장하지 않는다.
- **KuroSiwo:** S1 사건 재현 자산. 이미 본 test 사건은 다음 방법의 깨끗한 확증집합이 아니다.
  새 data revision·고정 head·binary/3-class 계약을 명시하고, 외부 숫자와 직접 순위를 매기지 않는다.
- **DEN:** 라벨 시점·실제 시간 이미지·센서/밴드/API를 감사한 뒤 시간별 map consistency 후보.
  월별 라벨이면 onset은 월 사이의 구간일 뿐 정확한 날짜가 아니다. 현재 S2 시계열이 있다고 가정하지 않는다.
- **한국:** `korea_shared_cache_3task_prereg_v1_amendment.json`의 기존 gate를 보존한다.
  하나의 updated state → land-cover/forest-loss/landslide D0 세 개. 한 head만 좋아진 것을
  “범용 표현 향상”으로 쓰지 않고 평균/최악 task·각 class·negative transfer를 함께 본다.
  readout-KL에 쓴 2task와 쓰지 않은 1task를 명시하면 **판독기 간 일반화**를 시험할 수 있다.
  세 개를 독립 학습한 updater와 shared updater, task별 전체 재인코딩 vs 공유 재인코딩을 비교한다.
  비용의 encoder는 실제 공유될 때만 한 번 센다. task label 시점이 없으면 static shared-state 시험으로 제한한다.

한국 label은 아직 열지 않는다. 위 3task는 이미 성공한 실험이 아니라 후속 설계다.

## 7. 판정·비용·CVPR 문장

권장 수치 초안(기존 사전등록이 아니며 launch 전 동결할 선택): 주지표는 equal-event/region AP,
matched 기준 대비 +.02 절대 개선과 95% paired cluster CI 하한>0, 고정 head retention 허용 손실
.01 AP, 고정-FP 예산 결과 병기. 데이터별 의미 있는 차이와 통계 검정력 검토 후 **결과 열기 전**
확정한다. 모두 실패를 작은 margin으로 구제하는 사후 변경은 금지한다. 회복50%/2배 speedup을
CVPR의 합격선으로 쓰지 않는다. 현 draft는 미결 gate가 있으므로 실행 불가다.

비용은 observation availability가 같은 시점끼리: 초기 인코딩, 새 영상 인코딩, 갱신, 모든 head,
실제 raw I/O·p50/p95·peak memory·state bytes, teacher 생성/추가학습의 amortization을 보고한다.
과거8개 '미래 관측' batch의 2.3배를 온라인 속도로 쓰지 않는다. 새 학습 비용도 숨기지 않는다.

CVPR에서 방어할 질문은 **“기존 GRU/PEFT/증류보다 어떤 조건에서 왜 개선되는가?”**다.
L1 하나의 양성, 센서 번역 하나의 복구, 과업 개수만으로 답할 수 없다. 불규칙 관측/센서결손에
맞춘 objective가 matched baseline을 이기고, 고정 readout과 외부 사건에서 재현되며 비용 이점이
남는다면 method 주장으로 발전할 근거가 된다. 현재는 그 실험을 준비한 상태다.

## 8. 다음 실행자가 할 일 — 준비와 실행을 혼동하지 않기

1. 이 문서 및 `config/observation_posttraining_v0_draft.json`을 읽고 unresolved 항목을 해소한다.
2. active queue와 보호 파일 상태 조회. Kuro revision/clean selection과 POST_ONLY 준비는 별도 output.
3. 설치 encoder module path·LoRA gradient·canonical-state boundary·teacher detach·forecast input
   cutoff를 CPU/small smoke로 검증. 원본 source/readout 손상 여부 테스트.
4. source split/actual-time manifest, normalization, λ/HPO, meaningful margins, external holdout,
   actual cost protocol을 **실행 전** 별도 prereg로 봉인한다. 기존 v0를 덮어쓰지 않는다.
5. snapshot 안의 코드 자체를 실행, 고유 OUTROOT, finite-validation/정상 checkpoint/예상 arm 완결 전
   DONE 금지. 파일 존재만으로 resume/skip하지 않는다. 서버 GPU1 규약은 유지한다.

**이번 준비에서 하지 않은 것:** 서버 접속/업로드·학습 실행·Korea 개봉·production 수정·commit/push.
새 모듈 성능·확증 결과·인코더 개선은 아직 없다.
