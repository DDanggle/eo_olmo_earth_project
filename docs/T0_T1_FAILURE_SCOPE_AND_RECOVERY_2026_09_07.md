# T0/T1 저녁 감사 — 실행 실패, 학습 실패, 정보 부족을 구분한다

2026-09-07. 로컬 시작 commit `4a18746`. 서버 확인: 12:03 UTC / 21:03 KST.
이번에 새 GPU 성능 실험을 실행하지 않았다. 기존 JSON을 내려받아 재집계했고, source-only CPU
표현 범위 진단과 신규 모듈 단위 테스트만 실행했다. 아래 최신성은 확인시각의 스냅샷이다.

후속 상태 확인 **13:28:49 UTC / 22:28:49 KST**: 다른 실행기가 13:16:44Z에
`train runner start (audit gate bug bypassed: audit_max=0.0 is a pass)`를 기록했고,
`artifacts/streaming_t1`이 생겼다. 따라서 T1은 이후 학습 재개 상태다. 이번 감사자가 재개하거나
활성 runner를 덮어쓴 것이 아니다. 새 학습의 완료/성능/전체 provenance 검증을 뜻하지 않는다.

## 1. 실제로 중단된 이유: 오차 0.0을 실패로 바꾸는 실행기

서버 T1 최초 중단 로그(21:03 KST 스냅샷):

- 12:03:13Z extraction rc=0.
- teacher/single 3,372/3,372, n_skipped=0, all_gates_pass=true.
- sampled teacher e12 vs sealed cache 최대 차이 **0.0**.
- 그런데 `streaming audit FAILED`; 그 시점 `artifacts/streaming_t1`도 없었다.

원인: `code/t1_chain.sh`의 `(a["audit_max"] or 1)<0.05`.
Python에서 0.0은 falsy이므로 1로 대체되어 검사에 실패한다. `None`과 0을 혼동한 버그다.
같은 코드가 `code/evening_queue_0907.sh`에도 있었다.

로컬 수정:

- `code/validate_temporal_cache_evidence.py`: 0 허용, None/NaN/Inf/음수/boolean 거부, audit sample
  수·중복·summary 최대값·추출 완결성 검증. T0 50개, T1 기존 30개 비교 표본 요구.
- 두 runner에서 helper 호출. 학습 명령 실패 시 실패 marker와 nonzero exit; 실패 후 DONE 방지.
- 기존 두 감사 JSON으로 새 검사 모두 통과: T1 0.0, T0 0.0032553672790527344.
- 단위 테스트 8개 + invalid-value subtest 5개 통과.

**서버의 활성 runner는 덮어쓰지 않았다.** 이 수정은 로컬에 있다. 학습 재개 전에는 v0의
T0 통과→T1 시작 조건과 별도 T1 등록의 관계를 dated amendment로 정리해야 한다.
버그 수정이 새 실험 승격이나 예산 승인을 자동으로 뜻하지 않는다.
resume 시 모든 기존 배열을 skip하면 audit sample list가 비어질 수 있으므로, 과거 유효 증거를
보존하거나 read-only replay를 해야 한다. 이 경우 검사를 없애거나 None을 0으로 바꾸면 안 된다.

## 2. T0 숫자는 맞지만, 상한이라는 해석은 틀리다

다운로드 스냅샷: `artifacts/t0t1_audit_20260907/temporal_t0/`.
재집계: `python3 code/inspect_t0_artifacts.py artifacts/t0t1_audit_20260907/temporal_t0`.

| chimanimani | seed 1 AP | seed 2 AP | seed 3 AP | 평균 AP | 평균 positive-tile IoU |
|---|---:|---:|---:|---:|---:|
| mean | .302764 | .371125 | .331281 | .335057 | .158343 |
| diffpca | .302889 | .248022 | .319588 | .290166 | .191110 |
| paired 차이 | +.000125 | −.123104 | −.011692 | −.044890 | +.032768 |

- 주지표 AP의 평균 하락은 실제다. IoU 상승으로 성공이라고 바꾸지 않는다.
- seed 2를 삭제하지 않는다. 다만 평균 하락 대부분이 해당 seed에서 나왔으므로 최적화·checkpoint
  민감도 확인 없이 표현의 정보량에 대한 결론을 내리지 않는다.
- mean의 선택 epoch는 25/4/11, diffpca는 25/31/25. 같은 모델 계열이어도 선택된 학습 단계가 다르다.
- AP가 주지표이지만 선택 기준은 val IoU@0.5다. 고정 recipe로서 허용 가능한 선택이며 결과 무효나
  누수를 뜻하지 않는다. 다만 AP 최적 readout을 시험한 것은 아니다. AP 선택 재실험은 새 recipe다.
- sketch는 당시 seed 1만(.285011), 다른 region/나머지 arm 미완료. 미완료를 음성으로 세지 않는다.

추론 경계: 입력을 추가한 모델이 기존 입력을 무시할 수 있다면, 함수군 수준에서 기존 해법은 여전히
존재한다. 실제 학습된 모델이 더 못한 것은 정보 없음뿐 아니라 추정·최적화·정규화 문제로도 설명된다.
**하나의 앞/뒤 절반 차분의 실패 ≠ 모든 시간 표현의 불필요 ≠ Bayes 상한 도달.**
더구나 앞/뒤 6장은 실제 재해 전/후로 정렬된 구간이 아니다. terminal mask의 분할과 변화시점 검출은 다르다.

T0와 T1의 질문도 다르다. T0는 같은 관측을 시간별로 읽으면 terminal-task AP가 오르는지,
T1은 새 관측으로 full-window를 다시 계산한 결과를 더 싸게 근사할 수 있는지를 묻는다.
따라서 T0가 실패했다고 T1의 계산 절약 가능성까지 논리적으로 부정되지는 않는다.
다만 이미 등록한 T0→T1 실행 조건은 조용히 무시할 수 없다. 이 estimand 차이를 명시한
dated amendment와 최초 v0 기록을 함께 보존해야 한다.

## 3. 구현에서 확인한 과학적 한계

### T0: 시간 정보를 추가한 방식도 공정하게 해석해야 한다

- time stack을 PCA-64 후 1×1 conv로 압축하는 것은 특정 readout이다. `full`도 raw temporal token
  전체를 무제약으로 쓰는 모델이 아니라 PCA·128채널 병목을 거친 reference다.
- 새 채널을 concat한 뒤 decoder 전체를 새로 초기화한다. 기존 mean predictor를 보존한 nested
  개선 실험이 아니다. 기존 source-trained mean head + zero-init residual을 source에서 학습하는
  대조가 필요하다. 기존 MS-114는 target few-shot adaptation이어서 이것과 학습 단계가 다르다.
- learned sketch의 32채널은 task/fold와 함께 학습된다. 64 KiB/tile은 해당 sketch를 학습 후
  materialize했을 때의 크기다. 다른 새 task에도 같은 sketch가 쓸모 있는지는 미측정이다.
  구현은 여전히 T×64 stack을 읽으므로 보편적 공유 캐시 64 KiB가 이미 성립했다고 말하면 안 된다.
- T0b single-acquisition embedding은 새로운 encoder 호출 계약이다. joint tokens가 contextualized라는
  사실만으로 차분이 무의미하다고 단정할 수 없다. joint mean과 single-mean은 별도 baseline이다.

### T1: 현재 구현은 제안의 좁은 MSE prototype이다

1. `F.mse_loss(m, teacher)/sc**2`만 최적화한다. frozen decoder AP는 마지막에만 평가하며
   task-space loss가 없다. reconstruction fidelity와 task fidelity를 구분하지 못한다.
2. 새 관측 두 개를 바로 평균내며, actual elapsed time·구름/결측 품질을 updater에 주지 않는다.
   encoder가 timestamp를 받는 것과 update module이 관측 간격을 받는 것은 다르다.
3. full-window에서 가장 맑은 12장을 고른 뒤 prefix를 자른다. future quality가 선택을 바꿀 수 있어
   진짜 causal acquisition stream이 아니다. retrospective curated-prefix 실험으로만 부른다.
4. GRU는 3,541,248개, residual은 4,721,664개 파라미터다. **budget-matched가 아니다.**
5. 구 GRU는 output projection 없이 `m'=(1-z)m+z*tanh(h)`다. 각 coordinate는
   `[min(m4,-1), max(m4,1)]` 밖으로 못 나간다. teacher 범위와 맞지 않을 가능성이 있다.
6. 원래 code는 누락 파일을 manifest에서 조용히 제외한다. count만이 아니라 ID hash와 exact equality 필요.
7. 끝점 c12만 채점하므로 업데이트 중간의 regression을 놓친다. c4/6/8/10/12별 fidelity를 보고할 것.
8. no-update와 teacher의 AP gap이 작거나 음수면 90% recovery 분모가 부적절하다.
   절대 AP 오차 허용폭과 실제 비용도 사전에 정해야 한다.

GRU 범위의 실제 source-only CPU 진단(각 fold source 10타일, query 미사용):

| fold의 source 표본 | 표현 불가능한 teacher coordinate 비율 | 피할 수 없는 relative feature MSE 하한 |
|---|---:|---:|
| chimanimani holdout source | 0.17972% | .0005124 |
| hiroshima holdout source | 0.17534% | .0003819 |

제약은 존재하지만 숫자는 작다. **이것이 AP 실패의 주원인이라고 주장하지 않는다.**
GRU에 output projection/source normalization을 둔 비교가 더 적절하다는 근거다.
재현 스크립트: `code/probe_streaming_gru_range.py`. source IDs는 실행 출력 및 review summary에 보존.

비용 정정: 등록의 36 vs 12 timestep-unit 비교는 초기 4개를 full 쪽에서만 뺐다.
동일한 c4/6/8/10/12 갱신 workload면 full 40, streaming 12다. 초기 encode 이후만 세면
36 vs 8이다. 끝점 c12만 필요하면 12 vs 12다. 어느 경우도 timestep 수를 GPU 시간 비율로
그대로 바꿀 수 없다. encoder attention·호출 overhead·updater 연산·I/O를 실제로 재야 한다.

## 4. 이번에 만든 개선 프로토타입

`code/task_aware_update_candidate.py` — 별도 파일이며 기존 T1을 대체하지 않는다.

- source-aligned state/observation + 실제 Δt + 관측 quality 입력.
- 새 관측마다 순서대로 호출; 둘을 평균내고 시작하지 않음.
- 768→64 bottleneck, depthwise spatial conv, 64→768 residual: **198,208 파라미터**.
- 마지막 layer를 0 초기화하여 시작점은 정확히 no-update. invalid/quality0은 상태 유지.
- frozen readout의 Bernoulli KL + source-normalized feature MSE loss helper.
  frozen head를 통해 student에는 gradient가 흐르고 teacher/head에는 흐르지 않는다.

단위 테스트 9개를 기존 서버 환경에서 **CUDA_VISIBLE_DEVICES=""로 CPU 실행**, 전부 통과.
identity, invalid NaN masking, quality 0 hold, loss gradient 경로, frozen head 검사를 포함한다.
업로드 위치는 `/home/work/data/olmoearth/review_20260907/`뿐이다. 기존 training code 경로는 수정하지 않았다.

이것은 학습된 방법이 아니라 **시험 가능한 부품**이다. no-update 시작은 학습 후 무회귀 보장이 아니며,
quality는 자동 추정된 위험도도 아니다. 기존 single/full-window 특징의 분포 정렬과 normalization을
source에서 먼저 맞춰야 한다. 이 코드만으로 정확도나 노벨티가 검증됐다고 말하면 안 된다.

새 설계: `config/streaming_task_preservation_v1_draft.json`. 미등록·미실행이며 v0와 별도다.
첫 비교는 두 모듈×MSE/MSE+task loss×노출 2지역×3seed=24회로 objective와 구조 효과를 분리한다.
시간·quality 이득을 주장하려면 각각을 제거한 대조를 추가 동결해야 한다. 여러 개선을 동시에 넣고
한 요소가 원인이라고 주장하지 않는다. 불필요한 18회가 자동으로 시작되도록 두지 말고 먼저
mean/freeze/teacher reference와 source train/val 1회 smoke를 확인한다.

**개선 목표도 분리한다.** T1이 full teacher를 싸게 근사하는 방법이라면, teacher AP 자체를
넘는 것이 필수 성공 조건은 아니다. full recompute에 대한 사전 정의된 성능 허용폭 안에서
실제 update latency/누적 GPU 비용을 줄이고, stale/단순 갱신보다 유용한지가 핵심이다.
반대로 정확도를 높이는 연구라면 teacher에 없는 유용한 관측/공간 정보 또는 source-stage 학습
기전이 있어야 하며, 그 목표를 feature reconstruction 실험 하나에 떠넘기지 않는다.
인코더 동결은 현재 실험 계약이지 구조 변경 불가능을 뜻하지 않는다. source-stage encoder PEFT는
별도 가능하지만 기존 target-head 소량 적응과 다른 실험이며, 흔한 PEFT 자체를 신규성으로 삼지 않는다.

## 5. Sherrie Wang / Andrew Markham 연구에서 실제로 가져올 것

### 시간과 분광의 구조를 task와 맞추기

[Tong & Wang, Invariant Features for Global Crop Type Classification](https://arxiv.org/abs/2509.03497),
2026-04-14 v3는 지역 전이에 joint spectral–temporal 구조와 phenology augmentation이 중요함을
보인다. crop 과업의 결과를 산사태에 그대로 일반화하지 않는다. 가져올 원칙은 **task를 구분하는
신호가 어느 축에 있고, 우리 요약·augmentation이 그것을 지우는가**를 먼저 묻는 것이다.

우리 적용: half-difference가 아니라 실제 acquisition interval, quality, event-relative ordering을
보존하는 readout. 계절 변화가 label을 보존하지 않는 사건 task에 crop용 time-warp를 그대로 쓰지 않는다.

### 물리 신호와 연산 병목을 함께 해결하기

[Růžička & Markham, HyperspectralViTs](https://arxiv.org/abs/2410.17248), 2025 accepted manuscript는
메탄/광물의 필요한 분광 정보가 초기 압축·중간 제품에서 사라지는 문제를 구조 변경으로 다루고,
synthetic-to-real 및 제한된 하드웨어에서 검증한다. “비싼 모델 대신 작은 모델”만이 아니라
어떤 신호가 없어지는지부터 규명한 사례다.

우리 적용은 메탄으로 즉시 방향을 바꾸는 것이 아니다. 실제 오류를 경계/오경보/가림/시간순서로
나눠 **모델이 필요한 관측을 가지고 있는지** 확인하고, 그 병목에 맞는 입력·학습 목적을 바꾸는 것이다.
Sentinel-2 몇 밴드로 EMIT 초분광 성능을 얻는다고 약속하지 않는다.

### 기존 방법을 새 발명이라고 부르지 않기

- [KalmanNet](https://arxiv.org/abs/2107.10043): 예상과 관측의 innovation을 학습해 갱신하는 원리는
  오래된 필터링 계열이다. 현재 `u-P(m)` gate를 “ours”로 붙인 것만으로 신규성이 성립하지 않는다.
- [From pixels to patches](https://www.microsoft.com/en-us/research/publication/from-pixels-to-patches-pooling-strategies-for-earth-embeddings/),
  2026-03: 여러 Earth embedding의 공간 pooling을 비교한다. mean이 보편적인 상한이라는 주장에
  반례가 되지만, 공간 classification 결과가 우리 temporal segmentation 개선을 보장하지 않는다.
- [RaVÆn](https://www.nature.com/articles/s41598-022-19437-5) 및
  [STTORM-CD](https://www.nature.com/articles/s41598-025-32598-3)는 저비용 변화탐지/관측 우선화의
  직접 선행연구다. onboard·event-priority라는 설정 자체를 최초 기여로 주장하지 않는다.

## 6. 방법을 평가할 과업을 바꾸되, 유리한 결과를 골라내지 않는다

Sen12 terminal mask는 **임베딩 갱신 fidelity 개발 시험**으로 유지한다. 시간 신호가 중요한
실제 사건 시험은 별도 지역/과업에서 사전에 정한다.

[KuroSiwo 공식 데이터](https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-Webdataset)는
2 pre-event + 1 post-event Sentinel-1, DEM, permanent-water/flood 구분을 제공한다.
따라서 “새 관측 뒤 물이 보인다”보다 **상시 물과 새 침수를 구분하는가**를 평가하기 좋다.
이는 2024 NeurIPS Datasets & Benchmarks 자료이며 새 알고리즘은 아니다.

- post-only OLMo, pre/post 단순 feature difference, full joint OLMo, learned update, classical SAR
  change baseline을 **같은 실제 관측·polarization·orbit·processing**으로 비교한다.
- 빠른 지역/센서 전이를 위해 차분 표준화·quality-aware update를 source에서 학습한다.
- task-space loss는 어떤 알려진 task를 보존하고 어떤 새 task에는 실패하는지까지 보고한다.
- 3 acquisition만으로 정확한 재해 onset/detection delay 곡선을 만들 수 없다. 관측시점의
  post-event mapping을 주장하고, 지연 평가는 더 촘촘한 dated evidence가 있을 때만 연다.
- Korea 3-task는 shared asset utility와 추가 task 비용 시험으로 유지. 새로운 development가
  실패할 때마다 sealed label을 조금씩 열지 않는다. 네팔은 별도 앱 repo 그대로 유지한다.

## 7. 현실적인 가능성과 중단선

| 방향 | 지금 근거 | 성립 가능성의 판단 기준 |
|---|---|---|
| 실행/평가 오류 수정 | 0.0 false-fail 실제 재현, local tests 통과 | 높은 확실성의 engineering 개선. 논문 성능 개선은 아님 |
| task-output 보존 loss | 현재 MSE-only와 평가 AP가 다름 | 타당한 후보. matched-module 비교 전 개선 크기 미지수 |
| time/quality-aware update | 현재 updater에 해당 정보가 없음 | 타당한 후보. metadata·실제 causal stream·ablation 필요 |
| 사건특화 transfer | terminal mask와 시간 사건 질문이 다름 | OLMo 활용 목적에 잘 맞지만 새 task에서 직접 검증 필요 |
| CVPR 신규 방법 | generic residual/KL/filtering 자체는 선행연구 존재 | 알려진 방법 대비 accuracy–cost 개선, 외부 event/지역 일반화가 있어야 함 |

실패하면 해당 objective/architecture/data setting을 닫는다. 기존 음성을 지우지 않으며, 새 조합을
끝없이 탐색하지 않는다. 동시에 “오늘 마지막 팔이 졌으니 모든 방법 불가능” 같은 전역 결론도 내리지 않는다.
이번 구현의 기능 테스트 통과와 학습 성공/CVPR novelty는 명확히 별개의 상태다.
