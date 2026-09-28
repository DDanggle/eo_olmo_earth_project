# E6 제안: 같은 EO 토큰을 쓰는 작은 판별 head와 E5 VLM 비교

작성: 2026-09-25 06:43 UTC. **설계 검토 초안이며 아직 실행 명세로 동결하지 않았고 실행하지 않았다.** E5 코드와 동결 명세, C0 코드와 명세, 입력 메타데이터만 읽었다. E5 학습 로그, 예측, 점수, 완료 상태는 읽지 않았고 모델·학습·실험 코드를 실행하지 않았다. 이미 노출된 개발 사건을 사용하는 탐색적 후속 실험이라는 한계는 유지된다.

**권고는 새로운 모델 계열을 탐색하는 대신, 공간 토큰을 유지하는 attention head 하나를 3개 seed로 학습하고 E5 `full/native`와 비교하는 것이다.** 입력 형식은 결과와 무관하게 지금 `full=[A,B,B−A]`로 고정한다. E5의 네 형식 중 사후 최고 성능을 고르지 않는다. 이 비교의 질문은 “이 고정된 이진 source-label 과제에서 frozen 7B LLM을 사용하는 시스템이 작은 판별 모델보다 무엇을 더 주는가?”이다. LLM의 언어 능력 전체나 EO 기억 시스템의 효용을 시험하는 실험은 아니다.

## 1. 실제 코드에서 확인한 출발점

- E5는 동일한 5,989개 source item 중 train 4,234개, test 1,755개를 사용한다. train/test의 tile, 홍수 사건, 산사태 지역은 분리된다. 홍수는 train 27/test 10개 사건, 산사태는 train 7/test 2개 지역이다.
- 각 item은 float32 `A,B ∈ R^(64×768)`이다. 각 64개 토큰은 원래 공간 특징을 CPU average pooling으로 만든 8×8 격자를 행 우선으로 펼친 것이다. S1은 48×48에서 kernel 6, S2는 32×32에서 kernel 4를 쓴다. 원본 encoder는 고정되어 있고 시간 정보가 이미 특징에 들어가므로, 이를 “날짜 정보 없는 순수 영상”이라고 부를 수 없다.
- full 입력은 `[A,B,B−A]`, 192개 토큰, type `[0]*64+[1]*64+[3]*64이다. E5는 `LayerNorm(768) → Linear(768,2048) → GELU → Linear(2048,H) → LayerNorm(H)`와 gain/type embedding을 학습하고 frozen Olmo-3-7B-Instruct를 거친다. H는 모델에서 읽으므로 이 검토에서는 임의로 확정하지 않는다.
- E5 질문은 현상과 센서, 두 날짜 외에는 일정하다: 두 관측 사이에 flood/landslide가 발생했는지 yes/no로 답하게 한다. 자유로운 질문 다양성, 장소 검색, 면적 추정, 근거 인용은 이 평가에 포함되지 않는다.
- E5는 세 seed 각각 3 epoch, batch 8, 1,590 update, 12,702 item exposure이다. 동일 seed 내 형식들은 저장된 batch 순서를 공유한다. loss는 yes/no 및 EOS의 언어모델 token loss이고, EOS를 포함한 생성에 대해 파싱 실패 가능성이 있다.
- 주평가는 C1의 품질 대칭 조건 `valid_frac ≥ .90`과 양쪽 클래스가 있는 동일 `(event,dates,slots)` 층으로 정한 902개: positive 445, hard negative 457, 8개 사건이다. 층별 BA를 사건 내 동등 평균하고 사건을 동등 평균한다. test 전체나 기존 positive/pre-pre-negative 914개 평가와 혼합하면 안 된다.

## 2. C0 선형 probe와 다른 점

| 항목 | C0-v1 | 제안 E6 |
|---|---|---|
| 영상 표현 | 각 시점 전체 공간 평균 768차원 | 8×8 토큰 두 시점과 명시적 D 유지 |
| 결정 함수 | convex 선형 logistic | 2층 attention + 비선형 token projection |
| 공간/시점 상호작용 | 평균 이후 선형 결합 | 위치별 특징과 시점 사이 상호작용 가능 |
| 학습 | 현상별 별도 fit, class-balanced BCE, L2, LBFGS 수렴 | 두 현상 공동 학습, E5 batch/노출 예산, unweighted BCE |
| 질문 부가정보 | 날짜·현상·센서 등을 feature로 사용하지 않음 | 원래 질문의 가변 정보를 명시적 숫자/범주로 제공 |
| 출력 | logit≥0 | logit≥0, 확률/logit도 보존 |

C0의 낮은 성능은 공간 정보 또는 비선형 결정 경계의 손실로 설명될 수 있다. 따라서 C0만으로 VLM 기여를 주장하기 어렵다. E6는 이 약한 비교군 문제를 줄인다. 다만 C0와 E6의 차이는 공간 보존 외에도 학습 목적·정규화·메타데이터·공동 학습이 달라, 두 점수 차이를 “공간 attention의 효과”로 해석할 수는 없다. C0와 수치 비교할 때도 반드시 같은 902개 및 같은 층별 집계로 다시 계산해야 한다.

## 3. 후보 하나: Metadata-conditioned Spatial Pair Head

**제안 아키텍처를 아래 한 개로 고정하며 architecture sweep은 하지 않는다.** 구현 전에는 재현 가능한 초기화/attention 연산/수치 설정까지 명세 파일에 옮겨야 한다.

1. float32 full 토큰 `[A,B,B−A]`에 token-wise `LayerNorm(768,eps=1e-5) → Linear(768,128,bias=True) → GELU(approximate='none') → LayerNorm(128,eps=1e-5)`를 적용한다. EO encoder, E5 projector, LLM embedding/가중치는 불러오지 않는다.
2. 학습 가능한 `Embedding(4,128)`에서 type 0/1/3을 더하고, 8×8 위치의 고정 2D sinusoidal vector를 더한다. 행/열 좌표는 각각 0…7. 각 축 u의 64개 성분은 k=0…31에 대해 `sin(u/10000^(k/32)), cos(u/10000^(k/32))`이며, 행 64개와 열 64개를 이어 128차원으로 한다. A/B/D의 같은 격자 위치는 같은 위치 vector를 쓴다. type과 위치를 더하는 배율은 1로 고정한다.
3. CLS를 앞에 붙여 총 193토큰으로 만든다. `CLS=c+Linear(8,128)(m)`이며 `m=[I(flood),I(landslide),(year_A−2000)/100,(month_A−1)/11,(day_A−1)/30,(year_B−2000)/100,(month_B−1)/11,(day_B−1)/30]`이다. ISO 날짜를 엄격히 파싱하고 누락/모순은 무효 처리한다. 현재 센서는 현상과 1:1 대응하므로 별도 중복 센서 feature는 필요 없다. 이 수치화는 날짜의 정보 내용을 유지하지만 LLM의 날짜 해석 방식과 동일하지 않다.
4. 두 개의 독립적으로 초기화된 pre-LayerNorm Transformer encoder block을 쓴다. width 128, head 4, FFN 256, GELU exact, 모든 dropout 0, 모든 linear bias=True, LayerNorm eps=1e-5. attention은 비인과적이며 CLS와 192개 토큰을 모두 본다. encoder block 하나만 쓰면 첫 CLS 집계 이전에 영상 토큰 간 상호작용이 제한되므로 두 개를 제안한다. 두 층의 초기 tensor를 의도치 않게 복제하지 않도록 별도 생성한다.
5. 최종 `LayerNorm(128) → Linear(128,1)`로 CLS logit을 낸다. sigmoid 확률과 raw logit을 모두 저장하며 `logit≥0`을 yes로 고정한다. 온도·threshold 조정은 없다.

위 명세의 학습 가능 parameter는 **367,361개**이다(수식으로 산출; 구현에서 다시 assert 필요). 입력 LN 1,536 + 입력 linear 98,432 + 입력후 LN 256 + type 512 + CLS 128 + metadata linear 1,152 + block당 132,480×2 + 최종 LN 256 + 출력 linear 129. 고정 위치 vector는 parameter가 아니다. 참고로 E5 projector 자체도 `1,576,449 + 2,055H`개의 학습 parameter가 있으므로, “동일 parameter 수 비교”라고 부르면 안 된다.

`id`, tile, 좌표, event/fold, kind, slot 문자열, 정답, flood/valid fraction, C1 membership, 캐시 경로는 입력에 넣지 않는다. `slots`는 평가 층 결정과 원자료 검증에만 사용한다. 훈련 label은 BCE target으로만 전달하고 inference API는 label을 받지 않게 한다. 어떤 종류의 답 token, pretrained 언어 embedding, E5 예측/score도 head 입력에 들어가면 안 된다.

메타데이터를 제공하는 이유는 E5에만 질문 정보를 주는 비대칭을 피하기 위해서다. 주평가 각 층에서는 현상과 날짜가 모두 같으므로, **그 정보만을 보는 결정론적 모델은 층 내 두 클래스를 구분할 수 없고 BA=.5가 된다.** 이 성질을 메타데이터 입력 동일성 및 합성 시험으로 검증하면 별도의 metadata-only 학습 sweep은 필요 없다. 다만 날짜가 영상 특징과 상호작용하거나 encoder 내부에 남은 교락까지 제거되는 것은 아니다.

## 4. 최소 실행과 통제 범위

최소안은 **full head 3개 학습 / 5,265개 test 예측**이다. pair/later/delta의 추가 E6 학습은 이 최초 실험에 포함하지 않는다. E5 네 모델 중 가장 잘 나온 것을 비교 대상으로 고르는 대신 `full/native`만 주대조로 고정한다. E5가 무효면 유효한 비교 결과가 나올 때까지 head 결과만 독립적으로 보존하고 “LLM 비교 성공/실패”를 판정하지 않는다.

- E5가 준비한 `pairs.npy`, `items.jsonl`, `ordered_ids.json`, `batches.json`, `eval_sets.json`을 byte-SHA로 연결한다. 다시 pooling하거나 재선정하지 않는다. 모든 4,234 train을 그대로 사용하며 primary 902개와 일치시키려고 훈련의 pre/pre negative를 제거하지 않는다.
- seed 1/2/3, E5와 동일한 저장 batch ID 순서, 3 epoch, batch 8, 1,590 update/12,702 exposure. AdamW lr=1e-4, weight_decay=.01, betas=(.9,.999), eps=1e-8, warmup/스케줄/early stopping/augmentation/class weighting 없음. float32 학습. BCEWithLogitsLoss의 batch mean; label yes=1/no=0. 같은 학습률이 서로 다른 구조의 동일한 최적화 난이도를 보장하지 않는다는 점을 명시한다.
- 각 seed의 초기 state, software/device, deterministic/TF32 설정, epoch batch 목록, loss/gradient/parameter finite, 최종 checkpoint를 동결·기록한다. 초기화는 명시된 순서로 모듈을 생성하고 각 seed당 한 번 저장한 tensor를 실제 시작 상태로 사용한다. 임의 중단 후 이어 학습하거나 결과가 좋은 seed로 대체하지 않는다.
- 최종 epoch 하나만 test에 한 번 적용한다. threshold는 0, parse 실패 대신 nonfinite logit은 실행 무효. label/type/ID 누락, 원본 SHA 차이, update 수 불일치도 전체 비교 무효이다. partial 결과와 실패 원인은 보존한다.
- wall time, update당 시간, peak GPU memory, trainable/전체 parameter, inference latency도 기록한다. latency는 양쪽 모델이 resident 상태에서 같은 장치/고정 batch와 동일 warmup·측정 규칙으로 재측정할 때만 직접 비교한다. 과거 E5 전체 wall time과 새 head latency를 단순 비율로 비교하지 않는다.

| 통제할 수 있는 것 | 이 최소 설계에서 같게 만들 수 없는 것 |
|---|---|
| EO encoder와 실제 A/B/D byte, source label, train/test/평가 support | 언어 pretraining, 전체/학습 가능 parameter 수, 표현 폭 |
| source 질문의 현상·센서·날짜 정보 내용 | 자연어 tokenization과 숫자 metadata 처리, 원래 LLM positional bias |
| 예제 순서·노출 수·update 수·optimizer hyperparameter | 최적 수렴 예산, architecture별 hyperparameter 적합성, FLOPs/시간 |
| source-label 평가식·사건 resample·seed 공개 | 생성 token/EOS loss와 binary BCE, 출력 vocabulary/파싱 비용 |

따라서 이는 **같은 source data와 예제 노출 예산에서의 두 시스템 비교**다. LLM만 제거한 완전한 인과적 ablation도, 같은 계산량·같은 parameter 수 비교도 아니다. 작은 head에 맞춰 classifier를 재학습하는 것이 비교 자체의 목적이므로 이 차이를 숨기지 않는다.

## 5. 주가설, 통계와 판정 제안

주 estimand는 각 seed의 `Δ=BA(E5 full/native)−BA(E6 full head)`이다. 양쪽 모두 동일 C1 902개에서 동일 층·사건 가중으로 집계하고, 8개 사건을 쌍으로 5,000회 resample한다. NumPy default_rng(20260925), linear percentile 95% CI를 E5와 맞춘다. seed는 3개의 독립 사건 표본이 아니므로 24개 사건처럼 늘려 세지 않는다. seed별 BA·Δ·CI와 8개 사건별 값을 모두 공개한다.

**현재 권고는 Δ와 CI를 주 결과로 먼저 명세하고, 새 성공 threshold를 만들지 않는 것이다.** 운영상 이분 판정이 꼭 필요하다면 아래는 아직 채택되지 않은 제안이며 실행 전 계획 파일에 확정해야 한다. E5 결과를 본 뒤 경계/대상 arm/seed 규칙을 바꾸면 해당 변경을 명시한 다음 버전으로 남겨야 한다.

- “VLM 시스템의 관측된 우위” 후보: 같은 seed에서 E5 BA≥.60이고 Δ CI의 하한>0인 경우가 2/3 이상. .60은 기존 E3–E5의 진단적 바닥값을 유지한 것일 뿐 실용적 효과 크기를 보증하지 않는다. 통계적으로 작은 양의 차이를 CVPR 기여나 실사용 가치로 포장하지 않는다.
- “이 head가 허용 오차 내 성능 보존” 후보: 같은 seed에서 양쪽 BA≥.60이고 `head−VLM` CI 하한≥−.05인 경우가 2/3 이상. 기존 .05 탐색 margin을 재사용한 제안이며 실제 피해 비용으로 검증된 동등성 범위가 아니다. 새로 근거 없이 더 넓은 margin을 만들지 않는다.
- 나머지는 혼합/불충분. 두 후보가 동시에 성립할 수 있다(예: 작지만 정밀한 VLM 우위). 둘을 억지로 배타적 성공/실패로 만들지 말고 연속 효과와 tolerance 해석을 함께 기록한다. “유의하지 않음”은 비열등성/동등성의 증거가 아니다.

Secondary는 기존 positive/pre-pre-negative 914개 사건 평균, hard-negative FPR, 산사태 2개 지역별 BA, full test 전체 예측 보존이다. 이들 중 좋은 subset을 새 primary로 올리지 않는다. head 확률의 AUROC/AP와 생성 답의 BA는 서로 다른 지표이므로 직접 차감하지 않는다. 확률 기반 분석을 추가하면 head의 보조 진단으로만 명시한다.

## 6. 실패 조건과 해석의 비대칭

1. **head가 비슷하거나 더 좋은 경우:** 이 고정 이진 source-label 평가에서 소형 판별 모델이 강한 대안임을 보여준다. LLM을 쓰는 이유를 자유 질문, 공간 근거, 의미 전이 등 실제 추가 기능으로 증명해야 한다. 하지만 이 결과만으로 LLM이 EO 전체에 쓸모없다는 결론도 안 된다.
2. **VLM이 더 좋은 경우:** 이 architecture와 동일 노출 예산에서의 시스템 우위다. 언어 지식의 효과, LLM의 필수성, 시간 추론, memory 이득은 분리되지 않는다. head의 capacity·학습 난이도·loss 차이와 추가 언어 pretraining이 대안 설명이다. 다음 단계가 필요하면 별도의 train-only 최적화/용량 통제 계획을 동결하고 새 평가 자료로 확인해야 한다.
3. **둘 다 낮거나 사건별 방향이 다른 경우:** 이 label/표현/지원집합에서 원래 질문에 답하기에 근거가 부족한 것이다. 신뢰구간을 좁히려고 seed 수를 사건 수처럼 늘리거나 유리한 사건만 남기지 않는다.
4. **head가 거의 상수만 예측하는 경우:** 현재 3 epoch의 최적화 실패 또는 표현의 분리 어려움을 구분할 수 없다. train loss·gradient·출력 분포를 보고하며, 낮은 head 성능을 곧장 VLM 기여로 세지 않는다. 실행 전 합성 task에서 gradient/label 방향/위치·시점 접근을 검증하는 것은 허용되지만, 실제 test 결과에 맞춘 hyperparameter 재시도는 최초 실험에 합치지 않는다.
5. **데이터/모델/완료 검증 실패:** 비교 실행 무효. 다른 데이터·seed·체크포인트로 조용히 교체하지 않는다.

이 실험은 VLM이 꼭 필요한 연구 문제를 선명하게 만드는 진단이다. 어떤 방향의 결과가 나와도, 현재 날짜의 신뢰성·source-label과 실제 변화의 간극·8개 개발 사건·센서/현상 교락은 남는다. “근거를 붙여 무엇이 언제 변했는지, 무엇을 모르는지 답하는 기억” 주장의 검증을 대신하지 않는다.

## 검토에 사용한 source와 SHA256

repo root: `/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project`. 아래는 계획/코드이며 E5 산출 점수 파일은 읽지 않았다.

- `code/e5_bundle_v0/e5_train_v0.py`: `3f33873711e6b94a8bb54d50ee4dbd5b5492da22c41d848ef4f664c0894b398b`; projector/source_prompt/label 마스킹/학습 예산.
- `code/e5_bundle_v0/e5_prepare_v0.py`: `508f444a512fcbd3f6937dd51b8ae2755c1f4deb334bdaed118c98e78840fd61`; pooling/입력 provenance/평가 층.
- `code/e5_bundle_v0/e5_equal_budget_prereg_v0.json`: `fc7866feebbf1e9fcd4e93de5154bf9ffa141f434e70dc4731390cf18dfd4696`.
- `config/c0_linear_view_prereg_v1.json`: `9664dbe590455dbae724a62d9e8b9f0cdb03fa098208386aa9de88dc339b4e43`.
- `code/c0_linear_view_probe_v1.py`: `5bea6041184b2593407229c3ff1ae6239d878002376b08ab165511acc643c612`; global mean/train-only standardization/class-balanced loss.
- 부가 read-only 확인: `code/e5_bundle_v0/frozen_inputs/items.jsonl` 앞 두 metadata row, `code/extract_kurosiwo_s1_cache.py` timestamp 입력 코드. E5 예측·점수·학습 로그는 사용하지 않았다.
