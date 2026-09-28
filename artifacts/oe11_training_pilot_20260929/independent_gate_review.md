# OE11 실제 가중치 학습·재개 파일럿 독립 검토

2026-09-29. 기존 문서, OE10 P2 v3 worker/protocol, frozen text-mask v2, OE11 matching head/adapter를 읽은 실행 전 제안이다. 서버·GPU·실제 자료를 열지 않았고 아래 숫자는 관측 성능이 아닌 사전 고정 가능한 진단 기준이다.

## 최소 범위

- 원본 pinned OlmoEarth에서 시작한다. 기존 P2 checkpoint를 이어 쓰지 않는다. 이번 경로는 Qwen의 **고정 text-only 표현 + mask BCE/Dice**이며 예전 P2의 native replay나 language CE는 없다. encoder adaptation 검사이며 native continued-pretraining 재현 또는 생성 VLM 검사는 아니다.
- 모델 점수 없이 train labels로 고정한 12 base cases를 권한다: 네 target 각각 target-present, target-absent/confuser-present, both-absent 하나씩. K1/K8 prefix를 모두 쓰면 24 episodes다. 존재는 최소64 유효 픽셀, 부재는0픽셀로 명시하고 positive case의 유효 비대상 영역도 확보한다. 가능한 세 부모를 포함하되 geometry/cloud로 사후 좋은 사례만 고르지 않는다. 불가능한 조합은 결손으로 보고하고 다른 클래스로 바꾸지 않는다.
- query는2/5, support는8 dates 그대로. context 조건 하나를 고정하고 모든 원본/반복/재개 경로가 동일한 case order, catalog SHA, context SHA, initialization을 사용한다. calibration 결과를 읽지 않는다.
- 학습성 작은 파일럿은48 updates=24 episode를2회 반복한다. 추가 업데이트를 성과에 따라 자동 연장하지 않는다. 전후24개의 고정 train probe만 측정한다. 최적 checkpoint 선택은 하지 않는다.

## 서로 다른 세 가지 판정

### 1. 계산 경로 통과: 반드시 필요한 공학 관문

- 모든 logits/loss, 존재하는 모든 gradient, pre-clip norm 및 optimizer state가 finite. AdamW는 원 recipe의 `eps=1e-6`, `foreach=False`, `fused=False`; clip norm1.0; FP32, TF32 off를 우선 유지한다. 간단한 파일럿은 encoder1e-5/head1e-3 고정 LR로 충분하다. 48-step 파일럿에 예전96-step warmup을 그대로 복사하지 않는다.
- B2의 mask-only backward에서 EO encoder와 head의 실제 nonzero gradient를 확인한다. 최초 K8 backward에서는 text_project, attention key 경로, scorer, background_scorer의 연결도 기록한다. K1 softmax는 단일 원소이므로 query_key/object_key/text_key gradient=0가 정상이다. 모든 encoder 파라미터가 nonzero여야 한다고 요구하지 않는다: 사용하지 않는 센서 경로가 있다.
- optimizer 전후 CPU clone으로 비교한다. B2에서 실제 encoder와 head 각각 하나 이상의 파라미터가 수치적으로 변경되고 모두 finite여야 한다. gradient가 있는 encoder anchor의 max abs delta를 기록해 weight decay만의 변화를 성공으로 세지 않는다. frozen B0 encoder, Qwen, unused connector는 grad=None 및 값 불변을 확인한다.
- B2 학습 중 EO feature cache는 비활성이다. 저장/재개 직후 EO/text cache를 비운다. encoder.eval()은 gradient freeze가 아니며 dropout/BN 조건을 고정하는 기존 동작임을 기록한다.
- finite/identity/role/cold-resume 실패는 즉시 stop. OOM/timeout은 incomplete이며 학습실패나 방법효과 없음으로 해석하지 않는다.

### 2. 학습 신호: 계산 성공과 별도 결과

계산 성공만으로 학습성 성공이라고 부르지 않는다. 다음 정도를 사전에 진단 기준으로 선택할 수 있다.

- 전후 고정24 training probes의 동일 가중 평균 mask loss가 최소5% 감소.
- 네 클래스의 target-present IoU를 동일 가중해 최소0.02 증가하고, 최소3/4 클래스가 악화하지 않음. threshold는 처음부터0.5 고정한다. K1/K8도 따로 표기한다.
- target-absent/confuser-present 및 both-absent의 threshold0.5 false-positive pixel fraction이 각각 평균0.02 넘게 악화하지 않음. target-present recall, 평균 positive/negative probability도 함께 기록해 all-background 붕괴를 드러낸다.

이 기준은 작은 train set에서의 **진단용 학습 신호**이며, 48 steps에서 미달하면 “이 예산에서 학습성 미입증”이다. 방법 폐기, 큰 데이터 수렴 실패 또는 우월성 판정이 아니다. raw before/after 값을 남기며 hardcoded PASS를 만들지 않는다.

### 3. 의미·효능: 이 파일럿으로 판정 불가

일반화·언어 지식 이해·교정 효율·CVPR 기여는 이번 모든 관문을 통과해도 미검증이다. 이후 같은 head의 EO fixed/update × names/description 대조와 평균-support head, 같은 supervision/관측/튜닝 기회가 필요하다. 클래스명이 사실상 고정 code로 작동하는 가능성도 남는다. 현재 source calibration을 새 지역 전이라고 부르지 않는다.

## 새 프로세스 재개: 작지만 실제로 검사

같은 물리 GPU/환경에서 uninterrupted update1→2와 update1 checkpoint→프로세스 종료→fresh process→update2를 비교한다. 첫 step K1, 두 번째 K8이면 attention 경로도 실제 optimizer 비교에 들어간다. fresh process는 다른 PID여야 하며 초기화 후 checkpoint를 읽고, RNG는 모든 초기화/사전 진단이 끝난 뒤 정확히 복원한다.

저장 대상은 모델 trainable state+buffers, AdamW state, LR/scheduler 또는 고정LR 선언, Python/NumPy/Torch CPU/CUDA RNG, global step, 다음 case cursor/order hash, 원본모델·입력·code·context·protocol identity다. 임시 파일 후 atomic rename하고 checkpoint SHA를 참조한다.

복원 직후 저장 tensor/state는 정확히 일치해야 한다. 이후 update2에서 모델 및 optimizer floating tensors, logits, loss의 max abs difference **<=1e-6**를 사전에 유지한다. integer step/cursor와 RNG는 정확히 일치해야 한다. loaded optimizer moments/step를 검증하지 않는 weight-only restore는 재개 성공이 아니다. 차이가 크면 tolerance를 결과를 보고 완화하지 말고 기록·원인 조사한다. 다른 GPU끼리 비교한 차이를 이 strict gate에 혼합하지 않는다.

## 예산과 공정성

과거 K8 B2의9.065초는 native/CE를 포함한 다른 모델 측정이다. 새로운 실제 K1/K8 시간을 동기화해서 측정하고, text 역할2회 prefill, EO forward/backward, checkpoint IO, peak allocated/reserved memory를 나눠 기록한다. 새 코드가 그보다 빠르거나 느릴 수 있으므로 처리량을 가정한 성능 주장을 하지 않는다.

각 worker hard timeout <=1800초, 전체 실행은 launcher가 사전 고정한 남은 pilot ledger 안에서 관리한다. 모델 load, probe, cold resume, 재시도도 총 점유 시간에 포함한다. 첫 두 workload 후 예상 잔여 시간이 초과하면 incomplete로 종료하거나 launch 전에 scope를 수정·다시 고정한다. 실제 GPU0/1 사용은 허용되어도 다른 작업을 종료하거나 자동 예약을 재개할 근거는 아니다.

EO fixed/update를 병행하더라도 같은 head 초기값/고정 cases/학습순서/steps/텍스트/관측이 기본이다. feature cache 사용 차이는 정보 노출을 바꾸지 않지만 시간 비교에 영향을 주므로 warm/cold 및 cache hits를 기록한다. encoder 갱신 파일럿과 frozen baseline의 미세한 train loss 차이를 효능 결과로 쓰지 않는다.
