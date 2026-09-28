# OE4 — 신규성 후보 검토와 실제 OlmoEarth–VLM 실행

> **9/27 15:43 KST GPU1 완료 — [실제 VLM 실행 결과와 다음 실험](OE4_GPU1_VLM_PROGRESS_20260927.md).** native8/32step 학습·저장복원과 실제 Qwen3-VL8B FP32 연결/역전파 검사 통과. 언어손실로 OlmoEarth211개·연결부4개tensor에gradient, 추적가중치변경과 원RGB경로보존 확인. BF16캐시검사실패는보존하며 FP32통과와구분한다. VLM checkpoint저장복원·새지역전이·성능개선은미검증. 이번GPU작업종료·결과/로그회수완료. 다음은PASTIS공개라벨입력2사례계약확인이다. 아래이전대기/실패표시는당시이력이다.

2026-09-27 02:45 KST snapshot. 이번 갱신은 기존 계획을 실제 자료·모델·학습 경로로 옮기는 작업이다. **연구를 진행할 이유는 있으나, 새 방법의 효과나 CVPR 채택 가능성이 입증된 것은 아니다.** 서버 receipt가 이 문서보다 더 최신일 수 있다.

## 무엇을 새로 주장할 수 있어야 하는가

목표는 사용자가 특정 식생·구조물의 예시나 교정을 주었을 때 OlmoEarth 기반 VLM이 새 지역에서 해당 대상을 찾고, 영역과 관측을 근거로 설명하는 모델이다. 측정은 적절한 해상도와 독립 참조자료가 있는 속성부터 다룬다. 10m 관측으로 개별 고사목·작은 해양쓰레기를 직접 식별한다고 약속하지 않으며, 단기간의 영상 차이를 climate 변화나 인과적 피해로 부르지 않는다.

문헌 검토는 9개 축, 21개 primary source를 정리했다. 별도 방법 검토와 구현 검토를 포함해 세 에이전트가 맡았다. [신규성 지도](../artifacts/oe4_native_v12_v0_20260927/novelty_map.md)와 [구체적 대조 실험](../artifacts/oe4_native_v12_v0_20260927/mechanism_candidates.md)에 근접 선행·반증 조건을 남겼다. 완전한 선행 부재 증명은 아니다.

| 후보 | 기존 연구와 겹치는 부분 | 아직 확인할 만한 부분 | 결정적인 대조 |
|---|---|---|---|
| 공간 개념의 교정과 전이 | 예시 segmentation, LoRA/SFT, retrieval 기반 model editing | 소수 교정이 새 지역·관계 질의에 전이되고 비관련 개념과 원 EO 능력은 유지되는가 | 같은 K개 라벨의 prototype, retrieval, LoRA, episodic learning; 새 지역의 실제 VLM 응답 |
| 관측에 따라 보존할 정보를 구분하는 학습 | band dropout, missing-modality robustness, uncertainty, query conditioning | 모든 관측을 같은 표현으로 만드는 학습보다 관측 가능한 속성과 사라진 세부 정보를 잘 구분하는가 | 공식 목적+band dropout, unconditional consistency, heteroscedastic baseline, post-hoc calibration |
| 영역·해상도에 맞는 측정 일관성 | multi-scale EO representation, 합·평균의 보존 | 경계·해상도 변화에서 의미와 측정의 일관성을 함께 높이는가 | mask+정확한 계산기, area-weighted resampling; 별도 독립 target |

첫째는 응용상 가장 중요하고, 둘째는 OlmoEarth 학습에 실제 결함이 있는지를 먼저 반증하기 좋다. 셋째는 주방법에 필요할 때 추가한다. query별 token pruning, 다중 task head, AI 여러 개의 합의 라벨, generic contrastive loss는 그 자체로 신규성이라고 내세우지 않는다.

TerraScope와 차이는 기능 목록으로 만들지 않는다. TerraScope가 이미 pixel grounding·측정·멀티센서/시간 reasoning을 다루므로, 우리 쪽은 **같은 감독량에서 OlmoEarth를 학습하는 방식이 달라졌을 때, 새 개념의 적응과 새 지역의 실제 VLM 답변이 더 좋아지는가**를 보여야 한다. mask가 좋아져 면적과 설명도 좋아진 것을 세 가지 독립 능력 전이로 세지 않는다. 새 연결부·새 VLM에서도 이점이 남는지 확인해야 encoder 학습의 재사용성을 주장할 수 있다.

## 이번에 실제로 만든 기반

- 공식 v1.2-Base checkpoint를 revision과 SHA256으로 고정했다. weights는 1,030,354,339 bytes이다.
- 공식 1k subset archive 17,855,037,440 bytes를 확보·검증·해제했다. H5 파일은 3,996개이다. 하나의 원 지역이 여러 파일로 나뉘므로 독립 지역 수로 세지 않는다. 원 pretraining corpus 일부이므로 외부 평가 자료도 아니다.
- 공식 source commit `0497dfbb6711ded4e6bf10cf089fc1e4d58c186b`의 loader, normalization, masking, loss를 연결했다. 공용 환경은 바꾸지 않고 필요한 dependency를 별도 디렉터리에 설치했다.
- train 64개, dev 8개 파일의 실제 입력 준비가 통과했다. 날짜·compact time·밴드 순서·정규화·원 crop hash를 기록했다. 이 split은 파일을 나눈 공학 진단이며 지리적 holdout이 아니다.
- 입력 4사례를 실제 RGB/WorldCover 그림으로 확인했다. 분홍색 농경지는 정상 코드 40이며, 점검한 4사례의 unknown class code는 0개이다. 정적 WorldCover를 날짜별 변화 정답으로 사용하지 않는다.
- 전체 3,996개 H5의 좌표를 읽은 결과 서로 다른 저장 좌표는 999개였다. 현재 72파일은 70좌표 그룹이며 train/dev 사이 동일 좌표 중복은 0개이다. dev 내부에는 같은 부모 지역 조각이 있다. 원 지역을 기준으로 묶는 후보 split을 별도로 만들었으며 기존 실행 manifest는 보존했다. 다른 좌표라는 이유만으로 공간적으로 독립이라고 판정하지 않는다.
- 실제 Qwen3-VL-8B에 별도 EO 슬롯을 넣는 코드와 real-H5 provider를 구현했다. 기존 RGB·DeepStack 경로를 보존하고 언어 손실이 OlmoEarth까지 도달하는지 검사한다. 이 단계의 일반 문장은 공학용 loss trigger이며 정답 라벨이 아니다.

### 학습 단계와 상태

실행 전 범위: [원 JSON 명세](../config/oe4_execution_scope_v0.json), [오류 수정 후 실행 개정](../config/oe4_execution_scope_v1.json). 파일별 hash와 원 log는 artifact에 보관한다.

| 단계 | 목적 | 규모 | 현 상태 |
|---|---|---|---|
| 원자료 준비 | 공식 H5 입력 계약 확인 | train64/dev8 | 완료 |
| CPU native 계약 검사 | 실제 forward/loss/update·저장 복원 | train2/dev2, 1step | v0 실패 보존, 수정 v1 통과 |
| Native smoke | 공식 모델 loss→encoder update→저장/복원 | train8/dev4, 8step | 유휴 GPU 대기 |
| Native development | 여러 실제 파일에서 같은 경로 안정성 | train64/dev8, 32step | 앞 단계 성공 후 실행 |
| 실제 VLM interface | Qwen3-VL loss→adapter→OlmoEarth, RGB 경로 보존 | 실제 1사례, 1step | 순차 대기 실행기 시작, 앞 단계 성공 대기 |
| 방법 비교 | 새로운 학습법의 우월성/전이 검증 | 독립 참조자료·matched arms 필요 | 미실행 |

Native run은 각기 원본 Base에서 시작한다. S2 context와 WorldCover target, 작은 crop과 두 시점만 사용하며 공식 trainer 전체 재현은 아니다. 공식 objective 구현을 사용하는 경로 검증이다. FP32 single-GPU, torch2.13 환경과 공식 요구 torch2.9의 차이도 기록한다. WorldCover가 균일하면 base loss가 0일 수 있어 contrastive와 base의 양수 step을 따로 기록한다. **loss가 줄었다는 이유만으로 통과시키지 않고**, gradient·실제 가중치 변화·공식 loader 복원 일치를 확인한다.

두 GPU에는 기존 작업이 있다. 새 controller는 기존 작업을 중단하지 않고, compute process가 없고 메모리/사용률 조건을 연속 확인한 후 실행한다. native 각 run의 최대 대기는 90분, worker는 최대 1,100초이며 실패하면 중단하고 원 로그를 보존한다. 실행 중 source는 snapshot으로 고정한다.

실행 중 발견한 두 마찰을 보존했다. controller v0는 GPU 사용률의 일시적 `[Not Found]`를 정수로 바꾸다 학습 전에 종료됐다. 알 수 없는 상태를 BUSY로 처리하도록 수정하고 7개 회귀 검사를 통과했다. 이어 실제 public checkpoint로 CPU 사전 검사를 하자 loss 설정의 registry `type`이 빠져 `KeyError`가 발생했다. 공식 `LossConfig.build()`가 `pop('type')`로 설정을 소비하고 그 뒤 설정을 저장하는 경로를 확인했다. 공식 recipe와 나머지 설정이 정확히 맞는지 확인해 메모리에서 누락 key를 복원하는 bridge를 만든다. 원 checkpoint/source를 고치지 않는다. 이 결함이 있는 코드가 GPU에서 실행되지 않도록 우리 controller v1만 종료했다. 기존 GPU 작업은 중단하지 않았다.

수정 runtime v1의 실제 CPU 1step 결과는 `PASS_bounded_encoder_update`이다. 213개 encoder 파라미터 텐서에 비영 gradient를 확인했고 실제 표본 가중치 변경과 저장/공식 loader 복원도 통과했다. base와 contrastive loss가 모두 활성화됐다. dev loss는 약 `+0.00001663` 증가했으며, 1step의 이 수치로 개선·열화를 주장하지 않는다. 데이터 4파일의 학습 경로 검증 결과이며 GPU/대규모/의미 성능 증거가 아니다.

서버 위치: `/home/work/data/olmoearth/oe4_native_v12_v0/`. 현재 native 상태는 `bounded_sequence_v2_status.json`, 로그는 `controller_v2.log`, 각 run 결과는 `runs/*/receipt.json`이다. VLM은 `vlm_sequence_v1_status.json`과 `vlm_controller_v1.log`를 확인한다. v0 실패와 종료한 v1 native 기록은 별도 보존한다. VLM 첫 패키지는 macOS 부가파일 `._*.py`를 source guard가 발견하여 학습 전에 거부했고, 부가파일 없는 새 snapshot v1으로 재시작했다. 현재 native는 `waiting_for_idle_gpu`, VLM은 `waiting_for_native_sequence`이다. VLM 실행기는 native 완료를 최대 3시간 기다리고 이후 유휴 GPU를 최대 30분 기다린 뒤 한 번 실행한다. 선행 실패·시간 초과에는 중단한다. 두 단계가 실제 완료되기 전에는 'GPU 학습 완료'라고 보고하지 않는다.

## 다음 비교를 어떻게 논문 근거로 만들 것인가

1. **원본 모델의 오류부터 고정한다.** native 학습/실제 VLM 연결이 통과하면 같은 지역·질문에서 EO 입력 유무, 잘못 연결한 EO, RGB만, 원본 OlmoEarth를 비교한다. 원본 OlmoEarth+adapter를 같은 데이터로 충분히 학습한 대조, 원 band/index 계산기·고전적 regressor, 동일 관측을 받는 RGB-only를 함께 둔다. 랜덤/1step adapter의 실패를 OlmoEarth 정보 손실로 귀속하지 않는다. 일반 문구/지역 메타데이터만으로 맞는 사례와 관측을 실제로 써야 하는 사례를 나눈다.
2. **정답의 출처를 먼저 확보한다.** 공개 parcel/dense annotation 등에서 원 지역·시점·라벨 단위를 확인한다. 서버에 있는 PASTIS 파일 두 개는 존재만 확인됐고 완전성/독자 검증이 끝나지 않았다. WorldCover/S2의 날짜 불일치를 변화 정답으로 바꾸지 않는다.
3. **동일한 감독을 쓰는 작은 두 arm부터 실행한다.** 일반 학습과 후보 학습에 같은 pair·질문·정답·token/forward 예산을 준다. feature consistency나 listwise contrast만으로 이점이 설명되면 더 큰 신규성 주장을 하지 않는다. 이 실행은 개발 실험으로 남기며, 결과를 본 뒤 본 test를 다시 설계하지 않는다.
4. **효과가 나오면 스케일을 늘린다.** 지역별 분리, 최소 3 seeds, K-label 적응곡선, 비관련 EO 능력 유지, 실제 VLM 답과 region grounding을 함께 평가한다. 지원 예시를 query와 섞지 않고, 원 지역 단위 bootstrap과 동일 budget 비교를 쓴다. 개발 효과가 불안정하면 데이터/목적을 고친 새 revision으로 다시 비교한다.
5. **주장을 단계별로 높인다.** 연결 성공 → 원본 오류 확인 → 단순 baseline 이상의 효과 → 새 지역/새 reader 전이 → 대규모 재현 순서다. 이 중 첫 단계만 끝났는데 나머지를 완료한 것처럼 쓰지 않는다.

원본의 독립 EO 기준선과 공식 목적의 추가 학습 비교는 계속 필수이다. 기존 21 core pipelines 계획은 전체 확증 비용 계획이며, 이번 8/32step 진단을 그 결과로 대체하지 않는다. 사람 예산 100만 원·공개 label audit 계획은 유지하고 아직 유료 API·인력 발주를 하지 않았다.

## 출처와 변경 이력

- [OlmoEarth v1.2](https://arxiv.org/html/2605.20804v3): 기존 band dropout과 single S2 bandset을 대조 기준으로 사용한다.
- [TerraScope](https://arxiv.org/html/2603.19039v1): pixel-grounded EO VLM과의 중복을 비교한다.
- [ReasonEdit](https://arxiv.org/html/2602.02408v5): 사람 reasoning의 retrieval 기반 교정 전이는 이미 존재한다.
- [Modality missingness faithfulness](https://arxiv.org/html/2609.07596v1): 관측 제거·복원과 자기 설명의 일치도 역시 선행이 있다.
- [전체 출처 장부](../artifacts/oe4_native_v12_v0_20260927/sources.json), [기존 CVPR 연구계획](CVPR_2027_OLMOEARTH_RESEARCH_PLAN_20260927.md).

기존 D1/OE1/OE2 판정과 결과를 수정하지 않는다. 이번 확보·전송·대기 상태는 이전 문서의 '다운로드/실행 없음' 시점 이후의 새 기록이다.


후속 설계: [지역·계절 맥락 학습 검토](REGIONAL_CONTEXT_LEARNING_20260927.md). Wikipedia/기후와 EO를 함께 학습한 선행을 반영한 후보 설계이며, 이 문서의 실행·판정은 변경하지 않는다.
