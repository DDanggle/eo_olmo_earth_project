# Robotics / world-model / annotation mechanisms for OlmoEarth VLM

검토일: 2026-09-27. 독립 조사 메모. 현재 연구계획 v4를 읽고, 원 논문·저자 저장소를 확인했다. 아래 7개는 메커니즘을 빌리거나 강한 대조를 만드는 데 유용한 후보이며, EO 성능을 입증하는 근거가 아니다. GPU 실행·가중치 다운로드·저장소 변경은 하지 않았다.

## 판단

가장 먼저 가져올 것은 **근거를 추가로 읽는 제한된 도구 루프**, **새 판독 모듈의 gradient가 원 표현에 주는 영향을 분리하는 학습**, **필요한 교정만 요청하는 검수 루프**다. 로봇 행동 생성이나 대규모 생성 세계모델을 먼저 이식할 필요는 없다. OlmoEarth의 가중치와 native EO 평가가 연구의 중심에 남아야 한다.

특히 SAM 3는 텍스트·예시·양성/음성 교정과 MLLM 도구 루프를 이미 제공한다. 따라서 “전문가 교정+VLM+반복” 자체를 새로움으로 부를 수 없다. 차별점은 **native 분광·불규칙 시계열에서 어떤 근거/교정을 추가하면 구별이 개선되는지 학습하고, 그 개선이 새 지역과 OlmoEarth 표현에 남는가**로 좁혀 검증하는 편이 낫다. 이는 제안이며 아직 관측된 결과가 아니다.

## 7개 원 연구와 실제 이용 상태

### 1. SAM 3 (2025-11): 직접적인 강한 비교군

- [논문](https://arxiv.org/html/2511.16719v1), [공식 저장소](https://github.com/facebookresearch/sam3), [공식 개요](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/).
- 텍스트/이미지 예시로 개념에 맞는 객체를 모두 찾고, 존재 판단과 위치 판독을 분리한다. 데이터 엔진은 mask 제안 뒤 적합성·누락을 검수한다. 부록 G의 SAM 3 Agent는 MLLM이 마스크를 제안·검사·선택/없음 반환하는 도구 루프이며 단순 자기평가 문장 반복이 아니다.
- **이용:** 코드·학습/추론 예제·agent notebook 공개. 가중치는 Hugging Face 접근 승인·인증 필요. 이 프로젝트에서 승인·다운로드한 상태는 아니다. 현재 저장소는 SAM 3.1도 안내하므로 재현 시 버전 고정 필요.
- **이식:** OlmoEarth dense head의 mask/관측을 도구 결과로 주고 Qwen이 다음 확인 대상을 고르는 대조. 없는 대상 반환과 누락 검수도 평가.
- **한계:** RGB 객체 개념 분할이 native S1/S2 분광·시간 의미를 보장하지 않는다. SAM 3+Qwen RGB 기준선과 같은 native 자료를 받는 OlmoEarth dense head+Qwen 도구 기준선을 모두 구분해야 한다. 부록의 일부 긴 반복 실행을 우리 예산으로 그대로 채택하지 않는다.

### 2. V-JEPA 2 (2025-06): 예측과 계획의 구분

- [논문](https://arxiv.org/abs/2506.09985), [코드·checkpoint·설정](https://github.com/facebookresearch/vjepa2).
- 잠재 표현 예측을 사전학습하고, 별도 action-conditioned 모델을 로봇 상호작용 자료로 학습해 이미지 목표에 대한 계획을 수행한다. VLM 정렬을 통한 영상 질의응답도 연구한다.
- **이용:** encoder·action-conditioned predictor·학습 설정 공개. 이번에 실행하지 않았다.
- **이식:** 실제로 관측 가능한 시점/영역 후보를 제한된 비용으로 더 읽고, 정답 영역의 불확실성이 줄었는지 확인하는 관측 선택 실험.
- **한계:** 로봇 action은 환경을 바꾸지만 “과거 SAR 영상을 더 읽기”는 환경 상태를 바꾸지 않는다. 이 둘을 같은 제어/인과 추론이라고 부르면 안 된다. 미래 EO 특징 예측도 기후/수리학 법칙을 학습했다는 증거가 아니다.

### 3. V-JEPA 2.1 (2026-03): dense 표현과 global 능력의 충돌을 측정

- [논문](https://arxiv.org/html/2603.14482v1), [공식 구현·80M/300M/1B/2B 가중치](https://github.com/facebookresearch/vjepa2).
- 가려진 token뿐 아니라 관측 context token에도 위치별 예측 손실을 주고 중간층에도 감독을 준다. 논문 ablation에서는 context loss 단독이 dense 판독을 개선하지만 global 분류를 낮추며, 여러 층의 감독이 이를 회복하는 양상을 보인다.
- **이용:** 공개 코드·모델·설정 확인. 논문 HTML은 v1 header와 내부 수정 날짜가 달라 실제 인용/재현 시 PDF/commit을 고정할 것.
- **이식:** 같은 EO 자료·예산에서 공식 목적만 추가학습 / dense 감독 추가 / 중간층 감독 추가를 비교하고 독립 EO 능력 보존을 함께 본다.
- **한계:** 이것을 그대로 추가해도 우리 신규 방법이 되지 않는다. OlmoEarth가 실제 사용하는 v1.2 목적·target·mask 경로와의 중복을 먼저 감사해야 한다. 이 논문의 image/video tokenizer는 언어용 BPE가 아니라 서로 다른 patch embedding이다.

### 4. Knowledge Insulation (2025-05): 무조건 공동학습 대신 gradient 경로를 검증

- [논문](https://arxiv.org/html/2505.23705v1), [공식 openpi](https://github.com/Physical-Intelligence/openpi).
- 새 continuous action expert의 gradient를 pretrained backbone에 모두 흘리는 방식과, 이를 차단하면서 별도 이산 action/언어 목적은 backbone을 학습시키는 방식을 비교한다. 원 표현을 완전히 동결하는 것과는 다르다.
- **이용:** 공식 openpi의 pi0.5와 관련 가중치·학습 경로 공개. 우리 EO 구조와 동일 구현은 아니다.
- **이식:** 새 EO dense head에서 올라오는 gradient, language loss, native loss의 충돌을 측정하고 head warmup·부분 stop-gradient·native replay를 강한 비교로 둔다.
- **한계:** 현재 OlmoEarth가 나쁘거나 gradient가 이미 손상을 준다는 근거는 없다. 이 논문의 로봇 결과를 우리 모델의 원인 설명으로 쓰지 않는다. OlmoEarth 전체를 고정하면 사용자가 원하는 표현 학습 주장을 시험하지 못한다.

### 5. π*0.6 / RECAP (2025-11): 검증된 경험으로 학습하는 루프

- [논문](https://arxiv.org/html/2511.14759v1), [공식 공개 모델 목록](https://github.com/Physical-Intelligence/openpi).
- 실제 수행 결과·전문가 개입·기존 시범을 모으고 value/advantage를 추정한 뒤 advantage 조건으로 정책을 학습한다. 학습 없는 agent 도구 반복과 구분되는 RL 과정이다.
- **이용:** 논문 공개. 확인한 공식 openpi README는 pi0/pi0-FAST/pi0.5만 공개 목록에 포함한다. RECAP 공식 checkpoint/전체 재현 코드를 확보했다고 말할 수 없다. 검색에 뜨는 RECAP fork는 저자 공식 구현이 아니다.
- **이식:** 교정 전후 실제 품질 변화를 학습지역의 분리 fold에서 계산해 “어떤 교정이 유효한가”를 감독할 수 있다. 실패 기록도 보존한다.
- **한계:** 모델 자체 점수나 여러 AI의 합의를 실제 reward로 부르지 않는다. annotation utility 회귀만 구현하면 offline supervised selection이며, 그 자체를 RECAP/RL 구현이라 부르지 않는다.

### 6. DreamGen (2025-05): 생성 자료와 실제 효과를 연결한 검증

- [논문](https://arxiv.org/abs/2505.12705), [저자 프로젝트](https://research.nvidia.com/labs/gear/dreamgen/), [공개 파이프라인 저장소](https://github.com/NVIDIA/GR00T-Dreams).
- 생성 영상에서 latent action 또는 inverse dynamics로 pseudo-action을 얻어 실제 데이터와 함께 학습한다. 영상 생성 품질을 downstream 실제 행동 성공과 연결해 평가한다.
- **이용:** 코드/파이프라인 문서는 공개. Cosmos 등 의존 모델·자료별 별도 접근·컴퓨트가 필요하므로 “작은 EO GPU 실험에 준비 완료”로 세지 않는다.
- **이식:** 관측 조건 변화나 도메인 변형을 만들더라도 실제 holdout 개선으로 유효성을 판단하는 원칙.
- **한계:** 생성 위성영상에 flood extent, SAR 산란, 농작물 생리 정답이 자동으로 생기지 않는다. 자연어로 만든 홍수 영상은 수리학 시뮬레이션이 아니다. 먼저 실제 자료의 label-preserving crop/결측 표현 같은 검증 가능한 변형만 쓰는 편이 낫다.

### 7. FAST (2025-01): '지구 단어'보다 먼저 rate–distortion 확인

- [논문](https://www.roboticsproceedings.org/rss21/p012.pdf), [저자 설명](https://www.pi.website/research/fast), [공식 tokenizer·학습 코드](https://huggingface.co/physical-intelligence/fast).
- 연속 행동 시계열에 DCT→양자화→BPE를 적용해 짧은 이산열로 표현한다. 공개 FAST+는 로봇 행동으로 학습됐으며 새 tokenizer를 학습하는 코드도 제공한다.
- **이식:** 날짜·센서·validity를 보존한 EO 구조 표현을 이산 코드로 압축할 수 있는지, 같은 byte/token/시간에서 연속 resampler와 비교하는 저비용 진단.
- **한계:** 로봇 행동 tokenizer를 EO 임베딩에 그대로 쓰는 것은 근거가 없다. 불규칙 취득 간격/구름 결측을 균일 시계열로 취급하면 안 된다. 코드 ID가 기존 언어 단어의 의미와 정렬됐다는 보장도 없다. token vocabulary를 Qwen에 붙여도 의미 정렬 학습과 원 독립 EO 평가가 필요하다. bag-of-words는 순서/위치를 버리므로 주 모델 후보보다 의도적인 약한 대조에 적합하다.

## 'Loop'는 네 가지를 분리해서 기록한다

| 종류 | 실제로 바뀌는 것 | 우리 작업의 정확한 이름 |
|---|---|---|
| 반복 추론/도구 검색 | 읽는 ROI·날짜·mask와 답변, 가중치 고정 | 제한된 근거 선택/검증 루프 |
| 사람 교정과 재학습 | 검수된 학습 데이터와 다음 모델 가중치 | active learning / correction loop |
| RL 정책 학습 | 실제 outcome reward를 이용한 관측/도구 선택 정책 | reward·rollout·정책 업데이트를 실제 구현했을 때만 RL |
| 새 위성 관측 도착 | 사용할 수 있는 센서/날짜 자료, 상태 추정 | streaming observation update; 자동으로 online learning이 아님 |

같은 답을 여러 번 말하게 하거나 AI끼리 동의하게 하는 것만으로 근거/정답이 추가되지 않는다. 모델 설명의 설득력 대신, 추가 관측/교정 이후 정답 mask·혼동 오류·근거 일치가 실제로 바뀌는지 측정한다.

## 낮은 비용으로 반증 가능한 3개 실험

아래 숫자는 개발 pilot 제안이며 검정력 계산이나 동결된 사전등록이 아니다. 먼저 부모 작업의 PASTIS 실제 입력·정답 계약과 학습 기준선이 통과해야 한다. 데이터 추가 구매·모델 대량 다운로드·GPU 예약을 암묵적으로 요구하지 않는다.

### E1. 필요한 관측을 다시 읽는 것이 그냥 많이 읽는 것보다 나은가

- 개발 20–40 episode에서 시작. 이미 가진 시계열 후보와 같은 positive/confusion support만 사용한다. 도구는 `read_region(date, roi)`, `compare_support(id)`, `return_mask`, `insufficient_evidence` 정도로 제한한다. 실제 unavailable 관측은 요청해도 반환하지 않는다.
- 같은 원자료 접근권과 총 관측량/token/GPU 시간 예산에서 단발 learned resampler, 균등 날짜/영역, uncertainty-greedy, 같은 Qwen 도구 baseline, 제안 학습 selector를 비교한다. 원격 API 비용은 추가하지 않는다.
- loop가 고른 최종 근거 집합을 단발 reader에 **그대로 재생하는 대조**를 둔다. 차이가 사라지면 이득은 반복 추론보다 선택된 자료에 있다. oracle 선택은 별도 상한이며 실현 가능한 모델로 세지 않는다.
- 결과: foreground IoU·혼동 오탐·대상 없음 오류 대 총 읽기 비용; 올바른 답→잘못된 답으로 바뀌는 harmful revision 비율. 더 많은 호출로만 좋아지거나 단순 greedy와 같으면 복잡한 루프 주장을 버린다.
- PASTIS 작물 정답만으로 변화 시점/재해 회복 정답을 생성하지 않는다. 이 단계에서 말할 수 있는 것은 관측 선택을 통한 작물 구별이다.

### E2. 새로운 dense/language 학습이 OlmoEarth에 도움을 주는 경로는 무엇인가

- 동일 개발 자료·정답·head·initial checkpoint에서 full joint, head warmup 후 joint, native+language는 encoder 업데이트하되 새 dense expert의 gradient만 차단하는 방식 비교. 새 모듈의 업데이트 횟수·총시간·encoder gradient 경로를 기록한다.
- 다음 단계에서 dense context/multilevel 목적을 추가한다면 공식 native 목적과 실제 중복을 먼저 확인하고 같은 계산비용으로 비교한다. 두 변경을 동시에 켜서 원인을 모르게 하지 않는다.
- 결과: 독립 EO probe/고정 평가, dense 판독, 실제 Qwen 조건 질의 정확도. gradient cosine은 진단일 뿐 성능 대용이 아니다. 언어 답변 개선과 native 능력 하락의 tradeoff를 함께 보고한다.
- 동일 native replay/distillation과 적절히 학습한 일반 connector가 효과를 설명하면 알려진 기법의 조합으로 해석한다. encoder 전체 freeze는 참고군이며 제안의 목표 달성이 아니다.

### E3. '오답을 줄인 교정'을 모으는 검수 루프

- 기존 예산의 20개 시간 측정 pilot 안에서 실제 수작업 분/교정 하나를 기록한다. crop/class/지역을 균형 있게 잡고 random, uncertainty, confusion-sensitive selection을 같은 시간 예산으로 비교한다.
- 자동 mask로 생성한 click/box correction은 **simulated annotator**로 명시한다. 사람 교정과 AI 후보, 정답 label의 출처를 각각 기록한다. 이런 합성 교정으로 훈련한 선택기를 다른 공개 라벨 지역에서 먼저 개발 검증한 뒤 사람 pilot에 적용할 수 있다.
- 독립 평가 사건·query 정답을 support 선택/utility 학습에 사용하지 않는다. utility cross-fitting은 parent 지역 단위로 수행하고 같은 utility를 generic router에도 제공한다.
- 결과: 교정 1개가 아니라 총 검수 분/전체 후보 library 비용에 따른 새 지역 품질. 정확도가 같은데 적게 검수하거나 같은 검수시간에 혼동 오류가 줄어야 의미가 있다. AI 합의율이나 조작된 confidence가 늘어나는 것은 성공이 아니다.

## 구조에 붙일 때의 최소 경계

`native EO + 날짜/센서/validity → OlmoEarth dense grid → 동일 dense head/continuous evidence store → [선택/검사 도구] → Qwen 답변`을 먼저 유지한다. 데이터 선택·이산 token·KV caching은 서로 다른 기능이다. inference cache는 같은 model/adapter/관측 버전일 때만 재사용하며, encoder 학습 중 stale embedding을 고정해서 encoder 개선을 주장하지 않는다. 학습 경로와 추론 경로를 구분한다.

우선순위는 E1과 E2의 최소 비교다. E3는 이미 계획된 사람 pilot에 붙인다. FAST식 이산화와 생성 세계모델은, 연속 특징/단순 selector가 실패하는 실제 근거가 확인되었을 때 별도 축으로 확장한다. '로봇에서 효과가 났으니 EO에서도 된다'가 아니라 같은 정보·연산·교정을 받은 강한 EO baseline을 이기는지를 묻는다.
