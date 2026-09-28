# OE2 실행 순서: 첫 VLM과 TerraScope 이후의 연구 질문

> **9/27 후속:** [실제 문제 감사·수정](OE2_PROBLEM_AUDIT_AND_FIXES_20260927.md)에서 질문 prior·채점·공간 계약을 검증하고20장CPU 표현 진단을 완료했다. 실행 순서는v2로 갱신했다. 아래는9/26 당시의 설계 기록이다.

2026-09-26. **현재 검증된 우월성은 없다.** 이 문서는 [OE2 v0](OE2_MULTIMODEL_SUPERVISION_AND_REPRESENTATION_PLAN_20260926.md)의 다음 결정을 구체화한 후속 설계다. 모델 후보 조사, 서버 읽기 확인, train 20개 후보 메타데이터 준비까지 수행했다. 새 모델 추론·학습·API 결제·공개 업로드는 수행하지 않았다. 기존 [진단 논문 골격](PAPER_EO_VLM_SKELETON_20260926.md)과 E5/D1 결과를 변경하지 않는다.

## 1. TerraScope와 무엇이 겹치고 무엇을 더 시험할 것인가

| 비교 축 | TerraScope에서 이미 다룬 것 | 우리 연구의 추가 검증 목표 |
|---|---|---|
| 공간 근거 | 마스크를 생성하고 해당 시각 특징을 다시 언어 생성에 사용 | OlmoEarth의 분광·공간 표현에서 근거를 회수할 수 있는지 |
| 측정·변화 | 면적·거리·경계·두 시점 변화; 답과 mask 평가 | 같은 원 관측에서 수치 오차와 잘못된 근거가 실제로 줄어드는지 |
| 입력 | optical RGB와 SAR 지원. 긴 연속 시계열은 향후 범위 | native 다중분광 입력. 추가 밴드의 이점과 학습법의 이점을 분리 |
| 학습 | InternVL3-8B 기반, vision encoder는 고정; projector·mask decoder·LLM LoRA 학습 | OlmoEarth encoder에 남는 변화, 새로운 연결부·reader로 옮긴 뒤의 성능 |

근거: [TerraScope §3.2·§5·Appendix A/J.2](https://arxiv.org/html/2603.19039v1). 위 오른쪽은 달성 결과가 아니라 검증할 목표다. '위성영상에 언어 연결', '근거 mask', '면적 설명', '다중 AI 라벨링', '시간 정보를 사용'은 단독 신규성이 아니다.

더 가까운 선행도 함께 비교한다.

- **[SPEX v2](https://arxiv.org/html/2508.05202v2):** 다중분광 encoder를 사전학습하고 언어·픽셀 감독으로 갱신한다. 따라서 'EO encoder도 학습한다'만으로 구별되지 않는다. SPEX의 TCP는 LLM 출력 description 특징을 압축해 mask decoder에 전달한다. 우리의 후보인 LLM 입력 전 EO 압축과 위치는 다르지만, 위치가 다르다는 사실만으로 방법 기여가 되지는 않는다. §III-B와 §IV-C를 비교한다.
- **[MS-CLIP](https://arxiv.org/abs/2503.15969):** 다중분광–언어 지속 사전학습의 선행. **[Spectral-LLaVA](https://arxiv.org/abs/2501.10144):** EO foundation encoder와 생성 LLM 연결의 선행. 넓은 최초 주장을 배제한다.
- **[Lightweight Adaptation, 2026-09](https://arxiv.org/abs/2609.02187):** 밴드·지수·SAR를 영상 뷰로 만들어 일반 VLM을 적응하고 입력 통제와 외부 전이를 시험했다. 단순 RGB 기준선만 이기는 결과로 OlmoEarth 경로의 필요성을 주장할 수 없다.

이번에 좁힌 연구 질문은 다음이다.

> 같은 관측·감독·학습 예산에서, OlmoEarth의 공간·분광 정보를 제한된 토큰으로 읽도록 학습하면, 일반적인 언어 정렬보다 측정·근거가 더 정확해지는가? 학습한 encoder를 다른 reader에 옮겨도 그 효과가 남고, 기존 EO 과제 성능을 유지하는가?

이 질문에는 연구 가치가 있지만, 실험 통제를 잘 갖췄다는 사실과 새 알고리즘을 제안했다는 사실은 다르다. **구체적인 보존 학습법은 아직 완성되지 않았다.** 토큰 압축을 거치며 어떤 목표 정보가 손실되는지 먼저 측정하고, 그 손실을 줄이는 설계를 일반 auxiliary head·기존 연결부와 비교해야 한다. 그 차이가 없으면 모듈을 추가하는 대신 주장을 축소한다.

## 2. 첫 모델은 이미 있는 Qwen3-VL-8B부터

이번 서버 읽기 확인에서 H200 두 장이 보였고 각 GPU의 사용 메모리는 0 MiB였다. 이는 조회 시점의 값이며 실행 직전에 재확인한다. 기존 `models/Qwen3-VL-8B-Instruct`의 config, 가중치 index와 4개 safetensors 파일의 존재를 확인했다. **파일 존재 확인이며 완전한 checksum·로드·backward 검증은 아니다.** Molmo2-O-7B는 config 존재만 이번에 확인했다. 기존 래퍼에서 SSH host-key 변경 경고도 출력됐으며 이번에 신뢰 설정을 변경하지 않았다.

| 순서 | 선택 | 목적 |
|---|---|---|
| 첫 연결 | **Qwen/Qwen3-VL-8B-Instruct** | 기존 서버 가중치를 우선 재사용. 실제 VLM의 영상 경로를 유지 |
| 메모리/구현 대안 | Qwen/Qwen3-VL-4B-Instruct | 8B의 실제 backward 프로파일이 맞지 않을 때. 처음부터 둘 다 학습하지 않음 |
| 주요 외부 비교 | OpenGVLab/InternVL3-8B, 공개 TerraScope | 같은 backbone의 대조와 실제 공개 시스템 비교 |
| 별도 reader 전이 | InternVL3-8B 우선; Molmo2-O-7B 후보 | 기존 연결부를 버린 뒤 EO 표현을 새 reader에서 재사용 |

[Qwen3-VL-8B 공식 카드](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B). 공식 revision과 소스는 [후보 장부](../config/oe2_vlm_candidates_20260926.json)에 기록했다. 서버의 기존 파일과 공식 revision 일치는 아직 검사하지 않았다.

InternVL3의 언어 모델은 Qwen2.5이므로 Qwen3-VL→InternVL3를 완전히 독립된 언어 계열 전이라고 부르지 않는다. 우선 다른 VLM 구조로의 전이다. Molmo 계열 전이는 더 넓은 검증 후보지만 새 EO adapter를 따로 구현해야 하며, 과거 inference loader가 있다는 것만으로 학습 호환이 보장되지 않는다.

공개 [TerraScope 구현](https://github.com/shuyansy/Earth-Observation-VLMs/tree/master/TerraScope)과 [가중치](https://huggingface.co/sy1998/TerraScope/tree/main)는 실제 공개되어 있다. 논문 3,837문항과 현재 [공개 benchmark](https://huggingface.co/datasets/sy1998/TerraScope-Bench)의 viewer 3,549문항이 달라 재현 때 revision·task population을 먼저 고정한다. 두 수량의 원인을 확정하지 않는다. 공개 모델을 우리 환경에서 실행하거나 논문 점수를 재현한 상태는 아니다.

## 3. 연결할 위치와 최초 통과 조건

Qwen3-VL은 RGB 영상의 여러 층 특징을 언어 모델에 넣는 DeepStack 경로가 있다. 이를 OlmoEarth 벡터로 단순 치환하지 않는다. 기본 RGB 경로와 영상 position 처리는 유지하고, 별도 EO 슬롯에 OlmoEarth 공간 특징을 투영하는 초기 구현을 검토한다. 이는 **검증할 구현안이며 지금 동작하는 adapter가 아니다.** [공식 Transformers 구현](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)

```text
동일한 실제 Sentinel-2 관측
 ├─ 고정 규격 RGB → Qwen3-VL의 기존 영상 경로 ───────────┐
 └─ 센서 원 밴드 → OlmoEarth → 공간 연결부 → EO 슬롯 ──┤
                                                     ↓
                           질문 → 답·수치·근거 영역/관측 참조
```

EO 입력 계약은 `(batch, time, bands, height, width)`, 실제 취득일, 밴드 순서, 유효 관측, 좌표·footprint다. Tensor layout은 기존 EO encoder에 맞게 변환하되 값의 의미는 보존한다. 구조화 출력은 task에 맞는 상태·수치·단위·분모·근거 참조·판단 유보다. 정답 mask/teacher 답/현장 보고서는 일반 영상 판독 입력에서 제외한다.

처음에는 기존 OlmoEarth Tiny로 입출력만 점검할 수 있다. 연구 본 비교에서는 같은 OlmoEarth Base 초기 checkpoint를 고정하는 후보를 유지한다. Tiny→Base, VLM 4B→8B, 데이터 10k→100k를 동시에 바꾸고 한 요인의 개선이라고 해석하지 않는다.

첫 구현 검사:

1. EO 슬롯을 만들지 않은 경로는 같은 입력에서 원 Qwen 모델의 logits와 수치 허용오차 내 일치해야 한다. 슬롯을 남기고 0으로 만든 조건은 별도 대조다.
2. native RGB placeholder와 DeepStack의 feature/mask 개수는 유지한다. EO 슬롯의 attention·position·teacher-forcing label·생성 cache를 학습과 추론에서 일치시킨다.
3. connector-only 단계는 EO 텐서 불변, joint 단계는 의도한 EO 층의 비영 gradient와 가중치 변화가 확인되어야 한다. frozen VLM도 입력으로 gradient를 보내야 하므로 forward 전체를 `no_grad`로 감싸면 안 된다.
4. 저장–복원, 단일/배치, padding 경로가 같은 문항에서 일치해야 한다. 작은 train 과적합은 연결 검사이며 일반화 성능이 아니다.
5. microbatch1의 실제 forward/backward/optimizer step로 peak memory·처리량을 측정한다. 가중치 파일 크기를 학습 메모리로 환산하지 않는다.

## 4. 최초 20개는 연결 검사, 공간 측정 정답은 별도 확보

이번에 [20개 후보 manifest](../artifacts/oe2_startup_v0_20260926/candidates20_manifest.json)를 만들었다. 기존 OE1 official-train512개에서 고정 hash 순위와 MGRS당1개 규칙으로 선정했다. **20개 지역 격자·40개 원문 질문**이며 모델 정답률을 보고 고르지 않았다. 원 metadata·질문·정답을 그대로 보존하고 출처와 파일/array hash를 기록했다. [재생성 코드](../code/oe2_build_candidates20_v0.py)를 함께 둔다.

현재 이 20개는 메타데이터 후보이고 로컬 raw 영상은0개다. 서버 receipt 경로는 있지만 이번에 원영상20개를 내려받아 검증하지 않았다. `ready_for_inference=false`다. 과거 OE1 train에 쓰인 표본이므로 새로운 평가셋이 아니다.

라벨은 장면의 클래스 존재 여부다. **이 자료만으로 면적·위치·변화·고사목 정답을 만들지 않는다.** 새 공간 평가에는 별도의 원 관측–dense/partial mask 정합 검증이 필요하다. 원 TIFF 내부60m transform과 외부10m grid의 기존 불일치도 유지 기록하고, 장면 preview에서 그럴듯해 보인다는 이유로 면적 정답을 계산하지 않는다.

## 5. 먼저 비교할 것: 입력 효과와 encoder 학습 효과

첫 개발 비교는 같은 연결부를 사용한다. 이전 v0의 연결부×목적 2×2를 즉시 전부 시작하기 전에 학습 목적의 필요성을 먼저 본다.

| 조건 | OlmoEarth 쪽 처리 | 판단 용도 |
|---|---|---|
| F | 원 pretrained encoder 고정, 연결부/reader 적응 | EO 표현의 출발 성능 |
| R | 기존 EO 목적의 추가학습, 이후 같은 reader 적응 | 단순 추가 EO 학습 효과 |
| L | 일반 언어·공간 감독으로 encoder 공동학습 | 일반적인 강한 공동학습 기준선 |
| P | 같은 감독으로 제안 보존 학습 | 주대조는 **P−L** |

L/P에는 **같은 공간·수치 라벨, 유효 영역, EO replay, 학습 가능한 parameter 예산, 연결부와 VLM 초기값**을 제공한다. L도 실제 auxiliary 공간/측정 loss를 사용한다. P만 정밀 라벨·dense mask를 받게 하지 않는다. R/F는 목표가 다른 참조 조건이므로 이들의 차이를 보존 loss의 인과 효과로 해석하지 않는다. 학습 단계가 다른 R의 총 관측 노출·GPU시간을 포함해 계산 한도를 맞춘다. 방법과 loss 수식은 미정이며 본 학습 전에 고정한다.

별도로 (a) RGB-only 실제 VLM, (b) 동일 원 관측의 여러 분광 뷰를 렌더링해 적응한 VLM, (c) EO segmentation/측정→계산기→같은 LLM을 비교한다. 이들은 입력·연산·구조 차이가 있어 주대조 P−L을 대신하지 않는다. 분광 뷰 대조는 NIR/SWIR 정보를 쉽게 보여주는 것만으로 해결되는지 확인하는 필수 강한 기준선이다. 모든 방법에 학습 자료·질의 분포를 공유하고 native-band vs rendered 정보 차이, 토큰·해상도·계산비를 보고한다.

## 6. 우리가 더 의미 있게 만들었다고 말할 수 있는 결과

**첫째, 같은 입력에서 측정 정보가 더 잘 읽혀야 한다.** 원 EO 특징과 압축 후 EO 토큰에는 같은 용량·학습 예산의 새 readout을 붙여 영역·수치·분광 구분의 회수 가능성을 본다. 최종 VLM 답은 별도 head로 교정하지 않고 원 응답의 수치·영역·근거를 직접 채점한다. 첫 P1은 EO 토큰64개를 개발 출발값으로 사용하고 일반 resampler와 비교하며 RGB 토큰 수를 고정한다. 처리량 문제로 바꾸면 모든 군에 공통 적용하고 본 학습 전에 고정한다. 16/64/256 전체 곡선은 별도 확장 실행표로 분리한다. 각 token budget에 맞춰 학습한 비교를 주로 쓰며, 추론 때만 token을 잘라 생긴 손실을 학습법 차이라고 하지 않는다. 여기서 측정하는 것은 지정한 목표의 회수 가능성이며 총정보량이 아니다.

**둘째, encoder를 새 reader에 옮겨도 이점이 남아야 한다.** F/R/L/P의 encoder를 고정하고 기존 연결부·VLM 적응 가중치를 버린다. 새 지역의 동일 적응 자료/예산으로 새 연결부를 붙인다. Qwen 기반 본 결과와 InternVL3 전이 결과를 구분해 제시한다. 모든 encoder에 같은 readout 용량·튜닝 기회를 주고 test를 적응에 사용하지 않는다.

**셋째, 언어 답변이 좋아지는 동안 기존 EO 능력을 잃지 않아야 한다.** 같은 초기 EO의 공간 분할·분광 구분 등 독립 과제를 기존 방식으로 재평가한다. label과 caption이 같은 원천에서 만들어진 과제만으로 '기존 능력 전체 보존'을 주장하지 않는다. 비열등 허용 한계와 최소 의미 효과를 dev에서 정한 뒤 test 전에 고정한다.

세 결과와 실제로 새로운 학습 설계가 함께 있어야 표현 학습 논문으로 설득력이 생긴다. 지역 검색은 이 결과를 보여주는 응용으로 두고, 고정 후보군의 조건부 검색·근거 확인 시간을 평가한다. 면적 추출이 나아졌다는 것만으로 검색 효율이나 생태적 원인 이해를 자동 주장하지 않는다.

## 7. 확장 순서와 제출 주장

1. **P0 /20개 연결 검사:** 원자료 checksum, RGB 렌더링, 실제 VLM, 별도 EO 슬롯, backward와 저장–복원. 이 단계는 유의미한 연구 성능을 판정하지 않는다.
2. **P1 /공간 정답 개발 집합:** 실제 날짜·geometry·mask 지원 범위를 감사하고 F/R/L/P의 목표 정보를 비교. 작은 pilot에서 압축 손실·일반 정렬의 손실이 없으면 보존 모듈의 필요성을 다시 판단한다. test는 열지 않는다.
3. **P2 /10k 학습:** 하나의 고정된 VLM 크기에서 P−L과 강한 분광 뷰/측정 파이프라인을 비교. 개발에서 정한 recipe와 실패 기준을 고정한다.
4. **P3 /100k·전이:** 같은 원천 혼합·유효 감독의 nested corpus, 여러 seed, 두 번째 VLM 구조와 새 지역·과제, 독립 사람 test. 여기서 v0의 2×2를 활용해 연결부의 추가 효과를 분해한다.
5. **P4 /선택 확장:** 1M 또는 긴 시계열. 앞 단계 결과·데이터·GPU 비용이 뒷받침할 때 선택한다. 긴 시계열·기후 원인·고사 원인을 현재 주장의 필수 항목으로 추가하지 않는다.

v0의24/30회 계획은 가능한 전체 비교 비용이며 지금 실행할 queue가 아니다. 새로운 F/R 조건을 전부 더해 run 수를 숨기지 않는다. P1 뒤 확정 실행표에서 기존 frozen 대조와 겹치는 실행을 합치고, R/readout/렌더링 비교의 추가 비용을 별도로 기록한다. 유리한 seed만 키우지 않는다. 사람·API100만 원 예산은 유지하며 GPU/개발 연산비와 구별한다.

TerraScope를 대상으로는 두 표가 필요하다. **공개 checkpoint 외부 비교**는 실제 사용 성능이지만 학습 자료와 backbone 차이를 포함한다. **같은 InternVL3·관측·감독의 방법 비교**는 방법의 이점을 시험한다. RGB vs native multispectral 비교는 추가 정보의 효과가 섞인다. 우리 점수가 더 높다는 한 줄로 세 차이를 합쳐 설명하지 않는다.

CVPR 방법 논문을 겨냥한 목표는 재사용 가능한 EO 표현 학습법과 그 근거다. 지금 완료한 것은 차별점 검토·실행 순서·20개 준비 대상·모델 가용성 확인이며, TerraScope/SPEX를 능가한 방법이나 결과는 아니다. 다음 실행은 P0이며 이후 주 실험의 모델 revision·loss·데이터·허용오차·예산을 [실행 순서 config](../config/oe2_first_vlm_route_v1_20260926.json)에서 별도로 동결한다.
