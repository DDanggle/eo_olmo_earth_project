# VLM을 본체로: EO 임베딩·메타정보·지역 기억의 갱신

2026-09-18. 사용자 추가 질문에 대한 연구 방향 노트. 새 실험·학습·prereg 변경 없음.

## 1. 결론과 초점 변경

가능하다. 사용자가 말한 ‘로봇 형태’는 우선 물리적 로봇을 새로 만들라는 뜻이 아니라, 로봇 VLM의 **관측→기억→추론→추가 정보 선택** 구조를 EO에 이식하는 것으로 해석한다.

이 설정에서는 latent 보간이 본체일 필요가 없다. 주 연구 대상은 **지역에 대한 답을 새 관측과 메타정보에 따라 갱신하는 memory-augmented EO VLM**이다. EO encoder는 지각 모듈이고, 임베딩은 기억의 재료이며, 언어 모델은 질문에 맞는 기억을 읽고 답·근거·보류를 출력한다.

앞선 보고서의 ‘T 검증 후 L 확장’은 한 실행 전략이었다. VLM 중심 연구가 기술적으로 불가능해서 필수였던 순서는 아니다. 사용자 의도를 반영하면 ‘VLM 시계열 추론/기억 갱신을 본체, 보간은 선택적 보조 과제’로 독립 연구를 설계할 수 있다.

다만 feasibility와 CVPR 신규성은 다르다. robotic memory를 EO에 적용했다는 사실만으로 충분하지 않으며, 이 노트가 새 방법의 성공이나 최초성을 입증하지 않는다.

## 2. 로봇 쪽에 실제로 가까운 구조가 있다

확인 수준: S는 관련 본문 일부 확인, M은 공식 초록/프로젝트/서지 확인이다. 전체 정독·독립 재현은 아니다.

| 참고 | 확인 | 가져올 요소와 구분 |
|---|---|---|
| [RAVEN: Long-Horizon Reasoning & Navigation with a Visuo-Spatio-Temporal Memory](https://arxiv.org/abs/2606.25206), 2026 preprint | S: §3–4 | visual embedding+위치+시각을 저장하고 VLM이 도구로 검색한다. 검색한 원영상도 VLM에 반환하므로 latent-only inference의 직접 증명은 아니다 |
| [MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](https://arxiv.org/abs/2508.19236), ICLR 2026 공식 기록 | S: §3 및 memory ablation | perceptual/cognitive tokens, 시간 조건부 retrieval, gate fusion, memory consolidation. EO에서는 실제 지리 좌표·불규칙 Δt·quality를 따로 설계해야 한다 |
| [ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation](https://arxiv.org/abs/2409.13682), 2024 공개 논문 기준 | S: memory building/querying, limitations | 위치/시각 검색과 장기 QA. caption 기반 기억은 baseline으로 유용하며 직접 EO embedding을 LLM에 정렬하는 방법과는 다르다 |
| [ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning](https://concept-graphs.github.io/), ICRA 2024 | M: 공식 프로젝트 | 공간 entity와 관계를 명시하는 memory 참고. EO patch/parcel identity와 동일하지 않으며 RGB-D/pose 계약을 그대로 복사하지 않는다 |
| [TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data](https://arxiv.org/abs/2410.06234), ICLR 2025 | M: 공식 초록/기존 조사 | EO 시계열→언어 정렬의 direct baseline. online memory와 답의 교정 성능은 별도 평가가 필요하다 |
| [EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](https://arxiv.org/abs/2412.15190), CVPR 2025 | M: 공식 논문/기존 조사 | EO multimodal/multitemporal instruction learning. 임베딩·센서·언어 결합 자체가 빈 분야는 아니다 |
| [Characterizing AlphaEarth Embedding Geometry for Agentic Environmental Reasoning](https://arxiv.org/abs/2604.18715), 2026 preprint | M: 초록 | embedding retrieval와 environmental tool agent가 이미 존재한다. ‘EO embedding+LLM+검색’만으로 최초성을 주장하지 않는다 |

MemoryVLA의 ICLR 2026 표기는 [공식 proceedings PDF 검색 기록](https://openreview.net/pdf?id=54U3XHf7qq)에서 확인했다. 직접 open에는 verification 화면이 나왔으므로 방법 읽기는 arXiv 본문을 사용했다. 최종 결과를 우리 EO 환경에서 재현한 것은 아니다.

중요한 차이: RAVEN의 semantic retrieval은 언어와 정렬된 multimodal embedding을 사용한다. OlmoEarth 벡터에 일반 text embedding의 cosine 검색을 그대로 적용해도 된다고 가정할 수 없다. query encoder와 EO retrieval space를 학습하거나, geo/time 필터·readout concept index를 이용한 기준선이 필요하다.

## 3. 로봇 구조를 EO로 옮기는 정확한 대응

| 로봇 | EO 연구에서의 대응 |
|---|---|
| 현재 카메라 관측 | 새 위성 관측의 공간 토큰 지도 |
| pose와 timestamp | CRS/좌표/footprint, acquisition·arrival 시각, 실제 Δt |
| 단기 working memory | 현재 질의·새 관측·관련 과거 토큰 |
| 장기 episodic memory | 지역별 시계열 임베딩과 관측 provenance |
| scene/entity memory | tile/parcel/region ID, 공간 관계 및 확인된 변화 기록 |
| 행동 제어 | 우선 답변/보류. 별도 확장에서는 과거 영상 조회·다른 센서 조회·검수 요청 |

EO에서 robot motor action을 그대로 학습할 이유는 없다. 첫 단계는 memory-VLM이다. 정보 선택 정책과 비용 대비 이득을 평가할 때에만 perception-action agent라고 확장한다. 위성 재방문을 자유롭게 명령할 수 있다는 가정도 하지 않는다.

## 4. 임베딩이 결과라는 것은 제약이지 불가능 조건이 아니다

벡터의 각 차원을 자연어 단어로 해석할 필요는 없다. 공간 embedding token에 좌표·시각·센서·품질 encoding을 붙여 learned projector 또는 cross-attention으로 언어 모델 입력 공간에 맞출 수 있다. pixel VLM에서도 encoder feature를 언어 모델에 연결한다는 점에서는 같은 원리다. 다만 OlmoEarth와 text의 정렬은 별도로 학습해야 한다.

후보 구조는 다음과 같다. 제안이며 구현·학습하지 않았다.

```text
새 EO 공간 임베딩 + 위치/시각/센서/quality
    → metadata-conditioned token adapter
    → 지역 memory의 관련 과거와 결합
    → memory update (원 관측과 근거 ID는 보존)

질문 + 갱신된 memory
    → spatial/temporal retrieval 또는 cross-attention
    → 언어 모델
    → 답 + 지역/기간 + 근거 관측 + 신뢰도/보류
```

메타정보의 역할을 분리한다.

- 조건: Δt, 계절, 위치, 센서, GSD, 관측 기하, 구름·유효 영역. token-level quality도 필요할 수 있다.
- 검색키: AOI/parcel ID, 좌표 범위, 기간, 관측 ID. 숫자를 모두 벡터 안에 숨길 필요는 없다.
- 외부 근거: 공공기록·정책 정의·토지 관리 정보. 영상에서 확인한 사실과 출처를 구분한다.

좌표가 같은데 crop grid가 다르면 memory가 같은 지점을 비교하도록 정합해야 한다. pooled vector 하나보다 공간 토큰/영역 토큰을 유지하는 후보가 유리할 수 있지만 실제 grounding 평가로 확인한다. 메타정보는 encoder가 잃은 세부 정보를 자동 복구하지 않는다.

‘메타정보 극대화’는 양이 아니라 **필요한 조건을 신뢰도와 이용 가능 시각에 맞게 사용하는 능력**이다. 특정 지역명이나 공공기록만 보고 답을 찍는 shortcut을 막기 위해 metadata-only 학습 기준선과 held-out 지역 평가가 필요하다.

## 5. 업데이트의 본체: 기억 추가뿐 아니라 근거에 맞는 판단 수정

예시 시나리오(실측 결과가 아님):

1. 정상 과거 관측: 산림 상태가 안정적이라고 답한다.
2. 일부 구름이 있는 새 관측: 변화 가능성은 있지만 확인 불가라고 답한다.
3. 이후 clear 관측: 관측 가능한 산림 감소의 위치와 기간 구간을 확인하고 이전 답의 불확실성을 해소한다.

이때 올바른 업데이트란 무조건 새 답으로 바꾸는 것이 아니다. 무관하거나 품질이 나쁜 영상에는 안정적이어야 하고, 유효한 반대 증거에는 바뀌어야 한다. 과거 날짜의 사실도 최신 상태로 덮어쓰면 안 된다. 늦게 도착한 영상은 acquisition 시각과 arrival 시각을 구분해 처리한다.

memory에는 두 층을 둘 수 있다.

- 시각적 기억: EO tokens와 geo/time/quality, 원 관측 링크.
- 판단 기록: 확인된 변화·미확인 가설·신뢰도 및 근거 ID.

LLM이 생성한 문장을 확인된 관측처럼 다시 저장하면 자기 환각을 강화할 수 있다. 원 관측은 보존하고 판단 기록을 versioning한다. ‘EO에서는 이런 dual memory가 아직 없다’는 주장은 추가 문헌 검토 없이 하지 않는다.

추천 주 질문:

> 제한된 기억·연산 예산에서 EO VLM이 불규칙한 새 관측을 이용해 지역에 대한 공간·시간적 답을 정확히 수정하면서, 근거 없는 수정과 과거 기억의 손실을 줄일 수 있는가?

quality-aware update, bounded memory, spatial grounding은 각각 기존 분야의 개념이다. 이 조합과 EO-specific failure를 다루는 메커니즘이 실질적으로 기여하는지 입증해야 한다. robotics checkpoint의 단순 encoder 교체를 신규 방법이라 주장하지 않는다.

## 6. 어떤 학습을 할 것인가

관측 때마다 거대 VLM 가중치를 다시 학습할 필요는 없다. 먼저 offline 학습으로 memory updater·alignment를 익히고 test stream에서는 memory/state만 갱신한다.

| 단계 | 학습 대상 | 데이터/목표 |
|---|---|---|
| A: EO–language alignment | projector/query adapter, 필요하면 LLM LoRA | 관측 가능한 속성·변화·지역·기간을 EO tokens와 연결. trained metadata-only와 비교 |
| B: sequential memory | updater/retriever, 공간·시간 adapter | 과거 prefix→새 관측→현재/과거 질문. 답·근거 선택·변화 location·uncertainty 평가 |
| C: 정보 선택(선택적) | retrieval/tool policy | 같은 비용 예산에서 필요한 과거 관측/센서를 조회. action GT가 없으면 oracle/SFT/RL 가정을 명시 |

신규 데이터는 ‘정답 문장’만 합성해도 충분하지 않다. 실제 지역/기간의 관측과 독립 라벨에서 event type·location·interval·evidence ID를 먼저 정한 뒤 질문 표현을 만들고 검수한다. LLM으로 답도 만들고 같은 LLM으로 평가하는 순환은 피한다.

첫 구현은 A+B 중 최소 모듈과 하나의 변화 현상에 한정한다. ‘환경파괴’는 관측 변화와 가치 해석을 나누고, 법적/정책적 판단은 독립 출처와 coverage caveat를 남긴다. 그날 정확히 발생했다는 라벨이 없으면 onset interval로 평가한다.

기존 L1은 A의 작은 pilot일 뿐이다. 로컬 코드에선 frozen LLM에 projector만 3 epoch 학습했고, 재귀적인 memory update가 없다. 좌표/quality 조건부 update나 learned retrieval도 없다. Q2는 학습 문항에서 제외되어 sequential temporal reasoning 실험도 아니다. 그 실패 판정은 보존하되 이 방향 전체의 실패라고 확대하지 않는다.

## 7. 한 가지 핵심 실험으로 시작한다

공간·시간 정답이 있는 하나의 변화 유형을 선택하고, 한 지역에서 시간 순서대로 관측을 제공한다. 지역 적응에는 과거 train만 사용한다. 새 관측 도착 전후 동일 질문을 평가하되, 매 시점의 gold는 해당 prefix에서 판단 가능한 정보로 정의한다. 뒤의 영상을 못 본 상황에서 확정 답을 강요하지 않는다.

질문은 예를 들어 ‘이 영역에 변화가 확인되는가?’, ‘어느 영역인가?’, ‘언제부터 관측으로 확인되는가?’, ‘이전 의심을 유지/철회할 근거가 있는가?’로 잡는다.

기준선:

- 최신 관측만 보는 EO projector-VLM.
- 같은 학습 예산의 metadata-only 모델.
- readout+structured record+LLM 또는 template.
- append-only visual memory의 geo/time 검색 + 원 칩 VLM(RAVEN-style).
- 단순 FIFO/mean/GRU memory, 고정 top-K retrieval.
- 가능한 context 한도에서 전체 prefix를 읽는 temporal VLM reference.
- 제안한 metadata-conditioned memory updater.

측정:

- 현재 및 과거 질문 정확도, 위치/시간 grounding, 근거 관측 ID precision/recall.
- 유효한 새 증거에서 올바르게 수정하는 비율.
- 무관한/저품질 입력에서 불필요하게 답을 바꾸는 비율.
- uncertainty/abstention, 과거 질문의 망각, 관측되지 않은 사실의 환각.
- 동일 memory/token/raw-I/O 예산의 accuracy–cost와 batch-1 latency.

날짜·위치·품질 metadata를 제거/변형하고, history를 바꾸되 최신 관측을 같게 하는 대조로 무엇을 읽는지 확인한다. EO tokens를 교환하는 intervention도 의미에 맞게 설계한다. coordinate/time invariance를 잘못 강제해 질문 자체의 정답을 바꾸지 않는다.

chrono/event/spatial split, 동일 질문 ID, paired 분석, 독립 검수, 미사용 미래/외부 지역 평가가 필요하다. 이 실험에서 simple trained memory와 metadata-only를 못 넘으면 새로운 updater 주장은 약하다. 외부 지역 적응 범위를 숨기지 않는다.

## 8. 이번 턴의 결정 범위

VLM 중심 연구는 가능하며 사용자 요청에 맞는 별도 본체로 설계했다. 보간/PDE는 선행 필수 단계가 아니다. 로봇 구조의 재사용 가능성을 보여주는 1차 자료가 있지만 EO에서의 성능·새 방법 신규성은 미검증이다.

새 학습 실행, 원격 서버 접속, 현재 queue 변경, 기존 T4/L1 gate 변경, 새 라벨 개봉, commit/push는 하지 않았다. 다음 실제 실행에는 변화 유형·독립 gold·prefix annotation·학습 budget·미사용 split의 새로운 계약이 필요하다.
