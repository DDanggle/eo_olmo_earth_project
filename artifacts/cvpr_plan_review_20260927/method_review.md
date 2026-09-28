# 독립 방법·실험 검토 — 2026-09-27

검토 대상: OlmoEarth의 native 공간·분광 표현과 실제 VLM을 연결하고, 소량의 support로 새 개념을 가르친 뒤 학습하지 않은 개념×관계×작업 조합으로 전이시키는 연구.

판정: **제한된 반증 실험에는 conditional go. 현 상태에서 CVPR 중심 기여가 확보되었다고 판단할 근거는 없다.** 20장 probe는 동작 확인이며 병목의 원인이나 표현 개선을 증명하지 않는다. 채택 확률은 산정하지 않는다. 이 문서는 새로운 실행 결과를 읽거나 실험을 수행한 보고서가 아니다.

## 1. A와 B 중 무엇을 중심에 둘 것인가

**A: support 개념 결합과 미학습 작업 조합 전이를 주 연구로 선택한다. B: 면적을 보존하는 압축은 필요한 경우에만 구현 부품으로 둔다.** 현재 사용자의 목적은 제한 토큰 압축 자체보다 새 EO 개념을 적은 예시로 배워 실제 작업에 재사용하는 데 있다.

A의 가장 큰 약점은 prototype/meta-learning, shared segmentation field, compositional supervision을 결합했다고 곧 새로워지지 않는다는 것이다. 의도적으로 concept×operation 표의 일부를 비우는 것 역시 일반적인 compositional evaluation 설계다. 이 구성만으로 논문 방법의 신규성을 주장해서는 안 된다.

B는 예측 면적의 손실 없는 전달을 만들 수 있지만, 면적 합을 스칼라로 넘기면 해결되는 부분이다. 수치 보존 또는 더 많은 토큰이 주된 원인이면 EO 표현 학습이라는 중심 주장은 성립하지 않는다. B를 독립적인 대형 CVPR 기여로 보는 것은 현재 근거로는 부적절하다.

**권할 중심 주장:** native EO를 사용하는 episodic 개념 결합 학습이 동일한 관측·support·학습량을 쓰는 기존 연결법보다 새로운 개념을 관계 조건에 맞게 검색하고 위치를 찾는 능력을 높이는가. 면적·설명은 같은 근거 표현에서 파생되는 응용 결과로 둔다. 네 작업의 점수를 네 개의 독립적인 일반화 증거처럼 세지 않는다.

## 2. 하나의 구체적인 메커니즘 후보

작업마다 별도 head를 두는 대신 **하나의 support-conditioned spatial field를 모든 작업이 의무적으로 참조**하도록 한다. 핵심 설계는 동일한 질문 배경에서 support와 공간 참조를 바꿨을 때 그 field가 올바르게 달라져야 한다는 학습이다. 다음은 검증할 제안이며 기존 방법 대비 신규성이 확인된 이름 붙은 방법은 아니다.

### 2.1 공통 spatial field

관측 X에는 원본 센서 밴드, 날짜, 유효 관측 마스크, 공간 좌표를 포함한다. 기존 RGB VLM 경로는 유지한다.

\[
z_i=E_\theta(X)_i,\qquad
c=H_\phi\left(\{E_\theta(X_s),y_s\}_{s=1}^{K}\right),\qquad
p_i(c\mid X,S)=\sigma\bigl(q_\phi(z_i,c)\bigr).
\]

S는 K개의 긍정·대조 support 영역이다. H는 mask/point로 지정한 영역을 읽는 작은 결합 모듈이고 E는 실제로 업데이트되는 OlmoEarth이다. 개념 이름은 primary evaluation에서 무작위 별칭으로 바꾸어 이미 알고 있는 단어의 뜻과 support 학습을 분리한다. 자연어 개념명은 별도 응용 평가에서 사용한다.

p_i를 확률로 학습했다면 이를 그대로 실제 면적 비율이라고 부르지 않는다. fractional cover 정답을 사용하고 보정한 경우에만 면적 합을 cover 추정으로 해석한다. 이 구분 없이 sigmoid 값의 합을 헥타르 단위 정답으로 주장하면 안 된다.

### 2.2 작업별 특징 추출을 금지하고 같은 field에 연산을 적용

관계 o와 anchor 영역 R로부터 공간 선택 가중치 w_i(o,R)를 만든다. 예를 들어 지정 거리 내·북쪽·영역 안 등의 연산은 실제 좌표계에서 계산한다. 점수의 단순 예는 다음과 같다.

\[
s(S,X,o,R)=
\frac{\sum_i a_i v_i\,w_i(o,R)\,p_i(c\mid X,S)}
{\sum_i a_i v_i\,w_i(o,R)+\epsilon}.
\]

a_i는 픽셀 면적, v_i는 관측 유효성이다. 검색은 후보 장면/영역의 동일 점수로, grounding은 p와 관계 마스크의 교집합으로 수행한다. VLM은 이미지·EO evidence token·질문을 읽고 개념과 연산을 연결하며 근거 영역 ID를 답한다. 면적과 문장은 동일 field의 별도 후처리 결과다. 언어모델이 만든 자유형 숫자를 measurement의 유일한 채점 대상으로 삼지 않는다.

공간 연산자를 고정하는 것은 메커니즘의 명료성을 위한 선택이다. 이 경우 새로운 공간 추론 알고리즘을 발명했다거나 VLM이 모든 기하학을 학습했다고 주장할 수 없다. 자연어를 연산자로 파싱하는 성능과 EO field의 성능은 분리하여 측정한다.

### 2.3 개념·위치가 실제로 결합되도록 episodic evidence contrast를 학습

각 학습 episode에서 올바른 tuple (S,X,o,R+)와 검증된 음성 tuple을 만든다. 음성은 비슷한 배경의 잘못된 support 개념, 같은 장면의 다른 anchor/영역, 조건이 충족되지 않는 후보 장면을 포함한다. 확실하지 않은 후보를 음성으로 강제하지 않는다.

\[
\mathcal L_{\mathrm{bind}} =
-\mathbb E_e\log
\frac{\exp(s(e^+)/\tau)}
{\sum_{e'\in\{e^+\}\cup\mathcal N_e}\exp(s(e')/\tau)}.
\]

이미 확보한 segmentation/point supervision으로 p를 식별 가능하게 학습하고, 위 episode를 통해 작업마다 다른 latent가 아니라 같은 p가 바뀌도록 한다. 원래 EO 학습 목적을 유지할지와 가중치는 개발 분할에서 고정하고, baseline에도 동일한 보존 학습 예산을 준다. 손실 항 개수 자체는 기여가 아니다.

이 메커니즘의 시험 가능한 차이는 **support의 뜻과 참조 영역을 바꾸는 통제에서 동일 field와 답이 함께 바뀌는가**, 그리고 이를 학습하지 않은 개념×관계 조합에서도 유지하는가다. 단순 class-name decoding, image-level pooling, 여러 head를 독립적으로 붙이는 방식과의 차이를 여기서 보여야 한다.

### 2.4 선택적인 B 부품: 보존되는 근거 토큰

토큰 압축이 실제 병목으로 확인된 경우에만 공간 할당 A_ij≥0, sum_j A_ij=1을 사용한다. 각 token의 관측 면적과 예측 cover mass를

\[
b_j=\sum_i a_i v_i A_{ij},\qquad
m_j=\sum_i a_i v_i A_{ij}p_i
\]

로 정의하면 sum_j m_j는 전체 관측 영역의 **예측값**을 보존한다. 정확한 실제 면적을 보장하지는 않는다. 토큰에는 공간 범위·시점·b_j·m_j·내용 특징을 함께 전달할 수 있다. 같은 숫자와 mask를 받는 단순 VLM baseline보다 이득이 없으면 B는 안전한 집계 구현일 뿐 표현 학습의 증거가 아니다.

## 3. 가장 강한 반증 baseline

다음 비교를 이기지 못하면 중심 기여를 낮춰야 한다.

1. **OlmoEarth + prototype/segmentation specialist + 정해진 연산 + VLM 설명.** 동일 support에서 얻은 mask를 검색·관계·면적으로 변환하고 동일 모델이 설명한다. 이 baseline은 mask+area+template의 더 강한 버전이다. 제안법이 이를 넘지 못하면 '여러 작업 전이'는 field의 당연한 재사용일 수 있다.
2. **동일 OlmoEarth·동일 실제 VLM + 표준 projector/concatenation.** 원본 밴드, 날짜, 지역, support, supervision, trainable parameter 범위, 시각 토큰 예산, 학습 steps를 맞춘다. baseline만 EO를 동결하고 제안법만 업데이트하면 불공정하다.
3. **동일 architecture와 같은 데이터에 ordinary multitask supervision만 적용.** episodic support/anchor contrast가 무엇을 바꾸는지 확인한다. 제안법만 hard negative를 더 많이 봤다면 데이터 이득과 목적함수 이득을 구분해야 한다.
4. **동일 p·mask·측정 숫자를 VLM에 제공하는 explicit-evidence baseline.** 언어 개선이 더 좋은 EO 표현 때문인지 정확한 답을 숫자로 제공했기 때문인지 분리한다. oracle mask는 진단 상한으로만 표시한다.
5. **실제 RGB VLM + 충분한 동일 센서 렌더링**은 응용상 비교로 유용하다. 그러나 12밴드 branch와 RGB-only를 비교한 이득을 동일 입력에서의 학습법 이득으로 주장하지 않는다. primary same-evidence 비교는 1~3이다.

최소한 frozen/joint encoder를 제안법과 표준 projector 양쪽에 적용하는 2×2 비교가 필요하다. EO 표현 학습을 주장하려면 joint 업데이트가 실제로 발생하고 support label efficiency나 field 전이에 기여해야 한다. VLM만 큰 모델로 바꾸거나 설명 문장이 좋아진 것은 해당 주장에 충분하지 않다.

## 4. 데이터 분할과 '미학습'의 정확한 의미

- base concepts, development concepts, final novel concepts를 먼저 고정한다. 기존 세부 class를 이름만 바꾼 경우 novel semantic concept라고 과장하지 않는다.
- 최종 개념의 K support와 query는 서로 다른 지리적 군집에서 뽑는다. 같은 원본 tile의 다른 crop, 중첩 tile, 같은 개체의 다른 날짜가 support/query 또는 train/test에 나뉘지 않게 원본 scene·AOI·event 단위로 묶는다.
- 같은 광역 AOI의 인접 지역은 pixel-level split로 충분하지 않다. buffer와 공간 군집을 고정하고, 공간·시간 holdout을 함께 선언한다. 날짜만 다르면 독립 표본이라는 가정도 피한다.
- 학습에서는 기본 연산자 각각을 가르칠 수 있지만 일부 concept×relation×operation 조합을 의도적으로 비운다. 이때 주장은 **알려진 연산자의 미학습 조합**이지 완전히 처음 보는 작업을 배우지 않고 수행했다는 뜻이 아니다.
- 새 개념의 query mask나 query에서 파생한 area/caption/ranking을 training tuple 생성·teacher prompt·hard-negative mining에 사용하면 누출이다. support annotation으로만 novel concept를 결합한다.
- 검색 gallery의 선택 자체가 정답을 드러내지 않도록 면적·계절·지역 배경이 유사한 검증된 distractor를 둔다. 항상 화면 중앙에 목표를 두거나 support에만 표식/범례를 넣는 시각 단서도 통제한다.
- test support의 K 증가에 따라 support가 중첩되는 nested design을 쓴다. 모든 방법에 같은 K개와 동일 annotation 종류·시간 예산을 준다. support masks와 query labels를 혼동하지 않는다.
- 최종 novel concepts와 split을 보고 유리한 개념만 골라 남기지 않는다. 저해상도에서 식별 불가능한 개념은 사전 판독 기준으로 제외하거나 observability failure로 별도 보고한다. Sentinel-2에서 단일 고사목을 항상 판독할 수 있다는 가정은 세우지 않는다.

## 5. 평가: 무엇을 주 결과로 둘 것인가

**Primary:** novel concept×관계 조합의 grounded retrieval와 localization. 개념별·지역별 macro 결과, K support 곡선, 같은 query에서의 paired 차이를 보고한다. 관계의 anchor를 틀리게 선택한 경우 mask 일부가 맞더라도 정답 처리하지 않는다.

**Secondary utility:** observed-area MAE/bias, 근거와 일치하는 설명의 비율, 숫자/시점/지역의 명시적 오류. area와 description은 같은 field에서 나온 종속 결과이며 서로 독립된 발견처럼 세지 않는다.

**진단:** support-label shuffle, empty support, wrong support concept, query observation swap, EO zero, anchor swap, 센서 밴드 ablation. 이들은 모델이 해당 입력에 의존하는지를 보여 준다. 그 자체로 실제 생태 변화의 원인이나 'encoder가 유일한 병목'을 증명하지는 않는다.

구름·누락 구역은 known observed area와 unknown area를 따로 보고한다. 관측되지 않은 구역을 음성으로 간주하지 않는다. 예측 면적의 불확실성과 미관측 면적을 구분한다. 원인을 설명하는 문장은 별도 현장 근거 없이 허용하지 않는다.

신뢰구간은 질문 행 수가 아니라 concept/AOI/event의 독립 군집을 기준으로 계산한다. 학습 seed 반복과 query bootstrap을 혼동하지 않는다. 최종 효과 크기 기준과 중단 조건은 최종 결과 전에 고정한다. 작은 pilot 결과를 보고 threshold를 이동시키지 않는다.

## 6. 큰 모델을 쓰는 main과 encoder 검증 support의 분리

Main은 **실제 VLM**에서 진행한다. 현재 캐시가 검증되고 한 step 메모리 측정이 통과한다면 Qwen3-VL-8B를 첫 모델로 사용해도 좋다. 4B는 자원/구현 fallback이며 '무조건 작은 모델부터'가 연구상 필수는 아니다. plain Olmo3-7B 기반 OE1 결과는 actual-VLM main claim의 대체 증거가 아니다.

TerraScope는 InternVL3-8B와 SAM2를 사용하므로 InternVL3-8B를 두 번째 학생으로 써 matched-backbone 비교를 둔다. 다만 공개 TerraScope는 별도 grounding/CoT 학습 데이터를 사용했다. 백본만 같다는 이유로 같은 학습 조건의 비교가 되는 것은 아니다. 또한 InternVL3-8B의 LLM은 Qwen2.5이므로 완전히 다른 언어 계열 일반화라고 부르지 않는다.

Encoder support 실험은 native 분광 정보가 필요한 heldout EO task·new-concept dense probe·sensor ablation·retention 평가다. 이 보조 결과는 왜 main의 EO branch가 유용한지를 뒷받침할 수 있다. 단독 linear-probe 개선을 VLM compositional transfer 결과로 대체할 수 없다.

VLM 교체, dataset 확대, K support 확대, 날짜 수 확대, EO token 수 확대를 동시에 하지 않는다. 무엇 때문에 좋아졌는지 알 수 없기 때문이다. 큰 모델 선택 자체는 신규성이 아니며, 데이터 규모도 mechanism ablation을 대신하지 않는다.

## 7. 단계별 반증과 중단 판단

1. **Interface gate:** 원래 RGB VLM 동작 재현, EO 슬롯의 shape/position/mask 일치, 입력 adapter와 의도한 EO 파라미터로 gradient 전달, zero/swap의 올바른 구현을 확인한다. 이 단계의 성공은 공학적 feasibility다.
2. **Falsification pilot:** semantic alias를 사용한 novel concepts에서 specialist+연산 baseline과 same-evidence projector를 먼저 비교한다. question-only나 잘못된 support로도 높은 점수가 나오면 데이터/평가를 고친다.
3. **Method gate:** 검증 군집에서 제안한 episodic 결합이 ordinary MTL 및 equivalent-hard-negative baseline보다 일관되게 나아야 대규모 학습으로 간다. 최종 기준은 pilot 전에 선언하고 필요한 표본 수를 결정한다.
4. **Main:** 고정된 최종 concept/AOI/event split, 최소 두 backbone 중 핵심 이전, support budget 곡선, primary metric과 실패 사례를 공개한다. 최종 test를 여러 번 보며 메커니즘을 수정하지 않는다.

다음이면 **현재 CVPR 방법 주장에는 no-go 또는 전환**이다: (a) mask+연산 baseline과 동등, (b) 추가 숫자·추가 밴드·추가 토큰·추가 annotation만으로 차이가 설명됨, (c) 새 개념 transfer가 class-name prior 또는 지리 누출에 의존, (d) EO 업데이트가 재현 가능한 이득을 주지 않고 ordinary projector가 충분함, (e) 독립 군집이 너무 적어 효과를 구별할 수 없음. 이때 유용한 시스템 구현, dataset/annotation 연구, negative result로 정직하게 범위를 바꾸는 선택은 가능하다.

## 8. 선행과 남은 신규성 질문

근거 mask를 생성하고 이를 설명에 사용하는 것 자체는 이미 TerraScope의 중심 내용이다. 검색/측정/설명을 한 UI에 넣는 것 또한 방법 기여를 자동으로 만들지 않는다. 가장 설득력 있는 남은 질문은 **native EO에서 새 개념을 적은 근거로 결합하는 학습이, 강한 mask-specialist와 동일 입력 projector가 못 하는 조합 전이를 제공하는가**다. 아직 답은 없다.

관련 사실의 1차 출처: [TerraScope, 특히 §3.2와 부록 G](https://arxiv.org/html/2603.19039v1), [InternVL3-8B 공식 모델 카드](https://huggingface.co/OpenGVLab/InternVL3-8B), [Qwen3-VL 공식 구현](https://raw.githubusercontent.com/huggingface/transformers/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py). 일반 meta-learning/개념 결합/조합 추론 선행과의 최종 차별성은 별도 prior-art 검토와 대조해야 한다. 이 문서는 그 차별성을 이미 입증했다고 주장하지 않는다.
