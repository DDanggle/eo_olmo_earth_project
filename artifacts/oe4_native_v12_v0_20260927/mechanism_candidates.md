# OlmoEarth + 실제 VLM: 검증할 메커니즘 후보 3개

작성 2026-09-27. 독립 방법·1차 선행 검토. 코드 구현, 서버 실행, 모델 다운로드, 결과 열람은 하지 않았다. 아래는 새로운 방법을 발견했다는 선언이 아니라 **기존 generic contrast에서 벗어날 수 있는 구체적인 가설과 반증 설계**다.

## 결정 요약

**첫 실제 pilot은 후보 1의 가장 작은 부분인 '질문별 관측 개입에 대한 답의 변화'부터 권한다.** 새 거대 모듈을 동시에 넣지 않는다. 후보 2는 사용자 목표인 소량 교정·제약 센서에 더 직접적이지만 관측 가능성 정답이 어려워 두 번째다. 후보 3은 독립적인 주 기여보다는 측정 신뢰성과 표현 감사 부품으로 둔다.

P-v0의 동일 tuple 독립 loss 대 listwise contrast는 좋은 반증 baseline이지만, supervised contrastive learning의 새 이름일 가능성이 높다. 이를 핵심 신규성으로 밀어붙이지 않는다. 단순 query conditioning, missing-modality augmentation, support mask 후처리, 면적 보존도 각각 이미 강한 선행이 있다.

## 1. 먼저 확인된 선행과 설계 제약

| 1차 출처 | 실제로 이미 한 일 | 본 연구에서 피할 주장 |
|---|---|---|
| [OlmoEarth v1.2 §2.2](https://arxiv.org/html/2605.20804v3) | modality당 단일 bandset, random band dropout, nonlinear projection. S2 밴드를 전부 함께 tokenize하며 target은 전체 밴드를 본다. | band dropout/결측 밴드 예측이 새 원리라는 주장; 출력 token을 밴드별 token으로 오해 |
| [OmniSat, ECCV 2024](https://arxiv.org/abs/2404.08351) | 정합된 다양한 EO modality의 self-supervised fusion; 단일 modality inference에도 도움 | multi-sensor 사전학습 후 일부 센서만 쓰는 것 자체의 신규성 |
| [SGMA, 2026 preprint](https://arxiv.org/html/2603.02505v1) | class prototype를 이용한 modality robustness 추정·가중 fusion·sampling | class별 sensor weight와 missing-modality robustness 자체의 신규성 |
| [DIS2, 2026](https://arxiv.org/html/2601.13502v1) | 빠진 modality 특징 보완 distillation, class-wise feature learning, 여러 해상도 fusion | full-modality teacher로 missing modality를 보완하는 것 자체의 신규성 |
| [InstructBLIP, NeurIPS 2023](https://arxiv.org/abs/2305.06500) | instruction-aware Q-Former로 질문에 맞는 시각 특징 추출 | text-conditioned connector가 처음이라는 주장 |
| [QA-ViT, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Ganz_Question_Aware_Vision_Transformer_for_Multimodal_Reasoning_CVPR_2024_paper.html) | 질문 정보를 vision encoder 내부에 넣는 question-aware encoding | 질문을 OlmoEarth 내부에 넣는다는 사실만으로 신규성 주장 |
| [TimeSenCLIP](https://arxiv.org/abs/2508.11919v3) | S2 시계열과 지상 영상을 temporal contrast로 연결; spectral/temporal 특징에 중점 | 시계열·분광을 언어/시각 의미와 연결하는 것 자체의 신규성 |
| [Scale-MAE](https://arxiv.org/abs/2212.14532) | 실제 지표면 scale를 반영한 위치 표현과 다중 scale 복원 | 지리 scale·다중 해상도를 추가한 사실 자체의 신규성 |
| [Learning To Count Everything, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Ranjan_Learning_To_Count_Everything_CVPR_2021_paper.html) | few-shot exemplar를 받아 새 category의 density map와 count를 예측 | 적은 support에서 density를 만들고 합산하는 것 자체의 신규성 |
| [Hard-Constrained Deep Learning for Climate Downscaling, JMLR](https://jmlr.org/papers/volume24/23-0158/23-0158.pdf) | 물리량의 집계 보존을 신경망에 강제 | 합산 보존 constraint가 처음이라는 주장 |
| [TerraScope](https://arxiv.org/html/2603.19039v1) | EO 근거 mask와 언어 reasoning·측정·다중 입력 결합 | grounded mask를 설명하거나 면적을 답하는 기능 자체의 신규성 |

이 표는 관련 선행의 제한된 대조이며 exhaustive novelty search는 아니다. 다른 연구가 없다는 결론을 내리지 않는다.

## 2. 후보 1 — 질문에 필요한 관측을 구별하도록 native EO 읽기를 학습

### 문제와 메커니즘

RGB에서 거의 같게 보이는 두 지역도 NIR/SWIR 또는 다른 시점에 따라 요청의 정답이 다를 수 있다. 기존 generic QA는 지역·문구 prior만으로도 답을 맞힐 수 있다. 목표는 **어떤 관측이 바뀌면 답도 바뀌어야 하고, 어떤 변화는 무시해야 하는지**를 encoder와 실제 VLM의 결합 경로에 학습시키는 것이다.

모델은 OlmoEarth Eθ, 기존 RGB 경로를 보존한 실제 VLM Vω, 작은 EO connector Aφ로 구성한다. Q는 질문, S는 support, M은 관측/밴드 availability다.

\[
z=E_\theta(X,M; q),\qquad
P(y\mid X,Q,S,M)=V_\omega(Q,S,\mathrm{RGB},A_\phi(z)).
\]

첫 버전은 E 내부의 q conditioning 없이도 시작할 수 있다. 다음 버전에서 질문 embedding을 projection의 저랭크 residual 또는 시간/공간 token reader에 넣는다. **v1.2의 S2 출력 token에는 밴드별 축이 없다.** band-specific routing을 주장하려면 band identity가 살아 있는 projection 이전 경로를 수정하거나 별도 band-preserving stem을 만들어야 한다. 혼합된 출력 token에 band 이름만 붙여서는 안 된다.

가능한 작은 pre-projection 잔차는 `W(q)=W0+U diag(Rq) V`다. 이는 generic conditional projection이므로 그 자체의 신규성은 아니다. 반드시 identity 초기화·원 native 경로 parity·공식 결측 처리·OlmoEarth gradient를 확인한다. 모든 입력을 이미 encode한 뒤 token을 줄였다면 encoder compute 절감을 주장할 수 없다.

### 학습을 구체적으로 달리하는 부분

record마다 원 관측 X와 정답이 검증된 개입 관측 Xa를 함께 사용한다. 실제 같은 현상의 다른 관측과 synthetic sensor intervention은 별도 태그로 둔다. 개입 후 정답 ya를 모르면 pair를 자동 생성하지 않는다.

예를 들어 A/B 후보의 VLM log-odds를 sθ라 할 때, 동일 tuple의 독립 QA loss에 다음과 같은 **paired response supervision**을 시험한다.

\[
L_{\rm dep}=\sum_{(X,X^a,Q)}
\rho\left([s_\theta(X^a,Q)-s_\theta(X,Q)]-\kappa\Delta_{\rm ref}\right).
\]

Δref는 reference가 정한 답의 변화 방향/불변 여부이며, ρ는 robust loss다. κ와 정규화는 dev에서 고정한다. 실제 contrastive loss와 수학적으로 같은 변형이면 별도 새 학습 원리라고 부르지 않는다. 더 중요한 질문은 이런 학습이 **훈련에 없던 개념·관계·관측 패턴으로까지 전달되는가**다.

### 결정 실험

- 동일 X/Xa, 질문, 정답, forward 수, token 수, encoder trainable 범위를 쓰는 **independent QA**와 비교한다. 제안군만 두 번째 관측을 보게 하면 안 된다.
- 강한 **InstructBLIP-style query reader**와 **QA-ViT-style query conditioning**을 conceptually aligned baseline으로 둔다. 일반 question-aware 연결만으로 충분하면 이 후보의 별도 기여는 약하다.
- EO conditioner에만 질문을 shuffle하고 VLM에는 원 질문을 준다. 모델이 여전히 같은 성능이면 native query reading의 역할이 없을 수 있다. 단, shuffle에 따른 분포 이탈만으로 인과적 우월성을 증명하지 않는다.
- 정답이 바뀌어야 하는 pair와 불변이어야 하는 pair를 모두 평가한다. flip 빈도만 높아져도 성공이 아니다.
- 최종 점수는 실제 VLM의 후보 ID/구조화 응답에서 계산한다. field-only 결과는 별도 진단이다.

**주장 가능한 후보 차이:** query conditioning을 발명한 것이 아니라, native spectral/time evidence에 대한 검증된 dependency를 학습하여 새로운 support 개념의 task transfer를 개선하는 것. **주요 실패 위험:** model-specific saliency를 정답처럼 사용, synthetic arithmetic에만 적응, 동일 증거 independent QA와 차이 없음.

## 3. 후보 2 — 관측 조건이 달라졌을 때 소량 교정이 어디까지 유효한지 학습

### 문제와 메커니즘

사용자가 충분한 센서를 보고 '이 패턴이 C17이다'라고 교정했더라도, NIR·계절 피크·SAR가 없는 query에 같은 확신을 강제하면 안 된다. 목표는 missing modality를 무조건 복원하는 것이 아니라 **support가 정의한 개념은 유지하면서 가용 관측에서 판독 가능한 정도를 구별**하는 것이다.

\[
c=H_\phi(S),\qquad
(p_i,u_i)=F_\theta(X_M,c,M),\qquad
\hat y=V_\omega(Q,\mathrm{RGB},A_\phi(p,u,z),M).
\]

한 support 개념을 여러 관측 조건 M에 적용하는 episode를 만든다. 충분한 조건에서는 교정의 효과가 유지되도록 하고, 검증된 불충분 조건에서는 posterior/답의 불확실성을 보존한다. 추가 관측이 들어오면 같은 c로 답을 회복하는지를 평가한다. 'unknown'이라는 별도 문구만 학습시키는 것으로 끝내지 않는다.

새 후보의 구조는 **관측 가능한 증거로 뒷받침되는 교정과 관측 조건에 의존하는 불확실성을 분리하는 shared concept + availability-conditioned posterior**다. 이를 위한 likelihood target을 (y, sufficient)로 정의하거나, 검증된 ambiguous set에 대해 soft target을 사용할 수 있다. 여기에 full-observation teacher의 단일 정답을 무조건 distill하지 않는다.

### 가장 어려운 정답 문제

밴드를 지웠다고 자동으로 판단 불가능한 것은 아니다. RGB·공간·계절 prior로도 추론할 수 있고, full teacher가 틀릴 수도 있다. 따라서 `band missing → unknown`을 gold로 만들어서는 안 된다. 미관측이 논리적으로 답을 결정하지 못하는 controlled task, 독립 판독 자료, 또는 명시적으로 weak라고 기록한 ambiguity reference가 필요하다. 같은 부족 관측에서 label이 다른 정확한 collision을 synthetic으로 만들면 식별 가능성 검사는 가능하지만 실제 지구 분포의 gold는 아니다.

### 결정 실험

- **공식 v1.2 band dropout + 동일 support learner + calibrated confidence**가 첫 baseline이다. SGMA-style modality reliability와 full-teacher distillation도 같은 예산으로 비교한다.
- full-query → partial-query → restored-query에서 같은 support를 고정한다. support 자체의 관측 제한과 query 제한을 별도 축으로 둔다.
- 동일 coverage에서 오류율, risk-coverage curve, calibration, restored accuracy, 새 개념의 K-support 곡선을 보고한다. 전부 abstain하는 모델은 개선이 아니다.
- 제공된 관측 이외의 센서를 teacher가 사용했으면 train privileged information으로 모든 비교에 동일하게 제공하고, inference 정보와 구별한다.

**주장 가능한 후보 차이:** class-specific sensor fusion을 넘어 새 support 교정의 유효 범위를 관측 조건에 따라 전이시키는가. **주요 실패 위험:** 단순 dropout/confidence calibration과 동일, ambiguity gold 부재, teacher의 hallucination을 distill. 독립 관측 가능성 계약 전에는 첫 pilot의 주 학습목표로 삼지 않는다.

## 4. 후보 3 — 지역을 자르거나 해상도를 바꾸어도 교정된 양이 일치하도록 학습

### 문제와 메커니즘

새 식생 피복 개념의 support 교정 후 같은 지역을 다른 crop/격자로 물으면 면적·순위·설명이 달라질 수 있다. 공통 concept field를 **물리 면적에 대해 적분 가능한 표현**으로 학습하고, 실제 VLM이 동일 근거를 읽도록 한다.

\[
\hat A_R(c)=\sum_{i\in R}a_i v_i p_i(c),\qquad
\hat A_R=\sum_k\hat A_{R_k}\;(R_k\text{가 서로 겹치지 않을 때}).
\]

단순 합산은 코드로도 강제할 수 있다. 학습 후보는 입력 해상도/시야가 달라져도 예측이 집계 관계를 만족하도록 하는 것이다. geometry-aware coarsening D를 쓰면

\[
L_{\rm grid}=\|D_a[p(X,c)]-p(D_X X,c)\|_{1,\mathrm{valid}}
\]

를 시험할 수 있다. foreground probability와 fractional cover를 구별하며, 같은 지역이라도 coarse grid에서 사라지는 작은 구조의 불확실성을 없애도록 강제해서는 안 된다. 센서 PSF·시점·registration이 다르면 단순 downsample은 실제 cross-sensor 관측과 동등하지 않다.

### 결정 실험과 우선순위

- 같은 segmentation field+정확한 면적 합산 baseline, 같은 scale augmentation을 쓰는 ordinary learner, Scale-MAE-style scale conditioning을 비교한다.
- 합산을 하드하게 맞춘 후에도 heldout resolution/domain의 **field 정확도와 실제 VLM의 근거 선택**이 더 좋아져야 표현 학습 기여다.
- 동일 mask/면적 숫자를 받는 VLM보다 나아지지 않으면 안전한 engineering 부품으로 남긴다. scalar를 입력으로 제공해 수치 응답이 좋아진 결과는 encoder 개선이 아니다.

few-shot density/합산과 hard conservation 선행이 강하므로 **현재 독립 주기여 우선순위는 낮다.** 또 관측 coverage의 가법성은 물리량 보존이며, 건강 악화·식생 지수·확률 자체에 무조건 질량 보존을 적용하는 것은 잘못이다.

## 5. 첫 실제 GPU pilot에 붙일 최소 학습

### 우선 실행할 것

현재 실제 VLM adapter가 준비 중이라면 구조를 크게 바꾸지 말고 **OlmoEarth v1.2-Base → 같은 connector → Qwen3-VL-8B**의 진짜 forward/backward부터 완료한다. 기존 RGB/DeepStack을 유지하고, EO encoder 업데이트와 저장·복원을 확인한다. VLM은 초기에는 frozen이어도 실제 모델 전체를 통과하며 EO까지 gradient가 돌아야 한다. 이는 큰 방법 결과가 아닌 feasibility다.

첫 비교는 두 arm으로 제한할 수 있다.

| Arm | 입력·감독·compute | 유일한 차이 |
|---|---|---|
| G-evidence | 동일 원본/개입 pair를 모두 forward, 같은 source QA | 각 tuple 독립 CE |
| P-evidence-v0 | G와 같은 pair·정답·forward·token·update 수 | 독립 CE + 검증된 pair의 response-change loss |

출력은 binary free text보다 A/B 또는 4개 후보 ID로 고정하고 후보 순서를 무작위화한다. 로그에는 raw generation과 제한된 후보 likelihood 점수를 둘 다 남기며, 유리한 쪽만 고르지 않는다. 100–300 optimizer steps·단일 seed는 엔지니어링/개발 진단이며 통계적 method success가 아니다. 표본 수·step 수는 root의 실제 GPU 예산 및 source 계약에 맞춰 동결한다.

### 당장 pair 정답이 부족하면

real PASTIS class label의 답은 NIR을 조작했다고 자동으로 바뀌지 않는다. 임의 band swap 뒤 crop label을 바꾸는 식으로 정답을 만들면 안 된다. 안전한 첫 확인은 실제 radiometry와 valid pixels가 검증된 native S2에서 **명시된 spectral quantity의 A/B 비교**다. 가령 reference NDVI 비교를 사용하되 BOA scaling/offset·band identity·0 denominator·cloud 처리를 먼저 확인한다. 이 task는 식생 건강/고사 원인의 정답이 아니다.

RGB는 그대로 두고 non-RGB 값만 바꾼 synthetic pair에서는 해당 quantity를 다시 계산하므로 정답 변화가 정의된다. 이 검사는 VLM이 EO branch의 non-RGB 정보를 읽는지 확인하는 controlled input test다. 자연 장면의 class/원인/새 개념 전이 결과로 발표하면 안 된다. source segmentation QA만 준비됐다면 우선 ordinary QA 학습을 실행하고, 정확한 paired targets가 준비될 때 loss를 추가한다. 라벨이 없는 pair를 만들며 학습을 서두를 이유는 없다.

### 첫 결정 기준

1. 실제 VLM 응답이 올바른 native EO 입력에 의존하는가: RGB 고정 non-RGB intervention, source-answer imbalance, candidate position을 통제한다.
2. P의 이득이 같은 pair를 독립적으로 학습한 G보다 있는가. 없다면 pair loss를 새 원리라고 밀지 않는다.
3. 다음 단계에서 **학습에 쓰지 않은 개념/관계**로 이득이 남는가. quantity 진단만 좋아지면 semantic transfer 주장은 아직 없다.
4. 그 뒤에만 query-conditioned projection을 추가하고, 일반 InstructBLIP/QA-ViT형 방법과 분리한다. 여러 후보를 한 번에 합쳐 어떤 요소가 필요한지 잃지 않는다.

## 6. 공통 공정성 계약과 최종 의견

모든 arm에 같은 source regions, dates, bands, support, native availability, labels, synthetic/real tags, 추가 teacher 정보, EO/RGB token 예산을 준다. 같은 forward 수와 parameter 범위를 우선 고정하며 loss마다 추가되는 연산량을 기록한다. 실제 wall time과 optimizer steps를 동시에 정확히 맞출 수 없으면 하나를 primary budget으로 사전 지정한다.

독립 spatial groups와 concept split을 유지하고, 원 모델의 pretraining 노출과 이번 추가학습의 감독/공간 노출을 구별한다. 최종 test를 열기 전에 후보 선택을 끝낸다. 후보들을 연속 시도해서 우연히 좋아진 최종 test만 남기지 않는다.

**지금 추천하는 것은 '세 개의 참신한 방법'이 아니라, 선행에 의해 이미 약해진 아이디어를 걷어내고 EO 고유 입력 의존성을 먼저 증명하는 순서다.** 후보 1은 구현·반증 가능성이 가장 높고, 후보 2는 성공하면 사용자 요구에 더 직접적이며, 후보 3은 측정 일관성에 유용하지만 주 신규성으로는 가장 약하다. 실제 evidence/교정 전이가 generic baselines를 넘지 못하면, 더 큰 학습이나 더 좋은 서술로 신규성을 만들어 낼 수는 없다.
