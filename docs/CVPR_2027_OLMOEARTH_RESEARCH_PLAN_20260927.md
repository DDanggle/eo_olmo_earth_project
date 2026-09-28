# CVPR 2027 연구계획 — 적은 교정으로 새 용도를 배우는 OlmoEarth 기반 VLM

> **09/28 23:18 KST — 사용자 요청으로 CVPR 기여 재검토.** [방법 가설·선행 반례·결정 실험](CVPR_CONTRIBUTION_DECISION_20260928.md)에 현재 결과와 입력 감사를 반영했다. 새 방법의 성능/신규성은 미입증이며 기존 protocol은 보존한다. 이번 수동 갱신으로 예약이나 GPU를 재개하지 않았다.

> **통합 연구 방향 v2:** [환경이 달라져도 전문가 교정을 재사용하는 OlmoEarth](OLMOEARTH_UNIFIED_CONTRIBUTION_20260927.md). 아래 문서의 개별 요소를 하나의 모델 목표와 결정 실험으로 연결했다. 제안 설계이며 기존 실행·판정·사전등록은 보존한다.


> 후속 실제 실행: [OE4 신규성 지도·공식 자료 확보·학습 경로 검사](OE4_NOVELTY_AND_NATIVE_VLM_EXECUTION_20260927.md). 이 문서의 미실행 상태는 작성 당시 이력이며 후속 상태는 OE4를 따른다.


작성: 2026-09-27 KST. 상태: **독립 검토를 반영한 연구 설계 v1; 구현·실험 결과 또는 동결 사전등록이 아니다.** 이전 D1/OE1/OE2 판정과 사전등록은 보존한다. 이 문서가 후속 연구의 큰 방향과 투자 순서를 갱신한다.

## 1. 결정과 목표

**목표는 OlmoEarth를 쉽게 가르치고 여러 업무에 재사용할 수 있는 EO–VLM 모델로 확장하는 것이다.** 원본 encoder의 linear probe 개선만으로 끝내지 않는다. 원본 EO 평가와 공식 목적 추가 학습은 전체 모델 개선을 설명하기 위한 대조군이다.

현재 결정은 **한정된 방법 반증 실험 진행**이다. 성공하면 100k 규모의 본 비교와 CVPR 제출로 확장한다. 아직 연구의 핵심 효과가 측정되지 않았으므로 채택 가능성 50% 이상이라고 계산할 근거가 없다. 세 검토자의 조건부 진행 의견도 채택 확률이 아니다.

사용 장면의 예: 사용자가 새로운 작물이나 피복 패턴의 양성·대조 영역 20개를 지정한다. 모델은 다른 지역에서 같은 대상을 찾고, 주어진 구역과의 관계에 맞춰 후보를 고르며, 관측한 범위의 면적과 근거를 설명한다. 해당 개념에 대한 관계 질문을 별도로 수백 개 라벨링하지 않아도 되는지가 연구 문제다. 다른 센서·시기로의 전이는 정합 자료를 확보한 뒤 추가로 검증한다.

성공했을 때 공개할 연구 자산은 네 가지다.

1. native EO encoder와 실제 VLM을 함께 학습한 모델 및 소량 support 적응 경로.
2. 새로운 개념을 다른 시각적 판단에 재사용하도록 만드는 학습 방법과 반증 실험.
3. 공개 EO 라벨의 출처·관측 범위·공간 분할을 추적하는 학습/평가 자료와 평가 코드.
4. 새로운 지역·개념에 적용할 수 있는 가중치/adapter·예제·재현 가능한 학습 recipe.

이는 제품의 기능 목록이 아니라 모델·학습·평가를 포함한 연구 프로그램이다. 논문의 중심 기여는 2번의 효과로 정한다. UI, 데이터 수, 모델 크기, 여러 loss를 붙인 사실만으로 신규성을 주장하지 않는다.

## 2. 지금까지 확인된 것과 아직 없는 것

| 항목 | 확인된 상태 | 이 계획에서의 역할 |
|---|---|---|
| OE1 | v1-Tiny+plain Olmo3-7B, 단일 seed·256개 dev QA; blind 66.80 / frozen 68.36 / joint 72.27 BA | encoder 공동학습 경로의 과거 진단. 실제 VLM·새 사전학습 원리의 증거가 아님 |
| OE1 편향 감사 | train class prior 68.36; 동률 규칙 차이 확인 | 소규모 yes/no 점수로 연구 가치를 판정하지 않는 이유 |
| OE2 | frozen Tiny 20장 CPU probe, 원공간/16/64토큰 MAE .1245/.1553/.1458 | 압축/readout 조합 진단. VLM 병목·불가역 정보손실·최신 OlmoEarth 결함의 증거가 아님 |
| 원본 EO 기준선 | Sen1Floods11·m-cashew-plant 공식 source/CLI inventory | 실제 평가 실행·성능값은 아직 없음 |
| 학습 준비 | full pretrain 환경·checkpoint loader bridge·자료 완전성 확인 필요 | 본 학습 실행 준비 완료 상태가 아님 |
| 이번 갱신 | 3개 독립 리뷰, 공식 논문/자료 조사, 공개 자산 metadata 확인 | 새 모델 학습·GPU 성능·유료 AI 라벨은 없음 |

기존 결과는 [OE1](OE1_BENTXT_FEASIBILITY_20260926.md), [OE2 감사](OE2_PROBLEM_AUDIT_AND_FIXES_20260927.md), [이전 큰 계획](OLMOEARTH_CAPABILITY_PROGRAM_20260927.md)에 보존한다. 긍정 결과가 나올 때까지 test를 반복하는 방식은 쓰지 않는다. 개발 실패는 기록하고 최종 평가를 열기 전에 방법을 확정한다.

## 3. TerraScope 등과 무엇이 달라야 하는가

| 가까운 선행 | 이미 다룬 부분 | 우리에게 남는 검증 질문 |
|---|---|---|
| [TerraScope](https://arxiv.org/html/2603.19039v1) | mask와 언어 생성, 영역 근거, 측정, optical/SAR, 다중시점 활용 | 새 개념에 소량의 근거만 제공했을 때 미학습 개념×관계 조합으로 얼마나 전이하는가 |
| [SPEX](https://arxiv.org/html/2508.05202v2), [MS-CLIP](https://arxiv.org/html/2503.15969v3), [DOFA-CLIP](https://arxiv.org/abs/2503.06312v2) | 분광/다중모달 입력과 언어 학습, encoder 적응, 검색/분할 | 단순 연결·분광 입력 증가를 넘는 학습상의 이득이 있는가 |
| [Spectral-LLaVA](https://arxiv.org/html/2501.10144v1), [RSCoVLM](https://arxiv.org/html/2511.21272v2) | EO 특징의 언어 연결, 여러 RS 작업의 통합 | 일반 연결부·공동학습보다 새 개념 재사용이 나아지는가 |
| [SegGPT](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_SegGPT_Towards_Segmenting_Everything_in_Context_ICCV_2023_paper.html), [SEEM](https://papers.neurips.cc/paper_files/paper/2023/hash/3ef61f7e4afacf9a2c5b71c726172b86-Abstract-Conference.html) | 예시 기반 분할, 시각/언어 prompt 조합 | prototype·prompt segmentation·generic episodic learning으로 충분한가 |
| [Composition-Aware Pretraining](https://arxiv.org/html/2608.30817v1) | 장면의 시각 구성 분포를 이용한 EO 학습 | 본 연구의 개념×연산 재사용과 구별되는 목적·효과를 보이는가 |

**중요한 반론:** mask가 좋아지면 면적 합산, 비교, 설명이 함께 좋아질 수 있다. 이는 응용상 가치가 있지만 네 개의 독립적인 능력 전이로 세지 않는다. 심지어 관계 검색도 mask+기하 연산으로 풀릴 수 있다. 따라서 같은 EO·support를 쓰는 분할 전문 모델+연산+동일 VLM을 반드시 비교한다. 이를 이기지 못하면 화려한 설명으로 신규성을 보충하지 않는다.

논문의 가설은 “우리가 처음 EO에 언어를 붙였다”가 아니다. **새 개념을 배울 때 생긴 시각적 지식이 여러 조건과 지역에서 재사용되도록 학습하는 것이, 일반 joint training이나 generic meta-learning보다 효과적인가**다. 현재 메커니즘의 신규성은 후보이며 입증된 사실이 아니다.

## 4. 모델: 최신 OlmoEarth와 실제 VLM이 주체

[현행 공식 논문](https://arxiv.org/html/2605.20804v3)은 OlmoEarth v1.2다. 신규 주모델 후보는 **v1.2-Base**이며 v1-Tiny는 과거 진단으로만 남긴다. 공식 HF 저장소 `allenai/OlmoEarth-v1_2-Base`, revision `2e99a734a30e9aeb993aaa39946c6dcf554739e0`의 공개 metadata를 확인했다. 가중치 다운로드/실제 로드 검증은 아직 안 했다.

실제 VLM 후보는 기존 서버에 파일이 있는 **Qwen3-VL-8B-Instruct**다. 기본 RGB 경로와 DeepStack을 보존하고 native EO 슬롯을 추가한다. 8B의 학습 처리량·메모리를 먼저 측정한다. 크기를 줄여야 하면 test 전에 명시적으로 4B 등으로 변경하고 전체 대조군에 동일하게 적용한다. plain Olmo3 결과를 실제 VLM 결과로 대체하지 않는다.

흐름:

```text
원본 S1/S2/Landsat 관측 + 날짜 + 관측/밴드 유효성
             → OlmoEarth v1.2 (학습)
             → 공간·시간 EO 표현
support 영역/점 + 양성·대조 예시 → 개념 결합 모듈
             → query에서 해당 개념의 위치별 예측
질문 + anchor + EO 근거 + 기존 RGB 경로 → 실제 VLM
             → 후보/영역/관측 범위의 수치/근거 설명
```

S1/S2/Landsat를 모두 쓸 수 있다는 구조와 실제 task에서 센서를 모두 검증했다는 주장은 구별한다. 첫 감독 task는 S2 중심이고 S1은 정합된 PASTIS-R 등에서 조건부 확장한다. Landsat task transfer와 장기 기후 영향은 별도 후속 범위다.

학습은 두 부분을 둔다. **A:** native EO의 원 목적을 유지하면서 support 기반 개념·영역·관계 학습을 추가하여 encoder 가중치를 바꾼다. **B:** 실제 VLM이 그 근거를 이용하도록 동일한 instruction 예산으로 학습한다. A 전후와 B 전후 checkpoint를 모두 남겨 공동적응과 EO 자체 개선을 구분한다. 언어 decoder는 LoRA 등 비용이 통제되는 동일 적응법을 우선하고 encoder는 실제 gradient/가중치 변화로 확인한다.

실행 단계를 다음처럼 고정해 조건 사이의 모호함을 줄인다.

- **A:** C는 공식 EO 목적만, S는 추가 직접 지도, I는 일반 instruction, G는 일반 episodic, P는 G에 결합 목적을 추가한다. I/G/P의 VLM 본체 가중치는 고정하지만 EO 입력까지 역전파 그래프를 유지한다. frozen VLM forward 전체를 `no_grad`로 감싸지 않는다. encoder와 허용된 field/연결 모듈은 학습한다.
- **B:** 모든 arm은 같은 instruction records와 예산으로 VLM LoRA 및 연결/field 모듈을 적응한다. encoder는 각 A의 종료 가중치에 고정한다. 연결부는 공통 초기 상태에서 시작하고 학습된 concept/field 모듈은 이어받는다. 따라서 주 결과는 전체 시스템 학습 효과이며, EO 자체 개선은 fresh field/readout 감사로 별도 확인한다.
- **I-frozen/P-frozen:** encoder가 A와 B 모두에서 원본 공개 가중치에 고정된다. B에서만 얼린 것을 frozen 대조로 부르지 않는다. B에서도 encoder를 다시 joint 학습하는 실험은 선택 후속이며 아래21회에 포함되지 않는다.
- C도 B에서는 instruction을 받으므로 “언어 감독이 전혀 없는 최종 모델”이 아니다. C는 **A단계의 공식 목적 추가 학습 대조**다. A/B 각각의 label exposure·steps·GPU 시간을 따로 기록한다.

원 목적 대조는 **선택한 v1.2의 공식 recipe**와 맞춘다. 기존 inventory의 v1 script를 v1.2에 그대로 붙여 “공식 목적 재현”이라고 하지 않는다. local official commit `0497dfbb6711ded4e6bf10cf089fc1e4d58c186b`의 `scripts/official/v1_2/base.py`와 모듈 import를 감사한다. public `weights.pth`와 distributed trainer checkpoint는 형식이 다르므로 공식 loader 연결 및 고정 입력 parity를 확인한다. optimizer를 새로 시작한 계속 사전학습이며 원 optimizer 상태까지 복원한 것은 아니다.

## 5. 주 방법 후보: 예시로 배운 개념을 다른 판단에 재사용하도록 학습

핵심 구현은 **support와 query를 구별하고, 모든 관련 작업이 동일한 개념 field를 참조하도록 만드는 것**이다. 초안 수식은 다음과 같다.

\[
Z=E_\theta(X),\quad c=H_\phi(\{E_\theta(X_s),y_s\}_{s=1}^{K}),\quad p_i=\sigma(q_\phi(Z_i,c)).
\]

`c`는 이름만 외운 클래스 ID가 아니라 양성·대조 support로 정의되는 개념이다. primary에서는 이름을 임의 별칭으로 가려 기존 언어 지식과 새 support 학습을 구분한다. 자연 이름을 주는 응용 평가도 따로 한다. query의 정답 mask는 입력으로 주지 않는다.

관계와 anchor는 동일 field에서 선택 범위를 정한다. 면적 합과 기하학은 명시적 연산으로 계산할 수 있다. 이때 VLM이 모든 기하학을 새로 발명했다거나 확률의 합이 실제 fractional cover 정답이라고 말하지 않는다. 예측 mask 면적과 관측 불가능 면적을 구별한다.

학습 episode는 `(support, query observation, operation, anchor, source label)`로 기록한다. support의 개념과 anchor를 바꾼 **정답이 확인된 대조 tuple**을 함께 사용한다. 모호하거나 미라벨인 곳을 음성으로 만들지 않는다.

\[
L=L_{\mathrm{official}}+\lambda_f L_{\mathrm{field}}
+\lambda_b L_{\mathrm{binding}}+\lambda_q L_{\mathrm{query}},
\qquad
L_{\mathrm{binding}}=-\log\frac{e^{s(e^+)/\tau}}{\sum_{e\in\{e^+\}\cup N}e^{s(e)/\tau}}.
\]

`L_query`는 support에서 제공한 감독 유형과 다른 연산의 query에도 같은 개념을 적용하도록 한다. 이것은 loss 항을 늘리는 것 자체가 기여라는 뜻이 아니다. 제안군의 source facts, 음성 tuple, 작업 분포, 추가 계산량을 ordinary multitask 및 generic episodic 대조에도 제공한다. 모든 arm의 원 목적 replay 비율도 맞춘다.

**시험할 차이:** support를 바꾸면 개념 field와 답이 함께 바뀌고, anchor를 바꾸면 올바른 하위 영역으로 선택이 바뀌며, 이런 결합이 학습하지 않은 개념×관계 조합에서도 유지되는가. 새 이름만 보고 찍거나 다른 작업 head가 별도로 외운 답으로는 설명할 수 없어야 한다.

첫 구현은 feed-forward support aggregation을 사용한다. MAML식 inner-loop까지 동시에 추가하지 않는다. test-time weight adaptation은 모든 방법에 동일 optimizer budget을 제공하는 별도 실험이다. K support inference와 K-shot finetuning을 같은 결과로 섞지 않는다.

### G와 P를 식별하는 첫 구현 명세

두 독립 리뷰가 “현재 수식은 generic prototype 학습으로도 읽힌다”고 지적했다. 이를 해결하기 위해 v0의 차이를 좁고 명확하게 고정한다. 이것은 **신규성이 확인된 해법이 아니라, 기존 방법으로 충분한지 확인할 최소 반증 구현**이다.

| 항목 | G: 강한 generic episodic 대조 | P-v0: 검증할 제안 |
|---|---|---|
| 데이터 | 동일 support/query/확인된 음성 tuple/작업 제외표 | G와 동일 |
| forward | 동일 encoder, support aggregator, 공통 field, 연산, VLM | G와 동일; 추가 파라미터 없음 |
| 기본 감독 | 공식 EO + field loss + 모든 tuple의 독립 정답/언어 loss | G와 동일 |
| 추가 결합 | 없음; 대신 동등 비용의 독립 tuple loss 연산/업데이트 통제 | 같은 query에서 support·anchor·연산이 달라지는 tuple의 grounded score를 listwise contrast로 결합 |
| gradient | 각 tuple의 독립 supervision에서 E/H/q로 역전파 | 위 경로 + tuple 사이 상대 점수의 gradient가 같은 E/H/q로 전달 |

```python
# 의사코드: 실제 실행 코드 아님. 두 arm에 동일 facts/negatives를 제공한다.
tuples = verified_episode_records(batch)
outputs = shared_model(tuples)  # 동일 architecture/모든 negative forward
common = official_loss + field_loss(outputs) + independent_target_loss(outputs)
loss_G = common
loss_P = common + lambda_binding * joint_tuple_contrast(outputs, tuples.groups)
```

이 차이는 그 자체로 기존 supervised contrastive/meta-learning과 구별되는 큰 신규성이 아니다. **generic supervised contrast로 동일 결과가 나면 새로운 원리라는 주장을 하지 않는다.** ordinary loss reweighting, 동등 hard negatives, capacity/compute를 맞춘 G가 우선 반증 대상이다. v0에서 의미 있는 전이가 나오면 어떤 결합/관측 표현이 필요한지 ablation으로 설명할 수 있어야 한다. G/P data→forward→loss→gradient 표와 실행 코드가 일치하지 않으면 본 학습을 열지 않는다. 더 복잡한 방법을 추가할 경우 새 버전·dev 실험으로 등록하고 v0 실패를 보존한다.

면적 보존 token 압축은 선택 부품이다. 성능 차이가 사실상 추가 mask/숫자/토큰에서 오면 같은 근거를 받는 대조군으로 드러내고 중심 방법 주장에서 제외한다.

## 6. 무엇으로 학습하고 무엇으로 평가하는가

두 자료 흐름을 분리한다. **공식 native EO corpus**는 원 목적 유지와 센서 표현 학습에 쓴다. **독립 참조 라벨 corpus**는 개념/영역/관계 감독과 평가에 쓴다. 두 corpus가 지리적으로 겹치는지 기록한다. 사전학습 원모델의 전체 노출을 완전히 배제할 수 없다면 “기반모델이 한 번도 본 적 없는 장소/개념”이라고 하지 않는다. 새로 설계한 추가 학습의 holdout과 supervised label의 holdout을 정확히 주장한다.

**새로 수행하는 A/B 학습에서는 locked query와 test support의 source footprint·공간 buffer·연결 날짜/사건을 모두 제외한다.** 좌표를 확인할 수 없는 공식 subset record는 겹침을 모르는 상태로 유지하지 않고 spatial-generalization 확증용 학습에서 제외한다. 원 공개 checkpoint의 과거 노출은 이 절차로 제거되지 않는다. native replay의 WorldCover/CDL 등 지도 target, parent class, negative label, caption이 held-out 개념을 다시 감독하는지도 ontology 장부로 감사한다. 관계가 남는 경우 “전 학습에서 미노출인 의미 개념” 대신 **downstream 추가 감독에서 보류한 support-defined target**이라고 범위를 제한한다. 단순 alias 교체로 이 문제를 해결했다고 하지 않는다.

| 자료 | 사용할 역할 | 계약/한계 | 실제 확보 상태 |
|---|---|---|---|
| 공식 OlmoEarth pretrain | C/S/I/G/P 공통 원 목적·native 관측 | 지도 target·sensor availability·날짜·CRS·원 recipe 일치 필요 | 1k H5 archive URL/17,855,037,440 bytes 확인; 미다운로드 |
| PASTIS / 조건부 PASTIS-R | 농업 개념의 support, parcel grounding, 관계 검색; 첫 공개 primary 후보 | crop의 계절 라벨을 매월 상태 변화·건강으로 해석하지 않음; S2 10밴드와 native 모델 입력 정합 | 서버 `geobench2/pastis` 이름 확인; 완전성/원 포맷 미검증 |
| MADOS | 농업 밖 해양/연안 분광 패턴의 외부 전이 | sparse label; 미라벨 배경은 음성이 아니며 전체 면적/완전 gallery relevance를 평가할 수 없음 | 다운로드·검수 미실행 |
| LUCAS Copernicus | 현장 판독 개념의 뜻·관측 가능성·support 확인 | 부분 polygon의 라벨이며 전체 scene mask가 아님; 작은 현장 영역의 위성 판독 가능성 확인 | 문헌·출처 확인 |
| Sen1Floods11, m-cashew-plant | 독립 EO 유지/전이 감사 | 원 official protocol/normalizer/입력 정합; 주 VLM 결과와 별개 | source inventory만 완료 |
| DynamicEarthNet | 향후 실제 시계열 변화 확장 | 기존 GeoBench2는 Planet 계열; S2 동시 자료·날짜·공간 연결을 별도로 검증 | 디렉터리 이름 확인만으로 native S2 준비 완료라고 하지 않음 |

출처: [공식 corpus](https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Pretraining-Dataset.md), [PASTIS](https://github.com/VSainteuf/pastis-benchmark), [MADOS](https://github.com/gkakogeorgiou/mados), [LUCAS 현장 자료](https://essd.copernicus.org/articles/16/5723/2024/), [DynamicEarthNet 논문](https://arxiv.org/abs/2203.12560).

PASTIS의 4개 광역 tile는 많은 parcel 수와 별개다. 광역 지역 일반화의 독립 표본을 10만 개라고 세지 않는다. 원 공식 fold는 그대로 보고하고 별도의 공간 holdout은 별도 protocol로 표시한다. domain transfer는 MADOS 등에서 지원된 label 범위만 비교한다. MADOS의 불완전 relevance를 전체 retrieval AP로 처리하지 않는다. 독립적인 두 번째 dense/검색 gold를 확보하지 못하면 농업 중심 범위를 그대로 명시한다.

고사목·쓰레기·기후는 이름을 붙이는 것만으로 gold가 생기지 않는다. Sentinel-2에서 개별 나무의 사인이나 작은 플라스틱 개체를 판별한다고 약속하지 않는다. 먼저 판독 가능한 피복/군집/오염 패턴에서 시작한다. 고해상도 현장자료와 연결되면 수목 건강, 장기 관측과 원인 참조가 생기면 기후 영향으로 확장한다. 실제 변화·계절 반복·관측 부족도 날짜별 reference가 있을 때만 평가한다.

## 7. 규모: 크게 하되 세는 단위를 바르게

공식 corpus의 원 지역 수는 약 285k이고 H5는 더 작은 subtile로 나뉜다. 지역, 관측 묶음, episode, 질문은 다른 단위다. 아래는 목표이며 확보량이 아니다.

| 단계 | native 관측 묶음 목표 | supervised 구성 | 목적 |
|---|---:|---|---|
| 환경/자료 검증 | 공식 1k subset | 실제 라벨 소량과 synthetic interface checks | 로더·loss·source alignment·처리량 |
| 반증 pilot | 10k 이하, 비용 실측 후 확정 | 확보한 참조 영역에서 동일한 support/query tuple | I/G/P/S 및 specialist가 충분한지 확인 |
| 본 연구 | 최대 100k distinct 관측 묶음 | 별도 확보량을 보고하는 audited labeled subset | 3 seeds, 최종 novel concept/조합, 기존 EO 유지 |
| instruction 확장 | 위 관측에서 최대 1M query records | 동일 mask에서 파생한 행은 같은 split/cluster | 문구·연산 다양성; 독립 관측 1M으로 부르지 않음 |

100k 모두에 전문가 dense mask가 있다는 가정은 없다. 데이터 장부에 `source_scene_count`, `unique_spatial_cells`, `observation_bundles`, `labeled_regions`, `concepts`, `queries`, `independent_eval_clusters`를 따로 기록한다. 같은 장면의 다른 질문을 더 많이 본 효과와 새 지역을 더 많이 본 효과는 분리한다.

학습용 AI는 검증된 사실을 질문·문장으로 바꾸거나 모호한 사례를 선별하는 용도다. 여러 AI의 합의를 지구 정답으로 쓰지 않는다. 최종 평가는 source annotation과 독립 사람 audit를 사용한다. teacher의 base/dev/test 접근과 prompt hash를 기록하며 test 정답을 생성 자료에 섞지 않는다.

## 8. 실험 설계: 가장 강한 반론부터 포함

### 8.1 다섯 학습 조건

| ID | 조건 | 답할 질문 |
|---|---|---|
| C | 같은 관측에서 v1.2 공식 목적 계속 사전학습; 공통 B 단계 | 더 오래 학습한 효과인가 |
| S | 공식 목적+같은 source facts의 일반 dense/직접 지도; 공통 B | 정답 mask를 더 준 효과인가 |
| I | 공식 목적+일반 EO–VLM 공동 instruction 학습 | 보통의 VLM finetuning이면 충분한가 |
| G | 같은 support/query/negative를 쓰는 generic episodic 개념 학습 | 일반 prototype/meta-learning이면 충분한가 |
| P | 제안한 공통 field·support/anchor 결합 학습 | 제안 학습 구조가 추가하는 효과가 있는가 |

P−I만으로 승리라고 하지 않는다. **P−G와 P−S, specialist+연산 비교**가 중요하다. C는 추가 감독이 없는 목적 비교이므로 P와 동일 감독이라고 부르지 않는다. S/I/G/P는 원 source facts와 negative exposure를 같게 하고, 표현 형식 차이는 기록한다. parameter 수, trainable 범위, 입력, EO/RGB token, replay, optimizer updates, validation 선택 횟수, FLOPs/GPU hours를 보고한다. 동일 step과 동일 FLOPs가 동시에 불가능하면 주 budget 기준을 사전에 정하고 보조 cost curve를 제시한다.

필수 외부/시스템 비교:

- 원본 encoder+같은 readout; frozen I/P와 EO-updated I/P로 A단계 EO 가중치 학습의 기여 확인. A/B 전체 동시 joint 학습이라는 뜻은 아니다.
- 같은 native EO와 K support를 쓴 강한 prototype/segmentation specialist → 동일 기하·검색 연산 → 동일 VLM. 학습 가능한 specialist의 튜닝/seed 비용도 장부에 넣는다.
- RGB-only 실제 VLM, 같은 센서를 충분히 렌더링한 VLM. 서로 입력 정보가 다르면 방법 우위의 primary 근거로 쓰지 않는다.
- 같은 예측 mask·숫자·anchor를 명시적으로 받는 VLM. oracle query mask는 상한 진단으로만 사용한다.
- TerraScope는 공통 입력/작업에서 공개 실행 경로를 검증한 뒤 비교한다. 다른 데이터의 논문 발표 점수를 직접 빼지 않는다.

### 8.2 실행 수를 숨기지 않는 장부

feasibility reviewer의 최초 C/S/I/P×3=12 제안에 generic G와 frozen interaction이 빠져 있었다. 원 리뷰는 보존하되 종합 계획은 아래로 수정한다.

| 묶음 | 계획 횟수 | 상태 |
|---|---:|---|
| C/S/I/G/P × 3 seeds | 15 | 본 학습 목표; 처리량/자료 검증 후 확정 |
| I-frozen/P-frozen × 3 | 6 | frozen/EO-updated 2×2 기여 감사; EO-updated는 위와 공유 |
| I/P의 작은 데이터 규모 × 3 | 6 | scaling 보조, 본 학습과 구별 |
| I/P에서 새 reader 적응 × 3 | 6 | EO checkpoint 재사용; 새 pretrain 6회가 아님 |
| specialist/원본 head/external baselines | 미정 | 튜닝·adapter·평가 비용 별도 측정 |

따라서 **핵심 21 training pipelines + 보조 12 adaptation/scale runs + baseline 비용**이다. 12회로 전체 논문이 끝난다고 예산을 축소하지 않는다. 이 전체가 7주 안에 가능하다는 실측은 없다. 처리량이 안 맞으면 최종 test를 보기 전에 규모와 주장을 줄이거나 다음 마감으로 옮긴다. generic baseline을 빼서 신규성이 있어 보이게 만들지 않는다.

### 8.3 데이터 분할과 새 개념

기본표는 `익숙한 개념/추가학습에서 감독을 제외한 개념 × 학습된 조합/감독하지 않은 개념-관계 조합`이다. 모든 primitive operation이 unseen인 zero-shot task라고 과장하지 않는다. K=0,5,20,100의 nested support를 가능한 개념에 사용하고, primary는 **K=20**으로 고정할 후보를 둔다. 충분한 지원 수가 없는 개념을 사후에 유리하게 제외하지 않는다.

K는 **서로 다른 source 관측 묶음에서 제공한 support 수**다. 묶음 하나에서 여러 질문/영역을 만들어 K를 늘리지 않는다. 양성 ceil(K/2)·대조 floor(K/2)를 기본 후보로 하여 K=20이면10+10이고, 같은 source footprint의 여러 날짜는 한 group으로 센다. 개념별 positive/contrast support와 독립 AOI 수가 충분한지 먼저 집계한다. K=0 alias 조건은 새 target의 의미가 주어지지 않는 대조이며 자연 이름을 제공하는 zero-shot과 구별한다. episode 내부의 operation withholding은 훈련법이고, 전체 학습에서 제외한 concept×operation 조합은 평가 조건이다.

원 scene·AOI·parcel/event·중복/overlap·연결된 모든 날짜와 파생 mask/수치/문장/질문을 묶는다. train/dev/locked-test concept 목록과 support/query 지리 분할을 manifest로 고정한다. alias를 써도 원 pretraining 지식 자체를 지우는 것은 아니다.

모든 파생 record는 `source_annotation_id`, `derivation_parent_ids`, `concept_id`, `operation`, `relation`, `anchor_id`, `scene_group`, `split`, `valid_support_geometry`, `label_timestamp`를 갖는다. test query에서 파생한 caption/area/ranking이 train이나 teacher prompt에 들어오면 차단한다. train의 mask로 학습되지 않은 관계를 계산할 수 있다는 사실은 별도로 인정하며 specialist 대조에 동일하게 허용한다.

### 8.4 평가와 통계

Primary 후보는 **새 개념×관계 조합의 검증된 gallery 검색 mAP, K=20**이다. query/positive relevance가 완전한 primary source에 한정한다. localization mIoU는 주요 secondary로 두고 둘을 임의 평균하지 않는다. 설명/면적은 같은 field에서 나오는 utility로 별도 보고한다. source가 이를 지원하지 못하면 dev 단계에서 primary protocol을 변경하고 변경 이유와 날짜를 기록한 후 동결한다.

여기서 primary ranking은 **실제 VLM이 자연어 요청·support·후보 관측을 받아 생성한 후보 ID 순서**다. 자연어의 정답 operation이나 요청에 제공되지 않은 oracle anchor를 추가로 넘겨 field를 합산한 순위를 main으로 대체하지 않는다. 사용자가 지정한 anchor는 아래와 같이 공통 입력으로 허용한다. 모든 모델에 같은 gallery/후보 입력 예산·순서 무작위화·출력 schema를 적용하고, 형식 오류·빠진 ID·중복·존재하지 않는 ID의 채점 규칙을 dev에서 동결한다. 생성을 실패한 사례를 평가에서 빼지 않는다. 자연어 paraphrase와 복합 조건도 source reference에 대응되는 범위에서 평가한다.

세 경로를 함께 보고한다: **(a) 정답 structured query + EO field/연산**은 시각 처리 상한/분해 진단, **(b) 동일 VLM parser + specialist/연산**은 강한 시스템 대조, **(c) 전체 EO–VLM 경로**는 main이다. 세 경로에 같은 관측·support·후보를 주고 언어 해석 오류, 개념 field 오류, grounding 오류를 분리한다. VLM 단순 문장화 결과만 좋아지면 실제 VLM 능력 확대라고 말하지 않는다.

primary gallery는 label로 목표를 잘라낸 crop 대신 관측 격자/고정 window로 구성하고, relevance만 숨겨 둔 reference에서 만든다. target의 GT parcel 경계는 query 입력에 제공하지 않는다. 사용자가 지정한 anchor polygon은 허용된 입력이며 모든 대조에 동일하게 준다. 모델이 anchor도 찾아야 하는 실험은 별도 조건으로 둔다. 보조 parcel-candidate 평가가 GT 경계를 제공하면 그 사실을 표시하고 localization 성능으로 합치지 않는다. gallery 크기·동시입력/분할ranking 방식·distractor·score tie 규칙은 실행 전 정해야 한다.

candidate pool을 먼저 줄여서 VLM에 주면 그 selection recall과 VLM reranking을 따로 기록한다. 제한된 후보 재정렬 점수를 지구 전체 검색 성능으로 해석하지 않는다. 최종 contrast에서는 모든 방법이 같은 후보와 context 예산을 받는다.

개념과 지리 cluster 단위 macro 평균 및 paired interval을 보고한다. 질문 수를 독립 n으로 계산하지 않는다. 독립 광역 tile 수가 너무 적으면 불안정한 bootstrap CI로 지역 일반화의 유의성을 포장하지 않고 tile별 전부를 보이며 주장을 제한한다. seed별 변화와 test cluster 불확실성은 별개다.

PASTIS 4개 광역 tile만으로 새로운 모든 지역에 대한 CI를 만들지 않는다. tile를 고정한 조건부 하위 spatial-cluster 분석과 tile별 전부를 보고하는 분석을 구별한다. 광역 일반화를 확증하려면 별도의 충분한 독립 지역이 필요하며, 확보 전에는 아래95%구간 문턱을 넓은 지역 일반화의 통과 판정에 사용하지 않는다. 타일 내부 buffer/cluster 크기·개념별 지원 수·gallery relevance는 10/10 이전 dev에서 확정한다.

support shuffle/empty, wrong concept, anchor swap, query swap, EO zero, sensor ablation을 진단한다. intervention의 기대 변화는 reference로 정한다. 불변이어야 할 사례와 답이 바뀌어야 할 사례를 섞어 무조건 flip을 성공으로 세지 않는다.

공식 EO 유지 평가는 source protocol에 맞춘 Sen1Floods11·m-cashew-plant에서 원본과 학습 후를 비교한다. fresh connector/저용량 head 및 두 번째 reader로 재사용성을 검사한다. 두 번째 reader 후보 InternVL3-8B는 TerraScope backbone 대응에 유용하지만 Qwen 계열 언어 backbone을 공유하므로 독립 LLM 계열 일반화라고 부르지 않는다.

## 9. 사람 100만 원은 어디에 쓰는가

기존 [예산](../config/eo_vlm_labeling_budget_v2_20260926.json)을 유지한다. 검수자60인시 72만 원, 전문가4인시 16만 원, API상한5만 원, 예비7만 원이다. 실제 인력/단가는 아직 확정하지 않았다. GPU와 개발 노동은 이 금액에 포함되지 않는다.

목표 train200/dev60/test120은 timed pilot 전의 가정이다. 독립 test120을 두 사람이 각8분 판독하면32인시이므로 복잡한 작업에서는 빠듯하다. 처음20건의 실제 시간을 재서 수량을 조정한다. 120건은 source label의 의미·관측 가능성·설명 오류 audit이며 작은3%p 효과를 검증하는 대규모 gold가 아니다. 공개 참조 라벨의 충분한 평가를 주 결과로 하고, 사람은 모델명을 숨긴 공통 reference를 만든다. 모델별 우열을 본 뒤 미해결 사례를 버리지 않는다.

AI 여러 개의 비교는 train 후보 선정과 오류 발견에 도움이 될 수 있다. API 호출 비용을 사전 계측하고 상한을 적용한다. 이번에는 유료 호출·채용·발주를 하지 않았다.

## 10. 진행 문턱과 실패했을 때의 결정

아래 수치는 **투자 판단 후보**이며 CVPR 채택 기준이 아니다. label 오류·개발 분할·처리량·표본력 검토 후 final test 전에 동결한다. test에 맞춰 낮추지 않는다.

| 단계 | 필요한 증거 | 안 되면 |
|---|---|---|
| 데이터/구현 | source 정합, v1.2 loader parity, RGB 동작 보존, EO gradient, 모든 arm의 동일 record/budget audit | 본 학습 보류, 결함 수정 |
| 방법 반증 | dev에서 P가 I/G 및 specialist 대조보다 실제 이득; support/anchor를 올바르게 사용 | 100k 자동 확대 중단; 원인 가설 수정 |
| 본 주장 | primary mAP P−G ≥.03 후보 및 cluster-aware95% 구간 하한>0; S/I/specialist 대조의 사전 지정 비교도 공개 | 차이가 없으면 새 재사용 학습법 주장을 닫음 |
| 기존 EO | metric별 사전 지정 non-inferiority; 예: −.01 margin 후보 | replay/학습 범위를 dev에서 조정; “유지” 자동 선언 금지 |
| 재사용 | 새 reader/connector 또는 저용량 readout에서 EO 학습 이득 확인 | 공동적응 시스템의 가치와 encoder 일반화 주장을 분리 |

main 비교의 검정 순서·보조 비교 다중성은 prereg에 고정한다. novel concept의 수가 너무 적거나 spatial units가 부족하면 .03과 CI 조건을 형식적으로 만족시켜도 넓은 일반화를 주장할 수 없다.

**specialist 대조도 본 주장 문턱에 포함한다.** dev에서 고른 가장 강한 specialist+동일 VLM 조합을 잠그고, 최종 P와의 primary 차이를 같은 cluster 평가로 보고한다. G를 이겨도 specialist와 동등하면 새로운 시각적 재사용 능력을 확보했다고 판정하지 않는다. 후보 confirmatory 순서는 P−G의 실용효과+양의구간 → P−S → P−specialist이며, 마지막 두 차이의 양의 구간 및 다중비교 처리는 test 전에 확정한다. 서로 다른 metric의 유의성 하나만 골라 성공을 선언하지 않는다.

성과별 논문 범위:

- **강한 결과:** P가 generic meta와 specialist를 넘고, 여러 개념/도메인·새 reader·EO 유지에서 반복된다 → OlmoEarth 기반 모델의 재사용 능력을 높이는 방법 논문으로 CVPR 본회의 목표.
- **중간 결과:** 한 domain/reader에서만 이득 → 범용 기반모델 주장 보류, 제한 범위와 추가 확증 필요.
- **시스템 결과:** 같은 mask+연산 baseline과 동등 → 유용한 engineering toolkit일 수 있지만 이를 새 시각 일반화 방법이라고 포장하지 않음.
- **무효 결과:** 누출·추가정보·관측 불가능 gold가 원인 → 성능 주장 철회하고 자료/평가를 수정. 기존 실패 기록 보존.

## 11. 일정과 컴퓨트

공식 CVPR 2027은 등록 **2026-11-10 AoE**, 본문 **11-16 AoE**, supplement **11-23 AoE**다. [공식 일정](https://cvpr.thecvf.com/Conferences/2027/Dates), [CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers). 현재 약7주다. 다음은 GPU 처리량을 재기 전 조건부 일정이다.

| 기간 | 완료물/결정 |
|---|---|
| 9/27–10/3 | v1.2 원본 기준선 실행 경로, 공식1k data contract, supervised source audit, 100–200step/recipe 처리량, split 설계 |
| 10/4–10/10 | 10k 이하 반증 비교 I/G/P/S와 specialist, 독립 구현 감사; 본 학습 진행 결정 |
| 10/11–10/25 | 동결 recipe·data/split·primary endpoint, 본 비교 및 EO 유지; human audit 병행 |
| 10/26–11/5 | reader/scale/ablation, 실패 분석, locked test 단일 확증, 분석 완료 |
| 11/6–11/16 | 원고·도표·재현 코드·독립 사실 감사, 등록/제출은 별도 사용자 단계 |

각 recipe에서 representative timesteps/해상도로 warm-up 후100–200 optimizer steps를 측정한다. median/p90 sec/step, 유효sample/s, peak allocated/reserved VRAM, IO/checkpoint/eval 시간을 기록한다. `ceil(N×epochs/effective_batch)×sec_per_step/3600`에 validation/IO를 더해 wall time, 여기에 world size를 곱해 GPU hours를 계산한다. 2 GPU 효율을 단일 GPU의 정확히2배로 가정하지 않는다.

**10/10 본 학습 결정은 방법 효과뿐 아니라 연산 예산 관문이다.** A의 C/S/I/G/P와 B, EO 평가·specialist·reader 적응·checkpoint/IO·실패 재실행 여유까지 실제 시간표에 넣는다. 10/11–25 창에 확증 실험이 들어가고11월 분석/집필 시간이 남을 때만 해당 규모를 동결한다. 현재21+12가 이 창에 맞는다는 근거는 없다.

GPU 점유는 실행 직전 확인하고 타 작업을 중단하지 않는다. 처리량상 seed·대조·집필 시간을 확보할 수 없으면 1M instruction/temporal extension/scaling 부가 실험부터 후속으로 이동한다. 결정적 generic baseline과 최종 검증을 빼고 규모만 키우지 않는다. 마감이 불가능하면 다음 학회로 이관하는 것이 현재 결과를 과장하는 것보다 연구 목표에 맞다.

## 12. 다음 실행을 구체적으로 준비하는 순서

1. v1.2-Base의 pin된 weights/config를 받는 스크립트와 official loader bridge를 구현한다. 고정 입력 parity 후 두 EO 기준선의 dev부터 실행한다.
2. 공식1k H5 subset을 서버 연구 output root에 받고 archive/source metadata를 보존한다. 전체 archive의 지역·modality·지도 target completeness를 감사한다. 현재17.9GB URL 확인은 download 완료가 아니다.
3. PASTIS 원자료/참조의 완전성·10밴드 처리·CRS·parcel/time 분할을 확인한다. 전문가 지원 개념을 고르기 전에 class별 독립지역/관측 가능성/지원 수를 집계한다.
4. immutable source annotation에서 episode manifest와 derivation graph를 만든다. synthetic geometry 계산 검사는 코드 검증이고 실제 지구 정답 검증과 구별한다.
5. actual VLM adapter, generic G, specialist pipeline부터 구현하고 P를 같은 경로에 넣는다. 원 RGB 생성, EO gradient·저장 복원, 지원/anchor intervention을 검사한다.
6. 개발용 bounded run으로 처리량과 반증 결과를 얻어 final prereg를 작성한다. 실패해도 original data/test를 재활용해 유리한 숫자를 찾지 않는다.

현재의 사용자 지시는 상세 계획과 검토를 승인한 것이다. 이전 GPU 사용 지시는 유지되지만 본 문서 자체가 미완성 large-run config를 실행시키지는 않는다. 새로운 공개 게시·유료 호출·발주는 이번 실행에 포함하지 않는다.

## 13. 독립 검토 기록

원 리뷰와 출처는 `artifacts/cvpr_plan_review_20260927/`에 저장한다. 리뷰어 세 명은 각각 선행·방법·실행 가능성을 검토했다. 원 초안 리뷰는 이전 capability plan과 이 계획의 구상에 대한 것이므로, 이 문서의 모든 문장을 검증했다고 하지 않는다. 후속 draft review를 별도로 기록한다.

세 가지 핵심 변경은 (1) mask의 파생 출력을 독립 전이로 세지 않기, (2) generic meta와 specialist+연산 대조 추가, (3) 최신 v1.2·실제 학습 비용·자료 상태 명시다. 채택 확률과 agent 투표를 혼동하지 않는다.


후속 설계: [지역·계절 맥락 학습 검토](REGIONAL_CONTEXT_LEARNING_20260927.md). Wikipedia/기후와 EO를 함께 학습한 선행을 반영한 후보 설계이며, 이 문서의 실행·판정은 변경하지 않는다.
