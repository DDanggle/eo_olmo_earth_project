# MS-112 · CVPR novelty · Korea shared-cache 감사 (2026-09-06)

## 결론

현재 연구는 **버릴 상태가 아니다.** 그러나 MS-112를 `selector 통과`로 쓰면 계산과 실험 단위에서
즉시 공격받는다. 정확한 판정은 다음과 같다.

- **이미 강한 결과**: OlmoEarth cache의 two-task reuse/few-shot utility와, 아무 Earth FM cache나
  같은 가치를 갖지는 않는다는 경계 관측.
- **MS-112의 역할**: support 구성·cache contract를 교차하면 행동 이질성이 생길 수 있다는
  **가설 생성용 개발 분석**.
- **아직 없는 결과**: 처음 보는 과업과 처음 보는 encoder family에서 query label 없이 행동을
  고르고 best-static보다 regret–cost를 개선한 selector.
- **CVPR 후보 주장**: “어느 모델이 좋은가”가 아니라 **이미 물질화된 Earth representation을
  reuse / head-adapt / raw / re-embed / abstain 중 어디에 배정할 것인가**.
- **Korea 3-task의 최적 역할**: 같은 물리 cache를 세 과업이 공유할 때 label efficiency와 실제
  amortization이 생기는지 보여주는 외부 systems/science case. 일반 selector의 독립 3과업으로
  세면 안 된다.

즉, 지금 제목을 붙인다면 다음이 가장 정확하다.

> **EarthCache: Deciding When to Reuse, Adapt, or Recompute Geospatial Representations**

## 1. MS-112 수치 재감사

### 1.1 실제 입력이 말하는 것

`artifacts/g0_dev/g0_dev_input_v3.json`의 episode 평균을 직접 재집계하면 다음과 같다.

| 개발 episode | CACHED | HEAD_ADAPT | RAW | 실제 승자 |
|---|---:|---:|---:|---|
| Sen12, random K=5 | .2578 | **.2885** | .1690 | HEAD_ADAPT |
| Sen12, stratified K=5 | .2578 | **.2937** | .1790 | HEAD_ADAPT |
| Solar, random K=5 | **.5907** | .4262 | .2598 | CACHED |
| Solar, stratified K=5 | **.5907** | .5816 | .2396 | CACHED |
| Sen12, Galileo cache | .1529 | 미측정 | **.1966** | RAW (잠정) |
| Sen12, Clay-256 cache | .1951 | 미측정 | **.1966** | 사실상 동률, 단일 seed |
| Sen12, OlmoEarth-base cache | **.2722** | 미측정 | .1966 | CACHED (잠정) |

위 표는 흥미롭지만, 아래 이유로 G0 통과 표는 아니다.

### 1.2 발견된 계산·설계 결함

1. **행동 교집합 버그**: 계산기는 모든 episode에 공통으로 존재하는 행동만 남겼다. cache-contract
   세 episode에 `HEAD_ADAPT`가 없으므로, .085 headroom은 실제로 CACHED 대 RAW 두 행동만의
   값이다. “승자 3종”과 같은 계산 결과가 아니다.
2. **비직사각 행렬**: `HEAD_ADAPT` 미측정은 사전 정의된 action ineligibility가 아니다. 결과를
   보지 않은 빈칸이다. 빈칸을 후보 제외로 바꾸면 oracle이 달라진다.
3. **seed 비대칭**: support episode는 3 seed지만 cache-contract episode의 cache와 raw 행은
   1 seed다. Clay–raw 차이 .00145를 승패로 부를 수 없다.
4. **독립성 과장**: 일곱 episode는 일곱 독립 과업이 아니다. 네 support episode는 두 데이터셋을
   반복하고, 세 cache episode는 동일 Sen12 fold/label/raw 결과를 반복한다.
5. **feature가 입력에 없음**: 규칙은 `support_positive_count`를 사용한다고 서술하지만 v3 행에는
   실제 count가 없고 `random/stratified` 라벨만 있다. 더구나 현재 episode 값은 8지역을 접은 값이라
   배포별 support 품질을 표현하지 않는다.
6. **family lookup의 누출 위험**: “등록된 cache-worthiness table에서 낮은 family면 RAW”는 개발
   결과를 외운 lookup baseline이다. leave-one-family-out에서 사용할 selector feature가 아니다.
7. **비용 미완성**: few-shot 일부 외 비용은 추정/반올림 값이다. .085는 아직 실제 운영 utility가
   아니다.
8. **no-raw-read headroom=0은 증거가 아님**: 비교 후보가 사실상 CACHED 하나만 남은 결과라
   구조적으로 0이다.

따라서 MS-112의 정정 문장은 이것이다.

> 개발 결과를 deployment 조건별로 재배열하자 support 구성과 cache family가 행동 이질성의 후보로
> 나타났다. 그러나 행렬 누락·seed 비대칭·상관 episode 때문에 selector 필요조건 통과로 해석할 수
> 없으며, 외부 직사각 action matrix에서 재검증해야 한다.

### 1.3 코드 조치

`code/geobench_action_headroom.py`에 `required_actions`와 `required_seed_count` 계약을 추가했다.
앞으로 선언된 행동 또는 seed가 하나라도 없으면 상태는
`INCOMPLETE_ACTION_MATRIX_DIAGNOSTIC_ONLY`이고 G0-A/G0-B/G0는 fail-closed다. 기존 v3 입력에
3행동×3seed를 요구해 재실행하면 cache-contract 세 episode의 HEAD_ADAPT가 없고 각 행동 seed가
1개여서 G0=False다.

## 2. 최근 연구와 겹치는 부분

| 이미 점유된 질문 | 대표 최근 연구 | EarthCache가 그대로 주장하면 생기는 문제 |
|---|---|---|
| 새 과업에 좋은 RSFM 예측 | [Capabilities Encoding, CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/MORSE/html/Adorni_Towards_Efficient_Benchmarking_of_Foundation_Models_in_Remote_Sensing_A_CVPRW_2025_paper.html) | generic model selector는 신규성이 약함 |
| modality·task·제약으로 모델 추천 | [REMSA, 2025](https://arxiv.org/abs/2511.17442) | metadata-only 규칙은 이미 강한 baseline이 있음 |
| 정확도·자원 기반 RS workflow 선택 | [Accuracy versus Efficiency, WACVW 2026](https://openaccess.thecvf.com/content/WACV2026W/CV4EO/html/Iseni_Accuracy_versus_Efficiency_in_Model_Selection_for_Remote_Sensing_Scientific_WACVW_2026_paper.html) | cost-aware model recommendation만으로는 겹침 |
| cross-domain PEFT 모듈 선택 | [CrossEarth-Gate, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_CrossEarth-Gate_Fisher-Guided_Adaptive_Tuning_Engine_for_Efficient_Adaptation_of_Cross-Domain_CVPR_2026_paper.html) | adapt module selector로 가면 정면 경쟁 |
| GeoFM PEFT 비교 | [Fine-tune Smarter, Not Harder, 2025](https://arxiv.org/abs/2504.17397) | LoRA/PEFT 자체를 novelty로 삼을 수 없음 |
| 여러 task에서 중간 embedding 재활용 | [Embedding Recycling, EACL Findings 2023](https://aclanthology.org/2023.findings-eacl.145/) | cache+adapter 자체는 EO 밖에서도 선행됨 |
| 광범위 GeoFM capability benchmark | [GEO-Bench-2, 2025/26](https://arxiv.org/abs/2511.15658) | “한 모델이 항상 최고가 아니다”는 이미 알려짐 |

2026년의 중요한 평가 기준도 더 엄격하다. [No One Knows the State of the Art in Geospatial
Foundation Models](https://arxiv.org/abs/2605.12678)는 152편을 감사해 동일 모델·benchmark·protocol의
대규모 불일치, 설정 파편화, weight 미공개를 지적하고 공통 harness·variance·data/architecture 통제를
요구한다. 따라서 seed 1 결과, 서로 다른 decoder/계약, copied와 rerun 혼합을 한 표에 놓으면 현재
리뷰 기준에서 특히 취약하다.

## 3. 남는 노벨티

### 3.1 핵심 차이

선행 model selection은 대체로 다음을 묻는다.

> “새 과업에 어느 pretrained model을 선택할까?”

EarthCache가 물어야 하는 것은 다르다.

> “raw cube를 다시 읽고 encoder를 다시 돌리기 전에, 이미 가진 cache·head·release contract와
> K개 support만으로 다음 운영 행동을 고를 수 있는가?”

이 차이가 실제 논문 기여가 되려면 다음 네 요소가 모두 있어야 한다.

1. 같은 episode의 **counterfactual action outcomes**: CACHED / ADAPT / RAW / REEMBED.
2. 정확도뿐 아니라 encoder, raw-I/O, cache storage/invalidation의 **실측 lifecycle cost**.
3. query label 없이 **held-out task와 held-out encoder family**에 일반화.
4. 확신이 없으면 `REQUEST_MORE_LABELS`로 빠지는 **selective policy**.

### 3.2 CVPR 판정

| 상태 | 현재 판정 |
|---|---|
| two-task cache utility/few-shot finding | 강한 empirical finding |
| cache family/scale 경계 | 유망하지만 주로 Sen12·seed 1 exploratory |
| MS-112 selector | 불성립, 개발 가설만 |
| corrected external selector | 설계상 CVPR 후보 |
| version bridge | 유용한 engineering appendix, 단독 novelty 약함 |
| Korea 3-task | 강한 external/system figure 후보, 단독 generality는 부족 |

냉정한 확률 언어로 말하면, **현재 결과만으로는 CVPR main보다 강한 workshop/TGRS empirical
study에 가깝다.** 반면 외부 task×family 직사각 행렬에서 best-static regret를 줄이고 실제 비용
frontier까지 통과하면 CVPR main의 method/systems story로 올라갈 수 있다. 새 MoE나 LoRA를 붙이는
것보다 이 검증이 훨씬 중요하다.

## 4. 수정 실험 설계

기계 판독 계약은 `config/geobench_cache_action_prereg_v1.json`이다.

### S — support decision

- 고정: OlmoEarth-base cache와 입력 계약.
- 교차: task × held-out target group × random support draw × K=5/20.
- 행동: CACHED_HEAD / HEAD_ADAPT / RAW_FINETUNE, 전부 3 seed.
- Z0와 K-track을 분리한다. Z0에서 positive count를 쓰면 label-free가 아니다.
- primary unit은 task이며 support draw와 region은 nested다.

### F — cache-family decision

- 교차: task × held-out group × cache family × K=5/20.
- 동일한 3행동을 모든 cell에 채운다.
- OlmoEarth-base/nano/Galileo부터 시작하되, encoder family ID lookup은 baseline에서만 쓴다.
- leave-one-family-out에서 보지 못한 family의 결과를 예측해야 method로 인정한다.

### R — release lifecycle

- identity reuse / bridge / re-embed를 같은 task와 raw cube에서 비교한다.
- S/F와 섞지 않는다. 현재 version mismatch는 “문제 존재”와 bridge 한계를 보여주지만,
  selector novelty를 대신하지 않는다.

### 실행 순서

1. DynamicEarthNet의 cache를 **단일 writer + atomic save + 전수 audit**으로 닫는다.
2. 각 external task의 chip/time/head/metric/anchor/cost harness를 outcome 전에 freeze한다.
3. S 행렬부터 CACHED/ADAPT/RAW를 3 seed로 채운다.
4. 독립 task-level robust winner가 둘 이상으로 갈리는지 G0를 재평가한다.
5. G0가 통과할 때만 factorized rule 또는 작은 ranker를 학습한다.
6. F 행렬과 leave-one-family-out을 연다.
7. 마지막에 Korea를 외부 shared-cache case로 한 번 개봉한다.

## 5. Korea shared-cache 3-task의 의미

### 5.1 왜 유의미한가

Korea는 세 개의 별도 데이터셋이 아니라 **같은 cube와 같은 cache에 세 label ontology가 붙은
환경**이다. 그래서 cross-dataset 비교에서 섞이는 pixel, GSD, acquisition, preprocessing 차이를
고정하고 다음 질문을 깨끗하게 볼 수 있다.

- land cover, deforestation, landslide가 같은 representation을 실제로 공유할 수 있는가?
- K=5/20에서 cache head와 raw model 중 어느 쪽이 label-efficient한가?
- encoder를 한 번 돌리고 task head만 추가할 때 N=1→2→3의 실제 비용 손익분기가 어디인가?
- 같은 pixel/cache인데 support composition과 task가 달라지면 행동이 바뀌는가?

이것은 EarthCache의 **shared infrastructure claim**을 가장 잘 보여주는 실험이다. 특히 AI-Hub
12-band materialization은 한국 운영 데이터 onboarding이라는 Ai2 취업 포트폴리오 축에도 직접
연결된다.

### 5.2 아직 열면 안 되는 이유

현재 확인된 상태는 2,536 cube materialized, 163 excluded(6.0%), materialized file의
shape/dtype/common-coverage fail 0이다. 그러나 transient error 6건 재시도와 exclusion selection-bias
감사가 남아 `experiment_eligible`이 아니다. 파일 무결성과 과학적 적격성을 혼동하면 안 된다.

또 v0 설계에는 두 문제가 있었다.

- T1/T2는 source head가 없으므로 T3와 같은 O0/R0 transfer 비교가 정의되지 않는다.
- rare positive가 없는 test cluster에 positive IoU 5/7 규칙을 강제하면 통계 단위가 잘못된다.

이를 `config/korea_shared_cache_3task_prereg_v1_amendment.json`에서 수정했다. T1/T2는 동일 K로
`CACHE_K vs RAW_K`, T3만 source transfer+adaptation을 본다. rare task는 all-cluster AP/FP-matched
metric과 positive-query-cluster conditional IoU를 분리한다.

### 5.3 Korea가 증명하지 못하는 것

- 세 task가 cube를 공유하므로 **세 독립 재현**이 아니다.
- Korea 하나로 general selector를 검증할 수 없다.
- label을 개봉한 뒤 selector 규칙이나 chip 계약을 고치면 external test가 아니다.
- 12-band sensitivity와 Sen12에 맞춘 10-band primary view를 섞어 같은 계약이라고 부를 수 없다.

## 6. 성공/실패 판정표

| Gate | 통과 기준 | 실패하면 |
|---|---|---|
| Matrix | 모든 required action×3 seed 완결 | selector headroom 보고 금지 |
| Heterogeneity | 독립 held-out task ≥2에서 robust 승자가 다름 | learned selector 중단 |
| Value | oracle headroom ≥.02, fixed anchor·실측 cost | characterization으로 후퇴 |
| Selector | best-static 대비 regret ≥30% 감소, task-group CI | rule은 baseline으로만 보존 |
| Family | unseen encoder family에서 방향 유지 | family-specific policy로 축소 |
| Korea readiness | error 6·selection bias·seal 모두 종결 | label 미개봉 유지 |
| Korea value | ≥2/3 task 방향+비용 손익분기 | 공유 cache의 한계로 보고 |

## 최종 교수 관점 권고

좋은 아이디어는 “에피소드를 늘려 selector를 살린다”가 아니다. **결정을 실제로 바꾸는 관측 가능한
상태 변수를 분리하고, 처음 보는 task/family에서 그 결정의 regret를 줄이는 것**이다. MS-112는 그
변수 후보를 발견했지만 검증은 하지 못했다.

따라서 다음 GPU를 더 많은 모델 탐색에 쓰지 않는다. 우선 DynamicEarthNet에서 동일 행동 3개를
동일 seed·head·anchor·cost 계약으로 채운다. 그 결과가 개발 lookup 없이 재현될 때만 selector로
간다. Korea 3-task는 그 뒤 **One cube, many tasks**라는 가장 이해하기 쉬운 외부 그림으로 연다.
