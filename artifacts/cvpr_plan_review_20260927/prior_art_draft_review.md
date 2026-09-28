# CVPR 종합 초안 후속 검토 — 선행·신규성

검토자: `/root/cvpr_prior_art_redteam`. 검토일: 2026-09-27.
검토 파일: `CVPR_2027_OLMOEARTH_RESEARCH_PLAN_20260927.md`.
범위: 초안 전체를 읽고 선행·기여·주장 수준을 검토했다. 원 리뷰는 변경하지 않았다. 데이터·코드·실험을 재현한 검토는 아니다.

## 판정

**실행 준비와 제한된 반증 파일럿 진행에 동의한다. 본 방법의 신규성 확정과 대규모 학습 착수에는 아직 동의하지 않는다.** 초안은 기존 결과를 진단으로 한정하고, v1.2 최신성·generic 대조·mask 파생 출력·21개 핵심 pipeline 비용을 명시했다. 큰 응용 모델 목표와 현재 결과 없음, 조건부 투자 판단을 정직하게 구별하고 있다. CVPR 채택 확률 50% 이상이라는 주장은 하지 않아 적절하다.

## 수정해야 할 두 가지 핵심 문제

### 1. 수식이 아직 일반적인 prototype/episodic learning과 구별되지 않는다

§5의 `c=H(support)`, `p=sigmoid(q(Z,c))`와 contrastive binding loss는 넓은 조건부 분할/메트릭 학습 형태다. `s(e)`가 무엇을 scoring하는지, `L_query`가 일반 support-query 감독과 어떻게 다른지, G와 P의 차이가 어느 computation/gradient 경로인지 정의되어 있지 않다. 현재 상태에서 loss 이름과 공통 field를 방법적 신규성으로 읽으면 과장이다.

**필요 수정:** §5에 다음을 명시한다.

> 현재 수식은 구현 골격이며 신규한 최적화 원리를 규정하지 않는다. 첫 설계 완료물은 G와 P의 데이터·순전파·손실·gradient 차이를 한 표와 pseudocode로 고정하는 것이다. 일반 prototype+같은 contrastive negative+같은 operation withholding으로 P를 재현할 수 있으면 별도 새 학습법 주장을 하지 않는다.

G가 일반적으로 작동하기 어렵게 일부 정보를 빼고 P에만 `binding negative`나 anchor에 대한 더 좋은 정답을 줘서는 안 된다. 이미 같은 negative exposure를 명시했으므로 이 원칙을 구현 감사로 연결하면 된다. EO band/mask 처리를 추가하더라도 현재 v1.2의 band dropout 재사용과 새 메커니즘을 구별해야 한다.

### 2. 주평가가 실제 VLM의 필요성을 검증하지 못할 수 있다

§8.4의 새로운 개념×관계 검색 mAP는 유효한 EO 평가 후보다. 그러나 query가 정형 relation/anchor로 주어지고 mask+기하 연산으로 검색할 수 있으면, 실제 VLM은 결과를 문장화하는 부품으로만 남을 수 있다. 초안은 specialist 대조를 넣어 이 위험을 인정했지만, **무엇을 보여야 전체 모델을 EO–VLM 연구라고 부를지**는 더 명확해야 한다.

**필요 수정:** structured-query primary를 유지하더라도 자연어/예시에서 그 query로 넘어가는 실제 모델 경로를 별도 평가한다. 사전 고정한 동의 표현과 복합 지시, 부정/대조 지원, 지원과 질문이 충돌하는 경우의 처리를 포함하고, (a) oracle structured query, (b) 같은 VLM parser+specialist, (c) 전체 joint 모델을 비교한다. 이는 기존 VLM보다 단지 문장이 유창해졌는지를 평가하려는 것이 아니라 실제로 EO 근거와 사용자 지시를 함께 활용했는지 확인하기 위한 대조다.

VLM을 제거해도 주 결과가 그대로라면 EO few-shot representation 연구로서 가치는 남는다. 다만 그 결과를 전체 EO–VLM 공동학습의 우위로 부르지 않는다. 반대로 전체 모델이 좋아지고 encoder-only가 비슷한 경우를 자동 실패로 보지 않는 현재 문구는 적절하다.

## 권장 보완

- PASTIS/MADOS에서 목표 개념과 연산이 충분히 정의되는지 확인하기 전에는 “새 작물 20개 예시로 여러 업무”를 보장하지 않는다. 현재 초안이 사용 시나리오로 표시한 것은 적절하다. K가 label region 수인지 support scene 수인지 정하고 중첩 parcel·같은 시계열의 중복 효과를 장부에 기록한다.
- 무작위 alias는 언어 이름의 힌트를 줄이는 통제이며 foundation model이 새로운 시각 개념을 처음 학습했다는 증거가 아니다. 이미 명시되어 있으므로 유지한다.
- P−G ≥.03이라는 투자 문턱은 임의의 후보이고 표본력/실질 효용 근거가 아직 없다. 현재 '후보, 사전 동결 전' 표기가 있어 치명적 과장은 아니다. 실제 variance를 본 뒤 dev에서만 확정하고 test를 보고 조정하지 않는다.
- “일부 작업 감독을 숨긴다”의 단위는 episode 내부 withholding과 전체 학습 corpus에서의 concept×operation holdout을 구별한다. 전자는 일반화 훈련, 후자는 평가 조건이다. 두 표현이 혼동되면 학습에서 이미 본 조합을 새 조합으로 오인하게 된다.

## 현행성·문헌 판정

TerraScope v1, SPEX v2, MS-CLIP v3, OlmoEarth v1.2 report v3의 표기는 직접 확인한 primary source와 맞는다. 수치 성능을 논문 간 직접 비교하지 않은 것도 적절하다. v1.2-Base의 실제 checkpoint revision과 공식 recipe 호환성은 문헌 검토가 아닌 loader 감사로 확인해야 한다. SegGPT/SEEM/RSCoVLM까지 대조 후보에 넣어 단순 조합의 신규성 위험을 충분히 인정했다.

**진행 범위 동의:** 원본 EO 평가·공식 자료 감사·v1.2 loader·generic G/specialist 기준선·실제 VLM 인터페이스 구현 및 소규모 처리량/반증 실험. **유보:** 제안 메커니즘이 동결되지 않은 상태의 100k 본 학습, CVPR-ready 또는 확률 보증. 큰 목표를 축소할 필요는 없으며, 위 두 식별 문제를 해결하는 것이 큰 모델 연구를 성립시키는 다음 작업이다.
