# 종합 초안 후속 독립 방법 리뷰 — 2026-09-27

검토 파일: `CVPR_2027_OLMOEARTH_RESEARCH_PLAN_20260927.md` 현재 초안. 원 `method_review.md`는 보존한다. 새 실험 결과나 가중치는 검사하지 않았다.

**진행 판정:** source/loader/실제 VLM interface 검증 및 작은 반증 pilot 준비에는 진행 가능. **현재 문서를 동결된 학습법 사전등록 또는 100k 본 학습 실행 명세로 사용하기에는 아직 부족하다.** 비용 장부의 핵심 21+보조 12+baseline, feed-forward support와 finetuning의 구별, frozen/joint 2×2, 두 번째 domain의 불완전성 명시는 적절하다. 이론상 계산 가능한 구조와 기존 방법보다 새롭고 유효한 학습법은 별개다.

## 실행 전에 해결할 결정적 사항

### 1. P와 G의 차이가 현재 수식으로는 식별되지 않는다 — §5, §8.1

`H(support) → shared field → operation`은 generic prototype/episodic learner도 구현할 수 있다. 초안은 G에도 같은 support/query/negative를 제공하므로, `s(e)`와 `L_query` 및 G의 학습 목적을 정의하지 않으면 사실상 같은 알고리즘을 다른 이름으로 비교하거나 약한 G를 만들 위험이 있다. inner-loop가 없는 선택 자체는 문제없다.

**수정:** G의 forward/목적/episode sampler를 먼저 고정하고, P가 바꾸는 단일 핵심을 코드 수준으로 적는다. 예를 들어 동일 shared-field G의 ordinary episodic supervision을 기준으로 P가 어떤 cross-operation pairing/consistency를 강제하는지 명시한다. G도 충분히 강한 shared-field variant를 포함해야 한다. 차이가 contrastive objective 하나라면 바로 그 범위로 신규성을 제한한다. `L_field`, `L_binding`, `L_query`가 같은 source label을 어떻게 중복 사용하는지와 정규화도 적는다. 구조적 차이를 아직 선택하지 못하면 P를 후보 상태로 유지하고 본 학습 시작을 보류한다.

### 2. Primary mAP가 실제 VLM을 평가하는지 불분명하다 — §4, §5, §8.4

field를 고정 기하 연산으로 걸러 순위를 계산하면 mAP는 EO binder/specialist의 점수다. VLM이 끝에서 설명만 만드는 경우 8B를 사용해도 primary 결과가 actual VLM 성능을 증명하지 않는다. 이는 잘못된 시스템은 아니지만 논문의 주체를 다르게 설명해야 한다.

**수정:** 각 method의 primary relevance score가 어느 tensor/최종 출력에서 나오는지 지정한다. 실제 VLM을 main으로 유지하려면 자연어 질문·support에서 최종 후보/영역을 선택하는 end-to-end 출력을 같은 인터페이스로 채점하고, oracle operation/anchor를 주는 field-only 점수를 보조로 둔다. 반대로 field score를 primary로 채택한다면 중심 주장은 EO concept learner이며 VLM은 설명·질의 인터페이스라고 명시한다. 인위적으로 VLM을 꼭 거치게 만드는 것 자체가 기여는 아니다. 두 수준의 점수를 섞으면 안 된다.

### 3. '추가학습에서 감독을 제외한 개념'은 native replay까지 감사해야 한다 — §6, §8.3

base-model 선행 노출을 인정한 것은 좋다. 그러나 새 추가학습의 공식 목적에도 지도 map target이 있을 수 있고, 그 class/상위 class/derived caption/음성 tuple이 heldout 개념의 의미를 다시 제공할 수 있다. downstream episode class 목록에서만 제거해서는 감독 holdout이 보장되지 않는다. 같은 query AOI가 추가학습 replay에 들어가면 공간 holdout의 의미도 약해진다.

**수정:** 모든 추가학습 source에 `target_ontology`, `concept_overlap`, `spatial_overlap`, `supervision_visibility`를 기록한다. 공식 replay의 지도 target·부모 class와 downstream split의 관계를 명시한다. 엄밀히 배제할 수 없으면 '추가학습 어디에서도 본 적 없는 개념' 대신 **해당 downstream 정의·support interface에서 query 감독을 제외한 개념**이라고 제한한다. imagery-only 노출, coarse parent-label 노출, exact class-label 노출은 서로 다른 상태로 보고한다. test-time K support 자체는 허용되는 감독이므로 그것과 query label 누출도 구별한다.

## 본 비교 고정 전에 구체화할 사항

### 4. Retrieval 단위와 anchor 정보가 평가 난이도를 바꾼다 — §8.3–8.4

gallery가 scene인지 window인지 parcel인지 아직 정해지지 않았다. 정답 parcel geometry로 만든 query crop/후보 mask를 입력으로 주면 대상의 경계 일부를 이미 알려 주는 셈이다. 모든 방법에 같게 주더라도 raw-image dense localization과는 다른 문제다. anchor가 ground truth인지 모델 예측인지에 따라서도 표현·파서·anchor 오류가 섞인다.

**수정:** 한 primary episode의 실제 JSON 예를 먼저 만든다. gallery item, 후보 경계의 출처, relevance 판정 단위, positive가 없는 query 처리, anchor source, distance/CRS, K의 양성/대조 개수, unknown relevance 처리를 명시한다. primary에서 query target mask를 input으로 넘기지 않는다는 규칙을 derivation graph 검사로 구현한다. `provided anchor/operation`과 `predicted anchor/operation`은 별도 결과로 표시한다. crop이나 polygon이 제공되는 평가라면 그 사실을 제목·지표 설명에 적는다.

### 5. Final gate가 가장 강한 반증군과 완전히 일치하지 않는다 — §10

표의 수치 gate는 P−G에 집중하고 S/I/specialist는 비교를 공개한다고만 되어 있다. P가 G를 이겨도 specialist+연산이 P와 동등하거나 더 좋으면, 앞서 적은 '새 시각 재사용 학습법'의 핵심 주장은 약해진다.

**수정:** primary 비교의 순서를 `P 대 강한 G`, 이어서 `P 대 specialist`처럼 동결하고, specialist와 동등할 때 남길 수 있는 주장을 미리 정한다. 더 적은 라벨/연산 비용으로 동등하다는 효율 주장을 채택하려면 비용 지표와 동등성 범위를 결과 전에 지정한다. 사후에 정확도 주장에서 효율 주장으로 슬쩍 바꾸지 않는다. S/I의 결과도 감독·일반 instruction만으로 설명되는지 판단하는 데 사용한다.

### 6. A/B checkpoint만 저장하면 A 효과가 자동으로 분리되지는 않는다 — §4, §8

공통 B에서 encoder도 업데이트하면 A의 차이가 사라지거나 새로 만들어질 수 있다. 또한 P의 A에서 instruction gradient가 이미 들어가고 B에서 다시 들어간다면 I와의 '동일 instruction 예산'을 세부적으로 맞춰야 한다.

**수정:** arm별로 A와 B의 optimizer update 수, unique facts/episode 노출 수, decoder·encoder의 trainable 범위, replay 비율을 표로 고정한다. A의 encoder 효과를 묻는 비교는 A checkpoint에 동일한 fresh readout/reader를 학습하는 통제를 둔다. 최종 시스템 성능은 B 이후로 보고한다. 현재 계획의 fresh connector 조항을 이 구체적인 비교에 연결하면 된다.

### 7. 작은 concept·지역 수에서는 3 seeds가 독립 평가 단위를 늘리지 않는다 — §6, §8.4

초안이 이미 광역 tile 4개와 sparse MADOS를 경고한 점은 타당하다. 남은 문제는 novel concept 수·공간 군집 수·support sampling 반복 수가 아직 없어 .03 차이와 95% 구간의 검출 가능성을 판단할 수 없다는 것이다. 개념별 질문을 수천 개 만드는 것으로 해결되지 않는다.

**수정:** split audit의 출력에 concept×독립 AOI×positive gallery unit×support pool 크기 표를 추가한다. primary macro 평균의 가중치와 어떤 수준을 resample할지 고정한다. support 집합을 여러 번 뽑으면 paired 방식으로 모든 방법에 같은 집합을 주고, support sampling 불확실성을 학습 seed와 구별한다. 표본력이 부족하면 threshold만 두지 말고 탐색적/농업 중심 범위로 제한한다.

## 수정 후의 가장 짧은 진행 경로

1. 실제 source에서 누출 없는 episode 1개와 gallery 계약을 완성한다.
2. generic G와 specialist를 먼저 구현하고 실제 8B VLM의 평가상 역할을 확정한다.
3. P가 바꾸는 핵심 한 가지를 정해 같은 record/budget의 반증 pilot을 한다.
4. 그 결과와 독립 평가 단위 수를 보고 100k·다중 seed·확장 비용을 확정한다.

**최종 의견:** 계획은 이전보다 정직하고 반증 가능해졌다. 아직 가장 중요한 미정 요소는 규모가 아니라 **P가 G와 다른 학습법인지, 그리고 주 점수가 VLM이 수행하는 무엇을 측정하는지**다. 이 두 가지를 해결하기 전에는 '큰 연구 계획이 완성되었으니 본 학습을 돌리면 된다'고 판단하지 않는다. 제한된 구현·자료 감사와 반증 pilot 준비는 계속할 수 있다.
