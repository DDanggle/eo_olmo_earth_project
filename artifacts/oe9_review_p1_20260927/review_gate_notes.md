# 교수 검토 대응 실행 관문: v6 유지

2026-09-27. `review_execution_gates.json`은 P2 성적을 보기 전에 작성한 **개발 관문 제안**이다. 확증 사전등록도 실제 성적 판정도 아니다. v7을 만들지 않고 v6의 입력→학습계약→B0/B2→방법 비교 순서를 유지한다. 이번 파일은 scorer와 연결할 판정 규칙만 구체화한다. 서버·GPU 실행이나 전역 문서 수정은 하지 않았다.

## 1. P2가 실제로 답하는 것

B0는 원 OlmoEarth를 고정하고 충분히 학습한 연결부/영역 head를 사용하는 기준이다. B2는 같은 계열의 판독 경로에서 encoder를 함께 학습하는 일반 공동학습이다. P2의 차이는 **이 작은 개발 조건에서 encoder 업데이트의 이점이 보이는가**이며, 새 제안 알고리즘의 신규성을 증명하지 않는다.

현재 실제 scoring 파일로 확인한 공통 K8 cohort는96개 query–directed-class-pair,16개 query patch, target-present64개/absent32개다. 포함 대상은 class1/2/3이다. class4는 K8 support 부족으로 공통 AUC에서 빠져 있으므로 작물4종 전체 성능이라고 보고하지 않는다. 같은 query의 여러 pair는 독립 공간 표본이 아니다.

scoring SHA와 공통 base ID hash를 JSON에 고정했다. K={1,2,4,8}의 선형 사다리꼴 AUC를7로 나눈다. target-present query만 IoU 평균에 사용하고, query→target class→parent의 macro 집계를 고정한다. counter class별 반복·K별 prefix를 동일하게 유지한다. absent32개는 별도 guardrail이며 빈 정답의 IoU1로 AUC를 올리지 않는다. lower-K 전체192개 성적은 별도 기술 표다.

## 2. 수치 판정: 개발 의사결정과 통계적 동등성을 분리

첫 개발 practical margin을 **absolute AUC0.02**로 제안한다. 데이터에서 추정한 효과폭이 아니라 다음 개발 투자를 결정하기 위한 사전 운영 임계치다. training seed는270927/270928/270929로 고정한다. root가 P2 이전에 채택·동결하고 config SHA를 run/scorer receipt에 남겨야 한다. 바꾸려면 비교 결과를 보기 전 이유·이력을 남긴다. 본 평가의 최소 실용 효과를 이 작은 pilot 값으로 자동 고정하지 않는다.

| 충분 학습·공정성·absent 관문을 통과한 후 | 개발 판정 |
|---|---|
| 같은3개 seed 모두 B2−B0≥+0.02 | 반복된 practical gain. 다른 독립 개발 부모에서 B0/B2부터 확인 |
| 같은3개 seed 모두 절대 차이<0.02 | 관측된 practical tie. 이번 개발 범위의 encoder 학습 이득 주장을 접고 B0 유지 |
| 같은3개 seed 모두 B2−B0≤−0.02 | 반복된 practical harm. B0 유지 및 이 범위의 공동학습 이점 주장 철회 |
| 그 외 혼합 결과 | inconclusive. 평균 상승만으로 GO, 비유의만으로 동등 판정하지 않음 |

여기서 tie는 **세 training 반복의 관측치가 사전 설정한 실용 범위 안에 있다**는 운영적 판정이다. 통계적 equivalence 검정이 아니다. 현재 부모1개를 seed3개로 대체해서 geography CI를 만들 수 없다. 등가성·비열등성의 확증 주장은 독립 지역 수, 정한 허용폭, uncertainty interval을 갖춘 별도 설계를 요구한다. 따라서 도구는 언제나 `formal_equivalence_test_performed=false`, `geographic_equivalence_proven=false`를 출력한다.

B0≈B2이면 이 개발 자료·학습량·reader·목적의 조합에서 encoder 업데이트 가치 주장을 보류/철회한다. **OlmoEarth가 어떤 EO 문제에서도 개선될 수 없다는 결론은 아니다.** 반대로 +0.02가 반복돼도 CVPR급 방법 신규성은 입증되지 않는다.

## 3. 먼저 통과해야 할 두 관문

**충분 학습:** 양쪽 모두 사전 지정 optimizer/LR 탐색 예산과 학습 일정을 마치고, save/resume·유한 gradient·비퇴화 supervised readout·B0 frozen/B2 update 검사를 통과해야 한다. 사전 정한 동일 평가 간격의 마지막3 AUC range≤0.005를 임시 안정성 기준으로 둔다. 시간 상한에 걸렸고 계속 개선 중이면 판정 보류다. plateau는 최적 성능의 증명이 아니므로 일반 기준선의 용량/최적화 실패 진단은 여전히 필요하다. 최소 updates/평가 간격은 P1 실제 처리량을 보고 성적 비교 전에 정한다.

**absent FP:** 동일32개 target-absent 사례에서 valid pixel을 분모로 한 예측 면적 비율과 FP 발생률을 각 seed·각 K마다 비교한다. 면적비 증가 최대0.005, 발생률 증가 최대0.02를 임시 guardrail로 둔다. FP 발생은 valid 면적의0.001을 초과한 예측으로 고정한다. 이 경계를 하나라도 넘으면 평균 IoU가 좋아도 encoder 이득으로 확대하지 않는다. 이것은 상대 악화 제한이며 양쪽 모델의 절대 안전성을 보장하는 값이 아니다.

같은 support·reader·head용량·관측권·정답감독·튜닝예산, query gold 차단, source-bank 감독 배제, ID/class mapping 제거, missing/acquisition guard 검증이 필수다. B0의 feature precompute도 계산비용에 넣는다. matched-data와 matched-total-compute 중 해당 비교의 축을 미리 고정하고 다른 축 비용도 기록한다. 동일 step이라고 비용 공정성이 자동 성립하지 않는다.

## 4. 외부 검토 대응의 나머지 실행 상태

- **두 번째 독립 dev 부모:** 첫 practical gain 이후 가장 먼저 확보할 대상이다. 현재 prepared dev는 t31tfm1개다. t30uxv를 자동 개봉하지 않고, 새 개발 후보의 과거 노출·공간 중복·분할 역할을 감사한다. dev2개가 생겨도 넓은 환경 확증 power가 충족됐다는 뜻은 아니다.
- **Kuro:** 원 날짜·파일·사건 노출, VV/VH 단위·전처리, 영구수역/홍수/void 의미, 개발/새 평가 사건 구분 감사가 끝나기 전 실행하지 않는다. 이미 본 test 사건은 개발용으로 유지한다.
- **EO 약한/강한 주장:** 약한 주장은 EO 비열등성과 새 reader 재사용 이점이다. 강한 주장은 노출 감사된 독립 EO 평가에서 원본·공식CPT·일반joint보다 개선된다는 것이다. frozen linear probe와 full finetune은 별개의 평가 프로토콜이며 이 주장 강약과 동일시하지 않는다. 두 프로토콜 모두 original/B1공식목적추가/B2일반joint에 같은 라벨·튜닝 예산을 적용하고, 실제 제안 checkpoint가 생긴 뒤 추가한다. EO 비열등성 margin은 아직 null이며 test 전에 개발 분산·과제 가치를 보고 동결한다. train loss가 독립 EO 점수를 대신하지 않는다.
- **resampler 설명:** 공간·시간 격자를 learned query로 고정 reader token 수에 압축하는 일반 연결부다. 충분히 학습해야 하고, flat16 평균은 진단용이다. full-grid는 실제 비용을 기록한 참조다. resampler 자체를 새 발명이라고 하지 않는다.
- **D1:** 미완료인 과거 연구 트랙으로 archive한다. 기존 사전등록·관찰은 보존하며 미완료를 pass/fail로 새로 판정하지 않는다. 현재 P2의 성공/실패로 과거 D1 결과를 대체하지 않는다.
- **범위 유지:** tokenizer/RL/물리/cache 신방법을 이 단계에 더 붙이지 않는다. P2 결과가 없는데 v7 연구기획으로 이동하지 않는다.

## 5. 실제 scorer 연결 계약

`evaluate_p2_gate.py --config review_execution_gates.json --summary ACTUAL_SUMMARY.json --out NEW_DECISION.json`으로 실행한다. 출력은 새 파일만 허용하여 기존 결정을 덮어쓰지 않는다. 잘못된 schema/hash/cohort/seed/NaN은 `invalid_input`, 학습·공정성 미달은 blocked, 정상 혼합 결과는 inconclusive로 구분한다. 모델 실행·서버 접근 기능은 없다.

scorer summary에는 다음이 필요하다.

- `schema_version=oe9_p2_scorer_summary_v1`, 실제 config SHA, protocol/seed가 결과 전에 잠겼다는 receipt 필드, scorer source/predictions manifest SHA.
- `aggregation`, absent-case 면적 threshold, common96 cohort의 숫자·class·parent·hash와 명시적 comparison axis.
- required fairness bool 전체, 미리 고정한 seed ID3개, arm×seed6개 run.
- run별 checkpoint/receipt/cohort SHA, `target_iou_by_k`, `absent_fp_area_by_k`, `absent_fp_case_rate_by_k`의0–1 점수. 도구가 AUC를 직접 재계산한다.
- 실제 학습 완료 checks, completed/minimum updates, 고정 eval interval과 완료시점까지 마지막3개 평가창(step/AUC), cap-while-improving 여부, 총 GPU·precompute·inference·support 관측 비용.

이 요약의 bool/집계는 증거를 대신하지 않는다. 실제 scorer와 receipt가 원 예측·원 labels·data guard 로그에서 생성해야 한다. 도구는 raw 모델 예측을 다시 채점하는 scorer가 아니라 그 검증된 집계에 개발 규칙을 적용한다. 테스트 fixture의 숫자는 전부 합성이며 실제 모델 성적으로 저장/보고하지 않는다.
