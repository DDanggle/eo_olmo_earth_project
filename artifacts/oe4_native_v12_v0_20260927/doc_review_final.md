# OE4 실행 문서 최종 독립 과학 검토

2026-09-27. 대상: `OE4_NOVELTY_AND_NATIVE_VLM_EXECUTION_20260927.md`, `oe4_execution_scope_v0.json`. 문서를 수정하거나 서버를 실행하지 않았다.

**판정: 실행 기반과 다음 연구 가설을 기록하는 문서로는 적절하다. 새로운 방법이나 CVPR 수준 효과를 주장하지 않는 구분도 유지된다. 최종 실행 상태를 갱신하고 아래 해석 제한을 지키면 게시에 동의한다. 현재 연구 성능 결과는 없다.**

## 게시 전에 반영할 사실

- parent가 확인한 controller v0의 `[Not Found]` 파싱 실패는 worker/GPU 학습이 시작되기 전의 실패다. 이를 보존하고, v1의 7개 negative/regression case 통과와 실제 GPU 학습 성공을 구분해야 한다. 유휴 대기·배포·실행·학습 완료를 각각 실제 receipt 시점으로 표기한다. 현재 본문의 “아래 시각 snapshot”에는 날짜만 있으므로 시간·시간대도 붙이는 편이 좋다.
- dev 1266/1267이 같은 parent 좌표라는 확인은 파일 수가 독립 지역 수가 아님을 구체적으로 보여준다. 이것만으로 train–dev overlap까지 확인됐다고 쓰지는 말고, 진행 중인 전체 72파일 parent-group audit 결과를 그대로 보고한다. 현재 engineering split을 바꿔 이전 run 결과를 덮어쓸 필요는 없다. 후속 과학 비교는 별도 parent-region split으로 고정해야 한다.

## 놓치기 쉬운 주장과 강한 대조

1. **연결 직후의 오류를 OlmoEarth의 한계로 귀속하지 않는다.** 랜덤 또는 1-step adapter에서는 연결부의 미학습이 가장 직접적인 설명이다. “원본 모델의 오류부터” 단계에는 같은 라벨/질문으로 충분히 학습한 **frozen OlmoEarth + adapter + 같은 VLM** 기준선이 필요하다. RGB-only, 올바른 EO, 틀린 EO, EO 없이도 맞는 질문 통제는 유용하지만 이 기준선을 대체하지 않는다.
2. **공식 목적 추가학습 + 같은 VLM 정렬**이 필수다. frozen 원본과 우리 joint 학습만 비교하면 추가 compute·supervision 효과와 encoder 목적의 효과를 분리하지 못한다. 최종 문단에는 이 요구가 있어 적절하다. JSON과 문서의 작은 두 arm은 개발용 후보 선별로만 해석해야 한다.
3. **관측 제거/보존 축에는 direct measurement와 강한 단순 회귀를 추가한다.** 필요한 밴드가 있을 때는 정확한 지수 계산기, 밴드 일부가 없을 때는 동일 입력과 availability mask를 받는 단순 regressor/heteroscedastic regressor가 강한 대조다. 빠진 밴드 값도 상관관계로 추정할 수 있으므로 모두 “몰라야 한다”고 채점하면 안 된다. 오차·calibration·risk/coverage를 본다. 위성 지수 정답을 식생 건강이나 고사목 정답으로 확대하지 않는다.
4. **RGB와 EO의 관측 예산을 명시한다.** 현재 smoke는 첫 시점 RGB와 두 시점 S2다. 과학 비교에서는 RGB baseline에도 비교 목적에 맞는 동일 시점 집합을 주거나 “추가 시점/밴드 정보의 효과”라고 구분한다. 추가 정보의 효과와 학습 목적의 효과를 섞으면 안 된다.

## 가장 실용적인 다음 실험

첫 단계는 연결 성공 뒤 **밴드 정보가 제거될 때 나타나는 오류를 제한된 범위에서 재현하는 개발 실험**이다. parent-region을 분리하고 하나의 명확한 native-spectral target을 정한다. 같은 crop의 전체 밴드, 필수 밴드 제거, 무관 밴드 제거를 같은 query에 대응시킨다. 전체 밴드의 직접 계산기, frozen OlmoEarth+학습된 head/adapter, availability-conditioned 단순 regressor를 먼저 실행한다. 후보 학습은 이 기준선에서 구체적인 실패가 남을 때 추가한다.

여기서 의미 있는 신호가 나오면 실제 언어 질의와 독립 semantic/region target으로 확장한다. 지수 회귀와 calibration 개선만으로 새로운 EO-VLM 사전학습 방법을 확정하지 않는다. 반대로 단순 회귀·calibration으로 같은 이점이 나면 해당 축의 방법 주장을 줄이고, 별도 우선 후보인 소수 공간 교정의 새 지역 전이로 투자 방향을 옮긴다.

## 유지할 좋은 제한

현재 문서는 original checkpoint 시작, generic CE target의 비정답성, full official recipe와의 차이, WorldCover base loss 비활성 가능성, 지역 비독립, encoder 갱신과 성능 향상의 구분을 적절히 명시한다. 두 문서의 native/VLM 규모도 일치한다. `forbidden_inferences`는 정직하며, 신규성 지도 자체를 신규성 입증으로 부르지 않는 점도 유지해야 한다.
