# 수정본 최종 독립 확인 — 2026-09-27

대상: `CVPR_2027_OLMOEARTH_RESEARCH_PLAN_20260927.md` 수정본. 원 방법 리뷰와 후속 리뷰를 보존한다. 새 실행 결과는 검사하지 않았다.

**판정: bounded 자료·interface 검증과 반증 pilot 준비에 conditional go. 수정본에서 이 범위를 막는 새로운 큰 내부 모순은 확인하지 못했다. 본 학습 run-ready 또는 CVPR 신규성 확보 판정은 아니다.**

이전 주요 지적은 다음과 같이 반영됐다.

- G와 P는 같은 forward·tuple·공통 감독을 사용하고, P-v0만 listwise binding을 추가한다. generic contrast와 같다면 새 원리가 아니라는 한계도 명시됐다. 이제 차이는 식별 가능하지만 효과·신규성이 증명된 것은 아니다.
- primary는 실제 VLM의 최종 후보 ID ranking으로 정의됐고, oracle structured query 및 같은 VLM parser+specialist 경로가 분리됐다.
- 새로운 A/B 학습의 test 공간·support/query·연결 날짜 제외와 native replay 지도 ontology 감사가 추가됐다.
- A에서 encoder 업데이트/VLM 동결, B에서 encoder 동결/LoRA·연결부 적응이 구분됐다. 전체 시스템 결과와 fresh readout의 EO 검증도 분리됐다.
- GT target parcel 경계 입력 금지, 제공된 anchor와 예측 anchor의 구별, specialist를 포함하는 최종 반증 기준이 추가됐다.
- 21 core+12 auxiliary+baseline 비용과 제한된 독립 지역 수를 그대로 드러내고 있다.

작은 문구 수정 두 가지를 권한다.

1. §8.4의 “정답 operation/anchor를 미리 넘기지 않는다”는 뒤의 “사용자 지정 anchor polygon은 허용”과 독자가 혼동할 수 있다. **“요청에 제공되지 않은 정답 operation/anchor를 oracle 입력으로 추가하지 않는다. 사용자 지정 anchor는 공통 입력이다”**로 쓰면 명확하다.
2. A/B가 순차 학습이므로 frozen/joint 표의 `joint`는 **`EO-updated`**, 반대는 **`EO-frozen`**이라고 명명하면 VLM과 encoder를 동시에 업데이트했다는 오해를 줄일 수 있다. 현 단계 A에서는 VLM이 동결된다는 명세를 우선한다.

남은 data contract, 실제 `s(e)`/loss 정규화·λ, gallery ranking 방식·출력 실패 채점, 독립 표본 수, A/B 처리량은 명시적으로 미정이다. 이를 정하기 전 100k 확증 실행을 시작해서는 안 된다는 현재 문서의 제한에 동의한다. **특히 P-v0는 이미 큰 방법을 발견했다는 안이 아니라, 기존 generic learner만으로 충분한지를 시험할 첫 반증 후보**라는 상태를 유지해야 한다.
