현재 가장 강한 결론은 **“홍수 source label을 읽는 신호는 있으나, E2 reader의 출력은 명시적으로 공급한 차분 입력을 0으로 만드는 개입에 취약하다”**이다. `history_sensitive_under_intervention` 판정은 사전 규칙과 일치하지만, 이 명칭을 “시간적 추론 입증”으로 번역하면 안 된다.

| 분석 | 실제 관측 | 허용되는 해석 |
|---|---|---|
| C1 | 같은 것으로 재구성된 prompt, 품질 기준을 맞춘 8사건에서 reader 평균 BA .803496, blind .500000; 차이 .303496, 사건 bootstrap 구간 [.180776,.412151] | 문구만으로 설명되지 않는 EO source-label association. 다른 tile의 사후 상태·배경 구별로도 가능 |
| E3 flood real | 10사건·57개 paired tile, seed별 BA .879583/.794167/.848333; 평균 .840694 | 이 노출된 부분집합에서 원래 pair label과의 일치 |
| E3 zero-delta arms | earlier_only, later_only, repeat_earlier, repeat_later, no_delta 모두 BA .5. 두 현상·3seed·다섯 arm의 실제 JSONL **2,676/2,676답이 no** | 독립적으로 다양한 실패가 아니라, 공통으로 D=0인 입력의 보편적 no 붕괴 |
| E3 reverse | flood source-label agreement .597917/.583333/.604167, 평균 .595139 | 슬롯 교환·차분 부호 반전·고정된 chronological 문구 충돌의 혼합 영향. 역순의 물리적 정답 정확도가 아님 |
| E3 landslide real | 두 지역 평균 BA .567708/.604167/.552083, 평균 .574653 | 약하고 지역·seed 의존적인 판독 신호. 두 현상 모두에서 강한 시간 이해라는 주장을 지지하지 않음 |

특히 **no_delta=[A,B,0]는 A와 B를 그대로 보존**한다. 두 관측만으로 원래 결정을 유지하지 못하고, 오히려 차분 입력 D를 0으로 만드는 것이 모든 붕괴 조건의 공통점이다. 실제 양성으로 답했던 항목의 later_only/repeat_later yes 유지율도 0이다. 이 결과는 차분 내용 의존, 학습된 입력 형식 의존, 0인 차분이 만드는 비정상 분포·기본 no 응답 등의 설명을 아직 분리하지 못한다. “D에 유용한 정보가 없다”거나 “모델이 아무 이미지도 보지 않는다”는 결론도 아니다.

hard-negative FPR가 real의 2/51, 2/51, 1/51에서 later_only/repeat_later의 0/51로 내려가는 것은 성공적인 specificity 개선으로 인용하면 안 된다. 해당 arm은 양성까지 전부 no로 답한다. Landslide의 실제 지역별 BA도 Hiroshima .71875/.625/.4375, Indonesia .41667/.58333/.66667로 일관된 지역 일반화를 보이지 않는다. region 두 개를 tile 수로 부풀린 신뢰구간을 만들면 안 된다.

C0는 중요한 반대 방향의 맥락을 준다. **입력 형태마다 따로 학습한** 선형 probe의 flood BA는 earlier .4935, later .6523, pair .5887이다. later−pair .0636의 사건 bootstrap 구간은 [-.1504,.3096]으로 우월성은 확정되지 않았다. pair는 hard-negative 사건 평균 FPR .0158로 later .2033보다 낮지만 민감도와 다른 음성 종류를 구별하는 순위에 비용이 있었다. 따라서 E3의 frozen-reader later_only=.5를 “B에는 정보가 없다”는 근거로 쓸 수 없다. C0와 E3는 학습 방식·표현·평가 부분집합이 달라 직접 성능 순위를 매길 수 없다. C1과 E3도 각각 pos 대 hard-negative와 같은 tile의 pos 대 pre/pre-neg를 평가하므로 .8035 대 .8407은 개선 전후 수치가 아니다.

CVPR 원고에서 현재 허용되는 핵심 주장의 상한:

> In three seeded instances of one frozen-backbone EO reader, source-label discrimination persists under reconstructed prompt matching, while zeroing the explicit difference input collapses decisions even when both observation embeddings remain available. Our controlled audit separates evidence of visual label association from unsupported claims of temporal competence.

이 문장은 현재 결과의 주장 범위이며 신규성이나 CVPR 채택 가능성을 입증한 문장이 아니다. bounded memory, 무엇이 언제 변했는지의 실제 시점 추론, 근거 공간 정합, uncertainty calibration, 새로운 사건 일반화는 아직 별도 검증이 필요하다. E3의 108개 flood tile은 57개 paired tile+51개 hard-negative tile이며, 108개의 독립 사건이나 108개의 모두 동일한 대조 묶음으로 표현하면 안 된다. Landslide 22 tile은 별도 설명용이다.

다음에 권하는 **단일 분별 실험은 학습·평가 입력 형식을 일치시킨 reader 비교**다. 동일한 현행 training ID, backbone/projector 규모, seed 3개, 학습 step·optimizer·192 token 위치·문구를 고정하고 입력 조건만 full=[A,B,D], AB=[A,B,0], B=[0,B,0], D=[0,0,D]의 네 수준으로 둔다. 각 조건을 처음부터 해당 형태로 학습·평가하므로, E3에서 제기된 “학습 때 보지 못한 0 블록 때문에 무너졌다”는 설명과 정보 부족을 구별할 수 있다. 원 full checkpoint와 숫자만 비교하지 말고 같은 재구성 학습 집합의 full 조건도 같이 학습해야 한다.

해석 규칙은 결과 전에 고정한다. AB가 full에 근접하면 명시적 D의 필요성은 표현·최적화 문제였을 가능성이 커진다. B가 근접하면 이 라벨 과제에서 과거 관측의 추가 이익이 입증되지 않은 것이다. D가 근접하면 원래 raw-view 토큰이 없어도 차분 표현이 현재 라벨 판독을 지탱할 수 있다. full이 모두 앞서더라도 실제 변화 추론의 확증은 아니다. 그 주장을 하려면 실제 취득 시각과 양 시점 상태를 독립 검수한, 개발에 노출되지 않은 사건에서 before/after gold를 정의해야 한다. 현재 E2 자료를 다시 쓰는 경우 이 새 비교도 development 진단으로 제한한다. 지금 데이터를 보고 threshold·seed·유리한 음성 종류를 고르면 안 된다.

이번 검토는 기존 JSONL과 scores를 읽어 해석한 것으로, 새 학습·GPU 추론·공유 저장소 수정은 하지 않았다. all-no 행 수는 원 JSONL에서 별도 합산했다. 참고 산출물은 실제 저장소의 `artifacts/e3_pair_dependence_v1_20260925/{scores.json,prereg.json,answers_*.jsonl}`, `artifacts/e3_independent_audit_20260925.json`, `artifacts/c1_same_prompt_flood_v0_20260925/results.json`, `artifacts/c0_linear_view_probe_v1_20260925/results.json`이다.
