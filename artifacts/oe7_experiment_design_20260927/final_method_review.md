# OE7 통합 초안 최종 방법 검토

2026-09-27. 검토 범위: 최소 학습 2×2, baseline과 selector의 기여 귀속. `OLMOEARTH_NEXT_EXPERIMENT_DESIGN_20260927.md`를 읽었다. 초안은 수정하지 않았다. 아래 1–2는 부모가 추가 예정이라고 알려 준 항목이며, 최종본에서 실제 반영을 확인하면 해결된다.

1. **2×2에서 날짜 학습 분포와 selector 학습을 고정해야 한다 (§6–7).** R0/R1은 동일한 randomized observed-subset 학습 순서를 사용하고, S0/S1은 최종 동일 checkpoint의 추론 정책으로만 바꾸는 첫 비교가 가장 명확하다. 그렇지 않으면 네 칸은 표현 loss뿐 아니라 서로 다른 관측 curriculum까지 비교한다. shared selector는 양쪽 representation의 train state를 같은 수로 받은 뒤 같은 training-only utility로 학습해 고정한다. 별도 selector를 각 R에 맞춰 다시 학습하면 그 비용/차이를 추가 요인으로 기록한다. 처음에는 2개 초기→총4개 관측을 양쪽 모두 사용하도록 고정하면 선택 효과가 명확하며, adaptive stop은 별도 비용–품질 실험으로 둔다.

2. **“full-grid/all-observation은 비용이 큰 참조”라는 단정은 삭제해야 한다 (§7).** 2→3→4 재인코딩은 9 frame 상당 입력을 처리한다. 같은 8후보를 한 번에 읽는 방식이 더 싸거나 비슷할 가능성이 있다. 각 경로의 실제 FLOPs/시간/VRAM을 측정해 예산 안에 들어오면 같은 비용의 강한 상대에 포함해야 한다. 모든 관측을 읽는 것과 full-grid token을 Qwen에 넣는 것은 서로 다른 비용 요인이다. all-8 + 일반 learned resampler도 비교할 수 있다. 원자료를 이미 CPU에 전체 로딩했다면 관측 선택의 I/O 절감도 주장하지 않는다.

3. **R1과 B4의 정의를 실행 전에 더 이상 선택지가 남지 않게 고정해야 한다 (§6).** 현재 `L_pair` 이름만 있고 B4는 “표준 contrastive/일반 distillation”으로 묶여 있다. R1의 첫 후보를 positive-minus-confusion cosine margin/softplus 같은 식 하나로 적고, B4의 **표준 supervised contrastive**는 정확히 같은 pair, query cell, label budget, projection, tuning 횟수를 받도록 명시한다. 일반 distillation은 나중의 grid→압축 보존 후보에 대한 별도 대조다. 두 일반 방법 중 유리한 것만 결과 후 선택해서 비교하면 기여가 모호해진다. R1이 B4와 같은 효과면 알려진 metric-learning 적용으로 해석한다는 현재 신규성 한계는 유지하면 된다.

4. **같은 Qwen을 쓰는 것과 같은 dense 출력을 주는 것은 다르다 (§5,9).** 실제 두 단계를 적어야 한다: 각 encoder가 만든 mask를 Qwen에 주는 비교는 end-to-end 품질이고, 저장한 **동일 mask·측정값·provenance ID**를 두 reader 경로에 그대로 주는 replay는 reader 기여 분리다. 현재 문장만으로는 더 좋은 mask가 만든 답변 개선을 VLM 추론 효과로 세기 쉽다. 선택된 날짜 replay는 별도로 유지하며, 이것은 selector가 준 정답 아닌 선택 결과를 공유하는 attribution ablation이고 배포 가능한 selector baseline은 아니다.

5. **U-TAE의 query/support 조건을 명시해야 한다 (§6 B3).** 현재 알려진 작물 과제에서 U-TAE의 class mask를 명시적 class query로 고르는 방식은 강한 전문 baseline이 될 수 있다. 그러나 이름 없는 예시 조건이나 target/confusion 역할 전환에서는 표준 closed-set U-TAE가 그 인터페이스를 원래 제공하지 않는다. 이 경우 동일 support를 쓰는 간단한 prototype/episodic head 또는 동일 class-binding 단계로 확장해 비용을 포함하거나, B3의 적용 범위를 named-target 조건으로 명확히 제한한다. 인터페이스가 없는 baseline을 0점 처리해서 “VLM이 필요하다”고 결론 내리면 안 된다. 실제 EO 관측 수, dense mask 감독, 해당 조건에서 이용 가능한 class 이름도 맞춘다.

위 수정을 전제로, 8후보·2초기·최대2추가의 변경 자체는 타당하다. R1을 학습 가설 검증으로 위치시키고, 공식 native 추가학습과 독립 EO/새 reader 평가를 남겨 둔 현재 구조에는 추가적인 필수 중단 사유가 없다. 한 dev 부모 지역과 synthetic support의 결과를 지역 일반화·사람 교정 시간 절감으로 확대하지 않는 제한도 적절하다.
