# OE10 독립 실행 결과 collector

`summarize_runs.py`는174줄의 읽기 전용 collector다. 규칙·threshold·gate 코드를 바꾸지 않는다. 새 GPU 실행이나 서버 파일 수정 기능이 없다.

```text
python summarize_runs.py --protocol PROTOCOL.json --run-root TRAIN_SEQUENCE_ROOT \
  --engineering-root ENGINEERING_ROOT --source-dir FROZEN_SOURCE_DIR \
  --gate-config GATE_CONFIG.json --out NEW_COLLECTION.json \
  [--evidence-json FAIRNESS_EVIDENCE_INDEX.json]
```

**부분 상태:** protocol이 없거나 engineering이 실패했거나 run/score/checkpoint가 없으면 `oe10_partial_run_collection_v1`을 쓴다. 완료된 run 일부는 사실대로 남기지만 oe9 gate summary로 승격하지 않는다. 6개 artifact가 있어도 controller가 실행 중이면 부분 상태다. 실패한 engineering_v0를 성공한 v1로 덮거나 혼합하지 않는다. 인자로 받은 engineering의 source가 training과 같고 B0/B2의 별도 PID 복원이 통과했는지 확인한다.

**완료 상태:** 두 arm×고정 seed3개가 끝나고 controller·source·protocol·training log·score·prediction·checkpoint 파일 계보가 맞을 때만 `oe9_p2_scorer_summary_v1`을 출력한다. 이것은 완료된 결과의 집계이지 PASS 판정이 아니다. 근거가 부족한 adequacy/fairness는 false로 두므로 기존 gate가 차단할 수 있다.

검증/산출 범위:

- 실제1..N update 로그, episode order digest, 동결 source/protocol, 모든 평가 interval과 최종 N점수의 존재를 확인한다.
- 각 score·scorer/loader·384개 prediction manifest/NPZ hash를 대조하고, 원 IoU curve에서 AUC를 다시 계산한다. 실제 mask 재채점은 독립 scorer의 책임이며 collector는 반복하지 않는다.
- native/mask/language/total loss와 grad norm의 유한성, head/connector nonzero, B0 encoder grad0·anchor변경0, B2 native+downstream gradient·anchor변경을 로그로 확인한다.
- 충분 학습 schedule 완료와 마지막 평가창을 만든다. 사전 optimizer/LR recipe를 실제 모든 step의 LR 값과 대조한다. protocol에 `base_learning_rates`와 `optimizer_lr_recipe_count=1`이 없거나 맞지 않으면 optimizer완료를 false로 둔다. 단일 recipe는 사전 지정한 시작 조건일 뿐 extensive tuning을 완료했다는 뜻이 아니다.
- `minimum_updates_per_run`이 없으면 이미 사전 지정된 전체 `updates_per_run`을 엄격한 최소값으로 사용한다. 임의의 더 작은 완료 기준을 만들지 않는다.
- readout 비퇴화는 실제 target-present prediction에 empty/full이 아닌 mask가 있고 AUC>0인지로 판정한다. 이것은 head가 의미 있게 일반화한다는 판정이 아니다.
- checkpoint는 실제 파일 SHA를 확인한다. collector가 torch.load로 optimizer/module 내부를 다시 검사한 것은 아니다. 상태 복원 근거는 같은 source의 engineering reference/resume 결과다.

비용 분모:

- `total_gpu_seconds_including_feature_precompute`는 run worker 전체 elapsed다. load/hash/cache/training/evaluation을 포함한 GPU 점유 시간의 기록이며 CUDA kernel시간이라고 부르지 않는다.
- `inference_gpu_seconds`는 worker가 기록한 evaluation_seconds다. CPU NPZ I/O와 별도 CPU scorer도 포함한 end-to-end 평가 시간이고 전체 elapsed의 부분집합이다. 둘을 합산해 총비용을 부풀리지 않는다.
- distinct query observation은 고유 query16개×2날짜=32개다. 반복 평가·K별 호출의 query observation instance 수는 별도 필드로 계산한다. support observation 필드는 반복 instance 수다. 실제 재인코딩/cache효과는 원 `model_costs_cumulative`도 함께 보존한다.

fairness evidence:

동일 cohort, 실제 paired episode order, 같은 원 EO checkpoint, 실제 language supervision 및 LR로그 등으로 확인할 수 있는 사실은 자동 산출한다. reader의 trainability, head 구조 동일성, ID/query-gold/source-bank/acquisition 차단처럼 추가 감사가 필요한 성질은 증거 없이 true로 채우지 않는다.

선택 인덱스는 `{CHECK_NAME:{"path":"actual_audit.json","sha256":"..."}}` 형식이다. 참조 감사는 `status="passed"`, 실제 통과한 항목의 `checks_passed` 배열, 해당 source의 `source_hashes`를 가져야 한다. collector는 hash·항목명·현재 source의 해당 hash를 검사한다. **실제로 수행한 감사 결과를 연결해야 하며 schema를 채우려고 PASS를 새로 작성해서는 안 된다.** 증거가 없으면 false와 `collector_blockers`가 남는다. 동일 이미지/관측권 항목은 감사 외에 실제 query2/support16K runtime metadata도 확인한다.

한계와 테스트:

5개 단위 테스트는 protocol/run 부족, source 변경, 한 run 완료, 여섯 artifact가 있으나 controller 미완료의 fail-closed 동작을 확인했다. 모두 통과했다. 실제 완료된 여섯 모델 run을 아직 집계한 것은 아니며 그 상태를 흉내 낸 성적을 만들지 않았다. 실제 결과가 들어온 뒤에도 이 collector는 기존 `evaluate_p2_gate.py`와 별도로 실행해야 한다.
