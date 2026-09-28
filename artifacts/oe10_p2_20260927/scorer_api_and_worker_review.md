# OE10 독립 scorer 계약 및 worker 평가 경로 검토

2026-09-27. 이 scorer는 모델 실행/학습 관문을 대신하지 않고, 저장된 원128×128 확률 mask와 고정 development 정답을 독립 채점한다. 전역 연구 파일·서버 수정 없음.

## 고정 입출력

CLI:

```text
python score_predictions.py --prepared-root PREP --episodes-root EP \
  --prediction-dir RUN_PRED --gate-config CONFIG --out NEW_SCORE.json
```

같은 source snapshot에 `episode_loader.py`가 있어야 한다. Python API는 `score_predictions(prepared_root=..., episodes_root=..., prediction_dir=..., gate_config_path=...)`이며 keyword-only다.

`RUN_PRED/predictions.jsonl`에는 common96×K4의 정확히384행이 필요하다. 행의 키는 `episode_id,base_id,k,npz_path,sha256` 다섯 개다. `npz_path`는 해당 prediction directory 내부의 상대경로이며, 파일은 `probability` 하나의 float128×128 배열만 갖는다. uint8/class mask·추가 key·NaN/Inf·범위 밖 값·잘못된 크기·경로 이탈·hash 변경은 거부한다. 동일 NPZ를 여러 행에서 참조하는 상수 baseline은 허용하되 모든 episode/base/K identity가 맞아야 한다.

scorer는 public K8 episode에서 공통 base를 직접 도출하고 모든 K1/2/4/8을 요구한다. raw prediction의 누락·추가·중복·join 불일치를 거부한다. source scoring JSONL hash는 잠긴 gate config와 대조한다. 정답은 scorer 안의 `EpisodeLoader.evaluation_target`으로 읽고 target/counter/valid mask·pixel count·label hash·evaluation audit identity를 다시 검증한다. model inference를 위해 gold를 읽거나 반환하는 기능은 없다.

주 출력 키:

- `target_iou_by_k`, `absent_fp_area_by_k`, `absent_fp_case_rate_by_k`
- **`target_iou_auc`** (`auc`라는 별칭은 없음)
- `cohort`, `aggregation_breakdown`, `per_episode_scoring_only`
- config/scorer/loader/prediction-manifest/source-scoring SHA

확률>0.5에서 mask를 만들며 void를 제외한다. target-present만 IoU에 넣는다. 같은 query/target의 counter 변형을 먼저 평균하고 query→target class→parent macro를 적용한다. absent-case는 valid 면적 대비 예측비율>0.001이며, 해당 지표에 관측 사례가 없는 class를0으로 채우지 않는다. 원 score와 per-class 분모를 함께 남긴다. AUC는 선형 K간격 사다리꼴 합/7이다.

## 검증 결과

- 합성 경계/집계/누출방지 검사22/22 PASS. 기본 시스템 Python은numpy가 없어 import 단계에서 실패했으나 bundled numpy runtime에서 모두 통과했다. 실제 모델 결과를 만든 테스트가 아니다.
- 실제 고정 OE8 metadata672개에서 common96×4=384개 identity와 scoring join을 별도 확인했다. source scoring SHA도 잠긴 config와 일치한다. 이 검사는 실제 prediction 채점이나 모델 성능 측정이 아니다.
- 출처 및 테스트 기록은 `scorer_test_receipt.json`에 저장했다.

## p2_worker.evaluate 정적 검토

검토한 worker는 development public catalog에서 K8 base96개를 고르고384개 prediction을 쓴다. `with torch.no_grad()` 안에서 `module(...,with_language=False)`가 내는 원128×128 logit을 sigmoid float32로 저장하므로 scorer shape/type 계약에 맞는다. 파일 hash를 행에 기록하고 manifest를 닫은 후 별도 subprocess scorer를 실행한다. scorer를 부르는 인자명과 수정된 `result['target_iou_auc']` 키도 일치한다.

evaluation cache는 model.eval/no_grad 아래에서만 사용하고 이후 비운다. B2 training에 평가 cache를 재사용하는 경로는 이 evaluate block에서 발견하지 못했다. 모델 함수에 dev target을 건네는 코드도 없다. raw NPZ8관측의 CPU I/O와 encoder로 전달하는 query2관측은 기존 loader 계약으로 구분된다.

**범위:** 이 단계의 채점은 mask 성능이다. 평가 중 언어 생성을 실행하지 않으므로 생성 답변의 정확도·근거 설명 성능이라고 보고하면 안 된다. training adequacy/fairness bool, seed 간 판정은 scorer가 만들지 않는다. 실제 receipt와 별도 gate 통합의 책임으로 남긴다. 이 정적 검토에서 worker evaluate block과 scorer 계약의 즉시 실행을 막을 불일치는 찾지 못했다.
