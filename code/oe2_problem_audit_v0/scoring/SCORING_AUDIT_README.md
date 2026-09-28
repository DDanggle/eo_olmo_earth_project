# OE1 채점 문제 재현과 다음 실행용 수정 사본

이 자료는 저장된 응답으로 수행한 CPU 감사다. 원 OE1 응답·점수·학습 코드·모델은 수정하지 않았고 새 학습/추론/서버 작업도 수행하지 않았다.

## 확인한 문제

원 `train_pilot.py`의 전체 어휘 argmax는 동일 logit이면 작은 token ID를 선택한다. yes=9891, no=2201이므로 canonical 두 토큰의 동률은 no다. 반면 보조 규칙 `p_yes >= 0.5`는 yes를 선택했다. 원 1,740응답에서 exact logit tie 75개와 보조 라벨 변경 75개를 재현했다. 이는 두 서로 다른 예측 규칙 사이의 일관성 문제이며 원 주지표가 잘못 계산됐다는 뜻은 아니다.

| 조건 | n | exact logit 동률 | 바뀌는 보조 답 | 원 주 BA | 원 보조 BA | 새 규칙을 저장 logits에 적용한 진단 BA | 남는 출력공간 차이 |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen/real | 256 | 8 | 8 | 68.3594% | 69.1406% | 68.3594% | 0 |
| frozen/zero | 256 | 4 | 4 | 51.5625% | 49.6094% | 48.8281% | 31 |
| frozen/observation_swap | 68 | 4 | 4 | 57.7056% | 54.5887% | 57.7056% | 0 |
| joint/real | 256 | 13 | 13 | 72.2656% | 75.0000% | 72.2656% | 0 |
| joint/zero | 256 | 11 | 11 | 49.6094% | 51.1719% | 50.0000% | 11 |
| joint/observation_swap | 68 | 5 | 5 | 61.8182% | 60.1299% | 61.8182% | 0 |
| blind/real | 256 | 13 | 13 | 66.7969% | 65.6250% | 66.7969% | 0 |
| blind/zero | 256 | 13 | 13 | 66.7969% | 65.6250% | 66.7969% | 0 |
| blind/observation_swap | 68 | 4 | 4 | 50.8658% | 50.6926% | 50.8658% | 0 |

위 마지막 BA는 원 실험 결과의 대체값이 아니다. 특히 real 조건의 원 보조 69.14/75.00/65.63%를 새 성능 개선 근거로 사용할 수 없다. 새 규칙 적용 시 원 주지표 68.36/72.27/66.80%와 같아진다.

확률이 반올림되어 0.5인 것과 logit이 정확히 같은 것은 별개다. 예컨대 yes logit=-1e-20, no logit=0은 확률 계산 결과가 0.5로 반올림될 수 있지만 no가 더 큰 logit이다. 새 함수는 확률을 판단에 사용하지 않는다. 저장된 이번 응답에서는 확률만 0.5이고 logits가 다른 행은 0개다. 원 BF16-derived logits의 잃어버린 정밀도는 float 변환으로 복구되지 않는다.

zero 조건에는 대문자 Yes/No 토큰 및 비응답 토큰 `**`가 있다. canonical 두 토큰만 비교하는 값과 전체 어휘의 정규화된 답은 서로 다른 예측 공간이므로, tie 정책을 통일해도 frozen/zero 31행·joint/zero 11행의 차이가 남는다. 이 42행을 임의로 일치시키거나 주지표를 대체하지 않는다.

## 이미 수정되어 있는 별도 결함

저장한 checkpoint를 두 번 로드했을 때 출력이 같다는 검사만으로는 저장 직전 학습 모델과 일치함을 증명하지 못한다. 같은 잘못된 가중치를 두 번 읽어도 통과하기 때문이다. 현재 원 OE1 trainer에는 저장 전 train 8문항의 logits를 보존하는 검사와 `validate_checkpoint_probe()`가 이미 반영되어 있다. `code/oe1_bentxt_v0/test_checkpoint_roundtrip.py`의 CPU 회귀검사 3개를 다시 실행해 통과했고, 변조된 encoder가 반복 로딩 일치는 만족하지만 실제 학습 상태 비교에서 거부되는 경우를 재현했다. 완료된 OE1 실행은 수정된 검사 경로를 사용했으므로 미해결 결함으로 보고하지 않는다.

## 산출물과 통합 범위

- `yes_no_scoring.py`: 새 실행용 direct-logit binary decoder, whole-first-token parser, invalid를 오답 분모에 포함하는 지표 함수.
- `train_pilot_scoring_v1_1.py`: 원 trainer의 독립 사본. 보조 라벨 선택을 공통 helper로 교체하고 scoring policy/output revision/helper source SHA를 기록한다. 원 전체 어휘 argmax와 주지표 parser·학습·데이터·예산은 보존한다.
- `eo_model.py`: 사본 trainer의 sibling import를 위해 원 파일을 byte-identical 복사했다.
- `trainer_scoring_only.diff`, `trainer_copy_manifest.json`: 변경 네 곳과 원/사본 SHA. 기존 exclusive output 생성이 유지되어 기존 결과 폴더를 재사용할 수 없다. 새 output에만 실행하고 원 결과를 자동 복제하지 않는다.
- `audit_oe1_scoring.py`: 원 receipt SHA/완결 상태/문항·donor 연결/분모/기존 점수를 확인하고 별도 진단을 기록한다. 이 감사기는 OE1의 과거 정책 재현용이며 새 policy의 결과를 과거 정책으로 검증하지 않는다.
- `historical_tie_audit/changed_secondary_rows.jsonl`: 변경 75행의 원 record 전체와 별도 정책 비교. `original_record_number`는 빈 줄을 제외한 JSON record 순번이며 1부터 시작한다.
- `historical_tie_audit/remaining_output_space_disagreements.jsonl`: 남은 42행의 원 record 및 설명용 분리 목록.

향후 계획 경로는 `code/oe2_problem_audit_v0/`다. 원 OE1 launcher/plan에는 새 사본을 자동 연결하지 않았다. 새 실행을 한다면 별도 사전 명세·source snapshot·output revision으로 이 entrypoint를 명시해야 한다. 이 작업에서는 GPU 실행을 하지 않았다.

검증: 새 pure-Python 회귀검사 21/21 PASS(극소 차이/매우 큰 logit/확률 반올림/반대 token ID 순서/invalid negative/원 record 보존/실제 trainer helper 연결). 원 checkpoint regression 3/3 PASS. 원 1,740응답의 주·보조 점수가 모두 기존 summary와 일치했고, 입력 receipt bytes가 감사 전후 같았다.
