# P1 수정 반영 확인 및 P2 관문 테스트 증빙

2026-09-27. 연구 저장소 `code/oe9_p1_v0/`의 최신 로컬 동결 source를 읽기 전용으로 재검토했다. `source_manifest.json`과 두 실제 파일 SHA가 일치한다.

| 파일 | SHA256 |
|---|---|
| p1_worker.py | `936d844e83a082655d71cf37b37cfe01c907a6e789aea056a0b103ec92eded7d` |
| run_p1_bounded.py | `eab4f32d2639f28ed43785ef66680aca85ca2c9da0948849a1a35c7cab94e9ca` |

## 기존 지적2개 반영 결과

1. **own worker 정리: 반영 확인.** controller62행 이후 `finally`에서 worker가 존재하고 `poll() is None`일 때만 자기 process group을 TERM하고30초 후에도 살아 있으면 KILL한다. 정상 종료·시간 상한·예외 경로에서 이 finally를 거친다. 가능한 protected/source after 검사와 상태 기록도 finally에 있다. 다른 작업의 PID를 선택해 종료하는 코드는 추가되지 않았다. 이는 코드 반영 확인이며 강제 예외를 주입한 runtime 테스트는 이번에 하지 않았다.

2. **P1 full gate와 실행 완료 구분: 반영 확인.** worker267행 이후 `full_pass`가 continuation parity, encoder anchor 변경, 두 학습 사례 IoU≥0.8을 모두 요구한다. parity/변경만 통과하고 head fit 기준이 안 되면 `completed_engineering_checks_head_fit_gate_unmet`를 기록한다. controller는 worker receipt의 `p1_full_gate_pass`를 읽고 그 값이 false인 완료 실행을 `completed_p1_gate_not_met`로 구분한다. 따라서 exit0/worker 종료만으로 P1 전체 관문을 통과했다고 보고할 필요가 없어졌다.

**현재 판정:** 두 수정은 확인한 동결 source에 반영됐다. 실제 GPU 결과와 checkpoint를 이 검토에서 읽지 않았으므로 P1 완료/성공을 판정하지 않는다. 원격 worker와 결과 receipt의 source 일치는 해당 결과를 회수했을 때 별도로 확인한다.

## P2 gate 테스트 기록

`gate_tests_receipt.json`은 이미 관찰한 `python3 -m unittest -v test_evaluate_p2_gate.py`의 최종18/18 PASS를 기록한다. 테스트를 이번에 다시 돌리지 않았다. 실행 tool이 보고한 exit0·18개·0.003초와 현재 해당 파일 SHA를 보존했다. 정확한 실행 시각은 당시 별도 취득하지 않아 null이며, 파일 mtime를 실행 시각으로 꾸미지 않는다.

이 테스트는 합성 scorer 입력의 경계·오류 처리, common96 혼합 차단, 세 seed practical 판정, absent FP·미수렴 차단을 확인한다. 실제 P2 모델 결과나 P1 GPU continuation 검증이 아니다. config는 현재 `proposed_development_rules_before_P2_results_not_confirmatory_preregistration` 상태이며 root의 채택·동결에 따른 향후 SHA는 이 테스트 시점의 SHA와 구분한다.
