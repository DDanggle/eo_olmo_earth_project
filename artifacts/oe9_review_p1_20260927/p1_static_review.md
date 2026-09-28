# OE9 P1 worker/controller bounded 정적 검토

2026-09-27. `p1_worker.py`, `run_p1_bounded.py`, loader 반환 계약을 검토했다. P1은 학습 자료2개에서 실제 supervised 경로·저장복원을 확인하는 공학 검사다. native replay가 없으므로 B2/P2 평가로 부르지 않는 해석은 맞다. GPU/서버 실행 및 source 수정 없음.

## 수정 권고

1. **controller 예외 경로의 own-worker 정리:** Popen 이후 `save()` 등에서 예외가 나면 현재 outer except는 상태 기록 후 raise만 하고 실행 중인 child를 정리하지 않는다. 그러면 worker.wait의7200초 상한에 도달하지 못해 own worker가 남을 수 있다. except/finally에서 `worker is not None and worker.poll() is None`인 경우에만 생성한 자기 process group을 TERM→bounded wait→KILL로 정리한다. 다른 GPU 작업을 건드리지 않는다. timeout/예외 경로에서도 가능한 범위의 protected/source after 상태를 기록한다.

2. **공학 실행 완료와 P1 gate PASS를 같은 상태로 읽지 않기:** head의 두 사례 IoU가0.8 미만이어도 continuation parity+anchor update가 통과하면 현재 worker는 `completed_p1_engineering`/exit0이 될 수 있고 controller는 `worker_completed`만 기록한다. receipt에 `p1_full_gate_pass`가 있어 사람 판독은 가능하지만 자동 후속 실행은 이를 반드시 읽어야 한다. worker에서 `completed_engineering_overfit_gate_failed` 등으로 구분하거나 controller가 receipt의 full gate를 확인해 다음 학습의 허가와 분리하는 편이 좋다.

## 확인한 정상 경로

- 실제 고정 salted-hash 선택을 로컬 scoring/manifest로 재현했다. query40271/40255와 사용한4개 support패치 모두8관측 no-missing=True다. query2관측과 support8관측의 원128입력 조건을 이 두 사례에서 만족한다. 밝거나 구름 없는 관측이라는 뜻은 아니다.
- EO 출력 `B,H,W,T,bandset,D`에서 시간·bandset 평균 후 공간 prototype을 계산하는 축은 코드상 맞다. 연간 작물 label을 매 날짜의 crop 상태 정답으로 쓰지 않는다.
- head prefit은 frozen EO feature cache만 쓰고, 이후 joint 단계는 query/support EO를 새로 실행한다. 오래된 frozen cache를 encoder update의 학습 입력으로 재사용하지 않는다.
- query gold 접근은 명시적 train-only loader 메서드와 train scoring으로 제한된다. sourcebank/dev는 이 P1 선택에 들어가지 않는다. target coverage의 CE 문장은 학습 정답 감독이며 평가 결과가 아니다.
- frozen Qwen embedding hook에서64개 EO slot을 실제 언어 forward에 넣고, CE만으로 encoder anchor gradient를 별도 확인한다. Qwen 가중치는 optimizer에 포함되지 않는다.
- step1에 module·optimizer·Python/NumPy/Torch/CUDA RNG를 저장하고, step2 연속 실행과 복원 후 step2를 비교한다. 같은 frozen VLM identity를 확인한다. 이 방식은 한 프로세스 안의 실제 상태 복원 검사이며 별도 새 프로세스 재시작까지 검증한 것은 아니다.
- receipt의 parity 판정은 mask·VLM logits·학습 module state·loss를 비교한다. IoU0.8은 두 학습 사례의 공학 overfit 기준이며 일반화 성적이 아니다.

**판정:** 위2개 실행/상태 관문 수정을 권고한다. 검토한 버전에서 즉시 실패를 확정할 encoder 축·prototype·CE 연결·optimizer/RNG 복원의 산술 오류는 찾지 못했다. 실제 API·메모리 적합성과 수치 parity는 GPU1 실행으로 확인할 사항이다.
