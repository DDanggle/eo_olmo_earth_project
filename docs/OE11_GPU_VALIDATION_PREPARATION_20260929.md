# OE11 실제 GPU 검증 준비 — 2026-09-29

**준비·CPU 검증은 완료했고 실제 GPU 실행은 대기 중이다.** 09/29 01:30 KST 확인에서 H200 GPU0/1이 모두 기존 프로세스로99% 사용 중이었다. 새 GPU 작업·optimizer update·예산 예약은0이다. 최신 사용자 요청으로 수동 GPU0/1 사용은 허용됐고, 예약 자동화는 계속 중지 상태다.

## 이번에 실제 완료한 것

| 검사 | 결과 | 의미 |
|---|---|---|
| 실제 고정 훈련 사례 | 12개 영상, 지역별4개, K1/K8 총24과제 | 초지·옥수수·포도밭·콩과마다 양성/혼동만/둘 다 부재 |
| 실제 입력 로딩 | 24/24통과 | query2날짜·support8날짜, 정답은 별도 훈련 경로 |
| 모델·소스 실물 검사 | 23파일,18,576,472,682bytes SHA 일치 | 기존 OlmoEarth·Qwen·정규화 소스와 같음; 모델 forward 아님 |
| 실제 Qwen tokenizer | 48역할 입력 검사, 고유문구8개,49~208토큰 | 1024토큰 이내, 잘림 없음; 의미 이해 증거 아님 |
| Mac CPU 재개 검사 | 6프로세스 성공·잘못된RNG2건 거부 | 실제 matching head와 대체 encoder/reader |
| 서버 Linux CPU 재개 검사 | 같은6프로세스 성공·잘못된RNG2건 거부 | Torch2.13.0+cu130 환경, GPU는 사용하지 않음 |
| 실행기 CPU 검사 | Mac/Linux 각각26개 통과 | 점유·예산·실패·타임아웃·자체 프로세스 종료 경계 |
| 최종 입력 사전 검사 | source/input57파일 및 model23파일 일치 | 실행기에 연결된 실제 경로까지 검증 |
| 최종 실행 사전 검사 | 두 GPU 점유로 차단 | 장부48초 그대로, 새 예약·작업 생성 없음 |

합성 CPU 검사는 B2+설명/B0+이름 각각 연속4단계와2단계 저장 후 새 프로세스2단계를 비교했다. 모델·AdamW moments/step·난수·다음 단계와 마지막 단계의 학습 전후 logits·loss가 모두 동일했다. 의도적으로 난수를 바꾼 경우는 실패했다. 이 결과를 실제 EO/Qwen 또는 CUDA 재개 성공으로 간주하지 않는다.

## 바로 실행할 첫 GPU 실험

[동결 프로토콜](../config/oe11_training_resume_protocol_20260929.json)은 **B2(OlmoEarth 갱신)+출처 있는 설명** 한 조건의 실제 학습/재개 검증이다. source/input/hash와 선택 사례·순서를 GPU 결과 전에 고정했다.

1. 네 대상×세 상태의12사례를 고정하고 K1/K8을 각각 구성했다. 세 지역에서4사례씩 선정됐고 훈련 영상 중복은 없다. 이는 사전 hash·가용성 규칙의 결과이며 독립 전문가 교정12건을 뜻하지 않는다.
2. 24과제를 두 번 섞은 고정 순서로48단계 학습한다. **K1만으로는 예시를 고르는 attention의 기울기가0이므로 K8도 실제로 학습**한다. 초안K1전용 목록은 보존했고 GPU 실행에는 사용하지 않았다.
3. 같은 초기 가중치·seed에서 별도 프로세스로 `연속48`과`24저장→새 프로세스24`를 비교한다. 비교 때문에 실제 optimizer update 총수는96이다.
4. 복원 직후 모델·AdamW·RNG는 정확히 같아야 한다. 25번째와48번째 단계의 모델·optimizer·logits·loss 차이 허용치는1e-6이며 RNG/순서는 정확히 같아야 한다. K8의 query/object/text 선택 경로 기울기, 실제 EO/head 갱신과 동결 모듈 불변도 검사한다.
5. 최초/마지막의 같은K1 훈련12사례에서 loss·양성 IoU/recall·부재 오탐을 별도로 기록한다. loss 감소만으로 학습 성공을 판정하지 않는다. 짧은 실행의 학습 신호와 계산/재개 통과를 분리한다.

AdamW 학습률은 EO1e-5/head1e-3, weight decay.01, eps1e-6, gradient clip1이며 짧은 검사에 기존96단계 warmup을 복사하지 않았다. 전체3프로세스는 하나의30분 상한을 공유한다. 기존 누적2시간 장부는48초 사용·7,152초 잔여다. 시간 초과/실패도 차감하고 임의 재시작·예산 초기화를 하지 않는다.

실행기는 GPU1을 먼저 확인하고 사용 중이면GPU0을 확인한다. 실제 UUID로 CUDA 장치를 고정하며 둘 다 사용 중이면 예약 전에 거부한다. 프로세스 그룹·부모 종료·전체 deadline을 관리하고 이번 실행의 자식만 종료한다. 다른 작업은 건드리지 않는다.

## 이후 충분한 비교로 넘어가는 순서

첫 파일럿은 모델 결합과 optimizer가 실제로 학습·복원되는지 확인하는 단계다. 48단계 성능을 논문 성능으로 쓰지 않는다. 통과하면 실제 K별 전체시간(I/O 포함), 최대 메모리와 훈련 곡선을 근거로 다음 비교의 업데이트·시드·시간 상한·학습 충분성 기준을 결과 전에 고정한다.

| 공통 구조의 비교 | OlmoEarth | 언어 입력 | 분리하는 효과 |
|---|---|---|---|
| E0 | 고정 | 작물명 | 공통 기준선 |
| E1 | 고정 | 이름+설명 | 고정 EO 위 설명 |
| E2 | 갱신 | 작물명 | EO 갱신 |
| E3 | 갱신 | 이름+설명 | E3−E2 설명 효과, E3−E1 EO 갱신 효과 |

정식 비교에는 이미 준비한 train192영상·bank48영상·내부검증48영상을 같은 조건으로 제공한다. 충분히 학습한 평균 support head를 구조 대조로 포함한다. 이 네 조건 비교는 아직 실행하지 않았으며 긴 학습 일정도 이번 엔지니어링 프로토콜에 확정하지 않았다.

그 다음 같은 구조·설명에서 pooled/cross_parent 학습 목록만 바꾸고, query2/4/8관측 확대는 별도 공통 대조로 시행한다. 현재 모델 wrapper는query2날짜다. 큰 지역·새 reader·독립 EO 재사용·생성 설명은 그 이후 별도 검증이다. 겨울밀 복구는 선행 조건이 아니고 기존 P2는 보존한다.

## 실행 명령과 경로

실제 연구 저장소에서 먼저 `./bin/nx tunnel up`을 실행한다. 고정된 launcher 인수는 [launcher_command_base.txt](../artifacts/oe11_training_pilot_20260929/launcher_command_base.txt)에 저장했다. 뒤에 `--preflight-only`는 GPU 포함 읽기 전용 검사, `--verify-inputs-only`는 점유와 별개인 입력 검사, `--execute`는 즉시 사전 검사를 다시 거쳐 한 작업을 실행한다. 긴 실행은 이 저장소의 nx를 통한 `setsid nohup`으로 분리하고 로그·PID를 서버 작업 루트에 남긴다. 예약 자동화나 GPU 대기열은 만들지 않았다.

- 서버 작업 루트: `/home/work/data/olmoearth/oe11_training_pilot_v0`
- 서버 worker: `snapshot/oe11_training_worker_v0/worker.py`
- 서버 launcher: `code_snapshot/oe11_training_launcher_v0/launcher.py`
- 첫 GPU job ID: `oe11_b2_facts_resume_01`; 아직 해당 학습 작업은 생성하지 않았다.
- [학습기](../code/oe11_training_worker_v0/README.md) · [실행기](../code/oe11_training_launcher_v0/README.md)
- [현재 실행 상태](../artifacts/oe11_training_pilot_20260929/execution_status.json)
- [입력 사전 검사](../artifacts/oe11_training_pilot_20260929/launcher_inputs_preflight_v1.json) · [GPU 점유 차단](../artifacts/oe11_training_pilot_20260929/launcher_gpu_preflight_v1.json)
- [서버 CPU 재개 검증](../artifacts/oe11_training_pilot_20260929/cpu_fresh_process_receipt.json) · [실행기26검사](../artifacts/oe11_training_pilot_20260929/launcher_linux_cpu_tests.log)

## 발견하고 수정한 실행 문제

전송 첫 시도에서 macOS AppleDouble `._` 파일6개가 서버 launcher 폴더에 포함돼 전체 소스 목록 검사가 실패했다. 검사를 완화하지 않고 실패 전송본을 `oe11_training_launcher_v0_transfer_with_appledouble`로 보존한 뒤 `COPYFILE_DISABLE=1 ./bin/nx push …`로 동일 코드의 깨끗한 사본을 전송했다. 최종 입력 검사는 통과했다. 기존4개 보호 소스의 mtime/SHA는 모든 전송 후 동일했다. 실제 학습 결과를 보기 전에 발견된 패키징 문제다.

## 해석 범위

현재 학습기는 mask loss만 사용한다. 공식 native replay, 언어 생성/CE, 실제 전문가 교정은 없다. 공유 manifest의 calibration 메타데이터는 읽지만 개발/최종 지역 영상·정답 payload나 모델 예측을 이번 학습·진단에 사용하지 않는다. 설명 문구8개는 클래스 코드로 작동할 수 있다. CPU 검사와 사전 검사는 연구 성능·수렴·새 지역 전이·OlmoEarth 독립 능력 보존 또는 CVPR 기여의 증거가 아니다.
