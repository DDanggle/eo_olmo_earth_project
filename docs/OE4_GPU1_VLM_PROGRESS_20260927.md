# GPU1 OlmoEarth–VLM 실행 결과와 다음 실험

2026-09-27 15:43 KST 완료, 이후 로그·결과 회수. **새 지역 전이·라벨 절감·VLM 정확도 향상은 아직 검증하지 않았다.** GPU1의 native 학습 경로와 실제 Qwen3-VL 연결·역전파 검사는 통과했다. GPU 장치 조회상 서버 모델은 H200이다. 이번 worker는 종료됐고, 종료 후 GPU 연산 프로세스 없음·사용 메모리 0 MiB를 확인했다.

## 실행 계보

| 단계 | 실제 실행 | 판정과 의미 |
|---|---|---|
| Native smoke | 공식 v1.2 Base, 실제 H5 8개, 8 step, GPU1 | 통과. 학습 손실·encoder gradient·가중치 변경·저장 후 공식 로더 복원 확인 |
| Native development | 같은 원본에서 별도로 시작, train 64 / dev 8, 32 step, GPU1 | 통과. 학습 경로 확인이며 방법 효과 실험은 아님 |
| VLM v2 | 원본 OlmoEarth + 연결부 + frozen Qwen3-VL 8B | 모델 로딩 전 CUDA 통계 초기화 오류. 초기화 순서를 수정한 v3로 이동 |
| VLM v3, BF16 | 실제 이미지와 EO token으로 캐시 생성 검사 | RGB 원 경로 보존·top-1 일치. 캐시와 전체 재계산 점수의 허용 오차 검사 실패, backward 전에 종료 |
| VLM v4, FP32 | 원 RGB cache 대조를 추가, 기존 EO 판정 기준 유지 | 통과. 실제 언어 손실의 encoder 역전파·1-step 가중치 변경·원 RGB 경로 보존 확인 |

v3의 캐시 최대 차이는 0.25, 평균 차이는 0.03626이었다. 판정은 `atol=.15`, `rtol=.01`과 top-1 일치를 모두 요구했다. v4에서 이 허용폭을 넓히지 않았다. 별도 FP32 성공도 BF16 문제가 해결됐다는 뜻은 아니다.

## 지금 실제로 연결한 것

```text
원본 OlmoEarth v1.2 Base ─ 원 밴드·날짜가 있는 H5 ─ EO 128 tokens
                                                   ↓ pooling
                                             EO 16 slots
                                                   ↓ LN + Linear
RGB 이미지 ─ Qwen3-VL 원래 영상 경로 ──────────── Qwen3-VL 8B
                                                   ↓ 언어 손실
                                   연결부와 OlmoEarth까지 역전파 검사
```

Qwen3-VL 자체 가중치는 고정하고, 연결부와 OlmoEarth를 업데이트한다. EO slot은 일반 텍스트 위치를 사용하며 새로운 공간 위치 인코딩을 구현한 결과가 아니다. 원 RGB/DeepStack 특징, 캐시 생성, EO 업데이트 후 원 RGB 출력 보존을 확인한다.

**Native 32-step checkpoint를 VLM에 이어 붙인 실행은 아니다.** native 두 실행과 VLM 검사는 각각 원본 OlmoEarth에서 시작한다. native 검사는 원 학습 경로를, VLM 검사는 언어 손실을 전달하는 경로를 확인한다.

## Native에서 관측한 수치

| 항목 | 8 step | 32 step |
|---|---:|---:|
| 각 step에서 nonzero gradient를 받은 encoder tensor | 213 | 213 |
| dev 목적함수, 학습 전 | 0.59702070 | 0.44859405 |
| dev 목적함수, 학습 후 | 0.59685770 | 0.44880767 |
| 저장·복원 후 token 최대 차이 | 0 | 0 |
| 저장·복원 후 dev 목적함수 차이 | 0 | 0 |

32-step dev 손실은 약 0.000214 증가했다. **성능 개선으로 보고할 수 없다.** S2와 WorldCover, 32×32 crop, 선택한 두 날짜의 작은 진단이다. 원 사전학습 corpus의 부분집합이며 공식 전체 추가 사전학습 재현이나 독립 EO 평가가 아니다. smoke/dev 실행 사이에 표본을 공유하므로 독립적인 두 과학 실험으로 세지 않는다. 저장물에 optimizer 상태가 없어 전체 trainer 재개를 검증한 것도 아니다.

## 실제 VLM v4 결과

실행 시각은 15:42:12~15:43:42 KST, worker 소요 89.88초, GPU 최대 할당 메모리 34.56 GiB였다. source·입력·기존 보호 파일의 전후 해시/mtime/크기는 동일하다.

| 확인 항목 | 실측 |
|---|---:|
| 원 RGB 캐시 vs 전체 재계산 최대 logit 차이 | 0.000026703 |
| EO를 넣은 캐시 vs 전체 재계산 최대 logit 차이 | 0.000023842 |
| 두 경로의 top-1 및 allclose 검사 | 모두 통과 |
| nonzero gradient를 받은 OlmoEarth tensor | 211개 |
| nonzero gradient를 받은 연결부 tensor | 4개 |
| 추적한 encoder 4개 tensor의 실제 최대 변경량 | 약 0.000001 |
| 추적한 연결부 4개 tensor의 실제 최대 변경량 | 약 0.0001 |
| 고정한 Qwen3-VL의 gradient / optimizer 등록 tensor | 0 / 0 |
| EO 업데이트 전후 원 RGB logit 차이 | 0, bitwise 일치 |

CE 손실은 6.23363, clip 전 전체 gradient norm은 454.3176이었다. norm 1로 clipping한 뒤 한 번 업데이트했다. 이 수치들은 실제 학습 경로가 작동했다는 검사 기록이며 정확도 점수가 아니다. encoder 211개 gradient는 native 목적함수의 213개와 다른 학습 경로에서 나온 수치다.

## VLM 결과의 주장 범위

VLM 검사는 한 이미지·한 optimizer step이다. 목표 문장 `The image shows land cover.`는 gradient 경로를 확인하기 위한 일반 문장이며 영상의 정답 라벨이 아니다. 따라서 통과하더라도 작물 분류·변화 설명·새 지역 일반화·전문가 교정 절감의 증거가 되지 않는다.

현재 VLM worker는 encoder/연결부 checkpoint의 저장·복원을 구현하지 않는다. native 저장·복원 통과와 구분하며, 다음 실제 학습 runner에서 별도 확인해야 한다. 생성 검사는 업데이트 전 두 token 범위이고, 긴 생성·다중 이미지·업데이트 후 EO 캐시 동등성은 아직 다루지 않는다.

## 논문 실험으로 이어지는 다음 단계

1. **바로 다음 한 작업은 PASTIS 공개 라벨로 정답이 정해지는 입력 2사례 준비다.** 기존 서버 shard의 SHA256부터 확인한다. 과거 다운로드 코드에는 shard 0001 해시 불일치 이력이 있어, 파일 존재만으로 완료로 세지 않는다. 통과한 자료에서 10밴드와 OlmoEarth 12밴드의 차이, 반사도·원 취득일·격자·부모 지역·mask/void를 확인한다. 연간 작물 라벨을 월별 상태·건강 정답으로 바꾸지 않는다. 산출물은 원 입력·RGB·참조 영역·질문/정답·출처 manifest 2건과 contract/preview이며, 누락 시 구체적인 blocker를 남긴다. 현재 이 묶음은 아직 없고 새 GPU 학습에 투입하지 않았다. `36 query`는 이후 준비 목표다.
2. 같은 query에서 같은 지역/다른 지역의 공개 라벨 기반 support가 얼마나 도움이 되거나 해가 되는지 먼저 측정한다. 실제 전문가가 수행한 교정이라고 부르지 않는다.
3. 실제 정답으로 일반 공동학습 VLM을 충분히 학습하고 저장·복원한다. 원본 frozen encoder+학습된 연결부와 비교한다.
4. 같은 자료·맥락·교정·연산 예산에서 제안 학습 P1과 일반 공동학습 G1을 비교한다. 이 차이가 생겨야 방법 기여를 주장한다. 다른 reader 전이와 독립 EO 성능은 후속 기여 귀속 검증이다.

연결 검사 통과만으로 `ready_to_train=true` 또는 CVPR 기여 검증 완료로 바꾸지 않는다. [첫 교정 전이 계획](OLMOEARTH_TRANSFER_FIRST_EXPERIMENT_20260927.md)의 데이터·분할 관문은 그대로 남는다.

독립 에이전트가 산출물·소스·실행 상태·gradient 행을 재집계한 35개 일관성 검사는 모두 통과했다. GPU 실행을 다시 재현한 검사는 아니다. [독립 감사](../artifacts/oe4_gpu1_retry_20260927/v4_independent_audit.md).

## 근거 위치

- 로컬 native 원본: `artifacts/oe4_gpu1_retry_20260927/snapshot_02/`
- 로컬 BF16 실패 원본: `artifacts/oe4_gpu1_retry_20260927/snapshot_03/`
- 로컬 FP32 성공 원본: `artifacts/oe4_gpu1_retry_20260927/snapshot_04/`
- 동결 VLM source: `code/oe4_native_v12_v0/vlm_snapshot_v4/`
- 원격 root: `/home/work/data/olmoearth/oe4_native_v12_v0`
- 원격 native 가중치: `runs/native_smoke_gpu1_v2/checkpoint_after/`, `runs/native_development_gpu1_v2/checkpoint_after/`
- 원격 VLM v4: `runs/vlm_interface_gpu1_v4/receipt.json`, `vlm_sequence_gpu1_v4_status.json`, `vlm_interface_gpu1_v4.log`

원 실패와 기존 사전등록은 보존했다. 새 source는 별도 경로에 동결했으며 기존 보호 파일 네 개의 해시·수정 시간은 실행 전후 검사한다.
