# E1 분석 계약 — crop 문맥 × decoder 용량

동결 시각: 2026-08-26 KST. 이 문서는 새 세 칸의 **test 결과를 읽기 전** 분석식을 고정한다.
단, 실행은 이미 시작됐고 첫 새 칸의 일부 validation curve를 관찰했으므로 완전한 사전등록이
아니라 `prospective analysis lock after run start`다. 모든 결과는 이미 노출된 Chimanimani
development fold의 원인진단으로만 읽고, confirmatory evidence 또는 지역 일반화로 쓰지 않는다.

## 질문과 요인

M30의 P4 열세가 기존 cache의 4×64 독립 crop 문맥과 작은 decoder 용량 중 어느 쪽과
관련되는지 2×2로 분리한다.

| 기호 | encoder cache | decoder | 상태 |
|---|---|---|---|
| `y00` | 4×64 tiled | small, 237,537 params | M30 관측: IoU 0.130582 |
| `y01` | 4×64 tiled | large convolutional | 실행 중 |
| `y10` | 1×128 full-context | small, 237,537 params | 실행 대기/중 |
| `y11` | 1×128 full-context | large convolutional | 실행 대기/중 |

`large`는 skip connection이나 OLMo intermediate feature를 쓰지 않는다. 따라서
**U-Net, multi-scale decoder, official AI2 decoder라고 부르지 않는다.** 이는 마지막-layer
feature에서 decoder parameter capacity만 늘린 대조군이다.

모든 칸은 같은 split, S12q 12시점, seed 1, 40 epoch, optimizer, loss, batch, augmentation,
0.5 threshold를 사용하고 best validation IoU checkpoint를 test에 한 번 평가한다. 각 칸은
고정 40 epoch를 모두 돌기 때문에 test 성능 비교는 맞지만, wall time은 fixed-budget cost다.

## 결과를 보기 전에 고정한 estimand

Primary outcome은 test pixel-micro IoU다. AUPRC, ld-IoU, positive-patch macro IoU, ECE와
학습시간은 secondary이며, ECE 단독 우위는 segmentation 성능 우위로 세지 않는다.

- 작은 decoder에서의 context contrast: `C_small = y10 - y00`
- 큰 decoder에서의 context contrast: `C_large = y11 - y01`
- context 평균효과: `C = (C_small + C_large) / 2`
- tiled cache에서의 capacity contrast: `D_tiled = y01 - y00`
- full cache에서의 capacity contrast: `D_full = y11 - y10`
- decoder 평균효과: `D = (D_tiled + D_full) / 2`
- 상호작용: `I = y11 - y10 - y01 + y00`

micro-IoU처럼 비선형인 지표에서 이는 모델계수 추정이 아니라 네 셀의 기술적 contrast다.
cell별 per-sample confusion을 봉인한 뒤 2.56/5.12/10.24/20.48 km 공간 블록 민감도 분석을
같이 보고한다. `bootstrap tail fraction`은 정식 p-value로 부르지 않는다. 20.48 km의 12개
블록 결과는 특히 불안정한 민감도 분석으로만 취급한다.

최종 계산은 `code/analyze_e1_factorial.py` 하나로 수행한다. 네 cell의 runner code SHA와 sample ID가
같지 않으면 분석을 거부하고, per-sample confusion에서 micro-IoU를 재계산해 pilot JSON과 대조한다.

## 판정 규칙

1. **context-supported**: `C_small > 0`과 `C_large > 0`이 모두 성립하고, 사전 지정한 네
   블록 크기 중 최소 3개에서 `C`의 paired bootstrap CI가 0을 제외한다.
2. **capacity-supported**: `D_tiled > 0`과 `D_full > 0`이 모두 성립하고, 최소 3개 블록
   크기에서 `D`의 paired bootstrap CI가 0을 제외한다.
3. 한 contrast만 양수면 `conditional/interaction`, 부호가 엇갈리면 main-effect 주장을 하지 않는다.
4. `y11`이 최고여도 이것만으로 resolution 또는 representation adaptation 가설을 기각하지 않는다.
5. `y11`이 P2보다 낮으면 다음 원인분해는 multi-level features/UNet decoder와 PEFT를 분리한다.

개발 fold의 **탐색적 parity reference**는 M30의 P2다.

- IoU: `0.95 × 0.159254 = 0.1512913`
- AUPRC: `0.95 × 0.174585 = 0.16585575`

`y11`이 두 기준을 모두 넘을 때만 이 한 fold에서 `exploratory parity`라고 쓴다. 이는 원래
G-P를 통과시키지 않는다. G-P는 미열람 지역의 region/event-macro 평가와 catastrophic-fold
검사가 필요하다. 기준을 못 넘겨도 E1은 원인진단으로 끝나며 router 필요성의 증거가 아니다.

## 비용 보고 계약

- encoder/cache extraction, head fit+validation, deployment inference를 분리한다.
- head 시간은 40-epoch fixed-budget로 우선 비교한다.
- practical early-stop cost는 모든 칸에 동일 patience를 사후 시뮬레이션해 별도 표로 낸다.
- large decoder가 빨라지거나 느려진 결과를 single-task end-to-end 비용으로 둔갑시키지 않는다.
- 공유 cache 손익분기는 실제 decoder별 시간으로 다시 계산하며, M30 small-decoder 기준은
  `1,130 + 641K < 1,491K`, 즉 정수 `K ≥ 2`다.

## 이 실험이 답하지 않는 것

- OLMoEarth가 다른 지역·task·backbone으로 전이되는가
- shared cache에 task별 heterogeneous risk가 있는가
- 40 m token support가 실질 병목인가
- intermediate layer 또는 skip feature가 성능을 회복하는가
- LoRA/partial unfreeze가 full fine-tuning보다 좋은가
- 한국 AI-Hub cube가 유효한가(M35 v2 materialization 전 사용 금지)

E1의 성공은 후속 가설을 좁힐 뿐 CVPR 기여가 아니다. 최종 방법 기여는 동일 cache의
다중 task에서 shift별 action value를 예측하고 accuracy–cost Pareto를 개선할 때 생긴다.
