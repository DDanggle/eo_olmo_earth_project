# G-P pilot 독립 검증 — M23 수치를 어디까지 믿을 수 있는가

최종 갱신: 2026-08-26

## 판정

M23의 “P4가 전 지표 1위”는 **8-epoch 개발 관측**으로만 보존한다. G-P 통과나 OLMo의 정확도
우위 증거로 쓰지 않는다. 8-epoch test를 본 뒤 epoch를 바꿨고, AUPRC 표본추출 결함·weak P2·
timestamp 정보 비대칭·비용 범위 누락이 있기 때문이다.

## 확인된 결함

| 심각도 | 결함 | 결과에 미치는 영향 | 조치 |
|---|---|---|---|
| 무효화 | AUPRC가 batch마다 같은 20,000 pixel offset만 반복 표본추출 | spatial sampling bias. `positive_pixel_frac`도 전체값이 아님 | 모든 pixel exact AP로 교체 |
| 무효화 | Chimanimani test를 8 epoch에서 본 뒤 40 epoch로 protocol 변경 | 이후 결과는 confirmatory test가 아님 | 이 fold를 development holdout으로 강등 |
| 차단 | P2가 공식 Sen12 3D-UNet이 아니라 265,649-param tiny 구현 | strong task baseline을 이겼다는 주장을 못 함 | 공식 3D-UNet과 U-TAE가 G-P 필수 |
| 차단 | P4만 실제 acquisition timestamp를 encoder에 전달 | 같은 timestep index여도 정보 계약이 다름 | raw baseline date encoding 또는 P4 timestamp ablation |
| 차단 | P3 U-TAE와 P5 다른 GeoFM 없음 | `max(P2,P3)` 및 backbone 일반화 gate 미측정 | full G-P 전 필수 |
| 과장 | P4 62초·0.32GB가 encoder/cache 비용을 제외 | cached-head 비용을 end-to-end 우위로 오독 | cold/cached/deployment 비용 분리 |
| 오류 | 공식 LD 조건 `>50`을 `>=50`으로 구현 | 경계 표본 cohort 불일치 | strictly `>50`으로 수정 |
| 재현성 | pos_weight는 train 전체가 아닌 300표본, RNG는 arm 실행 순서 의존 | arm별 초기화·shuffle과 loss weight가 바뀜 | 전체 train mask·arm별 RNG reset |
| 재현성 | RNG reset 뒤에도 CUDA strict determinism을 강제하지 않음 | 동일-seed P4 replay test IoU `0.122826→0.143442`(+16.8%) | deterministic algorithms/cuBLAS/cuDNN/TF32 계약 강제, bitwise 2회 smoke |
| 재현성 | best checkpoint·표본별 결과 미보존 | metric 독립 재계산·paired bootstrap 불가 | checkpoint/per-sample SHA-256 봉인 |
| 독립성 | test 1,133 patch 중 양성 423개가 모두 2019-03-15 Chimanimani event | 1,133 독립 사건으로 해석 불가 | region/canonical-event 단위 보고 |

## 원 benchmark 정정

공식 저장소 README의 IoU 0.4166(3D-UNet), 0.4474(U-TAE)는 Scientific Data 논문의 원 표가
아니라, 현재 저장소가 추가 제공하는 `S12LS-LD` binary benchmark다. 이 benchmark는 >50 positive
pixel 표본, random split, S2+DEM 15시점, 75 epoch BCEDice 설정이다. 우리 S12q LOCO와 직접 비교할
수 없다. Scientific Data 논문 자체는 50 epoch·cross-entropy 설정과 geographic cluster
leave-one-out 실험도 보고한다.

- 공식 저장소: https://github.com/PaulH97/Sen12Landslides
- 데이터 논문: https://www.nature.com/articles/s41597-025-06167-2

## 복구된 v2 계약

1. 6,834개 cache를 전수 검사해 exact file set, shape/dtype, embedding finite, raw reflectance
   `[0,10000]`, binary mask와 원 계약의 positive-pixel count, content SHA-256을 봉인한다.
2. seal이 없거나 하나라도 실패하면 학습을 거부한다.
3. arm마다 Python/NumPy/PyTorch/DataLoader RNG를 동일 seed로 독립 reset한다.
4. pos_weight는 5,542개 train mask 전체에서 계산한다.
5. val IoU로 best epoch을 고르고 checkpoint를 저장한다. test는 epoch 선택에 쓰지 않는다.
6. test/val은 모든 pixel exact AP, IoU/F1/precision/recall, pixel-micro ECE/Brier/NLL,
   positive-patch macro IoU를 내고 표본별 TP/FP/FN을 JSONL로 남긴다.
7. 결과에는 `development_only_not_confirmatory`와 test exposure를 기계 판독 가능하게 기록한다.

## 결과 계보

| 실행 | 상태 | 해석 |
|---|---|---|
| 8 epoch M23 | 보존·확정표 제외 | P4 all-metric win은 초기 개발 신호 |
| 기존 40 epoch | 보존·확정표 제외 | flawed AUPRC/RNG. P2 test IoU 0.1840 > P4 0.1440, P4 val IoU는 우세 |
| audit v2 | 보존·확정표 제외 | exact metric/RNG reset은 복구. P2 test IoU·AP 우세였지만 P4 replay IoU가 +16.8% 갈려 strict CUDA 재현성 실패 |
| strict smoke A/B | **통과** | metric/per-sample/checkpoint/tensor bitwise 동일, max-abs diff 0 |
| strict 40-epoch final | **개발 측정 완료** | artifact verifier 통과. P4-only 40-epoch metric/checkpoint/tensor bitwise 일치 |

### audit v2에서 보인 방향 — 최종 성능표가 아님

| arm | best China val IoU | Chimanimani test IoU | exact AP | F1 | positive-patch macro IoU | cached-head 초 |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 0.03885 | 0.060381 | 0.069175 | 0.113886 | 0.112544 | 334.7 |
| P2-tiny | 0.05795 | **0.139571** | **0.276568** | **0.244954** | **0.202185** | 668.3 |
| P4 frozen OLMo | **0.10893** | 0.122826 | 0.182354 | 0.218780 | 0.157508 | **286.3** |

P4는 P1보다 강하지만 P2-tiny 대비 test IoU 비율 88.0%, AP 비율 65.9%로 사전 95% 문턱보다
낮았다. 반대로 China val에서는 P4가 1위였다. 이는 한 region의 유의미한 이질성 신호지만,
P2가 공식 구조가 아니고 timestamp 계약도 다르며 CUDA replay까지 갈렸으므로 G-P를 통과/실패로
닫을 수 없다. 비용도 P4 head 286.3초만 보면 안 된다. embedding 전수 추출은 별도 **1,130.05초**,
10.75 GB였고 frozen encoder는 88.96M 파라미터다.

### strict final — 최종 코드 SHA `478c6af5…`

| arm | best China val IoU | Chimanimani IoU | exact AP | F1 | pos-patch macro IoU | LD IoU | head fit+val 초 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0.03685 | 0.054055 | 0.077737 | 0.102565 | 0.115962 | 0.14488 | **387.3** |
| P2-tiny-factorized | 0.06197 | 0.134989 | **0.286074** | 0.237869 | **0.195021** | 0.21327 | 1,455.7 |
| P4 frozen OLMo | **0.11181** | **0.141643** | 0.225115 | **0.248139** | 0.183479 | **0.23041** | 950.5 |

P4는 P1을 넘고 P2-tiny 대비 IoU 104.9%·F1 104.3%지만, threshold-free AP는 **78.7%**,
positive-patch macro IoU는 94.1%다. 사전 G-P는 IoU와 AUPRC 모두 95%를 요구하므로 AP 축에서
통과하지 못했다. 그러나 P2가 공식 strong baseline이 아니고 P3가 없으며 timestamp도 비대칭이라
이것은 **method 실패 판정도 아니다**. 정확한 상태는 `개발 viability 신호 있음 / G-P BLOCKED`다.

비용 순위도 “P4 전 지표·비용 1위”가 아니다. strict head 범위에서 P1이 가장 빠르고 메모리도
가장 작다. P4는 P2-tiny보다 빠르지만, cold OLMo extraction 1,130.05초·10.75 GB와 88.96M frozen
encoder를 포함하지 않은 값이다. artifact verifier는 세 checkpoint SHA, val/test JSONL SHA,
TP/FP/FN 합과 threshold 지표를 독립 재계산해 전부 통과했다. exact AP/ECE/Brier/NLL은 JSONL에
pixel score가 없어 verifier 범위 밖이며 strict evaluation replay로 검증했다. final full-run P4와
P4-only는 40 epoch history(시간 제외), val/test metric, per-sample SHA, checkpoint SHA와 모든 tensor가
bitwise 일치(max-abs diff 0)했다. 반면 cached fit+val 시간은 **950.5초 vs 520.0초**로 갈렸다.
따라서 정확도 재현성은 복구됐지만 비용은 randomized order·isolated repeat 전까지 비교 불가다.

## G-P를 실제로 닫는 다음 프로토콜

1. Chimanimani는 hyperparameter/debug 개발 fold로만 사용하고 primary confirmatory 평균에서 제외한다.
2. 공식 Sen12 3D-UNet·U-TAE를 같은 S12q/date contract로 구현하고 architecture parameter count와
   tensor regression test를 원 구현과 맞춘다.
   비교는 (a) 같은 loss/optimizer/search budget의 matched recipe와 (b) 저자 권장 native recipe를
   둘 다 두며, P4는 둘 중 더 강한 task baseline과 비교한다.
3. Chimanimani에서 고정한 loss·max epoch·early-stop·seed를 바꾸지 않고 아직 보지 않은 9지역을
   순서와 무관하게 실행한다. 각 outer test는 한 번만 연다.
4. primary는 region-macro IoU/AUPRC와 worst-region, canonical-event 보조 통계다. pixel micro는
   진단용이다.
   threshold-free AUPRC와 고정 0.5 IoU를 primary로 두고, val에서만 고른 threshold의 test IoU는
   secondary로 분리한다. test threshold tuning은 금지한다.
5. 비용은 cold cache extraction, cached head fit, deployment inference, task 수 K의 amortized
   Pareto로 보고한다.
6. 이 조건과 P3가 없으면 G-P 상태는 **BLOCKED**, CVPR 정확도 주장은 **0%**로 유지한다.
