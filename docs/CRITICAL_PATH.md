# 임계경로 한 장 — transferable · refreshable Earth embedding

갱신 2026-08-26. **이 문서가 실행 순서의 최종 근거다.** 자산의 실물 상태는
`docs/ASSET_INVENTORY.md`, arm·통계의 상세는 `docs/MOUNTAINSHIFT_EXPERIMENT_DESIGN.md`를 따른다.

## 북극성

> **한 번 계산해 여러 task가 공유하는 Earth embedding을 유지하면서, 지역·시점·센서·모델이
> 바뀔 때 task별 손실과 비용을 예측해 reuse / residual 추가 / refresh / recompute를 고른다.**

MountainShift는 이 방법을 만들기 위한 첫 과학 실험이다. 먼저 frozen OLMoEarth가 산사태 task에
실제로 쓸 만한지 판정하고, 그다음 공개 task-eligible 10지역→한국 전이에서 static/live evidence의 한계를 잰다.
FoldRefresh는 전체 논문의 목적이 아니라 `recompute` action 하나를 제공하는 기존 자산이다.

이 순서는 세 목표와 겹친다.

| 목표 | 남겨야 하는 증거 |
|---|---|
| AI2 취업 | OLMoEarth 실물 입력계약·release 재현·공개 benchmark·수정 가능한 코드 |
| 박사/CVPR | task/region/event 분리, 강한 baseline, 음성통제, 새 routing method |
| 비즈니스 | 정확도만이 아니라 GPU시간·저장량·응답지연을 포함한 refresh 의사결정 |

## 지금까지 작업 — 자산에서 실험으로

| 완료 | 의미 |
|---|---|
| M1·M8 release/mask 감사 | 같은 tag·release라도 cache 의미가 달라질 수 있음을 실측. refresh action의 근거 |
| M9·M10 한국 split 감사·봉인 | 공식 valid의 누수를 잡고 13 spatial cluster split을 동결. 한국 external test 기반 |
| M11 Sen12 접근·수신 | harmonized S2 **13,628 NetCDF / 38 GB** 확보. 공개 spine 실행 가능 |
| M12·M13 annotation 감사 | 저자 고정 후보 11지역, Italy는 annotation-shift arm으로 분리 |
| C0 task contract | 후보 중 Lanao 양성 0 → S12q headline **10지역**. label anomaly 2건 제외 봉인 |
| M16~M22 GK2A·ASOS | live residual 배관. 단, G-P와 static transfer 전에는 성능 기여로 세지 않음 |

과거 문서의 `Sen12 수신 없음`, `GPU 둘 다 점유`, `Italy→Korea가 headline`은 현재 사실이 아니다.
2026-08-25 재확인 시 **GPU0은 다른 프로젝트가 점유, GPU1은 0 MiB로 가용**했다. GPU 상태는
자산이 아니라 순간 상태이므로 매 실행 직전 다시 확인한다.

## task를 먼저 분리한다

같은 mask를 쓰더라도 세 질문은 다르다. 한 수치로 섞지 않는다.

| ID | task | 입력 시점 | 출력·지표 | 자리 |
|---|---|---|---|---|
| **S12q** | matched retrospective segmentation | SCL clear 상위 12개를 시간순 재정렬 | pixel mask · IoU/AUPRC/F1 | **G-P headline**. OLMo와 모든 baseline 동일 입력 |
| **S15-ref** | paper-reference segmentation | 15 timestep 전체 | pixel mask · IoU/AUPRC/F1 | Sen12 논문 수치 재현용. G-P 직접 비교 아님 |
| **S≤t** | cutoff-valid segmentation | cutoff까지 도착한 관측만 | pixel mask · IoU/AUPRC + lead time | operational/live 확장 |
| **R-event** | candidate retrieval | 같은 frozen cache | tile/event 순위 · Recall@K/nDCG | shared-cache 두 번째 task |

Sen12 `MASK`는 표본에서 **동일 event polygon이 15 timestep에 반복**되어 있었다. 따라서 15개
시점 라벨로 세지 않는다. S≤t에서 음성 patch는 event date가 없으므로 계절·지역을 맞춘
pseudo-cutoff 정책을 사전에 동결하기 전까지 열지 않는다. OLMo v1의 time embedding table은
12개라 15개 입력에서 shape error가 났다. 사후 편의 선택을 막기 위해 SCL quality만으로 top-12를
고르고 시간순으로 복원하는 S12q를 고정했으며 P1~P5 모두 같은 index를 쓴다.

**공정성 정정(2026-08-25)**: 같은 timestep index만으로 정보 계약이 완전히 같지는 않다. P4 cache는
OLMo에 실제 acquisition timestamp를 전달하지만 현재 P1/P2는 순서만 본다. matched G-P를 주장하려면
raw temporal baseline에도 같은 날짜/간격 encoding을 주거나 P4 timestamp ablation을 함께 둔다.

## 측정 사슬 — promotion gate를 통과해야 다음으로 간다

| # | 단계 | 무엇을 재는가 | 현재 |
|---|---|---|---|
| **0** | **C0 data contract** | 13,628파일 shape/band/time/static-mask/pre-post/SCL + LOCO 해시 | **통과** |
| **1a** | **G-P smoke** | GPU1, 64표본에서 OLMo 입력·메모리·cache runtime | **통과** |
| **1b** | **G-P full** | S12q에서 frozen OLMo가 matched task model의 95%에 닿는가 | 개발 pilot 측정 / **BLOCKED** |
| **1c** | **R-event probe** | 같은 cache가 retrieval에도 raw spectral보다 나은가 | 0% |
| **2** | **T-m** Höhn task-eligible 10지역 → Korea | annotation-matched zero/1/5/10% transfer | 0% |
| **3** | **T-x** Italy → Korea | annotation-mismatched transfer | 0% |
| **4** | **E_annotation** = T-m − T-x | 도화 기준 차이의 손실 | 0% |
| **5** | **E_static** | DEM/slope/기후평년을 더한 transfer 변화 | 0% |
| **6** | **E_live** | cutoff-valid 관측조건·강수 residual | 배관만 있음 |
| **7** | **R-cache** | task별 action 가치와 cost를 예측하는 router | 설계 후보 |
| **8** | external stress | 한국 untouched + 접근 통과 시 Swiss/Nepal | 한국 split만 봉인 |

0~1b 전에는 Italy/Korea 성능을 돌리지 않는다. **GeoFM 자체가 task에 부적합하면 residual의
성공도 실패도 해석할 수 없기 때문이다.** 5 전에는 6을 확장하지 않는다. 다만 GK2A는 2일만
보존되므로 `DAILY_OPS.md`의 최소 수집만 예외적으로 계속한다.

## C0와 G-P의 동결 계약

### C0 — CPU 전수 감사

`code/build_sen12_gp_contract.py`가 아래를 실물 NetCDF에서 검사한다.

- harmonized S2만 사용; `data_raw`와 혼합 금지
- 128×128, 15 timestep, B02--B12 10밴드 + SCL/MASK/DEM
- MASK 이진성·시간 불변성·`annotated` attr 일치
- pre/post index 범위·시간 순서·SCL clear fraction
- `center_lat/lon`은 값 범위와 CRS를 확인하기 전 위경도로 사용 금지
- annotation-matched 11지역 중 양·음성이 모두 있는 10지역을 outer test로 한 번씩 쓰는 10-fold.
  LanaoDelNorte 71개(양성 0)는 false-positive stress cohort로만 보존
- sample manifest와 각 fold train/val/test 목록의 SHA-256

C0 실측은 13,628/13,628 readable, 공통 schema gate 8/8 통과다. retrospective/R-event는
`hiroshima_s2_1427`, `hiroshima_s2_1428` 두 label anomaly를 fail-closed로 제외해
**13,626 전체 eligible / headline 6,834**로 봉인했다. 전체 양성의 단일 pre/post coverage는
5,397/6,737 = **80.11%**라 S≤t는 아직 열지 않는다.

### G-P — 비교가 공정해지는 최소 표

| arm | S12q segmentation | 목적 |
|---|---|---|
| P0 | all-negative / prevalence predictor | 불균형 하한 |
| P1 | raw spectral shallow U-Net | embedding 불필요 가능성 |
| P2 | **공식 Sen12 3D-UNet 구조** | 동일 12시점·날짜 encoding으로 재학습. tiny stand-in 금지 |
| P3 | **공식 U-TAE 구조** | 다른 temporal inductive bias. acquisition date를 동일 계약으로 사용 |
| P4 | **frozen OLMo v1 + 같은 용량의 spatial decoder** | G-P 대상 |
| P5 | frozen Prithvi-EO-2.0 + 같은 decoder | OLMo 한정 여부와 최신 근접연구 대조 |

OLMo v1.2는 12밴드 단일 group이고 Sen12는 B01·B09가 없는 10밴드라 v1과 대칭 입력이 안 된다.
따라서 **첫 task 자격 시험은 v1만** 쓴다. v1.2는 band imputation 실험이 아니라 입력계약 연구
(M8/FoldRefresh)에 남긴다. P4/P5의 decoder 깊이·parameter 수·학습 epoch·augmentation은 기록한다.

비용은 ① cold encoder/cache extraction ② cached head fit+validation ③ deployment inference로
분리한다. `trainable_params`를 쓸 때도 P4의 frozen encoder 총 파라미터와 10.75 GB cache를 함께
보고한다. task 수 `K`에 따른 amortized 표 전에는 end-to-end 비용 우위를 주장하지 않는다.

1 run 시간을 smoke에서 먼저 잰다. 그 값으로 전체 fold×seed 예산을 동결하며, 실행 중 성능을 보고
seed나 fold를 줄이지 않는다.

**G-P smoke 실측**: 10지역·양/음성 32/32의 64표본, 256 crop이 15.44초(모델 로드 4.75초 제외),
**4.146 sample/s**, peak CUDA **0.740 GB**, fp16 spatial cache **1,572,992 B/sample**, 첫 표본
재실행 max-abs diff **0.0**, shape 64/64 `768×32×32`, finite 64/64로 6/6 gate를 통과했다.
이 속도를 단순 외삽하면 headline 6,834개는 약 **27.5분 / 10.75 GB**다. 이는 embedding extraction
예산이지 decoder 학습시간이 아니며 pilot에서 따로 잰다.

**strict 개발 pilot 실측(M25)**: full cache extraction은 실제 1,130.05초(18.8분)였다. 이미 열람한
Chimanimani 한 fold에서 P4는 P1을 넘었고, P2-tiny 대비 test IoU 0.14164 vs 0.13499였지만 exact
AP는 0.22512 vs 0.28607(78.7%)였다. 따라서 viability 신호는 있으나 G-P의 IoU+AUPRC 95%를
동시에 통과하지 못했다. P2가 공식 구조가 아니고 P3·timestamp parity·미열람 9지역이 없으므로
최종 판정은 실패가 아니라 **BLOCKED**다. strict CUDA smoke는 checkpoint tensor max-abs diff 0,
artifact verifier는 checkpoint/per-sample SHA와 threshold aggregate 전부 통과했다. final P4-only
40-epoch replay도 metric/checkpoint/tensor가 bitwise 일치했지만 wall time은 950.5초 vs 520.0초로
갈렸으므로 G-C 비용은 isolated 반복 실험으로 다시 잰다.

## 결정성 × 공식성 — M26으로 해소됨

strict 모드에서 막히는 것은 **pooling backward 3개**(`max_pool3d`, `avg_pool3d`,
`adaptive_avg_pool3d`)뿐이다. `conv3d` stride 2의 backward는 결정적이고, U-TAE의
temporal attention과 SDPA도 결정적이다.

따라서 **선택지 C**를 택한다 — 공식 구조를 유지하고 **pooling만 결정적 연산으로 치환**한다.

| 선택지 | 공식성 | 결정성 | 판정 |
|---|---|---|---|
| A 경고 모드 후퇴 | O | X | 재현성 상실. 채택 안 함 |
| B tiny 분해 유지 | X | O | M25의 `BLOCKED` 원인. 유지 안 함 |
| **C deterministic-safe 재구현** | **O(치환 명시)** | **O** | **채택** |

C의 의무사항: 채널·depth·파라미터 수를 공식 config와 맞추고, 바꾼 연산과 이유를 산출물에
기록하며, **"공식과 동일"이라고 쓰지 않는다.** P3(U-TAE)까지 넣어야 게이트의
`max(P2,P3)`가 성립한다.

## 판정 기준

| gate | 통과 조건 | 실패 시 |
|---|---|---|
| C0 | core contract 전 파일 통과 + LOCO seal | 오류 유형·지역을 분리하고 고친 뒤 재봉인. 성능 실행 금지 |
| G-P | S12q region-macro IoU와 AUPRC에서 P4가 max(P2,P3)의 **95% 이상**, P1보다 우수, catastrophic fold 없음 | OLMo를 MountainShift backbone으로 쓰지 않음 |
| G-R | R-event에서 pooled OLMo가 raw spectral보다 우수 | “다중 task 공유 cache” 주장을 보류 |
| G-T | T-m zero/few-shot 곡선이 local-only/naive pooling보다 우수 | transfer 주장을 접고 Korea를 독립 배치 사례로만 보고 |
| G-S | E_static ≥+2%p, region CI 하한>0, worst-region 저하≤1%p | static residual method 주장 철회 |
| G-N | region shuffle / spatial shift / time shift에서 이득 소멸 | leakage로 판정하고 해당 주장 철회 |
| G-L | 미래정보 0건, cutoff provenance ≥95%, accuracy 또는 lead-time 개선 | live는 inference-fusion 배관으로만 보고 |
| G-CV | R-cache가 no-router/uncertainty-only/fixed schedule보다 accuracy-cost Pareto 우위 | system/benchmark 또는 워크숍으로 축소 |

논문에 보고할 독립 단위는 타일이 아니라 **region과 canonical event**다. pixel pooled 수치와 함께
region/event macro, fold 원자료, paired bootstrap 또는 randomization interval을 모두 둔다.

## 2026-08-25 prior-art 충돌과 CVPR 경로

`frozen VFM + terrain/material/rainfall residual`은 더 이상 충분한 novelty가 아니다.
2026-08-10 공개된 **GeoPhysAdapter**가 이미 Prithvi를 고정하고, 4개 공개 source·55 event·7,890
test sample에서 terrain/material/trigger를 pixel/object scale로 제한하며 misalignment·time-shift·
cross-anchor 통제까지 수행했다. 따라서 A3/A4가 좋아지는 것만으로는 CVPR method claim을 하지 않는다.

남아 있는 더 강한 질문은 아래다.

> task `q`, cache 상태 `c`, 새 관측/모델 변경 `d`, 비용 budget `B`가 있을 때,
> `reuse / add_static / refresh_live / recompute_release` 중 어느 action이 downstream loss를 가장
> 많이 줄이는가?

router는 `Δloss(q, action | drift, freshness, quality)`와 action cost를 예측한다. 평가는 단순 F1이
아니라 **oracle 대비 regret, calibration, GPU시간·storage·latency를 포함한 Pareto frontier**다.
FoldRefresh는 `recompute_release`, MountainShift는 `add_static/refresh_live`, M1·M8은 drift feature를
제공한다. 이 결합이 검증될 때만 세 프레임이 하나의 CVPR 방법이 된다.

## 지금 하지 않을 것

- C0/G-P 전에 264-run 전체 matrix 실행
- static/live feature를 그냥 10 m로 broadcast하고 novelty라고 주장
- GPU0 사용
- C0 전에 한국·Italy 결과로 하이퍼파라미터 선택
- 네팔·스위스를 headline 독립표본으로 과장
- OLMo v1.2의 빠진 B01·B09를 사후 편의대로 채우기

## 다음 세 행동

1. 공식 Sen12 3D U-Net·U-TAE를 원 구현 tensor/parameter regression test와 함께 이식한다.
2. raw baseline에 동일 acquisition date/gap encoding을 주고 Chimanimani에서 recipe를 동결한다.
3. 미열람 9지역을 한 번씩 열어 region/event-macro G-P와 G-R을 닫은 뒤 한국 T-m/T-x를 연다.

## 이번 재설계에 직접 사용한 공개 근거

- [Sen12Landslides data descriptor](https://www.nature.com/articles/s41597-025-06167-2) — 15시점,
  3D-UNet/U-TAE/U-ConvLSTM baseline과 기존 stratified split의 범위
- [GeoPhysAdapter](https://arxiv.org/abs/2608.09325) — frozen Prithvi + scale-matched physical prior가
  이미 직접 경쟁선임
- [rs-embed](https://arxiv.org/abs/2602.23678) — multi-model embedding 생성·표준화·batch cache는
  이미 존재하며, 우리의 차별점은 task-risk 기반 refresh decision이어야 함
- [How to Embed Matters](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Gilch_How_to_Embed_Matters_Evaluation_of_EO_Embedding_Design_Choices_CVPRW_2026_paper.html)
  — 한 embedding을 여러 task에 재사용하는 효율성은 중요하지만 task별 설계 차이가 성능을 바꿈
