# 임계경로 한 장 — transferable · refreshable Earth embedding

갱신 2026-09-01. **이 문서가 실행 순서의 최종 근거다.** 자산의 실물 상태는
`docs/ASSET_INVENTORY.md`, arm·통계의 상세는 `docs/MOUNTAINSHIFT_EXPERIMENT_DESIGN.md`를 따른다.

> **재시작 경계:** M65/`4862483` 이후 Nepal 전용 구현과 데이터는 sibling 저장소로 분리했다.
> 이 저장소의 활성 사슬은 1c Presto C1b native sensitivity → GeoContextGate development → nested label-budget audit → Korea
> input/split preflight → Korea untouched transfer다. Nepal은 외부 stress/operations sidecar이며
> 이 사슬의 선행조건이 아니다. MS-86 mechanism audit은 기존 72개 결과만 사용해 완료했으므로
> 새 GPU 실험이나 confirmatory evidence로 세지 않는다.

> **2026-09-01 MS-87:** C1a common-grid 6,834 cache·8지역×3seed 완료. C1a `.1092`는
> P4 `.2722`와 P2 `.1966`보다 8/8 지역에서 낮았다. 이는 효과가 아무 frozen GeoFM에나
> 생기지 않음을 보이나 Presto off-domain 계약·4×4 pooling·retrospective 비교이므로 universal
> OLMo superiority가 아니다. C1b와 Korea first-look가 남았다.

> **C1b 실행 경계:** 기존 P4 decoder는 32² 입력을 두 번 확대하도록 고정돼 있어 native 128²에
> 그대로 쓰면 512² 중간 표현을 만든 뒤 다시 128²로 축소한다. C1b는 같은 trainable layer를
> interpolation 없이 native grid에서 실행하는 `P4native`로만 수행한다. exact input shape,
> P4-parameter parity, native cache seal, immutable source snapshot, 새 OUTROOT가 모두 통과해야 GPU1을 쓴다.

> **2026-08-31 claim 교정:** P4는 새 2단 method가 아니라 frozen-cache arm이고, M57–M58의
> confirmatory 절차는 provenance protocol이다. P2는 SOTA가 아니라 matched official-architecture
> baseline이다. 한국 polygon 면적 유사성은 annotation equivalence를 증명하지 않으므로
> `T-m−T-x=annotation effect` 해석은 폐기한다. 현재 논문 SSOT는
> `docs/PAPER_NARRATIVE_2026_08_31.md`다.

> **2026-08-28 상태 정정:** frozen-v2 확증은 **8/8 지역·72실행**을 모두 마쳤고 post gate를
> 통과했다. 주지표 region-macro는 P4 reuse **.2722**, P2 raw-strong .1966, P3 raw-efficient
> .1834이며 P4−P2는 **+.0756**이다. 사전등록 per-region win은 6/8, strong-win은 5/8이다.
> Indonesia는 P4가 졌고 Itogon은 all-seed 규칙을 실패했으므로 지역 gain 이질성도 남았다.
> 이것은 frozen OLMoEarth의 **transfer viability**를 닫지만 OLMo 고유 우월성이나 label-free
> routing은 아니다. 다음 병목은 Presto matched control과 한국 untouched transfer다. Nepal live
> event는 `NEPAL_SIDECAR_HANDOFF_2026_08_28.md`에 주차했다.

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

**공정성 정정의 정정(2026-08-26, M39)**: "day·관측 간격이 정렬되지 않았다"는 우려를 실측으로
확인한 결과 **비대칭이 존재하지 않았다.** OLMo wrapper(`use_legacy_timestamps=False`)는 월을
보존한 채 날짜를 ±1~3일 옮겨도 임베딩이 5/5 비트 단위로 동일했다 — 시간 정보를 **월 해상도로
양자화**한다. 따라서 month 채널을 받는 P2/P3는 이미 인코더와 같은 시간 해상도를 가지며,
exact date/gap 정렬 실험은 없는 문제를 고치는 것이므로 하지 않는다. 남는 비대칭은 정보량이
아니라 부호화 형태(sinusoidal PE vs 스칼라 broadcast)뿐이고, 이는 별도 ablation 항목으로 둔다. 기존 pilot JSON의 `P1/P2 only receive order` 문구는 stale이며
`evidence/gp_official_bundle/bundle_manifest_v2.json`에서 정정했다.

## 측정 사슬 — promotion gate를 통과해야 다음으로 간다

| # | 단계 | 무엇을 재는가 | 현재 |
|---|---|---|---|
| **0** | **C0 data contract** | 13,628파일 shape/band/time/static-mask/pre-post/SCL + LOCO 해시 | **통과** |
| **1a** | **G-P smoke** | GPU1, 64표본에서 OLMo 입력·메모리·cache runtime | **통과** |
| **1b** | **G-P full** | S12q에서 frozen OLMo가 matched task model에 지역 전이되는가 | 개발 95% gate **FAIL(82.0%)**; 확증 8-region macro **P4 .2722 > P2 .1966**, 6/8 win — **완료** |
| **1c** | **C1 second GeoFM** | 같은 S12q·decoder에서 효과가 아무 frozen GeoFM에나 생기는가 | **C1a 완료 MS-87** — Presto `.1092`, P4/P2에 8/8 패배; C1b `P4native` runner CPU 준비, 서버 preflight·실행 남음 |
| **1c-0** | **mechanism audit** | 기존 72 test JSONL에서 8지역 FP·P2/P4 상보성이 재현되는가 | **완료 MS-86** — FP 7/8·중앙 5.02×, oracle +.02375·5/8 ≥.02 |
| **1d** | **R-event probe** | 같은 cache가 retrieval에도 raw spectral보다 나은가 | P@10 masked .538 > raw .432이나 사전 2×-base gate **FAIL**; 기존 AP@100 철회·재실행 대기 |
| **2** | **Korea contract gate** | input grid와 landslide ontology/time/provenance가 source task와 연결 가능한가 | split만 봉인; v2 mosaic·label-equivalence 미완 |
| **3** | **Korea external transfer** | joint geographic/dataset shift에서 zero/1/5/10% transfer | 0% |
| **4** | **Italy annotation stress** | 다른 annotator/MMU에서 성능이 얼마나 민감한가 | 0%; 인과적 annotation effect로 부르지 않음 |
| **5** | **E_static** | DEM/slope/기후평년을 더한 transfer 변화 | 0% |
| **6** | **E_live** | cutoff-valid 관측조건·강수 residual | 배관만 있음 |
| **7** | **R-cache** | task별 action 가치와 cost를 예측하는 router | 설계 후보 |
| **8** | external stress | 한국 + 접근·label provenance 통과 시 Nepal/Swiss | 한국 split 봉인; Nepal prospective input은 live seal **HOLD(S1 3/4)** |

0~1b는 닫혔다. Korea test는 C1 cache/decoder recipe와 한국 입력·split gate를 동결하기 전에는
열지 않는다. **GeoFM 자체가 task에 부적합하면 residual의 성공도 실패도 해석할 수 없기 때문이다.**
5 전에는 6을 확장하지 않는다. 다만 GK2A는 2일만
보존되므로 `DAILY_OPS.md`의 최소 수집만 예외적으로 계속한다.

Nepal 2026 live event는 새 데이터의 publication/selection/seal 지연을 prospectively 기록한
시간 제한 sidecar다. 현재 baseline/placebo embedding만 있고 `live_mode=null`, live cube는
S1 3/4라 invalid다. 본 queue에서는 주차했으며 `docs/NEPAL_SIDECAR_HANDOFF_2026_08_28.md`의
네 재개 조건을 모두 만족할 때만 별도로 연다.

## C0와 G-P의 동결 계약

### C0 — CPU 전수 감사

`code/build_sen12_gp_contract.py`가 아래를 실물 NetCDF에서 검사한다.

- harmonized S2만 사용; `data_raw`와 혼합 금지
- 128×128, 15 timestep, B02--B12 10밴드 + SCL/MASK/DEM
- MASK 이진성·시간 불변성·`annotated` attr 일치
- pre/post index 범위·시간 순서·SCL clear fraction
- `center_lat/lon`은 값 범위와 CRS를 확인하기 전 위경도로 사용 금지
- 동일 저자 annotation 후보 11지역 중 양·음성이 모두 있는 10지역을 outer test로 한 번씩 쓰는 10-fold.
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
| P4 | **frozen OLMo v1 last-layer cache + small spatial decoder** | 원래 G-P 대상; 237,537 params |
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

**공식 4-arm 개발 fold 실측(M30)**: cache extraction은 1,130.05초였다. 같은 40-epoch 실행에서
P2 official-safe UNet3D IoU/AUPRC는 **0.159254/0.174585**, P3 U-TAE는
0.120554/0.166852, P4 frozen last-layer cache는 **0.130582/0.151348**이었다.
P4/P2 IoU 비율은 **82.0%**로 사전 95% 게이트에 실패했고, P4가 이긴 segmentation 성능 지표는
없다(ECE만 더 낮음). 따라서 첫 fold의 frozen-small recipe는 FAIL이다. 이것은 OLMoEarth 전체나
adaptation 가능성을 기각하지 않고, **last-layer frozen cache + 해당 decoder 조합의 실패**를 뜻한다.
Chimanimani test는 이미 노출된 development fold이므로 E1 원인진단 결과도 확증으로 승격하지 않는다.
증거 번들은 per-sample·로그·checkpoint SHA를 봉인했고, v2 manifest가 blank-count·timing 문구를 정정한다.

## 2026-08-26 재구성 — 이야기의 축을 성능 우열에서 **판단 문제**로 옮긴다

### 지금 유의미한 사실은 metric 교차가 아니라 recipe 의존성이다

| 개발 evidence | IoU | AUPRC | 해석 |
|---|---:|---:|---|
| P2 official-safe UNet3D | **0.1593** | **0.1746** | 현재 strong raw baseline |
| P4 frozen last-layer + small decoder | 0.1306 | 0.1513 | 두 성능지표 모두 열세 |
| P4c frozen last-layer + large decoder | 0.1777 | 0.2136 | E1의 한 노출된 개발 셀; positive-patch macro·LD-IoU는 P2보다 낮음 |

P4-small에는 metric별 승자 교차가 없다. P4c의 회복 신호도 한 개발 지역·한 seed에서 decoder
용량을 바꾼 결과일 뿐, task별 위험 이질성이나 router 필요성의 증거가 아니다. 지금 유의미한 것은
**같은 embedding도 serving context와 decoder recipe에 따라 판정이 뒤집힐 수 있으므로 먼저 공정한
representation action을 동결해야 한다**는 점이다. RQ2는 이후 같은 목적함수에서 task별 action
gain의 순위가 실제로 달라지는지를 별도로 측정해야 한다.

### 논문의 사슬 — `기록`은 마지막 단계가 아니라 모든 단계의 증거층이다

```
관측된 실패
   ↓
비용이 걸린 의사결정 문제
   ↓
반증 가능한 가설
   ↓
가설을 구별하는 실험
   ↓
PASS / FAIL / BLOCKED 판정
   ↓
다음 질문 또는 방법 설계

각 단계 옆에 코드 · 해시 · split · checkpoint · 결과
```

논문 본문은 인과를 말하고, 저장소는 그 주장을 재현할 수 있게 한다.

### 의사결정 문제 — 현재는 두 극단뿐이다

```
전부 재사용    싸다 · 일부 task가 망가질 수 있다
전부 재계산    안전하다 · 느리고 비싸다
```

> **모델·지역·시점·센서가 변했을 때, task별 성능 위험을 사전에 예측해
> 어떤 embedding cache를 언제 갱신하는가?**

이것이 `EarthRoute`의 본체다. `FoldRefresh`는 전체 문제가 아니라 router가 고를 수 있는
**하나의 복구 action**이다.

```
reuse cache  →  cheap recalibration  →  partial refresh
             →  FoldRefresh repair   →  full re-embedding
```

## 실험 질문 5개 — 이 다섯 개로 고정한다

| RQ | 질문 | 현재 | kill 조건 |
|---|---|---|---|
| **RQ1** | cached embedding이 쓸 만한가 | frozen-small은 개발 gate FAIL. E1에서 tiled-large만 micro/AP 회복, full-context는 악화하고 positive-macro·비용 Pareto는 미회복 | exact-time·공통 seed 뒤에도 tiled-large가 불안정하면 multi-level/PEFT 한 축; 그래도 밀리면 backbone 제외 |
| **RQ2** | **위험이 정말 task별로 다른가** | **0%** | **모든 task가 비슷하게 망가지면 router 불필요 → method 논문 중단** |
| **RQ3** | target label 없이 task별 **action gain**을 사전 예측할 수 있는가 | 0% | contract·uncertainty·GdScore·ODD·agreement보다 regret이 낮지 않으면 method 중단 |
| **RQ4** | shared extraction cost까지 넣은 router가 정확도–비용 Pareto를 개선하는가 | 0% | oracle 대비 regret이 fixed refresh/best-single-action보다 나쁘면 system/benchmark로 강등 |
| **RQ5** | 다른 지역·모델·shift family로 transfer되는가 | 0% | 외부 지역에서 이득 소멸이면 한국/특정 backbone으로 축소 |

**RQ2가 사슬의 하중을 진다.** 여기서 이질성이 없으면 뒤의 셋이 전부 무의미해진다.

### RQ3–RQ4의 정확한 문제 계약 — generic router가 아니다 (2026-08-26)

최신 EarthShift·GEO-Bench-2·CrossEarth-Gate·DARN·GdScore·ODD·IUPM과 대조한 뒤 남는 질문은
`현재 성능을 추정`하거나 `PEFT module을 고른다`가 아니다.

> source/development label로 학습하되 새 target region/window label은 action 선택 때 보지 않고,
> `reuse / repair / re-embed / task-raw`의 task별 metric gain을 예측해 여러 task가 공유하는
> representation extraction 비용을 공동 배분할 수 있는가?

- action 단위는 pixel이 아니라 `spatial block × observation window`다.
- target label은 사후 utility/regret 평가에만 쓴다. `no labels anywhere`라고 쓰지 않는다.
- representation 재추출은 task 수 `K`에 한 번만 부과하고 head/raw 비용은 task별로 부과한다.
- 단순 seam·drift·entropy는 feature/baseline이지 method가 아니다.
- predictor support 밖에서는 `abstain / small audit-label request`를 별도 safety action으로 둔다.
- action set을 test 전에 hash로 봉인한다. 후보가 많아질수록 oracle gain이 자동으로 커지기 때문이다.

수식·claim ladder·강한 baseline·외부 지역 선택은
`docs/PAPER_CLAIM_EXPANSION_2026_08_26.md`가 SSOT다.

### RQ2의 데이터 제약 — 실측으로 확인했다 (2026-08-26)

RQ2는 **같은 타일에 여러 task 라벨**이 필요하다. 어디에 있는지 재봤다.

| 데이터셋 | land-cover | 벌목 | 산사태 |
|---|---|---|---|
| **Sen12Landslides** | 없음 | 없음 | 이진 마스크 1종만 (`label_positive`, `mask_*`) |
| **AI-Hub 71363** | 산림 541 · 밭 300 · 건물 287 · 도로 202 타일 | **167 타일** | **90 타일** |

**즉 Sen12만으로 RQ2는 불가능하고, 현재 확보 자산 중 AI-Hub가 유일한 3-task 후보**다. 다만 M35에서
v1 cube의 24.6% 심각한 0-fill과 선택편향을 확인했으므로, `T1 land-cover / T2 deforestation /
T3 landslide`는 v2 target-grid mosaic·12-band common coverage ≥99.9%·task별 제외율 gate를 통과한
표본에서만 성립한다. 아래 숫자는 label inventory이지 experiment-eligible count가 아니다.

동시에 **표본이 얇다** — 희소 task가 벌목 167 / 산사태 90 타일이다(전 군집 합계).
따라서 RQ2는 실행 가능하지만 **지역 단위 CI가 넓게 나올 것을 미리 인정하고**,
효과 크기를 절대값이 아니라 **task 사이 순위 역전**으로 판정한다.

### RQ3의 데이터 제약

RQ3의 정답은 **실제 downstream 하락량**이므로 shift 전후 **양쪽에 라벨**이 필요하다.
릴리스 shift(v1→v1.2)는 같은 라벨을 쓰므로 가능하고, 시점·센서 shift는 라벨이 시점마다
있어야 한다. AI-Hub는 타일당 1~8 날짜가 있어 시점 축이 가능하다.

## 감사 예산 규칙 (2026-08-26 신설)

측정·감사 항목은 M37까지 쌓였지만 method 결과는 아직 0개다. 이 비율을 유지하면 감사 논문이 된다.

> **main claim을 막지 않는 감사 항목은 새로 열지 않는다.**

새 감사를 열려면 `RQ1~RQ5 중 어느 것을 막고 있는가`를 먼저 적는다. 예외는
`DAILY_OPS.md`의 GK2A 수집뿐이다(2일 보존이라 미루면 소실).

## 현재 위치 — 정확히

| 작업 | 논문에서의 역할 | 상태 |
|---|---|---|
| split·SHA·LOCO·cache audit | 결과 신뢰성 기반 | 완료 |
| 잘못된 AP sampling 발견 | 허위 결론 방지 | 완료 |
| deterministic replay (bitwise) | 재현성 기반 | 완료 |
| P4 frozen OLMo pilot | cache viability | **원래 게이트 FAIL(82.0%)** |
| 공식 P2/P3 정렬 | 경쟁력 판정 | **완료(official-safe 이식)** |
| E1 crop-context × decoder capacity | 실패 원인 후보 분리 | **완료: full-context 음의 효과, decoder 부호 반전(M37)** |
| exact timestamp parity | 정보량 정렬 | **미완 ← 잔여 confound** |
| unseen 9지역 confirmatory | 지역 일반화 | **미완 ← 병목** |
| 3 task 위험 이질성 (RQ2) | router 필요성 | 미측정 |
| router Pareto (RQ4) | method contribution | 미측정 |
| 한국→Nepal 또는 Switzerland 한 곳 (RQ5) | external transfer | 미측정 |

지금까지의 감사는 헛일이 아니다 — 잘못된 "P4 전 지표 1위"를 제거하고 frozen-cache failure를
재현 가능한 결과로 만들었다. **현재 병목은 exact-time confound를 제거한 뒤 tiled-large와 strong
raw baseline을 공통 seed로 확인해 하나의 recipe를 개발 fold에서 동결하고, 미열람 지역을 더 이상
튜닝하지 않고 평가하는 것**이다.

## 실행 순서 (확정)

1. ~~raw arm exact date/gap 정렬~~ **M39로 해소** — 인코더가 월 해상도로 양자화하므로
   비대칭이 없다. 부호화 형태 ablation만 후순위로 남긴다.
2. E1의 tiled-large와 strong P2를 같은 seed 집합으로 반복하고, fixed threshold와 val-selected
   threshold를 구분해 recipe를 동결한다. full-context는 이 개발 계약에서 중단한다.
3. tiled-large가 positive-macro·비용까지 회복하지 못하면 **multi-level decoder 또는 PEFT 중 한 축만**
   열고 Chimanimani development recipe를 끝낸다. 여러 축을 동시에 사후 탐색하지 않는다.
4. 최종 recipe·seed·metric·중단 규칙을 사전등록하고 **미열람 지역을 순차 공개**한다.
5. AI-Hub v2 40표본 pilot→전수 health/selection-bias gate를 통과시킨다. v1 2,539큐브는 쓰지 않는다.
6. 같은 유효 cache 위에 AI-Hub 3 task를 구축해 RQ2의 shift×task degradation matrix를 잰다.
7. contract-only·entropy·drift·GdScore·ODD·agreement와 action-gain predictor를 tournament한다.
8. **shared extraction cost를 포함한 joint router + accuracy–cost Pareto**, 두 번째 backbone 순으로 연다.
9. recipe를 모두 동결한 뒤 Nepal 또는 Switzerland 한 곳만 external transfer로 공개한다. Nepal
   Koshi 2024는 U-Net-assisted + manual-QC silver label이므로 `untouched geography`와 `untouched gold`
   를 구분하고, 수동 adjudication subset 없이는 후자를 주장하지 않는다. swissEO 7-band를 쓰면
   regional transfer가 아니라 missing-band/source contract shift로 기록한다.

**실험 확장(2026-08-26 계획 승인)**: 위 순서의 2~8번을 E5(전제 완결: seed 폭·확률맵·FP율
정합·recipe 동결) → E6(Sen12 action matrix v1) → E7(public twin: R-event 승격 + PANGAEA 감사)
→ E8(AI-Hub v2 3-task) → E9(label-free predictor tournament) → E10(Gym 명세·post-training
타당성, 문서만)으로 구체화했다. 상세는 `docs/EARTHROUTE_GYM_SPEC.md`,
`docs/POSTTRAINING_FEASIBILITY.md`. LLM 학습은 이번 사이클에서 하지 않는다(사용자 결정).

**논문 질문 재설정 (2026-08-27, M61)**: "OlmoEarth frozen cache가 좋은가"는 약함 —
큰 EO 사전학습 모델이 작은 scratch baseline을 이기는 것은 예상 범위임.
질문을 **"지역이 바뀌었을 때 frozen / 경량 post-training / full fine-tuning 중 무엇이
정확도·오경보·비용에서 유리한가"**로 바꿈. 최우선 미착수 항목은 **C: 다른 frozen GeoFM과의
동일 decoder 비교** — 이것이 없으면 OLMo 고유 효과인지 일반 GeoFM 효과인지 말할 수 없음.
label budget 축(1/5/10/100%)을 더해 transfer frontier로 만듦. 상세는 M61.

지역 자산의 model-input/context/target 분리와 upstream 기여 연결은
`docs/OLMO_EXTERNAL_DATA_ONBOARDING_AND_PR_AUDIT_2026_08_26.md`가 SSOT다.

CVPR method paper가 되려면
`task별 위험 이질성 → 위험 예측 → refresh 의사결정 → 정확도·비용 Pareto → 외부 지역·두 번째 모델`
이 **하나의 인과 사슬로 닫혀야** 한다. 현재는
`좋은 재현성 감사 + 한 frozen-cache recipe의 반증 + 원인진단` 단계다.

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

**2026-08-26 추가 충돌**: OLMoEarth 원 논문·PANGAEA·PEFT 연구는 decoder/multi-level feature와
fine-tuning recipe가 frozen GFM 판정에 결정적임을 이미 보여준다. AlphaEarth·TESSERA·OlmoEarth
Studio는 shared embeddings-as-data를, TESSERA v2는 multi-task storage frontier와 seam artifact를,
Berkeley RALF는 downstream regret 기반 feature refresh를 이미 다룬다. 따라서 decoder 회복,
shared cache, seam, generic refresh는 headline novelty가 아니다. 남는 gap과 인용은
`docs/RECENT_LITERATURE_DECISION_2026_08_26.md`에 고정했다.

## 지금 하지 않을 것

- C0/G-P 전에 264-run 전체 matrix 실행
- static/live feature를 그냥 10 m로 broadcast하고 novelty라고 주장
- GPU0 사용
- C0 전에 한국·Italy 결과로 하이퍼파라미터 선택
- 네팔·스위스를 headline 독립표본으로 과장
- OLMo v1.2의 빠진 B01·B09를 사후 편의대로 채우기

## 다음 세 행동

1. exact date/gap parity를 닫고 tiled-large 대 P2를 공통 seed·공통 threshold-selection 규칙으로 반복한다.
2. 그 recipe가 positive-macro·비용까지 회복하지 못할 때만 multi-level decoder 또는 PEFT 한 축을 연다.
3. recipe를 동결하고 미열람 지역을 순차 평가한 후에만 AI-Hub v2/RQ2를 연다.

## 이번 재설계에 직접 사용한 공개 근거

- [Sen12Landslides data descriptor](https://www.nature.com/articles/s41597-025-06167-2) — 15시점,
  3D-UNet/U-TAE/U-ConvLSTM baseline과 기존 stratified split의 범위
- [GeoPhysAdapter](https://arxiv.org/abs/2608.09325) — frozen Prithvi + scale-matched physical prior가
  이미 직접 경쟁선임
- [rs-embed](https://arxiv.org/abs/2602.23678) — multi-model embedding 생성·표준화·batch cache는
  이미 존재하며, 우리의 차별점은 task-risk 기반 refresh decision이어야 함
- [How to Embed Matters](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Gilch_How_to_Embed_Matters_Evaluation_of_EO_Embedding_Design_Choices_CVPRW_2026_paper.html)
  — 한 embedding을 여러 task에 재사용하는 효율성은 중요하지만 task별 설계 차이가 성능을 바꿈
