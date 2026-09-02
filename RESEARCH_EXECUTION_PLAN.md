# 연구 실행계획 — MountainShift 우선순위 교정

갱신일: 2026-09-02
실행 자원: **H200 GPU1 전용** (2026-08-25 사용자 지시로 변경. 이전 표기는 GPU0였음).
모든 학습·추론은 `CUDA_VISIBLE_DEVICES=1`. GPU0은 건드리지 않는다.

> 활성 과학 기준점은 MS-93/`0dc68c3`. Nepal 전용 구현·데이터는 sibling 저장소로 이관됐으며
> 아래 R1–R6 queue를 선점하지 않는다. `RESTART_HERE.md`가 새 세션의 첫 문서다.

## 2026-08-28 실행 queue — transfer 본선 복귀

> **현재 질문:** 8-region에서 확인된 frozen OLMoEarth의 지역 전이 이득이 일반 frozen GeoFM
> 효과인지 OLMo 고유 효과인지 matched Presto 대조로 분리하고, 같은 recipe를 한국 spatial
> holdout에 처음 적용했을 때 유지되는가?

완료된 전제:

- frozen-v2 8-region 72실행·post gate 완료: P4/P2/P3 region-macro .2722/.1966/.1834,
  P4−P2 +.0756, per-region win 6/8(M65).
- Presto feasibility 8/8, upstream commit·code·weight·normalization `/10000` byte match, exact 12-month
  vector API 확인. `config/presto_c1_contract.json`으로 봉인.
- MS-87 C1a 완료: 6,834 cache seal, common-grid 8지역×3seed. C1a `.1092`, P4 `.2722`, P2 `.1966`.
- MS-93 C1b 완료: native-grid `.1261`; P4/P2에 8/8 패배. 24/24 원시 결과와 snapshot hash를
  compact artifact로 봉인했다.
- MS-90B/91/92 완료: 고정 융합과 learned gate 모두 불통과. FP-matched oracle headroom도 사라져
  stop rule 발동, v3 없음.
- Nepal live work는 `docs/NEPAL_SIDECAR_HANDOFF_2026_08_28.md`에 주차. `live_mode=null`이며
  CVPR queue를 대체하거나 선점하지 않는다.

| 순서 | 산출물 | 사전 gate / 중단 조건 |
|---|---|---|
| **R0 완료** | 8-region aggregate + M65 | 8 post manifest PASS, recipe SHA 일치, equal-region macro |
| **R0.5 완료** | 72 per-sample CPU mechanism audit(MS-86) | P4 empty-FP 7/8·중앙 5.02×, tile-oracle +.02375·5/8 ≥.02. 새 confirmatory/router로 부르지 않음 |
| **R1 완료** | 6,834 sample Presto cache + C1a common-grid | content seal, same IDs·decoder·3 seed; C1a `.1092`, P4/P2에 8/8 패배(MS-87) |
| **R2 완료** | Presto C1b native-grid 8지역×3seed | `.1261`; C1a 대비 +.0169이나 P4/P2에 8/8 패배. product sensitivity로만 보고 |
| **R3 종료** | naive fusion closure + GeoContextGate development | MS-90B/91/92 전부 FAIL. FP-matched oracle `+.008/-.004`; stop rule에 따라 method 후보 하차, v3 금지 |
| **R3.5** | raw baseline recipe audit | current 40ep BCE와 official-like 75ep BCEDice를 source-only val로 비교; P3를 SOTA로 부르지 않음 |
| **R4** | {1,5,10,100%} source-label curve | 3 subset seed×3 optimizer seed까지 포함하면 432 new runs. full val 라벨은 고정이므로 total-label claim 금지 |
| **R5** | Korea input/split/ontology preflight + sealed recipe | v2/대체 canonical S2, M10 split, label phenomenon/time/provenance를 감사하고 모든 arm 등록 후 test 개봉 |
| **R6** | Korea external transfer + cost table | cluster-macro/LOCO/CI, cold cache·warm head·deployment 비용 분리. paired re-annotation 없이는 annotation effect 주장 금지 |

2026-09-02 C1b는 `P4native`와 snapshot·cache seal·OUTROOT 규약 아래 24/24 완료했다. 사용자
규칙에 따라 GPU0은 계속 건드리지 않는다. 다음 GPU1 실행은 raw recipe·subset manifest·평가
작동점을 먼저 봉인한 뒤에만 시작하며, 실행 중 confirmatory 네 코드 경로는 수정하지 않는다.

**논문 경계:** R3까지는 “OLMo가 일반 GeoFM보다 낫다”를 금지한다. R5 전에는 한국 전이를
주장하지 않는다. 지역별 P4 gain 편차는 보였지만 label-free winner predictor가 없으므로 router는
method headline이 아니라 보류된 후속 질문이다.

---

## [이전 계획 보존] 2026-08-25 MountainShift 우선순위 교정

## 2026-08-25 현재 임계경로

> **글로벌 EO cache에 작은 region-static residual과 cutoff-valid live residual을 더하면
> 한국·네팔·스위스의 산악 disturbance segmentation·검색이 좋아지고, 두 국가에서 배운 표현이
> 봉인한 세 번째 국가로 0/1/5/10% 라벨만으로 전이되는가?**

이 질문은 FoldRefresh의 반복이 아니다. FoldRefresh는 encoder release가 바뀐 `z_global` page의
선택적 갱신을 맡고, 현재 실험은 새 지역의 의미 부족과 live context를 맡는다. AI-Hub C2-C는
한국 입력을 보강하는 최대 1일의 지원 gate이며 전체 queue를 막지 않는다.

### 2026-08-25 후속 보정 — headline spine을 공개 15지역으로 옮긴다

아래 7일 queue는 유지하되 **headline의 무게중심을 바꾼다.** 이유는 세 가지이고
근거·게이트는 `docs/MOUNTAINSHIFT_EXPERIMENT_DESIGN.md`가 authoritative하다.

1. 3-way leave-one-country-out은 **독립 표본이 3개**라 신뢰구간을 만들 수 없고,
   그중 둘(네팔·스위스)은 snapshot join을 한 번도 하지 않았다. 하나만 막혀도 headline이 죽는다.
2. Sen12Landslides는 **15지역** S1/S2+DEM·event date를 공개로 제공한다.
   leave-one-region-out을 15폴드로 돌릴 수 있고 재배포 제약도 없다.
3. 세 나라의 산사태 라벨은 **생성 과정이 다르다**(한국=S2 위 수동 폴리곤/촬영일,
   네팔=refined inventory/event date, 스위스=행정 cadastre/신고일). 이걸 통제하지 않으면
   국가 간 차이가 domain shift인지 annotation shift인지 분리되지 않는다.

```
P1  Sen12Landslides 15지역 LOCO        ← headline spine (공개, CI 가능)
P2  G-A annotation-process 감사         ← 3국 주장의 전제
P3  한국(봉인된 M10 holdout)을 16번째 지역으로
P4  네팔·스위스 access 통과분만 추가
P5  FoldRefresh continuity/cost (E_refresh)
```

**P1이 통과하지 못하면 P3~P5를 열지 않는다.** 공개 데이터에서 안 되는 방법을 제한
데이터로 살리지 않는다. 아래 7일 queue의 D1·D2(3국 mapping/access)는 P2·P4로 옮기고,
D0로 **Sen12Landslides 다운로드·라이선스·split 확인(G-0)** 을 앞에 둔다.

### 7일 queue

| 일자 | 산출물 | 통과 기준 |
|---|---|---|
| D1 | Sen12Landslides Nepal 20 + AI-Hub train/val 20의 canonical S2 10-band/time/GSD/label/DEM mapping | 동일 scale·missing-band 계약, 원본·파생 checksum과 manifest 봉인 |
| D1 병렬 | C2-C train+val 40표본 | 하루 안에 90% exact recovery; 실패하면 v1/10밴드 또는 새 S1/S2/DEM arm |
| D2 | Swiss Bern/SLF event 20건 access table | 15건 이상 geometry + cutoff time + pre/post EO 연결 |
| D3 | OLMo frozen linear/MLP probe + prototype retrieval | scratch/U-TAE 95% 또는 raw-spectral retrieval보다 우수한 출력 하나 |
| D4 | primary slope-failure의 local-only / naive pooled / shared+local-head baseline | country-cluster 단위 F1·AUPRC·Recall@20·nDCG@20 |
| D5 | DEM/slope/climate region-static residual, 3 seeds | F1 또는 Recall@20 `+2%p`, worst-country `-1%p` 이내 |
| D6 | historical live-source snapshot + cutoff replay | observed/published/retrieved 95% 이상; 미래정보 0건 |
| D7 | 3-way leave-one-country-out 표와 go/kill memo | zero-shot·1/5/10% label, `E_static/E_live/E_transfer/E_refresh` 분리 |

Backbone은 OLMoEarth가 중심이고, task-specific U-TAE/3D-UNet과 입력 계약이 맞는 Prithvi-EO-2.0
또는 TerraMind 하나를 비교한다. 여러 GeoFM을 장식처럼 늘리지 않는다. 상세 source·architecture·
promotion 기준은 `MOUNTAIN_EVIDENCE_TRANSFER.md`가 authoritative하다.

3-way country transfer의 primary task는 세 나라 모두에 있는 `slope-failure` 하나다. 한국 벌목,
네팔 GLOF, 스위스 눈사태는 local auxiliary head이며 공통 task로 합치지 않는다. 국가별 20건은
access gate이지 최종 논문 표본수가 아니다.

primary transfer는 한 OLMo release와 동일 input contract만 쓴다. 기본 후보는 v1 direct-model의
canonical S2 10밴드 + B01/B09 band-group missing mask다. 이 경로가 재현되지 않으면 세 나라 모두
12밴드 재물질화하거나 raw S1/S2/DEM baseline으로 전환한다. 국가마다 다른 상수 채움은 금지한다.
v1↔v1.2는 transfer 결과가 닫힌 뒤 FoldRefresh arm에서만 다시 연다.

## [LEGACY SUPPORT PLAN] 기존 K-EvidenceShift/K-ALIGN 계약

아래 계획은 삭제하지 않는다. 한국 evidence/compatibility branch와 negative evidence로 보존하지만,
2026-08-25 이후 현재 GPU queue나 논문 임계경로를 정의하지 않는다.

## 0. 지금 증명되지 않은 것

- 한국 공공데이터를 결합해 정확도가 올랐다는 supervised 증거는 아직 없다.
- 현재 14후보는 모델 순위로 선택된 retrospective audit pool이며 독립 정답이 0개다.
- 공식 사건 보강은 0/14이고 원인 라벨도 0/14이므로 원인규명 논문으로 쓰지 않는다.
- OlmoEarth v1/v1.2 legacy 입력 216×2 release audit은 완료했지만, BestClear 입력과 독립 task label이
  없어 공공 context·입력 합성·실제 세계 변화의 성능 효과는 계산되지 않았다.
- 연합학습은 실제 비반출 기관 3곳이 없으므로 본 실험에 포함하지 않는다.

### 이번 세션에서 닫힌 engineering gate

- GPU0에서 exact-input 8 site-years×OlmoEarth v1/v1.2를 순차 실행해 16/16 output과 완료 marker를
  만들었다. GPU1의 다른 학습은 건드리지 않았다.
- pooled site-year 거리 순위상관 0.889와 top-1/2 이웃 보존 0.75/1.00에 비해, window 내부
  spatial CKA는 평균 0.427(0.133–0.828)이었다. 전역 검색 구조와 국소 공간 표현이 서로 다른
  정도로 이동할 수 있다는 P0 신호다.
- 이 결과는 아래 연구를 시작할 수 있는 실행·계측 gate이지 transfer 성능 결과가 아니다.
- 실행 증거 자체는 서버 raw 228파일·7.85GB 재해시 758/758 check와 분석 JSON/CSV의
  byte-identical 재실행으로 닫혔다. 이후에도 허용되는 것은 8개 고정 입력의 기술통계뿐이다.
- 이후 full 216×2도 완료했다. same-release R@1=1.0과 달리 raw cross-release R@1=0.0,
  calibration-only affine ridge 0.6973/0.6089로 0.95 gate를 실패했다. 이는 representation-proxy
  cache 재사용 실패이며 task accuracy나 public-context utility가 아니다.

## 1. 논문을 두 개로 분리한다

### Paper A — 먼저 제출할 핵심 논문

가제: **K-ALIGN: Provenance-Aware Compatible Distillation for Earth Models under Asynchronous
Public Context**

핵심 질문:

> 시점·출처·coverage가 명시된 한국 공공 context로 여러 frozen Earth model의 지식을 EO-only
> stable bus에 증류하면, 지역/미래연도 task utility를 높이면서 model release가 바뀌어도 기존
> gallery와 head를 재사용하고 context만 별도로 갱신할 수 있는가?

기여 후보:

1. event/observed/published/retrieved time과 자연 coverage·누락·충돌을 분리한 site-event benchmark
2. Olmo v1/v1.2·TerraMind teacher를 stable EO-only bus로 옮기는 compatible distillation
3. stable cache와 timestamped Korea-context residual을 분리한 dual-speed representation
4. `E_repr / E_compat / E_fusion / E_refresh`의 지역/시간/cloud/missingness·비용 비교

Paper A에는 원인규명, 실시간 시스템, 연합학습, 전국 서비스, active acquisition, robotics/simulation
신기여를 넣지 않는다. public source-role 기본 계약은 `K_CONTEXT_FUSION_EXPERIMENT.md`, 통합
architecture·cache/update·gate의 authoritative 계약은 `KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`다.
`E_repr`와 `E_compat` 중 하나만 통과하면 한 논문으로 억지로 합치지 않고 context 또는 bus
단독 질문으로 축소한다.

### Paper B — 같은 데이터로 이어갈 방법론 논문

가제: **K-Evidence: Selective Change Detection under Incomplete Administrative Evidence and
Model Releases**

핵심 질문:

> 시각 변화와 공식 근거가 불완전·비동기일 때 시스템은 언제 변화를 보고하고, 언제 원인을
> 보류하며, 행정자료 누락 편향을 어떻게 측정해야 하는가?

Paper B의 중심 지표는 risk–coverage/AURC, evidence availability, false supported-cause rate,
release/input 변화에 따른 최종 결정 안정성이다.

### Paper A fallback / Paper D — 임베딩 전이와 embodied 확장 경계

compatibility만 통과할 때의 fallback 가제: **Compatible Earth Representation Bus**

> 여러 frozen EO model family/release의 spatial·temporal 지식을 compact student의 안정 좌표계로
> 옮기면, 독립 EO task utility를 유지하면서 cross-model gallery와 edge/cloud 비용을 함께 개선할
> 수 있는가?

- 현재 full-216 결과는 raw cross-release R@1=0과 ridge 0.6973/0.6089라는 **문제 동기**다.
  결과를 이미 본 sealed 64를 새 student의 test로 재사용하지 않는다.
- P0는 Olmo v1/v1.2 + TerraMind Base, 공통 S2 canonical view, 새 untouched 지역, student 1개,
  linear/MLP/relational bridge로 제한한다.
- 승격 gate는 best teacher task −1%p 이내, worst group −2%p 이내, bus-native 대비 cross-family
  R@1/mAP 95% 이상, latency/FLOPs 5× 또는 bytes 8×, backfill bytes 10× 절감이다.
- affine/per-teacher projector나 generic AM-RADIO식 distillation과 같으면 새 방법 주장을 중단한다.

Paper D 가제: **Release-Stable Earth-to-Embodied Transfer**

- satellite–drone/ground paired data가 생긴 뒤에만 GeoBridge·UniGeoRS·PAUL·MMGeo와 비교한다.
- paired image만 있으면 cross-view localization이다. 실제 action/pose trajectory와 Success/SPL·collision
  평가가 생긴 경우에만 navigation/world-model로 승격한다.
- simulation은 photorealism이 아니라 geometry/semantic consistency와 real policy 성능을 본다.

Paper A에 Paper D의 robotics·simulation을 합치지 않는다. 넓은 후보 문헌은
`EMBEDDING_TRANSFER_CVPR_TRACKS.md`, 한국 정렬형 중심 실험은
`KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`에 둔다.

## 2. 평가 단위와 라벨

기본 단위:

```text
(site_event_id, geometry, t0, t1, acquisition_ids,
 input_recipe, model_release, evidence_snapshot_cutoff)
```

라벨 축은 합치지 않는다.

| 축 | 값 | 만드는 사람/출처 | 주 태스크 사용 |
|---|---|---|---|
| `visual_change` | yes / no / uncertain | EO 블라인드 판독 | Paper A primary |
| `change_mask` | polygon/mask 또는 unavailable | 이중 판독+adjudication | 100건 이상 가능할 때 secondary |
| `official_event_supported` | supported / not_observed / conflicting | 날짜·geometry가 맞는 공식 기록 | Paper B |
| `evidence_available` | source별 available/missing/error/out-of-window | API/data manifest | Paper B |
| `cause` | official-only class / unknown | 공식 사건+독립 검증 | 탐색적; 사람 추정 금지 |

`uncertain`은 임의로 음성에 넣지 않는다. 학습에서는 제외/soft target/abstention을 사전 고정하고,
평가에서는 coverage와 함께 보고한다.

## 3. 표본을 세 자산으로 분리한다

### A. Sealed probability test — 최소 300건

- 지역 3곳: 제주, 강원 산림권, 수도권 외곽 개발권
- 전이구간 2개: 과거→중간, 중간→최신
- cloud proxy 2층: clear/mixed 이하, high/persistent
- 각 `3지역×2구간×2구름` cell에서 포함확률을 기록해 25건씩: 총 300건
- 같은 필지·500 m buffer·공유 Sentinel scene·동일 사건은 한 split에만 둔다.
- 모델 score·불확실성·공공근거를 보지 않고 추출하고, 최종 논문 전까지 봉인한다.

### B. Train/active pool — 별도 최소 300건

- random/stratified 100건으로 최초 head를 만든다.
- 나머지는 budget 25/50/100/200에서 각 acquisition method가 선택한다.
- 모델 disagreement로 뽑은 표본을 전국 prevalence 추정에 사용하지 않는다.

### C. Double-review set — 최소 120건

- `3지역×2 cloud strata×20건`
- 두 판독자가 서로의 결과와 공공근거를 보지 않고 먼저 판독한다.
- Cohen's kappa 또는 Krippendorff alpha 0.60 미만이면 label protocol을 고치고 모델표를 열지 않는다.
- conflict는 제3 adjudication 후 원판정 둘과 함께 보존한다.

## 4. 모델 비교는 두 표로 나눈다

### Table A — paired acquisition 비교

같은 AOI·t0/t1·Sentinel acquisition IDs·label·split을 사용한다. 모델별 공식 normalization과 band
adapter는 허용하되 최종 tensor hash를 저장한다.

| 입력 subtrack | 비교 모델 | 해석 |
|---|---|---|
| S2-12 @ 10 m | scratch U-Net, architecture-matched random ViT, OlmoEarth v1/v1.2, TerraMind Base | S2 다중밴드 전이 |
| HLS-like 6-band @ 30 m | scratch U-Net/ViT, OlmoEarth, Prithvi-EO-2.0-300M-TL | temporal pretraining 전이 |
| RGB-only | scratch/generic ViT, DINOv2-class generic vision, OlmoEarth RGB view | generic vision 대조 |

subtrack 사이의 절대 점수로 모델 우월성을 주장하지 않는다.

### Table B — native ceiling

- TerraMind: 공식 S2/S1/DEM 또는 지원 modality
- CROMA: 동시성 조건을 통과한 S1+S2가 있을 때만
- Prithvi: HLS 6-band/30 m/date-location contract
- OlmoEarth: 12-band/10 m multi-temporal contract

추가 modality로 좋아진 결과는 pretraining 효과가 아니라 `native system ceiling`으로 부른다.

## 5. 학습·평가 계약

모든 supervised 비교에 공통으로 적용한다.

- split: site/event/parcel/scene component + spatial buffer, 과거 train→미래 test
- label fractions: 1%, 5%, 10%, 50%, 100%
- seeds: 최소 5개
- head: frozen linear → frozen MLP → LoRA/adapter → full fine-tuning 순서
- compute matching: trial 수가 아니라 GPU-hours, processed samples, early-stop rule을 맞춘다.
- train-only: normalization, centering, calibration, threshold, Procrustes/linear bridge
- CI: 픽셀이 아니라 site/event 또는 spatial block clustered bootstrap
- model selection: test를 보지 않고 validation macro-AUPRC와 worst-group으로 선택

Primary metrics:

| 질문 | 지표 |
|---|---|
| 변화탐지 | event AUPRC, macro-F1, fixed-recall false alarm, mask가 있으면 mIoU |
| 전이효과 | `Delta_g = pretrained_g - matched_scratch_g`, cluster-bootstrap 95% CI |
| negative transfer | `Delta_g < 0`인 사전고정 그룹 비율과 worst-group delta |
| calibration/보류 | ECE, Brier, risk–coverage, AURC, coverage@fixed-risk |
| active labeling | budget별 AUPRC/worst-group, area under learning curve, target 도달 라벨 수 |
| 릴리스 연속성 | CKA, shift-null excess, within-release distance Spearman, neighbor overlap |
| 비용 | GPU-hours, peak VRAM, wall time, I/O time, label/review minutes |

## 6. 사전 promotion·kill gate

| 주장 | Promotion gate | 실패 시 |
|---|---|---|
| GeoFM 전이 유효 | matched scratch 대비 macro-AUPRC +2%p, 95% CI>0, 3지역 중 2곳 이상 같은 방향 | negative-transfer atlas만 보고 |
| PEFT 유효 | full FT 대비 -1%p 이내, trainable params ≤2%, GPU-hours/VRAM 절감 | frozen/full FT만 유지 |
| 능동 라벨 유효 | random 대비 같은 성능 도달 라벨 ≥20% 절감, worst-group -1%p 이내 | random/stratified baseline 유지 |
| 공공데이터 결합 유효 | 영상-only 대비 fixed recall false alarm ≥10% 감소 또는 AUPRC +2%p; clear 성능 -1%p 이내 | context UI로만 유지 |
| EO-only 표현 강화 | privileged-train student가 frozen Olmo보다 AUPRC +2%p 또는 label 20% 절감, CI>0, 3지역 중 2곳 같은 방향 | “embedding 강화” 주장 제거 |
| learned fusion 신기여 | best STACK/TOKEN-FUSE보다 +2%p 또는 같은 성능에서 label 20% 절감 | simple fusion 또는 benchmark 결과만 보고 |
| context 누출 반증 | region×year shuffle/time-shift에서 주장 이득 ≥80% 소멸, future sentinel 100% 차단 | location shortcut/leak로 판정하고 중단 |
| 선택적 보류 유효 | 같은 coverage에서 risk 감소, AURC 개선, unsupported-cause promotion 0 | 원인 축 제거 |
| release cache bridge | held-out spatial gallery에서 new/new 성능의 ≥95%, 새 모델 자체 손실 ≤1%p | 전체 backfill 필요로 결론 |
| FL | 실제 비반출 기관 3곳+, pooled 불가 근거, 실제 분산 1회 | 구현하지 않음 |

## 7. 정확한 6주 실행표

### Week 0–1 — 현재 GPU0에서 닫을 것

| Job | 입력 | 출력 | 완료 조건 |
|---|---|---|---|
| J0 release smoke | exact 8 site-years×v1/v1.2 | 16 GeoTIFF+SHA+logs | **완료**: 16/16, grid/mask gate pass |
| J1 smoke analysis | J0 output | CKA/null/neighbor/rank JSON+CSV | **완료**: 7-cluster limitation 포함 |
| J2 full legacy audit | 216 site-years×2 release | 432 outputs | **완료**: 432/432, paired hash/grid/mask/value-health 100%; cache proxy gate 실패 |
| J3 BestClear stress gate | 동결 8 site-years×12 periods | reflectance/SCL 96쌍+선택 trace+SHA+RGB | J2보다 먼저 설계; 96/96 `changed` 또는 `valid no-op`, contaminated 4건은 각각 ≥1 changed period, median bad-proxy ≥10% 감소, zero/mask 악화 ≤1%p, 2건 replay hash 100% |

산출 Figure: `release × year` 표현 연속성 heatmap. 성능 그림이 아니라 입력/릴리스 효과의
계산 가능성을 판정하는 engineering figure다.

### Week 2 — context headroom과 데이터셋을 여는 주

1. 3지역 최소 10,000 parcel/site-year sampling frame과 500 m/scene/PNU connected-component split 생성
2. source별 입력/privileged signal/label/evidence/coverage 역할과 prospective cutoff 동결
3. Olmo frozen, location/year-only, context-only, late concat, STACK/TOKEN-FUSE, context adapter,
   EO-only student의 P0 headroom 비교
4. shuffle/time-shift/missingness-only/future-leak sentinel control 실행
5. P0가 통과하면 sealed manifest와 120건 double-review UI/protocol 생성

Week 2 gate: P0 GO 미달이면 supervised scale-up과 모델 수 확장을 멈춘다. P0가 통과해도 독립 판독
120건의 합의도 0.60 미달이면 판독 protocol부터 수정한다.

### Week 3–4 — context representation/fusion 표

1. scratch U-Net + random-init ViT
2. OlmoEarth v1.2 frozen probe·SLR adapter·full FT
3. location/year-only, context-only, late concat, STACK/TOKEN-FUSE
4. provenance-aware EO+context teacher와 EO-only distilled student
5. GeoLink-style fusion과 TerraMind/Galileo native ceiling
6. label fractions×5 seeds, 지역·연도·cloud·natural coverage별 효과

필수 Table: `E_repr/E_fusion/E_decision`, 전체 평균, 3지역 macro, worst region, clear/high-cloud,
과거/미래, available/missing/error/conflict를 분리해 보고한다.

### Week 5 — provenance·누락·shortcut 반증

같은 split/model budget에서 다음을 비교한다.

1. region×year context shuffle
2. context timestamp ±1년 이동
3. location/year-only, missingness-mask-only, geometry-only
4. 미래 record leak-positive sentinel의 100% 차단
5. source ablation과 natural missingness strata
6. synthetic source dropout sensitivity
7. high-cloud에서 S2 / S2+S1 / S2+weather / full context

최종 점수는 처음 봉인한 probability/geographic-future test에서 한 번만 계산한다. active label
acquisition은 이 main matrix가 headroom을 보인 뒤 후속으로 연다.

### Week 6 — 논문 가능성 판정

필수 Figure/Table:

1. Dataset/split/evidence graph schematic
2. EO-only privileged-distillation label-efficiency curve
3. fusion effect: model×region×year×cloud×coverage heatmap
4. natural missingness risk–coverage/AURC curve
5. shortcut controls: real/shuffled/time-shift/location-only
6. matched-input simple/main method table
7. native-ceiling table
8. source-role·cloud/S1/weather·adapter loss ablation

다음 중 하나면 CVPR main 경로를 중단하고 EarthVision/IGARSS/TMLR로 전환한다.

- 3지역 sealed test를 확보하지 못함
- double review 120 또는 합의도 gate 미달
- EO-only student와 inference fusion 모두 strong 영상/simple-fusion baseline을 못 이김
- 개선이 shuffle/time-shift/location-only와 구분되지 않거나 미래정보에 의존함
- 공공데이터가 정확 필지·시점이 아니라 근처 행정문맥에만 머묾

## 8. GPU0 운용 원칙

GPU0은 이 프로젝트 전용으로 사용하지만 100% allocation과 100% utilization을 구분한다.
materialize/I/O 단계에서 GPU 사용률이 낮은 것은 실패가 아니며, 불필요한 dummy compute로 100%를
채우지 않는다.

- 모델 실행은 `CUDA_VISIBLE_DEVICES=0`
- GPU1 프로세스는 감시만 하고 종료·감속하지 않는다.
- feature extraction은 한 번만 하고 모든 probe/active 실험에서 hash로 재사용한다.
- batch size는 smoke 뒤 peak VRAM 80% 이하에서 올리고 OOM 재시작을 피한다.
- 긴 작업은 `setsid nohup`과 COMPLETE marker를 사용한다.
- 각 job은 input/checkpoint/config/output SHA, GPU-hours, wall time, peak VRAM을 남긴다.

## 9. 다음 사용자 입력이 실제로 필요한 시점

지금은 추가 API key가 필요 없다. 필요한 것은 Week 2의 판독자와 지역 선택 확정이다.

1. EO 변화 판독을 도와줄 두 번째 reviewer 1명
2. 제주 외 지역을 강원 산림권·수도권 개발권으로 잠글지 여부
3. NGII 전후 항공사진을 받을 사전고정 10–20 site
4. 실제 기관 비반출 데이터가 생길 경우에만 FL 파트너/계약 정보
