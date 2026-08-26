# 외부 지역 데이터를 OLMoEarth에 붙이는 법과 upstream PR 재감사

갱신: 2026-08-26. 이 문서는 한국·네팔·스위스 자산이 왜 OLMoEarth에 바로 들어가지 않는지,
무엇을 모델 입력·외부 context·label로 분리해야 하는지, 그리고 현재 upstream 기여 후보 중
실제로 제출 가치가 남은 것을 고정한다.

## 교수 판정

데이터가 부족해서 못 붙는 것이 아니다. **서로 다른 계약의 자산을 모두 `model input`이라고
부른 것**이 병목이다. 지역 데이터는 다음 셋 중 하나다.

1. **canonical EO input** — OLMoEarth가 기대하는 센서·밴드·순서·값·해상도·시간축을 만족한다.
2. **local context** — 강우·DEM·경보·행정자료처럼 별도 residual/adapter가 소비한다.
3. **target/evidence** — 산사태 polygon·토지피복 mask처럼 학습·평가에 쓰며 encoder 입력이 아니다.

따라서 “AI-Hub/ICIMOD/swissEO를 OLMoEarth에 넣는다”는 한 문장이 아니라, **지역 label은 canonical
Sentinel 입력과 조인하고, 지원되지 않는 공공 context는 별도 residual로 두며, 새 센서는 adapter/PEFT
후보로 승격한다**가 정확한 설계다. 이 분리는 불편한 전처리가 아니라 EarthRoute가 판정할
`contract shift`의 정의다.

## 공개 OLMoEarth 입력 계약 — 확인된 현재 경계

rslearn의 공개 OLMoEarth 가이드는 downstream 입력을 아래처럼 고정한다.

| modality | 공개 입력 계약 |
|---|---|
| Sentinel-2 L2A | `uint16`, 12밴드, 순서 `B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09` |
| Sentinel-1 IW GRD | `vv vh`, dB 변환, radiometric terrain correction 권장 |
| Landsat 8/9 | 11밴드 고정 순서 |
| 공통 | 1–12 timestep 권장, UTM, 10 m/pixel, modality별 normalization |

현재 rslearn v0.1.14 wrapper 코드는 `worldcover`와 `openstreetmap_raster` key도 인식하지만, 공개
사용 문서가 안정적으로 약속하는 입력·정규화 경로는 위 세 satellite modality다. KMA 강우,
GK2A, BIPAD, SLF, 임의 DEM, KOMPSAT/CAS500, 7-band swissEO를 채널에 덧붙이는 public contract는
없다. 채널을 단순 append하면 tokenization, normalization, channel embedding, checkpoint shape가
함께 바뀐다.

또 OLMoEarth pretraining이 derived map을 사용했다는 사실과, 외부 사용자가 임의의 지역 자료를
같은 modality로 투입할 수 있다는 말은 다르다. **pretraining modality의 이름·ontology·통계와
동일해야** checkpoint 의미가 유지된다.

## 지역별로 바로 붙지 않는 정확한 이유와 우회가 아닌 정식 경로

| 지역 자산 | 지금 가진 것 | 바로 못 붙는 이유 | 정식 연결 경로 | 논문 역할 |
|---|---|---|---|---|
| 한국 AI-Hub 71363 | land-cover·deforestation·landslide label, 원 archive, 별도 materialize한 S2 | label은 encoder input이 아니며 원 S2와 새 STAC S2의 scene/time/grid parity가 자동 보장되지 않음 | label geometry/time을 canonical 12-band S2 cube에 조인. KMA/GK2A/DEM은 residual branch | 같은 cube의 3-task action-risk 이질성 |
| Nepal Koshi 2024 | 산사태 inventory polygon, CC BY 4.0 | 입력 영상 tensor가 아니라 U-Net 자동탐지 후 수동 보정한 **silver target**. S2 기반 평가에서 label-generator 의존성이 생김 | canonical S2/S1을 새로 materialize하고 polygon을 target으로 사용. 수동 adjudication subset 또는 별도 manual inventory로 gold audit | 동일 task의 geography transfer; `untouched gold`가 아니라 `untouched geography + assisted labels` |
| swissEO S2-SR | STAC/COG, mask, registration, 7 reflectance bands | OLMo가 요구하는 12밴드가 아니며 20 m 밴드도 섞임. swissEO 자체에는 slope-failure target도 없음 | (A) 같은 지역·시점의 canonical Copernicus S2 12밴드 + 별도 Swiss label, 또는 (B) 7-band product를 의도적인 missing-band/source shift arm으로 사용 | A는 regional transfer, B는 contract-shift stress; operational system은 후속 |

### 중요한 정정

- Nepal은 **지역적으로 untouched**일 수 있지만 label 생성 과정까지 독립인 untouched gold test는 아니다.
  최종 external claim은 `independent geography transfer on silver labels`로 쓰고, 50–100개 stratified
  polygon의 수동 재판독이나 별도 manual inventory가 있어야 gold claim으로 승격한다.
- swissEO의 7밴드를 억지로 12밴드처럼 채우지 않는다. 지역 전이를 재려면 canonical 12밴드를 다시
  가져오고, missing-band robustness를 재려면 7밴드를 별도 shift family로 명시한다.
- AI-Hub exact-scene recovery는 논문의 본체가 아니다. label과 canonical input을 같은 관측으로
  맞추는 **한국 arm의 one-time eligibility gate**다. 실패하면 실패 원인을 source/reprocessing
  shift로 기록하고 canonical 새 관측으로 task 실험을 계속한다.

## “OLMoEarth 업데이트”를 네 가지로 분해한다

| 사용자가 원하는 효과 | 실제 action | core OLMo weight 변경 | 난이도 |
|---|---|---:|---:|
| 지역 label로 task 정확도 개선 | frozen cache + 새 head 또는 fine-tune | 선택적 | 낮음 |
| KMA/DEM/경보로 현지 성능 개선 | aligned context residual / cached adapter | 없음 | 중간 |
| swissEO 7밴드·새 위성 수용 | sensor adapter, imputation+mask, PEFT | 일부 | 높음 |
| 새 modality를 범용 foundation model에 내재화 | pretraining data schema + tokenizer + 재학습 | 있음 | 매우 높음 |

첫 논문은 첫 두 줄까지만 method action으로 닫는 것이 맞다. 새 센서 native support는 후속 논문이나
AI2 온보딩 기여다. 모든 것을 한 번에 core model에 넣으면 지역 효과, 센서 효과, adapter 효과를
분리할 수 없다.

## EarthRoute와 직접 연결되는 실험 단위

> EO의 지역·시간·센서·모델 shift에서 task별 손실을 target label 없이 예측하고, 비용을 포함해
> `reuse / cached adapter / re-embed / PEFT / raw model` 중 action을 고를 수 있는가?

이 질문에서 `target label 없이`는 **새 deployment block의 action 선택 시 target label을 보지
않는다**는 뜻이다. source/development label로 predictor를 학습하고, 봉인한 target label은 사후
utility와 regret 평가에만 쓴다.

각 `spatial block × observation window × task`에 아래를 남긴다.

- contract features: sensor/product/band availability/GSD/time coverage/cloud/normalization/model release
- unlabeled diagnostics: embedding drift, agreement, entropy, density/overlap, GdScore/ODD 계열
- action cost: shared extraction, adapter/head fit, task raw fit, serving latency, storage
- sealed outcome: action별 metric gain과 utility `gain - λ·cost`

### 가장 작은 유효 실험

1. AI-Hub v2에서 같은 유효 cube를 공유하는 세 task를 만든다.
2. 사전 동결 shift를 적용한다: time truncation, band/product missingness, release v1→v1.2,
   region holdout. 서로 다른 shift를 한 cell에 섞지 않는다.
3. 각 task에 `reuse / cached adapter / re-embed / task-raw`를 같은 head·metric·seed budget으로 실행한다.
   PEFT는 작은 action이 Pareto를 못 만들 때만 연다.
4. task 사이 action ranking이 실제로 역전되는지 먼저 본다. 역전이 없으면 router를 중단한다.
5. 역전이 있으면 target-unlabeled predictor와 fixed-policy baseline의 regret–cost를 비교한다.
6. predictor와 action set을 동결한 뒤 Nepal을 geography transfer로 한 번만 연다. silver-label audit를
   통과하지 못하면 Switzerland 운영 track으로 대체하는 것이 아니라 external claim을 낮춘다.

### M40은 무엇을 죽였고 무엇을 아직 못 죽였나

이 문서를 갱신하는 동안 M40 cheap oracle pilot이 추가됐다. 한 Chimanimani development task에서
P2와 P4 계열의 tile별 승자는 교차했지만, `mean_probability`와 predicted-positive-pixel만 쓴
in-sample threshold rule은 다수결보다 2.4%p만 높았다. 따라서 **현재 output-confidence proxy로는
label-free routing이 안 된다**는 음성 결과는 유효하다.

그러나 이것이 위 EarthRoute 질문 전체의 kill은 아니다.

- 한 task·한 region의 **model recipe 선택**이지 AI-Hub 세 task의 action-risk 이질성이 아니다.
- feature를 얻으려고 P2/P4c 출력을 모두 계산하면 후보 action 실행비를 이미 지불한다. pre-action
  feature로 쓸 것인지 cheap probe 비용으로 셀 것인지 계약이 없다.
- 같은 타일에서 threshold를 찾고 평가했으므로 out-of-region prediction evidence가 아니다.
- 코드가 tile-IoU로 arm을 고른 뒤 picked confusion count를 micro-IoU로 합친다. pairwise
  `oracle gain`이 음수가 되는 항목이 실제로 생기므로 **선택 목적과 보고 목적이 불일치**한다.
  additive block utility의 macro oracle 또는 micro-IoU를 직접 최적화하는 oracle로 다시 정의해야 한다.
- empty tile 62.7%에서는 boundary quality와 false-positive suppression을 같은 IoU winner로 섞지
  말고, positive-event utility와 no-event false-alarm utility를 분리해야 한다.

따라서 다음 gate는 더 많은 proxy를 즉흥 추가하는 것이 아니다. 먼저 seed-only noise-floor oracle,
FP-rate-matched threshold 비교, pre-action feature availability/cost, metric-aligned oracle을 고정한다.
그 뒤 AI-Hub에서 **task별 action ranking**이 재현될 때만 label-free predictor를 다시 연다.

## upstream PR 문서 재감사 — 2026-08-26

확인 기준은 `olmoearth_projects origin/main=23a3d7b`, rslearn `v0.1.14/master=c47952f`다.
olmoearth_projects의 공개 open PR 4건(#37, #42, #43, #64)과 공개 issue 목록에서 아래 첫 PR과
직접 중복은 없었다.

| 후보 | 현재 판정 | 이유와 다음 action |
|---|---|---|
| sample `es_*→oe_*` | **A, 첫 PR로 제출 가치 있음** | upstream main에 결함이 남고 로컬 `5e044ee`가 6 feature를, `21b658a`가 EOF를 고친다. Linux 6-window 기록 + current Linux replay 후 제출 |
| LFMC public checkpoint mismatch | **A, issue/report** | 공개 ckpt 951.9 vs 문서 580.6, 동일 공개 recipe 재학습 558.8. 기존 #45/#46과 주제가 다름. issue 생성 제한 때문에 maintainer email/PR에서 경로 문의 |
| SCL compositor dependency + categorical resampling | **A-, 두 개로 분리** | v0.1.14에서도 data source는 layer band set만 등록하고 scorer는 layer resampling을 그대로 사용. 먼저 `scoring_resampling=nearest` 소형 PR, dependency declaration은 issue/RFC |
| partial-band/release-aware mask | **B+, 연구 blocker지만 PR 미성숙** | v0.1.14 wrapper도 static modality band-set 수로 mask를 만든다. 그러나 public API의 10-band minimal repro와 기대 정책이 먼저 필요. 곧바로 큰 patch 금지 |
| lockfile v1.2 incompatibility | **B+, compatibility issue/PR** | 현재 project lock은 rslearn 0.0.23 + olmoearth-pretrain 0.0.2, 최신 rslearn은 0.1.14 + minimal ≥0.0.6. 무작정 lock bump 대신 v1/v1.2 matrix와 runner integration을 먼저 테스트 |
| `ingest:false + CONTAINS` NotImplementedError | **종결/폐기** | current rslearn direct materialization이 구현됐고 PC get-by-name 회귀 테스트도 존재. 과거 발견으로만 보존 |
| “미출시 rslearn API” | **#13에 병합** | API는 현재 출시됐다. 문제는 unreleased API가 아니라 오래된 project lock과 current config의 skew |
| embedding guide 12→4 ambiguity | **문서 clarification** | 현재 문서는 layer 수가 time range/query에 달린다고 설명한다. 버그 주장보다 exact consumed-layer manifest 제안으로 낮춤 |

### PR #1 제출 전 체크리스트

- [x] 현재 upstream main에도 legacy `es_*`가 남는지 확인
- [x] open PR/issue 직접 중복 없음 확인
- [x] 수정 범위가 한 GeoJSON 파일·6 feature뿐인지 확인
- [x] Linux 0.1.14에서 6 window 생성 기록
- [x] EOF newline 추가 후 JSON parse, 6 feature, legacy-key 0건, dict label 검사
- [x] macOS current runtime 재실행 — schema 이후 forkserver hang을 별도 #8로 재확인
- [ ] Linux current runtime에서 quick-start 한 번 재실행
- [ ] fork/push/PR 생성 — 외부 쓰기이므로 사용자 승인 뒤 수행

PR #1은 연구 main claim은 아니지만 AI2 취업 축에는 유의미하다. 작은 upstream defect를 현재 코드,
재현 명령, 호환 버전, 최소 diff로 닫는 능력을 보여준다. 반면 partial-band mask는 연구적으로 더
흥미롭지만 API 정책까지 건드려 첫 PR로는 위험하다.

## 금지 주장

- 지역 데이터가 있다는 이유만으로 “OLMoEarth가 한국/네팔/스위스를 지원한다”고 하지 않는다.
- U-Net-assisted Nepal inventory를 독립 gold label이라고 하지 않는다.
- 7-band swissEO 결과를 canonical 12-band transfer 성능으로 보고하지 않는다.
- pretraining에 derived maps가 있었다는 이유로 임의 KMA/DEM channel append를 허용하지 않는다.
- PR 후보를 제출 전 `review 중` 또는 `제출됨`으로 쓰지 않는다.

## 1차 출처

- rslearn OLMoEarth input contract: https://github.com/allenai/rslearn/blob/master/docs/foundation_models/OlmoEarth.md
- current rslearn wrapper: https://github.com/allenai/rslearn/blob/master/rslearn/models/olmoearth_pretrain/model.py
- OLMoEarth model/data summary: https://github.com/allenai/olmoearth_pretrain
- ICIMOD Koshi 2024 metadata/license: https://rds.icimod.org/metadata/af73da0a-885b-459d-95ba-2ea0662a7e7c
- swissEO S2-SR product content: https://www.swisstopo.admin.ch/en/satelliteimage-swisseo-s2-sr
- upstream open PR list: https://github.com/allenai/olmoearth_projects/pulls
- upstream issues: https://github.com/allenai/olmoearth_projects/issues
