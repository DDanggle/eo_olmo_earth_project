# Decision-Continuous Earth Intelligence

최종 갱신: 2026-08-24

## 한 문장 연구 프로그램

> **세계와 기반모델이 동시에 바뀔 때, 제한된 라벨과 컴퓨트만 가진 환경 조직이
> 지도 기반 의사결정을 과학적으로 유효하고 연속되게 유지하려면 무엇을 측정하고
> 무엇을 다시 계산해야 하는가?**

제주는 목적지가 아니라 첫 검증장이다. OlmoEarth는 연구 대상이자 공개 인프라이고,
`olmoearth-migrate`는 이 연구를 반복 가능하게 만드는 시스템 산출물이다.

## 현재 플래그십과 후속 프로그램의 경계

세 이름은 경쟁 프로젝트가 아니라 한 시스템의 서로 다른 층이다.

| 층 | 맡는 질문 | 이 문서에서의 상태 |
|---|---|---|
| **K-Earth** | 위성 변화에 한국 필지·행정·환경 근거를 붙였을 때 무엇을 말하고 보류할 수 있는가 | 현재 제주 플래그십 |
| **EarthEmbedContract** | 모델 릴리스·시간창·밴드·GSD·pooling 계약이 다른 embedding의 잘못된 재사용을 어떻게 탐지하고 막는가 | **첫 main-track 감사+방법 실험** |
| **FoldRefresh** | 모델 릴리스가 바뀌어도 어떤 모집단 통계·결정을 부분 재계산으로 유지할 수 있는가 | 별도 로컬 방법 자산, rslearn 이식 예정 |
| **EarthRoute** | 다음에 어떤 관측·모델·행정근거·사람검증을 구매해야 하는가 | 위 방법의 headroom 확인 뒤 여는 후속 프로그램 |

따라서 EarthRoute가 현재의 evidence-coverage 연구를 밀어내지 않는다. K-Earth가 안전한
decision target을 만들고, FoldRefresh가 `reuse` action을 제공한 뒤, EarthRoute가 이를 비용–위험
정책으로 일반화한다. 세부 action space·사업 가설·경쟁 경계는 `EARTHROUTE_PROGRAM_NOTE.md`,
문헌 검색 장부는 `PAPER_READING_LIST.md`에 둔다. 한국 전이 효과·공식근거 보류를 하나의 재현
가능한 평가 자산으로 만드는 schema는 `K_EVIDENCE_SHIFT_BENCHMARK.md`, 공공 context의 역할별
기본 계약은 `K_CONTEXT_FUSION_EXPERIMENT.md`, 이를 multi-teacher compatibility·cache refresh와
결합한 main-paper 계약은 `KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`가 담당한다.

## 현재 실행 상태

연구 질문은 넓지만 첫 실행은 의도적으로 좁혔다.

- `K-EvidenceShift Jeju pilot v0`: 14 candidate records, 13 independent spatial groups,
  independent human ground truth 0, official event corroboration 0, cause 0, abstain 14.
- leakage contract: 같은 site/500 m/window/scene/PNU를 묶고, t1 이후 RGB와 사후 API를
  prospective input에서 차단했다. 이 pilot은 benchmark score를 만들지 않는다.
- release P0: 54 windows × 4 years = 216 site-years, 162 adjacent-year events, clear/contaminated
  proxy smoke 8개를 label-free로 고정했다.
- OlmoEarth v1/v1.2는 HF commit과 weight SHA로 고정했고 smoke 뒤 216×2 full inference를 GPU0에서
  완료했다. raw cross-release R@1=0.0, 최선 affine ridge 0.6973/0.6089로 representation-proxy
  호환성 gate는 실패했다. 독립 label이 없어 task utility는 여전히 미검증이다.

따라서 당장 실행할 질문은 단순하다. **두 embedding이 정말 비교 가능한 계약에서 만들어졌는가?**
동일 입력의 model-release mismatch와 동일 모델의 time-window mismatch가 각각 cache 검색과 변화
후보를 무너뜨린다는 두 관측을 묶는다. 시스템은 비교 전에 REUSE/ADAPT/RECOMPUTE·ABSTAIN을
결정해야 한다. 비용·quantizer·공공 context는 이 판정 뒤의 해결수단과 검증자료다.

## 왜 이 질문인가

세 계보가 한 문제에서 만난다.

| 계보 | 이미 알려진 문제 | 이 프로젝트가 잇는 빈칸 |
|---|---|---|
| MIT Earth Intelligence Lab | 위성 지도는 정답이 아니라 오차가 있는 ML 예측이며, 지도만으로 회귀·면적을 추정하면 편향될 수 있다 | 모델 버전과 합성 레시피가 바뀌는 상황까지 오차원을 확장하고, 소량의 현장 검증으로 최종 의사결정을 보정한다 |
| Ai2 OlmoEarth | 소규모 조직도 빈번히 갱신 가능한 Earth Intelligence 인프라가 필요하다 | 새 릴리스가 정확도를 유지하더라도 기존 검색·우선순위·집계가 유지되는지는 별도 감사한다 |
| MARC형 현장 파트너 | 장기 현장조사로 서식지 이용과 인간활동 영향을 이해하고 연구 결과를 보전·정책으로 연결한다 | 위성은 동물을 탐지하지 않고, 현장조사의 공간·시간적 맥락과 조사 우선순위를 보조한다 |

이 연결은 “더 좋은 지도”가 아니라 **바뀌는 지도에서 여전히 유효한 결론**을 연구한다.

## 연구 대상: 네 축이 붙은 관측 단위

모든 결과는 아래 네 축을 잃지 않는 단위로 저장한다.

`장소·시점 × 원시 관측/합성 레시피 × 모델 릴리스·가중치 × 검증/의사결정 정의`

- 장소·시점: AOI, 기간, 좌표계, GSD
- 입력: 센서, 밴드, 모자이크 방식, cloud/nodata 처리
- 모델: 모델 ID, weight hash, code commit, embedding 후처리
- 결론: 라벨 표본설계, 지표, 신뢰구간, 파트너가 실제로 내리는 결정

이 네 축 중 하나라도 빠지면 세계 변화와 파이프라인 변화를 분리할 수 없다.

## 연구 질문과 반증 가능한 가설

### RQ1. 세계 변화와 측정 파이프라인 변화를 분리할 수 있는가?

- H1: 구름·결측이 큰 해안/몬순 지역에서는 모델 릴리스보다 **실제 장면선택·합성 개입**이
  임베딩과 변화 순위에 더 큰 영향을 준다.
- v5에서 `MOSAIC`와 `PER_PERIOD_MOSAIC` 문자열은 같은 handler·ordered item group·픽셀을
  만들었으므로 실험 수준으로 세지 않는다. 입력 축은 legacy selection과 v7의
  `기간당 coverage 최대 3 + Sentinel2SCLBestClear`처럼 실제 item/pixel hash가 달라진 셀만 인정한다.
- 반증: 사전 층화한 여러 AOI에서 v1↔v1.2 차이가 legacy↔SCL BestClear 차이보다 일관되게 크면
  “입력 개입이 더 크다”는 가설을 기각한다.

### RQ2. “drop-in replacement”가 의사결정에도 drop-in인가?

- H2: 평균 task 정확도가 유지돼도 최근접 이웃, Top-k 조사 후보, 행정/생태 구역 집계는
  비균일하게 이동한다.
- 반증: bootstrap 신뢰구간까지 포함해 모든 의사결정 지표의 변화가 사전 허용범위 안이면
  릴리스 드리프트 문제는 이 사례에서 중요하지 않다고 결론 낸다.

### RQ3. 적은 현장 라벨로 지도 기반 결론을 보정할 수 있는가?

- H3: 층화 확률표본과 Prediction-Powered Inference(PPI)를 결합하면 map-only 편향을
  보정하면서 labels-only보다 좁고 유효한 신뢰구간을 얻는다.
- 반증: nominal coverage가 무너지거나 labels-only 대비 구간 폭 이득이 없으면 사용하지 않는다.

### RQ4. 어디만 다시 계산해야 하는가?

- H4: 품질·릴리스 드리프트·의사결정 민감도가 큰 타일을 우선 갱신하면 전체의 25% 이하
  재계산으로 full refresh의 지역 집계와 조사 우선순위를 보존할 수 있다.
- 기술 사전 기준: full refresh 대비 집계 오차 5% 이하, Top-k Jaccard 0.90 이상.
  최종 기준은 파트너 인터뷰에서 false positive/누락 비용을 확인한 뒤 **실험 전에** 고정한다.

### RQ5. 제주 밖에서도 같은 현상과 선택 정책이 재현되는가?

- H5: 동일 감사 하네스가 최소 두 태스크군 × 두 지역 × v1/v1.2에서 입력/릴리스 드리프트를
  분해하고 선택적 갱신 대상을 찾는다. target region을 보고 조정한 셀은 무튜닝 전이로 세지 않는다.
- 평균 점수뿐 아니라 rare-class recall, worst-region risk, calibration, abstention coverage와
  label/I/O/GPU/검수 비용을 함께 보고한다.
- 두 번째 태스크나 지역에서 방향이 반복되지 않으면 `foundation-model transfer`를 지우고,
  방법 논문이 아니라 제주 사례 또는 task-specific adaptation으로 범위를 낮춘다.

### RQ6. 글로벌 GeoFM 전이는 한국에서 어디서 실패하고, 다음 라벨은 어디에 써야 하는가?

- H6a: OlmoEarth를 포함한 GeoFM의 평균 이득은 지역·연도·센서·구름·희귀 class별로 균일하지
  않으며, matched scratch baseline보다 나쁜 **negative transfer** 구간이 존재한다.
- H6b: 기존 분포와 중복된 라벨을 더 모으는 것보다, 품질 게이트를 통과한 cross-model·release·
  sensor disagreement와 공간 다양성·라벨 비용을 함께 쓴 acquisition이 같은 예산에서 worst-group
  risk를 더 빨리 줄인다.
- 전이 효과는 그룹 `g`, label budget `b`마다
  `score(pretrained+adapted) - score(compute/data-matched scratch)`로 정의하고 site/event 단위
  bootstrap CI를 붙인다. 평균 정확도 하나로 전이 성공을 선언하지 않는다.
- adaptive하게 얻은 라벨은 모집단 대표 표본이 아니다. active pool과 별개인 봉인 test·층화
  확률표본을 유지하고, active labels만으로 한국 전체 변화율이나 PPI 신뢰구간을 계산하지 않는다.
- PDE 연구의 `beta`, advection–diffusion, 모호성선, D-opt 해석은 전이하지 않는다. Earth에서는
  물리 parameter 대신 그룹별 전이효과, 물리적 경계 대신 경험적 disagreement region, D-opt 대신
  feature-diversity baseline으로 각각 새로 정의한다.
- 상세 schema·모델 매트릭스·acquisition baseline·promotion gate는
  `K_EVIDENCE_SHIFT_BENCHMARK.md`에 고정한다.

### RQ7. 시점·coverage가 있는 공공 context는 EO 표현 자체를 강화하는가?

- H7a: cutoff-valid public context를 train-time privileged supervision으로 사용하면, test에서
  context 없이 EO만 입력한 student도 frozen OlmoEarth보다 적은 라벨로 독립 visual task에 도달한다.
- H7b: 추론 시 context를 함께 쓰는 이득은 정적 위치 shortcut가 아니라 parcel/event 시점 정보에
  의존하며, region×year shuffle 또는 ±1년 time-shift에서 대부분 사라진다.
- H7c: source의 available/missing/error/out-of-window/conflict를 token과 평가 strata로 보존한
  adapter는 naive zero-imputation/late concat보다 high-cloud·미래연도·자연 누락의 worst-group
  risk와 AURC를 낮춘다.
- 반증: location/year-only 또는 hard-coded STACK/TOKEN-FUSE가 proposal과 같거나 더 좋고,
  adapter의 parameter/label 효율도 없다면 방법 기여를 중단한다. 개선이 미래 행정 record나 target
  label과 동일 source에서만 생겨도 즉시 누출로 판정한다.
- EO-only 표현 강화 `E_repr`, inference fusion `E_fusion`, post-prediction selective decision
  `E_decision`을 별도 표로 보고하며 하나의 “embedding 개선” 문장으로 합치지 않는다.
- 세부 source role, model/ablation, 3지역 split, 1,200-label 목표, promotion/kill gate는
  `K_CONTEXT_FUSION_EXPERIMENT.md`에 고정한다.

### RQ8. Earth embedding 지식을 다른 모델·view·policy로 안전하게 전이할 수 있는가?

- H8a: OlmoEarth·TerraMind·Galileo/Prithvi의 frozen teacher를 작은 student에 증류하면서 teacher별
  spatial/temporal relation과 task utility를 함께 보존하면, per-teacher linear bridge보다 높은
  cross-family/release retrieval과 5× 이상 query efficiency를 동시에 얻을 수 있다.
- H8b: 64/128/256/768d nested bus가 독립 저차원 모델·PCA/PQ보다 edge/cloud bandwidth–utility
  곡선을 개선하면서 old/new/family gallery와 직접 호환될 수 있다.
- H8c: satellite map teacher를 drone/ground student에 전이하는 효과는 cross-view localization이나
  action-conditioned navigation 성능으로만 판정한다. paired image는 localization 근거이고 실제
  trajectory·action이 없으면 robot-policy/world-model 주장은 하지 않는다.
- 반증: affine projector나 AM-RADIO식 generic distillation이 제안 방법과 같거나, best teacher 대비
  task −1%p·worst-group −2%p·native 대비 compatibility 95%·효율 5×/8× gate 중 하나라도 실패하면
  stable bus의 CVPR 방법 기여를 중단한다.
- 기존 full-216 sealed split은 이미 결과를 봤으므로 동기용 negative evidence만 허용한다. 새
  bridge/student는 다른 지역·acquisition의 untouched geographic-future test를 방법 선택 전에
  hash-freeze한다.
- cross-view robotics·latent world model·simulation은 EarthBus의 한 ablation이 아니라 paired
  aerial-ground 또는 trajectory 자산을 확보한 뒤 여는 별도 논문이다. 세부 트랙은
  `EMBEDDING_TRANSFER_CVPR_TRACKS.md`에 고정한다.

### RQ9. 산악지역 사이에서 무엇이 전이되고, 무엇은 지역 근거로 보정해야 하는가?

- H9a: HKH·알프스·한국의 원인 label은 달라도 `water/snow-ice/bare-debris/vegetation-loss/
  slope-failure` 시각 primitive 일부는 frozen Earth representation에서 저라벨로 전이된다.
- H9b: DEM·slope와 시점이 맞는 GLAMOS/ARPA/ICIMOD/한국 공공근거 residual은 EO-only보다
  unseen-region worst-group risk와 cloud/snow false positive를 낮추고, 불가능한 전이는 보류한다.
- 빙하·빙하호 task는 HKH↔Swiss Alps↔Monviso에서만 평가한다. 한국을 빙하 target으로 만들지 않고,
  한국은 산사태·산불·식생훼손·인간개입의 target/evidence 지역으로 둔다.
- frozen probe가 실패하고 raw S1/S2/DEM baseline만 성공하면 embedding adapter 주장을 중단하고
  재임베딩으로 전환한다. 지역명/위경도 shortcut만으로 이득이 재현돼도 전이를 기각한다.
- 데이터 자산, 두 task track, 단계별 실험과 kill gate는 `MOUNTAIN_EVIDENCE_TRANSFER.md`에 둔다.

### K-ALIGN main-paper 승격 규칙

RQ7의 `E_repr`와 RQ8의 `E_compat`가 모두 사전 gate를 통과할 때만 하나의 K-ALIGN main paper로
합친다. stable cache `z_stable`과 timestamped public-context residual `r_context`를 분리하고,
`E_repr / E_compat / E_fusion / E_refresh`를 별도 표로 보고한다. `E_fusion`만 통과하면
`Context Under Coverage`로, `E_compat`만 통과하면 `Compatible Earth Representation Bus`로 다시
분리한다. authoritative architecture·source role·4단계 효과·실행 queue는
`KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`에 둔다.

## 플래그십 실험: WorldShift × ModelShift

### 요인 설계

| 축 | 수준 |
|---|---|
| 실제 시간 | 제주 2023, 2024, 2025, rolling-2026 |
| 입력 장면선택·합성 | legacy ordered selection, coverage×3 + Sentinel2SCLBestClear |
| 모델 | OlmoEarth v1 Base, v1.2 Base |
| 품질 층 | land/sea, cloud/nodata, 연안/내륙, 토지피복 |

동일 픽셀 tensor를 두 모델에 넣는 셀과, 동일 모델에 item/pixel hash가 실제로 다른 두 입력을
넣는 셀을 모두 둔다. 그래야 `world shift`, `input shift`, `model shift`, 상호작용을 분리할 수
있다. 설정명만 다르고 item/pixel이 같은 셀은 재현성 감사에는 남기되 요인 실험에서는 제외한다.

### 베이스라인

1. sealed gallery의 raw cross-version cosine(negative compatibility baseline; 8건 smoke 금지)
2. train-only 공통 mean-centering
3. train-only 연도/도메인 centering(평가연도 통계 사용 금지)
4. spatial calibration split에 fit한 Orthogonal Procrustes
5. full recompute
6. 무작위 25% recompute
7. 품질만으로 고른 25% recompute
8. 제안 방법: 품질 × 릴리스 드리프트 × 의사결정 민감도

### 최종 지표

- 표현: CKA, 분포 거리, 공간 자기상관
- 검색: neighbor overlap, Top-k Jaccard, Kendall tau
- 지도: 구역별 면적/비율 오차와 bootstrap CI
- 통계: map-only / labels-only / PPI의 편향, coverage, interval width
- 시스템: GPU-second/km², 다운로드·materialize·추론 분해, 재계산 비율-품질 곡선
- 파트너: 검토 시간, false positive/누락 비용, 실제로 바뀌거나 확인된 결정 1건

### K-EvidenceShift로 확장되는 축

WorldShift × ModelShift는 release audit의 통제 실험이고, K-Context/K-EvidenceShift는 이를 한국
target domain의 공공 context 적응·선택적 결정 문제로 확장한다.

`시간 × 지역 × 센서/구름 × 입력 recipe × 모델 family/release × label budget × evidence coverage`

두 track은 같은 manifest와 site-event split을 공유하지만 논문 주기여는 분리한다.

1. CVPR/ICCV main stretch: 동적 public-context benchmark, provenance-aware adapter,
   EO-only privileged distillation, 자연 누락·지역/미래연도 OOD.
2. 후속 E&D/TMLR: 공식근거 누락·지연·충돌을 포함한 evidence-aware selective detection,
   PPI, release continuity.
3. 후속 active-label: 1의 headroom이 확인된 뒤 disagreement·공간 다양성·검수비용을 결합.

공공데이터 결합은 `입력 품질`, `표현`, `예측`, `선택적 결정` 중 어느 층을 바꿨는지 따로
보고한다. late fusion이나 검증만 바뀐 결과를 embedding 향상이라고 부르지 않는다.

## MARC 검증장의 정확한 역할

OlmoEarth 40m 임베딩은 돌고래 개체·행동을 관측하지 못한다. 따라서 이 프로젝트는
“위성으로 돌고래를 찾는다”고 주장하지 않는다.

관측 가능한 것은 연안 토지·수면의 변화, 육상양식장과 해안 인프라, 탁도/색 변화 같은
**서식지 압력의 맥락 후보**다. MARC의 목시·사진식별·행동·서식지 이용 연구가 있다면,
이 지도는 다음을 보조할 수 있다.

1. 현장조사에서 다시 볼 공간·시기를 우선순위화한다.
2. 장기 관찰 변화와 함께 검토할 인간활동/연안변화 가설을 제시한다.
3. 보전 캠페인·정책 의견서에 들어갈 전후 영상과 불확실성을 재현 가능하게 만든다.

현장 데이터 접근, 공동연구, 파트너 관계는 아직 성립하지 않았다고 명시한다.
민감한 개체 위치를 공개하지 않고, 인과효과는 별도 설계 없이는 주장하지 않는다.

### 한국형 evidence stack

제주 검증장은 하나의 지도를 정답으로 쓰지 않고 출처·시점·공간 단위가 다른 증거를 계층화한다.

| 층 | 현재 역할 | 확정할 수 없는 것 |
|---|---|---|
| Sentinel-2 2023–2026 | 같은 계절의 지속 지표 변화 탐지 | 개발 종류·허가·인과 |
| offline OSM 대한민국 snapshot | 현재 도로·건물·골프장·태양광·peak 문맥 | 실제 조성 시점, 공식 경계 |
| 제주특별자치도 오름현황 | 공식 명칭·주소·표고·면적 확인 | 오름 polygon 포함 여부 |
| 국토부 개발행위허가 | PNU·허가일·목적·용도지역 후보 | 누락 snapshot의 0건을 허가 없음으로 해석 |
| 2025 제주 FarmMap | 농경지 polygon·PNU·항공 관측일의 독립 상태 확인 | 개발 원인·오름 공식 경계·point miss의 음성 해석 |
| 다음: 지적/PNU·행정사건·항공사진 | 필지·사업구역·상태지도·시점 일치 | 현장 검증 없는 생태 영향 |

첫 4사이트 결합은 변화 판정을 번복하지 않았지만 2건의 개발/인프라 해석을 강화하고 2건에
대안 문맥을 추가했다. 특히 `r11`의 공식 고이악 416 m와 태양광 발전소 6개(419–951 m)는
후속 조사 순위를 바꿨다. 이 “결합 전/후 결정 변화” 자체를 파트너 지표와 논문 ablation으로
사용하며, 근접 증거가 경계·시점 증거로 승격되기 전에는 원인 주장을 금지한다.

첫 실제 FarmMap ingest는 이 경계를 더 선명하게 만들었다. 289,379개 공식 polygon을 전수 처리해
4개 변화좌표 중 `r08`만 밭 polygon과 정확히 겹쳤다. 그러나 실제 항공 관측일은 2022-12-30으로
변화 전 Sentinel 영상보다 503일 앞서므로, 이 edge는 **변화 전 상태 baseline B**이지 원인 B가
아니다. OSM 오름점 243개 중 7개 point hit는 공식 오름 경계가 없어 C로 제한했다. 따라서
상태근거가 늘어도 원인근거 0/368과 `367 abstain / 1 investigate` 결정은 바뀌지 않았다. 이것이
레이어 수가 아니라 weakest-link evidence policy를 논문 대상으로 삼아야 하는 직접 사례다.

공식 제공처 23개 연결 후보를 조사한 결과, 연결 순서는 `연속지적도 PNU → 환경영향평가·
건축인허가·개발행위허가·사유림사업 → 팜맵·토지피복·항공사진 → 보전/규제·기상`으로 고정한다.
PNU는 서로 다른 행정표를 잇는 spine이지 오름 경계나 원인 증명 자체가 아니다. 공식 오름 주소의
지번은 대표 필지일 수 있으므로 변화 footprint와 공식 사건 polygon의 실제 중첩이 없으면 B급으로
승격하지 않는다. 데이터별 접근조건·join·누락 규칙은 `KOREA_PUBLIC_DATA_CATALOG.md`에 고정했다.

새 연구 실험은 공공 레이어 수가 아니라 다음 evidence-source ablation이다.

`모델+OSM → +공식 PNU → +날짜별 상태지도 → +행정사건 → +항공/현장 독립검증`

각 단계에서 고정 분모 368개의 time-aligned coverage, 판정/보류율, 사람 판정 대비 selective risk,
지역·토지피복별 침묵률을 측정한다. 이로써 “공공데이터가 있으면 정확하다”가 아니라 **어느 출처의
누락이 어느 장소에서 모델의 발언 가능성을 없애는지**를 결과로 만든다.

### K-Earth Evidence Graph — 오름 368 전수 레지스트리

한국형 기여는 공공 레이어의 개수가 아니라 **이질적이고 누락된 근거 아래에서 모델이 언제
말하고 언제 보류해야 하는지**를 측정하는 것이다. 제주 공식 오름 368건을 고정 분모로 삼고,
각 레코드를 다음 상태기계로 저장한다.

`공식 목록 → 위치 해석 → 위성 screen → 필지·사업구역 근거 → 사람 검수 → 판정/조사/보류`

| 등급 | 의미 | 현재 예시 | 허용되는 주장 |
|---|---|---|---|
| A | 공식 레코드 또는 시점이 맞는 공식 polygon 중첩 | 제주 오름현황의 명칭·주소·속성 | 공식 목록에 등재됨 |
| B | 공식 원자료에서 재현한 공간·시간 파생근거 | `r08`의 변화 전 FarmMap 농지 상태 | 경계·시점이 검증된 상태/행정근거; 원인은 별도 |
| C | 현재 커뮤니티 지도 point/name | offline OSM peak | 위치 후보, 공식 경계 아님 |
| D | 같은 행정리·거리 근접 문맥 | 허가 지역명, 인접 시설 | 후속 조회 단서, 원인 아님 |
| M | OlmoEarth 변화 screen | 4기간·12기간 점별 순위 | 조사 우선순위, 인과 아님 |
| U | 미조회·누락·불명확 | 키 부재, 주소만 있음 | 보류 사유 |

2026-08-22 첫 실행에서 368/368을 상태화했고, 사용자 제공 제주시 210건 표는 공식 연번이
아닌 자체 순번임을 발견해 연번 결합을 폐기했다. 오름명·소재지·면적 복합키로 209건을 연결했으며
188건은 핵심 필드가 일치하고 21건은 주로 주소가 달랐고 `빈내오름` 1건은 공식 2024-03-31
목록에서 연결되지 않았다. offline OSM peak는 243/368만 보수적으로 위치화했다. 이 값들은
“오름 변화 243건 조사 완료”가 아니라 **위성 screen을 시도할 수 있는 현재 point coverage**다.

선택 정책은 사전에 고정한다.

1. 4기간·12기간 점수가 모두 위치화 오름의 상위 10%이고 최대 변화시점이 같을 때만
   `model-high-stable`로 둔다. 둘 중 하나만 높으면 모델 불안정으로 보류한다.
2. 필지 또는 환경영향평가 사업구역의 경계 중첩과 변화시점 일치가 없으면 원인을 말하지 않는다.
3. A/B급 원인 근거가 전체의 10% 미만이면 논문 질문을 원인 분류에서
   **불완전 행정기록 아래의 선택적 변화탐지와 abstention calibration**으로 자동 전환한다.
4. Top-k는 조사 우선순위로만 쓰고, 전체 변화율은 별도 층화 확률표본과 PPI 신뢰구간으로 추정한다.

따라서 첫 버전의 원인 근거 가용률은 0/368이고, 시스템은 실제로 선택적 변화탐지 모드로
전환된다. 이 0은 “개발이 없다”가 아니라 **현재 파이프라인이 원인을 검증할 공식 경계·시점
근거를 아직 확보하지 못했다**는 메타데이터다. 연구 기여 후보는 모델 정확도 하나가 아니라
`evidence coverage–risk–abstention` 곡선과, 자료 누락이 지역·오름 유형별 판정을 어떻게
선택적으로 침묵시키는지에 있다.

기존 v6 임베딩을 재사용한 첫 screen은 OSM 위치가 해결된 243/368을 모두 처리했다. 4기간과
12기간이 모두 상위 10%이고 최대 변화시점도 같은 `high-stable` 후보는 8건이었다. 그러나
각 연도 5월 최근접·고정 stretch RGB를 보니 8/8 모두 2023 구름 또는 해무를 공유한
false positive였고, 고확신 지속 지표 변화는 0건이었다. 기존 사람 후보가 416 m 떨어진 고이악
peak를 포함해 총 9건을 검수한 뒤 8건은 기각, 성산일출봉 1건만 입력 품질 부족으로 불확실하게
남았다. 즉 **모델 입력 변형 간 합의는 독립 증거가 아니며, 같은 오염을 안정적으로 재현할 수
있다.** 이후 선택 정책은 `4/12 합의`에 더해 SCL 입력품질 게이트와 RGB 검수를 필수로 둔다.

## 하나의 프로젝트가 남기는 네 산출물

| 산출물 | 사용자 | 성공의 증거 |
|---|---|---|
| 논문 A/C 통합 후보: *K-ALIGN* | CVPR/ICCV main stretch·박사 지원 | public-context EO-only distillation + multi-teacher compatible bus + refresh Pareto; 두 gate 동시 통과 |
| 논문 B: *When the Map Changes Twice* | E&D/TMLR/학계 | paired release/evidence benchmark, selective risk, CI, 두 번째 태스크 전이 |
| 논문 D: *Earth-to-Embodied* | CVPR/CoRL 후속 | paired satellite–drone/ground와 field-side no-retrain; trajectory가 있을 때만 navigation |
| `olmoearth-release-audit` | Ai2/엔지니어 | 작은 공개 sample, manifest, 재현 명령, 이슈/PR |
| Jeju Coastal Pressure Evidence Pack | MARC형 파트너 | 검증 칩, 후보·불확실성 지도, 결정 질문에 대한 답 |
| 기술 에세이/포트폴리오 | 채용·지원위원회 | 실패 v1→v5가 방법 설계로 이어진 서사와 공개 링크 |

“대단함”은 범위를 키우는 데서 나오지 않는다. 같은 실험이 논문 표, 공개 도구,
파트너 결정을 동시에 만들 때 생긴다.

## 박사 지원 계보와의 연결

| 타깃 | 이 프로젝트가 보여줄 fit | 제출 전 필요한 증거 |
|---|---|---|
| MIT Sherrie Wang / Earth Intelligence Lab | ML 지도 오차를 downstream 추론과 CI까지 전파 | PPI 표, 확률표본 설계, map-only 편향 사례 |
| UW Karen Chen / Yale Karen Seto | 지표·노출 정의가 환경/불평등 결론을 바꾸는 과정 | 두 합성/품질 정의가 지역 결론을 바꾸는 ablation |
| UT Austin Gengchen Mai | geospatial representation의 안정성·공간 전이 | v1↔v1.2 표현 정렬과 neighbor 안정성 |
| Ai2 OlmoEarth | 공개 모델을 파트너 배포까지 가져가는 재현성 엔지니어링 | LFMC 이슈, sample PR, release-audit 하네스 |
| MARC형 생태 파트너 | 모델이 아니라 실제 연구·보전 결정을 출발점으로 삼음 | 공동 정의한 질문, 검수 기록, 의사결정 변화 |

지원서의 중심 문장은 교수 이름이 아니라 다음 연구 궤적이어야 한다.

> “I study how changes in Earth, sensing pipelines, and foundation models propagate into
> environmental decisions, and how sparse field validation can keep those decisions statistically
> valid under limited compute.”

## 12주 실행 순서

### 0–1주: 공개 신용장과 입력 검증

- 제주 v5 계산 완주(216/216) ✅; 품질 감사에서 v1과 의미적으로 동일해 입력개선 가설 기각 ❌
- SCL BestClear 대표-window 합성 smoke test ✅ — bad proxy −95.64%, 고정 target 1.00→0.00,
  RGB 확인. 단, 1윈도우이므로 연도×오염도 사전 층화 다중-window 검증은 ⏳
- LFMC 재현성 보고 전달, sample schema PR 제출
- v1/v1.2 동일 입력 manifest와 smoke 8개 고정 ✅ — checkpoint/input SHA·비파괴 dataset view·
  GPU-busy fail-safe preflight 완료, embedding inference만 ⏳
- 제주 14후보 audit-only site-event schema·누수 gate·completion manifest ✅

### 2–4주: paired release audit

- 제주 v1/v1.2 × 입력 레시피 요인 실험
- neighbor/Top-k/집계 안정성 표와 failure atlas 작성
- 작은 공개 sample + 재현 스크립트 공개

- 제주 schema를 유지하면서 추가 2지역 sampling frame을 동결
- sealed probability test 300 + active/train pool 300을 분리하고, 최소 120건 이중판독·agreement 측정

### 5–7주: 통계적으로 유효한 결론

- 변화 점수 층화 확률표본 설계
- RGB 칩 블라인드 판정 프로토콜
- map-only / labels-only / PPI 비교와 신뢰구간
- matched scratch/generic vision/Olmo v1·v1.2/Prithvi/CROMA 또는 TerraMind의 frozen·PEFT baseline
- 지역·연도·구름별 전이효과와 negative-transfer CI

### 8–9주: 파트너 공동설계

- MARC 또는 독립 생태 파트너와 결정 인터뷰
- “무엇을 얼마나 자주 보고, FP/FN 비용이 무엇인가” 고정
- 민감 위치·데이터 권리·공개 범위 합의 전에는 데이터를 받지 않는다

### 10–12주: 선택적 갱신과 전이

- full/random/quality-only/proposed refresh 곡선
- LFMC 또는 양식장으로 두 번째 태스크 전이
- random/층화/uncertainty/disagreement/CLUE·cost-aware acquisition의 같은-budget offline replay
- 논문형 리포트, 파트너용 Evidence Pack, Ai2용 기술 보고를 같은 결과에서 생성

## 중단·축소 게이트

- v1↔v1.2 차이가 사전 허용범위보다 작으면 “릴리스 문제”를 과장하지 않고 입력 품질 감사로 축소한다.
- PPI가 labels-only보다 유효성/효율 이득이 없으면 현장 표본만 사용한다.
- 파트너가 실제 결정을 특정하지 못하면 MARC 브랜딩을 제거하고 방법 벤치마크로만 공개한다.
- 두 번째 태스크 전이에 실패하면 일반 방법이 아니라 제주 사례 연구로 쓴다.
- scratch보다 평균이 좋아도 사전 지역·연도·구름 그룹에서 반복되는 negative transfer를 숨기지
  않는다. active acquisition이 층화 random/CLUE의 AULC를 이기지 못하면 새 방법 주장을 삭제한다.
- 실제 독립 기관 3곳과 데이터 반출 제약이 없으면 federated learning은 본문에서 제외한다.
- 독립 파트너 2곳의 반복 수요 전에는 SaaS나 한반도 전체 materialize를 시작하지 않는다.

## 근거 자료

- MIT Earth Intelligence Lab: https://earthintelligence.mit.edu/
- Lu et al., *Regression coefficient estimation from remote sensing maps*:
  https://arxiv.org/abs/2407.13659
- Prediction-Powered Inference: https://arxiv.org/abs/2301.09633
- Ai2 OlmoEarth Platform: https://allenai.org/blog/olmoearth
- OlmoEarth v1.2: https://allenai.org/papers/olmoearth-v1-2
- Earth Embeddings: https://arxiv.org/abs/2608.03410
- EarthShift: https://arxiv.org/abs/2605.29330
- MARC 연구·보전 활동: https://marckorea718.org/
