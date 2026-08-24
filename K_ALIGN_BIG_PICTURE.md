# 큰 그림 — EarthKV와 Earth embedding 생명주기

작성: 2026-08-23  
최종 보정: 2026-08-24
상태: 장기 프로그램 spine. 첫 논문은 `EarthEmbedContract`로 더 좁게 유지

## 한 문장

> **아카이브된 Earth embedding은 모델 릴리스·시간창·밴드·해상도·풀링 계약 안에서만 의미가
> 있으며, EarthKV는 이 page-level latent를 검증·재사용·수리·재계산·보류하는 생명주기다.**

우리가 해결할 문제는 “더 복잡한 adapter를 만드는 것”이 아니라, **어떤 embedding을 안전하게
비교·재사용할 수 있고 무엇을 다시 계산해야 하는지 자동으로 판정하는 것**이다.

현재 실측은 계약이 바뀌어도 파일·차원·실행이 정상인 채 좌표 identity가 깨지고, 시간계약이
무효인 변화 후보가 높은 점수를 받을 수 있음을 보였다. **실제 downstream task의 고확신 오답은
아직 측정하지 않았다.** 이 구분을 지키지 않으면 representation proxy를 의사결정 오류로 과장하게 된다.

## 이름들은 서로를 대체하지 않는다

| 층위 | 이름 | 정확한 역할 | 현재 증거 |
|---|---|---|---|
| 장기 연구·시스템 프로그램 | **EarthKV** | `(space, time, release, contract)`로 주소화된 latent page의 admission·invalidation·repair·precision·eviction | contract audit와 repair 자산만 있음. 완성 시스템은 없음 |
| 첫 논문 | **EarthEmbedContract** | 비교 전에 `REUSE / ADAPT / RECOMPUTE / ABSTAIN`을 판정하고 task risk를 줄이는가 | M1–M5 완료, downstream task 미측정 |
| 수리 연산자 | **FoldRefresh** | 일부 page만 갱신해 통계량을 유지하는가 | 별도 프로젝트의 제출·실험 자산. 여기서는 재사용 |
| 정책 층 | **EarthRoute** | 다음에 cheap refresh·재계산·새 관측·사람검수 중 무엇을 살 것인가 | 설계만 있음 |
| 외부 평가 domain | **MountainShift** | 대륙·센서·근거밀도가 달라질 때 위 판정이 유지되는가 | 공개 benchmark 조사 단계 |

따라서 첫 논문 제목에 paging·eviction·distributed cache를 넣지 않는다. 구현·측정하지 않은
EarthKV 전체를 논문 기여처럼 쓰면 좋은 연구 프로그램이 약한 시스템 비유로 보인다.

## 2026-08-24 경쟁 경계 보정

`기존 embedding 제품에는 버전 의미가 없다`는 주장은 틀렸다.

- AlphaEarth 공개 컬렉션은 `MODEL_VERSION`, `PROCESSING_SOFTWARE_VERSION`,
  `DATASET_VERSION`을 제공한다.
- TESSERA Zarr convention은 `dataset_version`, `model_version`, `build_version`을 정의하고,
  서로 다른 model version의 store를 명시적 정렬·재임베딩 없이 섞지 말라고 경고한다.
- Major TOM OlmoEarth/Clay 두 제품에서는 우리가 검사한 8개 생성계약 필드가 기계 판독 스키마에
  없었고 `unique_id` 교집합도 0이었다. 이것은 **한 제품군의 실측 gap**이지 생태계 전체의 부재가 아니다.

따라서 novelty는 metadata field를 새로 발명하는 데 있지 않다. 남은 질문은
**버전 필드가 있어도 실제 task가 호환되는지 어떻게 검증하고, 불일치 때 어떤 action이 비용 대비
안전한지**다. 표준에는 validation semantics를, 논문에는 task-risk 감소를 기여해야 한다.

## 이미 발견한 두 증거

### 1. 모델 계약이 바뀌었다

동일한 제주 입력을 OlmoEarth v1과 v1.2에 넣었을 때 same-release 검색 R@1은 1.0이지만,
cross-release R@1은 양방향 0이었다. affine 정렬도 약 61–70%까지만 복구했다.

즉 파일 형식과 차원은 같아도 모델 릴리스가 다르면 같은 좌표계라고 볼 수 없다.

### 2. 시간 계약이 어긋났다

- 2025 연간창과 rolling-2026 창은 2025년 7–12월 **184일을 공유**한다.
- 기존 4기간 경로는 2023–2025의 가을·겨울과 rolling-2026의 봄을 비교했다.
- 4기간과 12기간의 Top-30 교집합은 5개, Jaccard는 0.091이었다.
- 기존 14후보 중 중첩 전이 5건, 4기간 source 5건, 합집합 **9/14**가 시간계약 결함에 노출됐다.

이 9건이 모두 시각적 false positive라는 뜻은 아니다. 실제 변화가 우연히 포함됐더라도,
**연도 변화탐지 결과로 해석할 자격이 없는 후보**라는 뜻이다.

## 해결책도 단순하다

모든 embedding에 다음 계약을 붙인다.

```text
model release + weight hash
time window + 실제 acquisition dates
band order + scaling
GSD/CRS + spatial support
temporal recipe + pooling
input/output content hash
```

그리고 검색·변화탐지 전에 세 가지 중 하나를 자동으로 선택한다.

1. **REUSE** — 계약이 동일하거나 호환성이 검증됨.
2. **ADAPT** — 별도 calibration에서 bridge가 task gate를 통과함.
3. **RECOMPUTE / ABSTAIN** — 시간 중첩·계절 불일치·미검증 릴리스라서 재계산하거나 보류.

핵심은 높은 z-score나 cosine보다 **비교 자격을 먼저 확인하는 것**이다.

## 한국 데이터의 역할

한국은 부록이 아니다. 제주에서 실제로 시간계약 오류가 고확신 변화 후보를 만들었고,
한국 공공데이터는 입력이 유효한 후보를 그 다음 단계에서 검증하는 계측기다.

다만 현재 `0/368`을 전부 공공데이터 부족으로 설명하지 않는다.

1. 후보 생성 시간계약이 먼저 오염됐다.
2. 공식 오름 자료가 점 위치 중심이라 모델 후보·필지·오름 경계를 정확히 잇지 못했다.
3. 개발행위허가 snapshot에는 제주 2023·2024 행이 0인 실제 coverage hole이 있다.

“오름은 보전지역이라 행정사건이 구조적으로 0”이라는 설명은 아직 증명되지 않았다. 오름 polygon과
제도별 모집단을 확보하기 전에는 geometry/coverage 미확정으로 둔다.

## 바로 고칠 것

1. 기존 rolling-2026을 연간 2026처럼 사용하지 않는다.
2. 4기간 후보 경로는 실패 재현 외 기본 실행을 금지한다.
3. 현재 12기간도 월 집합이 완전히 같지 않으므로 그대로 새 후보를 만들지 않는다.
4. 2025와 2026의 **동일한 계절 구간**을 새 계약으로 만들고 encoder를 다시 실행한다.
5. 새 후보와 기존 후보의 순위·오염·공공근거를 비교한다.

저장된 768채널은 월별 축이 아니라 융합된 feature이므로 사후에 특정 월만 잘라낼 수 없다.
하지만 원본 materialized raster를 재사용하면 전체 프로젝트를 다시 다운로드할 필요는 없다.
정확한 실행시간은 새 입력 view와 compact output으로 직접 잰다.

## 논문의 모양

가제:

> **EarthEmbedContract: Preventing Silent High-Confidence Errors in Reused Earth Embeddings**

논문 질문은 하나다.

> **Earth embedding의 생성 계약이 바뀌었을 때, 비교를 허용·보정·재계산·보류하는 판정이
> 잘못된 고확신 결정을 얼마나 줄이는가?**

Figure 1은 두 실패를 나란히 둔다.

- 같은 입력 + 다른 모델 릴리스 → cache 검색 붕괴.
- 같은 모델 + 다른 시간창 계약 → 변화 후보 붕괴.

그 다음 여러 EO model·공개 change/retrieval task에서 계약 mismatch를 의도적으로 만들고,
contract gate가 silent error를 줄이는지 측정한다. quantizer-aware adapter와 비용곡선은 필요할 때의
해결수단이지 논문의 중심 문장이 아니다.

### 논문을 살리는 결정적 실험

1. **Frozen-head silent error** — old release로 학습한 head를 그대로 고정하고 new release 또는
   계약 mutation을 넣어 정확도·calibration·고확신 오답을 측정한다. retrain upper bound도 함께 둔다.
2. **현실적인 mutation** — band order뿐 아니라 reflectance scale, acquisition/time recipe,
   pooling, GSD/resampling, nodata mask를 한 축씩 바꾼다. 단순 synthetic corruption과 분리한다.
3. **두 번째 family/release** — Major TOM의 OlmoEarth↔Clay는 cross-family라 release 복제가 아니다.
   Clay v1.0↔v1.5처럼 실제 release pair를 동일 입력에서 다시 계산하거나 다른 공개 release pair를 찾는다.
4. **Gate baseline** — version tag exact match, quality filter, full re-embed, Procrustes/ridge,
   query-side bridge, old/new dual index와 비교한다.
5. **Risk–cost curve** — unsafe reuse율·AURC·task delta와 재계산 비율·GPU/I/O 비용을 같은 표에 둔다.

`R@1=0`은 좌표 호환성 실패의 강한 증거지만 task failure 자체는 아니다. frozen-head 실험이 실패하면
논문 headline을 `silent high-confidence error`에서 `representation compatibility audit`로 낮춘다.

## 2026-08-24 확정 — P0 실험 경로 (C1·C2-A 종결 후)

M7·M8로 두 사실이 확정됐다. (1) M1의 same-token 비교는 유효하다. (2) v1.2에서는
partial-group missingness를 표현할 수단이 없고 그 무시가 조용하다.

두 번째 사실이 PhilEO를 그대로 쓸 수 없게 만든다. PhilEO S2 10밴드의 결측(B01·B09)은
v1에서는 band_set 부재로 선언 가능하지만 v1.2에서는 무시된다. 따라서 release×task를 한
데이터에서 바로 교차하면 처리 방식 차이가 release effect로 오인된다.

### 실행 순서 (확정)

| 단계 | 내용 | 통과/중단 기준 |
|---|---|---|
| ~~C2-A~~ | v1.2 mask 경로 폐쇄 | **완료. 게이트 6/6** (M8) |
| **C2-B (즉시)** | PhilEO **v1 단독** task-risk. land-cover seg / building-density reg / road-density reg 세 head를 frozen v1 embedding 위에 학습하고 perturbation(band-order dose, band_set 2 missing, embedding compression, spatial/token corruption)을 가한다 | 같은 손상에 세 task의 loss·calibration 순위가 **다르지 않으면** task-aware router 방향을 중단하거나 축소한다 |
| **C2-C (바로 다음)** | PhilEO 12밴드 재물질화 feasibility. 지역·split별 **20 sample 사전 고정** | 아래 4조건 전부 충족해야 통과 |
| C2-D | 통과 시 최종 P0: PhilEO full 12-band × v1/v1.2 × 3 task × old/new/mixed cache | — |

C2-C 통과 조건 (사전 등록):
1. 동일 acquisition product 식별 (CRS·transform·파일명·SCL에서 복구 → STAC 조회)
2. spatial grid 일치
3. 기존 10밴드가 재물질화 결과와 quantization/resampling 허용오차 내 일치
4. B01·B09의 실제 관측 복구 (합성·상수 채움이 아님)

실패 시: 실패 자체를 M-항목으로 기록하고, C2-B를 유지한 채 **다른 공개 12밴드 multi-task**를
찾아 release 축을 검증한다. 상수 채움은 headline이 아니라 **보조 ablation**으로만 쓴다
(두 릴리스가 상수를 다르게 해석하므로 순수 release effect가 아니다).

### 데이터셋 역할 고정

```
PhilEO         → 통제된 shared-cache / multi-task 실험 (P0)
EarthShift     → 센서·지역·시간 shift
AI-Hub 제주·지리산 → 외부 stress test와 operational demonstration (P0 아님)
```

AI-Hub를 P0로 올리면 접근 제한·ontology·co-registration을 동시에 풀어야 해서 모델 릴리스
연구보다 한국 데이터 정합 연구가 앞에 나온다. 공개 benchmark에서 방법을 고정한 뒤 AI-Hub에서
검증하면 한국 데이터가 훨씬 강한 외부 증거가 된다. 사용자가 데이터 확보를 진행 중이다.

### 갱신된 서사

> OlmoEarth v1→v1.2의 same-input audit에서 cache identity 붕괴를 발견했다. 원인을 추적하자
> 모델 크기뿐 아니라 Sentinel-2 tokenization contract도 3개 band set에서 단일 12-band group으로
> 바뀌었다. 이를 공개 multi-task 데이터 PhilEO에 적용하려 하자, 동일한 10밴드 제품의 missingness를
> 두 릴리스에 대칭적으로 표현할 수 없었다 — 게다가 그 실패는 조용하다. 따라서 EO cache upgrade는
> 단순 embedding alignment가 아니라 release-aware input admissibility와 task-specific risk를
> 함께 다뤄야 한다.

## 2026-08-25 재확정 — AI-Hub 71363 기반 P0 (M9 반영)

M9로 두 가지가 바뀌었다. (1) 실제 조합은 2,699쌍이고 곱집합 42,714가 아니다.
(2) 공식 split은 valid 110/110이 train과 공간 중첩이라 쓸 수 없다.

### 주장 범위를 좁힌다 (지금 단계에서 안전한 표현)

세 task는 **하나의 GeoJSON ontology에서 클래스를 나눈 것**이다. 따라서 리뷰어가
"multi-class segmentation 하나를 세 head로 쪼갠 것 아닌가"라고 물을 수 있다. 표현을 제한한다.

| task | 정의 |
|---|---|
| Task-LC | 일반 토지피복 semantic segmentation (산림·하천·밭·건물·도로) |
| Task-Logging | 벌목지 binary segmentation (665 타일) |
| Task-Landslide | 산사태·토석류 binary segmentation (373 타일) |

- **지금 주장하는 것**: shared-cache, heterogeneous-risk **three-head test**
- **아직 주장하지 않는 것**: 폭넓은 multi-task generalization
  → 공개 task(PhilEO·PANGAEA)를 추가한 뒤에 일반화를 주장한다

### 센서 축은 후속 stress test로 미룬다 (직전 제안 철회)

내가 SkySat을 고해상도 대리 센서 후보로 제안했다. **철회한다.**

- SkySat은 Planet의 위성이고 **KOMPSAT/아리랑이 아니다.** 메타데이터 확인 전에 혼용하면 안 된다
- 센서별 ontology 차이(드론 10 / SkySat 8 / S2 9 / Landsat 8)는 우선
  **sensor contract heterogeneity의 증거**일 뿐이다
- 센서 간 embedding cache 재사용 가능성의 증거는 아니다. 그것은
  동일 위치 · 유사 시점 · 공통 클래스 · 정합 해상도가 확보돼야 성립한다

### 실행 순서 (확정)

| 단계 | 내용 | 통과/중단 기준 |
|---|---|---|
| ~~C2-A~~ | v1.2 mask 경로 폐쇄 | **완료. 6/6** (M8) |
| ~~C2-P~~ | 71363 인벤토리·계약·split 감사 | **완료** (M9) |
| **C2-S (즉시)** | **AOI 군집 단위 spatial holdout 구축.** 13개 군집으로 leave-one-cluster-out. test 군집 **동결** | 군집 간 최소 이격이 1 타일 폭(10.24 km) 이상이어야 통과. 공식 split·탐색에 쓴 valid 300은 test로 쓰지 않는다 |
| **C2-C** | **AI-Hub exact-scene recovery gate.** 20~50 표본만 검사 | 아래 5조건 전부 |
| C2-B | three-head risk 차이 측정 | 사전 등록 기준(별도 고정) |
| C2-D | 통과 시 B01·B09 추가 + 2,699쌍으로 확장 | — |

### C2-C 표본 층화 (20~50개)

일반 토지피복 / 벌목 positive / 산사태 positive / task 동시등장 /
서로 다른 공원(군집)·연도·S2A·S2B — 각 층에서 뽑고 **사전에 고정**한다.

표본마다 보존할 값:

```
sample_id · geometry/CRS/affine transform · img_time · platform
원본 10밴드 checksum · label checksum · 클래스별 픽셀 면적
candidate STAC item ID
```

### C2-C 통과 조건 (사전 등록)

1. 같은 날짜·platform의 후보 STAC item을 **자동으로** 찾을 수 있다 (수작업 cherry-pick 금지)
2. 좌표계·grid·affine transform을 정확히 재현할 수 있다
3. 공통 10밴드가 **사전에 정한 오차** 안에서 일치한다
4. 후보 선택이 결정적이다 (같은 입력 → 같은 item)
5. 표본의 **90% 이상**이 복구 가능하다

**일치하지 않는데 B01·B09만 추가하면 12밴드 보강이 아니라 서로 다른 처리 파이프라인의
영상을 섞는 것이다.** 같은 날짜·bbox라도 제품·processing baseline·mosaic·구름 처리가 다를 수 있다.

**실패 시**: 데이터를 버리지 않는다. 실험 명칭을
`exact observation enrichment` → **`source/reprocessing shift`**로 바꾼다.
서로 다른 처리 파이프라인이 같은 캐시를 무효화하는지가 그 자체로 계약축이다.

### 논문 골격

```
M1 / M8      릴리스가 바뀌면 embedding·mask contract가 실제로 깨진다
   ↓
AI-Hub S2    같은 cache 변화라도 task별 손실이 다르다 (three-head test)
   ↓
router/gate  어떤 spatial representation을 언제 갱신할지 결정한다
   ↓
공개 task    일반화 + 독립 지역 외부 stress test
   ↓
SkySat/드론/Landsat   후속 sensor-contract 확장 (지금은 미포함)
```

### 현재 상태의 정확한 표현

"설계 가설이 증명됐다"가 아니다. **"P0를 실행할 수 있는 데이터 조건이 확인됐고,
공식 split은 쓸 수 없음을 측정했다"**가 정확하다.

## 중단 기준

- 시간창을 바로잡아도 후보가 동일하게 불안정하다.
- 모델 릴리스 오류가 preprocessing mismatch 하나로 완전히 설명된다.
- 계약 gate가 단순 quality filter보다 false alarm·task risk를 줄이지 못한다.
- 제주 한 사례에서만 나타나고 다른 model/task에서 반복되지 않는다.
- version tag exact-match 또는 full re-embed baseline이 같은 위험을 더 싸게 막는다.

이 경우 CVPR main을 주장하지 않고 제주 파이프라인 재현성·데이터 품질 보고서로 남긴다.
