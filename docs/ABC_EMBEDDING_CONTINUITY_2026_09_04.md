# A/B/C 확장 — Earth Embedding Continuity

작성: 2026-09-04  
상태: **설계 DRAFT — GPU 실행·sealed target 개봉 전**  
기계 판독 초안:

- A: `config/release_migration_prereg_draft_v0.json`
- B: `config/second_fm_cache_prereg_v1_draft.json`
- C: `config/safe_cache_action_prereg_draft_v0.json`

## 결론부터

세 방향을 독립 프로젝트로 벌리지 않는다.

> **세계 또는 embedding 제품이 바뀌었을 때, 이미 저장한 Earth embedding과 downstream head를
> 언제 그대로 쓰고, 조금 적응하고, 변환하고, 다시 계산할지를 label·risk·compute·cache
> invalidation 비용으로 결정한다.**

논문 구조에서 **A가 중심 방법/시스템 질문**, **B가 다른 제품으로의 외적 타당성**, **C가 실제
의사결정 정책**이다.

```text
                         변화가 무엇인가?
                 ┌──────────────┴──────────────┐
              세계/지역 변화                 모델/제품 변화
                 │                              │
       A0 그대로 / A1 head 적응          identity / bridge / re-embed
                 └──────────────┬──────────────┘
                                │
                   C: support·contract·cost만 보고
               REUSE / ADAPT / MIGRATE / RE-EMBED / REQUEST
```

## 1. 현재 손에 있는 과학적 출발점

### World shift

- Sen12Landslides: frozen OlmoEarth cache가 raw P2/P3보다 강했고, target label K=5/20에서
  A1(cache head 적응)이 raw A4w/A4h 적응보다 반복적으로 강했다(M65, MS-96/97).
- Solar Farm: zero-target P4가 raw P2보다 8/8 fold에서 강했고, A1은 raw 적응보다 강했다
  (MS-98/99). 그러나 A1이 A0보다 항상 좋은 것은 아니었다.
- 특히 Solar random K=5에서 양성 없는 support가 A1을 붕괴시켰다. 이 결과는 “항상 적응”이
  아니라 **안전한 action 선택**이 필요하다는 직접 근거다.

### Model shift

- M1은 같은 scene의 OlmoEarth v1/v1.2 token identity가 0이고, Procrustes와 affine ridge도
  등록 gate를 통과하지 못했음을 보였다. pooled CKA가 높아도 기존 검색/index 계약은 깨졌다.
- M85는 7지역 radar/optical probe에서 v1.2가 일관되게 우월하지 않음을 보인 **버전 성능 비교**다.
  M85는 downstream cache migration 실험이 아니다. A의 직접 출발점은 M1이다.

### 죽었고 다시 열지 않는 것

- P2/P4 prediction fusion·GeoContextGate: MS-90B/91/92 stop rule로 종료.
- label-free winner router: 기존 gate 실패. C는 target label을 전혀 안 쓰는 router가 아니라
  **사용자가 실제로 확보한 support label과 공개된 contract만 쓰는 안전 정책**이다.
- CacheTune A2 low-rank spatial residual: MS-94에서 A1보다 낮아 stop rule 발동. 이름을 바꿔
  되살리지 않는다.

## 2. A — Release migration: old head와 index를 살리는가

### 질문

> OlmoEarth가 v1에서 v1.2로 바뀌었을 때, 새 embedding을 라벨 없는 exact-scene pair로 old
> embedding 계약에 옮겨 기존 head·index를 유지할 수 있는가?

이것은 “v1.2가 더 정확한가?”와 다른 질문이다. 새 모델이 더 좋아도 기존 수백만 tile cache와
downstream head가 깨지면 운영 업그레이드 비용이 생긴다.

### 첫 testbed는 Solar Farm

Sen12는 B01·B09가 없어 v1과 v1.2 입력 missingness가 비대칭이다. 반면 Task-2 Solar cache의
OlmoEarth view는 12밴드가 모두 있어 같은 scene·밴드·시점으로 v1/v1.2를 비교할 수 있다.

첫 실험은 **query-side backward compatibility**다.

```text
기존: raw → v1 encoder → old cache/index → old task head
신규: 새 query raw → v1.2 encoder → bridge → old embedding contract → old index/head
                                               └ 과거 archive raw backfill 없음
```

cache-side old→new 변환은 저장된 전체 cache를 다시 쓰므로 “no backfill”이라고 부르지 않는다.

### 비교 arm

| arm | 의미 | label 사용 |
|---|---|---|
| R0 old reference | old head(v1 cache) | source head 학습에만 |
| R1 identity | old head(v1.2 cache) | 추가 없음 |
| R2 mean shift | channel 평균만 이동 | bridge label 0 |
| R3 Procrustes | paired orthogonal map | bridge label 0 |
| R4 affine ridge | paired affine map | bridge label 0 |
| R5 spatial stitch | 3×3 depthwise + 1×1 residual bridge | bridge label 0 |
| R6 new-native ceiling | 새 v1.2 head | source head 학습 |
| R7 full re-embed | 새 cache+새 head의 정확도·비용 상한 | source head 학습 |

R5는 처음부터 무조건 돌리지 않는다. 두 exposed fold에서 R4가 screen을 실패할 때만, 구조·loss를
별도 커밋한 뒤 한 번 연다. 단순 affine으로 충분하면 복잡한 stitch를 방법처럼 포장하지 않는다.

### 분리해야 할 두 성능

1. **representation compatibility**: same-token cosine, R@1, neighborhood overlap, CKA.
2. **decision continuity**: old head의 AP와 source-validation 고정 threshold 성능이 유지되는가.

M1은 1번을 측정했다. A가 새로 채우는 것은 2번이다. 높은 CKA를 task 호환성으로 대신하지 않는다.

### 학습·평가 분리

- bridge fit: target fold를 제외한 source의 paired v1/v1.2 token만. task label 0.
- bridge 선택: source validation fold.
- final: held-out target fold. label은 마지막 utility 평가에만 사용.
- threshold: source validation에서 정해 동결. 기존 query empty-label FP matching은 운영 지표로 쓰지 않는다.

### A 승급 조건 초안

- bridged old-head AP가 R0 대비 절대 `−0.02` 이내,
- fixed-threshold primary IoU가 R0의 95% 이상,
- 두 조건이 8fold 중 6개 이상,
- bridge가 identity보다 AP `+0.01` 이상인 fold가 6개 이상,
- bridge의 추가 비용이 과거 archive 전체를 v1.2로 다시 인코딩하는 backfill 비용의 10% 이하.
- 새 query를 v1.2 encoder에 넣기 위한 정상적인 raw read는 비용표에 포함한다. `raw read 0`이라고
  쓰지 않으며, 절감 대상은 **과거 raw archive의 재독출·재임베딩**이다.

실패하면 “v1→v1.2는 exact-scene linear/spatial bridge로 task continuity를 지킬 수 없다”가
결과다. threshold를 사후 완화하지 않는다.

### 선행연구와 남는 빈칸

일반 vision에는 backward-compatible representation과 model stitching이 이미 있다. 따라서
“작은 projection을 학습했다”는 novelty가 아니다. 남는 빈칸은 다음 조합이다.

- dense spatial EO token,
- spatiotemporal·multispectral input contract,
- search와 segmentation head를 동시에 보존,
- 여러 downstream task가 공유하는 cache의 backfill 비용,
- contract mismatch에서 fail closed하는 판정.

관련 근거:

- [Towards Backward-Compatible Representation Learning (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html)
- [Cross-modal backward-compatible representation learning (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Towards_Cross-modal_Backward-compatible_Representation_Learning_for_Vision-Language_Models_ICCV_2025_paper.html)
- [Revisiting Model Stitching in the Foundation Model Era (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Mai_Revisiting_Model_Stitching_In_the_Foundation_Model_Era_CVPR_2026_paper.html)

## 3. B — Product-independent protocol: 같은 시험에 넣을 수 있는가

embedding product는 벡터 차원만 다르지 않는다. 밴드·정규화·시간 수용장·물리적 token support·
결측 표현·양자화·업데이트 주기·미세조정 recipe와 함께 움직인다. 입력·시간·공간·출력 계약을
하나의 manifest로 만들고, 서로 다른 질문을 분리한다.

### 공식 사양 재확인

| 제품 | 공식적으로 확인된 값 | 우리 실험에서의 의미 |
|---|---|---|
| OlmoEarth v1 Base | S2/S1/Landsat, patch size 선택{1..8}, Base 768-d; patch 4이면 128 px 입력에서 32×32 | 현재 P4 기준 제품 |
| OlmoEarth v1.2 Base | Base encoder 114M, v1과 같은 768-d downstream output 계약 | A의 같은-family release migration pair |
| Clay v1.5 | S2 10밴드, patch 8, 1024-d, 256 px 예시에서 32×32 | 128 px에서는 native 16×16; 이를 32×32로 단순 확대하면 공정하지 않음 |
| AlphaEarth V1 annual | 2017–2025 COG, 10 m, 64-d, unit vector, CC-BY 4.0 | 사건시점 encoder가 아니라 연간 embedding product; static task 전용 비교 |
| Prithvi-EO-2.0 | HLS 6밴드, 30 m, temporal ViT | S2→HLS mapping과 GSD shift가 함께 생기는 sensitivity |

공식 근거:

- [OlmoEarth model summary and releases](https://github.com/allenai/olmoearth_pretrain/blob/main/README.md)
- [rslearn OlmoEarth input/output contract](https://github.com/allenai/rslearn/blob/master/docs/foundation_models/OlmoEarth.md)
- [Clay v1.5 specification](https://clay-foundation.github.io/model/release-notes/specification.html)
- [Clay embedding tutorial](https://clay-foundation.github.io/model/tutorials/embeddings.html)
- [AlphaEarth annual catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
- [AlphaEarth GCS/quantization contract](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme)
- [Prithvi-EO-2.0 model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M)

### 현재 second-FM v0의 세 문제

1. **Clay grid**: v0의 “모든 제품을 32×32로 bilinear resample”은 Clay 16×16에 정보가 생긴 것처럼
   보이게 한다.
2. **AlphaEarth 시간**: 연간 product를 Sen12 사건 전후 task와 같은 gate에 넣으면 temporal
   receptive field가 다르다. label 시점 이후 관측이 embedding에 들어갈 수도 있다.
3. **일반성 문장**: Clay 한 모델 통과를 “제품 무관 원리”로 부를 수 없다.

### B의 두 트랙

#### B1. common-physical-cache track — readout-controlled compact-cache 비교

- 공통 support: 80 m, 16×16 token grid.
- OlmoEarth: 32×32를 고정 2×2 arithmetic mean으로 16×16.
- Clay: native 16×16 유지.
- 양쪽 channel은 source-only Incremental PCA로 256차원 고정. 같은 decoder와 정확히 같은
  trainable parameter를 쓴다. 모델별 PCA가 서로 다른 정보를 보존할 수 있으므로 이것을
  `representation family의 순수 인과효과`라고 부르지 않는다. 정확한 표현은
  **공통 물리 support·동일 readout 예산의 compact-cache 비교**다.
- PCA 없는 결과는 B2에서 따로 낸다. B1과 B2 중 좋은 것을 사후 headline으로 바꾸지 않는다.

#### B2. native-product track — 사용 가능한 제품 비교

- 각 제품의 native spatial grid·channel·normalization·temporal aggregation을 유지한다.
- decoder는 동일 family를 쓰되 parameter/FLOP/cache bytes를 모두 보고한다.
- 이 트랙은 “어떤 제품을 사야/써야 하는가”에 답하지만 architecture 인과를 주장하지 않는다.

### 모델별 허용 범위

- **Clay v1.5**: Sen12와 Solar에서 B1/B2. 4/12시점은 per-timestep encode 후 사전등록 arithmetic
  mean. OlmoEarth joint-temporal과 동일한 모델 구조라고 주장하지 않는다.
- **AlphaEarth**: Solar와 후속 static land-cover에만 B2. event segmentation headline에서 제외.
- **Prithvi**: HLS 6-band/30 m contract sensitivity. Clay 뒤에만 실행.

### B의 claim ladder

1. Clay가 두 task에서 통과: “두 spatial GeoFM family에서 관측된 cache-pathway 효과.”
2. AlphaEarth가 static task에서 통과: “ready-made annual embedding product에도 static mapping
   workflow가 확장됨.”
3. 두 비-OLMo family와 독립 Task-3까지 통과해야만 “product-agnostic decision protocol”을 주장.

“Clay 하나가 이겼으므로 모든 Earth embedding cache에 일반적”은 금지한다.

## 4. C — SafeCacheAction: support-only action policy

### 질문

> target query label을 보지 않고, support K장과 cache/contract/cost만으로 A0 그대로 쓰기,
> A1 head 적응, A3 encoder 적응+재임베딩, 또는 label 추가 요청을 고를 수 있는가?

이 질문은 과거 label-free router와 다르다. MS-99에서 **action crossover와 실제 harm**가 이미
관측됐고, support label은 사용자가 제공한 합법적인 정보다.

### 왜 단순히 “양성이 있으면 A1”이 아닌가

- 양성 0장은 A1 붕괴의 충분한 위험 신호다.
- 그러나 양성 존재는 A1 이득의 충분조건이 아니다.
- Solar에서 primary FP-matched IoU는 K=20 A1이 조금 오르지만, tie-correct AP는 A0가 더 높다.
  따라서 action은 task 이름이 아니라 **배포 utility**와 threshold 계약까지 받아야 한다.

### C0 — deterministic safety policy

입력:

- K, positive tile 수, positive pixel 비율, component 수와 tile 간 label 다양성
- A0의 support loss/AP/예측 양성률/entropy
- source train cache와 target support cache의 mean/covariance distance
- 10-step adaptation의 loss slope, gradient norm, parameter displacement
- action별 measured latency·raw bytes·cache invalidation

출력:

- `A0_REUSE`
- `A1_HEAD_ADAPT`
- `REQUEST_MORE_LABELS`
- A3 측정 뒤에만 `A3_REEMBED`

첫 정책은 learned MoE가 아니다. `positive_count=0 → A0 또는 REQUEST`, 그 밖에는 support tile
leave-one-out gain의 하한이 0보다 클 때만 A1을 허용하는 **one-sided safety rule**이다.

### C1 — A3 encoder ceiling

A3가 비어 있으면 “왜 head만 적응했나?”를 닫을 수 없다.

- primary strong baseline: OlmoEarth 공식 문서가 권장하는 `LayerDecayAdamW` 기반 encoder+decoder
  적응.
- sensitivity: attention q/v LoRA rank 8. LoRA는 새 기여가 아니라 비용 대조군.
- 개발: Sen12 exposed 2지역 + Solar 개발 2fold, K=20, 3 seed.
- A3는 raw image read, encoder GPU, query/full-cache 재추출 비용과 새 cache bytes를 모두 포함한다.
- A3가 A1보다 좋지 않으면 “K≤20에서 cache를 깨지 않을 근거”가 된다. 조금 좋으면 Pareto,
  크게 좋으면 paper는 cache validity boundary로 이동한다.

현재 rslearn 공식 문서에서 확인된 권장안은 `LayerDecayAdamW`다. 기존 문서의 “rslearn에 APLA가
이미 있다”는 표현은 로컬 checkout과 공식 문서에서 확인되지 않았으므로 구현 전 사실로 쓰지 않는다.

### C2 — selector 평가

기존 16 holdout 결과는 이미 보았으므로 method의 untouched test로 재사용하지 않는다.

- 개발/meta-train: Sen12 8지역 + Solar 8fold의 기존 support draw.
- 최종 first-look: 독립 Task-3 또는 Korea sealed target.
- baseline: always A0, always A1, positive-count rule, embedding-distance rule, oracle.
- primary: oracle 대비 action regret.
- safety: harmful-adaptation rate(A1이 A0보다 0.01 이상 나쁜데 A1 선택).
- coverage: `REQUEST_MORE_LABELS`가 아닌 자동 결정 비율.
- 배포 metric threshold는 source validation/support-only로 정한다. query empty-label로 만든
  FP-matched threshold는 selector의 배포 성능 주장에 쓰지 않는다.

승급 초안:

- best fixed action보다 regret 30% 이상 감소,
- harmful-adaptation rate 5% 이하,
- coverage 60% 이상,
- untouched task/target에서 같은 방향.

하나라도 실패하면 learned selector로 확장하지 않고 deterministic guardrail만 결과로 남긴다.

### C3 — label-free placebo adaptation은 마지막

Nepal의 2–3 placebo window로는 self-adaptation을 검증할 수 없다. 최소 20–30 historical window와
독립 사건 라벨이 생기기 전까지 label-free 통계는 다음 용도로만 쓴다.

- drift 감지
- cache validity 경보
- `ABSTAIN/REQUEST` 또는 `RE-EMBED` 요청

pseudo-label로 head/encoder를 업데이트해 “정확도가 개선됐다”고 주장하지 않는다. MS-99의
support collapse는 잘못된 자기학습이 오히려 모델을 망칠 수 있음을 이미 경고한다.

## 5. 하나의 paper spine으로 묶는 법

### 중심 명제

> **Cached Earth representations need a continuity layer, not a blanket fine-tuning recipe.**

### 기여 구조

1. **측정**: world shift에서 cache pathway가 raw adaptation보다 강하지만 A0/A1 승자는 support와
   utility에 따라 달라진다(MS-96/97/99).
2. **방법/시스템(A)**: 모델 release가 바뀌어도 unlabeled paired spatial bridge로 old task head와
   index를 유지할 수 있는지 측정하고, 실패하면 fail closed한다.
3. **일반성(B)**: native/common-physical 두 계약에서 Clay·AlphaEarth까지 확장한다.
4. **정책(C)**: support-only 정보로 REUSE/ADAPT/MIGRATE/RE-EMBED/REQUEST를 고르고 regret와 실제
   I/O/GPU/cache invalidation 비용을 함께 낸다.

### 너무 넓어지는 것을 막는 선

- 물리 시뮬레이션, live disaster dashboard, prediction fusion은 본 논문에서 제외.
- MoE는 action complementarity와 untouched selector 성공 뒤에만 후속 연구.
- A/B/C 중 A가 실패해도 B+C empirical paper는 남지만, CVPR method headline은 약해진다.
- B가 실패하면 OLMoEarth-specific systems paper로 좁힌다.
- C가 실패하면 selector를 버리고 **contract-aware fail-closed migration benchmark**로 제출한다.

## 6. 실행 순서

### P0 — 기존 결과의 증거 구멍부터 닫기 (CPU)

1. Task-2 random support의 개별 support ID·양성 수·manifest SHA를 로컬 mirror.
2. Task-2 fold를 WGS84로 변환해 cross-CRS overlap/nearest-neighbor distance 감사.
3. Solar exact-query A4w0를 계산해 `Δcache=A1−A0`, `Δraw=A4w−A4w0` 분리.
4. 서버의 Task-2 source 48-run report/snapshot을 로컬 봉인.

### P1 — A 최소 실행

1. Solar 20-sample v1.2 smoke: 12-band·shape·finite·determinism·v1 exact-scene pairing.
2. 두 exposed fold의 v1.2 cache와 identity/Procrustes/ridge bridge.
3. old-head retention이 screen을 통과할 때만 8fold spatial bridge와 비용표.
4. retrieval R@1이 아니라 downstream fixed-threshold/AP를 primary로 판정.

### P2 — B Clay

0. **현재 v0 격리**: commit `e56561d..a346eab`의 실행은 Clay native 16×16을 32×32로 bilinear
   확대한다. 이 산출물은 `interpolated deployment-adapter baseline`으로만 보존하고 B1/B2의
   confirmatory 결과로 승격하지 않는다. 또한 Clay few-shot의 FP-matched IoU는 Clay A0가 만든
   FP budget을 사용하므로, 기존 OlmoEarth report의 raw A4와 서로 다른 budget으로 직접 비교하지
   않는다. tie-correct AP만 threshold-free 탐색 비교가 가능하다.
1. v0 서버 체인이 끝날 때까지 실행 경로를 push하지 않는다. 완료 뒤 checkpoint/code/metadata SHA,
   10-band order, 128 px→16×16×1024 실물 smoke를 봉인한다.
2. B1 common 80 m/256-d cache와 B2 native cache를 결과 보기 전 동시에 봉인한다.
3. B1/B2는 공통 source-validation threshold 또는 동일한 사전 FP budget으로 raw baseline을 같은
   report 안에서 재평가한다. historical report 숫자를 붙여 한 표로 만들지 않는다.
4. 개발 2지역/2fold 뒤 사전 gate 통과 시만 8fold·두 task.
5. AlphaEarth는 Solar static B2를 별도 실행; event task와 한 표에 강제 결합하지 않는다.

### P3 — C

1. A3 LayerDecayAdamW ceiling: exposed 4 units, K=20.
2. 기존 두 task에서 support-only feature와 action outcome table 생성.
3. deterministic safety rule 동결.
4. Task-3/Korea first-look에서 regret·harm·coverage 1회 판정.

## 7. 지금 당장 무엇을 하지 않는가

- Clay checkpoint와 v0 cache는 이미 만들어졌다. v0 결과는 위 격리 규칙을 적용하고, v1 B1/B2를
  열기 전 5-sample native-contract smoke와 checkpoint/code/metadata seal을 새로 만든다.
- AlphaEarth를 “실시간” 또는 “사건 전후” encoder라고 부르지 않는다.
- v1.2가 최신이라는 이유만으로 성능 우월을 가정하지 않는다.
- query label로 정한 FP threshold를 운영 selector 입력으로 쓰지 않는다.
- 세 방향을 동시에 GPU queue에 올리지 않는다. P0 → A screen → B Clay → C A3 순이다.

## 8. 목표별 가치

| 목표 | 가장 강한 산출물 |
|---|---|
| Ai2 취업 | OlmoEarth v1→v1.2 실제 migration harness, fail-closed contract manifest, 공식 권장 FT와 cache 비용 대조 |
| 박사/CVPR | WorldShift×ModelShift 아래 task continuity와 action regret를 함께 정의한 method/system paper |
| 사업 | 모델 공급자가 바뀌어도 기존 index/head를 유지할 수 있는지 사전에 계산하는 migration readiness report |

## 9. 최종 냉정 판정

A–C 확장은 유의미하다. 다만 **B부터 무작정 여러 모델을 돌리는 것보다 A를 먼저 닫는 편이
노벨티와 Ai2 연결성이 모두 높다.** 이미 M1이라는 failure와 MS-99라는 action crossover가 있어,
새 실험은 새 이야기를 발명하는 것이 아니라 두 기존 결과 사이의 비어 있는 인과 고리를 채운다.

현재 가장 좋은 순서는 다음이다.

> **P0 증거 복구 → A release migration → B Clay/AlphaEarth 외적 타당성 → C safe action →
> Korea/Task-3 first-look**

## 10. 2026-09-04 실행 중 발견한 B-v0 경계

- 서버의 `clay_cache`는 6,834/6,834 tile, 실패 0으로 완성됐지만 저장 shape `1024×32×32`는
  native `1024×16×16`을 bilinear 확대해 만든 것이다. `all_gates_pass=true`는 파일 완결성만
  보증하며 비교 공정성을 보증하지 않는다.
- source decoder chain과 few-shot wait chain이 실행 중이다. 확증 실행 중 코드 push 금지 규칙에
  따라 서버 코드는 건드리지 않았다.
- 로컬 실행기 감사에서 `clay_chain.sh`와 `clay_fewshot_chain.sh`가 실패 후에도 DONE marker를
  남길 수 있음을 발견했다. 로컬에는 exit-code/timeout guard를 추가했지만 현재 서버 실행에는
  push하지 않았다. 따라서 v0 완료 여부는 marker 하나가 아니라 24개 source run·각 report run 수·
  snapshot을 사후 검사해 판정한다.
- v0 결과의 허용 용도는 실행 가능성·메모리·대략적 성능 screen이다. 금지 용도는
  `common-resolution comparison`, `native-product comparison`, `Clay가 raw보다 우월`,
  `product-agnostic` headline이다.
- 특히 현재 few-shot chain은 Clay A0에서 구한 empty-FP budget을 쓰고, 비교하려는 기존 raw
  A4 report는 OlmoEarth A0의 budget을 썼다. 동일 작동점이 아니므로 FP-matched IoU의 교차-report
  비교는 무효다. B-v1에서는 raw/Olmo/Clay를 동일 report·threshold 계약으로 다시 채점한다.
