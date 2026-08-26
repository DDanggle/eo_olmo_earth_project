# 논문 주장 확장 계약 — M37 이후의 EarthRoute

조사·동결일: 2026-08-26

역할: 아이디어 문서가 아니라 **현재 증거가 허용하는 주장, 선행연구가 이미 점유한 영역, 다음 실험의
통과·중단 조건**을 한곳에 고정한다. 최신 결과는 `MEASURED_FINDINGS.md` M37을 따른다.

## 0. 한 문장 판정

현재 결과만으로는 CVPR method paper가 아니다. 그러나 M37을 단순 실패로 버릴 이유도 없다.
M37은 다음의 더 중요한 가능성을 드러냈다.

> **EO embedding의 seam·smoothness·drift 같은 표현 진단이 좋아져도 downstream utility는 악화될 수
> 있다. 따라서 여러 task가 공유하는 embedding product를 운영하려면, 관측 가능한 contract와
> representation 진단으로 각 maintenance action의 task별 이득을 예측하고, 공유 비용까지 포함해
> action을 선택해야 한다.**

이 문장의 앞 절은 아직 **한 지역·한 backbone·한 task의 반례**이고, 뒤 절은 **미검증 method
hypothesis**다. 둘을 섞어 쓰지 않는다.

논문의 작업명 `EarthRoute`는 유지하되, 방법의 기술적 핵심은 아래처럼 부른다.

> **Contract-Conditional Action Utility Estimation for Shared Earth Embeddings**

`router`라는 일반 명사는 novelty가 아니다. novelty 후보는 `EO contract 아래 task별 action utility를
target label 없이 추정`하는 것과 `한 번의 representation 비용을 여러 task에 공동 배분`하는 것이다.

---

## 1. M37이 실제로 허용하는 주장

### 1.1 실측 사실

동일 개발 지역 Chimanimani, 동일 OLMoEarth v1 last-layer embedding, 동일 test 1,133 tile에서
serving context와 decoder를 2×2로 교차했다.

| cache context | decoder | IoU | AUPRC | positive-patch macro IoU | 해석 |
|---|---|---:|---:|---:|---|
| tiled 4×64 | small | 0.130582 | 0.151348 | 0.159966 | 원래 frozen recipe |
| tiled 4×64 | large | **0.177727** | **0.213574** | 0.139172 | pixel-micro/AP 회복, 양성 macro는 악화 |
| full 1×128 | small | 0.116565 | 0.132972 | 0.141897 | wider context가 악화 |
| full 1×128 | large | 0.081419 | 0.080557 | 0.066738 | 가장 큰 악화 |

- context main effect는 평균 `-0.055162 IoU`이고 네 spatial block scale 모두 CI가 0 아래였다.
- decoder main effect는 `+0.005999`로 지지되지 않았다.
- interaction은 `-0.082291`이며 모든 block scale에서 음수였다.
- tiled-large는 P2 UNet3D의 pixel-micro IoU/AUPRC를 넘지만 positive-patch macro와 LD-IoU는 못 넘고,
  cache extraction까지 포함하면 accuracy–cost Pareto도 아니다.
- M34에서 full context는 인공 seam을 줄였지만 M37에서 utility는 떨어졌다.

### 1.2 지금 써도 되는 최소 claim

> 이 개발 fold에서 embedding의 국소 연속성 개선은 segmentation utility 개선을 보장하지 않았고,
> decoder 효과의 방향은 serving context에 따라 반전됐다.

이것은 `proxy–utility decoupling`의 **존재 반례**다. `full context가 나쁘다`, `tiled cache가 좋다`,
`OLMoEarth가 불안정하다`로 일반화하지 않는다.

### 1.3 아직 쓸 수 없는 claim

- seam 감소가 성능 하락의 원인이다.
- context가 일반적으로 해롭다.
- large decoder가 frozen embedding을 구한다.
- OLMoEarth가 raw U-Net보다 우월하거나 열등하다.
- task마다 cache 위험이 다르다.
- target label 없이 위험을 예측할 수 있다.
- router가 비용–정확도 Pareto를 개선한다.
- 한국 결과가 네팔·스위스로 transfer된다.

M37은 `task-specific risk`나 `routing gain`을 측정하지 않았다. 서로 다른 metric의 승자 교차도
routing 근거가 아니다.

---

## 2. 2026년 최신 선행과 충돌한 뒤 남는 빈칸

### 2.1 선행 충돌 표

| 선행 | 이미 점유한 질문 | 우리에게 남는 정확한 빈칸 | 필수 비교 |
|---|---|---|---|
| [EarthShift](https://earthshift.github.io/) (2026 preprint) | 8 GeoFM·11 task·5 현실 shift에서 ID/OOD 성능을 라벨로 측정; 평균 약 20% 저하 | **라벨 없는 새 batch에서** task×action gain을 미리 예측하고 action을 선택 | EarthShift protocol과 paired task를 그대로 재사용 |
| [GEO-Bench-2](https://arxiv.org/abs/2511.15658) (2025/2026 preprint) | 19 dataset의 capability 평가; task·modality·제약마다 최고 모델이 다름 | 고정 benchmark ranking이 아니라 **배포 중 cache state/action 선택** | capability group, strong supervised baseline |
| [PANGAEA](https://arxiv.org/abs/2412.04204) | GeoFM downstream 평가와 강한 supervised baseline | representation product의 lifecycle decision | PANGAEA recipe·U-Net/U-TAE |
| [How to Embed Matters](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Gilch_How_to_Embed_Matters_Evaluation_of_EO_Embedding_Design_Choices_CVPRW_2026_paper.html) | embedding 생성·aggregation recipe가 downstream 결과를 바꿈 | recipe 차이를 **사전 task risk와 action value**로 연결 | pooling/context/normalization ablation |
| [Earth Embeddings as Products](https://arxiv.org/abs/2601.13134) | embedding product taxonomy·표준 loader·상호운용성 | product를 읽는 법이 아니라 **언제 어느 product state를 열고 갱신할지** | product metadata contract |
| [TESSERA v2](https://arxiv.org/abs/2607.03949) | multi-task storage frontier와 seam/artifact 분석 | storage·seam proxy가 task utility를 대리할 수 있는지, 실패 시 action 선택 | dimension/storage/seam baseline |
| [RALF / feature-store freshness](https://escholarship.org/uc/item/5xk0f4z9) | downstream error feedback으로 stale feature의 regret를 추정해 update를 scheduling | 즉시 error feedback이 없는 EO에서 contract/action gain을 target-unlabeled로 예측; sensor/release/serving action과 multi-task 평가 | delayed-feedback RALF-style scheduler |
| [CrossEarth-Gate](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_CrossEarth-Gate_Fisher-Guided_Adaptive_Tuning_Engine_for_Efficient_Adaptation_of_Cross-Domain_CVPR_2026_paper.html) (CVPR 2026) | spatial/semantic/frequency PEFT module을 Fisher 정보로 선택; 18 cross-domain segmentation benchmark | tuning module 선택을 넘어 `reuse/repair/re-embed/raw`의 **cache lifecycle** 선택 | PEFT action ceiling·Fisher/gradient feature |
| [DARN](https://openaccess.thecvf.com/content/CVPR2026F/html/Yadav_DARN_Dynamic_Adaptive_Regularization_Networks_for_Efficient_and_Robust_Foundation_CVPRF_2026_paper.html) (CVPR Findings 2026) | sample difficulty에 따라 decoder dropout·capacity를 동적 조절 | decoder capacity gate가 아니라 **representation state와 maintenance cost** 결정 | DARN 또는 difficulty-only baseline |
| [DEFLECT](https://openaccess.thecvf.com/content/ICCV2025/html/Thoreau_Parameter-Efficient_Adaptation_of_Geospatial_Foundation_Models_through_Embedding_Deflection_ICCV_2025_paper.html) (ICCV 2025) | embedding deflection 기반 PEFT | PEFT는 후보 action이지 headline이 아님 | DEFLECT/LoRA/adapter 중 재현 가능한 하나 |
| [AnySat](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html)·[THOR](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Forgaard_THOR_A_Versatile_Foundation_Model_for_Earth_Observation_Climate_and_CVPRW_2026_paper.html) | 입력 sensor·resolution·scale과 compute를 한 모델에서 유연하게 처리 | 유연한 backbone에서도 **task별 refresh 필요성이 남는지** | flexible-model negative control |
| [MMEarth-Bench](https://mmearth-bench.com/) | 다섯 환경 task, geographic OOD, multimodal test-time training | adaptation 자체가 아니라 task별 action value와 joint cache cost | TTT-MMR를 adaptation ceiling으로 사용 |
| [ChronoEarth-492K](https://uiuctml.github.io/ChronoEarth492K/) (2026 preprint) | static/short/long horizon과 spatial-temporal OOD, cross-satellite transfer | temporal support를 cache action으로 바꿀 때 task별 utility 예측 | temporal-history/staleness stress 후보 |

### 2.2 label-free risk estimation과의 충돌

`라벨 없이 성능을 추정`하는 분야도 이미 존재한다.

| 선행 | 이미 하는 일 | EarthRoute가 추가로 증명해야 하는 것 |
|---|---|---|
| [GdScore](https://openreview.net/forum?id=FIWHRSuoos) (TMLR 2025) | pseudo-label cross-entropy의 classification-layer gradient norm으로 target accuracy 추정 | dense EO metric, 희소 event, cache-only/gradient-free 경우, action **gain** 추정 |
| [ODD](https://proceedings.mlr.press/v286/mishra25a.html) (UAI 2025) | source–target overlap을 domain classifier로 추정해 target error bound 개선 | spatial/event group과 task×action matrix에서의 비교 |
| [IUPM](https://proceedings.mlr.press/v258/koebler25a.html) (AISTATS 2025) | gradual shift를 optimal transport로 추적하고 불확실할 때 active label 개입 | EO temporal/contract shift와 label-request action |
| [Model Assessment under Temporal Shift](https://proceedings.mlr.press/v235/han24b.html) (ICML 2024) | adaptive rolling window로 temporal shift 아래 model loss와 pairwise model 차이를 추정·선택 | target-unlabeled/feedback-delay 조건과 EO action cost를 분리 |
| [Agreement-on-the-Line TTA](https://openreview.net/forum?id=iEFMwP5wng) (2024) | 모델 agreement로 label-free accuracy·TTA hyperparameter·calibration 선택 | 모델군 agreement가 희소 segmentation action ordering에도 유지되는지 |
| [Performance Prediction Under Dataset Shift](https://arxiv.org/abs/2206.10697) | 단순 shift metric보다 학습된 error predictor가 unseen domain에 낫다는 경험 결과 | EO-native shift family와 leave-one-shift-family-out 검증 |
| [Adapting Prediction Sets without Labels](https://proceedings.mlr.press/v286/kasa25a.html) (UAI 2025) | unlabeled target에서 conformal prediction set을 조정 | 위험 추정 불확실 시 `abstain/request-label` safety action |
| [Testable Learning with Distribution Shift](https://proceedings.mlr.press/v247/klivans24a.html) (COLT 2024) | 조건을 만족하는 shift에서만 성능을 certify하는 testable framework | 보편 label-free 보장이 아니라 **어느 EO shift family에서 유효한지** 명시 |

따라서 `uncertainty`, `embedding drift`, `seam score` 하나와 threshold를 붙인 것은 방법이 아니다.
GdScore·ODD·agreement·OT monitoring을 이겨야 한다. 더 중요한 것은 **현재 위험만 맞히는 것과 각
action을 취했을 때의 개선량을 맞히는 것은 다른 문제**라는 점이다.

delayed target label이 생기는 운영에서는 RALF-style feedback scheduler가 강한 비교군이다. EarthRoute가
주장할 구간은 label이 오기 전의 cold-start/action selection이며, label이 도착한 뒤에도 계속 우월하다고
가정하지 않는다.

### 2.3 Ai2 방향과의 실제 접점

Ai2의 [OlmoEarth embedding export](https://allenai.org/blog/olmoearth-embeddings)는 AOI, 1–12개월,
encoder variant, 10–80 m, S1/S2 입력을 선택해 int8 COG를 만든다. 같은 공식 글도 어려운 task에서는
SFT가 frozen embedding보다 높은 성능을 낼 수 있고 input imagery 품질을 검증해야 한다고 명시한다.
[2026년 7월 platform 글](https://allenai.org/blog/olmoearth-infrastructure)은 대륙 규모 inference,
모니터링, 시기·장소별 비용 효율을 운영 과제로 둔다.

즉 이 연구의 Ai2-facing 가치는 “OLMoEarth가 실시간이다”가 아니다. 모델 그 자체의 실시간성을
주장하지 않고 아래 운영 질문에 답하는 것이다.

> 같은 AOI에서 time span·sensor·resolution·model release·serving crop이 바뀔 때, 여러 downstream
> application이 소비하는 embedding COG를 그대로 둘지 다시 만들지 어떻게 검증하는가?

---

## 3. 제안 문제의 정확한 정의

### 3.1 단위

- `g`: 한 번에 cache action을 적용할 spatial block × observation window. pixel이 아니다.
- `t ∈ T`: land-cover, deforestation, landslide 등 downstream task.
- `s=(m,r,c,o)`: backbone family `m`, release `r`, serving contract `c`, observation state `o`로 정의한
  representation state.
- `a ∈ A`: representation/task maintenance action.
- `z(g,t,s,a)`: target label 없이 계산 가능한 contract·quality·representation·task 진단.
- `M_t(g,a)`: 사전 등록된 task primary metric. 연구 평가에서만 target label로 측정한다.
- `C(a)`: GPU time, I/O bytes, storage, latency, human-label cost의 벡터.

### 3.2 action set — 첫 논문은 넓히지 않는다

| action | 의미 | 공유 비용 |
|---|---|---|
| `A0 reuse` | 기존 cache와 기존 task head 유지 | 0에 가까움 |
| `A1 repair` | cache-compatible calibration/FoldRefresh/compatibility adapter | representation 재추출 없음; task별 비용 |
| `A2 re-embed` | 현재 observation·동결된 serving contract로 embedding 재생성 | **여러 task가 한 번을 공유** |
| `A3 task-raw` | task-specific raw imagery model 사용 | task마다 별도 비용 |
| `A4 abstain/audit` | 예측 구간이 불충분하면 자동결정하지 않고 소량 label/검수 요청 | 사람 비용; safety extension |

PEFT는 `A2`를 만드는 한 방식이며 별도 novelty가 아니다. 첫 제출에서는 LoRA/DEFLECT/CrossEarth-Gate
중 재현 가능한 하나만 둔다. action 수를 늘려 oracle headroom을 인위적으로 키우지 않는다.

### 3.3 supervision 계약

“label-free”의 뜻을 정확히 고정한다.

- source/development region의 라벨은 head, action candidate, risk estimator 학습에 사용한다.
- 새 target region/window의 라벨은 **action 선택 시점에는 보지 않는다**.
- target 라벨은 논문에서 실제 action utility와 regret를 사후 평가할 때만 공개한다.
- `A4` few-label extension을 열면 사용 label 수와 시점을 별도 예산으로 보고한다.

따라서 `no labels anywhere`가 아니라
`source-labeled meta-training + target-unlabeled action selection`이다.

### 3.4 action utility와 공동 비용

task `t`에서 reuse 대비 action gain은 다음으로 정의한다.

```text
Δ(g,t,a) = M_t(g,a) - M_t(g,A0)
```

risk estimator는 `z`로 `Δ`의 평균과 구간을 예측한다. 여러 task가 동일한 새 representation state를
사용하면 extraction 비용은 한 번만 낸다.

```text
joint utility = Σ_t predicted_metric(g,t,a_t)
                - λ_E · opened_representation_costs
                - Σ_t λ_t · task_specific_cost(a_t)
```

이 최적화는 facility-location/knapsack의 변형일 수 있으나 optimizer 자체를 novelty로 주장하지
않는다. 기여는 **label-free EO action-value matrix가 실제 oracle ordering을 복원하는지**다.

task metric 단위가 다르므로 논문 전체 집계는 held-out task의 best/worst action으로 정규화한 regret와
각 task 원 metric을 함께 보고한다. 원 metric을 하나의 임의 가중 평균으로 숨기지 않는다.

benchmark에서는 모든 후보 action을 같은 held-out unit에 실제 실행해 `M_t(g,a)`를 관측한다. 따라서
관측되지 않은 처치효과를 복원하는 causal claim은 하지 않는다. 향후 운영 로그처럼 선택된 action의
결과만 남는 setting으로 확장할 때에만 propensity/off-policy evaluation을 별도 문제로 연다.

### 3.5 예측 feature 계층

| 계층 | 예시 | 역할 |
|---|---|---|
| contract | model/release/build, band set, sensor, GSD, crop/context, temporal support | 어떤 배관이 바뀌었는지 |
| observation quality | valid coverage, cloud/SCL, missing dates, mosaic count, incidence/platform | 입력 증거가 충분한지 |
| representation | norm, anisotropy, neighbor stability, release drift, tiled/full discrepancy, quantization saturation | embedding state가 얼마나 달라졌는지 |
| task/head | entropy, margin, gradient norm, ensemble disagreement, object-scale prior, temporal horizon | 같은 shift가 task에 미치는 영향 |
| overlap | source–target domain classifier, OT distance, support test | risk predictor가 외삽 중인지 |

M34 seam score는 이 표의 한 feature일 뿐이다. M37은 seam/smoothness 하나만으로 utility를 추정하면
실패할 수 있음을 보여주는 첫 반례다.

---

## 4. 반증 가능한 claim ladder

| 단계 | 제출 시 주장 | 현재 증거 | 통과 조건 | 실패 시 정직한 산출물 |
|---|---|---|---|---|
| **C0 존재** | proxy 개선과 task utility가 분리될 수 있다 | M34→M37 한 셀에서 존재 | 2 backbone×3 task×3 shift 중 사전 정의 셀의 ≥1/3에서 proxy와 utility 순위 불일치; spatial/event CI | 한 OLMo serving-contract case study |
| **C1 이질성** | 같은 representation action의 gain 순위가 task에 따라 다르다 | 0 | 같은 spatial unit·동일 primary utility에서 task×action interaction과 양의 oracle joint-routing gain | router 중단; robustness/contract benchmark |
| **C2 예측** | target label 없이 task별 action gain을 예측할 수 있다 | 0 | leave-one-region 및 leave-one-shift-family-out에서 contract-only·entropy·GdScore·ODD·agreement보다 낮은 regret/오차 | label-free method 중단; diagnostic benchmark |
| **C3 결정** | 예측 action으로 정확도–비용 Pareto를 개선한다 | 0 | 고정 budget에서 fixed refresh와 best single action보다 metric 향상, 또는 동일 metric에서 compute/storage 감소; CI 포함 | monitoring/analysis paper |
| **C4 안전성** | predictor support 밖에서는 abstain/audit가 과신을 줄인다 | 0 | selective risk–coverage와 audit-label efficiency가 uncertainty-only보다 우수 | 보편 transfer claim 철회 |
| **C5 transfer** | policy ordering이 model family와 외부 지역에 유지된다 | 0 | second family + untouched country에서 top-action/regret ordering 유지 | 한국/특정 backbone으로 범위 축소 |

### 핵심 kill rule

`C1`이 실패하면 EarthRoute method를 중단한다. task들이 모든 shift에서 같은 action 순위를 가지면
shared cache는 task-conditioned router가 아니라 하나의 global refresh schedule로 충분하다.

`C2`가 실패해도 threshold를 사후 조정하지 않는다. 이 경우 M37과 대규모 실험은
`When representation diagnostics fail to predict downstream utility` 분석 논문으로 전환한다.

---

## 5. 최소 CVPR 실험 설계

### 5.1 3개 축을 분리한다

| 축 | 개발·반증 데이터 | 논문에서의 역할 |
|---|---|---|
| public shift | EarthShift의 location/temporal/sensor/data/scale 중 실제 재현 가능한 3개 | 공개 robustness와 risk-estimation baseline |
| shared multi-task | coverage-valid AI-Hub v2의 동일 cube 위 land-cover/deforestation/landslide | 동일 cache의 task×action heterogeneity와 joint cost |
| external transfer | **Nepal 또는 Switzerland 한 곳만** untouched | 한국에 맞춘 policy의 외부 검증 |

EarthShift의 task는 서로 다른 dataset에 있어 shared-cache 비용을 직접 증명하지 못한다. 반대로
AI-Hub는 세 task가 같은 cube를 쓰지만 원본 재배포가 제한된다. 두 축이 서로의 약점을 보완한다.
공개 benchmark를 빼고 AI-Hub만 쓰면 재현성이 약하고, AI-Hub를 빼면 multi-task shared-cache 주장이
약하다.

### 5.2 representation family

최소 두 family가 필요하다.

1. **OLMoEarth**: custom sub-annual embedding, exact-time·sensor·context·release action을 통제 가능.
2. **두 번째 family**:
   - 실행 가능한 우선순위: Prithvi/TerraMind/AnySat 중 입력 계약과 license가 맞는 하나.
   - embedding-product 대조: TESSERA v2 또는 AlphaEarth annual product.

annual product는 월별 산사태를 못 보는 열등 모델이 아니라 temporal support가 다른 제품이다. 같은
입력을 넣었다고 가장하면 안 되고, contract 차이 자체를 feature로 넣는다.

### 5.3 shift family

최소 세 가지를 사전에 고정한다.

1. **observation staleness**: 이전 window vs current window.
2. **input/sensor contract**: band missing, sensor/source, GSD/resampling 중 실제 shift 하나.
3. **generator/serving contract**: model release 또는 tiled/full crop. M37 셀을 포함하되 여기에 맞춰
   method를 튜닝하지 않는다.

synthetic corruption만으로 끝내지 않는다. EarthShift의 real paired shift와 AI-Hub repeated date가
최소 한 축씩 있어야 한다.

### 5.4 task와 primary metric

| task | primary metric | 보조 지표 | 독립 단위 |
|---|---|---|---|
| land-cover | class-macro IoU | per-class recall, ECE | spatial block/cluster |
| deforestation | event/positive-patch macro IoU | AUPRC, boundary/LD IoU | forest-loss event/cluster |
| landslide | event/positive-patch macro IoU | AUPRC, LD-IoU | landslide event/cluster |

pixel-micro IoU는 함께 보고하되 primary routing utility로 사용하지 않는다. 희소 task에서는 빈 타일이
많아 micro metric이 action gain을 왜곡할 수 있다.

### 5.5 비교군

**risk estimator**

1. always reuse / always re-embed / fixed schedule.
2. contract-only decision tree.
3. mean confidence, entropy, ECE proxy.
4. embedding drift만: MMD/Fréchet/neighbor stability/seam.
5. GdScore 또는 dense-task gradient analogue.
6. ODD/domain-overlap.
7. model agreement.
8. learned error predictor without task descriptor.
9. oracle action.

**representation/action**

1. raw UNet3D/U-TAE.
2. frozen small/large head with exact-time parity.
3. cache-compatible repair.
4. one PEFT action.
5. flexible-input backbone negative control(가능한 경우 AnySat/THOR).

### 5.6 split와 leakage 차단

- risk predictor train/val/test를 **tile이 아니라 region/event/shift family**로 나눈다.
- `leave-one-region-out`과 `leave-one-shift-family-out`을 둘 다 수행한다.
- target test label은 action 후보·risk predictor·threshold 선택에 사용하지 않는다.
- action 후보들은 동일 source label budget과 동일 observation cutoff를 쓴다.
- threshold는 fixed 0.5와 source-val selected를 분리한다.
- 적어도 3 optimization seed; spatial/event block interval은 seed uncertainty와 별도 보고한다.
- 여러 action을 사후 추가하면 oracle gain이 자동 증가하므로 action set hash를 test 전에 봉인한다.

### 5.7 평가값

**예측 품질**

- action gain MAE/RMSE와 Spearman rank correlation.
- top-action accuracy만 쓰지 않고 top-2 margin과 calibration interval coverage.
- worst-region·worst-task error.
- support test 통과/실패별 error.

**결정 품질**

- oracle-normalized regret.
- fixed compute/storage budget별 task metric.
- 동일 metric에서 GPU seconds, bytes read/written, output storage, latency.
- task 수 `K=1,2,3`에 따른 shared extraction amortization. M30 실측 손익분기 `K≥2`를 재검증한다.
- abstention/audit를 열면 risk–coverage와 label-request 수.

---

## 6. 한국·네팔·스위스의 역할

### 6.1 한국 — method의 shared-task 증명

AI-Hub v2는 세 task가 같은 Sentinel-2 cube를 소비한다는 점에서 핵심이다. 하지만 원본 외부 재배포가
제한되므로 공개 method 개발 데이터 전체를 대신할 수 없다. `AIHUB_CUBE_V2_CONTRACT.md`의 12-band,
same-date/platform, common validity ≥99.9%, selection-bias gate를 통과한 cube만 사용한다.

제주·지리산은 train geography가 아니라 한국 내부 operational holdout 또는 qualitative deployment
site로 둔다. label ontology와 observation date가 맞지 않으면 지도 사례로만 사용하고 metric에 넣지
않는다.

### 6.2 Nepal — semantic transfer가 강하지만 label 독립성을 감사한다

[ICIMOD Koshi Basin 2024 landslide inventory](https://rds.icimod.org/metadata/af73da0a-885b-459d-95ba-2ea0662a7e7c)는
2026년 공개, Sentinel-2 기반, 수동 보정·QC, CC BY 4.0이라 landslide transfer 후보로 매우 좋다.
[2015 earthquake inventory](https://rds.icimod.org/metadata/5e19b3a6-910e-4537-a49e-bce2e5da2e78)는
pre/co-seismic polygon과 CC BY 4.0을 제공한다. ICIMOD는 2026년 operational NepalLandslide NetCDF도
OpenDAP/WMS로 제공한다.

그러나 Koshi inventory는 U-Net 자동 탐지 후 수동 보정된 라벨이다. 동일 Sentinel-2 appearance와
segmentation-model bias를 완전히 독립 ground truth로 취급하면 안 된다. 최종 transfer에서는
수동보정 provenance strata, 2015 visually digitized inventory, event/date holdout을 분리한다.

### 6.3 Switzerland — contract·operations transfer가 가장 깨끗하다

[swissEO S2-SR](https://www.swisstopo.admin.ch/en/satelliteimage-swisseo-s2-sr)는 10 m COG, STAC,
cloud probability, mask, registration metadata와 2–3회/주 관측을 제공한다. swisstopo 기본 geodata는
open government data이고 출처표시 조건이 명확하다. [SLF data service](https://www.slf.ch/en/services-and-products/slf-data-service/)
는 live/historical station·model·avalanche bulletin을 CC BY 4.0으로 제공한다.

중요한 제한: swissEO S2-SR 공개 product의 reflectance 묶음은 10 m RGBN 4밴드와 20 m red-edge/NIR/
SWIR 3밴드다. OLMoEarth 12-band 입력과 동등하지 않다. full-band parity가 필요하면 별도 Copernicus
원 장면 경로를 써야 하고, swissEO를 그대로 넣는다면 그것은 missing-band/input-contract shift다.

따라서 Switzerland는 ingestion·registration·live metadata·license가 명확한 systems transfer에는
강하다. 다만 한국의 landslide/deforestation과 동일 task를 바로 만들기 어렵고 avalanche로 바꾸면
task semantics가 변한다.

### 6.4 첫 논문 선택

- **task-semantic transfer를 우선하면 Nepal**: landslide 동일 task, 기후·지형 shift가 큼.
- **재현 가능한 operational contract를 우선하면 Switzerland**: STAC·COG·mask·registration·license가
  명확함.

둘 다 넣어 breadth를 부풀리지 않는다. 현재 main claim에는 **Nepal 한 곳을 untouched task transfer**로
추천하고, Switzerland는 후속 operational paper/사업 demo로 남긴다. Nepal label-independence audit가
실패하면 순서를 바꾼다.

---

## 7. 즉시 실행할 P0 → P3

### P0 — RQ1을 공정하게 닫기

1. exact-time 정보 계약을 P2/P4 사이에 맞춘다.
2. tiled-large와 P2를 공통 seed·공통 threshold-selection으로 반복한다.
3. positive-event macro와 총비용을 포함해 한 recipe를 동결한다.
4. 여전히 불안정하면 multi-level 또는 PEFT **한 축만** 연다.
5. recipe·metric·중단 규칙을 봉인한 뒤 Sen12 미열람 지역에서 viability를 확인한다.

이 단계는 method 결과가 아니라 action library를 공정하게 만드는 단계다.

### P1 — oracle heterogeneity를 먼저 잰다

1. AI-Hub v2 40-pair pilot health/selection gate.
2. 세 task×사전 동결 shift×`A0–A3`의 실제 utility matrix 생성.
3. task×action interaction, action rank reversal, joint oracle gain 측정.

**P1에서 heterogeneity가 없으면 P2를 열지 않는다.**

### P2 — risk estimation baseline tournament

1. contract-only, entropy, drift, GdScore, ODD, agreement를 같은 split에서 실행.
2. 가장 단순한 선형/GBDT error predictor부터 시작한다.
3. leave-one-region·leave-one-shift-family-out 성능을 본 뒤에만 neural router를 연다.
4. uncertainty가 큰 곳은 `A4 abstain`으로 보낸다.

단순 GBDT가 충분하면 복잡한 신경망을 만들지 않는다. 방법 기여는 architecture 복잡도가 아니라
EO action-value formulation과 generalization evidence일 수 있다.

### P3 — joint policy와 외부 transfer

1. task 수 `K`별 shared extraction cost를 포함한 joint selection.
2. fixed schedule/best single action 대비 Pareto와 regret.
3. second representation family.
4. 모든 recipe 동결 후 Nepal을 한 번 공개.

---

## 8. CVPR 판정과 대체 논문

### CVPR main-track이 가능한 경우

아래 다섯 문장이 모두 데이터로 닫혀야 한다.

1. **Proxy failure:** representation-health proxy와 downstream utility가 여러 task/shift에서 분리된다.
2. **Need:** 같은 shared representation action의 이득 순위가 task마다 달라 joint oracle headroom이 있다.
3. **Method:** target-unlabeled action-value estimator가 강한 performance-estimation baseline을 이긴다.
4. **Decision:** 예측 policy가 fixed refresh/best-single-action보다 accuracy–cost Pareto를 개선한다.
5. **Generalization:** unseen shift family, second backbone, untouched country에서 regret 우위가 유지된다.

이 경우 예상 기여는 세 가지다.

- EO embedding product의 contract-conditional task risk/action-value formulation.
- spatial/event-aware label-free action-value estimator와 support/abstention mechanism.
- multi-task shared extraction cost를 포함한 benchmark 및 외부 transfer evidence.

### 결과별 정직한 착륙점

| 결과 | 적절한 논문 |
|---|---|
| C1–C5 통과 | CVPR/ICCV method + systems evidence |
| proxy–utility 분리 일반화, risk predictor 실패 | `When Earth Embedding Diagnostics Mislead` benchmark/analysis |
| task 이질성 없음 | global refresh schedule의 충분성에 대한 negative result; router 중단 |
| public shift는 되나 AI-Hub만 실패 | public robustness/performance-estimation paper; 한국 demo 분리 |
| 한국은 되나 외부 transfer 실패 | Korea-specific applied EO/remote-sensing journal |
| M37 한 지역만 유지 | EarthVision/workshop 또는 Ai2-facing engineering case study |

현재 냉정한 확률 판정은 다음과 같다.

- **문제의 유의미성: 높음.** embedding products와 continent-scale serving이 실제로 등장했고,
  EarthShift가 OOD 저하를 확인했다.
- **현재 novelty 확정도: 중간 이하.** 주변 분야가 매우 빠르게 차고 있으며 method evidence는 0이다.
- **CVPR 가능성: 조건부.** `C1 task heterogeneity`와 `C2 action-value prediction`이 가장 큰 관문이다.
- **취업 가치: 이미 높음.** contract, reproduction, failure analysis, cost accounting은 Ai2 platform 문제와
  직접 연결된다.
- **사업 가치: method 이후.** 아직 accuracy 보장·실시간 경보 제품을 판매할 단계가 아니라,
  embedding lifecycle audit/monitoring prototype을 만들 단계다.

---

## 9. 논문에서 금지할 표현

- “OLMoEarth는 실시간 예측 모델이다.”
- “라벨 없이 성능을 보장한다.”
- “seam을 줄이면 정확도가 오른다.”
- “큰 context/decoder가 더 좋다.”
- “다중 task cache/refresh/router를 최초 제안한다.”
- “Nepal/Switzerland로 transfer된다.” — 측정 전 금지.
- “AI-Hub 데이터는 인용만 하면 원본 재배포 가능하다.” — 논문 내 최소 이미지 인용과 원본 배포는
  다른 권리다.

대신 `under predefined shift families`, `target-unlabeled selection`, `source-labeled meta-training`,
`paired spatial/event evaluation`, `cost-aware action selection`을 정확히 쓴다.
