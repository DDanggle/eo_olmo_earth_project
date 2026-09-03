# MS-93 이후 교수 감사 — `Reuse or Retrain?`로 논문 등뼈 세우기

갱신: 2026-09-02. 적용 범위: Sen12Landslides S12q 본선, Presto C1, MS-90B/91/92,
다음 label-budget 및 Korea 외부 전이. Nepal은 운영 case study로만 참조한다.

## 결론부터

연구는 유의미하다. 그러나 현재 가장 강한 문장은 `OLMoEarth가 우월하다`도,
`두 모델을 합치면 낫다`도 아니다.

> **지리적으로 분리된 산사태 분할에서, 미리 계산한 frozen OLMoEarth 표현은 두 raw
> task-model 구조와 한 second-GeoFM control보다 더 잘 전이됐다. 이 이득이 라벨·계산 비용이
> 줄어들 때도 유지되는지, 그리고 새 외부 계약에서 재현되는지를 측정한다.**

현재 arm에서 비교하는 것은 cache를 다시 계산하는 일이 아니라, frozen cache 위 작은 head를
**재사용**할지 raw task model을 **새로 학습(retrain)**할지다. 따라서 이번 논문의 정확한 작업명은
`Reuse or Recompute?`보다 **`Reuse or Retrain?`**이다. `recompute`는 향후 모델 릴리스 교체나
센서 계약 변경에서 embedding을 다시 만들 때만 사용한다.

## 지금 단단한 것과 아직 약한 것

| 층 | 판정 | 정확한 근거/경계 |
|---|---|---|
| frozen 재사용 viability | **강함** | P4 `.2722` vs P2 `.1966`, P3 `.1834`; region mean 기준 최고 raw 대비 7/8 |
| second-GeoFM 대조 | **유용하지만 한정적** | Presto C1a `.1092`, C1b `.1261`; 둘 다 P4/P2에 8/8 패배. 단 Presto는 사건 중심 4시점 계약이 설계 영역 밖 |
| pooling 반론 | **순위 설명으로는 닫힘** | native C1b가 pooled C1a보다 `+.0169`지만 P4/P2와의 순위를 못 바꿈. “pooling 영향 0”은 아님 |
| raw baseline 방어 | **개선됐으나 미완** | UNet3D와 U-TAE 두 구조가 있음. 그러나 U-TAE는 reimplementation이고 공개 benchmark의 DEM·15시점·BCEDice·75 epoch와 다름 |
| 융합/게이트 | **증거로 종료** | MS-90B/91/92 불통과. FP 작동점을 맞추면 oracle조차 P4 대비 `+.008/-.004`; v3 금지 |
| label efficiency | **미측정** | 등록된 fraction은 있지만 subset manifest·실행·통계 모델이 없음 |
| 외부 transfer | **미측정** | Korea test는 아직 봉인. Nepal은 단일 flood proxy case이고 본선 외부 landslide test가 아님 |

P4−P2의 8개 지역 평균 격차는 `+.0756`, 지역 간 표본표준편차는 약 `.0603`이다. 7/8 지역에서
평균 격차가 양수지만, 지역이 통계적 단위이므로 24 seed run을 독립 표본처럼 세면 안 된다.
8지역 sign test는 단측으로는 약 `.035`, 양측으로는 약 `.070`이다. 효과 크기는 크지만 지역 수는
여전히 작다는 뜻이다.

## 사용자 요약에서 바로 고칠 네 문장

1. **“P3는 공개 최강급” 금지.** 공식 구조에 가까운 matched baseline이라고 쓴다. 저장소 코드도
   U-TAE가 원 구현과 bit-identical하지 않다고 명시한다. 공식 공개 수치는 S2+DEM, 15시점,
   random split, 75 epoch라 현재 LOCO 결과와 직접 비교할 수 없다.
2. **“baseline 리스크가 해소됐다” 대신 “구조 다양성 반론이 약해졌다.”** recipe tuning 반론은
   남는다. P2/P3가 모두 같은 40-epoch BCE 계열에서 낮은 것은 두 구조의 실패일 수도, recipe의
   실패일 수도 있다.
3. **“아무 FM이나 아님”은 한 second model에 한정한다.** Presto 하나가 패했다고 universal
   OlmoEarth specificity가 성립하지 않는다. 정확한 문장은 “이 효과가 matched Presto control로
   자동 확장되지는 않았다”다.
4. **M73 `AI > classical 9/9`를 본선 기둥으로 쓰지 않는다.** 해당 historical change screen의
   선택 baseline에 대한 결과다. Nepal NP-89에서는 strong post-NDWI가 AUPRC에서 Olmo를 이겼다.

## 144회가 아니라 432회인 이유

새 fraction만 세면 `3 fractions × 2 arms × 8 folds × 3 optimizer seeds = 144`다. 그러나 기존
계약은 label-sampling uncertainty를 따로 재기 위해 **subset seed도 3개**를 요구한다.

```text
3 fractions × 2 arms × 8 folds × 3 subset seeds × 3 optimizer seeds
= 432 new training runs
```

100%는 subset seed가 의미 없으므로 기존 P2/P4 48개 실행을 endpoint로 재사용할 수 있다. 단
recipe나 `drop_last`, loss, epoch 수를 바꾸면 같은 곡선 endpoint가 아니므로 100%도 다시 돌려야 한다.

## 더 중요한 문제 — 두 label-budget 질문은 다르다

### A. 지금 바로 가능한 질문: source-label efficiency

> 다른 지역에서 얻은 학습 라벨을 1/5/10%로 줄여도, frozen cache가 완전히 보지 못한 target
> region에 더 잘 전이되는가?

현재 LOCO 구조와 맞고 구현이 단순하다. 그러나 “새 지역에 라벨 N장이 있다”는 질문에는 답하지
않는다. target region label은 계속 0개다.

### B. 실제 운영 질문: target few-shot adaptation

> 새 지역에서 공간적으로 분리된 k개 라벨 타일을 얻었을 때, 기존 cache head만 적응할지 raw
> model을 다시 학습할지 어느 쪽이 label–compute–accuracy frontier가 좋은가?

이 질문은 더 강하고 원래 `reuse/retrain` 의사결정에 직접 연결된다. 대신 target region을
adaptation block과 untouched evaluation block으로 공간 분리해야 하며, 현재 LOCO test를 그대로
잘라 쓰면 인접 누수 위험이 있다.

권장: **A를 본선의 즉시 실행 가능한 label-efficiency 곡선으로 닫고, B를 Korea에서 spatial
few-shot final test로 설계한다.** 논문에는 두 축을 섞지 않고 각각 `source supervision`과
`target adaptation`으로 이름 붙인다.

## label-budget을 돌리기 전에 막아야 할 다섯 함정

1. **고정 validation 라벨 비용** — 현 실행기는 전체 val region 라벨로 매 epoch best model을
   고른다. 따라서 1%는 `total labels 1%`가 아니라 `train labels 1% + fixed full validation`이다.
   이 조건을 제목·x축에 명시하거나 validation도 함께 budget 안에 넣어야 한다.
2. **작동점 confound** — MS-91/92가 보여줬듯 IoU@0.5만으로 승자를 정하면 calibration 차이가
   방법 차이로 보인다. primary는 source-val에서 고른 threshold와 empty-tile FP cap을 함께 쓰고,
   exact AP와 IoU@0.5를 보조로 남긴다.
3. **unlabeled normalization access** — P4 embedding mean/std를 1% labeled subset에서 낼지,
   전체 unlabeled source pool에서 낼지 고정해야 한다. 운영 시나리오에는 전체 unlabeled pool이
   자연스럽지만 `few-shot supervised`가 아니라 `label-limited, unlabeled-available` 조건이 된다.
4. **subset uncertainty** — sample ID를 stable hash로 정렬해 region×positive/negative stratum별
   prefix를 취한다. 1→5→10%는 같은 순열의 nested prefix이고 두 arm이 동일 ID를 쓴다.
5. **학습 budget** — 40 epoch를 유지하면 sample exposure는 같지만 optimizer update 수는 fraction에
   따라 크게 줄어든다. 고정 epoch를 primary로 유지하되 update 수·본 tile 수·validation 비용을
   모두 보고한다. 결과를 본 뒤 fixed-step으로 바꾸지 않는다.

CPU preflight로 확인한 1% subset은 fold당 약 57–69타일, 양성 27–33타일이다. batch 16에서
zero-batch는 아니지만 epoch당 3–4 step뿐이다. `drop_last=True`는 매 epoch 최대 15장을 버리므로
subset manifest와 실제 loader exposure를 따로 기록해야 한다.

## baseline-strength 방어 순서

라벨 432회를 곧장 시작하기 전에 **raw recipe audit**를 source-only validation으로 닫는다.

- 비교 후보는 현재 40-epoch BCE recipe와 공식 저장소에 가까운 75-epoch BCEDice recipe다.
- P2와 P3를 같은 하나의 recipe에 강제로 묶기보다, 각 arm의 source-validation 최선 recipe를
  별도 보고한다. 동시에 현재 common-recipe 표도 controlled comparison으로 보존한다.
- target fold 라벨은 recipe 선택에 사용하지 않는다. 각 fold의 train/val만 사용한다.
- 이 audit 뒤 label curve primary raw arm을 P2로 고르면, “100%에서 P2가 더 좋아 골랐다”는
  retrospective selection을 명시한다. P3 100% endpoint는 계속 표에 둔다.

이 방어 없이 P2/P3 두 구조를 추가한 것만으로 supervised baseline 공격이 끝났다고 쓰면 위험하다.

## 최종 실험 순서

1. **완료 — C1b provenance 봉인**: 24/24 원시 JSON, 실행 snapshot hash, seed-level metric을
   `artifacts/c1b_presto_native_compact_v1.json`으로 로컬화한다.
2. **LB-0 raw recipe audit**: 현재 recipe vs official-like recipe. source validation으로만 선택.
3. **완료 — LB-1 manifest/preflight**: 3 subset seed의 nested ID, label count, positive count,
   batch count, checksum을 CPU에서 생성했다. 성능은 읽지 않았다. compact seal은
   `artifacts/source_label_budget_subsets_compact_v1.json`, full manifest는 서버에 있다.
4. **LB-2 staged execution**: 첫 subset seed 144회는 오직 실행/비용 screen으로 보고, 문제가 없으면
   결과 방향과 무관하게 남은 288회를 수행한다. 144회만으로 논문 결론을 내리지 않는다.
5. **LB-3 analysis**: 통계 단위는 region. region → subset seed → optimizer seed의 계층 bootstrap,
   per-region curve, simultaneous interval을 보고한다. 관측 구간에 교차가 없으면 억지 crossover를
   맞추지 않고 `no observed crossover up to 100%`로 쓴다.
6. **K-0 Korea preflight**: canonical mosaic/coverage, ontology, time, 10-band view를 닫는다.
7. **K-1 sealed external test**: source-label curve에서 고른 recipe와 target few-shot 규칙을 먼저
   봉인한 뒤 한 번만 개봉한다.

## 논문 기여를 이렇게 묶는다

1. **Measurement:** 8-region × 3-seed geographic transfer에서 cached frozen embedding과 두 raw
   temporal architectures를 비교한다.
2. **Representation control:** common/native readout의 Presto control로 model-family와 grid
   confound를 분리한다.
3. **Label–compute frontier:** source-label budget과 cold/warm cache 비용을 함께 보고, 어느 budget에서도
   raw가 따라잡지 못하면 그 자체를 결과로 둔다.
4. **Negative mechanism result:** false-alarm 작동점을 맞추면 fusion oracle headroom이 사라진다는
   재현 가능한 반증을 보고한다.
5. **External stress test:** Korea에서 acquisition+annotation shift를 숨기지 않고 final first-look를
   수행한다.

이 구성은 새 segmentation architecture를 내는 CVPR method paper가 아니다. 현재 그대로라면
TGRS/TMLR/NeurIPS Datasets & Benchmarks 성격이 더 자연스럽다. CVPR main 가능성을 올리려면
label–compute frontier가 매우 선명하고 Korea first-look가 생존하며, 평가 프로토콜 자체가 다른
embedding products에도 재사용 가능하도록 코드·manifest가 공개돼야 한다.

## 최근 문헌과의 위치

- [OlmoEarth (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Herzog_OlmoEarth_Stable_Latent_Image_Modeling_for_Multimodal_Earth_Observation_CVPR_2026_paper.html)
  원 논문은 여러 task에서 frozen embedding과 fine-tuning을 평가하지만, 이 저장소의
  S12q landslide LOCO 질문을 직접 닫지는 않는다.
- [PANGAEA](https://arxiv.org/abs/2412.04204)는 limited-label 평가와 supervised baseline을 이미
  요구하며 GeoFM이 항상 이기지 않는다고
  보고한다. 따라서 label curve 자체는 novelty가 아니다.
- [EarthShift](https://arxiv.org/abs/2605.29330)는 geography/time/sensor shift에서 모델들이 OOD
  성능을 잃는다고 보여준다. 따라서
  “지리 shift가 있다”도 novelty가 아니다.
- [Beyond Accuracy](https://arxiv.org/abs/2608.16614)는 shift 아래 calibration과 모델 순위가 함께
  흔들릴 수 있음을 보인다. MS-91/92의 FP-matched 재판정과 label-budget의 작동점 사전등록이
  주변 분석이 아니라 핵심 평가 계약인 이유다.
- [Earth Embeddings](https://arxiv.org/abs/2608.03410)는 reusable product의 spatial transfer,
  pooling, storage, reproducibility가 열린
  문제라고 정리한다. 본 연구의 강점은 바로 **동일 scene ID, readout contract, label sampling,
  false-alarm operating point, cache cost**를 한 산사태 실험에서 함께 봉인하는 데 있다.

## 최종 권고

**지금 GPU에 144회를 바로 올리지 않는다.** 먼저 논문 질문을 A(source-label efficiency)로 고정하고,
raw recipe audit·subset manifest·평가 작동점을 봉인한다. 그 다음 432회를 결과와 무관하게 완주한다.
Korea는 이를 통과한 뒤에만 열고, 새 지역 k-label 의사결정은 Korea의 spatial few-shot으로 답한다.

## 2026-09-02 method 확장 정정 — measurement를 버리지 않고 engineering novelty를 앞에 둔다

위 권고의 `432회 완주 → Korea에서만 target few-shot` 순서는 measurement paper를 닫는 데는
타당하지만, generic label curve의 선행연구 중복을 감안하면 CVPR method 가능성을 먼저 반증하는
순서로는 비효율적이다. 사용자 요청에 따라 다음처럼 **실행 순서만** 정정한다.

1. raw recipe audit와 봉인된 source-label manifest는 그대로 보존한다.
2. CPU에서 target support/query spatial split과 adaptation runner invariants를 먼저 닫는다.
3. exposed 2지역 36-run으로 low-rank spatial cache adapter가 head-only와 실제로 다른 이득을
   내는지 본다.
4. 통과하면 encoder APLA/LoRA ceiling과 source-label curve를 하나의 action frontier로 확장한다.
5. 실패하면 CacheTune·MoE를 즉시 접고 원래 432-run measurement queue로 돌아간다.
6. Korea는 어느 경우에도 protocol 선택에 쓰지 않고 final external test로 보존한다.

이 정정은 MS-93 결과나 sealed subset ID를 바꾸지 않는다. 새 method의 SSOT와 수치 gate는
`docs/CACHE_COMPATIBLE_POSTTRAINING_2026_09_02.md` 및
`config/cachetune_pt0_preregistration_v0.json`이다.

### 왜 LoRA와 MoE를 그대로 headline으로 쓰지 않는가

- GeoFM LoRA/adapter, limited-label curve, multimodal MoE는 이미 강한 선행이 있다.
- rslearn 환경에도 APLA callback이 있어 encoder PEFT 자체는 강한 baseline이지 새 기여가 아니다.
- MS-90B/91/92는 P2/P4 prediction mixture의 이용 가능한 상보성이 없음을 이미 보였다.

따라서 headline은 **stored Earth embedding product를 무효화하지 않는 target post-training**이다.
MoE는 지역/task별 작은 cache adapter가 실제로 전문화됐다는 oracle matrix가 나온 뒤에만 여는
parameterization이다. 이 분리가 과거 EarthRoute처럼 필요성 증명 전에 router부터 만드는 오류를 막는다.
