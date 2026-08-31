# 논문 서사 SSOT — M65 이후 transfer 본선

갱신: 2026-08-31. 이 문서는 M65 이후 논문 주장과 실험 우선순위의 단일 기준이다.
과거 router·live-residual·Nepal 계획은 역사로 보존하지만 이 문서의 실행 queue를 덮지 않는다.

## 한 줄 판정

> **현재 결과는 유의미한 cross-region measurement다. 아직 CVPR method는 아니다.**

M65는 frozen OlmoEarth v1 + small decoder가 8개 held-out 지역의 positive-tile macro IoU에서
P2/P3보다 높은 region-macro를 보였다는 사실을 봉인했다. 다음 질문은 이 현상이
`OlmoEarth 고유 / 일반 frozen GeoFM / decoder·해상도 계약 / 작동점` 중 무엇인지 분리하는 것이다.

## 논문의 현재 질문

> **재사용 가능한 frozen Earth embedding은 지리적으로 분리된 산사태 분할에서 언제 raw
> task model보다 유리하며, 그 이득은 모델 family·label budget·입력/라벨 계약이 바뀌어도 남는가?**

이 질문은 이미 기각된 label-free router를 되살리지 않는다. `reuse / refresh` 의사결정은 장기
프로그램에 남지만 이번 논문의 실증 단위는 **representation family × region × label budget**이다.

## 이미 닫힌 사실

| 사실 | 수치 | 허용 해석 |
|---|---:|---|
| 8-region region-macro | P4 .272166 · P2 .196558 · P3 .183436 | frozen OLMoEarth transfer viability |
| P4−P2 | +.075608 | 이 S12q·LOCO·decoder 계약에서의 관측 격차 |
| 지역 승리 | 6/8, strong 5/8 | 이득은 균일하지 않음 |
| 예외 | Indonesia 패배, Itogon all-seed 규칙 실패 | universal superiority 금지 |
| sensitivity | Thrissur 제외 P4−P2 +.068220 | 공개한 provenance deviation에만 의존하지 않음 |

## 반드시 고친 주장

1. **P4는 새 2단 방법이 아니다.** P4는 `frozen OlmoEarth last-layer spatial cache + small
   decoder` arm이다. M57–M58의 confirmatory 절차는 실행 provenance를 보호하는 과학 프로토콜이지
   segmentation architecture가 아니다. 따라서 “확증 절차가 FP를 줄였다” 또는 이를 방법 기여로
   쓰지 않는다.
2. **P2를 supervised SOTA라고 부르지 않는다.** P2는 공식 Sen12 UNet3D **architecture를 우리
   S12q·LOCO 계약에서 재학습한 matched raw baseline**이다. 공개 Sen12 수치는 S2+DEM, 15시점,
   다른 split·metric·75/100 epoch이어서 M65와 직접 우열 비교할 수 없다.
3. **FP 5–21배는 현재 8지역 headline이 아니다.** M59·M63·M64의 세 초기 지역, threshold 0.5
   관찰이다. 전 8지역 집계와 FP-budget-matched 곡선 전에는 “공간 문맥이 오경보를 억제했다”를
   가설로만 둔다.
4. **label-budget curve 자체는 novelty가 아니다.** PANGAEA, PhilEO 계열, 2026 ML4RS의
   precomputed-embedding label-efficiency 연구가 이미 이 질문을 다룬다. 여기서는
   `cross-region landslide + 동일 표본 + 모델 family + subset uncertainty`를 함께 닫을 때만 기여다.
5. **한국은 아직 annotation-matched transfer가 아니다.** polygon 면적/MMU 유사성은 ontology,
   촬영시점, 라벨 생성원, debris-flow 포함 여부를 같게 만들지 않는다. `T-m − T-x = annotation
   effect`는 인과적으로 성립하지 않는다. 같은 scene을 두 지침으로 재라벨하지 않는 한 Italy는
   **annotation-stress contrast**로만 부른다.
6. **From Pixels to Patches는 인접 문헌이지 P4의 방법 근거가 아니다.** 그 연구는 dense
   embedding을 patch label로 pooling하는 문제다. P4는 spatial grid를 유지한 pixel segmentation이다.

## 이번 논문의 기여 ladder

### A. 지금 이미 가능한 measurement paper

1. annotation-author confounding과 spatially disjoint S12q 계약.
2. 8지역·3seed의 frozen-vs-raw transfer 표와 지역 이질성.
3. 실행 provenance 위반을 탐지하고 snapshot 실행기로 고친 재현성 부록.

이 정도는 workshop/TMLR/TGRS형 측정 논문으로는 유의미하지만 CVPR main-track 방법 기여로는 약하다.

### B. 반드시 채워야 하는 방어벽

1. **Presto C1**: OLMo 고유 효과인지 일반 frozen representation 효과인지 분리.
2. **두 readout을 함께 보고**:
   - `common-grid`: Presto 128²를 32²로 사전등록 pooling해 P4와 동일 decoder/upscale 경로 사용.
   - `native-grid`: 각 encoder의 native spatial product로 실제 사용 성능 보고.
   common-grid가 representation 비교, native-grid가 product comparison이다.
3. **Label learning curve**: 1/5/10/100%만으로 “교차점 특정”이라 쓰지 않는다. nested subset을
   region×positive/negative로 층화하고, 최소 3개 **subset seed**를 둔다. optimizer seed만 3개로
   바꾸는 것은 label-sampling uncertainty를 측정하지 못한다. x축은 fraction과 함께 labeled tile 수,
   positive tile 수를 보고한다.
4. **Korea external transfer**: v2 mosaic/coverage gate 뒤 ontology·time·provenance audit를 먼저
   통과한다. 12밴드 cube를 보존하되 primary model view는 Sen12/Presto와 같은 10밴드이며
   B01/B09는 양쪽에서 missing 처리한다. 통과 전에는 동일 task가 아니라 joint
   geographic+annotation+acquisition shift다.

### C. CVPR method로 승격할 수 있는 단 하나의 단기 후보

**가칭 `GeoContextGate`: false-alarm-budgeted context/detail fusion.**

- frozen Earth embedding branch는 coarse context와 낮은 empty-tile FP 후보를 제공한다.
- raw temporal branch는 10 m detail을 제공한다.
- target label 없이 source-region에서 학습한 작은 gate가 `review / suppress / refine`를 고른다.
- 평가는 평균 IoU만이 아니라 고정 empty-tile FP budget에서 positive-tile IoU와 risk–coverage를 본다.

그러나 Clay-CNN hybrid와 일반 feature fusion이 이미 존재하므로 **바로 성능 실험으로 가지 않는다.**
먼저 기존 P2/P4 예측에서 per-tile oracle headroom을 재는 M86 screen을 실행했다. region-macro
headroom은 **+.023753**, 5/8 지역이 +.02 이상으로 사전 gate를 통과했다. empty-tile FP도 P4가
7/8 지역에서 낮았고 P2/P4 비 중앙값은 **5.0226×**였다. 따라서 development prototype은 열 수
있지만, target label oracle·threshold .5 분석이므로 deployable method 증거는 아니다.

Conformal review-budget은 그 다음 stretch다. spatial dependence와 unseen-region shift에서 단순
exchangeability가 깨지므로, calibration guarantee를 새로 증명하거나 block/group calibration을
검증하지 않으면 conformal이라는 이름만 붙이지 않는다.

## 실행 queue

| 순서 | 작업 | 판정 |
|---|---|---|
| **0 완료** | 기존 72 per-sample 결과 CPU-only 메커니즘 감사(M86) | FP screen PASS · fusion oracle screen PASS; 새 confirmatory/제품 주장 금지 |
| **1** | Presto 16/64/256-pixel smoke → 6,834 cache seal | GPU1 유휴, exact month/WGS84, deterministic/finite/content seal |
| **2** | C1 common-grid + native-grid, 8지역×3seed | OLMo 고유성·해상도 confound 분리; retrospective control 명시 |
| **3** | label-budget pilot → full curve | nested subsets, ≥3 subset seeds, same IDs across arms |
| **4** | Korea v2 40-sample preflight → full health/ontology audit | v1 0-fill cube 사용 금지; task equivalence 실패 시 stress test로 강등 |
| **5** | Korea sealed external test 1회 | P4/P2/P3/C1 recipe와 primary metric을 개봉 전에 봉인 |
| **6** | GeoContextGate 또는 집필 | M86은 필요조건만 통과. Presto 뒤 naive fusion을 먼저 이겨야 method로 승격. 승격 기준은 `config/geocontextgate_promotion_gate.json`(2026-08-31 사전등록)에 고정 — oracle headroom은 상한이므로 실현 이득 ≥ +0.01 또는 headroom의 50%, 고정 FP budget, label-free gate, 사후 완화 금지 |

현재 H200은 2026-08-31 확인 시 GPU0/1 모두 약 68.5 GiB, utilization 99%였다. 새 GPU 실행은
시작하지 않는다. confirmatory에는 72 test JSONL과 288 probability-map 파일이 있어 순서 0은 CPU로
가능하다.

## 이번 사이클에서 빼는 것

- Switzerland event join(0건), Nepal live event, physics simulation.
- label-free block/task router와 R-event headline(기존 gate 실패).
- AI-Hub 3-task ranking reversal(ontology·표본·v2 cube가 모두 먼저 필요).
- Clay v1/v1.5 release pair. Presto와 Korea가 닫힌 뒤 second-family breadth로만 검토.
- “OLMoEarth 우월성”, “지도학습 SOTA를 이김”, “annotation effect를 분리함” 문구.

## 최근 문헌과의 정확한 자리

- [OlmoEarth, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Herzog_OlmoEarth_Stable_Latent_Image_Modeling_for_Multimodal_Earth_Observation_CVPR_2026_paper.html):
  원 모델의 frozen probe/fine-tuning 기준. Sen12Landslides 직접 평가는 확인되지 않았다.
- [PANGAEA](https://arxiv.org/abs/2412.04204): GeoFM이 supervised baseline을 일관되게 이기지
  못하고 label-limited 평가가 중요함을 보였다.
- [EarthShift](https://arxiv.org/abs/2605.29330): 8 GeoFM·11 task에서 OOD가 평균 15–20% 낮아
  지역 shift 자체만으로는 novelty가 아님을 보여준다.
- [Clay-CNN Hybrids](https://arxiv.org/abs/2606.14081): landslide에서 generic hybrid/LoRA는 이미
  존재한다. 따라서 단순 OLMo+UNet 결합은 신규성이 없다.
- [From Pixels to Patches](https://arxiv.org/abs/2603.02080): pooling 선택이 spatial shift에서
  큰 차이를 만들므로 Presto common-grid pooling을 숨은 구현 상세로 두면 안 된다.
- [Earth Embeddings](https://arxiv.org/abs/2608.03410): reusable embedding의 storage,
  reproducibility, spatial transfer, uncertainty 공백을 정리한다. 본 논문의 product-level framing과 맞는다.
- [Conformal Prediction Sets for Instance Segmentation](https://arxiv.org/abs/2602.10045): review-budget
  stretch의 직접 선행. 단 semantic landslide mask와 spatially dependent cross-region setting은 별도다.

## 최종 claim template

> Under a sealed, spatially disjoint ten-region landslide protocol, a frozen OlmoEarth representation
> with a small decoder transferred better than two task-specific raw baselines in six of eight held-out
> regions. We then separate model-family, spatial-readout, label-sampling, and external-dataset effects;
> we do not treat the observed gain as universal GeoFM superiority.
