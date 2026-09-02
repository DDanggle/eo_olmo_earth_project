# 연구 주제 정렬 — 쉬운 버전

갱신: 2026-09-02 (CacheTune method 후보 반영). 이 문서는 실험 번호보다 먼저 읽는 연구 설명이다.

## 한 문장

> **한 번 계산한 OLMoEarth 임베딩 캐시를 새 지역에서도 버리지 않고 적응할 수 있는가? 라벨과
> 계산 예산이 달라질 때 reuse → cache-adapt → encoder-adapt → raw retrain 중 어디까지 가야 하는가?**

여기서 `임베딩`은 위성영상 여러 장을 OLMoEarth가 압축해 만든 공간 특징 지도이고, `재사용`은
encoder를 다시 학습하지 않고 이 특징 위에 작은 판독기만 붙이는 것이다.

## 지금까지의 이야기를 네 단계로

1. **쓸 수 있는가?** 8개 held-out 지역에서 frozen OLMoEarth(P4)는 raw UNet3D(P2)보다
   region-macro `.2722 vs .1966`, 지역 승리 `6/8`이었다(M65). 재사용 가능성은 확인됐다.
2. **아무 GeoFM이나 되는가?** Presto C1a/C1b는 `.1092/.1261`로 P4와 P2에 8/8 패배했다.
   pooling은 순위를 설명하지 못했지만 off-domain second model 하나이므로 universal 주장은 금지한다.
3. **둘을 그냥 섞으면 되는가?** 아니었다. MS-90B/91/92에서 고정·학습 융합이 모두 gate를
   실패했고 FP-matched oracle 이득도 사라졌다. 이 분기는 negative result로 닫고 v3를 만들지 않는다.
4. **새 나라에서도 유지되는가?** 한국 spatial holdout은 아직 열지 않았다. raw recipe와
   source-label curve를 먼저 봉인한 뒤 Korea first-look가 최종 외부 시험이다.

## Nepal의 정확한 자리

Nepal은 논문의 표본 수를 늘리는 아홉 번째 지역이 아니다. 한 실제 홍수 사건에서 `전후 임베딩
거리 → 평시 변화와 비교 → 사람 검토 순위`가 외부 flood proxy와 정합하는지 본 운영 case study다.

- 2.56 km 창 순위는 외부 라벨 기저율이 너무 높아 무판별이었다(NP-86).
- 40 m 토큰에서는 OLMo Δz와 세 기관 proxy가 정합했다(AUROC `.8459`, NP-88).
- 강한 post-event NDWI가 AUPRC에서 OLMo를 이겼고 공간 block 차이 CI가 0을 포함했다
  (NP-89A). 따라서 classical 우월성은 주장하지 않는다.
- 선택한 OSM simulation route의 150/300/600 m 버퍼 밖에서도 AUROC가 유지됐다(NP-89B).
  이는 그 **한 중심선**만 외운 설명을 약화하지만, 누락된 지류·전체 수계·단일 사건 문제를
  제거하지 않는다.

## 논문 기여는 결과에 따라 두 층으로 나눈다

### 반드시 남는 측정 논문

> Geographic shift 아래 frozen Earth embedding 재사용의 이득·실패·비용·계약을 강한 대조와
> 봉인된 외부 test로 측정한다.

M65, MS-87/MS-93, raw recipe audit, label-budget, Korea first-look가 이 논문의 최소 뼈대다.

### 종료된 method 후보

GeoContextGate는 사전 승급 기준과 stop rule에 따라 method 후보에서 내려갔다. 논문에는 작동점
불일치가 허위 fusion headroom을 만드는 negative/analysis 결과로만 남긴다.

### 새 method 후보 — CacheTune

prediction을 섞는 대신 저장된 공간 임베딩 자체에 작은 low-rank residual을 붙인다.

```text
새 지역 label 0개      → A0 기존 cache+head 재사용
새 지역 label 몇 개    → A1 head만 적응 / A2 cache residual 적응
cache 적응으로 부족함  → A3 encoder APLA·LoRA + 재임베딩
GFM이 부적합함         → A4 raw U-TAE/UNet3D 재학습
```

핵심은 low-rank라는 이름이 아니라 **A2는 기존 cache를 계속 유효하게 두지만 A3는 cache를
무효화한다**는 engineering 차이다. 현재 P4 decoder도 이미 768→128 projection을 학습하므로,
source decoder를 동결한 A2가 head-only A1과 구별될 때만 method로 승격한다. 상세 kill gate는
`docs/CACHE_COMPATIBLE_POSTTRAINING_2026_09_02.md`를 따른다.

MoE는 지금 만들지 않는다. 여러 지역/task용 작은 adapter가 실제로 서로 다른 곳에서 승리해
oracle 이득이 확인될 때만 adapter mixture로 연다. P2/P4 prediction fusion은 계속 종료 상태다.

## 다음 결정 순서

1. **완료 — C1b·fusion closure** — C1b 24/24와 MS-90B/91/92를 증거로 닫았다.
2. **raw recipe audit** — current 40ep BCE와 official-like 75ep BCEDice를 source-only val로 비교한다.
3. **CacheTune PT-0/PT-1** — target support/query를 공간 분리해 봉인하고, exposed 2지역 36-run으로
   head-only와 strict cache adapter를 먼저 반증한다.
4. **source-label baseline** — 봉인된 1/5/10/100%는 target label 0인 별도 곡선이다. 전체 432회를
   돌릴 경우 144회만으로 결론내지 않는다.
5. **encoder PEFT ceiling** — CacheTune gate 통과 때만 APLA/LoRA와 재임베딩 비용을 비교한다.
6. **Korea sealed first-look** — 모든 선택을 Sen12에서 고정한 뒤 spatial support로 적응하고 sealed
   query를 한 번만 연다. 동일 cache의 3-task 비용도 함께 잰다.

Nepal 다사건 확장, 전국 검색, 물리 시뮬레이션은 위 사슬을 대체하지 않는 portfolio/후속 연구다.

## 세 목표와의 연결

| 목표 | 이 연구가 주는 것 |
|---|---|
| AI2 취업 | OLMoEarth cache contract + rslearn-native PEFT comparator + sealed post-training pipeline |
| 박사/CVPR | geographic few-shot의 cache-compatible method + label–compute–invalidation frontier |
| 사업 | 새 국가/task마다 planetary cache를 다시 만들지 않고 작은 adapter artifact를 배포하는 구조 |

현재 사업 문장은 “재해를 자동 판정한다”가 아니라 **“한 번 만든 Earth embedding을 새 지역과
task에 작은 adapter로 맞추고, 부족할 때만 비싼 재임베딩을 선택하는 EO post-training layer”**다.


## 주장 정정 표 (2026-09-02, 교수 검토 반영 — 이 표가 본문과 충돌하면 이 표가 우선)

| 주제 | 쓰면 안 되는 표현 | 쓸 수 있는 표현 | 근거 |
|---|---|---|---|
| M65 헤드라인 | "7/8 승" | **6/8 사전등록 승(5/8 strong)**; 지역 평균 기준 최고 raw 대비 양수는 7/8 (관찰) | M65 |
| raw baseline 반론 | "종결", "공개 최강 baseline 격파" | **"서로 다른 두 raw temporal architecture(UNet3D·U-TAE 계열)에서 재현"** — P3 는 matched 재구현, 75-epoch BCEDice 감사 잔존 | MS-93 |
| Presto | "범용 GeoFM 우월" | "효과가 matched Presto control 로 확장되지 않았고 native grid 에서도 순위가 안 바뀜" (4시점 off-domain 계약, Presto 하한) | MS-87/93 |
| Nepal | "고전 변화탐지보다 우월" | "외부 flood proxy 와 40 m 규모로 정합한 embedding 기반 우선순위" — post-event NDWI AUPRC .291 > OLMo .255, 공간 block CI 0 포함, 600 m 밖 .873 은 route confound 하나만 약화 | NP-88/89 |
| MS-94 A1 | "첫 양성 paper result" | "두 exposed 지역의 강한 개발 신호(+.068/+.080)" — 확증은 A1 vs A4w 게이트 후 8지역에서 | MS-94, fewshot prereg |
| Tadi 대조 | 1.3% / 3.6% | **2.3% / 6.2%** (장면 밖 35% 0-채움 제외) | M77 정정 |
