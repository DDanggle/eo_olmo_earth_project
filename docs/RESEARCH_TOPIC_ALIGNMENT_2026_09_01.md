# 연구 주제 정렬 — 쉬운 버전

갱신: 2026-09-02. 이 문서는 실험 번호보다 먼저 읽는 연구 설명이다.

## 한 문장

> **새 지역에서도 한 번 계산한 OLMoEarth 임베딩 캐시를 믿고 재사용할 수 있는가? 라벨과 계산
> 예산이 달라질 때, 언제 원영상 task model을 새로 학습해야 하는가?**

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

## 다음 결정 순서

1. **완료 — C1b·fusion closure** — C1b 24/24와 MS-90B/91/92를 증거로 닫았다.
2. **raw recipe audit** — current 40ep BCE와 official-like 75ep BCEDice를 source-only val로 비교한다.
3. **nested source-label budget** — 1/5/10/100%, subset seed 3×optimizer seed 3. 새 실행은 432회다.
4. **Korea sealed first-look** — 모든 recipe를 고정한 뒤 test를 한 번만 연다. 새 지역 k-label
   adaptation은 이 단계의 spatially disjoint 별도 estimand다.

Nepal 다사건 확장, 전국 검색, 물리 시뮬레이션은 위 사슬을 대체하지 않는 portfolio/후속 연구다.

## 세 목표와의 연결

| 목표 | 이 연구가 주는 것 |
|---|---|
| AI2 취업 | OLMoEarth를 실제 계약으로 다루고, 실패·정정·외부 사건 적용까지 보여주는 증거 |
| 박사/CVPR | geographic shift의 label–compute reuse frontier + operating-point negative result + Korea 외부 전이 |
| 사업 | 여러 task가 공유할 cache와 재계산 비용 사이의 운영 결정을 수치로 설명할 기반 |

현재 사업 문장은 “재해를 자동 판정한다”가 아니라 **“어디를 먼저 검토하고, 언제 비싼 재계산을
해야 하는지 결정하는 EO evidence layer”**다.
