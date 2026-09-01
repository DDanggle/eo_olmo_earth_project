# 연구 주제 정렬 — 쉬운 버전

갱신: 2026-09-01. 이 문서는 실험 번호보다 먼저 읽는 연구 설명이다.

## 한 문장

> **새 지역에서도 한 번 계산한 OLMoEarth 임베딩 캐시를 믿고 재사용할 수 있는가? 그렇지 않은
> 경우에는 원영상 모델을 언제 함께 쓰거나 다시 계산해야 하는가?**

여기서 `임베딩`은 위성영상 여러 장을 OLMoEarth가 압축해 만든 공간 특징 지도이고, `재사용`은
encoder를 다시 학습하지 않고 이 특징 위에 작은 판독기만 붙이는 것이다.

## 지금까지의 이야기를 네 단계로

1. **쓸 수 있는가?** 8개 held-out 지역에서 frozen OLMoEarth(P4)는 raw UNet3D(P2)보다
   region-macro `.2722 vs .1966`, 지역 승리 `6/8`이었다(M65). 재사용 가능성은 확인됐다.
2. **아무 GeoFM이나 되는가?** 같은 계약의 Presto C1a는 `.1092`로 P4와 P2에 8/8 패배했다
   (MS-87). 하지만 Presto에 불리한 사건 중심 입력과 4×4 pooling이 있으므로 C1b가 남았다.
3. **둘을 그냥 섞으면 되는가?** 봉인 확률을 평균한 MS-90A는 macro `+.0063`이었지만
   사전 기준은 `3/8`만 통과했다. 고정 평균이 답은 아니다. 이것은 learned gate의 성공 증거가
   아니라, learned gate를 시험할 이유다.
4. **새 나라에서도 유지되는가?** 한국 spatial holdout은 아직 열지 않았다. C1b와 source-region
   method 결정을 먼저 봉인한 뒤 Korea first-look가 최종 외부 시험이다.

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

M65, MS-87, C1b, label-budget, Korea first-look가 이 논문의 최소 뼈대다.

### 통과할 때만 남는 method 논문

> 입력만 보고 frozen cache와 raw model 중 무엇을 얼마나 믿을지 정하는 GeoContextGate.

MS-90A의 단순 융합 실패는 필요성을 증명하지 않는다. 등록된 naive baseline을 모두 닫고,
source regions에서 promotion gate를 통과하며, Korea에서 최고 단일 arm보다 낮지 않아야 method로
승격한다. 하나라도 실패하면 분석 절로 내린다.

## 다음 결정 순서

1. **C1b native-grid 24실행** — C1a의 낮은 값이 pooling 때문인지 확인.
2. **MS-90B CPU baseline closure** — average·AND·OR·logit-mean, validation calibration,
   tie-correct AP, source hash를 채운다. MS-90A는 이 중 일부만 실행했다.
3. **GeoContextGate** — 1·2가 닫힌 뒤에만 GPU 개발. 성공은 보장하지 않는다.
4. **nested label budget** — 1/5/10/100%, subset seed 3개.
5. **Korea sealed first-look** — 모든 recipe를 고정한 뒤 test를 한 번만 연다.

Nepal 다사건 확장, 전국 검색, 물리 시뮬레이션은 위 사슬을 대체하지 않는 portfolio/후속 연구다.

## 세 목표와의 연결

| 목표 | 이 연구가 주는 것 |
|---|---|
| AI2 취업 | OLMoEarth를 실제 계약으로 다루고, 실패·정정·외부 사건 적용까지 보여주는 증거 |
| 박사/CVPR | geographic shift의 reuse 측정 + 통과 시 context-conditioned gate + Korea 외부 전이 |
| 사업 | 여러 task가 공유할 cache와 재계산 비용 사이의 운영 결정을 수치로 설명할 기반 |

현재 사업 문장은 “재해를 자동 판정한다”가 아니라 **“어디를 먼저 검토하고, 언제 비싼 재계산을
해야 하는지 결정하는 EO evidence layer”**다.
