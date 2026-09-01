# 교수 감사 — MS-87 Presto와 NP-88 Nepal 외부 라벨

갱신: 2026-09-01. 이 문서는 M65 이후 새 결과 두 개를 같은 주장으로 섞지 않기 위한 판정문이다.

## 먼저: 측정 ID namespace

두 저장소가 독립적으로 `M86` 이후 번호를 사용해 충돌했다. 기존 파일명과 커밋은 바꾸지 않되,
앞으로 본선은 `MS-`(MountainShift), Nepal sidecar는 `NP-`를 붙인다.

| 기존 표기 | 앞으로 쓰는 표기 | 뜻 |
|---|---|---|
| 본선 M86 | **MS-86** | P2/P4 FP·oracle mechanism audit |
| 본선 M87 | **MS-87** | Presto C1a common-grid control |
| Nepal M86 | **NP-86** | 2.56 km 창 외부 라벨 대조 |
| Nepal M88 | **NP-88** | 40 m 토큰 외부 라벨 정합 |
| 이번 감사 | **NP-89** | NP-88 강한 baseline·공간 의존성 사후 감사 |

## MS-87 판정

C1a는 6,834개 동일 S12q 표본에서 frozen Presto를 128×32×32 common grid로 고정하고 P4와 같은
decoder 경로·split·seed로 비교했다.

- region macro: P4 `.2722`, P2 `.1966`, C1a `.1092`.
- P4와 P2 모두 C1a를 8/8 지역에서 이겼다.
- 따라서 M65는 **아무 frozen GeoFM이나 붙여서 생기는 현상은 아니다.**

그러나 이것만으로 `OlmoEarth-specific`이라고 확정하지 않는다.

1. Presto는 12개월 연속 픽셀 시계열 모델이고 S12q는 사건 중심의 불규칙 12관측이다.
2. Presto 128²를 4×4 평균 풀링했으므로 native detail 손실이 섞여 있다.
3. 표현 차원과 receptive field도 128-d pixel / 768-d spatial-context로 다르다.
4. 결과가 열린 뒤 추가한 retrospective control이다.

허용 문장:

> The transfer gain did not extend to a matched, frozen Presto control under the same S12q event contract.

금지 문장:

> OlmoEarth is universally better than other GeoFMs.

C1b native-grid sensitivity와 최초 미열람 Korea OLMo-vs-Presto 비교가 이 경계를 닫는다.

## NP-88의 살아 있는 결과

Sentinel Asia가 공개한 IWM(PlanetScope), TASA(FORMOSAT-5), JAXA(ALOS-2) 홍수 proxy 합집합은
OlmoEarth 사건 전후 표현거리와 40 m 토큰에서 정합했다.

- pooled AUROC `.8459`, AUPRC `.2548`; label prevalence `.0547`.
- 기관별 AUROC: IWM `.8865`, TASA `.8783`, JAXA `.7961`.
- 이는 창 단위 NP-86이 무판별이었던 이유가 2.56 km aggregation임을 보여주는 유효한 사례다.

그러나 `122,558`은 독립 표본 수가 아니다. 겹치는 47개 창의 공간상관 토큰이며 독립 사건 수는
**1개**다. 외부 product도 현장 피해 정답이 아니라 8월 28일 영상 기반 flood proxy다.

## NP-89 강한 사후 감사

`code/audit_nepal_m88_robustness.py`가 NP-88의 동일 토큰·라벨을 바꾸지 않고 다음을 추가했다.
결과는 `artifacts/nepal_np89_robustness_audit_v1.json`에 source hash와 함께 봉인했다.

| 점수 | pooled AUROC | pooled AUPRC | 5.12 km block-macro AUROC |
|---|---:|---:|---:|
| **OlmoEarth Δz** | **.8459** | .2548 | **.8573** |
| 사건 후 NDWI | .8276 | **.2911** | .8497 |
| spectral angle | .7896 | .1493 | .7928 |
| |ΔNDVI| | .7503 | .1881 | .7518 |
| |ΔNDWI| | .7379 | .1729 | .7522 |

핵심 정정:

- NP-88의 “Olmo가 고전 기법보다 +.10~+.15 우월”은 약한 baseline 두 개에만 성립했다.
- 사건 후 NDWI는 AUPRC에서 Olmo를 이기며, block-macro AUROC 차이는 `.0077`뿐이다.
- 19개 공간 블록의 Olmo−post-NDWI 차이 bootstrap 95% CI는 `[-.0620, .0902]`로 0을 포함한다.
- 따라서 **AI 우월성 주장은 철회**한다.

강 위치 누출을 줄이기 위해 같은 창·같은 80 m 강거리 구간 안에서만 비교하면 conditional AUROC는
Olmo `.8006`, post-NDWI `.7518`이다. 이 신호는 흥미롭지만 NP-88 개봉 후 추가한 사후 분석이고
단일 사건이므로 메커니즘 가설까지만 허용한다.

## 실행 결정

1. Nepal은 더 이상 본선 GPU queue를 선점하지 않는다. NP-89로 case-study claim boundary를 닫는다.
2. 본선 GPU1의 다음 작업은 **C1b native-grid**다. 단 기존 P4 decoder는 128²를 512²까지 확대해
   잘못된 경로가 되므로 즉시 실행하지 않는다. 같은 layer를 interpolation 없이 128²에서 실행하는
   `P4native`와 source-snapshot runner를 먼저 봉인한다.
3. 이어 P2/P4 naive fusion과 GeoContextGate를 development source regions에서 시험한다.
4. method promotion gate를 통과한 recipe만 Korea test 개봉 전에 봉인한다.
5. label-budget curve는 그다음이며, 단일 subset seed 대신 3개 nested subset seed를 유지한다.

현재 GPU가 비어 있다는 사실은 과학적 준비 완료를 뜻하지 않는다. 이번 감사에서 C1b 전용 arm과
runner를 추가했지만, 서버에서 native cache seal·shape smoke·parameter parity·새 OUTROOT를 실제로
통과하기 전에는 24개 학습을 시작하지 않는다.

2026-09-01 read-only 서버 확인에서는 H200 GPU0/1이 각각 0 MiB였고 native cache 6,834파일,
manifest `aad49d14…`, spot-check PASS, `(128,128,128)` float16까지 일치했다. 즉 데이터 측 preflight는
통과했다. 코드 업로드·source snapshot·parameter parity·새 OUTROOT 확인과 실제 학습은 아직 0건이다.

## 논문에서의 자리

- **MS-87**: 본선의 model-family 방어벽.
- **NP-88/89**: 독립 실사건 case study 또는 portfolio demo. 산사태 segmentation headline의
  세 번째 test set이나 표본 수 증가로 세지 않는다.
- Nepal을 일반화 증거로 승격하려면 같은 계약을 다른 flood event에 그대로 적용하고 event-level
  first-look를 최소 한 번 더 확보해야 한다.
