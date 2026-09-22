# 지역 특화 단기 보간과 관측 교정: CVPR 문헌·실험 재감사

작성 기준일: 2026-09-18. 상태: 문헌 조사와 로컬 증거 감사 완료, 새 실험 미실행.

사용자 질문: “좁은 특정 지역을 대상으로 짧은 시간 간격을 보간하고, 실제 영상이 오면 확인해서 업데이트하면 되지 않는가? EO 임베딩·지역 메타정보를 언어/VLM과 연결할 수도 있는가?”

## 0. 먼저 답: 방향은 유의미하다. 그러나 ‘보간이 없다’는 전제는 틀리다

당신이 말하는 것은 단순한 보간기보다 **한 지역의 상태를 계속 추정하고 새 관측으로 교정하는 모델**에 가깝다. 기존 T4가 이 문제를 검증한 것은 아니다. 지역 특화·짧은 간격·실제 시간·온라인 교정·변화 탐지 효용이 기존 실험에서 한꺼번에 평가되지 않았다.

반대로 “시간 보간은 선행연구가 없을 것”이라고 논문을 쓰면 위험하다. AnytimeFormer/AGFlow/HLS-GPT는 날짜 질의 복원을, LIANet은 특정 지역의 신경 표현을, AlphaEarth는 시간 조건부 EO 표현을 다룬다. 2026-09-15 공개된 SPEAR NeXT는 이미 캐시된 frozen EO 임베딩 위에서 지역별 인과적 미래 상태 예측을 학습한다. 고전 S-CCD와 최근 latent assimilation 연구에는 새 관측으로 상태를 교정하는 방법도 있다.

따라서 추천 연구 질문은 다음이다. 이는 신규성 확인이 끝난 주장이 아니라 **검증할 가설**이다.

> 불규칙하고 구름이 낀 EO 관측이 순차적으로 도착할 때, 지역에 적응한 잠재 상태 모델이 계절 변화는 추정하면서 실제 급변은 지워버리지 않고, 새 관측으로 상태와 불확실성을 교정하여, 같은 오경보 수준에서 실제 변화를 더 빨리·더 저렴하게 확인할 수 있는가?

현재 자료만으로 CVPR main에 충분하다고 판단하지 않는다. 그러나 “+.029가 +.03을 못 넘었으니 보간의 상한이고 방법 연구를 접는다”, “언어는 제품일 뿐이다”, “두 GPU면 10주 안에 한 편”이라는 단정 역시 근거가 부족하다. 기존 실패 판정을 보존하면서 **다른 질문을 독립적인 새 프로토콜로** 시험해야 한다.

이 문서는 공식 proceedings, 저널/저자 페이지, arXiv, 저자 코드 등 1차 자료에 연결된 55편을 정리한다. 모든 논문을 전부 정독하거나 재현한 것은 아니다. 아래 S는 관련 본문 일부까지 확인, M은 서지·초록 또는 공식 프로젝트 설명 확인이다. 최신 preprint는 동료심사 결과로 취급하지 않는다.

## 1. 지금 섞여 있는 세 가지 문제

| 문제 | 쉬운 예 | 질의 시 사용할 수 있는 영상 | 검증 대상 |
|---|---|---|---|
| 과거 보간 / smoothing | 1일과 11일을 보고 6일 상태 복원 | 질의 날짜 이후 영상도 허용 | 실제로 가린 6일 관측·독립 상태 라벨 |
| 미래 예측 / forecasting | 1일까지 보고 6일 상태 예측 | 과거에 도착한 관측만 | 이후 실제 관측과 사건 |
| 관측 교정 / filtering | 6일 실제 영상이 오면 앞서 추정한 상태 수정 | 그 시점까지 도착한 관측만 | 교정 전후 오류·변화 보존·오경보·비용 |

기존 T4는 첫 번째다. 당신의 “확인해서 업데이트”는 세 번째다. 실제 운영에서는 두 번째와 세 번째가 이어진다. 이후 영상을 이용한 과거 보간이 잘 된다고 실시간 탐지가 잘 되는 것은 아니다. 반대로 미래의 벌목을 못 맞춘다고 새 관측에서 벌목을 빨리 확인하는 능력이 무의미한 것도 아니다.

또한 ‘업데이트’의 대상을 분리해야 한다.

- 상태 갱신: 모델 가중치는 그대로 두고 그 지역의 기억/상태만 바꾼다.
- 지역 적응: 그 지역의 과거 자료로 작은 모듈이나 지역 코드를 학습한다.
- 지속 학습: 운영 중 새 자료로 모델 가중치도 바꾼다. 망각·평가 누출까지 별도로 다뤄야 한다.

첫 논문에서는 **과거 자료로 지역 적응 → test에서는 가중치 고정 → 새 관측으로 상태만 갱신**을 권한다. 지역 적응과 continual learning을 처음부터 동시에 주장하지 않는다.

## 2. 기존 T/L 결과에서 무엇이 실제로 확인되었나

### 2.1 감사 범위와 확인 수준

읽은 주요 로컬 근거는 다음과 같다. 실행 서버에는 접속하지 않았고 현재 GPU 상태도 확인하지 않았다. `MEASURED_FINDINGS.md`에는 기준일 다음 날인 2026-09-19 표기 기록이 있다. 그 항목은 **로컬 기록에 보고된 값**으로 인용하며, 실제 실행 시각·원격 run 전체를 새로 인증한 것으로 해석하지 않는다.

- 설계: [기존 T/L 설계](RESEARCH_DESIGN_TIME_AND_LANGUAGE_2026_09_18.md), [자산 감사](CVPR_ASSET_INVENTORY_2026_09_17.md), [기존 streaming 관련연구](RELATED_WORK_STREAMING_2026_09_09.md).
- T0: `../code/analyze_postk_v3.py`, `../artifacts/frozen_sensitivity_v0/holdout_*_postk_v3.json`.
- T2: `../code/latent_interp_v0.py`, `../artifacts/latent_interp_v0/summary.json`.
- T4: `../code/latent_interp_train.py`, `../config/latent_interp_train_prereg_v0.json`, `../MEASURED_FINDINGS.md` MS-130. 학습 raw JSON/전체 per-row 파일은 이번 로컬 조회 경로에 없으므로 수치는 기록 기반이다.
- 근접 관측: `../artifacts/latent_noise_floor_v0/summary.json`, MS-132. 이는 반복성 통계이지 입증된 noise ceiling이 아니다.
- L: `../code/earthtalk_projector_train.py`, `../artifacts/earthtalk_projector_v0/scores_seed1.json`, `scores_textonly.json`, MS-131.
- 시간 부호화: 현재 로컬 설치 `olmoearth_pretrain/nn/flexi_vit.py`의 slot 위치 부호화와 month 부호화. v1.2는 논문 §2.5의 mixed 3D RoPE 및 month 설명도 확인했다.

### 2.2 T0는 유용한 관측 수 ablation이다. 아직 실제 탐지 지연 곡선은 아니다

Hiroshima의 21% / 75% / 96%는 사건 후 관측 1/2/3장을 남겼을 때 **전체 관측 control 대비 평균 IoU의 비율**이다. “사건을 탐지할 확률”이 아니다. 예를 들어 1장 arm의 평균 IoU는 .066, control은 .318이다.

별도 `tile_detection_rate`는 양성 타일에서 `bool(pred.any())`로 계산한다. 이는 예측 양성 픽셀이 하나라도 있으면 성공이다. 사건 후 영상이 없는 arm도 Hiroshima에서 .253을 얻는다. 위치 정확도나 음성 지역의 오경보를 통제한 조기 탐지 성공률이 아니다.

평가 집단도 사건 후 관측이 4장 이상인 양성 타일로 제한되며, Indonesia에서는 54개만 남는다. 전체 시간창에서 일부 관측을 가리는 실험이지, 도착 시각에 따른 prefix 평가 자체는 아니다.

허용: “이 pooled 표현과 고정 readout에서 초기 사건 후 관측을 추가하면 성능이 회복된다.”

미입증: “관측이 촘촘해져도 정확도 이득이 없다”, “2장이면 실시간 탐지에 충분하다”, “지역 단기 갱신 모델의 필요성이 없다.” 새 조기 탐지 평가는 음성 기간, 고정 오경보 예산, 실제 달력 지연, 미탐지 사건까지 포함해야 한다.

### 2.3 T2/T4는 당신이 제안한 hyperlocal short-gap 실험이 아니다

T2의 1,500타일·15,000 내부 질의에서 cosine은 nearest .8409, previous .8247, 날짜 가중 linear .8681, 다른 11장 평균 .8747이다. 두 이웃 사이 평균 cosine은 .7978이다.

특히 양쪽 이웃 간 간격 ≤20일은 405/15,000 = **2.7%**다. 기존 top-12 clear 관측 추출은 연간 자료를 희소하게 만들기 때문에 일 단위/수일 단위 연구를 대표하지 않는다. 현재 학습 코드의 날짜 차이는 `.days`로 계산해 시간 이하 차이를 버린다. 이 코드로 hourly interpolation을 시험했다고 말할 수 없다.

T4는 Hokkaido/Kyrgyzstan1/New Zealand를 학습 지역에서 제외하고, 다른 지역으로 학습한 모델의 보간을 평가한다. 각 공간 토큰에 잔차 MLP 또는 시간 문맥 attention을 적용한다. 한 지역의 과거로 적응해 그 지역의 미래 기간을 평가하거나 새 관측으로 상태를 재귀 교정하는 실험이 아니다. 공간 이웃을 명시적으로 모델링하는 네트워크도 아니다.

세 지역은 T2에서 이미 저성능 지역으로 확인해 선택한 dev 성격의 평가 대상이다. 학습에는 없었다는 사실과 연구자가 전혀 들여다보지 않은 최종 test라는 사실은 다르다. 새 주장에는 별도 미사용 기간/지역이 필요하다.

설계는 월 내 질의를 제외한다고 적지만 현재 학습 코드의 내부 슬롯 j=1..10에는 같은 달 제외 조건이 보이지 않는다. 실행 snapshot과 실제 row membership을 확인하기 전 프로토콜이 일치했다고 단정하지 않는다.

### 2.4 +.029와 사전등록 실패를 함께 정직하게 해석하기

MS-130의 학습 모델은 선형 .821에서 약 .850으로 개선했고, seed별 차이는 +.0290 / +.0299 / +.0291이다. seed1의 타일 단위 paired bootstrap CI는 [.0283, .0297]로 보고된다.

- 원래 규칙 “3 seed 중 2개가 ≥+.03”은 불통과다. 문턱을 사후에 낮추거나 성공으로 재분류하지 않는다.
- 그렇다고 “학습 효과가 없다”는 결과가 아니다. 보고된 평가 표본에서 개선은 일관된다.
- `1−cos`라는 각도 손실을 기준으로만 계산하면 .029/(1−.821) ≈ **16.2% 감소**다. 이는 물리 오차·일반 MSE·탐지 성능의 16.2% 개선이 아니다.
- 타일 단위 CI는 공간적 상관과 같은 사건·지역 내 상관을 모두 반영하지 않는다. 실질 효용과 외부 일반화는 아직 별도 질문이다.
- 두 소형 구조와 몇 seed가 비슷하게 수렴했다고 표현의 정보론적/구조적 상한이 입증되지 않는다. 입력, 공간 문맥, 지역 적응, 목표, 모델 클래스가 제한되어 있다.
- T2에서는 전체 평균이 선형보다 높다. T4와 같은 test에서 mean-all/계절 모델을 비교하지 않고 “강한 보간 기준선을 넘었다”고 확대하면 안 된다.

### 2.5 근접 관측 cosine은 noise ceiling이 아니다

MS-132의 ≤10일 쌍 cosine .848, 지역별 .75~.78은 **서로 다른 날짜 관측의 유사도**다. 그 차이에는 실제 생태 변화, 구름·대기, 관측 기하, 정합 오차, 센서 전처리 등이 함께 들어간다. 현재 통계만으로 어느 비율이 관측 노이즈인지 분리할 수 없다.

또한 “A와 B의 유사도”와 “A,C를 입력으로 B를 예측한 유사도”는 서로 다른 조건부 문제다. 전자가 후자의 최대치를 정하지 않는다. 예측 점수가 근접 관측 유사도보다 높아지는 것은 이 ceiling 해석을 정당화하지 않는다.

수학적 반례도 간단하다. 잡음 없는 상태가 `e(t)=(cos t, sin t)`라면 두 이웃 `e(−δ), e(+δ)`의 cosine은 `cos(2δ)<1`이다. 그러나 `0<δ<π/2`에서 둘의 합을 정규화하면 가운데 `e(0)`를 정확히 복원해 cosine 1을 얻는다. 가까운 관측의 유사도가 1보다 낮아도 복원 상한이 낮아지는 것은 아니다.

따라서 `(model−linear)/(close-pair-cos−linear)`를 최대 headroom 달성률로 쓰지 않는다. 반복 측정 모델, 동일 상태 가정, QA·정합 통제 및 독립적인 상태 검증 없이는 상한이라 부를 수 없다.

`1−linear_cos=.132` 역시 1까지 남은 지표 거리이지 달성 가능한 학습 headroom의 추정치가 아니다. window teacher로 바꿔도 “노이즈 없는 정답”이 되지 않는다. teacher는 자기 편향이 있고 사건을 평균으로 희석할 수도 있다.

### 2.6 월 부호화와 pooling: 약한 민감도 ≠ 시간 자체가 없는 모델

현재 확인한 v1 코드에는 시간 슬롯 위치 정보와 month 정보가 모두 있다. v1.2도 3D mixed RoPE와 month를 사용한다. 반면 정확한 날짜 차이·시간 간격을 이 month 설명만으로 충분히 표현한다고 할 수 없다.

같은 달 영상 순서를 바꾼 pooled cosine이 .9996/.9998이라는 결과는 **이 입력·pooling·readout 계약이 순서에 둔감하다**는 증거다. 모든 temporal token에 정보가 없거나 모든 OlmoEarth 시간 과업이 실패한다는 증거가 아니다. 현재 문서도 이 일반화를 금지한다.

주의할 provenance 문제도 있다. 문서는 합성 month를 caveat로 쓰지만 현재 로컬 추출 코드는 contract의 실제 timestamp를 읽는다. 캐시는 shape만 맞으면 재사용할 수 있다. 현재 코드만 보고 이미 생성된 캐시에 어느 timestamp가 쓰였는지는 확정할 수 없다. 실행 당시 immutable snapshot과 cache manifest를 먼저 대조해야 한다.

새 cache key에는 checkpoint/revision, 정확한 timestamp, 관측 선택, crop/좌표/CRS, band 정규화, cloud mask 계약을 포함한다.

### 2.7 L pilot: projector-only 국지화 실패이지 언어 연구의 불가능 판정은 아니다

로컬 JSON에서 확인되는 것은 7지역 2,072문항 학습, 2지역 768문항 test, projector-only 1 seed다. Q1 balanced accuracy .6068, 양성 recall .2604, FPR .0469다. Q3 Jaccard .2778, exact .0885다. 기존 등록 규칙상 L1 불통과는 유지한다.

그러나 비교는 조심해야 한다. 학습한 projector와 EO 토큰을 제거한 zero-shot 텍스트 LLM은 학습 예산이 맞지 않는다. TEOChat .581과의 약 .026 차이도 동일 문항 ID의 paired 평가·불확실성이 확인되지 않아 일반 우월성을 주장할 수 없다.

코드는 원래 32×32 토큰 지도에 pooling kernel 4를 적용해 **8×8=64 토큰**을 만든다. 4×4 토큰 지도가 아니다. 고정 순서와 LLM positional encoding은 남지만 명시적인 XY/region grounding 설계나 좌표 변환 학습은 확인되지 않는다. 작은 지역 국지화가 왜 실패했는지 pooling·좌표·학습량·LLM 적응을 분리해야 한다.

Q4의 관측 장수는 입력 텍스트에 제시되어 text-only도 1.0이다. 이것을 영상 기반 시간 추론 능력으로 점수화하면 안 된다. LoRA를 학습하지 않은 이 pilot로 “EO→LLM은 안 된다” 또는 “L은 반드시 workshop까지만”이라고 결론낼 수 없다.

## 3. 가장 직접적인 선행연구와 우리에게 남는 질문

아래 순서대로 읽는 것이 55편을 무작정 읽는 것보다 중요하다. 세부 서지와 링크는 §8에 있다.

| 먼저 볼 논문 | 이미 해결하려는 문제 | 우리가 구분해야 하는 부분 |
|---|---|---|
| SPEAR NeXT [08] | frozen EO latent + 지역 모델 + causal multi-horizon 예측 | 월별 정렬·완전 coverage에서 불규칙 arrival, 품질 교정, 사건 지연으로 넘어갈 때 실제 기여가 있는가 |
| LIANet [09] | 특정 지역의 공간 신경 표현과 시간 조건부 표현 | 지역 특화 자체는 새롭지 않다. acquisition별 시간 latent와 unseen-time/온라인 교정의 차이를 확인해야 한다 |
| AlphaEarth [03] | 시공간 EO 표현과 시간 조건부 source 복원 | 날짜 조건부 latent 자체를 최초라고 못 한다. 공개 annual product와 full model 접근성은 다르다 |
| AnytimeFormer [15] / AGFlow [16] | 비동기 SAR·optical로 임의 날짜 optical 복원 | frozen latent 공간이라는 이유만으로 문제 차이가 충분하지 않다. 입력·목표·causal 계약·효용을 비교해야 한다 |
| S-CCD [25] / RBC [26] | 관측이 늘어날 때 재귀 상태/분류와 변화 확인 | update와 조기 변화 탐지라는 개념은 오래됐다. EO FM 상태에서의 확장성과 우월성 증거가 필요하다 |
| LAINR [30] / KalmanNet [31] | latent assimilation·학습 관측 교정 | predict/correct를 새 블록 이름으로만 포장하면 신규성이 약하다 |
| U-TILISE [14] / UnCRtainTS [20] | 시계열 gap filling·cloud uncertainty | 선형 보간이 강한 데이터도 있다. 데이터별 난도와 품질 교정을 같이 진단해야 한다 |
| TEOChat [46] / EarthDial [47] | EO 영상/시계열을 projector와 LLM에 정렬 | 임베딩→언어 학습은 가능하다. observed/inferred 상태와 주장의 근거 연결이 추가 질문이다 |

LIANet은 CVPR 2026 **EarthVision workshop**이며 main이 아니다. AGFlow도 CVPR 2026 **MORSE workshop**이다. SA-STF는 CVPR 2026 **main**이다. workshop 논문도 기술적 선행연구이며 무시할 수 없다.

LIANet 본문에는 acquisition timestamp마다 학습한 시간 latent가 나온다. 제목의 “continuous”만으로 실제 미관측 날짜를 엄격히 hold-out 복원하거나, 새로운 미래 관측에 온라인으로 적응하는 능력이 입증되었다고 인용하지 않는다.

AGFlow의 flow-matching 경로 시간은 생성 모델의 내부 시간이다. 물리적인 지구 시간과 혼동하면 안 된다. 임의 날짜 생성 결과에는 항상 정합된 실제 optical GT가 있는 것도 아니다. dense output의 존재와 시간 해상도 검증은 별개다.

SPEAR NeXT는 공개된 지 며칠 안 된 preprint다. 결과를 독립 재현하지 않았지만, “캐시한 FM에 지역별 causal predictor를 얹는 것은 아직 아무도 안 했다”는 주장에 직접 반례이므로 지금 읽어야 한다.

## 4. 내가 권하는 연구 설계: 보간기보다 ‘관측 교정형 지역 상태’

### 4.1 무엇을 상태라고 부를 것인가

한 장의 embedding에는 토지피복 같은 느린 성분과 구름·센서·계절 같은 성분이 섞여 있다. 단일 관측 embedding을 그대로 ‘진짜 지구 상태’라고 부르지 않는다.

작업 가설로 지역의 느린 공간 성분과 시간에 따라 변하는 성분을 분리해 볼 수 있다. 하지만 분리가 식별된다는 보장은 없으므로 보조 물리량, 독립 change/readout, 재구성 목표와 ablation으로 확인해야 한다. 핵심은 단순히 cosine이 높은 latent를 만드는 게 아니라 **새 관측을 읽어 실제 변화를 확인하는 데 쓸 수 있는 상태**다.

### 4.2 최소 모델: 어디에 conditioning이 들어가는가

아래는 새 방법의 확정 명세가 아니라 기존 RNN/Kalman/assimilation과 비교할 후보 구조다.

```text
새 관측 I_i + 실제 시각 t_i + 센서 m_i + 품질 q_i
    → frozen EO encoder → 관측 embedding e_i

이전 교정 상태 h⁺_(i−1)
    → 시간 경과 Δt, 계절, 지역 코드, 이용 가능한 외생 변수로 전개
    → 관측 전 상태 h⁻_i 및 불확실성

e_i와 h⁻_i의 예상 관측 차이 + q_i
    → 관측 교정 모듈
    → 새 상태 h⁺_i 및 불확실성

질의 날짜/좌표
    → change/readout + 추정 상태
    → 관측됨 / 보간됨 / 예측됨 표시와 근거 관측 목록
```

사용자의 “conditioning이 부족하다”는 지적은 타당한 가설이다. 최소한 실수 단위 Δt, day-of-year, 위치/지역, 센서·band·GSD, cloud/quality, 관측 여부를 구분해야 한다. 지형이나 기상·수문 입력은 타깃에 필요한 경우 추가한다. 미래 기상 재분석을 사용하면 실시간 예측에서는 oracle 조건이라는 점을 분리한다.

관측과 상태를 연결하는 H_m이 필요하다면 센서별로 둔다. S1/S2 embedding을 무조건 동일한 물리 상태로 정렬했다고 가정하지 않는다. 우선 S2-only로 주장을 검증하고 S1은 정합과 actual availability가 확보된 독립 확장으로 다루는 편이 안전하다.

### 4.3 왜 무조건 부드러운 미분방정식이면 안 되는가

식생 성장이나 수면 온도처럼 일정 조건에서 연속적으로 변하는 양에는 continuous-time 모델이 유용하다. 그러나 벌목·산불·산사태·공사 시작은 외부 사건이다. 사건 입력이 없으면 앞뒤 두 영상에서 정확한 발생 시간을 하나로 식별할 수 없다.

시간 smoothness를 강하게 주면 실제 급변을 ‘부드러운 변화’로 지워 경보를 늦출 수 있다. 그래서 정상 구간의 smooth dynamics와 observation innovation의 급변을 구분하는 후보가 필요하다. jump gate는 한 가지 가설일 뿐이며, plain Δt-GRU도 같은 현상을 충분히 처리할 수 있다면 복잡한 모듈을 정당화할 수 없다.

GraphCast/NeuralGCM/Aurora는 실제 물리 변수와 강한 외생 입력·학습 자료를 사용한다. 이를 근거로 “EO 영상도 PDE만 넣으면 72시간을 매시간 정확히 복원한다”고 말할 수 없다. 센서 관측은 상태뿐 아니라 대기·광학·관측 기하의 함수다.

물리 제약을 넣으려면 타깃을 먼저 정한다. 예를 들어 흐르는 구름/물질에는 수송, 수면·온도에는 관련 보존/forcing을 검토할 수 있다. 정체가 불명확한 768차원 전체 embedding에 보존 법칙을 선언하지 않는다. 시뮬레이션 실험은 모델 작동 검증으로는 유용하지만 실제 EO 시간 해상도 향상의 증거를 대체하지 못한다.

### 4.4 GNN/equivariance의 역할과 한계

GNN은 불규칙 공간이나 다양한 해상도의 노드를 다루는 후보다. 그러나 graph의 permutation equivariance가 곧 지리적 20m translation equivariance는 아니다. 상대 좌표, 이웃 정의, sampling, 경계/crop 처리까지 일관되어야 한다.

40m 토큰의 격자를 20m 옮기면 반 토큰 위상 변화가 생긴다. 좌표 조건부 decoder나 연속 공간 표현은 query consistency를 돕지만, 기존 patch encoding에서 잃은 세부 정보를 자동 복원하지 않는다. 지리적으로 다른 지역으로 이동하는 것도 동일 장면의 좌표 이동과 다르다.

공간 이동이 논문의 주장이면 LEPA 및 INR류를 직접 비교해야 한다. 지금은 지역 상태와 사건 교정이 본체이므로 공간 consistency는 ablation으로 제한하는 것을 권한다. GNN·ODE·VLM을 모두 붙이는 것은 검증 질문을 흐릴 가능성이 높다.

## 5. ‘시간 해상도를 늘렸다’고 말하기 전에 필요한 데이터 계약

### 5.1 좁은 지역에서 시작하는 것은 좋다. 한 지역 하나의 성공으로 끝내면 약하다

특정 지역의 과거를 충분히 사용하면 지역 계절성·토양·관리 양식을 학습하기 쉽다. 이는 의도한 transductive/local deployment setting으로 정의할 수 있다. test 대상의 과거를 사용했다는 사실을 숨기지 않는다.

첫 실험은 한 AOI와 하나의 주 타깃으로 좁힌다. 환경 변화라면 관측으로 확인 가능한 산림 제거/수역 변화 등을 정한다. 우선 정상 계절 변화와 급변 구간을 함께 평가한다. 성공 후 다른 지역·다른 기간에서 재검증한다. 예를 들어 3지역·2현상은 제안되는 출발 범위이지 CVPR 채택을 보장하는 마법의 개수는 아니다.

기존 제주 자산에 사람 라벨이 없다고 전 연구를 포기할 필요는 없다. 공개 변화 데이터셋에서 프로토콜을 먼저 검증하고, 제주에는 별도 지역 적응·검증 표본을 만드는 선택지가 있다. 반대로 검증 없이 제주 결과에 ‘환경파괴’를 붙이는 것은 안 된다.

### 5.2 먼저 기존 cache가 아니라 실제 짧은 관측 간격을 세어야 한다

신규 모델 학습 전 AOI별 전체 실제 관측을 이용해 clear gap ≤3/7/14일, 계절별 결측, 사건 전후 usable observation 수를 센다. top-12 annual cache만으로 short-gap 가능성을 판단하지 않는다. 시간 단위 표현이면 `.total_seconds()` 기반 실제 시간 계약이 필요하다.

다음 조건을 모두 만족하는지 확인한다.

- acquisition 시각과 시스템에 들어온 arrival 시각을 분리할 수 있는가?
- 고품질 held-out 관측이 이웃 사이에 충분히 존재하는가?
- 사건이 없는 기간도 충분한가? 오경보를 잴 수 있는가?
- 가린 날짜의 영상이 encoder, cloud composite, teacher 입력에 간접적으로 들어가지 않는가?
- crop 정합과 quality mask가 모델별로 같은가?
- 시간 간격과 사건/지역을 섞지 않고 비교할 표본 수가 있는가?

### 5.3 데이터셋의 ‘daily’도 provenance를 봐야 한다

DynamicEarthNet은 CVPR main의 좋은 지역 시계열·의미 변화 데이터다. 하지만 본문 §3.1은 Planet Fusion이 날씨 occlusion을 제거하고 가까운 시점으로 **gap-fill**한다고 명시한다. daily product를 매일 독립 촬영한 깨끗한 ground truth와 동일시하면 안 된다. 라벨도 월별이다. 실시간 arrival나 정확한 사건 날짜 실험에는 별도 provenance 검토가 필요하다.

PASTIS-R은 실제 비동기 S1/S2와 시계열 모델 비교에 적합하지만 crop 라벨이 모든 날짜의 사건 정답은 아니다. EarthNet2021/GreenEarthNet은 기상 조건부 미래 vegetation/영상 예측에 유용하나 환경파괴 onset의 정답으로 바로 쓰지는 않는다. MultiEarth는 deforestation과 multimodal monitoring에 더 직접적이다. 데이터별로 푸는 질문을 분리한다.

### 5.4 hourly output와 hourly 검증은 다르다

Sentinel-3는 센서별 공간 해상도·재방문 조건이 다른 임무다. 3일 관측을 매시간 출력한다고 시간마다 새로운 관측 증거가 생기는 것은 아니다. Sentinel-3의 관측 정보만으로 작은 40m 객체의 실제 hourly 변화를 검증하는 주장은 특히 어렵다. [ESA Sentinel-3 사실 자료](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-3/Facts_and_figures).

GK2A는 전구 10분, 한반도 2분 관측이며 채널별 공간 해상도는 대략 0.5–2km 범위다. 시간 단위 입력을 주는 데는 유용하지만 작은 벌목/공사 변화의 ground truth를 대체하지 않는다. [기상청 관측 주기 설명](https://www.kma.go.kr/wnuri_help/html/image/gk2a.jsp), [기상청 국가기상위성센터 소개](https://nmsc.kma.go.kr/homepage/html/special/specialIntro.do?lang=ko).

처음에는 실제 검증 가능한 수일 간격을 잡고, hourly는 타깃과 독립 검증 자료가 확보된 별도 확장으로 둔다. 관측이 없는 날짜에는 uncertainty와 ‘추정’ 표시를 제공하는 것이 정직한 temporal densification이다.

## 6. CVPR 기여를 판별할 실험: 새 프로토콜 초안

이 절은 제안이며 prereg 등록·학습·라벨 개봉을 하지 않았다. 기존 T4/L1을 통과시키기 위한 사후 변경이 아니다.

### 6.1 분할과 누출 차단

지역 내 train=과거, validation=그 다음 기간, test=미사용 미래 기간으로 분할한다. 지역 적응은 train 기간만 사용한다. 시계열 창과 사건이 경계를 넘지 않도록 purge한다. 인접 타일·같은 사건을 무작위로 train/test에 나누지 않는다.

test에서는 가중치를 고정하고 시점마다 과거에 도착한 관측으로 상태만 업데이트한다. label-free test-time adaptation을 추가한다면 별도 arm으로 두고, 해당 관측을 보기 전 예측을 먼저 기록한다. 새 라벨로 업데이트한 뒤 같은 라벨을 평가하는 누출은 금지한다.

causal arm에서는 전체 연도를 보고 가장 맑은 12장을 미리 골라 주면 안 된다. 미래 관측 도착 여부, 품질, 기상도 미리 알면 안 된다. 뒤늦게 오는 관측의 backfill도 비용·정책을 기록한다. acquisition order와 arrival order가 다른 경우를 별도로 다룬다.

offline interpolation arm에는 미래 이웃을 허용하되 다른 표로 보고한다. 가린 실제 관측은 **모든 입력 생성 과정에서 먼저 제거**한다. full-window teacher에 target 영상이 들어간 경우에는 강한 distillation target이라고 명시하고 독립 라벨 평가를 추가한다. 이를 causal physical GT로 부르지 않는다.

### 6.2 반드시 넣을 기준선

| 평가 계약 | 최소 기준선 | 왜 필요한가 |
|---|---|---|
| 과거 보간 | nearest, 날짜 linear, mean-all, harmonic/seasonal model, GP | 정적 성분과 계절성만으로 점수가 높을 수 있다 |
| 지역 보간 | 동일 입력의 지역 MLP/INR, 적용 가능한 LIANet/AnytimeFormer/U-TILISE/AGFlow | latent라는 이름만으로 기존 pixel 복원 문제와 분리되지 않는다 |
| 순차 상태 | last observation, EMA, exact running mean + matched readout | 기존 기억이 아니라 최신 영상만 읽어도 되는지 확인한다 |
| 인과 예측/교정 | Δt-GRU, ODE-RNN 또는 Neural CDE, Kalman류, SPEAR-style causal predictor | 새로운 교정 블록이 일반 불규칙 시계열 모델을 넘는가 |
| 변화 경보 | seasonal residual/BFAST/S-CCD, recursive classifier, latest-only FM head | 고전 감시 방법과 단순 FM head보다 실제 경보가 나아야 한다 |
| 비용·표현 reference | 도착 prefix 전체를 OlmoEarth로 재인코딩 | 효율/호환성 기준이다. 물리적 정답이나 절대 상한은 아니다 |

각 방법이 요구하는 입력 modality와 학습 자료가 다르면 정면 수치 비교를 강요하지 않는다. shared-input 표와 richer-input 표를 나눈다. EO readout은 표현별로 같은 train label·같은 학습 예산을 주는 arm도 필요하다. 기존 decoder에만 맞는 latent를 평가하면 representation compatibility만 측정할 수 있다.

full-prefix encoder가 test에서도 미래를 쓰지 않게 하고, 기존 평균/GRU cache 자산과 동일한 계약으로 비교한다. frozen encoder와 LoRA 차이는 필요하면 독립 2×2로 추가하지만, 데이터·기준선이 확보되기 전에 학습 예산을 확장하지 않는다.

### 6.3 평가 지표: embedding cosine만으로 끝내지 않는다

1. 표현 복원: cosine 및 norm/MSE를 함께 보고, static baseline 대비 동적 residual·변화 구간 성능을 별도 산출한다. residual 정의와 정규화는 train에서 고정한다.
2. 관측량 복원: held-out clean pixel/NDVI 등 타깃에 적합한 독립 관측량의 오차. 광학 품질·정합 caveat를 기록한다. embedding에서 직접 계산할 수 없는 값은 독립 decoder가 필요하다.
3. 실제 변화: 위치 IoU/F1, 사건별 확인 성공률, 고정 false alarms/km²/time 또는 정해진 precision에서의 달력 지연. 변화 없는 긴 구간을 포함한다.
4. 불확실성: 예측 분포가 있으면 proper score/CRPS, coverage, calibration, risk–coverage. cosine 높음만으로 신뢰도를 대체하지 않는다.
5. 비용: batch=1 arrival workflow의 새 관측 encoding+상태 update+I/O, P50/P95 latency, 메모리·저장량·backfill 비용. 일괄 처리 wall time이나 논리 I/O 배수만으로 실시간 비용을 주장하지 않는다.

ground truth 사건 날짜를 정확히 모르면 “마지막 정상 관측~첫 변화 관측”의 구간으로 보고한다. 미탐지는 빼지 말고 censored/실패로 함께 보고한다. 수확·계절 변화는 실제 사건 경보의 hard negative다.

CI는 가능한 한 지역/사건/공간 block 단위의 paired 분석으로 잡고 per-region 결과도 공개한다. 적은 지역을 수천 독립 표본처럼 취급하지 않는다. hyperparameter와 실질적 개선 기준은 validation에서 고정하고 test를 반복해 문턱을 고르지 않는다.

### 6.4 어떤 결과면 논문을 밀고, 어떤 결과면 방향을 바꿀까

- 강한 기준선보다 동등한 오경보에서 실제 사건을 빨리 확인하며 비용도 줄이면, 상태 갱신/관측 교정 방법 주장을 검토한다. 모든 축을 이길 필요는 없지만 분명한 accuracy–latency–cost trade-off를 보여야 한다.
- 정상 계절 보간만 좋아지고 사건을 smear하면, 환경 변화 감시 주장으로 확장하지 않는다.
- latest-only FM head와 동일하다면, 긴 기억·교정 모듈의 기여가 약하다. 상태 예측과 운영 효율을 분리해 판단한다.
- cosine만 오르고 독립 readout·관측량·사건 성능이 안 오르면, latent geometry 결과이지 지구 시간 해상도 개선 결과가 아니다.
- 새 데이터/프로토콜이 여러 FM의 순위나 실패 유형을 바꾸고 실제 운영 의미를 드러내면 benchmark 논문도 가능하다. 단순 cosine 표와 기존 ablation의 묶음만으로 충분하다고 보지는 않는다.
- 개선이 없으면 지역/시간/품질별 반례를 남긴다. 같은 test를 계속 만져 유의성을 찾는 방식으로 진행하지 않는다.

CVPR에서의 최종 신규성과 강도는 결과 및 추가 문헌 정독에 달려 있다. 이 프로토콜 자체, Kalman식 innovation, continuous-time 모듈 중 하나만으로 새로운 방법이 보장되지 않는다.

### 6.5 실행 순서: GPU보다 데이터 계약이 먼저다

**P0 — 지금 해야 할 일:** 현재 cache의 timestamp/checkpoint provenance를 인증하고, 후보 AOI의 전체 관측 gap·사건/음성 기간·실제 정답 가용성을 조사한다. SPEAR NeXT, LIANet, AnytimeFormer, S-CCD, LAINR을 우선 정독한다.

**P1 — 첫 독립 실험:** 한 지역 chronological split에서 linear/mean/harmonic/last/Δt-GRU를 같은 input으로 비교한다. offline interpolation과 causal update를 분리한다. 기존 T4 반복이나 language LoRA부터 시작하지 않는다.

**P2 — 방법 후보:** 품질·실제 Δt conditioning과 observation corrector를 각각 추가해 효과를 분리한다. uncertainty와 abrupt-change preservation을 함께 측정한다. 충분한 validation 근거가 없으면 ODE/GNN을 추가하지 않는다.

**P3 — 외부 확인:** 미사용 미래 기간/다른 지역/다른 사건에서 frozen readout과 matched-readout 평가를 한다. 실제 arrival 비용을 잰다.

**P4 — 언어 확장:** 확인된 상태·변화·근거를 언어로 연결한다. T와 L을 동시에 대형 학습으로 시작하지 않는다.

현재 frozen cache와 작은 모듈은 두 GPU 환경에서 시작하기 유리하다. 하지만 dense extraction, imagery access, AOI 크기, 품질 검수, 라벨 및 저자 모델 접근성에 따라 비용이 달라져 10주·2 GPU로 완성을 보장할 수 없다. 이번 조사에서 위 단계를 실행하지 않았다.

## 7. 임베딩→언어/VLM: 가능하다. ‘표를 읽는 길만 있다’는 말은 틀리다

### 7.1 두 가지 정당한 경로

경로 A는 실용적이다: EO readout → 좌표/시기/변화량/품질/신뢰도/근거 영상의 구조화 기록 → LLM 질의·설명/RAG. 정량 주장과 근거를 통제하기 쉽다.

경로 B도 학습할 수 있다: spatial/temporal EO tokens + region/time metadata → learned projector 또는 cross-attention → LLM/VLM instruction tuning. TEOChat/EarthDial과 robotics의 OpenVLA가 encoder 출력을 언어 모델 입력에 정렬하는 직접적인 참고다. OlmoEarth embedding이 언어에 사전 정렬되지 않았다고 변환 학습까지 불가능한 것은 아니다.

다만 vector 숫자를 텍스트로 나열하면 바로 의미 있는 언어가 되는 것은 아니다. 좌표·시간·지역 수준을 보존하고 학습 데이터로 개념/주장을 접지해야 한다. metadata-only 모델도 강할 수 있어 visual latent의 추가 기여를 검증해야 한다.

### 7.2 robotics에서 가져올 것과 가져오면 안 되는 것

가져올 것은 structured tokens, multimodal alignment, memory, evidence-conditioned answer, uncertainty/abstention이다. EO의 ‘행동’은 관측 요청·재방문·추가 데이터 조회처럼 정의할 수 있다.

하지만 로봇의 명령 가능한 행동과 그 결과 데이터가 EO passive monitoring에 자동으로 존재하지 않는다. 단순 전후 영상 captioner를 Earth world model 또는 action model이라고 부르는 것은 과장이다. 관측 선택 policy를 주장하려면 실제 availability·비용과 policy 평가가 추가로 필요하다.

### 7.3 ‘환경파괴’ 대신 관측 가능한 주장부터 학습한다

“어디가 환경파괴가 심해졌는가”를 먼저 다음과 같이 분해한다.

```text
지역 / 좌표 / 면적
관측 가능한 변화 유형: 산림 감소, 수역 축소, 토양 노출 등
기간: 관측으로 확인되는 시작 구간과 지속 여부
증거: 실제 전후 관측 및 독립 기록
신뢰도와 대안 설명: 수확, 계절, 구름, 정합 등
정책/가치 해석: 별도 정의와 출처에 근거한 판단
```

관측되지 않은 날짜의 latent를 언어로 설명할 때는 observed / interpolated / forecast 상태를 반드시 구분해야 한다. “추정상 변화 가능성”을 “그날 벌목이 발생했다”로 바꾸지 않는다. 공공기록에 일치 항목이 없다고 불법이나 무허가라고 결론내리지도 않는다. 데이터 coverage 문제일 수 있다.

### 7.4 연구로 만들려면 무엇을 비교해야 하나

- 변화 score+template, readout+구조화 기록, 학습한 metadata-only LLM, EO projector-only, projector+XY/time, 적절한 LoRA, 원 영상 TEOChat/EarthDial 등을 같은 질문·예산 계약으로 비교한다.
- embedding/time/location을 바꾸거나 순서를 교환하는 intervention으로 실제 영상 증거를 읽는지 확인한다.
- claim precision/recall, 지역·시기·면적 grounding, evidence consistency, abstention, inferred-state hallucination을 독립 GT/검수로 평가한다.
- 환경 보고서는 사람/독립 자료의 gold 평가가 필요하다. LLM-as-judge만 사용하면 자기 설명을 자기 기준으로 평가하는 순환 위험이 있다.

그래서 L이 무조건 제품이라는 판단에는 동의하지 않는다. 다만 현재 generic projector+RAG의 조합은 신규성이 약하다. ‘관측과 추정을 구분해 시간·공간·근거에 접지된 주장을 한다’는 특정 난제를 해결하고 strong baseline을 넘어야 연구가 된다. 기존 L pilot의 실패를 유지하되 별도 연구 가능성을 배제하지 않는다.

## 8. 55편의 1차 자료 읽기 목록

표기: **S** = 이 조사에서 관련 method/result/limitation 본문 일부를 확인. **M** = 공식 서지·초록/프로젝트 설명 확인. S도 전체 정독·독립 재현을 뜻하지 않는다. ‘preprint’는 확인한 공식 자료에서 채택 venue를 확정하지 않았거나 공개 preprint로만 사용한다는 뜻이다. 온라인 공개 연도와 journal issue 연도가 다르면 따로 적었다.

### A. EO foundation 표현·시간·지역·메타정보 [01–13]

[01] **OlmoEarth: Stable Latent Image Modeling for Multimodal Earth Observation** — CVPR 2026 main. S(로컬 시간 부호화 코드 포함). [공식 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Herzog_OlmoEarth_Stable_Latent_Image_Modeling_for_Multimodal_Earth_Observation_CVPR_2026_paper.pdf), [arXiv](https://arxiv.org/abs/2511.13655). 현재 encoder의 pretraining/표현 계약이 출발점이다. single-observation, pooled-window, causal state를 같은 대상으로 취급하지 않는다.

[02] **OlmoEarth v1.2: A more efficient family of OlmoEarth models** — 2026 preprint/공식 업데이트. S(§2.3, §2.5). [보고서](https://arxiv.org/abs/2605.20804), [본문](https://arxiv.org/html/2605.20804v3), [Ai2 업데이트](https://allenai.org/blog/olmoearth-v1-1). frame masking과 mixed 3D RoPE를 확인한다. month 기반 표현의 한계를 논할 때 v1.2까지 무시하면 안 된다. 정확한 일/시간 간격 처리와 온라인 교정은 별도 검증이다.

[03] **AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data** — 2025 preprint. S(시간 조건부 source decoder 관련 §S16). [논문](https://arxiv.org/abs/2507.22291), [본문](https://arxiv.org/html/2507.22291v2). 시간 조건부 표현·관측 복원·텍스트 정렬이 이미 존재한다. 공개 annual embedding product가 full model이나 arbitrary-date 공개 inference API를 뜻하지 않는다.

[04] **TESSERA: Temporal Embeddings of Surface Spectra for Earth Representation and Analysis** — CVPR 2026 main(저자 publication 기록). M. [논문](https://arxiv.org/abs/2506.20380), [저자 기록](https://anil.recoil.org/papers/2025-tessera). S1/S2 temporal representation과 observation subsampling을 참고한다. annual field와 online dynamic state는 다르다.

[05] **TESSERA v2: Scaling Pixel-wise Earth Foundation Models** — 2026 preprint. M. [논문](https://arxiv.org/abs/2607.03949). 표현 학습 손실과 downstream 성능의 연결이 약할 수 있음을 확인해야 한다. 단순 latent metric 개선을 EO task 개선으로 간주하지 말라는 참고이며 현재 실험의 상한 증명은 아니다.

[06] **Temporal Sensitivity Analysis of Tessera Embeddings** — 2026-08 preprint. M. [논문](https://arxiv.org/abs/2608.27175). frozen Tessera의 입력 시간창을 줄였을 때 과업별 민감도가 다르다. 더 많은 날짜가 항상 같은 이득을 준다거나 어느 한 과업의 무해한 ablation이 모든 시간 연구를 배제한다는 주장을 피한다. 여러 FM 전체를 비교한 논문으로 잘못 인용하지 않는다.

[07] **TerraFlow: Multimodal, Multitemporal Representation Learning for Earth Observation** — 2026 preprint. S(temporal masking/forward-only 관련 부분). [논문](https://arxiv.org/abs/2603.12762), [본문](https://arxiv.org/html/2603.12762v1). temporal representation과 pre-event/causal protocol 설계에 가깝다. 특정 forward-only 실험의 실패를 causal EO 전반의 불가능으로 인용하지 않는다.

[08] **SPEAR NeXT Causal Latent Forecasting Across Multiple Horizons for Spectral Temporal Earth Representation Learning** — 2026-09-15 preprint. S(§3, §4.1, §5.2). [논문](https://arxiv.org/abs/2609.16871), [본문](https://arxiv.org/html/2609.16871v1). cached frozen SPEAR latent 위 causal transformer와 지역별 multi-horizon 예측을 다루는 가장 직접적인 신작이다. monthly aligned/complete sequences와 불규칙 cloud-aware arrival update의 차이를 검증해야 한다.

[09] **Location Is All You Need: Continuous Spatiotemporal Neural Representations of Earth Observation Data (LIANet)** — CVPR 2026 EarthVision workshop. S(§1, §3). [논문](https://arxiv.org/abs/2604.07092), [본문](https://arxiv.org/html/2604.07092), [공식 PDF](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/papers/Madadikhaljan_Location_Is_All_You_Need_Continuous_Spatiotemporal_Neural_Representations_of_CVPRW_2026_paper.pdf). 특정 지역·spatial hash encoding·acquisition별 temporal latent가 직접 경쟁점이다. ‘지역 특화’나 ‘latent vs pixel’만으로 차별화하지 않는다.

[10] **Paving the way toward foundation models for irregular and unaligned Satellite Image Time Series (ALISE)** — 2024 preprint. M. [논문](https://arxiv.org/abs/2407.08448). 불규칙/비정렬 시계열을 query projection으로 정렬하는 참고다. 관측 집합을 다루는 방법과 새 관측마다 고정 비용으로 상태를 교정하는 방법을 구분한다.

[11] **Atomizer: Generalizing to new modalities by breaking satellite images down to a set of scalars** — 2025 preprint. M. [논문](https://arxiv.org/abs/2506.13542). scalar measurement에 시간·GSD·spectral metadata를 결합하는 flexible EO 입력 표현이다. ‘메타정보를 극대화’하는 주장의 선행연구로 중요하다.

[12] **LEPA: Learning Geometric Equivariance in Satellite Remote Sensing Data with a Predictive Architecture** — 2026 preprint. M. [논문](https://arxiv.org/abs/2603.07246), [저자 코드](https://github.com/embed2scale/LEPA). EO embedding의 geometric transformation을 학습하는 가까운 prior다. 20m shift/cache 공간 변환이 본체라면 반드시 직접 검토한다.

[13] **Towards a Unified Copernicus Foundation Model for Earth Vision (Copernicus-FM)** — ICCV 2025 main. M. [논문](https://arxiv.org/abs/2503.11849), [공식 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Towards_a_Unified_Copernicus_Foundation_Model_for_Earth_Vision_ICCV_2025_paper.pdf). Sentinel 계열과 flexible spectral/geotemporal metadata conditioning이 선행한다. S3까지 고려할 때 relevant input baseline이다.

### B. Cloud removal·임의 날짜 복원·시공간 융합 [14–21]

[14] **U-TILISE: A Sequence-to-Sequence Model for Cloud Removal in Optical Satellite Time Series** — IEEE TGRS 61, 2023. S(results와 SEN12MS-CR-TS 관련 부분). [논문](https://arxiv.org/abs/2305.13277), [DOI](https://doi.org/10.1109/TGRS.2023.3333391), [본문](https://arxiv.org/html/2305.13277), [ETH 코드](https://github.com/prs-eth/U-TILISE). 학습 gap filling이 유용하지만 SEN12MS-CR-TS에서는 linear도 강하다. 한 데이터셋의 약한 gain을 전 지구 시간 보간 상한으로 해석하면 안 되는 직접적인 참고다.

[15] **AnytimeFormer: Fusing irregular and asynchronous SAR-optical time series to reconstruct reflectance at any given time** — Remote Sensing of Environment 333, 115120, 2026. M(공식 서지·저자 README; publisher 본문 접근 실패). [저널](https://www.sciencedirect.com/science/article/pii/S0034425725005243), [DOI](https://doi.org/10.1016/j.rse.2025.115120), [저자 코드](https://github.com/tangkai-RS/AnytimeFormer). 이름 그대로 임의 날짜 복원을 다루는 direct prior다. methods/results 전체 정독을 다음 우선순위로 남긴다.

[16] **Asynchronous Remote Sensing Time-Series Fusion for Cloud Removal and Anytime Reconstruction (AGFlow)** — CVPR 2026 MORSE workshop. S(methods/results/GT 제한). [논문](https://arxiv.org/abs/2605.27726), [본문](https://arxiv.org/html/2605.27726), [공식 PDF](https://openaccess.thecvf.com/content/CVPR2026W/MORSE/papers/Fallah_Asynchronous_Remote_Sensing_Time-Series_Fusion_for_Cloud_Removal_and_Anytime_CVPRW_2026_paper.pdf). query date와 비동기 S1/S2 flow matching 복원. arbitrary-date visualization과 정합된 실제 관측 평가를 분리해야 한다.

[17] **RESTORE-DiT: Reliable satellite image time series reconstruction by multimodal sequential diffusion transformer** — Remote Sensing of Environment 328, 114872, 2025. M. [저널](https://www.sciencedirect.com/science/article/pii/S0034425725002767), [DOI](https://doi.org/10.1016/j.rse.2025.114872), [저자 코드](https://github.com/SQD1/RESTORE-DiT). date-aware SAR/optical diffusion restoration을 다룬다. SAR를 더 넣은 복원기를 S2-only latent와 input 통제 없이 비교하지 않는다.

[18] **Semantic-Adaptive Diffusion for Dynamic Spatiotemporal Fusion (SA-STF)** — CVPR 2026 main. M(공식 초록). [공식 논문](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Semantic-Adaptive_Diffusion_for_Dynamic_Spatiotemporal_Fusion_CVPR_2026_paper.html). 시간에 촘촘한 coarse와 sparse fine을 결합하는 최신 main-conference prior다. 추가 센서 정보를 사용한 복원과 없는 정보를 상상하는 작업을 혼동하지 않는다.

[19] **Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series (MISR-S2)** — CVPR 2024 EarthVision workshop. M. [논문](https://arxiv.org/abs/2404.16409), [공식 페이지](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/html/Okabayashi_Cross-sensor_super-resolution_of_irregularly_sampled_Sentinel-2_time_series_CVPRW_2024_paper.html). irregular-time encoding과 fine-sensor supervision을 참고한다. spatial super-resolution과 hourly temporal truth는 다른 문제다.

[20] **UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series** — CVPR 2023 EarthVision workshop. M. [논문](https://arxiv.org/abs/2304.05464), [공식 PDF](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Ebel_UnCRtainTS_Uncertainty_Quantification_for_Cloud_Removal_in_Optical_Satellite_Time_CVPRW_2023_paper.pdf). cloud-aware aleatoric uncertainty가 이미 있다. uncertainty를 붙였다는 이유만으로 신규성을 주장하지 않는다.

[21] **HLS-GPT: A Generative Pretrained Transformer (GPT) for Continental-Scale NASA Harmonized Landsat and Sentinel-2 (HLS) Reflectance Reconstruction Across All Bands on Arbitrary Dates** — 2026-06 preprint; 저널 기록은 ISPRS JPRS 240의 2026-10 issue. M. [논문](https://arxiv.org/abs/2606.18115), [저널 기록](https://www.sciencedirect.com/science/article/pii/S0924271626003953). HLS multiband pixel 시계열의 arbitrary-date reconstruction이 direct prior다. 이미 공개된 자료를 인용하되 10월 issue가 현재 지났다고 쓰지 않는다.

### C. 재귀 변화 감시·자료 동화·continuous time·물리 [22–39]

[22] **Continuous Change Detection and Classification of Land Cover Using All Available Landsat Data (CCDC)** — RSE, 2014. M. [저널](https://www.sciencedirect.com/science/article/pii/S0034425714000248), [저자 PDF](https://gerslab.cahnr.uconn.edu/wp-content/uploads/sites/2514/2021/06/ZheZhu_CCDC.pdf). 계절·추세와 break를 계속 관찰한다. “새 관측으로 변화를 확인”하는 framing의 고전 baseline이다.

[23] **Near real-time disturbance detection using satellite image time series (BFAST Monitor)** — RSE 123, 98–108, 2012. M. [저자 PDF](https://bfast.r-forge.r-project.org/Verbesselt%2BZeileis%2BHerold-2012.pdf). 지역의 안정 과거를 모델링하고 순차 disturbance를 감시한다. model novelty 이전에 seasonal residual 탐지를 이기는지 봐야 한다.

[24] **Continuous Monitoring of Land Disturbance Based on Landsat Time Series (COLD)** — RSE 238, 111116, 2020 issue(2019 online). M. [저널](https://www.sciencedirect.com/science/article/pii/S0034425719301002), [저자 PDF](https://gerslab.cahnr.uconn.edu/wp-content/uploads/sites/2514/2021/06/cold_2020_zhu.pdf). disturbance 감시·confirmation delay를 평가할 때 참고한다. online/issue 연도를 구분한다.

[25] **A State Space Model for Continuous Change Detection in Satellite Image Time Series (S-CCD)** — RSE 252, 112167, 2021. M. [저널](https://www.sciencedirect.com/science/article/pii/S003442572030540X), [DOI](https://doi.org/10.1016/j.rse.2020.112167). state-space/Kalman 계열의 recursive update와 변화 감시가 직접 prior다. 새 방법도 실제 오경보와 지연을 함께 비교해야 한다.

[26] **Recursive classification of satellite imaging time-series: An application to land cover mapping** — 2023 공개 preprint 기준. S(recursion와 cost 부분). [논문](https://arxiv.org/abs/2301.01796), [본문](https://arxiv.org/html/2301.01796v4). online class posterior 갱신과 일정한 step 비용을 다룬다. class recursion과 EO latent dynamics는 다르지만 cache update novelty에는 가깝다.

[27] **4DVarNet-SSH: End-to-end learning of variational interpolation schemes for nadir and wide-swath satellite altimetry** — Geoscientific Model Development 16, 2119, 2023. M. [공식 논문](https://gmd.copernicus.org/articles/16/2119/2023/), [preprint](https://arxiv.org/abs/2211.05904). 특정 물리량의 sparse satellite interpolation/variational learning이 가능하다는 직접 사례다. 물리적 상태가 정의된 OSSE와 일반 RGB latent를 구분한다.

[28] **Observation-only learning of neural mapping schemes for gappy satellite-derived ocean colour parameters** — IEEE TGRS 63, 2025. M(공식 arXiv journal reference). [논문](https://arxiv.org/abs/2503.11532), [DOI](https://doi.org/10.1109/TGRS.2025.3624465). gap-free GT 없이 실제 gappy ocean-colour 관측으로 학습하는 접근을 읽는다. noisy teacher를 임의로 ‘무노이즈 상태’로 만드는 것과 다르다.

[29] **Generalization performance of neural mapping schemes for the space-time interpolation of satellite-derived ocean colour datasets** — IEEE JSTARS 18, 2025. M(공식 arXiv journal reference). [논문](https://arxiv.org/abs/2503.11588), [DOI](https://doi.org/10.1109/JSTARS.2025.3622311). 지역·변수 간 mapping generalization과 정규화가 중요하다. 지역 특화 실험을 다른 AOI로 확장할 때 읽을 자료다.

[30] **Latent assimilation with implicit neural representations for unknown dynamics (LAINR)** — Journal of Computational Physics 506, 112953, 2024. M. [논문](https://arxiv.org/abs/2309.09574), [DOI](https://doi.org/10.1016/j.jcp.2024.112953), [저자 코드](https://github.com/zylipku/LAINR). latent dynamics·INR·assimilation·uncertainty의 직접적인 방법 prior다. raw EO FM에 그대로 적용되었다는 뜻은 아니다.

[31] **KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics** — IEEE Transactions on Signal Processing, 2022. M. [논문](https://arxiv.org/abs/2107.10043). 학습한 gain/recursion으로 관측 교정을 한다. innovation gate를 붙이는 수준의 신규성 주장을 경계해야 한다.

[32] **Latent ODEs for Irregularly-Sampled Time Series** — NeurIPS 2019. M. [공식 PDF](https://proceedings.neurips.cc/paper_files/paper/2019/file/42a6845a557bef704ad8ac9cb4461d43-Paper.pdf). 실제 Δt를 통한 continuous latent evolution과 ODE-RNN을 baseline으로 둔다. smooth latent dynamics가 unpredictable disturbance를 자동 식별하지는 않는다.

[33] **Neural Controlled Differential Equations for Irregular Time Series** — NeurIPS 2020. M. [논문](https://arxiv.org/abs/2005.08926). 관측 경로가 latent를 구동하므로 autonomous ODE보다 사용자 문제에 가까울 수 있다. 온라인에서는 future-dependent path interpolation로 누출되지 않도록 계약을 정해야 한다.

[34] **ODE-RSSM: Learning Stochastic Recurrent State Space Model from Irregularly Sampled Data** — AAAI 2023. M. [공식 논문](https://ojs.aaai.org/index.php/AAAI/article/view/26310). 불규칙 입력/출력과 확률 latent dynamics를 함께 다루는 후보 baseline이다. EO 타깃 성능은 별도 검증이다.

[35] **Continuous PDE Dynamics Forecasting with Implicit Neural Representations (DINo)** — ICLR 2023. M. [논문](https://arxiv.org/abs/2209.14855), [저자 코드](https://github.com/mkirchmeyer/DINo). continuous space INR와 latent temporal dynamics를 결합한다. PDE simulation에서 arbitrary grid/time query가 가능하다고 EO hourly truth까지 주장하지 않는다.

[36] **Learning skillful medium-range global weather forecasting (GraphCast)** — Science, 2023. M. [논문](https://arxiv.org/abs/2212.12794), [저자 publication](https://deepmind.google/research/publications/22598/). graph 기반 물리 변수 예측의 대표 참고다. GNN 채택만으로 EO patch translation이나 생태 사건 예측이 해결되지 않는다.

[37] **Neural general circulation models for weather and climate (NeuralGCM)** — Nature, 2024. M. [공식 논문](https://www.nature.com/articles/s41586-024-07744-y). physical solver와 학습 parameterization을 결합한다. ‘미방을 가져온다’면 어떤 물리 상태/법칙과 observation operator가 있는지 먼저 답해야 한다.

[38] **A foundation model for the Earth system (Aurora)** — Nature, 2025. M. [공식 논문](https://www.nature.com/articles/s41586-025-09005-y). Earth-system variable conditioning과 광범위 예측을 참고한다. EO image foundation model과 Earth-system forecast model은 입력·정답·검증이 다르다.

[39] **From Surface Forecasting to Observability Forecasting: A Latent World Model for Cloud-Aware EO Monitoring (LeWM)** — 2026 preprint. S(§8.3–8.4). [논문](https://arxiv.org/abs/2607.13651), [본문](https://arxiv.org/html/2607.13651v1). cloud/observability forecast와 synthetic anomaly test를 구분해서 읽는다. 모든 real environmental change를 해석하는 Earth world model로 인용하면 과장이다.

### D. EO forecasting·시계열 변화 데이터 [40–44]

[40] **Multi-modal Learning for Geospatial Vegetation Forecasting (Contextformer / GreenEarthNet)** — CVPR 2024 main. M. [공식 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Benson_Multi-modal_Learning_for_Geospatial_Vegetation_Forecasting_CVPR_2024_paper.pdf), [논문](https://arxiv.org/abs/2303.16198). 기상·지형 conditioning을 넣는 EO vegetation forecast가 이미 main research다. 정확히 어떤 미래 forcing이 평가 때 주어지는지 읽어야 한다.

[41] **DynamicEarthNet: Daily Multi-Spectral Satellite Dataset for Semantic Change Segmentation** — CVPR 2022 main. S(§3.1–3.3). [공식 PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Toker_DynamicEarthNet_Daily_Multi-Spectral_Satellite_Dataset_for_Semantic_Change_Segmentation_CVPR_2022_paper.pdf), [논문](https://arxiv.org/abs/2203.12560). 75 AOI의 daily PlanetFusion imagery와 monthly 7-class labels. gap-filled product라 독립 daily raw GT/real-time arrival로 바로 사용하면 안 된다.

[42] **EarthNet2021: A Large-Scale Dataset and Challenge for Earth Surface Forecasting** — CVPR 2021 EarthVision workshop. M. [공식 PDF](https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/papers/Requena-Mesa_EarthNet2021_A_Large-Scale_Dataset_and_Challenge_for_Earth_Surface_Forecasting_CVPRW_2021_paper.pdf). 영상·기상 기반 미래 Earth surface forecast benchmark다. interpolation과 forecasting의 차이를 설명하기 좋다.

[43] **Multi-modal temporal attention models for crop mapping from satellite time series (PASTIS-R)** — ISPRS JPRS 187, 2022. M. [저널](https://www.sciencedirect.com/science/article/pii/S0924271622000855), [데이터](https://zenodo.org/records/5735646). 실제 비동기 S1/S2 reconstruction/crop model 비교에 유용하다. crop parcel labels를 매일의 change onset 정답으로 착각하지 않는다.

[44] **MultiEarth 2022: Multimodal Learning for Earth and Environment** — CVPR 2022 workshop/challenge. M. [논문](https://arxiv.org/abs/2204.07649), [공식 연구 페이지](https://www.microsoft.com/en-us/research/publication/multiearth-2022-multimodal-learning-for-earth-and-environment-workshop-and-challenge/?lang=zh-cn). deforestation·cloud·multimodal monitoring에 직접적인 데이터/문제다. 환경 변화에 관심 있다면 산사태 두 지역만 고집하기보다 검토할 가치가 있다.

### E. EO language·metadata alignment·robotics analogy [45–55]

[45] **GeoChat: Grounded Large Vision-Language Model for Remote Sensing** — CVPR 2024 main. M. [공식 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.pdf). spatial grounding과 EO instruction tuning의 직접 prior다. 언어를 붙이는 것 자체는 충분한 novelty가 아니다.

[46] **TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data** — ICLR 2025. M(공식 논문/저자 코드). [공식 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/ac3af725ae398b6184faae0828bdbd6c-Paper-Conference.pdf), [논문](https://arxiv.org/abs/2410.06234), [저자 코드](https://github.com/ermongroup/TEOChat). multi-image temporal EO instruction model로 직접 baseline이다. 몇 pilot 문항의 성능으로 temporal grounding 전반이 약하다고 일반화하지 않는다.

[47] **EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues** — CVPR 2025 main. M. [공식 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Soni_EarthDial_Turning_Multi-sensory_Earth_Observations_to_Interactive_Dialogues_CVPR_2025_paper.pdf), [논문](https://arxiv.org/abs/2412.15190). multimodal/multitemporal EO→LLM의 학습 경로가 실제로 존재한다는 직접 증거다. training volume과 sensor compatibility를 맞춰 비교해야 한다.

[48] **ChangeChat: An Interactive Model for Remote Sensing Change Analysis via Multimodal Instruction Tuning** — 2024 preprint. M. [논문](https://arxiv.org/abs/2409.08582). bitemporal change 이해·대화가 선행한다. 전후 차이를 말하는 수준을 넘어 state provenance·정량 grounding이 무엇을 추가하는지 정의해야 한다.

[49] **GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks** — ICCV 2025 main. M. [공식 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Danish_GEOBench-VLM_Benchmarking_Vision-Language_Models_for_Geospatial_Tasks_ICCV_2025_paper.pdf). 다양한 EO VLM 과업과 temporal/change 평가에 참고한다. VLM의 약점은 모델·과업별로 보고해야 한다.

[50] **Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence** — 2026 preprint. M. [논문](https://arxiv.org/abs/2602.10354). EO embedding→interpretable probes/vector retrieval→LLM 질의의 가까운 prior다. LLM judge 위주 평가를 독립적인 실제 환경 주장 검증과 동일시하지 않는다.

[51] **Characterizing AlphaEarth Embedding Geometry for Agentic Environmental Reasoning** — 2026 preprint. M. [논문](https://arxiv.org/abs/2604.18715). embedding retrieval·geometry·environmental agent의 직접 prior다. ‘embedding과 지역정보를 LLM에 연결’만으로 신규성을 주장하기 어렵다.

[52] **WildSAT: Learning Satellite Image Representations from Wildlife Observations** — ICCV 2025 main. M. [공식 페이지](https://openaccess.thecvf.com/content/ICCV2025/html/Daroya_WildSAT_Learning_Satellite_Image_Representations_from_Wildlife_Observations_ICCV_2025_paper.html), [논문](https://arxiv.org/abs/2412.14428). ecology/텍스트/위치 metadata로 EO representation을 정렬하는 prior다. metadata alignment도 학습 연구다.

[53] **SatCLIP: Global, General-Purpose Location Embeddings with Satellite Imagery** — AAAI 2025. M(공식 venue 확인). [공식 논문](https://ojs.aaai.org/index.php/AAAI/article/view/32457), [논문](https://arxiv.org/abs/2311.17179). satellite image와 coordinates를 정렬한 location representation이다. CLIP이라는 이름 때문에 text alignment 모델이라고 오인하지 않는다. location shortcut을 차단한 generalization 평가도 필요하다.

[54] **OpenVLA: An Open-Source Vision-Language-Action Model** — 2024 공개 preprint 기준. M. [공식 프로젝트](https://openvla.github.io/), [논문](https://arxiv.org/abs/2406.09246). image features→projector→LLM/action의 robotics analogy다. EO에서는 action 정의와 실제 interaction 데이터가 별도로 필요하다.

[55] **GEO-Bench-2: From Performance to Capability, Rethinking Evaluation in Geospatial AI** — 2025 공개 preprint/후속 revision 기준. M. [논문](https://arxiv.org/abs/2511.15658). 다과업 EO capability 평가를 참고한다. static capability benchmark가 causal arrival/update benchmark를 대체하지는 않는다.

## 9. 논문 서사의 초안과 아직 말하면 안 되는 것

잠정 제목: **Regional EO State Estimation with Observation-Grounded Temporal Updates**. 확정 제목·새 방법·결과 주장이 아니다.

서사 후보는 (1) pooled/annual EO representation의 offline 능력과 irregular-arrival monitoring 사이의 차이를 측정하고, (2) 지역 적응과 품질 조건부 교정이 normal dynamics와 abrupt change에서 각각 어떤 이득을 주는지 검증하며, (3) 실질 지연·오경보·불확실성·비용의 trade-off를 보여주는 것이다.

쓰면 안 되는 문장:

- “최초의 EO temporal interpolation / 최초의 지역 특화 embedding predictor / 최초의 recursive update.”
- “close-pair cosine이 noise ceiling이므로 우리의 모델이 상한에 도달했다.”
- “같은 달 swap이 무해하므로 OlmoEarth에는 temporal information이 없다.”
- “2장의 영상이면 사건을 75% 탐지한다.”
- “latent cosine을 높였으므로 실제 시간 해상도를 hourly로 향상했다.”
- “VLM은 데모뿐이며 CVPR 연구가 될 수 없다.”
- “캐시 임베딩이 있으므로 새 state model도 싸고 정확할 것이 확정됐다.”

가장 먼저 필요한 그림은 멋진 hourly 지도나 VLM demo가 아니다. **같은 지역의 미사용 미래 스트림에서, 새 영상 도착 전 추정과 도착 후 교정이 어떻게 달라지고 실제 사건을 어느 오경보·비용으로 확인하는지** 보여주는 paired 결과다. 그 결과가 좋으면 CVPR 방법/벤치마크 서사를 정할 수 있고, 아니면 단순 보간기의 범위를 정직하게 좁힐 수 있다.

## 10. 이번 작업의 종료 범위

- 55편의 1차 자료 링크와 확인 수준, 직접 경쟁 prior, 기존 T/L 판정의 적용 범위, 새 독립 실험 초안을 정리했다.
- 새 실험·서버 접속·GPU 작업·기존 prereg/코드 수정·새 라벨 개봉·commit/push는 하지 않았다.
- 기존 T4/L1 실패를 성공으로 바꾸지 않았으며 원본 측정 문서는 보존했다.
- 남은 일: 최우선 논문 정독, cache execution provenance 확인, AOI의 dense observation/독립 GT 계약, 새 validation/test 및 prereg 확정 후 실행.
