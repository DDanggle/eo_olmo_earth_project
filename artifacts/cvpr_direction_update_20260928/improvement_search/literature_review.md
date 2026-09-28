# OE10: 실제 성능 개선을 위한 문헌 검토 — 2026-09-28

전제는 root가 전달한 **query 관측 2시점, support 8개, 연간 crop mask, Qwen의 text-only 7개 문자열**이다. 실제 데이터나 실행 코드는 열지 않았다. 아래는 논문·공식 코드에서 확인한 방법을 현재 조건에 옮기는 제안이며, 성능 개선 결과 또는 최신 SOTA 주장이 아니다. GPU·서버 작업 없음.

**추천 순서는 ① 클래스 편향을 간단히 분리 → ② 영역별 support–언어 matching을 직접 학습 → ③ 날짜별 정보를 보존하는 집계다.** 언어 연결에 가장 직접적인 개선 후보는 ②다. 평균이 크게 오를지는 현재 근거로 보장할 수 없다.

## 1. 가장 간단한 후보: 클래스 편향 보정과 학습 부족을 먼저 구분

근거: [Long-tail Learning via Logit Adjustment, ICLR 2021, arXiv v2 2021-07-09](https://arxiv.org/html/2007.07314v2), [공식 코드](https://github.com/google-research/google-research/tree/master/logit_adjustment). 표준 CE로 학습한 모델의 logits에서 `tau * log(training_class_prior)`를 빼거나, 학습 CE 안에는 반대로 이 항을 더해 드문 클래스에 필요한 상대 마진을 바꾼다. 논문은 balanced classification error를 다루며, crop mIoU가 반드시 개선된다는 이론은 아니다.

우리 실험에 가져올 것:

- 겨울밀의 낮은 IoU만 보고 불균형이라고 단정하지 않는다. 먼저 train의 클래스별 loss·recall과 held-out 결과를 나눠, 못 배운 것인지 다른 클래스를 과선호하는 것인지 확인한다.
- 기존 CE 모델을 보존하고 작은 `tau` 후보 집합으로 post-hoc 보정부터 비교한다. train에서만 얻은 prior를 쓰고 smoothing한다. 8 support의 편향된 비율을 지역 전체 prior라고 취급하지 않는다.
- train 내 분리된 validation group이 없으면 최종 query에서 `tau`를 고르지 않는다. 고정값 또는 leave-one-support-group-out만 사용한다. binary segmentation이면 같은 목적의 단일 threshold 비교를 적용한다.
- 편향 보정이 효과적이면 그다음 학습 단계 logit-adjusted CE를 한 번 비교한다. 재표본화·weighted CE·focal·Dice를 한꺼번에 붙이면 무엇이 좋아졌는지 모른다.

이것은 새 방법이 아니라 **기존 성능 손실 중 얼마가 의사결정 편향인지 확인하는 싼 대조군**이다. 현재 loss가 CE가 아니거나 이미 균형 보정돼 있다면 원리 그대로 중복 적용하지 않는다. 겨울밀 recall만 올리고 false positive가 늘어 macro IoU가 떨어지는지도 함께 봐야 한다.

## 2. 가장 추천: 문자열을 붙이는 대신, 영역마다 텍스트와 support의 점수를 만들고 그 점수를 감독

근거 A: [DenseCLIP, CVPR 2022, arXiv v2 2022-03-21](https://arxiv.org/html/2112.01518v2), [공식 코드](https://github.com/raoyongming/DenseCLIP/blob/master/segmentation/denseclip/denseclip.py). 정규화한 visual/text 특징으로 `einsum('bchw,bkc->bkhw', ...)`의 pixel–text 점수 지도를 만들고 segmentation 학습에 사용한다. §4.3은 CLIP이 아닌 backbone도 비교한다. ADE20K 단일 스케일에서 RN50 38.6→41.0, Swin-T 44.5→45.4 mIoU다. **자연영상 결과이며 우리 예상 상승폭이 아니다.** 의미 있는 점은 EO와 텍스트가 원래 정렬되지 않았어도 supervised dense alignment를 시험할 근거가 있다는 것이다.

근거 B: [HDMNet, CVPR 2023, arXiv v1 2023-03-26](https://arxiv.org/html/2303.14652v1), [공식 코드](https://github.com/Pbihao/HDMNet). 하나의 평균 prototype에 의존하지 않고 mask로 선택한 support 특징과 query 사이 cosine correspondence를 학습한다. 계층별 matching과 correlation distillation을 분리해 검증했다. 이 논문은 자연영상 few-shot segmentation이며, 실제로 비슷한 클래스 구별이 남은 한계라고 명시한다. 따라서 target/confuser를 자동으로 해결한다고 인용하면 안 된다.

우리 실험의 작은 구현 제안:

1. OlmoEarth의 위치별 특징을 보존한다. 각 클래스/텍스트 조건에 대해 **텍스트 점수 지도**, **양성 support 점수 지도**, **혼동대상 support 점수 지도**를 별도로 만든다. support mask로 pooling하거나 소수의 subprototype을 만들되, 8개에서 큰 matching network를 처음부터 학습하지 않는다.
2. 초기에는 `visual-only score + learned_weight * text_score + support_positive_score - learned_weight * confuser_score` 정도의 작은 residual head면 충분하다. 비정렬 공간을 임의 cosine으로 비교하지 말고 작은 projection을 학습한다. frozen EO/head-only와 소량 EO update를 나눠 비교한다.
3. 연간 mask는 연간 crop identity의 위치별 감독으로 쓴다. mask만으로 특정 날짜의 생육 단계나 수확 여부 정답을 만들어내지 않는다. 원래 main CE에 text-score auxiliary CE를 더하는 정도에서 시작한다. hard negative는 train/support의 **실제 label이 확인된** 혼동 클래스에서 고른다.
4. 텍스트는 class name, 짧은 판별 속성, 검증된 지역·시기 문맥으로 나누고, 문장을 길게 쓰기보다 각 항목이 영역 점수에 영향을 주게 한다. Qwen의 7개 문자열이 이미지와 상호작용하지 않았다면, 현재는 VLM 시각 추론이 아니라 텍스트 특징 조건화다. 마지막 토큰 pooling 자체가 항상 틀렸다는 주장도 피한다. 핵심은 **어느 영역·support와 맞는지를 감독했는가**다.

필수 대조군은 같은 차원의 learned class-ID embedding, class-name만, 올바른 설명, 클래스 간 바꾼 설명이다. 단순 문자열 교환에서 성능이 변했다는 사실만으로 의미 이해를 입증할 수 없으므로 paraphrase와 미학습 구별에도 확인한다. support-only와 text-only를 반드시 남겨 언어가 더해준 값을 분리한다. 지역 문맥은 숫자/범주 metadata만 제공한 같은 정보량 대조군도 둔다. 도시명으로 작물 prior만 외우는 이득과 관측 판독 개선을 구분한다.

## 3. 두 날짜를 평균내지 말고, 날짜·가용 관측을 보존해 matching하기

근거: [U-TAE, ICCV 2021, 최신 arXiv v4 2022-06-27](https://arxiv.org/html/2107.07933v4), [공식 U-TAE 코드](https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py), [LTAE 코드](https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/ltae.py). 위치별 temporal attention으로 중요한 관측을 선택하고 다중 공간 해상도에 전달한다. 공식 구현은 `batch_positions`와 padding mask를 temporal encoder에 전달한다. 논문의 Appendix A.1은 33–61 관측의 PASTIS에서 test 관측을 8개로 제한할 때 −14.6 mIoU를 보고한다. 이는 우리 2시점 성능의 수치 예측이 아니라 **관측을 줄이면 잃는 정보가 상당할 수 있다는 경고 근거**다. 공식 저장소가 2022년 panoptic metric 버그를 수정했으므로 최신 v4를 기준으로 삼았다.

우리 실험에 가져올 것:

- OlmoEarth가 이미 timestamp를 쓰면 encoder에 날짜를 또 붙였다고 개선 방법으로 부르지 않는다. matching head까지 `z_date1`, `z_date2`, 실제 날짜 간격·유효 관측을 보존해 사용한다.
- 현재 두 날짜에는 작은 date-aware attention/gating부터 비교한다. 동급 baseline은 시간 평균, `[z1,z2,z2-z1]` + 같은 규모 MLP다. date shuffle은 정보 사용 여부 진단이며, 단독 성공 기준이 아니다.
- support의 다른 계절 표현을 한 prototype으로 무조건 평균내지 않는다. 가능한 경우 날짜별/생육시기별 support 유사도를 유지한다. 실제 지역 crop calendar가 확보된 뒤에만 그 정보를 조건화한다. Qwen이 만들어낸 지역별 수확 달력을 정답으로 쓰지 않는다.
- 두 관측이 겨울밀과 혼동 작물을 구별할 시기를 놓쳤다면, 언어가 누락된 관측을 복원해 주지는 못한다. 이후 동일 데이터의 추가 날짜를 쓸 수 있다면 2/4/8 시점의 정보량 비교가 큰 모델 교체보다 먼저다. 이 비교는 동일 정보량의 방법 비교와 분리해 보고한다.

## 다음 실행을 어떻게 해석할 것인가

작은 실험은 하나씩 바꾼다: 기존 방법 → 간단한 편향 보정 → 공간 support matching → 그 위에 dense text supervision → 날짜 집계. 앞 단계가 무효하면 다음 단계의 기준선도 무효 결과에 고정하지 않는다. 각 설정은 동일 support·split·학습량을 사용하고, macro IoU와 클래스별 IoU/recall/precision을 함께 보고한다. 8개 support에서는 support 선택 변화가 효과보다 클 수 있으므로 독립 support 구성 반복이 중요하다. 이때 train pixel 수를 독립 표본 수로 취급하지 않는다.

7개 문자열을 더 화려하게 바꾸는 것보다 **위치별 언어 감독과 실제 양성/혼동 사례의 비교**가 지금의 구체적인 개선 후보다. 먼저 이 작은 방식이 class-ID와 support-only 대조군을 이기는지 확인해야, 큰 EO–VLM 학습으로 확대할 실증 근거가 생긴다.
