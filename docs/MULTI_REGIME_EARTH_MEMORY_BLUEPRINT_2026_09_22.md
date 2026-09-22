# 홍수를 넘어: 다중 현상의 지역 상태와 근거를 갱신하는 EO–VLM

검토일: 2026-09-22. 문헌·설계 감사이며 새 학습 결과가 아니다. 기존 실험, 실패 판정,
사전등록, Poseidon 저장소를 변경하지 않았다. 논문 채택 가능성이나 최초성을 입증하지 않는다.

## 1. 결정 요약과 아직 없는 증거

권장 큰 그림은 **여러 현상을 예측하는 거대 모델**보다 **서로 다른 현상에서 같은 지역 기억을
갱신하고, 판단을 유지·수정·보류하는 공통 연산자**다. 홍수는 물리 연결을 시험하는 한 사례다.
빠른 재해, 느린 식생, 기상만으로 예측하기 어려운 토지이용 변화로 검증 범위를 확장한다.

아직 없는 것은 세 영역에서 공유 학습한 updater, 실제 관측 도착 전후의 판단 정답,
시간 가용성을 검증한 통합 데이터, 강한 retrieval/full-prefix 기준선 대비 이득이다.
따라서 지금 단계는 유망한 연구 가설이며 CVPR 방법 기여가 확보된 상태는 아니다.

대상 수를 늘리는 것만으로는 부족하다. EO-WM은 이미 기상 조건부 EO 예측을, LongEarth-R1은
여러 현상의 긴 시계열과 프레임·영역 근거를 다룬다. 우리 질문은 다음처럼 좁힌다.

> 현재까지 도착한 관측만으로 만든 지역 기억이, 새 증거가 들어올 때 무엇을 고치고 무엇을
> 유지해야 하는가? 제한된 기억과 계산량에서도 그 판단의 시각·공간 근거를 보존할 수 있는가?

## 2. 세 가지 변화 양식과 벤치마크

| 변화 양식 | 본체 후보 | 원래 제공하는 평가 | 새 연구에서의 역할 |
|---|---|---|---|
| 빠르게 전개되는 재해 | WildfireSpreadTS | 일별 시계열의 다음 날 active-fire 영역 예측 | 기상·지형 prior를 관측으로 교정하는 능력 |
| 느린 식생·계절 변화 | GreenEarthNet | 5일 간격 S2와 기상을 이용한 식생 예측 | 정상 계절성과 지속적 이상을 구분하는 기억 |
| 토지피복·인위적 변화 | DynamicEarthNet | 월별 의미 지도와 변화 분할 | 기상 prior로 설명되지 않는 새 관측을 수용하는 능력 |

이는 **센서별 입력 어댑터는 달라도 갱신 모듈의 가중치는 공유**하는 설계다.
같은 구조를 데이터마다 독립 학습한 결과만으로 범용성을 주장하지 않는다.
세 자료는 동일 지역에서 수집된 것이 아니므로 이들만으로 복합재난의 인과적 상호작용까지
학습·검증했다고 주장할 수 없다.

### 2.1 WildfireSpreadTS: 빠른 실제 시계열 확보

NeurIPS 2023 데이터셋은 미국 607개 산불, 2018–2021년의 일별 다중모달 자료와 기상·지형을
포함한다. 실제 관측 기반 다음 날 예측을 시험할 수 있다는 점에서 짧은 홍수 pre/post 쌍보다
시간 갱신 실험에 적합하다. 단, VIIRS 등 입력은 S2가 아니므로 OlmoEarth의 S2 밴드에 그대로
넣을 수 없다. 센서별 인코더를 두고 공유 상태에서 결합한다.
[논문](https://proceedings.neurips.cc/paper_files/paper/2023/file/ebd545176bdaa9cd5d45954947bd74b7-Paper-Datasets_and_Benchmarks.pdf).

공식 저장소는 2026년 2월 angle feature의 sin-only 인코딩 문제를 **미수정 주의사항**으로 적었다.
버전 봉인 후 sin/cos 수정 baseline을 따로 재현해야 한다. 데이터는 CC-BY-4.0으로 안내된다.
[공식 코드와 공지](https://github.com/SebastianGer/WildfireSpreadTS).
사건 중심 데이터의 오경보율은 전체 지표면의 운영 오경보율과 다르다. 후자를 주장하려면
비사건 시공간도 필요하다. daily 합성 입력·기상 자료의 실제 이용 가능 시점 역시 별도 감사한다.

### 2.2 GreenEarthNet: 기존 날씨–지표 연결을 강한 기준선으로

30개 5일 간격 프레임, 앞 10개 관측과 뒤 20개 예측, 20m RGB+NIR, daily E-OBS와 DEM이
핵심 계약이다. 기상에는 기본 강수·기압·온도 외 풍속·습도·일사도 포함된다. 공간·시간 분포 밖
평가가 제공된다. 관측된 미래 기상을 조건으로 주는 설정과 실제 발행 시점 기상예보만 쓰는
운영 설정을 분리한다. 현재의 12-band Olmo 입력과도 별도 대응이 필요하다.
[CVPR 2024 논문 §3.3](https://arxiv.org/html/2303.16198v2),
[공식 구현](https://github.com/vitusbenson/greenearthnet).

미래 NDVI/반사도는 평가할 수 있지만 매 시점의 사건 원인·자연어 설명 정답이 있는 자료는 아니다.
식생 감소를 관측했다고 자동으로 가뭄 피해, 벌목, 환경파괴라고 단정하지 않는다.
EarthNet과 GreenEarthNet은 일부 학습 위치를 공유하므로 서로 독립인 외부 검증 두 건으로 세지 않는다.

### 2.3 DynamicEarthNet: 시간 가용성 검사가 선행되어야 함

75개 지역, 2년 daily Planet Fusion 자료와 월별 7종 토지피복 라벨, 지역 단위 분할을 제공한다.
중요하게도 daily 영상은 모두 그날의 원시 실관측이 아니다. Appendix A1은 gap-fill 픽셀의
원관측 날짜 거리·방향을 QA로 제공한다고 명시한다. A2의 보조 S2는 **월 전체 영상의 composite**다.
미래 정보를 제거하거나 자료 이용 가능 시점을 뒤로 잡지 않고 온라인 평가하면 누수 위험이 있다.
[CVPR 2022 논문과 부록](https://arxiv.org/html/2203.12560),
[공식 데이터](https://mediatum.ub.tum.de/1650201).

월별 라벨로 정확한 사건 발생일을 만들 수 없다. 월/관측 구간 수준 평가로 한정한다.
QA만으로 모든 전처리의 시간 가용성이 입증되는 것도 아니므로 처리 계약까지 확인한다.
복원이 불가능하면 이 자료는 회고적 변화 평가에만 쓰고, 온라인 본체에는 원시 관측을 다시 모은다.

### 2.4 확장 후보는 역할을 나눠 사용

- **SpaceNet 7**: 도시 건물의 월별 변화·추적. 월별 모자이크와 고해상도 건물 주석을 제공하지만
  정확한 공사 시작일 정답은 아니다. 40m latent로 작은 건물을 설명하는 해상도 문제 때문에
  VHR 어댑터 또는 공간 집계가 필요하다. [공식 챌린지](https://spacenet.ai/sn7-challenge/).
- **MultiEarth 2022**: Amazon의 다중센서 기록과 성긴 날짜의 산림손실 주석. 별도 지역·현상
  검증에 유용하나 daily 사건 추적이라고 부르지 않는다.
  [논문](https://arxiv.org/abs/2204.07649), [공식 사이트](https://sites.google.com/view/rainforest-challenge/multiearth-2022).
- **PASTIS-R**: 불규칙 S1/S2 관측과 농작물 분류·필지 해석. 작기 단위 지도이지 시점별 변화
  정답이 아니므로 sensor dropout/early classification 진단으로 한정한다.
  [공식 저장소](https://github.com/VSainteuf/pastis-benchmark).
- **TS-SatFire**: active fire·burned area·진행 예측의 독립 산불 후보.
  [Scientific Data 2025](https://www.nature.com/articles/s41597-025-06271-3).
- **KuroSiwo/FloodCastBench**: 이전에 검토한 실제 SAR 교정/시뮬레이션 메커니즘 시험을 유지한다.
  단일 pre/post 출력이나 simulator frame을 dense 실관측 사건 정답으로 승격하지 않는다.
- **PANGAEA**: 공통 baseline과 데이터 어댑터를 만드는 데 참고할 평가 틀이다. 그 자체가
  순차 언어 판단 갱신 정답을 제공하는 것은 아니다.
  [공식 저장소](https://github.com/VMarsocci/pangaea-bench).

코드 license, 논문 license, 영상 원자료 license는 다르다. 위 자료를 결합·재배포하기 전에
각 provider 조건을 별도 기록하며, 지금 모든 재배포 권리를 확인했다고 주장하지 않는다.

## 3. 모델: 공통 기억 + 선택적 물리 prior + 근거 기반 언어

제안 인터페이스는 다음과 같다. 이것은 실험할 가설이며 새 구조의 성능 증거가 아니다.

```text
지역 context: 좌표·지형·기후·센서·시각·가용 시점
                       ↓
이전 지역 기억 ── 선택적 현상별 prior ── 예상 상태와 불확실성
       │                               │
       └──── 새 EO 관측·품질·실제 Δt ───┤
                                       ↓
                        공유 evidence-update 모듈
                              ↙             ↘
                    갱신된 지역 상태       근거 참조·주장 상태
                              └──────┬──────┘
                          공간 readout / VLM 질의
```

물리 예측은 수심·확산·식생 지표처럼 정의된 상태와 관측 연산자를 통해 연결한다.
768차원 임베딩 전체가 하나의 PDE를 따른다고 가정하지 않는다. 홍수의 유동 법칙이 곧 식생의
생리 모델이 아니며, 개발·벌목을 기상만으로 예측할 수 있다고도 가정하지 않는다.
새 관측이 prior와 다를 때는 실제 변화, 센서 이상, prior 오류를 구분할 수 있어야 한다.

Poseidon은 현상별 prior 후보 중 하나다. 기존 PDE의 단위·상태·forcing·경계조건과 목표 영역의
계약이 맞는지 먼저 시험한다. 잘못된 prior를 무시하는 능력도 범용 갱신의 평가 항목이다.
no-prior/계절 기준선이 낫다면 해당 영역에서 물리 모듈을 제거한다.

기억은 둘로 분리한다.

1. **지금의 추정 상태**: 공간 feature, 예측 분포, 불확실성. 관측으로 수정할 수 있다.
2. **불변 관측 기록**: obs ID, 획득/도착 시각, 위치·영역, 센서/GSD, 품질, 원본 pointer.
   압축 기억의 feature를 원시 증거로 인용하지 않는다. 과거 해석은 바뀔 수 있지만 원관측을 덮어쓰지 않는다.

bounded memory는 활성 state/token budget이 제한된다는 뜻이다. 원본 archive와 검색 비용까지
사라지는 것은 아니므로 저장량, retrieval I/O, 갱신, 질의 비용을 모두 보고한다.
초기에는 backbone을 고정하고 sensor adapter·shared updater·언어 연결기를 작게 학습한다.
평가 스트림 도중 모델 가중치를 계속 미세조정하는 것은 별도 문제로 제외한다.

## 4. VLM이 담당할 것은 classification 이상의 관계·수정 질의

미리 정의한 현상 label을 문장으로 바꾸는 것만으로는 언어 모델의 기여가 없다. 예를 들어:

- “지난달 식생이 감소한 곳 중, 최근 두 관측에서도 감소가 유지되는 영역을 보여줘.”
- “새 건물이 늘어난 영역과 일시적으로 물에 덮인 영역을 구분하고 근거 날짜를 알려줘.”
- “어제의 확산 예상 중 오늘 관측과 맞지 않는 곳은 어디인가?”
- “새 영상이 흐려졌는데, 기존 변화 판단을 철회할 만큼 반대 증거가 있는가?”

질의가 요구하는 다중 속성·공간 관계·기간 조건을 native EO feature와 근거 칩에 연결하고,
답변과 함께 영역, 기간, 근거 obs ID, observed/inferred/predicted/unknown 구분을 출력한다.
서로 다른 데이터셋에서만 학습한 속성들을 한 지역에서 결합하는 위 예시는 추가 paired 평가가
필요한 목표이며, 현재 데이터가 그대로 그 정답을 제공한다는 뜻은 아니다.

학습용 설명은 map/trajectory로 silver 문장을 만들 수 있다. 그러나 최종 갱신 평가는 사람이
**각 시점까지의 관측만 보고** 판정한 gold가 필요하다. 전체 영상을 본 최종 정답을 앞 시점의
정답으로 소급하지 않는다. 독립 검수와 애매함/판독 불가 라벨을 허용한다.
사회경제적 속성이나 ‘불법’, ‘환경파괴 원인’은 영상 형태만으로 확정하지 않고 별도 기록과 구분한다.

## 5. 신규성이 이미 있는 부분과 남은 연구 가설

| 직접 선행 | 이미 다루는 것 | 이 설계가 별도로 입증해야 할 것 |
|---|---|---|
| PhyDNet, CVPR 2020 | 물리적 동역학과 미지 요인 분리, 관측 교정 성격의 recurrent cell | physics+residual+gate 이상의 이득 |
| EO-WM, 2026 preprint | 기상 조건부 확률적 EO 예측, 계절·이상·누적 stress | 관측 도착 후 수정, 근거 보존, 여러 변화 양식 |
| EarthDial / TEOChat | 다중센서·다시점 언어 이해 | 매 prefix의 persistent state와 판단 수정 |
| TerraScope, 2026 preprint | 영역에 접지한 답변, bi-temporal 해석 | 긴 순차 스트림의 근거·판단 갱신 |
| LongEarth-R1, 2026 preprint | 여러 현상, 최대 30프레임, 프레임·영역 근거 | 고정 budget에서 미래 없이 유지·수정·보류 |

[PhyDNet](https://arxiv.org/abs/2003.01460), [EO-WM](https://arxiv.org/html/2606.27277),
[EarthDial](https://arxiv.org/html/2412.15190), [TEOChat](https://arxiv.org/html/2410.06234),
[TerraScope](https://arxiv.org/html/2603.19039), [LongEarth-R1](https://arxiv.org/html/2608.13344).

LongEarth-R1의 식(1)은 주어진 전체 시계열을 한 번에 읽는 입력이다. 따라서 매 prefix에서
다시 실행하는 강한 기준선으로 삼을 수 있다. 그때 미래 프레임은 제외한다. 출판된 프레임/영역
trace를 그대로 ‘새 관측 전후 판단 수정’ 정답이라고 부르지 않는다.

DynamicEarthNet Appendix B3도 이미 ‘이전 예측 오류를 고치는 일’과 ‘실제 지표 변화’를 구분한다.
그러므로 **판단 수정 자체가 최초**라고 주장하지 않는다. 제안된 차이는 근거 유형·시점 가용성·
기억 예산을 함께 제약한 멀티모달 언어 갱신이며, 이 조합의 신규성도 추가 선행 비교가 필요하다.

EO-WM 저장소에서 확인한 공개 범위는 benchmark CSV와 Earthformer reference 평가 예제다.
EO-WM 전체 학습 코드·가중치가 공개됐다고 가정하지 않는다.
[공개 범위](https://github.com/Luo-Z13/EO-WM/blob/main/README.md).
Aurora/ESFM처럼 넓은 Earth-system 예측 선행도 있어 ‘범용 지구 모델’ 명칭만으로 차별화되지 않는다.
[Aurora](https://www.nature.com/articles/s41586-025-09005-y), [ESFM](https://arxiv.org/abs/2605.00850).

## 6. 학습과 평가를 세 층으로 분리

### 층 A — 공개 benchmark의 원래 과제

공식 split·입력·metric을 먼저 유지해 원 논문 baseline과 비교한다. 산불의 다음 날 mask,
식생의 예측 오차, 월별 토지피복 변화 점수는 각각 보고한다. 단위가 다른 점수를 임의로
합쳐 성능 하나로 만들지 않는다. 별도 새 protocol 결과를 공식 leaderboard 점수라고 부르지 않는다.

### 층 B — 새 causal replay와 판단 갱신 정답

각 입력에 acquisition_time, available_time, support_interval을 기록한다. 예보는 issue_time과
valid_time을 분리한다. 주어진 평가 cutoff 뒤 정보는 teacher·정규화·계절 통계·샘플 선택에서도 제외한다.
과거 전부를 쓰는 full-prefix teacher는 가능하지만 미래까지 쓴 teacher는 온라인 정답이 아니다.

각 query/prefix의 gold는 claim, region/mask, 시간 구간, support/refutation/unresolved,
최소 근거 frame 집합을 포함한다. 실제 미래 결과는 별도 필드로 관리한다.
수정해야 하는 새 증거뿐 아니라 구름·중복·무관 관측에서도 이전 판단을 지키는 평가가 필요하다.
연속 시점의 label 변화는 실제 지표 변화일 수 있으므로 그 자체를 ‘모델 오류 수정’ gold로 삼지 않는다.

주요 점수: 적절한 수정 정확도·재현율, 틀린 철회율, 영역/근거 정확도, 답변 coverage와 오류율의
관계, 확률 보정, 관측/예측 혼동, 고정 오경보 조건에서의 확인 지연. 정확한 사건일이 없는 자료는
구간 수준 또는 새 증거 도착 후 지연으로 보고한다. LLM 문장 채점은 보조만 사용한다.

### 층 C — 공유 갱신 모듈의 일반화와 비용

- shared weights vs 현상별 독립 updater: 전체 parameter·compute 예산과 각 task 성능을 함께 비교.
- 한 현상을 학습에서 뺀 전이: 센서도 바뀌면 센서 이동과 현상 이동을 분리한 control 필요.
- 최신 1장, temporal mean, Δt-GRU, 계절 기준선, PhyDNet류를 역할에 맞게 비교.
- 학습 없는 quality/change/recency top-K + 같은 VLM, deterministic ledger+template를 포함.
- LongEarth/Qwen류에 매번 전체 **현재 prefix**를 넣는 재계산 baseline 포함.
- no-weather, no-physics, no-typed-evidence, state-only, metadata-only ablation 포함.
- 같은 시점의 같은 데이터, 활성 token budget과 전체 I/O/연산 비용을 맞춰 Pareto 비교.
- event/AOI/지역을 통계 단위로 삼는다. 같은 사건의 수천 타일을 독립 사건으로 세지 않는다.

개선 기준의 수치·CI·non-inferiority 허용 범위는 별도 실행 사전등록에서 고정한다. 이 문서를
이전 실패 기준을 바꾸거나 완료된 실험의 통과 판정을 바꾸는 근거로 사용하지 않는다.

## 7. 현재 자산에서 시작하는 순서

1. **입력 계약 + 최소 언어 평가**: 세 본체 후보의 센서·시각·누수 가능성을 검사하고,
   각 자료에서 maintain/revise/unknown을 검수할 작은 사람 gold를 확보한다.
   BigEarthNet.txt의 기존 Olmo 입력 성공은 언어 정확도 증거가 아니므로 grounding baseline도 유지한다.
2. **공유 updater**: 먼저 물리 모듈 없이 지역 상태와 증거 갱신을 학습한다. 이 단계에서
   retrieval/full-prefix 대비 효용과 영역별 negative transfer를 확인한다.
3. **선택적 물리 결합**: 기존 Poseidon 실험 자산으로 물리 연결을 작은 paired task에서 검사하고,
   날씨·지형 prior가 도움 되는 영역에만 확장한다. no-prior보다 낫지 않으면 강제하지 않는다.
4. **언어와 확장**: structured claim과 영역 근거가 맞은 뒤 자연어 관계 질의를 넓힌다.
   필요가 증명될 때만 7B급 LoRA를 진행한다. active sensing/행동 선택은 후속 연구다.

H100/H200은 이 과정의 frozen feature 추출, 입력 어댑터, 작은 updater, 필요시 LoRA에 쓸 수 있다.
대규모 기상 FM 사전학습부터 시작하지 않는다. 현재 자료의 접근·샘플 shape·throughput을
측정하지 않았으므로 GPU 시간이나 메모리 적합성을 확정하지 않는다. 이번에는 서버/GPU 실행 없음.

## 8. 논문으로 진행할 조건

세 현상의 표가 있다는 것보다, **하나의 공유 갱신 원리가 미지 지역에서도 근거를 잃지 않고
잘못된 예상을 교정하며, 강한 기준선 대비 정확도–비용 이득을 보이는가**가 중요하다.
새 관측을 무조건 신뢰하거나 prior를 무조건 유지하는 모델은 모두 탈락시킬 수 있어야 한다.

방법이 약하면 기여를 정직하게 benchmark/protocol로 축소할 수 있지만, 그것도 독립적 gold,
재현 가능한 baseline, 충분한 사건/지역 다양성이 있어야 한다. 단순 데이터셋 합본은 부족하다.
장기 비전은 지역을 지속적으로 이해하는 EO–VLM이고, 첫 논문의 단위는 그 안의 검증 가능한
공통 갱신 연산자와 근거 기반 판단 평가다.
