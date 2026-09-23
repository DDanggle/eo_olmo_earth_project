# Poseidon의 시간·물리 지식을 EO 관측과 언어 판단 갱신에 연결할 수 있는가?

> 2026-09-22 정리: 사용자 요청으로 PDE·기상 예측 후속 트랙에 분리했다.
> 아래는 9/21의 감사·설계 기록이며 현재 VLM reader/memory 실험의 실행 계획이 아니다.

검토일: 2026-09-21. 이전 저장소 `/Users/dgyi/dong/ai_projects/nips2026-1/v1-0426`는 읽기 전용으로
조사했다. 아래에서 **로컬 결과**, **외부 논문 보고**, **아직 실행하지 않은 제안**을 구분한다.
새 GPU 학습·시뮬레이션·서버 실행은 하지 않았다. 이 문서는 실행 사전등록이나 CVPR 신규성 확정이 아니다.

## 1. 결론을 쉬운 말로

가능하다. 다만 연결의 뜻은 **“PDE로 위성 임베딩 사이를 매끈하게 잇는다”**보다는 다음에 가깝다.

> 이곳의 지형·토지피복과 앞으로의 비를 알면, 물이 어떻게 움직일지 예상한다.
> 위성이 다시 관측하면 그 예상이 맞는지 확인하고 지역 상태와 설명을 고친다.

이때 OlmoEarth는 **지역과 관측을 읽는 부분**, Poseidon 계열은 **물리 상태의 변화를 예측하는 부분**,
VLM은 **어디서 무엇을 관측했고 무엇은 아직 예상인지 설명하는 부분**이다.
현재는 이 세 부분 사이의 paired training data와 관측 연결 모듈이 없다. 두 체크포인트를 붙인다고
연결이 완성되는 것은 아니다.

연구 방향은 살아 있다. 다만 기존 지역 언어 접지 연구를 당장 전부 버리고 기상 FM을 새로 만드는
방향은 권하지 않는다. **홍수 한 현상으로 물리 연결의 효용부터 검증**하고, 통과한 상태만 기존
근거 기억·언어 모듈에 연결하는 것이 가장 해석하기 쉽다. 이는 연구 판단이며 성능 예측이 아니다.

## 2. 이전 저장소에서 실제로 확인한 것

### 2.1 이미 Earth–PDE 연결 설계가 있었다 — 결과는 아니다

[`earth_robotics_bridge/design.md`](/Users/dgyi/dong/ai_projects/nips2026-1/v1-0426/earth_robotics_bridge/design.md:129)는
EO encoder × physics operator의 2×2 대조와 cross-region flood를 이미 제안했다.
특히 143행은 1-channel Darcy wrapper와 EO latent의 단순 연결을 경고하고,
204행 이후는 DEM·강우·경계조건과 EO context의 역할을 분리한다. 좋은 출발 설계다.
그러나 그 문서의 selector/robot 단계는 당시 계획이며, 최신 P1 결과가 그 성공을 보장하지 않는다.

### 2.2 세 종류의 ‘시간’을 혼동하면 안 된다

| 시간의 종류 | 이전 연구에서 의미 | 이번 질문에 주는 근거 |
|---|---|---|
| 실제 동역학 시간 | NS 등의 상태가 전개되는 시간, 시간 간격을 넣는 operator | 물리 상태의 시간 전개 학습을 재사용할 수 있음 |
| pseudo-time | 정상상태 Darcy 해를 반복 계산하면서 잔차가 줄어드는 정도 | 실제 비·침수·식생의 경과시간이 아님 |
| steady task의 고정 time 입력 | time=0, .25, .5, .75, 1 중 어떤 상수를 넣는가 | 인터페이스 민감도 실험이지 시간 보간의 증명이 아님 |

원래 Poseidon 논문의 all2all은 실제 trajectory의 여러 시작/끝 상태를 학습쌍으로 쓴다.
적절한 상태에서의 해 연산자 합성 성질을 활용한 것이며, 임의의 EO 임베딩이나 임의의 외력에도
그 성질이 성립한다는 뜻은 아니다. [Poseidon 원문](https://arxiv.org/html/2405.19101v2).

로컬 `multipde_transfer/corpus_gen.py`의 tau는 `-log(||r_i||/||r_0||)`라는 잔차 시계다.
[`G1 원시 판정`](/Users/dgyi/dong/ai_projects/nips2026-1/v1-0426/multipde_transfer/results_g1/g1_report.json)은 다음과 같다.

- `G1_PASS=false`: true clock은 constant 대비 1/3 seed, shuffled clock 대비 0/3 seed 우세.
- true-clock rel-L2는 seed 42/7/123에서 .1580/.1659/.1483.
- direct D0는 같은 seed에서 .0997/.0985/.1280으로 전부 더 좋았다.
- 따라서 이 결과를 “짧은 실제 시간 관계가 증명됐다”고 재사용할 수 없다.
  반대로 정상해 계산 시계의 실패가 실제 날씨·홍수 시간 전개의 실패를 뜻하지도 않는다.

별도의 실제 동적 NS 실험은 존재한다. 원장/논문 초안에는 입력·출력 인터페이스 보존의
rollout 개선이 기록되어 있다. 실제
[`run_ladder1_gac.py:188`](/Users/dgyi/dong/ai_projects/nips2026-1/v1-0426/transfer_learning/_workspace/run_gac_20260816/code/run_ladder1_gac.py:188)는
NS-PwC/NS-SVS를 **고정 Δt=.05**의 one-step task로 두고, 784행 이후 같은 time vector를
반복 사용해 horizon1/5/10/20을 평가한다. **장기 rollout 안정성**과 **시간 사이의 미관측 상태 복원**은
다른 평가다. variable-Δt, t1/t3→t2, irregular observation의 증거로 사용할 수 없다.
사용자 기억의 특정 실험을 단정하지 않고, 이번에는 확인한 실험 이름별 범위를 구분한다.

[`roadmap_final_v1.md:877`](/Users/dgyi/dong/ai_projects/nips2026-1/v1-0426/transfer_learning/roadmap_final_v1.md:877)의
C1은 Helmholtz에서 time=0이 time=1보다 좋았지만 Darcy에서는 기준 미달이다.
같은 원장의 후속 해석은 “사전학습 계약을 무조건 지켜라”라는 일반론도 철회했다.
또한 [`현재 wrapper:489`](/Users/dgyi/dong/ai_projects/nips2026-1/v1-0426/src/backbones/poseidon.py:489)는
batch에 고정 lead time을 넣으므로 그대로는 관측별 불규칙 Δt를 표현하지 못한다.

### 2.3 더 중요한 재사용 자산: 잘못된 메타정보 의존을 드러내는 실험

최신 판정은 `SUMMARY_20260820.md`가 아니라 8/22까지 갱신된
[`roadmap_final_v1.md:2381`](/Users/dgyi/dong/ai_projects/nips2026-1/v1-0426/transfer_learning/roadmap_final_v1.md:2381)를 우선한다.

핵심은 “훈련 데이터에서 두 설명이 항상 같이 움직이면, 정확도만으로 모델이 무엇을 배웠는지
구분할 수 없다”는 문제다. EO에서는 지역명·계절·강우·피해 유형이 엮이는 경우와 연결된다.
PDE에서의 원인 규명이 EO에서도 이미 성립했다는 뜻은 아니며, **검증 설계의 재사용**이다.

8/22 repair는 2–4개의 상관구조를 깨는 추가 시뮬레이션에서 교정 효용을 보였지만,
IID 악화 허용치는 2/4 arm에서 실패했다. 조건수로 효과 크기를 설명한다는 가설도 0/4로 반증됐다.
`dopt` 이름은 m≥2에서 엄밀한 공동 D-optimal 해법이 아니라 D-opt-inspired 정책으로 해석해야 한다.
4개 arm의 결과를 “보편적 repair 법칙”으로 승격하지 않는다.

## 3. 관련 문헌이 실제로 뒷받침하는 범위

| 1차 출처 | 확인한 연결 | 우리에게 남는 일 |
|---|---|---|
| [Poseidon, NeurIPS 2024](https://arxiv.org/abs/2405.19101) | PDE trajectory와 lead-time 조건부 전이 | 물리 변수·시간 단위·경계조건을 정의해야 함 |
| [Poseidon → Martian weather, 2026 preprint](https://arxiv.org/html/2602.15004v1) | OpenMARS 재분석의 바람·기온 예측으로 PDE FM 적응 | 위성 분광 임베딩이나 지표 변화로의 전이는 미검증 |
| [EarthNet2021](https://arxiv.org/abs/2104.10066) | S2 영상 + 지형 + 기상 조건의 미래 지표영상 예측 | 기상 조건부 EO 예측 자체는 새로운 문제가 아님 |
| [GreenEarthNet / Contextformer, CVPR 2024](https://arxiv.org/abs/2303.16198) | 지역 시각 맥락 + 기상 시계열로 식생 변화 예측 | 기존 시각·기상 fusion 기준선을 반드시 비교 |
| [FloodCastBench, Scientific Data 2025](https://www.nature.com/articles/s41597-025-04725-2) | 실제 지역의 입력과 촘촘한 수심 시뮬레이션 | 시뮬레이션 정답과 독립 실관측 평가를 분리 |
| [NeuralGCM, Nature 2024](https://www.nature.com/articles/s41586-024-07744-y) | 명시적 dynamics solver와 학습 모듈의 결합 | 모든 계산을 FM에 맡길 필요는 없다는 설계 근거 |
| [4DVarNet-SSH, GMD 2023](https://gmd.copernicus.org/articles/16/2119/2023/) | dynamics prior + 관측 연산자로 위성 고도계 자료 보간 | 양쪽 시점 관측을 쓰는 smoothing과 causal update를 구분 |
| [위성 홍수영역 동화, HESS 2014](https://hess.copernicus.org/articles/18/4325/2014/) | 2D 홍수 모델과 원격관측 영역 결합 | 홍수 관측동화 자체를 최초라고 주장할 수 없음 |

화성 논문은 4개 화성년, 약34GB 자료와 18개 수직층 확장 모델을 사용해 같은 구조 scratch 대비
validation loss 34.4% 감소를 보고한다. 훈련은 단일 A100/H100에서 중앙값13GPU시간이었다.
원자료는 5° 격자의 재분석이다. 이것은 **PDE 사전학습→대기 상태**의 직접 사례이지, EO 연결의
성능이나 우리 작업 비용 추정치가 아니다. 본문의 single-level 백분율은 표와 불일치해 인용하지 않았다.
[원문 §3–4](https://arxiv.org/html/2602.15004v1).

FloodCastBench는 4개 사건을 다루며 저장 간격은300초다. 공개 설정은 영국·호주30/60m,
저해상도 설정480m이고 파키스탄은 입력을480m로 낮춰 직접 계산한다.
강우 원자료는0.1°·30분 IMERG Final을 재표본화한 것이다. **30m·5분 강우를 실제 측정한 자료가 아니다.**
또한 관측 홍수지도에 맞춰 simulator를 보정했으므로 같은 지도로 평가하면 독립 검증이 되지 않는다.
데이터가 네 사건이라는 점도 타일 수로 해결되지 않는다. [원문 Methods/Data records](https://www.nature.com/articles/s41597-025-04725-2).

ERA5 역시 hourly이지만 통상 제공 격자는0.25°다. 시간 단위의 기상 조건은 줄 수 있어도
40m 토큰별 hourly 정답을 제공하지 않는다. 재분석은 실시간 발행 예보와 구분한다.
[ECMWF/Copernicus 공식 데이터 계약](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).

PDEBench의 shallow-water는 synthetic radial dam-break다. 파일 명세는128²·101시점·1,000표본·
수위1변수이며 실제 지역의 DEM·강우·위성 관측이 붙은 자료는 아니다.
수위 한 장은 일반적으로 유량까지 포함한 완전한 동역학 상태가 아니므로, 임의 중간 시점에서
같은 수위만으로 다음 상태가 유일하게 정해진다고 가정하면 안 된다.
[PDEBench 본문 및 데이터 명세](https://arxiv.org/html/2210.07182).

## 4. 실제로 연결할 때 필요한 최소 구조

### 4.1 임베딩과 물리 상태를 분리한다

예를 들어 홍수에서 물리 상태는 수심·유량이며, OlmoEarth latent의768개 채널은 그런 단위로
정의되어 있지 않다. latent에 바로 shallow-water PDE residual을 걸면 물리적으로 무엇을
제약하는지 알 수 없다. 학습으로 물리 해석 가능한 상태/decoder를 만들거나, 물리 상태를 별도로
유지하고 latent를 conditioning하는 구조가 필요하다.

설계 예:

```text
사건 전 EO 영상 ─ OlmoEarth ─ 지역 특성 c(x) ──────────┐
DEM·토지피복·강우·유입/경계조건 ─────────────────────┤
이전 물리 상태 + 실제 Δt ───────────── Poseidon/solver ─ 예상 상태
                                                         │
새 S1/S2·수위 관측 ─ 관측 모듈 ─────────────── 비교·교정 ──┤
                                                         ↓
                                      상태 추정 + 출처별 근거 기억
                                                         ↓
                                      VLM: 위치·변화·예측·보류 설명
```

물리 모델은 수심을 내는데 관측 모듈은 물 영역 확률만 내는 경우, 둘을 직접 빼지 않는다.
수심→관측 가능한 침수영역으로 바꾸는 관측 연산자 H를 정의해 같은 공간에서 비교한다.
광학/SAR의 탐지 한계와 구름·센서 잡음까지 모델링해야 하며, 단일 이진 mask가 수심·유량을
유일하게 결정한다고 가정하지 않는다. 불확실성이 남는 것이 정상이다.

첫 연구에서는 모든 multispectral band를 물리적으로 rendering하려 하지 않는다.
**침수영역 같은 관측 가능한 상태**부터 연결한다. roughness를 EO로 추정하더라도 실제 계수를
식별했다는 주장에는 별도 검증이 필요하며, 단순 조건부 adapter로만 보고할 수 있다.

### 4.2 시간 조건은 ‘몇 시간이 지났는가’만이 아니다

- 외부 강우가 바뀌는 계라면 구간 전체의 강우/유입 이력, 계절·시각, 경계조건도 필요하다.
- 같은24시간 후라도 비가 온24시간과 오지 않은24시간은 다른 입력이다.
- all2all 쌍을 만들 때도 해당 구간의 forcing을 같이 잘라야 한다.
- 서로 다른 simulator의 정규화 시간0.1을 모두 실제1시간으로 취급하지 않는다.
- 조각을 이어 예측하는 일관성은 동일한 forcing 이력·경계·충분한 상태라는 조건하에 평가한다.
  이 성질을 임의의 관측 latent에 강제로 적용하면 오히려 잘못된 제약이 될 수 있다.

### 4.3 보간과 온라인 갱신은 분리한다

- **사후 보간/smoothing:** 1일과3일 관측을 모두 보고2일 상태를 추정. 미래 관측 사용을 허용한 과제.
- **온라인 filtering/forecast:** 1일 현재 가진 자료만 쓰고,2일 관측이 도착하면 교정.
- 보간으로 생성한2일 임베딩/수심은2일의 실제 관측 증거가 아니다.
- ‘업데이트’는 우선 test-time 기억·상태 갱신이다. 가중치는 offline 학습 후 고정한다.
  관측마다 LoRA/전체 모델을 자동 재학습하는 별도 문제를 한꺼번에 추가하지 않는다.

## 5. 현재 VLM 방향과 어떻게 만나는가

목표는 물리 예측을 그럴듯한 문장으로 꾸미는 것이 아니다. 예를 들면 아래 세 종류를 분리해
말하고, 새 영상이 왔을 때 필요한 주장만 바꾸는 것이다. 다음은 실제 장소 결과가 아닌 출력 설계 예다.

- 관측: “동쪽 농경지에서 물 영역이 넓어졌다.” 영상시각·영역·obs ID 포함.
- 예측: “주어진 강우 조건에서는 하류 저지대로 확대될 가능성이 있다.” 예보시각·모델·조건 포함.
- 보류/수정: “서쪽은 관측이 부족하다 / 새 SAR 관측 때문에 이전 침수 추정을 수정한다.”

VLM 학습은 region token과 물리 상태 요약을 언어에 정렬할 수 있다. 다만 scalar를 문장으로
바꾸는 데 VLM이 필수는 아니다. `규칙+템플릿`, `metadata-only LLM`, `EO-only VLM`,
`full-history VLM`을 비교해 시각적 판단·근거 수정에서 실질적인 추가 가치를 보여야 한다.
외력과 현상 사이의 연관을 사용했다고 특정 사건의 원인을 입증한 것도 아니다.

로봇형 확장은 마지막에 자연스럽게 붙는다. 후보 물리 설명들이 엇갈리는 곳의 다음 관측을
선택하는 active sensing이다. 초기에 필요한 것은 이동 로봇 구현이 아니라 **어느 관측이
잘못된 믿음을 고치는가**를 측정하는 것이다. 기존8월 source selector의 실패를 숨긴 채
검증된 selector로 재사용해서는 안 된다.

## 6. 다음 한 가지 실험: 짧은 홍수 상태 갱신 pilot

**질문:** 동일한 희소 관측·DEM·강우를 줄 때, PDE 사전학습과 EO 지역 맥락이 다음 관측의
침수영역 및 중간 물리 상태 추정을 각각 개선하는가?

아래는 제안이다. 자료 다운로드·EO 조인·학습을 아직 실행하지 않았다.

1. **학습 전 계약 검사.** 공개 홍수 데이터에서 작은 구간을 골라 CRS·격자·단위·수심/forcing 시각·
   경계조건·보정에 쓰인 관측 ID를 확인한다. 같은 위치의 사건 전 EO를 실제 join할 수 있는지부터
   본다. 입출력만 H100/H200 한 장으로 smoke하고 throughput/메모리를 측정한 뒤 예산을 정한다.
2. **두 평가를 독립 구성.** simulator trajectory를 드문 관측처럼 가리는 실험은 controlled test로
   명시한다. 실제 위성으로 만든 관측은 별도의 observation-gap 실험이다. 둘의 점수를 합치지 않는다.
3. **작은 2×2.** 같은 scOT 구조 scratch/Poseidon 초기화 × raw physical context/raw+frozen EO context.
   EO 효과와 PDE pretraining 효과를 분리한다. 파라미터·업데이트·데이터 예산을 가능한 한 맞추고
   동일 용량의 무정보/위치-shuffled EO branch도 확인한다. 이후 EO encoder pretraining까지 주장하려면
   같은 encoder scratch/pretrained의 별도2×2가 필요하다.
4. **강한 비교.** persistence, 허용된 사후 과제의 선형보간, Δt-GRU/ConvGRU, 물리 predictor+단순
   관측 교정, scratch scOT. latent cosine만이 아니라 수심 오차·침수 IoU/경계·도달시각·불확실성을 본다.
5. **외부 분할.** 한 trajectory의 모든 all2all 쌍은 같은 split. 공간 buffer를 두고 사건/유역을 통째로
   나눈다. 네 사건만으로는 광범위한 일반화의 확증이라고 하지 않는다. 새로운 독립 관측이 필요하다.
6. **실패에 따른 결론.** raw physics만으로 동급이면 EO 기여를 제거한다. scratch와 동급이면 Poseidon
   기여를 제거한다. simulator에서만 개선하면 sim-to-real/EO 성능 주장을 하지 않는다. 이 관문을
   통과하기 전에7B VLM 학습까지 확장하지 않는다. 수치 문턱은 결과를 보기 전 별도 등록한다.

짧은 간격은 처음부터 ‘시간 단위’로 고정하고, 해상도와 예측 horizon을 실제 자료에 맞춰 선언한다.
구름/누락을 인위적으로 만드는 경우에도 센서별 현실적인 관측 가능성을 별도로 평가한다.
외력 제공은 (a) 사후 실제 강우를 아는 hindcast, (b) 발행 시점 예보만 쓰는 forecast로 분리한다.
Final rainfall/reanalysis와 사건 종료 후 보정된 simulator로 온라인 성능을 주장하지 않는다.

### 실제 EO 일반화가 우선이라면: 별도의 저위험 경로

시간당 물리 상태라는 목표를 내려놓고 **날씨 정보를 더 주면 실제 EO를 더 잘 읽고 예측하는가**부터
묻는다면 EarthNet2021/GreenEarthNet 계열이 더 직접적이다. EarthNet2021은20m·5일 간격·
4개 band의 S2와 기상/DEM을 제공한다. future weather는 실시간 예보가 아니라 사후 E-OBS를
사용하므로 oracle-weather-conditioned 평가다. hourly 실험이나 Poseidon weight transfer의 증거와
합치지 않는다. [EarthNet2021 §4](https://arxiv.org/html/2104.10066).

여기서는 같은 구조의 weather/no-weather prior와 실제 관측 도착 시 갱신 효과를 지역 holdout으로
비교한다. 다만 현재 OlmoEarth 추출 계약은12-band이므로4-band 배포본을 그대로 넣어도 된다고
가정하지 않는다. 원래 S2 bands 재취득 또는 명시적 missing-band 처리·비교 기준을 먼저 확인한다.
이 경로가 **현재 EO/VLM 본 논문의 일반화 평가에는 더 적합**할 수 있고, 홍수 pilot은 **Poseidon과
시간 단위 물리의 연결 가능성**을 시험한다. 두 경로를 한 번에 대규모 학습하자는 제안이 아니다.

### 가장 중요한 반사실 대조

지역ID·설명 문구만 바꾸면 관측된 수역 판단이 따라 바뀌는가? 물리적으로 허용된 강우/경계조건을
바꾸면 예상 상태는 달라지는가? 관측이 예측에 반대할 때 수정하는가? 흐린 영상만 들어와도
확신하는가? 앞 둘은 같지 않다. 정확한 metadata를 쓰는 것과 metadata를 지름길로 쓰는 것을
구분하는 실험이 이전 Poseidon 연구에서 가장 직접적으로 가져올 수 있는 부분이다.

## 7. CVPR 본체로 삼을 수 있는 조건

단순한 Poseidon+OlmoEarth+LLM 결합이나 “weather-conditioned EO forecasting” 자체는 부족하다.
후자는 이미 CVPR 관련 선행이 있다. 현재 남는 가설은 다음처럼 좁히는 것이 낫다.

> 불완전한 물리 예측과 불규칙·저품질 EO 관측이 충돌할 때, 지역별 상태와 언어적 주장을
> 올바른 근거에 따라 수정하는 방법.

일반 data assimilation 자체도 새롭지 않다. 논문이 되려면 기존 filtering/fusion 및 최신 EO VLM보다
새 지역에서 유리하고, 틀린 물리 prior·잘못된 context·관측 누락에서도 **정확한 수정과 보류**를
보여야 한다. 언어 평가에는 사람이 검증한 claim·위치·시간·근거가 필요하다.
시뮬레이터에서 생성한 설명만으로 실제 환경 판단의 사실성을 확정할 수 없다.

물리 prior가 실제로 도움이 없다면 기존 지역 속성 언어 접지 노선을 유지한다.
도움이 있다면 해당 상태 갱신을 VLM의 동적 확장으로 삼는다. 어느 경우에도 슬럼·벌목·불법개발 같은
사회적/인위적 사건 전반이 PDE로 예측된다고 확대하지 않는다.

## 8. 지금 재사용할 것과 가져오지 않을 것

| 재사용 가치가 높은 것 | 그대로 옮기지 않을 것 |
|---|---|
| Poseidon/scOT 로딩·same-architecture 비교 하네스 | Darcy1채널 wrapper를768채널EO dynamics로 간주 |
| manifest·seed·split·실패기록·solver 비용 기록 | residual-clock pseudo-time을실제시간으로 해석 |
| 실제 time-conditioned rollout 코드와시간-shuffle 대조 | 기존4EO사건과4홍수사건의타일수를독립사건수로 간주 |
| 계수·대리변수 의존을분리하는반사실 설계 | 이전toy repair를EO에서이미검증된법칙으로 인용 |
| 현재 OlmoEarth공간token·관측ID·언어projector | simulated/predicted상태를observed증거로 저장 |

핵심은 **버린 실험을 전부 부활시키는 것이 아니라, ‘물리적으로 가능한 변화’와 ‘실제로 본 증거’를
구분하는 새 문제에 검증 도구를 재사용하는 것**이다.
