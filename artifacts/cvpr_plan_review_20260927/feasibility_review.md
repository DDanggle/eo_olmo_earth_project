# 독립 실행 가능성 검토 — OlmoEarth 역량 확장, CVPR 2027

검토일: 2026-09-27. 역할: scientific feasibility reviewer. 근거는 공식 OlmoEarth clone `0497dfbb6711ded4e6bf10cf089fc1e4d58c186b`, OE2 기존 설계, OE3 공식 평가 inventory, 서버의 읽기 전용 inventory다. 이 검토는 실행·성능 검증이나 채택 확률 추정이 아니다. 첫 작성 시 root의 새 draft는 아직 없었다. 아래의 규모·문턱은 **검토자가 제안한 후보**이며 동결 사전등록이 아니다.

## 판정

**조건부 진행.** OlmoEarth에 적은 라벨로 새 개념·작업을 배울 수 있는 역량을 더한다는 연구 목표는 큰 연구로 확장할 수 있다. 단, 단일 VQA 미세조정 개선이나 기존 지도 이름을 문장으로 바꾼 학습만으로 이를 입증할 수 없다. 다음 세 가지를 연결해야 한다.

1. 충분한 고유 지역의 다중모달 관측·정합 감독으로 공통 표현을 실제 학습한다.
2. 학습에 쓰지 않은 개념과 작업 조합에서 적은 지원 예제로 적응한다.
3. 개선된 가중치가 실제 VLM 활용과 독립 EO 작업 양쪽에 도움이 되는지 확인한다.

encoder-only 평가는 연구 전체의 크기를 결정하는 것이 아니라 성능 향상의 소재를 확인하는 감사다. 주 연구는 새로운 활용 역량과 그 학습 방법이며, 모델 규모나 명목 instruction 수가 그 증거를 대신하지 않는다.

## Critical: 지금 먼저 해결해야 하는 것

### C1. 자료와 학습 환경은 아직 본 실행 준비가 아니다

서버 master 환경의 확인된 package metadata는 minimal 0.0.6, torch 2.13.0이다. full pretrain, ai2-olmo-core, geobench distribution metadata는 없었다. 이는 모든 다른 가상환경까지 없다는 뜻은 아니다. `geobench2` 디렉터리는 존재하지만 여섯 dataset 하위 디렉터리 이름만 확인했다. 완전성, split, labels, CRS와 날짜 정합은 확인되지 않았다. 예정 경로의 `geobench`, `floods`, `mados`, `pastis_r`, `research_benchmarks`는 없었으나 서버 전체 부재로 확대 해석하지 않는다.

공식 HF 번들은 `config.json + weights.pth`이고 현재 full evaluation sweep의 `--checkpoint_path`는 distributed checkpoint 로딩 경로다. 로더 bridge/변환과 고정 입력 동등성 확인 없이 대형 학습을 시작해서는 안 된다. 현재 v1-Tiny 파일의 checksum은 과거 기록과 일치했다. 확인 출처: `/private/tmp/oe3_capability_program_20260927/server_inventory_v0.json`.

### C2. 원래의 목적함수 대조군은 BEN S2만으로 재현되지 않는다

공식 v1은 파생 지도 target의 patch discrimination과 view 간 InfoNCE를 사용한다. WorldCover·OSM·SRTM·canopy·CDL·WorldCereal는 이미 의미·공간 감독이다. 따라서 같은 target을 텍스트로 읽어준다는 사실 자체는 새로운 정보나 독립 신규성을 보장하지 않는다. 제안 학습이 **동일 target의 직접 감독 및 평범한 instruction 공동학습**보다 도움이 되어야 한다.

### C3. 큰 목표와 실행 표가 분리되어야 한다

10k/100k/1M을 모두 독립 관측 수처럼 쓰면 실행량과 과학적 효과가 혼동된다. `고유 site 수 / 고유 episode 수 / 실제 sensor acquisition 수 / 질문 수 / 학습 노출 수`를 다섯 열로 따로 기록해야 한다. 공식 pretraining 공개 자료도 그중 어떤 단위를 제공하는지 inventory에서 확인해야 한다. 같은 관측의 crop·paraphrase로 1M을 만들 수 있으나 그것은 1M 독립 지역이 아니다.

## 규모와 데이터 출처에 대한 실행 판단

| 단계 | 제안 최소 준비량 | 의미 | 다음 단계 조건 |
|---|---|---|---|
| 정합·로더 시험 | 32–128개 train 관측 | 단위·밴드·mask·날짜·loss·gradient 검증 | 완전한 export/reload와 baseline forward |
| 개발 | 10k 고유 episode 목표, 여러 질문은 별도 계수 | 고정 region split, 처리량·학습 안정성·모델 선택용 | 같은 자료의 비교가 정상 작동, dev 성능/유지 성능 확인 |
| 주 비교 | 100k 고유 episode 목표 | 충분한 다양성에 대한 main recipe 비교 | 출처·권리·정합·compute 확보가 실제 확인됨 |
| 큰 instruction 규모 | 100k episode에서 1M question/example 생성 가능 | 질문 다양성/작업 구성 규모 | API 전수 호출 없이 source-grounded 변환과 표본 검수 |
| 1M 고유 episode | 이번 확증의 필수조건으로 두지 않음 | 별도의 데이터 확보·연산 연구 | 100k 결과와 추가 예산 확보 후 |

권장 출처 역할은 다음과 같다. 공식 사전학습 subset은 원 목적의 fair replay에, 공개 dense/field 자료는 확인 가능한 개념·영역·측정 감독에, 독립 공개 benchmark는 성능 평가에 쓴다. 동일 지역 또는 동일 source label을 supervision와 holdout 양쪽에서 쓰지 않도록 조인 장부가 필요하다. source의 양이 많아도 cloud/valid support/target date가 안 맞으면 사용 가능 수가 급감할 수 있다.

100k 목표를 못 채웠을 때 10k를 열 번 반복한 것을 100k 자료 실험으로 바꾸지 않는다. 10k 반복과 100k 고유자료를 같은 노출수로 비교하는 대조는 유용하지만 별도 실험이다. API 5만 원으로 100k–1M 이미지 판독을 약속하지 않는다. 기존 의미 지도에서 결정적으로 만들 수 있는 supervision를 먼저 사용하고, teacher는 언어 다양화·어려운 예제 검토에 제한한다.

## 최소 main experiment matrix

모델은 첫 개발에서 pinned v1-Tiny를 쓰고, 주 연구에서는 사용 가능한 최신 공식 버전의 한 규모를 **사전에 고정**한다. Tiny 개발 결과를 Base의 결과처럼 합치지 않는다. baseline마다 원 checkpoint, 입력 sensor, 데이터, supervision 사실, token budget, optimizer 노출수, 평가 절차가 같아야 한다.

| 조건 | 학습 | 분리하는 효과 | 권장 main seed 수 |
|---|---|---|---:|
| O | 공개 모델 그대로 | 원점 | 학습 0회; 평가 지원집합 seed 반복 |
| C | 원 목적 continued pretraining | 추가 데이터·추가 학습 효과 | 3 |
| S | C + 동일한 영역·측정·개념의 직접 지도 | 정답 추가 효과 | 3 |
| I | C + 평범한 grounded instruction 공동학습 | EO–VLM 결합·언어 정렬 효과 | 3 |
| P | I와 같은 감독에 제안 학습 원리 적용 | 제안 방법 자체 | 3 |

**100k에서 핵심은 12회 학습이다.** O는 학습 실행수가 아니다. I와 P가 주대조이고, S와 P가 언어 활용의 필요성을 검증한다. C는 원 목적 대조를 지키는 조건이다. P가 외부 teacher의 추가 정보·더 많은 token·더 긴 업데이트를 독점하면 이 matrix는 공정하지 않다.

추가 비교는 원인 주장에 필요한 것만 둔다.

- frozen EO + instruction 연결부 학습은 원 encoder가 이미 충분한지 확인하는 실용 기준선이다. 적어도 개발에서 실행하고, P의 성능이 encoder 변경에 기인한다고 강하게 주장하면 같은 3 seed의 추가 3회를 예산에 넣는다.
- 10k에서는 I/P 각 3 seed를 그대로 main으로 넣으면 추가 6회다. 자원이 부족하면 10k는 recipe 개발로만 쓰고 **규모에 따른 방법 이득 증가 주장**을 하지 않는다.
- 1M 고유 episode는 필수 표에서 제외한다. 1M instruction의 양은 실제 고유 관측 수와 함께 보고한다.
- 새 reader를 활용한 전이는 가장 강한 기준선과 P만 평가한다. 새 reader의 학습 조건을 같게 하고 적응 seed 수를 별도 장부에 기록한다. 기존 reader 복잡도를 곱한 전 factorial은 피한다.
- 원 표현의 독립 EO 평가와 support-set episode 평가는 대형 pretraining 실행수와 구분하되, embedding 생성·probe fit 비용을 반드시 더한다.

12회조차 처리량 확인 전 실행을 약속할 수 없다. 약 7주의 일정에서 24–30회 대형 학습과 reader·teacher·human 모든 교차 실험을 기본 약속으로 두는 것은 부적절하다.

## Few-shot 새 개념 × 새 작업: 무엇을 hold out할 것인가

권장 평가 표는 `seen concept/seen task`, `held-out concept/seen task`, `seen concept/held-out task`, `held-out concept/held-out task`의 네 칸이다. 이는 own adaptation corpus 기준이며 공개 foundation pretraining에서 완전히 미노출이라고 단정하지 않는다.

- 개념: 철자만 바꾼 동의어를 new concept으로 세지 않는다. ontology 수준에서 subtype/target family를 통째로 보류한다. 원 모델이 이미 WorldCover를 본 점을 공개한다.
- 작업: paraphrase나 출력 JSON 형식 교체는 새 작업이 아니다. 예를 들어 학습에서 존재·영역을 감독하고 새로운 support 예제로 검색·비교 또는 다른 측정 target을 적응하는 경우처럼 목표의 차이가 명확해야 한다. 기존 dense mask에서 면적을 계산하면 그 수학적 의존성을 숨기지 않는다.
- 지원 예제 수: k=0/5/20/100처럼 사전에 고정하고, k의 단위를 **고유 site 또는 관측 episode**로 한다. 질문 여러 개를 라벨 k개로 세지 않는다.
- 각 support/query split은 site/event group 단위로 분리한다. 같은 지역의 인접 tile나 같은 사건의 시점이 양쪽으로 섞이지 않게 한다. support sampling seed를 모델 사이에 공유한다.
- 최소 두 target family와 최소 두 실질적 task family를 둔다. 각 family의 query는 전체 human120과 별도로 공개 독립 정답으로 수백–수천 episode를 확보하는 방향이 적절하다. 정확한 최소 표본 수는 dev에서 관측한 모델 불일치·지역 군집으로 power simulation 후 확정해야 한다.
- 여러 support draws가 같은 query를 재사용하면 독립 test n이 늘어난 것이 아니다. training seed, support draw, query site를 구별해서 요약한다.

개념과 작업을 동시에 보류한 네 번째 칸이 의미 있게 작동하면 논문을 작고 고정된 benchmark fine-tuning에서 확장시킬 수 있다. 반대로 이 칸이 정답 형식 오류로만 실패하면 모델 능력과 interface 실패를 분리해 봐야 한다.

## Compute는 실측으로만 견적 낸다

2×H200 사용 가능성은 예산 자원이지 24시간 전용이라는 증거가 아니다. 현재 점유는 실행 직전에 확인하고 다른 작업을 종료하거나 메모리만 보고 동시 학습을 배치하지 않는다.

각 distinct recipe(C/S/I/P, frozen/joint, 해상도, timesteps, token budget)에 대해 warm-up 뒤 대표 길이의 최소 100–200 optimizer steps를 측정한다. wall seconds/step의 median·p90, 유효 sample 또는 token/s, peak allocated/reserved memory, data loading 대기, checkpoint 시간, validation 시간을 기록한다. 학습 비교 자체의 fair budget은 examples/updates와 FLOPs 또는 실제 GPU time을 모두 보고한다. 둘 다 완전히 같을 수 없으면 주 비교 기준과 비용 차이를 명시한다.

`예상 wall hours = 계획 optimizer steps × 실측 seconds/step ÷ 3600 + checkpoint/eval/IO 시간`

`예상 GPU hours = 각 실행의 world size × wall hours의 합`

계획 steps는 `unique episodes × epochs ÷ effective global batch`에서 반올림하며 gradient accumulation을 포함한다. 생성 데이터가 episode당 여러 instruction이면 실제 학습 길이 분포까지 포함한다. 단일 GPU 처리량을 두 배 한다는 방식으로 2 GPU 성능을 추정하지 않는다. 현 시점에서 구체적 GPU-hour 숫자를 주장할 근거는 없다.

후속 실행 표에는 모든 primary arm, 3 seeds, fresh reader, EO probes, 실패 재실행 여유를 포함한다. GPU 자원 가용성의 보수적 범위로 일정을 계산하고, 결과를 보고 유리한 조건만 반복하는 대신 locked test 전에 규모를 줄인다.

## Human120과 100만 원의 역할

기존 예산은 판독 60인시 72만 원, 도메인 검토 4인시 16만 원, API 상한 5만 원, 예비 7만 원이다. 단가·인력은 계획 가정이고 GPU/개발 인건비는 포함하지 않는다. 독립 120개에 각 두 사람×8분이면 32인시가 맞는다. 복잡한 다중시점 전문가 해석까지 동일 시간에 된다고 가정하지 않는다.

120개는 **출처 라벨의 의미·관측 가능성·설명 근거 오류를 점검하는 작은 독립 audit**다. 여러 concept×task×k cell의 정확도를 정밀하게 비교하거나 2–3%p 차이를 입증하기에는 부족하다. 독립 binary p=.75 가정에서 단일 비율의 대략적 95% 반폭만 약 7.7%p다. 실제 paired 차이는 모델 불일치와 site clustering에 따라 달라진다.

사람은 블라인드 공통 reference를 만들고 모든 모델에 같은 reference를 적용한다. 어느 모델이 이기는지 본 뒤 어려운 예제나 unresolved를 삭제하지 않는다. human train/dev/test의 관측은 분리하고, 독립 판독과 AI 답을 보고 한 수정은 구별한다. 판독자 합의와 image observability를 각각 보고한다. 합의율이 높다고 센서로 알 수 없는 원인 추정이 정답이 되지는 않는다.

## 성공 기준 후보 — 반드시 locked test 전에 확정

새 결과를 보지 않은 상태에서 검토자가 제안한 운영 후보다. 데이터 의미·pilot label 오차·처리량·power를 확인한 뒤 한 번 동결한다. test 결과에 맞춰 문턱을 낮추지 않는다.

1. **주 효과:** 한 개의 primary capability endpoint를 먼저 정한다. 예를 들어 held-out concept×task에서 동일 support budget의 검증 가능한 성공률 P−I가 3%p 이상이고 paired site/event bootstrap 95% 구간 하한이 0보다 큰 것을 후보로 둘 수 있다. 성공의 위치/수치/분모 조건은 원 정답의 해상도와 불확실성에 맞춰 dev에서 정한다. retrieval이면 mAP, dense task면 mIoU 등 자연 지표를 쓰고 다른 척도를 임의 평균하지 않는다.
2. **직접 감독 대비:** 동일 evidence의 S와 비교해 사전 지정한 capability 이득을 보여야 언어 학습의 역할을 주장할 수 있다. 이득이 I 대비에만 있고 S 대비에는 없으면 표현 일반화의 새로운 원리 주장보다 interface/recipe 개선으로 범위를 좁힌다.
3. **기존 EO 능력 유지:** 두 독립 EO task에서 non-inferiority를 평가한다. 예컨대 원 O 대비 −1%p를 후보 margin으로 두되, 해당 metric의 seed·label 불확실성과 실용 의미를 보고 test 전에 고정한다. 단순히 유의한 하락이 없다는 것만으로 유지했다고 하지 않는다.
4. **재사용:** 새 reader 또는 새로운 저용량 readout에서도 이득 방향이 남는지 확인한다. 여러 reader 전체 일반화를 주장하려면 한 번의 성공 이식만으로 부족하다.
5. **표현 기여:** decoder/connector를 초기화하거나 제거한 뒤 평가한다. 효과가 공동학습 연결부에만 남으면 encoder 개선 주장을 보류한다. 이것이 primary 논문 크기를 축소해야 한다는 뜻은 아니나 사용자의 OlmoEarth 기여 목표는 충족하지 못한다.

3 seeds는 최소 재현성 확인이지 seed 모집단의 안정적인 검정력을 보장하지 않는다. seed별 값과 paired 차이를 모두 보여주고 test-site uncertainty와 training-seed variability를 따로 보고한다. 같은 query를 반복한 결과를 sample 수로 부풀리지 않는다. primary contrast 순서와 보조 비교의 다중비교/탐색 상태를 사전에 정한다.

## 중단·피벗 규칙

- 데이터 조인의 위치/날짜/유효 support가 성립하지 않으면 그 source 사용을 중단한다. 단순한 질문 생성으로 대체하여 계속하지 않는다.
- 공식 objective 재현에 필요한 target이 없거나 baseline loss/gradient가 불건전하면 대형 P 학습을 보류하고 환경·data contract를 먼저 수정한다.
- 10k dev에서 P의 gradient는 정상이나 I/S 대비 반복 가능한 효과가 없으면 100k를 자동 확대하지 않는다. 정보·구성·최적화 중 무엇이 달라지는지 hypothesis를 수정하고 새 dev 실험으로 기록한다.
- P 효과가 raw-label S와 같으면 언어 생성 확대를 중단하고 task composition/representation 학습의 필요성을 다시 검토한다.
- 효과가 알려진 개념·작업 칸에만 있고 held-out 칸에 없으면 새로운 역량 습득 일반화 주장을 닫는다. 고정 과제의 효율 개선으로 축소할지 사용자와 연구 가치 기준으로 결정한다.
- EO 기존 능력 저하가 margin을 벗어나면 replay·학습 범위·recipe를 dev에서 수정하고 다시 검증한다. VLM 점수 상승만으로 유리하게 결론 내리지 않는다.
- compute 실측상 확증과 분석 시간을 확보할 수 없으면 seed를 사후 선택하거나 test를 여러 번 열기보다 claim/scale을 사전에 축소한다. 1M 고유 관측은 가장 먼저 후속으로 옮긴다.

## 일정에 관한 확인된 사실

공식 CFP와 Dates가 모두 **등록 2026-11-10 AoE, 본문 2026-11-16 AoE, supplementary 2026-11-23 AoE**를 명시한다. 확인일은 2026-09-27이다. [공식 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers), [공식 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates).

현재부터 약 7주이므로, 권장 순서는 첫 1주에 data/runtime/throughput과 freeze할 질문을 확정하고, 다음 단계에서 고정 recipe pilot, 나머지 기간에 main comparison·independent audit·분석·집필을 배치하는 것이다. 이는 처리량 실측 전 일정 가정이다. 마감 자체가 현재 연구를 대형 CVPR 결과로 만들 근거는 아니다. 채택 가능성을 수치로 추정할 자료는 없다.

## 우선순위

**P0:** source-aligned subset과 two-task original baseline를 실제 실행 가능하게 만들고, 새로운 concept×task split manifest를 정의한다. 여기까지는 준비다.

**P1:** 왜 일반 instruction 학습보다 나아져야 하는지 제안 mechanism을 하나로 정하고 C/S/I/P가 같은 자료에서 비교되게 구현한다. 단순 loss 나열이면 아직 방법 연구가 아니다.

**P2:** 10k에서 pipeline와 throughput을 검증한 뒤 100k main 12회, 추가 frozen/reader/scale 비교의 총예산을 확정한다. 새 능력의 독립 평가를 중심으로 두고 encoder-only 감사와 EO 유지 검증을 연결한다.
