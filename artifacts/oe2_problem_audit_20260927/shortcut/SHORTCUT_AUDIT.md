# OE1 질문·클래스 prior 사후 감사

이 분석은 기존 OE1 dev 결과가 나온 뒤 설계한 **사후 탐색적 재집계**다. 원 사전등록, 데이터, 모델, 저장 응답을 바꾸지 않았으며 새 추론을 실행하지 않았다. 학습 데이터의 답 빈도만 사용해 baseline을 만들었다. 동률과 train에서 보지 못한 질문은 모두 `no`로 고정했다. 비교는 저장된 free-vocabulary 첫 토큰의 `parsed`를 사용했다. 강제 yes/no 확률 점수로 바꾸지 않았다.

**문제는 분명하다. 전체 dev의 68.36%는 영상 없이 클래스별 다수 답만으로 얻는다.** 따라서 frozen의 68.36%나 joint의 72.27%를 그대로 영상 이해 성능의 크기로 해석하면 안 된다. 다만 같은 질문에서 양성과 음성을 구분해야 하는 작은 부분집합에서는 joint가 prior를 넘어서는 신호를 보였다.

| 시스템 | 동일 dev 256문항 pooled BA | 정답 수 |
|---|---:|---:|
| train 클래스별 majority, 문구로 19클래스 정규화 | 68.36% | 175 |
| train 동일 질문 문자열 majority | 61.72% | 158 |
| 학습된 blind | 66.80% | 171 |
| frozen EO + projector | 68.36% | 175 |
| joint EO + projector | 72.27% | 185 |

여기서 질문 majority는 dev 240/256문항의 정확한 문자열을 train에서 봤다. 미관측 16문항에는 `no`를 냈다. train에서 본 240문항만 대응 비교해도 BA는 질문 majority 61.49%, 클래스 majority 67.28%, blind 66.29%, frozen 68.97%, joint 71.31%다. joint의 전체 증가는 클래스 majority 대비 27문항을 새로 맞히고 17문항을 잃은 순증 10개(3.91%p), frozen 대비 28개 개선·18개 악화로 순증 10개, blind 대비 47개 개선·33개 악화로 순증 14개(5.47%p)다. 클래스 majority와 frozen은 총점만 같고 동일한 답을 낸다는 뜻은 아니다.

**질문을 통제한 평가:** dev의 동일 질문 문자열+동일 클래스 층 중 source yes/no가 모두 존재하는 26개 층, 68문항(33 yes / 35 no)만 평가했다. 각 층에서 BA를 계산한 뒤 층을 동일 가중했다. 질문만으로 일정한 답을 내는 baseline은 클래스별 양성비 차이로 이득을 얻을 수 없다. 문항 다운샘플링은 하지 않았다.

| 시스템 | 26개 질문 층 평균 BA | 동일 68문항 pooled BA |
|---|---:|---:|
| 클래스 majority | 50.00% | 52.55% |
| 동일 질문 majority | 50.00% | 51.21% |
| blind | 50.00% | 49.13% |
| frozen | 57.37% | 59.96% |
| joint | 66.99% | 67.62% |

blind는 dev의 178개 질문 그룹 전체에서 동일 질문에 동일한 답을 냈다(중복 질문 그룹 58개, 위반 0). 위 26개 층에서 frozen은 7개, joint는 11개 층에서 관측에 따라 답이 달랐다. joint는 동일 68문항에서 frozen보다 순증 5정답(12개 개선·7개 악화), blind보다 순증 13정답(21개 개선·8개 악화)을 얻었다. 이 부분집합은 원 donor-swap 가능 ID 68개와 정확히 일치하지만, 위 표는 **원 real 응답의 동일 source ID 비교**이며 전체 256문항 real과 68문항 swap을 직접 빼지 않았다.

더 넓게 동일 클래스에서 양성/음성이 모두 있는 11클래스·211문항의 클래스 평균 BA는 클래스 majority 50.00%, 질문 majority 42.87%, blind 56.81%, frozen 57.32%, joint 63.15%다. 나머지 8클래스·45문항은 dev에서 한 답만 존재해 해당 클래스 BA를 NA로 남겼다. 전체가 yes/no 128/128이라고 클래스별로 균형적인 것은 아니다. 예를 들어 train의 inland wetlands는 18 yes/0 no, coastal wetlands는 0 yes/14 no다. dev marine waters는 20 yes/1 no, inland waters는 2 yes/20 no라 몇 문항으로 클래스 BA가 크게 바뀐다. joint의 클래스 BA도 mixed forest 56.25%로 frozen 68.75%보다 낮고, complex cultivation patterns 38.33%로 frozen 43.33% 및 클래스 상수 baseline 50%보다 낮다. 여러 클래스가 일관되게 개선됐다는 주장은 성립하지 않는다.

**추가 데이터 계약 문제:** 원 `query_class`가 train 200/1024, dev 52/256문항에서 null이다. 누락은 `transitional woodlands or shrubs` 등 5개 명백한 질문 target 표현 때문이다. 원 파일을 고치지 않고, train 질문에 모두 등장하는 19개 target-phrase 사전을 정해 분석상 정규화했다. 질문의 답이나 patch label은 target 추정에 사용하지 않았다. 원 metadata만 쓴 baseline은 null에 고정 no를 내어 전체 64.84%; 원 class가 있는 동일 204문항에서는 class-majority 68.63%, blind 65.69%, frozen 69.12%, joint 72.55%다. 정규화 coverage와 원 metadata coverage를 구분해 JSON에 보존했다.

허용되는 결론은 **“이 한 번의 개발 실행에서, joint는 train의 클래스 빈도 baseline보다 전체 10정답 많았고, 동일 문구의 반대 답 사례를 일부 더 구분했다”**다. 이것은 관측 내용에 따른 구분의 단서이며, 새로운 사전학습 목적의 우월성·다른 reader 전이·픽셀 근거 위치·변화 이해의 증명이 아니다. EO 입력에는 취득일 인코딩도 있으므로 영상과 계절/날짜 신호를 분리하지 않았다. blind는 별도로 학습된 projector이고 zero는 분포 밖 입력일 수 있어 각각의 차이를 순수한 encoder 효과로 동일시할 수 없다.

표본은 단일 training seed 17, dev 128패치·11 MGRS 그룹이다. 질문 통제 부분집합은 전체의 26.56%, 58패치·9클래스·11 MGRS이고, 26층 중 15층이 겨우 2문항, 6층은 3문항, 5층은 4문항이다. 이 조건부 수치를 전체 모집단으로 일반화하거나 256개 독립 관측으로 유의성을 주장하지 않는다. train/dev MGRS는 분리됐지만 원 EO 사전학습 노출은 불명이고, 정답은 CLC2018 유래 class-presence로 새 전문가 판독이 아니다. 다음 평가에서는 사전에 고정한 질문/클래스별 yes/no 균형, 지역·계절 통제, 반복 seed와 새 분리 평가가 필요하다.

재현 명령:

```sh
python3 /private/tmp/oe2_problem_audit_20260926/shortcut/audit_question_shortcuts.py --out /private/tmp/oe2_shortcut_audit_reproduction
```

산출물 `shortcut_audit.json`은 모든 클래스·질문 층·MGRS의 지원 수와 대응 점수, train만으로 적합한 lookup table, before/after input SHA, 원 응답 9조건 점수 재검산, 자체 합성 검증 9개 결과를 담는다. `dev_predictions.jsonl`은 원 256문항 ID별 baseline/모델 응답과 포함 여부를 남긴다. 입력은 실제 artifact의 data JSONL, 완료 manifest/status, saved items/prompts/predictions/summary 및 기존 receipt audit에 hash로 연결했다. 이는 로컬 결과 파일의 재집계이며, 서버 checkpoint 재로드나 실제 영상을 독립 재검증한 것이 아니다.
