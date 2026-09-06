# 2026-09-06 논문 상태 — “된 것이 없는가?” 재판정

> **최신 판정(23:55)**: MS-112의 deployment-condition 재배열은 좋은 가설이지만 selector
> 결과가 아니다. 비직사각 action matrix·seed 비대칭·상관 episode를 수정한 외부 protocol은
> `docs/MS112_CVPR_AND_KOREA_AUDIT_2026_09_06.md`와
> `config/geobench_cache_action_prereg_v1.json`을 따른다. two-task cache finding은 유지되고,
> CVPR 승격 조건은 held-out task/family regret–cost 개선으로 좁혀졌다.

## 한 줄 판정

**된 것이 없는 상태는 아니다.** 이미 `캐시가 유리한 조건`과 `캐시가 조용히 실패하는 조건`을
측정했고, readout과 단순 모델 용량이라는 두 대안 설명도 약화시켰다. 현재 비어 있는 것은
**현상의 존재**가 아니라, 처음 보는 과업에서 그 조건을 읽고 `캐시/재학습`을 실제로 잘 고르는
규칙의 검증이다.

따라서 지금 성립하는 최소 논문은 **Earth embedding cache의 재사용 경계에 대한 benchmark/
characterization**이고, CVPR main으로 올리는 마지막 핵심은 **공개 untouched Task-3에서의 행동
선택과 실측 비용**이다.

## 무엇이 실제로 닫혔나

| 증거 층 | 현재 상태 | 실제 의미 |
|---|---|---|
| 릴리스가 바뀌면 캐시 identity가 깨짐 | **강** | v1→v1.1/v1.2에서 R@1=0. 관계구조가 높게 남아도 캐시 주소계는 연속적이지 않다. |
| 선형 bridge가 순위를 대부분 복구하지만 옛 결정과 동등하지 않음 | **강** | AP는 약 97% 복구하지만 등록된 fixed-decision gate는 실패한다. `복구됨`과 `그대로 사용 가능`은 다르다. |
| frozen OlmoEarth cache가 raw 학습을 이김 | **강** | Sen12와 Solar 두 과업에서 확인했다. cache-first가 단순 편의가 아니라 성능 선택지가 될 수 있다. |
| 적은 라벨에서 cache-head 적응이 raw 적응보다 유리 | **강** | K=5/20에서 두 과업 모두 A1>A4w 8/8. 다만 A1>A0는 K=5에서 5/8이므로 항상 적응하면 안 된다. |
| 다른 FM도 똑같이 이기는가 | **아니오, 중간 강도** | Clay는 입력 해상도 보정 후 raw와 대체로 동률, Galileo는 raw 미달이다. 캐시 가치는 FM의 보편 속성이 아니다. |
| Galileo 열세가 단순 평균 readout 때문인가 | **아니오, 중간 강도** | 3,840채널 group-concat으로 `.153→.168` 개선됐지만 raw `.197`을 넘지 못했다. |
| Galileo 열세가 단순 파라미터 수 때문인가 | **아니오, 중간 강도** | nano/tiny/base 계열의 matched-scale 비교에서도 OLMo–Galileo 격차가 남는다. 단, 같은 데이터·목적의 순수 용량 실험은 아니다. |
| OLMo 내부에도 cache/raw 경계가 있는가 | **관측됨, 아직 진단** | base `.272`, tiny `.228`, nano `.194`로 내려가며 nano 평균은 raw `.197` 아래다. 시드 1개와 작은 `.003` 차이 때문에 보편적 필요조건으로 부르지는 않는다. |

## 오늘 결과가 왜 “방어만”은 아닌가

MS-105와 MS-108은 새 방법을 만든 실험은 아니다. 그러나 연구 질문을 바꿀 정도의 발견을
만들었다.

기존의 약한 질문은 다음과 같았다.

> OlmoEarth cache가 raw보다 좋은가?

현재 증거가 지지하는 더 강하고 일반적인 질문은 다음이다.

> **Earth embedding cache의 task utility는 모델 family·scale·release·입력 계약에 따라 언제
> raw 학습 위/아래로 교차하며, 이 경계를 test label 없이 예측할 수 있는가?**

Galileo group-concat과 규모 축이 없었다면 OlmoEarth의 승리는 readout 실수나 큰 모델 효과일 수
있었다. 두 통제를 거친 뒤에는 **캐시 효용이 조건부이며, 그 조건이 실제 행동을 바꾼다**는
characterization이 남는다. 특히 `Olmo base에서는 cache`, `Olmo nano에서는 raw와 사실상 경계`,
`Galileo에서는 raw`라는 서로 다른 선택이 같은 과업 안에서 발생한다. 이것이 선택기가 필요한
이유다.

다만 다음 표현은 금지한다.

- “모델 용량이 캐시 가치의 필요조건이다.” nano와 raw 차이가 작고 한 과업·시드 1개뿐이다.
- “네 독립 실험에서 Galileo가 실패했다.” 네 설정은 같은 데이터·폴드·시드를 공유한 **상관된
  설정 반복**이다.
- “규모만으로 캐시 가치를 label-free 예측할 수 있다.” 규모는 후보 feature일 뿐, 독립 과업에서
  predictor로 검증되지 않았다.
- “OlmoEarth의 목적함수 때문에 이겼다.” 용량·readout 설명은 약화됐지만 pretraining data,
  objective, 입력 modality는 아직 교락돼 있다.

현재 경계를 한 장으로 만든 Figure 후보는
`artifacts/figures/cache_utility_boundary_ms108.png`이다. 0선 오른쪽은 cache가 raw보다 유리하고,
왼쪽은 raw가 유리하다. 이 그림 자체가 “항상 cache”나 “항상 raw”가 모두 틀릴 수 있음을 보여준다.

## 논문을 두 단계로 본다

### 지금 이미 성립하는 후퇴선

작업 제목:

> **EarthCacheBench: Characterizing When Geospatial Foundation-Model Embeddings Remain Reusable**

핵심 기여는 세 줄이다.

1. 모델 릴리스가 바뀌면 높은 CKA와 무관하게 cache identity와 fixed decision이 깨질 수 있다.
2. frozen cache의 raw 대비 가치는 task뿐 아니라 family·scale·readout·입력 계약에 조건부다.
3. support label budget에 따라 reuse와 adapt의 우열이 바뀌며, 무조건 adapt도 안전하지 않다.

이 형태는 이미 논리적으로 성립하지만, 현재 그대로는 **두 과업 + architecture 축 시드 1개**라서
CVPR main보다 강한 characterization/benchmark 초안에 가깝다.

추가로 Solar의 8폴드는 독립 site 8곳이 아니라 UTM-zone group이고, architecture 체인의 마지막
폴드는 다른 GPU 1 작업과 동시 실행돼 wall-clock이 오염됐다. 전자는 일반화 단위를 낮추고 후자는
비용 수치만 무효화한다. 성능 결과는 원시 report·fold·metric 계약으로 별도 검증한다.

### CVPR main으로 올리는 최소 추가 기여

다섯 행동을 한 번에 풀지 않는다. 먼저 가장 단순한 결정을 검증한다.

```text
입력: 모델/입력 계약 + K개의 support label + 값싼 cache 진단
출력: CACHE-HEAD / RAW-OR-REEMBED
선택 사항: K>0이고 support 이득이 안정적일 때만 ADAPT
```

공개 untouched Task-3에서 아래 네 개만 비교한다.

| 정책 | 의미 |
|---|---|
| always-cache | 항상 frozen cache head |
| always-raw | 항상 raw baseline |
| support-rule | test label 없이 support와 contract만 보고 선택 |
| oracle | test 결과를 본 도달 불가능 상한 |

주 지표는 “선택 정확도”가 아니라 **oracle 대비 task-regret + GPU/I/O 비용**이다. support-rule이
always-cache와 always-raw 중 더 나쁜 쪽을 피하고, oracle에 가까운 비용–성능 frontier를 만들면
EarthCache의 중심 주장이 처음으로 실증된다. 실패하면 selector 주장을 접고 위의
EarthCacheBench characterization으로 제출 범위를 낮춘다.

## 남은 일의 정확한 우선순위

1. **증거 무결성 복구**: 한국 cube error 6건, rare-class gate amendment, Solar cross-CRS,
   release frozen-threshold summary를 먼저 닫는다.
2. **공개 Task-3 계약 동결**: 데이터·split·K·metric·항상-cache/raw baseline을 결과 전에 고정한다.
3. **Task-3 기본 행렬**: cache, raw, 선택적 adapt를 동일 decoder/compute 계약으로 실행한다.
4. **한 번의 선택기 검증**: Sen12/Solar에서 만든 단순 support-rule을 Task-3 test에 한 번만 적용한다.
5. **실측 비용**: cold re-embed, warm cache read, head train, raw train, 저장량과 I/O를 같은 장비에서 잰다.
6. **한국 3-task**: 공개 결과를 대체하지 않고, 한 캐시·세 행정/환경 과업의 외부 사례로 한 번 개봉한다.

## 최종 판정

- **과학적 현상**: 있다. 두 과업과 릴리스 전환에서 강하다.
- **반론 통제**: 상당히 나아졌다. readout-only·parameter-count-only 설명은 현재 시험 범위에서
  지지되지 않는다.
- **새 방법**: 아직 없다. 현재 선택기는 개발 규칙일 뿐 untouched 성능이 0이다.
- **CVPR main**: 아직 조건부다. 공개 Task-3 action-regret와 실측 비용 중 하나만이 아니라 둘 다
  중심 그림에 들어가야 한다.
- **프로젝트 가치**: “한국 지도를 붙인 데모”가 아니라 **어떤 Earth cache를 믿고 재사용할 수
  있는지 결정하는 문제를 실제 실패와 경계로 발견했다**는 데 있다. 한국 공공데이터는 이 원리를
  검증하는 외부 응용이며, 아직 성능 기여로 세지 않는다.
