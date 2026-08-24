# K-ALIGN 프로그램 노트 — 중심축 확장과 승률 보정

작성 2026-08-23. 역할: 8월 EarthRoute 핸드오프 노트와 같은 층위의 문서다. 즉 "지금 무엇이
은행에 들어와 있고, 다음 프로그램은 무엇이며, 왜 그것이고, 무엇부터 하는가"를 한 파일에 둔다.

**2026-08-23 5차 전략 보정:** 이 문서의 integrated bus+public-context 계획은 조건부 후속축으로
보존한다. 현재 CVPR 1순위의 거시적 방향은 `K_ALIGN_BIG_PICTURE.md`, 상세 gate는
`K_ALIGN_CVPR_READINESS_AUDIT.md`가 맡는다. `KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`는
public-context 데이터 gate가 열린 뒤의 실험 계약이다.

---

## 0. 먼저 약점

- 이 문서는 실험 결과가 아니라 **계획 보정**이다. "한국 공공데이터가 다른 backbone 성능을
  올렸다"는 결과는 여전히 0건이다.
- 아래 여덟 개 보정 중 R1·R2·R4만 이미 있는 자산으로 즉시 착수 가능하고, 나머지는 새 실행이
  필요하다.
- 가장 큰 미검증 가정은 **"EO에서 유도되지 않는 한국 공공 context가 충분한 양으로 존재한다"**
  이다. 현재 반증 신호가 이미 하나 있다: 14후보에서 exact PNU 일치는 1건, 시간정렬 0건,
  EIA 직접중첩 0건이다. 이 가정이 깨지면 `E_repr`은 데이터가 없어서 실패한다(방법 실패가 아님).
- 마감은 **CVPR 통상 11월 초 → 10월 말 완료**로 사용자가 고정했다. 실제 CFP 날짜는 공개되면
  다시 확인한다. 오늘(2026-08-23)부터 **약 10주**다. 5절 A0–A7이 정확히 9주이므로 슬랙이 1주뿐이다.
- FoldRefresh는 **AAAI-27 AISI에 제출·갱신된 상태**다. 따라서 K-ALIGN에서 FoldRefresh는
  새 기여가 아니라 **인용하는 선행 자산**이며, 같은 내용을 두 번 제출하지 않는다.
  R4는 "FoldRefresh를 새로 만든다"가 아니라 "이미 심사 중인 방법을 K-ALIGN의 refresh 경로에
  **적용**해 보증을 붙인다"이다. 논문에는 concurrent/prior work로 명시한다.
- Prithvi를 두 번째 family로 넣자는 R8은 6-band·30 m HLS 계약이 S2 10 m와 달라 paired input이
  성립하지 않을 수 있다. A2 smoke에서 먼저 판정한다.

---

## 1. 지금 은행에 들어와 있는 것

"은행에 들어왔다" = 다시 실행하지 않아도 논문 표·면접·파트너 대화에 쓸 수 있는 상태.

### 1.1 재현 봉인된 수치 자산

| 자산 | 수치 | 어디에 쓰이나 |
|---|---|---|
| full-216 paired release audit | 216 site-years × v1/v1.2, 입력 5,616파일 56.68 GB SHA-256 전수 고정, 출력 432개 105.59 GB | Figure 1 문제 동기 |
| cross-release identity 실패 | native R@1 1.0000 / identity 0.0000 / mean-shift 0.00024 / Procrustes 0.49097·0.43604 / affine ridge 0.69727·0.60889 → 사전등록 8 gate 전부 실패 | K-ALIGN의 존재 이유 |
| 유사도 지표의 배신 | pooled CKA 0.97857, 거리 Spearman 0.95251, 그런데 동일 token raw cosine −0.00860 | 독립 주장 후보 (아래 R7) |
| 릴리스 간 실행비 비대칭 | v1 3,756.12초 55.26 crops/s vs v1.2 2,250.12초 92.25 crops/s (1.67×), peak VRAM 4,291 vs 2,719 MiB | `E_refresh` 비용축의 실측 근거 |
| 8-window smoke의 이중 구조 | pooled 거리 순위상관 0.889·top-1/2 이웃 0.75/1.00 vs window 내부 spatial CKA 평균 0.427 (0.133–0.828) | 전역 구조와 국소 좌표가 다르게 이동한다는 P0 신호 |
| 공공 API bounded snapshot v3 | HTTP 463/463, 의미상 성공 456, 유효 무항목 1, GK2A 제한 오류 6, ~32 MB, request/hash 보존 | provenance 계약의 실증 |
| 공공 원본 | FarmMap 289,379 polygon, BuildingHUB 8,794행·45 법정동, EIA 13 polygon, VWorld 256/257 feature·고유 PNU 235 | `r_context` 재료 |
| 시간축 manifest | 5,184행 (dataset × 4개년 × 54윈도우 × 12기간) | 비동기 benchmark의 시간 spine |
| 임베딩 검색 실증 | 82윈도우 768d ~540만 벡터, 제주 cropland p@2000 .816(×14.8) built .658(×18.3), 양식장 프로토타입 held-out 중앙 백분위 100.0·제주 교차 9/9 | bus가 실제 검색을 지탱한다는 사전 증거 |
| 재현성 감사 | 공개 LFMC 체크포인트 test 951.9 vs 문서 주장 580.6, 4-튜플 버전 스큐 3건 | 취업축 공개 증거 |
| 방법 자산 (형제 저장소) | FoldRefresh — cross-fitted partial refresh, 4예산 median RMSE −17.3~22.3%, 두 인접 릴리스 전이 +22.31/+23.46% | 아래 R4의 핵심 |

### 1.2 은행에 없는 것 (착각 금지)

- 사람 정답 라벨 0개. 제주 216은 task label이 0이고 sealed 64는 이미 결과를 봤다.
- 공공데이터가 표현이나 정확도를 개선했다는 증거 0건.
- 원인 A/B 근거 0/368, 14후보 0/14.
- 두 번째 지역·두 번째 태스크의 무튜닝 재현 0건.
- 파트너 인터뷰 0회, 유료 갱신 0회.

---

## 2. 한 문장과 이번 확장의 요지

> **여러 글로벌 EO model family/release의 지식을 시점·coverage가 명시된 한국 공공데이터로
> 정렬해 안정적인 EO-only representation bus로 증류하고, 모델 릴리스와 공공기록이 서로 다른
> 속도로 갱신돼도 기존 gallery와 task head를 안전하게 재사용할 수 있는가?**

이번 노트가 바꾸는 것은 질문이 아니라 **승률**이다. 현재 계약은 설계는 정확하지만 세 가지
구조적 취약점을 안고 있다.

| 취약점 | 구체적 실패 시나리오 | 보정 |
|---|---|---|
| V1. `E_repr`이 순환논증 또는 회복불가능 distillation으로 보인다 | EO 파생물은 독립정보가 아니고, EO에서 전혀 안 보이는 행정정보는 EO-only student로 옮기기 어렵다 | R1 `R/V/T` transferability triage |
| V2. `E_compat`이 통과해도 BCT의 재탕으로 보인다 | compat loss를 걸고 학습한 `S1`이 `S0` gallery에서 95%를 넘는 것은 BCT가 이미 하는 일이다 | R2 black-box 불가능성 실험을 앞으로 |
| V3. 라벨이 임계경로에 있다 | A5(1,200 라벨 + 이중판독 400)가 A6 앞을 막는다. 파트타임 10 h/주에서 이 하나로 마감이 무너진다 | R3 label-free core 선행 |

여기에 승률을 올리는 다섯 개를 더한다: R4 통계적 인증, R5 재현 가능한 비동기 시뮬레이터,
R6 asset 축, R7 바닥 논문 고정, R8 family×release 격자 완성.

---

## 3. 우리 것이라고 말할 수 있는 경계

선행연구가 이미 차지한 것을 다시 확인한다. 아래는 전부 `PAPER_READING_LIST.md`와
`EMBEDDING_TRANSFER_CVPR_TRACKS.md`에 M 등급 이상으로 기록돼 있다.

| 선행 | 점유한 것 | 남긴 빈칸 |
|---|---|---|
| AM-RADIO (CVPR 2024), Theia (CoRL 2024) | 이종 vision foundation model 다중 teacher 증류 | 증류 대상이 **재학습 불가능한 공개 release**이고 teacher마다 band/GSD/시간축 계약이 다른 경우 |
| BCT (CVPR 2020), FCT (CVPR 2022), LCE (ICCV 2021), AdvBCT | embedding 호환성 | BCT는 **구 모델의 분류기와 학습데이터 접근**을 가정하고 FCT는 구 모델 쪽 준비를 가정한다. 공개 black-box release에는 둘 다 성립하지 않는다 |
| Matryoshka (NeurIPS 2022) | 한 embedding의 prefix 다중 예산 | 차원축소 자체는 기여가 아님. family/release 호환성과 결합해야 함 |
| GeoBridge·UniGeoRS·PAUL (CVPR 2026) | satellite–drone–ground 정렬 | 이 논문의 주기여가 아님. Paper D로 분리 |

따라서 우리가 말할 수 있는 한 문장은 이것뿐이다.

> BCT/FCT는 구 모델을 우리가 통제할 수 있다고 가정한다. 공개 EO foundation model release는
> 통제할 수 없고, teacher끼리 입력 계약도 다르다. 우리는 그 조건에서 성립하는 첫 compatible
> bus를 만들고, 갱신 비용에 **유효한 통계적 보증**을 붙인다.

마지막 절(통계적 보증)이 AM-RADIO 계열과 BCT 계열 어느 쪽에도 없는 유일한 부분이다.
이것을 강조하지 않으면 이 논문은 "EO판 AM-RADIO+BCT"로 읽힌다.

---

## 4. 승률을 올리는 여덟 가지 보정

### R1. `E_repr` source를 recoverability·task value·transfer로 심사한다

**문제.** 연도별 토지피복, FarmMap, DEM은 대부분 EO 관측에서 만들어진 산출물이라 독립적인
새 정보로 해석하기 어렵다. 그러나 EO에서 잘 예측된다는 이유만으로 teacher에서 제외하는 것도
잘못이다. test 때 context가 없는 student로 옮기려면 signal 일부는 EO에서 회복 가능해야 한다.
반대로 EO에서 전혀 보이지 않는 행정정보는 inference residual에는 유용해도 EO-only embedding으로
증류되는 정보에 상한이 있다.

**보정.** 모든 public token type에 대해 세 값을 분리한다.

```text
R(source) = EO-only probe가 public token을 얼마나 회복하는가
V(source) = 독립 task Y에서 EO+context가 EO-only보다 주는 조건부 가치
T(source) = context를 train 때만 본 EO-only student의 no-context student 대비 이득
```

- `R` 높음·`V` 낮음은 중복/shortcut이므로 pseudo-label baseline으로만 쓴다.
- `R` 낮음·`V` 높음은 inference residual 후보이며 EO-only transfer를 주장하지 않는다.
- `R` 중간·`V/T` 양수일 때만 privileged `E_repr` teacher로 승격한다.
- `R/V/T` gate는 A0에서 고정하고 같은 EO 입력·같은 split·독립 label을 쓴다.

**이것이 왜 승률을 올리는가.** 세 가지를 동시에 얻는다.

1. EO 파생물의 weak-supervision 효과와 독립정보 효과를 분리한다.
2. context가 task에는 유용하지만 EO-only student로는 안 옮겨지는 경우를 정직하게 residual로 보낸다.
3. `E_repr`이 실패해도 coverage 부족·중복·회복불가능·transfer 실패 중 원인을 구분한다.

**사전 예상 (반드시 실험으로 뒤집힐 수 있게 기록).**

| source | 예상 recoverability `R` | 이유 |
|---|---|---|
| 건축 인허가·착공·사용승인 **일자** | 낮음 | 행정 시각은 물리적 관측에 없음 |
| EIA 사업구역 **경계** | 낮음 | 법적 경계는 지표 반사와 무관 |
| VWorld PNU 필지 경계 | 낮음–중간 | 지적선은 종종 물리 경계와 불일치 |
| GK2A 독립 구름 산출 | 중간 | S2 SCL과 상관 있으나 센서가 독립 |
| 연도별 토지피복 | **높음** | EO 파생 산출물 |
| FarmMap 필지 | 중간–높음 | 항공영상 기반 |
| DEM | 높음 | 지형은 EO에서 상당 부분 회복 |

이 표는 가설일 뿐 teacher 자격을 정하지 않는다. `V/T`와 유효 publication time·coverage가
없으면 `E_repr`은 열리지 않는다.

### R2. black-box 호환성의 불가능성을 먼저 증명한다

**문제.** `S1`에 compat loss를 걸었으니 `S1→S0` R@1 95%는 통과할 가능성이 높다. 그런데 그것만
보여주면 BCT의 재현이다.

**보정.** A3~A4에 다음 세 baseline을 **의무**로 넣고, 실패를 정량화한다.

1. `BCT-surrogate`: 구 모델의 분류기가 없으므로 `S0` gallery에서 surrogate head를 학습해 BCT
   influence loss를 건다. BCT 원 논문의 가정이 깨졌을 때 얼마나 떨어지는지 측정.
2. `FCT-posthoc`: 구 모델 쪽 준비(side-information)를 할 수 없으므로 사후 변환만으로 FCT를
   흉내낸다. 성립 여부 자체가 결과.
3. `contract-mismatch`: teacher의 band/GSD/temporal stride가 다를 때 위 두 방법이 추가로 얼마나
   떨어지는가.

**주장 형태.** "BCT/FCT가 나쁘다"가 아니라 **"BCT/FCT의 가정이 공개 EO release에서는 성립하지
않으며, 그 조건에서 우리 bus만 gate를 통과한다"**이다. 이 표가 없으면 novelty 문단을 쓸 수 없다.

**비용.** 라벨 0개, GPU 소량. **지금 은행에 있는 216×2 출력으로 예비 실행이 가능하다**
(단 sealed 64는 제외하고 calibration 120건에서만; 방법 선택은 새 untouched split에서 확정).

### R3. label-free core를 먼저 완결한다

현재 실행순서는 A5(라벨 1,200 + 이중판독 400)가 A6 앞을 막는다. 파트타임에서 이것은 단일
실패점이다. 네 estimand를 라벨 의존성으로 다시 나눈다.

| estimand | 사람 라벨 필요? | 대체 평가 |
|---|---|---|
| `E_compat` | **불필요** | cross query/gallery retrieval + frozen probe (공개 task) |
| `E_refresh` | **불필요** | bytes, wall/GPU-hour, latency + FoldRefresh 인증 |
| recoverability `R`·coverage (R1) | **불필요** | 공공기록 자체가 target |
| 조건부 value `V`·EO-only transfer `T` (R1) | **필요** | 독립 task label |
| `E_repr` | 필요 | 공개 dense task로 부분 대체 가능 |
| `E_fusion` | 필요 | 대체 불가 |

**결정: `E_compat + E_refresh + recoverability/coverage`를 먼저 닫고 그것만으로 서 있는 산출물을 만든다.**
라벨 수집은 그와 **병렬로** 시작하되 임계경로에서 뺀다. 라벨이 늦으면 논문이 축소될 뿐
사라지지 않는다.

### R4. `E_refresh`에 FoldRefresh 인증을 붙인다 — 지금 가장 큰 차별점

**문제.** 현재 계약의 `E_refresh`는 "backfill bytes 10× 절감" 같은 **비용 숫자**뿐이다. 비용
숫자는 엔지니어링 리포트지 CVPR 기여가 아니다.

**보정.** 형제 저장소에 이미 검증된 FoldRefresh(cross-fitted partial refresh)를 residual refresh
경로의 추정기로 이식한다. 그러면 주장이 이렇게 바뀐다.

> 갱신을 건너뛴 cache로 계산한 지도 수준 통계가 **유효한 유한모집단 보증**을 유지한 채
> backfill을 N× 줄인다.

BCT·FCT·AM-RADIO·Matryoshka 어느 논문에도 "부분 갱신된 embedding cache 위의 통계가 유효하다"는
보증은 없다. 이것은 **이미 만든 자산**이고, 붙이는 비용이 새 연구보다 훨씬 싸며, 두 논문
(AAAI 제출본 ↔ K-ALIGN)을 한 줄로 잇는다.

> 먼저 모델이 바뀐 뒤 호출의 75%를 건너뛰고도 통계를 유효하게 유지할 수 있음을 보였다.
> K-ALIGN은 그 "건너뛰기"를 **표현 좌표계 자체**로 확장한다.

**gate 추가.** `E_refresh`는 비용 절감과 **보증 유지**를 동시에 통과해야 한다. 절감만 되고
보증이 깨지면 KILL이 아니라 "engineering report로 축소"다.

### R5. 재현 가능한 비동기 시뮬레이터를 만든다

**문제.** 심사자는 VWorld/BuildingHUB/GK2A 키가 없다. 한국 API에 의존하는 dual-speed 주장은
"우리만 재현 가능한 결과"로 읽힌다. CVPR main에서 이것은 치명적이다.

**보정.** 공개 데이터만으로 **publication lag · coverage hole · conflicting record**를 합성하는
harness를 만든다.

- 기반: PANGAEA / GEO-Bench 계열 공개 dense task 1개 + 시점이 있는 공개 보조 레이어
  (예: ESA WorldCover 연도판, 공개 OSM history) — 후보는 A0에서 확정.
- 조작 변수: `published_time - event_time` 분포, coverage 누락률, 충돌률.
- 산출: 한국 실측 비동기 분포와 합성 분포를 같은 축에 겹친 그림 1장.

이러면 논문의 논리가 바뀐다. **일반 주장은 합성 harness로 증명하고, 한국은 그 합성이 실제
행정 시스템에서 어떻게 나타나는지 보이는 real-world stress case가 된다.** 지역성이 약점에서
강점으로 이동한다.

### R6. asset 축을 붙인다 — 바닥을 올리는 가장 확실한 수단

8월 노트의 수상 조사 결론은 명확했다: **asset · coverage · 정직한 감사** 중 둘 이상이 있어야
한다. 현재 K-ALIGN은 순수 method 논문이라 셋 중 하나(감사)만 있다.

이미 은행에 있는 것으로 asset을 만들 수 있다.

**KAB (Korea Asynchronous Bus) release** 후보 구성:

| 구성요소 | 현재 상태 | 추가로 필요한 것 |
|---|---|---|
| paired release-shift 출력 216 site-years × v1/v1.2 | 완료·해시 봉인 | 라이선스 검토, 익명화 불필요, 용량 축소본(pooled/저차원) |
| exact input freeze manifest (5,616파일 SHA-256) | 완료 | 공개 형식 정리 |
| 시간축 manifest 5,184행 | 완료 | 스키마 문서 |
| provenance-complete 공공 snapshot (v3, 463 request) | 완료 | 재배포 가능 필드만 선별, request manifest 공개 |
| failure atlas (구름 Top-k 오염, nodata 0값, 버전 스큐 3건) | 산재 | 한 곳으로 모으기 |
| source `R/V/T` 표 (R1) | 미실행 | R1 실행 결과 |

**왜 바닥을 올리는가.** 네 gate가 전부 실패해도 "비동기 갱신 benchmark + 실패 아틀라스 +
derivability 표"는 남는다. 지금 설계는 gate 실패 시 남는 것이 문서뿐이다.

### R7. 바닥 논문을 미리 못 박는다 — "CKA는 호환성이 아니다"

이미 봉인된 수치 하나가 그 자체로 주장이다.

> pooled CKA 0.97857, 거리 Spearman 0.95251, 그런데 같은 token의 raw cosine −0.00860이고
> cross-release R@1은 양방향 0.0000이다.

표현 유사도 지표(CKA)로 "모델이 호환된다"고 말하는 관행에 대한 반례다. 사전 등록한 8개
bridge gate가 전부 실패했다는 사실까지 포함하면, 짧은 논문 또는 workshop 한 편으로 독립
성립한다. **이것을 지금 "바닥"으로 선언해 두면 모든 후속 실패가 0이 아니게 된다.**

조건: 이 주장은 제주 한 지역·한 grid·label 0의 결과다. 일반화 문장을 붙이지 않는다.

### R8. family × release 격자를 완성한다

현재 teacher 구성은 Olmo{v1, v1.2} + TerraMind{Base}다. 이것은 **family 2개, release 축은 1개
family에만 있는** 격자다. "family/release 호환성"을 주장하려면 최소 2×2가 필요하다.

**제안.** 두 번째 family의 두 릴리스를 넣는다. Prithvi-EO {1.0, 2.0}은 둘 다 공개돼 있다.

| | release A | release B |
|---|---|---|
| Olmo | v1 | v1.2 |
| 두 번째 family | Prithvi-EO 1.0 | Prithvi-EO 2.0 |
| 세 번째 (선택) | TerraMind Base | — |

**위험.** Prithvi는 6-band·30 m HLS 계약이라 S2 10 m와 paired input이 성립하지 않을 수 있다
(`K_EVIDENCE_SHIFT_BENCHMARK.md`에 이미 기록된 제약). A2 teacher-contract smoke에서
**격자를 열기 전에** 판정한다. 성립하지 않으면 2×2를 포기하고 "1 family 2 release + 2 family
cross-sectional"로 주장을 낮춘다. 여기서 억지로 밀면 native ceiling 혼동이 생긴다.

### R9. 네 축을 한 논문에 넣지 않는다

사용자가 제기한 큰 그림 — "정확도 / 임베딩 / 속도 / 위성 유도" — 을 문헌으로 검증한 결과는
`K_GAIN_AXES.md`에 있다. 요지는 네 축이 **서로 다른 역할**이라는 것이다.

| 축 | 이 프로그램에서의 역할 | 근거 |
|---|---|---|
| A 정확도 | `E_repr`의 **평가 형태**. 주장은 "정확도 +"가 아니라 "라벨 절감" | PANGAEA는 full-label에서 supervised baseline 우세, 10% label에서만 GFM 우세를 보고 |
| B 임베딩 | **본편** | GeoLink·CLIP4Geo·WildSAT이 "비-EO 기록→EO 표현"을 점유. 남은 빈칸은 **공개 시각이 관리된 기록** |
| C 속도 | gate의 **단위**. 한국 데이터는 속도를 올리지 않는다 | binary 32×·PCA64+int8이 이미 sweet spot. 우리 빈칸은 "압축된 gallery를 릴리스 전환 때 어떻게 하는가" |
| D 위성 유도 | **다음 프로그램(EarthRoute)** | tip-and-cue·EO 스케줄링·온보드 배포가 각각 점유됨 |

특히 두 개는 **기여에서 뺀다**: ① 필지 경계(전지구 10 m 필지지도 241개국 31.7억 polygon 존재)
② 한국판 FLAIR-HUB. 후자의 이유는 **면적/예산이 아니라 장르 선점**이다 — FLAIR-HUB는
2,528 km²(제주의 약 1.4배)이고 "630억 픽셀"은 그 면적을 20 cm로 나눈 화소 수이지 사람의
판정 횟수가 아니다. 하지 않는 진짜 이유는 ⓐ 장르를 IGN이 이미 정의했고 ⓑ 한국판 dense 주석은
환경부 토지피복지도의 재포장이라 독립 정답이 아니며 ⓒ 비동기 provenance라는 우리 차별점과
무관하다는 것이다. 다만 **asset 형식(모달 정렬·CC BY-SA·벤치마크 동봉)은 템플릿으로 쓴다**.

반대로 `E-07` 온보드 임베딩 링크는 compatibility의 운영 동기로 **제한해** 쓴다. 논문의 약
598–690 B/query는 downlink JSON telemetry이고 hint gallery 업링크는 `N_hints × D × bytes`로
별도다. 따라서 “1 KB gallery budget 때문에 backfill이 물리적으로 불가능하다”고 쓰지 않고,
가정한 대역폭 아래의 simulation으로만 다룬다.

---

## 5. 다시 짠 실행 순서

기존 A0–A6을 라벨 의존성 기준으로 두 갈래로 나눈다. 왼쪽은 임계경로, 오른쪽은 병렬.

| 단계 | 기간 | 산출물 | 다음 단계 gate |
|---|---:|---|---|
| **A0** source/role/transferability 계약 freeze | 3일 | source×cutoff×role×license manifest, leak test, `R/V/T` gate 고정, 합성 harness 기반 데이터셋 확정 | 시간·role 불명확 source 제외 |
| **A1** 3지역 canonical anchor + untouched split hash-freeze | 1주 | ≥10k site-years, exact input/public snapshot hash | 두 번째 지역 join 실패 시 중단 |
| **A2** teacher contract smoke | 3일 | 32 windows × Olmo v1/v1.2 / TerraMind / Prithvi 1.0·2.0의 token·stride·mask·runtime | paired support 불일치면 R8 격자 축소 |
| **A3** source transferability triage (R1) | 4일+label pilot | source별 `R/V/T`, 유효 publication time·coverage | 독립 label 전에는 `R/coverage`만 보고하고 `E_repr` 개방 금지 |
| **A4** bridge + black-box 불가능성 (R2) | 1주 | identity/linear/MLP/relational + BCT-surrogate/FCT-posthoc/contract-mismatch 표 | affine=MLP면 nonlinear 확장 중단 |
| **A5** bus P0 + `E_compat` + `E_refresh` (R3·R4) | 2주 | `S0`/`S1`, 4칸 query/gallery, FoldRefresh 인증 refresh 곡선, 256/768d | `E_compat`+`E_refresh` 미달이면 bus 주장 중단 |
| **A6** 합성 비동기 harness (R5) | 1주 | 공개 task 위 lag/coverage/conflict 스윕, 한국 실측 분포 중첩 그림 | 합성에서 효과 방향이 없으면 dual-speed 주장 중단 |
| **A7** KAB asset 패키징 (R6) | 1주 | 공개 가능한 benchmark + failure atlas + source `R/V/T` 표 | 라이선스 미해결 항목은 제외하고 진행 |
| **B1** (병렬) 라벨 수집 | 착수 즉시, 4–8주 | NGII 전후 항공 신청, 블라인드 이중판독, train 600 / val 200 / sealed 400 | agreement <0.60이면 점수표를 열지 않음 |
| **B2** `E_repr` + `E_fusion` | B1 완료 후 2주 | 4 estimand 전체 표, 5 seeds, CI·Pareto | 사전 gate로만 판정 |

**핵심 변경 두 개.** ① 라벨(B1)이 label-free A5·A6·A7을 막지 않는다. ② A3의 `R/coverage`는
학습 전 데이터 feasibility를 판정하지만, `V/T`와 독립 label 없이는 `E_repr`을 열지 않는다.

### B1이 왜 어려운가 — 라벨 비용 실측

"라벨 1,200건"이 왜 임계경로에서 빠져야 하는지는 시간 계산보다 **구조적 막힘**이 더 크다.

**분량 계산.** 계약은 train 600 / validation 200 / sealed 400 = 1,200건이고, 이중판독은
sealed 400 전수 + train 최소 120 = 520건이다.

```text
총 판독 횟수 = 1,200 + 520 = 1,720 회
1회당 (전후 S2 칩 + 항공사진 대조 + 판정·사유 기록) 5–10분
        → 143 ~ 287 시간
불일치 조정(agreement 0.7 가정) 약 156건 추가
파트타임 10 h/주 → 판독만 14 ~ 29주
```

10주 마감 안에 들어갈 수 없다. 그러나 이건 두 번째 문제다.

**구조적 막힘 일곱 개** (위일수록 치명적):

| # | 막힘 | 근거 |
|---|---|---|
| 1 | **두 번째 독립 판독자가 없다** | 계약이 independent double review를 요구하는데 이 프로젝트는 1인이다. assistant pre-annotation은 leak으로 **명시 금지**돼 있으므로 보조로 메울 수 없다 |
| 2 | **10 m Sentinel-2로는 판정이 안 되는 사건이 많다** | 필지 단위 개발·소규모 벌채는 10 m에서 보이지 않는다 → NGII 항공사진 필수 |
| 3 | **NGII 항공사진은 수동 신청·승인이다** | 국토정보맵 로그인 → 주소/지명 → 연도 → 신청 → 승인 후 TIFF. 배치 API 없음. 1,200건 × before/after = 최대 2,400회 |
| 4 | **항공 촬영 주기가 라벨의 시간 해상도를 정한다** | 촬영이 연 1회면 "언제 변했는가"의 해상도가 1년이다. `t0~t1` 사이 시점을 확정할 수 없다 |
| 5 | **라벨 단위 geometry가 아직 없다** | 오름 368개 중 공식 polygon **0개**, OSM point seed 243개뿐. 무엇을 한 건으로 셀지부터 정의 작업이다 |
| 6 | **positive가 희소하다** | 368 고정분모에서 후보 14개. 무작위 1,200 표본은 positive가 몇 건 안 나온다 → 층화 필요 → 모집단 추정에 PPI 필요 |
| 7 | **행정기록으로 대체할 수 없다 (실증됨)** | 14후보에서 exact PNU 일치 1건, 시간정렬 0건, EIA 직접중첩 0건. "공공기록을 라벨로 쓰면 되잖아"는 이미 막혔다 |

**판정 자체가 어렵다는 자체 증거.** `jeju_v5_rgb_manual_review.json`은 사전 고정한 5쌍 전부
`no_improvement`로 gate가 실패했다. 지금까지 이 프로젝트의 사람 판독 실적은 RGB 직접 검수 9건과
14후보 assistant 검토가 전부다 — 즉 **누적 약 23건**이다. 목표는 1,200건이다.
우리 코퍼스의 `W-20` *Humans are Poor Few-Shot Classifiers for Sentinel-2 Land Cover*도 같은
방향을 경고한다.

**따라서 할 일은 규모를 줄이는 것이 아니라 막힘 1과 3을 먼저 푸는 것이다.**

1. **두 번째 판독자를 확보한다.** sealed 400건만 이중판독하면 400 × 7분 ≈ 47시간이다.
   이 정도는 파트너 또는 유료 판독자 1명에게 발주 가능한 규모다. 이것이 유일한 근본 해법이다.
2. **NGII 승인 lead time을 먼저 측정한다.** 우선 10–20건을 신청해 실제 소요일을 잰다.
   **이 숫자를 모르는 상태에서 B1 일정은 계산 자체가 불가능하다.** 그래서 9절에서 NGII 신청을
   임계경로가 아님에도 "오늘 착수"로 둔 것이다.
3. 그 측정 전까지 CVPR 제출본은 label-free core + 공개 dense task로 대체한 축소 `E_repr`로 간다.

기존 원칙은 유지: GPU0가 비어 있다는 이유로 A2부터 시작하지 않는다. A0/A1이 먼저 닫힌다.

---

## 6. 제출 사다리

**기준 마감: 2026-10-31 완료 (CVPR 통상 11월 초).** 오늘부터 약 10주, 슬랙 1주.

| 주차 | 날짜 | 닫혀야 하는 것 |
|---|---|---|
| W1 | ~08-30 | A0 계약 freeze, R6 라이선스 검토, R7 바닥 그림 |
| W2 | ~09-06 | A1 3지역 anchor + untouched split hash-freeze |
| W3 | ~09-13 | A2 teacher-contract smoke, A3 source transferability triage 착수 |
| W4 | ~09-20 | A3 완료 → **`E_repr` 개폐 결정** |
| W5 | ~09-27 | A4 bridge + black-box 불가능성 표 |
| W6–7 | ~10-11 | A5 bus P0 (`E_compat` + `E_refresh`) |
| W8 | ~10-18 | A6 합성 비동기 harness |
| W9 | ~10-25 | A7 asset 패키징 + 본문 작성 |
| W10 | ~10-31 | 예비 주차 (슬랙) |

**10-04(W5 종료) 체크포인트**: A4가 닫히지 않았으면 순위 0으로 전환한다. B1(라벨)은 이 일정
안에서 `E_repr`/`E_fusion`을 닫을 수 없으므로, **CVPR 제출본은 label-free core + 축소된
`E_repr`(공개 dense task 대체)로 간다**고 지금 결정한다. 한국 라벨은 다음 마감용이다.

| 순위 | 산출물 | 필요 조건 | 이것만으로 성립하는가 |
|---:|---|---|---|
| 0 | arXiv 기술 리포트 + KAB asset (R6·R7) | A0–A3 + A7 | 예 — 바닥 |
| 1 | **K-ALIGN 본편** | `E_repr` + `E_compat` 동시 통과 | CVPR main 목표 |
| 2 | Compatible Earth Representation Bus (fallback) | `E_compat`만 통과 | 예 |
| 3 | K-Context Fusion (fallback) | `E_fusion`만 통과 | 예 |
| 4 | Partial-Refresh Engineering Report | `E_refresh`만 통과 | 예 (논문 아님) |

**한 축만 통과하면 억지로 합치지 않는다**는 기존 규칙을 유지한다. 여기에 한 줄 추가:
**두 축이 통과해도 A6(합성 harness)이 없으면 CVPR main으로 올리지 않는다.** 재현 불가능한
지역 데이터에만 근거한 일반 주장은 리뷰에서 살아남지 못한다.

---

## 7. 프로그램 중단 조건

기존 estimand별 KILL은 계약이 유지한다. 여기에 **프로그램 수준** 중단 조건을 추가한다.

1. **데이터 중단.** A3에서 유효 publication time·geometry를 가진 source coverage가 사전 하한
   (대상 site-event의 5%)에 못 미치거나 독립 label에서 `T(source)`가 양수가 아니면 `E_repr`을
   열지 않고 R6+R7로 축소한다.
2. **호환성 중단.** A4에서 BCT-surrogate가 우리 bus와 통계적으로 같으면, 새 방법 주장을
   중단하고 "공개 release에 BCT를 적용하는 법"이라는 훨씬 작은 논문으로 낮춘다.
3. **일반성 중단.** A6 합성 harness에서 효과 방향이 사라지면 한국 전용 사례연구로 낮춘다.
4. **비용 중단.** `E_refresh`의 절감이 있으나 FoldRefresh 보증이 깨지면 통계 주장을 삭제하고
   비용만 보고한다.
5. **인력 중단.** B1의 이중판독 agreement가 0.60 미만이면 라벨 축 전체를 닫고 label-free
   산출물만 낸다. 낮은 agreement로 만든 점수표는 자기기만이다.
6. **시간 중단.** 마감 4주 전까지 A5가 닫히지 않으면 순위 0(arXiv+asset)으로 전환하고
   본편은 다음 마감으로 넘긴다. 마감을 맞추려고 gate를 낮추지 않는다.

---

## 8. 세 축 닫기 — 취업 × 박사 × 사업

각 estimand가 어떤 외부 증거로 닫히는지 미리 못 박는다. 코드를 썼다는 것은 성과가 아니다.

| 산출물 | 취업 (Ai2/EO 조직) | 박사 | 사업 |
|---|---|---|---|
| source `R/V/T` 표 (R1) | "회복성·조건부 가치·EO-only 전이를 분리했다" — 데이터 판단력의 증거 | source 역할을 가르는 표 | 어떤 공공 소스를 살지의 근거 |
| black-box 불가능성 (R2) | 공개 release 운영 경험 | novelty 문단의 뼈대 | GeoFM Release Audit 상품의 논거 |
| FoldRefresh 인증 refresh (R4) | 릴리스 전환 비용 절감 실측 | 두 논문을 잇는 한 줄 | "재계산비 30–40%↓" 판매 문장 |
| 합성 harness (R5) | 재현 가능한 공개 도구 | 일반성 방어 | 고객 데이터 없이 데모 가능 |
| KAB asset (R6) | 공개 자산 보유 | D&B 트랙 후보 | 방어 가능한 데이터 자산 |
| CKA 반례 (R7) | 기술 글 한 편 | 짧은 논문 | 릴리스 감사 필요성의 증거 |

각 칸은 **공개 증거(PR/이슈/릴리스) · 논문 표 · 파트너 결정** 중 하나로만 닫힌다 (L7).

---

## 9. 지금 당장 할 일

순서대로. 1~3은 서버 없이 로컬에서 가능하다.

1. **A0 계약 작성.** source×role×cutoff×license manifest에 `R/V/T`와 coverage gate 정의를
   추가한다. 합성 harness의 기반 공개 데이터셋을 후보 3개 중 하나로 확정한다.
2. **R7 바닥 고정.** 이미 봉인된 CKA 0.97857 / R@1 0.0000 대비를 한 그림·한 문단으로 만들어
   `artifacts/`에 넣는다. 이것이 Figure 1이자 최악의 경우의 산출물이다.
3. **R6 라이선스 검토.** VWorld / 공공데이터포털 / FarmMap / 국토지리정보원 각각의 재배포
   조건을 확인해 KAB에 넣을 수 있는 필드와 넣을 수 없는 필드를 나눈다. **여기서 막히면
   asset 축이 통째로 사라지므로 가장 먼저 확인한다.**
4. **A1 split 동결.** 3지역 anchor와 untouched split을 hash-freeze한다. 방법 선택 전에.
5. **A2 smoke.** Olmo v1/v1.2 + TerraMind + Prithvi 1.0/2.0의 32-window teacher-contract smoke.
   여기서 R8 격자가 성립하는지 판정한다.
6. **B1 착수.** NGII 전후 항공사진을 우선 10–20 후보에 대해 **오늘 신청한다.** 승인 대기가
   길기 때문에 임계경로가 아니어도 가장 먼저 넣어야 한다.
7. **마감일 확인.** CVPR 계열 및 대안 학회의 실제 마감을 확인해 6절 사다리에 날짜를 채운다.

---

## 10. 하지 않을 것

- 한국 지도를 더 많이 붙이는 것 자체를 기여로 세지 않는다.
- `E_fusion`만 좋아진 결과를 "EO 표현을 강화했다"로 쓰지 않는다.
- sealed 64 site-years를 새 방법의 test로 재사용하지 않는다.
- 로봇·simulation을 이 논문에 합치지 않는다 (Paper D로 분리 유지).
- 연합학습을 실제 비반출 기관 3곳 없이 넣지 않는다.
- CKA·코사인 유사도만으로 cache 호환성을 주장하지 않는다.
- 마감을 맞추려고 gate 수치를 사후에 낮추지 않는다 (L4).

---

## 11. 이 노트와 다른 문서의 관계

```text
K_ALIGN_PROGRAM_NOTE.md   ← 이 파일. 프로그램 수준 결정과 승률 보정
    │
    ├─ KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md   authoritative 실험 계약 (충돌 시 이쪽이 이김)
    ├─ K_CONTEXT_FUSION_EXPERIMENT.md          public source-role 기본 계약
    ├─ EMBEDDING_TRANSFER_CVPR_TRACKS.md       넓은 트랙 후보와 순위
    ├─ K_EVIDENCE_SHIFT_BENCHMARK.md           Paper B (selective evidence)
    ├─ PAPER_READING_LIST.md                   문헌 검증 장부
    └─ EARTHROUTE_PROGRAM_NOTE.md              후속 프로그램 (K-ALIGN 이후)
```

EarthRoute는 대체되지 않고 **뒤로 간다**. 순서는 그대로다.

```text
K-ALIGN: 무엇을 다시 계산하지 않고도 같은 좌표계에서 말할 수 있는가
    ↓
FoldRefresh: 그 부분 갱신 위의 통계가 유효한가
    ↓
EarthRoute: 다음에 관측·모델·행정근거·사람검수 중 무엇을 살 것인가
```
