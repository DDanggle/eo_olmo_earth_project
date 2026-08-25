# GK2A를 연구에 어떻게 엮는가 (2026-08-25)

> **이 문서는 측정 사슬 6번(`E_live`)임. 1~5가 닫히기 전에 열지 않음**
> (`docs/CRITICAL_PATH.md`). 예외는 `docs/DAILY_OPS.md`의 수집뿐임 —
> 2일 보존이라 미루면 원자료가 소실되므로 수집만 계속함.

수집은 시작했음(`docs/DAILY_OPS.md`). 이 문서는 **모은 것으로 무엇을 하는가**임.

## 0. 먼저 — GK2A가 답하는 질문을 잘못 잡으면 안 됨

| GK2A가 답하는 것 | GK2A가 답하지 **못하는** 것 |
|---|---|
| **볼 수 있나** (구름·안개·에어로졸·구름광학두께·구름형) | **왜 일어나나** (강우·적설·지진) |
| = 관측 가능성 (observability) | = 재해 강제력 (forcing) |

현재 수집하는 **다섯 경량화 산출물에는 강수가 없음.** 강수 forcing은 ASOS·레이더 등에서
따로 와야 한다. 따라서 이 다섯 산출물을 산사태 forcing으로 해석하면 틀린다.

### 그런데 이 구분이 오히려 기여가 됨

대부분의 EO 논문은 구름을 **지워야 할 방해물**로 다룸(마스킹하고 버림).
여기서는 구름이 **결정 변수**가 됨.

> 하늘이 보이지 않을 때 옳은 행동은 예측이 아니라 **보류(abstain)**임.

이것이 `EarthRoute`의 admission/abstain 정책과 정확히 맞음. 그리고 이 판단에는
**시점이 있는 관측조건**이 필요하므로, 정적 자료로는 대체할 수 없음.

## 1. 세 가지 구체적 사용처

| | 용도 | 어떻게 | 지금 가능한가 |
|---|---|---|---|
| **A** | **Admission/abstention** — 새 S2 관측을 처리·보류할지 판단 | S2 관측시각 ±60분의 CLD/FOG를 외부 보조관측으로 추가하고 S2 SCL-only 대비 조건부 이득 측정 | 공식 lat/lon grid 필요 (§2) |
| **B** | **`r_t` observability residual** — 설계의 E_live | AOI별 관측조건을 head의 confidence/abstention에 결합. Area 값이 행정구역 평균인지는 미확인 | 행정동코드와 의미 감사 필요 |
| **C** | **데이터 생명주기 결과** | 짧은 endpoint 창·별도 archive·snapshot provenance를 분리해 운영비용 측정 | **지금 가능** |

C가 즉시 가능하고, M9(split 누수)·M12(annotation 교락)와 같은 **계약 감사 계열**이다.
다만 KMA API Hub에 별도 L2 archive가 있으므로 “소급 연구가 불가능하다”는 결론은 철회한다.
경량화 응답과 archive 산출물이 같은지는 아직 미측정이다.

## 2. 좌표 결합 — 역공학 대신 공식 lat/lon grid

### 격자(All) 응답에 CRS가 없음

```
gridKm=2.0  xdim=320  ydim=397  x0=63.0  y0=333.0
value = 127,040개 값
```

**좌표계도, 원점 경위도도 응답에 없음.** `x0/y0`는 더 큰 격자의 오프셋으로 보이나
기준을 모름. 이걸 모르면 격자를 AOI에 붙일 수 없음.

### 행정구역(Area) 응답은 lon/lat을 줌 — 이것이 우회로

```
getGk2acldArea?...&dongCode=1111051500
→ {"lon":"126.97065","lat":"37.58414","value":"2"}
```

**단 코드 체계를 맞춰야 함.** 실측으로 확인했음:

| 코드 | 체계 | 결과 |
|---|---|---|
| `1111051500` (서울 종로구 청운효자동) | **행정동코드** | **OK** — lon/lat/value 반환 |
| `1114052000`, `2611051000`, `3111051000` | 행정동코드 | OK |
| `4885031029` (하동군 화개면, VWorld `level4LC`) | **법정동코드** | 실패 (`resultCode 11`) |

즉 GK2A는 **행정동코드**를 받음. VWorld 역지오코딩의 `level4LC`는 법정동코드라 안 맞음
(농촌 면 지역은 `level4AC` 행정동코드가 빈 값으로 옴).

### M15 역공학은 operational 경로에서 폐기

4개 고유 지점의 CLD 반복관측 96개에 `xo/yo`를 적합해 86/96(0.8958)을 얻었다. 반복 시간값은
고유 공간 anchor를 늘리지 않고, offset 선택과 평가를 같은 자료에서 했으므로 이것은 held-out
좌표계 검증점수가 아니다. 범주 분포도 0.33 균등이라고 보장되지 않는다. 따라서 “LCC 계열 확인,
row-major/y축 확정” 주장을 철회하고 **후보 적합 기록**으로만 남긴다.
초기 cache key에 `resultType`이 없어 FOG가 CLD를 덮어쓸 수 있던 코드 결함도 있었다. CLD-only
filter 후 수치는 우연히 동일했지만 설계의 외부 타당성 문제는 그대로다.

기상청 API Hub가 공식 경로를 제공한다.

```text
ASCII lat/lon:
https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-gk2a_latlon_api
  ?area=KO&grid=2&latlon={lon|lat}&disp=A&authKey=...

NetCDF lat/lon:
https://apihub.kma.go.kr/api/typ01/url/gk2a_latlon_file_down.php
  ?area=KO&grid=2&authKey=...
```

공식 문서는 좌상단→우하단 순서이며 “격자자료가 저장된 순서”와 같다고 명시한다. 다음 gate는
추정 정확도 0.90이 아니라 다음의 exact contract다.

1. KMA API Hub 접근 승인 및 KO/2 km lon·lat 원본의 SHA-256 고정
2. lon/lat/grid value의 `xdim × ydim = 127,040` 및 순서 exact 일치
3. 4개 Area 지점은 **fit이 아닌 sanity check**로만 사용
4. checksum이나 shape가 바뀌면 grid version 변경으로 보고 join을 중단

행정동코드는 이제 격자 좌표계를 푸는 열쇠가 아니라 **Area residual을 쓸 때만** 필요하다.
AOI와 직접 겹치는 소수 코드만 공식 코드표에서 고정하며, Area의 `lon/lat/value`가 경계 평균인지
대표점 값인지 활용가이드로 먼저 확인한다. `area_anchors.jsonl`은 parsed row이지 원본 응답이 아니다.

### admission 실험의 사전등록 단위

GK2A는 S2 SCL과 완전히 독립된 label이 아니라 같은 대기상태를 다른 플랫폼·알고리즘으로 본
외부 보조관측이다. 따라서 “GK2A를 넣어 정확도가 올랐다”보다 **incremental selective utility**를 잰다.

- 입력: S2-only quality score/SCL vs S2 + acquisition-time-matched GK2A(±60분)
- 평가: frozen 한국 spatial split의 failure-detection AUROC와 risk–coverage/AURC
- downstream 보조지표: admitted subset의 retrieval precision·segmentation mIoU, coverage와 함께 보고
- kill: S2-only 대비 held-out cluster에서 이득이 없거나 timestamp/official-grid gate가 깨지면
  GK2A를 performance method가 아니라 운영 provenance 사례로 내림

이 arm은 한국 operational demonstration이다. 네팔·스위스 transfer의 공통 입력인 척하지 않는다.

## 3. AOI 역지오코딩 결과 (부수 발견)

VWorld로 M10의 13개 군집 중심을 역지오코딩했음. **9/13이 육상**이고, 실제 국립공원으로 확인됨.

| 군집 | 타일 | 위치 | 산사태 타일 |
|---|---|---|---|
| C11 | 180 | 강원 평창군 평창읍 (오대산권) | 42 |
| C13 | 129 | 좌표가 해상 (한려해상권 도서 분산) | 14 |
| C01 | 84 | 좌표가 해상 (다도해권) | 0 |
| C10 | 38 | 경남 하동군 화개면 — **지리산** | 4 |
| C04 | 34 | 경북 경주시 하동 | 0 |
| C06 | 30 | 경남 거창군 웅양면 (덕유산권) | 0 |
| C03 | 28 | 좌표가 해상 (태안해안권) | 0 |
| C12 | 22 | 전남 장성군 북일면 (내장산권) | 12 |
| C07 | 14 | 경기 고양시 덕양구 | 0 |
| C09 | 12 | 경북 청송군 주왕산면 — **주왕산** | 8 |
| C02 | 9 | 좌표가 해상 (다도해권) | 0 |
| C05 | 8 | 대전 유성구 세동 | 4 |
| C08 | 6 | 전남 강진군 성전면 | 6 |

`NOT_FOUND` 4개(C01·C02·C03·C13)는 **군집 중심 좌표가 바다에 떨어진 것**임.
군집이 해상 전용이라는 뜻은 아님 — C13은 산사태 14타일을 가짐(도서 지역이 넓게 퍼져
중심이 물에 놓임). C01·C02·C03은 산사태가 0타일이라 연안·해상 성격과 일관됨.

**확인이 필요한 것**: 산사태 0타일 군집(C01·C02·C03·C04·C06·C07)을 headline task에서
빼야 하는지. M10의 holdout은 이미 동결됐으므로, 빼는 것이 아니라 **task별 유효 군집을
따로 보고**하는 방식이어야 함.

## 4. 지금 말할 수 있는 것과 없는 것

- **말할 수 있는 것**: 선택한 다섯 산출물은 observability 자료다. 8월 23·24일은 각
  57/57 terminal outcome으로 수집됐다. 공식 KO/2 km lat/lon API가 존재한다.
- **말할 수 없는 것**: 공식 lat/lon 파일을 아직 받지 않아 grid join은 0이다. Area aggregation
  의미와 행정동 매핑도 미완이다. GK2A 결합 성능 증거는 **0**이다.
- **하지 않을 것**: 4-anchor fitted offset을 operational 좌표로 쓰지 않는다. GK2A를 forcing 또는
  국가 간 공통 live feature로 부르지 않는다.
