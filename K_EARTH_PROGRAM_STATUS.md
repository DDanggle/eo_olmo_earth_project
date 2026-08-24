# K-Earth 프로그램 현황 — 데이터 신청부터 EarthRoute까지

최종 갱신: 2026-08-22  
범위: 제주 오름 368 고정분모, OlmoEarth 2023–2026 관측, 한국 공공·행정·환경 근거

## 한 줄 판정

**공공데이터와 시계열 관측을 함께 보유하면 연구·사업 가치가 커진다.** 다만 공개 원본 자체가
차별점은 아니다. 같은 장소를 반복 관측한 위성·항공·행정 snapshot, 누락 사유, 사람 판정,
모델 릴리스 결과를 하나의 versioned event store로 묶을 때 비로소 재현 가능한 한국형 데이터
자산이 된다.

`API 호출 가능`과 `데이터를 보유함`은 다르다. 이 프로젝트에서 보유는 다음 다섯 조건을 뜻한다.

1. 원본 파일 또는 합법적으로 재수집 가능한 request를 보존한다.
2. `snapshot_date / valid_from / valid_to / retrieved_at`을 분리한다.
3. SHA-256, schema, 좌표계, 라이선스와 모집단 coverage를 manifest로 고정한다.
4. PNU·공식 polygon·관측일로 같은 장소와 시점을 연결한다.
5. no-match와 사람의 보류 판정을 지우지 않고 다음 snapshot과 비교한다.

## 1. 현재 공공데이터 신청과 무엇이 필요한가

### 현재 접근 상태

아래 상태는 2026-08-22 실제 bounded API snapshot까지 반영한다. 키 값 자체는 출력·기록하지
않았고 request manifest에는 credential 변수명만 남겼다. **HTTP 성공**과 **API 본문 성공**도
분리했다. 현재 canonical v3는 HTTP 463/463, 의미상 성공 456, 유효 무항목 1,
과거 GK2A 제한 오류 6이다.

| 우선 | 신청/다운로드 | 계정·키 | 받아야 하는 필드 | 연구에서의 역할 | 현재 상태 |
|---:|---|---|---|---|---|
| 1 | [VWorld 연속지적도 2.0](https://www.vworld.kr/dev/v4dv_2ddataguide2_s003.do?svcIde=cadastral) | 별도 dataset 신청 없음; VWorld key+domain | PNU, polygon, CRS, 기준년월 | 모든 행정 사건의 geometry spine | **대표점 gate 통과, 256/257 feature**; 후보 14/14·오름 242/243 PNU |
| 2 | [환경영향평가 사업구역 WFS](https://www.data.go.kr/data/15142907/openapi.do) | 공공데이터포털 자동승인 + 서비스키 | `the_geom`, `MGTNO`, 사업명, 등록·수정일 | 변화 footprint와 공식 사업구역 교차 | **제주 bbox 성공: 13 polygon**, 14후보 직접중첩 0 |
| 3 | [건축HUB 건축인허가](https://www.data.go.kr/data/15136267/openapi.do) | 공공데이터포털 서비스키 | 대지/PNU, 허가·착공·사용승인·철거일, 용도 | 건축 사건의 시간축 | **45 법정동·111 page·8,794행**, 철거 조건 내 0행 |
| 4 | [기상청 위성자료 경량화 조회](https://www.data.go.kr/data/15077314/openapi.do) | 공공데이터포털 서비스키 | GK2A 구름 산출물, 관측시각, grid | Sentinel 공통 구름오류 독립감사 | 최신 grid 127,040값 성공; **과거 6관측일은 최근 2일 제한** |
| 5 | VWorld 용도지역·도시계획 | VWorld key+domain | polygon, 시설/용도 code, 고시·갱신정보 | 규제·계획 문맥; 원인 아님 | 연속지적 key 검증 완료; 별도 P1 probe 대기 |

VWorld는 등록 domain을 `.env`의 `VWORLD_API_DOMAIN=http://localhost`로 고정한 뒤 대표점
`status=OK`를 확인했다. 257점 확장에서 업무 오류는 0이고 `JJ-OREUM-190` 한 점만
`NOT_FOUND`였다. 이는 비개발 증거가 아니라 point coverage 누락이다. EIA WFS도 공유
공공데이터포털 키로 실제 13 feature를 반환했으므로 승인/adapter 단계는 통과했다.

### 키만으로 해결되지 않는 수동 확보

| 자료 | 필요한 행동 | 시간축 요구 | 주의 |
|---|---|---|---|
| [국토지리정보원 항공사진](https://www.data.go.kr/data/15059918/fileData.do) | 국토정보맵 로그인 → 주소/지명 → 항공사진 → 연도 → 신청·승인 후 TIFF | 후보별 before/after 실제 촬영일 | 사용자 링크 1347은 `지리OneView` 도구 공지이지 항공사진 다운로드 페이지가 아님 |
| [환경부 토지피복지도 WMS](https://aid.mcee.go.kr/api/land.do) | 공개 연도별 WMS를 먼저 probe; SHP가 필요할 때만 로그인 신청 | 레이어 연도와 원영상 기준일 모두 보존 | 공개 WMS에는 별도 ECVAM 키가 필요하지 않음 |
| 생태자연도·과거 고시도 | 최신판 다운로드, 과거판은 기관 협의 | 고시 version·유효기간 | 현행도를 과거 원인으로 사용 금지 |
| 사유림 사업·임상도·국가유산·보호지역 | 파일 snapshot 다운로드 | 사업연도/지정연도/version | 출처별 재배포·변경 조건 별도 |

### 사용자가 주면 가장 먼저 연결할 것

실제 secret 값을 문서나 채팅에 붙이지 않는다. 프로젝트에는 다음 **이름만** 설정한다.

```text
DATA_GO_KR_SERVICE_KEY
VWORLD_API_KEY
VWORLD_API_DOMAIN
```

`ECVAM_API_KEY`는 토지피복용이 아니라 [국토환경성평가지도 Open API](https://ecvam.neins.go.kr/api/apiGuide.do)의
보전가치·법제·환경생태 WMS 71종을 위한 **선택 P2 키**다. 이름·이메일·사용 URL을
[신청 페이지](https://ecvam.neins.go.kr/api/apiWrite.do)에 등록하고 이메일 인증 후 발급받는다.
현재 PNU/EIA/건축/GK2A 실험에는 필요하지 않으므로 비워둬도 된다.

VWorld 확장은 끝났다. 다음은 representative point parcel을 오름 경계로 오인하지 않고 실제 변화
footprint와 대조하며, NGII 전후 항공사진으로 변화시점을 독립 검수하는 것이다. BuildingHUB·EIA·
GK2A·토지피복은 원본·request hash·pagination을 보존했다. 과거 GK2A는 현재 endpoint로 소급
조회할 수 없으므로 Sentinel SCL 또는 별도 archive를 사용해야 한다.

## 2. 현재 상황은 어떤가

### 현재 로컬에 보존된 것

| 자산 | 현재 수치 | 근거 상태 | 남은 구멍 |
|---|---:|---|---|
| 제주 공식 오름 목록 | 368/368 상태화 | A급 목록 속성 | 공식 경계·좌표 없음 |
| offline OSM 위치 seed | 243/368 | C급 point/name | 공식 오름 polygon 아님 |
| OlmoEarth 점별 screen | 243/368 | M급 조사 우선순위 | 2023 공통 구름오류, v1.2 pair 미완료 |
| RGB 직접 검수 | 9건 | 오염/기각 8, 불확실 1 | 확률표본이 아니므로 전체 오류율 추정 불가 |
| 제주 FarmMap 원본 | 289,379 polygon | 2025-12-31 snapshot, 원본 ZIP+manifest 보존 | 농경지만 포함, 원인 사건 아님 |
| 개발행위허가 snapshot | 240행·223 PNU | 파일·coverage audit 보존 | 2023·2024 누락, no-match 해석 불가 |
| 제주시 산지이용 집계 | 2008–2026 19행 | 연간 집계 시계열 | 필지 join 불가 |
| API bounded snapshot v3 | HTTP 463/463·semantic 성공 456·무항목 1·오류 6·약 32 MB | raw/hash/request·입력 SHA·완주 marker 보존 | 과거 GK2A 6시점 제한 |
| VWorld 지적 | 257점 중 256 feature·고유 PNU 235 | 후보 14/14·오름 242/243 | 대표점 필지≠오름 경계; 14 PNU를 35점이 공유 |
| BuildingHUB·EIA | 8,794 event행·13 polygon | 45 법정동 pagination 소진 | 후보 exact PNU 1이나 시간정렬 0; EIA 중첩 0 |
| 토지피복·GK2A | 42 PNG·최신 127,040 grid값 | 연도/관측시각 상태근거 | 원인 아님; 역사 cloud grid 없음 |
| 원인 A/B 근거 | 0/368·14후보 0/14 | 현재 snapshot에 한정 | 실제 변화 footprint·전후 항공·사유림 사건 미연결 |

현재 원본 디렉터리는 약 112 MB이고 FarmMap ZIP·산지이용 CSV를 포함한다. 단, 제주 Sentinel
materialized tensor·원영상은 서버 중심이며 로컬에는 5,184행의 `dataset × 4개년 × 54윈도우 ×
12기간` time-axis manifest가 주로 남아 있다. v1/v5는 실제 item/pixel이 같았으므로 독립된 두
시계열로 세지 않는다. 현재는 **시계열 provenance 보유, 장기 원본 custody는 불완전**한 상태다.

### 다음 데이터 snapshot의 최소 스키마

```text
site_registry(site_id, official_name, official_geom?, representative_pnu?)
observation_event(site_id, sensor, scene_id, captured_at, quality, pixel_hash)
administrative_event(site_id?, pnu?, event_geom?, event_type, event_date, source_version)
context_state(site_id?, state_geom, class, observed_at, source_version)
human_review(site_id, imagery_ids, reviewer, decision, reason, reviewed_at)
model_run(site_id, input_hash, model_release, output_hash, calibration_version)
```

이 여섯 표가 같은 장소에서 여러 snapshot을 쌓을 때만 `행정기록 변화`, `현실 변화`, `센서 입력
변화`, `모델 변화`를 분리할 수 있다.

## 3. 비즈니스적인 가능성은 있는가

판정은 **가능성 있음, 시장 검증 전**이다. 판매 대상은 공공데이터 묶음이나 transfer learning
모델이 아니라 반복 결정의 감사 산출물이다.

| 상품 가설 | 고객 가설 | 고객이 사는 결과 | 현재 증거 | promotion gate |
|---|---|---|---|---|
| Post-EIA Evidence Pack | 환경영향평가 대행사·사업자 환경팀 | 사업구역의 전후 관측·공식근거·보류 사유 dossier | 법정 반복 workflow는 있음; 인터뷰 0 | 기존 검수시간 30%↓, 원인 오단정 0, 두 번째 유료 갱신 |
| GeoFM Release Audit | EO/GIS 업체·공공 지도 운영팀 | v1→v1.2 변경 때 유지/재계산/재검수 범위 | FoldRefresh 로컬 방법 자산 | 실제 재계산비 30–40%↓, 결정지표 손실 ≤1%p |
| Local Adaptation Sprint | 농업·산림·연안 조직 | 적은 현지 라벨로 지역 map과 보류정책 | 완도→제주 smoke만 있음 | 라벨 50%↓ 또는 5%p↑, 두 태스크·두 지역 반복 |

공공 원본은 경쟁사도 받을 수 있다. 방어력은 다음 누적물에서 생긴다.

- 동일 AOI의 여러 시점 raw snapshot과 request/manifest.
- 서로 다른 행정시스템의 누락 패턴과 `no-match interpretable` label.
- 사람이 본 before/after chip, 보류 이유와 판정 불일치.
- 모델 릴리스가 바뀔 때 실제로 변한 후보·집계·검수비용의 history.

따라서 API를 많이 연결한 대시보드는 사업이 아니고, **같은 고객이 두 번째 갱신 비용을 내는가**가
사업 증거다.

## 4. 한국 연구를 어떻게 봐야 하는가

한국형 연구의 핵심은 한국 지도를 많이 붙이는 것이 아니다.

> **불완전하고 서로 다른 시점의 행정자료 아래에서, Earth foundation model이 언제 말하고
> 언제 보류해야 하며 그 침묵이 어느 지역에 편향되는가?**

현재 가장 강한 논문 질문은 *Selective Change Detection under Incomplete Administrative
Evidence*다.

1. 고정분모 368에서 source별 `time-aligned coverage`를 측정한다.
2. `model+OSM → +PNU → +날짜별 상태 → +행정사건 → +항공/현장` ablation을 한다.
3. 각 단계의 risk–coverage, 보류율, 지역·토지피복별 침묵률을 본다.
4. Top-k와 전체 변화율을 분리하고, 전체 수치는 확률표본+Prediction-Powered Inference로 추정한다.
5. 제주 밖 두 번째 지역·두 번째 태스크에서 정책을 무튜닝 검증한다.

원인 A/B 근거가 계속 10% 미만이면 원인 분류 논문으로 확장하지 않는다. 그 자체를 행정자료
coverage와 선택적 예측의 결합 문제로 삼는다. 이 경계가 일반 변화탐지나 범용 OlmoEarth
fine-tuning과 구분되는 연구 기여다.

## 5. 8월 연구 노트는 어떻게 따로 확장할 수 있는가

8월 EarthRoute 노트는 현재 K-Earth를 대체하지 않고 후속 프로그램으로 둔다.

```text
K-Earth: 무엇을 말할 수 있는가
    ↓
FoldRefresh: 무엇을 다시 계산하지 않아도 되는가
    ↓
EarthRoute: 다음에 관측·모델·행정근거·사람검수 중 무엇을 살 것인가
```

첫 EarthRoute action은 `reuse / cheap_refresh / escalate` 세 개만 사용한다. provider·sensor·model·
resolution·agent를 한꺼번에 열지 않는다.

- Paper 1: K-Earth selective evidence와 missingness bias.
- Paper 2: FoldRefresh를 붙인 model-release decision continuity.
- Paper 3 후보: evidence acquisition까지 포함한 EarthRoute cost–risk policy.

Paper 3은 full run 대비 oracle end-to-end 비용 30–40% 절감, learned policy가 oracle saving의 70%
회수, 실제 runtime 1.5배 이상 개선을 먼저 보여야 연다. 실패해도 Paper 1과 Evidence Pack은 남는다.
세부 문헌·사업·license 경계는 `EARTHROUTE_PROGRAM_NOTE.md`, 논문 장부는
`PAPER_READING_LIST.md`가 맡는다.

## 지금 사용자에게 필요한 실제 행동

1. VWorld representative parcel과 OlmoEarth 변화 footprint의 공간관계를 후보별로 검수한다.
2. `r08`의 dated FarmMap↔current VWorld PNU 충돌을 source 기준일·경계 geometry로 해소한다.
3. NGII에서 우선 10–20개 사전 고정 후보의 before/after 항공사진을 신청한다.
4. 과거 GK2A 대신 사용할 archive 존재 여부를 확인하되, 없으면 Sentinel SCL 품질근거로 제한한다.
5. BuildingHUB/EIA/토지피복 동일 request를 분기·연간 재수집해 snapshot drift를 만든다.
6. ECVAM은 P0 완료 전에는 신청하지 않아도 된다.
