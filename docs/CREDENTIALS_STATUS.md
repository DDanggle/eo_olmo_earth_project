# 자격증명 실측 상태 (2026-08-25)

`.env` 위치: **`olmoearth_projects/.env`** (`_work` 밖, upstream clone 루트).
`git check-ignore` 통과, 추적 안 됨, `git status`에 안 뜸 → PR 오염 위험 없음.
권한이 `644`였으므로 **`600`으로 수정**했다.

값은 어디에도 기록하지 않는다. 아래는 전부 실제 호출 결과다.

| 키 | 길이 | 실측 결과 |
|---|---|---|
| `AIHUB_APIKEY` | 36 | **정상.** 데이터셋 873개 목록 수신, 71363·71361 확인 |
| `VWORLD_API_KEY` + `VWORLD_API_DOMAIN` | 36 / 16 | **정상.** `status: OK`, 한라산 검색 411건 |
| `DATA_GO_KR_SERVICE_KEY` | 96 (Encoding 키, `%` 포함) | **키 자체는 유효.** 서비스별 등록 상태가 갈린다 (아래) |
| `ECVAM_API_KEY` | 0 | 비어 있음. 선택 항목이므로 문제 아님 |

## AI-Hub 71363 — Sentinel-2 filekey 확정

`aihubshell -mode l -datasetkey 71363` 실측 결과. **Sentinel-2 부분만 합계 약 2.8 GB다.**

| 구분 | 파일 | 크기 | filekey |
|---|---|---|---|
| Train 원천 | `TS_03. Sentinel2.zip` | 1 GB | **491163** |
| Train 라벨(TIF) | `TL_01.LABEL_03. Sentinel2.zip` | 15 MB | **491167** |
| Train 라벨(JSON) | `TL_02.JSON_03. Sentinel2.zip` | 724 MB | **491171** |
| Valid 원천 | `VS_03. Sentinel2.zip` | 151 MB | **491175** |
| Valid 라벨(TIF) | `VL_01.LABEL_03. Sentinel2.zip` | 2 MB | **491179** |
| Valid 라벨(JSON) | `VL_02.JSON_03. Sentinel2.zip` | 82 MB | **491183** |
| 메타데이터 | `01.메타데이터_03. Sentinel2.zip` | 1 MB | **533616** |
| SHP | `02.SHP_03. Sentinel2.zip` | 794 MB | **533620** |

받지 않는 것: Drone 39 GB, Skysat 43 GB, Landsat 401 MB + 관련 라벨.
ontology가 다르고 실험에 쓰지 않는다. **전체는 약 85 GB, 우리는 3.3%만 받는다.**

원천데이터(TS/VS)도 받는 이유: 우리가 STAC로 물질화한 12밴드가 이들이 쓴 관측과 같은지
대조하는 검증(C2-C와 같은 성격)에 필요하다. 1.15 GB로 싸다.

```bash
bash /home/work/data/code/aihub_setup.sh get 71363 \
  '491163,491167,491171,491175,491179,491183,533616,533620'
```

## data.go.kr — 승인 3건 확인, 엔드포인트 확정, 오퍼레이션명 미확정

키는 `.env`와 포털 발급값이 일치한다 (앞8/뒤8 대조). 승인 3건 모두 2026-08-22 신청,
2028-08-22 만료, 개발계정 자동승인.

### 확정된 End Point (사용자 포털 화면 기준)

| 서비스 | End Point |
|---|---|
| 기상청 위성자료 경량화(기상산출물) | `https://apis.data.go.kr/1360000/CloudSatlitInfoService` |
| 국립환경과학원 환경영향평가 사업구역정보 | `https://apis.data.go.kr/1480523/BsnsAreaService` |
| 국토교통부 건축HUB 건축인허가정보 | `1613000/ArchPmsHubService` (실측으로 확인) |

**내가 처음 테스트한 `WthrSatlitInfoService`는 다른 서비스였다.** 웹 검색이 「위성자료(경량화)」와
「위성영상 조회서비스」를 계속 혼동했고, 그 결과 code 30(미등록)을 서비스 미승인으로 오해할
수 있었다. 실제 원인은 **엔드포인트가 달랐던 것**이다. 포털 화면이 유일한 정확한 출처다.

### 실측 상태 (2026-08-25, 오퍼레이션명 확보 후)

| 서비스 | 결과 |
|---|---|
| 건축HUB `ArchPmsHubService/getApBasisOulnInfo` | `SERVICETIMEOUT_ERROR` (05) — **인증 통과.** 상류 지연 |
| 기상청 `CloudSatlitInfoService/getGk2acldAll` | `resultCode 11 NO_MANDATORY_REQUEST_PARAMETERS_ERROR` — **인증 통과** |

**`resultCode 11`은 게이트웨이 오류가 아니라 서비스 자체 응답이다.** 즉 키·서비스·오퍼레이션이
모두 정상이고, 남은 것은 파라미터 값이다. code 30/12가 아니라는 점이 결정적이다.

`CloudSatlitInfoService` 상세기능 (포털 확인):

| 오퍼레이션 | 내용 |
|---|---|
| `getGk2acldAll` | 천리안위성2A호 구름탐지 한반도 |
| `getGk2aappsAll` | 에어로졸 산출물 한반도 |
| `getGk2afogAll` | 안개 한반도 |
| `getGk2adcoewAll` | 주간구름 산출물 한반도 |
| `getGk2aclaAll` | 구름분석 한반도 |
| `getGk2acldArea` | 구름탐지 **행정구역** |
| `getGk2aap…Area` | 에어로졸 행정구역 (이하 행정구역 계열) |

요청변수: `ServiceKey`, `pageNo`, `numOfRows`, `dataType`(XML/JSON),
`dateTime`(**참고 참조**), `resultType`(**참고 참조**).

### 10개 오퍼레이션 전부 통과 — 파라미터 규칙 확정 (2026-08-25)

`resultType`은 **출력 형식이 아니라 산출물 변수 코드**였다. 이것이 막혔던 이유다.
`dateTime`은 `YYYYMMDDHHMM` 10분 슬롯. 키는 **Encoding 형태를 쿼리스트링에 그대로** 넣는다
(재인코딩하면 실패). 행정구역 계열은 `dongCode`가 추가로 필수다.

| 오퍼레이션 | resultType | 추가 |
|---|---|---|
| `getGk2acldAll` / `…Area` | `CLD` (구름탐지) | Area는 `dongCode` |
| `getGk2aappsAll` / `…Area` | `AOD` (에어로졸) | 〃 |
| `getGk2afogAll` / `…Area` | `FOG` (안개) | 〃 |
| `getGk2adcoewAll` / `…Area` | `COT` (주간구름 광학두께) | 〃 |
| `getGk2aclaAll` / `…Area` | `CT` (구름분석·구름형) | 〃 |

10/10 모두 `resultCode 00 NORMAL_SERVICE`. 검증한 `dongCode` 예: `1111051500`.
재현: `code/probe_datagokr_gk2a.py` (규칙 발견) → `code/probe_datagokr_gk2a_codes.py` (코드표 확정).
결과: `artifacts/datagokr_gk2a_codes.json`.

응답 형태:

```
한반도(All)    2 km 격자. gridKm=2.0, xdim=320, ydim=397, x0/y0 오프셋 + value 평탄배열
행정구역(Area) lon/lat + value (야간 구름형은 -9999 = 결측)
```

### 정정 — "서버에서 차단됨"은 내 오진이었음

처음에 서버에서 60초 타임아웃(http=000)이 나서 "한국 공공API가 이 클라우드 대역을 막는다"고
적었음. **틀렸음.** 원인은 내가 `numOfRows=200000`으로 요청해 **API의 동시호출 락**이 걸려
있던 것이었음 — 앞서 본 `resultCode 99 "이미 호출중에 있습니다"`와 같은 증상임.

락이 풀린 뒤 서버에서 재검증한 결과:

| | 결과 |
|---|---|
| `apis.data.go.kr` 루트 | http 400, 0.085 s (파라미터 없는 정상 반응) |
| `getGk2acldAll` 실제 호출 ×3 | **3/3 http 200, 0.22~0.31 s, 254 KB** |
| 대조: `api.vworld.kr` / `api.aihub.or.kr` / `www.kma.go.kr` | 전부 200/302, 0.1~0.3 s |

**차단 없음.** 교훈: 타임아웃을 네트워크 차단으로 단정하기 전에 요청 자체가 과대한지 본다.

### 수집 운영 — 어디에 어떻게 쌓이는가

```
~/dong/ai_projects/data/gk2a/YYYY/MM/DD/
  ├─ getGk2acldAll_CLD_0000.json.gz     구름탐지 00시
  ├─ getGk2aappsAll_AOD_0000.json.gz    에어로졸
  ├─ getGk2afogAll_FOG_0000.json.gz     안개
  ├─ getGk2adcoewAll_COT_0000.json.gz   주간구름 광학두께
  ├─ getGk2aclaAll_CT_0000.json.gz      구름분석
  │   … 03/06/09/12/15/18/21시 반복 (하루 5종 × 8슬롯 = 40개)
  └─ manifest.jsonl                      시각·바이트·sha256·http상태·resultCode
```

응답 원본을 gzip으로 그대로 보존함. 파생 가공은 나중에 함.
저장 위치는 `GK2A_ROOT` 환경변수로 바꿀 수 있음.

**얼마나 자주**: 보존이 2일이므로 **2일에 한 번**이면 충분함. 매일 아니어도 됨.

```bash
cd ~/dong/ai_projects/olmoearth_projects && set -a && . ./.env && set +a
python3 _work/code/gk2a_snapshot.py --gaps     # D-1, D-2 중 빠진 것만 보충
python3 _work/code/gk2a_snapshot.py --status    # 쌓인 상태·빠진 날·용량
```

`--gaps`는 이미 받은 파일을 건너뛰므로 여러 번 돌려도 안전함(idempotent).

**현재 상태 (2026-08-25)**: 2일분 70파일 0.8 MB. 하루 평균 0.38 MB → **1년 약 0.14 GB.**
용량은 사실상 문제가 아님.

**하루 40개 중 5개는 항상 실패함** — `resultCode 03 NO_DATA`임. 야간(15·18·21시 등)에
주간구름 산출물(COT)이 존재하지 않는 정상 결측이며 오류가 아님.

**서버에 cron이 없고 세션이 GPU 6시간 미사용 시 회수되므로** 서버를 상주 수집 호스트로
쓰지 않음. 2일 여유가 있으니 수동으로 충분함.

### 치명적 제약 — 이 API는 71363에 결합할 수 없다

```
dateTime=202608240000  -> code 00 (정상)
dateTime=202608170000  -> code 99 "최대 조회 기간은 오늘 기준으로 2일 전까지입니다."
dateTime=202210290300  -> code 99 (같음)
```

**보존 기간이 2일이다.** AI-Hub 71363의 촬영일은 2019-01-03 ~ 2022-10-29이므로
이 API로 그 시점의 구름·안개·에어로졸 조건을 **소급 조회할 수 없다.**

따라서 역할이 바뀐다.

| 원래 계획 | 실제 가능한 것 |
|---|---|
| 71363 타일의 관측조건 맥락 레이어 | **불가** (2일 보존) |
| — | **전향적(prospective) 운영 데모**: 실시간 구름·안개 조건에 따른 캐시 갱신 판단 |

71363의 관측조건은 다른 경로로 얻어야 한다 — Sentinel-2 자체의 SCL 밴드,
또는 Planetary Computer STAC 아이템의 `eo:cloud_cover` 메타데이터.
어느 쪽이든 12밴드 물질화(P0)에서 같이 나온다.

## 실험에서의 위치 — 이것들은 P0가 아니다

| 자산 | 역할 |
|---|---|
| AI-Hub 71363 (10m S2) | 한국 외부 stress test. 승인 나면 즉시 받는다 |
| VWorld | 행정경계·POI 보조. 검증용 |
| data.go.kr GK2A | **소급 조회 불가(2일 보존).** 전향적 운영 데모 전용. headline claim에 쓰지 않는다 |
| 71363 관측조건 | Sentinel-2 SCL 밴드 또는 STAC `eo:cloud_cover`로 대체 — P0에서 함께 확보 |
| 건축HUB | 건물 라벨 교차확인 후보 |

headline claim은 공개 benchmark(PhilEO·AvalCD·Sen12Landslides)에서 성립해야 한다.
따라서 data.go.kr 서비스가 하나도 안 열려도 논문은 진행된다.
