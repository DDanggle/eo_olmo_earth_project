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

### 남은 것 — 참고문서의 코드표 2개

`dateTime`과 `resultType`은 명세가 "참고 참조"이고 값 목록이 참고문서에만 있다.
형식 3종(`YYYYMMDDHHMM`/`YYYYMMDDHH`/`YYYYMMDD`) × `resultType` 7종을 시도했으나 전부
`resultCode 11`이었다. 추측을 반복하지 않는다.

필요한 것: 활용신청 상세 페이지의 **참고문서(OpenAPI 활용가이드)** 안
① `dateTime` 형식·유효범위 ② `resultType` 허용값 목록. 이 둘만 있으면 즉시 닫힌다.

**우선순위: P2.** 이 서비스는 관측조건(구름·안개·에어로졸) 맥락 레이어이고 headline claim에
쓰지 않는다. 닫히지 않아도 논문은 진행된다.

### 보안 — 인증키 재발급 권고

2026-08-25 대화 중 인증키 전문이 채팅에 입력됐다. 즉각적 위험은 낮으나 대화 기록에 남는다.
포털 마이페이지에서 **일반 인증키 재발급** 후 `.env`를 갱신하는 것을 권한다.
재발급하면 이 문서의 실측 결과는 유지되고 키만 교체된다.

## 실험에서의 위치 — 이것들은 P0가 아니다

| 자산 | 역할 |
|---|---|
| AI-Hub 71363 (10m S2) | 한국 외부 stress test. 승인 나면 즉시 받는다 |
| VWorld | 행정경계·POI 보조. 검증용 |
| data.go.kr 기상 | 관측조건(구름·시점) 맥락. **headline claim에 쓰지 않는다** |
| 건축HUB | 건물 라벨 교차확인 후보 |

headline claim은 공개 benchmark(PhilEO·AvalCD·Sen12Landslides)에서 성립해야 한다.
따라서 data.go.kr 서비스가 하나도 안 열려도 논문은 진행된다.
