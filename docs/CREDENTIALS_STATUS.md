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

## data.go.kr — 키는 유효, 서비스 등록이 갈린다

동일 키로 서비스별 응답이 다르다. 이것이 원인을 특정한다.

| 서비스 | 엔드포인트 | 응답 | 해석 |
|---|---|---|---|
| 건축HUB 건축인허가 | `1613000/ArchPmsHubService/getApBasisOulnInfo` | `SERVICETIMEOUT_ERROR` (05) | **인증 통과.** 상류 서버 지연 → 재시도 사안 |
| 기상청 위성자료(경량화) | `1360000/WthrSatlitInfoService/getGk2aIrAll` | `SERVICE_KEY_IS_NOT_REGISTERED_ERROR` (30) | 오퍼레이션은 존재하나 **이 키에 서비스 미등록** |

같은 서비스의 다른 오퍼레이션 6개(`getGk2aLimbAll`, `getGk2aKmaL2`, `getGk2aL2Ctps`,
`getGk2aL2Cld`, `getGk2aVi006`, `getSatlitImgInfo`)는 모두 `NO_OPENAPI_SERVICE_ERROR` (12)
= 존재하지 않음. 즉 `getGk2aIrAll`이 올바른 이름이고, 문제는 **오퍼레이션이 아니라 서비스 등록**이다.

키 형태 문제도 아니다. Encoding(96자)과 decoding(88자) 양쪽 모두 동일하게 code 30이 나왔다.

### 남은 원인 두 가지

1. **승인 전파 지연.** data.go.kr은 자동승인 후 활용까지 시간이 걸린다. 방금 승인이면 대기.
2. **승인한 데이터셋이 다른 것.** 「위성자료(경량화)」와 「위성영상 조회서비스」는 별개 서비스다.
   승인 목록의 제목만으로는 어느 쪽인지 확정할 수 없다.

### 사용자가 확인해줄 것 (내가 볼 수 없는 화면)

마이페이지 → 데이터활용 → **활용신청 현황** → 해당 서비스 상세에서 다음 두 줄을 알려주면
바로 재검증한다.

- **요청주소(엔드포인트 전체 경로)**
- **일반 인증키 발급 상태 / 승인일시**

미확인 서비스 3개(환경영향평가 사업구역, 국토지리정보원 항공사진,
국립환경과학원 기상자료)도 요청주소를 알려주면 같이 테스트한다.

## 실험에서의 위치 — 이것들은 P0가 아니다

| 자산 | 역할 |
|---|---|
| AI-Hub 71363 (10m S2) | 한국 외부 stress test. 승인 나면 즉시 받는다 |
| VWorld | 행정경계·POI 보조. 검증용 |
| data.go.kr 기상 | 관측조건(구름·시점) 맥락. **headline claim에 쓰지 않는다** |
| 건축HUB | 건물 라벨 교차확인 후보 |

headline claim은 공개 benchmark(PhilEO·AvalCD·Sen12Landslides)에서 성립해야 한다.
따라서 data.go.kr 서비스가 하나도 안 열려도 논문은 진행된다.
