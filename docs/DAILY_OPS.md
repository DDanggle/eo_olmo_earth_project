# 짧은 조회창 데이터 운영

> 현재 등록된 recurring time-critical capture는 GK2A 경량화 endpoint 하나다.
> “저장소에서 유일한 되돌릴 수 없는 작업” 또는 “놓치면 원자료가 영구 소실”이라고는 말하지 않는다.

## GK2A 스냅샷 — 매일 권장, D-2가 마지막 경량화 창

상태 확인에는 API 키가 필요 없다.

```bash
cd ~/dong/ai_projects/olmoearth_projects
python3 _work/code/gk2a_snapshot.py --status

set -a && . ./.env && set +a
python3 _work/code/gk2a_snapshot.py --gaps
python3 _work/code/gk2a_snapshot.py --status
```

`--gaps`는 현행 57개 예정 슬롯에 대해 성공 파일 또는 명시적 `NO_DATA`가 있으면 건너뛴다.
따라서 여러 번 실행해도 같은 성공 응답을 덮어쓰지 않는다.

### 정확히 무엇이 2일인가

data.go.kr의 **경량화 API endpoint**에서 다음 응답을 실측했다.

```
dateTime=202608170000 → resultCode 99
                        "최대 조회 기간은 오늘 기준으로 2일 전까지입니다."
```

이것은 해당 endpoint의 접근 창이지 GK2A 원자료 전체의 보존기간 증거가 아니다. 기상청 API Hub는
GK2A L2 binary 다운로드와 2 km KO grid의 공식 lat/lon 조회·NetCDF 다운로드를 별도로 제공한다.
놓친 날짜가 그 경로에서 동일 산출물로 복구되는지는 아직 product/processing-version 계약을
감사하지 않았다. 따라서 현재의 정확한 표현은 다음과 같다.

- D-1/D-2: 이 스크립트로 경량화 응답을 직접 보충할 수 있음
- D-3 이전: 이 endpoint에서는 보충 불가
- 대체 archive: 존재하지만 접근·동일성·라이선스 미검증

매일 실행을 권장한다. 이틀 간격은 시간대·장애·산출 지연에 대한 여유가 없다.

### 완전성 계약

현행 스케줄은 **57파일 보장**이 아니라 **57개 요청 슬롯의 terminal outcome**이다.

| 산출물 | 슬롯 |
|---|---:|
| CLD, FOG | 각 2시간 간격 12개 = 24 |
| AOD, COT, CT | 각 08~18시 11개 = 33 |
| 합계 | **57** |

terminal outcome은 둘 중 하나다.

1. 원본 JSON을 gzip한 파일 + SHA-256이 manifest에 존재
2. API가 `resultCode 03 NO_DATA`를 명시했고 manifest에 보존

2026-08-23·24 실측은 각각 `data 54 + NO_DATA 3 = 57/57`로 완전하다. 두 날짜 모두 15시
AOD/COT/CT가 NO_DATA였다. 디렉터리에는 구 스케줄 파일 18개가 더 있어 실제 파일은 72개지만,
그 수를 현행 계약의 분모로 쓰지 않는다.

### 저장 위치

```
~/dong/ai_projects/data/gk2a/YYYY/MM/DD/
  ├─ getGk2acldAll_CLD_HHMM.json.gz
  ├─ getGk2afogAll_FOG_HHMM.json.gz
  ├─ getGk2aappsAll_AOD_HHMM.json.gz
  ├─ getGk2adcoewAll_COT_HHMM.json.gz
  ├─ getGk2aclaAll_CT_HHMM.json.gz
  └─ manifest.jsonl

~/dong/ai_projects/data/gk2a/_crs/area_anchors.jsonl
```

일별 gzip은 원본 응답이다. `area_anchors.jsonl`은 Area 응답에서 lon/lat/value만 파싱해 누적한
별도 산출물이며 원본 응답 보존이라고 부르지 않는다. `GK2A_ROOT`로 위치를 바꿀 수 있다.

현재 2일 실측 총량은 144파일·2.1 MB(구 스케줄 extra 포함), 약 1.07 MB/일이다. 스케줄이
안정된 7일 뒤 현행 57-slot 기준 용량을 다시 추정한다.

### 서버 동기화

로컬·서버 모두 API 호출 성공을 확인했다. 다만 GPU 세션 회수와 cron 부재 때문에 서버를
상주 수집 호스트로 가정하지 않는다.

```bash
rsync -a ~/dong/ai_projects/data/gk2a/  <서버>:/home/work/data/olmoearth/gk2a/
```

### 정상인데 실패로 보이는 것

- **`resultCode 03 NO_DATA`**: 해당 시각 산출물이 없다는 terminal outcome. 현재는 재시도하지 않음
- **AOD/COT/CT 야간 `-9999`**: 주간 광학 산출물이므로 현행 스케줄에서 야간을 제외함
- **AOD 낮은 유효율**: clear-sky retrieval의 missingness일 수 있으므로 버리지 않고 품질 변수로 보존

## 아직 없는 운영 보장

README 배너는 scheduler가 아니다. launchd/별도 상시 호스트, 마지막 성공시각 alert, 원격 replica,
KMA archive recovery drill은 아직 없다. 따라서 현재 상태는 **2일치 완전 수집 + 수동 운영**이지
무인 production pipeline이 아니다.

## 2026-09-04 갱신 — 자동화 등록
- 2026-08-28 ~ 09-01 **5일 미수집**(수동 실행이 끊김). 경량화 endpoint 창(D-2)을 넘어 이 경로로는 복구 불가. 대체 archive(API Hub L2/NetCDF)에서의 동일성은 미감사.
- 09-02·09-03은 오늘 `--gaps`로 보충함(결과는 `~/Library/Logs/gk2a/daily.log`).
- launchd 등록: `~/Library/LaunchAgents/kr.dgyi.gk2a.daily.plist` → `bin/gk2a_daily.sh`, 매일 09:30·21:30(놓친 회차 대비 하루 2회, 멱등). 맥이 꺼져 있으면 실행 안 됨 → 이틀 연속 꺼두면 손실. 확인: `launchctl list | grep gk2a`, `tail ~/Library/Logs/gk2a/daily.log`.
