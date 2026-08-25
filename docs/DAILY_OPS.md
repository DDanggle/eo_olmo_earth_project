# 매일(또는 2일마다) 해야 하는 것

> **이 저장소에서 미루면 영구히 손실되는 작업은 이것 하나뿐임.**
> 실험·학습·집필은 나중에 해도 됨. GK2A 수집은 안 됨.

## GK2A 스냅샷 — 2일에 한 번

```bash
cd ~/dong/ai_projects/olmoearth_projects && set -a && . ./.env && set +a
python3 _work/code/gk2a_snapshot.py --gaps      # 받고
python3 _work/code/gk2a_snapshot.py --status     # 확인
```

`--gaps`는 이미 받은 파일을 건너뛰므로 여러 번 돌려도 안전함.

### 왜 미룰 수 없는가

기상청 GK2A 경량화 API의 응답:

```
dateTime=202608170000 → resultCode 99
                        "최대 조회 기간은 오늘 기준으로 2일 전까지입니다."
```

**보존이 2일임.** 오늘 받지 않은 날짜는 다시 받을 방법이 없음. 돈으로도, 나중에도 안 됨.

### 얼마나 자주

| 마지막 실행 후 | 상태 |
|---|---|
| 1일 이내 | 안전 |
| 2일 | **한계.** 지금 돌려야 함 |
| 3일 이상 | 하루 이상 영구 소실됨 |

`--status`가 "빠진 날 N일"을 보여줌. N이 0이 아니고 그 날짜가 D-2보다 오래됐으면 복구 불가임.

### 규모 (걱정할 것 없음)

```
하루 72파일 · 1.07 MB   →   1년 0.39 GB   →   10년 4 GB
```

### 저장 위치

```
~/dong/ai_projects/data/gk2a/YYYY/MM/DD/
  ├─ getGk2acldAll_CLD_HHMM.json.gz     구름탐지   2시간 간격 × 24h
  ├─ getGk2afogAll_FOG_HHMM.json.gz     안개       2시간 간격 × 24h
  ├─ getGk2aappsAll_AOD_HHMM.json.gz    에어로졸   1시간 간격 08~18h
  ├─ getGk2adcoewAll_COT_HHMM.json.gz   구름광학두께 1시간 간격 08~18h
  ├─ getGk2aclaAll_CT_HHMM.json.gz      구름분석   1시간 간격 08~18h
  └─ manifest.jsonl                     시각·바이트·sha256·http·resultCode
```

응답 원본을 gzip으로 그대로 보존함. 가공은 나중에 함 — 지금 가공하면 되돌릴 수 없음.
`GK2A_ROOT` 환경변수로 위치를 바꿀 수 있음.

### 나중에 서버로 옮길 때

서버에서도 API가 정상 동작함(실측 3/3, 0.22 s). 다만 서버에 cron이 없고 세션이
**GPU 6시간 미사용 시 회수**되므로 상주 수집 호스트로는 신뢰할 수 없음.
옮길 때는 데이터만 동기하고 수집 주체는 유지하는 편이 안전함.

```bash
rsync -a ~/dong/ai_projects/data/gk2a/  <서버>:/home/work/data/olmoearth/gk2a/
```

## 정상인데 실패로 보이는 것

- **`resultCode 03 NO_DATA`**: 그 시각에 산출물이 없음. 정상임
- **AOD/COT/CT의 야간 슬롯**: 태양광 반사가 필요해 격자 전체가 `-9999`임.
  그래서 주간(08~18시)만 받도록 일정을 분리했음
- **AOD 유효율 20~33%**: 에어로졸 산출은 맑은 하늘에서만 가능함. 낮은 게 정상임
