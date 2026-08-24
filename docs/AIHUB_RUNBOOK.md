# AI-Hub → H200 직접 수신 런북 (2026-08-25 검증)

## 결론: 다른 컴퓨터를 거칠 필요가 없다

AI-Hub에는 공식 CLI API가 있고(`aihubshell`), **H200에서 직접 받는다.**
노트북으로 받아 다시 올리는 경로는 쓰지 않는다 — 수십 GB를 두 번 옮기게 되고,
중간 사본이 남아 재배포 위반 위험만 늘어난다.

### 서버에서 실측한 도달성 (2026-08-25)

| 항목 | 결과 |
|---|---|
| `https://api.aihub.or.kr/api/aihubshell.do` | **http 200**, 7,824 bytes, 0.25 s |
| 받은 스크립트 버전 | `aihubshell version 25.09.19 v0.6` |
| `https://aihub.or.kr/` | http 200 |
| `/home/work/data` 여유 | **7.9 T** (9.1 T 중 14% 사용) |

용량·회선 모두 문제없다.

### aihubshell v0.6에서 확정한 동작 (스크립트 직접 확인)

```
인증   -aihubapikey '<키>'   또는   환경변수 AIHUB_APIKEY   ← line 94의 fallback
모드   l 조회 / d 다운로드 / pl 패키지조회 / pd 패키지다운로드
선택   -filekey <a,b,c>      기본값이 "all" 이므로 생략하면 전체를 받는다
동작   CWD에 download.tar 수신 → tar -xvf → .part 파일 자동 병합
```

**환경변수 이름은 `AIHUB_APIKEY`다.** (초기에 `AIHUB_API_KEY`로 잘못 적었다가
스크립트 확인 후 고쳤다.) 이 변수를 export하면 키를 명령행에 쓰지 않아도 되고,
셸 히스토리와 `ps` 목록에 남지 않는다.

## 절차

### 1. 승인 확인
`71363`, `71361` 상세페이지에서 다운로드 상태가 **승인**인지 확인한다.
API key만으로는 받아지지 않는다.

### 2. 키를 서버에 심는다 (한 번만)

키를 대화창·커밋·로그에 붙여넣지 않는다. 서버에서 직접 실행한다.

```bash
umask 077
printf 'AIHUB_APIKEY=%s\n' '<이메일로 받은 키>' > /home/work/data/olmoearth/.env.aihub
chmod 600 /home/work/data/olmoearth/.env.aihub
```

앞에 공백 한 칸을 두고 입력하면 zsh/bash 히스토리에도 남지 않는다
(`HISTCONTROL=ignorespace` 환경). 확실히 하려면 심은 뒤 `history -c`.

### 3. 인증 확인

```bash
set -a; . /home/work/data/olmoearth/.env.aihub; set +a
bash /home/work/data/code/aihub_setup.sh check
```

### 4. 무엇을 받을지 먼저 본다 — 전체를 받지 않는다

```bash
bash /home/work/data/code/aihub_setup.sh list 71363
```

`71363`은 드론 0.1m 20,000장 + Skysat 0.5m 25,000장 + **Sentinel-2 10m 3,000장**
+ Landsat 30m 2,000장이다. 우리에게 필요한 것은 **10m Sentinel-2 부분만**이다.
filetree 출력에서 해당 filekey를 골라낸다.

### 5. 부분 다운로드 + manifest 생성

```bash
setsid nohup bash /home/work/data/code/aihub_setup.sh get 71363 '<filekey,...>' \
  > /home/work/data/olmoearth/aihub/logs/get_71363.out 2>&1 &
```

`get`은 `filekey`를 **필수**로 받는다. 실수로 전체를 받는 것을 막기 위해 기본값을 두지 않았다.
완료 후 `manifest_71363.sha256`이 생성된다 — 논문에서 공개하려는 것은 원본이 아니라
이 manifest다 (`AIHUB_INQUIRY.md` 질문 2).

### 6. 받은 뒤 검증할 것 (다운로드 자체가 목적이 아니다)

| # | 확인 | 실패 시 |
|---|---|---|
| 1 | 10m 부분에 촬영시점(YYYYMMDD)이 파일·JSON에 실제로 있는가 | STAC 조회 불가 → 12밴드 물질화 불가 → 이 데이터셋 P0에서 제외 |
| 2 | EPSG:32652 폴리곤 좌표가 타일별로 유효한가 | 같음 |
| 3 | land-cover / 산사태·토석류 / 벌목지 라벨이 **같은 타일에서 겹치는가** | multi-task 구성 불가 → 단일 task로 축소 |
| 4 | 희소 클래스(산사태·벌목지) 양이 head 학습에 충분한가 | task를 탐지에서 존재여부 분류로 강등 |

3·4는 설계 가설이고 아직 미검증이다. 데이터를 보기 전에 실험 설계를 확정하지 않는다.

## 하지 않는 것

- 원본을 로컬로 내리거나 저장소에 커밋하지 않는다 (`.gitignore`가 `raw/*.zip`을 막지만 신뢰하지 않는다)
- 드론·Skysat·Landsat 부분을 받지 않는다 (ontology가 다르고 실험에 쓰지 않는다)
- 승인 회신 전에 파생물을 공개하지 않는다 (`AIHUB_INQUIRY.md`의 분기 표를 따른다)
