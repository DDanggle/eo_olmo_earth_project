# CLAUDE.md — 이 작업공간에서 일하는 방법

OlmoEarth 연구 작업공간입니다. **`README.md`를 먼저 읽으세요** (전체 정리·실험 대장·재현 절차).

## 두 저장소의 역할

| 위치 | 역할 |
|---|---|
| `~/dong/ai_projects/h100-setup` | 서버 접속 전용 (nexus CLI). 프로젝트 파일을 두지 않는다 |
| `.` (이 폴더, `olmoearth_projects/_work`) | 연구 문서·실험 코드. 자체 git 저장소 |
| `..` (`olmoearth_projects`) | Ai2 원본 레포 클론 — **PR 작업만**. 우리 파일을 여기 두지 않는다 |

## 작업 규약

1. 작업 시작 전 `GOAL.md`를 읽고, 오늘 할 일을 Worklog에 계획으로 먼저 적는다.
2. **서버 조작은 `./bin/nx`로만.** 긴 작업은 `./bin/nx sh` + `setsid nohup`으로 분리한다
   (SSH가 끊기면 전경 작업은 같이 죽는다). 산출물은 `/home/work/data/olmoearth/` 아래에만.
3. **분석 코드는 먼저 `code/`에 파일로 쓰고** 서버로 전송해 실행한다. SSH에 즉석 코드를
   흘려보내면 재현이 불가능해진다 (2026-08-21에 이 규칙이 생긴 이유).
4. 모든 서버 실행은 `env -u PYTHONPATH`로 감싼다 (다른 프로젝트의 PYTHONPATH가 venv를 오염).
5. 작업 종료 시 `GOAL.md` Worklog에 결과(수치 포함)·마찰·다음 단계를 기록하고, PR 후보는
   `## PR 후보` 절에 누적한다.
6. 새 개념을 부딪히면 `STUDY.md`에 개념 카드(+확인 질문)를 남긴다. 교과서식 선행 학습 금지 —
   실제 작업에서 만난 것만 카드로 만든다.
7. 결과를 보고할 때는 **먼저 약점·미검증 부분**을 말하고, 주장마다 근거 상태를 밝힌다.
   용어는 처음 나올 때 풀어서 쓴다.

## 자주 쓰는 명령

```bash
./bin/nx status                  # 세션 상태
./bin/nx tunnel up               # 터널 (컨테이너 바뀌면 필수)
./bin/nx sh 'nvidia-smi'         # 서버에서 한 줄 실행 (PYTHONPATH 차단됨)
./bin/nx push $PWD/code olmoearth/   # 코드 업로드
./watch.sh                       # GPU/작업/디스크 대시보드
```
