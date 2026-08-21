# OlmoEarth 프로젝트 — 전체 정리 및 인수인계

최종 갱신: 2026-08-21

이 파일이 **입구**입니다. 처음 보는 사람(또는 새 컴퓨터의 나)은 여기부터 읽습니다.

## 이 폴더의 위치와 두 저장소의 관계

```
~/dong/ai_projects/
├── h100-setup/                    # [접속 전용] kt cloud AI Nexus 세션·터널·전송 (nexus CLI)
└── olmoearth_projects/            # Ai2 원본 레포 클론 (PR 작업용)
    └── _work/                     # ★ 여기 — 우리 연구 작업공간 (자체 git 저장소)
        ├── README.md GOAL.md STUDY.md PAPER_NOTES_v1.md ISSUE_DRAFT_lfmc.md
        ├── code/                  # 실험 스크립트
        └── bin/nx                 # 서버 조작 래퍼 (h100-setup의 nexus 호출)
```

- `_work/`는 바깥 클론의 `.git/info/exclude`에 등록돼 **원본 레포 git에 보이지 않습니다** →
  PR 작업(브랜치 `fix/sample-annotation-oe-schema` 등)이 우리 파일로 더러워지지 않습니다.
- `_work/`는 **자체 git 저장소**라 우리 작업만 따로 버전 관리됩니다.
- 서버 조작은 `./bin/nx`로 합니다. `h100-setup` 위치가 다르면 `H100_SETUP_DIR` 환경변수로 지정.

| 파일 | 역할 |
|---|---|
| **README.md** (이 파일) | 전체 요약 + 실험 대장 + 새 컴퓨터 재현 절차 |
| `GOAL.md` | 살아있는 계획서(SSOT) — 미션, 로드맵, Worklog, PR 후보, 확장 백로그 |
| `STUDY.md` | 공부 시스템 — 개념 카드 15장 + 확인 질문(= 면접 예상 질문) |
| `PAPER_NOTES_v1.md` | OlmoEarth v1 논문(arXiv 2511.13655) 상세 노트 |
| `ISSUE_DRAFT_lfmc.md` | Ai2에 보낼 이슈 초안 (제출 대기, 검토 필요) |
| `code/` | 모든 실험 스크립트 (아래 표) |
| `bootstrap.sh` | H200 세션 환경 복원 (멱등) |
| `watch.sh` | GPU/작업/디스크 모니터 |

---

## 1. 한 문단 요약

Ai2의 지구관측 파운데이션 모델 **OlmoEarth**를 실제로 돌려보며 배우고 개선하는 프로젝트.
2주간 ① 환경 구축 ② 산불연료·산림손실 모델 재현 ③ 공개 체크포인트의 재현성 결함 발견
④ 라벨 없는 위성 유사도 검색엔진(Korea Earth Search) 구축 ⑤ 제주 다개년 변화탐지 시도와
5차례의 체계적 실패·진단까지 진행했다. 실행 환경은 kt cloud AI Nexus의 **H200 ×2**.

핵심 성과 두 가지:
- **재현성 감사**: Ai2가 공개한 LFMC 체크포인트가 문서 성능의 60% 수준(MSE 951.9 vs 문서
  580.6)이고, 같은 공개 데이터로 우리가 재학습한 모델이 **558.8**로 문서를 웃돎 →
  "공개된 체크포인트 파일 자체가 잘못됐다"는 결론을 통제 실험으로 확정.
- **검색 방법론**: 파운데이션 임베딩의 이방성을 mean-centering으로 교정하면 라벨 0개로
  최대 ×26 리프트의 유사지 검색이 가능함을 정량 검증. few-shot 프로토타입은 지역·유형을
  건너뛰어 전이됨(완도 해상 김양식 → 제주 육상 수조, 9/9가 상위 4% 내).

---

## 2. 실험 대장 (전부 실측)

### A. 모델 재현 / 재현성 감사

| # | 실험 | 결과 | 산출물 |
|---|---|---|---|
| A1 | forest_loss_driver 추론 엔드투엔드 (페루 100건 → 10클래스) | ✅ 완주. agriculture 74 / mining 16 / logging 3 / road 3 / hurricane 2 / burned 1 / river 1. GPU 추론 2.3초, 전체는 데이터 다운로드가 지배 | `result.geojson`, 대시보드 아티팩트 |
| A2 | 5차례 실행 실패의 원인 규명 | 내부 API 의존, fail-fast 부재, rslearn `ingest:false` 미구현, 미출시 API 사용, SimpleTimeSeries 채널 시맨틱 불일치 | GOAL.md PR 후보 9건 |
| A3 | lfmc 파인튜닝 재현 (44,022 윈도우) | val 832→652(최저), ep36에서 중단. 동결 구간 **IO-bound(GPU 0%)**, 해동 후 GPU-bound | `trainer_checkpoints/epoch=33` |
| A4 | 공개 ckpt vs 우리 ckpt 통제 실험 | 공개 val 995.3 / master 환경 995.4(버전 효과 기각) / test 951.9 / **우리 ep33 test 558.8** | `ISSUE_DRAFT_lfmc.md` |

### B. Korea Earth Search (임베딩 검색)

| # | 실험 | 결과 |
|---|---|---|
| B1 | 임베딩 스토어 구축 | 완도 12 + 제주 54 + 지리산 16 + 제주 다개년 162 = **244 윈도우**, 768차원 × 40m 격자 |
| B2 | 이방성 교정 (v1 vs v2) | v1 실패(모든 cosine ~0.7) → mean-centering 후 판별력 확보 |
| B3 | WorldCover 정량 평가 | 제주: crop ×14.8, built ×18.3, grass ×14.1, tree ×3.6 / 완도: built ×26.0 |
| B4 | 제주 쿼리 3종 | 한라산 아고산 쿼리가 정상부를 정확히 구획(구상나무 프로브 절반 통과) / 오름·바다 쿼리는 특이도 낮음 |
| B5 | 양식장 프로토타입 (few-shot) | held-out 20곳 중앙값 백분위 **100.0** / 제주 교차지역 9/9가 **96.0~99.8** |

### C. 제주 다개년 변화탐지 — 실패 계보 (각 실패가 다음 설계의 근거)

| 버전 | 방법 | 결과 | 교훈 |
|---|---|---|---|
| v1 | 연도 간 cosine 거리 | ❌ Top-20 전부 바다 | 파도·반사로 바다 지문이 매년 변함 (std 0.094 vs 육지 0.042~0.060) |
| v2 | 계단형 검출 + 토지피복 층화 z | ⚠️ 육지화 성공, 26/30이 한 시점 집중 → **육안 검증 5/5 구름** | 층화는 요동만 잡고 오염은 못 잡음 |
| v3 | 구름비율 **평균** ≤0.20 마스킹 | ⚠️ 분산됨, **진짜 변화 1건 발견**(33.5087N 126.5747E 벌채·개발), 3/5 구름 잔존 | 평균은 1/12 오염을 희석 |
| v4 | **최악 모자이크** ≤0.35 | ❌ 생존 1.2%, 전부 바다 | 제주 최악-모자이크 평균 0.53~0.84 → **사후 마스킹 원리적 불가** |
| v5 | 입력 수정(`PER_PERIOD_MOSAIC`) + PPI 검증 | 🔄 2026-08-21 착수 | 합성 레시피는 manifest 필수 항목 |

---

## 3. 코드 지도 (`code/`)

실행 순서대로. 서버(H200) 실행은 전부 `env -u PYTHONPATH`로 감싼다
(다른 프로젝트가 심어둔 `PYTHONPATH`가 우리 venv를 오염시킨다).

| 파일 | 어디서 | 무엇을 |
|---|---|---|
| `setup_embed_store.sh` | 서버 | 검색용 임베딩 스토어 구축 (윈도우 생성 → 다운로드 → 임베딩 추출) |
| `setup_jeju_v2.sh` | 서버 | **변화탐지용** 제주 4개년 재수집 (구름 강건 합성) + 임베딩 |
| `model_s2.yaml` / `model.yaml` | 서버 | 임베딩 추출 설정 (S2-only / S1+S2). 디코더 = `EmbeddingHead` |
| `config.json` | 서버 | v1 스토어의 데이터셋 정의 (참고용, 스크립트가 자체 생성) |
| `fetch_osm_aquaculture.py` | 로컬 | OSM 양식장 좌표 수집 → JSON |
| `search_similarity.py` | 서버 | 이방성 교정 v1/v2 비교 + WorldCover precision@k |
| `farm_prototype.py` | 서버 | 양식장 프로토타입 검색 + 교차지역 전이 평가 |
| `change_4yr.py` | 서버 | 변화탐지 v1 (연도 간 거리 — 실패 기록용) |
| `change_v2_step.py` | 서버 | 변화탐지 v2 (계단형 + 층화) |
| `cloud_mask_v3.py` | 서버 | 구름 마스크(평균 기준) + 재순위 |
| `cloud_mask_v4.py` | 서버 | 구름 마스크(최악 모자이크 기준) + 재순위, `cloud_stats.npz` 생성 |
| `verify_candidates.py` | 서버 | 후보 지점의 연도별 RGB 칩 그리드 (육안 검증) |

---

## 4. 새 컴퓨터에서 그대로 돌리기

### 4.1 로컬 준비 (맥/리눅스)

두 저장소를 모두 가져온다.

```bash
cd ~/dong/ai_projects
# 1-a) 접속 도구
git clone <h100-setup 저장소> h100-setup
# 1-b) Ai2 원본 레포 + 우리 작업공간
git clone https://github.com/allenai/olmoearth_projects
git clone git@github.com-dong:DDanggle/eo_olmo_earth_project.git olmoearth_projects/_work
printf '_work/\n.DS_Store\n' >> olmoearth_projects/.git/info/exclude

cd h100-setup

# 2) 자격증명 — .env는 저장소에 없다 (gitignore). 기존 컴퓨터에서 복사하거나 새로 작성
cp .env.example .env && $EDITOR .env    # NIPA 포털 ID/PW 입력

# 3) nexus CLI 의존성 (Python 3.13 필요 — backend.ai 클라이언트 26.4.9 고정 때문)
./nexus doctor        # 필요한 것을 자동 설치하고 어디서 막혔는지 알려준다
```

`.env`에 넣을 것: `NIPA_KTCLOUD_ID`, `NIPA_KTCLOUD_PASSWORD`. 나머지 기본값은
`.env.example` 그대로 두면 된다 (그룹 NIPA-H200, GPU 2, 이미지 등).

### 4.2 서버 세션 + 환경 복원

```bash
cd ~/dong/ai_projects/olmoearth_projects/_work
./bin/nx up                # 세션 시작 (있으면 재사용)
./bin/nx tunnel up         # SSH 터널 (컨테이너가 바뀌면 반드시 재실행)
./bin/nx push $PWD/bootstrap.sh olmoearth/
./bin/nx run "bash /home/work/data/olmoearth/bootstrap.sh"
```

`bootstrap.sh`는 멱등이며, 영구 저장소(`/home/work/data/olmoearth`)에
`uv.lock` 고정 버전의 Python 3.11 venv를 만든다. 세션이 회수돼도 이 한 줄이면 복원된다.

**추가 venv**: 임베딩 실험은 rslearn git master가 필요하다(`EmbeddingTask`가 PyPI에 없음).
```bash
./bin/nx ssh
export UV_CACHE_DIR=/home/work/data/.cache/uv PATH=/home/work/data/.local/bin:$PATH
uv venv --python 3.11 /home/work/data/olmoearth/.venv-master
VIRTUAL_ENV=/home/work/data/olmoearth/.venv-master \
  uv pip install "rslearn[extra] @ git+https://github.com/allenai/rslearn" wandb
```

### 4.3 실험 재현

```bash
# 코드 업로드
./bin/nx push $PWD/code olmoearth/

# (A) LFMC 재현성 감사 — 데이터 50GB 다운로드 + NFS 압축해제 ~4시간 주의
#     docs/lfmc.md의 공개 dataset.tar와 HF 체크포인트를 받아 model test 실행
#     상세 절차와 통제 실험 구성은 ISSUE_DRAFT_lfmc.md 참고

# (B) 임베딩 검색 스토어
./bin/nx sh "bash /home/work/data/olmoearth/code/setup_embed_store.sh"
python code/fetch_osm_aquaculture.py wando 34.15 126.55 34.50 126.95
./bin/nx push $PWD/osm_aqua_wando.json olmoearth/embed_search/
./bin/nx sh "/home/work/data/olmoearth/.venv-master/bin/python \
  /home/work/data/olmoearth/code/search_similarity.py wando"

# (C) 변화탐지 v5 (권장 경로)
./bin/nx sh "bash /home/work/data/olmoearth/code/setup_jeju_v2.sh"
```

**긴 작업은 반드시 백그라운드로.** SSH가 끊기면 전경 작업은 같이 죽는다:
```bash
./bin/nx ssh     # 또는:  ./bin/nx sh '명령'  (PYTHONPATH 자동 차단)
setsid nohup env -u PYTHONPATH <명령> > /home/work/data/.jobs/이름.log 2>&1 &
echo $! > /home/work/data/.jobs/이름.pid
```

### 4.4 이 환경에서 밟으면 아픈 것들 (실측)

1. **`PYTHONPATH` 오염** — 다른 프로젝트가 `.bashrc`에 심어둔 경로가 우리 venv의 lightning을
   덮어써서 설정 파싱이 깨진다. 모든 실행을 `env -u PYTHONPATH`로 감쌀 것.
2. **임베딩 추출 OOM** — rslearn 가이드 기본값(workers 16 + `load_all_crops`)은 1024px
   윈도우에서 **트레이스백 없이** 죽는다. `--data.init_args.num_workers=6 --batch_size=4`.
3. **NFS 압축해제** — 50GB tar가 다운로드 24분 vs 해제 4시간 (파일 100만 개). 파일 수를 먼저 확인.
4. **Sentinel-1 지역 공백** — Planetary Computer에 제주 2024년 S1이 0장(완도는 정상).
   모달리티가 지역마다 다르면 임베딩 공간이 갈라진다 → S2-only로 통일.
5. **세션 회수** — 6시간 평균 GPU 사용률 1% 미만이면 회수. 산출물은 `/home/work/data`에만.
6. **버전 스큐** — 레포 main의 설정이 미출시 rslearn API를 쓴다. `uv sync --frozen` 준수하고,
   필요한 설정 패치는 GOAL.md PR 후보 참고.

---

## 5. 공부 재개 지점

1. `GOAL.md` Worklog 통독 (무엇을 왜 했는지의 서사)
2. **`STUDY.md`의 확인 질문에 직접 답 써보기** ← 진짜 재개 지점. 특히
   #3(버전 스큐 재발 방지), #5(재사용 vs 재계산), #13(요동 vs 오염),
   #15(PPI로 valid한 결론) — 이 넷이 프로젝트의 뼈대이자 면접 질문이다.
3. `PAPER_NOTES_v1.md`는 필요할 때 참조 (암기 불필요)
4. 다음 커리큘럼: v1.2 릴리스 변경점, FoldRefresh 재정식화

## 6. 대기 중인 사람의 결정

- `ISSUE_DRAFT_lfmc.md` 검토 후 Ai2에 제출 (웹 붙여넣기 또는 `gh` 설치 후)
- `olmoearth_projects` 레포의 `fix/sample-annotation-oe-schema` 브랜치 PR 제출
- MARC/테크포임팩트에 제주 변화 데모를 보여줄지 (v5 검증 통과 후)
