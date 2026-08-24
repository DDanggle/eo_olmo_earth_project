# OlmoEarth 프로젝트 — 전체 정리 및 인수인계

최종 갱신: 2026-08-24

이 파일이 **입구**입니다. 처음 보는 사람(또는 새 컴퓨터의 나)은 여기부터 읽습니다.

## 이 폴더의 위치와 두 저장소의 관계

```
~/DongDong/ai_projects/
├── h100-setup/                    # [접속 전용] kt cloud AI Nexus 세션·터널·전송 (nexus CLI)
└── olmoearth_projects/            # Ai2 원본 레포 클론 (PR 작업용)
    └── _work/                     # ★ 여기 — 우리 연구 작업공간 (자체 git 저장소)
        ├── README.md GOAL.md STUDY.md RESEARCH_STRATEGY.md
        ├── K_ALIGN_PROGRAM_NOTE.md KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md
        ├── EARTHROUTE_PROGRAM_NOTE.md PAPER_READING_LIST.md PAPER_NOTES_v1.md
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
| `RESEARCH_STRATEGY.md` | 박사 연구 프로그램 — WorldShift × ModelShift, 가설·베이스라인·12주 실행 |
| `RESEARCH_EXECUTION_PLAN.md` | 1·2·6주 실행계획 — 표본 3종, 모델 subtrack, 지표, 표/figure, promotion/kill gate, GPU0 queue |
| `K_CONTEXT_FUSION_EXPERIMENT.md` | 메인 논문 실험 계약 — 동적 공공 context, EO-only privileged distillation, simple/native baseline, 3지역 split·kill gate |
| **`MEASURED_FINDINGS.md`** | **측정 장부 — 실행해서 나온 수치만.** M1 릴리스 identity 실패 / M2 Major TOM 계약 감사 / M3 dose–response / M4 취약성 반증 / M5 복제 실패 |
| `MOUNTAIN_EVIDENCE_TRANSFER.md` | MountainShift — 알프스·HKH·한국 산악 전이 설계, TRANSFER/LOCAL-ADAPT/RE-EMBED/ABSTAIN, 중단 기준 |
| `K_ALIGN_BIG_PICTURE.md` | **EarthKV 프로그램 spine** — EarthEmbedContract·FoldRefresh·EarthRoute·MountainShift의 층위와 REUSE/ADAPT/RECOMPUTE 판정 |
| `K_ALIGN_CVPR_READINESS_AUDIT.md` | CVPR 준비도 감사 — 필요성·예측가능성·fixed quantizer, 14일 P0 |
| `K_ALIGN_WIDE_ANGLE.md` | **광각 보정** — 계약 불일치 계측기(dose–response), 진단 눈멂 행렬, Major TOM 실험대, ADC killer baseline, embeddings-STAC gap |
| `K_GAIN_AXES.md` | **네 축 감사** — 정확도·임베딩·속도·위성유도 각각의 기전, 이미 점유된 선행연구, 남은 빈칸, 계약 수정 5건 |
| `K_ALIGN_PROGRAM_NOTE.md` | **K-ALIGN 프로그램 노트** — 은행에 든 자산, 승률을 올리는 8보정(R1–R8), 제출 사다리, 프로그램 중단 조건, 세 축 닫기 |
| `KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md` | **K-ALIGN 중심 계약(authoritative)** — stable cache, multi-teacher compatible distillation, timestamped public residual, 4 estimand·gate, derivability screen, black-box baseline, A0–A7/B1–B2 실행순서 |
| `EMBEDDING_TRANSFER_CVPR_TRACKS.md` | multi-teacher EO 호환 전이, edge/cloud 효율, cross-view robotics·world-model·simulation 후보와 kill gate |
| `K_EARTH_PROGRAM_STATUS.md` | 5축 현재판 — 공공데이터 신청, 보유/시계열 상태, 사업, 한국 연구, EarthRoute 확장 |
| `config/kearth_public_access.json` | secret 없는 접근상태 SSOT — 사용자 확인·키 존재·실제 API probe를 분리 |
| `EARTHROUTE_PROGRAM_NOTE.md` | 8월 EarthRoute 노트의 canonical 확장판 — K-Earth·FoldRefresh·전이·사업·중단 기준 |
| `PAPER_READING_LIST.md` | 최신 논문 검색·독서 장부 — 선점 영역, 반증 문헌, 읽기 순서, 실험 변경점 |
| `KOREA_PUBLIC_DATA_CATALOG.md` | 공식 공공데이터 23개 연결 후보 — 접근성·PNU/공간/시간 join·누락 계약·우선 구현 |
| `PARTNER_BRIEF_MARC.md` | MARC형 생태 파트너 공동설계 초안 — 결정 질문·파일럿·금지 주장 |
| `STUDY.md` | 공부 시스템 — 실제 실험에서 만난 개념 카드 + 확인 질문(= 면접 예상 질문) |
| `PAPER_NOTES_v1.md` | OlmoEarth v1 논문(arXiv 2511.13655) 상세 노트 |
| `ISSUE_DRAFT_lfmc.md` | Ai2에 보낼 이슈 초안 (제출 대기, 검토 필요) |
| `code/` | 모든 실험 스크립트 (아래 표) |
| `bootstrap.sh` | H200 세션 환경 복원 (멱등) |
| `watch.sh` | GPU/작업/디스크 모니터 |

---

## 1. 한 문단 요약

Ai2의 지구관측 파운데이션 모델 **OlmoEarth**를 실제로 돌려보며 배우고 개선하는 프로젝트.
2주간 ① 환경 구축 ② 산불연료·산림손실 모델 재현 ③ 공개 체크포인트의 재현성 결함 발견
④ 라벨 없는 위성 유사도 검색엔진(Korea Earth Search) 구축 ⑤ 제주 다개년 변화탐지의
v1~v6 실패·민감도 진단 ⑥ SCL 장면선택 v7 golden-window 검증까지 진행했다. 실행 환경은
kt cloud AI Nexus의 **H200 ×2**.

핵심 성과 두 가지:
- **재현성 감사**: Ai2가 공개한 LFMC 체크포인트가 문서 성능의 60% 수준(MSE 951.9 vs 문서
  580.6)이고, 같은 공개 데이터로 우리가 재학습한 모델이 **558.8**로 문서를 웃돎 →
  "공개된 체크포인트 파일 자체가 잘못됐다"는 결론을 통제 실험으로 확정.
- **검색 방법론**: 파운데이션 임베딩의 이방성을 mean-centering으로 교정하면 라벨 0개로
  최대 ×26 리프트의 유사지 검색이 가능함을 정량 검증. few-shot 프로토타입은 지역·유형을
  건너뛰어 전이됨(완도 해상 김양식 → 제주 육상 수조, 9/9가 상위 4% 내).

### 2026-08-24 냉정한 현재판

이 프로젝트는 **유의미한 연구 프로그램이지만 아직 논문 결론도 제품도 아니다.** 강점은 계획의
크기가 아니라, 주장을 스스로 기각한 증거 사슬이다. v1/v1.2 exact-input 216건, Major TOM 248,719
paired audit, band-order dose response, Gaussian 반증 대조군, v1.2 복제 실패까지 `MEASURED_FINDINGS.md`
한 장으로 닫혀 있다. 로컬 전체 suite는 올바른 parent venv에서 **128 tests 통과, optional 1건 skip**이다.

장기 이름은 **EarthKV**로 묶되 첫 논문은 `EarthEmbedContract`에 고정한다. EarthKV의 paging·eviction·
admission·distributed cache는 아직 구현·측정하지 않았고 기여로 세지 않는다. 가장 큰 미완성은
`R@1=0`이 실제 frozen downstream head의 정확도·calibration·고확신 오답으로 이어지는지다.

최신 조사로 두 전제도 고쳤다. AlphaEarth와 TESSERA는 이미 model/data/build version을 명시하므로
novelty는 `version metadata` 자체가 아니라 **task-level compatibility validation과 risk–cost action**이다.
MountainShift는 기관별 원자료를 먼저 합치지 않고 2025–2026 공개 Sen12Landslides·AvalCD에서
region holdout을 먼저 반증한다. AI-Hub 국립공원 4해상도는 sensor별 class가 달라, 동일 장면·동일
polygon audit 전에는 resolution ladder라고 부르지 않는다.

현재 우선순위는 둘뿐이다: ① public task의 frozen-head silent-error 표 ② Ai2에 보일 수 있는
외부 증거(sample schema PR 또는 LFMC report) 한 건. 이 둘 전에는 새 VLM·federated·분산 EarthKV
트랙을 열지 않는다.

### 2026-08-23 연구 프로그램 보정과 첫 실행

이제 제주 검색/변화 데모 자체를 최종 목표로 삼지 않는다. 플래그십 질문은
**"세계, 입력 합성, 기반모델 릴리스가 함께 바뀔 때 지도 기반 의사결정을 어떻게 분해·보정·
선택적으로 갱신할 것인가"**다. Sherrie Wang 계보의 ML 지도 오차·Prediction-Powered
Inference, Ai2의 파트너 중심 공개 인프라, MARC형 장기 생태 현장조사를 하나의 실험 체계로
연결한다. 현재 플래그십 설계는 `RESEARCH_STRATEGY.md`, 그 이후 관측·계산·근거 수집을 함께
선택하는 후속 프로그램과 사업 gate는 `EARTHROUTE_PROGRAM_NOTE.md`에 있다. 한국 지역·연도·
센서별 전이 실패를 scratch·다른 GeoFM과 비교하고 다음 라벨을 능동적으로 고르는 benchmark
schema와 CVPR/후속 논문 경계는 `K_EVIDENCE_SHIFT_BENCHMARK.md`에 고정했다.

첫 실행 가능한 `K-EvidenceShift Jeju pilot v0`도 추가했다. 14 candidate record를 13개 독립
공간사건으로 묶고, 공유 window·scene·PNU, t1 이후 RGB, 사후 수집 API, assistant pre-annotation을
누수 검사한다. 결과는 공식 사건 보강 근거 0/14, 원인 라벨 0, 보류 14/14이며 성능 benchmark가
아니다. 동시에 OlmoEarth v1/v1.2 P0용 제주 216 site-years·162 adjacent-year events, label-free
smoke 8개, exact input/checkpoint SHA를 고정했다. GPU0에서 v1/v1.2 16개 출력을 완주했고,
GPU1의 다른 프로젝트는 건드리지 않았다. label-free 8표본에서 pooled 거리 순위상관은 0.889,
top-1/2 이웃 보존은 0.75/1.00이었지만 창 내부 spatial CKA는 평균 0.427(0.133–0.828)로
불균일했다. 이는 정확도나 cache 호환성 증거가 아니라 릴리스 표현 이동의 기술통계다.
서버 원본 입력·체크포인트·출력 228파일(7,851,565,383 bytes)을 다시 해시한 증거 폐쇄 검사도
758/758 통과했고, 분석 JSON/CSV의 독립 재실행 결과가 기존 파일과 byte-identical했다.
고정 hash·preflight·분리 실행·완료 판정은 `OLMO_RELEASE_AUDIT_P0.md`에 재현 명령으로 남겼다.

이후 GPU0 full audit를 54위치×4년=216 site-years 전체로 확장했다. v1/v1.2 432개 출력
(105,591,415,295 bytes)을 모두 hash·grid·mask·finite/nonzero로 봉인했고, 보정 30위치와 이전에
보지 않은 sealed 16위치를 분리했다. same-release R@1은 1.0이지만 raw cross-version R@1은
양방향 0.0, 최선 affine ridge도 v1.2→v1 0.6973·v1→v1.2 0.6089로 사전 0.95 gate를 실패했다.
sealed pooled CKA 0.9786·거리 Spearman 0.9525처럼 216개 패널 내부 관계 구조는 남지만 token
cache identity는 깨진다. 따라서 이 결과는 **단순 v1→v1.2 cache 재사용을 승인하지 못한
representation-proxy 증거**이지 정확도·구름 강건성·한국 일반화의 증거가 아니다. 전체 계약과 금지 주장은
`artifacts/release_audit_full216_v1/README.md`에 있다.

full audit 이후 메인 논문 질문도 보정했다. 단순 `EO+지도/날씨` 결합은 GeoLink·MMEarth·Galileo가
이미 강하게 점유하므로, 새 주장은 **event/observed/published/retrieved time과 자연 coverage·누락·
충돌을 보존한 한국 공공 context가 inference-time fusion뿐 아니라 EO-only student 표현까지
강화하는가**로 좁혔다. 영상-only, location/year-only, context-only, hard-coded STACK/TOKEN-FUSE,
GeoLink-style fusion, native multimodal model을 같은 split에서 비교하고, 미래정보 누출·shuffle·
time-shift control을 통과해야 한다. authoritative 계약은 `K_CONTEXT_FUSION_EXPERIMENT.md`에 있다.

full audit과 시간축 재감사 뒤 K-ALIGN을 더 단순하게 좁혔다. **Earth embedding은 모델 릴리스·
시간창·밴드·GSD·pooling 계약 안에서만 의미가 있고, 계약이 다른 cache를 조용히 재사용하면
고확신 오류가 생긴다.** 실제로 같은 입력의 v1/v1.2는 cross-release R@1=0이었고, 같은 모델에서도
2025/rolling-2026 창이 184일 겹치며 4기간/12기간 Top-30 Jaccard가 0.091이었다. 14후보 중 9건은
두 시간계약 결함 중 하나에 노출돼 annual-change claim 자격이 없다. 새 프로그램은 비교 전에
계약을 검사해 REUSE/ADAPT/RECOMPUTE·ABSTAIN을 고르는 `EarthEmbedContract`다. quantizer와
비용곡선은 필요할 때의 해결수단이지 중심 문장이 아니다.

한국 public-context는 중요한 후속축이지만 현재 0 time-aligned support와 `published_time` 부재 때문에
main에 강제로 합치지 않는다. BuildingHUB/EIA event-first 표본과 독립 label이 확보될 때만
`R_source`(EO 회복성), `V_source`(독립 task 가치), `T_source`(EO-only student 전이)를 검증한다.
로봇·simulation도 paired trajectory가 생긴 뒤 별도 논문으로 연다. 거시적 판정과 P0는
`K_ALIGN_CVPR_READINESS_AUDIT.md`, public-context authoritative 계약은
`KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`에 있다.

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
| B2 | 검색 파이프라인 v1 vs v2(Olmo 모델 릴리스명이 아님) | v1 실패(모든 cosine ~0.7) → mean-centering 후 판별력 확보 |
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
| v5 | `PER_PERIOD_MOSAIC` 재수집 시도 | ❌ **의미적 중복**: 계산은 216/216 완료했지만 v1↔v5 cloud/zero 지표와 blind RGB 5/5가 동일. 2,592 source-group hash와 고정 픽셀/임베딩 표본도 동일 | 설정 문자열이 아니라 실행 handler·item hash·픽셀을 먼저 비교해야 함 |
| v6 | 4기간↔12기간 입력 민감도 | ⚠️ Top-30 교집합 5, Jaccard 0.091. 역시간순·계절 불일치가 함께 발견됨 | 시간축 manifest와 계절 통제 전에는 변화 정답으로 해석 금지 |
| **v7 smoke** | 최대 3 coverage 중 SCL clear-cover가 가장 높은 장면 선택 | ✅ 한 golden window에서 첫 4기간 bad proxy **95.64% 감소**, 고정 target 1.00→0.00, RGB 확인 | 실제 입력 개입이 품질을 바꾼 첫 증거. 다중-window 일반화는 아직 미검증 |
| **v7.1 human audit** | v3/v6 순위로 사전 고정한 14후보의 5월 정렬 RGB | ✅ 고확신 지속 변화 5 records/**4 unique sites**, 추가 의심 3, 구름·계절성·불명확 6. 동일 개발지 중복 1쌍 | 동부 중산간 집단도 개발·경작·구름이 섞임. 특정 오름/시설 원인은 외부 근거 전까지 금지 |
| **v7.2 Korean context** | 4개 고유 고확신 site × 공식 오름 368건 × 국토부 제주 허가 240건 × offline OSM | ✅ 개발/인프라 해석 강화 2, 문맥만 추가 2, RGB 변화 판정 번복 0 | r11은 공식 고이악 416m·태양광 plant 6개 419–951m. 허가 CSV는 제주 2023/24가 없어 불일치를 음성 증거로 사용 금지 |
| **v7.3 K-Earth 368** | 공식 오름 368 고정 분모 × offline OSM 위치 × 기존 4/12기간 점 screen × RGB 보류 | ⚠️ 목록 368/368, 위치·screen 243/368. 모델 high-stable 8개였으나 RGB 지속 변화 확인 0, 구름/해무 기각 8, 불확실 1 | 모델 두 입력의 합의가 동일 오염을 안정적으로 재현할 수 있음. A/B급 원인 근거 0%라 선택적 변화탐지 모드 유지 |
| **v7.4 public-data contract** | 공식 데이터 23개 연결 후보의 접근·geometry·time·coverage 계약 조사 | ✅ PNU 기하 spine, 행정사건, 독립 상태관측, 규제/교란 층을 분리하고 P0 3패키지 정의 | 대표 지번≠오름 경계. 출처 coverage가 입증되지 않은 no-match는 U이며 evidence-source ablation을 다음 논문 실험으로 고정 |
| **v7.5 official ingestion** | 2025 제주 FarmMap 289,379 polygon + 개발행위 240행 + 산지이용 19년을 결정적 evidence edge로 정규화 | ✅ 4 변화좌표 중 `r08` 1건 B급 변화 전 농지 상태, OSM 오름점 243개 중 7건 C급 농경지 상태, FarmMap↔허가 exact PNU 50개 | `r08` 항공 관측은 변화 전 503일이라 원인이 아닌 baseline. A/B급 **원인** 근거는 여전히 0/368이며 367 보류·1 조사 유지 |
| **v7.6 live API snapshot** | BuildingHUB·EIA WFS·GK2A·환경부 토지피복·VWorld를 secret-safe request manifest와 bounded pagination으로 실제 수집 | ✅ HTTP 207, 의미상 성공 200. 건축 8,794행·EIA 13 polygon·토지피복 42장·최신 구름 grid 127,040값. 기존 PNU 58개 중 건축 exact 9 | 14후보 EIA 직접중첩 0, `r08` 같은 법정동 77건이나 exact PNU 0. VWorld는 로컬/VM 모두 key 거부, 과거 GK2A는 최근 2일 제한 → **14/14 보류** |
| **v7.7 VWorld cadastral snapshot** | 재승인된 key를 대표점 `status=OK`로 gate한 뒤 후보 14점+오름 243점 조회, v7.6 raw와 offline 재결합 | ✅ VWorld 256/257 필지·고유 PNU 235, 후보 14/14·오름 242/243. 전체 HTTP 463/463, semantic 성공 456·유효 무항목 1·과거 GK2A 오류 6 | `r04` exact PNU 건축사건 1건은 마지막 관측 뒤 2026-07 사건이라 시간정렬 0. `r08` FarmMap↔VWorld PNU 충돌 1건을 보존. A/B급 원인 **0/14·0/368**, 전부 보류 유지 |
| **v7.8 K-EvidenceShift pilot/P0** | 14후보 audit schema·누수 gate + 216 site-year release manifest + v1/v1.2 immutable checkpoint/input SHA | ✅ GPU0 smoke 16/16. 증거 폐쇄 758/758, raw 228파일·7.85GB 재해시, 분석 JSON/CSV byte-identical 재실행. pooled distance Spearman 0.889, top-1/2 overlap 0.75/1.00, spatial CKA 평균 0.427 | 독립 GT 0, cause 0, scene-disjoint quality split 불가. 8 label-free·7 spatial cluster이므로 정확도·한국 일반화·cache 호환성 주장은 금지 |
| **v7.9 full release audit** | 216 site-year×v1/v1.2 exact-input paired 실행 + spatial calibration/embargo/sealed split + 4개 bridge | ❌ representation proxy gate 실패. native R@1 1.0, no-bridge 0.0, 최선 ridge 0.697/0.609로 0.95 미달. sealed CKA 0.979·거리 Spearman 0.953 | 패널 관계구조 보존≠token identity 호환. 라벨 0이므로 task 정확도·구름·공공데이터·한국 일반화 주장은 금지; 새 bridge는 새 untouched split 필요 |
| **v7.10 time-contract audit** | window 날짜·실제 월 순서·candidate lineage 재감사 | ❌ 2025/rolling-2026 184일 중첩, first-4 계절정렬 false, all-12 월집합도 false. 4/12 Top-30 Jaccard 0.091; 14후보 중 중첩 5·4기간 5·합집합 9 contract-exposed | 9건 모두 false라는 뜻은 아니나 annual-change claim 자격 없음. 옛 후보 스크립트 default 실행을 차단하고 valid time manifest 뒤 재추론 필요 |

---

## 3. 코드 지도 (`code/`)

실행 순서대로. 서버(H200) 실행은 전부 `env -u PYTHONPATH`로 감싼다
(다른 프로젝트가 심어둔 `PYTHONPATH`가 우리 venv를 오염시킨다).

| 파일 | 어디서 | 무엇을 |
|---|---|---|
| `setup_embed_store.sh` | 서버 | 검색용 임베딩 스토어 구축 (윈도우 생성 → 다운로드 → 임베딩 추출) |
| `setup_jeju_v2.sh` | 서버 | v5 의미적 중복·시간창 중첩 실패 재현용. 명시적 override 없이는 실행 거부 |
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
| `change_v5.py` | 서버 | PER_PERIOD 입력 변화탐지와 v1 합성 레시피 통제(회수된 실행 코드) |
| `model_s2_t12.yaml` / `extract_jeju_t12.sh` / `change_v6_t12.py` | 서버 | 역사적 4↔12기간 민감도 재현용. 현재 시간계약 실패로 명시적 override 없이는 실행 거부 |
| `audit_jeju_v5_quality.py` | 서버 | v1↔v5 cloud/zero proxy 전수 감사 + blind RGB 쌍 |
| `audit_jeju_time_axis.py` | 서버 | 실제 source-item 날짜·순서 hash와 계절 정렬 감사 |
| `scl_compositor.py` / `setup_jeju_v7_smoke.sh` | 서버 | SCL은 nearest로 점수화하고 반사도는 bilinear로 유지하는 1-window 입력 개입 |
| `evaluate_jeju_v7_smoke.py` | 서버 | 사전 고정 target·전체-window bad/zero proxy·RGB의 v1↔v7 판정 |
| `build_jeju_human_review.py` | 서버 | **역사적 실패 감사 전용**. 시간계약 결함이 있는 v3/v6 후보 RGB 재현; 명시적 override 없이는 실행 거부 |
| `prepare_korea_public_data.py` | 로컬 | 제주 공식 오름 CSV 정규화 + 국토부 전국 개발행위허가에서 제주 행 추출·품질 요약 |
| `enrich_jeju_offline_osm.py` | 로컬 | 대한민국 전체 OSM을 받은 뒤 후보 좌표는 로컬에서만 공간 결합, 공공데이터 evidence dashboard 생성 |
| `build_kearth_oreum_registry.py` | 로컬 | 공식 368개 오름 고정 분모 + 첨부 제주시 210건 복합키 대조 + offline OSM 보수적 위치화 + 근거등급/보류 대시보드 |
| `ingest_kearth_public_data.py` | 로컬 | 공식 FarmMap SHP·개발행위·산지이용을 manifest/evidence edge로 정규화하고 point-in-polygon·exact PNU·시간축을 감사 |
| `kearth_public/` | 로컬 | PNU·날짜구간·coverage·결정적 JSON/SHA·안전한 ZIP·FarmMap offline join의 재사용 가능한 ingestion core |
| `collect_kearth_api_snapshot.py` / `kearth_public/api_snapshot.py` | 로컬/VM | 공공 API credential을 manifest에서 제외하고 raw 응답을 redaction·hash하며 제주 bounded pagination 수집 |
| `derive_kearth_api_snapshot.py` / `render_kearth_api_dashboard.py` | 로컬 | API raw를 기존 FarmMap PNU·OlmoEarth 후보와 재결합하고 GK2A/토지피복 context 및 14후보 보류 dashboard 생성 |
| `merge_kearth_api_snapshots.py` | 로컬 | v7.6 비-VWorld raw와 gate된 VWorld 257점 snapshot을 네트워크 없이 결합하고, 입력 SHA·응답 lineage·FarmMap/VWorld PNU 충돌·완주 marker를 보존 |
| `build_k_evidence_shift_pilot.py` / `kearth_benchmark/` | 로컬 | 제주 14후보를 audit-only site-event로 만들고 공간/window/scene/PNU/시간/API 누수와 promotion gate를 검사 |
| `build_olmo_release_audit_manifest.py` / `hash_olmo_release_inputs.py` | 로컬→서버 | 제주 216 site-year manifest와 label-free smoke 8개를 선택하고 입력 tensor/metadata를 SHA-256으로 고정 |
| `resolve_olmo_release_checkpoints.py` / `run_olmo_release_smoke.py` | 서버 | v1/v1.2 HF commit·weight SHA를 고정하고, 선택한 GPU에 active process가 있으면 실행을 거부한 뒤 exact-input paired smoke를 순차 실행 |
| `analyze_olmo_release_smoke.py` | 서버 | 완료 marker·output SHA/mtime·sample/input/grid/mask 정합을 검사한 뒤 CKA(+spatial-shift null)·k=1/2 이웃 보존·거리 순위상관·spatial-cluster LOO를 계산 |
| `verify_olmo_release_bundle.py` | 로컬/서버 | 실행 preflight→checkpoint/exact-input→2×8 output→analysis marker를 결속하고 raw 228파일 SHA·GPU0 로그·identity·금지 주장을 fail-closed 검증 |
| `prepare_olmo_release_audit_view.py` | 서버 | 원본 dataset을 수정하지 않고 smoke 입력 layer만 symlink한 독립 rslearn output view 생성 |
| `run_olmo_release_batch_gate.py` / `analyze_olmo_release_batch_gate.py` / `finalize_olmo_batch_contract.py` | 서버 | batch·worker 후보를 exact output equivalence, 반복성, GPU/runtime/code provenance로 승격. full audit는 batch8/workers4 선택 |
| `freeze_olmo_release_spatial_split.py` / `run_olmo_release_full.py` | 로컬→서버 | 54위치의 calibration/embargo/sealed/disclosed split을 출력 열람 전에 동결하고, GPU0에서 v1/v1.2 각 216개를 fresh root에 fail-closed 실행 |
| `finalize_olmo_release_full.py` / `analyze_olmo_release_full.py` | 서버 | 5,616입력·432출력 SHA/grid/mask/value health를 결속하고 calibration-only bridge·sealed identity retrieval·금지 주장을 봉인 분석 |
| `olmo_release_semantic_contract.py` / `olmo_release_raster_contract.py` | 로컬/서버 | 모델·checkpoint·runtime·rslearn/code identity와 768×256×256 finite/nonzero GeoTIFF 계약의 공용 validator |
| `score_oreum_existing_embeddings.py` | 서버 | **역사적 실패 감사 전용**. 기존 4/12기간 시간계약이 무효라 명시적 override 없이는 새 오름 후보 생성 거부 |
| `audit_jeju_candidate_time_contract.py` | 로컬 | 창 중첩·계절 불일치와 14후보 lineage를 재집계해 annual-change claim 자격을 fail-closed 감사 |
| `render_oreum_screen_review.py` | 서버 | 선택된 오름 후보를 각 연도 5월 최근접·고정 RGB stretch·두 공간 scale로 렌더해 모델 합의를 육안 검증 |

---

## 4. 새 컴퓨터에서 그대로 돌리기

### 4.1 로컬 준비 (맥/리눅스)

두 저장소를 모두 가져온다.

```bash
cd ~/DongDong/ai_projects
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
cd ~/DongDong/ai_projects/olmoearth_projects/_work
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

# (C) 변화탐지 v5 (실패 재현용 — 품질 개선 경로로 사용 금지)
./bin/nx sh "ALLOW_HISTORICAL_INVALID_JEJU_TIME_WINDOWS=1 \
  bash /home/work/data/olmoearth/code/setup_jeju_v2.sh"

# (D) v7 SCL golden-window smoke (전체 제주 확장 전 게이트)
./bin/nx sh "bash /home/work/data/olmoearth/code/setup_jeju_v7_smoke.sh"
./bin/nx sh "/home/work/data/olmoearth/.venv-master/bin/python \
  /home/work/data/olmoearth/code/evaluate_jeju_v7_smoke.py"

# (E) K-Earth 공식데이터 ingestion (공식 원본 확보 뒤 실행은 네트워크 불필요)
python3 -m venv /tmp/kearth-public-venv
/tmp/kearth-public-venv/bin/python -m pip install -r requirements-public-data.txt
/tmp/kearth-public-venv/bin/python code/ingest_kearth_public_data.py \
  --forest-use-csv artifacts/external_data/kearth_public_ingest_v1/raw/jeju_forest_use_20260630.csv \
  --farmmap-zip artifacts/external_data/kearth_public_ingest_v1/raw/jeju_farmmap_20251231.zip \
  --development-permits-csv artifacts/external_data/korea_public_v1/jeju_development_permits_20260819.csv \
  --candidate-context-json artifacts/external_data/korea_public_v1/candidate_public_context.json \
  --candidate-manifest-json artifacts/human_review_v1/manifest.json \
  --oreum-registry-json artifacts/external_data/kearth_oreum_v1/oreum_evidence_registry.json \
  --out-dir artifacts/external_data/kearth_public_ingest_v1 \
  --retrieved-at 2026-08-22T07:22:00Z
/tmp/kearth-public-venv/bin/python -m unittest -v tests/test_kearth_public.py

# 기존 registry JSON에서 UI만 안전하게 재생성
python3 code/render_kearth_dashboard.py \
  --registry artifacts/external_data/kearth_oreum_v1/oreum_evidence_registry.json \
  --access-status config/kearth_public_access.json \
  --output artifacts/external_data/kearth_oreum_v1/dashboard.html
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
2. `RESEARCH_STRATEGY.md` → `K_EVIDENCE_SHIFT_BENCHMARK.md` →
   `K_ALIGN_PROGRAM_NOTE.md` → `KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md` →
   `EMBEDDING_TRANSFER_CVPR_TRACKS.md` → `EARTHROUTE_PROGRAM_NOTE.md` 순서로
   프로그램 결정·실험 계약·임베딩 전이·후속 프로그램 구분
3. `PAPER_READING_LIST.md`의 **먼저 읽을 26편**에서 당장 바꿀 실험이 있는 문헌만 정독
4. **`STUDY.md`의 확인 질문에 직접 답 써보기** ← 진짜 이해 점검. 특히
   #3(버전 스큐 재발 방지), #5(재사용 vs 재계산), #13(요동 vs 오염),
   #15(PPI로 valid한 결론) — 이 넷이 프로젝트의 뼈대이자 면접 질문이다.
5. `PAPER_NOTES_v1.md`는 OlmoEarth **v1 전용** 상세 수치가 필요할 때만 참조
6. 다음 실험: matched scratch/Olmo v1·v1.2/다른 GeoFM 전이표 → active-label offline replay →
   v1↔v1.2 paired audit·확률표본/PPI·FoldRefresh rslearn 이식

## 6. 대기 중인 사람의 결정

- `ISSUE_DRAFT_lfmc.md` 검토 후 Ai2에 제출 (웹 붙여넣기 또는 `gh` 설치 후)
- `olmoearth_projects` 레포의 `fix/sample-annotation-oe-schema` 브랜치 PR 제출
- MARC/테크포임팩트에 제주 변화 데모를 보여줄지 (v5 검증 통과 후)
