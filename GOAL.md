# OlmoEarth 연구 목표 — 살아있는 계획서

> 이 파일이 이 프로젝트의 단일 진실 공급원(SSOT)입니다.
> 에이전트든 사람이든, 작업 시작 전에 읽고 / 끝나면 Worklog와 상태를 갱신합니다.

## 미션

**"컴퓨트가 부족한 아시아 조직도 최신 OlmoEarth 릴리스를 쓸 수 있게 만드는 도구"**
(가칭 `olmoearth-migrate` — 아시아 파트너 온보딩 킷)

최종 산출물 세 개:

| # | 산출물 | 핵심 수치 | 증명하는 것 |
|---|---|---|---|
| A | 릴리스 벤치마크 | v1/v1.1/v1.2 × Nano~Large: 정확도 × GPU-초/km² × 비용 | 서빙 최적화 |
| B | 부분 갱신 실행기 | 25% 호출로 지역 집계 유지 (신뢰구간 포함) | FoldRefresh |
| C | 한국 산불 LFMC + 도시 열 헤드 | 실제 AOI GeoTIFF + 뷰어 + 변화 귀인 화면 | 도메인 + 태스크 이식성 |

원칙: **파악하면서 개선한다.** 매 루프 = 돌린다 → 마찰을 만난다 → 개선(PR/스크립트/문서)으로
남긴다 → 그것이 A/B/C의 부품이 된다. PR은 찾아다니지 않고 부딪힌 것만 올린다.

## 세 목표 운영체계 — 취업 × 박사 × 비즈니스 (2026-08-21)

로드맵의 실행 순서는 유지하되, 모든 큰 작업은 아래 세 축에서 무엇을 남기는지 먼저 적는다.
세 개를 별도 프로젝트로 벌리지 않고 하나의 질문으로 묶는다.

> **모델 릴리스와 현실 세계가 동시에 바뀔 때, 컴퓨트와 전문 인력이 부족한 조직이
> 의사결정의 연속성을 어떻게 지킬 수 있는가?**

| 목표 | 반드시 남길 외부 증거 | 이 프로젝트에서의 증명 방식 |
|---|---|---|
| Ai2/OlmoEarth 취업 | 재현 가능한 이슈·PR, PyTorch/rslearn 디버깅, 파트너 문제를 배포까지 가져간 기록, 명료한 기술 글 | LFMC 아티팩트 감사, 공개 파이프라인 수정, v1→v1.2 마이그레이션 하네스, 한국 파트너 케이스 |
| 박사 | 일반화 가능한 연구 질문, 강한 베이스라인·통제 실험·불확실성, 공개 데이터/코드, 논문 표 | **세계 변화 vs 모델 변화 분해**, 릴리스 간 이웃·순위·집계 안정성, 선택적 재계산(FoldRefresh) |
| 비즈니스 | 실제 의사결정자, 반복되는 고통, 유료/공동 파일럿, 갱신 주기와 실패 비용 | 아시아 데이터 가용성 감사, 릴리스 전환 감사, 지역 정답지 오버레이, 변화 후보 검수 리포트 |

운영 규칙:

1. 플래그십 작업은 세 축을 모두, 기반 작업은 최소 두 축을 만족해야 한다.
2. 코드 자체를 성과로 세지 않는다. **공개 증거 / 논문 표 / 파트너 결정** 중 하나로 닫혀야 한다.
3. Ai2·ESA·파트너 접촉은 콜드 피치보다 재현 리포트·PR·검증된 지도 한 장을 먼저 만든 뒤 한다.
4. 비즈니스는 범용 SaaS부터 만들지 않는다. 같은 고통을 독립 파트너 2곳에서 확인하기 전에는
   `olmoearth-migrate`를 서비스형 감사 도구로 유지한다.
5. 새 응용을 추가할 때는 기존 하네스의 head·AOI·라벨 교체만으로 가능한지 확인한다.

## 로드맵과 현재 상태

- [x] **루프 1 (주 1–2): 바닥 다지기** — 2026-08-14 완료
  - [x] H200 세션에 olmoearth_projects 부트스트랩 (`bootstrap.sh`) — 2026-08-13 완료
  - [x] 소규모 AOI 엔드투엔드 추론 1회 완주 (forest_loss_driver 100건)
  - [x] lfmc 설정으로 학습 재현 1회 (ep33 test MSE 558.8)
  - [x] 마찰 기록 → 문서/코드 PR 후보 정리
- [ ] **루프 2 (주 3–6): A 벤치마크** ← 지금 여기 — 릴리스 × 모델 크기 매트릭스,
      프로파일링하며 코드 해부
- [ ] **루프 3 (주 7–9): B 부분 갱신** — RslearnWriter 출력에 FoldRefresh 물리기, 4-튜플 버전 태깅
- [ ] **루프 4 (주 10–12): C 케이스** — 한국 AOI LFMC + 도시 열 헤드 + 변화 귀인 뷰어

## 환경 요약 (자세한 건 h100-setup/CLAUDE.md)

- kt cloud AI Nexus, **H200 ×2** (143 GB each), 이미지: NGC PyTorch 2.7 / py312 / CUDA 12.8
- 영구 저장소는 `/home/work/data` 뿐. 프로젝트 루트: `/home/work/data/olmoearth/`
- 학습·장기 작업은 반드시 `./nexus job start`로 (터미널 분리). 체크포인트는 resume 가능하게.
- 레포(로컬): `~/dong/ai_projects/olmoearth_projects` — 설정 3종(dataset.json / model.yaml /
  olmoearth_run.yaml) 구조와 아키텍처 분석은 그쪽 세션에서 완료됨.

## 작업 규약

1. 작업 시작 전: 이 파일을 읽고, 오늘 할 일을 Worklog에 계획으로 먼저 적는다.
2. GPU 작업은 `h100-setup`의 `./nexus`로만. 산출물은 `/home/work/data/olmoearth/` 아래에만.
3. 작업 종료 시: Worklog에 결과·마찰·다음 단계를 기록하고 체크박스를 갱신한다.
4. PR 후보 마찰은 `## PR 후보` 절에 누적한다 (레포, 파일, 증상, 재현법).
5. 작업에서 새 개념을 부딪히면 `STUDY.md`에 개념 카드(+확인 질문)로 남기고,
   세션 종료 시 스터디 로그를 갱신한다. 확인 질문 = 면접 예상 질문.

## 아시아 확장 벤치마크 (산출물 C의 근거 데이터)

핵심 질문: **"릴리스 브릿지의 가정이 아시아에서 깨지는가?"** — 가설별로 측정해 누적한다.

| 가설 | 실험 | 상태 · 결과 |
|---|---|---|
| H0: 파이프라인이 아시아를 지원하는가 | E1: 데이터 소스 커버리지 감사 | ✅ **깨짐 확인** (2026-08-13): GLAD-S2 알림 타일은 남미(40W–90W)뿐 — forest_loss_driver는 아시아에서 알림 소스부터 교체 필요 (RADD 등). 구조적 갭 1호 |
| H1: 몬순 구름 → 시계열 결측 | E2: S2 유효관측 밀도 프로브 (14d×12구간, cc<50% 기준, 2025-01~07) | ✅ **확인** (2026-08-13): 빈 구간 페루 2/12 vs **보르네오 5/12** vs 강원 1/12. 총 장면수 308 vs 84 vs 75. PER_PERIOD_MOSAIC이 보르네오에선 타임스텝의 ~42%가 결측/오염 |
| H2: 소규모 필지 → 공간 상관 드리프트 | E3: 아시아 AOI 추론 + 릴리스 간 델타의 공간 자기상관 측정 | 예정 (루프 2~3, v1.1/v1.2 벤치마크와 결합) |
| H3: SAR 의존 태스크의 릴리스 드리프트 증폭 | E4: S1 위주 태스크(논벼/산악)에서 v1→v1.2 델타 비교 | 예정 |
| H4: 한국 LFMC 실효성 | E5: 한국 산불 AOI LFMC + FoldRefresh 마이그레이션 비용 | 예정 (루프 4, 클라이맥스) |

E2는 Planetary Computer STAC 직접 질의로 측정 — 추후 `olmoearth-migrate probe`의
데이터 가용성 모듈로 승격 예정 (probe CLI의 첫 부품).

## 요구 사양 실측 (온보딩 킷 "requirements" 페이지 초안, 2026-08-15)

- **추론**: GPU 사실상 불필요 수준 (Base 100윈도우 = GPU 2.3초, VRAM ~5GB; Nano/Tiny는 CPU 가능).
  지배 비용은 데이터 materialize(네트워크) — 대역폭 > GPU.
- **파인튜닝(Base)**: batch 32 → VRAM ~55GB / batch 축소 시 24GB 카드 가능.
  동결 구간은 IO-bound(NFS에서 GPU 0% 실측) → **로컬 NVMe가 GPU보다 중요**.
  44k 윈도우 × ~35에폭 ≈ 반나절(H200 1장).
- **사전학습**: Base = H100 2,989 GPUh (논문) — 파트너 범위 밖.
- 결론: 장벽은 하드웨어가 아니라 버전 정합·데이터 접근·스토리지 특성·재현 절차.

## Korea Earth Search 확장 로드맵 — 릴리스 인지형 Earth Embedding (2026-08-21 보정)

### 최신 조사로 폐기된 전제

**"Major TOM에 OlmoEarth 임베딩이 아직 없다"는 전제는 틀렸다.** Major TOM은 이미
`Core-S2L2A-249k-OlmoEarth-Base`를 공개했다. 전지구 균등 표본 248,719개, 단일 시점
S2-L2A 384×384 crop, OlmoEarth-Base 768차원 임베딩이다. 따라서 단순
`Korea-S2L2A-OlmoEarth` 복제는 최초성도, 충분한 연구 기여도 없다.

남아 있는 빈칸은 **다시 계산하는 법**이 아니라 **릴리스가 바뀌어도 의사결정 결과를
유지하는 법**이다. 기존 제품이 단일 시점·단일 릴리스 중심인 데 비해 우리는 다음을 묶는다.

- v1↔v1.2 쌍으로 계산한 동일 입력과 모델/코드/입력 스키마의 버전 manifest
- 2023–2026 다년 dense embedding에서 세계 변화와 모델 변화의 2×2 분해
- 구름·nodata·모달리티 결측이 큰 아시아 조건의 품질 마스크와 불확실성
- 전체 재계산 없이 검색 순위·Top-k·행정구역 집계를 보존하는 FoldRefresh

이 포지셔닝은 최신 흐름과 직접 맞닿는다. OlmoEarth v1.2는 RoPE로 임베딩 줄무늬를 줄이고
Base 추론 MACs를 v1 대비 2.9배 절감한 drop-in 릴리스다. 동시에 GFM 문헌은 평가 프로토콜
불일치, 실제 분포 이동에서 15–20% 성능 저하, shift 아래 과신을 보고한다. 조사한 범위에서는
**모델 릴리스 이동과 실제 지표의 시간 변화를 분해하고 선택적 재계산 비용까지 다룬 공개
벤치마크는 확인하지 못했다** — 이 문장은 계속 선행연구 검색으로 반증을 시도한다.

핵심 참고 자료:

- Major TOM OlmoEarth 249k: https://huggingface.co/datasets/Major-TOM/Core-S2L2A-249k
- OlmoEarth v1.2: https://allenai.org/blog/olmoearth-v1-1
- Ai2 대규모 추론 구조: https://allenai.org/blog/olmoearth-infrastructure
- Earth Embeddings: arXiv:2608.03410 / Earth Embeddings as Products: arXiv:2601.13134
- 평가 재현성 감사: arXiv:2605.12678 / EarthShift: arXiv:2605.29330
- shift·calibration: arXiv:2608.16614

### 세 목표 판정표

| 작업 | 취업 | 박사 | 비즈니스 | 우선순위 |
|---|---:|---:|---:|---|
| LFMC 재현 리포트 + sample 스키마 PR | 매우 높음 | 보조 사례 | 낮음 | **P0: 신용장** |
| v1/v1.2 릴리스 감사 하네스 + 4-튜플 manifest | 매우 높음 | 매우 높음 | 매우 높음 | **P0: 플래그십** |
| 제주 4개년 paired embedding + 검증된 변화 Top-k | 높음 | 매우 높음 | 높음 | **P1: 첫 파트너 증거** |
| Major TOM 변환기·스키마 호환성 테스트 | 높음 | 높음 | 중간 | P1: 공개 인프라 |
| 한반도 대규모 임베딩 복제 | 중간 | 낮음 | 검증 전 낮음 | P2: 수요 확인 전 보류 |
| 도시 열 등 새 head 추가 | 중간 | 중간 | 미확인 | P3: 기존 게이트 후 |

### Phase 0 — 공개 신용장 (지금부터 7일)

- [x] LFMC ep33 test 평가: 공개 ckpt 951.9 vs 재학습 558.8, 이슈 초안 완료
- [ ] `olmoearth_projects` 작업트리 정리 후 sample 스키마 수정 PR을 단독 제출
- [ ] `olmoearth_projects`는 신규 이슈 작성이 제한되어 있으므로 LFMC 보고서는
      `olmoearth@allenai.org` 또는 관련 PR 설명으로 전달하고 maintainer 경로를 요청
- [ ] Earth Embeddings(2608.03410)·v1.2 보고서 정독 → STUDY 카드와 실험 변수 확정
- [x] Major TOM의 기존 OlmoEarth 249k 공개본 확인 — "최초 확장판" 주장 폐기

### Phase 1 — 릴리스 감사 하네스 (2~3주)

- 동일 원시 입력을 v1/v1.2에 넣는 paired 실행을 고정한다. manifest 최소 필드:
  `model_id + weight hash + code commit + input modalities/bands/timesteps + AOI/time +
  CRS/GSD + normalization/centering + cloud/nodata mask + output schema`.
- 베이스라인은 raw cosine, 공통 mean-centering, 연도별 mean-centering, Orthogonal Procrustes.
- 지표는 embedding CKA/분포 거리뿐 아니라 **neighbor overlap, Top-k Jaccard, Kendall τ,
  행정구역 집계 오차, 공간 자기상관, bootstrap CI, 재계산 비율-품질 곡선**을 포함한다.
- 기존 Major TOM 249k를 전지구 참조/상호운용성 테스트로 재사용하고 불필요한 재계산을 피한다.
- 공개 단위는 `olmoearth-release-audit` 재현 스크립트 + 작은 sample + 결과표로 먼저 잡는다.

### Phase 2 — 파트너 증거와 두 개의 문 (3~6주)

- 제주 변화지도는 구름/nodata 마스크와 사람이 확인한 정답 이벤트를 통과한 뒤에만 데모로 쓴다.
- MARC·테크포임팩트 및 독립 후보를 포함해 최소 3회 문제 인터뷰:
  **누가 어떤 결정을, 얼마나 자주, false positive/누락 각각 얼마의 비용으로 내리는가?**
- Ai2에는 이슈/PR, 재현성 매트릭스, 아시아 결측 프로브, 파트너용 지도 1장을 묶어 접촉한다.
  최신 Senior Research Engineer 공고가 요구하는 partner deployment, task head, rslearn 개선,
  기술 커뮤니케이션을 이 패키지가 직접 증명하게 한다.
- Φ-lab/CloudFerro에는 "최초 OlmoEarth 확장"이 아니라 **paired temporal/versioned extension의
  Major TOM 호환 schema 리뷰**를 요청한다.

### Phase 3 — 논문·파일럿 패키지 (6~12주)

- 논문 주기여: ① 세계 변화/모델 변화 분해 ② 릴리스 간 검색·집계 안정성 벤치마크
  ③ 선택적 재계산의 비용-오차 보장 ④ 아시아 결측·구름 조건 ⑤ 재현 가능한 artifact manifest.
- 데이터 공개가 가능하면 이름은 범용 복제보다 범위를 드러내는
  `Korea-Temporal-OlmoEarth-v1-v1.2`로 하고, 작은 paired benchmark부터 공개한다.
- 비즈니스 오퍼는 모델 판매가 아니라 세 단계 서비스로 검증한다:
  **Release Readiness Audit → Local Evidence Pack → Recurring Decision-Continuity Refresh**.
- OlmoEarth Artifact License는 사용·수정·파생물 배포를 허용하지만 군사/국방·감시/치안 및
  석유·가스·광업·산림파괴 같은 extractive 용도를 금지한다. 유료 파일럿 전 데이터 권리와
  downstream 제한 승계 여부를 별도 법률 검토한다.

### Phase 4 — 확장 게이트

다음 세 조건 전에는 한반도 전체 materialize, 범용 SaaS, 새 응용 여러 개로 확장하지 않는다.

1. 독립 파트너 2곳이 같은 릴리스/갱신 고통을 확인한다.
2. 최소 1곳이 데이터·검수 시간·LOI 또는 파일럿 비용 중 하나를 약속한다.
3. 동일 방법이 제주 연안 외 두 번째 태스크(LFMC 또는 양식장)에서도 재현된다.

### 제주 변화탐지 — v1~v4 실패 계보와 v5 설계 (2026-08-21)

각 단계가 다음 단계의 근거가 된 실패 사슬. 전 과정이 논문 §"왜 순진한 임베딩
변화탐지가 아시아에서 실패하는가"의 재료.

| 버전 | 방법 | 결과 | 배운 것 |
|---|---|---|---|
| v1 | 연도 간 cosine 거리 | ❌ Top-20 전부 바다 | 바다는 파도/반사로 매년 지문이 변함 (바다 점수 std 0.094 vs 육지 0.042~0.060) |
| v2 | 계단형(그룹간−그룹내) + WorldCover 층화 z | ⚠️ 육지화 성공, 그러나 26/30이 2023→2024에 집중 → **육안 검증 5/5 구름** | 층화·계단형은 요동은 잡지만 오염은 못 잡음 |
| v3 | 구름비율 **평균** ≤0.20 마스킹 | ⚠️ 시점/공간 분산, 실제 변화 1건(33.5087N 126.5747E 벌채·개발 진행) 발견, 그러나 5곳 중 3곳 구름 잔존 | 평균은 1/12 오염을 희석해 못 걸름 |
| v4 | **최악 모자이크** ≤0.35 마스킹 | ❌ 생존 픽셀 1.2%, 전부 바다 → 육지 후보 0 | 제주 최악-모자이크 평균 0.53~0.84 → **사후 마스킹 원리적 불가** |
| **v5** | 입력 단계 수정: `PER_PERIOD_MOSAIC` 재수집 + PPI 검증 | 🔄 2026-08-21 22:10 착수 (216윈도우, 밤새) | 합성 레시피는 manifest 필수 항목 |

v5 설계 (문헌 반영):
1. **입력 수정** — rslearn 임베딩 가이드 기본값 `space_mode: MOSAIC`(기간당 1장면)이
   구름 지역에 부적합. Ai2 실전 설정(lfmc)의 `PER_PERIOD_MOSAIC`(기간당 다장면 합성)으로
   제주 4개년 재수집 → `embed_jeju_v2`. **v1 데이터는 보존** → "합성 레시피가 임베딩과
   하류 결론을 얼마나 바꾸는가" 통제 실험(입력 스키마도 4-튜플의 일부라는 주장의 직접 증거).
2. **검증 프로토콜 (Wang et al., RSE 2025 / arXiv:2407.13659)** — 변화 점수로 층화
   무작위 표본 → 시계열 RGB 칩 육안 판정 → **Prediction-Powered Inference**로 변화 면적과
   신뢰구간 추정. Top-k는 "조사 우선순위", 면적은 "PPI+CI"로 분리 보고.
   Top-k 정밀도만 보고하면 선택 편향으로 면적 추정이 무효.
3. **품질 산출물** — `cloud_stats.npz`(제주 4개년 픽셀별 구름 평균·최댓값)를 공개 자산으로
   유지. Earth Embeddings 서베이가 지적한 "임베딩 제품에 없는 품질 마스크"의 실체.

### 병행 트랙 — 제주 연안 서식지 감시 (MARC 연계 후보, 2026-08-21 착수)

- 목적: 남방큰돌고래 서식지(대정~모슬포) 연안의 이용 압력 변화 감시 —
  카카오임팩트/테크포임팩트·MARC 접점 활용, Ai2식 "파트너 케이스"의 한국판.
- 상태: 제주 54윈도우 × 4개년 materialize와 v1 S2-only 임베딩 추출 완료.
  `change_4yr.py` 후처리는 4개년 각 54윈도우 로딩 후 CPU 계산 단계까지 확인.
- 다음: ① cloud/nodata mask ② 공통 중심화 vs 연도별 중심화 ablation
  ③ v1.2 paired 추출 ④ 변화 Top-k의 사람이 확인 가능한 근거 칩 생성
  ⑤ 어장정보도·인허가 오버레이 후 파트너 리포트.
- 수요 확인 원칙: 검증된 데모 1장과 "이 지도가 바꿀 결정" 질문을 먼저 보여주고 확장 결정.

## 확장 백로그 (SDG/ODA 프레임, 2026-08-14 브레인스토밍)

원칙: 새 프로젝트가 아니라 **같은 파이프라인에 head/지역 교체**로 되는 것만 승격한다.

- **해조류 양식장 매핑 (한국→동남아)** — 루프 4 후보로 승격. 라벨 공짜(해수부 어장정보도
  공공데이터), S1 SAR에 구조물 선명 → H3(SAR 태스크 드리프트) 검증 태스크 겸용.
  블루카본+SDG14+연안생계+한국 종주국 서사. 선행 연구 공백 = 선점 가능.
- **도시 열 노출 헤드** — 기존 계획 유지, Tzu-Hsin Karen Chen 계보(도시 열/비공식 정착지
  EO+DL)의 후속으로 프레이밍. 차별점 = 릴리스 마이그레이션 관리(연도 비교 안정성).
- **SDG 지표 파이프라인 프레임** — 지도→행정구역 지표 집계→릴리스 불변성 관리.
  SDG 보고는 연도 비교가 생명이라 "모델 변화 vs 세상 변화" 문제가 실무 요구가 되는 지점.
  FoldRefresh(B)의 응용 서사로 사용.
- (백로그) Seto류 도시 확장 예측 — 스코프 큼, 플래그십 이후 논문감.
- (백로그) 기후 트래킹(ETH NCCM류 구름/기후 프로덕트) × 위성 종합 × 헬스 데이터 접목 —
  아이디어만 기록 (2026-08-14, 확장 금지 원칙에 따라 보류).
- (백로그) 비공식 정착지 × 홍수 노출 인구 (SDG 11, Sen1Floods11 연계).
- 기아/작물(SDG 2)은 케냐/모잠비크 기존 프로젝트를 벤치마크 태스크로만 활용.

## PR 후보

1. **`olmoearth-runner`의 `requires-python <3.12` 상한** — NGC PyTorch 컨테이너(py3.12)를 쓰는
   조직은 시스템 파이썬으로 설치 불가. `pip install olmoearth-runner` →
   "No matching distribution found". 상한을 풀거나 문서에 명시할 것을 제안할 가치.
   (우리는 uv로 py3.11 venv를 만들어 우회 — bootstrap.sh 참고)
2. **sample 프로젝트의 반쪽 스키마 마이그레이션 (확실한 버그)** —
   `olmoearth_run_data/sample/annotation_task_features.geojson`은 `oe_*` 접두사로
   마이그레이션됐는데 `annotation_features.geojson`은 legacy `es_*` + 스칼라 `es_label`인
   채로 남아 runner 0.1.14의 pydantic 검증(`oe_annotations_task_id`, dict형 `oe_labels`)에서
   실패. README대로 실행하면 바로 깨짐. **로컬 레포에 수정 적용 완료** (es_→oe_,
   `es_label: 1` → `oe_labels: {category: 1}`) — 이대로 PR 가능.
3. `olmoearth-runner`가 라이브러리가 아니라 완전 고정(`==`) 의존성의 잠긴 앱으로 배포됨 —
   requires-python 상한과 함께 논의 가치 (#1과 묶어서).
4. **forest_loss_driver가 외부에서 실행 불가 (문서 vs 실제 불일치, 중요)** —
   `docs/forest_loss_driver.md`는 외부 사용자에게 추론 실행법을 안내하지만, `dataset.json`의
   pre/post_sentinel2 레이어가 `olmoearth_datasets.sentinel2_l2a.Sentinel2L2A`
   (Ai2 내부 API, `https://datasets.olmoearth.allenai.org`, Bearer 토큰 필수 → 익명 401)를
   사용. 게다가 `OEDATASETS_API_URL` 기본값이 `""`라서 미설정 시 명확한 에러 대신
   "Invalid URL '/api/v1/items/search'" **무한 재시도 루프**에 빠짐 (이중 버그).
   수정안: (a) dataset.json을 공개 `planetary_computer.Sentinel2`로 교체 또는 문서에 명시,
   (b) olmoearth_run에서 URL 미설정 시 fail-fast. **로컬에서 (a) 패치 적용 완료.**
5. (예비) `get_local_checkpoint()`의 캐시 레이스/부분 다운로드 문제 — 실제로 부딪히면 기록
6. **macOS에서 `olmoearth_projects.main`이 행** — `utils/mp.py`의 `init_mp()`가
   forkserver + torch preload를 강제하는데 macOS에서 Pool 생성이 안 돌아옴.
   러너 직접 호출로 우회 가능. 재현: README의 prepare_labeled_windows 명령을 맥에서 실행.
7. **✅ 결론 확정 (2026-08-21): HF의 LFMC 체크포인트가 잘못된 파일** —
   최종 매트릭스: 문서 주장 580.6 / 공개 ckpt 실측 **951.9** / 공개 데이터·레시피로
   우리가 재학습(ep33, 그들의 1/3 학습량) **558.8**. 데이터·코드·레시피는 정상이고
   체크포인트 업로드만 불량이라는 결론. Ai2 이슈 등록 준비 완료 (ISSUE_DRAFT_lfmc.md).
   (조사 이력) 공개 LFMC 아티팩트의 문서 수치 재현 불가 —
   docs/lfmc.md는 "test set MSE 580.6"을 주장하나, 공개 ckpt + 공개 dataset.tar(20251029) +
   레포 model.yaml로 실측 시 test split(4,585윈도우) MSE **951.9**. 통제 실험으로
   버전 효과 기각(rslearn 0.0.27 vs master: 995.3 vs 995.4, val split), split 효과 기각.
   단서: ckpt는 에폭당 662스텝, 공개 데이터는 655스텝 → ~1% 다른 데이터 스냅샷으로
   학습된 가중치로 추정. 우리 파인튜닝 완료 후 최종 비교로 확정할 것. (2026-08-15)
8. **runner가 상대 project_path에서 깨진 심링크 생성** — `_setup_project_env`가
   심링크 타깃을 resolve하지 않아 상대경로 입력 시 "file not found". 한 줄 수정감
   (`target.resolve()`), olmoearth_run 레포. 로컬 재현 확인 (2026-08-14).

## Worklog

### 2026-08-21 — 취업 × 박사 × 비즈니스 전략 보정

- 계획: 기존 로드맵 순서를 보존하면서 세 목표가 같은 산출물을 공유하도록 최신 생태계 조사.
- 확인: Ai2가 OlmoEarth partner deployment·task head·rslearn 개선을 담당할 Senior Research
  Engineer를 채용 중이며, 플랫폼도 global embedding precompute와 run-anywhere를 다음 축으로 명시.
- 중요 수정: Major TOM에 248,719개 `OlmoEarth-Base` 임베딩이 이미 공개돼 있어
  "최초 OlmoEarth 확장판" 포지셔닝을 폐기. paired temporal/versioned benchmark로 전환.
- 연구 근거: 재현성 불일치(arXiv:2605.12678), 현실 분포 이동(EarthShift), shift 아래 calibration
  악화(2608.16614)를 확인. 모델 릴리스 이동과 세계 변화를 함께 분해하는 문제로 기여를 좁힘.
- 사업 결론: Ai2 플랫폼과 범용 추론으로 경쟁하지 않고 아시아 데이터 가용성·현지 정답지·
  릴리스 전환 검증을 묶은 서비스형 감사로 시작. 파트너 2곳 반복 수요 전 SaaS 확장 금지.
- 다음: 작업트리 정리 → sample PR → LFMC 보고 전달 경로 확보 → v1/v1.2 paired manifest와
  제주 cloud/nodata 검증을 먼저 완결.

### 2026-08-13
- 계획: 프로젝트 구조 생성 (GOAL.md, bootstrap.sh, 에이전트 파일). H200 부트스트랩.
- 결과: **부트스트랩 성공.** H200 ×2 인식, torch 2.7.1+cu126, rslearn 0.0.27 /
  olmoearth-runner 0.1.14 / olmoearth-pretrain 0.0.2 설치 완료.
  환경은 `/home/work/data/olmoearth/.venv`(py3.11), 캐시·uv 전부 영구 저장소.
- 마찰: olmoearth-runner requires-python <3.12 → PR 후보 #1로 기록. 시스템 py3.12 대신
  uv venv 방식으로 bootstrap.sh 재설계.
- 다음: sample 프로젝트 소규모 AOI 엔드투엔드 추론.

### 2026-08-13 (2차)
- 계획: sample 프로젝트 `prepare_labeled_windows` 완주.
- 결과: **성공.** 6개 라벨 윈도우 생성 (`/home/work/data/olmoearth/scratch/sample/dataset`,
  LA 롱비치 AOI, EPSG:32611, 10m, train/val 분할). rslearn 윈도우 구조
  (`windows/<group>/<name>/metadata.json` + layers/) 확인.
- 마찰: PR 후보 #2 (es_→oe_ 반쪽 마이그레이션) 발견·수정. olmoearth-runner의 고정 의존성
  구조 확인 (PR 후보 #3).
- 다음: 공개 체크포인트가 있는 프로젝트(forest_loss_driver 또는 mangrove)로 실제 GPU 추론
  엔드투엔드 (`partition → build_dataset → run_inference → postprocess → combine`).

### 2026-08-13 (3차) — 진행 중
- 계획: forest_loss_driver 실전 GPU 추론 엔드투엔드. 페루 아마존 산림손실 이벤트 100건을
  10개 원인 클래스로 분류 (레포 동봉 prediction_request_geometry.geojson 사용).
- 실행: 공개 체크포인트(HF, 364MB) 다운로드 → `./nexus job start fld-infer`로 백그라운드
  추론 시작 (18:39 KST). 로그: `/home/work/data/.jobs/fld-infer.log` (영구).
- 마찰 1: 1차 실행이 `Invalid URL '/api/v1/items/search'` 무한 재시도에 빠짐 →
  원인: dataset.json이 Ai2 내부 API(olmoearth_datasets, 토큰 필수) 의존 + URL 미설정 시
  fail-fast 없음 → **PR 후보 #4 (이중 버그, 이번 발견 중 최중량)**.
  dataset.json을 공개 planetary_computer.Sentinel2로 패치 후 19:4x 재시작.
- 마찰 2: 로컬→H200 SSH 터널이 조용히 죽어 job 제어 불가 → `./nexus tunnel down/up`으로
  복구. watch/모니터에 터널 헬스체크 필요 (개선 아이디어).
- 병행: 아시아 확장 벤치마크 E1·E2 완료 (위 표 참고) — GLAD-S2는 남미 전용(H0 깨짐),
  보르네오 몬순 결측 5/12 구간 확인(H1 확인). E2 프로브는 `olmoearth-migrate probe`의
  첫 부품으로 승격 예정.
- 마찰 3: 2차 실행도 실패 — rslearn `ingest: false`(직접 materialize) 경로가
  planetary_computer.Sentinel2 + `space_mode: CONTAINS` 조합에서
  `DirectMaterializeDataSource.get_item_by_name` 미구현으로 `NotImplementedError`
  (rslearn `direct_materialize_data_source.py:92`) → **PR 후보 #6 (rslearn 레포)**.
  `ingest: true`로 우회, 3차 재시작 (GPU1 배정 — GPU0은 p1_anchor 실험 사용 중).
- 마찰 4~5 (3~5차 실행): model.yaml이 **미출시 rslearn(git master) 전용 API** 사용 —
  `enable_confusion_matrix`, `BestLastCheckpoint`는 PyPI 어느 릴리스(0.0.23/0.0.27)에도
  없고, `SimpleTimeSeries` 채널 시맨틱도 릴리스별로 달라 ZeroDivisionError 발생.
  runner(PyPI)는 반대로 옛 rslearn API를 요구(git rslearn과 비호환) → 삼각 스큐.
  **PR 후보 #7**: 공개 체크포인트 + main 설정 + PyPI 패키지 조합이 상호 비호환 —
  설정/체크포인트에 호환 버전 명시 필요 (4-튜플 레지스트리 논거의 실증).
- 정착 조합: **runner 0.1.14 + rslearn 0.0.27 + model.yaml 최소 패치 2건**
  (enable_confusion_matrix 제거, BestLastCheckpoint→ModelCheckpoint).
- 기타: `./nexus job start`가 이전 job 미종료 시 조용히 실패하는 문제,
  pgrep 자기매칭 함정 확인 → 직접 SSH(setsid nohup) 방식으로 전환.

### 2026-08-20 — "Korea Earth Search" 임베딩 검색 엔진 (완도→제주+지리산)
- 구축: 완도(12)+제주(54)+지리산(16) = 82윈도우, S2-only 12모자이크, OlmoEarth-Base
  임베딩(768d, 40m 격자, ~540만 벡터) 영구 저장 (`/home/work/data/olmoearth/embed_search`).
- 방법 확립: **mean-centering이 결정적** (이방성 교정 전 cosine이 0.7에 몰림 → 교정 후
  판별력 확보). 평가는 ESA WorldCover 정답, 클래스당 무작위 5쿼리 precision@2000.
- 수치: 제주 — cropland .816(×14.8), built .658(×18.3), grassland .371(×14.1),
  tree .756(×3.6), water 1.000. 완도 — built ×26, cropland ×8.3.
- 데모: 한라산 아고산 쿼리가 백록담 일대를 정확히 구획 (구상나무 프로브 절반 통과 —
  아고산대가 임베딩 공간에서 분리됨 확인). 교차지역 검색 성립하나 의미 정밀도는 과제.
- 마찰/교훈: ① rslearn 임베딩 가이드 기본값(워커16+load_all_crops)이 대형 윈도우에서
  조용한 OOM (28/82에서 사망, 트레이스백 없음) → 워커6/배치4로 해결. ② **PC의 S1이
  제주 2024 완전 공백**(54/54 윈도우 0장, 완도는 정상 — 궤도 커버리지 경계) →
  모달리티 불일치는 임베딩 공간을 가르므로 S2-only로 전체 통일 재추출. 온보딩 킷 함정 2건.
- 다음: 뷰어(클릭 쿼리 웹앱), FAISS 인덱스, 팜맵/어장정보도 라벨로 의미 특이적 쿼리.

### 2026-08-21 — 의미 특이적 쿼리 (양식장 프로토타입) + 상세 리포트
- OSM 양식장 폴리곤(완도 48, 제주 9)으로 few-shot 프로토타입 검색 구현:
  완도 19곳 평균 벡터 → **held-out 20곳 중앙값 백분위 100.0** (75%가 상위 1%),
  **제주 교차 지역 9/9 전부 백분위 96.0~99.8** (해상 김양식→육상 수조 유형 전이 성립).
- 한계 기록: 제주 히트맵이 해안 일반에도 반응(특이도 불완전), WorldCover 2021 vs
  임베딩 2024 시차, OSM 라벨 커버리지 편향 → 어장정보도 전수 폴리곤으로 재평가 예정.
- 상세 리포트 아티팩트: https://claude.ai/code/artifact/fe29beda-4856-460a-b0ac-9a7a6346f382

### 2026-08-15 — lfmc 학습 재현 + 공개 아티팩트 정합성 조사
- 학습: 44,022윈도우(공개 dataset.tar), 원본 레시피. val 곡선: 832(ep0)→680(ep6)→
  687(ep20 해동)→**652(최저)**→667(ep35). ep36에서 사용자 GPU 필요로 의도적 중단.
  체크포인트 보존: best=epoch33, last(~ep36) — trainer_checkpoints/.
- 실측: 동결 구간 = NFS IO-bound(GPU 0%, 에폭 18~24분), 해동 후 = GPU-bound 전환
  (GPU 100%/55GB). "MACs ≠ 실배포 비용"의 직접 증거 (벤치마크 A 원자료).
- 정합성 매트릭스: 공개ckpt val 995.3 / 우리ep0 val 832.3 / 공개ckpt@master 995.4
  (버전 기각) / 공개ckpt test 951.9 vs **문서 주장 580.6** (재현 실패, PR 후보 #7).
- 남은 한 수: best(ep33)를 test split 평가 → 3자 대조로 원인 확정 (GPU 여유 시).
- 데이터 교훈: 49.8GB 다운로드 24분 vs NFS tar 해제 ~4시간 (파일 100만+ 개) — 온보딩 킷 기록.

### 2026-08-14 — ✅ 루프 1 완료: 첫 엔드투엔드 GPU 추론 성공
- **결과: 페루 마드레 데 디오스 산림손실 100건 분류 완료.**
  agriculture 74 / mining 16 / logging 3 / road 3 / hurricane 2 / burned 1 / river 1
  — 금광 지대 특성과 부합. 추론 자체는 H200에서 2.3초(13 batch, 5.7it/s);
  전체 시간은 데이터 materialize가 지배.
- 산출물: `results/result.geojson` (H200 영구 저장소), 시각화 대시보드
  https://claude.ai/code/artifact/b5a54835-93cd-4ebc-9905-c2a7813b977e
- 다음(루프 2): 이 파이프라인을 하네스로 벤치마크 매트릭스(A) 설계 —
  v1 × Nano/Tiny/Base/Large × precision, GPU-초/km² 측정. + lfmc 학습 재현.
