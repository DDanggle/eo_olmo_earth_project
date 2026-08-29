# OlmoEarth 연구 목표 — 살아있는 계획서

> 이 파일이 이 프로젝트의 단일 진실 공급원(SSOT)입니다.
> 에이전트든 사람이든, 작업 시작 전에 읽고 / 끝나면 Worklog와 상태를 갱신합니다.

## 미션 — Decision-Continuous Earth Intelligence

**"세계와 기반모델이 동시에 바뀌어도, 컴퓨트와 라벨이 부족한 환경 조직이
지도 기반 의사결정을 과학적으로 유효하고 연속되게 유지하게 한다."**

연구 질문은 “새 모델이 더 정확한가?”가 아니라 **세계 변화, 입력 변화, 모델 릴리스 변화가
최종 검색·집계·보전 결정에 각각 얼마나 기여하는가**다. 제주는 첫 검증장이고,
`olmoearth-migrate`/`olmoearth-release-audit`는 이 연구의 공개 시스템 산출물이다.
전체 연구 설계는 `RESEARCH_STRATEGY.md`, 후속 관측·계산·근거 라우팅과 사업 gate는
`EARTHROUTE_PROGRAM_NOTE.md`, 문헌 검증 장부는 `PAPER_READING_LIST.md`, 파트너 공동설계 초안은
`PARTNER_BRIEF_MARC.md`가 맡는다.

최종 산출물 세 개:

| # | 산출물 | 핵심 수치 | 증명하는 것 |
|---|---|---|---|
| A | WorldShift × ModelShift 벤치마크 | v1/v1.2 × 입력 레시피 × 4개년: 이웃·Top-k·집계·비용 | 세계/입력/모델 변화 분해 |
| B | 유효한 선택적 갱신기 | PPI 신뢰구간 + 25% 이하 재계산으로 결정 지표 유지 | 통계적 유효성 + FoldRefresh |
| C | 파트너 Evidence Pack | 제주 연안 압력 후보 + 현장 검수 + 두 번째 태스크 전이 | 의사결정·생태·일반화 |

### 2026-08-25 현재 우선순위 — MountainShift가 실행 임계경로

아래 2026-08-24 교수 판정의 **측정 결과와 약점은 유지**하되, 실행 우선순위는 보정한다.
EarthEmbedContract와 FoldRefresh를 다시 발명하지 않는다. 현재 새 연구 기여는 한국·네팔·스위스에서
`global EO cache + region-static residual + cutoff-valid live residual`이 segmentation·event retrieval과
저라벨 국가 전이를 개선하는지 검증하는 `MountainShift`다.

| 목표 | 이번 track이 남길 증거 |
|---|---|
| **Ai2/OlmoEarth 취업** | OLMo 중심 + 타 GeoFM 1개 재현 benchmark, rslearn/data-contract 마찰 PR, live regional deployment report |
| **박사/CVPR** | 3-way leave-one-country-out, 0/1/5/10% label curve, `E_static/E_live/E_transfer/E_refresh` 분리, negative-transfer·kill 결과 |
| **비즈니스** | 같은 event schema로 한국·네팔·스위스 live source를 ingest하고, 검색 우선순위·freshness·보류를 보여주는 prospective pilot |

한국 C2-C는 최대 1일의 지원 gate다. 통과하지 않아도 v1/10밴드 probe 또는 새 S1/S2/DEM
materialization으로 한국 arm을 계속하며 네팔·스위스를 막지 않는다. FoldRefresh는 encoder release가
바뀔 때 `z_global` page를 갱신하는 continuity/cost arm이고, local/live 정확도 기여로 세지 않는다.

### 2026-08-24 교수 판정 — 유의미하지만 아직 논문·제품으로 닫히지 않았다

| 축 | 현재 판정 | 근거 | 다음 승격 조건 |
|---|---|---|---|
| 과학적 실행 | **강함** | 216 exact-input release audit, 248,719 paired product audit, 사전 gate로 M5 일반 주장 철회, 128 tests 통과 | public downstream task에서 silent error와 gate 이득 재현 |
| 첫 논문 | **중간 이하** | 좌표 호환성 실패는 강하지만 task 정확도·calibration 0건 | frozen old head + realistic mismatch + 두 번째 release family |
| Ai2 취업 | **높은 잠재력, 외부 신호 미완성** | rslearn/runner 마찰·재현성 감사·파트너형 문제와 공고 요구가 정합 | PR/이슈/기술글 중 최소 1건 외부 merge·응답 |
| 박사 프로그램 | **유의미** | release-aware representation lifecycle이라는 일반 질문과 반증 가능한 단계가 있음 | 논문 1을 metadata audit가 아닌 task-risk science로 만들기 |
| 비즈니스 | **가설 단계** | audit offer는 명확하지만 독립 고객 인터뷰·LOI·유료 검증 0 | 같은 손실을 말하는 파트너 2곳 + 1곳의 데이터/시간/비용 약속 |

장기 프로그램 이름은 **EarthKV**로 사용한다. 단 층위를 섞지 않는다.

```text
EarthKV (program/system abstraction)
├─ EarthEmbedContract (compatibility guard; task-risk 자산)
├─ FoldRefresh (repair operator; 별도 검증 자산 재사용)
├─ EarthRoute (admission/escalation policy; 후속)
└─ MountainShift (현재 method/transfer 실행 track)
```

AlphaEarth와 TESSERA는 이미 version metadata를 제공하므로 `버전 필드가 없다`를 novelty로 쓰지 않는다.
현재 novelty 후보는 **task-level compatibility 검증 + action별 risk–cost 곡선**이다. paging·eviction·
distributed/federated EarthKV는 구현 전까지 문서의 장기 확장일 뿐 현재 기여가 아니다.

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
- [ ] **루프 2 (주 3–6): MountainShift transfer benchmark** ← 지금 여기 — **한국·네팔·스위스
      cross-region task gate가 임계경로**. public mountain task와 봉인한 한국 split에서 frozen
      embedding probe·event retrieval·region-static/live residual을 비교하고, 3-way
      leave-one-country-out과 target label 0/1/5/10% 곡선을 만든다. release silent-error 표는
      EarthEmbedContract guard로 병렬 유지하되 C2-C 입력 복구가 전체 실행을 막지 않는다.
- [ ] **루프 3 (주 7–9): B 부분 갱신** — 별도 `decision-ready-earth-ai`에서 검증된 FoldRefresh
      방법을 RslearnWriter 출력·K-Earth 결정지표에 이식하고 4-튜플 버전 태깅
- [ ] **루프 4 (주 10–12): C Evidence Pack** — 제주 연안/오름의 공식근거·사람검수 dossier와
      LFMC 또는 양식장 두 번째 태스크 전이; 실제 파트너 결정·검수시간으로 닫기

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
- cache 호환성 베이스라인은 sealed gallery가 생긴 뒤 raw cross-version cosine, train-only 공통
  mean-centering, Orthogonal Procrustes를 비교한다. 연도별 mean-centering은 미래/평가연도 통계를
  쓰지 않는 경우에만 둔다. 현재 8건 smoke에는 이 alignment baseline을 fit하지 않는다.
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

### 제주 변화탐지 — v1~v7 증거 계보 (2026-08-21~22)

각 단계가 다음 단계의 근거가 된 실패 사슬. 전 과정이 논문 §"왜 순진한 임베딩
변화탐지가 아시아에서 실패하는가"의 재료.

| 버전 | 방법 | 결과 | 배운 것 |
|---|---|---|---|
| v1 | 연도 간 cosine 거리 | ❌ Top-20 전부 바다 | 바다는 파도/반사로 매년 지문이 변함 (바다 점수 std 0.094 vs 육지 0.042~0.060) |
| v2 | 계단형(그룹간−그룹내) + WorldCover 층화 z | ⚠️ 육지화 성공, 그러나 26/30이 2023→2024에 집중 → **육안 검증 5/5 구름** | 층화·계단형은 요동은 잡지만 오염은 못 잡음 |
| v3 | 구름비율 **평균** ≤0.20 마스킹 | ⚠️ 시점/공간 분산, 실제 변화 1건(33.5087N 126.5747E 벌채·개발 진행) 발견, 그러나 5곳 중 3곳 구름 잔존 | 평균은 1/12 오염을 희석해 못 걸름 |
| v4 | **최악 모자이크** ≤0.35 마스킹 | ❌ 생존 픽셀 1.2%, 전부 바다 → 육지 후보 0 | 제주 최악-모자이크 평균 0.53~0.84 → **사후 마스킹 원리적 불가** |
| **v5** | 입력 수정 시도: `PER_PERIOD_MOSAIC` 재수집 | ❌ **가설 기각 · 의미적 중복**: 216/216 계산은 완료했으나 결과 래스터가 v1과 **md5까지 동일**. v1↔v5 cloud/zero 지표와 blind RGB 5/5, 2,592 source-group hash, 고정 픽셀/임베딩 표본 전부 일치 | 설정 문자열이 아니라 실행 handler·item hash·픽셀을 먼저 비교해야 한다 |
| **v6** | 모델 입력 타임스텝 4 ↔ 12 민감도 | ⚠️ Top-30 교집합 5, **Jaccard 0.091**. 단 실제 첫 4기간은 역시간순·계절 불일치 | timestep 수가 후보 지도를 크게 바꾸므로 계절 정렬 전에는 변화 정답으로 해석 금지 |
| **v7 smoke** | 기간당 최대 3 coverage + SCL BestClear | ✅ **golden window 사전 기준 전부 통과**: 첫 4기간 bad proxy 95.64% 감소, 결측 −0.0082%p, 고정 target 1.00→0.00, RGB 구름 제거 확인 | 실제 장면 선택 개입이 입력 품질을 바꾼 첫 증거. 아직 1윈도우이므로 제주 전체 주장 금지 |

v5 실패 기록과 후속 감사 (2026-08-22):

`PER_PERIOD_MOSAIC`로 216윈도우를 2시간에 걸쳐 재수집했으나 **결과 래스터가 v1과 md5까지 동일**했다.
`items.json` 확인 결과 두 모드 모두 기간당 장면이 1개다. 원인은 `space_mode`가 윈도우를 *공간적으로*
덮는 데 필요한 장면을 모자이크하는 옵션이고, 같은 기간의 여러 장면을 겹쳐 구름을 메우는 *시간적*
합성이 아니기 때문이다. 우리 윈도우(1024px=10km)는 S2 타일(110km) 하나에 완전히 들어가므로 두
모드가 같아진다. Ai2의 lfmc 설정도 동일 구조이므로 **Ai2 자신의 프로덕션 설정도 구름 노출은 같다**
(지도학습 회귀는 모델 강건성에 의존).

후속 감사에서 더 정확한 원인이 확인됐다. rslearn 0.1.13(commit `bbbc18b`)에서 두 SpaceMode는 같은
`match_with_space_mode_mosaic` handler를 쓴다. 정규화 설정, 2,592개 ordered item group, B02 전수
품질 지표, 원본 24쌍 전체 밴드, 임베딩 24쌍 공간표본이 모두 같았다. 따라서 v5는 새 합성 레시피
실험이 아니라 **설정 별칭으로 만든 중복 실행**이다.

부수 소득: 같은 입력 → 같은 임베딩이 바이트 단위로 재현됐다 = **파이프라인 결정성 확인**.
이것이 이후 v1 vs v1.2 차이를 모델 효과로 귀속할 수 있는 근거가 됐고, 2026-08-24 dose-response
실험의 `dose 0 byte-identical 8/8` 타당성 검사도 같은 성질에 기댄다.

v6 설계 (2026-08-22, 타임스텝 교정):
1. **타임스텝 수 교정** — 우리 `model_s2.yaml`이 `layers: [sentinel2_l2a, .1, .2, .3]`으로
   **12개 중 앞 4개만** 사용했다. 구름 낀 모자이크 1장이 입력의 25%를 차지하고 계절 신호도
   반년치가 누락된다. rslearn 임베딩 가이드 예제(4개 레이어)를 그대로 따른 결과다.
   → `model_s2_t12.yaml`(12타임스텝, 출력 레이어 `embeddings_t12`)로 재추출하고, 4타임스텝 결과와
   나란히 두어 **"입력 타임스텝 수가 변화탐지 결론을 얼마나 바꾸는가"** 통제 실험으로 사용한다
   (입력 스키마 = 4-튜플의 일부라는 주장의 직접 증거). 실제 결과는 Jaccard 0.091이었다.
   단 첫 4기간이 역시간순·계절 불일치이므로 이 경로는 이후 `REFUSED` 가드로 차단했다.
2. **검증 프로토콜 (Wang et al., RSE 2025 / arXiv:2407.13659)** — 변화 점수로 층화
   무작위 표본 → 시계열 RGB 칩 육안 판정 → **Prediction-Powered Inference**로 변화 면적과
   신뢰구간 추정. Top-k는 "조사 우선순위", 면적은 "PPI+CI"로 분리 보고.
   Top-k 정밀도만 보고하면 선택 편향으로 면적 추정이 무효.
3. **품질 산출물** — `cloud_stats.npz`(제주 4개년 픽셀별 구름 평균·최댓값)를 공개 자산으로
   유지. Earth Embeddings 서베이가 지적한 "임베딩 제품에 없는 품질 마스크"의 실체.

v7은 SCL(Scene Classification Layer)을 실제 장면 선택에 연결하고, 범주형 SCL은 nearest,
반사도는 bilinear로 읽는 대표-window smoke test를 통과했다. 다음 단계는 이 한 건을 일반화하지
않고, 연도·기존 bad proxy 층화로 사전 선택한 다중 window에서 효과 크기와 실패율을 추정하는
것이다. 그 검증을 통과하기 전에는 전체 216윈도우 재계산과 파트너 데모를 시작하지 않는다.

### 병행 트랙 — 제주 연안 서식지 감시 (MARC 연계 후보, 2026-08-21 착수)

- 목적: 남방큰돌고래를 위성으로 탐지하는 것이 아니라, 대정~모슬포 연안의 양식장·해안
  인프라·토지/수면 변화 같은 **서식지 압력의 공간적 맥락 후보**를 장기 현장조사와 함께
  검토할 수 있게 한다. MARC 협력·데이터 접근은 아직 성립하지 않은 후보 단계다.
- 상태: 제주 54윈도우 × 4개년 materialize와 v1 S2-only 임베딩 추출 완료.
  `change_4yr.py` 후처리는 4개년 각 54윈도우 로딩 후 CPU 계산 단계까지 확인.
- 다음: ① cloud/nodata mask ② 공통 중심화 vs 연도별 중심화 ablation
  ③ v1.2 paired 추출 ④ 변화 Top-k의 사람이 확인 가능한 근거 칩 생성
  ⑤ 어장정보도·인허가 오버레이 ⑥ 층화 확률표본+PPI 후 파트너 Evidence Pack.
- 수요 확인 원칙: 지도부터 피칭하지 않는다. 조사 우선순위/정책 의견/장기 모니터링 중
  **어떤 결정을 얼마나 자주 내리는지**, FP/FN 비용과 민감 위치 공개 범위를 먼저 묻는다.
- 금지 주장: 돌고래 개체·행동 직접 탐지, 관찰자료만으로 양식장/개발의 인과효과 주장,
  파트너 동의 전 기관명·데이터를 공개 산출물에 사용.

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

**2026-08-26 current-upstream 재감사**: 아래는 발견 계보라 번호를 지우지 않는다. 실제 제출 큐는
`PR_DOSSIER.md`가 SSOT다. 첫 sample schema PR은 current main에도 유효하고 미제출 상태다.
rslearn direct-materialize `NotImplementedError`는 v0.1.14에서 해소됐고, `미출시 API` 문제는
project lock(`rslearn 0.0.23 + olmoearth-pretrain 0.0.2`)과 current release의 skew로 재분류했다.
SCL 후보는 categorical-nearest 최소 PR과 auxiliary dependency RFC로 분리한다. partial-band mask는
public API end-to-end repro 전에는 연구 blocker이지 제출 가능한 PR이 아니다.

1. **`olmoearth-runner`의 `requires-python <3.12` 상한** — NGC PyTorch 컨테이너(py3.12)를 쓰는
   조직은 시스템 파이썬으로 설치 불가. `pip install olmoearth-runner` →
   "No matching distribution found". 상한을 풀거나 문서에 명시할 것을 제안할 가치.
   (우리는 uv로 py3.11 venv를 만들어 우회 — bootstrap.sh 참고)
2. **sample 프로젝트의 반쪽 스키마 마이그레이션 (확실한 버그)** —
   `olmoearth_run_data/sample/annotation_task_features.geojson`은 `oe_*` 접두사로
   마이그레이션됐는데 `annotation_features.geojson`은 legacy `es_*` + 스칼라 `es_label`인
   채로 남아 runner 0.1.14의 pydantic 검증(`oe_annotations_task_id`, dict형 `oe_labels`)에서
   실패. README대로 실행하면 바로 깨짐. **로컬 레포에 수정 적용 완료** (es_→oe_,
   `es_label: 1` → `oe_labels: {category: 1}`) — local commits `5e044ee`, `21b658a`; EOF newline·schema
   gate 통과. current upstream/open PR 중복 없음. Linux current-runtime replay 뒤 첫 PR로 제출 가능.
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
9. **rslearn SCL compositor의 숨은 보조-band/보간 계약** — `Sentinel2SCLBestClear`를
   반사도 레이어에 설정해도 `SCL`이 `band_sets`에 없으면 Planetary Computer 데이터소스가
   SCL 자산을 타일 저장소에 등록하지 않아 `missing scoring bands ['SCL']`로 실패한다.
   등록 후에도 compositor가 레이어의 bilinear resampling을 범주형 SCL 점수에 그대로 넘겨
   class ID equality를 왜곡할 수 있다. v7에서 SCL 보조 band set + nearest 점수 adapter로
   재현·해결. compositor가 SCL 의존성을 선언하고 categorical nearest를 강제하거나 문서화할
   공개 기여 후보 (`code/scl_compositor.py`, 세 실패 로그와 성공 로그 보존). current v0.1.14에서도
   `_score_item`이 layer resampling을 그대로 받는 것을 재확인했다. nearest scoring fix를 작은 PR로,
   보조-band dependency declaration은 별도 issue/RFC로 낸다.
10. **Sen12Landslides metadata/label issue 후보** — harmonized S2 13,628파일 전수 감사에서
    `center_lat/lon`이 13,628/13,628 모두 위경도 범위를 벗어난 projected coordinate였고,
    `hiroshima_s2_1427/1428`은 `annotated=True`·`ann_id=1964`인데 MASK 양성 픽셀이 0이었다.
    LanaoDelNorte 71패치도 inventory에는 annotation 후보가 있으나 harmonized S2 MASK는 전부 0이다.
    원본을 수정하지 않고 task cohort에서 fail-closed 처리했으며, 재현 manifest와 sample ID를 붙여
    upstream data issue로 보고할 가치가 있다.

## Worklog

### 2026-08-27 — Nepal OLMoEarth Live Twin GIS/WASM 공개 데모

- 계획: 사용자 지정 좌표 `28.2786794, 85.3780644`와 인접점
  `28.2828546, 85.3763336`을 Rasuwagadhi impact AOI로 지도에 고정하고, 원거리 좌표
  `27.8790412, 84.3103107`은 즉시 같은 유로로 가정하지 않고 거리·지명·하천 연결을 감사한다.
- 구현: 이미 물질화한 Sentinel-1/2 baseline과 향후 post-event 장면을 같은 georeferenced overlay
  계약으로 내보내는 Python compiler, 브라우저에서 직접 실행되는 Rust/WASM illustrative flow,
  OLMoEarth baseline/live-delta 상태와 acquisition timeline을 결합한 MapLibre GIS를 만든다.
- 주장 경계: WASM 흐름은 물리 예측이 아니라 UI/kinematic preview이며, `r.avaflow`·SFINCS 등 검증된
  물리 모델 산출물이 들어오기 전에는 hazard forecast로 표시하지 않는다. OLMoEarth 숫자도 실제
  embedding 산출물 전에는 invented similarity를 쓰지 않고 `baseline ready / post pending`으로 둔다.
- 검증·공개: mobile/desktop layout, Python manifest 재생성, Rust wasm build, frontend lint/build를
  통과한 정확한 소스 snapshot만 private site로 배포하고 URL과 재현 명령을 남긴다.
- 결과 — 좌표 감사: A는 Rasuwagadhi의 Pasang Lhamu Highway, B는 중국 Gyirong의 G216으로
  국경 양쪽 0.49 km 쌍이다. C는 Gandaki Province Tanahun의 Rishing-03이며 A에서 **113.79 km**라
  같은 사건 유로 endpoint에서 제외하고 별도 transfer/reference AOI로 표시했다.
- 결과 — 데이터/엔진: 로컬 물질화본에서 S1 4장·S2 4장을 256×256 georeferenced RGBA로 생성했고,
  5개 OLMo input anchor polygon을 별도 GeoJSON으로 냈다. OSM Bhote Koshi→Trishuli way
  `201928141→809865767→24624604`를 78점으로 봉인하고 Rust raw-WASM이 280입자를 그 경로에서
  브라우저 내 계산한다. Python은 build-time data plane이고 배포 runtime은 JS/WASM만 사용한다.
- 결과 — 주장 경계: UI와 manifest 모두 `embedding_status=not_run_in_this_web_snapshot`,
  `post_event_delta=blocked_until_post_scene`, `illustrative_kinematic_preview_not_hazard_forecast`를
  기계적으로 보존한다. 즉 OLMo 입력 준비와 임베딩 결과를 혼동하지 않고, 물리 모델 결과도 꾸미지 않는다.
- 검증/공개: scene 8·anchor 5·route 78·WASM particle 280 invariant, ESLint, `tsc --noEmit`,
  vinext production build 통과. 독립 site source commit `5b5a189…`, Sites version 1을 owner-only로
  배포했다: `https://olmoearth-nepal-live-twin.seeso.chatgpt.site`.
- 배포 감사/정정: v1 뒤 local preview 재감사에서 MapLibre가 번들러가 만든 상대 worker 경로를
  찾지 못할 수 있음을 발견했다. worker entry와 shared bundle(합계 508,167 B)을 self-hosted asset으로
  봉인하고 invariant에 크기 검사를 추가했다. ESLint·typecheck·production build를 다시 통과한 독립
  source commit `99bdddf…`를 Sites version 2로 저장해 같은 owner-only production URL에 재배포했다.
- 가시성 감사/정정: 사용자 스크린샷에서 v2의 중앙 지도가 사실상 검게 보이는 것을 확인했다. 초기
  카메라가 2.56 km 장면보다 넓은 corridor를 보면서 canvas brightness를 0.62로 낮췄고, 외부 OSM tile이
  비면 빈 배경이 지배하는 설계 결함이었다. 기본 카메라와 모든 ready-scene 전환을 장면 bounds에 자동
  fit하고 brightness 0.98, local background fallback, `FOCUS SATELLITE / RIVER CORRIDOR` 분리, 현재
  표시 장면 readout을 추가했다. pending 날짜도 지도를 지우지 않고 최신 baseline을 유지한다. 독립 source
  commit `563260e…`, Sites version 3으로 같은 owner-only production URL에 교체 배포했다.
- 다음: post-event S2/S1가 catalog에 생기면 Python compiler에 같은 AOI overlay를 추가하고,
  봉인한 OLMo runner로 baseline/post embedding과 Δ layer를 실제 계산한다. r.avaflow/SFINCS 결과가
  생기면 WASM preview를 대체하지 않고 별도 `physics_result` layer로 나란히 비교한다.

### 2026-08-27 — 확증 sweep 진행 기록

- **sweep 정의**: 동결 recipe(v2)로 미열람 8지역을 등록 순서대로 한 번씩 여는 확증 실험.
  지역당 3 arm(P4/P2/P3) × 3 seed = 9실행, pre gate → snapshot 봉인 → 봉인본 실행 → post gate → 판독.
- 완료: thrissur(강한 승리, +0.127, provenance 예외 공개), hiroshima(강한 승리, +0.062, 첫 완전 clean).
  중단 규칙(첫 3지역 2승) **2/2로 통과 확정**.
- 진행: hokkaido 3/9 (순차 runner, ~17:35 예상). 캐시 prefetch 7/9 지역 완료, 잔여 2지역 ~16:40 예상.
- indonesia부터 병렬 runner(seed당 3 arm 동시, 지역당 ~80분) → 8지역 완주 ~내일 01:40 예상.
- 일관 패턴: 오경보 격차(5~21배)가 주지표 격차보다 큼(2지역 연속). 지역별 이득 편차 2배(+0.127 vs +0.062).
- 위치 확인: sweep은 A~G 표의 A·B행 확증이지 논문 novelty가 아님(M61). 한국은 recipe 동결 후
  untouched 3-task로 개봉함(큐브 v2 재물질화 선행).


### 2026-08-25 — ASSET_INVENTORY × CRITICAL_PATH 재동기화와 CVPR 최소 실험 고정

- 계획: `docs/ASSET_INVENTORY.md`의 실측 보유자산을 authoritative `docs/CRITICAL_PATH.md`의
  단계 1~7에 재매핑한다. 특히 Sen12 수신 완료 이후에도 남은 “수신 없음/다운로드가 유일한 다음
  행동”을 제거하고, 진짜 병목을 G-P sample/label/input contract → smoke runtime → frozen probe로
  고정한다.
- 실험 설계: Italy 5,321패치를 곧바로 headline source로 쓰지 않고, 먼저 annotation-matched Höhn
  11지역의 region-held-out G-P에서 frozen OLMo의 task 자격을 판정한다. 입력·label·split·head·
  task-specific baseline·runtime budget·promotion/kill 기준을 실행 전에 봉인하고, 통과한 경우에만
  Italy→Korea(T-x), Höhn→Korea(T-m), `E_annotation`, static/live residual 순으로 연다.
- CVPR 감사: 최신 관련 benchmark/method와 비교해 “산악 데이터 결합”이 아니라 어떤 식별 가능한
  method contribution이 남는지, 필요한 region 수·baseline·통계·외부 한국 holdout·ablation을
  명시한다. G-P가 실패하거나 residual 이득이 +2%p/CI gate를 못 넘으면 method paper 주장을
  축소하고 benchmark/negative-transfer 결과로 전환한다.
- 결과 — 자산/임계경로 정합: Sen12 수신 없음·GPU 둘 다 점유라는 stale 상태를 제거했다. GPU0은
  다른 프로젝트가 62,585 MiB 사용 중이라 건드리지 않았고 GPU1은 0 MiB 가용이었다. Italy→Korea를
  headline에서 annotation-shift arm으로 내리고, 공개 task contract→G-P→T-m/T-x→static/live→
  R-cache의 promotion chain으로 `ASSET_INVENTORY`, `CRITICAL_PATH`, MountainShift 설계를 동기화했다.
- 결과 — C0 전수 계약: `code/build_sen12_gp_contract.py`로 13,628/13,628 NetCDF의 128×128·15시점·
  10밴드·SCL/MASK/DEM·시간순서·이진/시간불변 MASK가 통과했다. Hiroshima 2건의
  `annotated=True, MASK=0`을 원본 변경 없이 제외했다. annotation-matched 후보 11지역 중
  LanaoDelNorte 71패치는 양성 MASK=0이라 negative-only stress로 내리고, **task headline을 10지역
  6,834표본**으로 봉인했다. 전체 양성 6,737 중 단일 pre/post 유효는 5,397(**80.11%**)라 S≤t는
  계속 차단한다. sample contract SHA `dcdfef9a…`, anomaly SHA `bf086042…`.
- 결과 — G-P embedding smoke: 첫 15시점 forward가 OLMo v1 time embedding `12 != 15`로 실패했다.
  이를 숨기지 않고 SCL clear 상위 12개를 label-independent하게 고른 뒤 시간순 정렬하는 **S12q**로
  바꾸고 모든 baseline 입력을 맞췄다. 10지역·양/음성 32/32, 64표본/256crop이 15.44초,
  **4.146 sample/s**, peak CUDA 0.740 GB, fp16 cache 1,572,992 B/sample, shape `768×32×32`,
  replay max-abs diff 0으로 6/6 통과했다. headline 단순 외삽은 약 27.5분/10.75 GB다.
- 결과 — 연구/CVPR 재판정: 2026-08-10 GeoPhysAdapter가 frozen Prithvi + terrain/material/rainfall,
  event-isolated 55 event, pixel/object scale 및 misalignment 통제까지 이미 공개했다. 따라서 A3/A4의
  정확도 향상만으로는 CVPR novelty가 아니다. CVPR 후보를 task별
  `reuse/add_static/refresh_live/recompute_release`의 Δloss와 cost를 예측하는 **R-cache router**로
  좁히고 oracle regret·calibration·accuracy-cost Pareto, 2 backbone·3 task·한국 external holdout을
  통과 조건으로 고정했다.
- 검증/산출물: `artifacts/sen12_gp_contract/{summary,loco_folds}.json`,
  `artifacts/sen12_olmo_v1_smoke_64/summary.json` 회수. outer venv 전체 suite
  **153 passed, 1 skipped, 10 subtests**. system Python의 PyYAML 5개 collection failure는 환경
  차이이며 outer venv에서 전체 검증했다.
- 다음: smoke cache로 raw spectral/P4 head tensor 계약을 닫고, 한 10-region LOCO fold×1 seed에서
  P1/P2/P4 runtime·effect pilot을 먼저 잰다. 성능을 보기 전 full fold×seed 예산을 동결한다.

### 2026-08-25 — GK2A 운영·격자 감사와 국가 센서 축 정합성 재점검

- 계획: 최신 GK2A 수집 상태와 `--gaps`/`--status` 구현을 실제 산출물 기준으로 감사하고,
  `DAILY_OPS.md`의 2일 창·복구 불가능성·72파일/일 주장을 측정 범위에 맞게 축소한다.
  M15는 반복 관측 96개가 아니라 **고유 공간 anchor 4개**라는 점을 기준으로 LCC/offset 결론의
  과적합 여부를 재판정하고, 공식 행정동 코드 및 held-out spatial validation 전에는 좌표 변환을
  operational path에 넣지 않는다.
- 큰 그림 연결: GK2A는 강우 forcing이 아니라 시간 정합된 외부 observability residual로만 두고,
  admission/retrieval/segmentation에 더하는 조건부 이득을 frozen split에서 평가한다. 국가 센서 축은
  KOMPSAT-5↔Sentinel-1의 주파수·편파 계약과 CAS500-4의 spectral/GSD 계약을 공식 사양으로 다시
  확인해, 플랫폼-only 또는 pure-GSD 실험이라는 과장된 표현을 제거한다.
- 산출물: 기존 SSOT와 설계 문서만 갱신하고, 필요한 invariant/상태 검사를 추가한다. bulk 데이터나
  frozen split은 변경하지 않되 status가 D-1/D-2 gap을 찾으면 기존 운영계약 안에서만 보충한다.
- 결과 — 운영: `--status`의 credential 선검사를 제거하고 날짜 폴더 존재가 아니라 현행
  **57-slot terminal contract**를 검사하게 했다. 8월 23·24일은 각각 data 54 + NO_DATA 3으로
  57/57이며, 구 스케줄 extra 18개 때문에 보이던 “72파일/일”을 현행 분모에서 제거했다.
  정상 NO_DATA는 재시도하지 않고, 성공 manifest만 있고 파일이 사라지면 다시 받도록 고쳤다.
- 결과 — 수명 주장: “저장소의 유일한 비가역 작업/돈으로도 복구 불가”를 철회했다. D-1/D-2는
  data.go.kr 경량화 endpoint의 창이고, KMA API Hub에 별도 L2 download가 존재한다. archive와
  경량화 산출물의 parity는 미측정이라 매일 snapshot은 계속하되 정확한 endpoint provenance
  보존으로 의미를 제한했다.
- 결과 — M15: KMA가 KO/2 km 공식 lat/lon ASCII·NetCDF를 grid 저장순서대로 제공함을 확인해
  4-anchor LCC 역공학을 operational 경로에서 폐기했다. 기존 cache key에 `resultType`이 없어
  FOG가 CLD를 덮어쓸 수 있던 결함을 고쳤고, CLD-only 재실행도 0.8958이었지만 4개 공간점에
  fit/eval을 같이 했으므로 projection 확인 주장은 철회했다.
- 결과 — 큰 그림: GK2A는 한국 arm의 forcing/정확도 feature가 아니라 acquisition-time-matched
  observability residual이다. S2-only 대비 failure AUROC와 risk–coverage/AURC의 조건부 이득으로
  승격 여부를 판단하고, 네팔·스위스 공통 feature로 부르지 않는다.
- 결과 — 국가 센서: KOMPSAT-5는 X-band 9.66 GHz single-pol이고 Sentinel-1은 C-band 5.405 GHz
  VV+VH 계약이므로 platform-only 가설을 폐기했다. CAS500-4↔S2도 pure-GSD가 아니다. 한 release의
  sensor adapter → task utility → release×sensor 2×2 순서로만 연다.
- 검증: GK2A contract unit test 5개 통과(credential-free status, partial/NO_DATA/raw-SHA 포함).
  outer repo venv의 전체 suite **133 passed, 1 skipped**. 로컬 실데이터 status는 두 날짜 모두
  57/57 OK. system Python suite의 PyYAML import 5건 실패는 코드 회귀가 아니라 환경 차이였다.
- PR 판정: 새 Ai2 upstream defect는 발견하지 않았다. K5/CAS500 adapter는 제품 사양·접근·task
  utility를 먼저 증명해야 하는 연구 arm이지 현 시점의 upstream PR 후보가 아니다.

### 2026-08-25 — 큰 그림 복귀: 한국·네팔·스위스 Live Mountain Transfer를 임계경로로 승격

- 계획: C2-C exact-scene 복구를 논문의 임계경로에서 지원용 한국 ingestion gate로 내린다.
  FoldRefresh는 이미 확보한 릴리스/부분갱신 연산자로 두고, 새 중심 질문을 **글로벌 EO cache에
  지역 정적 residual과 시점·freshness가 있는 live residual을 더하면 한국·네팔·스위스에서
  산악 disturbance 검색·segmentation이 좋아지고, 봉인한 새 국가로 적은 라벨만으로 전이되는가**로
  고정한다.
- 이번에 고정할 것: ① FoldRefresh와 새 방법의 비중복 경계 ② 세 국가의 역사 label/static/live
  source 역할 ③ `E_static/E_live/E_transfer/E_refresh` 네 estimand ④ retrieval+segmentation
  leave-one-country-out 평가 ⑤ local-only/naive-pooled/GeoFM/adapter/live-residual baseline
  ⑥ 7일 promotion/kill gate. “실시간”은 위성 초단위 추론이 아니라 느린 EO cache 위에
  timestamped near-real-time residual을 갱신하는 것으로 제한한다.
- 무효 조건: 서로 다른 산악 현상을 같은 class로 합치거나, 미래 alert/사후 inventory를 과거
  예측 입력에 넣거나, 국가 ID/위경도 shortcut을 물리적 전이라고 부르거나, 한국 C2-C가 막혔다는
  이유로 네팔·스위스 public/live probe까지 멈추는 경우.
- 결과: `MOUNTAIN_EVIDENCE_TRANSFER.md`를 현재 authoritative track으로 승격하고 세 국가의
  historical/static/live/evidence 역할, dual-speed architecture, 7개 baseline, 성공·kill 기준,
  3-way leave-one-country-out을 고정했다. `README.md`, `RESEARCH_STRATEGY.md`,
  `K_ALIGN_BIG_PICTURE.md`, 본 SSOT의 루프 2와 프로그램 층위를 같은 순서로 동기화했다.
- task 경계: 세 나라 headline은 공통 `slope-failure` segmentation·retrieval만 사용한다. 한국 벌목,
  네팔 GLOF, 스위스 눈사태는 local auxiliary head이며 cross-country 동일 task라고 부르지 않는다.
- 입력 경계: primary transfer에는 한 OLMo release와 동일 canonical S2 10-band
  scale/GSD/time/missing-band 계약만 쓴다. v1↔v1.2는 FoldRefresh arm에서만 다시 열고, 국가별로
  다른 B01/B09 상수 채움은 금지한다.
- 외부 실행가능성 확인: 네팔 BIPAD alert/event/geohazard/streamflow API와 ICIMOD NepalLandslide
  OPeNDAP/WMS/HTTP catalog, 스위스 MeteoSwiss STAC와 SLF live measurement/bulletin/warning-region
  API가 존재한다. 아직 sample join과 snapshot 시간필드 audit은 안 했으므로 실시간 성능 주장은 0이다.
- 다음 7일: Sen12Landslides Nepal 20 + AI-Hub train/val 20 mapping 봉인 → Swiss 20 event 중 15건의
  geometry/cutoff/pre-post EO 연결 → frozen OLMo segmentation·prototype retrieval → static residual
  3-seed gate → cutoff-valid live replay. C2-C에는 최대 1일만 쓴다.
- 검증: 로컬 전체 suite `137 passed, 1 skipped, 10 subtests passed`; 문서 diff whitespace 검사 통과.

### 2026-08-24 (8차) — EarthKV 통합 판정 + 최신 경쟁·데이터 보정

- 계획: 첨부된 EarthKV 통합안을 현재 M1–M5, 최신 공개 embedding product, 산악 공개 benchmark,
  Ai2 채용 요구와 대조하고 과장된 전제를 고친다. 새 전략 문서는 만들지 않고 기존 SSOT에 통합한다.
- **판정**: EarthKV는 장기 프로그램 spine으로 유의미하다. 다만 첫 논문을 EarthKV 전체로 넓히지 않고
  `EarthEmbedContract`의 task-risk gate로 고정한다. FoldRefresh는 repair operator, EarthRoute는 후속
  policy, MountainShift는 Paper 2 평가 domain이다.
- **경쟁 보정**: AlphaEarth는 model/process/data version을, TESSERA convention은 dataset/model/build
  version과 cross-version 혼합 금지를 이미 제공한다. 따라서 `경쟁제품은 버전 의미가 없다`를 철회.
  Major TOM의 8필드 부재와 `unique_id` trap은 해당 제품 두 개의 실측으로만 유지한다.
- **논문 최대 위협**: M1의 identity R@1=0과 M3/M5의 dose curve는 representation proxy다.
  downstream task·고확신 오답·calibration을 측정하지 않았으므로 `silent error 감소`는 아직 주장할
  수 없다. 다음 임계 실험은 old-release frozen head + retrain upper bound + full re-embed/dual-index/
  simple version gate baseline이다.
- **MountainShift 보정**: 2026 AvalCD(4지역 bi-temporal avalanche)와 2025 Sen12Landslides(15지역,
  refined 74,956 events)가 있어 기관별 원자료 결합 전 public region-holdout을 먼저 할 수 있다.
  AI-Hub 50k의 네 해상도는 class ontology가 서로 다르고 Landsat에는 산사태 class가 없어,
  co-registration 감사 전 `동일 라벨 resolution ladder` 주장을 금지했다.
- **재현 확인**: parent `olmoearth_projects/.venv`로 전체 suite **128 tests OK, optional 1 skip**.
  system Python은 PyYAML 부재로 5 import error였으므로 재현 명령에 interpreter 계약이 중요하다.
- 세 목표 결론: 취업 신호는 강하지만 외부 PR/이슈가 아직 닫히지 않았고, 박사 질문은 살아 있으나
  task-risk 표가 필요하며, 비즈니스는 파트너 2곳 반복 수요 전까지 `Release Readiness Audit` 서비스형
  검증에 머문다.
- 다음: ① public downstream frozen-head table ② sample schema PR 또는 LFMC report 외부 전달
  ③ AvalCD/Sen12 20-sample transform/license gate. 그 전 VLM·federated·distributed EarthKV 금지.

### 2026-08-24 (7차) — **사전 등록 gate 실패. W2 일반 주장 철회**

야간 체인 완료. `replicates_across_releases = **False**`.
**어젯밤 강해 보였던 `진단 눈멂` 주장은 두 번째 릴리스에서 재현되지 않았다.**

| dose | 이동칸 | v1 CKA / R@1 | v1.2 CKA / R@1 |
|---|---:|---:|---:|
| 1 | 2 | 0.9923 / 0.9818 | **0.7928** / 0.7556 |
| 2 | 4 | 0.9873 / 0.9246 | **0.5274** / 0.1487 |
| 3 | 6 | 0.9797 / 0.6019 | **0.5193** / 0.0898 |
| 6 | 12 | **0.9720** / 0.2456 | **0.4172** / 0.0216 |
| reverse | 12 | **0.9595** / 0.1613 | **0.2749** / 0.0020 |

`blind_doses_by_release = {"v1": ["6", "reverse"], "v1_2": []}`

**v1.2에서는 CKA가 손상을 정확히 따라간다.** dose가 커질수록 CKA도 0.79 → 0.53 → 0.52 → 0.42 →
0.27로 같이 무너진다. 즉 눈멀지 않았다.

**타당성은 확인했다** — v1.2 dose 0도 frozen `embeddings_audit_v1_2_legacy`와
**byte-identical 8/8**이다. 데이터 문제가 아니라 실제 차이다.

#### 사전 규칙대로 처리한다

`analyze_contract_dose_response.py`에 미리 적어둔 규칙:
> 모든 dose에서 CKA와 R@1이 같이 무너지면 값싼 진단으로 충분하다는 뜻이므로 **W2 주장을 철회한다.**

이 규칙이 v1.2에 적용된다. 따라서:

- **철회**: `CKA는 계약 불일치에 눈멀었다` / `CKA의 순서가 운영 결과에 대해 역전된다`를
  **일반 주장으로 쓰지 않는다.** v1 한 릴리스에서 관측된 현상이다.
- **유지**: v1에서 관측된 역전 자체는 사실이고 재현 가능하다(무작위 잡음 30%: CKA 0.9505 / R@1
  1.0000 vs band-order dose 6: CKA 0.9720 / R@1 0.2456). 다만 **모델 의존적 현상**으로 기술한다.
- **유지**: 용량–반응 단조성은 두 릴리스 모두에서 성립한다. 계측기는 작동한다.
- **유지**: 취약성 기각(잡음 30%에서 오검색 0건)도 그대로다.

#### 대신 나온 것 — 다만 사전 등록하지 않은 탐색적 관측

**같은 계약 불일치에 대해 v1.2가 훨씬 취약하다.**

| dose | v1 R@1 | v1.2 R@1 | 배율 |
|---|---:|---:|---:|
| 2 (4칸) | 0.9246 | 0.1487 | **6.2×** |
| 3 (6칸) | 0.6019 | 0.0898 | 6.7× |
| 6 (12칸) | 0.2456 | 0.0216 | 11× |
| reverse | 0.1613 | 0.0020 | 80× |

운영적으로 읽으면: **파이프라인에 잠복한 밴드 순서 버그가 있을 때, v1→v1.2로 올리면 그 버그의
피해가 6~80배 커진다.** 이건 계약 축의 원래 이야기(릴리스 전환의 숨은 비용)와 오히려 더 잘 맞는다.

**그러나 이것을 오늘의 결과로 승격하지 않는다.**
- 릴리스 2개뿐이다. `비교`이지 `법칙`이 아니다.
- 사전 등록하지 않은 탐색적 관측이다. 별도 축·별도 릴리스에서 사전 등록 후 확인해야 한다.
- 원인 가설(예: v1.2 표현이 밴드에 더 특화돼 순열이 전역 구조까지 흔든다)은 아직 가설이다.
  effective rank·밴드별 ablation 민감도로 시험할 수 있다.

#### 남은 위협

1. downstream task 미측정(어제와 동일).
2. 제주 8 site-years·축 1개. 모집단 추정이 아니다.
3. 새 관측(v1.2 취약성)은 사전 등록되지 않았다.

**교훈**: 어젯밤 `사전 등록한 주장이 통과했다`고 보고했는데, 하루 만에 그 주장의 일반성이
자기 규칙에 의해 기각됐다. 사전 등록과 복제 gate를 걸어두지 않았으면 v1 결과만 들고
논문을 썼을 것이다. **L4·L6이 실제로 작동한 사례로 남긴다.**

### 2026-08-24 (6차) — v1 dose-response 결과 + 반증 대조군 착수

**사전 등록한 W2 주장이 v1에서 통과했다.** 단, 아직 해석을 확정하면 안 되는 이유가 있어
반증 대조군을 바로 걸었다.

| dose | 이동칸 | same-token cos | linear CKA | dist Spearman | **R@1** |
|---|---:|---:|---:|---:|---:|
| 1 | 2 | +0.9643 | 0.9923 | +0.9617 | 0.9818 |
| 2 | 4 | +0.9584 | 0.9873 | +0.9275 | 0.9246 |
| 3 | 6 | +0.9430 | 0.9797 | +0.8925 | 0.6019 |
| 6 | 12 | +0.9145 | **0.9720** | +0.8318 | **0.2456** ← BLIND |
| reverse | 12 | +0.8628 | **0.9595** | +0.7743 | **0.1613** ← BLIND |

- **용량–반응 단조성 확인.** R@1 0.98 → 0.92 → 0.60 → 0.25 → 0.16. 계측기가 작동한다.
- **감도 비대칭.** 전 구간에서 CKA는 0.9923→0.9595로 **3.3%p**만 움직이는데 R@1은
  0.9818→0.1613으로 **82%p** 무너진다. 약 **25배** 차이다. 코사인도 눈멀었다 —
  reverse에서 same-token cosine 0.8628인데 R@1은 0.1613이다.
- **대조군 작동.** dose 6과 reverse는 이동 칸수가 **똑같이 12**인데 reverse가 더 파괴적이다
  (R@1 0.2456 vs 0.1613, CKA 0.9720 vs 0.9595). **이동 개수가 아니라 이동 거리가 중요하다.**
  즉 displaced_positions 하나로 손상을 예측할 수 없다.
- **기존 릴리스 발견과 같은 서명.** v1→v1.2는 CKA 0.978 / R@1 0.000이었고, 지금 통제된
  band-order dose 6은 CKA 0.972 / R@1 0.246이다. **높은 CKA + 무너진 검색**이라는 같은 패턴을
  이제 원하는 심각도로 생성할 수 있다.

#### 아직 유의미하다고 말하면 안 되는 이유 셋 (사용자 질문에 대한 정직한 답)

1. **task가 없다.** 여기 R@1은 `이 토큰이 자기 자신을 찾는가`이며 표현 프록시다. 자기검색이
   75% 깨질 때 실제 downstream task가 망가지는지는 **측정하지 않았다.**
2. **R@1이 구조적으로 취약한 지표일 수 있다.** 한 window의 토큰은 공간적으로 인접해 서로 거의
   같다. 작은 섭동만으로 최근접이 옆 토큰으로 넘어가면, R@1 붕괴는 `표현이 깨졌다`가 아니라
   `지표가 원래 잘 깨진다`가 된다. **이것이 이 발견의 가장 큰 위협이다.**
3. **CKA의 둔감성은 부분적으로 알려진 성질이다.** CKA는 구조 유사도, R@1은 좌표 identity라
   원래 다르게 움직인다. `당연한 것 아니냐`는 반론이 가능하다.

#### 위 2번을 정면으로 시험하는 반증 대조군 — **완료. 위협 2 기각됨**

`code/dose_brittleness_control.py` — **GPU를 쓰지 않는다.** dose 0 임베딩에만 작업한다.

- dose 0에 크기를 아는 Gaussian 잡음(0 / 0.1 / 0.3 / 1 / 3 / 10 / 30%)을 넣고 같은 곡선을 그린다.
- **사전 판정 규칙**: dose 6의 R@1(0.2456)과 같은 수준을 만드는 최소 잡음이 **1% 이하**이면
  R@1은 구조적으로 취약한 지표이며, dose 실험의 R@1 붕괴를 `표현이 깨졌다`로 해석하면 안 된다.
- 추가로 **틀린 이웃의 공간 거리**를 잰다. `fraction_misses_adjacent`가 높으면(>0.5) 틀린 이웃이
  대부분 바로 옆 토큰이라는 뜻이므로 좌표계 붕괴가 아니라 **국소 혼동**이다.

**결과 (`artifacts/results/dose_brittleness_control.json`)**

| 잡음 | same-token cos | linear CKA | **R@1** | 틀린 이웃 거리 |
|---:|---:|---:|---:|---:|
| 0.0 | +1.0000 | 1.0000 | 1.0000 | (오검색 0건) |
| 0.01 | +0.9999 | 1.0000 | 1.0000 | (오검색 0건) |
| 0.03 | +0.9995 | 0.9997 | 1.0000 | (오검색 0건) |
| 0.1 | +0.9950 | 0.9967 | 1.0000 | (오검색 0건) |
| **0.3** | **+0.9577** | **0.9505** | **1.0000** | **(오검색 0건)** |

`smallest_noise_matching_dose6 = None` — **시험한 어떤 잡음도 dose 6의 R@1(0.2456)을 재현하지
못했다.** 30% 잡음에서도 자기검색이 4,096 토큰 전부 정확했다(오검색 0건이라 거리 통계는 nan).

**위협 2는 기각됐다.** 그리고 부수적으로 `토큰이 공간적으로 인접해 거의 같다`는 내 우려도
틀렸음이 확인됐다 — 30% 잡음에도 오검색이 한 건도 없다는 것은 토큰이 임베딩 공간에서
충분히 분리돼 있다는 뜻이다.

#### 그런데 이 대조군이 발견을 **훨씬 강하게** 만들었다

| | cos | **CKA** | **R@1** |
|---|---:|---:|---:|
| 무작위 잡음 0.3 | +0.9577 | **0.9505** | **1.0000** |
| band-order dose 6 | +0.9145 | **0.9720** | **0.2456** |

**CKA는 순서를 거꾸로 매긴다.** 검색이 완벽한 무작위 잡음(R@1 1.000)을 CKA 0.9505로,
검색의 75%가 깨진 계약 불일치(R@1 0.2456)를 CKA 0.9720으로 평가한다.
즉 CKA 기준으로는 **무해한 잡음이 치명적 불일치보다 더 달라 보인다.**

따라서 주장이 `CKA는 덜 민감하다`(이건 알려진 성질이라 약했다)에서
**`CKA의 순서가 운영 결과에 대해 역전돼 있다`**로 바뀐다. 이건 알려진 성질이 아니다.
위협 3(“CKA 둔감성은 당연하다”)도 함께 약해졌다.

**남은 위협은 1번 하나다 — downstream task를 아직 측정하지 않았다.**

#### 사용자 질문: 스위스·네팔 데이터를 썼는가 — **아니다**

이번 실험 입력은 **제주 8 site-years Sentinel-2 영상뿐**이다.
GLAMOS·swissALTI3D **0건**, ICIMOD/HKH **0건**, 한국 공공데이터(BuildingHUB·EIA·PNU·GK2A) **0건**.
MountainShift는 Phase 0 다운로드·라이선스 확인조차 시작하지 않았다(우선순위 3번, 미착수).
이 실험은 **계약 축**이라 지역 데이터가 필요하지 않았다. 두 축을 혼동하지 않는다.

### 2026-08-24 (5차) — 첫 측정 착지: Major TOM 계약 감사 완료

**나흘간의 계획 끝에 처음으로 실제 수치가 나왔다.**

- 실행: `code/audit_majortom_contract.py`를 서버 CPU에서 실행(GPU 미사용).
  `.venv-master`(재현성 계약)를 건드리지 않으려고 별도 `.venv-data`를 uv로 만들고
  pyarrow 25.0.1 + huggingface_hub 1.28.0만 설치했다.
- **Q1 판정: PAIRED — 단, 조인 키가 `unique_id`가 아니다.**

  | 조인 키 | 1:1 | 교집합 |
  |---|---|---:|
  | `unique_id` | **False** | **0** |
  | `grid_cell + product_id` | True | **248,719** |
  | `grid_cell` | True | 248,719 |

  두 데이터셋 모두 248,719행, **동일 스키마 15개 컬럼**, 첫 3행의 `grid_cell`·`product_id`가
  정확히 일치하며 행 순서까지 정렬돼 있다. 그런데 **`unique_id`는 교집합이 0**이다.
  값을 보면 데이터셋별 64자 content hash다(예: olmo `ba84ab36…` vs clay `91c8ee89…`,
  같은 `grid=921D_252L`·같은 `prod=S2B_MSIL2A_20221115T161819_N0400_R111_T01CDJ`).
- **이것 자체가 우리 논문의 명제다.** `unique_id`라는 이름과 동일한 스키마 위치를 가진 컬럼이
  두 제품에 다 있는데 공유 식별자가 아니다. 조인 키로 쓰면 조용히 빈 결과가 나온다.
  스크립트에 `unique_id_is_a_trap` 필드로 박아뒀다.
- **Q2 판정: 8개 계약 필드가 전부 기계가 읽을 수 있는 스키마에 없다.**
  `model_weights_hash`, `acquisition_dates`, `temporal_recipe`, `band_order`,
  `normalization`, `pooling`, `input_content_hash`, `output_content_hash` — **8/8 부재.**
  실제로는 두 제품이 차원(768 vs 1024), pooling(unmasked token 평균 vs CLS),
  밴드(12 vs 10), 정규화(OlmoEarth normalizer vs Clay mean/std)에서 다른데,
  그 차이는 데이터셋 카드의 **산문에만** 있다.
  → `geo-embeddings/embeddings-stac-specification` v0.0.1 gap 분석(W9)의 **1차 증거**가 확보됐다.
- 산출물: `artifacts/results/majortom_contract_audit.json` (로컬·서버 양쪽),
  원본 parquet 캐시는 서버 `artifacts/external_data/majortom_cache`.
- **파생 결론**: paired 전제가 살아났으므로 gallery-size 곡선을 `10³–10⁵` 구간에서 **실측**으로
  그릴 수 있다(지금까지 216 site-years에서 외삽하던 구간). 단 앞서 적은 불가능 4가지
  (release pair 아님·모델 우열 비교 불가·token 수준 분석 불가·라벨 없음)는 그대로다.
- 약점: 249k 두 개만 확인했다. SatCLIP·SigLIP·DINOv2·FarSLIP·MMEarth·AlphaEarth·UniverSat은
  미확인이다. 또한 `grid_cell` 단독 조인이 이 subset에서 우연히 1:1이지만, Major TOM Core 전체에서는
  한 grid cell에 여러 product가 있을 수 있으므로 **`grid_cell` 단독을 일반 조인 키로 쓰면 안 된다.**

### 2026-08-24 (5차, 병행) — GPU1 dose-response 착수

- GPU0은 다른 프로젝트가 점유(62 GiB, 99%) → 사용자 지시대로 **GPU1**에서 실행.
- `code/contract_dose_response.py` 신규. 축은 **밴드 순서**다. `olmo_release_v1_legacy.yaml`이
  밴드 순서를 **두 곳**에 선언한다는 점을 이용한다 — `data.inputs.sentinel2_l2a.bands`와
  정규화기 `OlmoEarthNormalize.band_names`. **입력 쪽만 k쌍 치환하고 정규화기는 그대로 둔다.**
  그러면 정규화 통계가 다른 밴드에 붙는다. 파일도 정상, 실행도 성공, 차원도 동일 — **조용히 틀린다.**
- dose: 0(0개 이동) / 1(2개) / 2(4개) / 3(6개) / 6(12개) / reverse(12개).
  **dose 6과 reverse는 이동 수가 같고 치환이 다르므로 좋은 대조군**이다.
- 대상은 이미 공개된 disclosed-audit smoke 8 site-years다. sealed 64는 건드리지 않는다.
  원본 raster를 재사용하므로 재다운로드·재materialize가 없다.
- 마찰: rslearn은 dataset `config.json`에 선언되지 않은 `output_layer`에 쓰지 않는다
  (`KeyError: Output layer 'embeddings_dose_0' not found`). 스크립트에 `register_layers()`를 추가해
  기존 `embeddings_audit_v1_legacy` 스펙을 템플릿으로 dose 레이어 6개만 추가하고,
  변경 전 config를 백업·해시 기록했다. 기존 레이어와 frozen 입력은 수정하지 않았다.
- 분석기 `code/analyze_contract_dose_response.py`도 작성했다. dose 0 대비
  same-token cosine·linear CKA·pairwise-distance Spearman·**R@1(dose→base)**을 낸다.
  **핵심 판정 지표는 `cka_stays_high_while_recall_collapses`** — CKA ≥ 0.90인데 R@1 ≤ 0.50인
  dose가 있으면, 표현 유사도 지표가 계약 불일치에 눈멀었다는 직접 증거다.
  모든 dose에서 CKA와 R@1이 같이 무너지면 **W2 주장을 철회한다**(사전 기록).
- **타당성 검사 통과 (중요)**: dose 0은 기존 frozen `embeddings_audit_v1_legacy`와 설정이
  출력 레이어 이름만 다르므로 동일해야 한다. 8 site-years 전부 **byte-identical 8/8**
  (244,294,379 / 244,565,065 / 244,442,973 / 244,824,942 / 244,771,762 / 244,719,531 /
  244,426,449 / 244,326,009 bytes). 두 가지가 확인됐다 — ① **하네스가 교란을 넣지 않으므로
  dose ≥1의 차이는 전부 밴드 순서 불일치에 귀속된다** ② 이 설정의 추론은 결정론적이다.
  이 검사를 안 했으면 dose 곡선의 원인을 특정할 수 없었다.
- 실행 시간: dose당 약 191초(0=191.562s, 1=194.351s, 2=190.604s, 3=190.99s, 6=191.876s).

### 2026-08-24 (5차, 야간) — 무인 체인 가동

사용자가 취침하므로 판단이 필요 없는 작업만 순차 실행하도록 `code/overnight_contract_chain.sh`를
만들어 GPU1에서 백그라운드로 걸었다.

| 단계 | 내용 |
|---|---|
| 1 | v1 dose run 완료 대기 (최대 60분) |
| 2 | v1 분석 → `artifacts/results/contract_dose_v1_analysis.json` |
| 3 | **v1.2 dose run** — 같은 8 site-years·같은 밴드 순서 축 |
| 4 | v1.2 분석 → `artifacts/results/contract_dose_v12_analysis.json` |
| 5 | `OVERNIGHT_COMPLETE.json` — 두 릴리스 요약과 `replicates_across_releases` 판정 |

**3단계를 넣은 이유**: 진단 눈멂이 v1 한 릴리스에서만 나타나면 그 릴리스의 특성일 수 있어
일반 주장을 할 수 없다. **두 번째 릴리스에서 재현되어야 계약 축 주장이 선다.**
`replicates_across_releases` 필드로 사전 판정하도록 marker 생성기에 박아뒀다.

안전장치: 단계별 `set -euo pipefail`, 실패 시 즉시 중단 + `FAILED.json` marker,
dose 스크립트 자체가 선택 GPU에 다른 프로세스가 있으면 거부. GPU0의 타 프로젝트 작업은
건드리지 않는다. 분석기 의존성(numpy 2.4.6 / rasterio 1.4.4)은 사전 확인했다.
`--model-env`를 파라미터화해 v1.2의 `OLMO_V1_2_MODEL_PATH`를 쓸 수 있게 스크립트를 수정했다.

### 2026-08-24 (4차) — MountainShift 검토 + 다섯 방향 통합 우선순위

> **후속 보정(8차):** 아래는 당시의 의사결정 기록이다. 공개 Phase 0의 현재 1순위는
> Glacial-Lake-Bench/Landslide4Sense가 아니라 AvalCD/Sen12Landslides이며, 현재 실행 queue는
> 이 파일 상단 `교수 판정`과 8차 Worklog를 따른다.

- 판정: **좋은 방향이다. 다만 아직 세지 않은 비용이 하나 있고, 범위가 지금까지 중 가장 크다.**
- 좋은 점 셋: ① **ETH 정합성** — GLAMOS는 ETH 공동운영이고, 다섯 방향 중 지원 목표와 자산이
  겹치는 유일한 방향이다. 경력 논거이지 과학 논거가 아니지만 **진짜 논거이므로 그렇게 부르고 쓴다.**
  ② **Phase 0이 라벨 병목을 통과한다** — Glacial-Lake-Bench(19,115 pairs, leave-one-region-out 내장)와
  Landslide4Sense(3,799 patches)는 다운로드로 끝난다. 한국 트랙은 라벨 1,200 + NGII 승인이 필요했다.
  **실제 transfer 수치에 가장 빨리 닿는 경로다.** ③ dual-speed(`z_global/z_region/r_t`)가 K-ALIGN의
  stable/residual 분리와 같은 구조라 새로 만드는 게 아니다.
- **아무도 안 센 비용 — MountainShift는 우리의 가장 희소한 자산을 버린다.**
  한국의 희소 자산은 산악이 아니라 **필지에 결속된 시점 있는 행정기록**(BuildingHUB 8,794 event행 ×
  PNU × 허가·착공·사용승인 일자)이다. **산사태·산불에는 건축 인허가가 없다.** MountainShift에서
  한국은 "또 하나의 산사태 지역"이 되고 cell D·coverage 편향·`published_time` 계측기가 대부분
  적용되지 않는다. 즉 **희소 자산을 흔한 도메인 전이 연구와 맞바꾸는 것**이며, 그 교환 사실이
  문서 어디에도 없었다.
  **완화책**: Track B/C에서 한국을 "산사태 지역"이 아니라 **"행정근거가 있는 유일한 지역"**으로 둔다.
  알프스·HKH에는 GLAMOS/ARPA/ICIMOD inventory가 있지만 **필지 단위 인허가 시각은 없다.**
- 비용 2 — 범위: GLAMOS·swissALTI3D·ARPA SIFraP(약 36,000건)·눈사태 portal·ICIMOD RDS·산림청
  3종이 새로 붙고 각각 라이선스·다운로드·시간정렬 확인이 필요하다. 오늘 **GPU 0장**, 마감 10주.
- 비용 3 — "다지역이 낫다"는 부분적으로 이미 알려진 답이다(PANGAEA 저라벨 GFM 우세, AnySat 다센서
  전이). headline을 거기 두면 약하다. 새로운 것은 **worst-region 비악화·API 누락 시 보류·location
  shortcut 배제**이며 문서의 사전 성공 기준은 이미 그렇게 적혀 있다. **제목만 그쪽으로 옮기면 된다.**
- **반대로 하나는 MountainShift에서 더 강해진다**: 제주에서 6개월 창 offset은 구름·식생 차이였지만
  **알프스에서는 적설 유무를 통째로 뒤집는다.** W1 dose–response를 산악에서 돌리면 인공물이 제주보다
  훨씬 크고 육안으로 명백하다. **계약 축은 버려지는 게 아니라 증폭된다.** 두 방향의 합류점이다.
- **통합 우선순위 확정** (원칙: 결정력×저렴함 ÷ 새 의존성. 하드룰: **측정 하나가 착지하기 전에는
  새 방향을 열지 않는다**). 각 작업이 몇 방향에 동시에 쓰이는지 세어보면 **W1 dose–response만이
  다섯 칸(compat·계약·한국·VLM·Mountain) 전부에 들어간다.**
  1. **Major TOM 계약 감사** — GPU 0장인 오늘 가능한 유일한 결정적 측정. `code/audit_majortom_contract.py` 작성 완료
  2. **W1 dose–response 최소판**(밴드 순서·정규화) — GPU 나면 즉시
  3. **Phase 0 다운로드·라이선스 확인** — MountainShift 전체의 gate. 하루. GPU 불필요
  4. **NGII 신청 + 한국 event universe** — 승인 대기가 길어 지금 안 넣으면 다음 주기도 못 엶
  5. frozen probe(제주 임베딩에서 water/snow/debris) 6. ADC baseline 7. 14일 방법 P0
  8. MountainShift Phase 1(3 통과 시). **보류**: VLM 트랙(하루 5-way 사전검사만), 5-cell 다지역 격자
- **순서 검증**: 1·2·3이 전부 음성이면? Major TOM이 paired가 아니면 규모 축을 잃되 제주 216 token
  축은 남고, dose–response 무반응이면 **계약 축 전체가 죽고 MountainShift가 주 방향이 되며**,
  Phase 0이 막히면 MountainShift가 죽고 계약·한국 축이 남는다. **셋이 동시에 죽을 확률은 낮고 어느
  하나가 죽어도 나머지가 산다.** 반대로 5-cell 격자부터 시작하면 GPU를 다 쓰고도 세 축 중 어느
  것도 판정하지 못한다.
- 서버 상태(확인): `h200-dev` RUNNING이나 **GPU 0·1 둘 다 사용 중**(각 68 GiB, util 54%/53%,
  다른 프로젝트 PID 2개). 규약상 GPU0 전용 실행 불가 → 오늘은 CPU 작업만.
- 약점: 위 판정은 전부 설계 검토이고 측정이 아니다. Glacial-Lake-Bench는 2026 ESSD **preprint**이며
  실제 다운로드·라이선스를 확인하지 않았다. GLAMOS/ARPA/ICIMOD도 마찬가지다.
- 다음: 1번(Major TOM 감사) 실행 → 2·3·4 병행.

### 2026-08-24 (4차) — 지역 embedding + 실시간 API + 다지역 공동학습

- 계획: 산악 자연보존 문제에서 `천천히 바뀌는 지역 표현`과 `빠르게 바뀌는 API evidence`를
  분리하고, 단일지역 모델·naive pooled 모델·공유 backbone+지역 adapter·동적 residual 모델을 같은
  split/label/compute에서 비교하는 최소 계약을 만든다. 기존 K-ALIGN dual-speed 설계와 중복되는
  부분은 새 시스템으로 만들지 않고 산악 use case로 구체화한다.
- 사전 판정 기준: API의 미래/사후 record를 과거 예측 입력에 넣지 않는다. pooled 평균이 좋아도
  worst-region 또는 unseen-region이 나빠지면 `여러 데이터가 더 좋다`고 선언하지 않는다. 지역 ID나
  위경도 shortcut만으로 이득이 재현되면 물리적 지역특성 전이로 세지 않는다.
- 설계 결론: `z_global`(S1/S2) + `z_region`(DEM·slope·지질·기후평년 adapter) +
  `r_t`(관측시각·freshness·missingness가 있는 실시간 API residual)의 dual-speed 구조로 닫는다.
  공공 API 갱신은 residual만 refresh하고 stable gallery는 유지한다. 사후 확정자료는 label/evidence이며
  과거 inference input에 넣지 않는다.
- 비교 계약: local-only / naive pooled / shared backbone+local head / shared+local adapter /
  adapter+timestamped residual 5칸. label 1/5/10/50/100%, unseen-region, future-year,
  API missing/stale에서 macro와 worst-region을 함께 본다. `E_repr`과 `E_fusion`은 분리한다.
- 문헌 경계: AnySat(CVPR 2025)은 이종 5 datasets·11 sensors 공동학습 가능성을 보였지만,
  PANGAEA는 GeoFM이 supervised baseline을 항상 이기지 않음을 보였다. 따라서 naive pooling이 아니라
  조건부 공유/지역 보존과 negative-transfer 감사가 기여다.
- 반영: `MOUNTAIN_EVIDENCE_TRANSFER.md`에 dual-speed 수식, 데이터 속도표, 5-model matrix,
  promotion/kill gate를 추가했다. GPU/데이터 실행은 하지 않았고 실제 개선은 미검증이다.

### 2026-08-24 (3차) — VLM 방향 판정: 구조는 맞고 과녁이 틀렸다

- 질문: `계약 gate + VLM + 한국 공공근거 → REUSE/ADAPT/RECOMPUTE/ABSTAIN` 구조와
  "Earth VLM은 두 EO 임베딩을 비교하면 안 되는 순간을 알 수 있는가?"라는 논문 질문에 대한 평가.
- **판정 1 — 제목 질문은 "아니오, 그리고 알 필요도 없다"로 답해질 것이다.**
  계약 불일치는 시각적 사실이 아니라 **메타데이터 사실**이다. mean pooling vs CLS, 밴드 12 vs 10,
  가중치 v1 vs v1.2 — **어느 것도 픽셀에 없어서 VLM이 원리적으로 볼 수 없다.** 반대로 메타데이터가
  있으면 10줄 결정론적 검사가 100%로 답한다. 따라서 5-way ablation에서 `계약 gate만`이
  `gate+VLM`을 이길 가능성이 높고, 논문이 자기 제목에 "아니오"라고 답하게 된다.
  (예외: 계약 필드가 없는 아카이브에서 *결과*를 보고 역추정하는 것은 가능하나, 그건 계약 탐지가
  아니라 결과 이상 탐지이고 통계 검정이 더 잘한다.)
- **판정 2 — 제안 안에 강한 논문이 이미 있다: 역할 2번.** "영상 변화는 있지만 공식 근거는 없음"
  으로 구조화하고 근거 없으면 원인을 지어내지 않고 보류 — 이름을 붙이면
  **행정근거 결손 아래의 원인 환각(fabricated cause attribution)**이다.
- **선행 조사(신규)**: 인접 영역은 붐빈다 — GeoChat(CVPR 2024)·EarthDial(CVPR 2025)이 EO 대화·
  grounding을, GEOBench-VLM(ICCV 2025, 최고 MCQ 41.7%)이 벤치마크를, **RSHallu·DDFAV/RSPOPE·
  UHR-Micro·VLRS-Bench·CHOICE가 RS VLM 환각 벤치마크**를, ChangeVLM·VLM-BCD·ViLaCD-R1·
  Decoding the Delta가 VLM 변화탐지를, "Knowing When Not to Answer"가 다중모달 abstention을 점유.
  **그러나 기존 RS 환각 벤치마크는 전부 객체 존재(POPE 계열)를 검사하고, 변화의 원인 주장을
  외부 기록으로 검증하는 것은 검색에 나오지 않았다.** 이유는 명확하다 — 시점이 찍힌 행정기록을
  가진 팀이 거의 없다. 한국은 있다.
- **바꿔야 할 것은 구조가 아니라 과녁.** 아키텍처(규칙 먼저 → VLM → 공공근거)는 옳다.
  역할만 정확히 나눈다: 계약 gate = 이 비교가 **유효한가**(메타데이터, 100%) /
  VLM = 유효한 비교에서 **진짜 변화인가**(지각, 불안정) /
  **한국 공공근거 = VLM의 원인 설명이 사실인가(여기가 기여).**
  새 제목 질문: **"시점이 찍힌 행정근거는 Earth VLM의 허위 원인 설명을 줄이는가? 그리고 근거
  coverage가 비어 있는 곳에서 환각은 누구에게 집중되는가?"**
  장점: ① 계약 gate가 버려지지 않고 **상류 필터**가 된다(184일 중첩 5건·4기간 5건이 그 실증)
  ② **cell D와 직결** — 변화는 있는데 기록이 없는 칸이 VLM이 원인을 지어낼 바로 그 칸이고 정답은
  보류다 ③ **coverage 편향이 결과가 된다** — 개발행위허가 제주 2023·2024 0행을 이용해 "근거가
  없는 해·지역에서 환각이 증가하는가"를 측정 ④ 41.7%짜리 teacher로 증류하지 않는다(역할 3 계속 보류).
- **하루짜리 사전 검사를 먼저 한다.** 새 데이터 없이 이미 있는 약 20~29건(14 candidates +
  v5 blind pair 5쌍 + 184일 중첩 노출 5건 + 4기간 source 5건)으로 5-way를 돌린다.
  **사전 판정 규칙: `계약 gate만`이 `gate+VLM`과 잘못된 REUSE 비율에서 같으면 "VLM이 계약
  불일치를 안다"는 주장을 버리고 원인 환각 축으로만 간다.** 이 표본은 사람 판독·구조 결함 분류가
  끝나 있어 정답이 있고, GPU도 새 라벨도 필요 없다.
- **정직한 지적**: 이건 4일 만의 네 번째 방향이다(compat/method → 한국 event-first → wide-angle
  계측기 → VLM). 아이디어는 매번 좋아지고 있으나 마감 10주·1인·14일 P0는 이미 방법 트랙에 배정됐다.
  위 하루 검사는 GPU가 필요 없어 P0와 경쟁하지 않으므로 **그것만 먼저 하고, 어느 쪽 결과든
  D1–14는 건드리지 않는다.**
- 다음: ① 하루 5-way 사전 검사 ② 결과에 따라 VLM 범위 확정(원인 환각 축으로 축소 또는 승격)
  ③ 기존 14일 P0와 한국 데이터 병행 P0는 예정대로.

### 2026-08-24 (3차) — 알프스·Monviso·HKH·한국 산악 전이 연구 검토

- 계획: ETH권 알프스, Ostana/Monviso, ICIMOD의 Hindu Kush Himalaya(HKH), 한국 산악
  공공데이터에서 실제 확보 가능한 공식 산출물을 확인하고, 직접 전이하면 안 되는 현상과 공통으로
  전이 가능한 산악 변화 primitive를 분리한다. 기존 Earth embedding의 보정 가능 범위도
  `adapter로 가능한 것 / 원영상 재임베딩이 필요한 것`으로 나눠 최소 실험을 설계한다.
- 사전 판정 기준: 한국에 없는 빙하 현상을 한국 target label로 포장하지 않는다. 지역별 공간해상도·
  관측주기·발행시점이 다른 자료는 같은 정답표에 직접 합치지 않는다. 최소 3개 지역의 source→target
  전이와 scratch/frozen/adapter 대조군, negative transfer, 근거 부족 시 보류를 측정할 수 있을 때만
  연구 트랙으로 남긴다.
- 공식 자산 확인: ETH 공동운영 GLAMOS는 glacier inventory·길이·질량·부피 자료를 다운로드하고,
  swisstopo는 0.5/2 m DEM을 제공한다. ARPA Piemonte는 약 36,000 산사태 SIFraP, Monviso 빙하
  현장조사·눈사태·암벽붕괴 사진측량을 제공한다. ICIMOD RDS는 1990–2020 빙하 변화, 1533–2025
  GLOF 766건, 2000–2022 토지피복을 노출한다. 한국은 10 m 산사태 위험지도·발령, 산불 이력·위험,
  기존 기상/토지피복/항공/행정근거를 연결할 수 있다. 개별 license·다운로드는 아직 미감사다.
- 설계 판정: 빙하 track은 HKH↔Swiss Alps↔Monviso, 전 지역 공통 track은 water/snow-ice/
  bare-debris/vegetation-loss/slope-failure로 분리한다. 원인은 local evidence head가 맡고, 한국을
  빙하 target으로 만들지 않는다.
- 보정 사다리: frozen embedding probe → DEM/공공근거 residual adapter → train-time privileged
  distillation → 정보가 없으면 S1/S2/DEM 재임베딩. adapter가 fused embedding에서 소실된 월·공간
  정보를 복원한다는 주장은 금지한다.
- 문서화: `MOUNTAIN_EVIDENCE_TRANSFER.md` 신규, `RESEARCH_STRATEGY.md` RQ9,
  `PAPER_READING_LIST.md` 산악·cryosphere 4편을 추가했다. 실제 성능은 미검증이다.

### 2026-08-24 (2차) — "한국식은 포기한 건가"에 대한 판정 + event-first 설계 보정

- 질문: 축적된 결정들(CVPR main에서 한국 트랙 분리, 필지 경계 제외, 한국판 FLAIR-HUB 포기,
  label-free core 선행, B1 임계경로 제외)이 사실상 한국 트랙 포기가 아닌가.
- **판정: 포기 아님. 그러나 지금 상태로 두면 포기가 된다.**
  - 빠진 셋(필지 경계·한국판 FLAIR-HUB·정확도 headline)은 전부 **이미 남이 차지했거나
    순환논증**이라 빠졌다. 한국 고유 기여는 하나도 반증되지 않았다.
  - 남은 것: CVPR 논문의 Figure 1 failure atlas(184일 중첩 → z=10.6 인공물)·stress case,
    그리고 `E_repr`/`E_fusion`과 event-first 재설계는 **연기**된 상태다.
  - **진짜 위험은 연기의 누적이다.** 3일 동안 네 번 밀렸고(derivability 도입 → 문헌 감사 →
    B1 비용 실측 → CVPR 단일 마감), **네 번 다 근거는 옳았으나 네 번 다 날짜를 붙이지 않았다.**
    근거 있는 연기가 날짜 없이 쌓이는 것이 조용한 포기다. L4를 일정에도 적용해야 한다.
- **고침: 14일 P0에 한국 데이터 P0(§3.3)를 병행으로 넣는다.** 감사 문서 스스로 "이 단계는
  GPU보다 데이터 정의가 병목"이라고 적었으므로 GPU-bound인 D1–14와 경쟁하지 않는다.
  D1–5 event universe 재구성 / D6–8 unique event·`published_time` 필드 실재 확인 /
  D9–11 200 event 층화추출 가용성 측정 / D12–14 NGII lead time 회수.
  1인 분량이 안 되면 오른쪽 열을 절반으로 줄이되 **NGII 신청만은 D1에 넣는다**(승인 대기가 길어
  지금 안 넣으면 다음 주기도 못 연다). **14일 안에 착수 못 하면 그때는 문서에 "포기"라고 적는다.**
- **W11 — event-first 설계에 빠진 칸: cell D.** BuildingHUB에서 표집을 시작하면 행정기록 존재가
  표본틀의 정의가 되어, 행정기록을 보는 모델이 자동으로 이긴다. 2×2가 필요하다.

  | | 기록 있음 | 기록 없음 |
  |---|---|---|
  | 변화 O | A 허가+착공 (공짜) | **D 무허가·미기록 변화** |
  | 변화 X | B 허가 후 미착공 (공짜) | C 배경 (무작위, 공짜) |

  **A·B·C만 있으면 만든 것은 허가기록 검증기이지 연구가 아니다. D가 요점 전부다.**
  D가 비어 있지 않다는 증거는 이미 있다 — **개발행위허가 제주 2023·2024가 0행**이므로 그 두 해의
  모든 실제 변화가 정의상 cell D다. 즉 D는 잔여가 아니라 구조적으로 큰 모집단이다.
  표집은 모델로 D를 고르면 편향되므로 **행정 coverage 상태 × 연도로 층화 후 층 내 무작위 →
  블라인드 판독**한다. 이 층화 자체가 Paper B(행정근거 coverage가 누가 감시받는지를 결정한다)의 주 결과다.
  **부수 효과: B1이 싸진다** — old 설계는 사람이 변화를 *발견*해야 했으나(368 분모, 희소 positive),
  새 설계에서 A·B·C는 *확인*이다. 비용은 D에 집중되며 그건 줄일 게 아니라 의도적으로 쓸 예산이다.
- **W12 — `R/V/T` 분해의 경고등.** 단일 derivability를 `R_source`/`V_source`/`T_source`로 나눈 것은
  기존 hard exclusion보다 명백히 낫다. 여기에 정보이론적 제약을 명시했다:
  **공공정보가 EO에서 정말로 회복 불가능하면(`R≈0`) EO-only student는 추론 때 그것을 운반할 수
  없으므로 `T≈0`이어야 한다.** 따라서 `R≈0`인데 `T>0`이면 셋 중 하나이며 전부 조사 대상이다 —
  ① 누수 ② `R` 측정 실패(probe가 약함) ③ context가 정보원이 아니라 **정규화·커리큘럼**으로 작동.
  ③도 실제 발견이지만 **다른 주장**이므로("공공 context가 학습 정규화로 작동했다") 같은 표에
  섞지 않는다. 역으로 `R`이 매우 높은 source는 `T>0`이어도 기여가 약하다.
  **사용자가 정한 "R 중간 + V·T 양수" 창이 옳고, 창이 좁은 이유가 바로 이것이다** —
  이유를 문서에 남겨야 나중에 스스로 창을 넓히지 않는다.
- 동의: 14일 P0 구성(D1–5/D6–8/D9–11/D12–14)과 4개 즉시중단 조건, 4.9 TB → pooled + 16×16 FP16
  token lattice 저장 설계, 병목이 VRAM이 아니라 I/O·teacher feature 추출이라는 판정.
- 약점: W10–W12는 설계 제안이고 실행 결과가 아니다. cell D의 실제 크기(층별 유병률)는 측정 전이며,
  층화 표집이 몇 건을 요구하는지도 아직 계산하지 않았다.
- 다음: ① NGII 신청 D1 착수 ② BuildingHUB event universe 재구성 ③ `published_time` 필드 실재
  확인(없으면 prospective 포기·retrospective residual로 격하) ④ cell D 유병률 파일럿 20건.

### 2026-08-24 — Earth 모델 × VLM 아이디어 경계 확인

- 계획: 기존 `EarthEmbedContract` 문제에 Vision-Language Model(VLM)을 붙일 때의 역할을
  ① 계약·근거 감사 ② 후보 설명·분류 ③ EO 표현 증류로 나누고, 최신 원 논문 기준으로 이미 해결된
  범위와 남는 연구 질문을 확인한다. 아이디어 검토이므로 새 GPU 실행은 하지 않는다.
- 사전 판정 기준: VLM이 결정론적 시간창·밴드·GSD·가중치 hash 검사를 대신하면 기각한다.
  VLM은 복구 불가능한 입력을 복구했다고 주장할 수 없고, 생성 문장은 공식 근거로 세지 않는다.
  현재 프로젝트에 가장 작은 실험으로 연결되고 명확한 대조군이 있을 때만 후속 후보로 남긴다.
- 결과: **가능하지만, 일반 EO 대화형 VLM은 이미 붐빈다.** GeoChat(CVPR 2024)은 영역 대화·시각
  grounding을, EarthDial(CVPR 2025)은 다중분광·다중시점·다중해상도 입력과 변화탐지까지 다룬다.
  GEOBench-VLM(ICCV 2025)에서는 최고 모델도 MCQ 41.7%라, VLM 단독 판정기를 신뢰하기도 어렵다.
- 남는 가장 자연스러운 역할은 `contract-grounded auditor`: 원본/전후 thumbnail, 기계 판독 가능한
  임베딩 계약, 한국 공공자료의 시점·공간 근거를 함께 받아 `REUSE / ADAPT / RECOMPUTE / ABSTAIN`을
  구조화 출력한다. 시간창 중첩·밴드 순서·weight hash는 결정론적 코드 gate가 먼저 막고, VLM은
  구름·계절·해무 같은 시각적 모호성과 근거 설명을 보조한다. 생성 설명은 evidence가 아니다.
- 최소 실험 후보: 같은 패널에 대해 `EO-only`, `VLM-only`, `hard contract gate`,
  `gate+VLM`, `gate+VLM+한국 공공근거`를 비교한다. 시간창 중첩·계절 불일치·release 불일치·정상
  대조군에서 **unsafe reuse율, artifact 분류, 근거 일치율, risk–coverage/AURC**를 본다.
- 논문성 판단: 단순 projector/대화 데모는 약하다. 여러 EO backbone·release·시간 recipe에서
  contract-aware hybrid가 조용한 오판을 일관되게 줄이는 benchmark+method이면 CVPR형으로 커질 수 있다.
  VLM을 EO embedding teacher로 쓰는 증류는 가능하지만 독립 task label이 필요하므로 두 번째 단계다.

### 2026-08-24 — Major TOM 249k 정밀 확인 + 마감 단일화(CVPR)

- 계획: 전날 "동일 chip 위에 여러 모델 임베딩이 공개돼 있다"고 쓴 근거가 조직 페이지 요약뿐이라,
  실제 데이터셋 카드 두 개를 직접 열어 **정말 paired인지** 확인한다. 동시에 AAAI 마감 종료를 반영한다.
- 사전 판정 기준: `unique_id`/`grid_cell`로 조인 가능하고 chip 수·원본이 같아야 paired로 인정한다.
  전처리 계약이 다르면 그 차이를 표로 먼저 적고, 대조 없이 한 표에 올리지 않는다.
- **결과 — paired 확인됨.** `249k-OlmoEarth-Base`와 `249k-Clay-v1_5` 모두 **248,719 chip,
  384×384**, 동일 스키마(`unique_id, embedding, timestamp, product_id, grid_cell, grid_row_u,
  grid_col_r, geometry, centre_lat/lon, utm_footprint, utm_crs, pixel_bbox, parquet_row, parquet_url`),
  둘 다 **CC-BY-SA-4.0**. Clay 카드가 "동일한 249k grid cell과 동일 원본 영상을 다른 Major TOM
  249k 임베딩 데이터셋과 공유한다"고 명시한다. 용량은 824 MB / 1.08 GB로 노트북 규모다.
- **그런데 계약이 서로 다르다 — 그리고 그게 우리 증거다.**

  | | OlmoEarth-Base | Clay v1.5 |
  |---|---|---|
  | 차원 | 768 | 1024 |
  | pooling | unmasked spatial token **평균** | **CLS** token |
  | 밴드 | **12개 전부**(재정렬) | **10개**(B01·B09 없음) |
  | 정규화 | OlmoEarth 사전학습 normalizer | Clay S2 mean/std |
  | L2 정규화 | 미적용 명시 | 미기재 |

  같은 chip 위의 두 공개 제품이 pooling·밴드 수·정규화에서 다른데, 이 차이는 데이터셋 카드의
  **산문**에만 있고 기계가 읽을 수 있는 필드에는 없다. 모델 가중치 hash는 아예 없다.
  → `embeddings-stac-specification` v0.0.1 gap 분석(W9)의 직접 증거로 들어간다.
- **가능/불가능 경계를 문서에 명시했다.** 가능: cross-model paired 집합, gallery-size 곡선을
  `10³–10⁵` **실측**으로(지금까지 216에서 외삽하던 구간), 진단 눈멂 행렬을 실제 공개 제품에서 실행.
  **불가능 4가지**: ① release pair가 아님(Major TOM의 OlmoEarth 릴리스는 **하나뿐**이라
  `S1→S0` 질문에 직접 답하지 않음) ② 밴드·pooling·정규화가 달라 모델 우열 비교 불가
  ③ **chip당 벡터 하나뿐이라 token/공간 수준 분석 불가** — 우리 실패(R@1 0.0000, 동일 token
  cosine −0.00860, spatial CKA 0.427)는 token 수준이므로 여전히 로컬 216 raster에서만 가능
  ④ 라벨 없음. **두 자산은 대체재가 아니라 보완재다** — Major TOM은 넓고 얕게, 제주 216은 좁고 깊게.
- **사전 예측 기록(실행 전).** 우리 216 파이프라인은 768×256×256 dense token raster를 만들고
  Major TOM은 unmasked token 평균 1개를 만든다. → **동일하게 mean-pool하지 않으면 일치하지 않고,
  동일하게 pool해도 밴드 재정렬·normalizer가 정확히 같아야 일치한다.** 맞으면 "같은 모델·같은
  영상인데 계약이 달라 값이 다르다"의 공개 제품 사례가 되고, 그냥 일치하면 계약 재현 절차가
  산출물이 된다. 어느 쪽이든 손해가 없다.
- **마감 단일화**: 사용자 확인으로 **AAAI 마감은 이미 종료**. 이 주기의 목표는 **CVPR 하나**
  (통상 11월 초 → 10-31 완료). 한국 트랙 venue 결정은 보류하고 CVPR 결과·B1 진척을 보고 정한다.
  부수 효과로 "두 트랙 동시 제출" 압력이 사라졌으므로, 한국 데이터는 이번 주기에 CVPR 논문의
  **failure atlas·stress case**로만 기여하면 된다. FoldRefresh의 AAAI-27 제출 건은 심사 중이며
  K-ALIGN에서는 인용 선행 자산으로만 다룬다(중복 제출 아님).
- 약점: 두 카드만 열었다. SatCLIP·SigLIP·DINOv2·FarSLIP·MMEarth·AlphaEarth·UniverSat의 계약은
  아직 확인하지 않았다. 조인이 실제로 1:1로 떨어지는지도 파일을 받아 확인해야 한다(카드 기재와
  실제 행 정렬은 다를 수 있다).
- 다음: ① 두 parquet를 실제로 받아 `unique_id` 조인이 1:1인지 확인 ② 계약 대조표를 나머지
  249k 데이터셋으로 확장 ③ 그 표를 embeddings-stac gap 분석 초안에 그대로 사용
  ④ W1 dose–response 최소판(밴드 순서·정규화 축) ⑤ ADC baseline.

### 2026-08-23 (4차) — 광각 보정 + 미확인 항목 4건 실제 검증

- 계획: 사용자 피드백(시간창 184일 중첩, 4기간 계절편향, Jaccard 0.091, FEATURES=768 제약)과
  새 문서 두 개(`K_ALIGN_BIG_PICTURE.md`, `K_ALIGN_CVPR_READINESS_AUDIT.md`)를 읽고, **그 두
  문서 바깥**에 있는 프로그램 수준 보정만 더한다. 그 뒤 "확인하지 않았다"고 적은 항목을 실제로 확인한다.
- 사전 판정 기준: 기존 문서와 중복되면 쓰지 않는다. 각 보정은 ① 방법 성패와 무관하게 쓰이거나
  ② 답할 수 없는 질문을 답할 수 있게 만들거나 ③ 리뷰어의 결정적 반론을 미리 막아야 한다.
- 근거 재확인(직접): `setup_jeju_v2.sh:55-58`의 jeju25↔jeju26r **184일 중첩** 실재,
  `change_v6_t12.py`에 이미 `REFUSED` 가드가 적용돼 있음(고칠 순서 1·3이 부분 반영됨),
  `olmo_release_raster_contract.py:19 FEATURES=768`, `jeju_change_v6_top.json` control
  intersection=5 / jaccard=0.091.
- 결과 — `K_ALIGN_WIDE_ANGLE.md` 신규(326행) W1–W9. 핵심 셋:
  - **W1 중첩은 버그가 아니라 계측기다.** 184일은 우연히 만들어진 "50% 용량 1회 투여"다.
    중첩 0/25/50/75/100%, 계절 offset, 밴드 순서, 정규화, pooling을 조절하면 **dose–response
    곡선**이 된다. Figure 1이 일화에서 곡선이 되고, ground truth가 알려진 합성 release pair를
    무제한 생산하므로 **W5의 pair 부족 문제까지 푼다**. raster 재사용이라 가장 싸다.
  - **W2 기여는 계약 명세가 아니라 "기존 진단이 전부 눈멀었다"는 것.** CKA 0.97857·거리
    Spearman 0.95251·z=10.6 셋 다 통과시켰는데 실제는 R@1 0.0000·cosine −0.00860·100% 인공물이었다.
    진단 K개 × 불일치 유형 M개 행렬에서 **대부분이 "탐지 못함"인 표 자체가 결과**다.
    이 프레이밍이면 remote sensing 논문이 아니라 표현 평가 논문이 된다.
  - **W7 바닥을 다시 매긴다.** 전처리 불일치가 R@1=0을 전부 설명해도 남는 것은 "위로상"이
    아니라 감사 논문이다. 따라서 W1·W2를 방법보다 **먼저** 한다.
- **미확인 4건 검증 결과 — 둘은 결론이 바뀌었다:**
  - **W9 ✅ 결론 변경.** "STAC 확장을 새로 쓰자"고 제안했으나 자리는 이미 있었다.
    [`stac-extensions/mlm`](https://github.com/stac-extensions/mlm)(활성),
    [`stac-extensions/ml-model`](https://github.com/stac-extensions/ml-model)(deprecated),
    [`geo-embeddings/embeddings-stac-specification`](https://github.com/geo-embeddings/embeddings-stac-specification)
    (**Proposal, v0.0.1**). 그런데 **그 v0.0.1이 빠뜨린 필드가 정확히 우리가 증명한 두 실패다** —
    모델 가중치 hash(v1→v1.2 R@1 0), 실제 acquisition 날짜·temporal recipe(184일 중첩),
    밴드 순서, input/output content hash. 있는 것은 `emb:temporal_resolution`, `gsd`,
    `emb:preprocessing/postprocessing`, `processing:version`뿐이다.
    → 새 확장 작성이 아니라 **기존 Proposal에 gap 분석 + 실패 증거를 붙인 이슈/PR**로 바뀐다.
    더 싸고 채택 가능성이 높으며 지금이 열려 있는 시점이다.
  - **W4 ✅ 규모 확대.** `Core-S2L2A-249k-OlmoEarth-Base`는 실재한다(2026-08-21 기록이 맞았다).
    게다가 **동일한 249k chip 위에 Clay-v1_5·SatCLIP·SigLIP·DINOv2·FarSLIP 임베딩이 이미
    공개돼 있다**(조직 전체 25개 데이터셋, 그 밖에 MMEarth·AlphaEarth·UniverSat·SSL4EO·DeCUR).
    즉 **paired-input cross-model 실험대가 이미 공짜로 존재**하고, gallery-size 곡선을
    216 site-years가 아니라 248,719 chip 실측으로 그릴 수 있다. 단 이들은 cross-family이지
    release pair가 아니며, 각 데이터셋의 전처리 계약이 다르므로 **대조 없이 한 표에 올리면
    우리가 경고하는 오류를 우리가 저지르는 것**이다.
  - **W5 ✅ 부분 해결.** Clay v1.0(2024-06-06) / v1.5(가중치 2024-11-19)가 둘 다 공개된 실재
    release pair이고 Major TOM에 v1.5 임베딩도 있다. 확실한 release pair는 Olmo·Prithvi·Clay
    **3개**. 예측기 fit + held-out에는 여전히 부족 → W1 합성 pair 필요성이 재확인됐다.
  - **W6 ✅ 출처 확인.** ADC는 Jégou·Douze·Schmid, TPAMI 33(1):117–128, 2011.
    "computes the approximate distance between a vector and a code" — **query를 양자화하지 않는
    것이 PQ의 원래 권장 사용법**이다. 15년 된 표준이므로 quantizer-aware 방법을 제안하기 전에
    `새 query → 선형 map → old float 공간 → old codebook에 ADC` baseline을 반드시 먼저 돌린다.
- 동의하고 넘어간 것: `R@1=0 → affine 61–70%`를 비선형 잔차 증거로 쓰지 않음, 37.5분/216건을
  재임베딩 비용으로 외삽하지 않음(49.1 GiB GeoTIFF 쓰기 포함), "오름 보전지역이라 행정사건 0"을
  미증명으로 둠, 한국 트랙을 CVPR main에서 분리.
- 약점: W1–W3·W7·W8은 여전히 **제안**이고 실행 결과가 아니다. W8(AAAI AISI에 FoldRefresh와
  같은 회차 두 편 제출 가능 여부)은 확인하지 못했다. `mlm`과 `embeddings-stac`의 역할 경계도
  미확인이라 PR 전에 두 저장소 이슈를 읽어야 한다.
- 다음: ① Major TOM 249k 데이터셋들의 **전처리 계약 대조표**부터 (재계산 검증의 전제)
  ② W1 dose–response 최소판(밴드 순서·정규화 축은 창 재정의도 불필요) ③ embeddings-stac
  gap 분석 초안 ④ ADC baseline ⑤ 기존 Day 0 compact 재임베딩 비용곡선은 그대로 병렬 유지.

### 2026-08-23 (6차) — 시간계약 오류를 먼저 고치는 단순한 EO 연구축

- 계획: 제주 변화탐지의 ① 2025/rolling-2026 6개월 중첩 ② 4기간 계절 불일치 ③ 4/12기간 후보
  불안정성 ④ 14후보 오염률을 원 코드·manifest·산출물에서 재검증한다. 검증되면 후보 생성에서
  잘못된 전이를 fail-closed로 차단하고 12기간 단일 경로를 canonical로 지정하며, 시간축 감사 실패를
  사전 gate로 승격한다. 연구 큰 그림은 “임베딩은 release×time-window×band×GSD×pooling 계약
  안에서만 의미가 있고, 계약이 어긋난 재사용은 조용한 고확신 오류를 만든다”로 단순화한다.
- 사전 성공 기준: ① 중첩 기간·계절 불일치·Top-30 Jaccard를 파일에서 재현 ② 잘못된 2025→2026
  및 4기간 후보 경로가 기본 실행에서 거부됨 ③ 실패 이유가 후보 JSON/문서에 전파됨 ④ 기존
  release-audit 테스트는 유지 ⑤ 재실행이 필요한 범위와 금지 주장을 명확히 기록.
- 무효/중단 조건: 실제 날짜가 사용자 요약과 다름, 4기간이 후보 생성에 쓰이지 않음, 오염 9/14가
  candidate lineage로 재현되지 않음, 또는 기존 216 임베딩만으로 월별 재구성이 가능하다고 확인되는
  경우. 이때는 가설을 수정하고 코드를 억지로 막지 않는다.
- 결과: 원파일에서 네 발견을 재현했다. `jeju25`/`jeju26r` overlap은 184일,
  `model_first4_season_aligned_across_years=false`, `all12_cover_same_calendar_month_set=false`,
  4/12 Top-30 intersection 5·Jaccard 0.091이다. 14후보는 overlap 전이 5, v3 4기간 source 5,
  합집합 9, 두 검사 비노출 5로 재집계됐다.
- 해석 보정: 9/14는 모두 false positive라는 뜻이 아니라 **annual-change claim lineage 부적격**이다.
  실제 May RGB에서 지속 변화가 보인 record도 포함돼 있다. 또한 “오름은 보전지역이라 행정사건이
  구조적으로 0”은 미검증이다. 공식 오름 polygon 부재·OSM point/필지 geometry mismatch와 제도별
  coverage를 먼저 분리한다. 개발행위허가 제주 2023/24 0행은 확인된 source hole이다.
- 코드: `audit_jeju_candidate_time_contract.py`와 hash-linked audit JSON을 추가했다. 역사적
  `change_v2_step.py`, `change_v5.py`, `change_v6_t12.py`, `build_jeju_human_review.py`,
  `score_oreum_existing_embeddings.py`는 명시적 failure-reproduction override 없이는 실행을 거부한다.
  새 후보는 non-overlap·season alignment·actual acquisition hash가 통과한 manifest 뒤에만 허용한다.
- 큰 그림: `K_ALIGN_BIG_PICTURE.md`를 **contract-bound Earth embedding** 중심으로 단순화했다.
  모델 릴리스 mismatch와 시간창 mismatch를 같은 silent high-confidence reuse error로 묶고,
  비교 전 `REUSE / ADAPT / RECOMPUTE·ABSTAIN` 판정을 연구한다. quantizer·비용·공공 context는
  해결수단/검증층으로 내렸다.

### 2026-08-23 (5차) — 숲 재정렬: cost-first characterization × quantized compatibility

- 계획: K-ALIGN을 조건을 많이 붙인 adapter 논문이 아니라 ① 재임베딩이 실제로 비싼/불가능한
  운영 구간을 먼저 측정하고 ② release 간 사후 정렬 가능성을 예측하며 ③ 고정 quantizer가 있는
  구간에서만 새 방법을 제시하는 프로그램으로 재정렬한다. 현재 216 full-raster 실행시간은 compact
  production re-embedding 비용을 과대평가할 수 있으므로 근거로 바로 외삽하지 않는다.
- 사전 성공 기준: ① 방법 구현 전 re-embed/adapter/dual-index의 `N×Q` 비용 경계 ② gallery size별
  retrieval degradation ③ band order·normalization·pooling·timestamp 등 사소한 mismatch 재감사
  ④ layer CKA/probe 등 alignment feasibility predictor의 held-out 검정 ⑤ quantized old gallery에서
  최신 compatibility baseline보다 task–retrieval–cost Pareto 우위.
- 무효/중단 조건: compact re-embedding이 현실적 archive에서 더 싸고 raw imagery도 항상 접근 가능,
  alignment predictor가 새 family/release에서 일반화하지 않음, quantizer-aware 방법이 기존 post-hoc
  adapter/PQ 재구축/dual-index와 같음, 216개 작은 gallery 결과만 보고 scale compatibility 주장,
  또는 한국 public-context/published-time 문제가 해결되지 않았는데 CVPR main에 합치는 경우.
- 결과: `K_ALIGN_BIG_PICTURE.md`를 만들어 프로그램을 **Necessity → Predictability → Intervention**으로
  재정렬했다. 현재 1순위는 cost-first characterization이며 public-context는 조건부 후속이다.
- P0 순서를 뒤집었다. D1에 compact re-embed/dual-index/adapter `N×Q` 비용곡선, D2–4에
  preprocessing mismatch 재감사와 gallery-size protocol, D5–8에 held-out alignment predictor,
  D9–11에 fixed-quantizer 방법, D12–14에는 public task 1개의 frozen probe만 둔다.
- 문헌 추가 감사: UniBCT(IJCAI 2022), BiCT, Darwinian Model Upgrades(AAAI 2023), WACV 2025 online
  backfill이 open-set/raw-image 부재/gallery evolution을 이미 다룬다. 따라서 raw raster 부재나
  부분 gallery 갱신만으로 novelty를 주장하지 않는다. fixed old PQ code와 frozen EO release의
  결합은 아직 유망한 경계지만 systematic search·재현 전에는 “최초”라고 쓰지 않는다.
- 비용 해석 보정: v1.2 216건 2,250초는 49.1 GiB full token GeoTIFF 쓰기를 포함하므로 compact
  production re-embedding 비용의 상한성 관측일 뿐이다. compact writer를 별도 계측하기 전에는
  adapter가 경제적이라는 주장을 열지 않는다.

### 2026-08-23 (4차) — 첨부 embedding 아이디어 26편 근거감사·CVPR 실험/엔지니어링 판정

- 계획: 사용자 첨부 노트의 26편·아이디어를 끝까지 읽고, 초록/본문 확인 수준과 검색 snippet 수준을
  분리한다. CVPR main 후보마다 선행연구가 이미 점유한 부분, 정확한 새 연구질문, 필요한 데이터·
  라벨·모델·compute, matched baseline, leakage 없는 split, primary metric, promotion/kill gate,
  구현 모듈과 예상 엔지니어링 병목을 정리한다. K-ALIGN의 stable cache·multi-teacher distillation·
  한국 public residual 계약과 겹치거나 충돌하는지도 대조한다.
- 사전 성공 기준: ① 첨부의 모든 아이디어를 evidence tier로 재분류 ② main 후보 1–2개와 버릴/후속
  후보를 명확히 순위화 ③ 공식 proceedings/paper/repo로 핵심 선행 8편 이상 재검증 ④ 데이터가 없는
  주장을 실험 가능으로 포장하지 않음 ⑤ P0/P1/full main matrix와 GPU/스토리지/라벨 예상치를 분리
  ⑥ engineering contribution과 단순 plumbing을 구분하고 CVPR·다른 venue fit을 조건부 판정.
- 무효/중단 조건: snippet을 재현된 사실로 사용, `binary token`·`causal`·`federated` 같은 용어만
  붙여 novelty를 주장, 기존 sealed 제주 split을 새 방법 test로 재사용, 한국 public record를 feature와
  label에 중복 사용, 서로 다른 모델 native input을 paired 우월성으로 해석, 로봇/시뮬레이션을 데이터
  없이 주기여로 추가, 모델 수·API 수·코드량만으로 main-track 기여를 주장하는 경우.
- 결과: `K_ALIGN_CVPR_READINESS_AUDIT.md`에 근거감사·두 논문 track·14일 P0·6–8주 full matrix·
  엔지니어링 모듈·GPU/스토리지 추정·promotion/kill gate를 작성했다.
- **현재 main 판정은 낮음.** 216쌍 release audit은 재현·누출방지·provenance 측면에서 강하지만
  label 0, 제주 54 cluster 한 격자, downstream task 0, multi-model trainer 0이다. 시스템 감사만으로
  CVPR vision method 기여가 되지는 않는다.
- **이번 주기 1순위**를 frozen third-party EO model upgrade로 좁혔다. Olmo v1/v1.2와
  Prithvi 1.0/2.0, 공개 task 3개, compressed gallery에서 새 모델 task utility와 old-gallery
  compatibility를 동시에 보존하는 post-hoc adapter/student를 검증한다. BCT/FCT/LCE/AdvBCT/BT²,
  hyperbolic BCRL과 강한 post-hoc bridge를 이기지 못하면 main 주장을 중단한다.
- **한국 public-context track은 event-first data gate 뒤로 이동.** 현재 candidate-first join은
  14후보 중 time-aligned exact support 0건이고 오름 전체 causal evidence 0/368이다. BuildingHUB
  8,794행/EIA 13 polygon에서 event universe를 먼저 만들고, EO before/after와 matched control,
  source와 독립인 label을 붙인다. `published_time`은 현재 BuildingHUB snapshot에 실제 필드가 없어
  `created_date`/`retrieved_at`으로 대체하지 않는다.
- **기전 수정:** 단일 derivability 제외 규칙을 폐기하고 `R_source`(EO 회복가능성),
  `V_source`(독립 task 추가가치), `T_source`(EO-only student 전이효과)를 분리했다. 완전히 EO에서
  회복 불가능한 행정정보는 distillation보다 inference residual/abstention 역할이 맞다.
- **E-07 정정:** ICLR 2026 공식 OpenReview 본문을 확인했다. 약 598–690 B는 주로 downlink JSON
  telemetry이고 hint/gallery upload는 별도다. 따라서 온보드 link는 운영 동기일 뿐 1 KB gallery
  backfill 불가능의 증거가 아니다. 3차 worklog의 당시 해석은 실패 계보로 남기되 본 계약에서는 폐기했다.
- 공식 1차 소스로 PANGAEA, GeoLink, MMEarth, OmniSat, SatMIP, Galileo, WildSAT, Auxiliary
  Modality Learning, AM-RADIO, BCT/FCT/LCE/AdvBCT/BT², ICML 2025 BCRL, XBT, NeuCo-Bench,
  E-07을 재확인했다. `M`은 저자 주장 확인이지 우리 환경 재현 완료가 아니다.
- 14일 P0 gate: 새 untouched anchor 2,048개, Olmo/Prithvi 공통 adapter, identity/Procrustes/ridge/
  MLP/compatibility baseline, 공개 task 1개의 old/new/cross 4-cell, PCA64+int8/PQ 비용표까지 닫는다.
  MLP가 affine를 못 이기거나 cross-task utility가 무너지거나 dual-index가 Pareto 우위면 확장하지 않는다.

### 2026-08-23 (3차) — 네 축 문헌 감사: 무엇이 좋아지는가를 네 주장으로 분해

- 계획: 사용자가 제기한 큰 그림(`OlmoEarth × 한국 공공데이터 → 정확도 / 임베딩 / 속도 /
  위성 유도`)을 문헌으로 검증한다. `earth_paper` 코퍼스 185편을 훑고 웹 검색으로 2025–2026
  최신을 보강한다. 마감(CVPR 통상 11월 초 → 10월 말 완료)과 FoldRefresh AAAI 제출 상태를 반영한다.
- 사전 판정 기준: 각 축은 ① 기전 ② 이미 점유한 선행연구 ③ 남은 빈칸 ④ 현재 자산 실현가능성이
  모두 적혀야 한다. 하나라도 못 채우면 그 축은 "가능성"으로 세지 않는다.
- 무효 조건: 검색 스니펫의 수치를 원문 확인 없이 논문 근거로 승격하는 경우, 네 축을 한 논문에
  합치는 경우.
- 결과 — 네 축 판정 (`K_GAIN_AXES.md` 신규 275행):
  - **A 정확도 = 조건부.** PANGAEA는 full-label에서 UNet 등 supervised baseline이 대부분의 GFM을
    이기고 10% label에서만 GFM이 이긴다고 보고한다. → `E_repr`의 primary를 **라벨 절감**으로
    바꾸고 정확도 +2%p를 secondary로 내렸다.
  - **B 임베딩 = 본편이되 빈칸이 좁다.** GeoLink·CLIP4Geo·WildSAT이 "비-EO 기록으로 EO 표현
    강화"를 이미 점유했다. WildSAT은 야생동물 관찰기록으로 위성 표현을 학습한다. 남은 빈칸은
    **`published_time`이 관리되는 기록** 하나뿐이다.
  - **C 속도 = 직접 기전 없음.** 한국 데이터는 추론을 빠르게 하지 않는다. 게다가 압축은 이미
    binary quantization 32×(float32 NN의 약 65% 회복), PCA(64)+int8이 sweet spot이다.
    → `E_refresh`의 bytes 기준선을 float32에서 **PCA(64)+int8**로 교체했다. 이걸 안 바꿨으면
    이미 알려진 압축을 우리 성과로 셀 뻔했다.
  - **D 위성 유도 = EarthRoute로 이월.** tip-and-cue·EO 스케줄링·온보드 RSFM 배포가 각각 점유됨.
- 기여에서 **뺀 것 두 개**:
  - **필지 경계** — 전지구 10 m 필지 경계 지도가 241개국 **31.7억 polygon**으로 공개됐다.
    FarmMap 289,379 polygon은 이제 기여가 아니라 anchor다.
  - **한국판 FLAIR-HUB** — IGN이 항공·S1/S2·SPOT·지형·과거항공 6모달, **630억 수동 주석 픽셀**,
    OA 78.2/mIoU 65.8로 이미 만들었다. 주석 예산에서 이길 수 없다.
- 당시 기여로 **끌어온 것 하나**(4차 감사에서 해석 폐기): 우리 코퍼스의 `E-07` [Embedding-Only Uplink for Onboard Retrieval
  Under Shift](https://arxiv.org/abs/2604.03301) (초록 확인). 지상국이 임베딩만 업링크하고
  궤도상에서 벡터 검색을 하며 **질의당 1 KB 미만**이다. 궤도상 gallery는 backfill이 물리적으로
  불가능하므로, backward compatibility의 가장 극단적인 운영 근거가 된다. `E_refresh` 동기
  문단에 넣되 **시뮬레이션된 대역폭 예산으로만** 다룬다.
- GK2A 판정: 2 km·고빈도는 한국의 진짜 물리 자산이지만 정지궤도×극궤도 NDVI gap filling은 이미
  연구 중이고, 과거 GK2A는 현재 endpoint로 소급 조회가 안 된다(최근 2일 제한, 6관측일 실패 실측).
  → 용도를 **구름 상태의 센서 독립 감사**로 한정하고 융합·초해상으로 확장하지 않는다.
- 마감 반영: **2026-10-31 완료 기준, 오늘부터 약 10주, 슬랙 1주**. 주차별 표를 프로그램 노트
  6절에 넣었다. **10-04 체크포인트**에 A4가 안 닫히면 순위 0으로 전환한다. B1(한국 라벨)은 이
  일정 안에 `E_repr`/`E_fusion`을 닫을 수 없으므로, **CVPR 제출본은 label-free core + 공개
  dense task로 대체한 축소 `E_repr`로 간다**고 지금 결정했다. 한국 라벨은 다음 마감용이다.
- FoldRefresh 상태 보정: **AAAI-27 AISI에 제출·갱신됨**. 따라서 K-ALIGN에서 FoldRefresh는 새
  기여가 아니라 인용하는 선행 자산이고 같은 내용을 두 번 제출하지 않는다. R4는 "만든다"가
  아니라 "refresh 경로에 **적용**해 보증을 붙인다"로 다시 썼다.
- 계약 수정 5건 반영: `E_repr` primary 교체, `E_refresh` bytes 기준선 교체, 필지 경계 제외,
  baseline 확장(CLIP4Geo·WildSAT·Beyond-Pixels·JDCNet·InfraNet·binary quantization·NeuCo-Bench),
  온보드 업링크 동기 추가. `PAPER_READING_LIST.md`에 결정 9건 표를 추가했다.
- **정정 (같은 날 확인)**: FLAIR-HUB를 "주석 예산에서 이길 수 없다"고 쓴 근거가 틀렸다.
  초록을 열어보니 면적은 프랑스 전토가 아니라 **2,528 km²**(제주 약 1,850 km²의 1.4배)이고,
  "630억 픽셀"은 그 면적을 20 cm로 나눈 dense raster 화소 수이지 사람의 판정 횟수가 아니다.
  하지 않을 진짜 이유는 ⓐ 장르를 IGN이 이미 정의했고 ⓑ 한국판 dense 주석은 환경부
  토지피복지도의 재포장이라 독립 정답이 아니며 ⓒ 비동기 provenance와 무관하다는 것이다.
  다만 **asset 형식(모달 정렬·CC BY-SA 4.0·벤치마크 동봉)은 템플릿으로 차용**한다.
- **B1 라벨 비용 실측** (프로그램 노트 5절에 추가): 총 판독 1,720회(1,200 + 이중판독 520),
  회당 5–10분 → 143–287시간, 파트타임 10 h/주면 판독만 14–29주. 10주 마감에 못 들어간다.
  그러나 더 큰 것은 구조적 막힘 7개다 — ① **두 번째 독립 판독자 부재**(1인 프로젝트인데 계약이
  independent double review를 요구하고 assistant pre-annotation은 leak으로 금지) ② 10 m S2로
  판정 불가한 사건 ③ NGII 항공사진 수동 신청·승인(배치 API 없음, 최대 2,400회) ④ 항공 촬영
  주기가 라벨의 시간 해상도를 1년으로 고정 ⑤ 오름 368개 공식 polygon 0개 — 라벨 단위 geometry
  미정 ⑥ positive 희소(368 중 후보 14) → 층화 → PPI 필요 ⑦ 행정기록 대체 불가(실증: exact PNU
  1건, 시간정렬 0건, EIA 중첩 0건). 자체 증거로 v5 blind pair 5/5 `no_improvement` gate 실패,
  누적 사람 판독 실적 약 23건(RGB 9 + 후보 14) vs 목표 1,200건, 코퍼스의 `W-20`
  *Humans are Poor Few-Shot Classifiers for Sentinel-2 Land Cover* 경고.
  → **해법은 규모 축소가 아니라 막힘 1·3 해소**다: sealed 400건만 이중판독하면 약 47시간이라
  파트너 또는 유료 판독자 1명에게 발주 가능하고, NGII 승인 lead time은 10–20건 파일럿으로
  먼저 측정해야 한다(**이 수치 없이는 B1 일정 계산 자체가 불가능**).
- 약점: **초록까지 확인한 것은 `E-07`과 FLAIR-HUB 둘**이다. 나머지 25편은 제목·URL·검색 스니펫 수준(`W`)이며
  수치를 논문에 인용하기 전에 원문을 열어야 한다. 특히 binary 32×/65% 회복과 PCA64+int8 sweet
  spot은 실무 블로그 보고이므로 학술 인용으로 쓰면 안 된다. PANGAEA·FLAIR-HUB·압축 3편은
  gate 수치를 직접 바꾸므로 **정독이 A0의 일부**다.
- 다음: ① `E-07` 정독(PDF 보유) ② Beyond-Pixels(raster×vector, arXiv 2606.02374) — 우리
  아이디어와 가장 가까움 ③ PANGAEA로 `E_repr` 주장 형태 확정 ④ FLAIR-HUB로 "만들지 않을 것"
  확정 ⑤ 압축 3편으로 `E_refresh` gate 수치 확정 ⑥ 그 다음 A0 계약 freeze.

### 2026-08-23 (2차) — K-ALIGN 승률 보정: 프로그램 노트 신설 + 중심 계약 개정

- 계획: 8월 EarthRoute 핸드오프 노트와 같은 층위의 프로그램 문서를 K-ALIGN에도 만들고,
  "설계는 맞지만 승률이 낮은" 지점을 진단해 중심 계약(`KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`)을
  실제로 고친다. 새 실험은 돌리지 않는다.
- 사전 판정 기준: 보정은 ① 심사자가 던질 반론을 실험 설계로 미리 막거나 ② 실패 시 남는 산출물의
  바닥을 올리거나 ③ 임계경로를 줄여야 한다. 셋 중 하나도 아니면 넣지 않는다.
- 무효 조건: 새 수치를 만들지 않고 문서만 늘리는 경우, 이미 결과를 본 sealed 64를 다시 test로
  끌어오는 경우, 마감 압박으로 gate 수치를 낮추는 경우.
- 진단한 구조적 취약점 3개:
  - **V1 순환논증** — 토지피복·FarmMap·DEM은 EO 파생물이다. 이것을 privileged teacher로 쓰면
    `E_repr`은 "EO 정보를 EO 모델에 되돌려 넣었다"는 한 줄로 소멸한다.
  - **V2 BCT 재탕** — `S1`에 compat loss를 걸었으니 `S1→S0` R@1 95%는 통과할 확률이 높다.
    그것만으로는 BCT의 재현이고 novelty 문단을 쓸 수 없다.
  - **V3 라벨이 임계경로** — 기존 A5(라벨 1,200 + 이중판독 400)가 A6 앞을 막아 파트타임
    10 h/주에서 단일 실패점이었다.
- 결과 — 여덟 보정(R1–R8)을 `K_ALIGN_PROGRAM_NOTE.md`(신규 402행)에 기록하고 계약에 반영:
  - **R1 derivability screen** — 모든 public source에 EO-only probe `D(source)`를 사전 측정해
    `D >= tau`면 privileged teacher에서 제외한다. `tau` 제안값 R² 0.60 / AUPRC lift 2.0 /
    ±30일. 통과 source의 시간정렬 coverage가 대상 site-event의 5% 미만이면 `E_repr`을 열지 않는다.
    이 표 자체가 독립 산출물이다.
  - **R2 black-box 불가능성 baseline 필수화** — `BCT-surrogate`, `FCT-posthoc`,
    `contract-mismatch` 세 개를 §7에 의무로 넣었다. 주장 형태는 "BCT가 나쁘다"가 아니라
    "BCT/FCT의 가정이 공개 release에서 성립하지 않는다"이다.
  - **R3 label-free core 선행** — `E_compat`·`E_refresh`·derivability는 라벨 0개로 평가된다.
    실행순서를 A0–A7(임계경로) / B1–B2(라벨 병렬)로 재편했다.
  - **R4 FoldRefresh 인증** — `E_refresh`를 비용 숫자에서 "부분 갱신된 cache 위 통계의 유효한
    유한모집단 보증"으로 승격. BCT·FCT·AM-RADIO·Matryoshka 어디에도 없는 유일한 차별점이고
    형제 저장소에 이미 검증된 자산이다. 계약 §12의 금지 주장에도 재현 전 사용 금지로 못박았다.
  - **R5 합성 비동기 harness** — 심사자에게 VWorld/BuildingHUB/GK2A 접근권이 없다. 공개
    데이터만으로 publication lag·coverage hole·conflict를 합성하는 A6을 신설하고, **A6 없이는
    두 축이 통과해도 main으로 올리지 않는다**는 규칙을 넣었다.
  - **R6 asset 축** — 216×2 paired 출력, 5,616파일 exact-input freeze, 5,184행 시간축 manifest,
    463 request provenance snapshot, failure atlas를 공개 benchmark로 묶는 A7 신설.
  - **R7 바닥 논문 고정** — CKA 0.97857 / 거리 Spearman 0.95251 vs 동일 token cosine −0.00860,
    cross-release R@1 0.0000, 사전등록 8 gate 전부 실패. "CKA는 호환성이 아니다"를 최악의
    경우에도 남는 산출물로 선언했다.
  - **R8 family×release 격자** — 현재 구성은 release 축이 Olmo 한 family에만 있다. 두 번째
    family의 두 릴리스(Prithvi-EO 1.0/2.0)를 후보로 넣되 6-band·30 m HLS 계약이 S2 10 m와
    paired input을 만들 수 있는지 A2에서 먼저 판정한다.
- 계약 변경 실물: §0에 위험 3개, §2에 라벨 의존성 분리, §4에 derivability screen 절,
  §7에 black-box baseline과 FoldRefresh 인증, §9에 gate 4행 추가, §10 figure 3개 추가,
  §11 실행순서·family 격자·제출 사다리 전면 교체, §12 금지 주장 3개 추가, §13 프로그램 수준
  중단 조건 6개 신설. 문서는 290 → 417행.
- 약점: 이번 세션은 계획 보정이고 실험 결과가 아니다. R1의 예상 derivability 표는 아직 예상일
  뿐이며, 저-derivability source의 coverage가 이미 나쁜 신호를 보인다(14후보 exact PNU 1건,
  시간정렬 0건, EIA 직접중첩 0건). 마감일도 확인하지 않아 사다리에 날짜가 없다.
- 다음: ① A0 계약에 `tau`와 합성 harness 기반 데이터셋 확정 ② R7 바닥 그림을 `artifacts/`에
  고정 ③ **공공데이터 재배포 라이선스 검토** — 여기서 막히면 asset 축이 통째로 사라지므로
  가장 먼저 확인한다 ④ A1 3지역 split hash-freeze ⑤ A2 teacher-contract smoke
  ⑥ NGII 전후 항공사진 신청은 임계경로가 아니어도 승인 대기가 길어 즉시 착수.

### 2026-08-23 — 한국 공공데이터 정렬형 compatible Earth bus로 초점 수렴

- 계획: 사용자가 `한국을 살리는 transfer`, `좌표계를 맞춘 cache 재사용`, `다른 backbone의 teacher
  signal 증류`, `한국 공공데이터 alignment 극대화`를 중심으로 좁혀 달라고 요청했다. 직전의 다섯
  광범위 트랙을 그대로 병합하지 않고, stable EO bus와 timestamped Korea-context residual의
  이중속도 표현 하나로 재정식화한다. 기존 K-Context의 provenance/natural-missingness 계약과
  EarthBus의 multi-teacher compatibility 계약을 같은 평가 단위에서 연결한다.
- 사전 성공 기준: ① stable cache와 dynamic context residual의 역할·갱신주기를 분리 ② Olmo
  v1/v1.2·TerraMind teacher, EO-only student, public-context teacher의 train/test 입력 계약 명시
  ③ `E_repr / E_compat / E_fusion / E_refresh`를 별도 estimand로 정의 ④ 한국 public source별
  event/observed/published/retrieved time·coverage·conflict를 보존 ⑤ identity/Procrustes/ridge,
  single/multi-teacher, GeoLink-style fusion, MRL/PQ, full backfill을 포함한 baseline·gate 고정
  ⑥ 기존 full-216 sealed 재사용 금지와 새 3지역 untouched test 명시.
- 무효/중단 조건: 행정 record를 feature와 label에 동시에 사용, 미래 published/retrieved record 누출,
  cache 호환성 없이 CKA만 개선, context residual이 없을 때 task가 붕괴, simple late fusion이나 affine
  bridge가 동일 결과, 모델마다 다른 acquisition으로 teacher 우월성을 주장, 제주 한 지역 결과를
  한국 전체로 일반화하는 경우.
- 결과 — 중심 질문: 광범위 후보를 **K-ALIGN: Provenance-Aware Compatible Distillation for Earth
  Models under Asynchronous Public Context** 하나로 수렴했다. Olmo v1로 `S0` stable bus를 만들고,
  Olmo v1.2+TerraMind teacher로 `S1`을 갱신하되 frozen `S0` gallery/head와 직접 호환되도록 한다.
  cutoff-valid 한국 공공 context는 train-time privileged signal과 별도 `r_context` residual로 사용해
  model release·새 EO 관측·record publication의 갱신 시계를 분리한다.
- 결과 — 네 estimand: `E_repr`(public teacher가 EO-only student를 강화), `E_compat`(S1 query→S0
  gallery/head 유지), `E_fusion`(추론 시 residual 정보 이득), `E_refresh`(full backfill 대비 residual/
  query-only 비용)를 별도 표로 고정했다. `E_repr+E_compat`가 모두 gate를 통과해야 통합 main paper로,
  한 축만 통과하면 Context Under Coverage 또는 Compatible Earth Bus로 다시 분리한다.
- 결과 — public alignment: VWorld PNU/geometry는 공간 anchor, SCL·GK2A/KMA는 관측품질,
  토지피복·FarmMap·DEM은 상태 auxiliary, BuildingHUB/EIA는 target과 독립인 auxiliary 또는 post-
  evidence, NGII/블라인드 EO 판독만 task label로 역할을 고정했다. 모든 token에 event/observed/
  published/retrieved time·prediction cutoff·coverage·raw SHA를 보존하고 duplicate-role/future sentinel을
  100% 차단한다.
- 결과 — 실행계약: `KOREA_ALIGNED_EARTH_BUS_EXPERIMENT.md`를 새 authoritative 문서로 만들고,
  S0/S0·S1/S1·S1/S0·S0/S1 네 query/gallery, 3지역 10k unlabeled·1,200 labels·외부 task 1개,
  baseline/loss/반증 control/7개 Figure·Table·A0–A6 queue를 고정했다. K-Context·EarthBus 문서는
  구성요소/넓은 후속 탐색으로 낮추고 `RESEARCH_STRATEGY.md`, `RESEARCH_EXECUTION_PLAN.md`,
  `PAPER_READING_LIST.md`, `README.md`, `STUDY.md` 카드 #43을 같은 우선순위로 정렬했다.
- 검증: 새 계약 section 0–12와 모든 문서의 authoritative link, STUDY #43을 구조 검사했다. 전체
  124테스트는 122통과·선택적 rasterio/geospatial 2개 skip, `compileall`, `git diff --check` 통과.
  이번 갱신은 실험 사전계약이며 한국 public alignment나 다른 backbone transfer 개선 수치는 아직 없다.

### 2026-08-23 — EO embedding transfer·robotics/simulation CVPR 트랙 확장 감사

- 계획: 사용자가 `earth_paper/dashboard.html`과 paper corpus를 함께 보고 OlmoEarth embedding을
  다른 backbone·policy·simulator로 전이하는 가능성, 로봇/시뮬레이션까지 연결한 CVPR main-track
  후보를 광범위하게 조사해 달라고 요청했다. earth_paper의 실제 연구 지도·아이디어·paper note를
  먼저 감사하고, 최신 primary-source의 representation translation/distillation, geospatial world
  model, embodied navigation, sim-to-real/domain adaptation, aerial/ground cross-view learning과 대조한다.
- 사전 성공 기준: ① `cache 좌표 정렬 / teacher→student 표현 증류 / downstream policy state`를
  분리 ② 기존 full-216 sealed 결과를 새 방법 선택에 재사용하지 않는 untouched-test 계약 ③ 각
  트랙마다 novelty, 최소 데이터, 강한 baseline, primary metric, promotion/kill gate, 현재 자산으로
  가능한 P0를 명시 ④ Earth paper corpus에서 연결되는 연구 계보를 paper ID/노트로 추적 ⑤ CVPR
  main 가능성과 robotics/remote-sensing venue 적합도를 과장 없이 순위화 ⑥ paper list와 실행 문서에
  실제 결정을 바꾸는 문헌·트랙만 반영.
- 무효/중단 조건: 높은 CKA를 transfer 성공으로 간주, 동일 sealed split을 본 뒤 고른 bridge를 같은
  test에서 평가, EO embedding을 로봇 state로 넣었다는 사실만으로 embodied contribution 주장,
  실제 paired aerial-ground/trajectory가 없는 synthetic-only demo, 서로 다른 입력·compute의 모델을
  한 표에서 우월 비교, robotics·simulation·public context·FL을 한 논문에 모두 주기여로 넣는 경우.
- 결과 — `earth_paper` 감사: 실제 파일은 `../earth_paper/dashboard.html`이고 `.hmtl`은 오타다.
  2026-08-10 생성 dashboard/INDEX는 185편·PDF 110편·완독 0·읽는 중 1·아이디어 3개다. robotics/
  simulation 전용 collection은 없고 연결 가능한 `G-08` Decision Transformer, `W-13` street-view,
  `K-05` dashcam flood, `O-02` onboard satellite, `O-09` climate simulation, `M-07` Spatial-Agent,
  `E-03/E-06` pooling/AlphaEarth note도 대부분 stub였다. 기존 corpus를 근거로 embodied claim을
  확장하지 않고 별도 문헌축이 필요하다고 판정했다.
- 결과 — 전이 가능성: AM-RADIO와 Theia는 서로 다른 frozen VFM의 지식을 한 compact/robot student로
  증류할 수 있음을 보여주므로 가능성은 있다. 다만 BCT/FCT/LCE가 compatibility를, MRL이 nested
  dimension을 이미 점유하므로 `feature 회귀 / cross-model cache / task utility / efficiency`를 분리했다.
  현재 full-216 sealed는 raw R@1=0·ridge 0.6973/0.6089라는 동기만 제공하며 새 bridge/student의 test는
  새로운 geographic-future split을 방법 선택 전에 hash-freeze한다.
- 결과 — CVPR 트랙 순위: ① 현재 exact-input·release audit와 직접 이어지는 `Compatible Multi-Teacher
  Earth Representation Bus`를 1순위 ② paired satellite–drone/ground와 field-side no-retrain을 묻는
  `Earth-to-Embodied`를 2순위 ③ Matryoshka edge/cloud 효율을 1의 보조축 ④ action trajectory가 생긴
  뒤 EO-conditioned latent world model ⑤ real policy fidelity가 생긴 뒤 satellite-to-ground simulation
  순으로 고정했다. GeoBridge·UniGeoRS·PAUL은 generic cross-view 질문을, DINO-WM·Navigation World
  Models·Vid2Sim은 latent planning/simulation 경계를 이미 점유하므로 단순 결합을 novelty에서 제거했다.
- 결과 — 실행 경계: Track 1의 4주 P0는 Olmo v1/v1.2+TerraMind Base, 공통 S2 view, 새 untouched
  region, linear/MLP/relational bridge, student 1개·256/768d로 제한한다. best-teacher task −1%p 이내,
  worst-group −2%p 이내, bus-native 대비 cross-family R@1/mAP 95%, latency/FLOPs 5× 또는 bytes 8×,
  backfill bytes 10×를 모두 통과해야 full paper matrix로 승격한다. paired image만 있으면 localization,
  action trajectory와 Success/SPL·collision이 있을 때만 navigation/world-model로 부른다.
- 산출물: `EMBEDDING_TRANSFER_CVPR_TRACKS.md`를 canonical 트랙 문서로 만들고,
  `PAPER_READING_LIST.md`를 먼저 읽을 26편과 새 section 9로 확장했다. `RESEARCH_STRATEGY.md` RQ8·
  EarthBus/Paper C·D, `RESEARCH_EXECUTION_PLAN.md`의 논문 경계, `README.md` index, `STUDY.md`
  카드 #40–#42를 같은 claim/gate로 정렬했다.
- 마찰/검증: in-app Browser의 URL 보안 정책이 로컬 `file://` navigation을 차단해 우회하지 않고
  dashboard/INDEX/IDEAS와 paper note 원문을 filesystem에서 감사했다. `PAPER_READING_LIST.md`의 우선
  독서표 26행·section 1–9·STUDY 카드 #40–#42를 구조 확인했고, 전체 124테스트는 122통과·선택적
  rasterio/geospatial 2개 skip, `compileall`, `git diff --check`까지 모두 통과했다. 이번 작업은
  연구 설계·문헌 감사이며 새 GPU/model 성능 실험은 실행하지 않았다.

### 2026-08-23 — 한국 공공데이터 조건부 EO 표현 강화 메인 논문 재설계

- 계획: full-216 감사가 단순 cross-release cache identity를 기각했으므로, 임베딩 보정 자체를
  목적화하지 않고 한국 공공데이터가 영상-only 모델의 관측 모호성을 실제로 줄이는 조건을 최신
  주류 학회·공식 코드와 대조한다. 기존 `PAPER_READING_LIST.md`, `K_EVIDENCE_SHIFT_BENCHMARK.md`,
  `RESEARCH_STRATEGY.md`, `RESEARCH_EXECUTION_PLAN.md`의 범위 중복을 감사하고, workshop이 아닌
  main-track 수준의 한 중심 질문·최소 모델/ablation·라벨/split·통계·kill gate로 다시 고정한다.
- 사전 성공 기준: ① 최신 primary source에서 EO 다중모달 fusion·metadata/context conditioning·
  missing-modality·전이/강건성 baseline을 확인 ② 공공데이터를 입력·약한 라벨·근거·coverage로
  분리하고 source leakage를 차단 ③ 영상-only 대비 정보 이득을 EO embedding·task prediction·
  selective decision으로 분해 ④ 현재 보유 데이터로 가능한 P0와 새 라벨/자료가 필요한 P1을 분리
  ⑤ 평균 정확도뿐 아니라 high-cloud·missing-evidence·지역/시간 OOD·calibration·AURC와
  paired spatial bootstrap을 사전 고정 ⑥ 기존 paper list에 실제 실험 결정을 바꾸는 문헌만 추가.
- 무효/중단 조건: 행정 사건을 입력과 정답에 동시 사용, 같은 API의 no-match를 음성 라벨로 사용,
  label 0 release geometry를 downstream 정확도로 포장, 모델마다 다른 입력을 주고 pretraining
  우월성으로 해석, 제주 단일 split의 작은 평균 개선만으로 한국형 Earth Intelligence를 주장,
  실제 사일로 없이 FL을 주기여로 추가하는 경우.
- 결과 — 경쟁 경계: NeurIPS 2025 GeoLink가 127만 EO–OSM pair의 region/object alignment와
  object-patch fusion을, ECCV 2024 MMEarth·SatMIP와 ICML 2025 Galileo가 multimodal·time/location
  supervision을 이미 점유함을 공식 proceedings에서 확인했다. 따라서 단순 `EO+지도/날씨`를
  novelty에서 제거하고 동적 공식 record의 시점·자연 누락·지연·충돌을 핵심 빈칸으로 고정했다.
- 결과 — 중심 질문: `E_repr`(train context→EO-only student), `E_fusion`(test EO+context),
  `E_decision`(예측 후 evidence/abstention)을 분리한 `K_CONTEXT_FUSION_EXPERIMENT.md`를 만들었다.
  제안 방법은 동결 Olmo v1.2의 provenance-aware 경량 adapter와 privileged distillation이며,
  location/year-only, context-only, STACK/TOKEN-FUSE, GeoLink-style fusion, native multimodal ceiling을
  같은 split에서 비교한다.
- 결과 — 실행 계약: 3지역 최소 10,000 unlabeled parcel/site-years의 P0와 권장 1,200 독립 label의
  P1을 분리했다. shuffle/time-shift/missingness-only/future-leak sentinel, 지역·미래연도·cloud·
  자연 coverage, site/event clustered bootstrap과 promotion/kill gate를 실험 전에 고정했다.
  P0에서 EO-only student +2%p 또는 label 20% 절감이 없거나 simple fusion을 이기지 못하면 대규모
  GPU/라벨 실험을 중단한다.
- 결과 — 문서 통합: `PAPER_READING_LIST.md`의 우선 독서를 18편으로 갱신하고 GeoLink, MMEarth,
  Galileo, SatMIP, MMEarth-Bench, Rao–Rolf를 역할별로 추가했다. `K_EVIDENCE_SHIFT_BENCHMARK.md`,
  `RESEARCH_EXECUTION_PLAN.md`, `RESEARCH_STRATEGY.md`, `README.md`를 새 Paper A와 full-216 실측
  상태에 맞췄고 `STUDY.md` 카드 #38–#39에 estimand 분리와 자연 누락 교훈을 남겼다.
- 보류 — 실행: 현재 독립 supervised label이 0이고 source별 publication/observed time 계약이 먼저라
  GPU를 추가로 돌리지 않았다. 다음 실제 작업은 C0 source-role/cutoff manifest와 C1 3지역 frame이며,
  이것이 통과한 뒤에만 C2 P0 adapter headroom을 GPU0에서 실행한다.

### 2026-08-23 — GPU0 full-216 릴리스·캐시 호환성 감사

- 계획: 사용자가 GPU0를 강하게 사용해 실제 연구 산출물을 만들라고 요청했다. 독립 GT 없이도
  엄밀히 닫을 수 있는 가장 큰 현재 자산인 제주 54 spatial windows×4 site-years 전체에서
  OlmoEarth v1/v1.2 release drift와 old/new cache compatibility를 측정한다. 먼저 216 입력 전체
  content hash와 batch-throughput 동등성 gate를 통과하고, 그 뒤에만 GPU0에서 두 release를
  순차 실행한다. GPU1의 다른 프로젝트는 건드리지 않는다.
- 사전 성공 기준: ① 216 site-years×12 periods의 tensor/metadata/items/window file을 SHA-256으로
  고정하고 COMPLETE marker 생성 ② 새 full audit view 216/216·2,592 input symlink, 원본 수정 0
  ③ GPU0 selected-UUID idle gate·출력 경로 비존재·예상 저장공간 확인 ④ batch1 대비 후보
  batch size의 output grid/mask/identity 100%와 pooled numeric drift 허용오차를 사전 고정하고 smoke
  통과 ⑤ full output 216×2=432, mtime/SHA/config/checkpoint/log/COMPLETE 100% ⑥ 54 spatial
  cluster 단위 통계와 공간 calibration/test split을 사용한 no-bridge·orthogonal bridge·linear/ridge
  cache baseline ⑦ 새 분석을 재실행해 결정성 확인 ⑧ label-free이므로 accuracy·한국 일반화·
  구름 강건성 주장은 금지.
- 무효/중단 조건: 전체 입력 hash 1건이라도 불일치, GPU0 active process, 기존 output/stale file,
  batch smoke 수치 drift가 사전 허용오차 초과, output completion <100%, spatial split leakage,
  calibration과 evaluation을 같은 공간창에서 수행, GPU 사용 자체를 성과로 해석하는 경우.
- 실행 완료: 54위치×4년의 216 site-years를 v1/v1.2 각각 GPU0에서 순차 실행했다. 입력
  5,616파일·56,684,540,847 bytes와 출력 432파일·105,591,415,295 bytes를 전수 hash로 닫았고,
  두 릴리스 모두 216/216, 768×256×256 float32, EPSG:32652, finite·usable·nonzero 100%였다.
  실행 계약과 선택 장치는 GPU0 UUID에 고정됐고 GPU1은 이 추론의 선택 GPU가 아니었다.
- 자원 실측: 사전 batch gate가 선택한 batch 8/workers 4에서 v1은 3,756.12초·55.26 crops/s
  (GPU0 util p50/p90 88/89%, peak 4,291 MiB), v1.2는 2,250.12초·92.25 crops/s
  (72/77%, 2,719 MiB)였다. 이 exact workload의 end-to-end 처리량에서 v1.2가 약 1.67배 빨랐다.
- 봉인 분석: calibration 30위치/120건에서만 bridge와 ridge alpha를 맞추고, smoke에 노출되지 않은
  sealed 16위치/64건에서만 headline을 계산했다. same-release native R@1은 양방향 1.0이지만
  no-bridge cross-version R@1은 0.0이었다. Procrustes는 v1.2→v1 0.4910·반대 0.4360,
  최선 affine ridge도 0.6973·0.6089에 그쳐 사전 0.95 gate를 8/8 실패했다.
- 해석: sealed pooled CKA 0.9786·거리 Spearman 0.9525와 동일 token raw cosine 평균 −0.0086이
  동시에 관찰됐다. 216개 패널의 관계 구조는 남지만 좌표 identity가 깨진 구조적 release
  shift이며, `full_cache_compatibility_promoted=false`다. 사전 등록한 네 bridge 모두
  representation proxy를 승격하지 못했으므로 downstream task 평가 전에는 운영 cache 정책을
  결정하지 않는다.
- 약점과 다음 gate: 라벨 0·제주 단일 grid·legacy 입력만 있으므로 정확도, negative transfer,
  구름/공공데이터 효과, 변화탐지, 한국 일반화는 미검증이다. 이번 sealed 결과를 이미 보았으므로
  nonlinear bridge/distillation은 새 untouched geographic split을 먼저 동결한 뒤에만 평가한다.
  상세 증거와 금지 주장은 `artifacts/release_audit_full216_v1/README.md`에 고정했다.
- 사전 기준 보정: full headline analyzer는 output raster를 읽기 전에 one-time
  `PREANALYSIS_LOCK`을 쓰고 sealed 결과를 한 번만 여는 계약으로 강화했으므로, 사전 기준 ⑦의
  두 번째 sealed 재실행은 하지 않았다. 대신 분석 코드·NumPy/BLAS/rasterio/GDAL runtime을
  시작/종료에 재검증하고 metric 회귀 테스트를 통과시켰다. 같은 sealed 결과의 반복 실행을 새
  독립 증거로 세지 않는다.
- 증거: paired evidence SHA `9931cddf…e2f1`, analysis summary SHA `56030ea0…185d`.
  로컬 compact evidence의 marker·상호 SHA·216 pair·432 output closure·CSV 행 수 핵심 검사를
  18/18 통과했다. 서버의 약 98 GiB raw GeoTIFF는 삭제하지 않고 유지한다.
- 종료 상태: 모든 inference·finalizer·analyzer가 완료 marker를 쓴 뒤 종료된 것을 확인했다. 최종
  재접속 시 `nx tunnel up`은 `h200-dev` 세션이 더 이상 실행 중이 아니라고 보고했다. 작업을
  재시작하지 않았으며 raw는 영구 저장소 경로에 남긴 상태다.

### 2026-08-23 — 검증 가능한 단계만 진행하는 verified-only gate

- 계획: 사용자의 요청에 따라 연구 아이디어를 넓히거나 대규모 GPU 작업을 먼저 시작하지 않는다.
  현재 산출물과 다음 후보인 BestClear 대표 8-window gate를 독립적으로 재감사하고, 입력·코드·
  성공 기준이 모두 고정된 최소 단계만 실행한다. 통과하지 못하면 실패 원인과 필요한 선행조건만
  남기고 멈춘다.
- 사전 성공 기준: ① 기존 release 결과의 COMPLETE/SHA를 재검증 ② GPU/서버 작업과 무관한
  deterministic test를 로컬에서 통과 ③ BestClear 검증 표본은 기존 score/사람 판독을 보지 않고
  고정되었음을 증명 ④ 96기간을 `changed`/`valid no-op`로 전부 설명하고 contaminated 4건은 각각
  최소 1기간의 array hash 변화 ⑤ cloud proxy 개선과 zero/nodata 악화가 함께 보고됨
  ⑥ RGB와 SCL 원자료 육안 확인
  ⑦ 어느 하나라도 불충분하면 216×2 full run·정확도·구름 강건성 주장을 시작하지 않음.
- 무효 조건: golden window 한 건을 8건 일반화로 재사용, 기존 후보 순위로 검증 표본을 선택,
  설정명 차이를 입력 차이로 간주, output metric만 보고 source item/pixel hash를 생략, GPU를 채우기
  위해 검증되지 않은 대규모 실행을 추가하는 경우.
- 감사 결과 — 기존 증거: 로컬 `preflight.json`이 과거 `ready=false` 보류본인데 문서는 실행본으로
  설명하는 불일치를 발견했다. 서버 실행본과 launcher 첫 JSON의 SHA
  `b63c8c60e7314fb77be579657a5a0a5c5e49bcb277c4ee69676fea249f1a2a2b`가 같은 것을 확인하고
  로컬 파일을 실제 `ready=true`, `selected_gpu=0` 실행본으로 교체했다.
- 구현: `verify_olmo_release_bundle.py`를 추가해 preflight→checkpoint/exact input→두 config/log→
  2×8 output inventory→COMPLETE→analysis marker를 fail-closed로 결속했다. 로컬 raw가 없을 때는
  `PARTIAL_VERIFIED`, `--require-raw`에서 하나라도 없거나 다르면 `FAILED`이며 기존 결과를
  overwrite하지 않는다.
- 검증 결과: 서버에서 exact input 208 + checkpoint 4 + output GeoTIFF 16 = 228 raw files,
  7,851,565,383 bytes를 전부 재해시했다. 758/758 checks, failure/missing 0으로
  `FULL_EVIDENCE_VERIFIED`; verification SHA는
  `b543f1b4b43750d510ff36a51e0a2f80ac9b258dc7a29fe8411a9bdd5dc0d34f`다.
- 결정성: 기존 analyzer를 새 디렉터리에서 다시 실행해 `analysis_summary.json` SHA
  `7bfeac8d…`와 `per_window_metrics.csv` SHA `8a25a7cf…`가 기존 파일과 byte-identical임을 확인했다.
- BestClear 중단 판정: 현재 코드는 1-window×4-period에 하드코딩됐고 `max_matches=4`, stale-output
  거부·선택 item sidecar·SCL/reflectance 96쌍 SHA·grid/time/mask·replay 결정성 테스트가 없다.
  따라서 이번에는 materialize/GPU 실행을 시작하지 않았다. 기존 golden window는 positive
  regression control일 뿐 새 stress 분모에 넣지 않는다.
- 다음 승격 계약: 사람/공공근거를 보지 않고 legacy bad-proxy 양끝에서 이미 동결한 8 site-year
  (2023/2026, clear 4/contaminated 4, 7 spatial clusters)를 대표 표본이 아닌 label-free stress set으로
  쓴다. 8×12=96기간 완료, 선택 trace·SCL/reflectance SHA, 96/96 grid/time/mask, contaminated 각
  ≥1 changed period, median bad-proxy ≥10% 감소, zero/mask 악화 ≤1%p, 고정 RGB, 2건 replay
  hash 100%를 모두 구현·통과할 때만 다음 계산으로 간다.
- 최종 검증: 로컬 `unittest` 68개 중 66 pass·2 optional dependency skip, 서버 신규 verifier
  3/3 pass, `compileall`, `git diff --check`, secret-like value scan 0건, 로컬 verification marker SHA,
  preflight 실행본 SHA, 재분석 JSON/CSV byte identity를 모두 재확인했다.

### 2026-08-23 — GPU0 전용 P0 실측·연구 실행표 구체화

- 계획: 사용자가 GPU0 전체 사용을 허용했으므로 아이디어 확장보다 이미 고정한 exact-input
  OlmoEarth v1/v1.2 release smoke를 GPU0에서 실측하고, 그 결과와 무관하게 1주·2주·6주 단위의
  데이터·모델·지표·표본·중단 기준을 실행 명령 수준으로 구체화한다.
- 사전 성공 기준: ① GPU0의 기존 프로세스를 확인하고 사용자 작업을 임의 종료하지 않음
  ② exact input/checkpoint SHA가 기존 manifest와 100% 일치 ③ 8 sample×2 release=16 output과
  COMPLETE marker 생성 ④ output SHA/mtime·sample/input/grid/mask gate 통과 ⑤ 7 spatial cluster를
  표본 단위로 둔 CKA·shift-null·neighbor/rank 진단 생성 ⑥ accuracy·negative transfer 주장은
  독립 라벨 전까지 금지 ⑦ 다음 6주의 표/figure·owner input·promotion/kill gate를 문서로 고정.
- 무효 조건: GPU0의 다른 프로젝트 프로세스를 강제 종료, active output을 재사용해 stale 결과를
  새 실행으로 오인, 8 label-free smoke를 정확도나 한국 일반화로 해석, 픽셀을 독립 표본으로 CI를
  계산, BestClear 미완료 상태에서 input effect와 release effect를 비교하는 경우.
- 결과 — 자원: H200 세션과 영구 저장소는 정상. GPU0은 0 MiB로 비어 있었고 GPU1은
  `knee-proj`가 약 62.6 GiB·70%를 사용 중이었다. 실행기의 전역 active-process gate가 GPU1 때문에
  GPU0까지 막는 결함을 찾아 index→UUID 기반 selected-device gate로 고쳤고 GPU1은 건드리지 않았다.
- 결과 — 실측: exact input/checkpoint SHA preflight `ready=true`, 8 sample×2 release=16 output과
  COMPLETE marker 생성. v1 202.633초, v1.2 196.830초. output SHA·mtime·sample/input/grid/mask
  검증을 모두 통과했다.
- 결과 — 표현 감사: pooled linear CKA 0.981, pairwise distance Spearman 0.889, top-1/2 neighbor
  overlap 0.75/1.00. 반면 per-window spatial CKA는 평균 0.427(0.133–0.828), shift-null 초과분
  평균 0.247이었다. 전역 이웃 구조 보존과 국소 공간 표현 이동이 동시에 나타났지만 8 label-free·7
  cluster 기술통계이므로 정확도·구름 강건성·한국 일반화·cache 호환성은 여전히 미검증이다.
- 결과 — 실행계획: `RESEARCH_EXECUTION_PLAN.md`에 Paper A/B, sealed probability test 300,
  train/active pool 300, double-review 120, 3개 matched-input subtrack, 5-seed/label-fraction 계약,
  1·2·6주 표·figure·promotion/kill gate와 GPU0 운용 원칙을 고정했다. J0/J1은 완료, J2 full
  432-output은 사용자가 연구를 재개할 때까지 보류했다.
- 검증: 로컬 `unittest` 65개 중 63 pass·2 optional dependency skip, `compileall`,
  `git diff --check` 통과. 서버 release/raster 계약 16/16 pass. 서버 전체 discover는 pilot 입력
  artifact를 서버에 복사하지 않아 50개 중 1 setUp error·1 optional skip이었으며 코드 실패로
  세지 않았다.
- 다음: 연구 재개 시 full legacy 432개부터 무작정 돌리지 않고, 먼저 BestClear 대표 8 window의
  pixel-hash gate와 sealed 3지역 표본·이중판독 protocol을 닫는다. 그 뒤에만 216×2와 다중모델
  frozen probe를 GPU0 queue에 올린다.

### 2026-08-23 — K-EvidenceShift 실행화 0주차: Jeju pilot manifest·누수 gate

- 계획: 사용자가 CVPR형 transfer/active-label/evidence 연구를 실제로 함께 진행해 달라고 했으므로,
  기존 설계 문서를 첫 실행 가능한 benchmark 자산으로 내린다. 현재 14후보·공공 API v3·사람
  pre-review·OlmoEarth 관측 provenance를 결합하되, 성능표를 꾸미거나 368개를 라벨된 표본으로
  세지 않는다. 동시에 H200의 현재 모델/데이터 자산을 read-only 감사해 첫 paired model run을
  고른다.
- 사전 성공 기준: ① 기본 단위를 `(site, t0, t1, input revision, evidence snapshot)`으로 고정
  ② `visual_change / official_event_supported / evidence_available / cause`와 label provenance를 분리
  ③ t1 뒤 행정사건·사람검수·evidence snapshot을 prospective feature로 쓰지 못하게 role/time gate
  ④ 같은 site·parcel·event가 split을 넘지 않는 group/buffer audit ⑤ cloud/evidence/PNU-conflict
  strata와 현재 coverage를 실제 수치로 출력 ⑥ model/adaptation/input matrix는 실행 전 빈 결과와
  promotion gate를 명시 ⑦ 결정적 JSON/CSV·SHA·CLI·단위테스트·문서 갱신.
- 무효 조건: API no-match를 `visual_change=0`으로 변환, assistant pre-review를 독립 ground truth로
  승격, 제주 14건 결과를 한국 성능으로 일반화, 픽셀/중복 후보를 독립 표본으로 집계, post-t1
  evidence leakage, 서로 다른 native modality를 paired-input 성능으로 비교, 실제 silo 없이 FL을
  핵심기여로 구현하는 경우.
- pilot 결과: build `0c9968ef33d47027`로 14 records를 13개 500 m spatial group·8개 source-window
  group·1개 scene component로 고정했다. 독립 human GT 0, transition-aligned 공식사건 0/14, cause
  0/14, PNU conflict 1, 보류 14/14다. assistant 판독은 pre-annotation으로만, t1 뒤 EO frame은
  `future_after_t1_review_only`로, 사후 API snapshot은 prospective input 불가로 고정했다. 동일
  scene graph 때문에 현재 pool로 scene-disjoint cloud test를 만들 수 없고 sealed test도 없다.
- 누수/승격 보강: upstream 전체기간 BuildingHUB 정렬을 신뢰하지 않고 각 후보 t0→t1 exact PNU
  event를 재계산한다. EIA는 transition date가 없으면 spatial context에만 둔다. API v3 COMPLETE·raw
  SHA, assistant manifest protocol hash, 좌표·관측일, 2023/24/25 토지피복 semantic success를 모두
  검증한다. 성능표 gate에는 3지역·독립 label 300·이중판독 120·sealed probability test·동결 common
  input contract·4 paired baselines·4 immutable checkpoints를 명시했다.
- release P0: 제주 54 windows×4년=216 site-years와 인접연도 162 events를 metadata manifest로
  만들고, label/evidence-free smoke 8건의 96 S2 layer·208 tensor/metadata 파일을 exact SHA로
  고정했다. 원본은 수정하지 않고 symlink audit view를 만들었다. v1/v1.2는 양쪽 모두 12 S2 bands,
  patch 4, legacy timestamps, 같은 crop/head로 고정했으며 smoke 예정 output은 16개다.
- checkpoint 증거: v1 commit `93589e2d…`, weights SHA `551c1cc5…`; v1.2 commit `581aa9ba…`,
  weights SHA `57f7b66f…`. exact-input manifest SHA는 `bc149353…`, checkpoint manifest SHA는
  `f325e3f6…`이며 2026-08-23 서버 preflight에서도 다시 일치했다.
- 분석 계약: smoke 8건은 7개 spatial cluster다. output 16개를 `sample_id`·input bundle·cluster로
  exact-pair하고 SHA/mtime·expected grid·nodata mask가 하나라도 다르면 실패한다. raw cross-version
  cosine과 smoke-fit Procrustes는 금지하고, raw/row-normalized spatial CKA·toroidal-shift null,
  pooled Euclidean-distance Spearman, k=1/2 chance-corrected neighbor overlap과 7-cluster LOO만
  descriptive metric으로 사전 고정했다.
- 서버 판정: `./bin/nx`의 실제 `~/DongDong/ai_projects/h100-setup` 탐색을 고쳐 status/doctor/원격
  shell/영구저장소는 정상이다. 마지막 preflight에서 두 H200에 다른 `knee-proj` python PID
  1230427(68,990 MiB), 1248837(62,562 MiB)가 있어 `ready=false`; 실행기는 GPU 선점을 거부했고
  실제 v1/v1.2 output은 시작하지 않았다.
- 마찰: 첫 HF resolver가 `Path.resolve()`로 snapshot symlink를 blob까지 풀어 commit provenance를
  잃었다. raw snapshot path에서 revision을 읽고 blob SHA는 별도 검증하도록 고쳤으며 회귀 테스트를
  추가했다. local→server tar의 macOS provenance xattr 경고는 파일 업로드를 막지 않았다.
- 최종 QA: 로컬 전체 64 tests 중 62 pass·선택적 geospatial/raster dependency 2 skip. 서버에서는
  rasterio 1.4.4로 exact sample/input/grid/mask와 synthetic 16-GeoTIFF 분석을 포함한 release 계약
  15/15가 통과했다. `compileall`,
  `git diff --check`, pilot output 8파일 SHA/COMPLETE, 신규 benchmark/release artifact secret-like
  pattern 0건을 통과했다. 실행 계약은 `OLMO_RELEASE_AUDIT_P0.md`에 고정했다.
- 다음: GPU가 빌 때만 8×2 paired release smoke 실행 → output SHA·embedding drift·neighbor overlap
  계산 → 216 site-years full hash/audit → SCL BestClear 216 materialize로 input×release 2×2 완성.
  정확도·negative transfer는 sealed 300 + 별도 active/train 300, 그중 double review 120과 공통 입력
  계약이 생길 때까지 보류한다. FL은 실제 비반출 기관 silo 3곳 전에는 구현하지 않는다.

### 2026-08-22 — VWorld 재승인 후 257점 지적 snapshot·공식근거 재결합

- 계획: 사용자가 VWorld API key 허용설정을 갱신했으므로 기존 `INCORRECT_KEY`를 그대로 덮어쓰지
  않고 새 snapshot에서 대표점 1건을 먼저 probe한다. 본문 `status=OK`와 feature/PNU를 확인한
  경우에만 위치화 오름 243점과 기존 변화후보 14점을 확장한다. 성공 후 기존 BuildingHUB·EIA·
  토지피복·GK2A snapshot과 결정적으로 재결합하고 dashboard의 고정 실패 문구를 실제 상태 기반으로
  바꾼다.
- 사전 성공 기준: ① credential/domain을 artifact·로그에 노출하지 않음 ② HTTP 200과 VWorld
  `OK/NOT_FOUND/API error`를 분리 ③ 257 요청의 target별 PNU·주소·response hash 보존 ④ 기존
  8,794 BuildingHUB event와 exact PNU/time join 재계산 ⑤ 대표 지적 필지를 오름 경계나 변화원인으로
  승격하지 않음 ⑥ 원본 v1 실패 snapshot 보존, 새 combined snapshot과 테스트·secret scan 통과.
- 무효 조건: 대표점 실패인데 전수 요청, `NOT_FOUND`를 key 오류나 비개발 음성으로 해석, 같은
  PNU를 공유한 여러 오름점을 서로 독립 필지로 집계, 변화 관측 뒤의 건축사건을 원인 B급으로 승격,
  dashboard에 과거 `VWorld 실패` 또는 고정 `0/14`를 남겨 실제 JSON과 불일치시키는 경우.
- 실행 결과: `.env`의 실제 key 값은 읽거나 출력하지 않고 등록 domain `http://localhost`만 고정했다.
  대표 후보 1점이 HTTP 200·본문 `status=OK`·feature 1을 반환한 뒤에만 257점으로 확장했다.
  VWorld는 HTTP 257/257, `api_success` 256·`api_no_features` 1·API 오류 0이며 후보 14/14,
  위치화 오름 242/243의 대표 PNU를 확보했다. `JJ-OREUM-190`의 `NOT_FOUND`는 비개발 음성이
  아니라 point coverage 누락으로 남겼다.
- 결합 결과: v7.6의 비-VWorld 206응답과 새 VWorld 257응답을 네트워크 호출 없이 v3로 결합해
  전체 HTTP 463/463, semantic 성공 456·유효 무항목 1·과거 GK2A 오류 6을 보존했다. VWorld
  feature 256개는 고유 PNU 235개이며, 14개 PNU를 35점이 공유한다. 따라서 256개 독립 필지로
  집계하지 않는다.
- 공식근거 판정: 후보 exact PNU BuildingHUB 사건은 `oreum_v6_r04` 1건이지만 허가 2026-07-06·
  착공 2026-07-22로 마지막 EO 관측 2026-05-06 뒤라 시간정렬은 0건이다. `oreum_v6_r08`은
  dated FarmMap PNU `5013025324202000000`과 current VWorld PNU `5013025324201990000`이 달라
  어느 한쪽도 덮어쓰지 않고 출처 충돌로 보존했다. 결과적으로 A/B급 원인 근거는 **0/14·0/368**,
  후보 14/14 보류를 유지한다.
- 구현/검증: `OK/NOT_FOUND/error`와 feature count를 함께 검증하고, VWorld target 257개 완전성·
  단일 feature 계약·dual-anchor 보존·입력 snapshot SHA/시각·request identity와 raw response
  lineage·완주 marker를 v3에 추가했다. 원본 실패 v1, 성공 VWorld-only, 결함이 발견된 v2를 모두
  보존하고 메인 보드 링크만 v3로 승격했다.
- 최종 QA: `python3 -m unittest discover -s tests -v` 34개 중 33 통과·선택적 geospatial 의존성
  1개 skip, `compileall`, `git diff --check`, artifact secret scan 0건을 통과했다. `COMPLETE.json`의
  requests/run-summary SHA와 실제 파일 hash, merge script SHA도 일치한다. localhost 8766의 메인·
  API 보드를 새로고침해 463/463, VWorld 256/1/0, PNU 충돌 1, 14/14 보류 문구를 DOM에서 확인했다.

### 2026-08-22 — Korea Temporal Evidence Benchmark × GeoFM 전이·연합학습 논문 설계

- 계획: 현재 WorldShift×ModelShift·K-Earth selective evidence 프로그램과 사용자의
  `OlmoEarth + 한국 실시간/시계열 공공데이터` 아이디어를 하나의 제출 가능한 논문 설계로
  합친다. 최신 GeoFM benchmark·한국/비한국 시계열 데이터·구름/입력 shift·모델 릴리스
  호환성·transfer/domain adaptation·federated learning 문헌을 공식 원문으로 다시 감사한다.
- 사전 성공 기준: ① “공공데이터를 붙이면 정확도 상승”을 입력·표현·예측·결정 단계의 서로 다른
  개선으로 분리 ② 공개 benchmark 재사용/새 benchmark 필요성을 증거로 판정 ③ OlmoEarth 외
  최소 3 model family와 supervised/non-FM baseline 포함 ④ spatial/temporal/region/model-release
  leakage 없는 split·지표·ablation·통계 검정 고정 ⑤ federated learning은 실제 기관 분산·privacy
  조건과 중앙학습 상한을 갖출 때만 핵심기여로 승격 ⑥ CVPR형 최소 논문과 박사 프로그램 확장을
  분리하고 8–12주 실행·중단 gate를 제시.
- 무효 조건: 한국 공공데이터가 존재한다는 사실만 novelty로 주장, API 상태지도/행정기록을
  ground truth로 오인, 서로 다른 센서·해상도·시점·라벨 budget을 불공정 비교, 제주 한 지역의
  개선을 한국 전체/실시간으로 일반화, 모델명만 늘리고 paired input 통제를 잃거나, 실제 다기관
  silo 없이 simulated federated learning을 개인정보·현장협업 기여로 과장하는 경우.
- 산출물: canonical 논문 설계·benchmark data card, `RESEARCH_STRATEGY.md`와
  `PAPER_READING_LIST.md`의 최신 문헌/모델 비교 갱신, 실행 우선순위와 CVPR readiness 판정.
- 결과 — 질문 수렴: CVPR형 중심 질문을 **“글로벌 GeoFM은 한국의 어떤 지역·시기·센서·구름
  shift에서 matched scratch보다 negative transfer를 만들며, 제한된 target label을 어디에 추가해야
  worst-group 실패를 가장 빨리 줄이는가?”**로 고정했다. 공공근거 누락·선택적 보류·release
  continuity는 같은 자산을 쓰는 Paper B/박사 프로그램으로 분리했다.
- 결과 — canonical 설계: `K_EVIDENCE_SHIFT_BENCHMARK.md`를 만들고 site-event 단위 schema,
  `visual_change / official_event_supported / evidence_available / cause` 분리, source 역할과 temporal
  leakage 금지, spatial/temporal/cloud/evidence/release split, matched-input/native-ceiling model track,
  transfer effect CI와 active-label acquisition baseline·negative controls·promotion gate를 고정했다.
- 결과 — 비교군: 최소 publishable set은 task-specific U-Net/ViT scratch + generic vision +
  OlmoEarth v1/v1.2 + Prithvi-EO-2.0 + CROMA 또는 TerraMind로 잡고, AnySat은 확장축으로 뒀다.
  GEO-Bench-2/PANGAEA harness, EarthShift OOD protocol, AllClear/CloudSEN12 cloud strata를 재사용한다.
- 결과 — active acquisition: random/층화/uncertainty/release·cross-family disagreement/k-center/
  log-det/CLUE/cost-aware spatial baseline과 동일 pool·oracle·budget·5-seed 계약을 뒀다. 제주 8/8 cloud
  공통오류 때문에 quality gate와 spatial dedup을 query 전에 강제하며, adaptive set과 모집단 추론용
  확률표본을 분리했다. PDE의 beta·advection–diffusion·모호성선·D-opt 해석은 Earth에 직접 이식하지
  않고 그룹별 transfer effect·경험적 disagreement·diversity baseline으로 새로 정의했다.
- 결과 — FL/CVPR 판정: 공개 API를 시도별로 나눈 것은 실제 silo가 아니므로 FL은 독립 기관 3곳,
  반출불가 데이터, 실제 분산실행과 중앙화 불가 근거가 모두 생길 때만 승격한다. CVPR 2027 공식
  논문마감은 2026-08-22 현재 미발표이므로 내부 10월 31일 초록 동결·11월 6일 제출 가능본을 쓴다.
  현재 상태 그대로는 main 가능성이 낮고, 6주차까지 3지역·multi-model baseline·300-label audit·
  재생성 계약을 못 만들면 IGARSS/TMLR/EarthVision 경로로 전환한다.
- 검증/문헌: `PAPER_READING_LIST.md`에 Copernicus-Bench, REOBench, AllClear/CloudSEN12, CLUE,
  RIPU, Active Learning under Label Shift, Active-DDC, FedRS-Bench/FedSense/FedAG를 추가했고,
  `RESEARCH_STRATEGY.md`, `EARTHROUTE_PROGRAM_NOTE.md`, `README.md`, `STUDY.md`의 질문·로드맵·개념
  카드를 같은 경계로 정렬했다. 문헌 수치는 primary paper/proceedings·공식 repo의 저자 주장이고,
  우리 환경 재현 전에는 실험 사실로 승격하지 않는다.

### 2026-08-22 — 제주 공공 API 실제 수집·시공간 결합 v1

- 계획: 확보된 credential로 전국 dump가 아닌 제주 bounded snapshot을 만든다. 코드에 source별
  adapter와 secret-safe request manifest를 먼저 쓰고, VWorld 연속지적도는 등록된 VM에서,
  BuildingHUB·GK2A·EIA는 승인/응답이 허용되는 범위에서, 환경부 토지피복은 공개 WMS로 probe한다.
  원본 응답을 보존한 뒤 PNU·geometry·captured/event time으로 기존 오름 368·OlmoEarth 시계열과
  결합하고, 접근 실패도 삭제하지 않고 source coverage 결과로 남긴다.
- 사전 성공 기준: ① secret/전체 query string이 로그·manifest에 없음 ② source별 1건 이상 실제
  응답 또는 구조화된 실패 증거 ③ request hash·retrieved_at·HTTP/content type·schema/CRS·row/tile
  수 보존 ④ VWorld PNU/geometry, BuildingHUB 사건일, GK2A 관측시각, EIA 사업 polygon, 토지피복
  연도 중 실제 제공 필드만 공통 event model로 정규화 ⑤ 기존 368 denominator와 원인 10% gate 유지
  ⑥ adapter 단위테스트·재실행 결정성·dashboard 결과 확인.
- 무효 조건: 전국 무제한 수집, 페이지네이션/시간범위를 모른 채 no-match를 음성으로 해석,
  key/서비스 URL query 노출, 서로 다른 좌표계의 직접 교차, 승인된 API를 데이터 확보 완료로 표현,
  상태지도·기상자료만으로 개발 원인을 확정하는 경우.
- 실행 순서: 공식 API 계약/endpoint 확인 → adapter·fixture test → 최소 probe → 제주 범위 snapshot →
  공통 schema·coverage audit → 기존 registry 결합 → dashboard/문서/Worklog 갱신.
- 결과: secret-safe adapter와 bounded collector를 구현하고 **207개 HTTP 응답**을 원본·SHA-256·
  secret 없는 request hash로 보존했다. 의미상 성공은 200건이다. BuildingHUB는 기존 PNU에서 나온
  제주 45 법정동의 2023–2026 기본개요를 **111페이지·8,794행** 전부 받았고, 철거·멸실 endpoint는
  같은 범위 45요청 성공·0행이었다. EIA WFS는 제주 bbox **13 polygon**, 환경부 토지피복은
  OlmoEarth 14후보×2023–2025 **42 PNG**, 최신 GK2A는 2 km **127,040 grid값**을 반환했다.
- 결합 결과: 기존 PNU 58개 중 BuildingHUB exact PNU가 있는 것은 **9개**였다. `oreum_v6_r08`은
  기존 FarmMap PNU로 같은 법정동 건축사건 77건까지 좁혀졌지만 exact PNU는 0건, EIA와 직접 겹친
  14후보도 0건이었다. 따라서 새 A/B급 원인 corroboration은 **0/14**, 14/14 보류를 유지한다.
  데이터량 증가는 원인 판정을 늘리기보다 “왜 보류하는가”를 출처별로 구체화했다.
- 실패/coverage: VWorld 대표점은 로컬과 H100 VM 양쪽 모두 HTTP 200 본문
  `INCORRECT_KEY`로 실패해 257점 반복요청을 중단했다. GK2A는 OlmoEarth 과거 6관측일을 최근
  2일 제한으로 거부해 역사 cloud audit에 사용할 수 없고 최신 grid만 보존했다. NGII 항공사진은
  수동 신청 채널이라 자동 수집 범위 밖이다. 이 셋 때문에 no-match는 계속 U다.
- 구현/검증: `api_snapshot.py`, collector·derivation·dashboard renderer와 테스트를 추가했다.
  API 실패를 HTTP 성공과 분리하고 BuildingHUB 응답 page size 100을 따라 pagination을 소진했다.
  전체 27테스트 중 26 통과·선택 geospatial dependency 1 skip, compileall/JSON/diff check 통과.
  브라우저에서 8766 메인→API 보드 링크와 `r08` 토지피복 2023/24/25 전환을 직접 확인했다.
- 다음: VWorld 개발키와 등록 URL/domain을 함께 재발급·설정한 뒤 대표 1점이 `status=OK`일 때만
  243점으로 확장한다. 그 다음 사전 고정 10–20후보의 NGII 전후 항공 TIFF를 받아 14후보 중
  실제 변화 시기와 parcel footprint를 독립 검수한다. 기존 0/368 gate는 이 두 단계 전까지 유지.

### 2026-08-22 — 사용자 제공 공공 API 승인상태 반영·ECVAM 제거 판정

- 계획: 사용자가 확인한 VWorld 연속지적도·건축HUB·기상청·VWorld 접근상태와 NGII 항공사진
  안내를 공식 페이지 및 로컬 secret 존재 여부와 대조한다. secret 값은 읽거나 출력하지 않고
  변수명별 설정 여부만 확인한다. 환경부용으로 임시 기재된 `ECVAM_API_KEY`의 실제 사용처를
  코드·공식 서비스에서 찾지 못하면 필수키 목록에서 제거하고, 환경영향평가 WFS 승인상태는
  사용자의 명시가 없으므로 `확인 필요`로 유지한다.
- 사전 성공 기준: ① 사용자 확인/로컬 key 존재/API 실제 호출 성공을 서로 다른 상태로 표시
  ② 연속지적도 경로를 data.go.kr과 VWorld 중 실제 사용할 경로로 수정 ③ 항공사진의 신청·다운로드
  절차를 공식 안내와 연결 ④ 환경부 토지피복/생태자연도는 키가 확인되기 전 수동 자료로 유지
  ⑤ 대시보드·canonical 현황·환경변수 템플릿·테스트가 같은 상태를 말한다.
- 무효 조건: key가 있다는 이유로 API schema/coverage 검증까지 완료됐다고 쓰거나, EIA WFS를
  승인 완료로 추정하거나, secret 값을 로그·문서·Git에 노출하거나, 출처가 불명확한 환경부 key를
  필수요건으로 남기는 경우.
- 확인 결과: 로컬 `.env`에는 `DATA_GO_KR_SERVICE_KEY`와 `VWORLD_API_KEY`가 설정돼 있고
  `ECVAM_API_KEY`는 비어 있었다. 값은 읽거나 출력하지 않았고 `.env`의 gitignore도 확인했다.
  사용자 확인은 BuildingHUB·GK2A 승인, VWorld key 확보·VM 실행이며 EIA는 완료 표시가 없어
  미확인으로 유지했다. 이 셋과 `실제 API 응답 성공`은 아직 별도다.
- 공식 페이지 감사: VWorld 연속지적도 2.0은 별도 dataset 신청 대신 `domain+key`와
  `geomFilter/attrFilter`를 받아 PNU·polygon·주소·기준년월을 반환한다. 사용자 제공 NGII 공지
  1347은 항공사진 자료가 아니라 `지리OneView` 설치도구 공지였다. 실제 항공 TIFF는 국토정보맵
  로그인→주소/지명→항공사진→연도→신청 후 승인·문자 알림 경로다.
- 환경부/ECVAM 판정: 환경공간정보서비스는 2021–2025 등의 연도별 토지피복 WMS를 공개하며
  별도 인증키가 필요하지 않다. 반면 `ECVAM_API_KEY`는 토지피복 키가 아니라 국토환경성평가지도의
  법제·환경생태 등 WMS 71종용 선택 P2 key다. 이름·이메일·사용 URL 신청과 이메일 본인인증 후
  발급되지만 현재 PNU/EIA/건축/GK2A gate에는 필요하지 않으므로 신청을 보류했다.
- 구현: secret 없는 `config/kearth_public_access.json`을 접근상태 SSOT로 만들고 사용자 확인·
  credential 존재·실제 probe 상태를 분리했다. dashboard renderer가 이 manifest를 검증·읽도록
  수정했으며 `.env.example`, `K_EARTH_PROGRAM_STATUS.md`, `KOREA_PUBLIC_DATA_CATALOG.md`,
  `README.md`와 포트 8766 대시보드를 같은 상태로 갱신했다. secret 필드가 status manifest에
  들어오면 실패하는 검증도 추가했다.
- 검증: dashboard/public-data 테스트 20개 중 19 통과·선택 geospatial dependency 1개 skip,
  JSON/compileall/`git diff --check` 통과. 브라우저에서 `접근 3종 확인 · EIA 미확인`, VWorld VM,
  BuildingHUB·GK2A 승인, 토지피복 무키, ECVAM 선택 P2 표기를 직접 확인했고 기존 368행을 보존했다.
- 약점/다음: 키가 준비됐어도 실제 제주 request/response, pagination, 날짜 null, CRS, coverage는
  아직 검증되지 않았다. 다음 실행은 VM의 VWorld 점 1개·소형 bbox → BuildingHUB/GK2A 최소
  1페이지 → EIA 승인 확인/소형 WFS → 공개 토지피복 WMS 2023–2025 → NGII 사전고정 10–20후보
  순이다. raw response와 request hash를 보존하기 전에는 `데이터 확보`로 승격하지 않는다.

### 2026-08-22 — 데이터 소유·공공/시계열 결합 중심 5축 대시보드 재정렬

- 계획: 현재 K-Earth 대시보드와 문서를 ① 공공데이터 신청/필요자료 ② 현재 확보·실험 상태
  ③ 비즈니스 가능성 ④ 한국형 연구 질문 ⑤ 8월 EarthRoute 노트의 독립 확장으로 나눈다.
  사용자가 직접 보유할 원본·snapshot·시계열 label이 무엇인지와, 단순 API 연결이 아니라
  재현 가능한 데이터 자산이 되기 위한 provenance·시간축·권리 조건을 명시한다.
- 사전 성공 기준: ① 각 축에 `현재/다음/판정 gate`가 보임 ② 신청처·키·원본 파일·시간 필드·
  공간 join을 혼동하지 않음 ③ 확보/미확보/접근키 필요/파트너 필요를 분리 ④ 공공+시계열이
  연구·사업에 주는 추가가치를 반증 가능한 지표로 연결 ⑤ 기존 368 오름·FarmMap·선택적 보류
  수치를 보존 ⑥ 현재 로컬 대시보드에서 다섯 섹션과 모바일/콘솔 오류를 직접 확인.
- 무효 조건: 데이터셋 개수만 늘리거나, 신청 가능을 확보 완료로 표현하거나, 시계열 원본의
  snapshot/date/version을 잃거나, 전국 확장·원인 규명·사업성을 현재 증거보다 앞서 주장하는 경우.
- 실행 순서: 포트 8766의 실제 문서/서빙 경로 확인 → 기존 catalog·ingest·dashboard 상태 감사 →
  5축 정보구조와 데이터 소유 체크리스트 구현 → 문서·README/GOAL 연결 → 브라우저 visual/DOM/console
  검증 → 수치·미검증·다음 신청 항목 기록.
- 결과: `K_EARTH_PROGRAM_STATUS.md`를 5축 canonical 현황판으로 만들고, 현재 대시보드를
  `K-Earth Program Board v2`로 재구성했다. 새 상단은 신청 상태·보유 경계·사업 gate·논문 질문·
  EarthRoute 분리를 보여주며, 기존 368건 지도/필터/개별 검수와 `367 보류·1 조사` 판정은 그대로
  보존했다. 신청 가능과 실제 승인을 구분해 대시보드 상태를 `활용신청 상태 미확인`으로 고정했다.
- 데이터 감사 수치: 공식 목록 368/368, OSM 위치·OlmoEarth screen 243/368, RGB 검수 9건
  (오염/기각 8·불확실 1), 원인 A/B급 0/368이다. 로컬 공공 원본은 약 112 MB이며 FarmMap
  289,379 polygon, 개발행위 240행·223 PNU, 산지이용 2008–2026 19행을 보존한다. 위성 쪽은
  5,184행 time-axis manifest를 보유하지만 materialized tensor/원영상은 서버 중심이고 v1/v5도
  동일 입력이므로, `시계열 provenance 보유·장기 raw custody 불완전`으로 판정했다.
- 신청/수집 판정: 공공데이터포털의 연속지적도·환경영향평가 WFS·건축HUB·GK2A와 VWorld를
  우선 schema probe 대상으로 고정했다. 국토지리정보원 항공사진과 환경부 토지피복지도는
  key API가 아니라 사전 고정 10–20후보의 연도별 수동 원본 확보 대상으로 분리했다. 실제 계정의
  활용신청 승인 목록은 로컬에서 확인할 수 없고 key adapter도 아직 없으므로 확보 완료로 세지 않는다.
- 사업/연구 판정: 사업은 `Post-EIA Evidence Pack / GeoFM Release Audit / Local Adaptation
  Sprint`의 세 가설만 유지하며 인터뷰·유료 고객은 0이다. 논문 플래그십은 지도를 많이 붙이는
  것이 아니라 **불완전한 행정근거 아래의 선택적 변화탐지와 누락 편향 측정**이다. EarthRoute는
  이 gate를 통과한 뒤 `reuse / cheap_refresh / escalate`를 고르는 별도 후속으로 남겼다.
- 검증: dashboard 단위 테스트 2/2 통과, 공공 ingest 테스트 17개 중 16 통과·선택 geospatial
  dependency 1개 skip, 전체 compile과 `git diff --check` 통과. 브라우저에서 프로그램 카드 5개,
  오름 행 368개, `위치 미해결` 125개 필터와 전체 복귀, console warning/error 0개를 확인했다.
  661 px에서 카드가 가로로 잘리는 반응형 결함을 발견해 1열로 보정했고 1,440 px 5열도 확인했다.
- 약점/마찰: 실제 API 승인상태·지적/EIA/건축/항공 원본·장기 EO 원본 custody·확률표본 100건·
  파트너 검수와 유료 반복은 아직 없다. 선택 geospatial 테스트 1개는 의존성 부재로 skip했고
  로컬에 Ruff가 없어 lint는 실행하지 못했다. 이번 작업은 정보구조와 보유상태 감사였으므로
  실험에서 새로 부딪힌 연구 개념이 없어 `STUDY.md` 카드는 추가하지 않았다.
- 다음 단계: 사용자가 secret을 공유하지 않고 승인된 서비스명만 확인 → `.env`에 key를 보관한
  뒤 제주 bbox 소량 schema probe 구현 → 항공/토지피복 10–20후보 수동 확보 → 100개 층화 표본의
  coverage·보류율·검수시간을 먼저 측정한다. 이 gate 전에는 전국 수집이나 원인 분류로 확장하지 않는다.

### 2026-08-22 — EarthRoute 노트 복원·문헌 지도·사업 가능성 감사

- 계획: 2026-08-04 작성한 `EarthRoute` handoff 노트를 현재 Jeju K-Earth·OlmoEarth 실험과
  연결해 독립 문서로 복원한다. 노트의 선행연구·수치·최초성 주장을 최신 1차 출처로 다시
  확인하고, transfer learning을 모델 점수 개선이 아니라 `릴리스 이전 안정성·선택적 재계산·
  지역 근거 검증`의 연구/사업 질문으로 좁힌다. 별도 논문 검색 리스트에는 읽기 순서·가설·
  우리 실험에 미치는 결정을 함께 기록한다.
- 사전 성공 기준: ① 첨부 노트의 아이디어를 현재 프로젝트와 중복 없이 하나의 프로그램으로 정리
  ② 최소 20편의 실제 논문/공식 프로젝트를 핵심·인접·반증 문헌으로 분류하고 DOI/arXiv/공식 URL
  확인 ③ `고객 문제→유료 wedge→검증 산출물→반복 제품`과 6–12주 proof gate 제시
  ④ transfer가 두 태스크/두 지역에서 재현되지 않으면 표현을 낮추는 kill rule 유지
  ⑤ 근거가 불확실한 award·최초성·시장 수치는 사실처럼 남기지 않는다.
- 무효 조건: 논문 제목만 나열하거나, `fine-tuning 성능 상승`을 곧바로 사업 수요로 간주하거나,
  MARC 같은 연구 파트너를 지불 고객으로 가정하거나, 제주 단일 사례로 전국/글로벌 transfer를
  주장하거나, 기존 Earth platform·embedding API와 정면 경쟁하는 범용 SaaS를 설계하는 경우.
- 실행 순서: 첨부·현재 문서 대조 → 최신 1차 문헌/공식 제품 조사 → 문헌 검색 지도와 확장 노트 작성
  → README/연구전략 링크·Worklog 결과 갱신 → 링크·중복·주장 경계 검수.
- 결과: Slack UI와 중복 본문을 제거한 canonical `EARTHROUTE_PROGRAM_NOTE.md`를 만들고,
  K-Earth(말할 수 있는 근거) → FoldRefresh(릴리스 결과 재사용) → EarthRoute(다음 관측·모델·
  행정근거·사람검증 선택)의 세 층으로 통합했다. 첫 action space는 9축에서
  `reuse / cheap_refresh / escalate` 세 개로 줄였고, 예측 risk·모집단 CI·evidence coverage를
  서로 다른 보증으로 분리했다.
- 문헌 결과: `PAPER_READING_LIST.md`에 50개가 넘는 primary-source-linked 문헌/시스템 레코드를
  모델·전이/shift·유효 추론·adaptive execution·compatibility·impact asset으로 분류했다.
  EarthShift/PANGAEA는 GeoFM transfer가 자동 이득이 아님을, THOR/EO-Gym/OlmoEarth Platform은
  범용 routing이 이미 혼잡함을, PPI/RSE 2025/CRC는 결정·통계 보증을 분리해야 함을 보여준다.
- 사업 판정: `transfer learning` 판매는 기각하고, 첫 wedge를 환경영향평가/GIS 파트너 대상
  `Decision Continuity Audit`·`K-Earth Evidence Pack`으로 좁혔다. 현재는 기술 중, 제품·시장 하이며
  유료 수요 0건이다. 90일 gate는 9개 문제 인터뷰, 최소 100후보 workflow, 검수시간 30% 절감,
  잘못된 원인 단정 0, 같은 고객의 두 번째 유료 refresh다.
- FoldRefresh 감사: 별도 `../decision-ready-earth-ai/`에서 재현 가이드·preregistration·결과 JSON·
  claim verifier·AAAI build chain을 확인했다. 다만 공개 OpenReview receipt/ID를 확인하지 않았고
  로컬 checklist의 수동 항목이 남아 있어 `local artifact verified / venue status externally
  unverified`로 낮췄다. 이 저장소의 미완료 항목은 방법 발명이 아니라 rslearn/K-Earth 이식이다.
- 마찰·약점: 2026 preprint가 많아 출판상태가 바뀔 수 있고, `최초 joint router` 부재 주장은 아직
  검색 가설이다. EIA workflow는 법령상 반복되지만 지불의사와 Evidence Pack의 시간절감은 미검증,
  OlmoEarth local artifact와 hosted Platform은 상업화 조건이 다르다. 모델 릴리스별 parameter
  수치를 섞지 않도록 `PAPER_NOTES_v1.md`에 v1 전용 경고를 추가했다.
- 검수: 읽기 장부는 59개 상태행·54개 고유 링크이며 task 문서의 로컬 Markdown 참조와 표 열 수,
  `git diff --check`를 통과했다. 별도 Markdown linter는 설치돼 있지 않아 실행하지 못했다.
  v5에서 동일 handler로 확인된 `MOSAIC↔PER_PERIOD`는 요인 축에서 제거하고, 실제 item/pixel이
  바뀐 `legacy↔coverage×3+SCL BestClear`만 입력 개입으로 고쳤다. 이번 작업은 문헌 종합이므로
  실제 실험 마찰에서만 만드는 `STUDY.md` 개념 카드는 추가하지 않았다.
- 다음: 먼저 EarthShift·PANGAEA·Backward-Compatible Prediction Updates·RSE PPI를 정독해
  v1↔v1.2 paired audit/확률표본 표를 잠근다. 병행해 EIA 3곳, EO/GIS 2곳, 공공 2곳, 보전 2곳에
  문제 인터뷰를 하고, 유료/LOI/실제 검수시간 중 하나가 나오기 전에는 SaaS를 만들지 않는다.

### 2026-08-22 — K-Earth 공식데이터 ingestion core와 첫 실제 연결

- 계획: `KOREA_PUBLIC_DATA_CATALOG.md`의 설계를 실행 가능한 코드로 내린다. 원본 파일과 API를
  동일한 provenance 계약으로 다루는 작은 ingestion core를 만들고, 우선 키 없이 받을 수 있는
  공식 CSV/SHP 또는 이미 보유한 개발행위허가를 실제로 정규화한다. PNU 형식·날짜·coverage·
  snapshot을 검증하고, 368 오름 registry와의 연결은 exact PNU/polygon/time이 없으면 D/U에서
  승격하지 않는다. 핵심 로직은 import 가능한 모듈과 단위 테스트로 분리한다.
- 사전 성공 기준: ① `source_manifest`와 `evidence_edge`를 JSON Schema 또는 검증 가능한 Python
  model로 구현 ② SHA-256·schema hash·row count·시공간 coverage를 결정적으로 생성 ③ PNU 19자리
  검증과 날짜구간 overlap/no-match eligibility를 경계값 테스트 ④ 실제 공식 데이터 최소 1개를
  end-to-end ingest해 반복 실행 hash가 같음 ⑤ 실패 시 partial 결과를 정답처럼 남기지 않고
  actionable error와 U 상태를 출력 ⑥ formatter/lint에 준하는 정적 검사와 전체 테스트 통과.
- 무효 조건: notebook/SSH 즉석 코드만 남기거나, 데이터별 column명을 핵심 알고리즘에 하드코딩해
  재사용이 불가능하거나, float/row 순서 때문에 manifest가 흔들리거나, 대표지번을 오름 경계로
  간주하거나, 시간필드가 없는 record를 causal match로 승격하거나, 원본 라이선스·URL·기준일을
  잃는 경우.
- 실행 순서: 기존 스크립트·로컬 Python 의존성 점검 → core/model/adapter/CLI와 fixture test 작성 →
  키 없는 공식 원본 획득 또는 보유 snapshot ingest → 결정성·coverage·오류경로 검증 → 368 ledger
  연결 결과와 한계 기록 → README/Worklog/Study 갱신.
- 구현 결과: `code/kearth_public/`에 결정적 JSON/SHA, 검증 model, PNU, 날짜구간·coverage,
  CP949 CSV, 안전한 ZIP 해제, FarmMap offline join을 분리하고 `ingest_kearth_public_data.py` CLI와
  고정 의존성 파일을 추가했다. B급 edge는 공간방법과 시간방법이 모두 없으면 생성되지 않으며,
  no-match는 네 coverage 조건이 모두 참일 때만 해석 가능하다.
- 실제 원본 결과: 제주 FarmMap 2개 SHP 289,379건(제주시 156,122·서귀포 133,257), 유효 PNU
  289,367건·placeholder 12건을 읽었다. 분류는 밭 146,482·과수 74,446·비경지 34,799·시설
  33,539·논 113건이다. 제주시 산지이용은 2008–2026 19행이며 2023 714건/230.6 ha,
  2024 542건/74.2 ha로 기존 개발행위 snapshot의 0건을 음성으로 쓰지 못한다는 경보를 유지했다.
- 연결 결과: 변화 후보 4좌표 중 `oreum_v6_r08`만 FarmMap 밭 polygon과 point-in-polygon으로
  연결됐다. PNU `5013025324202000000`, 항공 2022-12-30, 갱신 2023-12-08이며 변화 구간
  2024-05-16→2025-05-13보다 503일 앞선 **B급 변화 전 상태**다. 원인·허가 증거는 아니다.
  OSM 오름점 243개 중 7건은 C급 FarmMap point 상태다. 개발행위와 exact PNU는 206행·50 PNU·
  144 FarmMap polygon이지만 사건시간·변화 footprint가 없어 교차출처 문맥으로만 저장했다.
- 레지스트리/결정: 368 고정 분모의 FarmMap 상태 C가 7건으로 늘었지만 A/B급 원인 근거는
  0/368이고 `abstain 367 / investigate 1`과 선택적 변화탐지 모드는 변하지 않았다. 4사이트
  dashboard에는 `r08`의 변화 전 상태·503일 gap과 다른 3건의 해석 불가능한 point miss를 표시했다.
- 품질 검증: 단위·통합 테스트 17/17, Ruff check/format, `compileall`, 대시보드 정적 QA를 통과했다.
  같은 원본·고정 retrieval time으로 새 디렉터리에 재실행한 8개 산출물이 모두 byte-identical했다.
  raw SHA-256은 FarmMap `977f840e...ac637bf`, 산지이용 `ce50643a...bcfc4e`로 manifest에 보존했다.
- 약점/미검증: FarmMap은 법적 경계가 아니며 point 결합은 변화 footprint 면적중첩보다 약하다.
  항공 관측연도가 오래된 polygon이 많고, 항공사진 blind review·GK2A 구름·공식 지적·EIA·
  BuildingHUB·사유림 사건을 아직 연결하지 않았다. 98 MB raw ZIP은 Git 반영 전 LFS/객체저장소
  정책도 필요하다. 따라서 `cause_supported`는 0건이며 데이터 증가를 성능 향상으로 주장하지 않는다.
- 다음 게이트: 연속지적도에서 368 대표 PNU와 실제 변화 footprint geometry를 만들고,
  EIA/BuildingHUB/사유림 사건을 30/90/180일 민감도로 결합한다. 그 다음 NGII 전후 항공사진을
  모델 score blind로 판독해 `상태 B + 사건 B + 사람검수`가 처음 닫히는지 확인한다.

### 2026-08-22 — 한국 공공데이터 연결 가능성 전수 탐색

- 계획: 제주 오름 368 evidence ledger를 기준으로 한국의 공식 공공데이터를 `필지·인허가·사업구역 /
  환경·보전 상태 / 재난·기상 관측 / 현장·행정 검증`으로 나눠 검색한다. 각 데이터는 공식
  제공기관·기준시점·공간단위·시간필드·접근방식·인증키·라이선스·결합키를 확인하고, 현재
  레지스트리의 어느 단계에 어떤 근거등급(A/B/D/U)으로 연결되는지 데이터 계약으로 남긴다.
- 사전 성공 기준: ① 최소 12개 공식 데이터 후보와 직접 공식 URL ② PNU·공간중첩·시간중첩 등
  재현 가능한 join 경로 ③ 즉시 사용/키 필요/협의 필요의 접근성 분류 ④ `no match`를 음성
  증거로 사용할 수 있는 조건과 누락 편향 명시 ⑤ 우선순위 3개와 실제 ingestion 스키마 제안.
- 무효 조건: 포털 검색 결과나 블로그만 근거로 삼거나, 데이터 이름만 나열하거나, 현행 레이어를
  과거 원인으로 해석하거나, 주소·행정동 근접을 필지 일치로 승격하거나, API 키 부재를 데이터
  부재로 기록하는 경우. 공식 문서에서 공간·시간·접근 계약을 확인하지 못하면 `탐색 후보`로
  낮춰 기록한다.
- 실행 순서: 기존 결합 공백 확인 → 공식 제공처·API 문서 검색 → 데이터별 evidence-role 및
  join contract 작성 → 368 레지스트리용 우선순위/누락편향 표 작성 → 연구전략·아티팩트 색인과
  Worklog에 결과·마찰·다음 게이트 기록.
- 결과: 중앙·지방정부 공식 페이지에서 **23개 연결 후보**를 확인해
  `KOREA_PUBLIC_DATA_CATALOG.md`에 제공기관·직접 URL·접근방식·공간/시간키·evidence 역할·한계를
  기록했다. 연결 구조는 `연속지적도 PNU → 환경영향평가/BuildingHUB/개발행위/사유림 행정사건 →
  팜맵/토지피복/임상도/항공사진 독립 상태관측 → 생태·보호·국가유산 영향 문맥 → GK2A/SCL·기상
  입력품질`이다. 최소 12개 기준을 넘었지만 목록 수 자체는 성과로 세지 않는다.
- 핵심 발견: 현재 개발행위허가 snapshot의 제주 2023·2024행은 0이지만, 행정 개념이 다른
  제주시 산지이용지정현황에는 **2023년 714건·230.6 ha / 2024년 542건·74.2 ha**가 있다. 이는
  두 데이터의 직접 모순이 아니라 행위유형·모집단이 다른 행정 시스템의 0건을 “활동 없음”으로
  일반화할 수 없다는 coverage 경보다. no-match는 공간·기간·행위 모집단·pagination/이력·join
  필드가 모두 완전할 때만 음성 근거로 허용한다.
- join contract: 공식 오름 주소의 지번은 대표필지일 수 있으므로 PNU를 오름 경계로 간주하지
  않는다. exact PNU, polygon intersection, 행정리/최근접을 별도 저장하고, 변화 전후 구간에
  30/90/180일 사전 민감도 창을 적용한다. `source_manifest`와 `evidence_edge` 스키마를 정의해
  snapshot·유효기간·CRS·라이선스·schema hash·겹침면적·day gap·no-match 해석 가능성을 보존한다.
- 연구 전환: `모델+OSM → +공식 PNU → +상태지도 → +행정사건 → +항공/현장` ablation을 고정하고
  단계별 time-aligned coverage, decide/abstain, selective risk, 지역·토지피복별 침묵률을 368
  분모에서 측정한다. 이는 공공데이터 개수를 늘리는 데모가 아니라 **자료 누락이 Earth model의
  안전한 발언 가능성을 어떻게 편향시키는지**를 측정하는 논문 실험이다.
- 마찰·약점: 일부 서비스는 키·Digital OnePass·도엽 신청·과거판 기관 협의가 필요하고, 산불 API는
  좌표정밀도, BuildingHUB·산림사업은 실제 실행일 필드를 아직 원본 schema로 확인하지 못했다.
  이들은 A/B로 선반영하지 않고 `탐색 후보` 또는 U로 유지했다. 임상도는 변경금지 조건을 포함한
  라이선스 검토가 필요하며, 현행 용도지역·보호지도는 과거 원인으로 쓰지 않는다.
- 다음 게이트: ① 연속지적도 키로 368 주소의 PNU exact/review/unresolved coverage 산출
  ② 2025 제주 팜맵 SHP를 내려받아 현재 변화 footprint와 교차 ③ EIA WFS·BuildingHUB·사유림사업
  schema의 날짜/geometry coverage 감사 ④ 항공사진 blind review 표본 설계. 이 네 단계 전까지
  A/B급 원인근거는 계속 0/368이며 선택적 보류 모드를 유지한다.

### 2026-08-22 — K-Earth Evidence: 제주 368개 오름 전수 레지스트리와 선택적 판정

- 계획: 제주 공식 오름 368건을 분모로 고정하고, 사용자가 제공한 오름 속성 표를 재현 가능하게
  정규화·대조한다. 각 오름에 대해 `목록 등재 → 위치 해석 → 위성 관측 가능 → 변화점수 산출 →
  공식 근거 결합 → 사람 검수`의 단계를 별도 상태로 저장하며, 기존 4개 변화 후보의 근접 근거는
  이 전수 레지스트리에 연결한다. 아직 수행하지 않은 위성 판정을 “조사 완료”로 표시하지 않는다.
- 사전 성공 기준: ① 공식 368건 모두에 고유 ID와 출처·기준일·주소·속성을 보존 ② 첨부 표와의
  일치/불일치를 수치화 ③ 전수 분모에서 공식 근거 가용률과 상태별 누락률 계산 ④ 근거 가용률이
  10% 미만이면 원인 규명 대신 `조사 우선 / 보류 / 판정 가능` 선택적 변화탐지 모드로 자동 전환
  ⑤ 대시보드에서 368건 검색·필터·근거 등급·보류 이유·연구 게이트를 확인할 수 있음.
- 무효 조건: 주소만 있는 오름에 임의 좌표를 부여하거나, 현재 OSM peak 근접을 공식 경계·과거
  상태·인과 근거로 간주하거나, 미조회/키 부재/시차를 “개발 없음”으로 코딩하거나, 기존 4개
  후보의 근거 가용률을 368개 전체 가용률로 일반화하는 경우. 전수 목록 커버리지와 전수 위성
  판정 완료율을 반드시 분리한다.
- 실행 순서: 첨부 HTML 표 parser와 provenance manifest 작성 → 공식 CSV record linkage →
  offline OSM의 오름 위치 후보 연결 → evidence coverage/abstention 정책 산출 → 368개 레지스트리
  및 연구 대시보드 생성 → 브라우저 육안·상호작용 검증 → 서버에서 전체 위성 실행의 비용·입력
  준비 상태를 점검하고, 장기 실행은 다중-window v7 게이트를 통과한 뒤에만 시작한다.
- 결과 — 고정 분모와 연결: 공식 오름 **368/368**을 상태화했다. 첨부 제주시 210행은
  **209행 연결 / 188행 핵심 필드 대조 / 21행 주소 등 충돌 / 1행(`빈내오름`) 미연결**이다.
  첨부 `번호`가 공식 `연번`과 같은 키라는 초기 가정을 폐기하고 이름·주소·면적 복합키로
  교체했다. offline OSM peak는 **243/368**, 같은 리 단위 허가 문맥은 **183/368**이며
  각각 C/D급 탐색 근거일 뿐 공식 경계·필지 원인으로 승격하지 않았다.
- 결과 — 모델과 육안 검수: H200에 이미 생성된 v6 embedding만 재사용해 위치가 연결된
  **243/243**을 점별 screen했다. `high_stable 8 / high_unstable 34 / moderate_stable 4 /
  low_or_unstable 197`이었다. 모델 안정 후보 8개와 기존 후보 인접 1개, 총 9개를 2023–2026
  동일 월·고정 stretch·두 공간 축척 RGB로 검수한 결과 **지속 변화 확정 0 / 구름·연무로
  기각 8 / 불확실 1(성산일출봉)**이었다. 최종 출력은 `조사 우선 1 / 보류 367`이다.
- 핵심 발견: 4기간 점수와 12기간 점수가 모두 높은 8건은 독립된 두 증거가 아니었다. 두
  계산이 공유한 오염된 2023 입력 때문에 **8/8이 같은 구름 오차를 안정적으로 재현**했다.
  따라서 모델 합의는 입력·오류가 독립적일 때만 강화 증거이며, v7 SCL(Scene Classification
  Layer) 품질 게이트와 사람 검수 전에는 오름 변화 주장으로 사용할 수 없다.
- 정책 판정: 필지·환경영향평가 경계와 시점이 맞는 A/B급 공식 원인 근거는 **0/368(0%)**로
  사전 10% 문턱보다 낮다. 시스템은 자동으로 `selective_change_detection` 모드가 되었으며,
  현재 허용 주장은 전수 증거 상태·누락률·조사 우선순위뿐이다. “368개 원인 규명”과
  “오름 훼손 전수 확인”은 금지한다.
- 마찰·개선: ① 부분집합 내부 순번을 공식 연번으로 붙인 첫 결합은 206개 가짜 충돌을 만들어
  linkage audit와 복합키로 교정했다. ② 섬 전체 embedding center를 매번 읽는 첫 score 방식은
  NFS I/O 병목으로 중단하고, `code/`의 point-only grouped read로 바꿔 243건을 완주했다.
  ③ 첫 RGB figure의 한글 glyph 경고는 영문 ID 제목으로 교정해 경고 없이 재생성했다.
- 브라우저 검증: 메인 화면 **368행**, 위치 미해결 필터 **125행**, `빈내오름` 검색 **0행**,
  판정 변경의 localStorage 저장·reload 지속을 확인했다. RGB 검수 화면은 article/image
  **9/9**, 모든 이미지 `naturalWidth > 0`, 콘솔 오류 0이다.
- 종료 시 서버 상태: `h200-dev`와 영구 data volume은 RUNNING/ready다. 오름 score·RGB render
  프로세스는 남아 있지 않다. H200 GPU0은 메모리 0 MiB로 비어 있고, GPU1은 다른 작업이
  38%·68,501/143,771 MiB를 사용 중이다. 오름 산출물은 `/home/work/data/olmoearth/` 아래에
  보존했다. 접속 시 실제 대문자 경로의 `H100_SETUP_DIR` 명시가 계속 필요하다.
- 약점·다음 게이트: **125개는 공식 좌표/경계가 없어 모델 screen도 미완료**이고, 243개도
  현행 OSM point다. v6는 계절·시간축 교란이 남고, 허가 스냅샷은 2023·2024 제주 행이 없어
  `no match`가 음성 증거가 아니다. 다음은 ① 브이월드 PNU·환경영향평가 공식 polygon 확보
  ② 대표 window에서 v7 SCL 다중시점 게이트 통과 ③ top-k가 아닌 비후보 포함 층화 확률표본과
  PPI(Prediction-Powered Inference) 신뢰구간 ④ 현장 파트너의 독립 판정 순으로 진행한다.

### 2026-08-22 — 한국 공공데이터 결합 전후 증거력 비교

- 계획: `human_review_v1`의 후보·순위·사람 판정을 고정한 채, 한국 공식 지도/공공데이터와
  개방형 지도 레이어를 후보 주변에 결합한다. 먼저 4개 고유 고확신 변화지를 대상으로
  오름/지명, 도로·건물, 토지피복·용도지역, 개발 관련 공개 기록의 실제 접근성과 기준 연도를
  조사하고, 키 없이 재현 가능한 레이어부터 evidence pack과 검수 UI에 붙인다.
- 사전 성공 기준: ① 원래 RGB-only 판정을 덮어쓰지 않고 결합 전/후 판정을 분리 ② 레이어마다
  제공기관·기준일·라이선스·공간해상도/축척·조회 시각 보존 ③ 4개 사이트 모두에 “추가 증거 있음
  / 없음 / 조회 불가”를 기록 ④ 최소 1건에서 외부 레이어가 원래 해석을 강화·약화·변경하거나,
  전부 무정보라는 결론을 재현 가능하게 남김 ⑤ 브라우저에서 사용자가 근거를 보고 수정·내보냄.
- 무효 조건: 현재 지도 객체를 과거에도 존재한 것으로 간주하거나, 지목·용도지역을 실제 이용과
  동일시하거나, 점/선 객체의 근접만으로 개발 원인·허가·위법·생태 영향을 확정하는 경우.
  API 키 부재/시간 불일치는 음성 증거가 아니며 명시적인 `unavailable`로 남긴다.
- 데이터 결합: 제주특별자치도 오름현황 **368건**(기준일 2024-03-31, 주소·속성만 있고
  좌표/경계 없음)과 국토부 토지이음 개발행위허가 최신 전국 ZIP(2026-08-19)에서 제주
  **240건**을 UTF-8로 정규화했다. 허가 행은 2023·2024가 0, 2025가 4뿐이어서 네 후보의
  행정경계명 일치 0건을 “허가 없음”으로 사용할 수 없다. 브이월드 지적/PNU WFS와 환경부
  토지피복·환경영향평가 API는 인증키가 필요한 다음 레이어로 기록했다.
- privacy-preserving spatial join: 후보 정밀좌표를 Nominatim/Overpass에 보내는 호출은 보안
  검토에서 중단했다. 대신 Geofabrik 대한민국 전체 OSM 스냅샷(MD5 검증,
  2026-08-21T20:21:11Z)을 받고 제주를 추출해 **113,547 feature를 전부 로컬 결합**했다.
  후보 좌표의 제3자 전송은 0건이다.
- 결과: RGB 지속 변화 판정이 뒤집힌 site는 0/4지만 문맥은 4/4에 추가됐다. 개발 대조는
  삼양동 도시 가장자리(500 m 건물 26·도로 객체 32)라 개발 해석이 강화됐다. `r08`은 수망리
  현행 wood·더클래식컨트리클럽 경계 289 m·공식 마은이 1.64 km, `r10`은 가시리·동 골프장
  997 m·공식 마은이옆 1.24 km여서 특정 오름 정상부 훼손 주장은 약해지고 관리형 중산간
  문맥이 생겼다. `r11`은 공식 고이악/OSM 고이오름 416 m와 현행 태양광 발전소 폴리곤
  6개(419–951 m)가 있어 인프라 연관 가설이 강화되어 후속 1순위가 됐다. 최종 판정은
  **강화 2 / 문맥만 추가 2 / 변화 판정 번복 0**이다.
- UI·검증: `artifacts/external_data/korea_public_v1/evidence_dashboard.html`에 4개년 RGB와
  3 km offline vector map을 병치하고 OSM ODbL 출처, 공식 오름명·거리, 건물/도로, 허가
  한계를 표시했다. 브라우저에서 4/4 이미지·SVG·초기 판정, 선택 변경→reload 지속→복원을
  확인했고 좁은 self-contained 폴더만 localhost로 제공했다.
- 실패·개선: 공식 허가 CSV의 null `지자체코드`로 첫 정규화가 실패해 null-safe parser로
  고쳤다. 근접 지명 화북이동을 포함 경계 삼양동 대신 허가 연결한 초기 규칙은 false join으로
  폐기하고 포함 행정경계만 허용했다. 날짜 고정 Geofabrik URL 404는 `latest`+MD5로 교체했고,
  오름 별칭 set 순서 때문에 출력 hash가 두 값 사이에서 흔들린 문제는 정렬을 강제해 2회 연속
  동일 SHA-256으로 닫았다.
- 약점·다음: OSM은 현재 커뮤니티 지도이며 시설 등록일이 실제 조성일이 아니다. 공식 오름
  파일에는 경계가 없고 peak 점만 연결했으며, 허가는 필지 PNU와 후보 좌표가 아직 연결되지
  않았다. 다음 승격 조건은 브이월드 지적/PNU 또는 동등한 공식 cadastral layer, 환경부
  토지피복/환경영향평가, 과거 항공사진 중 하나로 **거리 근접을 경계 중첩+시점 일치**로 바꾸는 것이다.

### 2026-08-22 — 제주 변화 후보 human-in-the-loop 육안 감사

- 계획: 기존 v3/v6 변화 후보를 결과를 본 뒤 재선택하지 않고 입력 목록으로 고정한다. 확인된
  개발 후보(33.5087N, 126.5747E)를 양성 대조로 포함하고, 오름·초지/산림권과 해안·도시권을
  공간적으로 층화해 2023~2026 동일 stretch RGB 시계열 아틀라스를 만든다. 각 후보를
  `개발·벌채 / 오름·초지 계절성 / 구름·해무 / 바다·반사 / 불명확`으로 판정하며 사용자가
  JSON에서 판정을 수정할 수 있게 provenance와 좌표를 함께 남긴다.
- 사전 성공 기준: ① 후보 선택 근거·source ranking을 보존 ② 모든 후보에 4개년 동일 위치 RGB
  제공 ③ 자동 점수와 사람 판정을 분리 ④ 최소 1건의 지속적 지표 변화 또는 “유효 후보 없음”을
  증거와 함께 판정 ⑤ 구름·계절성 후보를 개발로 오인하지 않음.
- 무효 조건: RGB를 본 뒤 성공 사례만 골라 분모를 바꾸거나, 오름의 식생 계절성·그림자·화산
  지형 차이를 개발로 부르거나, 현장·고해상도 자료 없이 원인을 확정하는 경우. 산출물은
  조사 우선순위 후보이며 생태 영향·인과효과 주장이 아니다.
- 결과: RGB를 보기 전 규칙으로 **14개**를 고정했다(개발 대조 4, 동부 중산간 tree/grass
  후보 6, v3 공간 대조 4). 각 연도 5월 15일에 가장 가까운 관측, 동일 0–3000 DN stretch,
  1.28 km 맥락+400 m 상세로 전부 육안 판정했다. 고확신 지속 변화 record는 5개지만
  `dev_control_v3_r02`와 `built_v6_r16`이 약 60 m 간격의 같은 변화지라 **고유 사이트는 4개**다.
  추가 중간/불확실 변화 3개, 구름·농경 계절성·안정/불명확 6개로 판정했다.
- 핵심 관찰: 동부 중산간 집단을 “오름 변화” 하나로 묶을 수 없었다. `oreum_v6_r08/r10/r11`은
  2024~2026에 갈색 절개→회색 대면적 표면 또는 확대된 나지가 지속되는 고확신 토지전환
  형태였지만, `r04`는 2023 구름 뒤 안정된 경작/피복지였다. v3 공간 대조 4개는 2개 구름,
  2개 농경·식생 변화로 닫혀 기존 false-positive 계보를 재확인했다.
- 사용자 검수 산출물: `artifacts/human_review_v1/dashboard.html`에 Codex 1차 판정을 채우고
  select/메모를 직접 수정한 뒤 JSON으로 내보낼 수 있게 했다. 브라우저에서 14/14 초기값,
  선택 변경·복원, 상대 이미지 로딩을 확인했다. 알고리즘 manifest와 사람 판정은 별도 JSON으로
  유지한다.
- 약점·다음: 10 m RGB는 개발 종류·허가 여부·생태 영향·특정 오름 경계를 확정하지 못한다.
  외부 역지오코딩은 정밀 후보 좌표를 제3자 서비스로 전송하는 문제가 있어 수행하지 않았다.
  사용자 판정과 동의된 공공 인허가/고해상도 레이어를 받은 뒤 4개 고유 고확신 지점을
  evidence pack 후보로 승격하고, 그 전에는 좌표 기반 조사 우선순위로만 유지한다.
- 종료 검증: manifest·사람 판정·PNG가 각각 14개로 일치하고 manifest SHA-256 고정값과
  Python AST, `git diff --check`가 모두 통과했다. 서버의 검수 생성 프로세스는 0개다.
  H200 2장의 현재 점유(약 81/73 GiB)는 무릎 학습 2개와 별도 `p1_world` 적응 실험 2개이며,
  제주 검수 작업이 아니다. 영구 볼륨은 9.1 TiB 중 8.2 TiB가 남아 있다.

### 2026-08-22 — v7 SCL-aware 합성 golden-window smoke test

- 계획: 전체 제주 재계산을 금지하고, v1 blind audit에서 구름 100%였던 2025년
  `30720_-372736` 한 window만 사용한다. 기간당 3개 spatial coverage를 후보로 넣고,
  rslearn의 `Sentinel2SCLBestClear`를 SCL nearest-neighbor 점수로 보정해 window 내부 clear
  cover가 가장 높은 장면을 선택한다. 기존 v1과 같은 최근 4기간만 비교한다.
- 사전 성공 기준: ① v1과 다른 ordered item group/출력 픽셀 확인 ② 첫 4기간 전체-window
  bad proxy 상대 10% 이상 감소 ③ 사전 선택 target block(2025 period index 3)의 bad proxy
  1.0→0.5 이하 ④ zero/mask proxy 악화 1%p 이하 ⑤ 고정 stretch RGB에서 target cloud 감소.
- 무효 조건: SCL을 bilinear로 보간해 class 점수를 왜곡하거나, v7 결과를 보고 target을 다시
  고르거나, 한 window 성공을 제주 전체/생태 결과로 일반화하는 경우. 수치·RGB가 실패하면
  overlaps/합성기를 수정하되 같은 target과 기준을 유지한다.
- 실행 마찰(실패도 보존): attempt 1은 enum 대소문자(`BILINEAR`) 검증 실패, attempt 2는
  console entrypoint의 `sys.path`에서 로컬 compositor import 실패, attempt 3은 SCL 자산 URL이
  items에 있어도 `band_sets`에 없어서 tile store 등록이 안 되어 실패했다. 각각 설정을 고친 뒤
  같은 window·target·판정 기준으로 재실행했으며 로그는 `artifacts/results/jeju-v7-smoke*.log`에
  보존했다.
- 결과: materialize **1/1, 4기간 완료, 실패 0**. 네 기간 모두 ordered source group이 v1과
  달랐고 3/4기간의 반사도 픽셀이 달라졌다. 첫 4기간 mean bad proxy 상대 감소는 **95.64%**,
  zero/mask proxy 변화는 **−0.0082%p**, 사전 고정 target(period 3)은 **1.00→0.00**이었다.
  고정 stretch RGB에서도 period 3의 큰 밝은 구름과 period 0의 구름 패치가 제거되어 수치·육안
  게이트가 모두 통과했다.
- 약점·다음: 이는 사전 선택한 1개 golden window의 장면선택 성공이지 제주 전체 성능이나
  변화탐지 정밀도가 아니다. 어두운 clear scene이 임베딩 분포를 바꿀 가능성도 남는다. 다음은
  v5 전수 감사에서 연도×bad-proxy로 **사전 층화한 다중 window**에 같은 기준을 적용하고,
  효과 분포·실패율·embedding/Top-k 안정성을 본 뒤에만 216윈도우로 확장한다.
- 종료 상태(11:00 KST): `h200-dev` RUNNING, 영구 data ready. v7 디렉터리는 60 MiB이고
  완료 marker 4개, 잔여 v7 프로세스 0이다. attempt 1이 남긴 독립 process group 1074688은
  소유·PGID를 확인한 뒤 종료했다. `/home/work/data`는 9.1 TiB 중 1.0 TiB 사용(11%, 8.2 TiB
  여유). H200 사용률 69%/41%, 메모리 80.9/72.7 GiB는 다른 사용자 작업이 계속 점유 중이다.

### 2026-08-22 — 제주 v5 입력 품질 감사 착수

- 계획: `code/audit_jeju_v5_quality.py`를 먼저 작성해 v1(MOSAIC)과 v5
  (PER_PERIOD_MOSAIC)의 동일 4개년·동일 격자를 비교한다. 구조 완전성, B02 기반 구름 proxy,
  0/mask 기반 nodata proxy, 최악 기간 품질, 엄격 clean coverage를 수치화하고 동일 위치 RGB
  쌍을 생성한다. 서버에는 파일로 전송한 코드만 실행한다.
- 사전 성공 기준: ① 4년×54 = 216 matched windows와 각 12기간 확보 ② v5 평균 cloud/bad
  proxy가 v1보다 상대 **25% 이상 감소** ③ v4의 strict clean coverage 1.2% 대비 **5배 이상**
  증가(≥6%) ④ zero/nodata proxy가 **1%p 넘게 악화되지 않음** ⑤ RGB 5쌍 중 4쌍 이상에서
  구름/결측 감소를 사람이 확인. 수치가 좋아도 RGB 판정 전에는 통과로 닫지 않는다.
- 무효 조건: B02>1800 휴리스틱을 실제 cloud mask로 오인하거나, 개선이 큰 위치만 골라 전체
  품질을 주장하거나, FIRST_VALID의 0을 자동으로 nodata라고 단정하는 경우. 결과에는 proxy
  한계와 전수/표본 범위를 함께 남긴다.
- 전수 결과: 구조는 216윈도우×12기간으로 완전했지만 v1↔v5의 cloud/zero/bad proxy와
  strict clean coverage가 **소수점 이하까지 동일**했다. cloud/bad 감소 0%, all-12 strict clean
  1.235%→1.235%(×1), blind RGB 수치 개선 0/5이며 육안으로도 5/5가 같은 구름 형태였다.
  따라서 사전 성공 기준 ②·③·⑤를 실패했고, v5를 품질 개선으로 판정하지 않는다.
- 후속 진단 사전 기준: `code/diagnose_jeju_v5_equivalence.py`로 ① 2,592개 period source-item
  순서 hash 전수 비교 ② 현재 rslearn에서 두 SpaceMode가 동일 handler인지 확인 ③ 고정 seed의
  원본 24쌍 전체 밴드와 임베딩 공간표본 비교를 한다. 모두 같으면 “v5는 새 입력이 아니라
  설정 별칭으로 만든 중복 실행”으로 닫고, 하나라도 다르면 그 단계부터 차이를 추적한다.
- 새 시간축 마찰: 실제 item 순서는 역시간순이다. 첫 4기간은 2023~2025가 대체로 12→9월,
  rolling-2026이 6→3월이라 계절 정렬에 실패한다. 기존 12기간 실험은 4기간과 Top-30 교집합
  5곳(Jaccard 0.091)이지만, 이 값은 민감도 증거이지 변화 정답이 아니다.
- 등가성 진단 결과: rslearn 0.1.13(commit `bbbc18b`)에서 두 설정은 동일
  `match_with_space_mode_mosaic` handler로 정규화된다. ordered item hash **2,592/2,592**,
  원본 전체 12밴드 표본 **24/24**, 임베딩 32×32 표본 **24/24**가 정확히 같았다.
  v5를 의미적 중복으로 닫고 SCL/cloud mask가 실제 pixel validity를 바꾸는 v7 smoke test로 간다.

### 2026-08-22 — 제주 v5 완료 확인·서버 상태 재점검 (10:12 KST)

- 계획: Nexus 세션·GPU·v5 프로세스·임베딩 완료 표식·오류 로그·영구 저장소를 원격에서
  교차 확인한다. PID가 없더라도 `PIPELINE_DONE`, 216개 완료 표식, traceback 0이 함께
  확인될 때만 계산 완료로 판정한다.
- 결과: `h200-dev`와 영구 데이터 폴더는 정상이다. v5는 00:41 KST에 `PIPELINE_DONE`을 남겼고,
  materialize **216/216(실패 0)**, 임베딩 TIFF/완료 표식 **216/216**, 0-byte 임베딩 0,
  traceback 0으로 계산을 마쳤다. 산출물 디렉터리는 **152 GiB**다. 기존 부모·추론 PID는
  종료돼 좀비/잔여 작업도 없다.
- 서버 여유: `/home/work/data`는 9.1 TiB 중 약 1.0 TiB 사용, **8.2 TiB 여유(11%)**다.
  H200은 51%/28%, 메모리 74.0/72.6 GiB 사용 중이나 현재 점유자는 `knee-proj` 학습과
  `p1_world` 적응 실험이며 제주 v5가 아니다.
- 약점: `FIRST_VALID` nodata 기본값, 폐기 예정 `PER_PERIOD_MOSAIC`, 시간순서/legacy timestep
  경고는 계산 완료로 해소되지 않았다. 따라서 v5를 “구름 강건 입력 검증 완료”라고 부르지 않는다.
- 다음: 재현 가능한 품질 감사 코드를 `code/`에 먼저 작성해 ① 0/nodata·유효관측률
  ② v1↔v5 구름 감소 ③ 대표/상위 후보 RGB 육안 검증을 닫은 뒤 paired release audit로 간다.

### 2026-08-22 — OlmoEarth 프로젝트를 박사 연구 프로그램으로 재정렬

- 계획: `earth_paper/dashboard.html`의 세 계보(EO 측정·GeoFM 표현·의사결정)를 현재 실험과
  연결해, 제주를 목적이 아닌 검증장으로 재정의한다. Sherrie Wang식 통계적 추론,
  Ai2 OlmoEarth식 공개 인프라, MARC식 현장 결정을 하나의 연구 질문·산출물·게이트로 묶는다.
- 만들 것: ① 연구 프로그램 문서 `RESEARCH_STRATEGY.md` ② README/GOAL의 미션·로드맵 연결
  ③ 지원·논문·파트너 증거가 어떻게 한 실험에서 나오는지 보여주는 연구 포트폴리오 시각화.
- 사전 판정 기준: 새 방향은 (a) 한 문장 연구 질문 (b) 반증 가능한 가설과 강한 베이스라인
  (c) 제주 외 두 번째 태스크 전이 (d) 공개 코드/표 (e) 파트너가 바꾸는 결정까지 포함해야 한다.
- 무효 조건: 교수 이름을 붙인 독서 목록, 제주 사례의 과장, 범용 플랫폼/SaaS 확장, 또는
  v1↔v1.2 paired 결과 없이 “릴리스 안정성”을 주장하는 경우.
- 결과: 연구 질문을 **Decision-Continuous Earth Intelligence**로 고정하고, 제주를 목적이 아닌
  첫 검증장으로 재정의했다. `RESEARCH_STRATEGY.md`에 5개 가설·강한 베이스라인·기술/파트너
  성공 기준·12주 실행·중단 게이트를, `PARTNER_BRIEF_MARC.md`에 실제 MARC 연구 의제와 맞춘
  첫 미팅 질문·6주 파일럿·금지 주장을 기록했다. 지원·논문·파트너 증거가 한 실험에서 나오는
  상호작용형 연구 지도를 함께 만들고 모바일/다크모드까지 렌더 검증했다.
- 실행 증거(00:21 KST): v5 materialize는 **216/216, 실패 0**으로 끝나 임베딩 단계에 진입했고,
  부모 PID 984897과 `rslearn model predict` 자식 PID 1002472가 살아 있다. 임베딩 GeoTIFF가
  계속 생성되며 완료 표식 **68/216**을 확인했으나 `PIPELINE_DONE`은 아직 없어 전체 완료
  판정은 보류한다.
- 새 마찰: 현재 rslearn은 `PER_PERIOD_MOSAIC`를 폐기 예정으로 표시하고
  `MOSAIC + period_duration`을 권고한다. 장면 시간순서 기본값과 legacy timestep도 변경 예정
  경고가 있어, 같은 YAML 이름만으로 실행 의미가 고정되지 않는다. 실행 중인 통제군은 건드리지
  않고 버전·정규화 설정·장면순서 해시를 manifest에 포함하는 후속 검증으로 넘긴다.
- 다음: v5 완주 확인 → nodata/구름/RGB 육안 검증 → 동일 원시 입력으로 v1↔v1.2 paired audit
  → 층화 표본·PPI 추정. MARC에는 이 세 단계가 통과되기 전 협력·성능·생태효과를 주장하지 않는다.

### 2026-08-21 — 제주 v5 진척 재점검·프로젝트 유의미성 판정

- 계획: 제주 v5의 PID·로그·단계·산출물·GPU 경합을 다시 확인하고, 프로젝트의 현재 증거를
  취업(공개 이슈/PR), 연구(일반화·통제·불확실성), 사업(실제 의사결정자) 세 축으로 판정한다.
- 사전 판정 기준: v5는 로그와 산출물이 전진하고 오류 없이 임베딩 단계/완료 표식에 도달해야
  실행 성공으로 본다. 프로젝트는 제주 데모 단독이 아니라 ① v1↔v1.2 paired 결과표
  ② 구름/nodata 육안·정량 검증 ③ 외부 공개 증거 또는 파트너 결정 중 최소 둘을 남겨야
  “충분히 유의미”하다고 판정한다.
- 무효 조건: 살아 있는 PID만으로 성공을 선언하거나, 단일 Top-k 사례·자체 정답·최초성 주장만으로
  연구/사업 가치를 과대평가하는 경우.
- 실행 결과(23:58 KST): v5 materialize가 195→212/216(90→98%)로 계속 전진했고,
  PID·자식 프로세스·로그 갱신 모두 정상이며 오류/트레이스백은 없다. 산출물은 53 GiB.
  아직 임베딩 파일 0개, `=== embeddings`/`PIPELINE_DONE` 없음이므로 성공 판정은 보류한다.
- 외부 대조: Ai2 Studio는 이미 임베딩 export·유사도 검색·변화탐지를 제공하고 Major TOM도
  OlmoEarth 249k 임베딩을 공개했으므로, “한국 임베딩 검색/제주 변화 데모” 자체는 기여가 약하다.
  반면 최신 Earth Embeddings 문헌은 제품 간 재현성·불확실성·벤치마킹을 열린 문제로 두고,
  EarthShift와 GFM 재현성 감사는 시간/지역 shift와 평가 불일치가 실제 문제임을 정량화한다.
- 판정: 현재도 재현성 디버깅·실패 계보·아시아 입력 품질 감사라는 **취업 포트폴리오 가치는 높음**.
  그러나 핵심 차별점인 v1↔v1.2 paired 실행 코드/결과표가 아직 없고, 공개 이슈/PR 및 파트너
  결정도 닫히지 않아 **논문 기여는 유망하지만 미완성, 사업 가치는 미검증**이다.
- 다음 게이트: ① v5 완주 뒤 nodata/구름/RGB 육안 검증 ② 같은 원시 입력의 v1↔v1.2 paired
  결과와 이웃·Top-k·집계 안정성 표 ③ LFMC 이슈/스키마 PR 공개 ④ 파트너 1곳의 실제 결정
  질문 검증. 이 네 개 중 앞의 세 개가 닫히면 “충분히 유의미한 공개 프로젝트”로 판정한다.

### 2026-08-21 — H100 접속·실행 상태 점검

- 계획: 실제 접속 저장소가 `~/DongDong/ai_projects/h100-setup`에 있으므로
  `H100_SETUP_DIR`을 명시하고, 프로젝트 규약대로 `./bin/nx`를 통해서만 점검한다.
- 사전 판정 기준: ① Nexus 세션 조회 성공 ② 터널을 통한 원격 셸 응답 ③ H200 2장과
  사용률·메모리 확인 ④ 실행 중 프로세스/PID·최근 로그 확인 ⑤ 영구 저장소 용량 확인.
- 무효 조건: 로컬 `.state`만 읽고 원격 명령이 통하지 않거나, 오래된 PID/로그만으로 작업이
  살아 있다고 판단하는 경우. 원격 프로세스와 로그 갱신 시각을 함께 대조한다.
- 결과(23:48 KST): `doctor` 전 항목 통과, `h200-dev` RUNNING, 원격 셸 응답 확인.
  H200 2장(각 143,771 MiB)은 59%/38%, 메모리 80,957/62,587 MiB 사용 중이었다.
  `/home/work` 47 GiB, 영구 저장소 8.3 TiB 여유(사용률 10%).
- 제주 v5: PID 984897과 materialize 자식 프로세스가 살아 있고, 로그가 195/216(90%)까지
  계속 갱신됨. 오류·트레이스백 없음. 로컬/서버 스크립트 SHA-256 일치, 현재 산출물 50 GiB.
  아직 materialize 단계라 embedding 파일과 `PIPELINE_DONE`은 없음.
- 마찰·약점: `bin/nx` 기본 경로가 실제 대문자 경로와 달라 `H100_SETUP_DIR`이 없으면 실패.
  두 GPU는 다른 학습 작업도 각각 약 80.9/66.8 GiB 점유하므로 v5가 GPU0 임베딩 단계로
  넘어갈 때 메모리 경합을 관찰해야 한다. 또한 `FIRST_VALID` 합성에서 nodata 메타데이터가
  없어 0을 쓰는 경고가 반복됨 — 결과 해석 전 0값/마스크 의미 검증 필요(STUDY #16).
- 다음: 실행은 그대로 둔다. materialize 완료 후 `=== embeddings` 진입과 최종
  `PIPELINE_DONE`을 확인하고, 그 뒤 구름 감소율·nodata 비율·육안 RGB 칩을 v1과 비교한다.

### 2026-08-25 — AI-Hub 71363 수신 → 계약 감사 → spatial holdout 동결 (M9·M10)

**한 일**
1. AI-Hub 인증 방식을 포털에서 검증하고 H200에서 직접 수신 (Sentinel-2 8 zip, 3.32 GB).
   filekey 491163/491167/491171/491175/491179/491183/533616/533620 — 전체 85 GB의 3.9%만.
2. 수신 후 검증 4/4 통과. 촬영시점 3,000건 전부 존재(63일, 2019-01-03~2022-10-29),
   좌표 3,000/3,000 EPSG:32652, task 중첩 20쌍, 클래스 9/9가 100타일 기준 통과.
3. **M9** — 인벤토리·계약 감사. 내가 쓴 "678×63"은 틀렸고 실제 조합은 **2,699쌍**이다.
   메타 `coordinates`가 좌상단임을 라벨 폴리곤과 대조해 확정(중위거리 4.2e-05 m).
   **공식 split 사용 불가** — valid 타일 110/110이 train과 공간 중첩(642쌍).
4. **M10** — AOI 군집 13개 기준 spatial holdout 구축·동결. 게이트 6/6.
   train 393 / val 84 / test 113 / excluded 4 타일, 군집 간 최소이격 20,480 m.
   각 split 타일목록 SHA-256 + LOCO 13폴드 산출.
5. 데이터 계약 4층 봉인 (원본 zip / 파생 / 내용 / 코드), seal `5b088ada…`.
6. `tests/test_aihub_split_invariants.py` 10개 추가 — 동결이 깨지면 CI가 잡는다.
7. data.go.kr GK2A 10/10 오퍼레이션 통과. 당시에는 “보존기간 2일 → 소급 결합 불가”로
   판정했으나, **2026-08-25 정정**: 이것은 경량화 endpoint의 D-1/D-2 창이다. KMA 별도 L2
   archive가 존재하며 product parity는 미검증. 전향적 observability demo 역할은 유지한다.

**마찰**
- 라벨 파서를 클래스 키 추측으로 짜서 0건이 나왔다. 실제는 GeoJSON `ANN_CD`/`ANN_NM`이고
  인코딩이 파일 종류마다 다르다(라벨 UTF-8, 메타 cp949).
- 군집 간 거리를 bounding box로 재서 `min_gap=0`이 나왔다. 타일 단위로 재야 했다.
- 희소 클래스를 '존재 여부'로 세어 val 산사태가 4타일뿐이었다. '타일 수'로 바꿨다.
- SSH 터널이 9시간 뒤 죽어 push/실행이 조용히 실패했다. `nx tunnel up` 재시작으로 복구.

**다음**
- **[2026-08-25 우선순위 보정]** C2-C exact-scene recovery는 train+val 40표본·최대 1일의
  한국 지원 gate로만 실행한다. test 113타일에는 동결된 파이프라인을 1회만 적용한다.
- 주 임계경로는 Sen12Landslides Nepal + AI-Hub Korea + Swiss event의 3-country probe·static/live
  residual·leave-one-country-out이다. 공식 split 누수 성능 차이는 한국 arm의 보조 결과로 측정한다.

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

### 2026-08-25~26 — G-P 첫 성능 결과 독립 검증·공정 비교 복구 (완료)

- 계획: `holdout_chimanimani` P1/P2/P4 결과를 숫자표가 아니라 데이터→split→모델→학습→선택→평가
  전체 경로로 감사한다. 첨부 보고서의 8/40 epoch 주장과 서버 산출물을 독립 재계산하고,
  best-checkpoint test, 손실·불균형 처리, AUPRC/ECE 구현, LD 부분집합, 파라미터·시간·메모리·
  사전계산 비용을 확인한다.
- 사전 판정 기준: ① 세 arm의 sample ID·S12q 입력·mask·split이 동일 ② test는 모델/epoch/threshold
  선택에 미사용 ③ 최종 test는 val로 고른 best checkpoint에서 단 한 번 산출 ④ exact 또는 충분히
  정밀한 pixel AUPRC와 명시된 ECE ⑤ P4 비용은 cache extraction amortization을 포함해 보고
  ⑥ 코드·산출물 해시와 독립 재실행 테스트가 남아야 유효한 pilot으로 인정한다.
- 무효 조건: 마지막 epoch test를 best-epoch 성능으로 부르기, test를 매 epoch 조회하기, arm별 다른
  표본/augmentation/threshold 사용, 사후 epoch 연장 결과를 최초 사전등록 게이트처럼 표현하기,
  또는 frozen backbone 사전계산 비용을 0으로 보고하는 경우.
- 실행 순서: 로컬 정적 감사와 단위 테스트 → GPU1 작업/로그 비침습 점검 → 필요한 코드 수정 →
  작은 synthetic 회귀 테스트 → 기존 결과 재평가 또는 GPU1 재실행 → 약점부터 결과 기록.
- 감사 결과: 최초 8-epoch “P4 전 지표 1위”는 확정표에서 제외했다. batch마다 같은 20k pixel
  offset을 뽑는 AUPRC 편향, test 열람 뒤 epoch 변경, 265,649-param tiny P2를 공식 3D U-Net처럼
  부른 문제, P4만 timestamp를 받은 정보 비대칭, encoder/cache를 뺀 비용 과장, `>=50` LD 경계,
  300-mask pos_weight, arm 순서 의존 RNG, checkpoint/per-sample 부재를 확인했다.
- cache 복구: 6,834/6,834의 file set·shape/dtype·finite/range·binary mask·원 계약 positive count와
  content SHA를 전수 감사해 4/4 gate를 통과했다. cache seal 없이는 runner가 실행되지 않는다.
- strict final(code SHA `478c6af5…`): P1 IoU/AP 0.05406/0.07774, P2-tiny-factorized
  0.13499/0.28607, P4 frozen OLMo **0.14164/0.22512**. P4는 P1과 P2 IoU를 넘지만 AP는 P2의
  **78.7%**라 사전 95% G-P를 통과하지 못했다. P2가 official strong baseline이 아니고 P3·
  timestamp parity·미열람 9지역이 없으므로 상태는 실패가 아니라 **G-P BLOCKED**다.
- 재현성: RNG reset만 한 v2 P4 replay는 IoU가 0.12283→0.14344(+16.8%)로 갈렸다. strict
  algorithms/cuBLAS/cuDNN/TF32 계약과 factorized deterministic P2-tiny로 복구했다. final full P4와
  P4-only 40-epoch는 history(시간 제외), 모든 지표, per-sample/checkpoint SHA, 모든 tensor가 bitwise
  일치(max-abs diff 0). artifact verifier도 threshold aggregate를 독립 재계산해 통과했다.
- 비용 정정: P4 cache extraction 1,130.05초·10.75 GB·frozen encoder 88.96M을 별도 기록했다.
  같은 P4 cached fit도 full sequence 950.5초 vs 단독 520.0초라 wall-time 우위는 미측정이다.
- 검증: 전용 metric/artifact 테스트 6개와 전체 **159 passed, 1 skipped, 10 subtests**, py_compile,
  `git diff --check`를 통과한다. system Python은 PyYAML이 없어 collection 5건이 실패했으며,
  프로젝트 환경 `../.venv` 결과만 유효하다.
- 다음: Chimanimani 추가 튜닝을 멈춘다. 공식 Sen12 3D U-Net/U-TAE tensor regression → raw date/gap
  encoding 또는 P4 timestamp ablation → recipe 동결 → 미열람 9지역 region/event-macro G-P 순서다.
  그 뒤 같은 cache의 R-event를 닫아야 한국 T-m/T-x와 다중-task cache 주장을 연다.

### 2026-08-26 — M32–M36 주장 정정·E1 배관 복구·최근 연구 대비 재설계

- 작업 계획: 최신 evidence bundle과 M31–M35를 독립 재감사하고, 결과를 읽기 전에 E1 2×2의
  estimand·판정식·비용 규칙을 고정한다. 과장된 원인진단과 manifest 집계 오류를 코드·문서에서
  수정하고, AI-Hub v2 materialization 계약을 만든다. 그 뒤 최신 OLMo/PANGAEA/PEFT/embedding
  product/feature refresh 연구와 대조해 CVPR 잔여 gap을 다시 판정한다.
- M32 정정: 0.4987/0.6071은 모든 decoder가 넘지 못하는 token-grid ceiling이 아니라
  **block-constant label oracle 참고값**이다. 실제 embedding 정보 보존과 40 m support 병목은
  미판정이다. 면적 quartile만으로 thin-scar 가설도 기각할 수 없어 제목·코드·금지 주장을 고쳤다.
- M33/M34 정정: spatial block bootstrap의 tail fraction을 p-value로 부르지 않고 20.48 km 12블록의
  불안정성을 명시했다. seam v1은 ID 정렬 첫 300개·CI 없음이라 확증에서 관측으로 낮췄고,
  지역 균형 30개씩 + tiled/full difference-in-differences + 계층 bootstrap v2 코드를 추가했다.
- 증거 정정: v1 manifest의 `n_blank=1`은 blank unique value 수를 센 버그였다. P2 test 기준
  ann_id blank row는 **710/1,133**, nonblank distinct 422다. 원본 v1은 보존하고 correction manifest
  v2와 미래 seal script를 추가했다. timing 계약도 P2/P3 month 채널, P1 mean-month로 정정하되
  exact day/gap mismatch는 남겼다.
- E1 analysis lock: `y00/y01/y10/y11`, context·decoder contrast·interaction, 4개 spatial block CI,
  IoU/AUPRC 95% 탐색 기준(0.1512913/0.16585575), fixed/practical cost를 결과 전에 문서화했다.
  첫 새 셀 validation 일부를 이미 본 뒤라 완전 preregistration이 아니라
  `prospective analysis lock after run start`로 명시했다.
- M36 실측: tiled + 2.989M large convolutional decoder(P4c)는 IoU/AUPRC
  **0.177727/0.213574**로 small P4(0.130582/0.151348)와 P2(0.159254/0.174585)를 micro 지표에서
  넘었다. 그러나 positive-patch macro IoU는 **0.139172**로 small P4(0.159966), P2(0.194446)보다
  낮고 LD-IoU도 P2보다 낮다. 한 개발 지역·seed 1·test 노출 결과라 우월성으로 쓰지 않는다.
- E1 중단 원인: full128 root에는 embedding만 있고 base mask/raw/month/audit가 없는데 단일
  `--cache`로 결합돼 두 번째 셀이 시작 전 종료됐다. runner에 `--emb-cache`를 분리하고,
  alternate embedding exact set/shape/dtype/finite/content SHA를 base seal에 묶는 audit를 추가했다.
  최종 2×2는 동일 수정 code SHA로 재실행한다.
- AI-Hub M35 후속: 1,912는 usable이 아니라 post-hoc low-zero 후보로 낮췄다. v2는 같은 날짜·플랫폼
  item을 target EPSG:32652 grid에 warp/mosaic하고 alpha/source validity 기반 12-band common
  coverage **≥99.9%**만 통과한다. v1을 덮어쓰지 않고 40표본 층화 pilot→전수 health/class
  selection-bias audit 뒤에만 RQ2에 사용한다. 계약과 v2 materializer를 추가했다.
- 최신 연구 판정: strong decoder·PEFT, shared embeddings-as-data, tile seam, generic downstream
  regret refresh는 각각 OLMo/PANGAEA/PEFT, AlphaEarth/TESSERA, TESSERA v2, Berkeley RALF가
  선점했다. 남는 gap은 **label-free EO task-risk prediction + reuse/cached-adapter/re-embed/PEFT/raw
  multi-action policy + external-region regret–cost Pareto**로 좁혔다.
- 검증: project parent venv에서 **161 passed, 1 skipped, 10 subtests**, 수정 스크립트 py_compile과
  관련 18 tests 통과. system Python의 PyYAML 부재 collection failure는 환경 문제로 재확인했다.
- 원격 마찰: localhost:9922 SSH host fingerprint가 기존 known_hosts와 달라졌다
  (`SHA256:y6YfMhYlugocey1MiM5fS2Ggg0RFtQSMCexH2ryWziI`). read-only로 E1 로그·산출물은 확인했지만
  Backend.AI 세션 identity를 재확인하기 전 code push·GPU 재실행은 중단했다.
- 원격 재개: Backend.AI session ID를 로컬 state와 대조하고 위 fingerprint를 봉인한 뒤 code SHA
  4개가 로컬과 서버에서 일치함을 확인했다. 첫 재실행은 runner가 `/home/work/data`에서 상대
  `code/...`를 찾는 경로 결함으로 학습 전에 종료됐다. 결과·GPU 상태를 오염시키지 않았으며,
  project root로 `cd`하도록 고치고 셸 검사·추가 커밋 후 처음부터 재실행한다.
- 현재/다음: host identity·code push·full embedding seal은 완료했다. 동일 수정 code SHA로 E1 네 셀을
  처음부터 재실행 중이며, 완료 뒤 per-sample 검증과 factorial CI를 한 번 계산한다. 그 결과에 따라
  multi-level cached-token decoder 또는 PEFT 한 축만 열고, exact-time parity 뒤 미열람 region
  protocol을 동결한다.
- E1 완결(M37): 동일 runner SHA·paired 1,133 test tile로 `tiled-small 0.130582`,
  `tiled-large 0.177727`, `full-small 0.116565`, `full-large 0.081419` IoU를 얻었다. full-context
  평균효과 -0.055162의 공간 CI는 네 scale 모두 0 아래였고, decoder 효과는 tiled +0.047145에서
  full -0.035146으로 반전(interaction -0.082291)했다. 따라서 seam smoothness→성능, context와
  capacity의 독립 가산 개선을 모두 기각한다.
- 비용/주장 판정: tiled-large는 P2보다 micro/AP가 높지만 positive-patch macro·LD-IoU가 낮고,
  head fit 1,596.2초 + cache 1,130초라 P2 1,491초 대비 Pareto 우위가 없다. 한 노출 지역·seed 1의
  개발 결과이므로 우월성·일반화 주장은 금지한다. 증거 1.3 MB와 입력 SHA를
  `evidence/e1_factorial_v2/`에 봉인했다.
- 다음: full-context는 이 개발 계약에서 중단한다. exact-time parity를 먼저 닫고 tiled-large와 P2를
  공통 seed·공통 threshold-selection 규칙으로 반복한다. positive-macro·비용이 여전히 약할 때만
  multi-level decoder 또는 PEFT 한 축을 열고, 그 뒤 미열람 지역 protocol을 동결한다.
- 검증: evidence 입력파일 SHA·analysis code SHA·runner code SHA를 로컬 테스트에서 다시 대조했고,
  project venv 전체 **164 passed, 1 skipped, 10 subtests**, `git diff --check`를 통과했다.

### 2026-08-26 — M37 이후 논문 claim 확장·최신 경쟁선 재조사

- 목적: M37의 context×decoder interaction을 과장하지 않으면서 CVPR method로 확장 가능한 정확한
  빈칸을 찾는다. EarthShift/GEO-Bench-2/embedding product, CrossEarth-Gate/DARN/DEFLECT adaptation,
  GdScore/ODD/IUPM/agreement/conformal/testable-learning을 1차 출처로 대조했다.
- 증거 경계: M37이 허용하는 것은 Chimanimani 한 개발 fold에서 `seam/smoothness 개선 ≠ task utility
  개선`이라는 존재 반례와 serving-context×decoder interaction뿐이다. context·decoder·OLMoEarth의
  일반 우열, task heterogeneity, label-free prediction, router Pareto는 모두 미측정으로 유지한다.
- claim 재정의: generic refresh router가 아니라 **Contract-Conditional Action Utility Estimation for
  Shared Earth Embeddings**로 좁혔다. source/development label은 쓰되 새 target label은 action 선택에
  쓰지 않고, `reuse / repair / re-embed / task-raw`의 task별 gain을 예측한다.
- 비용 재정의: representation extraction은 여러 task가 한 번을 공유하고 task head/raw 비용은 각각
  부담한다. facility-location/knapsack optimizer 자체가 아니라 EO action-value matrix의 예측 가능성과
  regret/Pareto가 기여 후보다.
- 강한 경쟁선: EarthShift는 labeled robustness, CrossEarth-Gate는 Fisher-guided PEFT selection,
  DARN은 difficulty-aware decoder gate, GdScore/ODD/IUPM은 label-free performance monitoring을 이미
  점유했다. 따라서 contract-only·entropy·drift·gradient·overlap·agreement baseline을 모두 둔다.
- 이론 경계: `no labels anywhere`나 보편 보장을 금지한다. predefined EO shift family에서
  `source-labeled meta-training + target-unlabeled selection`으로 한정하고, support 밖에서는
  abstain/small audit-label action을 둔다.
- 지역 결정: AI-Hub v2는 같은 cube의 3-task heterogeneity, EarthShift는 공개 shift 재현을 담당한다.
  external test는 둘을 동시에 열지 않는다. Nepal은 동일 landslide semantics가 강하지만 2024 Koshi
  inventory의 U-Net-assisted label provenance를 감사해야 하고, Switzerland는 STAC/COG/mask/registration/
  license가 깨끗하지만 avalanche로 task가 달라진다. 현재 추천은 Nepal 한 곳 untouched, Switzerland는
  후속 operational track이다.
- kill path: task×action ordering이 같으면 router를 즉시 중단한다. action-gain predictor가 GdScore·
  ODD·agreement 등을 못 이기면 `When Earth Embedding Diagnostics Mislead` analysis/benchmark로
  전환한다. CVPR에는 proxy failure→heterogeneity→label-free action value→joint Pareto→second
  backbone/untouched country가 모두 필요하다.
- 산출물: `docs/PAPER_CLAIM_EXPANSION_2026_08_26.md`를 claim SSOT로 추가하고 README,
  `docs/CRITICAL_PATH.md`, `PAPER_READING_LIST.md`를 동기화했다.

### 2026-08-26 — 외부 지역 onboarding 계약·upstream PR 큐 재감사

- 작업 계획: 한국·네팔·스위스 자산을 OLMoEarth input/context/target으로 다시 분류하고, 현재
  rslearn/olmoearth_projects upstream에서 PR 후보가 여전히 살아 있는지 재검증한다. 제출 상태를
  로컬 브랜치와 혼동하지 않고, 의미 있는 후보만 current queue로 남긴다.
- 입력계약 판정: 공개 rslearn OLMoEarth 경로는 canonical S2 12밴드/S1/Landsat의 band order,
  normalization, 10 m, time contract를 요구한다. AI-Hub label은 canonical S2 target으로 조인하고,
  KMA/GK2A/DEM은 residual, swissEO 7-band는 canonical transfer가 아니라 별도 missing-band/source
  shift로 둔다.
- Nepal 정정: Koshi 2024는 CC BY 4.0이고 geography transfer에는 유용하지만 U-Net 자동탐지 후
  manual QC된 silver label이다. 과거 Worklog의 `Nepal untouched`는 `untouched geography`로만
  정정하며, manual adjudication 없이는 untouched gold claim을 금지한다.
- upstream 재감사: `olmoearth_projects origin/main=23a3d7b`, rslearn
  `v0.1.14/master=c47952f`를 기준으로 확인했다. sample schema #1은 upstream/open PR과 중복 없이
  유효하다. direct-materialize #4는 current에서 해소, 미출시-API #5는 lock skew #13에 병합,
  docs #9는 clarification으로 강등했다. SCL #10은 nearest-only PR과 dependency RFC로 분리한다.
- PR #1 보완: GeoJSON EOF newline을 추가했고 JSON parse·6 feature·legacy key 0·dict label gate를
  통과했다. macOS current runtime은 schema 이후 forkserver에서 다시 hang해 #8을 재확인했다.
  Linux 0.1.14의 기존 6-window 완주는 유효하지만 제출 전 current Linux replay는 남아 있다.
- 동시 M40 재감사: current output-confidence rule의 label-free winner prediction 실패는 유효한
  음성 결과다. 다만 한 task/region의 recipe gate이고, 후보 출력 자체의 pre-action availability/cost가
  정의되지 않았으며, tile-IoU로 선택한 뒤 micro-IoU를 보고해 pairwise oracle gain이 음수가 되는
  objective mismatch가 있다. metric-aligned oracle과 noise-floor/FP-rate 통제 전에는 EarthRoute
  전체 kill로 승격하지 않는다.
- 산출물: `docs/OLMO_EXTERNAL_DATA_ONBOARDING_AND_PR_AUDIT_2026_08_26.md`, 갱신된
  `PR_DOSSIER.md`, `PR_REVIEW_NOTES.md`, PR body, README/CRITICAL_PATH/STUDY.
- 다음: AI-Hub v2 eligibility를 닫은 뒤 세 task의 action ranking 이질성을 먼저 잰다. PR 축은
  Linux current replay→사용자 승인 후 fork/push 순서다.

### 2026-08-28 — Nepal live data 갱신·검은 지도 원인 폐쇄

- 계획: 배포 화면의 검은 배경을 브라우저에서 재현하고, 지도 렌더·타일·장면 데이터 상태를
  분리 진단한다. Copernicus catalog를 새 immutable snapshot으로 갱신한 뒤, `게시됨`,
  `provider가 선택함`, `OLMo 입력 물질화`, `임베딩 계산`을 서로 다른 게이트로 표시한다.
- 배경 원인: OSM direct tile이 HTTP 200이지만 실제로는 동일 6,933-byte 차단 응답을 반환했다.
  외부 raster/WebGL과 무관하게 최신 로컬 S2 PNG를 full-screen DOM backdrop으로 렌더하고,
  MapLibre canvas resize와 WebGL2 fallback을 보강했다. 로컬 브라우저에서 전 화면 위성 배경,
  패널·timeline·Rust/WASM flow를 함께 확인했고 console error 0이다.
- 최신 catalog: snapshot `20260828T000910Z`, seal SHA-256
  `1c3be74f71e6c43d99c5cbf2ec6eaaefd42314e781dd4cd2ecb9bcfe53934ba2`.
  S1 GRD 15, S2 L2A 31 acquisitions. S2B 08/27 L2A는 18:33 KST 게시,
  product `S2B_MSIL2A_20260827T045659_N0512_R119_T45RUM_20260827T084453.SAFE`,
  tile cloud 78.471315%다. 이 수치는 AOI clear coverage가 아니다.
- 실측 차단: official CDSE에는 제품이 있지만 Planetary Computer STAC가 아직 08/27 장면을
  반환하지 않아 rslearn은 5/5 앵커에서 08/24를 선택했다. 따라서 post cube·embedding delta는
  미측정이며 화면도 `CATALOG / CUBE WAIT`로 표시한다.
- 개선: `check_nepal_live_selection.py`를 추가해 live item date가 5/5 앵커에 없으면
  materialization 전에 exit 4로 중단한다. `prepare_nepal_olmo_live.sh`의 잘못된 materialize
  `--force` 전달도 prepare 전용으로 분리했다. partial `s2_live`는 unsealed 상태로 유지한다.
- 다음: 08/28 21:19 KST S1D 획득 뒤 official catalog와 provider STAC를 재조회한다. S2/S1
  selection preflight가 5/5를 통과할 때만 materialize→seal→OLMo v1 embedding→placebo-calibrated
  delta로 진행한다. 카탈로그 metadata만으로 damage/anomaly를 주장하지 않는다.

### 2026-08-28 — Nepal evidence operations 독립 재감사 (완료)

- 계획: `c5872b6`의 OPERATIONS LOG·AOI 관측성·M17 검색 보조지표를 원 산출물과 독립 대조한다.
  EarthRanger·Skylight·Copernicus EMS·FIRMS·Global Forest Watch의 공식 운영 문법을 비교해,
  단순 이벤트 피드가 아니라 `관측 → 계약 게이트 → 후보 → 독립 증거 → 사람 검토 → incident`로
  승격할 최소 기능과 연구 평가를 고정한다.
- 사전 판정 기준: ① UI의 `OLMo READY`는 selection preflight뿐 아니라 5/5 materialization seal과
  4기간×모달리티 계약을 모두 요구한다 ② `bright pixel`을 cloud-free로 해석하지 않는다
  ③ 검색 AP는 표준 AP@K 정의와 단위 테스트를 통과해야 한다 ④ 두 placebo만으로 95% 이상치나
  재해 탐지를 주장하지 않는다 ⑤ 모든 이벤트는 관측시각·기록시각·근거 URI를 분리한다.
- 현재 발견: `s2_live`는 S2 4/4지만 S1 3/4라 seal invalid인데 scenario가 preflight만 보고
  `olmo_ready=true`를 냈다. 또한 M17의 기존 `AP@100`은 top-100 안에서 찾은 양성 수를 분모로 써
  표준 AP@100이 아니다. 두 값은 교정 전까지 주장·판정에서 제외한다.
- 구현: scenario compiler의 readiness를 selection+materialization seal로 묶고, 화면 상단에
  `DO NOT EMBED / S1 3/4 · S2 4/4` 결정·다음 gate·허용 claim을 표시했다. ops event마다
  event ID·관측/기록 시각·evidence URI를 넣었고 site verifier가 invalid seal과 중복 ID를 막는다.
- 관측성 정정: B02 임계의 보완값을 `clear_dark_frac`에서 `not_bright_frac_of_valid`로 바꾸고
  threshold·scene·raster SHA·“cloud classifier가 아님”을 schema v2에 봉인했다. Rasuwagadhi
  2.52% bright는 재현되지만 cloud-free 97.48%라는 뜻이 아니다.
- 검색 정정: 기존 AP@100 세 수치를 철회하고 표준 AP@K(분모 `min(total_relevant,k)`)와
  Recall@100을 구현했다. 이전 오류를 재현하는 회귀 테스트를 포함해 4/4 통과했다. P@10과
  preregistered gate 실패는 영향 없음; 새 AP 수치는 동일 산출물 재실행 전까지 공란이다.
- 제품 승격: EarthRanger incident/history, Skylight schedule/review/feedback, CEMS 제품 버전,
  Global Nature Watch 다중증거 confidence를 `SCHEDULED→CATALOGUED→SELECTED→SEALED→EMBEDDED→
  CANDIDATE→CORROBORATED→REVIEWED` 상태기계로 결합했다. 상세는
  `docs/NEPAL_EVIDENCE_OPERATIONS_REVIEW_2026_08_28.md`.
- 실험 경계: materialized placebo가 A/B 두 개뿐이므로 95 percentile anomaly는 금지했다.
  첫 sealed delta는 descriptive candidate change만 허용하고, 최소 20개(권장 30개+) historical
  placebo와 multi-event A0–A4 비교 전에는 CVPR method claim으로 승격하지 않는다.
- 검증: retrieval metric pytest 4 passed, site `verify`·`lint`·production build 통과. 현재 서버
  `./bin/nx status` 조회가 실패해 GPU/queue 순간 상태는 확인되지 않았으며 추정하지 않는다.

### 2026-08-28 — Nepal sidecar 주차·CVPR transfer 임계경로 복귀 (완료)

- 계획: Nepal live event는 현재 산출물·금지 주장·재개 조건을 immutable handoff로 남기고 주
  실험 queue에서 제거한다. Sen12 confirmatory 8-region, 두 번째 GeoFM(Presto), 한국 transfer의
  실물 폴더·manifest·서버 상태를 대조해 정확한 재개점을 고정한다.
- 사전 판정 기준: ① 실행 중인 확증 작업이나 다른 GPU 프로세스를 건드리지 않는다 ② 서버 결과는
  post gate·snapshot·sample-set seal을 통과하기 전 로컬 측정 장부로 승격하지 않는다 ③ 8-region
  headline을 닫기 전 Nepal 단일-event 결과나 새 adapter/physics 실험을 열지 않는다 ④ OLMo 고유
  주장은 Presto 정규화·commit·동일 decoder 비교 전 금지 ⑤ Korea transfer는 8-region aggregate와
  C-arm 결과를 읽은 뒤 사전등록한다.
- 첫 상태 확인: 로컬은 Thrissur·Hiroshima·Hokkaido 3지역만 판독된 것으로 보였으나, 서버에는
  **8/8 confirmatory 지역 디렉터리와 각 9개 arm×seed 로그·read_summary가 모두 존재**한다.
  현재 학습 프로세스는 grep상 없지만 GPU0/1은 각각 34.8/34.9 GiB를 사용 중이므로 새 GPU 작업은
  금지한다. 남은 일은 실행이 아니라 원격 5지역의 post-gate·출처 검증과 안전한 회수다.
- 확증 회수·판정: 8지역 모두 post manifest PASS. Thrissur는 공개된 M57 snapshot 예외를 유지하고,
  나머지 7지역은 snapshot timing/required files/SHA를 포함한 13/13 gate를 통과했다. 8-region
  region-macro P4/P2/P3는 **.272166/.196558/.183436**, P4−P2 **+.075608**. 사전등록 win 6/8,
  strong-win 5/8이며 Indonesia는 −.011294, Itogon은 +.004014이나 seed 3 음수로 non-win이다.
  Thrissur 제외 sensitivity도 +.068220으로 방향이 유지된다.
- 봉인: `code/summarize_confirmatory_8region.py`가 recipe SHA·arm/seed·gap arithmetic·post gate·
  snapshot을 검증하고 `artifacts/confirmatory_8region_summary.json`을 생성한다. M65와 README,
  CRITICAL_PATH, ASSET_INVENTORY, 실행계획을 같은 수치로 동기화했다.
- Presto 감사: 공식 upstream commit `11e207a…`와 서버 code/weight/normalization이 byte-identical이고
  S2는 shift 0, `/10000`이 맞다. 2-D month tensor로 S12q 실제 월 벡터를 넣을 수 있음도 확인했다.
  `config/presto_c1_contract.json`에 commit·세 SHA·band/month/latlon/decoder 계약을 고정했다.
  기존 8지역은 이미 P2/P3/P4 결과가 열렸으므로 C1은 matched retrospective control로 강등하고,
  최초 untouched OLMo-vs-Presto 비교는 한국이 맡는다.
- Nepal 주차: baseline/placebo 3 mode×5앵커 embedding은 봉인됐지만 event `live_mode=null`이고
  S2 live cube는 S1 3/4라 invalid다. 금지 주장과 네 재개 조건을
  `docs/NEPAL_SIDECAR_HANDOFF_2026_08_28.md`에 고정했다.
- 다음 실행: GPU1이 비면 Presto 16/64/256픽셀+1타일 smoke(exact month·WGS84 lat/lon) →
  6,834 cache seal → 결과 관찰 전 recipe v3 → 8-region×3seed C1 → Korea recipe 동결 순이다.
  현재 두 GPU가 타 작업에 점유돼 있어 새 GPU 실행은 의도적으로 시작하지 않았다.

### 2026-08-29 — Nepal live twin 재개: 사건·AI2·물리·스토리 검증 (진행 중)

- 계획: 사용자가 갱신한 `apps/nepal-olmo-gis`의 MapTiler/스토리/포인트/flow/data 상태를 실물과
  대조한다. 2026-08-26 Rasuwagadhi 사건과 08-27 전후 Tibet/HKH 추가 사건의 공식 보도·좌표·
  원인 확정도를 다시 검색하고, glacier/landslide/flood corridor를 관측·추정·미확인으로 분리한다.
  OLMoEarth가 수행할 수 있는 embedding extraction/change/retrieval과 할 수 없는 물리 예측을
  구분하고, DEM·SAR·강우·빙하/호수 inventory·mass-flow simulation의 결합 계약과 평가표를 만든다.
  마지막으로 Snow Fall/Upshot/Economist식 스크롤 서사를 실제 증거 상태에 맞춰 강화하고 배포한다.
- 사전 판정 기준: ① 사건 날짜·좌표·원인은 2개 이상 독립 출처 또는 1차 기관 없이는 확정하지 않는다
  ② OLMo embedding은 valid multimodal seal과 event/post mode 없이는 live 변화로 표시하지 않는다
  ③ 물리 애니메이션은 calibrated runout이 아니면 route/illustrative simulation으로 명시한다
  ④ Tibet 사건과 Nepal 본 사건은 공간·수계 연결이 입증되지 않으면 별도 event로 유지한다
  ⑤ AI2 가치 평가는 탐지 성능뿐 아니라 latency, abstention, retrieval, spatial false alarm,
  evidence-to-decision time을 포함한다 ⑥ 본 CVPR transfer queue의 GPU/확증 코드는 건드리지 않는다.
- 예상 산출물: 최신 source ledger와 claim matrix, OLMo×physics 결합 평가 설계, 강화된 story sections,
  데이터/앱 검증 결과, immutable handoff와 재개 조건. 외부 기사 문구를 시각 카피로 옮길 때는
  사실과 편집적 해석을 분리하고 출처를 화면 가까이에 둔다.
- 결과: USGS·CGS·ICIMOD·AP·중국 국무원 자료를 교차해 source를 Nepal-side Langtang Lirung으로
  정정했다. 08-27 별도 Tibet landslide 근거는 없으며, 같은 26일 사건의 second signal과
  barrier-lake aftermath로 분리했다. 속도·피해구조물·clear-AOI 등 출처 범위를 넘은 카피를 제거했다.
- 최신 관측: catalog snapshot `20260828T151656Z`와 regional-footprint audit
  `20260828T152324Z`를 봉인했다. 08-28 S1D 인접 제품 2개가 AOI를 사이에 두고 지나가
  `aoi_covering=0`, `MISSED_COVERAGE`로 판정했다. 화면과 decision engine이 놓친 pass를 기다리지
  않고 08-31 S1D를 다음 gate로 가리킨다.
- OLMo 경계: Nepal event embedding은 S1 3/4·S2 4/4라 계속 `DO NOT EMBED`다. M66은 related
  S2-only historical pilot, M67은 pre-event susceptibility `not detected`로 화면에서 분리했다.
  OLMo는 source/change/analogue proposal, r.avaflow는 runout, D-Claw는 독립 확인,
  LISFLOOD-FP/BASEMENT는 조건부 downstream stage를 맡도록 계약했다.
- 제품/평가: storyboard를 evidence-now→event clock→optical/gaps→OLMo→physics→A0–A5 test→
  next clock→ledger로 확장했다. 운영 주지표는 matched recall에서 invalid action·analyst minutes,
  과학 지표는 event AUPRC·source error·runout IoU·maximum-runout error·interval coverage다.
- 검증: Python compiler, asset verifier, TypeScript, ESLint(오류 0; 정적 image 권고 6), Rust/WASM
  280 particles, vinext production build를 통과했다. 로컬 MapTiler 지도·MISSED evidence card·영/한
  story를 브라우저에서 확인했고 스타일 reload race를 제거했다. 상세는
  `docs/NEPAL_OLMO_PHYSICS_STORYBOARD_UPDATE_2026_08_29.md`와 handoff 재개 감사에 기록했다.
- 상태: **완료 — live embedding은 의도적으로 미실행**. 08-31 실제 footprint→selection→4+4 seal이
  통과하기 전까지 재계산하지 않으며 CVPR 본선 GPU queue는 변경하지 않았다.

### 2026-08-29 — Nepal live twin: 스펙트럼 섹션·전체 검증·배포 준비 (이 세션)

- 병렬 세션의 감사 결과(`4bb36f2`: Langtang Lirung 정정, MISSED_COVERAGE, 물리 결합 계약,
  스토리 11섹션)를 실물 대조로 확인했음. 재개 조건 4개 중 1·2가 여전히 미충족이므로
  live embedding은 실행하지 않았음.
- 스토리에 04b THE SPECTRA를 추가했음(`f2d533f`): 같은 08-27 Rasuwagadhi 창의
  트루컬러/SWIR(B12·B8A·B04)/NDWI 3연 패널 + 사건 전 08-12 SWIR 비교. SWIR에서
  debris 회랑(분홍-갈색)과 내부 물길(청색)이 트루컬러보다 명확함을 실물 확인했음.
- 전체 검증 통과: pnpm data(route 79) · tsc 0오류 · eslint 0오류/10경고 · asset verifier
  (particles 280, worker 508,167B) · production build · 헤드리스 스크린샷(메인/스토리 12섹션).
- 신규 관측 없음: 08-28 12:00 UTC 이후 PC STAC에서 AOI 포함 S1/S2 제품 0건 재확인.
  다음 게이트는 08-31 00:07 UTC S1 후보이며 footprint containment로 재판정함.
- 배포 준비 완료: 최신 소스는 이 저장소 HEAD. seeso Sites 재배포는 해당 세션의
  퍼블리시 도구에서 이 HEAD 기준으로 version 7을 올리면 됨.

### 2026-08-29 — Nepal live twin: 사건 인과선·OLMo 가치·physics fusion 재설계 (진행 중)

- 계획: UI의 A/B/C 문자 중심 문법을 폐기하고 `SOURCE → IMPACT → BORDER → DOWNSTREAM`의
  사건 단계와 `NEGATIVE CONTROL`을 색·순서·지도 라벨로 분리한다. C는 본 수계와 무관한
  대조군이며 본 사건 경로에 포함되지 않는다는 사실을 모든 화면에서 강제한다.
- 데이터 감사: Bidur를 OLMo 5-anchor 봉인 계약에 사후 삽입하지 않고, 별도 visual-only
  downstream anchor로 취급한다. 로컬 STAC catalog에서 실제 S1/S2 footprint와 날짜를 감사한 뒤
  장면이 있으면 재료화하고, 없으면 `NO COVERAGE / NOT MATERIALIZED`를 구분해 표시한다.
- 제품 재설계: 현재의 HOLD 중심 우측 rail을 `baseline embeddings READY`, `8-region transfer
  6/8 wins`, `Nepal live delta WAITING FOR S1`, `physics ensemble DESIGN`으로 분리한다. 메인 서사는
  사건 사슬, 위성 시간×거리 행렬, OLMo의 검증된 현재 가치, OLMo×mass-flow×observation-operator
  검증 루프, 다음 실행 우선순위의 다섯 장으로 축소한다.
- 과학 경계: OLMo embedding이 마찰계수·유량·도달시간을 직접 출력한다고 주장하지 않는다.
  OLMo는 source/change/analogue proposal, r.avaflow 또는 D-Claw는 runout ensemble, 위성 observation
  operator는 `시뮬레이션 결과가 S1/S2에서 어떻게 보여야 하는가`를 담당하고, 실제 post-event
  관측과의 일치도로 ensemble을 재순위화한다.
- 성공 기준: ① source/impact/control을 5초 안에 구별 ② OLMo의 READY 성과가 HOLD보다 먼저 보임
  ③ Bidur 공백의 원인이 명시됨 ④ 물리 애니메이션이 단순 강 중심선 장식이 아니라 관측으로
  반증 가능한 experiment graph로 보임 ⑤ tsc·lint·production build·브라우저 QA·비공개 배포 통과.

#### 결과

- 사건 사슬을 `E source(red) → D secondary hazard(purple) → A impact(orange) / B border(yellow)
  → F Bidur(blue)`로 재구성하고 C control(gray)을 사건 밖으로 분리했다. 지도 라벨도 긴 역할명이
  아닌 `E · SOURCE`, `A · IMPACT`, `F · BIDUR`로 축약해 전체 회랑 줌에서 읽히게 했다.
- Bidur 공백은 영상 부재가 아니라 MGRS 경계 결함이었다. 기존 `45RUM` 조회 밖의 인접 `45RUL`에서
  08-12/08-27 실제 S2 L2A 장면을 찾고 동일 2.56 km 창으로 물질화했다. visual-only manifest와
  checksum을 저장했으며 5-anchor OLMo 계약에는 사후 삽입하지 않았다.
- 우측 rail과 story를 baseline 5-anchor READY, confirmatory transfer 6/8 wins(0.272 vs 0.197),
  Nepal live Δ WAIT S1으로 분리했다. confirmatory 결과는 OLMo-vs-raw이며 Presto 통제 전에는
  OLMo-specific superiority로 쓰지 않는 경계를 UI와 roadmap 양쪽에 넣었다.
- 검증 루프를 `OLMo proposal → r.avaflow ensemble → D-Claw check → semantic observation operator
  → Rasuwagadhi/Bidur actual observations`로 고정했다. 첫 위성 시뮬레이션은 가짜 사진이 아니라
  water/debris/visibility mask다.
- `pnpm data`, TypeScript, asset invariant, WASM build, production build 통과. 실제 브라우저에서
  한국어 사건 카드, Bidur 전후 영상, OLMo 성과, physics-fusion 도해를 시각 검증했다.
- 상세 설계와 P0–P4 우선순위는
  `docs/NEPAL_AI2_IMPACT_ENGINEERING_ROADMAP_2026_08_29.md`에 저장했다.
- 14:15 KST 갱신: 08-28 S1D가 공식 카탈로그에 늦게 나타났고, 강화된 감사에서 제품 2개가
  5/5 anchor를 모두 덮었다. 다만 rslearn/Planetary Computer는 아직 08-24를 선택해 08-28 match가
  0/5였다. `PREPARE_ONLY=1` preflight에서 대용량 다운로드를 중단했고 UI gate를
  `WAIT FOR S1`에서 `WAIT FOR PROVIDER SYNC`로 정정했다. 08-29 S2C는
  `acquired_pending_catalog` 상태다.
- 앱 소스 `c198aa5`를 Sites 저장소에 push하고 production build archive로 version 8을 저장했다.
  기존 사이트가 public이므로 새 version의 공개 배포는 사용자 명시 승인 전까지 시작하지 않았다.

### 2026-08-29 — Nepal live twin: planetary response stack 확장 (완료)

- 계획: 단일 OLMo 데모를 `관측 → 다중 GeoFM 후보 → 물리 타당성 → 인간 노출·보건 영향 →
  독립 증거·사람 검토`의 planetary response stack으로 확장한다. OLMoEarth를 중심 표현으로
  유지하되 Prithvi/Presto/TerraMind/AlphaEarth 계열은 입력에 억지로 끼우지 않고, late fusion·
  candidate consensus·teacher/student·abstention 중 계약에 맞는 결합만 허용한다.
- 사전 판정 기준: ① 다른 모델은 동일 입력·라벨·decoder 통제가 없으면 `보강 후보`일 뿐 OLMo
  우월성 근거가 아니다 ② 물리모델은 DEM·초기조건·불확실성 ensemble과 사후 관측 검증 없이는
  경로 애니메이션을 넘는 예측이 아니다 ③ 질병·사망·개인 위치는 원격탐사로 직접 추론하지 않고
  시설 접근성·WASH·인구노출의 검토 대기 후보만 만든다 ④ SNS는 공식 embed/API 또는 사용자가
  등록한 공개 URL만 provenance card로 쓰고 스크래핑·신원추적·ground truth 취급을 금지한다
  ⑤ 현행 CVPR transfer GPU queue와 확인 실행 코드는 건드리지 않는다.
- 예상 산출물: 최신 1차 자료 기반 모델/서비스 결합표, candidate funnel과 평가 계약,
  humanitarian/health 안전 경계, 앱의 response-stack·candidate queue·human-impact story,
  TypeScript/lint/build/browser 검증, 공개 배포 전 승인 가능한 새 Sites version.

#### 결과

- 최신 공식 catalog를 다시 봉인했다(`20260829T052655Z`). 08-28 S1은 공식 footprint 5/5지만
  rslearn provider selection은 계속 08-24를 골라 exact-period match 0/5다. 따라서 대용량
  materialization과 Nepal post-event OLMo embedding은 실행하지 않았다.
- 앱을 `OBSERVE → REPRESENT → EXPLAIN → IMPACT` stack, C0–C6 candidate funnel,
  WHO-verified human-impact/access lens로 확장했다. 다른 GeoFM은 OLMo 입력밴드처럼 가장하지 않고
  matched control·late fusion·candidate cascade로만 연결한다.
- 사용자의 “AI가 무엇을 했는가” 지적을 반영해 hash-linked `ai_run_ledger`를 데이터 생성기에
  추가했다. 실제 상태는 6행으로 강제한다: Nepal pre-event OLMo `EXECUTED`, 8-region transfer
  `MEASURED`, historical delta `MEASURED_PILOT`, prospective susceptibility `NEGATIVE_RESULT`,
  Nepal live delta `WAITING_INPUT`, Presto matched control `NOT_RUN`.
- 기존 `5 × 768-d` 표기는 실제 artifact를 지나치게 축약했으므로 바로잡았다. 봉인 산출물은
  baseline/placebo 2개 × 5 anchor = **15 embedding GeoTIFF**, 각 **768×64×64 float32 spatial
  grid**다. 지도·story·Rust/WASM particle은 AI 출력이 아니라 이 산출물의 전달/감사 UI임을
  화면과 `docs/NEPAL_WHAT_THE_AI_ACTUALLY_DOES_2026_08_29.md`에 명시했다.
- 연구 종합은 `docs/PLANETARY_RESPONSE_STACK_RESEARCH_2026_08_29.md`에 저장했다. 제품 주지표는
  정확도 단독이 아니라 matched recall의 false-candidate area, analyst minutes, invalid-action rate다.
- 검증: `pnpm data`, TypeScript, ESLint(오류 0; 기존 `<img>` 권고 6), asset verifier, Rust/WASM,
  vinext production build 통과. 생성 JSON에서 AI ledger 6행·15 raster·상태 불변식을 검사한다.
  브라우저는 이전 localhost connection-failure 페이지의 URL policy 때문에 최종 재진입이 차단되어
  우회하지 않았고, 직전 화면 구조 QA와 정적/빌드 검증을 근거로 남겼다.
- 상태: **구현·연구 문서 완료, 공개 배포 보류**. 기존 public site는 사용자 명시 승인 없이
  교체하지 않는다. 과학적 다음 P0는 UI가 아니라 Presto matched control과 Nepal post-event
  exact-period seal이다.

### 2026-08-29 — 이 세션: 카탈로그 재감사·M68 전수 확장·Planet 공개영상·스토리 기사체 (완료)

- 카탈로그 재감사: 공식 CDSE에 08-28 S1D(12:21:41–12:22:06) 제품이 Rasuwagadhi를 포함해 있고
  Planetary Computer에도 GRD 원본은 인덱싱됨. 파이프라인 입력인 RTC 파생물만 미생성
  (PC RTC 최신 08-28 10:46). 따라서 "카탈로그 문제"가 아니라 RTC 생성 지연임. GRD 대체 입력은
  사건 전 계약과 달라지므로 쓰지 않음.
- Planet 재난 공개 데이터(source.coop, CC-BY-NC-4.0) 발견: PlanetScope 3.8m 08-26/28,
  SkySat 0.5m·Pelican 0.55m 08-27. 앵커 창을 잘라 실물 확인 — SkySat은 Rasuwagadhi 전면 구름,
  PlanetScope 08-28은 합류부가 맑음(밝은 픽셀 5%). 참고 영상으로만 앱에 넣고 AI 입력엔 미사용.
- M68: GPU1 유휴 시간에 다지점 Δz를 15지역으로 확장(863패치). 9지역 판정·6지역 데이터 부적격.
  다중 날짜 event_date로 1회 크래시 → 제외 규칙 추가 후 재실행. 보호 4파일 mtime 불변 확인.
- 앱: 스펙트럼 패널 복원, 스토리 한국어 29개 문단을 기사체로 개작, PlanetScope 영상을 2장과
  팝업에 추가. tsc/eslint/asset/build 통과, DOM 확인. 커밋 `51fa327` 외.
- 미해결: RTC 생성 대기(자동 재개 금지 유지), Sites 공개 배포는 사용자 승인 대기.

### 2026-08-29 — 이 세션 후반: 회랑 후보 지도(M69)·flow 가시성 수정·Planet 영상

- M69: 27 자동 창 S2-only Δ 순위. v1 결함(NaN 1위·과엄격 마스크) 보존 후 v2 재실행.
  상위 = Timure·Trishuli Bazar·Devighat, 사용자 앵커 Rasuwagadhi는 12위(placebo 높음).
- 앱: flow 입자가 밝은 배경에서 가산혼합 민트색이라 안 보이던 결함 수정(source-over·진한 색).
  AI 후보 층·목록 추가. PlanetScope 08-28 3.8m 참고영상(CC-BY-NC) 2장·팝업.
- 커밋: `cd1fb51`(flow), 후보 지도 커밋, M69 문서 커밋. 공개 배포는 여전히 사용자 승인 대기.

### 2026-08-29 — 이 세션: RTC 도착 → 봉인 s1_live 임베딩 → 첫 라이브 Δz (M70), M69 후보 지도·검색, UX

- RTC 지연을 실측(24h 배치)하고 도착 감시 → 도착 즉시 catalog·preflight(5/5)·materialize·seal(valid)·
  GPU1 임베딩(seal 661b19c8…)·delta 판정까지 완주함. 3/5 앵커 candidate change, 2/5 not detected.
- M69 회랑 27창 S2-only 후보 지도(v1 결함 보존, v2), 변화-벡터 검색(질의 1,324토큰, Timure·Lingling·
  Tupche 상위, Thulo Bharkhu 신규 부상). 앱: 후보 카드(전·후·AI Δ·지명·거리), 라이트박스 슬라이더,
  번호 마커, 검색 목록, flow 입자 가시성·레이어 재시도·카메라 복귀 결함 수정, 스토리 기사체.
- 미해결: s2_live S1 4번째 레이어 미물질화(불필요), Galchhi까지 창 확장, placebo 20+ 확보 전
  percentile 금지 유지, 공개 배포 승인 대기.

### 2026-08-29 (밤) — placebo 10개·매칭 설계(M72), 스캔 v2 100창(M71), 회랑 봉인 계약 진행

- placebo 확장 8창 물질화·봉인·임베딩 완료(터널 단절로 1회 전면 재업로드). 기존 분석기로는
  5/5 not detected → M70 철회. 매칭 1기간 쌍(n=9)에서는 rasuwagadhi만 1/10(0.0002 차).
  결론: 앵커 평균 Δ는 무딘 지표, 토큰 수준 근거로 이동. 앱 판정 카드 "NOT DETECTED" 로 내림.
- 스캔 v2: 연속 강변+Lhende+산사면 격자 100창, 47 판정/53 관측불가(산사면 43). 강변 밖
  lead v064(Salê)·v056. 앱: 종류별 색, 산사면 목록, 100창 자산·지명, GO→지도 드레이프,
  헤드라인 카드, Galchhi까지 강 선.
- 진행: 회랑 27창 봉인 계약(baseline 물질화 중 → s1_live). 공개 배포는 사용자 사이트로 정리 후.
