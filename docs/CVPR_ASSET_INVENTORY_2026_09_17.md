# CVPR 준비 자산 목록 — 2026-09-17 재집중

작성: 2026-09-17. 목적: PR·Ai2 지원·Nepal 사이드카를 걷어내고, CVPR 논문용으로 이미 만든 것만 추림.
새 실험·GPU·서버 조회 없음. 장부(`MEASURED_FINDINGS.md`)와 docs 재독 기반. 수치는 장부 인용.

## 0. 한 줄 현재 위치

- 논문 질문이 세 번 바뀜: (1) reuse-or-retrain 측정 → (2) EarthCache selector/벤치마크 → (3) 갱신형 캐시(streaming update).
- 가장 최근 강한 결과는 (3) MS-116/117. (1)(2)는 empirical base로 살아 있음.
- 9/13 이후 논문 작업 중단 상태. 마지막 논문 문서는 `docs/PAPER_STATE_2026_09_06.md`, `docs/STREAMING_RESEARCH_UPDATE_2026_09_09.md`.

## 1. 살아 있는 측정 결과 (논문 표에 쓸 수 있음)

| # | 결과 | 수치 | 상태 | 장부 |
|---|---|---|---|---|
| R1 | frozen OlmoEarth cache + small decoder > raw UNet3D, 8지역 LOCO 산사태 | region-macro .272 vs .197, 6/8 승 | 강 | M65 |
| R2 | 두 번째 과업(Solar) 확증: OlmoEarth cache > Galileo cache | 8/8 | 강 | MS-109 |
| R3 | few-shot: cache-head 적응 > raw 적응 (K=5/20) | 두 과업 8/8; A1>A0는 K=5에서 5/8 | 강 | MS-113 계열 |
| R4 | family 경계: Clay 동률·Galileo 미달·Prithvi 붕괴 | Clay .195, Galileo .168(groupcat), Prithvi .025 | 중 (seed 1) | MS-102/105 |
| R5 | scale 축: nano/tiny/base로 cache/raw 경계 이동 | .194/.228/.272 | 중 (seed 1) | MS-108 |
| R6 | release 전환 시 cache identity 붕괴, 선형 bridge로 AP 97% 복구하나 fixed-decision gate 실패 | v1→v1.1/v1.2 R@1=0 | 강 | MS-100 계열 |
| R7 | 갱신형 캐시(GRU updater) 4지역 개발 | teacher downstream 77~100%+ 회복, 관측0 ≈0 | 개발 | MS-116 |
| R8 | KuroSiwo 외부 재현(S1 홍수, 10사건) | 회복 107~114%, 판독기3×갱신기3 gate 통과 | 조건부 (NaN 제외, 사건별 7/3) | MS-117 |
| R9 | 갱신형 캐시 실측 비용 | 벽시계 2.3배·raw 읽기 4.5배 유리(공정 배치 시) | 범위 한정 | MS-116-비용 |
| R10 | Presto C1a matched control 8/8 패배 | .109 | 강 | MS-87 |

## 2. 음성/경계 결과 (논문에 경계 조건으로 씀)

| # | 결과 | 장부 |
|---|---|---|
| N1 | 토큰 해상도 지렛대(patch-2) 기각, 두 날짜 조건 모두 | MS-115 |
| N2 | 교차 센서 갱신(S1→S2 cache) 실패, 사영기로도 물리적 한계 | MS-118/121 |
| N3 | 이탈리아 봉인 지역 .089, 등록 범위 하회; 해상도 분해도 동일 | MS-119/120 |
| N4 | label-free 프로브는 예측기가 아니라 family 판별기 | MS-110 |
| N5 | selector G0 개발 데이터에서 불통과; MS-112 "승자 3종" 철회 | MS-111/112 |
| N6 | 갱신기 아키텍처 변형(Δt·공간게이트·어텐션) 등록 기준 미달 | MS-116-arch |
| N7 | T0 시간 차분 스케치 무효 | MS-115-T0 |
| N8 | R-event 검색 gate 실패, 2번째 shared-cache task 주장 불가 | M17-2 |

## 3. 논문 서사·감사 문서

| 문서 | 역할 |
|---|---|
| `docs/PAPER_NARRATIVE_2026_08_31.md` | 서사 SSOT(Reuse or Retrain 시점), 금지 표현 6개 |
| `docs/CVPR_REVIEW_2026_08_31.md` | 살아 있는 결과/깨진 것/미실행 복기 |
| `docs/CVPR_BIG_PICTURE_AUDIT_2026_09_05.md` | 약점 8개 선언, 요약↔원시 6/6 대조 |
| `docs/PAPER_STATE_2026_09_06.md` | EarthCacheBench 후퇴선 + CVPR main 승격 조건(Task-3 regret+비용) |
| `docs/MS112_CVPR_AND_KOREA_AUDIT_2026_09_06.md` | 제목안 "EarthCache: Reuse/Adapt/Recompute", 경쟁 문헌표 |
| `docs/NOVELTY_ARCHITECTURE_OPTIONS_2026_09_07.md` | 선택 연구→재사용 정보 학습 연구 전환 제안 |
| `docs/BIG_PICTURE_STREAMING_EARTH_2026_09_08.md` | streaming 큰 그림, 자산 역할표 |
| `docs/STREAMING_RESEARCH_UPDATE_2026_09_09.md` | 최신 수치판(MS-117~120) |
| `docs/RELATED_WORK_STREAMING_2026_09_09.md` | streaming 관련연구 |
| `docs/POSTTRAINING_JEPA_UPDATE_2026_09_09.md` | 다음 학습 설계(encoder frozen/LoRA × forecast loss) |
| `docs/PAPER_CLAIM_EXPANSION_2026_08_26.md`, `docs/RECENT_LITERATURE_DECISION_2026_08_26.md` | claim 확장·문헌 결정 |
| `PAPER_READING_LIST.md`(60KB), `PAPER_NOTES_v1.md` | 문헌 |
| `EMBEDDING_TRANSFER_CVPR_TRACKS.md`, `K_ALIGN_CVPR_READINESS_AUDIT.md` | 8/24 초기 트랙 정의(역사) |
| `docs/talk/olmoearth_session_2026_09.html`, `build_deck.py` | 발표 덱 |

## 4. 사전등록 config (config/, 실행 여부)

- 실행됨: `geobench_cache_action_prereg_v0/v1.json`, `streaming_updater_arch_prereg_v0.json`, `cross_sensor_update_prereg_v0.json`, `cross_sensor_projector_prereg_v0.json`, `italy_sealed_region_prereg_v0.json`, `italy_resolution_decomposition_prereg_v0.json`, `task2_extension_prereg_v0.json`
- 초안(launch_allowed=false): `kurosiwo_streaming_prereg_v0_draft.json`, `streaming_task_preservation_v1_draft.json`, `observation_posttraining_v0_draft.json`, `second_fm_cache_prereg_v1_draft.json`, `label_budget_curve_draft_v1.json`, `release_migration_prereg_draft_v0.json`, `safe_cache_action_prereg_draft_v0.json`, `t1_evidence_and_fidelity_review_v1_draft.json`

## 5. 그림 후보 (artifacts/figures)

- `cache_utility_boundary_ms108.png` — family×scale 경계 (논문 Fig 후보)
- `region_bars_landslide_solar.png`, `before_after_landslide_solar.png` — R1/R2
- `bv1_diagnostics.png` — family 진단
- `fig_streaming_material_hiroshima.npz`, `fig_kurosiwo_material.npz` — R7/R8 그림 재료(미렌더)
- jeju_* — Korea 데모, 논문 성능 표 아님

## 6. 코드 자산 (code/, 324 파일 중 논문 직결)

- 캐시 추출: `extract_sen12_fold_cache.py`, `build_olmo_pool16.py`, second-FM 추출 계열
- 학습/평가: `cache_decoder_train.py`, `streaming_update_train.py`, `t1_cost_measure.py`
- selector/행렬: `geobench_action_headroom.py`, `build_action_matrix.py`, `build_g0_dev_input_v3.py`
- 감사: `audit_summary_vs_raw.py`, `audit_streaming_t1_reports.py`, `audit_release_gate_contract.py`, `bootstrap_spatial_block.py`
- 체인: `bv1_chain.sh`, `arch_axes_chain.sh`

## 7. 논문에 세지 않는 것 (분리 유지)

- Korea AI-Hub 3-task: 파일 무결성 통과, 성능 기여 0. 외부 사례로만.
- Nepal Live Twin: 별도 저장소, 응용 자산.
- PR/sample schema/LFMC issue: Ai2 기여, 논문 무관.
- Ai2 지원 문서(`AI2_ONE_PAGER`, `Ai2_Meeting_Cards`): 논문 무관.

## 8. 미결 (PAPER_STATE 9/6 + 9/9 기준)

1. 공개 untouched Task-3 부재 → selector 트랙은 두 과업뿐.
2. R8 clean 확증: NaN 제외 전 decoder 선택, FAR 분모 수정.
3. R4/R5 seed 1개.
4. 실측 비용 곡선: Solar wall-clock 오염, FLOPs 기준만 유효.
5. 한 updater로 다중 과업 갱신 결과 없음.
6. release-gate 등록 지표(`iou_frozen_thr`) 재집계.
