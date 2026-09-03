# OLMoEarth 연구 재시작 지점

갱신: 2026-09-02
활성 과학 기준점: **MS-93 (C1b 24/24 완료, prediction fusion 종료) + CacheTune PT-0 설계**

이 파일은 새 세션의 첫 진입점이다. Nepal 대응 데모와 그 후속 실험은 삭제하지
않았지만 현재 CVPR/transfer 임계경로가 아니다. Nepal 전용 코드·문서·원본·중간 산출물은
독립 저장소 `/Users/dgyi/dong/ai_projects/nepal-live-twin`이 소유한다.

## 지금까지 과학적으로 닫힌 것

- Sen12Landslides S12q 계약: 10개 task-eligible 지역, 6,834 표본, 8개 held-out 지역 확증.
- 주지표 positive-tile macro IoU의 region-macro:
  - P4 frozen OLMoEarth v1 + small decoder: **0.272166**
  - P2 UNet3D: **0.196558**
  - P3 U-TAE: **0.183436**
  - P4−P2: **+0.075608**
- 사전등록 지역 승리 6/8, strong win 5/8. Indonesia에서는 P4가 패배했고 Itogon은
  all-seed 승리 규칙을 통과하지 못했다.
- 이 결과가 지지하는 것은 **frozen OLMoEarth 표현의 지역 전이 viability**다.
- MS-87 retrospective control: 같은 S12q·split·seed·decoder 경로에서 Presto C1a `.1092`,
  P4 `.2722`, P2 `.1966`; P4와 P2가 C1a를 8/8 지역에서 이겼다. 이것은 효과가 아무 frozen
  GeoFM에나 생기지 않음을 보이지만, off-domain Presto 하나로 OLMo universal superiority를
  주장하지 않는다.
- MS-93 C1b: Presto native grid는 `.1261`로 pooled C1a `.1092`보다 `+.0169` 올랐지만 P4와
  P2에 8/8 패배했다. pooling이 Presto의 낮은 순위를 설명하지 못한다. 24개 원시 JSON과 실행
  snapshot은 `artifacts/c1b_presto_native_compact_v1.json`으로 로컬 봉인했다.
- MS-90B/91/92: 등록 naive fusion과 GeoContextGate v1/v2가 모두 승급 gate를 실패했다. FP
  작동점을 맞추면 label-peeking oracle도 P4 대비 `+.008/-.004`여서 상보성 자체가 거의 없었다.
  등록 stop rule에 따라 v3는 만들지 않고 융합은 negative/analysis 절로 닫는다.

## 아직 주장할 수 없는 것

- OLMoEarth가 다른 GeoFM보다 보편적으로 우월하다는 주장: common/native Presto 한 모델의
  retrospective/off-domain 결과만 있고 미열람 외부 비교가 남았다.
- label-free router가 승자를 고른다는 주장: 기존 task/block routing과 winner prediction gate는
  실패했거나 기각됐다.
- M65를 실시간 재난 탐지·물리 위험 예측으로 확장하는 주장.
- Nepal 단일 사건 결과를 CVPR 본선의 독립 transfer로 세는 것.
- low-rank cache adapter가 head-only 또는 encoder PEFT보다 낫다는 주장: 2026-09-02 현재
  preregistration draft만 있고 학습 실행은 0회다.
- MoE가 필요하다는 주장: adapter별 oracle complementarity를 아직 측정하지 않았다.

## 새 method 후보 — CacheTune

> frozen OLMoEarth cache를 버리지 않고 target few-shot label로 low-rank spatial residual만
> post-train하고, head-only / encoder APLA·LoRA / raw retraining과 label–compute–cache-invalidation
> frontier를 비교한다.

중심 설계는 `docs/CACHE_COMPATIBLE_POSTTRAINING_2026_09_02.md`, 실행 전 기계 판정 계약은
`config/cachetune_pt0_preregistration_v0.json`을 따른다. generic LoRA나 MoE 자체를 novelty로
세지 않는다. 현재 P4 decoder의 첫 `1×1 768→128` 병목과 adapter가 섞이지 않도록 primary A2는
source decoder를 동결한다. P2/P4 prediction fusion은 MS-90B/91/92에서 닫혔으며 다시 열지 않는다.

## 다음 실행 순서

0. **완료 — MS-86 메커니즘 감사**: P4 empty-FP 7/8, 중앙 P2/P4 5.02×; tile-oracle
   headroom +.02375, 5/8 지역 ≥.02. 두 screen은 통과했지만 새 확증/라우터 결과가 아니다.
1. **완료 — C1b/MS-90B/MS-91/MS-92**: second-GeoFM native sensitivity를 닫았고 fusion method는
   사전 stop rule로 종료했다. 이 분기를 다시 열지 않는다.
2. **Raw recipe audit**: P2/P3가 공식 구조에 가까워도 현재 40-epoch BCE recipe는 공개
   75-epoch BCEDice benchmark와 다르다. source-only validation으로 current vs official-like recipe를
   비교해 supervised-baseline 공격을 닫는다. CacheTune A4 비교군에도 필요하다.
3. **CacheTune PT-0 (CPU)**: target support/query를 event/component+spatial buffer로 나누고,
   별도 adaptation runner의 checkpoint-init/freeze/subset/normalization invariants를 봉인한다.
4. **CacheTune PT-1 (GPU1, 36-run development screen)**: exposed 2지역 × K={5,20} ×
   A1 head-only/A2 strict/A2 non-spatial × 3 반복. A2가 사전 gate를 못 넘으면 method·MoE를 중단한다.
5. **Source-label manifest + curve**: 봉인된 1/5/10% ID는 보존한다. 전체는 432 new runs이며
   144회만으로 결론내지 않는다. 이 곡선은 target label 0인 baseline이고 CacheTune target few-shot과
   섞지 않는다. raw audit와 PT-1 결과 뒤 실행 순서를 최종 동결한다.
6. **Encoder PEFT ceiling**: PT-1 통과 시에만 rslearn APLA/LoRA를 열고 raw I/O·재임베딩 비용을
   포함해 A2와 비교한다.
7. **Korea preflight**: M10 spatial split seal을 유지하고, AI-Hub S2 v1의 0-fill 오염을 재사용하지
   않는 canonical input/mosaic와 label ontology/time/provenance 계약을 닫는다.
8. **Korea sealed external transfer**: Sen12에서 CacheTune recipe까지 고정한 뒤 spatial support로
   적응하고 sealed query를 한 번만 연다. 같은 cache의 land-cover/deforestation/landslide 3-task
   amortization도 측정한다. paired re-annotation 없이는 `annotation effect`라고 부르지 않는다.
9. **Upstream evidence**: 공식 Ai2 저장소의 sample annotation schema PR 후보를 별도 checkout에서
   검증·정리한다.

새 방법·새 지역·새 앱을 추가하기 전에 위 2–3이 막히는 이유를 먼저 기록한다. Nepal 작업을
재개하려면 sibling 저장소에서 별도 세션으로 수행하고 이 GPU queue를 선점하지 않는다.

## 읽는 순서

1. 이 파일
2. `docs/CRITICAL_PATH.md`
3. `docs/ASSET_INVENTORY.md`
4. `docs/EXPERIMENT_C_SECOND_GEOFM.md`
5. `RESEARCH_EXECUTION_PLAN.md`
6. M65 근거: `artifacts/confirmatory_8region_summary.json`, `MEASURED_FINDINGS.md`
7. 논문 주장 SSOT: `docs/PAPER_NARRATIVE_2026_08_31.md`
8. 쉬운 연구 주제 정렬: `docs/RESEARCH_TOPIC_ALIGNMENT_2026_09_01.md`
9. MS-87/NP-88 교수 감사: `docs/PROFESSOR_AUDIT_M87_NP88_2026_09_01.md`
10. method 확장 SSOT: `docs/CACHE_COMPATIBLE_POSTTRAINING_2026_09_02.md`

공식 Ai2 checkout `..`에는 사용자 수정이 남아 있다. 연구 재시작 작업에서
`olmoearth_run_data/forest_loss_driver/{dataset.json,model.yaml}`과 `.pnpm-store/`를 건드리지 않는다.
