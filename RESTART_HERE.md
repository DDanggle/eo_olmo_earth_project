# OLMoEarth 연구 재시작 지점

갱신: 2026-08-31
활성 과학 기준점: **M65 / commit `4862483` (`Close 8-region transfer and park Nepal sidecar`)**

이 파일은 새 세션의 첫 진입점이다. Nepal 대응 데모와 그 후속 실험 M66–M85는 삭제하지
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

## 아직 주장할 수 없는 것

- OLMoEarth가 다른 GeoFM보다 우월하다는 주장: matched second-GeoFM full comparison이 없다.
- label-free router가 승자를 고른다는 주장: 기존 task/block routing과 winner prediction gate는
  실패했거나 기각됐다.
- M65를 실시간 재난 탐지·물리 위험 예측으로 확장하는 주장.
- Nepal 단일 사건 결과를 CVPR 본선의 독립 transfer로 세는 것.

## 다음 실행 순서

0. **완료 — M86 메커니즘 감사**: P4 empty-FP 7/8, 중앙 P2/P4 5.02×; tile-oracle
   headroom +.02375, 5/8 지역 ≥.02. 두 screen은 통과했지만 새 확증/라우터 결과가 아니다.
1. **Presto matched control**: 16/64/256-pixel cache smoke → 6,834 sample cache seal →
   common-grid와 native-grid readout을 같은 IDs·3 seed로 비교한다.
2. **Label-budget curve**: C1 뒤 nested region/class subset, 최소 3 subset seed로 1/5/10/100%를
   잰다. 단일 subset seed로 crossover를 주장하지 않는다.
3. **Korea preflight**: M10 spatial split seal을 유지하고, AI-Hub S2 v1의 0-fill 오염을 재사용하지
   않는 canonical input/mosaic와 label ontology/time/provenance 계약을 닫는다.
4. **Korea sealed external transfer**: P4/P2/P3/C1을 모두 등록한 뒤 test를 한 번만 연다.
   paired re-annotation 없이는 `annotation effect` 또는 `annotation-matched`라고 부르지 않는다.
5. **Upstream evidence**: 공식 Ai2 저장소의 sample annotation schema PR 후보를 별도 checkout에서
   검증·정리한다.

새 방법·새 지역·새 앱을 추가하기 전에 위 1–3이 막히는 이유를 먼저 기록한다. Nepal 작업을
재개하려면 sibling 저장소에서 별도 세션으로 수행하고 이 GPU queue를 선점하지 않는다.

## 읽는 순서

1. 이 파일
2. `docs/CRITICAL_PATH.md`
3. `docs/ASSET_INVENTORY.md`
4. `docs/EXPERIMENT_C_SECOND_GEOFM.md`
5. `RESEARCH_EXECUTION_PLAN.md`
6. M65 근거: `artifacts/confirmatory_8region_summary.json`, `MEASURED_FINDINGS.md`
7. 논문 주장 SSOT: `docs/PAPER_NARRATIVE_2026_08_31.md`

공식 Ai2 checkout `..`에는 사용자 수정이 남아 있다. 연구 재시작 작업에서
`olmoearth_run_data/forest_loss_driver/{dataset.json,model.yaml}`과 `.pnpm-store/`를 건드리지 않는다.
