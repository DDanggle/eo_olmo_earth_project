# OLMoEarth 연구 재시작 지점

갱신: 2026-09-04
활성 과학 기준점: **MS-96/97 + MS-98/99 + Earth Embedding Continuity A/B/C 설계**
로컬 HEAD: `a346eab` — Clay v0 실행기까지. 아래 A/B/C 문서·config는 아직 미커밋 DRAFT다.

이 파일은 새 세션의 첫 진입점이다. Nepal 대응 데모는 현재 CVPR/transfer 임계경로가 아니며,
전용 코드·문서·원본·중간 산출물은 sibling 저장소
`/Users/dgyi/dong/ai_projects/nepal-live-twin`이 소유한다.

## 한 문장 연구 질문

> **세계·과업·모델 release가 바뀔 때, 저장된 Earth embedding과 downstream head를 그대로 쓰고,
> head만 적응하고, 표현을 migration하고, 재임베딩하거나 라벨을 더 요청할 시점을 정확도와
> label·GPU·raw-I/O·cache invalidation 비용으로 결정할 수 있는가?**

세 축을 별도 논문으로 벌리지 않는다.

- **A — release migration**: OlmoEarth v1→v1.2에서 old cache/index/head를 살리는가.
- **B — product validity**: Clay·AlphaEarth 등 다른 embedding product에서도 같은 결정 문제가
  성립하는가.
- **C — safe action**: support label과 contract만 보고 A0 reuse/A1 head-adapt/A3 re-embed/
  REQUEST를 고르는가.

설계 SSOT는 `docs/ABC_EMBEDDING_CONTINUITY_2026_09_04.md`다.

## 지금까지 닫힌 양성 결과

### Task 1 — Sen12Landslides, 8개 geographic holdout

- source-only frozen OlmoEarth cache + decoder(P4): region-macro `.2722`.
- raw UNet3D(P2) `.1966`, raw U-TAE(P3) `.1834`; 최고 raw 대비 7/8 지역 우위.
- target tile K=5/20에서 A1 cache-head adaptation은 raw full A4w를 8/8, parameter-matched
  A4h를 방향 기준 8/8 이겼다(MS-96/97). fixed-exposure에서도 A1>A4w 8/8이다.
- 그러나 K=5에서 A1>A0는 5/8뿐이다. `라벨 5장이면 항상 적응`은 금지한다.

### Task 2 — Solar Farm, 8개 UTM-zone fold group

- A0 cache no-adapt `.591`, stratified A1 K=5 `.582`, K=20 `.609`.
- raw A4w K=5/20 `.240/.245`, A4h `.257/.291`; cache pathway가 raw adaptation을 8/8
  이겼다(MS-98/99).
- random K=5에서 support 12/24가 positive tile 0장이었고 A1이 `.426`으로 붕괴했다.
- tie-correct AP는 A0 `.9252`가 A1 K=5/20 `.8651/.9044`보다 높다. 따라서 action은 task뿐
  아니라 support 구성과 배포 utility/threshold 계약에 의존한다.
- Solar fold는 독립 지역 8개가 아니라 UTM-zone 기반 group이다. Sen12의 `8지역`과 같은
  일반화 단위라고 부르지 않는다.

### Model shift

- M1: 같은 scene의 OlmoEarth v1/v1.2 token identity R@1은 양방향 0. Procrustes·affine ridge도
  등록된 retrieval compatibility gate를 실패했다. pooled CKA `.979`는 task continuity 증거가 아니다.
- M85는 v1/v1.2 radar/optical 성능 비교이지 cache migration 실험이 아니다.

## 닫힌 음성 방향 — 이름을 바꿔 되살리지 않는다

- CacheTune A2 low-rank spatial residual: MS-94에서 A1보다 `.05–.08` 낮아 stop rule 발동.
- prediction fusion·GeoContextGate: MS-90B/91/92 종료. FP-matched oracle headroom도 거의 0.
- label-free winner router와 block routing: 등록 gate 실패.
- MoE: action complementarity와 untouched selector 성공 전에는 열지 않는다.

## 2026-09-04 Clay v0 실행 경계

서버에서 확인된 사실:

- `clay_cache`: 6,834/6,834, 실패 0, Clay v1.5, 1024-d.
- 그러나 native `16×16`을 bilinear로 `32×32`에 확대한 cache다. audit의
  `all_gates_pass=true`는 파일 완결성만 보증하며 비교 공정성을 보증하지 않는다.
- source decoder chain과 few-shot wait chain이 실행 중이었다. 확증 실행 중 서버 코드 push 금지
  규칙에 따라 이 검토에서는 서버 코드를 건드리지 않았다.
- 이 v0 결과는 **interpolated deployment-adapter exploratory baseline**으로만 보존한다.
  B1 common-physical 또는 B2 native confirmatory 결과로 쓰지 않는다.
- current Clay few-shot FP-matched IoU는 Clay A0의 FP budget을 쓰고, 비교하려는 historical raw
  A4는 OlmoEarth A0 budget을 썼다. 서로 다른 작동점이므로 report 간 primary IoU 비교는 금지한다.
  threshold-free tie-correct AP만 탐색 비교할 수 있다.

## 아직 주장할 수 없는 것

- A1이 A0보다 항상 낫다.
- Clay 하나로 product-agnostic 원리가 증명됐다.
- v1→v1.2 bridge가 old task head/index를 보존한다. 아직 downstream migration은 0회다.
- AlphaEarth 연간 embedding으로 사건 전후/실시간 변화를 측정한다.
- support label 없이 self-training하면 정확도가 개선된다.
- Korea sealed target 또는 독립 Task-3에서 safe action policy가 통과했다.
- OLMoEarth가 모든 GeoFM보다 보편적으로 우월하다.

## 다음 실행 순서

1. **P0 증거 복구(CPU)**: Solar random support ID/양성 수/SHA, WGS84 cross-CRS distance,
   exact-query A4w0, 서버 48-run 원시 report/snapshot을 로컬 봉인한다.
2. **A release migration**: full-12-band Solar의 exposed 2fold에서 v1/v1.2 exact-scene bridge가
   old-head AP·fixed-threshold IoU를 보존하는지 screen한다. 새 query raw read는 정상 비용이고,
   피하는 것은 과거 archive 전체의 raw backfill이다.
3. **B-v1 Clay**: v0 종료 뒤 native 16×16 smoke를 새로 봉인한다. B1은 80 m·16×16·256-d
   compact cache, B2는 native product로 분리하고 같은 report/threshold에서 raw를 재평가한다.
4. **C A3 ceiling + deterministic guardrail**: exposed 4 unit, K=20에서 공식
   LayerDecayAdamW와 q/v LoRA sensitivity를 측정한다. positive 0이면 A0/REQUEST, 그 외에도
   leave-one-tile-out 하한이 0보다 클 때만 A1을 허용한다.
5. **untouched first-look**: policy와 threshold를 동결한 뒤 독립 Task-3 또는 Korea sealed target을
   한 번만 연다.

AlphaEarth는 Solar/static mapping B2 뒤에만 둔다. 연간 64-d product이므로 event Sen12와 같은
시간 gate에 넣지 않는다. Prithvi는 Clay 뒤 contract-shift sensitivity다.

## 읽는 순서

1. 이 파일
2. `docs/ABC_EMBEDDING_CONTINUITY_2026_09_04.md`
3. `docs/ASSET_INVENTORY.md`
4. `docs/CRITICAL_PATH.md`
5. `MEASURED_FINDINGS.md`
6. `docs/PAPER_NARRATIVE_2026_08_31.md` — 이전 narrative, 현재 A/B/C 문서가 실행 방향을 대체
7. `GOAL.md` 마지막 Worklog

기계 판독 DRAFT:

- `config/release_migration_prereg_draft_v0.json`
- `config/second_fm_cache_prereg_v1_draft.json`
- `config/safe_cache_action_prereg_draft_v0.json`

`DRAFT`는 실험 전 커밋되어야 active preregistration이 된다. 결과가 나온 뒤 문구·gate를 고쳐
commit하면 사전등록이 아니다.

공식 Ai2 checkout `..`에는 사용자 수정이 남아 있다. 연구 재시작 작업에서
`olmoearth_run_data/forest_loss_driver/{dataset.json,model.yaml}`과 `.pnpm-store/`를 건드리지 않는다.
