# OLMoEarth upstream 기여 후보 — 재진입 목록 (최종본, 2026-09-10)

정리일 2026-09-10. 로컬 준비물(branch·본문·초안)을 재확인하고 **같은 날 upstream·HF·PyPI 에 대조함**
(`olmoearth_projects origin/main=23a3d7b` 변동 없음, `rslearn` 최신 release v0.1.14 = 2026-08-25).
제출은 아직 하지 않았음. 연구 결과 홍보와 좁은 upstream 수정은 분리함. SSOT 는 [PR_DOSSIER.md](../PR_DOSSIER.md).

## 오늘 확인한 사실(제출 판단의 근거)

| 확인 항목 | 결과 |
|---|---|
| sample `annotation_features.geojson` (origin/main) | 여전히 `es_annotations_task_id`·`es_label` 등 legacy key, 6 feature |
| runner 가 `oe_*` 를 요구하는가 | PyPI `olmoearth_runner-0.1.12` wheel(`uv.lock` 고정 버전) `annotation_features.py` 에 `oe_annotations_task_id: UUID`, `oe_labels: dict[str,int\|float\|None]` 필수. 0.1.14 도 동일 |
| open PR / issue 중복 | open PR 4건(#37·#42·#43·#64), open issue 13건. #1·#2 와 직접 중복 없음 |
| 이슈 작성 권한 | **열려 있음.** 외부 계정이 #40~#65 를 작성. 8/26 명세서의 "제한" 은 오류 |
| LFMC HF ckpt | `model.ckpt` 1,139,505,083 B, lastModified 2025-11-03 → 측정 뒤 교체 없음. `docs/lfmc.md` 580.6 유지 |
| rslearn SCL(#3·#4) | v0.1.14 `sentinel2_scl.py` `_score_item` 이 layer `resampling_method` 를 SCL 읽기에 전달(L115), `missing scoring bands` ValueError 유지(L103/L267). PC Sentinel2 는 layer `band_sets` 교차 자산만 등록 |
| lock 스큐(#5) | origin/main `uv.lock`: rslearn 0.0.23 · olmoearth-pretrain 0.0.2 · runner 0.1.12 |
| partial-band mask(#6) | v0.1.14 `model.py` 도 `Modality.get(...).band_sets` 정적 수로 mask 생성 |
| forest_loss_driver | 내부 `olmoearth_datasets.sentinel2_l2a` 레이어 2개 유지. main 최신 머지 #39 가 Ai2 내부 배포용 |
| 로컬 branch | `fix/sample-annotation-oe-schema` HEAD 21b658a = origin/main + 1 file(+36 −24). fork(`DDanggle/olmoearth_projects`) 없음 |

## 제출 순서와 형식

| 순서 | 후보 / 대상 | 형식 | 준비 상태 | 남은 것 |
|---|---|---|---|---|
| 1 | sample annotation `es_* → oe_*` / olmoearth_projects | **PR** | branch·본문([pr_bodies/01_sample_schema.md](../pr_bodies/01_sample_schema.md), 버전 주장 wheel 로 검증) 완료. 8월 Linux 6-window 재현 | fork → push → PR. 원하면 current runner 0.1.14 로 quick-start 1회 재실행 후 제출 |
| 2 | LFMC 공개 ckpt MSE 불일치 / olmoearth_projects | **Issue** | [ISSUE_DRAFT_lfmc.md](../ISSUE_DRAFT_lfmc.md) 최종본: 절대 MSE 제목, "확인 요청" 프레임, 옛 API 문구 삭제, HF revision 기록 | `gh issue create` |
| 3 | forest_loss_driver 외부 실행 불가 + URL 미설정 무한 재시도 | **Issue** | 증상·401 확인·fail-fast 제안 기록(dossier #3·#6) | 영문 본문 작성. 소스 교체 PR 은 내부 배포(#39)와 충돌 가능해 금지 |
| 4 | SCL categorical scoring 에 nearest / rslearn | PR(소) | 결함 위치 재확인(v0.1.14 L115) | nearest 강제 최소 patch + 합성 회귀 테스트(bilinear 반사도 + SCL BestClear) |
| 5 | SCL 보조-band 의존성 선언 / rslearn | Issue/RFC | 설계 gap 재확인 | #4 와 분리해 작성 |
| 6 | release–lock–runner 호환성 / olmoearth_projects·rslearn | Issue → PR | lock 스큐·v1.2 로드 실패 기록 | runner 포함 compatibility matrix 실측 후 |
| 7 | partial-band mask 릴리스별 처리 / rslearn | 보류 | 내부 probe 만 | 공개 config 경로 최소 재현 + 정책 합의 전엔 제출 안 함 |

**백로그(제출 안 함):** macOS multiprocessing hang(#8, 재현 환경 제한), 상대 경로 심링크(#7, 비공개 레포), consumed-timestep 문서 clarification(#9), runner `requires-python<3.12` 상한(설계 결정일 수 있음).

## 제출 명령(승인 후 실행)

```bash
cd ~/dong/ai_projects/olmoearth_projects
gh repo fork allenai/olmoearth_projects --remote --remote-name fork
git push fork fix/sample-annotation-oe-schema
gh pr create --repo allenai/olmoearth_projects \
  --head DDanggle:fix/sample-annotation-oe-schema --base main \
  --title "Migrate sample annotation_features.geojson to oe_* schema" \
  --body-file _work/pr_bodies/01_sample_schema.md
# LFMC (본문은 ISSUE_DRAFT_lfmc.md 의 Body 절만)
gh issue create --repo allenai/olmoearth_projects --title "<Title 줄>" --body-file <body.md>
```

## 이전 초안에서 고친 것

- LFMC: "문서 성능의 60%" → MSE 절대값 불일치. "잘못된 업로드" 단정 → 가설 + revision 확인 요청. 말미의 "어떤 PyPI rslearn 에도 API 없음" 문구 삭제(8/26 정정 반영). HF 파일 크기·oid·수정일 추가.
- PR 본문: "runner ≥0.1.12" 를 wheel 파일 경로·필드로 구체화.
- 한국 KR-4 감사([KOREA_KR4_AUDIT_2026_09_10.md](KOREA_KR4_AUDIT_2026_09_10.md))의 결함(test 64칩 시간 창 비대칭, FULL 노출량 2배, seed·mIoU 정의)은 **우리 연구 코드 문제**로 Ai2 후보와 분리함. 코드 v2 에 `--full-bs`·창 정합 제외를 넣었고 재실행 전임.

이 문서는 준비 목록이며 제출 권한을 새로 부여하거나 자동 제출하지 않음.
