# PR 리뷰 노트 — 제출 후 대응 기록

역할 분담: `PR_DOSSIER.md`는 **제출 전** 명세(증상·원인·수정·검증), 이 파일은 **제출 후**
리뷰 과정(리뷰어 질문, 우리 답변, 후속 커밋, 최종 결과)을 기록한다.

원칙:
- 리뷰어 질문에는 **재현 명령과 근거 파일**로 답한다. 의견이 아니라 실행 결과.
- 우리가 틀렸으면 즉시 인정하고 수정한다. 방어하지 않는다.
- 답변에 우리 저장소의 사설 문서(K-ALIGN 등)를 링크하지 않는다. 상류 레포에 필요한 최소 정보만.

---

## 제출 현황

| # | 대상 | 제목 | 제출일 | URL/번호 | 상태 |
|---|---|---|---|---|---|
| 1 | olmoearth_projects | Migrate sample annotation_features.geojson to oe_* schema | (기입) | (기입) | 제출됨 |
| 2 | (경로 확인 필요) | LFMC 공개 체크포인트 재현성 리포트 | 미제출 | — | 초안 `ISSUE_DRAFT_lfmc.md` |
| 4 | rslearn | ingest:false + PC Sentinel2 + CONTAINS → NotImplementedError | 미제출 | — | 재현 명확 |
| 10 | rslearn | `space_mode` 의미 모호 (MOSAIC ≡ PER_PERIOD_MOSAIC for sub-tile windows) | 미제출 | — | md5 증거 보유 |
| 11 | rslearn | 임베딩 가이드 예제가 12개 모자이크 중 4개만 사용 | 미제출 | — | Jaccard 0.091 증거 보유 |

## PR #1 (sample 스키마) — 예상 질문과 답변 준비

| 리뷰어가 물을 만한 것 | 준비된 답 |
|---|---|
| 왜 `oe_labels: {category: N}` 형태인가? | 샘플의 `olmoearth_run.yaml`이 `label_property: "category"`를 선언하고, runner의 `AnnotationFeatureProperties.oe_labels`가 `dict[str, int\|float\|None]`을 요구한다 |
| 어느 runner 버전에서 깨지는가? | 0.1.12/0.1.14 양쪽에서 재현. 0.1.12는 macOS, 0.1.14는 Linux(py3.11)에서 확인 |
| 검증했는가? | `prepare_labeled_windows` 완주 + 윈도우 6개 생성 (EPSG:32611, 10m, train/val). 서버·맥 양쪽 |
| task_features는 왜 안 고쳤나? | 이미 `oe_*` 스키마다. 짝 파일 중 하나만 마이그레이션돼 있었던 것이 이 버그의 원인 |

## 리뷰 로그

(리뷰어 코멘트가 오면 날짜·요지·우리 대응·후속 커밋을 여기에 시간순으로 append)

```text
YYYY-MM-DD | reviewer | 요지 | 우리 대응 | 커밋/링크
```

## 상류에 보내지 말 것

- 사설 전략 문서(K_ALIGN_*, RESEARCH_*, MEASURED_FINDINGS 등) 링크
- 미공개 측정치(M1~M5의 sealed test 수치)
- 서버 경로·자격증명 변수명
