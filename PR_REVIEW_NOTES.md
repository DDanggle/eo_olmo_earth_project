# PR 리뷰 노트 — 제출 후 대응 기록

최종 갱신: 2026-09-11 (upstream 재대조·우선순위 교정, 제출은 아직 없음).

현재 실행 권고는 **sample → SCL → LFMC**다.
[9/11 우선순위·검증 조건](docs/PR_PRIORITIES_AND_EO_GAPS_2026_09_11.md)을 따른다.
아래 번호는 리뷰 티켓이나 GitHub PR 번호가 아니라 준비 큐다.

역할 분담: `PR_DOSSIER.md`는 **제출 전** 명세(증상·원인·수정·검증), 이 파일은 **제출 후**
리뷰 과정(리뷰어 질문, 우리 답변, 후속 커밋, 최종 결과)을 기록한다.

원칙:
- 리뷰어 질문에는 **재현 명령과 근거 파일**로 답한다. 의견이 아니라 실행 결과.
- 우리가 틀렸으면 즉시 인정하고 수정한다. 방어하지 않는다.
- 답변에 우리 저장소의 사설 문서(K-ALIGN 등)를 링크하지 않는다. 상류 레포에 필요한 최소 정보만.

---

## 제출 현황 (2026-09-16 PR 1건·이슈 1건 제출, DDanggle 계정)

| # | 대상 | 제목 | 제출일 | URL/번호 | 상태 |
|---|---|---|---|---|---|
| 1 | olmoearth_projects | Migrate sample annotation_features.geojson to oe_* schema | 2026-09-16 | https://github.com/allenai/olmoearth_projects/pull/68 | open, 리뷰 대기 |
| 2 | olmoearth_projects | LFMC released checkpoint gives test MSE 951.9, docs say 580.6 | 2026-09-16 | https://github.com/allenai/olmoearth_projects/issues/69 | open, 답변 대기 |

이전 판에는 #1을 `제출됨`으로 잘못 적었지만 fork·push·PR URL이 없고 로컬 브랜치만 존재한다.
2026-08-26 현재 정확한 상태는 아래 pre-submit queue다. 후보 상세와 current-upstream 판정은
`PR_DOSSIER.md`가 SSOT다.

## pre-submit queue — 이 표는 리뷰 상태가 아니다

| 순서 | 후보 | 현재 상태 | 제출 전 남은 것 |
|---|---|---|---|
| 1 | sample `es_*→oe_*` | 로컬 branch `fix/sample-annotation-oe-schema`, commits `5e044ee`, `21b658a`; 9/10 upstream 여전히 legacy, runner 0.1.12 wheel 로 요구 필드 확인 | fork/push 승인(선택: runner 0.1.14 quick-start 재실행) |
| 2 | SCL categorical scoring | v0.1.14 FirstValid·BestClear 경로 재확인(9/11) | nearest-only 최소 patch와 실제 재격자 synthetic test |
| 3 | LFMC checkpoint mismatch | 영어 issue 교정본(9/11), 원인 단정 제거·파일 날짜 수정 | config·로그·과거 평가 파일 hash 묶음, 현재 계정 권한 확인 |
| 4 | SCL auxiliary dependency | 설계 gap 재확인 | 별도 issue/RFC; #3과 한 PR 금지 |
| 5 | lockfile/v1.2 compatibility | current lock skew 재확인 | runner 포함 compatibility matrix |
| 6 | partial-band release mask | internal probe만 완료 | public API end-to-end repro + 기대 정책 |

제출 큐에서 내린 것: direct-materialize NotImplementedError는 current upstream에서 해소됨,
`미출시 API`는 lock skew에 병합, embedding 12→4는 documentation clarification으로 강등.

## 예정 PR #1 (sample 스키마) — 예상 질문과 답변 준비

| 리뷰어가 물을 만한 것 | 준비된 답 |
|---|---|
| 왜 `oe_labels: {category: N}` 형태인가? | 샘플의 `olmoearth_run.yaml`이 `label_property: "category"`를 선언하고, runner의 `AnnotationFeatureProperties.oe_labels`가 `dict[str, int\|float\|None]`을 요구한다 |
| 어느 runner 버전에서 깨지는가? | 0.1.12/0.1.14의 요구 필드는 wheel에서 확인. 과거 macOS schema 실패와 Linux 0.1.14 E2E를 구분하고, 오늘 재실행으로 부르지 않음 |
| 검증했는가? | Linux에서 `prepare_labeled_windows` 완주 + 6 windows. macOS는 schema gate 후 별도 forkserver hang이 재현돼 full verification으로 세지 않음 |
| task_features는 왜 안 고쳤나? | 이미 `oe_*` 스키마다. 짝 파일 중 하나만 마이그레이션돼 있었던 것이 이 버그의 원인 |
| upstream에서 이미 고쳐졌나? | 2026-09-10 `origin/main=23a3d7b`(변동 없음)에도 legacy key 6 feature 유지. open PR 4건·open issue 13건과 직접 중복 없음 |
| 왜 큰 regression test를 추가하지 않나? | 첫 PR은 data-only schema repair로 최소화한다. 최신 runner quick-start를 Tests에 적고 maintainer가 원하면 schema test를 후속 커밋 |

## 리뷰 로그

(리뷰어 코멘트가 오면 날짜·요지·우리 대응·후속 커밋을 여기에 시간순으로 append)

```text
YYYY-MM-DD | reviewer | 요지 | 우리 대응 | 커밋/링크
```

## 상류에 보내지 말 것

- 사설 전략 문서(K_ALIGN_*, RESEARCH_*, MEASURED_FINDINGS 등) 링크
- 미공개 측정치(M1~M5의 sealed test 수치)
- 서버 경로·자격증명 변수명
