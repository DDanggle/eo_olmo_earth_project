# PR 리뷰 노트 — 제출 후 대응 기록

최종 갱신: 2026-08-26.

역할 분담: `PR_DOSSIER.md`는 **제출 전** 명세(증상·원인·수정·검증), 이 파일은 **제출 후**
리뷰 과정(리뷰어 질문, 우리 답변, 후속 커밋, 최종 결과)을 기록한다.

원칙:
- 리뷰어 질문에는 **재현 명령과 근거 파일**로 답한다. 의견이 아니라 실행 결과.
- 우리가 틀렸으면 즉시 인정하고 수정한다. 방어하지 않는다.
- 답변에 우리 저장소의 사설 문서(K-ALIGN 등)를 링크하지 않는다. 상류 레포에 필요한 최소 정보만.

---

## 제출 현황 — 아직 제출된 항목 없음

| # | 대상 | 제목 | 제출일 | URL/번호 | 상태 |
|---|---|---|---|---|---|
| — | — | — | — | — | 제출 후에만 행을 추가한다 |

이전 판에는 #1을 `제출됨`으로 잘못 적었지만 fork·push·PR URL이 없고 로컬 브랜치만 존재한다.
2026-08-26 현재 정확한 상태는 아래 pre-submit queue다. 후보 상세와 current-upstream 판정은
`PR_DOSSIER.md`가 SSOT다.

## pre-submit queue — 이 표는 리뷰 상태가 아니다

| 순서 | 후보 | 현재 상태 | 제출 전 남은 것 |
|---|---|---|---|
| 1 | sample `es_*→oe_*` | 로컬 branch `fix/sample-annotation-oe-schema`, commits `5e044ee`, `21b658a` | Linux current-runtime replay, fork/push 승인 |
| 2 | LFMC checkpoint mismatch | 영어 issue 초안 완료 | maintainer contact path 확인 |
| 3 | SCL categorical scoring | current v0.1.14에서 결함 재확인 | nearest-only 최소 patch와 synthetic test |
| 4 | SCL auxiliary dependency | 설계 gap 재확인 | 별도 issue/RFC; #3과 한 PR 금지 |
| 5 | lockfile/v1.2 compatibility | current lock skew 재확인 | runner 포함 compatibility matrix |
| 6 | partial-band release mask | internal probe만 완료 | public API end-to-end repro + 기대 정책 |

제출 큐에서 내린 것: direct-materialize NotImplementedError는 current upstream에서 해소됨,
`미출시 API`는 lock skew에 병합, embedding 12→4는 documentation clarification으로 강등.

## 예정 PR #1 (sample 스키마) — 예상 질문과 답변 준비

| 리뷰어가 물을 만한 것 | 준비된 답 |
|---|---|
| 왜 `oe_labels: {category: N}` 형태인가? | 샘플의 `olmoearth_run.yaml`이 `label_property: "category"`를 선언하고, runner의 `AnnotationFeatureProperties.oe_labels`가 `dict[str, int\|float\|None]`을 요구한다 |
| 어느 runner 버전에서 깨지는가? | 0.1.12/0.1.14 양쪽에서 재현. 0.1.12는 macOS, 0.1.14는 Linux(py3.11)에서 확인 |
| 검증했는가? | Linux에서 `prepare_labeled_windows` 완주 + 6 windows. macOS는 schema gate 후 별도 forkserver hang이 재현돼 full verification으로 세지 않음 |
| task_features는 왜 안 고쳤나? | 이미 `oe_*` 스키마다. 짝 파일 중 하나만 마이그레이션돼 있었던 것이 이 버그의 원인 |
| upstream에서 이미 고쳐졌나? | 2026-08-26 `origin/main=23a3d7b`에도 legacy key가 남고 open PR 4건과 직접 중복 없음 |
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
