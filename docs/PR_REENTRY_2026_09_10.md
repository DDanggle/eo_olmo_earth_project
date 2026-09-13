# OLMoEarth upstream 기여 후보 — 재진입 목록 (최종본, 2026-09-10)

> **9/11 재검토:** [실질 가치·새 우선순위·다른 EO와의 비교](PR_PRIORITIES_AND_EO_GAPS_2026_09_11.md).
> 첫 PR은 sample, 대표 기술 PR은 SCL(FirstValid·BestClear 모두), LFMC는 원인 확인 이슈다.
> 아래 표는 9/11 판단을 반영했다. 제품 코드/실험/게시 변경 없이 준비 문서만 갱신했다.

정리일 2026-09-10. 로컬 준비물(branch·본문·초안)을 재확인하고 **같은 날 upstream·HF·PyPI 에 대조함**
(`olmoearth_projects origin/main=23a3d7b` 변동 없음, `rslearn` 최신 release v0.1.14 = 2026-08-25).
제출은 아직 하지 않았음. 연구 결과 홍보와 좁은 upstream 수정은 분리함. SSOT 는 [PR_DOSSIER.md](../PR_DOSSIER.md).

## 오늘 확인한 사실(제출 판단의 근거)

| 확인 항목 | 결과 |
|---|---|
| sample `annotation_features.geojson` (origin/main) | 여전히 `es_annotations_task_id`·`es_label` 등 legacy key, 6 feature |
| runner 가 `oe_*` 를 요구하는가 | PyPI `olmoearth_runner-0.1.12` wheel(`uv.lock` 고정 버전) `annotation_features.py` 에 `oe_annotations_task_id: UUID`, `oe_labels: dict[str,int\|float\|None]` 필수. 0.1.14 도 동일 |
| open PR / issue 중복 | open PR 4건(#37·#42·#43·#64), open issue 13건. #1·#2 와 직접 중복 없음 |
| 이슈 작성 권한 | 외부 계정의 작성 이력 확인. **9/11 교정:** 현재 사용자 계정의 작성 권한은 로그인 후 확인해야 함 |
| LFMC HF ckpt | `model.ckpt` 1,139,505,083 B·LFS oid 유지. **9/11 교정:** repo lastModified는 2025-11-03, 파일 lastCommit은 2025-10-30. `docs/lfmc.md` 580.6 유지 |
| rslearn SCL(dossier #10a/b) | v0.1.14 FirstValid L115·BestClear L279 둘 다 layer resampling을 scoring read에 전달. PC 자산 의존성은 별도 문제 |
| lock 스큐(#5) | origin/main `uv.lock`: rslearn 0.0.23 · olmoearth-pretrain 0.0.2 · runner 0.1.12 |
| partial-band mask(#6) | v0.1.14 `model.py` 도 `Modality.get(...).band_sets` 정적 수로 mask 생성 |
| forest_loss_driver | 내부 `olmoearth_datasets.sentinel2_l2a` 레이어 2개 유지. main 최신 머지 #39 가 Ai2 내부 배포용 |
| 로컬 branch | `fix/sample-annotation-oe-schema` HEAD 21b658a = origin/main + 1 file(+36 −24). fork(`DDanggle/olmoearth_projects`) 없음 |

## 제출 순서와 형식 (9/11 재정렬)

| 순서 | 후보 / 대상 | 형식 | 준비 상태 | 남은 것 |
|---|---|---|---|---|
| 1 | sample annotation `es_* → oe_*` / olmoearth_projects | **PR** | branch·본문([pr_bodies/01_sample_schema.md](../pr_bodies/01_sample_schema.md), 버전 주장 wheel 로 검증) 완료. 8월 Linux 6-window 재현 | fork → push → PR. 원하면 current runner 0.1.14 로 quick-start 1회 재실행 후 제출 |
| 2 | SCL categorical scoring 에 nearest / rslearn | PR(소) | 두 scoring 경로를 현행 소스로 확인 | 두 클래스 최소 patch + 실제 재격자 회귀 테스트. 반사도 bilinear 유지 |
| 3 | LFMC 공개 ckpt MSE 불일치 / olmoearth_projects | **Issue** | [ISSUE_DRAFT_lfmc.md](../ISSUE_DRAFT_lfmc.md) 9/11 교정: 원인 단정 제거, repo/file 날짜 분리 | 실행 config·로그·과거 평가 파일 hash 묶음 확인 후 제출 승인 |
| 4 | release–lock–runner 호환성 / olmoearth_projects·rslearn | Issue → PR | lock 스큐·v1.2 로드 실패 기록 | runner 포함 compatibility matrix; 오래된 lock 자체를 버그로 단정하지 않음 |
| 5 | forest_loss_driver 외부 실행 불가 + URL 미설정 재시도 | **Issue** | 과거 증상·401·fail-fast 제안(dossier #3·#6) | 영문 본문·현행 runner 최소 재현. 내부 소스 교체 PR 금지 |
| 6 | SCL 보조-band 의존성 선언 / rslearn | 문서 → Issue/RFC | 공개 경로 gap 재확인 | 완전한 PC 설정 예제부터. nearest 수정과 분리 |
| 7 | embedding 입력·품질·시간 receipt | 튜토리얼 제안 | 제주 자산·기존 metadata 재사용 가능 | 팀 수요/중복 확인 후. 상세는 9/11 우선순위 문서 |
| 8 | partial-band mask 릴리스별 처리 / rslearn | 보류 | 내부 probe 만 | 공개 config 경로 최소 재현 + 정책 합의 전엔 제출 안 함 |

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

## Ai2 OlmoEarth 팀 관점 판정 (2026-09-13 추가) — 메인테이너·채용 담당자의 눈으로

기준 두 개. (a) **메인테이너 효용**: 머지·확인하면 그들의 일이 줄거나 사용자 불만이 사라지는가. (b) **채용 신호**: 이 기여가 "코드를 읽고, 재현하고, 통제 실험을 한 사람"임을 보여주는가. 두 기준이 다르게 나올 수 있어 분리해 적음. 2026-09-13 추가 확인: rslearn 레이어 `resampling_method` 기본값은 **bilinear**(`config/dataset.py` L643) → SCL 범주형 보간 버그는 기본 설정 사용자 전원에게 살아 있음. SCL 문서는 "SCL band 접근 필요, 없으면 오류"를 이미 명시함.

### A. 팀에 실제로 유의미 — 제출

| 후보 | 메인테이너 효용 | 채용 신호 | 팀이 볼 때의 주의점 |
|---|---|---|---|
| **LFMC 공개 ckpt 불일치(#2)** | **최상.** 자기들이 공개한 가중치가 문서 수치를 못 내는 건 팀이 가장 먼저 알고 싶은 정보. 사용자가 이걸 밟으면 모델 전체를 불신함 | **최상.** 재학습으로 데이터·레시피 정상을 입증하고, 버전·split·steps/epoch 통제까지 돌린 건 연구자의 일하는 방식 그 자체 | 단정 금지. "다른 run 일 가능성" 은 가설로, revision 확인을 요청. 558.8 ckpt 공유 제안은 좋은 마무리 |
| **sample 스키마 PR(#1)** | 상. quick-start 첫 명령이 죽는 건 신규 사용자 이탈 원인. 1파일이라 리뷰 비용 0 | 중. 난이도는 낮지만 "실제로 튜토리얼을 끝까지 돌린 사람"이라는 증거. 첫 PR 로 정확히 맞는 크기 | 본문에 wheel 필드 근거·6 window 재현을 넣었으니 그대로. 테스트 추가 요구가 오면 후속 커밋 |
| **SCL 범주형 scoring 에 nearest(#4)** | 상. 기본값 bilinear 라 `Sentinel2SCLBestClear` 사용자 전원이 class ID 보간된 점수로 장면을 고름. 조용한 품질 버그 | 상. rslearn 내부(compositor → tile store 읽기 경로)를 따라가 원인을 특정한 것은 코드 리딩 능력의 직접 증거 | scoring 읽기만 nearest 로 고정하는 10줄 내외 patch + 합성 테스트(bilinear 반사도 레이어 + SCL). 반사도 출력은 건드리지 않음을 본문에 명시 |

### B. 팀엔 약하거나 이미 아는 것 — 이슈로 축소하거나 보류

| 후보 | 왜 약한가 | 처리 |
|---|---|---|
| SCL 보조 band 의존성 선언(#5) | 문서가 이미 "SCL band 필요, 없으면 오류"라고 적음. 자동 전달은 설계 변경이라 외부 RFC 로는 먹히기 어려움 | 제출 안 함. #4 patch 본문에 한 줄 "auto-registration 은 별도 논의"로만 언급 |
| forest_loss_driver 내부 API(#3) | 팀은 내부 배포용임을 앎(#39). "외부에서 안 된다"는 이슈는 그들에겐 새 정보가 아님 | **URL 미설정 시 무한 재시도(#6) 만** 작은 이슈로. 문서에 "internal data source" 한 줄 추가 제안 포함. PC 로 바꾸는 패치는 우리 로컬용 |
| lock 스큐·v1.2 로드 실패(#13) | lock 이 오래된 건 그들도 앎(#49 인접). 외부가 만든 호환표는 금방 낡음 | 제출한다면 "구 패키지가 v1.2 config 를 조용히 무시하고 v1 을 짓는다 → 명시적 오류를 내달라"는 **좁은 이슈**만. 표 만들기는 우리 내부용 |
| partial-band mask(#12) | 팀 답은 "모델이 기대하는 band set 을 넣어라"일 가능성이 큼. 공개 config 재현이 없으니 버그로 못 냄 | 제출 안 함. 대신 면접·토론용 카드로 보관: "v1.2 단일 bandset 은 B01/B09 결측을 표현할 수 없다"는 관찰은 기술 대화에서 강함 |
| macOS hang(#8), python 상한, 심링크(#7), timestep 문서(#9) | 환경 특수·설계 결정·비공개 레포·사용자 오해. 팀 일이 늘기만 함 | 제출 안 함 |

### C. 우리(모델러)가 스스로 해야 하는 것 — upstream 에 넘기지 않음

- **한국 KR-4 결함 전부.** test 64칩 시간 창 비대칭, FULL 배치 32/16 노출 차이, seed 가 초기화를 통제 못 함, GT-present mIoU 정의. 우리 trainer 문제이고 v2 코드에 이미 수정이 들어감. 재실행 후 KR-5.
- **로컬 어댑터.** SCL nearest 어댑터(`code/scl_compositor.py`), forest_loss_driver 의 Planetary Computer 교체, lock 우회 조합(runner 0.1.14 + rslearn 0.0.27). 이건 우리 환경을 돌리기 위한 것이지 상류 수정이 아님.
- **12→4 timestep 오해.** 문서를 더 읽었어야 할 문제. 배운 것은 STUDY 카드로만.
- **band set 정책.** 10밴드 제품에 B01/B09 를 어떻게 채울지는 우리 실험 계약(MISSING 마스크 + 0)으로 이미 정했음. 상류에 물을 게 아니라 논문에 명시할 것.

### 팀이 이 사람을 어떻게 볼지 — 정직한 추정

- 강점으로 읽힐 것: LFMC 재현(통제 실험 3종), SCL 버그의 코드 경로 특정, 첫 PR 을 작게 낸 절제.
- 약점으로 읽힐 위험: 후보를 한꺼번에 쏟아내면 "이슈 폭탄"으로 보임. 한 번에 하나, 답이 오면 다음. 사설 문서·연구 수치 링크 금지(REVIEW_NOTES 규칙).
- 순서 확정: **#1 PR → 답 확인 → #2 이슈 → #4 patch**. 나머지는 답이 오는 속도에 따라.
