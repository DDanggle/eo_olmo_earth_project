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

## 미팅 카드(Ai2_Meeting_Cards_DG.md, 2026-09-11)와의 연결 — 말한 것을 기여로 갚기 (2026-09-13 추가)

전제: 카드는 미팅 준비본이고 실제 대화가 어디까지 갔는지는 기록이 없음. 아래는 "카드대로 말했다"고 가정한 정리이며, 실제로 안 꺼낸 항목은 빼면 됨.

### 1. 미팅에서 한 약속 → 지켜야 할 것

| 카드에서 한 말 | 팀이 기대할 후속 | 우리 후보 | 기한 |
|---|---|---|---|
| ⑥ "I found a few issues... I'm planning to send pull requests on GitHub" | 며칠 안에 실제 PR/이슈가 오는가. 안 오면 말뿐인 사람 | **#1 sample PR → #2 LFMC 이슈 → #4 SCL patch** | Research Engineer 마감 **2026-09-16**. 그 전에 #1 은 반드시 올라가 있어야 함 |
| ⑧ "I'd be happy to share a short technical example and my CV" | 이메일 1통: CV + 기술 사례 1개 링크 | 기술 사례 = Rasuwa 공개 저장소(eo-rasuwa) + 이 재진입 문서의 A 항목 3개 요약(내부 링크 금지) | 미팅 후 48시간 안 |
| ⑧ "I'd also be happy to volunteer on case studies" / Asia 3 ideas | 구체 제안 1개 | 한국 3-task 를 "파트너형 end-to-end 예제"로 다듬어 제안(아래 3절) | 첫 답이 온 뒤 |

### 2. 개선 요청 6가지(T9~T14) → 어떤 형식이 맞는가

메인테이너 시각으로 나누면 **패치로 낼 수 있는 것 1, 측정치를 붙인 이슈로 낼 수 있는 것 3, 연구 주제로 남길 것 2**임.

| 요청 | 카드 내용 | 형식 | 우리가 이미 가진 근거 | 주의 |
|---|---|---|---|---|
| T9 구름 | SCL 마스크·20% 규칙을 밖에서 붙였고 시간이 가장 많이 들었다 | **패치(#4)** + 모델 요청은 토론 | SCL 범주형 보간 버그는 정확히 "구름 처리에서 밟은 것". 미팅 서사와 PR 이 일치함 | 토큰별 cloud 신호는 모델 설계 요청이라 PR 이 아님. #4 본문에 한 줄 동기로만 |
| T10 눈금 | 1−cos 크기가 모델마다 다름(Nano .10 · Base .25 · Large .02), placebo p99 를 만들어야 했다 | **이슈(모델 카드 제안)** | 세 모델의 평상시 Δz 실측치 | 우리 수치는 한 사건 기준. "typical Δz 를 모델 카드에 적어 달라"는 요청 + 우리 측정 표를 참고로 |
| T11 계절 | 몬순 평시 변화가 문턱을 끌어올림 | 연구 주제 | placebo 쌍 방법 | 제출 안 함. 논문·케이스 스터디 재료 |
| T12 센서 | S1·S2 가 한 공간이라지만 같은 장소의 Δz 분포가 다름 → 센서별 정규화 | **이슈(측정 요청)** | Nepal Δz 분포 + MS-118/121: S1 단일 → S2 캐시 갱신 회복 0~3%, 사영기로 cos .62→.81 이어도 회복 안 됨 | MS 수치는 미공개 개발 결과. 넣으려면 사용자 결정 필요. 넣지 않아도 Nepal 분포 차이만으로 이슈 성립 |
| T13 40 m | 집·도로·다리가 40 m 칸에서 안 보임 → 10 m 모드 | 토론(신중) | MS-115/120: 20 m 토큰(patch 2)으로 촘촘히 뽑아도 소형 물체 IoU 개선 없음(8지역 1승, 이탈리아 ≈.09) | **우리 결과가 요청을 약화시킴.** "토큰만 촘촘하면 된다"가 아니라는 걸 우리가 이미 봤으니, 요청하려면 "현재 patch 2 로는 안 되더라"를 같이 말해야 정직함 |
| T14 크기 안내 | Nano 로 충분했는데 다들 Base 받음(52,768 vs 8,517) | **이슈(모델 카드 한 줄)** | Nano=Large 겹침 점수(제한된 검사) | 작고 무해. T10 과 한 이슈로 묶어 "model card: typical Δz + size guidance" |

### 3. Pain point 2 "end-to-end 파트너 예제가 없다" → 우리가 만들 수 있는 가장 큰 기여

- 팀 문서는 단계별 튜토리얼은 있지만 raw 입력 → 검증된 결과까지의 파트너 예제가 없다고 우리가 말했음. 그 빈칸을 우리가 채우는 것이 PR 6건보다 큰 신호임.
- 후보: **한국 AI-Hub 3-task**(공개 데이터·공간 홀드아웃·3 head·비용 실측)를 `olmoearth_projects` 예제 형식(`olmoearth_run_data/<project>/` + docs)으로 재구성. 단 KR-4 결함(시간 창·배치 노출·seed) 수정 후 재실행(KR-5)이 먼저.
- 대안: Rasuwa(Nepal) 를 "frozen embedding 변화 검토" 예제로. 라벨 없는 워크플로라 예제로는 가볍고, 이미 공개 저장소가 있음.
- 순서: 이메일에는 "예제 기여 의향"만 적고, 실제 제안은 KR-5 뒤.

### 4. 미팅 뒤 실행 순서(날짜)

1. 09-13~14: fork → push → **#1 PR 제출**. 본문 끝에 "found while working through the sample quick-start" 한 줄.
2. 09-14: **#2 LFMC 이슈** 제출(최종본 그대로).
3. 09-14~15: 이메일(CV + eo-rasuwa 링크 + #1/#2 링크). Research Engineer 지원서 제출(마감 09-16).
4. 09-16~20: **#4 SCL nearest patch** + 합성 테스트 → rslearn PR.
5. 그 다음: T10+T14 모델 카드 이슈 1건, T12 센서 이슈 1건(수치 범위는 사용자 결정).
6. KR-5 완료 후: end-to-end 예제 제안.

### 5. 카드에서 고칠 문장 1개

- ⑥ "I'm planning to send pull requests" 는 #1 이 올라간 순간부터 "I've sent the first one, and two more are coming" 으로 바꿔 말함. 이메일에도 같은 문장.
