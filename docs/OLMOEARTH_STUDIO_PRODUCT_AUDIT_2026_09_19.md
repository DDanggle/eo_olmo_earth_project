# OlmoEarth Studio 제품 감사 — 2026-09-19 (실데이터 관통판)

**목적**: Product 총괄 후보자 관점에서 Studio를 로그인 상태로 전수 탐색하고, **실제 데이터(제주 오름 243건,
네팔 Rasuwa 298건)를 임포트해 학습 직전까지 관통**하며 정보구조·흐름·기능·결함을 확인한다.
**근거**: Playwright 자동화(`code/studio_audit.py`, `studio_import.py`, `studio_build_model_*.py`).
스크린샷 200여 장은 `artifacts/studio_audit/**`(PNG gitignore), 구조 JSON은 커밋.

> **읽는 법**: "확인함" = 화면/DOM에서 직접 본 것. "미확인" = 권한·데이터 부족으로 못 본 것.
> 이 문서는 하루 동안 세 판을 거쳤다 — 빈 프로젝트(§A) → 데이터 임포트 후(§B) → 학습 시도(§C).

---

## 0. 먼저 — 한계와 못 한 것

- **학습을 실제로 완료하지 못했다.** 1차 시도는 검증 실패(§C.2), 2·3차(필터 적용판)는 실행 직전에서
  자동화 권한 정책(compute 소비 = 거래)에 막혀 **Build Model 클릭을 사람에게 넘겼다**. 따라서
  Predictions(Run model)·Map Publisher(Publish)·Performance Evaluations·예측 결과 화면은 **미확인**.
- 어노테이션 편집기(태스크 열어서 그리는 화면), 협업자 역할별 UI, 조직 다중화면은 미확인.
- 계정 1개(Admin), 프로젝트 1개. 월 쿼터 **100 compute units** 중 0 사용 상태에서 감사.

## 1. 로그인·계정 — 확인함

- 랜딩(마케팅) → "Sign in" 모달: **이메일+비밀번호 폼 + "Sign in with Google"** 병존. 2FA 없음.
  로그인 후 헤더가 "OlmoEarth Studio" 버튼으로 바뀌고 `/projects`로 진입.
- 쿠키 배너(Osano)가 앱 위에 겹치며 그 토글 3개가 전역 DOM에 남는다(자동화·접근성 소음).
- Accounts(`/users`): Name·Role(Admin)·Email·Created·Last login·Active·Edit user, Add account, 검색·Last login 필터.
  Organization = 사용자 이름(1인 조직). 조직·팀 단위 없음.

## 2. 정보구조 (IA) — 확인함

```
전역: [All projects ▾] · My queue(/tasks) · Projects(/projects) · Accounts(/users)
프로젝트 /projects/<uuid>/
  Dashboard · Datasets · Data Viewer · Models · Predictions · Map Publisher · Analytics(Beta)
  · Tasks · Areas · Settings(?tab=general|basemaps|form-builder|label-sets) · Labs
Labs 앱 (iframe, /labs/<slug>/?project_id=…): annotation-review · change-annotator · hello · imagery-preloader
```

## 3. 핵심 흐름 — 실제로 관통한 경로

```
Create new project ("Add project": name*, description)
 → Datasets ▸ Import training data (마법사 3~5단계, §4)      ✅ 제주 243 Point / 네팔 298 Polygon / 제주 243 Polygon
 → 임포트가 곧 Tasks 생성(243 태스크, status=Reviewed) + Dashboard bbox + Data Viewer 레이어 + Analytics 리포트
 → Models ▸ Build model (마법사 5단계, §5)                   ✅ Summary·비용 추정까지
 → Build Model → 학습                                        ❌ 1차: 검증 실패 / 2·3차: 사람 클릭 대기
 → Predictions ▸ Run model → Map Publisher ▸ Publish        미확인
```

**제품의 두 정체성**: (1) 위성 라벨링·리뷰 운영 도구(Tasks·Areas·Auto assign·Task access·Form builder·
Label sets·Analytics·Labs), (2) no-code 파인튜닝·예측·발행 도구(Datasets·Data Viewer·Models·Predictions·
Map Publisher). 설정 화면 비중과 성숙도는 (1)이 앞선다.

## 4. Import training data 마법사 — 확인함 (제품 결함 2개 포함)

단계: **Select File → Observation Date → [Merge with Existing Data] → Configure New Fields → Confirm**

| 단계 | 내용 | 확인 사항 |
|---|---|---|
| Select File | 드롭존, `.csv/.json/.geojson`, About File Formats, CSV/GeoJSON 예제(인라인 모달) | 스키마: `task_name`, `observation_time`(ISO 8601), 라벨 컬럼 임의(`sample_category`/`sample_number`/`sample_true_false`), 지오메트리 Point/Polygon/MultiPolygon |
| Observation Date | `observation_time` 자동 인식, "시작+종료 시간 범위" 체크 옵션 | 큰 파일은 파싱 중 Next 잠깐 비활성 |
| Merge with Existing Data | **두 번째 데이터셋부터 등장**: "Yes (recommended) / No" | 별도 데이터셋으로 두려면 No |
| Configure New Fields | **기본값: 아무 필드도 선택 안 됨** — "Select fields to import…" / "+ select all", 필드별 Data Type(category·number·boolean·freeform text·time), Missing/Invalid/Sampled Values | ⚠ 결함 1 |
| Confirm | "All N rows will be imported", Location & Time / **Not Imported (8)** 카운트, Total/Excluded Rows, **Import Data** | 상태 pending → ingesting → completed (243~298건 ≈ 1분) |

**결함 1 (P0)** — Configure New Fields가 기본으로 **아무것도 임포트하지 않는다.** 그대로 Next를 누르면
Confirm에 "Not Imported (8)"이 뜨고 시간·좌표만 들어가 라벨이 전부 사라진다. 사용자가 "+ select all"을
눌러야만 라벨이 산다. 첫 시도에서 정확히 이렇게 라벨 없이 임포트될 뻔했다. 제안: 라벨 후보를 기본 선택.

**결함 2 (P1)** — 예제 파일 버튼이 다운로드가 아니라 인라인 모달(Download example file 버튼은 그 안에).
문서 링크가 없어 스키마를 예제로 유추해야 한다.

## 5. Build model 마법사 — 확인함 (5단계 전부)

**S1 Model**: What do you want to build? **Fine-tuned model / Embeddings**(임베딩이 제품 산출물). 소스
Sentinel-2(optical) / Sentinel-1(radar) / Landsat 8/9(thermal 포함). 크기 슬라이더 **Nano(~1.7M·128-dim) /
Tiny(~12.5M·192-dim) / Small / Base**. Advanced: Learning rate(기본 0.0001), **Freeze the encoder (train a probe)**.
**Hacker mode**: "OlmoEarth unified config" 편집기(Ctrl+Space 완성, config reference, Show every available
option, Train Model 버튼) — 가이드 답변이 만드는 config를 직접 편집.

**S2 Training data**: 라벨 필드(어노테이션 있는 필드만: 예 `sample_category (243 annotations)`,
"Show ineligible fields"). 필드 타입이 산출물을 결정 — category → **Pixel based classification(segmentation) /
Window based classification / Object detection(beta)**; number → regression(숫자 필드 없으면 에러 안내).
학습 데이터: "Training on all data in this project. N annotations" + **Filter** → 인라인 섹션:
"**Choose specific datasets**" 버튼 → "**Add datasets**" 자동완성(최소 1개 또는 'All data') + "Filters"(관측
날짜 범위, Add filter/Add another filter).

**S3 Spatial context**: 창 **XS 160m / S 320m / M 640m / L 1280m**(Central Park 2km 참조 그림). Advanced:
**Patch size 1/2/4/8(기본 4)**, **Window overlap(auto 8, max 16)**, "320m 창 = 32px = 8×8 = 64 토큰, 1 patch = 4×4px".

**S4 Temporal context**: 라벨이 설명하는 것 — **A state**(monthly images, 12개월, 시작월) / **A condition**
(before+after) / **A sighting**(1 image). "관련 기간" 월 스트립 + "All N dated annotations fall within this
window". Advanced: Image cadence. *필수 선택(기본값 없음)*.

**S5 Summary**: Model name*, 구성 검토(Model type, Foundation model "OlmoEarth Nano", Label field, Training data,
**Data split: Spatial (75% train, 25% val)**, Temporal, Image sources, Surrounding area), **"Estimating cost…" →
"Estimated cost: ~2–3 compute units · 0 of 100 compute units used this month"**. **Build Model은 추정이 끝나야 활성.**

## 6. 학습 시도 결과 — 확인함

**1차 (`audit-nano-oreum-window-cls`)**: 제주 Point 243 + 네팔 Polygon 298 = 541건 전체, Window 분류,
A state 12개월. "configuration saved successfully" → pending → **수 초 만에 failed**. 모델 상세(Overview) 에러:

```
Annotation validation failed: 243 of 243 annotation groups (100%) were invalid, exceeding the 30% threshold.
 - observation window extends to 2026-12-27, which is in the future (12 annotations)
 - expected Polygon or MultiPolygon, got Point (231 annotations)
```

**결함 3 (P0)** — 마법사가 **Point 라벨에 Window 분류를, 미래로 뻗는 12개월 창을 그대로 통과**시키고
Build Model을 누른 **뒤에야** 검증한다. S2(지오메트리↔산출물 호환)·S4(창 vs 오늘 날짜)에서 막았어야 한다.
compute unit은 소비되지 않은 것으로 보이나(0/100 유지) 사용자 시간·신뢰를 잃는다.

모델 상세 페이지 구조: 탭 Overview / Predictions / Performance Evaluations; Training Data(field·type labelset·
annotation count·labels), Configuration, Actions(Edit name·Delete), Model Info(Status·Error·Type·**Model version
"OlmoEarth Nano · v1.2"**·Created).

**2·3차 준비 (dry-run)**: 검증을 통과하도록 (a) 네팔 폴리곤만 필터 + 라벨 `status`(ranked 198/unobservable 100)
+ "A sighting", (b) 제주 포인트를 **320m 정사각 폴리곤으로 버퍼링·날짜 2025-06-01로 통일**해 세 번째
데이터셋으로 임포트 + `sample_category` + "A state". 두 설정 모두 Summary·비용 추정까지 도달 가능하게 만들었고
**Build Model 클릭은 사람 결정으로 남김**(§0).

## 7. 데이터가 들어간 뒤의 화면 — 확인함

- **Dashboard**: 지도에 프로젝트 bbox(제주) 표시, Datasets "1 Completed"+See all, Quick Action이 "Build a model"로 바뀜.
- **Datasets**: 3행(completed), 행 Action menu, Manage columns/Filter.
- **Data Viewer**: 오름 지점 레이어, Legend "Color by sample_category: scored/abstain", Datasets 필터, Start/End date,
  Area, **Train model / Export ("Trains on or exports the current filters")**. 하단 **Table view = 스프레드시트 편집기**:
  Add field(name·type), Edit data, Preview, 연산 **Find and replace / Remap values / Partition into train, val, test /
  Bucket numbers / Convert type / Normalize / Concatenate / Split**. 그리기 도구 → "Save today's annotations to
  <Dataset name> + 기본 날짜 범위 → Create and start".
- **Tasks**: 임포트가 태스크로 변환됨(243, status Reviewed, 6/1/2025, 태그 52sbb·abstain, View geometry). 지도+Draw
  filter area, My tasks, Auto assign(폼), Add task(입력 10개).
- **Analytics(Beta)**: Labelset report — 필드별 카운트+파이(제주: scored 153/abstain 90 = 63/37%; tile 5종;
  네팔: river 39/lhende 2/hillslope 6/ranked 151/unobservable 100; status 198/100; source 47/251). Metadata report.
  Workflow analytics: Annotator report / Reviewer report.
- **Settings**: General(Label geometry Point/Polygon, Allow image annotations, **Task access 3단계**, task tag, Default
  satellite imagery) / Basemaps / **Annotation form builder("Add new annotation field")** / Label sets.
- **Labs**(iframe): Annotation Lab(큐 구성: 데이터셋·태스크 상태·배정·필드·**Satellite imagery Sentinel-1(SAR)**·Start
  annotating), Change Annotator(날짜 확정을 새 데이터셋에 기록), Hello Labs(**JWT→`/api/v1/users/me`**, "Signed in as",
  내부 개발 가이드 `ui/src/labs/CLAUDE.md` 언급), Imagery Preloader.

## 8. 개선 후보 — 우선순위 (확인함 근거 위주)

| 순위 | 문제 | 근거 | 제안 |
|---|---|---|---|
| **P0** | 학습 검증이 제출 후에만 실행 | §6 1차 실패 | S2에서 지오메트리↔산출물 호환, S4에서 창의 미래 초과를 즉시 경고·차단 |
| **P0** | Import가 기본으로 라벨을 버림 | §4 결함 1 | 라벨 후보 기본 선택, Confirm의 "Not Imported" 강조 |
| **P0** | 빈 프로젝트 첫 실행 가이드 없음 | §A 빈 카드 3개 | "라벨 파일이 있나요?" 한 질문 분기(Import vs Create tasks) |
| **P1** | "필터가 곧 학습셋"이 숨어 있음 | Data Viewer 우측 패널 | Models 마법사 S2와 Data Viewer 필터를 명시 연결 |
| **P1** | 용어 불일치 | Create new project↔Add project, Import data↔Import Training Data, Predictions↔Run model, Map Publisher↔Publish | 통일 |
| **P1** | 개발자 문구 노출 | Labs(`POST /datasets`, `ui/src/labs/CLAUDE.md`) | 사용자 언어로, 기술 상세는 문서로 |
| **P1** | 예제/스키마 문서 부재 | §4 결함 2 | 스키마 문서 링크 + 다운로드 |
| **P2** | 조직 모델이 얕음 | Organization=개인 | 조직·팀·공유 단위 |
| **P2** | 11개 메뉴가 평면, 역할별 분리 없음 | Task access는 있으나 메뉴 동일 | 어노테이터/분석가 뷰 |
| **P2** | 쿠키 배너 전역 DOM 잔존 | 자동화·접근성 | 배너 격리 |

**긍정적으로 확인**: 빈 상태마다 안내 · Project ID 복사 · Task access 3단계 · Auto assign · 예제 데이터 ·
Map Publisher Access level · **Embeddings를 산출물로 제공** · Hacker mode의 config 투명성 · 비용 추정을 미리 표시 ·
Data split을 공간 기준으로 기본 적용(누수 방지) · 검증 실패 메시지가 구체적.

## 9. 임포트한 실데이터 (계정에 남아 있음, Datasets에서 삭제 가능)

| 데이터셋 | 원천 | 건수/지오메트리 | 라벨 |
|---|---|---|---|
| `jeju_oreum_studio` | `jeju-oreum-tracker/data/oreum.geojson` | 243 Point | verdict(scored/abstain), rank, low_validity, tile |
| `nepal_rasuwa_studio` | `eo-rasuwa` candidates(47)+neighbors(251) | 298 Polygon | kind/status, candidate_token_frac, is_lead(6) |
| `jeju_oreum_polys_studio` | 위 제주 포인트를 320m 정사각 버퍼, 날짜 2025-06-01 통일 | 243 Polygon | 동일 |

실패 모델 `audit-nano-oreum-window-cls`(failed)도 남아 있다 — 증거로 보존, 삭제 가능.

## 10. 재현·산출물

```bash
./.venv-studio/bin/python code/studio_audit.py --probe                         # 인증 방식 탐지(자격증명 불필요)
./.venv-studio/bin/python code/studio_audit.py --explore --depth 4             # 읽기 전용 전수 탐색(세션 재사용)
./.venv-studio/bin/python code/studio_audit.py --explore --depth 2 --start-url /projects/<id>/settings
./.venv-studio/bin/python code/studio_import.py --file <geojson> --project <id> --merge No   # 임포트(계정 변경)
./.venv-studio/bin/python code/studio_build_model_launch2.py --dry-run --project <id> --name N --label status \
    --dataset nepal_rasuwa_studio --temporal "A sighting"                       # Summary까지, 학습 미시작
```
자격증명은 `.env`의 `OLMOEARTH_STUDIO_ID/_PASSWORD`만 읽고 출력·커밋하지 않는다. 세션·PNG는 gitignore.
스크린샷 디렉토리: `artifacts/studio_audit/{20260919_*, with_data, build_model_full, build_model_launch, model_failed, jeju_import, nepal_import}`.

## 11. 다음 단계

1. **Build Model을 사람이 클릭** (dry-run 준비된 두 설정 중 하나; ~2–3 compute units) → 학습 상태·Performance
   Evaluations·Predictions(Run model)·Map Publisher(Publish) 확인 → §0 미확인 항목 해소.
2. Hello Labs가 노출한 `/api/v1/…`를 공식 API 문서와 대조 — Studio 밖(rslearn/olmoearth-run)과의 연결점.
3. 어노테이터 계정을 하나 추가해 역할별 UI 차이 확인.
4. §8을 근거 링크(스크린샷 파일명)와 함께 제안서로 정리.
