# PR·이슈 명세서 — 2026-08-26 upstream 재감사판

최종 갱신: 2026-08-26. 원칙: **PR을 찾아다니지 않는다.** 전부 실제로 부딪혀서 진단한 것만.
판단 기준 하나: *"이게 머지되면 메인테이너의 일이 줄어드는가?"*

감사 기준: `olmoearth_projects origin/main=23a3d7b`,
`rslearn v0.1.14/master=c47952f`. 과거에 맞았어도 current upstream에서 해소됐으면 제출 큐에서
내린다. 아직 fork·push·PR을 하지 않은 후보는 `제출됨`이나 `review 중`으로 부르지 않는다.

## 요약표

| # | 대상 | 제목 | 등급 | 상태 |
|---|---|---|---|---|
| 1 | olmoearth_projects | sample 예제의 반쪽 스키마 마이그레이션 | **A (첫 PR 추천)** | 로컬 커밋, 미제출 |
| 2 | olmoearth_projects | LFMC 공개 체크포인트가 문서 성능의 60% | **A (이슈 초안 완료)** | `ISSUE_DRAFT_lfmc.md` |
| 3 | olmoearth_projects | forest_loss_driver를 외부에서 실행 불가 | A | 패치 보유, 미제출 |
| 4 | rslearn | `ingest:false` + PC Sentinel2 + CONTAINS → NotImplementedError | **종결** | current upstream에서 구현됨 |
| 5 | olmoearth_projects | model.yaml과 lockfile의 rslearn API skew | B+ | #13에 병합 |
| 6 | olmoearth_run(비공개) | `OEDATASETS_API_URL` 미설정 시 무한 재시도 | B | 이슈만 가능 |
| 7 | olmoearth_run(비공개) | 상대 `project_path` → 깨진 심링크 | B (한 줄 수정) | 이슈만 가능 |
| 8 | olmoearth_projects | macOS에서 `main.py` 행(hang) | C | 참고용 |
| 9 | rslearn / 문서 | 임베딩 가이드의 consumed-timestep provenance | C+ | clarification으로 낮춤 |
| 10 | rslearn | SCL 보조자산·범주형 resampling 계약 | **A-** | 작은 PR + RFC로 분리 |
| 12 | rslearn | partial-band mask가 release별로 달리 소비됨 | B+ | public API repro 필요 |
| 13 | olmoearth_projects | lockfile에서 v1.2 로드 불가 | B+ | compatibility matrix 먼저 |
| — | olmoearth-runner | `requires-python <3.12` 상한 | 주의 | PR 금지, 이슈로만 |

---

## 2026-08-26 재감사 결론

- `olmoearth_projects` 공개 open PR 4건(#37, #42, #43, #64)과 open issue 목록에 #1의 직접
  중복은 없다. upstream sample 파일도 legacy schema 그대로라 #1은 살아 있다.
- #4는 current rslearn의 direct materialization 구현과 `get_item_by_name` 회귀 테스트로 해소됐다.
  발견 계보는 보존하지만 제출하지 않는다.
- #5의 `미출시 API` 표현은 더 이상 맞지 않는다. current rslearn에는 API가 출시됐고, 실제 문제는
  project lock(`rslearn 0.0.23`, `olmoearth-pretrain 0.0.2`)과 current config/model release의
  compatibility skew다. #13에 병합한다.
- #10은 v0.1.14에서도 남는다. 다만 보조자산 dependency declaration과 categorical-nearest를
  한 PR에 넣지 않는다. nearest scoring을 최소 patch로 먼저 제안하고 dependency API는 RFC로 연다.
- #12는 연구적으로 중요하지만 public user path에서 10-band 입력→명시적 missing mask의 최소
  재현이 아직 없다. 내부 tensor probe만으로 upstream API를 설계하지 않는다.

상세 연구·데이터 연결 판정은
`docs/OLMO_EXTERNAL_DATA_ONBOARDING_AND_PR_AUDIT_2026_08_26.md`를 따른다.

---

## 1. sample 예제의 반쪽 스키마 마이그레이션 — 첫 PR 추천, 미제출

**대상**: `allenai/olmoearth_projects` / `olmoearth_run_data/sample/annotation_features.geojson`

**증상**: README의 첫 예제 명령이 pydantic 검증 실패로 즉사.
```
Field required [type=missing] oe_annotations_task_id
```

**원인**: ES→OlmoEarth 리브랜딩 때 짝 파일 중 하나만 마이그레이션됨.
`annotation_task_features.geojson`은 신형 `oe_*`인데 `annotation_features.geojson`은
구형 `es_*` + 스칼라 `es_label`로 남음. runner ≥0.1.12의
`AnnotationFeatureProperties`는 `oe_annotations_task_id` + dict형 `oe_labels`를 요구.

**수정**: `es_* → oe_*`, `es_label: 1 → oe_labels: {category: 1}` (6개 feature)

**검증**: 수정 후 Linux 서버(py3.11, runner 0.1.14)에서 `prepare_labeled_windows`가 완주하며
라벨 윈도우 6개 생성(LA 롱비치, EPSG:32611, 10m, train/val 분할). macOS에서는 JSON/schema
gate가 통과하지만 CLI가 이후 forkserver multiprocessing에서 멈춘다. 이는 별도 #8이며 #1의
end-to-end macOS 검증으로 세지 않는다.

**제출 상태**: 브랜치 `fix/sample-annotation-oe-schema`, 기능 커밋 `5e044ee` + newline 커밋
`21b658a` (전체 1 file, +37 −25). 로컬에만 있음 — fork·push·PR 생성 대기. 2026-08-26 current upstream과
open PR/issue를 재확인했고 직접 중복은 없었다. EOF newline·static schema gate는 보완했다.
current Linux runtime quick-start를 한 번 더 확인한 뒤 제출한다(macOS는 별도 #8에서 멈춤).

```bash
# 제출 절차 (gh CLI 필요: brew install gh && gh auth login)
cd ~/dong/ai_projects/olmoearth_projects
gh repo fork allenai/olmoearth_projects --remote --remote-name fork
git push fork fix/sample-annotation-oe-schema
gh pr create --repo allenai/olmoearth_projects \
  --head <내계정>:fix/sample-annotation-oe-schema --base main \
  --title "Migrate sample annotation_features.geojson to oe_* schema" \
  --body-file _work/pr_bodies/01_sample_schema.md
```

---

## 2. LFMC 공개 체크포인트가 문서 성능의 60% — 이슈 초안 완료

**본문**: `ISSUE_DRAFT_lfmc.md` (영어, 재현 절차·통제 실험·제안 포함)

**핵심 매트릭스** (전부 같은 데이터·같은 평가 코드):

| 가중치 | split | 환경 | MSE |
|---|---|---|---|
| 문서 주장 (`docs/lfmc.md`) | test | — | **580.6** |
| 공개 ckpt (HF, epoch 91) | val | rslearn 0.0.27 | 995.3 |
| 공개 ckpt | val | rslearn master | 995.4 ← 버전 효과 기각 |
| 공개 ckpt | **test** | rslearn 0.0.27 | **951.9** |
| **우리 재학습 (epoch 33/100)** | **test** | rslearn 0.0.27 | **558.8** |

**결론**: 데이터·설정·레시피는 정상. 우리가 그들 학습량의 1/3로도 문서 수치를 웃돎 →
**HF에 업로드된 체크포인트 파일이 문서의 실험 산출물이 아닐 가능성**이 높다.
보조 단서: 공개 ckpt는 60,260스텝/91에폭 = 662 steps/epoch, 공개 데이터는 655 → ~1% 다른
데이터 스냅샷.

**제출 경로 주의**: `olmoearth_projects`는 신규 이슈 작성이 제한돼 있음. 따라서
① `olmoearth@allenai.org`로 리포트 전송 또는 ② PR #1 설명에 함께 언급하며 maintainer
경로 요청. 둘 중 하나를 택한다.

---

## 3. forest_loss_driver를 외부에서 실행 불가

**증상**: `docs/forest_loss_driver.md`는 외부 사용자용 추론 절차를 안내하지만,
`dataset.json`의 pre/post_sentinel2 레이어가 Ai2 내부 API
(`olmoearth_datasets.sentinel2_l2a.Sentinel2L2A` → `https://datasets.olmoearth.allenai.org`)를
사용. 익명 요청은 **401**(직접 확인). 게다가 `OEDATASETS_API_URL` 기본값이 `""`라
미설정 시 명확한 에러 대신 `Invalid URL '/api/v1/items/search'` **무한 재시도**.

**수정안**: (a) 공개 `planetary_computer.Sentinel2`로 교체하거나 문서에 내부 전용임을 명시,
(b) URL 미설정 시 fail-fast. **(a) 패치 로컬 보유** (작업트리의 `dataset.json` 변경).

**검증**: 패치 후 페루 100건 추론 완주 (agriculture 74/mining 16/... GPU 2.3초).

---

## 4. rslearn — `ingest:false` 직접 materialize 경로 미구현 [UPSTREAM 종결]

**대상**: `allenai/rslearn` / `rslearn/data_sources/direct_materialize_data_source.py:92`

**증상**: `ingest: false` + `planetary_computer.Sentinel2` + `space_mode: CONTAINS` 조합에서
`DirectMaterializeDataSource.get_item_by_name`이 `raise NotImplementedError`.
호출 경로: `materialize.py:80 read_raster_window_from_tiles → tile_store.py:233
get_raster_bounds → direct_materialize_data_source.py:163 → :92`.

**재현**: forest_loss_driver 설정 그대로 materialize 실행. 모자이크 모드(lfmc)에서는
안 밟히고 CONTAINS 모드에서만 발생.

**우회**: `ingest: true`.

**2026-08-26 재판정**: current rslearn v0.1.14의 `DirectMaterializeDataSource`는 원격 COG를
직접 읽는 `get_raster_bounds/read_raster/materialize` 경로를 구현했고 Planetary Computer
`get_item_by_name` 회귀 테스트도 있다. 과거 pinned 환경의 마찰로 보존하되 새 PR/issue는 금지한다.

---

## 5. model.yaml과 lockfile의 rslearn API skew [#13에 병합]

레포 main의 `olmoearth_run_data/forest_loss_driver/model.yaml`이
`enable_confusion_matrix`(ClassificationTask), `rslearn.train.callbacks.checkpointing.
BestLastCheckpoint`를 사용하나 **어떤 PyPI 릴리스에도 없음**(0.0.23/0.0.27 확인).
결과: 문서대로 실행 시 `rslearn exited with code 2`. 반대로 rslearn git master를 쓰면
`olmoearth-runner`(PyPI)가 요구하는 `get_window_layer_dir`가 없어 ImportError — **삼각 스큐**.
정착 조합: runner 0.1.14 + rslearn 0.0.27 + 설정 패치 2건.

**2026-08-26 정정**: `어떤 PyPI 릴리스에도 없음`은 당시 조사 범위에서는 맞았지만 current
rslearn v0.1.14에는 해당 계열 API가 존재한다. 독립 버그가 아니라 오래된 project lock과 current
config의 version skew로 재분류해 #13과 합친다.

---

## 6·7. olmoearth_run (비공개 레포 — 이슈만 가능)

- **6**: `OEDATASETS_API_URL` 기본값 `""` → fail-fast 없음 (3번과 세트)
- **7**: `_setup_project_env`가 심링크 타깃을 resolve하지 않아, 상대 `project_path`로
  러너를 만들면 깨진 심링크 생성 → "file not found". 한 줄 수정(`target.resolve()`).
  로컬에서 재현 확인.

## 8. macOS에서 `main.py` 행

`olmoearth_projects/utils/mp.py`의 `init_mp()`가 forkserver + torch preload를 강제 →
macOS에서 Pool 생성이 돌아오지 않음. 러너 직접 호출로 우회 가능.
재현: README의 `prepare_labeled_windows` 명령을 맥에서 실행.

## 9. 임베딩 가이드의 시간축·합성 의미가 모호함 [clarification으로 강등]

**대상**: rslearn `docs/examples/OlmoEarthEmbeddings.md`

**증상**: 가이드 예제는 12기간을 materialize하지만 모델 입력은 앞 4개 레이어만 명시한다.
또 현재 rslearn 0.1.13에서 `MOSAIC + period_duration`과 폐기 예정
`PER_PERIOD_MOSAIC + period_duration`은 동일 handler다. 기본 시간순서는 역순이며 변경 예정이라,
설정 이름만 보면 실제 사용 기간·합성 의미를 오해하기 쉽다.

**근거 (제주 4개년 실측)**: 두 설정으로 216윈도우를 각각 계산했지만 cloud/zero 지표와
blind RGB가 동일했다. ordered source group 2,592/2,592, 원본 12밴드 표본 24/24,
임베딩 표본 24/24가 동일했다. 4기간↔12기간 Top-30 Jaccard는 0.091이고, 실제 첫 4기간은
2023~2025와 rolling-2026 사이 계절이 정렬되지 않았다.

**제안**: 예제에 실제 소비 timestep, 시간순서, `PER_PERIOD_MOSAIC` alias/deprecation을
명시하고 item-order manifest를 남긴다. 구름 개선은 SpaceMode 이름 변경이 아니라 SCL/cloud
mask를 pixel validity에 연결하는 합성 예제와 품질 마스크로 안내한다.

**부가**: `load_all_crops` + workers 16 기본값이 1024px 윈도우에서 **트레이스백 없이**
OOM 사망(28/82에서 중단). workers 6 / batch 4로 해결. 문서 주의사항 후보.

**비용**: 이 오해로 **재다운로드 2시간 + GPU 1시간**을 소모했다.

**2026-08-26 재판정**: current 문서는 item-group 수가 window time range와 `query_config`에
달린다고 이미 설명한다. 따라서 `12개를 materialize하고 4개만 쓰는 버그`라고 제출하지 않는다.
실제 consumed layer/time manifest와 시간 정렬 예제를 더 명시해 달라는 문서 clarification만 남긴다.

**기각된 초기 진단 (기록 보존)**: 처음에는 원인을 `space_mode`의 기하로 설명했다 —
`PER_PERIOD_MOSAIC`은 윈도우를 *공간적으로* 덮는 데 필요한 장면만 모자이크하므로,
윈도우(1024px=10km)가 S2 타일(110km) 하나에 들어가면 `MOSAIC`과 같아진다는 것이다.
후속 감사에서 더 단순한 사실이 확인됐다 — rslearn 0.1.13에서 두 SpaceMode는 **같은
`match_with_space_mode_mosaic` handler를 호출한다.** 따라서 이 절의 진단은 기하가 아니라
handler alias다. 초기 가설도 실패 계보로 남긴다(L3).

---

## 10. rslearn — SCL compositor의 숨은 자산 의존성과 categorical resampling

**대상**: `allenai/rslearn` / `rslearn/dataset/sentinel2_scl.py`,
`rslearn/data_sources/planetary_computer.py`

**재현 증상**: 반사도 12밴드 layer에 `Sentinel2SCLBestClear`를 지정하면 STAC item의
`asset_urls`에 `SCL`이 있어도 `missing scoring bands ['SCL']`로 materialize가 실패한다.
Sentinel-2 데이터소스가 layer `band_sets`와 교차하는 자산만 tile store에 등록하므로,
compositor의 보조-band 의존성이 자동 전달되지 않기 때문이다.

**두 번째 문제**: SCL을 band set에 추가하면 실행은 되지만, compositor `_score_item`은
반사도 layer의 `resampling_method`를 범주형 SCL read에 그대로 전달한다. 반사도에 일반적인
bilinear를 쓰면 class ID가 보간되어 `SCL in {4,5,6}` 점수가 왜곡될 수 있다.

**로컬 검증**: `code/scl_compositor.py` adapter로 SCL score만 nearest, 선택된 반사도 출력은
bilinear로 유지했다. 사전 고정 제주 golden window에서 source group과 pixel이 실제로 바뀌고,
첫 4기간 bad proxy 95.64% 감소·target 1.00→0.00·RGB 구름 제거를 확인했다. 실패 3회와 성공
로그를 `artifacts/results/jeju-v7-smoke*.log`에 보존했다.

**제안**: compositor가 보조-band 의존성을 data source context에 선언할 수 있게 하거나 SCL
band-set 요구를 config error로 명시하고, categorical SCL scoring은 nearest를 기본/강제로
분리한다. 최소 회귀 테스트는 bilinear reflectance + SCL BestClear 조합이다.

---

## 제출하지 말 것

**`olmoearth-runner`의 `requires-python <3.12` 상한**: 버그가 아니라 배포 설계 결정일 수
있다(그들의 컨테이너 이미지가 py3.11). 249개 의존성을 `==`로 못 박은 "잠긴 앱" 형태로
배포되는 것도 의도일 가능성. PR로 밀지 말고, "NGC 컨테이너(py3.12) 사용자가 설치 불가"라는
사실만 이슈로 보고하는 것이 예의.

---

## 제출 순서 (2026-08-26 권장)

1. **#1** — 무해하고 명확하다. EOF newline 완료; current Linux quick-start 확인 뒤 첫 기여로 제출
2. **#2** — 완결된 매트릭스. issue 작성 제한 때문에 maintainer 경로 확인 후 report
3. **#10a** — SCL scoring만 nearest로 분리하는 최소 rslearn patch + synthetic regression
4. **#10b** — compositor auxiliary-band dependency declaration은 별도 issue/RFC
5. **#13** — current runner integration까지 통과한 compatibility matrix/lock update
6. **#12** — public API 최소 재현과 maintainer가 원하는 missing-band policy가 정해진 뒤

#4는 해소됐고 #5는 #13에 병합했으며 #9는 clarification으로 낮췄다.

## 12. band_set 부재 선언이 릴리스에 따라 조용히 무시된다 (2026-08-24 실측, M8)

**대상**: `allenai/rslearn` — `rslearn/models/olmoearth_pretrain/model.py:_prepare_modality_inputs`
와 그것이 만드는 mask를 소비하는 `olmoearth_pretrain` 토큰화 경로

**증상**: rslearn은 modality mask를 `(b,h,w,t,S)`로 만들고 `S`를 **정적** 정의
`len(Modality.get(modality).band_sets)` = sentinel2_l2a면 항상 3으로 잡는다. 그런데 소비 측은
**로드된 모델의** `tokenization_config.get_num_bandsets(modality)`만큼만 순회한다.

| 릴리스 | 모델 bandset 수 | 읽히는 mask slice | slice 2를 MISSING으로 표시하면 |
|---|---|---|---|
| OlmoEarth v1 | 3 | 0,1,2 | 출력 변화 (max\|Δ\| 4.79) |
| OlmoEarth v1.2 | 1 (12밴드 단일 group) | **0만** | **출력 byte-identical — 무시됨** |

게다가 이 무시가 조용하다. `fast_pass`는 입력 mask 전체를 보고 꺼지므로 pooling이
masked-average 경로로 전환되지만, 출력 mask에는 MISSING이 없어 결과가 baseline과 동일하다.
사용자는 밴드 부재를 선언했다고 믿고, 경고는 없다.

**영향**: 10밴드 S2 제품 사용자 전체. 예로 PhilEO-downstream S2는
`B02 B03 B04 B08 / B05 B06 B07 B8A B11 B12`로 v1의 `band_set 0+1`과 정확히 일치하고
없는 `B01 B09`가 `band_set 2` 전체다. v1에서는 그 set을 MISSING으로 표현할 수 있으나
v1.2에서는 표현 수단 자체가 없다. 같은 입력을 두 릴리스에 대칭적으로 줄 수 없다.

**제안**: 정적 3-band-set mask를 만드는 대신, 로드된 모델의 tokenization config와 실제 band
availability를 함께 사용해 **release-aware input contract**를 구성한다. 모델의 bandset 분할이
데이터의 결측 경계와 일치하지 않는 partial-group missingness에는 명시적 imputation policy를
요구하고, 요구가 충족되지 않으면 조용히 통과시키지 않고 오류를 낸다.

**재현**: `code/probe_mask_path_c2a.py` (합성 입력, seed 고정, 게이트 6개)

**2026-08-26 제출 경계**: current v0.1.14도 wrapper mask를 static `Modality.get(...).band_sets`
수로 만든다. 다만 공개 loader는 필요한 12밴드를 못 읽으면 먼저 오류를 내므로, 사용자가 실제
config에서 10밴드 결측을 선언하는 end-to-end public API repro는 아직 없다. 이는 연구 blocker와
설계 gap으로는 유효하지만 곧바로 PR할 수 있는 좁은 버그 수정은 아니다. 먼저 fail-closed API와
partial-group imputation/mask 정책을 maintainer와 합의한다.

## 13. lockfile 환경으로는 OlmoEarth v1.2를 로드할 수 없고, 실패가 불투명하다 (2026-08-24 실측)

**대상**: `allenai/olmoearth_projects` (uv.lock) + `rslearn` 로딩 경로

**증상**: 레포 current `uv.lock`이 고정하는 `rslearn 0.0.23 + olmoearth_pretrain 0.0.2` 환경에서
`ModelID`에 v1.1·v1.2 엔트리가 없고, `OlmoEarth(model_path=<v1.2 snapshot>)`는
원인 안내 없이 state_dict 오류로 죽는다.

```
size mismatch for encoder.composite_encodings.per_modality_channel_embeddings.sentinel2_l2a:
  checkpoint [1,192] vs current model [3,192]
Unexpected key(s): ...attn.rope_mixed_freqs, ...pixel_proj...
Missing key(s): ...sentinel2_l2a__1..., ...sentinel2_l2a__2...
```

`load_model_from_path`는 snapshot의 `config.json`을 읽지만, 구버전 패키지가 v1.2 필드
(`tokenization_config.overrides`, `use_linear_patch_embed`, `temporal_rope_dim_frac`)를
조용히 무시하고 v1 아키텍처를 짓는다. `rslearn 0.1.x` + `olmoearth_pretrain_minimal`에서는
정상 로드된다(v1.2 Base 113.99M).

**제안**: 지원되지 않는 릴리스에 대해 "이 패키지 버전은 v1.2 config를 지원하지 않는다"는
명시적 오류를 내거나, 문서에 릴리스별 최소 버전 표를 넣는다.

**2026-08-26 current 비교**: rslearn v0.1.14는
`olmoearth-pretrain-minimal>=0.0.6`과 v1.1/v1.2 `ModelID`를 지원한다. 따라서 해결책은 단순히
`uv.lock` 하나를 덮어쓰는 것이 아니라, `olmoearth-runner`와 project examples가 함께 통과하는
release compatibility matrix를 만들고 그 결과로 dependency lower bound/lock을 갱신하는 것이다.
