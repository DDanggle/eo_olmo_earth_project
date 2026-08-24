# PR·이슈 명세서 — 발견 9건과 제출 준비 상태

최종 갱신: 2026-08-21. 원칙: **PR을 찾아다니지 않는다.** 전부 실제로 부딪혀서 진단한 것만.
판단 기준 하나: *"이게 머지되면 메인테이너의 일이 줄어드는가?"*

## 요약표

| # | 대상 | 제목 | 등급 | 상태 |
|---|---|---|---|---|
| 1 | olmoearth_projects | sample 예제의 반쪽 스키마 마이그레이션 | **A (제출 준비 완료)** | 커밋됨, 브랜치 대기 |
| 2 | olmoearth_projects | LFMC 공개 체크포인트가 문서 성능의 60% | **A (이슈 초안 완료)** | `ISSUE_DRAFT_lfmc.md` |
| 3 | olmoearth_projects | forest_loss_driver를 외부에서 실행 불가 | A | 패치 보유, 미제출 |
| 4 | rslearn | `ingest:false` + PC Sentinel2 + CONTAINS → NotImplementedError | A | 재현 명확, 미제출 |
| 5 | olmoearth_projects | model.yaml이 미출시 rslearn API 사용 | B | 이슈 소재 |
| 6 | olmoearth_run(비공개) | `OEDATASETS_API_URL` 미설정 시 무한 재시도 | B | 이슈만 가능 |
| 7 | olmoearth_run(비공개) | 상대 `project_path` → 깨진 심링크 | B (한 줄 수정) | 이슈만 가능 |
| 8 | olmoearth_projects | macOS에서 `main.py` 행(hang) | C | 참고용 |
| 9 | rslearn / 문서 | 임베딩 가이드 기본값이 구름 지역에서 부적합 | B (신규) | 근거 확보됨 |
| — | olmoearth-runner | `requires-python <3.12` 상한 | 주의 | PR 금지, 이슈로만 |

---

## 1. sample 예제의 반쪽 스키마 마이그레이션 — 제출 준비 완료

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

**검증**: 수정 후 `prepare_labeled_windows`가 완주하며 라벨 윈도우 6개 생성
(LA 롱비치, EPSG:32611, 10m, train/val 분할). 서버(py3.11)와 맥 양쪽에서 확인.

**제출 상태**: 브랜치 `fix/sample-annotation-oe-schema`, 커밋 `5e044ee`
(1 file, +37 −25). 로컬에만 있음 — fork·push·PR 생성 대기.

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

## 4. rslearn — `ingest:false` 직접 materialize 경로 미구현

**대상**: `allenai/rslearn` / `rslearn/data_sources/direct_materialize_data_source.py:92`

**증상**: `ingest: false` + `planetary_computer.Sentinel2` + `space_mode: CONTAINS` 조합에서
`DirectMaterializeDataSource.get_item_by_name`이 `raise NotImplementedError`.
호출 경로: `materialize.py:80 read_raster_window_from_tiles → tile_store.py:233
get_raster_bounds → direct_materialize_data_source.py:163 → :92`.

**재현**: forest_loss_driver 설정 그대로 materialize 실행. 모자이크 모드(lfmc)에서는
안 밟히고 CONTAINS 모드에서만 발생.

**우회**: `ingest: true`.

---

## 5. model.yaml이 미출시 rslearn API 사용

레포 main의 `olmoearth_run_data/forest_loss_driver/model.yaml`이
`enable_confusion_matrix`(ClassificationTask), `rslearn.train.callbacks.checkpointing.
BestLastCheckpoint`를 사용하나 **어떤 PyPI 릴리스에도 없음**(0.0.23/0.0.27 확인).
결과: 문서대로 실행 시 `rslearn exited with code 2`. 반대로 rslearn git master를 쓰면
`olmoearth-runner`(PyPI)가 요구하는 `get_window_layer_dir`가 없어 ImportError — **삼각 스큐**.
정착 조합: runner 0.1.14 + rslearn 0.0.27 + 설정 패치 2건.

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

## 9. 임베딩 가이드의 시간축·합성 의미가 모호함 (보정, 2026-08-22)

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

## 제출 순서 (권장)

1. **#1** — 무해하고 명확, 첫 기여로 최적
2. **#4** — 재현 스크립트가 명확한 진짜 버그 (rslearn은 공개 레포)
3. **#2** — 가장 무겁다. 완결된 매트릭스로 정중하게, maintainer 경로 확인 후
4. **#10** — golden-window 재현과 회귀 테스트가 있는 rslearn categorical-resampling 개선
5. #3, #9 — 문서·설정 개선 (#2와 함께 묶어도 좋다)
6. 나머지는 이슈로 축약

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

## 13. lockfile 환경으로는 OlmoEarth v1.2를 로드할 수 없고, 실패가 불투명하다 (2026-08-24 실측)

**대상**: `allenai/olmoearth_projects` (uv.lock) + `rslearn` 로딩 경로

**증상**: 레포 lockfile이 고정하는 `rslearn 0.0.27 + olmoearth_pretrain 0.0.2` 환경에서
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
