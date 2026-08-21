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

## 9. 임베딩 가이드 기본값이 구름 지역에서 부적합 (신규, 2026-08-21)

**대상**: rslearn `docs/examples/OlmoEarthEmbeddings.md`

**증상**: 가이드 예제의 `query_config`는 `space_mode: MOSAIC`(기간당 장면 1개).
구름이 많은 지역에서는 그 기간의 최선 장면이 흐리면 그대로 오염되고, 하류 변화탐지가
전부 구름을 검출한다. Ai2 자신의 실전 설정(`olmoearth_run_data/lfmc`)은
`PER_PERIOD_MOSAIC`(기간당 다장면 합성)을 쓴다.

**근거 (제주 4개년 실측)**: 연도별 "최악 모자이크" 구름 비율 평균 0.53~0.84 →
거의 모든 픽셀이 매년 최소 1장은 절반 이상 구름. 사후 마스킹 시 생존 픽셀 1.2%(전부 바다).
변화탐지 Top-30 육안 검증에서 5/5(v2), 3/5(v3)가 구름으로 판명.

**제안**: 가이드에 구름 많은 지역용 권장 설정과 함께, 임베딩 산출물에 품질 마스크를
동반하라는 주의를 추가. (Earth Embeddings 서베이 arXiv:2608.03410도 같은 공백을 지적)

**부가**: `load_all_crops` + workers 16 기본값이 1024px 윈도우에서 **트레이스백 없이**
OOM 사망(28/82에서 중단). workers 6 / batch 4로 해결. 문서 주의사항 후보.

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
4. #3, #9 — 문서·설정 개선 (#2와 함께 묶어도 좋다)
5. 나머지는 이슈로 축약

---

## 10. rslearn — `space_mode` 의미가 오해를 유발 (2026-08-22 실측)

**대상**: `allenai/rslearn` 문서 + `docs/examples/OlmoEarthEmbeddings.md`

**증상**: `space_mode: MOSAIC`과 `PER_PERIOD_MOSAIC`이 **서브타일 윈도우에서 완전히 동일한
결과**를 낸다. 제주 216윈도우를 두 설정으로 각각 materialize한 결과 래스터가
**md5까지 일치**(`items.json`의 기간별 장면 수도 양쪽 모두 1). 사용자는
"PER_PERIOD_MOSAIC = 같은 기간의 여러 장면을 겹쳐 구름을 메운다"로 읽기 쉽지만,
실제로는 윈도우를 *공간적으로* 덮는 데 필요한 장면만 모자이크한다. 윈도우(1024px=10km)가
S2 타일(110km) 하나에 들어가면 두 모드가 같아진다.

**비용**: 이 오해로 2시간 재다운로드 + 1시간 GPU를 소모했다.

**제안**: 두 모드의 차이를 문서에 명시하고, "구름을 줄이려면 무엇을 해야 하는가"
(타임스텝 수 늘리기 / 구름 마스크 밴드 / 짧은 기간 + 다수 후보에서 선택)를 안내.

## 11. 임베딩 가이드 예제가 12개 모자이크 중 4개만 사용 (2026-08-22 실측)

**대상**: `allenai/rslearn` / `docs/examples/OlmoEarthEmbeddings.md`

**증상**: 가이드의 `model.yaml` 예제는 `query_config.max_matches: 12`로 12개 기간의
모자이크를 받도록 데이터셋을 정의하면서, 모델 입력은
`layers: ["sentinel2_l2a", "sentinel2_l2a.1", "sentinel2_l2a.2", "sentinel2_l2a.3"]`로
**앞 4개만** 사용한다. 사용자는 12개를 다 쓰는 것으로 오해하기 쉽고, 그 결과:
- 구름 낀 모자이크 1장이 모델 입력의 **25%**를 차지 (12개를 쓰면 8%)
- 연중 계절 신호의 2/3가 누락
- 임베딩 기반 변화탐지가 사실상 "1~4월 구름 상태"를 측정

**근거**: 제주 4개년 변화탐지 Top-30이 특정 연도의 첫 모자이크 구름에 지배됨
(육안 검증 v2 5/5, v3 3/5). 12타임스텝으로 재추출해 비교 중.

**제안**: 예제를 12개 레이어로 맞추거나, 왜 4개만 쓰는지(비용?) 주석으로 명시.
