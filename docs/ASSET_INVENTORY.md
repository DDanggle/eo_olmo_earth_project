# 보유 자산 — 2026-08-31 재시작 감사

> Nepal 전용 앱·코드·raw/intermediate artifact는
> `/Users/dgyi/dong/ai_projects/nepal-live-twin`로 이관했다. 아래 Nepal 절은 역사적 인벤토리이며
> 이 저장소의 로컬 자산을 뜻하지 않는다. 활성 본선 자산은 Sen12, M65 확증 번들, Presto 계약,
> AI-Hub split seal이다.

`docs/CRITICAL_PATH.md`의 측정 사슬 1~7에 매핑함. **"있다"가 아니라 "어느 단계에 쓰이는가"로 정리함.**

## 1. 핵심 경로에 바로 쓰이는 것

### Sen12Landslides harmonized S2 — **수신 완료** (사슬 1·2·3·4)

| | |
|---|---|
| 위치 | 서버 `/home/work/data/sen12landslides` |
| 용량 | **38 GB** (28 파트, `data_harmonized/s2/`만) |
| 파일 | **`.nc` 13,628개** — 문서상 S2 패치 수와 정확히 일치 |
| 구조 | 128×128 @10 m, **15 timestep**, B02–B12(10밴드) + SCL + MASK + DEM |
| 속성 | `event_date`, `date_confidence`, `pre_post_dates`, `crs`, `center_lat/lon` |
| 라이선스 | **CC BY 4.0** — 파생물 공개 가능 |
| PB04 | harmonized는 +1000 DN offset 보정됨. **`data_raw`와 섞지 말 것** (M11) |

2026-08-25 실물 64파일 contract smoke에서 추가로 확인했다.

- `MASK`는 이진이며 한 표본의 15 timestep에 **같은 event polygon이 반복**된다. 시점별 독립
  라벨 15개로 세면 안 된다.
- 양성 23/23은 `pre_post_dates`가 유효했다. 음성은 event cutoff가 없으므로 operational task에는
  별도의 pseudo-cutoff 정책이 필요하다.
- 64/64의 `center_lat/lon` 값이 위경도 범위를 벗어났다. 이름과 달리 projected coordinate이므로
  CRS 변환 없이 위경도로 쓰지 않는다.
- `code/build_sen12_gp_contract.py` 전수 결과 13,628/13,628파일의 shape/band/time/static-mask 계약이
  통과했다. `annotated=True`지만 mask=0인 Hiroshima 2개는 원본을 고치지 않고 S12q/R-event에서 제외했다.
- annotation-matched 11지역 중 **LanaoDelNorte 71개는 양성 MASK가 0**이다. annotation 감사 cohort에는
  남지만 segmentation headline은 양·음성이 모두 있는 **10지역 LOCO / 6,834 eligible sample**이다.
- 전체 양성 6,737개 중 유효한 단일 pre/post pair는 5,397개(**80.11%**). 나머지 1,340개는 주로 한
  patch에 여러 event/date가 들어가 S≤t에서 제외 또는 별도 정책이 필요하다.
- OLMo v1은 15시점 입력에서 time embedding `12 != 15` shape error가 났다. G-P는 SCL clear 상위
  12개를 label-independent하게 고른 **S12q**로 고정하고 모든 baseline 입력을 맞춘다.
- **OLMo v1 smoke 6/6 통과**: 10지역·양/음성 32/32, 64표본/256crop, 15.44초(4.146 sample/s),
  peak CUDA 0.740 GB, fp16 cache 1,572,992 B/sample, `768×32×32`, deterministic replay diff 0.
  full 6,834개 cache도 완료되어 **1,130.05초(18.8분), 6.0475 sample/s, 10.75 GB**로 실측됐다.
- 전수 manifest SHA-256 `dcdfef9a…`, label anomaly 목록 SHA-256 `bf086042…`로 봉인했다.
- full-context alternate cache도 6,834/6,834 content seal을 통과했다. 다만 E1에서 tiled 대비
  IoU가 small -0.0140, large -0.0963으로 악화됐으므로 **보유 자산이지 채택 recipe가 아니다.**
  네 E1 cell의 pilot/per-sample과 분석 JSON 1.3 MB는 `evidence/e1_factorial_v2/`에 봉인했다.

지역별 패치 수 (`region_index.jsonl`, 중복 제거 후):

| 역할 | 지역 | 패치 |
|---|---|---|
| **annotation-matched 후보** | kyrgyzstan1 1566 · newzealand 1173 · chimanimani 1133 · hiroshima 864 · indonesia 573 · thrissur 427 · kyrgyzstan2 416 · hokkaido 290 · itogon 235 · china 159 · lanaodelnorte 71 | **6,907** |
| **T-x arm** | **italy** | **5,321** |
| 제외 (다른 저자) | dominicamaria 1148 · usa 238 · nepal 14 | 1,400 |

`nepal`이 14패치뿐인 것이 M12의 "네팔 폴리곤 8개"와 일관됨. 파일명 접두는 `usa` 하나로
합쳐져 있어 inventory의 `USA_PuertoRico`/`USA_Alaska` 구분과 다름 — 결합 시 주의.

### AI-Hub 71363 (한국) — **수신·봉인 완료** (사슬 2·3의 목표 지역)

| | |
|---|---|
| 원본 | 서버 `/home/work/data/olmoearth/aihub` **3.2 GB** (Sentinel-2 8 zip) |
| 실제 조합 | **2,699 (타일, 날짜) 쌍** — 594 타일 × 60 날짜, 타일당 1~8일 |
| 좌표 | EPSG:32652, `coordinates`는 **좌상단** (M9, 중위 4.2e-05 m 정확일치) |
| 플랫폼 | SENTINEL-2A 1,961 / 2B 738 |
| split | **동결됨** — train 393 / val 84 / test 113 / excluded 4 타일 |
| 계약 봉인 | 4층 (원본 zip / 파생 / 내용 / 코드), seal `5b088ada…` |
| CI | `tests/test_aihub_split_invariants.py` 10개 |

### OlmoEarth 체크포인트 — 있음 (사슬 1)

`.cache/huggingface/hub/` 에 `allenai--OlmoEarth-v1-Base`, `allenai--OlmoEarth-v1_2-Base`.
그 외 timm/convnext/vit/swin 등 20여 개(다른 프로젝트 것), Poseidon, TinyLlama, bge-m3.

### 파이썬 환경 — 두 개, 용도가 다름

| | 패키지 | v1.2 로드 |
|---|---|---|
| `.venv` (레포 lockfile) | rslearn 0.0.27 + `olmoearth_pretrain` 0.0.2 | **불가** (M7) |
| `.venv-master` | rslearn 0.1.13 + `olmoearth_pretrain_minimal` 0.0.6 | 가능 |

**릴리스 축 실험은 `.venv-master`에서만.**

### 계산 자원 — 순간 상태와 자산을 분리함

2026-08-28 재확인: GPU0·GPU1 모두 약 34.8 GiB를 쓰는 다른 Python 작업이 있었다. Sen12/
confirmatory/Nepal 프로세스는 아니었고 새 GPU 작업을 시작하지 않았다. 이는 영구 자산 상태가
아니므로 모든 실행 직전 `./bin/nx status`와 `nvidia-smi`를 다시 확인하고 GPU1만 사용한다.

### Nepal OLMoEarth live event — 입력 자산과 차단 상태

이 자산은 Sen12 8-region 확증을 대체하는 headline dataset이 아니라 prospective operations
sidecar다. 웹 화면의 READY 표시는 아래 실물 계약과 항상 일치해야 한다.

| cube | 앵커 | S1/S2 기간 | 파일 / bytes | seal | 용도 |
|---|---:|---:|---:|---|---|
| baseline | 5/5 | 4/4 | 91 / 48,859,900 | valid | pre-event memory |
| placebo A | 5/5 | 4/4 | 91 / 49,313,623 | valid | descriptive control 1 |
| placebo B | 5/5 | 4/4 | 91 / 48,680,635 | valid | descriptive control 2 |
| S2 live | 5/5 | **S1 3/4 · S2 4/4** | 81 / 45,754,625 | **invalid** | 08/27 S2 pixels only; live embedding 금지 |

- live selection preflight는 5/5를 통과했지만 materialization seal은 S1 3/4 때문에 실패했다.
- OLMo v1 baseline/placebo embedding은 **3 mode × 5앵커**가 valid manifest로 봉인됐다.
  `nepal_delta_report.json`의 `live_mode=null`이므로 **event pre/post embedding·delta는 0건**이다.
- placebo 두 개는 95 percentile anomaly를 정의하기에 부족하다. 최소 20개, 권장 30개 이상의
  label-independent historical windows를 동결하기 전에는 descriptive rank만 허용한다.
- AOI 관측성 산출물은 `B02 > 2600 DN` 밝기 진단이며 cloud classifier가 아니다. SCL/CLD
  sidecar 전에는 cloud-free coverage를 주장하지 않는다.
- M17 기존 AP@100 세 수치는 비표준 분모 결함으로 철회했다. P@10과 사전 gate 실패는 유지하며,
  표준 AP@K·Recall@K 재실행은 아직 없다.
- 운영·연구 승격 설계: `docs/NEPAL_EVIDENCE_OPERATIONS_REVIEW_2026_08_28.md`.

## 2. 사슬 6(E_live)용 — 1~5가 닫힌 뒤 씀

| 자산 | 위치 | 규모 | 상태 |
|---|---|---|---|
| GK2A 경량화 산출물 | 로컬 `~/dong/ai_projects/data/gk2a` | **19 MB**, `.json.gz` 144 (2일분) | 매 2일 수집 필요 (2일 보존) |
| GK2A KO/2km 격자 | 같은 곳 `_grid/` | lon·lat 각 ~9 MB, 900×900 | **봉인 보류** — `x0` 해석 불가 (M20) |
| ASOS `era5_10` residual | 로컬 `~/dong/ai_projects/data/asos` | **13 MB**, 3,898행 (주시각 780) | 5/6 변수 커버리지 ~100%, 강수만 99.1% 공백 (M22) |
| ASOS 96지점 좌표 | 같은 곳 | `stn_inf_sfc.txt` | AOI 결합 완료, 중위 17.7 km (M21) |
| data.go.kr GK2A 코드표 | `artifacts/datagokr_gk2a_codes.json` | 10 오퍼레이션 | 전부 검증 |

## 3. 과거 실험 산출물 — 재사용 가능

| 자산 | 위치 | 규모 | 쓸 곳 |
|---|---|---|---|
| 릴리스 감사 216 site-year | 서버 `olmoearth/release_audit_p0` | **26 GB** | M1의 근거. FoldRefresh arm(사슬 밖) |
| mask 경로 실측 | `olmoearth/mask_path_c2a` | 12 KB | M8. 입력계약 근거 |
| HF 캐시 전체 | `.cache/huggingface` | **66 GB** | 다른 프로젝트 것 포함 |

## 4. 저장소

| | 수 |
|---|---|
| 측정 장부 | **M1–M65** (legacy M17 번호 중복은 보존) |
| 코드 파일 | 123 |
| 테스트 파일 / 통과 | 19 / **164 passed, 1 skipped, 10 subtests** |
| artifacts | 19 |
| docs | 8 |
| STUDY 카드 | 54 |
| PR 후보 | 12 |
| 커밋 | 43 |

## 5. 디스크

서버 `/home/work/data` **9.1 T 중 1.4 T 사용 (15%)**, 여유 7.8 T. 용량 제약 없음.

## 6. 없는 것 — 사슬 진행을 막는 것

| 없는 것 | 막는 단계 | 비고 |
|---|---|---|
| **Sen12 전수 task contract** | **0 (C0)** | **통과** — 13,628 readable, retrospective 2건 fail-closed 제외, 10-region LOCO 봉인 |
| **full frozen OLMo probe 결과** | **1 (G-P)** | **완료** — 8-region P4/P2/P3 .2722/.1966/.1834, P4−P2 +.0756, 6/8 win |
| **두 번째 frozen GeoFM full 결과** | **1c (C1)** | Presto 계약·probe는 닫힘. 128×128 cache smoke·seal·matched 3-seed가 미실행 |
| **matched head/baseline 1 run 시간** | **G-C** | fixed 40-epoch: P2 1,491초; E1 tiled small/large 866.6/1,596.2초 + cache 1,130초. practical early-stop·deployment inference 미측정 |
| 71363의 12밴드 물질화 | 2·3의 대안 경로 | v1 2,539 파일은 M35에서 철회. 624개 severe zero hole; <1% zero 1,912도 후보일 뿐. v2 mosaic/coverage contract 필요 |
| Sen12 ↔ 71363 공통 입력 계약 | 2·3 | 양쪽 다 B02–B12 10밴드라 **같은 결측 구조**임 — 유리함 |
| GK2A 격자 대응 | 6 | Area 경로로 우회 가능 |
| 선행강우지수 | 6 | 설계 미착수 (사슬 1~5 전에는 열지 않음) |

## 7. 한 줄 판정

> **Sen12 확증은 8/8 지역으로 닫혔다.** 현재 병목은 **Presto 128×128 cache smoke·matched
> 3-seed control → 그 결과를 포함한 Korea recipe 동결 → 한국 untouched transfer**다. Nepal은
> baseline/placebo만 봉인된 parked sidecar이며 본 경로를 막지 않는다.

---

# 증보 — 2026-08-27 실측

## A. 이번 이틀간 새로 만든 파생 자산

| 자산 | 위치(서버) | 크기 | 상태 |
|---|---|---|---|
| AI-Hub 대응 12밴드 S2 큐브 **v1** | `olmoearth/aihub/s2_12band` | **63.9 GB**, 2,539큐브 | **오염 판정** — 24.6%가 격자 밖 0 채움(M35). v2(모자이크) 재물질화 전까지 사용 금지 |
| OLMo v1 캐시 — 9개 LOCO fold 전부 | `olmoearth/sen12_pilot/holdout_*` | fold당 emb 10.75 GB + raw/mask (chimanimani 실측 36 GB) | prefetch 완료. 확증 sweep의 기반 |
| full-context(1×128) 캐시 | `sen12_pilot_full128` | 10.3 GB | 보유하되 **비채택**(M37·M52) |
| 확증 산출물 8지역 | `olmoearth/confirmatory/` | 72 arm×seed 실행, 지역당 pilot JSON·per-sample·체크포인트·확률맵·코드 스냅샷 | **8/8 post gate PASS**. compact 집계 `artifacts/confirmatory_8region_summary.json` |
| Presto 코드+가중치 | `olmoearth/models/presto` | 3.3 MB, 822K 파라미터 | probe 8/8 통과(M62), upstream commit·code·weight·정규화 byte match 완료. full cache 미실행 |
| 증거 번들 | 로컬 `evidence/` | ~10 MB, 90+ 파일 | per-sample 재계산 검증 통과분만 |

## B. 사용자 제공·신청으로 확보한 원천 (변동 없음 — 08-26 실사 유지)

| 원천 | 상태 | 역할 |
|---|---|---|
| AI-Hub 71363 (산사태 위성영상) | 라벨 GeoJSON + 원천 S2는 **3밴드 RGB**(M28) — 라벨·타일 정의만 사용 | 한국 3-task(E8)의 target |
| KMA APIhub 인증키 + 승인 15건 | 유효 | live residual(E_live) 배관 |
| data.go.kr 키 + GK2A 코드표 10종 | 유효 (키 회전 권고 유지) | GK2A 수집 |
| GK2A 경량 산출물 | 로컬 19 MB, **2일 보존 예외 수집 계속** | 〃 |
| ASOS `era5_10` residual | 로컬 13 MB, 5/6 변수 커버 | 〃 |

## C. 사용 판정 요약

- **지금 쓰는 것**: Sen12 38 GB + OLMo 캐시 9 fold + 8지역 확증 산출물
- **다음**: Presto full cache smoke→matched decoder, 그 뒤 AI-Hub/Korea untouched transfer
- **대기**: Clay(Presto 뒤), AI-Hub 라벨(큐브 v2/입력계약 gate 뒤)
- **폐기·비채택**: s2_12band v1(오염), full-context 캐시(성능 열세), P2_tiny stand-in
- **기상(GK2A·ASOS)**: 아직 성능 기여로 세지 않음 — E_live는 static transfer(사슬 5) 뒤 순서 유지
