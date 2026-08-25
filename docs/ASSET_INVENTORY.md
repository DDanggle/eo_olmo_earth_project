# 보유 자산 — 2026-08-26 실측

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

2026-08-25 재확인: GPU0은 다른 프로젝트가 62,585 MiB를 사용 중이므로 건드리지 않는다.
**GPU1은 0 MiB로 가용**했다. 이는 영구 자산 상태가 아니므로 모든 실행 직전 `nvidia-smi`와
`CUDA_VISIBLE_DEVICES=1`을 다시 확인한다.

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
| 측정 M-항목 | **25** |
| 코드 파일 | 92 |
| 테스트 파일 / 통과 | 19 / **159 passed, 1 skipped, 10 subtests** |
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
| **full frozen OLMo probe 결과** | **1 (G-P)** | strict 개발 pilot은 측정. P4 IoU 0.1416/AP 0.2251. 공식 P2/P3·9 unseen region은 **BLOCKED** |
| **matched head/baseline 1 run 시간** | **G-C** | cached fit+val: P1 387.3초 / P2-tiny 1,455.7초 / P4 950.5초. native strong baseline은 미측정 |
| 71363의 12밴드 물질화 | 2·3의 대안 경로 | C2-C 미실행. v1 + B01·B09 missing mask로 우회 가능 |
| Sen12 ↔ 71363 공통 입력 계약 | 2·3 | 양쪽 다 B02–B12 10밴드라 **같은 결측 구조**임 — 유리함 |
| GK2A 격자 대응 | 6 | Area 경로로 우회 가능 |
| 선행강우지수 | 6 | 설계 미착수 (사슬 1~5 전에는 열지 않음) |

## 7. 한 줄 판정

> **사슬 1~4에 필요한 데이터는 전부 손에 있음** (Sen12 13,628패치 + 한국 2,699쌍 + 체크포인트
> + 동결된 split). C0·cache audit·strict 개발 pilot은 닫혔다. 현재 병목은 **공식 3D U-Net/U-TAE와
> timestamp parity → 미열람 9지역 full G-P**다. 개발 fold 수치는 존재하지만 confirmatory 성능은 0이다.
