# CVPR 큰그림 재감사 — 2026-09-05

범위: 읽기·분석·문서화 + 안전한 상태 점검. GPU 실행·sealed 개봉·보호 4파일 수정 없음.
계획은 `GOAL.md` 2026-09-05 항목. 기준점은 `docs/BIG_PICTURE_2026_09_04.md`와
`MEASURED_FINDINGS.md`(MS-104까지).

> **2026-09-06 갱신**: 진행 중이던 architecture-axis 체인이 56/56으로 완결돼 §1.5·§1.8을
> 최종값으로 갱신했다. 새 GPU job은 시작하지 않았다. 쉬운 큰그림과 과장 금지 문구는
> `docs/PAPER_STATE_2026_09_06.md`를 우선한다.

> **읽는 법**: 이 문서에서 "확인함"은 오늘 내가 원시 산출물을 직접 다시 계산했거나
> 서버 실물을 조회한 것만이다. "장부 인용"은 이전 실행의 기록을 그대로 옮긴 것이며
> 오늘 재검증하지 않았다.

---

## 0. 먼저 — 약점부터

1. **한국·스위스·네팔 공공데이터의 모델 성능 기여는 아직 0이다.** 접근·인벤토리·계약
   감사(M9·M10·M28·M104)는 했지만 어떤 성능표에도 들어가지 않았다. 논문 몸통을
   "한국 shared-cache 3-task"로 잡아둔 상태에서 이건 가장 큰 미결이다.
2. **MS-102의 핵심 결론은 시드 1개다.** 오늘 readout 반론은 닫혔지만 시드 수는 그대로다.
3. **A(릴리스 브리지)는 "동등"에 도달하지 못했고, R5까지 써도 개선이 없었다.** 부록의
   지위는 확정이며 이걸 본문 주장으로 올리면 안 된다.
4. **실행 체인의 성공 로그가 신뢰 불가였다**(아래 §4). 오늘까지의 결과는 독립 감사로
   구제됐지만, 진행 중인 체인은 아직 그 보호가 없다.
5. **한국 cube 전수 파일은 대부분 정상이나 scientific gate는 아직 통과가 아니다.** excluded
   163건 중 `error` 6건은 계약상 재시도 대상인데, 현재 audit와 shard generator가 이를
   deterministic exclusion/done으로 취급한다. split/class selection-bias gate도 남았다.
6. **release migration 저장 gate는 등록 지표와 다르다.** `release_gate_summary.py`가
   `iou_frozen_thr`가 아니라 target-test label을 쓰는 `iou_fp_matched`를 집계한다. AP 복구 결론은
   유지되지만 decision-equivalence count는 registered metric으로 다시 읽어야 한다.
7. **공개 untouched Task-3가 없다.** Sen12와 Solar 두 과업만으로는 PANGAEA, GEO-Bench-2,
   EarthShift 이후의 CVPR main benchmark 규모와 외적 타당성이 부족하다.
8. **운영비의 실제 곡선이 없다.** 기존 FLOPs 계산은 UNet3D 대비 2 task, U-TAE 대비 8~12 task로
   손익분기가 달라진다. Korea 3-task만으로 "cache가 더 싸다"고 말할 수 없다.

---

## 1. 오늘 확인한 것 (first-hand)

### 1.1 요약 수치 ↔ 원시 산출물 일치 — **통과**

`code/audit_summary_vs_raw.py` (신규). `MEASURED_FINDINGS.md` MS-102 표의 macro 값을
`bv1_runs/<cache>/holdout_<fold>_seed1.json`의 `test.positive_patch_macro_iou`에서
8폴드 평균으로 재계산해 대조.

| 캐시 | 재계산 | 장부 | 일치(3자리) |
|---|---:|---:|:--:|
| olmo_cache_pool16 | .2193 | .219 | ✅ |
| clay_cache_native16 | .1547 | .155 | ✅ |
| clay_cache_native16_last | .1098 | .110 | ✅ |
| clay_cache_in256 | .1951 | .195 | ✅ |
| galileo_cache | .1529 | .153 | ✅ |
| prithvi_cache | .0246 | .025 | ✅ |

6/6 일치. 산출물 `artifacts/summary_vs_raw_audit.json`.

> 이 스크립트의 초판에는 **지표 키를 못 찾으면 0건을 검증하고도 "통과"를 출력하는 결함**이
> 있었다. 실제로 첫 실행이 그렇게 나왔다. 수정판은 대조 건수가 0이면 "검증 불능"으로
> 판정한다. 같은 부류의 결함이 §4에도 있다 — 이 프로젝트의 반복되는 실패 양식이다.

### 1.2 Galileo group-concat readout — **MS-102 유지, 반론 종결** (신규 측정)

MS-102-감사가 남긴 마지막 의심("Galileo 토큰을 밴드그룹·시점에 단순 평균한 readout이
정보를 뭉갰다")에 대한 등록 실행이 완료됐다. `logs/bv1_chain.log` → `BV1_CHAIN4_DONE` 09:14Z.

| 캐시 | macro | > raw (.197) | > OlmoEarth (.272) |
|---|---:|---:|---:|
| galileo_cache (평균 readout) | .153 | 1/8 | 0/8 |
| **galileo_cache_groupcat (3840ch concat)** | **.168** | **1/8** | **0/8** |

readout을 고쳐도 raw를 넘지 못한다(+.015). 등록 규칙대로 **MS-102 결론 유지**:
시험 범위 안에서 cache-first는 OlmoEarth 특이 성질이다.

**단, 표현을 정확히 해야 한다.** `clay_cache_in256`는 .195로 raw .197과 사실상 동률이고
4/8에서 raw를 이긴다. 따라서 쓸 수 있는 문장은 "OlmoEarth만 캐시 가치가 있다"가 아니라
**"raw를 확실히 넘는 것은 OlmoEarth뿐이고, Clay는 입력 해상도를 맞춰주면 raw와 동률까지
올라오지만 넘지는 못한다"**이다.

### 1.3 한국 AI-Hub 큐브 v2 전량 — **파일 무결성 통과, scientific gate 미완료**

`logs/aihub_v2_full.log` → `AIHUB_V2_FULL_DONE` 10:14Z.
`aihub/s2_12band_v2/audit_full.json` (444B, sha256 `c8da0a3b125a15bc…`):

```
n_manifest 2536 / n_excluded 163 / n_fail 0 / fails []
coverage_min 0.99978352 / coverage_p05 1.0 / gate_pass true
제외 사유: no_stac_item 148, insufficient_common_coverage 5, error 6, no_candidate_under_cloud_max 4
```

M35 오염(all-band-zero를 0으로 채워 "성공"으로 세던 v1)을 대체하는 2,536개 coverage-valid
후보가 확보됐다. 그러나 `docs/AIHUB_CUBE_V2_CONTRACT.md`는 network/reader `error`를 과학적
제외가 아닌 재시도 대상으로 정의한다. 현재 6건의 `error`가 excluded에 섞였고,
`code/aihub_v2_shards.py`는 reason과 무관하게 excluded key를 모두 done으로 처리한다.
`code/audit_aihub_v2.py`도 error를 deterministic exclusion이라고 설명하면서 `gate_pass=true`를
낸다. 따라서 이 결과의 정확한 상태는 `materialized + coverage_valid 후보`이며,
split/class selection-bias 감사까지 끝나기 전 `experiment_eligible`은 아니다.

### 1.4 제외 163건의 선택 편향 — **시간 편향이 지배적이며 잔여 지리 효과도 있음** (신규 측정)

`code/analyze_aihub_v2_exclusion_bias.py` (신규, 사전 등록 기준 G1–G4를 결과 보기 전에 고정).
산출물 `aihub/s2_12band_v2/exclusion_bias.json`.

| 기준 | 값 | 발동 |
|---|---|:--:|
| G1 전체 date를 잃은 tile | 6개 (`SB1300000000/01/02`, `SB1300010000/01/02`) | 🔴 |
| G2 상위 10% tile이 제외의 50%↑ | 42.3% | ⚪ |
| G3 단일 date가 제외의 30%↑ | `20220824` 57건 = 35.0% | 🔴 |
| G4 split 제외율 격차 3%p↑ | train 5.13% vs valid **13.33%** = 8.2%p | 🔴 |

**진짜 원인은 하나다.** 4개 날짜가 선택된 키를 100% 잃었다:

| date | 제외/선택 | 비율 |
|---|---|---:|
| 20220824 | 57/57 | 1.00 |
| 20220220 | 51/51 | 1.00 |
| 20201229 | 33/33 | 1.00 |
| 20190513 | 12/12 | 1.00 |

합계 153건 = 전체 제외의 **93.9%**, 전부 `no_stac_item` 계열. 주원인은 특정 날짜의 STAC
아이템 부재다. 다만 이로 인해 `SB13*` 인접 6 tile이 통째로 소실되므로 "지리 편향이 없다"고
단정하지 않는다. 정확한 표현은 **원인이 시간 결측에 집중됐지만 결과에는 공간 소실도 남는다**다.
G1·G4는 이것의 파생이다:

- **G1**: 소실된 6개 tile은 전부 `SB13*` 인접 블록(129.07–129.35°E, 36.31–36.47°N,
  경북 동해안), 전부 train, 그리고 **각 tile이 date를 딱 1개만 가졌는데 그게 죽은 날짜**였다.
  594 tile 중 6개(1.0%) 소실.
- **G4**: valid는 date 종류가 13개뿐이라 죽은 날짜 2개를 잃으면 13.3%가 날아간다.
  train은 56개 중 3개라 4.75%다. **date 다양성: train 56→53, valid 13→11 (−15%).**

**결정에 미치는 영향**: 계절성이 있는 task(작물·토지피복)라면 valid의 계절 커버리지가
비례적으로 더 얇아졌다. 한국 3-task 라벨 개봉 **전에** 처리해야 한다(§5의 D1).

### 1.5 architecture-axis 체인 — **추출 7/7, decoder 56/56 완결** (09-06 01:05 KST 갱신)

일반 sandbox의 DNS/SSH 제한 때문에 `nx status/tunnel`이 실패해 한 차례 세션 종료로 오판했다.
권한 있는 터널을 복구하고 직접 SSH로 PID·child·GPU·실물 report를 다시 확인했다.

```
202719 bash code/arch_axes_chain.sh
213594 cache_decoder_train.py --cache clay_in256_half --fold holdout_hokkaido --seed 1
decoder reports 41/56
GPU1 1,307 MiB / 27%        GPU0 0 MiB
```

**09-06 01:05 갱신**: `ARCH_AXES_DONE` 2026-09-05T16:05:15Z. 추출 7/7과 decoder
**56/56**이 모두 끝났다. 강화한 `code/verify_arch_axes.py`가 JSON parse, cache/fold/seed,
primary metric의 유한 범위, embedding shape와 파일 SHA-256을 검사했고 missing/invalid는 0이다.
검증/집계 로컬 해시는 `ca1d7730…` / `405cb800…`이다.

실측 decoder 속도는 폴드당 ~2–4.5분으로 초기 추정(20분)의 1/5–1/10이었다. 초기 ETA "내일 오전"은
철회한다.

**판정 규칙(신설)**: 체인 생사는 `./bin/nx status`만으로 판단하지 않는다. sandbox network와
터널 상태를 구분하고, 터널 복구 뒤 `pgrep` + child process + GPU + log mtime + expected report
count를 함께 본다. 완료는 `code/verify_arch_axes.py`가 실물 56/56을 확인한 뒤에만 선언한다.

보호 4파일은 오늘 3회 push 전후로 sha256·mtime 불변 확인 (규약 4c):
`pilot_sen12_gp_heads 1fb3fd66…/1787715166`, `sen12_official_baselines 19232b2a…/1787683824`,
`extract_sen12_fold_cache 197895a0…/1787659265`, `audit_sen12_fold_cache 4287dbcd…/1787668636`.

### 1.8 architecture-axis 결과 — **cache/raw 경계는 scale·family에 조건부이고 parameter count만으로 설명되지 않는다** (신규 측정)

`code/arch_axes_summary.py`(신규) → `artifacts/arch_axes_summary.json`. 시드 1, 동일 8폴드,
동일 디코더·지표. 기준선은 봉인 pilot OlmoEarth P4 `.272`, raw P2 `.197`.

| 캐시 | family | 축 | n | macro | full 대비 | >raw | >olmo |
|---|---|---|---:|---:|---:|---:|---:|
| olmo_tiny | OlmoEarth | scale=tiny | 8 | **.228** | −.044 | 5/8 | 2/8 |
| olmo_base_half | OlmoEarth | depth=50% | 8 | **.221** | −.051 | 5/8 | 2/8 |
| olmo_nano | OlmoEarth | scale=nano | 8 | **.194** | −.078 | 3/8 | 1/8 |
| clay_in256_half | Clay | depth=50% | 8 | .124 | −.071 | 1/8 | 0/8 |
| galileo_tiny | Galileo | scale=tiny | 8 | .129 | −.024 | 0/8 | 0/8 |
| galileo_nano | Galileo | scale=nano | 8 | .112 | −.041 | 0/8 | 0/8 |
| galileo_base_half | Galileo | depth=50% | 8 | .138 | −.015 | 0/8 | 0/8 |

읽기 — 세 가지가 동시에 나온다.

1. **관측된 scale 경계다.** OlmoEarth cache 성능은 규모·깊이를 줄인 설정에서 낮아지고
   **nano에서 raw 아래로 내려간다(.194 < .197, >raw 3/8)**. 사전 6/8 gate를 안정적으로 넘은 것은
   full base뿐이다. 단, `.003` 차이·과업 1개·시드 1개이므로 이를 보편적인 **용량 필요조건**이라
   부르지 않는다. cache/raw 행동이 달라지는 경험적 경계가 관측됐다고 쓴다.
2. **parameter count만으로 설명되지 않는다.** Galileo는 어떤 규모에서도 raw를 **0/8**로 못 넘고,
   full base(.153)가 OlmoEarth **nano**(.194)보다도 낮다. 용량을 맞춰줘도 family 격차가 남는다.
   → MS-102의 "OlmoEarth 특이성"은 **용량 교란으로 설명되지 않는다**. 이것이 오늘의 핵심 소득이다.
3. **깊이 절반은 Clay에서만 파괴적이다**(−.071, .195→.124). OlmoEarth는 −.051로 버틴다.
   중간층 표현의 안정성이 family마다 다르다.

**왜 이 축이 선행연구에 먹히지 않는가.** 동시 세션 보고서는
[How to Embed Matters](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Gilch_How_to_Embed_Matters_Evaluation_of_EO_Embedding_Design_Choices_CVPRW_2026_paper.html)가
pooling·backbone·depth를 점유했다고 적었고, 그 판단 자체는 옳다. 그러나 그 논문이 재는 것은
**임베딩 품질의 설계 선택**이고, 여기서 재는 것은 **raw 재학습 대비 캐시 재사용의 손익**이다.
산출물이 "어떤 설계가 좋은가"가 아니라 **"어느 지점 아래로는 캐시하면 손해인가"**라는
결정 경계다. 그리고 그 경계가 실제로 관측됐다 — OlmoEarth nano가 그 선이다.

**EarthCache에서의 자리**: 모델 규모·깊이는 **라벨 없이 읽히는 계약 필드**다. 따라서 D3의
label-free 판별자에 넣을 **candidate feature**가 된다. 그러나 이 표는 feature 선택의 근거일 뿐,
독립 과업에서 cache 가치를 예측한 결과가 아니므로 predictor 검증으로 세지 않는다.

**말할 수 없는 것**: 시드 1. nano/tiny는 사전학습 체크포인트가 다른 것이지 동일 모델의
절단이 아니므로 "규모"는
사전학습 예산과 교락돼 있다. depth=50%만이 동일 가중치 내 절단이다.

마지막 Galileo base-half/Thrissur fold 도중 다른 GPU 1 작업이 약 72 GiB를 사용하기 시작했다.
metric report는 정상 완결됐지만 해당 fold의 wall-clock은 비용 근거로 쓰지 않는다.

---

## 2. 주장의 3단 분류

### 2.1 확증된 것 (봉인 또는 사전등록 게이트 통과)

| 주장 | 근거 | 강도 |
|---|---|---|
| 릴리스가 바뀌면 임베딩 좌표계 identity가 완전히 깨진다 | M1 (8 gate 전부 실패, R@1 0.0000 양방향) | 강 |
| 관계구조는 남는다 (CKA .979, 거리 Spearman .953) — 그러나 task 연속성 증거가 아니다 | M1 | 강 |
| 라벨 없는 선형 브리지가 붕괴를 대부분 복구한다 (AP .015 → .901, R0의 97.6%) | MS-100-확증 8폴드 | 강 |
| 그러나 옛 head와 **동등**은 못 된다 (등록 frozen-threshold gate: v1.2 R3/R4/R5=2/8·3/8·3/8) | 9/5 재감사 | 강 |
| 공간 스티치(R5)를 더해도 개선이 없다 (.901 = .901) | MS-103 | 강 |
| 릴리스 거리가 짧아도(v1→v1.1) 붕괴·복구 폭이 같다 | MS-103 | 중 |
| frozen 캐시 + 디코더가 raw 재학습을 이긴다 (Sen12 8지역, Solar 8폴드) | M65, MS-96/97/98/99 | 강 |
| K=5/20 few-shot에서 A1 > A4w는 두 task 모두 8/8; A1>A4h는 Sen12 7/8, Solar 8/8 | MS-96/97/98/99 | 강 |
| **단** K=5에서 A1 > A0는 5/8뿐 — "라벨 5장이면 항상 적응"은 금지 | MS-96 정정 | 강 |
| raw를 확실히 넘는 캐시는 시험 4 family 중 OlmoEarth뿐 | MS-102 + 오늘 §1.2 | 중(시드1) |
| Clay는 입력 256px로 맞추면 raw와 동률(.195 vs .197)까지 온다 | MS-102, 오늘 재계산 | 중(시드1) |
| Galileo의 열세는 readout 인공물이 아니다 | **오늘 §1.2** | 중(시드1) |
| 한국 큐브 v2 2,536개가 shape/dtype/coverage 파일 감사를 통과했다 | **오늘 §1.3** | 강 |
| v2 제외는 4개 날짜 STAC 결측이 지배(153/163), 그 결과 6 spatial tile도 소실 | **오늘 §1.4** | 강 |
| 시험한 OlmoEarth scale series에서 full base만 6/8 gate를 안정적으로 넘고 nano는 raw 경계 아래(.194 < .197) | **오늘 §1.8** | 중(시드1) |
| Galileo는 어떤 규모에서도 raw를 0/8로 못 넘고 full base(.153) < OlmoEarth nano(.194) — parameter count만으로 family 차이를 설명하지 못함 | **오늘 §1.8** | 중(시드1) |

### 2.2 아직 가설인 것

| 가설 | 왜 아직 가설인가 | 닫는 방법 |
|---|---|---|
| 한국 공공데이터가 표현을 개선한다 (OlmoEarth-KR) | 성능 기여 측정 **0** | 3-task 개봉 (§5 D2) |
| support와 contract만으로 최소 충분 행동을 판별할 수 있다 | C0-dev뿐, untouched 검증 없음 | 공개 Task-3 action regret |
| 하나의 캐시로 세 head가 동시에 선다 (shared-cache) | 미실행 | §5 D2 |
| OlmoEarth 우위의 **원인**이 사전학습 데이터·목적이다 | §1.8이 parameter-count-only 설명을 약화했으나 데이터·목적은 아직 분리 안 됨 | 동일 데이터·다른 목적 사전학습(비쌈) 또는 관찰적 논증 |
| 모델 규모·깊이를 포함한 contract feature로 캐시 가치를 **라벨 없이** 예측할 수 있다 | §1.8은 feature 후보를 줬을 뿐 독립 과업 예측은 0회 | D3 판별자에 투입 후 untouched 평가 |
| Sen12 → 한국 → Nepal head 전이 | 미실행 | 선택 |
| 정책 C가 support+계약만으로 행동을 고른다 | C0-dev 개발 화면뿐 | 부록 |

### 2.3 CVPR 본회의를 막는 결함

| # | 결함 | 왜 치명적인가 | 상태 |
|---|---|---|---|
| **F1** | 논문 몸통(한국 3-task)에 결과가 0 | 심사자가 볼 "본문"이 없음. 현재 있는 건 전부 부록 재료 | 🔴 |
| **F2** | MS-102 시드 1 | "어느 표현이 캐시 가치가 있나"가 핵심 열인데 근거가 단일 시드 | 🟡 |
| **F3** | Solar 폴드가 독립 지역이 아니라 UTM-zone 그룹 | Sen12의 "8지역"과 같은 일반화 단위로 부르면 오류 — 이미 장부에 명시됨, 원고에서 지켜야 | 🟡 |
| **F4** | 벤치마크를 표방하는데 공개 task가 Sen12·Solar 2개 | GEO-Bench-2(19), PANGAEA와 나란히 놓일 때 규모가 약함 | 🟡 |
| **F5** | 비용 축이 **FLOPs 모형(M38)** 과 릴리스 처리량 1건(M1 1.67×)뿐 — 실측 cold/warm/re-embed/저장 곡선 없음 | "비용으로 결정한다"가 논문 한 문장. M38은 forward FLOPs만 세고 backward를 일괄 2배로 가정했으며 손익분기가 baseline 선택에 의존한다(UNet3D 대비 2 task, U-TAE 대비 8–12 task) | 🟡 |
| **F6** | Korea error 6건·selection-bias gate·희소 class 통계 단위 미해결 | label 개봉 전 계약 위반 위험 | 🔴 |
| **F7** | release summary가 target-test-dependent IoU를 사용 | preregistration 불일치, decision-equivalence 과대/왜곡 | 🔴 |
| **F8** | 공개 untouched Task-3 없음 | action selector와 외적 타당성을 독립 검증할 곳이 없음 | 🔴 |

---

## 3. 제출 가능한 한 문장 — 최종 권장

> **저장된 Earth embedding의 가치는 정적인 평균 정확도가 아니라, 새 지역·새 과업·새 릴리스에서
> 제한된 support label과 데이터 계약만으로 REUSE/ADAPT/RE-EMBED/REQUEST 중 가장 싼 안전 행동을
> 선택했을 때의 utility–harm–cost regret로 평가해야 한다.**

현재 결과는 이 질문의 필요성을 지지하지만, selector가 unseen unit에서 작동한다는 부분은 아직
빈칸이다. F1/F8을 못 닫을 때의 후퇴선은 **release migration characterization + 두 task cache
reuse boundary**이며, 그 경우 CVPR main에 충분하다고 미리 단정하지 않는다.

---

## 4. 실행 인프라 결함 — `rc=$?`

`arch_axes_chain.sh`와 `aihub_v2_full_chain.sh` 양쪽에 동일한 패턴이 있다:

```bash
run code/extract_... ; echo "$(date -u +%FT%TZ) name rc=$?" >> $LOG
```

`$(date)`가 먼저 실행돼 `$?`를 덮어쓴다. 로컬 확인:

```
$ bash -c 'false; echo "$(date -u +%FT%TZ) test rc=$?"'
2026-09-05T11:11:40Z test rc=0      ← false인데 rc=0
```

따라서 `logs/arch_axes.log`와 `logs/aihub_v2_full.log`의 **모든 `rc=0`은 `date`의 종료코드**다.

- **오늘까지의 결과는 무사하다.** 추출은 로그의 `all_gates_pass` JSON으로, 한국 v2는
  `audit_aihub_v2.py`의 독립 전수 감사로 실물 검증됐다. rc에 의존한 판단이 없었다.
- **위험은 앞으로다.** 디코더 56회에는 독립 감사기가 없어서, 크래시해도 `rc=0`만 남는다.
- **조치**: 돌아가는 스크립트는 손대지 않는다(bash는 실행 중 파일을 바이트 오프셋으로
  이어 읽으므로 편집하면 실행이 깨진다). 대신 `code/verify_arch_axes.py`로 실물을 센다.
  스크립트의 `rc=$?` → `rc=$?; ... "rc=$rc"` 수정은 **다음 실행분부터** 적용한다.

**계보**: 이건 §1.1의 "0건 검증하고 통과 출력"과 같은 부류다 — *검증이 아무것도 안 해도
성공으로 보이는 형태*. 앞으로 체인·감사 스크립트는 "무엇을 몇 건 검사했는가"를
반드시 출력하고, 0건이면 통과라고 말하지 않는다. (`L3` 실패 계보에 등재)

---

## 5. 가장 작은 결정 실험 (D1–D4)

가장 적은 GPU로 §2.3의 결함을 닫는 순서.

### D1 — Korea label 개봉 전 data gate 복구 (GPU 0)
`20220824 / 20220220 / 20201229 / 20190513`을 다른 STAC 엔드포인트 또는 넓은 시간창으로
재질의한다.
- **회수되면**: 163건 중 153건이 살아나고 G1·G4가 동시에 해소된다. valid date 다양성 13종 복구.
- **회수 안 되면**: 결정론적 결측으로 확정하고 gate 6에 그대로 보고. `SB13*` 6 tile 소실과
  valid −15% date 다양성을 논문 limitation에 명시.
- **판정 기준(사전)**: 4개 날짜 중 2개 이상이 회수되면 재물질화, 아니면 현 상태 동결.

별도로 `reason=error` 6건은 반드시 재시도한다. 회수 실패 시 network/reader terminal failure로
분리하고 raw exception provenance를 보존한다. audit는 `error>0`이면 fail해야 하며, shard todo는
error key를 done으로 세지 않는다. 그 뒤 split/class exclusion rate를 다시 계산한다.

### D2 — 한국 shared-cache 3-task 개봉 (F1을 닫는 유일한 실험)
전제: D1 종결 + 이미 `6f47156`에 커밋된 `config/korea_shared_cache_3task_prereg_v0.json`의
dated amendment. amendment는 (a) 128-chip unit, (b) rare-class eligible cluster macro,
(c) negative-only FPR, (d) T1/T2에는 O0가 없음을 결과 개봉 전에 고정한다.
실험은 O0/O1/R0/R1/FULL × 해당 task × K=5/20 × 3시드다.
**이것이 임계경로다.** 다른 모든 실험은 이것에 양보한다.

### D3 — 공개 untouched Task-3 + support-only 행동 선택 (F8을 닫음)
Sen1Floods11을 우선 후보로 contract audit하고, label을 보기 전에 split·threshold·K=0/5/20·action
rule을 고정한다. Sen12/Solar는 selector 개발에만 쓰고 Task-3는 한 번만 연다.

최소 selector는 (1) contract fail-closed, (2) positive support 0이면 A1 금지,
(3) leave-one-support-out A1-A0 하한이 양수일 때만 A1, (4) 불안정하면 A0/REQUEST로 둔다.
always-reuse/adapt/reembed와 oracle 대비 utility regret, harm rate, abstention coverage, cost를 측정한다.

토큰 geometry로 cache 성능을 label-free 예측하는 D3 구상은 14개 cache가 같은 데이터/decoder에서
파생돼 독립 표본이 아니므로 headline으로 쓰지 않는다. exploratory appendix로만 허용한다.

### D4 — A3 ceiling·MS-102 시드 2·3·실제 비용 (F2/F5)
7 캐시 × 8 폴드 × 2 시드 = 112회. 구조 축 체인(56회)의 뒤에 붙이면 GPU ~1.5일.
그 전에 exposed unit에서 공식 LayerDecayAdamW 또는 q/v LoRA의 encoder PEFT+re-embed ceiling을
측정한다. cold/warm/raw read/cache write/re-embed/storage/latency도 함께 기록한다.
**D2/D3보다 뒤다.** F1/F8이 F2보다 치명적이다.

---

## 6. 장부에 등재할 항목 (MEASURED_FINDINGS.md)

- **MS-105** — Galileo group-concat readout .168, raw 미달 1/8. MS-102 유지, readout 반론 종결.
- **M104-전량** — 한국 큐브 v2 2,536칩, 파일 shape/dtype/coverage fail 0, 제외 163 (6.0%).
  단 error 6건과 selection-bias gate 때문에 experiment eligibility는 미완료.
- **M106** — v2 제외 편향: 시간(4개 date 100% 결측 = 93.9%)이 지배하며,
  파생 공간 소실도 있음.
  파생 피해 = `SB13*` 6 tile 소실 + valid date 다양성 −15%.
- **감사** — MS-102 표 6/6이 원시 산출물에서 재계산 일치.

---

## 7. 실행 순서 (CVPR 2027 본문 마감 11/16 AoE)

| 주 | 일 | 중단 판정 |
|---|---|---|
| W1 (9/5–9/11) | D1: error 재시도·date 재조회·selection-bias·Korea metric amendment. release gate 재생성. Solar cross-CRS 감사 | P0가 안 닫히면 해당 결과 headline 보류 |
| W2 (9/12–9/18) | **D3 공개 Task-3** contract/prereg/cache/raw/A0/A1 + exposed A3 ceiling | Task-3가 모두 음성이면 범용 cache 주장 중단 |
| W3–W4 (9/19–10/2) | **D2 Korea 3-task 1회 개봉** + cold/warm/re-embed/storage 비용 | rare class unit가 불충분하면 그 task는 사례로 하향 |
| W5 (10/3–10/9) | action selector untouched 평가, CVPR go/no-go | 공개 3-task/action/cost 중 2개 미완료면 main 강행 금지 |
| W6–W8 (10/10–10/31) | D4 필요한 시드만 확장, Figure/Table·본문·artifact dry-run | 10/20 이후 headline 실험 추가 금지 |
| 제출 (11/10·11/16·11/23) | 등록·본문·supplement | 공식 일정 준수 |

---

## 8. Figure/Table 설계 (동결 후보)

| # | 내용 | 데이터 상태 |
|---|---|---|
| F1 | 릴리스 전환의 붕괴·복구: R1 identity .015 → R3/R4 .90 → R0 .925, 8폴드 | ✅ 있음 |
| F2 | 그러나 동등은 아니다: registered frozen-threshold gate 2/8~3/8, 잔여 decision gap | ✅ 재감사 있음 |
| T1 | 캐시 가치의 비보편성: 7 캐시 × 8 폴드 macro IoU vs raw/OlmoEarth | ✅ 있음(+오늘 groupcat) |
| T2 | few-shot 결정표: A0/A1/A4w/A4h × K=0/5/20 × 2 과업 | ✅ 있음 |
| F3 | untouched Task-3 action regret/harm/coverage/cost | 🔴 D3 — 없음 |
| T3 | **한국 3-task shared-cache** | 🔴 D2 — 없음 |
| T4 | 비용 축: 재임베딩 GPU·I/O·인덱스 | 🟡 M1 1.67× 1건뿐 |

---

## 9. 이번 감사에서 하지 않은 것

- sealed Korea / Task-3 라벨 개봉 — 안 함.
- GPU 실행 — 안 함(모든 신규 스크립트는 CPU 읽기 전용).
- 보호 4파일 수정 — 안 함(해시·mtime 불변 확인).
- CVPR 공식 일정·선행연구 — 후속 통합 감사에서 공식 CVPR·CVF·원 논문을 재확인했고 §10에 반영.
- Solar 폴드 독립성 — `task2_contract`와 audit를 재확인했으며 58 UTM zone을 8 group으로 묶은
  단위다. same-CRS overlap 201건은 merge했지만 WGS84 cross-CRS 감사 산출물은 아직 없다.

---

## 10. 2026-09-05 최신 문헌과 정확히 겹쳐보기

CVPR 2027 공식 일정은 paper registration 2026-11-10, submission 11-16, supplement 11-23
AoE이며, 본회의는 2027-06-22~25 Seattle이다. 오늘부터 본문까지 72일이다.
[CVPR 2027 공식 일정](https://cvpr.thecvf.com/Conferences/2027/Dates)

OlmoEarth 공식 저장소 기준 최신 공개 family는 v1.2이며 Nano/Tiny/Small/Base가 있다. v1.2 Base는
114M encoder이고 v1·v1.1·v1.2가 별도 weight/script로 유지된다.
[OlmoEarth 공식 저장소](https://github.com/allenai/olmoearth_pretrain)

최신 문헌은 우리 아이디어를 없애기보다 **평범한 주장들을 제거**한다.

| 선행연구 | 이미 점유한 질문 | 이 프로젝트의 남은 질문 |
|---|---|---|
| [OlmoEarth, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Herzog_OlmoEarth_Stable_Latent_Image_Modeling_for_Multimodal_Earth_Observation_CVPR_2026_paper.html) | 대규모 task에서 강한 multimodal EO foundation model | 새 모델 품질이 아니라 저장된 결과의 재사용 수명 |
| [PANGAEA](https://arxiv.org/abs/2412.04204) | supervised baseline·limited-label을 포함한 GFM benchmark | K=5/20만으로 novelty 주장 금지; deployment action으로 이동 |
| [GEO-Bench-2](https://arxiv.org/abs/2511.15658) | 19 permissive dataset, capability-aware model ranking | static ranking이 아닌 cache/action/cost |
| [EarthShift](https://earthshift.github.io/) | 8 GFM·11 task·5 real shift, 약 20% OOD 하락 | shift 존재 확인이 아니라 shift에서 최소 충분 행동 선택 |
| [Earth Embeddings](https://arxiv.org/abs/2608.03410) | embedding 유형·저장·압축·pooling·spatial transfer | embedding 제품을 언제 유지·폐기·갱신하는가 |
| [Earth Embeddings as Products](https://arxiv.org/abs/2601.13134) | 제품 taxonomy와 통합 loader | API/표준화가 아니라 lifecycle decision |
| [How to Embed Matters](https://openaccess.thecvf.com/content/CVPR2026W/EarthVision/html/Gilch_How_to_Embed_Matters_Evaluation_of_EO_Embedding_Design_Choices_CVPRW_2026_paper.html) | backbone·objective·depth·pooling·representation 결합 | 현재 architecture-axis는 ablation일 뿐 headline 아님 |
| [Better Together](https://arxiv.org/abs/2605.18667) | 4 Earth embedding의 aligned fusion, 6 task 중 4개 개선 | 단순 concat/multi-model fusion은 novelty 아님 |
| [TESSERA v2](https://arxiv.org/abs/2607.03949) | scaling, distillation, Matryoshka; 16d가 128d 성능 92% | 차원 압축보다 utility 유지와 행동 결정 |
| [Model Stitching, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Mai_Revisiting_Model_Stitching_In_the_Foundation_Model_Era_CVPR_2026_paper.html) | heterogeneous VFM feature stitching | bridge 자체보다 EO contract와 fixed-decision 보존 |
| [BCT, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Towards_Backward-Compatible_Representation_Learning_CVPR_2020_paper.html) | backfill-free old/new embedding compatibility | dense EO cache의 band/time/GSD/release 비용 경계 |
| [XBT, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Jang_Towards_Cross-modal_Backward-compatible_Representation_Learning_for_Vision-Language_Models_ICCV_2025_paper.html) | VLM cross-modal backward compatibility | release bridge를 main method로 과장하지 않음 |
| [Brewing Stronger Features, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wolf_Brewing_Stronger_Features_Dual-Teacher_Distillation_for_Multispectral_Earth_Observation_CVPR_2026_paper.html) | multispectral dual-teacher distillation | 다른 backbone teacher 전이는 후속축 |
| [CrossEarth-Gate, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_CrossEarth-Gate_Fisher-Guided_Adaptive_Tuning_Engine_for_Efficient_Adaptation_of_Cross-Domain_CVPR_2026_paper.html) | cross-domain EO parameter-efficient tuning | PEFT는 A3 arm/ceiling이지 단독 기여 아님 |

그러므로 아래 여섯 문장을 논문의 novelty로 쓰지 않는다.

1. 한국 지도와 위성 영상을 결합했다.
2. 다른 foundation model 임베딩을 concat했다.
3. teacher signal을 student에 증류했다.
4. embedding 차원을 줄였다.
5. 지역 이동에서 성능이 떨어졌다.
6. 릴리스 좌표계를 linear map으로 맞췄다.

이 요소들은 모두 action space나 ablation으로는 쓸 수 있다. 주기여는 **이미 cache가 존재하는
배포자가 다음 계산/라벨 행동을 어떻게 결정하는가**다.

## 11. 최종 논문 형태

### 11.1 권장 제목

> **EarthCache: When to Reuse, Adapt, or Recompute Geospatial Foundation-Model Embeddings**

대안:

> **One Cache, Many Earth Tasks: Safe Reuse under Geographic, Task, and Model Shift**

### 11.2 RQ

**RQ1 — Cache value.** 같은 frozen Earth representation이 서로 다른 task와 새 geography에서 strong
raw baseline 대비 언제 가치를 유지하는가?

**RQ2 — Minimum sufficient action.** K=0/5/20 support와 contract metadata만 보고 REUSE,
ADAPT, PEFT/RE-EMBED, RAW, REQUEST 중 가장 싼 안전 행동을 고를 수 있는가?

**RQ3 — Lifecycle robustness.** release가 바뀌거나 local public context가 추가될 때 기존 cache와
결정은 얼마나 유지되며 어느 시점에 bridge보다 재계산이 낫나?

RQ1·RQ2가 main이고 RQ3는 Solar release migration과 Korea context의 제한된 case study다.

### 11.3 기여 세 개

1. **EarthCache protocol**: world/task/model shift에서 cache의 reuse/adapt/recompute를 accuracy,
   harm, labels, FLOPs, bytes, latency로 같은 action table에서 비교한다.
2. **Support-only safe action**: contract·positive coverage·leave-one-support-out stability로 test label
   없이 최소 행동을 고르고 oracle action 대비 regret을 측정한다.
3. **실증**: Sen12·Solar·공개 Task-3와 제한된 한국 one-cache/three-task 외부 사례에서 positive
   transfer, negative transfer, support failure, release failure를 함께 공개한다.

새 neural block이 반드시 필요한 것은 아니다. 단순 규칙도 untouched Task-3에서 harmful update를
줄이고 비용을 절약하면 method contribution이 된다. 반대로 규칙이 실패하면 새 모델 이름을 붙이지
않고 benchmark/characterization paper로 하향한다.

## 12. 실험 계약 — 논문 표가 흔들리지 않게

### 12.1 데이터 역할

| 역할 | 데이터 | 현재 상태 | 최종 역할 |
|---|---|---|---|
| 확증 1 | Sen12Landslides | 3 seed 완료 | geographic shift와 hazard |
| 확증 2 | Solar Farm | 3 seed 완료, cross-CRS P0 | task replication과 empty-support failure |
| untouched 공개 | Sen1Floods11 우선 | 미실행 | action selector first-look |
| 공개 대안 | PASTIS-R | 감사 후보 | 시간이 허용될 때 multi-class/multimodal |
| 제한 외부 | AI-Hub 71363 Korea | 2,536 cube 후보, gate 미완료 | one cache/3 task, Sen12→Korea |
| model shift | Solar v1→v1.1/v1.2 | 실행, summary protocol 수정 필요 | appendix lifecycle case |

Sen1Floods11 우선은 결과를 보지 않고 data size, task 적합성, 72일 기한으로 정한다. contract audit에서
치명적 leakage/label 문제가 나오면 그 근거를 기록하고 PASTIS-R로 한 번만 전환한다.

### 12.2 action arms

| Arm | 모델 행동 | 측정할 비용 |
|---|---|---|
| A0 REUSE | old/frozen cache + source head | cache read + head inference |
| A1 ADAPT | cache 고정, head만 K support 적응 | label + cache read + head train |
| A3 PEFT | encoder를 조금 조정하고 target 재임베딩 | raw read + encoder fwd/bwd + 새 cache |
| A4 RAW | strong raw model 학습/적응 | raw read + full train |
| A5 REQUEST | 보류하고 label 추가 | annotation count/time proxy |
| MIGRATE | new release를 old cache/head contract로 bridge | paired anchor + fit/infer + backfill bytes |

A3는 공식 LayerDecayAdamW 또는 q/v LoRA를 exposed unit에서 먼저 평가한다. A3가 없으면 리뷰어의
“encoder를 조금만 고치면 되는 것 아닌가?”에 답하지 못한다.

### 12.3 공정성 규칙

1. 모든 arm은 같은 geometry와 support IDs를 쓴다.
2. threshold는 source-val/target-val/support-only에서 고정하고 test label로 맞추지 않는다.
3. trainable parameters, update 수, observed examples, raw bytes를 각각 공개한다.
4. official raw, parameter-matched raw, PEFT/full-FT ceiling을 분리한다.
5. band/GSD/time/pooling/release가 다른 model은 common-physical과 native-product 표를 분리한다.
6. diagnostic 1 seed는 confirmatory 3 seed 표에 섞지 않는다.
7. target test region/cluster는 selector나 hyperparameter를 고르는 데 사용하지 않는다.
8. raw가 비싼 상황과 raw 접근 자체가 불가능한 상황을 분리한다.

### 12.4 지표

| 축 | primary | secondary |
|---|---|---|
| dense utility | positive-unit macro IoU | tie-correct AP, calibration |
| safety | A0보다 악화한 unit의 비율(harm rate) | worst-unit gap |
| decision | oracle action 대비 utility–cost regret | action confusion matrix |
| abstention | risk–coverage | REQUEST count |
| labels | K=0/5/20 curve | positive support count/effective size |
| compute | cold/warm GPU seconds + FLOPs | peak VRAM |
| movement | raw bytes read + cache bytes read/write | archive backfill bytes |
| release | AP/IoU retention at frozen threshold | R@1 vs gallery N |

### 12.5 통계 단위

- pixel을 독립 표본처럼 세지 않는다. Sen12는 region, Solar는 UTM group, Korea는 spatial cluster다.
- per-unit paired difference 뒤 region→seed hierarchical bootstrap을 사용한다.
- final arm은 최소 3 seed다.
- 평균, worst unit, negative-transfer count를 함께 보고한다.
- positive가 없는 cluster는 IoU에 억지로 0을 넣지 않는다. positive metric과 negative-only FPR을
  분리한다.
- A1의 순수 적응 이득은 `A1-A0`, raw 이득은 `A4-A4w0`로 보고한다.

Korea test의 rare label 분포는 특히 중요하다.

- deforestation positives: C02/C05/C06/C08/C12, 즉 5/7 cluster
- landslide positives: C05/C08/C12, 즉 3/7 cluster

현재 prereg의 T3 `O0>=R0 in 5/7`은 positive-tile metric이라면 구조적으로 불가능하거나 undefined다.
label을 열기 전에 **class-eligible cluster macro + negative-only FPR**로 amendment한다.

## 13. Support-only selector의 최소 모양

복잡한 meta-learning부터 시작하지 않는다.

```text
contract 위반? ── yes → RE-EMBED 또는 REQUEST
       no
support positive=0? ── yes → REUSE 또는 REQUEST
       no
leave-one-support-out A1-A0 하한 > 0? ── yes → ADAPT
       no
A0 risk 허용? ── yes → REUSE
       no  → PEFT/RAW/REQUEST 중 비용 최소
```

개발 task에서 동결한 뒤 untouched Task-3와 Korea에 한 번 적용한다. 성공 기준 후보는:

- always-adapt 대비 harm rate를 절반 이하로 감소
- oracle action utility gap의 70% 이상 회복
- always-PEFT/raw 대비 GPU 또는 raw-I/O cost 50% 이상 절감
- abstention을 숨기지 않고 coverage와 함께 보고

이 숫자는 Task-3 test label을 보기 전에 개발 결과와 statistical power로 확정한다. 결과 후 바꾸면
dated amendment로 공개하고 primary에는 쓰지 않는다.

## 14. 비용 연구 — 반드시 main figure로

M38의 현재 forward FLOPs/sample:

| 구성 | GFLOP/sample |
|---|---:|
| P4 small decoder | 2.01 |
| U-TAE | 38.72 |
| UNet3D | 270.76 |
| OlmoEarth encoder 1×128 | 26,454.32 |

기존 학습 배율에서 cache 손익분기는 UNet3D 대비 2 task, U-TAE 대비 8~12 task다. 즉 “one cache,
three tasks”는 비싼 raw baseline에는 경제적이지만 싼 U-TAE에는 아직 아니다.

최종 비용 곡선은 네 시나리오를 분리한다.

1. **cold**: raw download/read + encoder + cache write + head train
2. **warm**: cache read + head train
3. **release**: archive N개 re-embedding vs bridge vs dual-index
4. **raw unavailable**: 원본 삭제/권한 부재로 re-embedding 자체가 불가능

gallery/archive 크기 N과 task 수 K를 축으로 실제 교차점을 그린다. 현실적 N/K에서 cache 또는
bridge가 이기지 않으면 비용 주장을 중단한다. wall-clock은 전용 GPU 구간에서만 재고, 기존 공유 GPU
벽시계는 오염된 값으로 유지한다.

## 15. 한국 공공데이터·실시간 API의 정확한 자리

한국형 Earth Intelligence는 세 층으로 분리한다.

| 층 | 예 | 역할 | CVPR 2027 |
|---|---|---|---|
| stable visual bus | Sentinel-1/2, Landsat, DEM | 여러 task가 공유하는 cache | main |
| privileged training context | 지적도·토지피복·건축·환경·기상 이력 | EO-only representation을 지도 | 작은 ablation 또는 후속 |
| live evidence residual | GK2A·허가·건축·환경 API | 최신성·원인 근거·보류 | demo/appendix |

“API를 붙이면 embedding이 좋아진다”는 아직 미측정이다. 가장 작은 유효 실험은 입력 EO를 고정하고
다음을 비교하는 것이다.

1. EO only
2. location/year only
3. cutoff-valid static public context
4. cutoff-valid dynamic context
5. shuffled context
6. missing/stale context

train-time privileged signal과 inference-time fusion을 분리하고 `event_time / observed_time /
published_time / retrieved_time`을 기록한다. 이 표가 없으면 public API는 논문 abstract가 아니라
제품 evidence dashboard에 둔다.

현재 제주 오름의 행정 근거 0/368은 모델 성능 문제가 아니라 (a) 초기 변화후보의 시간창/구름 오염,
(b) 보전지역의 구조적 행정 사건 부재, (c) 원천자료의 연도 누락이 겹친 것이다. 오름을 원인규명
benchmark로 밀지 않고, 선택적 변화탐지와 abstention의 실패 사례로 보존한다.

## 16. Alps·Ostana/Monviso·ICIMOD·VLM은 어떻게 이어지는가

제주, ETH 인근 Alps/Monviso, Hindu Kush–Himalaya는 서로 다른 snow/cloud/terrain/sensor/public-record
shift를 제공한다. 장기 프로그램은 공통 EO bus와 지역별 residual로 구성할 수 있다.

```text
공통 EO cache
  ├─ 제주: 오름·산림·개발·토지피복·GK2A
  ├─ Alps/Monviso: snow·rock·vegetation·field observations
  └─ ICIMOD/HKH: glacier lake·landslide·snow/ice·hazard inventory
                ↓
       지역별 head + live residual + abstention
```

하지만 72일 안에 세 지역을 다 열면 데이터 계약만 만들다가 끝난다. CVPR 2027에는 Korea 한 곳만
external case로, 공개 Task-3를 재현 축으로 둔다. Alps/ICIMOD는 같은 protocol을 재사용할 후속 외부
검증이다.

VLM은 기술적으로 가능하다. 적절한 역할은 후보 검수, map legend/행정 문서 정규화, 사람에게 보류
사유 설명이다. pixel detection을 VLM 말하기로 대체하지 않는다. main에 넣으려면 grounded
localization, citation faithfulness, temporal leakage, abstention accuracy가 별도 필요하므로 이번에는
future work/demo다.

Federated learning은 실제 반출 불가 기관이 적어도 3곳 있고 기관별 local training을 실행할 때만
연다. 한 서버에서 지역을 인위적으로 silo로 나누는 것은 이번 main 기여가 아니다.

## 17. 최종 Figure/Table

### Figure 1 — 한 cache, 여러 행동

한 location-time EO cube에서 cache를 한 번 만들고 task/region이 A0/A1/A3/A4/REQUEST를 선택한다.
model release는 작은 branch, Korean public context는 별도 residual layer로 그린다.

### Figure 2 — 이미 있는 positive와 failure

Sen12 region별 P4-P2 paired gap에서 Indonesia negative를 표시한다. Solar K=0/5/20에서 stratified와
random support를 비교해 12/24 positive-free support 실패를 보여준다.

### Figure 3 — Cache value frontier

task K 또는 archive N에 따른 cold/warm/reembed/raw/bridge 비용과 utility Pareto. 실제 교차점이
없으면 그 사실을 그대로 보고한다.

### Figure 4 — Untouched action policy

Task-3/Korea에서 always-reuse, always-adapt, always-PEFT/raw, selector, oracle의 utility, harm,
coverage, cost를 비교한다. 이 그림이 CVPR method contribution의 생사를 결정한다.

### Figure 5 — Release appendix

identity AP 붕괴 → bridge AP 복구 → frozen-threshold equivalence 실패를 한 그림에 놓는다.

필수 표는 Dataset×shift×unit, action×utility×cost, per-unit paired result, contract/error taxonomy,
second-FM native diagnostic이다.

## 18. 예상 리뷰와 방어

**“그냥 다시 임베딩하면 되지 않나?”**  
N/K별 실제 비용 곡선으로 답한다. 현실적 교차점이 없으면 bridge 경제성 주장을 중단한다.

**“PANGAEA/EarthShift와 뭐가 다른가?”**  
그들은 static model quality와 shift robustness를 잰다. 우리는 이미 cache를 가진 배포자의 다음 행동과
그 regret/cost를 잰다. 이 차이는 untouched action test로 증명해야 한다.

**“OlmoEarth head가 raw보다 구조적으로 유리한 것 아닌가?”**  
official raw, parameter-matched raw, 동일 support/update, A3 PEFT/full-FT ceiling을 함께 보고한다.
현재 결과만으로 pretraining의 순수 인과효과를 주장하지 않는다.

**“두 task로 일반화 가능한가?”**  
아직 아니다. 공개 Task-3가 제출 gate다. restricted Korea는 이를 대체하지 않는다.

**“한국 데이터를 공개 못 하는데 재현 가능한가?”**  
headline protocol은 공개 3 task에서 end-to-end 재현하고 Korea는 frozen policy의 external validation으로
둔다.

**“AP 97% 복구면 migration 성공 아닌가?”**  
ranking은 복구됐지만 registered fixed-threshold decision gate는 2/8~3/8이다. geometry/ranking/
decision equivalence를 구분한 것이 결과다.

## 19. 엔지니어링 완료 조건

현재 pre-run snapshot, hash, failure genealogy는 강한 자산이다. 다음을 추가로 요구한다.

1. AI-Hub `error`를 retryable로 분리하고 error>0이면 audit fail.
2. shard summary를 전체 2,536/163으로 집계하고 split/class bias report 생성.
3. release summary를 `iou_frozen_thr`로 생성하고 v1.1 R6 invalid 표기.
4. Solar WGS84 cross-CRS leakage와 support provenance 봉인.
5. shell chain은 `set -euo pipefail`, 실제 `$?` 저장, expected file/run/hash를 통과한 뒤에만 DONE.
6. 모든 audit는 검사 건수 0이면 fail.
7. public Task-3 download→manifest→cache→train→report가 한 명령으로 재현.
8. API key는 core benchmark dependency에서 제외하고 mock/redacted request schema 제공.

권장 공개 구조:

```text
earthcache/
  contracts/       # band/time/GSD/pooling/release
  datasets/        # public builders + immutable manifests
  encoders/        # model adapters
  actions/         # A0/A1/A3/A4/REQUEST
  metrics/         # IoU/AP/regret/harm/cost
  prereg/          # frozen configs + dated amendments
  reports/         # JSON + paper tables
  tests/           # leakage/threshold/hash/fail-closed/secret scan
```

## 20. Claim ladder

### 지금 말해도 되는 것

1. 두 dense EO task에서 frozen OlmoEarth cache pathway가 등록된 raw baseline보다 강했다.
2. low-shot A1은 raw adaptation보다 강했지만 A0보다 항상 강하지 않았다.
3. Solar random K=5의 절반은 positive-free였고 A1이 크게 무너졌다.
4. v1→v1.1/v1.2 identity는 실패했고 linear bridge가 AP 대부분을 복구했다.
5. AP 복구는 frozen-threshold decision equivalence를 보장하지 않았다.
6. 현재 single-seed second-FM diagnostic에서 cache value는 자동으로 나타나지 않았다.
7. Korea cube 2,536개가 coverage file audit를 통과했지만 experiment eligibility는 미완료다.

### 실험 뒤에만 말할 것

1. selector가 unseen task에서 안전한 최소 행동을 고른다.
2. 한 Korea cache가 세 task를 비용 효율적으로 지원한다.
3. public context/API가 EO representation 또는 inference utility를 개선한다.
4. 현실적 archive N에서 bridge가 re-embedding보다 싸다.
5. OlmoEarth 외 family에도 원리가 일반화된다.

### 금지 문장

- OlmoEarth가 모든 GeoFM보다 우월하다.
- 8 Solar fold가 8 독립 지역이다.
- API를 많이 붙여 원인을 규명했다.
- AP 97% 복구가 decision 97% 보존을 뜻한다.
- cube 2,536개가 있으므로 Korea gate가 끝났다.
- GPU를 많이 썼으므로 연구 기여가 충분하다.

## 21. 초록의 조건부 뼈대

> Geospatial foundation models are increasingly consumed as stored embedding products, yet their value is
> usually evaluated as static accuracy rather than as a deployment decision. We study when an existing Earth
> embedding cache should be reused, adapted with a few target labels, or recomputed under geographic, task,
> and model-release shifts. We introduce EarthCache, a contract- and support-aware protocol that measures
> utility, harm, label cost, raw I/O, cache storage, and compute in a common action space. Across three public
> dense-prediction tasks and a restricted Korean multi-task external study, we show that frozen caches can
> outperform raw adaptation while naive few-shot adaptation can fail when support coverage is poor. A
> support-only selector reduces harmful updates and approaches the oracle utility–cost frontier on unseen
> units. Release-migration experiments further show that label-free bridges recover ranking quality while
> failing to preserve fixed decision boundaries.

`three public`, `Korean multi-task`, `selector reduces`는 아직 실험 빈칸이다. 통과 전에는 초록에 넣지
않는다.

## 22. 최종 결론

프로젝트는 충분히 유의미하고 CVPR 2027 도전도 가능하다. 다만 가능한 이유는 “한국 데이터를 많이
모았다”거나 “OlmoEarth가 좋은 모델이다”가 아니다.

> **새 지역·task가 왔을 때 저장된 Earth 표현을 그냥 쓸지, head만 고칠지, encoder를 다시 돌릴지,
> 라벨을 더 요청할지를 test label 없이 고르고, 그 선택의 utility·harm·비용을 공개 task에서
> 검증한다.**

이 한 문장을 공개 Task-3, Korea one-cache/three-task, 실제 비용 곡선으로 닫으면 CVPR main의 형태가
된다. 제주·공공 API·Alps·ICIMOD·VLM은 이 중심축을 넓히는 다음 연구 프로그램이지, 이번 제출 gate를
대체하는 장식이 아니다.

---

## 23. 두 컴퓨터 교차검증 — EarthCache 보고서 대조 (2026-09-06 00:30)

두 컴퓨터가 같은 감사를 병렬로 돌렸다. 이 절은 그 보고서의 각 주장을 이 세션의 직접 관측과
대조한 결과다. **채택 5, 정정 2, 보완 3.**

### 채택 — 그쪽이 맞고 내가 틀렸던 것

| # | 주장 | 판정 |
|---|---|---|
| A1 | Korea excluded 163건에 재시도 대상 `error` 6건이 섞였고 audit가 deterministic exclusion으로 잘못 통과시킨다 | **채택.** 6건 실물 확인 — 전부 Planetary Computer 일시 오류. `aihub_v2_shards.py`가 reason 무관하게 done 처리한다는 메커니즘까지 정확 |
| A2 | A1>A4h는 Sen12 **7/8**(내가 8/8로 적었음) | **채택.** MS-97 원문 `representation_vs_capacity: 7/8 (K=5), 7/8 (K=20)` 확인. AP 기준으로는 8/8 |
| A3 | release summary가 등록 지표 대신 target-test-dependent IoU를 사용했다 | **채택.** 단 결론 불변 — 등록 임계로 재계산해도 요구 6/8에 최대 4/8 (§1.6) |
| A4 | CVPR 2027 본문 마감 11/16 AoE | **채택.** 내가 갖고 있던 "11월 초"보다 정확. 일정표 전면 교체 |
| A5 | 논문 중심을 `EarthCache` 하나로 고정 | **채택.** §3의 한 문장과 정합 |

### 정정 — 내가 직접 관측으로 반박하는 것

**C1. 서버 상태가 낡았다.** 보고서는 과거 시점의 `decoder 41/56`을 인용했지만, 현재는
`ARCH_AXES_DONE`과 실물 **56/56** 검증까지 완료됐다(§1.5·§1.8). 진행 중 숫자를 최종 근거로
재사용하지 않는다.

**C2. architecture-axis 축이 증거표에서 통째로 빠졌다.** 보고서의 「현재 실제 증거」 8행에
이 축이 없다. 그런데 이 축은 GPU 5시간을 이미 쓴 실행이고, **MS-102의 미해결 원인 질문을
직접 좁힌다**(§1.8): cache/raw 경계가 scale에 따라 움직이지만 parameter count만으로 family
차이를 설명하지 못한다. 보고서가 같은 문단에서
"pooling·backbone·depth는 [How to Embed Matters]가 점유했다"고 적었기 때문에, 이대로 두면
**돌고 있는 실험이 선행연구에 먹힌 것으로 오분류되어 폐기된다.** §1.8에 그 구분을 명시했다 —
그 논문은 *임베딩 품질의 설계 선택*을, 우리는 *raw 재학습 대비 캐시 손익의 결정 경계*를 잰다.

### 보완 — 그쪽 주장 중 근거를 더 밝혀야 하는 것

**B1. "한국 landslide positive가 test 7 cluster 중 3개뿐"은 봉인을 건드린 관측일 수 있다.**
`config/korea_shared_cache_3task_prereg_v0.json`은 `status: labels sealed`이고 T3는
`binary mask (90 labelled tiles)`, test는 7 cluster(C02·C03·C05·C06·C07·C08·C12)다.
**cluster별 양성 분포를 알려면 라벨 메타데이터를 읽어야 한다.** 이건 방어 가능한 행위다 —
k-of-n 성공 규칙은 n을 모르면 사전등록할 수 없으므로 설계 시점에 필요한 구조적 사실이다.
그러나 **묵시적으로 하면 안 된다.** M1의 `disclosed-audit` 선례대로 (1) 무엇을 읽었는지,
(2) 왜 결과가 아니라 설계 사실인지, (3) 이후 어떤 판정에 쓰지 않을 것인지를 명시해
개봉 기록으로 남겨야 한다. 그러지 않으면 "규칙을 결과 보고 바꿨다"는 반론을 막을 수 없다(L4).

**B2. 비용 축은 "없음"이 아니라 "M38이 있고 한계가 있음"이다.** 보고서가 인용한
"UNet3D 대비 2 task, U-TAE 대비 8–12 task"는 새 계산이 아니라 **M38(장부 1738–1830행)** 이다.
M38은 이미 스스로 세 가지를 자백했다 — (a) 벽시계는 경합으로 오염돼 FLOPs로 대체했고,
(b) forward FLOPs만 세고 backward를 일괄 2배로 가정했으며, (c) 손익분기가 baseline 선택에
의존한다. 따라서 F5의 정확한 표현은 "비용 곡선 없음"이 아니라
**"FLOPs 모형은 있으나 실측 cold/warm/re-embed/저장 곡선이 없고, 기존 모형의 가정이 미검증"**이다.
새 실험을 설계할 때 M38을 baseline으로 놓고 그 가정 세 개를 표적으로 삼는 편이 싸다.

**B3. "Clay/Galileo가 raw를 확실히 못 넘음 = 비자명성 증거"는 §1.8로 더 강해진다.**
보고서는 이를 1-seed 진단이라고만 적었다. 실제로는 오늘 **용량 축 6개가 추가로 닫혔고**,
Galileo가 nano·tiny·base-half 어디서도 0/8이라는 사실은 parameter-count-only 설명을 약화한다.
그러나 네 설정은 같은 데이터·폴드·시드를 공유한 **상관된 configuration 반복**이다. 이를
독립 반복이나 시드 불확실성 해소로 쓰지 않는다.

### 합의된 다음 순서 (변경 없음)

P0 무결성 복구(error 6 재시도 · selection bias · rare-class amendment · Solar cross-CRS ·
frozen-threshold summary 재생성) → 공개 untouched Task-3(Sen1Floods11) → Korea 3-task 1회 개봉
→ 실측 비용 곡선. **10/09 go/no-go.** 여기에 두 가지만 추가한다:

- MS-108은 `ARCH_AXES_DONE`·56/56 schema/hash 검사·최종 summary 재생성까지 확정됐다.
- B1의 개봉 기록을 P0에 포함한다(라벨 메타데이터 열람 사실의 명시).
